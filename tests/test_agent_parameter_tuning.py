"""Tests for bounded parameter tuning agent execution."""

import asyncio
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from scarf.agent.parameter_tuning import (
    ArtifactRecord,
    CandidateComparison,
    ParameterCandidate,
    ParameterCandidateEvaluation,
    ParameterMetrics,
    ParameterSearchPlan,
    ParameterTuningAgent,
    ParameterTuningDependencies,
    ParameterTuningNeedsInput,
    ParameterTuningReport,
    build_initial_parameter_candidates,
    evaluate_parameter_candidate,
    execute_parameter_candidate,
    get_default_parameter_candidates,
    parameter_search_prompt,
    parameter_search_system_prompt,
    parameter_tuning_prompt,
    parameter_tuning_system_prompt,
    tune_parameters,
    validate_parameter_search_plan,
    validate_parameter_tuning_report,
)
from scarf.agent.types import (
    AgentDataModel,
    ArtifactReferenceModel,
    BatchSafetyEvidence,
    ExperimentalTuningHandoff,
)
from scarf.storage.refs import ArtifactRef


def _artifact(kind: str, token: int) -> ArtifactRef:
    return ArtifactRef(
        scope="assay",
        kind=kind,
        artifact_id=f"{token:064x}",
        assay="RNA",
    )


def _cell_selection(token: int = 8) -> ArtifactRef:
    return ArtifactRef(
        scope="datastore",
        kind="cell_selection",
        artifact_id=f"{token:064x}",
    )


class _FakeStore:
    def __init__(self, *, cluster_values: np.ndarray | None = None) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.normalized = _artifact("normalized", 1)
        self.cell_selection = _cell_selection()
        self.cluster_values = (
            np.asarray(cluster_values)
            if cluster_values is not None
            else np.asarray([0] * 60 + [1] * 40)
        )
        self._artifacts = {
            "pca": _artifact("reduction", 2),
            "harmony": _artifact("batch_correction", 3),
            "ann": _artifact("ann_index", 4),
            "neighbors": _artifact("neighbors", 5),
            "graph": _artifact("connectivity_map", 6),
            "clusters": _artifact("cluster_labels", 7),
        }

    def _record(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append((name, args, kwargs))

    def inspect_artifact(self, normalized: ArtifactRef) -> Any:
        self._record("inspect_artifact", normalized)
        assert normalized == self.normalized
        return SimpleNamespace(
            exists=True,
            complete=True,
            inputs={"cell_selection": self.cell_selection.to_dict()},
        )

    def run_pca(
        self,
        normalized: ArtifactRef,
        *,
        dims: int,
        feat_scaling: bool,
        show_elbow_plot: bool,
        invalidate_cache: bool,
    ) -> ArtifactRef:
        self._record(
            "run_pca",
            normalized,
            dims=dims,
            feat_scaling=feat_scaling,
            show_elbow_plot=show_elbow_plot,
            invalidate_cache=invalidate_cache,
        )
        return self._artifacts["pca"]

    def run_harmony(
        self,
        reduction: ArtifactRef,
        batch_columns: list[str],
        *,
        invalidate_cache: bool,
    ) -> ArtifactRef:
        self._record(
            "run_harmony",
            reduction,
            batch_columns,
            invalidate_cache=invalidate_cache,
        )
        return self._artifacts["harmony"]

    def build_ann_index(
        self,
        coordinates: ArtifactRef,
        *,
        ann_metric: str,
        ann_parallel: bool,
        rand_state: int,
        invalidate_cache: bool,
    ) -> ArtifactRef:
        self._record(
            "build_ann_index",
            coordinates,
            ann_metric=ann_metric,
            ann_parallel=ann_parallel,
            rand_state=rand_state,
            invalidate_cache=invalidate_cache,
        )
        return self._artifacts["ann"]

    def query_neighbors(
        self,
        ann_index: ArtifactRef,
        *,
        coordinates: ArtifactRef,
        k: int,
        invalidate_cache: bool,
    ) -> ArtifactRef:
        self._record(
            "query_neighbors",
            ann_index,
            coordinates=coordinates,
            k=k,
            invalidate_cache=invalidate_cache,
        )
        return self._artifacts["neighbors"]

    def build_connectivity_map(
        self,
        neighbors: ArtifactRef,
        *,
        local_connectivity: float,
        bandwidth: float,
        invalidate_cache: bool,
    ) -> ArtifactRef:
        self._record(
            "build_connectivity_map",
            neighbors,
            local_connectivity=local_connectivity,
            bandwidth=bandwidth,
            invalidate_cache=invalidate_cache,
        )
        return self._artifacts["graph"]

    def run_leiden_clustering(
        self,
        graph: ArtifactRef,
        *,
        resolution: float,
        backend: str,
        symmetric_graph: bool,
        graph_upper_only: bool,
        random_seed: int,
        invalidate_cache: bool,
    ) -> ArtifactRef:
        self._record(
            "run_leiden_clustering",
            graph,
            resolution=resolution,
            backend=backend,
            symmetric_graph=symmetric_graph,
            graph_upper_only=graph_upper_only,
            random_seed=random_seed,
            invalidate_cache=invalidate_cache,
        )
        return self._artifacts["clusters"]

    def load_artifact(self, ref: ArtifactRef) -> dict[str, np.ndarray]:
        self._record("load_artifact", ref)
        assert ref == self._artifacts["clusters"]
        return {"values": self.cluster_values}

    def metric_graph_silhouette(
        self,
        neighbors: ArtifactRef,
        clusters: ArtifactRef,
        *,
        random_seed: int,
        sample_size: int,
    ) -> np.ndarray:
        self._record(
            "metric_graph_silhouette",
            neighbors,
            clusters,
            random_seed=random_seed,
            sample_size=sample_size,
        )
        return np.asarray([0.2, 0.4])

    def metric_cluster_separability(
        self,
        pca: ArtifactRef,
        clusters: dict[str, ArtifactRef],
        *,
        random_seed: int,
    ) -> Any:
        self._record(
            "metric_cluster_separability",
            pca,
            clusters,
            random_seed=random_seed,
        )
        cluster_name = next(iter(clusters))
        return SimpleNamespace(
            clustering_scores=pd.DataFrame(
                {
                    "clustering": [cluster_name],
                    "silhouette_score": [0.35],
                    "macro_f1_mean": [0.8],
                    "weighted_f1_mean": [0.85],
                }
            )
        )

    def metric_proportional_batch_mixing(
        self,
        label_column: str,
        neighbors: ArtifactRef,
        *,
        perplexity: float,
    ) -> float:
        self._record(
            "metric_proportional_batch_mixing",
            label_column,
            neighbors,
            perplexity=perplexity,
        )
        return 0.7

    def metric_clisi(
        self,
        label_column: str,
        neighbors: ArtifactRef,
        *,
        perplexity: float | None,
        scale: bool,
    ) -> float:
        self._record(
            "metric_clisi",
            label_column,
            neighbors,
            perplexity=perplexity,
            scale=scale,
        )
        return 0.9

    def metric_graph_connectivity(
        self,
        label_column: str,
        graph: ArtifactRef,
    ) -> float:
        self._record("metric_graph_connectivity", label_column, graph)
        return 0.95


def _dependencies(
    store: _FakeStore,
    *,
    candidates: list[ParameterCandidate] | None = None,
    max_candidates: int = 5,
    min_cluster_cells: int = 20,
) -> ParameterTuningDependencies:
    candidate_values = candidates or [ParameterCandidate.get_example()]
    return ParameterTuningDependencies(
        store=store,
        normalized=store.normalized,
        cellSelection=store.cell_selection,
        fromAssay="RNA",
        candidates={value.candidateId: value for value in candidate_values},
        batchColumns=("batch",),
        preservationColumns=("cell_type",),
        maxCandidates=max_candidates,
        minClusterCells=min_cluster_cells,
    )


def _context(deps: ParameterTuningDependencies) -> Any:
    return SimpleNamespace(deps=deps)


def _evaluate(
    deps: ParameterTuningDependencies,
    candidate_id: str,
) -> ParameterCandidateEvaluation:
    return asyncio.run(evaluate_parameter_candidate(_context(deps), candidate_id))


@pytest.mark.parametrize(
    "model_type",
    [
        ArtifactRecord,
        CandidateComparison,
        ParameterCandidate,
        ParameterMetrics,
        ParameterCandidateEvaluation,
        ParameterSearchPlan,
        ParameterTuningNeedsInput,
        ParameterTuningReport,
        ParameterTuningDependencies,
    ],
)
def test_parameter_models_have_blank_and_example(
    model_type: type[AgentDataModel],
) -> None:
    assert isinstance(model_type.get_blank(), model_type)
    assert isinstance(model_type.get_example(), model_type)
    assert all("_" not in field for field in model_type.model_fields)


def test_default_candidates_are_explicit_unique_one_factor_variants() -> None:
    candidates = get_default_parameter_candidates()
    assert [candidate.candidateId for candidate in candidates] == [
        "baseline",
        "pca_15",
        "pca_30",
        "leiden_0_5",
        "leiden_1_5",
    ]
    baseline = candidates[0]
    assert (
        sum(candidate.dimensions != baseline.dimensions for candidate in candidates)
        == 2
    )
    assert (
        sum(
            candidate.leidenResolution != baseline.leidenResolution
            for candidate in candidates
        )
        == 2
    )


def test_harmony_pairing_covers_every_initial_candidate() -> None:
    seeds = get_default_parameter_candidates()

    candidates = build_initial_parameter_candidates(seeds, pair_harmony=True)

    assert len(candidates) == 10
    for index, seed in enumerate(seeds):
        uncorrected, corrected = candidates[index * 2 : index * 2 + 2]
        assert uncorrected == seed
        assert corrected.candidateId == f"{seed.candidateId}_harmony"
        assert corrected.useHarmony is True
        assert corrected.model_dump(exclude={"candidateId", "useHarmony"}) == (
            seed.model_dump(exclude={"candidateId", "useHarmony"})
        )


def test_prompts_include_only_explicit_candidate_context() -> None:
    evaluation = ParameterCandidateEvaluation.get_example()
    plan = ParameterSearchPlan(status="complete")
    cell_selection = ArtifactReferenceModel.from_artifact_ref(_cell_selection())
    planning_system_prompt = parameter_search_system_prompt()
    planning_prompt = parameter_search_prompt(
        from_assay="RNA",
        cell_selection=cell_selection,
        evaluations=[evaluation],
        batch_columns=["batch"],
        preservation_columns=["cell_type"],
        harmony_authorized=True,
        max_refined_candidates=3,
    )
    system_prompt = parameter_tuning_system_prompt(20)
    user_prompt = parameter_tuning_prompt(
        from_assay="RNA",
        cell_selection=cell_selection,
        evaluations=[evaluation],
        batch_columns=["batch"],
        preservation_columns=["cell_type"],
        search_plan=plan,
    )

    assert planning_system_prompt.startswith(
        "You are planning one bounded refinement pass"
    )
    assert '"candidateId": "baseline"' in planning_prompt
    assert '"harmony"' in planning_prompt
    assert "Maximum refined candidates: 3" in planning_prompt
    assert "status=complete with candidates=[]" in planning_system_prompt
    assert "status=refine with one or more candidates" in planning_system_prompt
    assert system_prompt.startswith("You are Scarf's parameter tuning selection agent.")
    assert "20 cells" in system_prompt
    assert '"candidateId": "baseline"' in user_prompt
    assert cell_selection.artifactId in planning_prompt
    assert cell_selection.artifactId in user_prompt
    assert "Do not request tools" in system_prompt
    assert "one comparison for every non-selected successful candidate" in system_prompt
    assert "Validated refinement plan" in user_prompt


def test_unknown_candidate_is_rejected_without_store_calls() -> None:
    store = _FakeStore()
    deps = _dependencies(store)

    result = _evaluate(deps, "invented")

    assert result.status == "failed"
    assert "Unknown candidate" in (result.error or "")
    assert store.calls == []
    assert deps.executionOrder == []


def test_candidate_execution_routes_exact_artifacts_without_state_updates() -> None:
    store = _FakeStore()
    deps = _dependencies(store)

    result = _evaluate(deps, "baseline")

    assert result.status == "done"
    assert result.eligible is True
    assert result.cellSelection == ArtifactReferenceModel.from_artifact_ref(
        store.cell_selection
    )
    assert "clusterColumn" not in ParameterCandidateEvaluation.model_fields
    assert result.metrics.nClusters == 2
    assert result.metrics.minClusterCells == 40
    assert result.metrics.graphSilhouetteMedian == pytest.approx(0.3)
    assert result.metrics.pcaSilhouette == pytest.approx(0.35)
    assert result.metrics.batchMixing == {"batch": 0.7}
    assert result.metrics.biologicalPreservation == {
        "cell_type": {"clisi": 0.9, "graphConnectivity": 0.95}
    }
    assert set(result.artifacts) == {
        "pca",
        "annIndex",
        "neighbors",
        "connectivityMap",
        "clusters",
    }

    assert [name for name, _args, _kwargs in store.calls] == [
        "run_pca",
        "build_ann_index",
        "query_neighbors",
        "build_connectivity_map",
        "run_leiden_clustering",
        "load_artifact",
        "metric_graph_silhouette",
        "metric_cluster_separability",
        "metric_proportional_batch_mixing",
        "metric_clisi",
        "metric_graph_connectivity",
    ]
    assert all("update_state" not in kwargs for _name, _args, kwargs in store.calls)

    call_map = {name: (args, kwargs) for name, args, kwargs in store.calls}
    artifacts = store._artifacts
    assert call_map["run_pca"] == (
        (deps.normalized,),
        {
            "dims": 21,
            "feat_scaling": True,
            "show_elbow_plot": False,
            "invalidate_cache": False,
        },
    )
    assert call_map["build_ann_index"] == (
        (artifacts["pca"],),
        {
            "ann_metric": "l2",
            "ann_parallel": False,
            "rand_state": 4466,
            "invalidate_cache": False,
        },
    )
    assert call_map["query_neighbors"] == (
        (artifacts["ann"],),
        {
            "coordinates": artifacts["pca"],
            "k": 11,
            "invalidate_cache": False,
        },
    )
    assert call_map["build_connectivity_map"] == (
        (artifacts["neighbors"],),
        {
            "local_connectivity": 1.0,
            "bandwidth": 1.5,
            "invalidate_cache": False,
        },
    )
    assert call_map["run_leiden_clustering"] == (
        (artifacts["graph"],),
        {
            "resolution": 1.0,
            "backend": "igraph",
            "symmetric_graph": False,
            "graph_upper_only": False,
            "random_seed": 4444,
            "invalidate_cache": False,
        },
    )
    assert call_map["load_artifact"] == ((artifacts["clusters"],), {})
    assert call_map["metric_graph_silhouette"][0] == (
        artifacts["neighbors"],
        artifacts["clusters"],
    )
    assert call_map["metric_cluster_separability"][0] == (
        artifacts["pca"],
        {"clusters": artifacts["clusters"]},
    )
    assert call_map["metric_proportional_batch_mixing"][0] == (
        "batch",
        artifacts["neighbors"],
    )
    assert call_map["metric_clisi"][0] == (
        "cell_type",
        artifacts["neighbors"],
    )
    assert call_map["metric_graph_connectivity"] == (
        ("cell_type", artifacts["graph"]),
        {},
    )


def test_duplicate_candidate_returns_recorded_execution_without_rerun() -> None:
    store = _FakeStore()
    deps = _dependencies(store)

    first = _evaluate(deps, "baseline")
    call_count = len(store.calls)
    second = _evaluate(deps, "baseline")

    assert second is first
    assert len(store.calls) == call_count
    assert deps.executionOrder == ["baseline"]


def test_candidate_budget_prevents_another_execution() -> None:
    store = _FakeStore()
    candidates = [
        ParameterCandidate.get_example(),
        ParameterCandidate(candidateId="pca_15", dimensions=15),
    ]
    deps = _dependencies(store, candidates=candidates, max_candidates=1)

    _evaluate(deps, "baseline")
    call_count = len(store.calls)
    result = _evaluate(deps, "pca_15")

    assert result.status == "failed"
    assert "limit 1 reached" in (result.error or "")
    assert len(store.calls) == call_count


def test_refinement_plan_is_bounded_by_initial_evidence_and_envelope() -> None:
    store = _FakeStore()
    candidates = [
        ParameterCandidate.get_example(),
        ParameterCandidate(candidateId="pca_15", dimensions=15),
    ]
    deps = _dependencies(store, candidates=candidates, max_candidates=3)
    baseline = execute_parameter_candidate(deps, "baseline")
    pca_15 = execute_parameter_candidate(deps, "pca_15")
    plan = ParameterSearchPlan(
        status="refine",
        candidates=[ParameterCandidate(candidateId="refined_pca_18", dimensions=18)],
        basedOnCandidateIds=["baseline", "pca_15"],
        objectives=["Resolve the dimension tradeoff."],
        rationale="The initial candidates bracket a narrower dimension choice.",
        evidenceIds=[baseline.evidenceIds[0], pca_15.evidenceIds[0]],
        stoppingCriteria=["Execute the proposed candidate once."],
    )

    validated = validate_parameter_search_plan(
        plan,
        deps,
        initial_candidate_ids=["baseline", "pca_15"],
        max_refined_candidates=2,
    )

    assert validated == plan
    plan.candidates[0].dimensions = 30
    with pytest.raises(ValueError, match="initial search envelope"):
        validate_parameter_search_plan(
            plan,
            deps,
            initial_candidate_ids=["baseline", "pca_15"],
            max_refined_candidates=2,
        )


def test_refinement_plan_canonicalizes_status_from_candidate_presence() -> None:
    store = _FakeStore()
    candidates = [
        ParameterCandidate.get_example(),
        ParameterCandidate(candidateId="pca_15", dimensions=15),
    ]
    deps = _dependencies(store, candidates=candidates, max_candidates=3)
    baseline = execute_parameter_candidate(deps, "baseline")
    pca_15 = execute_parameter_candidate(deps, "pca_15")
    plan = ParameterSearchPlan.model_validate(
        {
            "candidates": [
                ParameterCandidate(
                    candidateId="refined_pca_18",
                    dimensions=18,
                ).model_dump()
            ],
            "basedOnCandidateIds": ["baseline", "pca_15"],
            "objectives": ["Resolve the dimension tradeoff."],
            "rationale": (
                "The initial candidates bracket a narrower dimension choice."
            ),
            "evidenceIds": [baseline.evidenceIds[0], pca_15.evidenceIds[0]],
            "stoppingCriteria": ["Execute the proposed candidate once."],
        }
    )

    validated = validate_parameter_search_plan(
        plan,
        deps,
        initial_candidate_ids=["baseline", "pca_15"],
        max_refined_candidates=2,
    )
    complete = validate_parameter_search_plan(
        ParameterSearchPlan(status="refine"),
        deps,
        initial_candidate_ids=["baseline", "pca_15"],
        max_refined_candidates=2,
    )

    assert "status" not in plan.model_fields_set
    assert validated.status == "refine"
    assert validated.candidates == plan.candidates
    assert complete.status == "complete"


def test_refinement_plan_requires_authorized_matched_harmony_evidence() -> None:
    store = _FakeStore()
    candidates = [
        ParameterCandidate.get_example(),
        ParameterCandidate(candidateId="pca_15", dimensions=15),
    ]
    deps = _dependencies(store, candidates=candidates, max_candidates=3)
    baseline = execute_parameter_candidate(deps, "baseline")
    pca_15 = execute_parameter_candidate(deps, "pca_15")
    plan = ParameterSearchPlan(
        status="refine",
        candidates=[
            ParameterCandidate(
                candidateId="refined_pca_18_harmony",
                dimensions=18,
                useHarmony=True,
            )
        ],
        basedOnCandidateIds=["baseline", "pca_15"],
        objectives=["Resolve the dimension tradeoff."],
        rationale="The initial candidates bracket a narrower dimension choice.",
        evidenceIds=[baseline.evidenceIds[0], pca_15.evidenceIds[0]],
        stoppingCriteria=["Execute the proposed candidate once."],
    )

    with pytest.raises(ValueError, match="not authorized for Harmony"):
        validate_parameter_search_plan(
            plan,
            deps,
            initial_candidate_ids=["baseline", "pca_15"],
            max_refined_candidates=2,
        )

    corrected = [
        ParameterCandidate(
            candidateId="baseline_harmony",
            useHarmony=True,
        ),
        ParameterCandidate(
            candidateId="pca_15_harmony",
            dimensions=15,
            useHarmony=True,
        ),
    ]
    deps.candidates.update(
        {candidate.candidateId: candidate for candidate in corrected}
    )
    deps.maxCandidates = 4
    deps.harmonyAuthorized = True
    baseline_harmony = execute_parameter_candidate(deps, "baseline_harmony")
    pca_15_harmony = execute_parameter_candidate(deps, "pca_15_harmony")
    initial_ids = ["baseline", "pca_15", "baseline_harmony", "pca_15_harmony"]
    plan.basedOnCandidateIds = ["baseline_harmony", "pca_15_harmony"]
    plan.evidenceIds = [
        baseline_harmony.evidenceIds[0],
        pca_15_harmony.evidenceIds[0],
    ]
    with pytest.raises(ValueError, match="matched corrected and uncorrected"):
        validate_parameter_search_plan(
            plan,
            deps,
            initial_candidate_ids=initial_ids,
            max_refined_candidates=2,
        )

    plan.basedOnCandidateIds = ["pca_15", "pca_15_harmony"]
    plan.evidenceIds = [pca_15.evidenceIds[0], pca_15_harmony.evidenceIds[0]]
    plan.harmonyBatchColumns = ["other"]
    with pytest.raises(ValueError, match="cannot modify"):
        validate_parameter_search_plan(
            plan,
            deps,
            initial_candidate_ids=initial_ids,
            max_refined_candidates=2,
        )
    plan.harmonyBatchColumns = []
    validated = validate_parameter_search_plan(
        plan,
        deps,
        initial_candidate_ids=initial_ids,
        max_refined_candidates=2,
    )
    assert validated.harmonyBatchColumns == ["batch"]


def test_small_cluster_marks_candidate_ineligible() -> None:
    store = _FakeStore(cluster_values=np.asarray([0] * 98 + [1] * 2))
    deps = _dependencies(store, min_cluster_cells=20)

    result = _evaluate(deps, "baseline")

    assert result.status == "done"
    assert result.eligible is False
    assert result.eligibilityReasons == ["smallest cluster has 2 cells; minimum is 20"]


def test_report_validation_uses_only_executed_results_and_artifacts() -> None:
    store = _FakeStore()
    deps = _dependencies(store)
    evaluation = _evaluate(deps, "baseline")
    report = ParameterTuningReport(
        status="done",
        evaluations=[ParameterCandidateEvaluation.get_blank()],
        recommendedCandidateId="baseline",
        selectedArtifacts={"invented": ArtifactRecord.get_example()},
        confidence="medium",
        rationale="Balanced candidate.",
        evidenceIds=[evaluation.evidenceIds[0]],
        stopReason="Enough candidates evaluated.",
    )

    validated = validate_parameter_tuning_report(report, deps)

    assert validated.evaluations == [evaluation]
    assert validated.selectedArtifacts == evaluation.artifacts
    assert validated.cellSelection == evaluation.cellSelection
    handoff = validated.to_biological_handoff()
    assert handoff.cellSelection == evaluation.cellSelection
    assert "clusterColumn" not in type(handoff).model_fields
    assert handoff.clusterArtifact is not None
    assert (
        handoff.clusterArtifact.artifactId
        == evaluation.artifacts["clusters"].artifactId
    )


def test_report_validation_rejects_unknown_evidence_and_ineligible_choice() -> None:
    store = _FakeStore(cluster_values=np.asarray([0] * 98 + [1] * 2))
    deps = _dependencies(store)
    evaluation = _evaluate(deps, "baseline")

    with pytest.raises(ValueError, match="unknown evidence"):
        validate_parameter_tuning_report(
            ParameterTuningReport(
                status="done",
                recommendedCandidateId=None,
                evidenceIds=["candidate:invented:metric"],
            ),
            deps,
        )

    with pytest.raises(ValueError, match="not eligible"):
        validate_parameter_tuning_report(
            ParameterTuningReport(
                status="done",
                recommendedCandidateId="baseline",
                evidenceIds=[evaluation.evidenceIds[0]],
            ),
            deps,
        )


def test_done_report_requires_a_successful_comparator_when_available() -> None:
    store = _FakeStore()
    candidates = [
        ParameterCandidate.get_example(),
        ParameterCandidate(candidateId="pca_15", dimensions=15),
    ]
    deps = _dependencies(store, candidates=candidates, max_candidates=2)
    baseline = _evaluate(deps, "baseline")
    report = ParameterTuningReport(
        status="done",
        recommendedCandidateId="baseline",
        evidenceIds=[baseline.evidenceIds[0]],
    )

    with pytest.raises(ValueError, match="at least two successful"):
        validate_parameter_tuning_report(report, deps)

    comparator = _evaluate(deps, "pca_15")
    with pytest.raises(ValueError, match="comparisons for every successful"):
        validate_parameter_tuning_report(report, deps)

    report.comparisons = [
        CandidateComparison(
            candidateId="pca_15",
            summary="Baseline keeps larger minimum clusters.",
            evidenceIds=[
                baseline.evidenceIds[0],
                comparator.evidenceIds[0],
            ],
        )
    ]
    validated = validate_parameter_tuning_report(report, deps)

    assert [item.candidateId for item in validated.evaluations] == [
        "baseline",
        "pca_15",
    ]
    assert baseline.cellSelection == comparator.cellSelection
    assert baseline.evidenceIds[0] != comparator.evidenceIds[0]


def test_comparison_requires_selected_and_comparator_evidence() -> None:
    store = _FakeStore()
    candidates = [
        ParameterCandidate.get_example(),
        ParameterCandidate(candidateId="pca_15", dimensions=15),
    ]
    deps = _dependencies(store, candidates=candidates, max_candidates=2)
    baseline = _evaluate(deps, "baseline")
    comparator = _evaluate(deps, "pca_15")
    report = ParameterTuningReport(
        status="done",
        recommendedCandidateId="baseline",
        evidenceIds=[baseline.evidenceIds[0]],
        comparisons=[
            CandidateComparison(
                candidateId="pca_15",
                evidenceIds=[baseline.evidenceIds[0]],
            )
        ],
    )

    with pytest.raises(ValueError, match="evidence from its comparator"):
        validate_parameter_tuning_report(report, deps)

    report.comparisons[0].evidenceIds = [comparator.evidenceIds[0]]
    with pytest.raises(ValueError, match="evidence from the selected"):
        validate_parameter_tuning_report(report, deps)

    report.comparisons[0].evidenceIds = [
        baseline.evidenceIds[0],
        comparator.evidenceIds[0],
    ]
    with pytest.raises(ValueError, match="grounded summary"):
        validate_parameter_tuning_report(report, deps)


def test_comparison_rejects_unknown_or_duplicate_candidate_ids() -> None:
    store = _FakeStore()
    candidates = [
        ParameterCandidate.get_example(),
        ParameterCandidate(candidateId="pca_15", dimensions=15),
    ]
    deps = _dependencies(store, candidates=candidates, max_candidates=2)
    baseline = _evaluate(deps, "baseline")
    comparator = _evaluate(deps, "pca_15")
    evidence_ids = [baseline.evidenceIds[0], comparator.evidenceIds[0]]
    report = ParameterTuningReport(
        status="done",
        recommendedCandidateId="baseline",
        evidenceIds=[baseline.evidenceIds[0]],
        comparisons=[
            CandidateComparison(
                candidateId="pca_15",
                evidenceIds=evidence_ids,
            ),
            CandidateComparison(
                candidateId="invented",
                evidenceIds=evidence_ids,
            ),
        ],
    )

    with pytest.raises(ValueError, match="successful non-selected"):
        validate_parameter_tuning_report(report, deps)

    report.comparisons[1].candidateId = "pca_15"
    with pytest.raises(ValueError, match="Duplicate candidate comparisons"):
        validate_parameter_tuning_report(report, deps)


def test_candidate_execution_does_not_call_assay_state_apis() -> None:
    class StateTrackingStore(_FakeStore):
        def __init__(self) -> None:
            super().__init__()
            self.state_calls: list[tuple[str, tuple[Any, ...]]] = []

        def get_assay_state(self, assay: str) -> object:
            self.state_calls.append(("get_assay_state", (assay,)))
            return object()

        def update_assay_state(self, assay: str, state: object) -> None:
            self.state_calls.append(("update_assay_state", (assay, state)))

    store = StateTrackingStore()
    deps = _dependencies(store)

    result = _evaluate(deps, "baseline")

    assert result.status == "done"
    assert store.state_calls == []


def test_tune_parameters_validates_candidate_ids_before_model_call() -> None:
    with pytest.raises(ValueError, match="candidateId"):
        tune_parameters(
            _FakeStore(),
            model=object(),
            normalized=_artifact("normalized", 1),
            candidates=[ParameterCandidate(candidateId="not allowed")],
        )


def test_harmony_candidate_requires_authorized_batch_columns() -> None:
    with pytest.raises(ValueError, match="requires batch_columns"):
        tune_parameters(
            _FakeStore(),
            model=object(),
            normalized=_artifact("normalized", 1),
            candidates=[ParameterCandidate(candidateId="harmony", useHarmony=True)],
        )


def test_tuning_handoff_rejects_conflicts_and_unauthorized_harmony() -> None:
    safe_handoff = ExperimentalTuningHandoff(
        cellSelection=ArtifactReferenceModel.from_artifact_ref(_cell_selection()),
        batchAction="evaluateHarmony",
        batchColumns=["batch"],
        preservationColumns=["disease"],
        coefficientsOfInterest=["disease"],
        batchSafety=[
            BatchSafetyEvidence(
                coefficient="disease",
                coefficientKind="categorical",
                observationUnit="sample",
                batchColumns=["batch"],
                unitConstantBatchColumns=["batch"],
                status="safe",
                evidenceId="batchEstimability:disease:batch",
            )
        ],
        evidenceIds=["batchEstimability:disease:batch"],
    )
    with pytest.raises(ValueError, match="batch_columns conflict"):
        tune_parameters(
            _FakeStore(),
            model=object(),
            normalized=_artifact("normalized", 1),
            batch_columns=["other"],
            experimental_handoff=safe_handoff,
        )

    unsafe_handoff = safe_handoff.model_copy(
        update={
            "batchAction": "unsafe",
            "batchSafety": [
                safe_handoff.batchSafety[0].model_copy(update={"status": "unsafe"})
            ],
        }
    )
    with pytest.raises(ValueError, match="not authorized for Harmony"):
        tune_parameters(
            _FakeStore(),
            model=object(),
            normalized=_artifact("normalized", 1),
            candidates=[ParameterCandidate(candidateId="harmony", useHarmony=True)],
            experimental_handoff=unsafe_handoff,
        )


def test_biology_handoff_requires_selected_cluster_artifact() -> None:
    cell_selection = ArtifactReferenceModel.from_artifact_ref(_cell_selection())
    report = ParameterTuningReport(
        status="done",
        fromAssay="RNA",
        cellSelection=cell_selection,
        recommendedCandidateId="baseline",
        evaluations=[
            ParameterCandidateEvaluation(
                candidateId="baseline",
                status="done",
                eligible=True,
                cellSelection=cell_selection,
            )
        ],
    )

    with pytest.raises(ValueError, match="lacks an exact cluster artifact"):
        report.to_biological_handoff()


def test_parameter_tuning_agent_delegates_with_its_model_and_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scarf.agent import parameter_tuning as module

    model = object()
    expected = ParameterTuningReport.get_blank()
    captured: dict[str, Any] = {}

    def fake_tune(store: Any, **kwargs: Any) -> ParameterTuningReport:
        captured["store"] = store
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(module, "tune_parameters", fake_tune)
    store = _FakeStore()
    normalized = _artifact("normalized", 1)
    agent = ParameterTuningAgent(model)

    result = agent.run(
        store,
        normalized=normalized,
    )

    assert result is expected
    assert captured["store"] is store
    assert captured["model"] is model
    assert captured["normalized"] == normalized
    assert captured["config"] is agent.config
    assert "from_assay" not in captured


def test_parameter_tuning_skips_unrequested_refinement_planning() -> None:
    model_calls = 0
    seen_function_tools: list[set[str]] = []

    async def reply(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        seen_function_tools.append({tool.name for tool in info.function_tools})
        report = ParameterTuningReport(
            status="done",
            recommendedCandidateId="baseline",
            confidence="medium",
            rationale="The executed baseline is eligible.",
            evidenceIds=["candidate:baseline:clusters"],
            stopReason="The authorized candidate was evaluated.",
        )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args=report.model_dump(),
                )
            ]
        )

    result = ParameterTuningAgent(FunctionModel(reply)).run(
        _FakeStore(),
        normalized=_artifact("normalized", 1),
        candidates=[ParameterCandidate.get_example()],
        experimental_handoff=ExperimentalTuningHandoff(
            cellSelection=ArtifactReferenceModel.from_artifact_ref(_cell_selection()),
            batchAction="skip",
        ),
        max_candidates=1,
    )

    assert result.status == "done"
    assert result.recommendedCandidateId == "baseline"
    assert result.searchPlan is not None
    assert result.searchPlan.status == "complete"
    assert result.searchPlan.rationale.startswith("Refinement was not authorized")
    assert result.searchPlan.runInfo.usage.requests == 0
    assert result.runInfo.agentName == "parameter_tuning"
    assert result.runInfo.toolCalls == []
    assert model_calls == 1
    assert seen_function_tools == [set()]


def test_default_parameter_screen_uses_one_selection_model_call() -> None:
    candidate_ids = [
        candidate.candidateId for candidate in get_default_parameter_candidates()
    ]
    model_calls = 0

    async def reply(
        _messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        selected_evidence = "candidate:baseline:clusters"
        report = ParameterTuningReport(
            status="done",
            recommendedCandidateId="baseline",
            confidence="medium",
            rationale="The baseline is the most balanced eligible branch.",
            evidenceIds=[selected_evidence],
            comparisons=[
                CandidateComparison(
                    candidateId=candidate_id,
                    summary="The baseline retains the preferred balance.",
                    evidenceIds=[
                        selected_evidence,
                        f"candidate:{candidate_id}:clusters",
                    ],
                )
                for candidate_id in candidate_ids
                if candidate_id != "baseline"
            ],
            stopReason="The deterministic initial screen is complete.",
        )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args=report.model_dump(),
                )
            ]
        )

    result = ParameterTuningAgent(FunctionModel(reply)).run(
        _FakeStore(),
        normalized=_artifact("normalized", 1),
        experimental_handoff=ExperimentalTuningHandoff(
            cellSelection=ArtifactReferenceModel.from_artifact_ref(_cell_selection()),
            batchAction="skip",
        ),
    )

    assert model_calls == 1
    assert [item.candidateId for item in result.evaluations] == candidate_ids
    assert result.searchPlan is not None
    assert result.searchPlan.status == "complete"
    assert result.recommendedCandidateId == "baseline"


def test_two_pass_tuning_pairs_exact_multicolumn_harmony_and_runs_refinement() -> None:
    initial_ids = [
        "baseline",
        "baseline_harmony",
        "pca_15",
        "pca_15_harmony",
        "pca_30",
        "pca_30_harmony",
        "leiden_0_5",
        "leiden_0_5_harmony",
        "leiden_1_5",
        "leiden_1_5_harmony",
    ]
    selected_id = "refined_pca_18_harmony"
    model_calls = 0

    async def reply(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        assert info.function_tools == []
        if model_calls == 1:
            plan = ParameterSearchPlan(
                status="complete",
                candidates=[
                    ParameterCandidate(
                        candidateId=selected_id,
                        dimensions=18,
                        leidenResolution=1.0,
                        neighborsK=11,
                        useHarmony=True,
                    )
                ],
                basedOnCandidateIds=["pca_15", "pca_15_harmony"],
                objectives=["Resolve the corrected dimension tradeoff."],
                rationale="The corrected initial branches bracket a narrower choice.",
                evidenceIds=[
                    "candidate:pca_15:clusters",
                    "candidate:pca_15_harmony:clusters",
                ],
                stoppingCriteria=["Execute the proposed corrected branch once."],
            )
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=info.output_tools[0].name,
                        args=plan.model_dump(),
                    )
                ]
            )
        selected_evidence = f"candidate:{selected_id}:clusters"
        report = ParameterTuningReport(
            status="done",
            recommendedCandidateId=selected_id,
            confidence="medium",
            rationale="The refined corrected branch is eligible.",
            evidenceIds=[selected_evidence],
            comparisons=[
                CandidateComparison(
                    candidateId=candidate_id,
                    summary="The refined branch was selected after bounded comparison.",
                    evidenceIds=[
                        selected_evidence,
                        f"candidate:{candidate_id}:clusters",
                    ],
                )
                for candidate_id in initial_ids
            ],
            stopReason="The validated refinement pass completed.",
        )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args=report.model_dump(),
                )
            ]
        )

    handoff = ExperimentalTuningHandoff(
        cellSelection=ArtifactReferenceModel.from_artifact_ref(_cell_selection()),
        batchAction="evaluateHarmony",
        batchColumns=["batch", "site"],
        preservationColumns=["disease"],
        coefficientsOfInterest=["disease"],
        batchSafety=[
            BatchSafetyEvidence(
                coefficient="disease",
                coefficientKind="categorical",
                observationUnit="sample",
                batchColumns=["batch", "site"],
                unitConstantBatchColumns=["batch", "site"],
                status="safe",
                evidenceId="batchEstimability:disease:batch,site",
            )
        ],
        evidenceIds=["batchEstimability:disease:batch,site"],
    )
    store = _FakeStore()

    result = ParameterTuningAgent(FunctionModel(reply)).run(
        store,
        normalized=_artifact("normalized", 1),
        experimental_handoff=handoff,
        max_refined_candidates=2,
    )

    assert model_calls == 2
    assert result.searchPlan is not None
    assert result.searchPlan.status == "refine"
    assert [evaluation.candidateId for evaluation in result.evaluations] == [
        *initial_ids,
        selected_id,
    ]
    assert [evaluation.phase for evaluation in result.evaluations] == [
        *(["initial"] * len(initial_ids)),
        "refined",
    ]
    harmony_calls = [
        (args, kwargs) for name, args, kwargs in store.calls if name == "run_harmony"
    ]
    assert len(harmony_calls) == 6
    assert all(
        args == (store._artifacts["pca"], ["batch", "site"])
        for args, _kwargs in harmony_calls
    )
    assert all(
        evaluation.harmonyBatchColumns == ["batch", "site"]
        for evaluation in result.evaluations
        if evaluation.parameters.useHarmony
    )
    assert all(
        evaluation.harmonyBatchColumns == []
        for evaluation in result.evaluations
        if not evaluation.parameters.useHarmony
    )
    assert result.searchPlan is not None
    assert result.searchPlan.candidates[0].useHarmony is True
    assert result.searchPlan.harmonyBatchColumns == ["batch", "site"]
    assert result.searchPlan.runInfo.agentName == "parameter_search_planning"
    assert result.recommendedCandidateId == selected_id
