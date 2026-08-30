"""Tests for bounded parameter tuning agent execution."""

import asyncio
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic_ai import UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

import scarf.agent.parameter_tuning as parameter_tuning_module
from scarf.agent.parameter_tuning import (
    ArtifactRecord,
    CandidateComparison,
    ParameterCandidate,
    ParameterCandidateEvaluation,
    ParameterMetrics,
    ParameterSearchPlan,
    ParameterTuningAssayInput,
    ParameterTuningAgent,
    ParameterTuningBatchSearchPlan,
    ParameterTuningDependencies,
    ParameterTuningNeedsInput,
    ParameterTuningReport,
    build_initial_parameter_candidates,
    evaluate_parameter_candidate,
    execute_parameter_candidate,
    FinalGraphComparison,
    FinalGraphNeedsInput,
    FinalGraphSelection,
    finalize_parameter_tuning_selection,
    get_default_parameter_candidates,
    IntegrationCandidateEvaluation,
    IntegrationMetrics,
    parameter_batch_selection_prompt,
    parameter_search_prompt,
    parameter_search_system_prompt,
    parameter_tuning_prompt,
    parameter_tuning_system_prompt,
    promote_parameter_candidate,
    select_final_parameter_graph,
    tune_parameters,
    tune_parameters_batch,
    validate_parameter_candidate_rank,
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


def _artifact(kind: str, token: int, assay: str = "RNA") -> ArtifactRef:
    return ArtifactRef(
        scope="assay",
        kind=kind,
        artifact_id=f"{token:064x}",
        assay=assay,
    )


def _cell_selection(token: int = 8) -> ArtifactRef:
    return ArtifactRef(
        scope="datastore",
        kind="cell_selection",
        artifact_id=f"{token:064x}",
    )


class _FakeStore:
    def __init__(
        self,
        *,
        cluster_values: np.ndarray | None = None,
        normalized_shape: tuple[int, int] = (100, 50),
    ) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.normalized = _artifact("normalized", 1)
        self.cell_selection = _cell_selection()
        self.cluster_values = (
            np.asarray(cluster_values)
            if cluster_values is not None
            else np.asarray([0] * 60 + [1] * 40)
        )
        self.normalized_shape = normalized_shape
        self._artifacts = {
            "pca": _artifact("reduction", 2),
            "lsi": _artifact("reduction", 8),
            "identity": _artifact("reduction", 9),
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
        assert normalized.kind == "normalized"
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

    def run_lsi(self, *args: Any, **kwargs: Any) -> ArtifactRef:
        self._record("run_lsi", *args, **kwargs)
        return self._artifacts["lsi"]

    def run_custom_reduction(self, *args: Any, **kwargs: Any) -> ArtifactRef:
        self._record("run_custom_reduction", *args, **kwargs)
        return self._artifacts["identity"]

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

    def load_artifact(self, ref: ArtifactRef) -> dict[str, Any]:
        self._record("load_artifact", ref)
        if ref.kind == "normalized":
            return {"data": SimpleNamespace(shape=self.normalized_shape)}
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
        normalizedShape=store.normalized_shape,
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
        FinalGraphComparison,
        FinalGraphNeedsInput,
        FinalGraphSelection,
        IntegrationMetrics,
        IntegrationCandidateEvaluation,
        ParameterSearchPlan,
        ParameterTuningBatchSearchPlan,
        ParameterTuningAssayInput,
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
    assert result.clusterColumn == "RNA_agent_tuning_baseline"
    assert result.clusterLabel == "agent_tuning_baseline"
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
        {"RNA_agent_tuning_baseline": artifacts["clusters"]},
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


@pytest.mark.parametrize(
    ("candidate", "expected_call", "artifact_key"),
    [
        (
            ParameterCandidate(
                candidateId="atac_lsi",
                reductionMethod="lsi",
                dimensions=15,
            ),
            "run_lsi",
            "lsi",
        ),
        (
            ParameterCandidate(
                candidateId="adt_identity",
                reductionMethod="identity",
                dimensions=50,
            ),
            "run_custom_reduction",
            "identity",
        ),
    ],
)
def test_candidate_dispatches_modality_reduction_without_pca_metrics(
    candidate: ParameterCandidate,
    expected_call: str,
    artifact_key: str,
) -> None:
    store = _FakeStore()
    deps = _dependencies(store, candidates=[candidate])

    result = _evaluate(deps, candidate.candidateId)

    assert result.status == "done"
    assert result.effectiveDimensions == candidate.dimensions
    assert artifact_key in result.artifacts
    call_names = [name for name, _args, _kwargs in store.calls]
    assert expected_call in call_names
    assert "run_pca" not in call_names
    assert "metric_cluster_separability" not in call_names
    assert result.metrics.pcaSilhouette is None
    if candidate.reductionMethod == "lsi":
        lsi_kwargs = next(
            kwargs for name, _args, kwargs in store.calls if name == "run_lsi"
        )
        assert lsi_kwargs["skip_first"] is True
    else:
        _name, args, _kwargs = next(
            call for call in store.calls if call[0] == "run_custom_reduction"
        )
        np.testing.assert_array_equal(args[0], np.eye(50))


@pytest.mark.parametrize(
    "candidate",
    [
        ParameterCandidate(candidateId="pca_rank", dimensions=50),
        ParameterCandidate(
            candidateId="lsi_rank",
            reductionMethod="lsi",
            dimensions=50,
        ),
        ParameterCandidate(candidateId="neighbor_rank", neighborsK=100),
        ParameterCandidate(
            candidateId="identity_rank",
            reductionMethod="identity",
            dimensions=49,
        ),
    ],
)
def test_rank_invalid_candidates_fail_before_branch_operations(
    candidate: ParameterCandidate,
) -> None:
    store = _FakeStore()
    deps = _dependencies(store, candidates=[candidate])

    result = _evaluate(deps, candidate.candidateId)

    assert result.status == "failed"
    assert result.error
    assert store.calls == []


def test_identity_reduction_enforces_feature_limit() -> None:
    candidate = ParameterCandidate(
        candidateId="identity",
        reductionMethod="identity",
        dimensions=50,
    )

    with pytest.raises(ValueError, match="at most 32"):
        validate_parameter_candidate_rank(
            candidate,
            (100, 50),
            identity_feature_limit=32,
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


def test_selected_branch_resolution_reuses_exact_artifacts_without_replay() -> None:
    store = _FakeStore()
    deps = _dependencies(store)
    evaluation = _evaluate(deps, "baseline")
    report = validate_parameter_tuning_report(
        ParameterTuningReport(
            status="done",
            recommendedCandidateId="baseline",
            evidenceIds=[evaluation.evidenceIds[0]],
        ),
        deps,
    )
    store.calls.clear()

    promoted = promote_parameter_candidate(
        store,
        report=report,
        normalized=_artifact("normalized", 1),
    )

    assert promoted.artifacts == evaluation.artifacts
    assert [name for name, _args, _kwargs in store.calls] == ["inspect_artifact"]


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


def test_integrated_final_selection_separates_graph_and_marker_assays() -> None:
    native = ParameterTuningReport.get_example()
    aggregate = ParameterTuningReport(
        status="done",
        fromAssay="RNA",
        cellSelection=native.cellSelection,
        assayReports={"RNA": native},
        recommendedByAssay={"RNA": "baseline"},
    )
    integration = IntegrationCandidateEvaluation(
        integrationId="wnn_1",
        method="wnn",
        assays=["RNA", "ADT"],
        status="done",
        eligible=True,
        cellSelection=native.cellSelection,
        graphArtifact=ArtifactRecord(
            scope="datastore",
            kind="integrated_graph",
            artifactId="8" * 64,
        ),
        clusterArtifact=ArtifactRecord(
            scope="datastore",
            kind="cluster_labels",
            artifactId="9" * 64,
        ),
        clusterColumn="agent_wnn_cluster",
        metrics=IntegrationMetrics(
            nClusters=6,
            minClusterCells=40,
            modalityWeightsValid=True,
        ),
        evidenceIds=["integration:wnn_1:clusters"],
    )

    finalized = finalize_parameter_tuning_selection(
        aggregate,
        marker_assay="RNA",
        integration_evaluations=[integration],
        recommended_integration_id="wnn_1",
    )
    handoff = finalized.to_biological_handoff()

    assert finalized.graphAssay is None
    assert handoff.graphAssay is None
    assert handoff.markerAssay == "RNA"
    assert handoff.fromAssay == "RNA"
    assert handoff.clusterArtifact is not None
    assert handoff.clusterArtifact.scope == "datastore"
    assert "clusterColumn" not in type(handoff).model_fields


def test_integrated_final_selection_requires_marker_assay() -> None:
    report = ParameterTuningReport(
        status="done",
        cellSelection=ArtifactReferenceModel.from_artifact_ref(_cell_selection()),
        finalClusterColumn="integrated_cluster",
        finalClusterArtifact=ArtifactRecord(
            scope="datastore",
            kind="cluster_labels",
            artifactId="9" * 64,
        ),
    )

    with pytest.raises(ValueError, match="marker assay"):
        report.to_biological_handoff()


def test_final_graph_selector_uses_one_grounded_provider_request() -> None:
    native_evaluation = ParameterCandidateEvaluation.get_example()
    native_evaluation.artifacts["clusters"] = ArtifactRecord(
        assay="RNA",
        kind="cluster_labels",
        artifactId="7" * 64,
    )
    rna_report = ParameterTuningReport(
        status="done",
        fromAssay="RNA",
        cellSelection=native_evaluation.cellSelection,
        evaluations=[native_evaluation],
        recommendedCandidateId="baseline",
        evidenceIds=["candidate:baseline:clusters"],
    )
    adt_evaluation = native_evaluation.model_copy(
        update={
            "clusterColumn": "ADT_agent_tuning_baseline",
            "artifacts": {
                **native_evaluation.artifacts,
                "clusters": ArtifactRecord(
                    assay="ADT",
                    kind="cluster_labels",
                    artifactId="6" * 64,
                ),
            },
        }
    )
    adt_report = ParameterTuningReport(
        status="done",
        fromAssay="ADT",
        cellSelection=adt_evaluation.cellSelection,
        evaluations=[adt_evaluation],
        recommendedCandidateId="baseline",
        evidenceIds=["candidate:baseline:clusters"],
    )
    report = ParameterTuningReport(
        status="done",
        fromAssay="RNA",
        cellSelection=native_evaluation.cellSelection,
        assayReports={"RNA": rna_report, "ADT": adt_report},
        recommendedByAssay={"RNA": "baseline", "ADT": "baseline"},
    )
    integration = IntegrationCandidateEvaluation(
        integrationId="wnn_1",
        method="wnn",
        assays=["RNA", "ADT"],
        status="done",
        eligible=True,
        cellSelection=native_evaluation.cellSelection,
        graphArtifact=ArtifactRecord(
            scope="datastore",
            kind="integrated_graph",
            artifactId="8" * 64,
        ),
        clusterArtifact=ArtifactRecord(
            scope="datastore",
            kind="cluster_labels",
            artifactId="9" * 64,
        ),
        clusterColumn="agent_wnn_cluster",
        metrics=IntegrationMetrics(modalityWeightsValid=True),
        evidenceIds=["integration:wnn_1:clusters"],
    )
    model_calls = 0

    async def reply(
        _messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        selected_evidence = "integration:wnn_1:clusters"
        selection = FinalGraphSelection(
            status="done",
            selectedOptionId="integration:wnn_1",
            graphMethod="wnn",
            integrationId="wnn_1",
            markerAssay="invented",
            confidence="medium",
            rationale="The eligible WNN graph best balances the supplied evidence.",
            evidenceIds=[selected_evidence],
            comparisons=[
                FinalGraphComparison(
                    optionId=f"native:{assay}:baseline",
                    summary="The integrated option was preferred to this native graph.",
                    evidenceIds=[
                        selected_evidence,
                        f"native:{assay}:candidate:baseline:clusters",
                    ],
                )
                for assay in ("RNA", "ADT")
            ],
        )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args=selection.model_dump(),
                )
            ]
        )

    selected_report = select_final_parameter_graph(
        model=FunctionModel(reply),
        report=report,
        integration_evaluations=[integration],
        marker_assay="RNA",
    )

    assert model_calls == 1
    assert selected_report.finalSelection is not None
    assert selected_report.finalSelection.runInfo.agentName == (
        "parameter_tuning_final_graph"
    )
    assert selected_report.finalSelection.markerAssay == "RNA"
    assert selected_report.recommendedIntegrationId == "wnn_1"
    assert selected_report.finalClusterArtifact == integration.clusterArtifact


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


def test_batched_tuning_selects_all_assays_in_one_model_request() -> None:
    model_calls = 0

    async def reply(
        _messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        assay_reports = {
            assay: ParameterTuningReport(
                status="done",
                recommendedCandidateId="baseline",
                confidence="medium",
                rationale=f"The {assay} baseline is eligible.",
                evidenceIds=["candidate:baseline:clusters"],
                stopReason="The authorized branch was evaluated.",
            )
            for assay in ("RNA", "ADT")
        }
        report = ParameterTuningReport(
            status="done",
            assayReports=assay_reports,
            rationale="Both native modality screens completed.",
            evidenceIds=["candidate:baseline:clusters"],
            stopReason="Native selection is complete.",
        )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args=report.model_dump(),
                )
            ]
        )

    result = tune_parameters_batch(
        _FakeStore(),
        model=FunctionModel(reply),
        assays=[
            ParameterTuningAssayInput(
                normalized=_artifact("normalized", 1, "RNA"),
                candidates=[ParameterCandidate.get_example()],
                maxCandidates=1,
            ),
            ParameterTuningAssayInput(
                normalized=_artifact("normalized", 10, "ADT"),
                candidates=[ParameterCandidate.get_example()],
                maxCandidates=1,
            ),
        ],
        primary_assay="RNA",
    )

    assert model_calls == 1
    assert result.status == "done"
    assert set(result.assayReports) == {"RNA", "ADT"}
    assert result.recommendedByAssay == {"RNA": "baseline", "ADT": "baseline"}
    assert result.totalCandidates == 2
    assert result.fromAssay == "RNA"
    assert result.runInfo.agentName == "parameter_tuning_batch"


def test_batched_refinement_planning_allows_a_validator_retry() -> None:
    model_calls = 0

    async def reply(
        _messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            output: AgentDataModel = ParameterTuningBatchSearchPlan()
        elif model_calls == 2:
            output = ParameterTuningBatchSearchPlan(
                assayPlans={"RNA": ParameterSearchPlan(status="complete")}
            )
        else:
            output = ParameterTuningReport(
                status="done",
                assayReports={
                    "RNA": ParameterTuningReport(
                        status="done",
                        recommendedCandidateId="baseline",
                        confidence="medium",
                        rationale="The baseline candidate is eligible.",
                        evidenceIds=["candidate:baseline:clusters"],
                        stopReason="The bounded screen completed.",
                    )
                },
                rationale="The RNA native screen completed.",
                evidenceIds=["candidate:baseline:clusters"],
                stopReason="Native selection is complete.",
            )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args=output.model_dump(),
                )
            ]
        )

    result = tune_parameters_batch(
        _FakeStore(),
        model=FunctionModel(reply),
        assays=[
            ParameterTuningAssayInput(
                normalized=_artifact("normalized", 1),
                candidates=[ParameterCandidate.get_example()],
                maxCandidates=2,
                maxRefinedCandidates=1,
            )
        ],
        primary_assay="RNA",
    )

    assert model_calls == 3
    assert result.status == "done"
    assert result.recommendedByAssay == {"RNA": "baseline"}


def test_batched_tuning_falls_back_after_structured_output_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scarf.agent import parameter_tuning as module

    calls: list[str] = []

    def unavailable_structured_output(**kwargs: Any) -> None:
        calls.append(kwargs["name"])
        raise UnexpectedModelBehavior("structured output unavailable")

    monkeypatch.setattr(module, "run_agent_sync", unavailable_structured_output)
    result = tune_parameters_batch(
        _FakeStore(),
        model=object(),
        assays=[
            ParameterTuningAssayInput(
                normalized=_artifact("normalized", 1),
                candidates=[
                    ParameterCandidate.get_example(),
                    ParameterCandidate(candidateId="pca_15", dimensions=15),
                ],
                maxCandidates=3,
                maxRefinedCandidates=1,
            )
        ],
        primary_assay="RNA",
    )

    assert calls == ["parameter_batch_search_planning", "parameter_tuning_batch"]
    assert result.status == "done"
    assert result.recommendedByAssay == {"RNA": "baseline"}
    assert result.assayReports["RNA"].confidence == "low"
    assert result.assayReports["RNA"].comparisons[0].candidateId == "pca_15"
    assert result.searchPlan is not None
    assert result.searchPlan.status == "complete"
    assert result.runInfo.agentName == "parameter_tuning_batch_fallback"


def test_single_tuning_falls_back_after_structured_output_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scarf.agent import parameter_tuning as module

    def unavailable_structured_output(**_kwargs: Any) -> None:
        raise UnexpectedModelBehavior("structured output unavailable")

    monkeypatch.setattr(module, "run_agent_sync", unavailable_structured_output)
    result = tune_parameters(
        _FakeStore(),
        model=object(),
        normalized=_artifact("normalized", 1),
        candidates=[
            ParameterCandidate.get_example(),
            ParameterCandidate(candidateId="pca_15", dimensions=15),
        ],
        max_candidates=3,
        max_refined_candidates=1,
    )

    assert result.status == "done"
    assert result.recommendedCandidateId == "baseline"
    assert result.confidence == "low"
    assert result.runInfo.agentName == "parameter_tuning_fallback"


def test_single_eligible_final_graph_skips_provider_selection() -> None:
    evaluation = ParameterCandidateEvaluation.get_example()
    evaluation.artifacts["clusters"] = ArtifactRecord(
        assay="RNA",
        kind="cluster_labels",
        artifactId="7" * 64,
    )
    native_report = ParameterTuningReport(
        status="done",
        fromAssay="RNA",
        cellSelection=evaluation.cellSelection,
        evaluations=[evaluation],
        recommendedCandidateId="baseline",
        evidenceIds=["candidate:baseline:clusters"],
    )
    report = ParameterTuningReport(
        status="done",
        fromAssay="RNA",
        cellSelection=evaluation.cellSelection,
        assayReports={"RNA": native_report},
        recommendedByAssay={"RNA": "baseline"},
    )

    selected = select_final_parameter_graph(
        model=object(),
        report=report,
        integration_evaluations=[],
        marker_assay="RNA",
    )

    assert selected.status == "done"
    assert selected.finalSelection is not None
    assert selected.finalSelection.selectedOptionId == "native:RNA:baseline"
    assert selected.finalSelection.runInfo.agentName == (
        "parameter_tuning_final_graph_deterministic"
    )


def test_final_graph_retry_exhaustion_pauses_when_multiple_options_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scarf.agent import parameter_tuning as module

    reports: dict[str, ParameterTuningReport] = {}
    for assay, token in (("RNA", "7"), ("ADT", "8")):
        evaluation = ParameterCandidateEvaluation.get_example().model_copy(
            update={
                "clusterColumn": f"{assay}_agent_tuning_baseline",
                "artifacts": {
                    "connectivityMap": ArtifactRecord(
                        assay=assay,
                        kind="connectivity_map",
                        artifactId=token * 64,
                    ),
                    "clusters": ArtifactRecord(
                        assay=assay,
                        kind="cluster_labels",
                        artifactId=token * 64,
                    ),
                },
            }
        )
        reports[assay] = ParameterTuningReport(
            status="done",
            fromAssay=assay,
            cellSelection=evaluation.cellSelection,
            evaluations=[evaluation],
            recommendedCandidateId="baseline",
            evidenceIds=["candidate:baseline:clusters"],
        )
    report = ParameterTuningReport(
        status="done",
        fromAssay="RNA",
        cellSelection=next(iter(reports.values())).cellSelection,
        assayReports=reports,
        recommendedByAssay={"RNA": "baseline", "ADT": "baseline"},
    )

    def unavailable_structured_output(**_kwargs: Any) -> None:
        raise UnexpectedModelBehavior("structured output unavailable")

    monkeypatch.setattr(module, "run_agent_sync", unavailable_structured_output)
    selected = select_final_parameter_graph(
        model=object(),
        report=report,
        integration_evaluations=[],
        marker_assay="RNA",
    )

    assert selected.status == "needsInput"
    assert selected.needsInput is not None
    assert selected.needsInput.options == [
        "native:ADT:baseline",
        "native:RNA:baseline",
    ]
    assert selected.finalSelection is not None
    assert selected.finalSelection.runInfo.agentName == (
        "parameter_tuning_final_graph_fallback"
    )


def test_batched_selection_prompt_includes_persisted_resume_directions() -> None:
    dependencies = {"RNA": _dependencies(_FakeStore(), max_candidates=1)}
    dependencies["RNA"].evaluations["baseline"] = execute_parameter_candidate(
        dependencies["RNA"], "baseline"
    )
    prompt = parameter_batch_selection_prompt(
        dependencies,
        {"RNA": ParameterSearchPlan(status="complete")},
        "RNA",
        "Prefer the eligible branch that preserves the trusted label.",
    )

    assert "Prefer the eligible branch that preserves the trusted label." in prompt


def test_batched_tuning_enforces_global_candidate_limit_before_execution() -> None:
    store = _FakeStore()
    assays = [
        ParameterTuningAssayInput(
            normalized=_artifact("normalized", 1, assay),
            candidates=[ParameterCandidate.get_example()],
            maxCandidates=1,
        )
        for assay in ("RNA", "ADT")
    ]

    with pytest.raises(ValueError, match="global limit"):
        tune_parameters_batch(
            store,
            model=object(),
            assays=assays,
            max_total_candidates=1,
        )

    assert store.calls == []


def test_parameter_tuning_handoff_validation_edges() -> None:
    report = ParameterTuningReport.get_example()
    with pytest.raises(ValueError, match="must be done"):
        report.model_copy(update={"status": "failed"}).to_biological_handoff()
    with pytest.raises(ValueError, match="must recommend"):
        report.model_copy(
            update={"recommendedCandidateId": None, "finalClusterArtifact": None}
        ).to_biological_handoff()
    with pytest.raises(ValueError, match="not an eligible execution"):
        report.model_copy(
            update={
                "recommendedCandidateId": "missing",
                "finalClusterArtifact": None,
            }
        ).to_biological_handoff()

    evaluation = report.evaluations[0]
    wrong_cluster = evaluation.artifacts["clusters"].model_copy(update={"assay": "ADT"})
    wrong_evaluation = evaluation.model_copy(
        update={"artifacts": {**evaluation.artifacts, "clusters": wrong_cluster}}
    )
    with pytest.raises(ValueError, match="does not match the assay"):
        report.model_copy(
            update={"evaluations": [wrong_evaluation], "finalClusterArtifact": None}
        ).to_biological_handoff()

    integrated_cluster = ArtifactRecord(
        scope="datastore",
        kind="cluster_labels",
        artifactId="d" * 64,
    )
    integrated = report.model_copy(
        update={
            "finalClusterArtifact": integrated_cluster,
            "finalSelection": None,
            "recommendedIntegrationId": None,
            "graphAssay": None,
            "markerAssay": "RNA",
            "evidenceIds": ["candidate:baseline:clusters"],
        }
    )
    handoff = integrated.to_biological_handoff()
    assert handoff.clusterArtifact is not None
    assert handoff.clusterArtifact.artifactId == integrated_cluster.artifactId
    assert handoff.evidenceIds == ["candidate:baseline:clusters"]

    with pytest.raises(ValueError, match="exact cell selection"):
        integrated.model_copy(update={"cellSelection": None}).to_biological_handoff()
    named_datastore_cluster = integrated_cluster.model_copy(update={"assay": "RNA"})
    with pytest.raises(ValueError, match="must not name an assay"):
        integrated.model_copy(
            update={"finalClusterArtifact": named_datastore_cluster}
        ).to_biological_handoff()
    wrong_native_cluster = ArtifactRecord(
        scope="assay",
        assay="ADT",
        kind="cluster_labels",
        artifactId="e" * 64,
    )
    with pytest.raises(ValueError, match="does not match graphAssay"):
        integrated.model_copy(
            update={
                "finalClusterArtifact": wrong_native_cluster,
                "graphAssay": "RNA",
            }
        ).to_biological_handoff()


def test_final_graph_option_filtering_and_validation_edges() -> None:
    report = ParameterTuningReport.get_example()
    with pytest.raises(ValueError, match="lacks an exact cell selection"):
        parameter_tuning_module.final_graph_options(
            report.model_copy(update={"cellSelection": None}),
            [],
        )

    skipped_report = report.model_copy(
        update={
            "recommendedCandidateId": None,
            "evaluations": [],
            "assayReports": {},
        }
    )
    assert parameter_tuning_module.final_graph_options(skipped_report, []) == {}
    ineligible_report = report.model_copy(
        update={
            "evaluations": [
                report.evaluations[0].model_copy(update={"eligible": False})
            ],
            "assayReports": {},
        }
    )
    assert parameter_tuning_module.final_graph_options(ineligible_report, []) == {}
    mismatched_native = report.evaluations[0].model_copy(
        update={
            "cellSelection": ArtifactReferenceModel(
                scope="datastore",
                kind="cell_selection",
                artifactId="f" * 64,
            )
        }
    )
    with pytest.raises(ValueError, match="Native graph option uses a different"):
        parameter_tuning_module.final_graph_options(
            report.model_copy(
                update={"evaluations": [mismatched_native], "assayReports": {}}
            ),
            [],
        )

    integration = IntegrationCandidateEvaluation.get_example()
    for ignored in (
        integration.model_copy(update={"status": "failed"}),
        integration.model_copy(update={"graphArtifact": None}),
        integration.model_copy(update={"evidenceIds": []}),
        integration.model_copy(
            update={
                "method": "wnn",
                "metrics": integration.metrics.model_copy(
                    update={"modalityWeightsValid": False}
                ),
            }
        ),
    ):
        options = parameter_tuning_module.final_graph_options(report, [ignored])
        assert all(not key.startswith("integration:") for key in options)

    with pytest.raises(ValueError, match="Integrated graph option uses a different"):
        parameter_tuning_module.final_graph_options(
            report,
            [
                integration.model_copy(
                    update={
                        "cellSelection": ArtifactReferenceModel(
                            scope="datastore",
                            kind="cell_selection",
                            artifactId="1" * 64,
                        )
                    }
                )
            ],
        )
    with pytest.raises(ValueError, match="require integrationId"):
        parameter_tuning_module.final_graph_options(
            report,
            [integration.model_copy(update={"integrationId": ""})],
        )
    with pytest.raises(ValueError, match="must be datastore-scoped"):
        parameter_tuning_module.final_graph_options(
            report,
            [
                integration.model_copy(
                    update={
                        "graphArtifact": ArtifactRecord(
                            scope="assay",
                            assay="RNA",
                            kind="integrated_graph",
                            artifactId="2" * 64,
                        )
                    }
                )
            ],
        )
    with pytest.raises(ValueError, match="Duplicate integration id"):
        parameter_tuning_module.final_graph_options(report, [integration, integration])


def test_normalized_shape_and_candidate_metric_failure_edges() -> None:
    class ShapeStore:
        def __init__(self, payload: dict[str, Any]) -> None:
            self.payload = payload

        def load_artifact(self, _ref: Any) -> dict[str, Any]:
            return self.payload

    with pytest.raises(ValueError, match="does not contain"):
        parameter_tuning_module.normalized_artifact_shape(ShapeStore({}), object())
    with pytest.raises(ValueError, match="two-dimensional"):
        parameter_tuning_module.normalized_artifact_shape(
            ShapeStore({"data": SimpleNamespace(shape=(4,))}),
            object(),
        )
    with pytest.raises(ValueError, match="at least two cells"):
        parameter_tuning_module.normalized_artifact_shape(
            ShapeStore({"data": SimpleNamespace(shape=(1, 4))}),
            object(),
        )

    class FailingMetricStore(_FakeStore):
        def metric_graph_silhouette(self, *_args: Any, **_kwargs: Any) -> Any:
            raise ValueError("graph unavailable")

        def metric_cluster_separability(self, *_args: Any, **_kwargs: Any) -> Any:
            raise KeyError("separability unavailable")

        def metric_proportional_batch_mixing(
            self, *_args: Any, **_kwargs: Any
        ) -> float:
            raise TypeError("mixing unavailable")

        def metric_clisi(self, *_args: Any, **_kwargs: Any) -> float:
            raise ValueError("clisi unavailable")

        def metric_graph_connectivity(self, *_args: Any, **_kwargs: Any) -> float:
            raise KeyError("connectivity unavailable")

    store = FailingMetricStore(cluster_values=np.zeros(100, dtype=int))
    evaluation = _evaluate(_dependencies(store), "baseline")
    assert evaluation.status == "done"
    assert evaluation.eligibilityReasons == ["fewer than two clusters"]
    assert len(evaluation.warnings) == 5

    for values, message in (
        (np.asarray([[0, 1]]), "one non-empty label vector"),
        (np.asarray([0, -1]), "negative labels"),
    ):
        invalid = _evaluate(
            _dependencies(_FakeStore(cluster_values=values), min_cluster_cells=1),
            "baseline",
        )
        assert invalid.status == "failed"
        assert message in (invalid.error or "")

    harmony = ParameterCandidate.get_example().model_copy(
        update={"candidateId": "baseline_harmony", "useHarmony": True}
    )
    harmony_deps = _dependencies(_FakeStore(), candidates=[harmony])
    harmony_deps.batchColumns = ()
    denied = _evaluate(harmony_deps, harmony.candidateId)
    assert denied.status == "failed"
    assert "requires at least one authorized batch" in (denied.error or "")


def test_parameter_search_plan_validation_edges() -> None:
    candidates = [
        ParameterCandidate.get_example(),
        ParameterCandidate(candidateId="pca_15", dimensions=15),
    ]
    deps = _dependencies(_FakeStore(), candidates=candidates, max_candidates=4)
    baseline = execute_parameter_candidate(deps, "baseline")
    pca_15 = execute_parameter_candidate(deps, "pca_15")
    base = ParameterSearchPlan(
        status="refine",
        candidates=[ParameterCandidate(candidateId="refined_pca_18", dimensions=18)],
        basedOnCandidateIds=["baseline", "pca_15"],
        objectives=["Resolve the bounded tradeoff."],
        rationale="The initial screen brackets the proposed branch.",
        evidenceIds=[baseline.evidenceIds[0], pca_15.evidenceIds[0]],
        stoppingCriteria=["Execute the proposal once."],
    )
    initial_ids = ["baseline", "pca_15"]

    def validate(plan: ParameterSearchPlan, *, limit: int = 2) -> None:
        validate_parameter_search_plan(
            plan,
            deps,
            initial_candidate_ids=initial_ids,
            max_refined_candidates=limit,
        )

    changes: list[tuple[dict[str, Any], str, int]] = [
        ({"evidenceIds": ["unknown"]}, "unknown evidence", 2),
        (
            {
                "candidates": [
                    *base.candidates,
                    base.candidates[0].model_copy(
                        update={"candidateId": "refined_second"}
                    ),
                ]
            },
            "exceeds the refined candidate limit",
            1,
        ),
        ({"rationale": ""}, "requires a rationale", 2),
        ({"objectives": []}, "requires focused objectives", 2),
        ({"stoppingCriteria": []}, "requires stopping criteria", 2),
        ({"evidenceIds": []}, "requires initial-screen evidence", 2),
        ({"basedOnCandidateIds": []}, "identify its initial candidates", 2),
        (
            {"basedOnCandidateIds": ["baseline", "baseline"]},
            "Duplicate refinement parent",
            2,
        ),
        ({"basedOnCandidateIds": ["missing"]}, "successful initial", 2),
        (
            {
                "basedOnCandidateIds": ["baseline", "pca_15"],
                "evidenceIds": [baseline.evidenceIds[0]],
            },
            "cite every parent candidate",
            2,
        ),
        (
            {
                "candidates": [
                    base.candidates[0].model_copy(update={"candidateId": "bad-id"})
                ]
            },
            "only ASCII letters",
            2,
        ),
        (
            {
                "candidates": [
                    base.candidates[0].model_copy(update={"candidateId": "baseline"})
                ]
            },
            "Duplicate refined candidateId",
            2,
        ),
        (
            {
                "candidates": [
                    base.candidates[0].model_copy(
                        update={"candidateId": "refined_lsi", "reductionMethod": "lsi"}
                    )
                ]
            },
            "untested reduction method",
            2,
        ),
        (
            {
                "candidates": [
                    base.candidates[0].model_copy(update={"leidenResolution": 2.0})
                ]
            },
            "Leiden resolution",
            2,
        ),
        (
            {"candidates": [base.candidates[0].model_copy(update={"neighborsK": 20})]},
            "neighbor count",
            2,
        ),
        (
            {
                "candidates": [
                    base.candidates[0].model_copy(
                        update={
                            "candidateId": "duplicate_signature",
                            "dimensions": 21,
                        }
                    )
                ]
            },
            "duplicates an evaluated",
            2,
        ),
    ]
    for updates, message, limit in changes:
        with pytest.raises(ValueError, match=message):
            validate(base.model_copy(deep=True, update=updates), limit=limit)

    duplicate_proposals = base.model_copy(
        update={
            "candidates": [
                base.candidates[0],
                base.candidates[0].model_copy(update={"dimensions": 19}),
            ]
        }
    )
    with pytest.raises(ValueError, match="Duplicate refined candidateId"):
        validate(duplicate_proposals)


def test_parameter_tuning_report_validation_status_edges() -> None:
    store = _FakeStore()
    deps = _dependencies(store)
    evaluation = execute_parameter_candidate(deps, "baseline")
    evidence = evaluation.evidenceIds[0]

    with pytest.raises(ValueError, match="unknown evidence"):
        validate_parameter_tuning_report(
            ParameterTuningReport(
                status="needsInput",
                needsInput=ParameterTuningNeedsInput(
                    question="Choose.",
                    evidenceIds=["unknown"],
                ),
            ),
            deps,
        )
    with pytest.raises(ValueError, match="must recommend"):
        validate_parameter_tuning_report(
            ParameterTuningReport(status="done", evidenceIds=[evidence]),
            deps,
        )
    with pytest.raises(ValueError, match="concrete question"):
        validate_parameter_tuning_report(
            ParameterTuningReport(status="needsInput"),
            deps,
        )
    with pytest.raises(ValueError, match="recommendation evidence"):
        validate_parameter_tuning_report(
            ParameterTuningReport(status="done", recommendedCandidateId="baseline"),
            deps,
        )
    with pytest.raises(ValueError, match="was not executed"):
        validate_parameter_tuning_report(
            ParameterTuningReport(
                status="failed",
                recommendedCandidateId="missing",
            ),
            deps,
        )

    failed_deps = _dependencies(_FakeStore())
    failed_deps.evaluations["baseline"] = ParameterCandidateEvaluation(
        candidateId="baseline",
        status="failed",
        evidenceIds=["candidate:baseline:error"],
    )
    failed_deps.executionOrder = ["baseline"]
    with pytest.raises(ValueError, match="execution failed"):
        validate_parameter_tuning_report(
            ParameterTuningReport(
                status="failed",
                recommendedCandidateId="baseline",
                evidenceIds=["candidate:baseline:error"],
            ),
            failed_deps,
        )

    with pytest.raises(ValueError, match="must include the selected"):
        validate_parameter_tuning_report(
            ParameterTuningReport(
                status="done",
                recommendedCandidateId="baseline",
                evidenceIds=["other:evidence"],
            ),
            deps.model_copy(
                update={
                    "evaluations": {
                        "baseline": evaluation.model_copy(
                            update={"evidenceIds": ["other:evidence"]}
                        )
                    }
                }
            ),
        )

    with pytest.raises(ValueError, match="comparisons require"):
        validate_parameter_tuning_report(
            ParameterTuningReport(
                status="done",
                recommendedCandidateId="baseline",
                evidenceIds=[evidence],
                comparisons=[
                    CandidateComparison(
                        candidateId="unused",
                        summary="Not allowed for a one-candidate screen.",
                        evidenceIds=[evidence],
                    )
                ],
            ),
            deps,
        )


def test_final_graph_selection_validation_edges() -> None:
    report = ParameterTuningReport.get_example()
    integration = IntegrationCandidateEvaluation.get_example()
    native_evidence = "native:RNA:candidate:baseline:clusters"
    integration_evidence = integration.evidenceIds[0]
    comparison = FinalGraphComparison(
        optionId=f"integration:{integration.integrationId}",
        summary="Both exact executor branches were compared.",
        evidenceIds=[native_evidence, integration_evidence],
    )
    valid = FinalGraphSelection(
        status="done",
        selectedOptionId="native:RNA:baseline",
        markerAssay="RNA",
        evidenceIds=[native_evidence],
        comparisons=[comparison],
    )
    validated = parameter_tuning_module.validate_final_graph_selection(
        valid,
        report,
        integration_evaluations=[integration],
        marker_assay="RNA",
    )
    assert validated.graphMethod == "native"

    with pytest.raises(ValueError, match="must finish"):
        parameter_tuning_module.validate_final_graph_selection(
            valid,
            report.model_copy(update={"status": "failed"}),
            integration_evaluations=[integration],
            marker_assay="RNA",
        )
    no_options = report.model_copy(
        update={"evaluations": [], "recommendedCandidateId": None, "assayReports": {}}
    )
    with pytest.raises(ValueError, match="No eligible"):
        parameter_tuning_module.validate_final_graph_selection(
            valid,
            no_options,
            integration_evaluations=[],
            marker_assay="RNA",
        )
    with pytest.raises(ValueError, match="unknown evidence"):
        parameter_tuning_module.validate_final_graph_selection(
            valid.model_copy(update={"evidenceIds": ["unknown"]}),
            report,
            integration_evaluations=[integration],
            marker_assay="RNA",
        )
    with pytest.raises(ValueError, match="concrete question"):
        parameter_tuning_module.validate_final_graph_selection(
            FinalGraphSelection(status="needsInput"),
            report,
            integration_evaluations=[integration],
            marker_assay="RNA",
        )
    with pytest.raises(ValueError, match="done or needsInput"):
        parameter_tuning_module.validate_final_graph_selection(
            valid.model_copy(update={"status": "failed"}),
            report,
            integration_evaluations=[integration],
            marker_assay="RNA",
        )
    with pytest.raises(ValueError, match="not eligible"):
        parameter_tuning_module.validate_final_graph_selection(
            valid.model_copy(update={"selectedOptionId": "missing"}),
            report,
            integration_evaluations=[integration],
            marker_assay="RNA",
        )
    with pytest.raises(ValueError, match="cite selected-option"):
        parameter_tuning_module.validate_final_graph_selection(
            valid.model_copy(update={"evidenceIds": []}),
            report,
            integration_evaluations=[integration],
            marker_assay="RNA",
        )
    with pytest.raises(ValueError, match="must not contain duplicates"):
        parameter_tuning_module.validate_final_graph_selection(
            valid.model_copy(update={"comparisons": [comparison, comparison]}),
            report,
            integration_evaluations=[integration],
            marker_assay="RNA",
        )
    with pytest.raises(ValueError, match="one comparison for every"):
        parameter_tuning_module.validate_final_graph_selection(
            valid.model_copy(update={"comparisons": []}),
            report,
            integration_evaluations=[integration],
            marker_assay="RNA",
        )
    with pytest.raises(ValueError, match="cite selected-option evidence"):
        parameter_tuning_module.validate_final_graph_selection(
            valid.model_copy(
                update={
                    "comparisons": [
                        comparison.model_copy(
                            update={"evidenceIds": [integration_evidence]}
                        )
                    ]
                }
            ),
            report,
            integration_evaluations=[integration],
            marker_assay="RNA",
        )
    with pytest.raises(ValueError, match="cite comparator evidence"):
        parameter_tuning_module.validate_final_graph_selection(
            valid.model_copy(
                update={
                    "comparisons": [
                        comparison.model_copy(update={"evidenceIds": [native_evidence]})
                    ]
                }
            ),
            report,
            integration_evaluations=[integration],
            marker_assay="RNA",
        )
    with pytest.raises(ValueError, match="requires a summary"):
        parameter_tuning_module.validate_final_graph_selection(
            valid.model_copy(
                update={"comparisons": [comparison.model_copy(update={"summary": ""})]}
            ),
            report,
            integration_evaluations=[integration],
            marker_assay="RNA",
        )


def test_experimental_tuning_handoff_resolution_edges() -> None:
    selection = _cell_selection()
    safety = BatchSafetyEvidence(
        coefficient="condition",
        coefficientKind="categorical",
        observationUnit="sample",
        batchColumns=["batch"],
        status="safe",
        evidenceId="batchEstimability:condition:batch",
    )
    handoff = ExperimentalTuningHandoff(
        cellSelection=ArtifactReferenceModel.from_artifact_ref(selection),
        batchAction="evaluateHarmony",
        batchColumns=["batch"],
        preservationColumns=["condition"],
        coefficientsOfInterest=["condition"],
        batchSafety=[safety],
        evidenceIds=[safety.evidenceId],
    )
    resolved = parameter_tuning_module._resolve_experimental_tuning_handoff(
        normalized_cell_selection=selection,
        batch_columns=[],
        preservation_columns=[],
        experimental_handoff=handoff,
    )
    assert resolved == (selection, ["batch"], ["condition"])

    changes: list[tuple[dict[str, Any], str]] = [
        ({"batchColumns": ["batch", "batch"]}, "must be unique"),
        ({"cellSelection": None}, "lacks an exact cell selection"),
        (
            {
                "cellSelection": ArtifactReferenceModel.from_artifact_ref(
                    _cell_selection(9)
                )
            },
            "selection conflicts",
        ),
        ({"batchAction": "needsInput"}, "requires input"),
        ({"batchAction": "skip"}, "skip handoff must not contain"),
        ({"batchSafety": []}, "lacks safe evidence"),
        (
            {
                "batchAction": "unsafe",
                "batchSafety": [safety.model_copy(update={"status": "safe"})],
            },
            "lacks exact unsafe",
        ),
        ({"evidenceIds": []}, "does not cite"),
    ]
    for updates, message in changes:
        with pytest.raises(ValueError, match=message):
            parameter_tuning_module._resolve_experimental_tuning_handoff(
                normalized_cell_selection=selection,
                batch_columns=[],
                preservation_columns=[],
                experimental_handoff=handoff.model_copy(update=updates),
            )
    with pytest.raises(ValueError, match="batch_columns conflict"):
        parameter_tuning_module._resolve_experimental_tuning_handoff(
            normalized_cell_selection=selection,
            batch_columns=["other"],
            preservation_columns=[],
            experimental_handoff=handoff,
        )
    with pytest.raises(ValueError, match="preservation_columns conflict"):
        parameter_tuning_module._resolve_experimental_tuning_handoff(
            normalized_cell_selection=selection,
            batch_columns=[],
            preservation_columns=["other"],
            experimental_handoff=handoff,
        )


def test_prepare_parameter_tuning_dependencies_validation_edges() -> None:
    store = _FakeStore()
    normalized = store.normalized
    candidate = ParameterCandidate.get_example()

    for kwargs, message in (
        ({"max_candidates": 0}, "max_candidates"),
        ({"max_refined_candidates": -1}, "max_refined_candidates"),
        ({"min_cluster_cells": 0}, "min_cluster_cells"),
        ({"identity_feature_limit": 1}, "identity_feature_limit"),
    ):
        with pytest.raises(ValueError, match=message):
            parameter_tuning_module.prepare_parameter_tuning_dependencies(
                store,
                normalized=normalized,
                **kwargs,
            )
    with pytest.raises(TypeError, match="normalized ArtifactRef"):
        parameter_tuning_module.prepare_parameter_tuning_dependencies(
            store,
            normalized=_artifact("reduction", 10),
        )
    with pytest.raises(ValueError, match="has no assay"):
        parameter_tuning_module.prepare_parameter_tuning_dependencies(
            store,
            normalized=ArtifactRef(
                scope="datastore",
                kind="normalized",
                artifact_id="a" * 64,
            ),
        )

    class StatusStore(_FakeStore):
        def __init__(self, status: Any) -> None:
            super().__init__()
            self.status = status

        def inspect_artifact(self, _normalized: ArtifactRef) -> Any:
            return self.status

    statuses = (
        (SimpleNamespace(exists=False, complete=True, inputs={}), "does not exist"),
        (SimpleNamespace(exists=True, complete=False, inputs={}), "is incomplete"),
        (SimpleNamespace(exists=True, complete=True, inputs={}), "no cell-selection"),
        (
            SimpleNamespace(
                exists=True,
                complete=True,
                inputs={
                    "cell_selection": ArtifactRef(
                        scope="assay",
                        assay="RNA",
                        kind="cell_selection",
                        artifact_id="b" * 64,
                    ).to_dict()
                },
            ),
            "invalid cell-selection",
        ),
    )
    for status, message in statuses:
        with pytest.raises(ValueError, match=message):
            invalid_store = StatusStore(status)
            parameter_tuning_module.prepare_parameter_tuning_dependencies(
                invalid_store,
                normalized=invalid_store.normalized,
            )

    with pytest.raises(ValueError, match="batch_columns must be unique"):
        parameter_tuning_module.prepare_parameter_tuning_dependencies(
            store,
            normalized=normalized,
            batch_columns=["batch", "batch"],
        )
    with pytest.raises(ValueError, match="candidates must be non-empty"):
        parameter_tuning_module.prepare_parameter_tuning_dependencies(
            store,
            normalized=normalized,
            candidates=[],
        )
    with pytest.raises(ValueError, match="exceeds max_candidates"):
        parameter_tuning_module.prepare_parameter_tuning_dependencies(
            store,
            normalized=normalized,
            candidates=[
                candidate,
                candidate.model_copy(update={"candidateId": "other"}),
            ],
            max_candidates=1,
        )
    with pytest.raises(ValueError, match="only ASCII"):
        parameter_tuning_module.prepare_parameter_tuning_dependencies(
            store,
            normalized=normalized,
            candidates=[candidate.model_copy(update={"candidateId": "bad-id"})],
        )
    with pytest.raises(ValueError, match="Duplicate candidateId"):
        parameter_tuning_module.prepare_parameter_tuning_dependencies(
            store,
            normalized=normalized,
            candidates=[candidate, candidate],
        )
