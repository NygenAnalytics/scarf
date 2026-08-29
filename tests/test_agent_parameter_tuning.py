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


class _FakeStore:
    def __init__(
        self,
        *,
        cluster_values: np.ndarray | None = None,
        normalized_shape: tuple[int, int] = (100, 50),
    ) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.state: object = object()
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

    def get_assay_state(self, assay: str) -> object:
        assert assay
        return self.state

    def run_pca(self, *args: Any, **kwargs: Any) -> ArtifactRef:
        self._record("run_pca", *args, **kwargs)
        return self._artifacts["pca"]

    def run_lsi(self, *args: Any, **kwargs: Any) -> ArtifactRef:
        self._record("run_lsi", *args, **kwargs)
        return self._artifacts["lsi"]

    def run_custom_reduction(self, *args: Any, **kwargs: Any) -> ArtifactRef:
        self._record("run_custom_reduction", *args, **kwargs)
        return self._artifacts["identity"]

    def run_harmony(self, *args: Any, **kwargs: Any) -> ArtifactRef:
        self._record("run_harmony", *args, **kwargs)
        return self._artifacts["harmony"]

    def build_ann_index(self, *args: Any, **kwargs: Any) -> ArtifactRef:
        self._record("build_ann_index", *args, **kwargs)
        return self._artifacts["ann"]

    def query_neighbors(self, *args: Any, **kwargs: Any) -> ArtifactRef:
        self._record("query_neighbors", *args, **kwargs)
        return self._artifacts["neighbors"]

    def build_connectivity_map(self, *args: Any, **kwargs: Any) -> ArtifactRef:
        self._record("build_connectivity_map", *args, **kwargs)
        return self._artifacts["graph"]

    def run_leiden_clustering(self, *args: Any, **kwargs: Any) -> ArtifactRef:
        self._record("run_leiden_clustering", *args, **kwargs)
        return self._artifacts["clusters"]

    def load_artifact(self, ref: ArtifactRef) -> dict[str, Any]:
        if ref.kind == "normalized":
            return {"data": SimpleNamespace(shape=self.normalized_shape)}
        assert ref == self._artifacts["clusters"]
        return {"values": self.cluster_values}

    def metric_graph_silhouette(self, *args: Any, **kwargs: Any) -> np.ndarray:
        self._record("metric_graph_silhouette", *args, **kwargs)
        return np.asarray([0.2, 0.4])

    def metric_cluster_separability(self, *args: Any, **kwargs: Any) -> Any:
        self._record("metric_cluster_separability", *args, **kwargs)
        cluster_column = args[1][0]
        return SimpleNamespace(
            clustering_scores=pd.DataFrame(
                {
                    "clustering": [cluster_column],
                    "silhouette_score": [0.35],
                    "macro_f1_mean": [0.8],
                    "weighted_f1_mean": [0.85],
                }
            )
        )

    def metric_proportional_batch_mixing(self, *args: Any, **kwargs: Any) -> float:
        self._record("metric_proportional_batch_mixing", *args, **kwargs)
        return 0.7

    def metric_clisi(self, *args: Any, **kwargs: Any) -> float:
        self._record("metric_clisi", *args, **kwargs)
        return 0.9

    def metric_graph_connectivity(self, *args: Any, **kwargs: Any) -> float:
        self._record("metric_graph_connectivity", *args, **kwargs)
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
        normalized=_artifact("normalized", 1),
        fromAssay="RNA",
        cellKey="I",
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
    planning_system_prompt = parameter_search_system_prompt()
    planning_prompt = parameter_search_prompt(
        from_assay="RNA",
        cell_key="I",
        evaluations=[evaluation],
        batch_columns=["batch"],
        preservation_columns=["cell_type"],
        harmony_authorized=True,
        max_refined_candidates=3,
    )
    system_prompt = parameter_tuning_system_prompt(20)
    user_prompt = parameter_tuning_prompt(
        from_assay="RNA",
        cell_key="I",
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


def test_candidate_execution_is_branch_safe_and_returns_bounded_metrics() -> None:
    store = _FakeStore()
    deps = _dependencies(store)

    result = _evaluate(deps, "baseline")

    assert result.status == "done"
    assert result.eligible is True
    assert result.clusterColumn is not None
    assert result.clusterColumn.startswith("RNA_agent_tuning_baseline_")
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

    call_map = {name: kwargs for name, _args, kwargs in store.calls}
    for name in (
        "run_pca",
        "build_ann_index",
        "query_neighbors",
        "build_connectivity_map",
    ):
        assert call_map[name]["update_state"] is False
        assert call_map[name]["invalidate_cache"] is False
    assert call_map["build_ann_index"]["ann_parallel"] is False
    assert call_map["build_ann_index"]["rand_state"] == 4466
    assert call_map["run_leiden_clustering"]["random_seed"] == 4444
    assert call_map["run_leiden_clustering"]["label"].startswith(
        "agent_tuning_baseline_"
    )
    assert result.clusterColumn == f"RNA_{call_map['run_leiden_clustering']['label']}"
    assert call_map["run_leiden_clustering"]["invalidate_cache"] is False
    assert store.get_assay_state("RNA") is store.state


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
    handoff = validated.to_biological_handoff()
    assert handoff.clusterColumn == evaluation.clusterColumn
    assert handoff.clusterArtifact is not None
    assert (
        handoff.clusterArtifact.artifactId
        == evaluation.artifacts["clusters"].artifactId
    )


def test_selected_branch_promotion_reuses_exact_artifacts_and_updates_state() -> None:
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
    updating_calls = {
        name: kwargs
        for name, _args, kwargs in store.calls
        if name
        in {
            "run_pca",
            "build_ann_index",
            "query_neighbors",
            "build_connectivity_map",
        }
    }
    assert updating_calls
    assert all(kwargs["update_state"] is True for kwargs in updating_calls.values())
    cluster_kwargs = next(
        kwargs for name, _args, kwargs in store.calls if name == "run_leiden_clustering"
    )
    assert cluster_kwargs["label"] == evaluation.clusterLabel


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
    assert baseline.clusterColumn != comparator.clusterColumn


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


def test_candidate_state_mutation_is_not_hidden_by_a_failed_execution() -> None:
    class MutatingStore(_FakeStore):
        def run_pca(self, *args: Any, **kwargs: Any) -> ArtifactRef:
            self._record("run_pca", *args, **kwargs)
            self.state = object()
            raise ValueError("candidate failed after mutating state")

    store = MutatingStore()
    deps = _dependencies(store)

    with pytest.raises(RuntimeError, match="unexpectedly changed"):
        _evaluate(deps, "baseline")


def test_tune_parameters_validates_candidate_ids_before_model_call() -> None:
    with pytest.raises(ValueError, match="candidateId"):
        tune_parameters(
            _FakeStore(),
            model=object(),
            normalized=_artifact("normalized", 1),
            from_assay="RNA",
            candidates=[ParameterCandidate(candidateId="not allowed")],
        )


def test_harmony_candidate_requires_authorized_batch_columns() -> None:
    with pytest.raises(ValueError, match="requires batch_columns"):
        tune_parameters(
            _FakeStore(),
            model=object(),
            normalized=_artifact("normalized", 1),
            from_assay="RNA",
            candidates=[ParameterCandidate(candidateId="harmony", useHarmony=True)],
        )


def test_tuning_handoff_rejects_conflicts_and_unauthorized_harmony() -> None:
    safe_handoff = ExperimentalTuningHandoff(
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
            from_assay="RNA",
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
            from_assay="RNA",
            candidates=[ParameterCandidate(candidateId="harmony", useHarmony=True)],
            experimental_handoff=unsafe_handoff,
        )


def test_biology_handoff_requires_selected_cluster_artifact() -> None:
    report = ParameterTuningReport(
        status="done",
        fromAssay="RNA",
        recommendedCandidateId="baseline",
        evaluations=[
            ParameterCandidateEvaluation(
                candidateId="baseline",
                status="done",
                eligible=True,
                clusterColumn="RNA_cluster",
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
        cellKey="I",
        assayReports={"RNA": native},
        recommendedByAssay={"RNA": "baseline"},
    )
    integration = IntegrationCandidateEvaluation(
        integrationId="wnn_1",
        method="wnn",
        assays=["RNA", "ADT"],
        status="done",
        eligible=True,
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
    assert handoff.clusterColumn == "agent_wnn_cluster"


def test_integrated_final_selection_requires_marker_assay() -> None:
    report = ParameterTuningReport(
        status="done",
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
        evaluations=[adt_evaluation],
        recommendedCandidateId="baseline",
        evidenceIds=["candidate:baseline:clusters"],
    )
    report = ParameterTuningReport(
        status="done",
        fromAssay="RNA",
        assayReports={"RNA": rna_report, "ADT": adt_report},
        recommendedByAssay={"RNA": "baseline", "ADT": "baseline"},
    )
    integration = IntegrationCandidateEvaluation(
        integrationId="wnn_1",
        method="wnn",
        assays=["RNA", "ADT"],
        status="done",
        eligible=True,
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
        from_assay="RNA",
    )

    assert result is expected
    assert captured["store"] is store
    assert captured["model"] is model
    assert captured["normalized"] == normalized
    assert captured["config"] is agent.config


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
        from_assay="RNA",
        candidates=[ParameterCandidate.get_example()],
        experimental_handoff=ExperimentalTuningHandoff(batchAction="skip"),
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
        from_assay="RNA",
        experimental_handoff=ExperimentalTuningHandoff(batchAction="skip"),
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
        from_assay="RNA",
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
    assert all(args[0] == ["batch", "site"] for args, _kwargs in harmony_calls)
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
                fromAssay="RNA",
                candidates=[ParameterCandidate.get_example()],
                maxCandidates=1,
            ),
            ParameterTuningAssayInput(
                normalized=_artifact("normalized", 10, "ADT"),
                fromAssay="ADT",
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
                fromAssay="RNA",
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
                fromAssay="RNA",
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
        from_assay="RNA",
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
        evaluations=[evaluation],
        recommendedCandidateId="baseline",
        evidenceIds=["candidate:baseline:clusters"],
    )
    report = ParameterTuningReport(
        status="done",
        fromAssay="RNA",
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
            evaluations=[evaluation],
            recommendedCandidateId="baseline",
            evidenceIds=["candidate:baseline:clusters"],
        )
    report = ParameterTuningReport(
        status="done",
        fromAssay="RNA",
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
            fromAssay=assay,
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
