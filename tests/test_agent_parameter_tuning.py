"""Tests for bounded parameter tuning agent execution."""

from tests.agent_examples import example

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

import scarf.agent.parameter_tuning.agent as parameter_tuning_agent
import scarf.agent.parameter_tuning.execution as parameter_tuning_execution
from scarf.agent.parameter_tuning import (
    ArtifactRecord,
    CandidateComparison,
    ParameterCandidate,
    ParameterCandidateEvaluation,
    ParameterMetrics,
    ParameterSearchPlan,
    ParameterTuningDependencies,
    ParameterTuningNeedsInput,
    ParameterTuningReport,
    build_initial_parameter_candidates,
    execute_parameter_candidate,
    FinalGraphComparison,
    FinalGraphNeedsInput,
    FinalGraphSelection,
    finalize_parameter_tuning_selection,
    get_default_parameter_candidates,
    IntegrationCandidateEvaluation,
    IntegrationMetrics,
    promote_parameter_candidate,
    validate_parameter_candidate_rank,
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
    candidate_values = candidates or [example(ParameterCandidate)]
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
        ParameterTuningNeedsInput,
        ParameterTuningReport,
        ParameterTuningDependencies,
    ],
)
def test_parameter_models_have_blank_and_example(
    model_type: type[AgentDataModel],
) -> None:
    assert isinstance(model_type.get_blank(), model_type)
    assert isinstance(example(model_type), model_type)
    assert all("_" not in field for field in model_type.model_fields)


def test_blank_integration_evaluation_uses_wnn_default() -> None:
    assert IntegrationCandidateEvaluation.get_blank().method == "wnn"


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


def test_unknown_candidate_is_rejected_without_store_calls() -> None:
    store = _FakeStore()
    deps = _dependencies(store)

    result = execute_parameter_candidate(deps, "invented")

    assert result.status == "failed"
    assert "Unknown candidate" in (result.error or "")
    assert store.calls == []
    assert deps.executionOrder == []


def test_candidate_execution_routes_exact_artifacts_without_state_updates() -> None:
    store = _FakeStore()
    deps = _dependencies(store)

    result = execute_parameter_candidate(deps, "baseline")

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

    result = execute_parameter_candidate(deps, candidate.candidateId)

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

    result = execute_parameter_candidate(deps, candidate.candidateId)

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

    first = execute_parameter_candidate(deps, "baseline")
    call_count = len(store.calls)
    second = execute_parameter_candidate(deps, "baseline")

    assert second is first
    assert len(store.calls) == call_count
    assert deps.executionOrder == ["baseline"]


def test_candidate_budget_prevents_another_execution() -> None:
    store = _FakeStore()
    candidates = [
        example(ParameterCandidate),
        ParameterCandidate(candidateId="pca_15", dimensions=15),
    ]
    deps = _dependencies(store, candidates=candidates, max_candidates=1)

    execute_parameter_candidate(deps, "baseline")
    call_count = len(store.calls)
    result = execute_parameter_candidate(deps, "pca_15")

    assert result.status == "failed"
    assert "limit 1 reached" in (result.error or "")
    assert len(store.calls) == call_count


def test_small_cluster_marks_candidate_ineligible() -> None:
    store = _FakeStore(cluster_values=np.asarray([0] * 98 + [1] * 2))
    deps = _dependencies(store, min_cluster_cells=20)

    result = execute_parameter_candidate(deps, "baseline")

    assert result.status == "done"
    assert result.eligible is False
    assert result.eligibilityReasons == ["smallest cluster has 2 cells; minimum is 20"]


def test_selected_branch_resolution_reuses_exact_artifacts_without_replay() -> None:
    store = _FakeStore()
    deps = _dependencies(store)
    evaluation = execute_parameter_candidate(deps, "baseline")
    report = ParameterTuningReport(
        status="done",
        fromAssay="RNA",
        cellSelection=evaluation.cellSelection,
        evaluations=[evaluation],
        recommendedCandidateId="baseline",
        evidenceIds=[evaluation.evidenceIds[0]],
    )
    store.calls.clear()

    promoted = promote_parameter_candidate(
        store,
        report=report,
        normalized=_artifact("normalized", 1),
    )

    assert promoted.artifacts == evaluation.artifacts
    assert [name for name, _args, _kwargs in store.calls] == ["inspect_artifact"]


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

    result = execute_parameter_candidate(deps, "baseline")

    assert result.status == "done"
    assert store.state_calls == []


def test_harmony_candidate_requires_authorized_batch_columns() -> None:
    with pytest.raises(ValueError, match="requires batch_columns"):
        parameter_tuning_agent.prepare_parameter_tuning_dependencies(
            _FakeStore(),
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
        parameter_tuning_agent.prepare_parameter_tuning_dependencies(
            _FakeStore(),
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
        parameter_tuning_agent.prepare_parameter_tuning_dependencies(
            _FakeStore(),
            normalized=_artifact("normalized", 1),
            candidates=[ParameterCandidate(candidateId="harmony", useHarmony=True)],
            experimental_handoff=unsafe_handoff,
        )


def test_integrated_final_selection_separates_graph_and_marker_assays() -> None:
    native = example(ParameterTuningReport)
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

    assert finalized.graphAssay is None
    assert finalized.markerAssay == "RNA"
    assert finalized.fromAssay == "RNA"
    assert finalized.finalClusterArtifact == integration.clusterArtifact
    assert finalized.finalClusterColumn == "agent_wnn_cluster"


def test_harmony_candidate_uses_exact_multicolumn_batch_columns() -> None:
    store = _FakeStore()
    candidates = [
        example(ParameterCandidate),
        ParameterCandidate(candidateId="baseline_harmony", useHarmony=True),
    ]
    deps = _dependencies(store, candidates=candidates)
    deps.batchColumns = ("batch", "site")

    native = execute_parameter_candidate(deps, "baseline")
    corrected = execute_parameter_candidate(deps, "baseline_harmony")

    assert native.harmonyBatchColumns == []
    assert corrected.harmonyBatchColumns == ["batch", "site"]
    assert [args for name, args, _kwargs in store.calls if name == "run_harmony"] == [
        (store._artifacts["pca"], ["batch", "site"])
    ]


def test_normalized_shape_and_candidate_metric_failure_edges() -> None:
    class ShapeStore:
        def __init__(self, payload: dict[str, Any]) -> None:
            self.payload = payload

        def load_artifact(self, _ref: Any) -> dict[str, Any]:
            return self.payload

    with pytest.raises(ValueError, match="does not contain"):
        parameter_tuning_execution.normalized_artifact_shape(ShapeStore({}), object())
    with pytest.raises(ValueError, match="two-dimensional"):
        parameter_tuning_execution.normalized_artifact_shape(
            ShapeStore({"data": SimpleNamespace(shape=(4,))}),
            object(),
        )
    with pytest.raises(ValueError, match="at least two cells"):
        parameter_tuning_execution.normalized_artifact_shape(
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
    evaluation = execute_parameter_candidate(_dependencies(store), "baseline")
    assert evaluation.status == "done"
    assert evaluation.eligibilityReasons == ["fewer than two clusters"]
    assert len(evaluation.warnings) == 5

    for values, message in (
        (np.asarray([[0, 1]]), "one non-empty label vector"),
        (np.asarray([0, -1]), "negative labels"),
    ):
        invalid = execute_parameter_candidate(
            _dependencies(_FakeStore(cluster_values=values), min_cluster_cells=1),
            "baseline",
        )
        assert invalid.status == "failed"
        assert message in (invalid.error or "")

    harmony = example(ParameterCandidate).model_copy(
        update={"candidateId": "baseline_harmony", "useHarmony": True}
    )
    harmony_deps = _dependencies(_FakeStore(), candidates=[harmony])
    harmony_deps.batchColumns = ()
    denied = execute_parameter_candidate(harmony_deps, harmony.candidateId)
    assert denied.status == "failed"
    assert "requires at least one authorized batch" in (denied.error or "")


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
    resolved = parameter_tuning_agent._resolve_experimental_tuning_handoff(
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
            parameter_tuning_agent._resolve_experimental_tuning_handoff(
                normalized_cell_selection=selection,
                batch_columns=[],
                preservation_columns=[],
                experimental_handoff=handoff.model_copy(update=updates),
            )
    with pytest.raises(ValueError, match="batch_columns conflict"):
        parameter_tuning_agent._resolve_experimental_tuning_handoff(
            normalized_cell_selection=selection,
            batch_columns=["other"],
            preservation_columns=[],
            experimental_handoff=handoff,
        )
    with pytest.raises(ValueError, match="preservation_columns conflict"):
        parameter_tuning_agent._resolve_experimental_tuning_handoff(
            normalized_cell_selection=selection,
            batch_columns=[],
            preservation_columns=["other"],
            experimental_handoff=handoff,
        )


def test_prepare_parameter_tuning_dependencies_validation_edges() -> None:
    store = _FakeStore()
    normalized = store.normalized
    candidate = example(ParameterCandidate)

    for kwargs, message in (
        ({"max_candidates": 0}, "max_candidates"),
        ({"max_refined_candidates": -1}, "max_refined_candidates"),
        ({"min_cluster_cells": 0}, "min_cluster_cells"),
        ({"identity_feature_limit": 1}, "identity_feature_limit"),
    ):
        with pytest.raises(ValueError, match=message):
            parameter_tuning_agent.prepare_parameter_tuning_dependencies(
                store,
                normalized=normalized,
                **kwargs,
            )
    with pytest.raises(TypeError, match="normalized ArtifactRef"):
        parameter_tuning_agent.prepare_parameter_tuning_dependencies(
            store,
            normalized=_artifact("reduction", 10),
        )
    with pytest.raises(ValueError, match="has no assay"):
        parameter_tuning_agent.prepare_parameter_tuning_dependencies(
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
            parameter_tuning_agent.prepare_parameter_tuning_dependencies(
                invalid_store,
                normalized=invalid_store.normalized,
            )

    with pytest.raises(ValueError, match="batch_columns must be unique"):
        parameter_tuning_agent.prepare_parameter_tuning_dependencies(
            store,
            normalized=normalized,
            batch_columns=["batch", "batch"],
        )
    with pytest.raises(ValueError, match="candidates must be non-empty"):
        parameter_tuning_agent.prepare_parameter_tuning_dependencies(
            store,
            normalized=normalized,
            candidates=[],
        )
    with pytest.raises(ValueError, match="exceeds max_candidates"):
        parameter_tuning_agent.prepare_parameter_tuning_dependencies(
            store,
            normalized=normalized,
            candidates=[
                candidate,
                candidate.model_copy(update={"candidateId": "other"}),
            ],
            max_candidates=1,
        )
    with pytest.raises(ValueError, match="only ASCII"):
        parameter_tuning_agent.prepare_parameter_tuning_dependencies(
            store,
            normalized=normalized,
            candidates=[candidate.model_copy(update={"candidateId": "bad-id"})],
        )
    with pytest.raises(ValueError, match="Duplicate candidateId"):
        parameter_tuning_agent.prepare_parameter_tuning_dependencies(
            store,
            normalized=normalized,
            candidates=[candidate, candidate],
        )
