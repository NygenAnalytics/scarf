import time
from pathlib import Path
from typing import Any

import pytest

from profiling.config import (
    GRAPH_CONSTRUCTION_STAGE_ORDER,
    StageName,
    StageResources,
    WorkflowParameters,
)
from profiling.stages import _run_analysis, run_stage
from scarf import DataStore
from scarf.storage import ArtifactRef


_FEATURE_REF = ArtifactRef(
    scope="assay",
    assay="RNA",
    kind="feature_selection",
    artifact_id="a" * 64,
)


def _resources() -> StageResources:
    return StageResources(
        modalMemoryRequestMb=4096,
        modalMemoryLimitMb=4096,
        modalCpuRequest=1.0,
        modalCpuLimit=1.0,
        scarfMemoryBudget=2 * 1024**3,
        workers=3,
        timeoutSeconds=600,
        ephemeralDiskMb=524_288,
    )


class _RecordingStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.resolved_features: list[tuple[str, str]] = []

    def resolve_features(self, assay: str, features: str) -> ArtifactRef:
        self.resolved_features.append((assay, features))
        return _FEATURE_REF

    @staticmethod
    def _get_assay(_assay: str) -> object:
        return object()

    @staticmethod
    def _ensure_all_features(_assay: object) -> ArtifactRef:
        return _FEATURE_REF

    def __getattr__(self, name: str) -> Any:
        def record(*args: Any, **kwargs: Any) -> object:
            self.calls.append((name, args, kwargs))
            return object()

        return record


@pytest.mark.parametrize(
    ("stage", "method", "expected"),
    [
        (
            "runNormalization",
            "run_normalization",
            {
                "from_assay": "RNA",
                "cell_key": "I",
                "features": _FEATURE_REF,
                "update_state": True,
                "invalidate_cache": False,
            },
        ),
        (
            "runPca",
            "run_pca",
            {
                "from_assay": "RNA",
                "dims": 37,
                "local_cache": False,
                "show_elbow_plot": False,
                "update_state": True,
                "invalidate_cache": False,
            },
        ),
        (
            "buildEmbeddingInitialization",
            "build_embedding_initialization",
            {
                "from_assay": "RNA",
                "n_centroids": 321,
                "rand_state": 99,
                "kmeans_sampling": 0.2,
                "kmeans_batch_size": 5_000,
                "update_state": True,
                "invalidate_cache": False,
            },
        ),
        (
            "buildAnnIndex",
            "build_ann_index",
            {
                "from_assay": "RNA",
                "ann_efc": 51,
                "ann_ef": 51,
                "ann_m": 55,
                "ann_parallel": True,
                "rand_state": 99,
                "update_state": True,
                "invalidate_cache": False,
            },
        ),
        (
            "queryNeighbors",
            "query_neighbors",
            {
                "from_assay": "RNA",
                "k": 17,
                "update_state": True,
                "invalidate_cache": False,
            },
        ),
        (
            "buildConnectivityMap",
            "build_connectivity_map",
            {
                "from_assay": "RNA",
                "update_state": True,
                "invalidate_cache": False,
            },
        ),
    ],
)
def test_graph_construction_profile_stage_selects_state_and_preserves_parameters(
    stage: StageName,
    method: str,
    expected: dict[str, Any],
) -> None:
    store = _RecordingStore()
    workflow = WorkflowParameters(
        dims=37,
        nCentroids=321,
        graphSeed=99,
        kmeansSampling=0.2,
        kmeansBatchSize=5_000,
        annParallel=True,
        k=17,
        graphLocalCache=False,
    )

    _run_analysis(
        stage,
        store,
        workflow,
        _resources(),
    )

    assert store.calls == [(method, (), expected)]
    if stage == "runNormalization":
        assert store.resolved_features == [("RNA", "hvgs")]


def test_profile_normalization_consumes_returned_hvg_ref() -> None:
    store = _RecordingStore()

    _run_analysis(
        "runNormalization",
        store,
        WorkflowParameters(),
        _resources(),
        hvgRef=_FEATURE_REF,
    )

    assert store.resolved_features == []
    assert store.calls == [
        (
            "run_normalization",
            (),
            {
                "from_assay": "RNA",
                "cell_key": "I",
                "features": _FEATURE_REF,
                "update_state": True,
                "invalidate_cache": False,
            },
        )
    ]


@pytest.mark.parametrize("marker_features", ["all_features", "marker_panel"])
def test_profile_marker_search_resolves_explicit_features(
    marker_features: str,
) -> None:
    store = _RecordingStore()

    _run_analysis(
        "findMarkers",
        store,
        WorkflowParameters(markerFeatures=marker_features),
        _resources(),
    )

    assert store.resolved_features == [("RNA", marker_features)]
    marker_calls = [call for call in store.calls if call[0] == "run_marker_search"]
    assert len(marker_calls) == 1
    assert marker_calls[0][2]["features"] == _FEATURE_REF


def test_forced_profile_stages_invalidate_reusable_artifacts() -> None:
    workflow = WorkflowParameters()
    for stage in (*GRAPH_CONSTRUCTION_STAGE_ORDER, "findMarkers"):
        store = _RecordingStore()
        _run_analysis(
            stage,
            store,
            workflow,
            _resources(),
            invalidateCache=True,
        )
        assert store.calls
        assert all(
            kwargs.get("invalidate_cache") is True
            for method, _args, kwargs in store.calls
            if not method.startswith("_")
        )


def test_run_stage_reports_store_open_as_input_setup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = object()
    analysis_kwargs: dict[str, Any] = {}

    def open_store(*_args: Any, **_kwargs: Any) -> object:
        time.sleep(0.02)
        return store

    def run_analysis(*_args: Any, **kwargs: Any) -> None:
        analysis_kwargs.update(kwargs)
        time.sleep(0.02)

    monkeypatch.setattr("profiling.stages._open_datastore", open_store)
    monkeypatch.setattr("profiling.stages._run_analysis", run_analysis)

    result = run_stage(
        "runNormalization",
        nRows=10_000,
        storeUri=str(tmp_path / "store.zarr"),
        workflow=WorkflowParameters(),
        resources=_resources(),
        sampleIntervalSeconds=0.005,
        invalidateCache=True,
    )

    assert result.status == "ok"
    assert result.inputSetupSeconds is not None
    assert result.inputSetupSeconds >= 0.015
    assert result.seconds is not None
    assert result.seconds >= 0.015
    assert result.wholeFunctionSeconds is not None
    assert result.wholeFunctionSeconds >= (result.inputSetupSeconds + result.seconds)
    assert result.modalMemoryMb == 4096
    assert result.modalCpuRequest == 1.0
    assert result.modalCpuLimit == 1.0
    assert analysis_kwargs["invalidateCache"] is True


def test_run_stage_session_carries_hvg_ref_to_normalization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = object()
    seen_refs: list[ArtifactRef | None] = []

    monkeypatch.setattr(
        "profiling.stages._open_datastore",
        lambda *_args, **_kwargs: store,
    )

    def run_analysis(
        stage: StageName,
        *_args: Any,
        hvgRef: ArtifactRef | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any] | None:
        seen_refs.append(hvgRef)
        if stage == "markHvgs":
            return {"artifact": _FEATURE_REF.to_dict()}
        return None

    monkeypatch.setattr("profiling.stages._run_analysis", run_analysis)
    session: dict[str, Any] = {}
    common = {
        "nRows": 10_000,
        "storeUri": str(tmp_path / "store.zarr"),
        "workflow": WorkflowParameters(),
        "resources": _resources(),
        "sampleIntervalSeconds": 0.005,
        "recordStoreOperations": False,
        "session": session,
    }

    marked = run_stage("markHvgs", **common)
    normalized = run_stage("runNormalization", **common)

    assert marked.status == "ok"
    assert normalized.status == "ok"
    assert seen_refs == [None, _FEATURE_REF]
    assert session["hvgRef"] == _FEATURE_REF


@pytest.mark.slow
def test_graph_construction_profile_stages_chain_through_persisted_assay_state(
    datastore_ephemeral: DataStore,
) -> None:
    datastore_ephemeral.auto_filter_cells(show_qc_plots=False)
    datastore_ephemeral.mark_hvgs(
        from_assay="RNA",
        cell_key="I",
        top_n=100,
        label="profile_hvgs",
        show_plot=False,
    )
    store_uri = str(datastore_ephemeral.zarr_loc)
    workflow = WorkflowParameters(
        hvgLabel="profile_hvgs",
        dims=5,
        nCentroids=20,
        k=3,
        graphLocalCache=False,
    )

    for stage in GRAPH_CONSTRUCTION_STAGE_ORDER:
        result = run_stage(
            stage,
            nRows=datastore_ephemeral.cells.N,
            storeUri=store_uri,
            workflow=workflow,
            resources=_resources(),
            sampleIntervalSeconds=0.01,
        )
        assert result.status == "ok", result.error

    reopened = DataStore(store_uri)
    state = reopened.get_assay_state("RNA")
    assert state is not None
    assert state.normalized is not None
    assert state.reduction is not None
    assert state.embedding_initialization is not None
    assert state.ann_index is not None
    assert state.neighbors is not None
    assert state.connectivity_map is not None


def test_run_stage_persists_every_execution_report_and_wire_counts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from scarf.storage.budget import ResourceBudget
    from scarf.storage.execution import (
        ExecutionReport,
        WorkShape,
        clear_execution_reports,
        plan_operation,
        record_execution_report,
    )

    from profiling.stages import run_stage

    clear_execution_reports()
    plan = plan_operation(
        ResourceBudget(8 * 1024 * 1024, 4),
        WorkShape(nUnits=4, unitBytes=1024),
    )

    def open_store(*_args: Any, **_kwargs: Any) -> object:
        return object()

    def run_analysis(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        record_execution_report(
            ExecutionReport(
                plan=plan,
                unitKind="countsRowBlock",
                actualReadWorkers=1,
                actualComputeWorkers=1,
                actualWriteWorkers=1,
            )
        )
        record_execution_report(
            ExecutionReport(
                plan=plan,
                unitKind="countsTCellBand",
                actualReadWorkers=2,
                actualComputeWorkers=1,
                actualWriteWorkers=1,
                fetchSeconds=0.5,
                readerWaitSeconds=0.25,
            )
        )
        return {"consume": {"unitKind": "countsTCellBand"}}

    monkeypatch.setattr("profiling.stages._open_datastore", open_store)
    monkeypatch.setattr("profiling.stages._run_analysis", run_analysis)

    result = run_stage(
        "markHvgs",
        nRows=1000,
        storeUri=str(tmp_path / "store.zarr"),
        workflow=WorkflowParameters(),
        resources=_resources(),
        sampleIntervalSeconds=0.01,
    )

    assert result.status == "ok"
    assert result.details is not None
    reports = result.details["executionReports"]
    assert "countsRowBlock" in reports
    assert reports["countsTCellBand"][-1]["fetchSeconds"] == 0.5
    assert reports["countsTCellBand"][-1]["readerWaitSeconds"] == 0.25
    assert result.details["storeOperations"]["gets"] == 0
    assert result.details["consume"]["unitKind"] == "countsTCellBand"
