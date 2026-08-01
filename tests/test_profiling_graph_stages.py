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
                "feat_key": "hvgs",
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
        assert all(
            kwargs.get("invalidate_cache") is True
            for _method, _args, kwargs in store.calls
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


@pytest.mark.slow
def test_graph_construction_profile_stages_chain_through_persisted_assay_state(
    datastore_ephemeral: DataStore,
) -> None:
    datastore_ephemeral.auto_filter_cells(show_qc_plots=False)
    assay = datastore_ephemeral.get_assay("RNA")
    if "I__profile_hvgs" not in assay.feats.columns:
        datastore_ephemeral.mark_hvgs(
            from_assay="RNA",
            cell_key="I",
            top_n=100,
            hvg_key_name="profile_hvgs",
            show_plot=False,
        )
    store_uri = str(datastore_ephemeral.zarr_loc)
    workflow = WorkflowParameters(
        hvgKey="profile_hvgs",
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
