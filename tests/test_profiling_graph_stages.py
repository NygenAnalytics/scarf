import time
from pathlib import Path
from types import SimpleNamespace
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


def _ref(kind: str, value: str, *, scope: str = "assay") -> ArtifactRef:
    return ArtifactRef(
        scope=scope,
        assay="RNA" if scope == "assay" else None,
        kind=kind,
        artifact_id=value * 64,
    )


_CELL_SELECTION = _ref("cell_selection", "b", scope="datastore")
_NORMALIZED = _ref("normalized", "c")
_REDUCTION = _ref("reduction", "d")
_INITIALIZATION = _ref("embedding_initialization", "e")
_ANN_INDEX = _ref("ann_index", "f")
_NEIGHBORS = _ref("neighbors", "1")
_CONNECTIVITY = _ref("connectivity_map", "2")
_EMBEDDING = _ref("embedding", "3")
_CLUSTERS = _ref("cluster_labels", "4")

_ARTIFACTS_BY_KIND = {
    ref.kind: ref
    for ref in (
        _NORMALIZED,
        _REDUCTION,
        _INITIALIZATION,
        _ANN_INDEX,
        _NEIGHBORS,
        _CONNECTIVITY,
    )
}

_RESULT_BY_METHOD = {
    "run_normalization": _NORMALIZED,
    "run_pca": _REDUCTION,
    "build_embedding_initialization": _INITIALIZATION,
    "build_ann_index": _ANN_INDEX,
    "query_neighbors": _NEIGHBORS,
    "build_connectivity_map": _CONNECTIVITY,
    "run_umap": _EMBEDDING,
}

_INPUTS_BY_STAGE: dict[StageName, dict[str, ArtifactRef]] = {
    "runNormalization": {
        "cells": _CELL_SELECTION,
        "features": _FEATURE_REF,
    },
    "runPca": {"normalized": _NORMALIZED},
    "buildEmbeddingInitialization": {"coordinates": _REDUCTION},
    "buildAnnIndex": {"coordinates": _REDUCTION},
    "queryNeighbors": {"ann_index": _ANN_INDEX},
    "buildConnectivityMap": {"neighbors": _NEIGHBORS},
    "runUmap": {
        "graph": _CONNECTIVITY,
        "initialization": _INITIALIZATION,
    },
    "findMarkers": {"clusters": _CLUSTERS},
}


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

    def snapshot_cell_selection(self, _cell_key: str) -> ArtifactRef:
        return _CELL_SELECTION

    def list_artifacts(self, *, kind: str, **_kwargs: Any) -> list[ArtifactRef]:
        return [_ARTIFACTS_BY_KIND[kind]]

    @staticmethod
    def _get_assay(_assay: str) -> object:
        return object()

    @staticmethod
    def get_assay(_assay: str) -> object:
        return SimpleNamespace(feats=SimpleNamespace(N=10))

    @staticmethod
    def select_all_features(*, from_assay: str | None = None) -> ArtifactRef:
        assert from_assay == "RNA"
        return _FEATURE_REF

    def __getattr__(self, name: str) -> Any:
        def record(*args: Any, **kwargs: Any) -> ArtifactRef:
            self.calls.append((name, args, kwargs))
            return _RESULT_BY_METHOD.get(name, _FEATURE_REF)

        return record


@pytest.mark.parametrize(
    ("stage", "method", "expected_args", "expected_kwargs"),
    [
        (
            "runNormalization",
            "run_normalization",
            (_CELL_SELECTION, _FEATURE_REF),
            {
                "invalidate_cache": False,
            },
        ),
        (
            "runPca",
            "run_pca",
            (_NORMALIZED,),
            {
                "dims": 37,
                "local_cache": False,
                "show_elbow_plot": False,
                "invalidate_cache": False,
            },
        ),
        (
            "buildEmbeddingInitialization",
            "build_embedding_initialization",
            (_REDUCTION,),
            {
                "n_centroids": 321,
                "rand_state": 99,
                "kmeans_sampling": 0.2,
                "kmeans_batch_size": 5_000,
                "invalidate_cache": False,
            },
        ),
        (
            "buildAnnIndex",
            "build_ann_index",
            (_REDUCTION,),
            {
                "ann_efc": 51,
                "ann_ef": 51,
                "ann_m": 55,
                "ann_parallel": True,
                "rand_state": 99,
                "invalidate_cache": False,
            },
        ),
        (
            "queryNeighbors",
            "query_neighbors",
            (_ANN_INDEX,),
            {
                "k": 17,
                "invalidate_cache": False,
            },
        ),
        (
            "buildConnectivityMap",
            "build_connectivity_map",
            (_NEIGHBORS,),
            {
                "invalidate_cache": False,
            },
        ),
    ],
)
def test_graph_construction_profile_stage_uses_explicit_refs_and_parameters(
    stage: StageName,
    method: str,
    expected_args: tuple[ArtifactRef, ...],
    expected_kwargs: dict[str, Any],
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
        inputRefs=_INPUTS_BY_STAGE[stage],
    )

    assert store.calls == [(method, expected_args, expected_kwargs)]
    assert store.resolved_features == []


def test_profile_normalization_consumes_explicit_feature_ref() -> None:
    store = _RecordingStore()

    _run_analysis(
        "runNormalization",
        store,
        WorkflowParameters(),
        _resources(),
        inputRefs={"cells": _CELL_SELECTION, "features": _FEATURE_REF},
    )

    assert store.resolved_features == []
    assert store.calls == [
        (
            "run_normalization",
            (_CELL_SELECTION, _FEATURE_REF),
            {
                "invalidate_cache": False,
            },
        )
    ]


def test_profile_marker_search_uses_explicit_cluster_and_feature_refs() -> None:
    store = _RecordingStore()

    _run_analysis(
        "findMarkers",
        store,
        WorkflowParameters(),
        _resources(),
        inputRefs={"clusters": _CLUSTERS},
    )

    assert store.resolved_features == []
    marker_calls = [call for call in store.calls if call[0] == "run_marker_search"]
    assert len(marker_calls) == 1
    assert marker_calls[0][1] == (_CLUSTERS,)
    assert marker_calls[0][2]["features"] == _FEATURE_REF
    assert "group_key" not in marker_calls[0][2]


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
            inputRefs=_INPUTS_BY_STAGE.get(stage),
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


def test_run_stage_session_carries_exact_ref_to_normalization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = object()
    seen_refs: list[dict[str, ArtifactRef]] = []

    monkeypatch.setattr(
        "profiling.stages._open_datastore",
        lambda *_args, **_kwargs: store,
    )

    def run_analysis(
        stage: StageName,
        *_args: Any,
        inputRefs: dict[str, ArtifactRef] | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any] | None:
        seen_refs.append(dict(inputRefs or {}))
        if stage == "filterCells":
            return {"artifact": _CELL_SELECTION.to_dict()}
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

    filtered = run_stage("filterCells", **common)
    marked = run_stage("markHvgs", **common)
    normalized = run_stage("runNormalization", **common)

    assert filtered.status == "ok"
    assert marked.status == "ok"
    assert normalized.status == "ok"
    assert seen_refs == [
        {},
        {"cells": _CELL_SELECTION},
        {"cells": _CELL_SELECTION, "features": _FEATURE_REF},
    ]
    assert session["artifactRefs"]["filterCells"] == _CELL_SELECTION
    assert session["artifactRefs"]["markHvgs"] == _FEATURE_REF


@pytest.mark.slow
def test_graph_construction_profile_stages_chain_through_explicit_artifacts(
    datastore_ephemeral: DataStore,
) -> None:
    cell_selection = datastore_ephemeral.auto_filter_cells()
    features = datastore_ephemeral.select_hvgs(
        cell_selection,
        from_assay="RNA",
        top_n=100,
        show_plot=False,
    )
    store_uri = str(datastore_ephemeral.zarr_loc)
    workflow = WorkflowParameters(
        dims=5,
        nCentroids=20,
        k=3,
        graphLocalCache=False,
    )

    session: dict[str, Any] = {
        "artifactRefs": {
            "filterCells": cell_selection,
            "markHvgs": features,
        }
    }
    for stage in GRAPH_CONSTRUCTION_STAGE_ORDER:
        result = run_stage(
            stage,
            nRows=datastore_ephemeral.cells.N,
            storeUri=store_uri,
            workflow=workflow,
            resources=_resources(),
            sampleIntervalSeconds=0.01,
            session=session,
        )
        assert result.status == "ok", result.error

    reopened = DataStore(store_uri)
    for kind in (
        "normalized",
        "reduction",
        "embedding_initialization",
        "ann_index",
        "neighbors",
        "connectivity_map",
    ):
        refs = reopened.list_artifacts(
            kind=kind,
            from_assay="RNA",
            scope="assay",
            complete_only=True,
        )
        assert refs
        assert reopened.inspect_artifact(refs[-1]).complete


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
