import inspect
import pickle
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf import PipelineExecutionError, PipelineRun
from scarf.datastore._pipeline_cluster_selection import run_cluster_selection
from scarf.datastore._pipeline_fields import (
    categorical_array_display,
    continuous_array_display,
)
from scarf.datastore._pipeline_ledger import PipelineEventEmitter
from scarf.datastore.pipeline_accessor import PipelineEvent
from scarf.storage.artifact_writer import finish_artifact, plan_artifact, start_artifact
from scarf.storage.artifacts import (
    ArtifactRef,
    artifact_group,
    fingerprint_stored_arrays,
    fingerprint_stored_strings,
    require_complete_artifact,
)
from scarf.storage.refs import ArtifactScope
from scarf.storage.selections import resolve_stored_selection_artifact
from scarf.utils.shutdown import ShutdownRequested, current_shutdown_token


def _minimal_run_options() -> dict[str, Any]:
    return {
        "filtering": False,
        "cell_cycle": False,
        "hvg_count": 50,
        "pca_dims": 3,
        "neighbors_k": 3,
        "umap": False,
        "leiden": False,
        "paris": False,
        "doublets": False,
        "markers": False,
    }


def test_pipeline_callback_baseexceptions_do_not_escape() -> None:
    events: list[PipelineEvent] = []

    def callback(event: PipelineEvent) -> None:
        events.append(event)
        raise KeyboardInterrupt("callback interruption")

    PipelineEventEmitter(callback).emit("stage_started", "input_snapshot")

    assert [event.kind for event in events] == ["stage_started"]


def test_pipeline_event_keeps_its_public_pickle_identity() -> None:
    event = PipelineEvent(kind="stage_completed", stage="pca")

    assert PipelineEvent.__module__ == "scarf.datastore.pipeline_accessor"
    assert pickle.loads(pickle.dumps(event)) == event


def _insert_nullable_cell_column(
    datastore: Any,
    name: str,
    values: np.ndarray,
    missing: np.ndarray,
) -> None:
    cell_data = datastore.zw["cellData"]
    missing_name = f"__scarf_missing__{name}"
    cell_data.create_array(name, data=np.asarray(values))
    cell_data.create_array(missing_name, data=np.asarray(missing, dtype=bool))
    cell_data[name].attrs["missing_mask"] = missing_name


class _ClusterSelectionStore:
    def __init__(self, root: Any) -> None:
        self.zw = root
        self.memoryBytes = 64 * 1024 * 1024


def test_pipeline_field_display_summaries_read_stored_chunks() -> None:
    class ChunkedArray:
        def __init__(self, values: np.ndarray, chunk_rows: int) -> None:
            self._values = values
            self.shape = values.shape
            self.dtype = values.dtype
            self.chunks = (chunk_rows, *values.shape[1:])
            self.reads: list[slice] = []

        def __getitem__(self, key):
            row_slice = key if isinstance(key, slice) else key[0]
            assert isinstance(row_slice, slice)
            assert row_slice.start is not None and row_slice.stop is not None
            assert row_slice.stop - row_slice.start <= self.chunks[0]
            self.reads.append(row_slice)
            return self._values[key]

    coordinates = ChunkedArray(
        np.asarray(
            [[3.0, 8.0], [np.nan, 4.0], [-2.0, 6.0], [1.0, np.inf]],
            dtype=np.float64,
        ),
        2,
    )
    display = continuous_array_display(coordinates, value_index=0)
    assert display["minimum"] == -2.0
    assert display["maximum"] == 3.0
    assert len(coordinates.reads) == 2

    labels = ChunkedArray(
        np.asarray([2.0, np.nan, 1.0, 2.0, 3.0]),
        2,
    )
    categorical = categorical_array_display(labels)
    assert [item["value"] for item in categorical["categories"]] == [1.0, 2.0, 3.0]
    assert categorical["missing_label"] == "NA"
    assert len(labels.reads) == 3


def _array_artifact(
    root: Any,
    *,
    kind: str,
    values: dict[str, np.ndarray],
    inputs: dict[str, Any] | None = None,
    scope: ArtifactScope = "assay",
    operation: str | None = None,
) -> ArtifactRef:
    planned = plan_artifact(
        root,
        scope=scope,
        assay="RNA" if scope == "assay" else None,
        kind=kind,
        operation=f"test_{kind}" if operation is None else operation,
        parameters={
            "payload": {
                name: np.asarray(data).tolist() for name, data in values.items()
            }
        },
        inputs={} if inputs is None else inputs,
        execution_options={},
    )
    group = start_artifact(root, planned)
    for name, data in values.items():
        group.create_array(name, data=data)
    finish_artifact(group, planned)
    return planned.ref


def _cluster_selection_lineage(
    root: Any,
    *,
    n_cells: int,
    coordinates: np.ndarray,
    harmony: bool = False,
) -> dict[str, ArtifactRef]:
    cell_ids = np.asarray([f"c{index}" for index in range(n_cells)])
    feature_ids = np.asarray(["g0", "g1"])
    cell_data = root.create_group("cellData")
    cell_data.create_array("ids", data=cell_ids)
    cell_data.create_array("I", data=np.ones(n_cells, dtype=bool))
    feature_data = root.create_group("RNA").create_group("featureData")
    feature_data.create_array("ids", data=feature_ids)
    cell_selection = resolve_stored_selection_artifact(
        root,
        table_path="cellData",
        id_column="ids",
        source_column="I",
        scope="datastore",
        kind="cell_selection",
        operation="test_cell_selection",
        parameters={},
        inputs={},
    )
    feature_fingerprint = fingerprint_stored_strings(feature_data["ids"])
    feature_payload = np.ones(len(feature_ids), dtype=bool)
    feature_selection = _rewrite_feature_universe(
        root,
        values=feature_payload,
        fingerprint=feature_fingerprint,
    )
    normalized = _array_artifact(
        root,
        kind="normalized",
        operation="run_normalization",
        values={"values": np.zeros((n_cells, 2), dtype=np.float32)},
        inputs={
            "cell_selection": cell_selection,
            "feature_selection": feature_selection,
        },
    )
    pca = _array_artifact(
        root,
        kind="reduction",
        operation="run_pca",
        values={"data": np.asarray(coordinates, dtype=np.float32)},
        inputs={"normalized": normalized},
    )
    scored = pca
    if harmony:
        scored = _array_artifact(
            root,
            kind="batch_correction",
            operation="run_harmony",
            values={"data": np.asarray(coordinates, dtype=np.float32) + 1},
            inputs={"reduction": pca},
        )
    ann_index = _array_artifact(
        root,
        kind="ann_index",
        operation="build_ann_index",
        values={"index": np.arange(n_cells, dtype=np.int32)},
        inputs={"coordinates": scored},
    )
    neighbors = _array_artifact(
        root,
        kind="neighbors",
        operation="query_neighbors",
        values={"indices": np.zeros((n_cells, 1), dtype=np.int32)},
        inputs={"ann_index": ann_index, "coordinates": scored},
    )
    connectivity = _array_artifact(
        root,
        kind="connectivity_map",
        operation="build_connectivity_map",
        values={"data": np.ones(n_cells, dtype=np.float32)},
        inputs={"neighbors": neighbors},
    )
    return {
        "cell_selection": cell_selection,
        "pca": pca,
        "coordinates": scored,
        "connectivity_map": connectivity,
        "feature_selection": feature_selection,
        "normalized": normalized,
    }


def _connectivity_from_coordinates(
    root: Any,
    *,
    coordinates: ArtifactRef,
    n_cells: int,
) -> ArtifactRef:
    ann_index = _array_artifact(
        root,
        kind="ann_index",
        operation="build_ann_index",
        values={"index": np.arange(n_cells, dtype=np.int32)},
        inputs={"coordinates": coordinates},
    )
    neighbors = _array_artifact(
        root,
        kind="neighbors",
        operation="query_neighbors",
        values={"indices": np.zeros((n_cells, 1), dtype=np.int32)},
        inputs={"ann_index": ann_index, "coordinates": coordinates},
    )
    return _array_artifact(
        root,
        kind="connectivity_map",
        operation="build_connectivity_map",
        values={"data": np.ones(n_cells, dtype=np.float32)},
        inputs={"neighbors": neighbors},
    )


def _rewrite_feature_universe(
    root: Any,
    *,
    values: np.ndarray,
    fingerprint: str,
) -> ArtifactRef:
    planned = plan_artifact(
        root,
        scope="assay",
        assay="RNA",
        kind="feature_selection",
        operation="create_all_features",
        parameters={
            "dataset_fingerprint": "test-dataset",
            "ordered_feature_ids_fingerprint": fingerprint,
        },
        inputs={},
        execution_options={},
    )
    group = start_artifact(root, planned)
    group.create_array("values", data=values)
    group.attrs["ordered_feature_ids_fingerprint"] = fingerprint
    group.attrs["payload_fingerprint"] = fingerprint_stored_arrays(group, ("values",))
    finish_artifact(group, planned)
    return planned.ref


def _cluster_labels(
    root: Any,
    *,
    values: np.ndarray,
    cell_selection: ArtifactRef,
    graph: ArtifactRef,
) -> ArtifactRef:
    return _array_artifact(
        root,
        kind="cluster_labels",
        operation="run_leiden_clustering",
        values={"values": values},
        inputs={"graph": graph, "cell_selection": cell_selection},
    )


def _select_clusters(
    store: Any,
    lineage: Mapping[str, ArtifactRef],
    candidates: tuple[tuple[str, ArtifactRef], ...],
    **kwargs: Any,
) -> tuple[ArtifactRef, str, ArtifactRef]:
    return run_cluster_selection(
        store,
        coordinates=lineage["coordinates"],
        connectivity_map=lineage["connectivity_map"],
        cell_selection=lineage["cell_selection"],
        candidates=candidates,
        **kwargs,
    )


def test_pipeline_run_has_one_small_public_invocation() -> None:
    signature = inspect.signature(PipelineRun)
    assert tuple(signature.parameters) == ("owner", "record")

    from scarf.datastore.pipeline_accessor import PipelineAccessor

    run_signature = inspect.signature(PipelineAccessor.run)
    assert tuple(run_signature.parameters) == (
        "self",
        "assay",
        "label",
        "cell_key",
        "filtering",
        "harmony_batch_columns",
        "hvg_count",
        "pca_dims",
        "neighbors_k",
        "umap",
        "leiden",
        "cell_cycle",
        "paris",
        "doublets",
        "markers",
        "snapshot_columns",
        "callback",
    )
    assert tuple(inspect.signature(PipelineAccessor.open).parameters) == (
        "self",
        "run_id",
        "label",
    )
    assert not hasattr(PipelineAccessor, "publish")


def test_pipeline_preflight_rejects_ambiguous_recipes_without_writes(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    before = tuple(run.run_id for run in datastore.pipeline.list_runs(limit=100))

    with pytest.raises(ValueError, match="exactly 'partitions'"):
        datastore.pipeline.run(
            leiden={"partitions": [0.5], "primary": 0.5, "automatic": True},
            doublets=False,
            markers=False,
        )
    with pytest.raises(
        ValueError,
        match="doublets and markers require at least one Leiden candidate",
    ):
        datastore.pipeline.run(leiden=False, paris=False)
    with pytest.raises(
        ValueError,
        match="doublets and markers require at least one Leiden candidate",
    ):
        datastore.pipeline.run(leiden=False, paris=True)
    with pytest.raises(TypeError, match="leiden must be a mapping or bool"):
        datastore.pipeline.run(leiden=None, doublets=False, markers=False)
    with pytest.raises(TypeError, match="filtering must be a mapping or bool"):
        datastore.pipeline.run(filtering=None)
    with pytest.raises(ValueError, match="reserved run fields"):
        datastore.pipeline.run(snapshot_columns=("clusters",))
    with pytest.raises(ValueError, match="reserved run fields"):
        datastore.pipeline.run(snapshot_columns=("highly_variable_features",))
    with pytest.raises(ValueError, match="duplicate resolutions"):
        datastore.pipeline.run(
            leiden={"partitions": [1, 1.0]},
            doublets=False,
            markers=False,
        )

    after = tuple(run.run_id for run in datastore.pipeline.list_runs(limit=100))
    assert after == before


@pytest.mark.parametrize(
    ("filtering", "message"),
    (
        ({"attrs": ["RNA_nCounts"], "min_p": "0.1"}, "min_p"),
        ({"attrs": ["RNA_nCounts"], "max_p": True}, "max_p"),
        ({"attrs": ["RNA_nCounts"], "n_mads": "3"}, "n_mads"),
        (
            {
                "method": "manual",
                "attrs": ["RNA_nCounts"],
                "lows": [0],
                "highs": [100],
                "keep_bounds": 1,
            },
            "keep_bounds",
        ),
    ),
)
def test_pipeline_filtering_rejects_coercible_scalar_types_without_writes(
    datastore_ephemeral,
    filtering: dict[str, object],
    message: str,
) -> None:
    datastore = datastore_ephemeral
    before = tuple(run.run_id for run in datastore.pipeline.list_runs(limit=100))

    with pytest.raises(TypeError, match=message):
        datastore.pipeline.run(filtering=filtering)

    after = tuple(run.run_id for run in datastore.pipeline.list_runs(limit=100))
    assert after == before


def test_pipeline_requested_filtering_requires_at_least_one_qc_column(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    for suffix in ("nCounts", "nFeatures", "percentMito", "percentRibo"):
        column = f"RNA_{suffix}"
        if column in datastore.cells.columns:
            datastore.cells.drop(column)
    before = tuple(run.run_id for run in datastore.pipeline.list_runs(limit=100))

    with pytest.raises(ValueError, match="pass filtering=False"):
        datastore.pipeline.run(**{**_minimal_run_options(), "filtering": True})

    after = tuple(run.run_id for run in datastore.pipeline.list_runs(limit=100))
    assert after == before


def test_minimal_pipeline_is_artifact_only_cold_openable_and_ordered(
    datastore_ephemeral,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datastore = datastore_ephemeral

    def reject_embedding_initialization(*_args, **_kwargs):
        raise AssertionError("umap=False must not build embedding initialization")

    monkeypatch.setattr(
        type(datastore),
        "build_embedding_initialization",
        reject_embedding_initialization,
    )
    cells_before = frozenset(datastore.cells.columns)
    assay = datastore.get_assay("RNA")
    features_before = frozenset(assay.feats.columns)
    assay_attrs_before = dict(assay.attrs)
    events: list[PipelineEvent] = []

    run = datastore.pipeline.run(
        label="baseline",
        callback=events.append,
        **_minimal_run_options(),
    )

    assert isinstance(run, PipelineRun)
    assert run.status == "completed"
    assert run.label == "baseline"
    assert list(run) == [
        "input_cell_selection",
        "analysis_cell_selection",
        "feature_universe",
        "highly_variable_features",
        "normalized",
        "pca",
        "ann_index",
        "neighbors",
        "connectivity_map",
    ]
    assert run["analysis_cell_selection"] == run["input_cell_selection"]
    assert all(isinstance(ref, ArtifactRef) for ref in run.values())
    assert all(datastore.inspect_artifact(ref).complete for ref in run.values())
    assert run.cells.columns == ("I", "ids", "names")
    assert run.features.columns == (
        "I",
        "ids",
        "names",
        "highly_variable_features",
    )
    assert frozenset(datastore.cells.columns) == cells_before
    assert frozenset(assay.feats.columns) == features_before
    assert dict(assay.attrs) == assay_attrs_before

    expected_stages = (
        "input_snapshot",
        "highly_variable_features",
        "normalization",
        "pca",
        "ann_index",
        "neighbors",
        "connectivity",
    )
    assert [(event.kind, event.stage) for event in events] == [
        (kind, stage)
        for stage in expected_stages
        for kind in ("stage_started", "stage_completed")
    ]

    reopened = datastore.pipeline.open(label="baseline")
    assert reopened.run_id == run.run_id
    assert datastore.pipeline.open(run_id=run.run_id).run_id == run.run_id
    with pytest.raises(ValueError, match="exactly one"):
        datastore.pipeline.open()
    with pytest.raises(ValueError, match="exactly one"):
        datastore.pipeline.open(run_id=run.run_id, label="baseline")
    assert datastore.pipeline.list_runs(status="completed")[0].run_id == run.run_id
    report = reopened.report()
    assert isinstance(report, Mapping)
    assert report["run"]["status"] == "completed"
    assert all(stage["metrics"] is not None for stage in report["stages"])
    assert all("plans" in stage for stage in report["stages"])
    stages = {stage["stage"]: stage for stage in report["stages"]}
    assert stages["embedding_initialization"]["status"] == "skipped"
    assert stages["umap"]["status"] == "skipped"
    markdown = reopened.report(format="markdown")
    assert "# Pipeline run" in markdown
    assert f"- Scarf version: `{report['run']['scarfVersion']}`" in markdown
    assert not hasattr(reopened, "plots")
    assert not hasattr(reopened, "markers")
    assert not hasattr(reopened, "publish")

    with pytest.raises(ValueError, match="already committed"):
        datastore.pipeline.run(label="baseline", **_minimal_run_options())
    assert len(datastore.pipeline.list_runs(limit=100)) == 1


def test_pipeline_failure_exposes_openable_run_and_original_cause(
    datastore_ephemeral,
    monkeypatch,
) -> None:
    datastore = datastore_ephemeral
    expected = RuntimeError("deliberate PCA failure")
    events: list[PipelineEvent] = []

    def fail_pca(self, *_args, **_kwargs):
        raise expected

    monkeypatch.setattr(type(datastore), "run_pca", fail_pca)
    with pytest.raises(PipelineExecutionError) as caught:
        datastore.pipeline.run(
            label="failed-baseline",
            callback=events.append,
            **_minimal_run_options(),
        )

    error = caught.value
    assert error.stage == "pca"
    assert error.__cause__ is expected
    failed = datastore.pipeline.open(run_id=error.run_id)
    assert failed.status == "failed"
    assert failed.label is None
    report = failed.report()
    assert report["run"]["requestedLabel"] == "failed-baseline"
    assert report["run"]["error"]["type"] == "RuntimeError"
    assert events[-1].kind == "stage_failed"
    with pytest.raises(RuntimeError, match="requires a completed run"):
        list(failed)
    with pytest.raises(KeyError, match="No completed pipeline run"):
        datastore.pipeline.open(label="failed-baseline")


def test_skipped_stage_bookkeeping_failure_commits_failed_stage_and_run(
    datastore_ephemeral,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scarf.datastore._pipeline_ledger as pipeline_module

    datastore = datastore_ephemeral
    expected = RuntimeError("deliberate skipped-stage commit failure")
    original_finish = pipeline_module.finish_pipeline_stage_record
    events: list[PipelineEvent] = []

    def fail_skipped_stage(*args, **kwargs):
        if kwargs.get("status") == "skipped":
            raise expected
        return original_finish(*args, **kwargs)

    monkeypatch.setattr(
        pipeline_module,
        "finish_pipeline_stage_record",
        fail_skipped_stage,
    )
    with pytest.raises(PipelineExecutionError) as caught:
        datastore.pipeline.run(callback=events.append, **_minimal_run_options())

    error = caught.value
    assert error.stage == "filtering"
    assert error.__cause__ is expected
    failed = datastore.pipeline.open(run_id=error.run_id)
    assert failed.status == "failed"
    filtering = next(
        stage for stage in failed.report()["stages"] if stage["stage"] == "filtering"
    )
    assert filtering["status"] == "failed"
    assert filtering["complete"] is True
    assert filtering["error"] == {
        "type": "RuntimeError",
        "message": str(expected),
    }
    assert events[-1].kind == "stage_failed"
    assert events[-1].stage == "filtering"


def test_completed_stage_bookkeeping_failure_commits_failed_stage_and_run(
    datastore_ephemeral,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scarf.datastore._pipeline_ledger as pipeline_module

    datastore = datastore_ephemeral
    expected = RuntimeError("deliberate completed-stage commit failure")
    original_finish = pipeline_module.finish_pipeline_stage_record
    failed_once = False
    events: list[PipelineEvent] = []

    def fail_completed_stage_once(*args, **kwargs):
        nonlocal failed_once
        if kwargs.get("status") == "completed" and not failed_once:
            failed_once = True
            raise expected
        return original_finish(*args, **kwargs)

    monkeypatch.setattr(
        pipeline_module,
        "finish_pipeline_stage_record",
        fail_completed_stage_once,
    )
    with pytest.raises(PipelineExecutionError) as caught:
        datastore.pipeline.run(callback=events.append, **_minimal_run_options())

    error = caught.value
    assert error.stage == "input_snapshot"
    assert error.__cause__ is expected
    failed = datastore.pipeline.open(run_id=error.run_id)
    assert failed.status == "failed"
    input_snapshot = failed.report()["stages"][0]
    assert input_snapshot["stage"] == "input_snapshot"
    assert input_snapshot["status"] == "failed"
    assert input_snapshot["complete"] is True
    assert input_snapshot["error"] == {
        "type": "RuntimeError",
        "message": str(expected),
    }
    assert [(event.kind, event.stage) for event in events] == [
        ("stage_started", "input_snapshot"),
        ("stage_failed", "input_snapshot"),
    ]
    assert events[-1].error is expected


def test_pipeline_interruption_is_durable_before_callbacks(
    datastore_ephemeral,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datastore = datastore_ephemeral
    events: list[PipelineEvent] = []

    def interrupt_pca(self, *_args, **_kwargs):
        raise KeyboardInterrupt("stop pipeline")

    monkeypatch.setattr(type(datastore), "run_pca", interrupt_pca)
    with pytest.raises(KeyboardInterrupt, match="stop pipeline"):
        datastore.pipeline.run(callback=events.append, **_minimal_run_options())

    interrupted = datastore.pipeline.list_runs(status="interrupted")
    assert len(interrupted) == 1
    report = interrupted[0].report()
    assert report["run"]["complete"] is True
    assert report["run"]["interruption"]["kind"] == "keyboard_interrupt"
    pca_stage = next(stage for stage in report["stages"] if stage["stage"] == "pca")
    assert pca_stage["status"] == "interrupted"
    assert pca_stage["complete"] is True
    assert [event.kind for event in events[-2:]] == [
        "stage_interrupted",
        "pipeline_interrupted",
    ]


def test_pending_shutdown_propagates_when_interruption_cleanup_fails(
    datastore_ephemeral,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datastore = datastore_ephemeral

    def request_shutdown(
        _accessor,
        _recipe,
        _callback,
        *,
        active_run_id,
        **_kwargs,
    ) -> None:
        active_run_id.append("a" * 64)
        token = current_shutdown_token()
        assert token is not None
        token.request(reason="test shutdown")
        token.checkpoint()

    monkeypatch.setattr(type(datastore.pipeline), "_execute_recipe", request_shutdown)
    monkeypatch.setattr(
        "scarf.datastore.pipeline_accessor.load_pipeline_run_record",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("cleanup bookkeeping failed")
        ),
    )

    with pytest.raises(ShutdownRequested, match="test shutdown"):
        datastore.pipeline.run(**_minimal_run_options())


def test_automatic_filtering_threads_one_distinct_analysis_selection(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    input_count = int(np.count_nonzero(datastore.cells.fetch_all("I")))
    options = _minimal_run_options()
    options["filtering"] = True

    run = datastore.pipeline.run(label="filtered", **options)

    assert run["analysis_cell_selection"] != run["input_cell_selection"]
    analysis_count = int(np.count_nonzero(run.cells.fetch_all("I")))
    assert 0 < analysis_count < input_count
    assert len(run.cells.fetch("ids")) == analysis_count
    normalized = datastore.inspect_artifact(run["normalized"])
    assert (
        ArtifactRef.from_dict(normalized.inputs["cell_selection"])
        == run["analysis_cell_selection"]
    )


def test_pipeline_filtering_excludes_nullable_integer_metric_rows(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    active = np.asarray(datastore.cells.fetch_all("I"), dtype=bool)
    missing = np.zeros(active.shape, dtype=bool)
    missing_index = int(np.flatnonzero(active)[0])
    missing[missing_index] = True
    _insert_nullable_cell_column(
        datastore,
        "nullable_qc",
        np.full(active.shape, 10, dtype=np.int32),
        missing,
    )
    options = _minimal_run_options()
    options["filtering"] = {
        "method": "manual",
        "attrs": ["nullable_qc"],
        "lows": [0],
        "highs": [20],
        "keep_bounds": True,
    }

    run = datastore.pipeline.run(**options)

    expected = active.copy()
    expected[missing_index] = False
    np.testing.assert_array_equal(run.cells.fetch_all("I"), expected)


def test_pipeline_filtering_rejects_nullable_active_sample_label(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    active = np.asarray(datastore.cells.fetch_all("I"), dtype=bool)
    missing = np.zeros(active.shape, dtype=bool)
    missing[int(np.flatnonzero(active)[0])] = True
    _insert_nullable_cell_column(
        datastore,
        "nullable_sample",
        np.full(active.shape, "sample-a"),
        missing,
    )
    options = _minimal_run_options()
    options["filtering"] = {
        "attrs": ["RNA_nCounts"],
        "sample_column": "nullable_sample",
        "min_cells_per_sample": 2,
    }

    with pytest.raises(PipelineExecutionError) as caught:
        datastore.pipeline.run(**options)

    assert caught.value.stage == "filtering"
    assert isinstance(caught.value.__cause__, ValueError)
    assert "contains missing labels among active cells" in str(caught.value.__cause__)


def test_default_clustering_is_selected_by_a_persisted_silhouette_decision(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    cells_before = {
        column: np.asarray(datastore.cells.fetch_all(column)).copy()
        for column in datastore.cells.columns
    }
    assay = datastore.get_assay("RNA")
    features_before = {
        column: np.asarray(assay.feats.fetch_all(column)).copy()
        for column in assay.feats.columns
    }
    options = _minimal_run_options()
    options["leiden"] = True
    run = datastore.pipeline.run(label="leiden-defaults", **options)

    partitions = ("leiden_0.5", "leiden_0.75", "leiden_1.0", "leiden_1.25")
    assert all(key in run for key in partitions)
    assert "cluster_selection" in run
    decision = artifact_group(datastore.zw, run["cluster_selection"])
    assert tuple(decision.attrs["candidateKeys"]) == partitions
    assert tuple(decision.attrs["tieOrder"]) == partitions
    assert "paris" not in decision.attrs["candidateKeys"]
    selected_key = decision.attrs["selectedKey"]
    assert selected_key in partitions
    assert run["clusters"] == run[selected_key]
    scores = np.asarray(decision["scores"][:], dtype=float)
    assert scores.shape == (4,)
    assert np.isfinite(scores).any()
    assert decision["sample_indices"].shape[0] <= 10_000
    assert dict(decision.attrs["sampleDefinition"])["sampleStrategy"] == (
        "sharedClusterQuota"
    )
    assert dict(decision.attrs["sampleDefinition"])["minClusterQuota"] == 2
    status = require_complete_artifact(datastore.zw, run["cluster_selection"])
    assert status.inputs is not None
    assert ArtifactRef.from_dict(status.inputs["coordinates"]) == run["pca"]
    assert (
        ArtifactRef.from_dict(status.inputs["connectivityMap"])
        == run["connectivity_map"]
    )
    assert run.cells.columns[-5:] == (*partitions, "clusters")
    assert "RNA_clusters" not in datastore.cells.columns
    assert "hvgs" not in datastore.get_assay("RNA").feats.columns

    assert set(datastore.cells.columns) == set(cells_before)
    assert set(assay.feats.columns) == set(features_before)
    for column, values in cells_before.items():
        np.testing.assert_equal(datastore.cells.fetch_all(column), values)
    for column, values in features_before.items():
        np.testing.assert_equal(assay.feats.fetch_all(column), values)


def test_custom_leiden_uses_only_the_requested_candidate_set(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    options = _minimal_run_options()
    options["leiden"] = {"partitions": [0.4, 0.8]}
    run = datastore.pipeline.run(**options)

    assert "leiden_0.4" in run
    assert "leiden_0.8" in run
    assert run["clusters"] in {run["leiden_0.4"], run["leiden_0.8"]}
    decision = artifact_group(datastore.zw, run["cluster_selection"])
    assert tuple(decision.attrs["candidateKeys"]) == (
        "leiden_0.4",
        "leiden_0.8",
    )
    assert "leiden_1.0" not in run


def test_cluster_selection_records_invalid_candidates_ties_and_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    store = _ClusterSelectionStore(root)
    coordinates = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [5.0, 5.0],
            [5.1, 5.0],
            [5.0, 5.1],
        ],
        dtype=np.float32,
    )
    lineage = _cluster_selection_lineage(root, n_cells=6, coordinates=coordinates)
    selection = lineage["cell_selection"]
    graph = lineage["connectivity_map"]
    invalid = _cluster_labels(
        root,
        values=np.zeros(6, dtype=np.int32),
        cell_selection=selection,
        graph=graph,
    )
    first = _cluster_labels(
        root,
        values=np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32),
        cell_selection=selection,
        graph=graph,
    )
    second = _cluster_labels(
        root,
        values=np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int32),
        cell_selection=selection,
        graph=graph,
    )
    monkeypatch.setattr(
        "sklearn.metrics.silhouette_score",
        lambda *_args, **_kwargs: 0.25,
    )
    candidates = (("invalid", invalid), ("first", first), ("second", second))

    decision, selected_key, selected = _select_clusters(store, lineage, candidates)
    group = artifact_group(root, decision)
    assert selected_key == "first"
    assert selected == first
    assert group.attrs["invalidReasons"][0] == (
        "sample contains fewer than two clusters"
    )
    assert tuple(group.attrs["tieOrder"]) == ("invalid", "first", "second")
    np.testing.assert_allclose(group["scores"][:], [np.nan, 0.25, 0.25])

    reused, reused_key, reused_selected = _select_clusters(store, lineage, candidates)
    assert reused == decision
    assert reused_key == selected_key
    assert reused_selected == selected
    status = require_complete_artifact(root, decision)
    assert status.inputs is not None
    assert ArtifactRef.from_dict(status.inputs["coordinates"]) == lineage["pca"]
    assert ArtifactRef.from_dict(status.inputs["connectivityMap"]) == graph
    assert dict(group.attrs["sampleDefinition"]) == {
        "seed": 4466,
        "populationSize": 6,
        "sampleSize": 6,
        "maxSampleSize": 10_000,
        "sampleStrategy": "sharedClusterQuota",
        "minClusterQuota": 2,
    }

    group.attrs["selectedKey"] = "second"
    replacement, replacement_key, _replacement_selected = _select_clusters(
        store,
        lineage,
        candidates,
    )
    assert replacement != decision
    assert replacement_key == "first"

    replacement_group = artifact_group(root, replacement)
    replacement_group["sample_indices"][0] = 1
    resampled, resampled_key, _resampled_selected = _select_clusters(
        store,
        lineage,
        candidates,
    )
    assert resampled != replacement
    assert resampled_key == "first"

    resampled_group = artifact_group(root, resampled)
    resampled_group["scores"][1] = np.nan
    rescored, rescored_key, _rescored_selected = _select_clusters(
        store,
        lineage,
        candidates,
    )
    assert rescored != resampled
    assert rescored_key == "first"

    rescored_group = artifact_group(root, rescored)
    rescored_group.attrs["candidateRefs"] = [
        second.to_dict(),
        first.to_dict(),
        invalid.to_dict(),
    ]
    rereferenced, rereferenced_key, _rereferenced_selected = _select_clusters(
        store,
        lineage,
        candidates,
    )
    assert rereferenced != rescored
    assert rereferenced_key == "first"

    def replace_after_attr_corruption(
        current: ArtifactRef,
        attribute: str,
        value: object,
    ) -> ArtifactRef:
        artifact_group(root, current).attrs[attribute] = value
        replacement, replacement_key, _replacement_selected = _select_clusters(
            store,
            lineage,
            candidates,
        )
        assert replacement != current
        assert replacement_key == "first"
        return replacement

    rereferenced = replace_after_attr_corruption(
        rereferenced,
        "candidateKeys",
        ["wrong", "first", "second"],
    )
    rereferenced = replace_after_attr_corruption(
        rereferenced,
        "tieOrder",
        ["second", "first", "invalid"],
    )
    rereferenced = replace_after_attr_corruption(
        rereferenced,
        "invalidReasons",
        ["too short"],
    )
    replace_after_attr_corruption(
        rereferenced,
        "sampleDefinition",
        {
            "seed": 4466,
            "populationSize": 6,
            "sampleSize": 6,
            "maxSampleSize": 10_000,
        },
    )


def test_cluster_selection_adapter_rejects_detached_lineage() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    store = _ClusterSelectionStore(root)
    coordinates = np.arange(8, dtype=np.float32).reshape(4, 2)
    lineage = _cluster_selection_lineage(root, n_cells=4, coordinates=coordinates)
    selection = lineage["cell_selection"]
    graph = lineage["connectivity_map"]
    labels = _cluster_labels(
        root,
        values=np.asarray([0, 0, 1, 1], dtype=np.int32),
        cell_selection=selection,
        graph=graph,
    )
    candidates = (("clusters", labels),)

    detached_pca = _array_artifact(
        root,
        kind="reduction",
        values={"data": coordinates},
        inputs={
            "normalized": _array_artifact(
                root,
                kind="normalized",
                values={"values": np.zeros((4, 2), dtype=np.float32)},
                inputs={
                    "cell_selection": selection,
                    "feature_selection": lineage["feature_selection"],
                },
            )
        },
    )
    with pytest.raises(ValueError, match="must reference a PCA artifact"):
        run_cluster_selection(
            store,
            coordinates=detached_pca,
            connectivity_map=graph,
            cell_selection=selection,
            candidates=candidates,
        )

    cell_data = root["cellData"]
    cell_data.create_array(
        "I2",
        data=np.asarray([True, True, True, False]),
    )
    other_selection = resolve_stored_selection_artifact(
        root,
        table_path="cellData",
        id_column="ids",
        source_column="I2",
        scope="datastore",
        kind="cell_selection",
        operation="test_other_cell_selection",
        parameters={},
        inputs={},
    )
    detached_labels = _cluster_labels(
        root,
        values=np.asarray([0, 0, 1, 1], dtype=np.int32),
        cell_selection=other_selection,
        graph=graph,
    )
    with pytest.raises(ValueError, match="does not use the requested cell selection"):
        _select_clusters(store, lineage, (("clusters", detached_labels),))

    with pytest.raises(ValueError, match="coordinates do not use the requested"):
        run_cluster_selection(
            store,
            coordinates=lineage["pca"],
            connectivity_map=graph,
            cell_selection=other_selection,
            candidates=candidates,
        )

    other_graph = _array_artifact(
        root,
        kind="connectivity_map",
        values={"data": np.ones(4, dtype=np.float32)},
        inputs={"neighbors": lineage["connectivity_map"]},
    )
    foreign_labels = _cluster_labels(
        root,
        values=np.asarray([0, 0, 1, 1], dtype=np.int32),
        cell_selection=selection,
        graph=other_graph,
    )
    with pytest.raises(
        ValueError,
        match="was not partitioned from the requested connectivity map",
    ):
        _select_clusters(store, lineage, (("clusters", foreign_labels),))

    wrong_scope = _array_artifact(
        root,
        scope="datastore",
        kind="cluster_labels",
        values={"values": np.asarray([0, 0, 1, 1], dtype=np.int32)},
        inputs={"graph": graph, "cell_selection": selection},
    )
    with pytest.raises(ValueError, match="assay-scoped Leiden cluster-label artifact"):
        _select_clusters(store, lineage, (("clusters", wrong_scope),))

    with pytest.raises(TypeError, match="coordinates must be an ArtifactRef"):
        run_cluster_selection(
            store,
            coordinates="pca",  # type: ignore[arg-type]
            connectivity_map=graph,
            cell_selection=selection,
            candidates=candidates,
        )
    with pytest.raises(TypeError, match="connectivity_map must be an ArtifactRef"):
        run_cluster_selection(
            store,
            coordinates=lineage["pca"],
            connectivity_map="graph",  # type: ignore[arg-type]
            cell_selection=selection,
            candidates=candidates,
        )
    with pytest.raises(TypeError, match="cell_selection must be an ArtifactRef"):
        run_cluster_selection(
            store,
            coordinates=lineage["pca"],
            connectivity_map=graph,
            cell_selection="I",  # type: ignore[arg-type]
            candidates=candidates,
        )
    with pytest.raises(TypeError, match="seed must be an integer"):
        _select_clusters(store, lineage, candidates, seed=1.5)
    with pytest.raises(ValueError, match="min_cluster_quota must be at least 1"):
        _select_clusters(store, lineage, candidates, min_cluster_quota=0)

    embedding = _array_artifact(
        root,
        kind="embedding",
        values={"data": coordinates},
        inputs={"coordinates": lineage["pca"]},
    )
    with pytest.raises(
        ValueError,
        match="assay-scoped PCA reduction or Harmony",
    ):
        run_cluster_selection(
            store,
            coordinates=embedding,
            connectivity_map=graph,
            cell_selection=selection,
            candidates=candidates,
        )

    harmony_wrong_operation = _array_artifact(
        root,
        kind="batch_correction",
        operation="test_not_harmony",
        values={"data": coordinates},
        inputs={"reduction": lineage["pca"]},
    )
    with pytest.raises(ValueError, match="must reference a Harmony artifact"):
        run_cluster_selection(
            store,
            coordinates=harmony_wrong_operation,
            connectivity_map=graph,
            cell_selection=selection,
            candidates=candidates,
        )

    with pytest.raises(ValueError, match="Native graph source must be"):
        run_cluster_selection(
            store,
            coordinates=lineage["pca"],
            connectivity_map=lineage["pca"],
            cell_selection=selection,
            candidates=candidates,
        )

    missing_data = _array_artifact(
        root,
        kind="reduction",
        operation="run_pca",
        values={"payload": coordinates},
        inputs={"normalized": lineage["normalized"]},
    )
    missing_graph = _connectivity_from_coordinates(
        root,
        coordinates=missing_data,
        n_cells=4,
    )
    with pytest.raises(ValueError, match="missing its data array"):
        run_cluster_selection(
            store,
            coordinates=missing_data,
            connectivity_map=missing_graph,
            cell_selection=selection,
            candidates=candidates,
        )

    flat = _array_artifact(
        root,
        kind="reduction",
        operation="run_pca",
        values={"data": np.arange(4, dtype=np.float32)},
        inputs={"normalized": lineage["normalized"]},
    )
    flat_graph = _connectivity_from_coordinates(root, coordinates=flat, n_cells=4)
    with pytest.raises(ValueError, match="non-empty two-dimensional array"):
        run_cluster_selection(
            store,
            coordinates=flat,
            connectivity_map=flat_graph,
            cell_selection=selection,
            candidates=candidates,
        )

    with pytest.raises(ValueError, match="at least one candidate"):
        _select_clusters(store, lineage, ())
    with pytest.raises(TypeError, match=r"\(key, ArtifactRef\) tuples"):
        _select_clusters(store, lineage, ("clusters", labels))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="non-empty strings"):
        _select_clusters(store, lineage, (("", labels),))
    with pytest.raises(TypeError, match="must be an ArtifactRef"):
        _select_clusters(store, lineage, (("clusters", "labels"),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be unique"):
        _select_clusters(store, lineage, (("clusters", labels), ("clusters", labels)))

    short_labels = _cluster_labels(
        root,
        values=np.asarray([0, 0, 1], dtype=np.int32),
        cell_selection=selection,
        graph=graph,
    )
    with pytest.raises(ValueError, match="does not align with coordinate rows"):
        _select_clusters(store, lineage, (("clusters", short_labels),))

    missing_selection = _array_artifact(
        root,
        kind="cluster_labels",
        operation="run_leiden_clustering",
        values={"values": np.asarray([0, 0, 1, 1], dtype=np.int32)},
        inputs={"graph": graph},
    )
    with pytest.raises(ValueError, match="has no 'cell_selection' artifact input"):
        _select_clusters(store, lineage, (("clusters", missing_selection),))

    malformed_selection = _array_artifact(
        root,
        kind="cluster_labels",
        operation="run_leiden_clustering",
        values={"values": np.asarray([0, 0, 1, 1], dtype=np.int32)},
        inputs={
            "graph": graph,
            "cell_selection": {"type": "artifact", "scope": "nope"},
        },
    )
    with pytest.raises(ValueError, match="malformed 'cell_selection' artifact input"):
        _select_clusters(store, lineage, (("clusters", malformed_selection),))

    paris_cut = _array_artifact(
        root,
        kind="cluster_cut",
        operation="cut_paris_tree",
        values={"labels": np.asarray([0, 0, 1, 1], dtype=np.int32)},
        inputs={"connectivity_map": graph, "cell_selection": selection},
    )
    with pytest.raises(ValueError, match="assay-scoped Leiden cluster-label artifact"):
        _select_clusters(store, lineage, (("paris", paris_cut),))

    imported_labels = _array_artifact(
        root,
        kind="cluster_labels",
        operation="import_cluster_labels",
        values={"values": np.asarray([0, 0, 1, 1], dtype=np.int32)},
        inputs={"graph": graph, "cell_selection": selection},
    )
    with pytest.raises(ValueError, match="must reference a Leiden clustering artifact"):
        _select_clusters(store, lineage, (("imported", imported_labels),))


def test_empty_filtering_mapping_uses_automatic_defaults(
    datastore_ephemeral,
) -> None:
    from scarf.datastore._pipeline_recipe import resolve_pipeline_recipe

    recipe = resolve_pipeline_recipe(
        datastore_ephemeral,
        assay=None,
        label=None,
        cell_key="I",
        filtering={},
        harmony_batch_columns=None,
        hvg_count=50,
        pca_dims=3,
        neighbors_k=3,
        umap=False,
        leiden=False,
        cell_cycle=False,
        paris=False,
        doublets=False,
        markers=False,
        snapshot_columns=(),
    )

    assert recipe.filtering["enabled"] is True
    assert recipe.filtering["method"] == "auto"
    assert recipe.leiden_partitions == ()


def test_cluster_selection_scores_harmony_coordinates_not_pca(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    store = _ClusterSelectionStore(root)
    pca_values = np.arange(8, dtype=np.float32).reshape(4, 2)
    lineage = _cluster_selection_lineage(
        root,
        n_cells=4,
        coordinates=pca_values,
        harmony=True,
    )
    labels = _cluster_labels(
        root,
        values=np.asarray([0, 0, 1, 1], dtype=np.int32),
        cell_selection=lineage["cell_selection"],
        graph=lineage["connectivity_map"],
    )
    seen: list[np.ndarray] = []

    def capture_score(sampled_coordinates: np.ndarray, *_args, **_kwargs) -> float:
        seen.append(np.asarray(sampled_coordinates, dtype=np.float64).copy())
        return 0.4

    monkeypatch.setattr("sklearn.metrics.silhouette_score", capture_score)
    decision, selected_key, selected = _select_clusters(
        store,
        lineage,
        (("clusters", labels),),
    )
    assert selected_key == "clusters"
    assert selected == labels
    status = require_complete_artifact(root, decision)
    assert status.inputs is not None
    assert ArtifactRef.from_dict(status.inputs["coordinates"]) == lineage["coordinates"]
    assert lineage["coordinates"] != lineage["pca"]
    assert seen
    harmony_values = np.asarray(
        artifact_group(root, lineage["coordinates"])["data"][:],
        dtype=np.float64,
    )
    np.testing.assert_allclose(seen[0], harmony_values)

    with pytest.raises(
        ValueError,
        match="was not built from the scored coordinates",
    ):
        run_cluster_selection(
            store,
            coordinates=lineage["pca"],
            connectivity_map=lineage["connectivity_map"],
            cell_selection=lineage["cell_selection"],
            candidates=(("clusters", labels),),
        )


def test_rich_pipeline_views_plots_and_markers_remain_frozen_after_live_i_drift(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    initial = np.asarray(datastore.cells.fetch_all("I"), dtype=bool)
    selected = np.flatnonzero(initial)[:250]
    limited = np.zeros(datastore.cells.N, dtype=bool)
    limited[selected] = True
    datastore.cells.insert("I", limited, overwrite=True, force=True)

    assay = datastore.get_assay("RNA")
    cell_values_before = {
        column: np.asarray(datastore.cells.fetch_all(column)).copy()
        for column in datastore.cells.columns
    }
    feature_values_before = {
        column: np.asarray(assay.feats.fetch_all(column)).copy()
        for column in assay.feats.columns
    }
    cell_attrs_before = dict(datastore.cells.locations["primary"].attrs)
    feature_attrs_before = dict(assay.feats.locations["primary"].attrs)
    assay_attrs_before = dict(assay.attrs)

    run = datastore.pipeline.run(
        label="rich-frozen",
        filtering=False,
        hvg_count=50,
        pca_dims=3,
        neighbors_k=3,
    )

    assert list(run) == [
        "input_cell_selection",
        "analysis_cell_selection",
        "feature_universe",
        "cell_cycle",
        "highly_variable_features",
        "normalized",
        "pca",
        "ann_index",
        "neighbors",
        "connectivity_map",
        "embedding_initialization",
        "umap",
        "leiden_0.5",
        "leiden_0.75",
        "leiden_1.0",
        "leiden_1.25",
        "paris",
        "cluster_selection",
        "clusters",
        "doublets",
        "markers",
    ]
    decision = artifact_group(datastore.zw, run["cluster_selection"])
    assert run["clusters"] == run[decision.attrs["selectedKey"]]
    assert decision.attrs["selectedKey"].startswith("leiden_")
    assert "paris" not in decision.attrs["candidateKeys"]
    assert "paris" in run
    assert run["clusters"] != run["paris"]
    assert set(datastore.cells.columns) == set(cell_values_before)
    assert set(assay.feats.columns) == set(feature_values_before)
    for column, values in cell_values_before.items():
        np.testing.assert_equal(datastore.cells.fetch_all(column), values)
    for column, values in feature_values_before.items():
        np.testing.assert_equal(assay.feats.fetch_all(column), values)
    assert dict(datastore.cells.locations["primary"].attrs) == cell_attrs_before
    assert dict(assay.feats.locations["primary"].attrs) == feature_attrs_before
    assert dict(assay.attrs) == assay_attrs_before

    datastore.cells.insert(
        "I",
        np.zeros(datastore.cells.N, dtype=bool),
        overwrite=True,
        force=True,
    )
    reopened = datastore.pipeline.open(label="rich-frozen")
    frame = reopened.cells.to_pandas_dataframe(
        ["umap_1", "umap_2", "clusters", "doublet_score"]
    )
    assert len(frame) == len(selected)
    assert np.isfinite(frame[["umap_1", "umap_2"]].to_numpy()).all()

    plot = datastore.plots.embedding(
        run=reopened,
        layout="umap",
        color_by="clusters",
        show=False,
        rasterize_threshold=0,
    )
    assert plot.figure is not None
    plot.close()
    raster = datastore.plots.embedding_raster(
        run=reopened,
        layout="umap",
        color_by="doublet_score",
        pixels=32,
        block_rows=32,
        show=False,
    )
    assert "artifact_layout" in raster.provenance.notes
    assert "frozen_run_fields" in raster.provenance.notes
    assert raster.provenance.n_cells == len(selected)
    raster.close()
    markers = datastore.get_markers(
        marker=reopened["markers"],
        min_score=0,
        min_frac_exp=0,
    )
    assert {
        "group_id",
        "feature_name",
        "feature_index",
        "score",
        "frac_exp",
    }.issubset(markers.columns)


def test_harmony_doublets_record_an_uncorrected_internal_graph_only(
    datastore_ephemeral,
    monkeypatch,
) -> None:
    datastore = datastore_ephemeral
    batch = np.where(np.arange(datastore.cells.N) % 2, "b", "a")
    datastore.cells.insert("pipeline_batch", batch)
    captured: dict[str, Any] = {}
    original_doublets = type(datastore)._run_doublet_detection_artifact

    def capture_doublet_inputs(self, **kwargs):
        captured.update(kwargs)
        return original_doublets(self, **kwargs)

    monkeypatch.setattr(
        type(datastore),
        "_run_doublet_detection_artifact",
        capture_doublet_inputs,
    )
    run = datastore.pipeline.run(
        label="harmony-doublets",
        filtering=False,
        harmony_batch_columns=("pipeline_batch",),
        hvg_count=50,
        pca_dims=3,
        neighbors_k=3,
        umap=False,
        leiden={"partitions": [1.0]},
        cell_cycle=False,
        paris=False,
        doublets=True,
        markers=False,
    )

    assert run["clusters"] == captured["clusters"]
    assert captured["connectivity"] != run["connectivity_map"]
    decision = artifact_group(datastore.zw, run["cluster_selection"])
    status = require_complete_artifact(datastore.zw, run["cluster_selection"])
    assert status.inputs is not None
    assert ArtifactRef.from_dict(status.inputs["coordinates"]) == run["harmony"]
    assert (
        ArtifactRef.from_dict(status.inputs["connectivityMap"])
        == run["connectivity_map"]
    )
    assert tuple(decision.attrs["candidateKeys"]) == ("leiden_1.0",)
    assert not any(key.startswith("uncorrected_") for key in run)
    stage = next(
        item for item in run.report()["stages"] if item["stage"] == "doublet_graph"
    )
    stage_outputs = {
        item["outputKey"]: ArtifactRef.from_dict(item["artifact"])
        for item in stage["outputs"]
    }
    assert set(stage_outputs) == {
        "uncorrected_ann_index",
        "uncorrected_neighbors",
        "uncorrected_connectivity_map",
    }
    assert captured["connectivity"] == stage_outputs["uncorrected_connectivity_map"]


def test_paris_only_pipeline_keeps_paris_as_a_diagnostic_without_clusters(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    run = datastore.pipeline.run(
        **{
            **_minimal_run_options(),
            "leiden": False,
            "paris": True,
        }
    )

    assert "paris" in run
    assert "clusters" not in run
    assert "cluster_selection" not in run
    assert "leiden_1.0" not in run
    assert list(run).count("paris") == 1
