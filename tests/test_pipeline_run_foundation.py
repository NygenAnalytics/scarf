import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import Any

import numpy as np
import pytest
import zarr
from zarr.storage import FsspecStore, LoggingStore, MemoryStore, ZipStore

import scarf.storage.pipeline_runs as pipeline_run_storage
from scarf.datastore.pipeline_accessor import PipelineAccessor
from scarf.datastore.pipeline_run import (
    PipelineAxisView,
    PipelineExecutionError,
    PipelineRun,
    list_pipeline_runs,
    open_pipeline_run,
)
from scarf.storage.artifact_writer import (
    finish_artifact,
    plan_artifact,
    start_artifact,
)
from scarf.storage.artifacts import (
    artifact_group,
    fingerprint_array,
    fingerprint_strings,
)
from scarf.storage.errors import ArtifactResolutionError
from scarf.storage.feature_selection import (
    _feature_selection_plan,
    _write_feature_selection,
)
from scarf.storage.pipeline_runs import (
    PipelineFieldDescriptor,
    PipelineInterruptionRecord,
    PipelineOutputRecord,
    PipelineRunRecord,
    PipelineStageMetrics,
    PipelineStageOutputRecord,
    complete_pipeline_run_record,
    create_pipeline_run_record,
    fail_pipeline_run_record,
    finish_pipeline_stage_record,
    interrupt_pipeline_run_record,
    list_pipeline_run_records,
    load_pipeline_stage_record,
    load_pipeline_stage_records,
    load_pipeline_run_record,
    open_pipeline_run_record,
    start_pipeline_stage_record,
)
from scarf.storage.refs import ArtifactRef
from scarf.storage.selections import (
    resolve_selection_artifact,
    snapshot_run_metadata,
)
from scarf.storage.types import as_zarr_array


class _Table:
    def __init__(self, group: zarr.Group) -> None:
        self._group = group
        self.N = int(as_zarr_array(group["ids"], name="ids").shape[0])

    def fetch_all(self, column: str) -> np.ndarray:
        return np.asarray(as_zarr_array(self._group[column], name=column)[:])

    def _get_array(self, column: str) -> zarr.Array:
        return as_zarr_array(self._group[column], name=column)


class _Assay:
    def __init__(self, feats: _Table) -> None:
        self.feats = feats


class _Owner:
    def __init__(self, root: zarr.Group) -> None:
        self.zw = root
        self.cells = _Table(root["cellData"])
        self._assay = _Assay(_Table(root["RNA/featureData"]))

    def get_assay(self, assay_name: str) -> _Assay:
        if assay_name != "RNA":
            raise KeyError(assay_name)
        return self._assay


def _metrics() -> PipelineStageMetrics:
    return PipelineStageMetrics(
        wall_seconds=0.01,
        rss_baseline_bytes=100,
        rss_peak_bytes=120,
        rss_incremental_peak_bytes=20,
        sample_interval_seconds=0.1,
        sample_count=2,
        sampling_error_count=0,
        rss_unavailable_reason=None,
    )


def _root(*, store: Any | None = None) -> zarr.Group:
    root = zarr.open_group(
        store=MemoryStore() if store is None else store,
        mode="w",
    )
    cell_data = root.create_group("cellData")
    cell_data.create_array("ids", data=np.asarray(["c1", "c2", "c3", "c4"]))
    cell_data.create_array(
        "names",
        data=np.asarray(["a", "b", "c", "d"], dtype="U2"),
    )
    cell_data.create_array("I", data=np.asarray([True, True, True, True]))
    feature_data = root.create_group("RNA/featureData")
    feature_data.create_array("ids", data=np.asarray(["g1", "g2", "g3"]))
    feature_data.create_array(
        "names",
        data=np.asarray(["A", "B", "C"], dtype="U2"),
    )
    feature_data.create_array("I", data=np.asarray([True, True, False]))
    return root


def _complete_labeled_run_in_process(
    store_path: str,
    run_id: str,
    artifact_value: dict[str, Any],
    barrier: Any,
    result_queue: Any,
) -> None:
    root = zarr.open_group(store=store_path, mode="r+")
    artifact = ArtifactRef.from_dict(artifact_value)
    original_check = pipeline_run_storage.ensure_pipeline_label_available

    def synchronized_check(
        root: zarr.Group,
        label: str,
        *,
        exclude_run_id: str | None = None,
    ) -> None:
        original_check(root, label, exclude_run_id=exclude_run_id)
        if exclude_run_id is not None:
            barrier.wait(timeout=10)

    pipeline_run_storage.ensure_pipeline_label_available = synchronized_check
    try:
        completed = complete_pipeline_run_record(
            root,
            run_id=run_id,
            outputs=(PipelineOutputRecord("selection", artifact),),
            fields=(),
        )
    except Exception as exc:
        result_queue.put(("error", type(exc).__name__, str(exc)))
    else:
        result_queue.put(("completed", completed.run_id, ""))


def _artifact(
    root: zarr.Group,
    *,
    kind: str,
    values: dict[str, np.ndarray],
    inputs: dict[str, Any],
    scope: str = "datastore",
    assay: str | None = None,
) -> ArtifactRef:
    planned = plan_artifact(
        root,
        scope=scope,
        assay=assay,
        kind=kind,
        operation=f"test_{kind}",
        parameters={},
        inputs=inputs,
        execution_options={},
        invalidate_cache=True,
    )
    group = start_artifact(root, planned)
    for name, data in values.items():
        group.create_array(name, data=data)
    finish_artifact(group, planned)
    return planned.ref


def _ready_labeled_runs(
    root: zarr.Group,
    label: str,
    *,
    count: int = 2,
) -> tuple[ArtifactRef, tuple[PipelineRunRecord, ...]]:
    artifact = _artifact(
        root,
        kind="cell_selection",
        values={"values": np.asarray([True, True, True, True])},
        inputs={
            "ordered_row_ids_fingerprint": fingerprint_strings(
                np.asarray(["c1", "c2", "c3", "c4"])
            )
        },
    )
    runs = tuple(
        create_pipeline_run_record(
            root,
            recipe="basic_rna_analysis",
            requested_label=label,
            assay="RNA",
            config={},
            stage_order=("input_snapshot",),
            scarf_version="1.0.0",
            started_at_ns=100 + index,
        )
        for index in range(count)
    )
    for index, record in enumerate(runs):
        start_pipeline_stage_record(
            root,
            run_id=record.run_id,
            ordinal=0,
            stage="input_snapshot",
            started_at_ns=110 + index,
        )
        finish_pipeline_stage_record(
            root,
            run_id=record.run_id,
            ordinal=0,
            status="completed",
            outputs=(PipelineStageOutputRecord("selection", artifact, True),),
            metrics=_metrics(),
            finished_at_ns=120 + index,
        )
    return artifact, runs


def _fixture_feature_selection(
    root: zarr.Group,
    values: np.ndarray,
    *,
    operation: str,
    parameters: dict[str, Any],
    inputs: dict[str, Any],
) -> ArtifactRef:
    mask = np.asarray(values, dtype=bool)
    feature_ids = np.asarray(root["RNA/featureData/ids"][:]).astype(str)
    row_fingerprint = fingerprint_strings(feature_ids)
    payload_fingerprint = fingerprint_array(mask)
    planned = _feature_selection_plan(
        root,
        assay="RNA",
        n_features=len(mask),
        ordered_feature_ids_fingerprint=row_fingerprint,
        operation=operation,
        parameters=parameters,
        inputs=inputs,
        execution_options={},
        expected_payload_fingerprint=payload_fingerprint,
        invalidate_cache=True,
    )
    _write_feature_selection(
        root,
        planned,
        ordered_feature_ids_fingerprint=row_fingerprint,
        payload={"values": mask},
    )
    return planned.ref


def _completed_run(root: zarr.Group) -> PipelineRun:
    cell_ids = np.asarray(["c1", "c2", "c3", "c4"])
    feature_ids = np.asarray(["g1", "g2", "g3"])
    feature_fingerprint = fingerprint_strings(feature_ids)
    cell_selection = resolve_selection_artifact(
        root,
        scope="datastore",
        kind="cell_selection",
        values=np.asarray([True, False, True, False]),
        row_ids=cell_ids,
        operation="test_pipeline_selection",
        parameters={},
        inputs={},
        source_column="I",
    )
    cell_data = root["cellData"]
    cell_data["names"][:] = np.asarray(["a0", "b0", "c0", "d0"])
    cell_data.create_array("batch", data=np.asarray(["x", "y", "x", "y"]))
    cell_data.create_array(
        "nullable_score",
        data=np.asarray([10, 20, 30, 40], dtype=np.int32),
    )
    cell_data.create_array(
        "__scarf_missing__nullable_score",
        data=np.asarray([False, False, True, False]),
    )
    cell_data["nullable_score"].attrs["missing_mask"] = (
        "__scarf_missing__nullable_score"
    )
    cell_snapshot = snapshot_run_metadata(
        root,
        table_path="cellData",
        id_column="ids",
        columns=("names", "batch", "nullable_score"),
        axis="cell",
    )
    cell_data["names"][:] = np.asarray(["a", "b", "c", "d"])
    feature_data = root["RNA/featureData"]
    feature_data["names"][:] = np.asarray(["A0", "B0", "C0"])
    feature_snapshot = snapshot_run_metadata(
        root,
        table_path="RNA/featureData",
        id_column="ids",
        columns=("names",),
        axis="feature",
        assay="RNA",
    )
    feature_data["names"][:] = np.asarray(["A", "B", "C"])
    embedding = _artifact(
        root,
        scope="assay",
        assay="RNA",
        kind="embedding",
        values={"values": np.asarray([[1.0, 10.0], [3.0, 30.0]], dtype=np.float32)},
        inputs={"cell_selection": cell_selection.to_dict()},
    )
    clusters = _artifact(
        root,
        scope="assay",
        assay="RNA",
        kind="cluster_labels",
        values={"values": np.asarray([0, 2], dtype=np.int32)},
        inputs={"cell_selection": cell_selection.to_dict()},
    )
    feature_universe = _fixture_feature_selection(
        root,
        np.ones(3, dtype=bool),
        operation="create_all_features",
        parameters={
            "dataset_fingerprint": "fixture",
            "ordered_feature_ids_fingerprint": feature_fingerprint,
        },
        inputs={},
    )
    hvg_values = np.asarray([True, False, True])
    hvgs = _fixture_feature_selection(
        root,
        hvg_values,
        operation="set_feature_selection",
        parameters={"values_fingerprint": fingerprint_array(hvg_values)},
        inputs={"all_features": feature_universe},
    )
    run = create_pipeline_run_record(
        root,
        recipe="basic_rna_analysis",
        requested_label="baseline",
        assay="RNA",
        config={"pcaDims": 21},
        stage_order=("input_snapshot",),
        scarf_version="1.0.0",
        started_at_ns=100,
    )
    start_pipeline_stage_record(
        root,
        run_id=run.run_id,
        ordinal=0,
        stage="input_snapshot",
        started_at_ns=110,
    )
    finish_pipeline_stage_record(
        root,
        run_id=run.run_id,
        ordinal=0,
        status="completed",
        outputs=(
            PipelineStageOutputRecord(
                output_key="analysis_cell_selection",
                artifact=cell_selection,
                reused=False,
            ),
        ),
        metrics=_metrics(),
        finished_at_ns=120,
    )
    descriptors = (
        PipelineFieldDescriptor(
            key="I",
            axis="cells",
            artifact=cell_selection,
            source_value="values",
            value_index=None,
            dtype=np.dtype(bool).str,
            fill=False,
            missing_mask=None,
            display=None,
        ),
        PipelineFieldDescriptor(
            key="ids",
            axis="cells",
            artifact=cell_snapshot,
            source_value="ids",
            value_index=None,
            dtype=np.asarray(cell_ids).dtype.str,
            fill="",
            missing_mask=None,
            display=None,
        ),
        PipelineFieldDescriptor(
            key="names",
            axis="cells",
            artifact=cell_snapshot,
            source_value="names",
            value_index=None,
            dtype=np.asarray(["a0", "b0", "c0", "d0"]).dtype.str,
            fill="",
            missing_mask=None,
            display=None,
        ),
        PipelineFieldDescriptor(
            key="batch",
            axis="cells",
            artifact=cell_snapshot,
            source_value="batch",
            value_index=None,
            dtype=np.asarray(["x", "y", "x", "y"]).dtype.str,
            fill="",
            missing_mask=None,
            display=None,
        ),
        PipelineFieldDescriptor(
            key="nullable_score",
            axis="cells",
            artifact=cell_snapshot,
            source_value="nullable_score",
            value_index=None,
            dtype=np.dtype(np.int32).str,
            fill=-1,
            missing_mask="__scarf_missing__nullable_score",
            display={
                "kind": "continuous",
                "colormap": "viridis",
                "minimum": 10.0,
                "maximum": 40.0,
                "scale": "linear",
            },
        ),
        PipelineFieldDescriptor(
            key="umap_1",
            axis="cells",
            artifact=embedding,
            source_value="values",
            value_index=0,
            dtype=np.dtype(np.float32).str,
            fill="nan",
            missing_mask=None,
            display={"kind": "continuous"},
        ),
        PipelineFieldDescriptor(
            key="clusters",
            axis="cells",
            artifact=clusters,
            source_value="values",
            value_index=None,
            dtype=np.dtype(np.int32).str,
            fill=-1,
            missing_mask=None,
            display={"kind": "categorical"},
        ),
        PipelineFieldDescriptor(
            key="I",
            axis="features",
            artifact=feature_universe,
            source_value="values",
            value_index=None,
            dtype=np.dtype(bool).str,
            fill=False,
            missing_mask=None,
            display=None,
        ),
        PipelineFieldDescriptor(
            key="ids",
            axis="features",
            artifact=feature_snapshot,
            source_value="ids",
            value_index=None,
            dtype=np.asarray(feature_ids).dtype.str,
            fill="",
            missing_mask=None,
            display=None,
        ),
        PipelineFieldDescriptor(
            key="names",
            axis="features",
            artifact=feature_snapshot,
            source_value="names",
            value_index=None,
            dtype=np.asarray(["A0", "B0", "C0"]).dtype.str,
            fill="",
            missing_mask=None,
            display=None,
        ),
        PipelineFieldDescriptor(
            key="highly_variable_features",
            axis="features",
            artifact=hvgs,
            source_value="values",
            value_index=None,
            dtype=np.dtype(bool).str,
            fill=False,
            missing_mask=None,
            display=None,
        ),
    )
    completed = complete_pipeline_run_record(
        root,
        run_id=run.run_id,
        outputs=(
            PipelineOutputRecord("analysis_cell_selection", cell_selection),
            PipelineOutputRecord("feature_universe", feature_universe),
            PipelineOutputRecord("umap", embedding),
            PipelineOutputRecord("clusters", clusters),
            PipelineOutputRecord("highly_variable_features", hvgs),
        ),
        fields=descriptors,
        finished_at_ns=130,
    )
    return PipelineRun(_Owner(root), completed)


def test_strict_run_lifecycle_and_scan_based_label_lookup() -> None:
    root = _root()
    completed = _completed_run(root)

    stored = load_pipeline_run_record(root, completed.run_id)
    assert set(root[f"pipeline/runs/{completed.run_id}"].attrs) == {
        "runId",
        "recipe",
        "requestedLabel",
        "label",
        "assay",
        "startedAtNs",
        "finishedAtNs",
        "status",
        "complete",
        "scarfVersion",
        "config",
        "stageOrder",
        "outputs",
        "fields",
        "error",
        "interruption",
    }
    assert stored.successfully_completed
    assert open_pipeline_run_record(root, label="baseline") == stored
    assert list_pipeline_run_records(root)[0] == stored

    owner = _Owner(root)
    assert open_pipeline_run(owner, label="baseline").run_id == completed.run_id
    assert list_pipeline_runs(owner)[0].run_id == completed.run_id


def test_run_catalog_skips_torn_and_corrupt_children() -> None:
    root = _root()
    completed = _completed_run(root)
    runs = root["pipeline/runs"]
    torn_id = "a" * 64
    runs.create_group(torn_id)
    corrupt_id = "b" * 64
    runs.create_group(corrupt_id).create_group("stages")
    runs.create_group("not-a-run")

    assert list_pipeline_run_records(root) == (
        load_pipeline_run_record(root, completed.run_id),
    )
    assert open_pipeline_run_record(root, label="baseline").run_id == completed.run_id
    assert list_pipeline_runs(_Owner(root))[0].run_id == completed.run_id
    with pytest.raises(ValueError, match="has no stages group"):
        open_pipeline_run_record(root, run_id=torn_id)

    fresh = create_pipeline_run_record(
        root,
        recipe="basic_rna_analysis",
        requested_label="fresh",
        assay="RNA",
        config={},
        stage_order=("input_snapshot",),
        scarf_version="1.0.0",
    )
    assert fresh.status == "running"


def test_stage_record_paths_load_the_run_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root()
    run = create_pipeline_run_record(
        root,
        recipe="basic_rna_analysis",
        requested_label=None,
        assay="RNA",
        config={},
        stage_order=("first", "second"),
        scarf_version="1.0.0",
        started_at_ns=100,
    )
    original_load = pipeline_run_storage.load_pipeline_run_record
    load_count = 0

    def counted_load(
        root: zarr.Group,
        run_id: str,
    ) -> pipeline_run_storage.PipelineRunRecord:
        nonlocal load_count
        load_count += 1
        return original_load(root, run_id)

    monkeypatch.setattr(
        pipeline_run_storage,
        "load_pipeline_run_record",
        counted_load,
    )

    start_pipeline_stage_record(
        root,
        run_id=run.run_id,
        ordinal=0,
        stage="first",
        started_at_ns=110,
    )
    assert load_count == 1

    load_count = 0
    finish_pipeline_stage_record(
        root,
        run_id=run.run_id,
        ordinal=0,
        status="completed",
        metrics=_metrics(),
        finished_at_ns=120,
    )
    assert load_count == 1

    load_count = 0
    assert load_pipeline_stage_record(root, run.run_id, 0).stage == "first"
    assert load_count == 1

    load_count = 0
    assert tuple(
        stage.stage for stage in load_pipeline_stage_records(root, run.run_id)
    ) == ("first",)
    assert load_count == 1

    load_count = 0
    start_pipeline_stage_record(
        root,
        run_id=run.run_id,
        ordinal=1,
        stage="second",
        started_at_ns=130,
    )
    assert load_count == 1


def test_labeled_run_rejects_unsupported_backend_before_record_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root()
    monkeypatch.setattr(
        pipeline_run_storage,
        "_store_supports_atomic_label_claims",
        lambda _store: False,
    )

    with pytest.raises(RuntimeError, match="atomic set_if_not_exists"):
        create_pipeline_run_record(
            root,
            recipe="basic_rna_analysis",
            requested_label="unsupported",
            assay="RNA",
            config={},
            stage_order=("input_snapshot",),
            scarf_version="1.0.0",
        )

    assert "pipeline" not in root
    unlabeled = create_pipeline_run_record(
        root,
        recipe="basic_rna_analysis",
        requested_label=None,
        assay="RNA",
        config={},
        stage_order=("input_snapshot",),
        scarf_version="1.0.0",
    )
    assert unlabeled.status == "running"


@pytest.mark.parametrize("container_kind", ("group", "nonempty", "wrong_dtype"))
def test_labeled_run_rejects_incompatible_claim_container_before_record_creation(
    container_kind: str,
) -> None:
    root = _root()
    runs = root.create_group("pipeline/runs")
    if container_kind == "group":
        runs.create_group(".label-claims")
    elif container_kind == "nonempty":
        runs.create_array(".label-claims", shape=(1,), dtype="uint8")
    else:
        runs.create_array(".label-claims", shape=(0,), dtype="int8")
    children_before = set(runs.keys())

    with pytest.raises(ValueError, match="claim container is incompatible"):
        create_pipeline_run_record(
            root,
            recipe="basic_rna_analysis",
            requested_label="blocked",
            assay="RNA",
            config={},
            stage_order=("input_snapshot",),
            scarf_version="1.0.0",
        )

    assert set(runs.keys()) == children_before


def test_raw_label_claim_reader_rejects_incomplete_and_cyclic_chains() -> None:
    root = _root()
    label = "corrupt-claim-chain"
    head = pipeline_run_storage._pipeline_label_claim_path(root, label, "head")

    with pytest.raises(ValueError, match="incomplete durable claim"):
        pipeline_run_storage._read_pipeline_label_claim(head, label)

    run_id = "a" * 64
    payload = pipeline_run_storage._pipeline_label_claim_bytes(label, run_id)
    pipeline_run_storage.sync(head.set(payload))
    successor = pipeline_run_storage._pipeline_label_claim_path(root, label, run_id)
    pipeline_run_storage.sync(successor.set(payload))

    with pytest.raises(ValueError, match="cyclic durable claim"):
        pipeline_run_storage._pipeline_label_claim_owner(root, label)


def test_label_claim_preflight_advances_past_a_missing_owner() -> None:
    root = _root()
    label = "missing-owner"
    owner = create_pipeline_run_record(
        root,
        recipe="basic_rna_analysis",
        requested_label=label,
        assay="RNA",
        config={},
        stage_order=("input_snapshot",),
        scarf_version="1.0.0",
    )
    pipeline_run_storage._claim_pipeline_label(root, label, owner.run_id)
    del root[f"pipeline/runs/{owner.run_id}"]

    replacement = create_pipeline_run_record(
        root,
        recipe="basic_rna_analysis",
        requested_label=label,
        assay="RNA",
        config={},
        stage_order=("input_snapshot",),
        scarf_version="1.0.0",
    )

    assert replacement.status == "running"


def test_terminal_label_conflict_fails_one_concurrent_thread(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_path = str(tmp_path / "threaded-labels.zarr")
    root = _root(store=store_path)
    artifact, (first, second) = _ready_labeled_runs(root, "same")
    scan_barrier = Barrier(2)
    original_check = pipeline_run_storage.ensure_pipeline_label_available

    def synchronized_check(
        root: zarr.Group,
        label: str,
        *,
        exclude_run_id: str | None = None,
    ) -> None:
        original_check(root, label, exclude_run_id=exclude_run_id)
        if exclude_run_id is not None:
            scan_barrier.wait(timeout=10)

    monkeypatch.setattr(
        pipeline_run_storage,
        "ensure_pipeline_label_available",
        synchronized_check,
    )

    def complete(run_id: str) -> PipelineRunRecord:
        worker_root = zarr.open_group(store=store_path, mode="r+")
        return complete_pipeline_run_record(
            worker_root,
            run_id=run_id,
            outputs=(PipelineOutputRecord("selection", artifact),),
            fields=(),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(complete, run.run_id) for run in (first, second)]
    completed = []
    failures = []
    for future in futures:
        try:
            completed.append(future.result())
        except ValueError as exc:
            failures.append(exc)

    assert len(completed) == 1
    assert len(failures) == 1
    assert "Pipeline label 'same'" in str(failures[0])
    loser_id = second.run_id if completed[0].run_id == first.run_id else first.run_id
    failed = load_pipeline_run_record(root, loser_id)
    assert failed.status == "failed"
    assert failed.complete
    assert failed.label is None
    assert failed.error is not None
    assert open_pipeline_run_record(root, label="same") == completed[0]


def test_terminal_label_claim_is_process_safe(tmp_path: Any) -> None:
    store_path = str(tmp_path / "process-labels.zarr")
    root = _root(store=store_path)
    artifact, runs = _ready_labeled_runs(root, "process-safe")
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_complete_labeled_run_in_process,
            args=(
                store_path,
                run.run_id,
                artifact.to_dict(),
                barrier,
                result_queue,
            ),
        )
        for run in runs
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=20)
        assert all(not process.is_alive() for process in processes)
        assert all(process.exitcode == 0 for process in processes)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    results = [result_queue.get(timeout=5) for _ in processes]
    assert [result[0] for result in results].count("completed") == 1
    assert [result[0] for result in results].count("error") == 1
    assert {load_pipeline_run_record(root, run.run_id).status for run in runs} == {
        "completed",
        "failed",
    }
    owner = open_pipeline_run_record(root, label="process-safe")
    assert owner.run_id == next(
        result[1] for result in results if result[0] == "completed"
    )


@pytest.mark.parametrize("stale_kind", ("failed", "interrupted", "orphaned"))
def test_terminal_label_claim_advances_past_non_owner(
    stale_kind: str,
) -> None:
    root = _root()
    label = f"retry-{stale_kind}"
    artifact, (first, second) = _ready_labeled_runs(root, label)
    pipeline_run_storage._claim_pipeline_label(root, label, first.run_id)
    with pytest.raises(KeyError, match="No completed pipeline run"):
        open_pipeline_run_record(root, label=label)

    if stale_kind == "failed":
        fail_pipeline_run_record(root, run_id=first.run_id, error=RuntimeError("x"))
    elif stale_kind == "interrupted":
        interrupt_pipeline_run_record(
            root,
            run_id=first.run_id,
            interruption=PipelineInterruptionRecord(
                kind="cancelled",
                message="cancelled",
                requested_at_ns=130,
            ),
        )
    else:
        del root[f"pipeline/runs/{first.run_id}"]

    completed = complete_pipeline_run_record(
        root,
        run_id=second.run_id,
        outputs=(PipelineOutputRecord("selection", artifact),),
        fields=(),
    )
    assert open_pipeline_run_record(root, label=label) == completed


def test_running_label_claim_requires_explicit_exact_owner_abandonment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root()
    label = "abandoned-finalizer"
    artifact, (first,) = _ready_labeled_runs(root, label, count=1)
    pipeline_run_storage._claim_pipeline_label(root, label, first.run_id)
    run_ids_before = set(root["pipeline/runs"].group_keys())

    with pytest.raises(RuntimeError, match="pipeline.abandon_label_claim"):
        create_pipeline_run_record(
            root,
            recipe="basic_rna_analysis",
            requested_label=label,
            assay="RNA",
            config={},
            stage_order=("input_snapshot",),
            scarf_version="1.0.0",
        )
    assert set(root["pipeline/runs"].group_keys()) == run_ids_before

    owner = _Owner(root)
    accessor = PipelineAccessor(owner)
    owner.zarr_mode = "r"
    with pytest.raises(
        PermissionError,
        match=r"zarr_mode='r\+'",
    ):
        accessor.abandon_label_claim(
            label=label,
            run_id=first.run_id,
            reason="read-only recovery must fail",
        )
    owner.zarr_mode = "r+"
    with monkeypatch.context() as unsupported_backend:
        unsupported_backend.setattr(
            pipeline_run_storage,
            "_store_supports_atomic_label_claims",
            lambda _store: False,
        )
        with pytest.raises(RuntimeError, match="atomic set_if_not_exists"):
            accessor.abandon_label_claim(
                label=label,
                run_id=first.run_id,
                reason="unsupported backend must fail",
            )

    run_group = root[f"pipeline/runs/{first.run_id}"]
    run_group.attrs["runId"] = "invalid"
    with pytest.raises(ValueError, match="invalid claim-owner record"):
        pipeline_run_storage.ensure_pipeline_label_claimable(root, label)
    run_group.attrs["runId"] = first.run_id
    run_group.attrs["requestedLabel"] = "different-label"
    with pytest.raises(ValueError, match="claim from an incompatible run"):
        pipeline_run_storage.ensure_pipeline_label_claimable(root, label)
    with pytest.raises(ValueError, match="claim from an incompatible run"):
        accessor.abandon_label_claim(
            label=label,
            run_id=first.run_id,
            reason="mismatched owner must fail",
        )
    run_group.attrs["requestedLabel"] = label

    with pytest.raises(ValueError, match="is owned by run"):
        accessor.abandon_label_claim(
            label=label,
            run_id="f" * 64,
            reason="wrong owner",
        )
    with pytest.raises(KeyError, match="has no durable claim"):
        accessor.abandon_label_claim(
            label="wrong-label",
            run_id=first.run_id,
            reason="wrong label",
        )
    with pytest.raises(TypeError, match="reason must be a non-empty string"):
        accessor.abandon_label_claim(
            label=label,
            run_id=first.run_id,
            reason="",
        )

    abandoned = accessor.abandon_label_claim(
        label=label,
        run_id=first.run_id,
        reason="worker was terminated and confirmed stopped",
    )
    assert abandoned.status == "interrupted"
    interruption = abandoned.report()["run"]["interruption"]
    assert interruption["kind"] == "abandoned_label_claim"
    assert interruption["message"] == "worker was terminated and confirmed stopped"

    second = create_pipeline_run_record(
        root,
        recipe="basic_rna_analysis",
        requested_label=label,
        assay="RNA",
        config={},
        stage_order=("input_snapshot",),
        scarf_version="1.0.0",
    )
    start_pipeline_stage_record(
        root,
        run_id=second.run_id,
        ordinal=0,
        stage="input_snapshot",
    )
    finish_pipeline_stage_record(
        root,
        run_id=second.run_id,
        ordinal=0,
        status="completed",
        outputs=(PipelineStageOutputRecord("selection", artifact, True),),
        metrics=_metrics(),
    )
    completed = complete_pipeline_run_record(
        root,
        run_id=second.run_id,
        outputs=(PipelineOutputRecord("selection", artifact),),
        fields=(),
    )
    assert open_pipeline_run_record(root, label=label) == completed


def test_label_claim_preflight_rechecks_an_owner_completed_during_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root()
    label = "completed-during-preflight"
    artifact, (owner,) = _ready_labeled_runs(root, label, count=1)
    pipeline_run_storage._claim_pipeline_label(root, label, owner.run_id)
    original_check = pipeline_run_storage.ensure_pipeline_label_available

    def complete_after_scan(
        root: zarr.Group,
        label: str,
        *,
        exclude_run_id: str | None = None,
    ) -> None:
        original_check(root, label, exclude_run_id=exclude_run_id)
        if exclude_run_id is None:
            complete_pipeline_run_record(
                root,
                run_id=owner.run_id,
                outputs=(PipelineOutputRecord("selection", artifact),),
                fields=(),
            )

    monkeypatch.setattr(
        pipeline_run_storage,
        "ensure_pipeline_label_available",
        complete_after_scan,
    )

    with pytest.raises(ValueError, match="already committed"):
        pipeline_run_storage.ensure_pipeline_label_claimable(root, label)


def test_torn_terminal_label_claim_can_be_explicitly_abandoned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root()
    label = "torn-terminal-finalizer"
    artifact, (first,) = _ready_labeled_runs(root, label, count=1)
    original_write = pipeline_run_storage._write_terminal_attrs

    def write_without_final_complete(
        group: zarr.Group,
        value: dict[str, Any],
    ) -> None:
        payload = dict(value)
        payload["complete"] = False
        group.attrs.update(payload)

    monkeypatch.setattr(
        pipeline_run_storage,
        "_write_terminal_attrs",
        write_without_final_complete,
    )
    complete_pipeline_run_record(
        root,
        run_id=first.run_id,
        outputs=(PipelineOutputRecord("selection", artifact),),
        fields=(),
    )
    torn = load_pipeline_run_record(root, first.run_id)
    assert torn.status == "completed"
    assert torn.complete is False

    monkeypatch.setattr(
        pipeline_run_storage,
        "_write_terminal_attrs",
        original_write,
    )
    owner = _Owner(root)
    owner.zarr_mode = "r+"
    recovered = PipelineAccessor(owner).abandon_label_claim(
        label=label,
        run_id=first.run_id,
        reason="worker stopped during the terminal commit",
    )
    assert recovered.status == "interrupted"
    assert load_pipeline_run_record(root, first.run_id).complete is True

    retry = create_pipeline_run_record(
        root,
        recipe="basic_rna_analysis",
        requested_label=label,
        assay="RNA",
        config={},
        stage_order=("input_snapshot",),
        scarf_version="1.0.0",
    )
    assert retry.status == "running"


@pytest.mark.parametrize("terminal_status", ("completed", "failed"))
def test_label_claim_abandonment_refuses_terminal_owner(
    terminal_status: str,
) -> None:
    root = _root()
    label = f"terminal-{terminal_status}"
    artifact, (record,) = _ready_labeled_runs(root, label, count=1)
    pipeline_run_storage._claim_pipeline_label(root, label, record.run_id)
    if terminal_status == "completed":
        complete_pipeline_run_record(
            root,
            run_id=record.run_id,
            outputs=(PipelineOutputRecord("selection", artifact),),
            fields=(),
        )
    else:
        fail_pipeline_run_record(root, run_id=record.run_id, error=RuntimeError("x"))

    owner = _Owner(root)
    owner.zarr_mode = "r+"
    with pytest.raises(ValueError, match="not held by an unfinished run"):
        PipelineAccessor(owner).abandon_label_claim(
            label=label,
            run_id=record.run_id,
            reason="must refuse terminal owners",
        )


def test_terminal_label_claim_backend_failure_is_durable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root()
    artifact, (record,) = _ready_labeled_runs(root, "unsupported", count=1)

    def reject_claim(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("atomic conditional creation is unavailable")

    monkeypatch.setattr(
        pipeline_run_storage,
        "_claim_pipeline_label",
        reject_claim,
    )
    with pytest.raises(RuntimeError, match="atomic conditional creation"):
        complete_pipeline_run_record(
            root,
            run_id=record.run_id,
            outputs=(PipelineOutputRecord("selection", artifact),),
            fields=(),
        )

    failed = load_pipeline_run_record(root, record.run_id)
    assert failed.status == "failed"
    assert failed.complete
    assert failed.error is not None
    assert failed.error.type == "PipelineLabelClaimUnavailable"


def test_atomic_label_claim_backend_detection(tmp_path: Any) -> None:
    memory_store = MemoryStore()
    safe_wrapper = LoggingStore(memory_store)
    fsspec_store = FsspecStore.from_url(f"memory://scarf-label-claims/{tmp_path.name}")
    unsafe_wrapper = LoggingStore(fsspec_store)
    zip_store = ZipStore(tmp_path / "label-claims.zip", mode="w")
    try:
        assert pipeline_run_storage._store_supports_atomic_label_claims(memory_store)
        assert pipeline_run_storage._store_supports_atomic_label_claims(safe_wrapper)
        assert not pipeline_run_storage._store_supports_atomic_label_claims(
            fsspec_store
        )
        assert not pipeline_run_storage._store_supports_atomic_label_claims(
            unsafe_wrapper
        )
        assert not pipeline_run_storage._store_supports_atomic_label_claims(zip_store)
    finally:
        safe_wrapper.close()
        unsafe_wrapper.close()
        # zarr 3.2.0 ZipStore.close() requires _lock, which exists only after open.
        if getattr(zip_store, "_is_open", False):
            zip_store.close()


def test_run_view_reads_frozen_fields_and_ignores_live_i_drift() -> None:
    root = _root()
    run = _completed_run(root)
    cells = run.cells

    np.testing.assert_array_equal(cells.fetch_all("I"), [True, False, True, False])
    np.testing.assert_array_equal(cells.fetch("names"), ["a0", "c0"])
    np.testing.assert_array_equal(cells.fetch("clusters"), [0, 2])
    np.testing.assert_allclose(
        cells.fetch_all("umap_1"),
        [1.0, np.nan, 3.0, np.nan],
        equal_nan=True,
    )
    frame = cells.to_pandas_dataframe(["names", "batch", "clusters"])
    assert frame.to_dict(orient="list") == {
        "names": ["a0", "c0"],
        "batch": ["x", "x"],
        "clusters": [0, 2],
    }

    root["cellData/I"][:] = False
    np.testing.assert_array_equal(run.cells.fetch("ids"), ["c1", "c3"])
    np.testing.assert_array_equal(
        run.features.fetch("highly_variable_features"),
        [True, False, True],
    )


@pytest.mark.parametrize("required_key", ("I", "ids", "names"))
def test_run_view_requires_persisted_axis_identity_descriptors(
    required_key: str,
) -> None:
    root = _root()
    run = _completed_run(root)
    run_group = root[f"pipeline/runs/{run.run_id}"]
    fields = list(run_group.attrs["fields"])
    run_group.attrs["fields"] = [
        field
        for field in fields
        if not (field["axis"] == "cells" and field["key"] == required_key)
    ]

    reopened = open_pipeline_run(_Owner(root), run_id=run.run_id)
    with pytest.raises(ArtifactResolutionError) as caught:
        _ = reopened.cells
    assert caught.value.code == "pipeline_view_required_fields_missing"


def test_run_view_rejects_tampered_selection_values() -> None:
    root = _root()
    run = _completed_run(root)
    selection = run["analysis_cell_selection"]
    artifact_group(root, selection)["values"][0] = False

    with pytest.raises(ArtifactResolutionError) as caught:
        _ = run.cells
    assert caught.value.code == "selection_values_changed"


def test_run_view_rejects_tampered_feature_universe() -> None:
    root = _root()
    run = _completed_run(root)
    feature_universe = run["feature_universe"]
    artifact_group(root, feature_universe)["values"][0] = False

    with pytest.raises(ArtifactResolutionError) as caught:
        _ = run.features
    assert caught.value.code == "corrupt_payload"


@pytest.mark.parametrize("axis", ("cells", "features"))
def test_run_view_rejects_tampered_metadata_snapshot(axis: str) -> None:
    root = _root()
    run = _completed_run(root)
    record = load_pipeline_run_record(root, run.run_id)
    names = next(
        field for field in record.fields if field.axis == axis and field.key == "names"
    )
    artifact_group(root, names.artifact)["names"][0] = "changed"

    with pytest.raises(ArtifactResolutionError) as caught:
        _ = getattr(run, axis)
    assert caught.value.code == "snapshot_values_changed"


def test_run_view_rejects_compact_field_from_another_selection() -> None:
    root = _root()
    run = _completed_run(root)
    other_selection = resolve_selection_artifact(
        root,
        scope="datastore",
        kind="cell_selection",
        values=np.asarray([False, True, False, True]),
        row_ids=np.asarray(["c1", "c2", "c3", "c4"]),
        operation="test_other_pipeline_selection",
        parameters={},
        inputs={},
        source_column="I",
    )
    clusters = run["clusters"]
    group = artifact_group(root, clusters)
    provenance = dict(group.attrs["provenance"])
    inputs = dict(provenance["inputs"])
    inputs["cell_selection"] = other_selection.to_dict()
    provenance["inputs"] = inputs
    group.attrs["provenance"] = provenance

    with pytest.raises(ArtifactResolutionError) as caught:
        _ = run.cells
    assert caught.value.code == "pipeline_field_selection_mismatch"


def test_run_view_rejects_full_axis_cell_field_from_another_selection() -> None:
    root = _root()
    run = _completed_run(root)
    other_selection = resolve_selection_artifact(
        root,
        scope="datastore",
        kind="cell_selection",
        values=np.asarray([False, True, False, True]),
        row_ids=np.asarray(["c1", "c2", "c3", "c4"]),
        operation="test_other_full_axis_selection",
        parameters={},
        inputs={},
        source_column="I",
    )
    foreign = _artifact(
        root,
        scope="assay",
        assay="RNA",
        kind="cluster_labels",
        values={"values": np.asarray([0, 1, 2, 3], dtype=np.int32)},
        inputs={"cell_selection": other_selection.to_dict()},
    )
    run_group = root[f"pipeline/runs/{run.run_id}"]
    fields = list(run_group.attrs["fields"])
    run_group.attrs["fields"] = [
        {**field, "artifact": foreign.to_dict()}
        if field["axis"] == "cells" and field["key"] == "clusters"
        else field
        for field in fields
    ]

    reopened = open_pipeline_run(_Owner(root), run_id=run.run_id)
    with pytest.raises(ArtifactResolutionError) as caught:
        _ = reopened.cells
    assert caught.value.code == "pipeline_field_selection_mismatch"


def test_run_view_validates_feature_result_payload_integrity() -> None:
    root = _root()
    run = _completed_run(root)
    hvg = run["highly_variable_features"]
    artifact_group(root, hvg)["values"][0] = False

    with pytest.raises(ArtifactResolutionError) as caught:
        _ = run.features
    assert caught.value.code == "corrupt_payload"


def test_run_view_rejects_cell_field_owned_by_another_assay() -> None:
    root = _root()
    run = _completed_run(root)
    foreign = _artifact(
        root,
        scope="assay",
        assay="ADT",
        kind="cluster_labels",
        values={"values": np.asarray([0, 2], dtype=np.int32)},
        inputs={"cell_selection": run["analysis_cell_selection"].to_dict()},
    )
    run_group = root[f"pipeline/runs/{run.run_id}"]
    fields = list(run_group.attrs["fields"])
    run_group.attrs["fields"] = [
        {**field, "artifact": foreign.to_dict()}
        if field["axis"] == "cells" and field["key"] == "clusters"
        else field
        for field in fields
    ]

    reopened = open_pipeline_run(_Owner(root), run_id=run.run_id)
    with pytest.raises(ArtifactResolutionError) as caught:
        _ = reopened.cells
    assert caught.value.code == "pipeline_field_axis_mismatch"


def test_run_view_rejects_datastore_scoped_feature_field() -> None:
    root = _root()
    run = _completed_run(root)
    foreign = _artifact(
        root,
        kind="feature_selection",
        values={"values": np.asarray([True, False, True])},
        inputs={},
    )
    run_group = root[f"pipeline/runs/{run.run_id}"]
    fields = list(run_group.attrs["fields"])
    run_group.attrs["fields"] = [
        {**field, "artifact": foreign.to_dict()}
        if field["axis"] == "features" and field["key"] == "highly_variable_features"
        else field
        for field in fields
    ]

    reopened = open_pipeline_run(_Owner(root), run_id=run.run_id)
    with pytest.raises(ArtifactResolutionError) as caught:
        _ = reopened.features
    assert caught.value.code == "pipeline_field_axis_mismatch"


def test_run_view_selected_blocks_and_head_are_bounded() -> None:
    from scarf.datastore._plot_accessor import _FrozenRunPlotStore
    from scarf.plotting._display import stored_display_metadata

    run = _completed_run(_root())

    blocks = tuple(
        run.cells._iter_selected_blocks(
            ("names", "umap_1", "nullable_score"),
            block_rows=1,
        )
    )
    assert [block.active_global_indices.tolist() for block in blocks] == [
        [0],
        [],
        [2],
        [],
    ]
    assert blocks[0].values["nullable_score"].tolist() == [10.0]
    assert np.isnan(blocks[2].values["nullable_score"][0])
    np.testing.assert_allclose(
        run.cells._plot_fetch_all("nullable_score"),
        [10.0, 20.0, np.nan, 40.0],
        equal_nan=True,
    )
    assert run.cells._field_display("nullable_score") == {
        "kind": "continuous",
        "colormap": "viridis",
        "minimum": 10.0,
        "maximum": 40.0,
        "scale": "linear",
    }
    plot_store = _FrozenRunPlotStore(
        run._owner,
        assay=run.assay,
        cells=run.cells,
    )
    np.testing.assert_allclose(
        plot_store.cells.fetch_all("nullable_score"),
        [10.0, 20.0, np.nan, 40.0],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        plot_store.cells.fetch("nullable_score"),
        [10.0, np.nan],
        equal_nan=True,
    )
    np.testing.assert_array_equal(plot_store.cells.fetch("ids"), ["c1", "c3"])
    np.testing.assert_array_equal(plot_store.cells.fetch("clusters"), [0, 2])
    assert stored_display_metadata(plot_store, "nullable_score") == {
        "kind": "continuous",
        "colormap": "viridis",
        "minimum": 10.0,
        "maximum": 40.0,
        "scale": "linear",
    }
    assert run.cells.head(1).to_dict(orient="list") == {
        "I": [True],
        "ids": ["c1"],
        "names": ["a0"],
        "batch": ["x"],
        "nullable_score": [10],
        "umap_1": [1.0],
        "clusters": [0],
    }


@pytest.mark.parametrize(
    ("axis", "path", "replacement"),
    (
        ("cells", "cellData/ids", ["c2", "c1", "c3", "c4"]),
        ("features", "RNA/featureData/ids", ["g2", "g1", "g3"]),
    ),
)
def test_run_view_fails_closed_after_ordered_ids_change(
    axis: str,
    path: str,
    replacement: list[str],
) -> None:
    root = _root()
    run = _completed_run(root)
    root[path][:] = np.asarray(replacement)

    with pytest.raises(ArtifactResolutionError) as caught:
        _ = getattr(run, axis)
    assert caught.value.code == "row_identity_mismatch"


def test_failed_run_supports_reports_but_not_completed_surfaces() -> None:
    root = _root()
    record = create_pipeline_run_record(
        root,
        recipe="basic_rna_analysis",
        requested_label="failed_attempt",
        assay="RNA",
        config={"markers": True},
        stage_order=("input_snapshot",),
        scarf_version="1.0.0",
        started_at_ns=100,
    )
    start_pipeline_stage_record(
        root,
        run_id=record.run_id,
        ordinal=0,
        stage="input_snapshot",
        started_at_ns=110,
    )
    finish_pipeline_stage_record(
        root,
        run_id=record.run_id,
        ordinal=0,
        status="failed",
        metrics=_metrics(),
        error=ValueError("bad snapshot"),
        finished_at_ns=120,
    )
    failed = fail_pipeline_run_record(
        root,
        run_id=record.run_id,
        error=ValueError("bad snapshot"),
        finished_at_ns=121,
    )
    run = PipelineRun(_Owner(root), failed)

    assert run.report()["run"]["error"] == {
        "type": "ValueError",
        "message": "bad snapshot",
    }
    assert "No completed outputs." in run.report(format="markdown")
    with pytest.raises(RuntimeError, match="requires a completed run"):
        _ = run.cells
    with pytest.raises(RuntimeError, match="requires a completed run"):
        _ = len(run)


def test_markdown_report_loads_stage_records_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _completed_run(_root())
    load_count = 0

    def counted_load(
        root: zarr.Group,
        run_id: str,
    ) -> tuple[pipeline_run_storage.PipelineStageRecord, ...]:
        nonlocal load_count
        load_count += 1
        return load_pipeline_stage_records(root, run_id)

    monkeypatch.setattr(
        "scarf.datastore.pipeline_run.load_pipeline_stage_records",
        counted_load,
    )

    assert "# Pipeline run" in run.report(format="markdown")
    assert load_count == 1


def test_pipeline_run_has_no_convenience_consumers_or_commit_internals() -> None:
    run = _completed_run(_root())

    assert not hasattr(run, "plots")
    assert not hasattr(run, "markers")
    assert not hasattr(run, "complete")
    assert not hasattr(run, "requested_label")


def test_pipeline_records_reject_extra_persisted_fields() -> None:
    root = _root()
    record = create_pipeline_run_record(
        root,
        recipe="basic_rna_analysis",
        requested_label=None,
        assay="RNA",
        config={},
        stage_order=("input_snapshot",),
        scarf_version="1.0.0",
        started_at_ns=100,
    )
    root[f"pipeline/runs/{record.run_id}"].attrs["unexpected"] = True

    with pytest.raises(ValueError, match="extra"):
        load_pipeline_run_record(root, record.run_id)


def test_run_view_is_cached_and_rechecks_live_ids_on_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root()
    run = _completed_run(root)
    validations: list[str] = []
    original = PipelineAxisView._validate_contract

    def spy(self) -> None:
        validations.append(self._axis)
        original(self)

    monkeypatch.setattr(PipelineAxisView, "_validate_contract", spy)
    cells = run.cells
    again = run.cells
    features = run.features
    same_features = run.features

    assert cells is again
    assert features is same_features
    assert validations == ["cells", "features"]
    np.testing.assert_array_equal(cells.fetch("clusters"), [0, 2])
    root["cellData/ids"][:] = np.asarray(["c2", "c1", "c3", "c4"])
    with pytest.raises(ArtifactResolutionError) as caught:
        cells.fetch("clusters")
    assert caught.value.code == "row_identity_mismatch"


def test_run_view_fetch_does_not_expand_compact_cluster_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root()
    run = _completed_run(root)
    expanded: list[str] = []
    original = PipelineAxisView._full_axis_values

    def spy(self, descriptor):
        expanded.append(descriptor.key)
        return original(self, descriptor)

    monkeypatch.setattr(PipelineAxisView, "_full_axis_values", spy)
    np.testing.assert_array_equal(run.cells.fetch("clusters"), [0, 2])
    np.testing.assert_allclose(run.cells.fetch("umap_1"), [1.0, 3.0])
    assert expanded == []
    run.cells.fetch_all("clusters")
    assert expanded == ["clusters"]


def test_pipeline_execution_error_exposes_durable_identity() -> None:
    cause = ValueError("bad pca")
    error = PipelineExecutionError(
        run_id="a" * 64,
        stage="pca",
        cause=cause,
    )

    assert error.run_id == "a" * 64
    assert error.stage == "pca"
    assert str(error) == f"Pipeline run {'a' * 64} failed during stage 'pca': bad pca"
