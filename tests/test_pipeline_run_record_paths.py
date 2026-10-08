"""Durable pipeline-run records, label claims, and pipeline execution outcomes."""

import hashlib
import json
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync
from zarr.storage import FsspecStore, MemoryStore

import scarf.storage.pipeline_runs as pipeline_run_storage
from scarf import DataStore, PipelineExecutionError, PipelineRun
from scarf.datastore.pipeline_accessor import PipelineEvent
from scarf.storage.artifacts import ArtifactRef, new_artifact_id
from scarf.storage.pipeline_runs import (
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
    load_pipeline_run_record,
    load_pipeline_stage_record,
    load_pipeline_stage_records,
    open_pipeline_run_record,
    start_pipeline_stage_record,
)
from scarf.storage.schema import create_cell_data, create_zarr_count_assay
from scarf.storage.stores import load_zarr
from scarf.tools.repack_zarr import repack_store
from scarf.utils.logging import logger
from scarf.utils.shutdown import ShutdownRequested, current_shutdown_token
from scarf.writers.counts_t import finalize_writer_counts_t
from tests.storage_helpers import finalize_test_counts, write_count_store

_CLAIMS = "pipeline/runs/.label-claims"
_FAST_RUN = {
    "filtering": False,
    "cell_cycle": False,
    "umap": False,
    "leiden": False,
    "paris": False,
    "doublets": False,
    "markers": False,
}
# Every graph stage runs, and Paris supplies a cluster cut for tree plots.
_GRAPH_RUN = {
    **_FAST_RUN,
    "hvg_count": 20,
    "pca_dims": 3,
    "neighbors_k": 5,
    "paris": True,
}


def _metrics() -> PipelineStageMetrics:
    return PipelineStageMetrics(
        wall_seconds=0.01,
        rss_baseline_bytes=None,
        rss_peak_bytes=None,
        rss_incremental_peak_bytes=None,
        sample_interval_seconds=0.1,
        sample_count=0,
        sampling_error_count=0,
        rss_unavailable_reason="not sampled in tests",
    )


def _root() -> zarr.Group:
    return zarr.open_group(store=MemoryStore(), mode="w")


def _new_run(
    root: zarr.Group,
    *,
    label: str | None = None,
    stages: tuple[str, ...] = ("one",),
) -> PipelineRunRecord:
    return create_pipeline_run_record(
        root,
        recipe="basic_rna_analysis",
        requested_label=label,
        assay="RNA",
        config={},
        stage_order=stages,
        scarf_version="1.0.0",
    )


def _run_stage(root: zarr.Group, run: PipelineRunRecord, ordinal: int) -> None:
    start_pipeline_stage_record(
        root,
        run_id=run.run_id,
        ordinal=ordinal,
        stage=run.stage_order[ordinal],
    )
    finish_pipeline_stage_record(
        root,
        run_id=run.run_id,
        ordinal=ordinal,
        status="completed",
        metrics=_metrics(),
    )


def _complete(root: zarr.Group, run: PipelineRunRecord) -> PipelineRunRecord:
    return complete_pipeline_run_record(root, run_id=run.run_id, outputs=(), fields=())


def _committed_run(root: zarr.Group, label: str | None) -> PipelineRunRecord:
    run = _new_run(root, label=label)
    _run_stage(root, run, 0)
    return _complete(root, run)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_raw(root: zarr.Group, key: str, payload: bytes) -> None:
    buffer = default_buffer_prototype().buffer.from_bytes(payload)
    sync(root.store.set(key, buffer))


def _claim_payload(label: str, run_id: str) -> bytes:
    return json.dumps({"label": label, "runId": run_id}).encode("utf-8")


def _write_claim(
    root: zarr.Group,
    *,
    label: str,
    predecessor: str,
    run_id: str,
) -> None:
    """Write one durable label claim as a finalizer stores it."""
    _write_raw(
        root,
        f"{_CLAIMS}/{_digest(label)}/{predecessor}.json",
        _claim_payload(label, run_id),
    )


def _copy_run(source: zarr.Group, destination: zarr.Group, run_id: str) -> None:
    runs = destination.require_group("pipeline/runs")
    source_run = source[f"pipeline/runs/{run_id}"]
    copied = runs.create_group(run_id, attributes=dict(source_run.attrs))
    stages = copied.create_group("stages")
    for name, stage in source_run["stages"].groups():
        stages.create_group(name, attributes=dict(stage.attrs))


@contextmanager
def _captured_warnings() -> Iterator[list[str]]:
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="WARNING",
    )
    try:
        yield messages
    finally:
        logger.remove(sink)


def _small_store(path: Path) -> str:
    counts = np.random.default_rng(3).poisson(3.0, size=(16, 8)).astype(np.uint32)
    write_count_store(str(path), {"RNA": counts}, np.uint32)
    return str(path)


def _prepared_store(path: Path) -> zarr.Group:
    location = _small_store(path)
    # The first writable open prepares a fresh import.
    DataStore(location, default_assay="RNA", min_features_per_cell=1)
    return zarr.open_group(location, mode="r+")


def test_run_record_stored_under_another_run_id_is_rejected_and_skipped() -> None:
    root = _root()
    run = _committed_run(root, "baseline")
    copy_id = "f" * 64
    runs = root["pipeline/runs"]
    runs.create_group(copy_id, attributes=dict(runs[run.run_id].attrs)).create_group(
        "stages"
    )

    with pytest.raises(ValueError, match=f"{copy_id!r} contains {run.run_id!r}"):
        load_pipeline_run_record(root, copy_id)
    assert list_pipeline_run_records(root) == (run,)
    assert open_pipeline_run_record(root, label="baseline") == run


def test_torn_pipeline_group_without_runs_reads_as_an_empty_catalog() -> None:
    root = _root()
    # The catalog is created one group at a time, so a stop can leave only
    # the pipeline group.
    root.create_group("pipeline")

    assert list_pipeline_run_records(root) == ()
    with pytest.raises(KeyError, match="No completed pipeline run"):
        open_pipeline_run_record(root, label="baseline")

    run = _committed_run(root, "baseline")
    assert open_pipeline_run_record(root, label="baseline") == run


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b'{"label": "baseline", "runId"', "invalid durable claim"),
        (_claim_payload("other", "a" * 64), "collides with another durable claim"),
    ],
    ids=["unreadable", "foreign_label"],
)
def test_labeled_run_refuses_to_start_over_a_corrupt_durable_claim(
    payload: bytes,
    message: str,
) -> None:
    root = _root()
    _committed_run(root, "previous")
    _write_raw(root, f"{_CLAIMS}/{_digest('baseline')}/head.json", payload)
    runs_before = set(root["pipeline/runs"].group_keys())

    with pytest.raises(ValueError, match=message):
        _new_run(root, label="baseline")

    assert set(root["pipeline/runs"].group_keys()) == runs_before
    assert _new_run(root, label="unaffected").status == "running"


@pytest.mark.parametrize("corruption", ["cycle", "foreign_owner"])
def test_commit_fails_durably_when_the_claim_chain_changed_during_the_run(
    corruption: str,
) -> None:
    root = _root()
    # An earlier labeled commit created the claim namespace.
    _committed_run(root, "previous")
    if corruption == "cycle":
        stale = _new_run(root, label="baseline")
        fail_pipeline_run_record(root, run_id=stale.run_id, error=RuntimeError("x"))
        run = _new_run(root, label="baseline")
        _write_claim(root, label="baseline", predecessor="head", run_id=stale.run_id)
        _write_claim(
            root, label="baseline", predecessor=stale.run_id, run_id=stale.run_id
        )
        message = "cyclic durable claim"
    else:
        foreign = _new_run(root, label="other")
        run = _new_run(root, label="baseline")
        _write_claim(root, label="baseline", predecessor="head", run_id=foreign.run_id)
        message = "claim from an incompatible run"
    _run_stage(root, run, 0)

    with pytest.raises(ValueError, match=message):
        _complete(root, run)

    failed = load_pipeline_run_record(root, run.run_id)
    assert failed.status == "failed"
    assert failed.complete
    assert failed.label is None
    assert failed.error is not None
    assert failed.error.type == "PipelineLabelConflict"


def test_commit_through_a_backend_without_atomic_claims_fails_durably(
    tmp_path: Path,
) -> None:
    location = str(tmp_path / "runs.zarr")
    root = zarr.open_group(location, mode="w")
    run = _new_run(root, label="baseline")
    _run_stage(root, run, 0)
    # The same files, opened through a store without conditional writes.
    fsspec_root = zarr.open_group(
        store=FsspecStore.from_url(f"file://{location}"),
        mode="r+",
    )

    with pytest.raises(RuntimeError, match="atomic set_if_not_exists"):
        _complete(fsspec_root, run)

    failed = load_pipeline_run_record(root, run.run_id)
    assert failed.status == "failed"
    assert failed.error is not None
    assert failed.error.type == "PipelineLabelClaimUnavailable"
    with pytest.raises(KeyError, match="No completed pipeline run"):
        open_pipeline_run_record(root, label="baseline")


def test_commit_shares_a_claim_container_created_by_a_concurrent_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root()
    run = _new_run(root, label="baseline")
    _run_stage(root, run, 0)
    create_array = zarr.Group.create_array

    def create_after_competitor(group: zarr.Group, name: str, **options: Any) -> Any:
        if name == ".label-claims":
            # Another first labeled commit creates the container after this
            # one found it missing.
            create_array(group, name, **options)
        return create_array(group, name, **options)

    monkeypatch.setattr(zarr.Group, "create_array", create_after_competitor)
    completed = _complete(root, run)

    assert completed.label == "baseline"
    assert open_pipeline_run_record(root, label="baseline") == completed


def test_stage_records_must_match_their_path_and_the_run_stage_order() -> None:
    root = _root()
    run = _new_run(root, stages=("one", "two"))
    _run_stage(root, run, 0)
    stages = root[f"pipeline/runs/{run.run_id}/stages"]
    stage = stages["0"]
    stored = dict(stage.attrs)

    stage.attrs.put({**stored, "ordinal": 1})
    with pytest.raises(ValueError, match="Pipeline stage path 0 contains 1"):
        load_pipeline_stage_record(root, run.run_id, 0)
    stage.attrs.put({**stored, "stage": "two"})
    with pytest.raises(ValueError, match="does not match its run stage order"):
        load_pipeline_stage_records(root, run.run_id)
    stage.attrs.put(stored)
    stages.create_group("2")
    with pytest.raises(ValueError, match="ordinal is out of range: 2"):
        load_pipeline_stage_records(root, run.run_id)


def test_stage_lifecycle_rejects_unplanned_repeated_and_late_writes() -> None:
    root = _root()
    run = _new_run(root, stages=("one", "two", "three"))
    for ordinal, stage in ((0, "two"), (3, "one")):
        with pytest.raises(ValueError, match="do not match the persisted stage order"):
            start_pipeline_stage_record(
                root, run_id=run.run_id, ordinal=ordinal, stage=stage
            )
    start_pipeline_stage_record(root, run_id=run.run_id, ordinal=0, stage="one")
    selection = ArtifactRef(
        scope="datastore",
        kind="cell_selection",
        artifact_id=new_artifact_id(),
    )
    with pytest.raises(TypeError, match="PipelineStageOutputRecord"):
        finish_pipeline_stage_record(
            root,
            run_id=run.run_id,
            ordinal=0,
            status="completed",
            outputs=(PipelineOutputRecord("selection", selection),),  # type: ignore[arg-type]
            metrics=_metrics(),
        )
    finish_pipeline_stage_record(
        root, run_id=run.run_id, ordinal=0, status="completed", metrics=_metrics()
    )
    with pytest.raises(ValueError, match="Pipeline stage is already terminal"):
        finish_pipeline_stage_record(
            root, run_id=run.run_id, ordinal=0, status="failed", metrics=_metrics()
        )
    start_pipeline_stage_record(root, run_id=run.run_id, ordinal=1, stage="two")

    # A worker that outlives its run may not record more stage results.
    fail_pipeline_run_record(root, run_id=run.run_id, error=RuntimeError("stopped"))
    with pytest.raises(ValueError, match="Cannot finish a stage on a terminal"):
        finish_pipeline_stage_record(
            root, run_id=run.run_id, ordinal=1, status="completed", metrics=_metrics()
        )
    with pytest.raises(ValueError, match="Cannot start a stage on a terminal"):
        start_pipeline_stage_record(root, run_id=run.run_id, ordinal=2, stage="three")
    assert load_pipeline_stage_record(root, run.run_id, 1).status == "running"


def test_stage_started_by_another_writer_after_the_check_already_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root()
    run = _new_run(root, stages=("one", "two"))
    _run_stage(root, run, 0)
    load_stages = pipeline_run_storage._load_pipeline_stage_records_for_run

    def start_competing_stage(
        store_root: zarr.Group, record: PipelineRunRecord
    ) -> tuple[Any, ...]:
        prior = load_stages(store_root, record)
        # Another writer starts the stage after this one listed the stages.
        store_root[f"pipeline/runs/{record.run_id}/stages"].create_group("1")
        return prior

    monkeypatch.setattr(
        pipeline_run_storage,
        "_load_pipeline_stage_records_for_run",
        start_competing_stage,
    )

    with pytest.raises(FileExistsError, match="Pipeline stage 1 already exists"):
        start_pipeline_stage_record(root, run_id=run.run_id, ordinal=1, stage="two")


def test_terminal_runs_cannot_end_again() -> None:
    root = _root()
    completed = _committed_run(root, None)
    interruption = PipelineInterruptionRecord(
        kind="keyboard_interrupt",
        message="stop",
        requested_at_ns=1,
    )

    with pytest.raises(ValueError, match="already terminal"):
        _complete(root, completed)
    with pytest.raises(ValueError, match="already terminal"):
        fail_pipeline_run_record(root, run_id=completed.run_id, error=ValueError("x"))
    with pytest.raises(ValueError, match="already terminal"):
        interrupt_pipeline_run_record(
            root, run_id=completed.run_id, interruption=interruption
        )
    assert load_pipeline_run_record(root, completed.run_id) == completed


def test_completion_requires_every_stage_to_succeed_and_typed_results() -> None:
    root = _root()
    run = _new_run(root, stages=("one", "two"))
    _run_stage(root, run, 0)
    with pytest.raises(ValueError, match="requires every stage record"):
        _complete(root, run)
    start_pipeline_stage_record(root, run_id=run.run_id, ordinal=1, stage="two")
    with pytest.raises(ValueError, match="requires terminal successful stages"):
        _complete(root, run)
    finish_pipeline_stage_record(
        root,
        run_id=run.run_id,
        ordinal=1,
        status="failed",
        metrics=_metrics(),
        error=RuntimeError("stage failed"),
    )
    with pytest.raises(ValueError, match="requires terminal successful stages"):
        _complete(root, run)

    finished = _new_run(root)
    _run_stage(root, finished, 0)
    selection = ArtifactRef(
        scope="datastore",
        kind="cell_selection",
        artifact_id=new_artifact_id(),
    )
    with pytest.raises(TypeError, match="outputs must contain PipelineOutputRecord"):
        complete_pipeline_run_record(
            root,
            run_id=finished.run_id,
            outputs=(PipelineStageOutputRecord("selection", selection, False),),  # type: ignore[arg-type]
            fields=(),
        )
    with pytest.raises(TypeError, match="fields must contain PipelineFieldDescriptor"):
        complete_pipeline_run_record(
            root,
            run_id=finished.run_id,
            outputs=(),
            fields=(PipelineOutputRecord("selection", selection),),  # type: ignore[arg-type]
        )
    assert load_pipeline_run_record(root, finished.run_id).status == "running"


def test_label_lookup_fails_closed_when_copied_runs_share_a_label() -> None:
    root, other = _root(), _root()
    kept = _committed_run(root, "baseline")
    copied = _committed_run(other, "baseline")
    _copy_run(other, root, copied.run_id)

    with pytest.raises(ValueError, match="'baseline' is duplicated by runs"):
        open_pipeline_run_record(root, label="baseline")
    assert {record.run_id for record in list_pipeline_run_records(root)} == {
        kept.run_id,
        copied.run_id,
    }


def test_repack_copies_runs_from_namespaces_without_label_claims(
    tmp_path: Path,
) -> None:
    root = _prepared_store(tmp_path / "source.zarr")
    unlabeled = _committed_run(root, None)
    # A workspace whose run catalog was only partly created.
    root.create_group("analysis/pipeline")
    output = str(tmp_path / "repacked.zarr")

    repack_store(str(tmp_path / "source.zarr"), output)

    result = zarr.open_group(output, mode="r")
    assert load_pipeline_run_record(result, unlabeled.run_id) == unlabeled
    assert ".label-claims" not in result["pipeline/runs"]
    assert list_pipeline_run_records(result["analysis"]) == ()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"not a claim", "Pipeline label claim is invalid"),
        (
            _claim_payload("baseline", "a" * 64),
            "Pipeline label claim digest is invalid",
        ),
    ],
    ids=["unreadable", "misfiled"],
)
def test_repack_refuses_to_copy_corrupt_label_claims(
    tmp_path: Path,
    payload: bytes,
    message: str,
) -> None:
    source = tmp_path / "source.zarr"
    root = _prepared_store(source)
    _committed_run(root, "baseline")
    _write_raw(root, f"{_CLAIMS}/{_digest('other')}/head.json", payload)

    with pytest.raises(ValueError, match=message):
        repack_store(str(source), str(tmp_path / "repacked.zarr"))


def test_repack_refuses_a_label_claim_deleted_after_it_was_listed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zarr"
    _committed_run(_prepared_store(source), "baseline")
    claim = source / _CLAIMS / _digest("baseline") / "head.json"
    list_claims = pipeline_run_storage.collect_aiterator

    def list_then_delete(keys: Any) -> tuple[Any, ...]:
        listed = list_claims(keys)
        # Another process deletes the claim after the copy listed it.
        claim.unlink()
        return listed

    monkeypatch.setattr(pipeline_run_storage, "collect_aiterator", list_then_delete)

    with pytest.raises(
        ValueError, match="Pipeline label claim disappeared during copy"
    ):
        repack_store(str(source), str(tmp_path / "repacked.zarr"))


def test_pipeline_run_rejects_a_bad_callback_and_read_only_stores(
    tmp_path: Path,
) -> None:
    location = _small_store(tmp_path / "data.zarr")
    datastore = DataStore(location, default_assay="RNA", min_features_per_cell=1)
    with pytest.raises(TypeError, match="callback must be callable"):
        datastore.pipeline.run(callback="print", **_FAST_RUN)  # type: ignore[arg-type]

    read_only = DataStore(location, default_assay="RNA", zarr_mode="r")
    with pytest.raises(PermissionError, match=r"zarr_mode='r\+'"):
        read_only.pipeline.run(**_FAST_RUN)
    assert datastore.pipeline.list_runs() == ()


def test_shutdown_requested_between_stages_ends_the_run_interrupted(
    tmp_path: Path,
) -> None:
    location = _small_store(tmp_path / "data.zarr")
    datastore = DataStore(location, default_assay="RNA", min_features_per_cell=1)
    events: list[tuple[str, str]] = []

    def stop_after_first_stage(event: PipelineEvent) -> None:
        events.append((event.kind, event.stage))
        if event.kind == "stage_completed":
            token = current_shutdown_token()
            assert token is not None
            token.request(reason="stop between stages")

    with pytest.raises(ShutdownRequested, match="stop between stages"):
        datastore.pipeline.run(callback=stop_after_first_stage, **_FAST_RUN)

    assert events == [
        ("stage_started", "input_snapshot"),
        ("stage_completed", "input_snapshot"),
        ("pipeline_interrupted", "between_stages"),
    ]
    assert datastore.pipeline.list_runs(status="running") == ()
    (interrupted,) = datastore.pipeline.list_runs(status="interrupted")
    report = interrupted.report()
    assert report["run"]["complete"] is True
    assert report["run"]["interruption"]["kind"] == "shutdown_request"
    assert report["run"]["interruption"]["message"] == "stop between stages"
    assert [(stage["stage"], stage["status"]) for stage in report["stages"]] == [
        ("input_snapshot", "completed")
    ]


_N_CELLS = 120
_N_FEATURES = 40


def _graph_counts() -> np.ndarray:
    rng = np.random.default_rng(11)
    rates = rng.uniform(0.5, 4.0, size=_N_FEATURES)
    counts = rng.poisson(rates, size=(_N_CELLS, _N_FEATURES)).astype(np.uint32)
    # Both features named DUP repeat feature A.
    counts[:, 1] = counts[:, 0]
    counts[:, 2] = counts[:, 0]
    return counts


_GRAPH_COUNTS = _graph_counts()
_GRAPH_NAMES = np.asarray(
    ["A", "DUP", "DUP", *(f"G{index}" for index in range(3, _N_FEATURES))]
)


def _write_graph_store(path: Path) -> str:
    root = load_zarr(zarr_loc=str(path), mode="w")
    cell_ids = np.asarray([f"cell{index}" for index in range(_N_CELLS)])
    create_cell_data(root, None, ids=cell_ids, names=cell_ids)
    counts = create_zarr_count_assay(
        root,
        "RNA",
        None,
        _N_CELLS,
        feat_ids=np.asarray([f"gene{index}" for index in range(_N_FEATURES)]),
        feat_names=_GRAPH_NAMES,
        dtype="uint32",
    )
    counts[:] = _GRAPH_COUNTS
    finalize_test_counts(counts)
    finalize_writer_counts_t(root, "RNA", None)
    return str(path)


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory: Any) -> tuple[DataStore, PipelineRun, str]:
    location = _write_graph_store(tmp_path_factory.mktemp("graph_run") / "data.zarr")
    datastore = DataStore(location, default_assay="RNA", min_features_per_cell=1)
    run = datastore.pipeline.run(label="baseline", **_GRAPH_RUN)
    assert run.status == "completed"
    return datastore, run, location


@pytest.mark.slow
def test_run_aware_export_aligns_layers_to_the_frozen_features(
    completed_run: tuple[DataStore, PipelineRun, str],
) -> None:
    datastore, run, _location = completed_run

    exported = datastore.to_anndata(run=run, matrix="normed", layers={"raw": "RNA"})

    # Normalized values are the run's own: the stored float32 values of its
    # normalized artifact, which cover its highly variable features.
    cells = np.flatnonzero(run.cells.fetch_all("I"))
    features = np.flatnonzero(run.features.fetch_all("highly_variable_features"))
    stored = np.asarray(datastore.load_artifact(run["normalized"])["data"][:])
    feature_ids = datastore.RNA.feats.fetch_all("ids").astype(str)
    expected = _GRAPH_COUNTS[np.ix_(cells, features)]
    assert exported.X.dtype == np.float32
    np.testing.assert_array_equal(exported.X.toarray(), stored)
    assert exported.var_names.tolist() == feature_ids[features].tolist()
    np.testing.assert_array_equal(exported.layers["raw"].toarray(), expected)
    assert not np.array_equal(exported.X.toarray(), expected)


@pytest.mark.slow
def test_cluster_tree_fill_values_resolve_feature_names(
    completed_run: tuple[DataStore, PipelineRun, str],
) -> None:
    datastore, run, _location = completed_run
    graph, cut = run["connectivity_map"], run["paris"]

    with pytest.raises(ValueError, match="MISSING not found in RNA assay"):
        datastore.plots.cluster_tree(
            graph=graph, clusters=cut, fill_by_value="MISSING", show=False
        )
    single = datastore.plots.cluster_tree(
        graph=graph, clusters=cut, fill_by_value="A", show=False
    )
    with _captured_warnings() as messages:
        duplicated = datastore.plots.cluster_tree(
            graph=graph, clusters=cut, fill_by_value="DUP", show=False
        )
    try:
        assert "Plotting mean of 2 features because DUP is not unique." in messages
        # Both DUP features repeat A, so their mean colors the tree like A.
        assert duplicated.legends[0].label == "DUP"
        assert duplicated.legends[0].extras == single.legends[0].extras
    finally:
        single.close()
        duplicated.close()

    missing_cut = ArtifactRef(
        scope=graph.scope,
        assay=graph.assay,
        kind="cluster_cut",
        artifact_id=new_artifact_id(),
    )
    with pytest.raises(ValueError, match="complete Paris cluster-cut artifact"):
        datastore.plots.cluster_tree(graph=graph, clusters=missing_cut, show=False)


@pytest.mark.slow
def test_label_committed_by_another_run_fails_this_run_at_finalize(
    completed_run: tuple[DataStore, PipelineRun, str],
    tmp_path: Path,
) -> None:
    _datastore, _run, location = completed_run
    copy = tmp_path / "copy.zarr"
    shutil.copytree(location, copy)
    datastore = DataStore(str(copy), default_assay="RNA", min_features_per_cell=1)
    competitor: list[str] = []

    def commit_competing_run(event: PipelineEvent) -> None:
        # Another process commits the same label while this run executes.
        if event.kind == "stage_completed" and not competitor:
            competitor.append(_committed_run(datastore.zw, "contested").run_id)

    with pytest.raises(PipelineExecutionError) as caught:
        datastore.pipeline.run(
            label="contested", callback=commit_competing_run, **_GRAPH_RUN
        )

    error = caught.value
    assert error.stage == "finalize"
    assert isinstance(error.__cause__, ValueError)
    assert "already committed" in str(error.__cause__)
    failed = datastore.pipeline.open(run_id=error.run_id)
    assert failed.status == "failed"
    assert failed.label is None
    assert failed.report()["run"]["error"]["type"] == "PipelineLabelConflict"
    assert all(stage["status"] != "running" for stage in failed.report()["stages"])
    assert datastore.pipeline.open(label="contested").run_id == competitor[0]
