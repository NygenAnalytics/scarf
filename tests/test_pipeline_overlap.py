"""Pipeline stages that run on a worker thread beside later stages."""

import shutil
import threading
import time
from importlib.machinery import ExtensionFileLoader
from typing import Any

import numpy as np
import pytest
import zarr
from threadpoolctl import ThreadpoolController
from zarr.storage import MemoryStore

import scarf.utils.background as background
from scarf.datastore._pipeline_ledger import PipelineEvent, RunLedger
from scarf.datastore.datastore import DataStore
from scarf.datastore.pipeline_run import PipelineExecutionError
from scarf.storage.artifacts import artifact_group
from scarf.storage.pipeline_runs import (
    create_pipeline_run_record,
    finish_pipeline_stage_record,
    load_pipeline_run_record,
    load_pipeline_stage_records,
    start_pipeline_stage_record,
)
from tests.test_pipeline_contract_edge_coverage import _metrics

# Generous, so a loaded test worker never turns a slow thread into a failure.
_WAIT_SECONDS = 30


def _ledger(*stages: str) -> tuple[zarr.Group, RunLedger, list[PipelineEvent]]:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    run = create_pipeline_run_record(
        root,
        recipe="basic_rna_analysis",
        requested_label=None,
        assay="RNA",
        config={},
        stage_order=stages,
        scarf_version="1.0.0",
    )
    events: list[PipelineEvent] = []
    return root, RunLedger(root, run.run_id, events.append), events


def _statuses(root: zarr.Group, ledger: RunLedger) -> dict[str, str]:
    return {
        record.stage: record.status
        for record in load_pipeline_stage_records(root, ledger.run_id)
    }


def test_overlapping_stage_runs_beside_later_stages() -> None:
    root, ledger, events = _ledger("background", "first", "second")
    first_ran = threading.Event()
    threads: dict[str, threading.Thread] = {}

    def background_stage() -> tuple[()]:
        threads["background"] = threading.current_thread()
        # Returns only if a later stage runs while this one is still running.
        if not first_ran.wait(_WAIT_SECONDS):
            raise TimeoutError("later stages did not run beside this stage")
        return ()

    def first_stage() -> tuple[()]:
        threads["first"] = threading.current_thread()
        first_ran.set()
        return ()

    with ledger.overlapping("background", background_stage):
        ledger.run("first", first_stage)
        ledger.run("second", lambda: ())

    assert threads["first"] is threading.current_thread()
    assert threads["background"] is not threading.current_thread()
    assert _statuses(root, ledger) == {
        "background": "completed",
        "first": "completed",
        "second": "completed",
    }
    assert [(event.kind, event.stage) for event in events] == [
        ("stage_started", "background"),
        ("stage_started", "first"),
        ("stage_completed", "first"),
        ("stage_started", "second"),
        ("stage_completed", "second"),
        ("stage_completed", "background"),
    ]


def test_failed_stage_records_the_background_stage_before_the_run_fails() -> None:
    root, ledger, events = _ledger("background", "first")
    failing = threading.Event()

    def background_stage() -> tuple[()]:
        if not failing.wait(_WAIT_SECONDS):
            raise TimeoutError("the failing stage never ran")
        time.sleep(0.2)  # still running when the failure is handled
        return ()

    def first_stage() -> tuple[()]:
        failing.set()
        raise RuntimeError("deliberate failure")

    with pytest.raises(PipelineExecutionError) as caught:
        with ledger.overlapping("background", background_stage):
            ledger.run("first", first_stage)

    assert caught.value.stage == "first"
    assert load_pipeline_run_record(root, ledger.run_id).status == "failed"
    assert _statuses(root, ledger) == {"background": "completed", "first": "failed"}
    assert [(event.kind, event.stage) for event in events][-2:] == [
        ("stage_completed", "background"),
        ("stage_failed", "first"),
    ]


def test_background_failure_ends_the_run() -> None:
    root, ledger, _events = _ledger("background", "first", "second")
    expected = RuntimeError("deliberate background failure")

    def background_stage() -> tuple[()]:
        raise expected

    with pytest.raises(PipelineExecutionError) as caught:
        with ledger.overlapping("background", background_stage):
            ledger.run("first", lambda: ())
            ledger.run("second", lambda: ())

    assert caught.value.stage == "background"
    assert caught.value.__cause__ is expected
    assert load_pipeline_run_record(root, ledger.run_id).status == "failed"
    assert _statuses(root, ledger)["background"] == "failed"


def test_interrupted_stage_records_the_background_stage_first() -> None:
    root, ledger, events = _ledger("background", "first")

    def first_stage() -> tuple[()]:
        raise KeyboardInterrupt("stop")

    with pytest.raises(KeyboardInterrupt):
        with ledger.overlapping("background", lambda: ()):
            ledger.run("first", first_stage)

    assert load_pipeline_run_record(root, ledger.run_id).status == "interrupted"
    assert _statuses(root, ledger) == {
        "background": "completed",
        "first": "interrupted",
    }
    assert [event.kind for event in events][-2:] == [
        "stage_interrupted",
        "pipeline_interrupted",
    ]


def test_overlap_runs_inline_without_a_threadsafe_numba_layer(monkeypatch) -> None:
    monkeypatch.setattr(background, "threadsafe_threading_layer", lambda: False)
    root, ledger, _events = _ledger("background", "first")
    threads: list[threading.Thread] = []

    def background_stage() -> tuple[()]:
        threads.append(threading.current_thread())
        return ()

    with ledger.overlapping("background", background_stage):
        ledger.run("first", lambda: ())

    assert threads == [threading.current_thread()]
    assert _statuses(root, ledger) == {"background": "completed", "first": "completed"}


def test_stages_start_in_order_but_never_after_a_failed_stage() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    run = create_pipeline_run_record(
        root,
        recipe="basic_rna_analysis",
        requested_label=None,
        assay="RNA",
        config={},
        stage_order=("a", "b", "c"),
        scarf_version="1.0.0",
    )
    start_pipeline_stage_record(root, run_id=run.run_id, ordinal=0, stage="a")
    with pytest.raises(ValueError, match="in order"):
        start_pipeline_stage_record(root, run_id=run.run_id, ordinal=2, stage="c")
    # "a" is still running; a later stage may start beside it.
    start_pipeline_stage_record(root, run_id=run.run_id, ordinal=1, stage="b")
    finish_pipeline_stage_record(
        root,
        run_id=run.run_id,
        ordinal=1,
        status="failed",
        metrics=_metrics(),
        error=RuntimeError("deliberate failure"),
    )
    with pytest.raises(ValueError, match="after successful stages"):
        start_pipeline_stage_record(root, run_id=run.run_id, ordinal=2, stage="c")


def _payloads(datastore: DataStore, ref: Any) -> dict[str, np.ndarray]:
    group = artifact_group(datastore.zw, ref)
    return {
        path: np.asarray(node[...])
        for path, node in group.members(max_depth=None)
        if isinstance(node, zarr.Array)
    }


def test_overlapped_pipeline_matches_the_sequential_pipeline(
    datastore_zarr_root,
    tmp_path,
    monkeypatch,
) -> None:
    umap_threads: list[threading.Thread] = []
    run_umap = DataStore._run_umap_artifact

    def recorded_umap(self, *args, **kwargs):
        umap_threads.append(threading.current_thread())
        return run_umap(self, *args, **kwargs)

    monkeypatch.setattr(DataStore, "_run_umap_artifact", recorded_umap)
    # A thread-pool scan and a first extension import deadlock across threads,
    # so the background stage must do neither.
    unsafe_threads: list[threading.Thread] = []
    scan = ThreadpoolController._find_libraries_with_dl_iterate_phdr
    create_module = ExtensionFileLoader.create_module

    def recorded_scan(self):
        unsafe_threads.append(threading.current_thread())
        return scan(self)

    def recorded_create(self, spec):
        unsafe_threads.append(threading.current_thread())
        return create_module(self, spec)

    monkeypatch.setattr(
        ThreadpoolController, "_find_libraries_with_dl_iterate_phdr", recorded_scan
    )
    monkeypatch.setattr(ExtensionFileLoader, "create_module", recorded_create)
    options: dict[str, Any] = {
        "filtering": False,
        "cell_cycle": False,
        "hvg_count": 100,
        "pca_dims": 5,
        "neighbors_k": 5,
        "leiden": {"partitions": [1.0]},
        "paris": False,
        "doublets": False,
    }

    runs = []
    for name, overlap in (("overlapped", True), ("sequential", False)):
        if not overlap:
            monkeypatch.setattr(background, "threadsafe_threading_layer", lambda: False)
        path = tmp_path / name
        shutil.copytree(datastore_zarr_root, path)
        datastore = DataStore(str(path), default_assay="RNA")
        runs.append((datastore, datastore.pipeline.run(**options)))

    assert umap_threads[0] is not threading.current_thread()
    assert umap_threads[1] is threading.current_thread()
    assert umap_threads[0] not in unsafe_threads
    (overlapped_store, overlapped), (sequential_store, sequential) = runs
    assert list(overlapped) == list(sequential)
    for key in ("embedding_initialization", "umap", "leiden_1.0", "markers"):
        expected = _payloads(sequential_store, sequential[key])
        actual = _payloads(overlapped_store, overlapped[key])
        assert expected.keys() == actual.keys(), key
        for path, values in expected.items():
            np.testing.assert_array_equal(actual[path], values, err_msg=key)
