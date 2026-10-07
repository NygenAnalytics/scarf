"""Offline tests for the Cytebase dataset workers and the queued pipeline run."""

import hashlib
import json
import logging
import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("modal")
pytest.importorskip("fastapi")
pytest.importorskip("duckdb")

import duckdb
import modal

from scarf.cytebase.pipeline import app
from scarf.cytebase.pipeline import build as pipeline_build
from scarf.cytebase.pipeline import catalog as pipeline_catalog
from scarf.cytebase.pipeline import download as pipeline_download
from scarf.cytebase.pipeline.models import DatasetRecord
from scarf.cytebase.pipeline.resources import PROCESS_RESOURCES, ImportMemoryRefusal
from tests.fixtures_cytebase import (
    BUCKET_ID,
    CALL_ID,
    COLLECTION_ID,
    CYTEBASE_ID,
    NEW_VERSION_ID,
    NOW,
    PIPELINE_VERSION,
    SOURCE_URL,
    VERSION_ID,
    FakeFunction,
    FakeProgressStore,
    cellxgene_collection,
    cellxgene_dataset,
    dataset_record,
    source_details,
    write_h5ad,
)

pytestmark = pytest.mark.usefixtures("cytebase_offline")

RUN_ID = "fc-run-1"
WORKER_ID = "fc-process-1"
CATALOG_WORKER_ID = "fc-catalog-1"
PROCESS_KEY = f"{CYTEBASE_ID}:process"
PREFIX = f"datasets/{CYTEBASE_ID}"
RECORD_PATH = f"{PREFIX}/dataset.json"
SOURCE_PATH = f"{PREFIX}/cellxgene/dataset.json"
CATALOG_PATH = "catalog/cytebase.duckdb"
OTHER_ID = "doe_2023_blood_atlas_55555555"
OTHER_DATASET_ID = "55555555-5555-4555-8555-555555555555"
OTHER_COLLECTION_ID = "99999999-9999-4999-8999-999999999999"
READY = {
    "status": "ready",
    "processedVersionId": VERSION_ID,
    "zarrUri": "/published/data.zarr",
}
APPROVAL = "Review these exact generated paths and resubmit with approvedDeletionPaths"
INTERRUPTED = "Interrupted run reset after explicit worker-drain confirmation"
UNOWNED = "This worker no longer owns the current pipeline stage"
GIB = 1024**3
MIB = 1024**2


class Clock:
    """A manual stand-in for the app module's monotonic clock."""

    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class Progress:
    """Collects stage updates the way a worker's progress callback receives them."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, stage: str, **counters: Any) -> None:
        self.calls.append((stage, counters))

    @property
    def stages(self) -> list[str]:
        return [stage for stage, _ in self.calls]


class Calls:
    """Stands in for ``modal.FunctionCall``; each call ID returns or raises."""

    def __init__(self, outcomes: dict[str, Any] | None = None) -> None:
        self.outcomes = outcomes or {}
        self.polled: list[tuple[str, float | None]] = []

    def from_id(self, call_id: str) -> SimpleNamespace:
        def get(timeout: float | None = None) -> Any:
            self.polled.append((call_id, timeout))
            outcome = self.outcomes.get(call_id)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        return SimpleNamespace(get=get)


class Records(logging.Handler):
    """Keeps the app logger's records; the logger does not propagate to caplog."""

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def messages(self, level: int = logging.INFO) -> list[str]:
        return [
            record.getMessage() for record in self.records if record.levelno == level
        ]


@dataclass
class Worker:
    """Stand-ins for the download, build and publish steps of one dataset."""

    hub: Any
    events: list[SimpleNamespace] = field(default_factory=list)
    write_source: bool = True
    download_error: Exception | None = None
    build_errors: list[Exception | None] = field(default_factory=list)
    conversion: str = "done"
    result: dict[str, Any] = field(default_factory=lambda: {"outcome": "succeeded"})

    @property
    def steps(self) -> list[str]:
        return [event.step for event in self.events]

    def download_h5ad(
        self, url, destination, expected_bytes=None, *, progress=None, timings=None
    ):
        self.events.append(
            SimpleNamespace(
                step="download",
                url=url,
                destination=destination,
                expected_bytes=expected_bytes,
                saved=self.hub.read_json(RECORD_PATH),
                progress=progress,
                timings=timings,
            )
        )
        progress("downloading", completed=4, total=4, unit="bytes")
        if self.download_error is not None:
            raise self.download_error
        if self.write_source:
            destination.write_bytes(b"h5ad")
        return 4, "checksum"

    def build_local(
        self, record, source, store, raw, size, checksum, progress, *, resources
    ):
        self.events.append(
            SimpleNamespace(
                step="build",
                record=record,
                source=source,
                store=store,
                raw=raw,
                size=size,
                checksum=checksum,
                progress=progress,
                resources=resources,
            )
        )
        if self.build_errors and (error := self.build_errors.pop(0)) is not None:
            raise error
        store.mkdir()
        return "manifest", {"status": self.conversion}

    def publish_store(
        self, record, request, storage, store, manifest, converted, progress, check
    ):
        self.events.append(
            SimpleNamespace(
                step="publish",
                record=record,
                request=request,
                storage=storage,
                store=store,
                manifest=manifest,
                converted=converted,
                progress=progress,
                check=check,
                source_exists=(store.parent / "source.h5ad").exists(),
                store_exists=store.is_dir(),
            )
        )
        return self.result


@pytest.fixture
def app_logs():
    handler = Records()
    app.logger.addHandler(handler)
    yield handler
    app.logger.removeHandler(handler)


@pytest.fixture
def progress_store(monkeypatch) -> FakeProgressStore:
    store = FakeProgressStore()
    monkeypatch.setattr(app, "progress_store", store)
    return store


@pytest.fixture(autouse=True)
def run_store(monkeypatch) -> FakeProgressStore:
    """Keep every test off the deployed run store; the harness installs its own."""
    store = FakeProgressStore()
    monkeypatch.setattr(app, "run_store", store)
    return store


@pytest.fixture
def worker(fake_hub, monkeypatch) -> Worker:
    stub = Worker(fake_hub)
    monkeypatch.setattr(pipeline_download, "download_h5ad", stub.download_h5ad)
    monkeypatch.setattr(pipeline_build, "build_local", stub.build_local)
    monkeypatch.setattr(pipeline_build, "publish_store", stub.publish_store)
    return stub


@contextmanager
def _as_call(call_id: str):
    token = CALL_ID.set(call_id)
    try:
        yield
    finally:
        CALL_ID.reset(token)


def _unexpected(*args: Any, **kwargs: Any) -> None:
    raise AssertionError("This step must not run")


def _record(**overrides: Any) -> DatasetRecord:
    return DatasetRecord.model_validate(dataset_record(**overrides))


def _register(hub, **overrides: Any) -> dict[str, Any]:
    """Save a registered record and the CELLxGENE metadata it was built from."""
    record = dataset_record(**overrides)
    prefix = f"datasets/{record['cytebaseId']}"
    hub.put(f"{prefix}/dataset.json", record)
    hub.put(
        f"{prefix}/cellxgene/dataset.json",
        cellxgene_dataset(
            dataset_id=record["datasetId"],
            dataset_version_id=record["latestVersionId"],
        ),
    )
    return record


def _run_file(
    children: dict[str, Any] | None = None,
    *,
    run_id: str = RUN_ID,
    state: str = "running",
) -> dict[str, Any]:
    return {
        "runId": run_id,
        "callId": run_id,
        "action": "process",
        "state": state,
        "startedAt": NOW,
        "updatedAt": NOW,
        "children": children or {},
    }


def _process_child(
    cytebase_id: str = CYTEBASE_ID,
    *,
    call_id: str | None = WORKER_ID,
    state: str = "running",
) -> dict[str, Any]:
    return {
        "cytebaseId": cytebase_id,
        "stage": "process",
        "callId": call_id,
        "state": state,
    }


def _catalog_child(
    *, call_id: str | None = CATALOG_WORKER_ID, state: str = "running"
) -> dict[str, Any]:
    return {"stage": "catalog", "callId": call_id, "state": state}


def _seed_runs(store: FakeProgressStore, run: dict[str, Any] | None) -> None:
    """Mirror a run file into the run store the way the orchestrator keeps it."""
    if run is None:
        return
    store.put(app.CURRENT_RUN, {"runId": run["runId"], "state": run["state"]})
    for key, child in run["children"].items():
        store.put(app._child_key(run["runId"], key), child)


def _snapshot(hub) -> dict[str, bytes]:
    return {path: hub.read(path) for path in hub.files()}


def _writes(hub) -> list[list[str]]:
    return [call[2] for call in hub.calls if call[0] == "batch_bucket_files"]


def _catalog_rows(hub) -> list[tuple[str, str]]:
    with duckdb.connect(str(hub.path(CATALOG_PATH)), read_only=True) as database:
        return database.execute(
            "SELECT cytebase_id, status FROM datasets ORDER BY cytebase_id"
        ).fetchall()


def _process_worker(
    harness, request: dict | None = None, *, resource_tier: int = 0
) -> dict[str, Any]:
    """Run the real ``process_dataset`` body as the reserved Modal call."""
    with _as_call(WORKER_ID):
        return harness.process_dataset.target(
            CYTEBASE_ID, RUN_ID, request or {}, resource_tier
        )


def _format(message: str, *args: Any, exc_info: Any = None) -> str:
    record = logging.LogRecord(
        "cytebase", logging.INFO, __file__, 1, message, args, exc_info
    )
    return app._LogFormatter("%(message)s").format(record)


def test_log_formatter_redacts_tokens_and_remote_urls(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "plain-secret")
    text = _format(
        "env=%s literal=hf_Abc123 source=https://example.org/a.h5ad?sig=1 "
        "store=hf://buckets/ns/name/data.zarr quoted='http://example.org/b' end",
        "plain-secret",
    )
    assert text == (
        "env=[redacted] literal=[redacted] source=[remote URL] "
        "store=[remote URL] quoted='[remote URL]' end"
    )


def test_log_formatter_without_hf_token_still_redacts_literal_tokens():
    assert _format("key=plain-secret token=hf_Abc123") == (
        "key=plain-secret token=[redacted]"
    )


def test_log_formatter_redacts_tracebacks(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "plain-secret")
    try:
        raise RuntimeError("GET https://example.org/x failed for plain-secret")
    except RuntimeError:
        text = _format("Dataset worker failed", exc_info=sys.exc_info())
    assert text.startswith("Dataset worker failed\nTraceback")
    assert text.endswith("RuntimeError: GET [remote URL] failed for [redacted]")
    assert "plain-secret" not in text
    assert "https://" not in text


def test_module_logger_only_writes_through_its_redacting_handler():
    assert app.logger.propagate is False
    assert app.logger.level == logging.INFO
    assert any(
        isinstance(handler.formatter, app._LogFormatter)
        for handler in app.logger.handlers
    )


@pytest.mark.parametrize(("value", "expected"), [(None, 4), ("1", 1), ("16", 16)])
def test_limit_reads_a_positive_container_count(monkeypatch, value, expected):
    if value is not None:
        monkeypatch.setenv("CYTEBASE_PROCESS_CONTAINERS", value)
    assert app._limit("CYTEBASE_PROCESS_CONTAINERS") == expected


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ("0", "CYTEBASE_PROCESS_CONTAINERS must be positive"),
        ("-2", "CYTEBASE_PROCESS_CONTAINERS must be positive"),
        ("four", "invalid literal for int"),
    ],
)
def test_limit_rejects_other_container_counts(monkeypatch, value, match):
    monkeypatch.setenv("CYTEBASE_PROCESS_CONTAINERS", value)
    with pytest.raises(ValueError, match=match):
        app._limit("CYTEBASE_PROCESS_CONTAINERS")


@pytest.mark.parametrize("token", [None, "", "   "])
def test_storage_requires_an_hf_token(monkeypatch, token):
    monkeypatch.setenv("CYTEBASE_BUCKET", BUCKET_ID)
    if token is not None:
        monkeypatch.setenv("HF_TOKEN", token)
    with pytest.raises(RuntimeError, match="Missing required environment variable"):
        app._storage()


def test_storage_opens_the_configured_bucket_with_the_worker_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "  hf_workerToken  ")
    monkeypatch.setenv("CYTEBASE_BUCKET", BUCKET_ID)
    bucket = app._storage()
    assert (bucket.bucket_id, bucket.token, bucket.root) == (
        BUCKET_ID,
        "hf_workerToken",
        f"hf://buckets/{BUCKET_ID}",
    )


def test_storage_requires_a_bucket(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_workerToken")
    with pytest.raises(
        RuntimeError, match="Missing required environment variable: CYTEBASE_BUCKET$"
    ):
        app._storage()


def test_dev_storage_opens_the_dev_bucket(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_workerToken")
    monkeypatch.setenv("CYTEBASE_BUCKET_KEY", "CYTEBASE_BUCKET_DEV")
    monkeypatch.setenv("CYTEBASE_BUCKET", BUCKET_ID)
    monkeypatch.setenv("CYTEBASE_BUCKET_DEV", "test/cytebase-dev")
    assert app._storage().bucket_id == "test/cytebase-dev"


def test_dev_storage_never_falls_back_to_the_production_bucket(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_workerToken")
    monkeypatch.setenv("CYTEBASE_BUCKET_KEY", "CYTEBASE_BUCKET_DEV")
    monkeypatch.setenv("CYTEBASE_BUCKET", BUCKET_ID)
    with pytest.raises(RuntimeError, match="CYTEBASE_BUCKET_DEV"):
        app._storage()


@pytest.mark.parametrize("key", ["HF_TOKEN", "CYTEBASE_BUCKET_PROD", ""])
def test_bucket_key_rejects_other_secret_keys(monkeypatch, key):
    monkeypatch.setenv("CYTEBASE_BUCKET_KEY", key)
    with pytest.raises(ValueError, match="CYTEBASE_BUCKET_KEY must be one of"):
        app._bucket_key()


def test_now_is_an_aware_utc_timestamp():
    before = datetime.now(UTC)
    moment = datetime.fromisoformat(app._now())
    assert moment.utcoffset() == timedelta(0)
    assert before <= moment <= datetime.now(UTC)


@pytest.mark.parametrize(
    ("counters", "summary"),
    [
        pytest.param(None, "", id="none"),
        pytest.param({}, "", id="empty"),
        pytest.param({"percent": None, "message": None}, "", id="unset-values"),
        pytest.param(
            {"completed": GIB, "total": 4 * GIB, "unit": "bytes"},
            "downloaded=1.00/4.00 GiB",
            id="download",
        ),
        pytest.param(
            {
                "completed": GIB // 2,
                "total": None,
                "unit": "bytes",
                "phase": "checksum",
                "downloadedBytes": 3 * GIB,
            },
            "hashed=0.50/? GiB downloaded=3.00 GiB phase=checksum",
            id="checksum-of-unknown-size",
        ),
        pytest.param(
            {"completed": 1_234_567, "total": 2_000_000, "unit": "values"},
            "checked=1,234,567/2,000,000 values",
            id="counted-values",
        ),
        pytest.param(
            {"completed": 7, "total": None, "unit": "rows"},
            "checked=7/? rows",
            id="unknown-count",
        ),
        pytest.param(
            {"phase": "download", "downloadedBytes": GIB},
            "phase=download",
            id="downloaded-bytes-only-while-hashing",
        ),
        pytest.param(
            {
                "downloadSpeedBytesPerSecond": 10 * MIB,
                "message": "Hashing",
                "storedValuesChecked": 7,
                "phase": "checksum",
                "activeConnections": 2,
                "downloadedBytes": 4 * GIB,
                "percent": 50.0,
                "unit": "bytes",
                "total": 4 * GIB,
                "completed": 2 * GIB,
            },
            "hashed=2.00/4.00 GiB 50.0% downloaded=4.00 GiB activeConnections=2 "
            "phase=checksum storedValuesChecked=7 message=Hashing speed=10.0 MiB/s",
            id="fixed-order",
        ),
    ],
)
def test_progress_summary_describes_worker_counters(counters, summary):
    assert app._progress_summary(counters) == summary


@pytest.mark.parametrize(
    "state",
    [
        pytest.param(None, id="no-run-file"),
        pytest.param(
            _run_file({"catalog": _catalog_child()}, run_id="fc-run-2"), id="new-run"
        ),
        pytest.param(
            _run_file({"catalog": _catalog_child()}, state="reset"), id="reset-run"
        ),
        # Missing reservations fail the ownership test before the reservation
        # test, so "Pipeline has no reservation" is unreachable.
        pytest.param(_run_file(), id="no-reservation"),
        pytest.param(
            _run_file({"catalog": _catalog_child(state="succeeded")}),
            id="finished-stage",
        ),
        pytest.param(
            _run_file({"catalog": _catalog_child(call_id="fc-catalog-2")}),
            id="other-worker",
        ),
    ],
)
def test_owner_rejects_workers_without_the_current_reservation(run_store, state):
    _seed_runs(run_store, state)
    with pytest.raises(RuntimeError, match=UNOWNED):
        app._owner(RUN_ID, "catalog", CATALOG_WORKER_ID)


@pytest.mark.parametrize(
    "child",
    [
        _catalog_child(call_id=None, state="pending"),
        _catalog_child(call_id=CATALOG_WORKER_ID, state="running"),
    ],
)
def test_owner_accepts_the_reserved_worker(run_store, child):
    _seed_runs(run_store, _run_file({"catalog": child}))
    app._owner(RUN_ID, "catalog", CATALOG_WORKER_ID)


WORKER = SimpleNamespace(
    runId=RUN_ID, cytebaseId=CYTEBASE_ID, callId=WORKER_ID, attempt=2
)
PROGRESS_KEY = f"{RUN_ID}:{CYTEBASE_ID}:process"


def test_progress_publishes_stage_changes_with_elapsed_times(
    monkeypatch, progress_store, app_logs
):
    clock = Clock()
    monkeypatch.setattr(app, "monotonic", clock)
    with app._progress(WORKER, "process") as update:
        clock.advance(5)
        update("downloading", completed=GIB // 2, total=GIB, unit="bytes")
        clock.advance(2)
        update("downloading", completed=GIB, total=GIB, unit="bytes")
        clock.advance(3)
        update("inspecting", message="Inspecting H5AD")
    first = progress_store.puts[0][1]
    assert first == {
        "runId": RUN_ID,
        "callId": WORKER_ID,
        "stage": "process",
        "attempt": 2,
        "heartbeatAt": first["heartbeatAt"],
        "progress": None,
    }
    assert [
        (key, value["stage"], value["progress"]) for key, value in progress_store.puts
    ] == [
        (PROGRESS_KEY, "process", None),
        (
            PROGRESS_KEY,
            "downloading",
            {"completed": GIB // 2, "total": GIB, "unit": "bytes", "percent": 50.0},
        ),
        (PROGRESS_KEY, "inspecting", {"message": "Inspecting H5AD", "percent": None}),
    ]
    dataset = f"dataset={CYTEBASE_ID}"
    assert app_logs.messages() == [
        f"{dataset} attempt=2 stage=process elapsed=0.0s stageElapsed=0.0s ",
        f"{dataset} stage transition process -> downloading after 5.0s; ",
        f"{dataset} attempt=2 stage=downloading elapsed=5.0s stageElapsed=0.0s "
        "downloaded=0.50/1.00 GiB 50.0%",
        f"{dataset} stage transition downloading -> inspecting after 5.0s; "
        "downloaded=1.00/1.00 GiB 100.0%",
        f"{dataset} attempt=2 stage=inspecting elapsed=10.0s stageElapsed=0.0s "
        "message=Inspecting H5AD",
    ]


@pytest.mark.parametrize(
    ("counters", "percent"),
    [
        ({"completed": 5, "total": 10}, 50.0),
        ({"completed": 15, "total": 10}, 100),
        ({"completed": 5, "total": 0}, None),
        ({"completed": 5}, None),
        ({"total": 10}, None),
    ],
)
def test_progress_percent_is_capped_and_needs_a_known_total(
    progress_store, counters, percent
):
    with app._progress(WORKER, "process") as update:
        update("converting", **counters)
    assert progress_store.puts[-1][1]["progress"] == counters | {"percent": percent}


def test_progress_store_failures_only_log_warnings(monkeypatch, app_logs):
    monkeypatch.setattr(app, "progress_store", FakeProgressStore(fail=True))
    with app._progress(WORKER, "process") as update:
        update("downloading", completed=1, total=2)
    assert (
        app_logs.messages(logging.WARNING)
        == ["Progress update failed: RuntimeError: progress store unavailable"] * 2
    )


def test_progress_heartbeat_republishes_the_latest_counters(
    monkeypatch, progress_store
):
    events = []

    class Heartbeat(threading.Event):
        """The first 15 second wait ends when the test releases it."""

        def __init__(self) -> None:
            super().__init__()
            self.release = threading.Event()
            self.published = threading.Event()
            self.waits = 0
            events.append(self)

        def wait(self, timeout: float | None = None) -> bool:
            self.waits += 1
            if self.waits == 1:
                assert timeout == 15
                assert self.release.wait(5)
                return False
            self.published.set()
            return super().wait(timeout)

    monkeypatch.setattr(app, "Event", Heartbeat)
    with app._progress(WORKER, "process") as update:
        update("process", completed=1, total=4)
        assert len(progress_store.puts) == 1
        events[0].release.set()
        assert events[0].published.wait(5)
        assert len(progress_store.puts) == 2
    initial, heartbeat = (value for _, value in progress_store.puts)
    assert heartbeat["stage"] == "process"
    assert heartbeat["progress"] == {"completed": 1, "total": 4, "percent": 25.0}
    assert heartbeat["heartbeatAt"] >= initial["heartbeatAt"]


def test_progress_stops_its_heartbeat_when_the_work_fails(monkeypatch, progress_store):
    threads = []

    class RecordedThread(threading.Thread):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            threads.append(self)

    monkeypatch.setattr(app, "Thread", RecordedThread)
    with pytest.raises(ValueError, match="conversion failed"):
        with app._progress(WORKER, "process"):
            raise ValueError("conversion failed")
    [thread] = threads
    assert thread.daemon
    assert not thread.is_alive()


def test_cleanup_local_removes_the_workspace_and_times_it(monkeypatch, tmp_path):
    clock = Clock()
    monkeypatch.setattr(app, "monotonic", clock)
    workspace = TemporaryDirectory(dir=tmp_path)
    (Path(workspace.name) / "source.h5ad").write_bytes(b"h5ad")
    calls = []

    def progress(stage: str, **counters: Any) -> None:
        calls.append((stage, counters))
        clock.advance(2.5)

    record = SimpleNamespace(timings={})
    app._cleanup_local(workspace, record, progress)
    assert calls == [
        ("cleaning_local", {"message": "Removing temporary source and store files"})
    ]
    assert not Path(workspace.name).exists()
    assert record.timings == {"cleanupSeconds": 2.5}


def test_cleanup_local_logs_failures_without_raising(app_logs):
    def progress(stage: str, **counters: Any) -> None:
        raise RuntimeError("progress store closed")

    def cleanup() -> None:
        raise OSError("device busy")

    record = SimpleNamespace(timings={})
    app._cleanup_local(SimpleNamespace(cleanup=cleanup), record, progress)
    assert app_logs.messages(logging.WARNING) == [
        "Cleanup progress failed: RuntimeError: progress store closed",
        "Local workspace cleanup failed: OSError: device busy",
    ]
    assert "cleanupSeconds" in record.timings


def test_run_dataset_skips_a_current_ready_store(fake_hub, worker):
    progress = Progress()
    result = app._run_dataset(
        _record(**READY), {}, fake_hub.bucket(), progress, _unexpected
    )
    assert result == {
        "outcome": "skipped",
        "message": "The registered version already has a ready Scarf store",
    }
    assert progress.calls == []
    assert worker.events == []
    assert fake_hub.calls == []


@pytest.mark.parametrize(
    ("overrides", "request_"),
    [
        pytest.param(READY, {"force": True}, id="forced"),
        pytest.param(READY | {"latestVersionId": NEW_VERSION_ID}, {}, id="new-version"),
        pytest.param(READY | {"zarrUri": None}, {}, id="no-store"),
        pytest.param(READY | {"status": "update_available"}, {}, id="not-ready"),
    ],
)
def test_run_dataset_rechecks_sources_unless_the_store_is_current(
    fake_hub, worker, overrides, request_
):
    # No CELLxGENE metadata is saved, so preflight is where each run stops.
    progress = Progress()
    with pytest.raises(ValueError, match="source metadata is missing or has changed"):
        app._run_dataset(
            _record(**overrides), request_, fake_hub.bucket(), progress, _unexpected
        )
    assert progress.stages == ["preflight"]


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(None, id="missing"),
        pytest.param(cellxgene_dataset(dataset_id=OTHER_DATASET_ID), id="dataset"),
        pytest.param(
            cellxgene_dataset(dataset_version_id=NEW_VERSION_ID), id="version"
        ),
    ],
)
def test_run_dataset_requires_the_registered_source_metadata(fake_hub, worker, raw):
    if raw is not None:
        fake_hub.put(SOURCE_PATH, raw)
    with pytest.raises(ValueError, match="source metadata is missing or has changed"):
        app._run_dataset(_record(), {}, fake_hub.bucket(), Progress(), _unexpected)
    assert _writes(fake_hub) == []
    assert worker.events == []


def test_run_dataset_needs_approval_for_each_generated_path(fake_hub, worker):
    fake_hub.put(SOURCE_PATH, cellxgene_dataset())
    for path in (
        "data.zarr/zarr.json",
        "metadata/obs.json",
        "scarf_ingest.json",
        "README.md",
    ):
        fake_hub.put(f"{PREFIX}/{path}", "{}")
    request = {"approvedDeletionPaths": [f"{PREFIX}/metadata/obs.json"]}
    result = app._run_dataset(
        _record(), request, fake_hub.bucket(), Progress(), _unexpected
    )
    assert result == {
        "outcome": "needsApproval",
        "message": APPROVAL,
        "deletionPaths": [
            f"{PREFIX}/data.zarr/zarr.json",
            f"{PREFIX}/scarf_ingest.json",
        ],
    }
    assert _writes(fake_hub) == []
    assert worker.events == []


@pytest.mark.parametrize(
    ("overrides", "request_", "saved_status"),
    [
        pytest.param({}, {}, "processing", id="registered"),
        pytest.param(
            {"status": "needsInput", "needsInput": {"question": "Which counts?"}},
            {},
            "processing",
            id="answered-question",
        ),
        pytest.param(READY, {"force": True}, "ready", id="forced-ready-store"),
    ],
)
def test_run_dataset_downloads_builds_and_publishes_in_a_private_workspace(
    fake_hub, worker, overrides, request_, saved_status
):
    fake_hub.put(SOURCE_PATH, cellxgene_dataset())
    record, storage, progress, checks = (
        _record(**overrides),
        fake_hub.bucket(),
        Progress(),
        [],
    )

    def check() -> None:
        checks.append(progress.stages[-1])

    result = app._run_dataset(record, request_, storage, progress, check)

    download, build, publish = worker.events
    workspace = download.destination.parent
    assert result is worker.result
    assert (download.url, download.destination.name, download.expected_bytes) == (
        SOURCE_URL,
        "source.h5ad",
        1234,
    )
    # The unavailable state is saved before the download starts.
    assert download.saved["status"] == saved_status
    assert download.saved["needsInput"] is None
    assert download.progress is progress
    assert download.timings is record.timings
    assert build.record is record
    assert (build.source, build.store) == (
        download.destination,
        workspace / "data.zarr",
    )
    assert (build.raw, build.size, build.checksum) == (
        cellxgene_dataset(),
        4,
        "checksum",
    )
    assert build.progress is progress
    assert build.resources == PROCESS_RESOURCES[0]
    assert publish.record is record
    assert publish.request is request_
    assert publish.storage is storage
    assert (publish.store, publish.manifest, publish.converted) == (
        workspace / "data.zarr",
        "manifest",
        {"status": "done"},
    )
    assert publish.progress is progress
    assert publish.check is check
    assert publish.store_exists
    assert not publish.source_exists
    # Ownership is checked before saving and again before the build.
    assert checks == ["preflight", "downloading"]
    assert progress.stages == ["preflight", "downloading", "cleaning_local"]
    assert not workspace.exists()
    assert set(record.timings) == {"downloadSeconds", "cleanupSeconds"}
    assert (record.status, record.needsInput) == (saved_status, None)


def test_run_dataset_keeps_the_source_when_conversion_needs_input(fake_hub, worker):
    fake_hub.put(SOURCE_PATH, cellxgene_dataset())
    worker.conversion = "needsInput"
    app._run_dataset(_record(), {}, fake_hub.bucket(), Progress(), lambda: None)
    assert worker.steps == ["download", "build", "publish"]
    assert worker.events[-1].source_exists
    assert not worker.events[0].destination.parent.exists()


def test_run_dataset_warns_when_the_source_is_already_gone(fake_hub, worker, app_logs):
    fake_hub.put(SOURCE_PATH, cellxgene_dataset())
    worker.write_source = False
    result = app._run_dataset(
        _record(), {}, fake_hub.bucket(), Progress(), lambda: None
    )
    assert result == {"outcome": "succeeded"}
    [warning] = app_logs.messages(logging.WARNING)
    assert warning.startswith("Local source cleanup failed: FileNotFoundError")


@pytest.mark.parametrize("failure", ["download", "ownership"])
def test_run_dataset_cleans_up_after_a_failed_step(fake_hub, worker, failure):
    fake_hub.put(SOURCE_PATH, cellxgene_dataset())
    checks = []
    if failure == "download":
        worker.download_error = RuntimeError("connection reset")

    def check() -> None:
        checks.append(len(checks))
        if failure == "ownership" and len(checks) == 2:
            raise RuntimeError(UNOWNED)

    record, progress = _record(), Progress()
    with pytest.raises(RuntimeError, match="connection reset|no longer owns"):
        app._run_dataset(record, {}, fake_hub.bucket(), progress, check)
    assert worker.steps == ["download"]
    assert not worker.events[0].destination.parent.exists()
    assert progress.stages[-1] == "cleaning_local"
    assert set(record.timings) == {"downloadSeconds", "cleanupSeconds"}


@pytest.mark.parametrize("resource_tier", range(3))
def test_process_dataset_records_the_attempt_before_and_after_work(
    modal_harness, monkeypatch, resource_tier
):
    hub = modal_harness.hub
    _register(
        hub,
        attempt=2,
        error="old failure",
        pipelineVersion="old-pipeline",
        timings={"downloadSeconds": 9.0},
    )
    _seed_runs(modal_harness.run_store, _run_file({PROCESS_KEY: _process_child()}))
    seen = {}

    def run_dataset(record, request, storage, progress, check, *, resources):
        check()
        seen.update(
            request=request,
            storage=storage,
            started=hub.read_json(RECORD_PATH),
            bound=storage.progress is progress,
            resources=resources,
        )
        progress("downloading", completed=1, total=2, unit="bytes")
        return {"outcome": "succeeded"}

    monkeypatch.setattr(app, "_run_dataset", run_dataset)
    result = _process_worker(
        modal_harness, {"force": True}, resource_tier=resource_tier
    )

    started, saved = seen["started"], hub.read_json(RECORD_PATH)
    assert {
        key: started[key]
        for key in (
            "attempt",
            "runId",
            "callId",
            "stage",
            "stageOutcome",
            "pipelineVersion",
            "error",
            "timings",
        )
    } == {
        "attempt": 3,
        "runId": RUN_ID,
        "callId": WORKER_ID,
        "stage": "process",
        "stageOutcome": "running",
        "pipelineVersion": PIPELINE_VERSION,
        "error": None,
        "timings": {},
    }
    assert started["startedAt"] is not None
    assert seen["request"] == {"force": True}
    assert seen["resources"] == PROCESS_RESOURCES[resource_tier]
    assert seen["bound"]
    assert seen["storage"].progress is None
    assert result == {
        "outcome": "succeeded",
        "cytebaseId": CYTEBASE_ID,
        "status": "registered",
        "record": saved,
    }
    assert (saved["stageOutcome"], saved["error"], list(saved["timings"])) == (
        "succeeded",
        None,
        ["processSeconds"],
    )
    assert [
        (key, value["stage"], value["attempt"], value["callId"])
        for key, value in modal_harness.progress_store.puts
    ] == [
        (PROGRESS_KEY, "process", 3, WORKER_ID),
        (PROGRESS_KEY, "downloading", 3, WORKER_ID),
    ]


@pytest.mark.parametrize("resource_tier", range(3))
@pytest.mark.parametrize("status", ["registered", "ready"])
def test_process_dataset_keeps_the_status_when_a_refusal_is_retried(
    modal_harness, monkeypatch, resource_tier, status
):
    hub = modal_harness.hub
    _register(hub, **(READY if status == "ready" else {}))
    _seed_runs(modal_harness.run_store, _run_file({PROCESS_KEY: _process_child()}))

    def run_dataset(record, *args, **kwargs):
        # The worker commits processing before it downloads the source.
        record.status = "processing"
        raise ImportMemoryRefusal("count layout exceeds budget")

    monkeypatch.setattr(app, "_run_dataset", run_dataset)
    result = _process_worker(modal_harness, resource_tier=resource_tier)

    saved = hub.read_json(RECORD_PATH)
    retried = resource_tier < len(PROCESS_RESOURCES) - 1
    assert (result["outcome"], result["_retryableMemory"]) == ("failed", True)
    assert saved["status"] == result["status"] == (status if retried else "failed")
    assert (saved["stageOutcome"], saved["error"]) == (
        "failed",
        "ImportMemoryRefusal: count layout exceeds budget",
    )


def test_process_dataset_records_its_container_and_peak_memory(
    modal_harness, monkeypatch
):
    hub = modal_harness.hub
    _register(hub)
    _seed_runs(modal_harness.run_store, _run_file({PROCESS_KEY: _process_child()}))
    seen = {}

    @contextmanager
    def peak_memory():
        yield lambda: 3 * GIB

    def run_dataset(record, request, storage, progress, check, *, resources):
        seen["started"] = hub.read_json(RECORD_PATH)
        return {"outcome": "succeeded"}

    monkeypatch.setattr(app, "_peak_memory", peak_memory)
    monkeypatch.setattr(app, "_run_dataset", run_dataset)
    result = _process_worker(modal_harness, resource_tier=1)

    container = {"cpu": 8, "memoryMiB": 32_768, "memBudget": "24G"}
    assert seen["started"]["resources"] == container | {"peakMemoryBytes": None}
    saved = hub.read_json(RECORD_PATH)
    assert saved["resources"] == result["record"]["resources"]
    assert saved["resources"] == container | {"peakMemoryBytes": 3 * GIB}


@pytest.mark.skipif(not Path("/proc/self/statm").exists(), reason="needs /proc")
def test_peak_memory_includes_child_processes():
    size = 64 * MIB
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            f"import sys, time; block = b'x' * {size}; print('ready', flush=True); "
            "time.sleep(60)",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline() == "ready\n"
        assert app._process_tree_bytes(child.pid) >= size
        with app._peak_memory(interval=0.01) as peak:
            assert app._process_tree_bytes(os.getpid()) >= size
        assert peak() is not None and peak() >= size
    finally:
        child.kill()
        child.wait()


@pytest.mark.parametrize(
    ("outcome", "error"),
    [
        pytest.param(
            {"outcome": "skipped", "message": "Already ready"}, None, id="skip"
        ),
        pytest.param(
            {
                "outcome": "needsApproval",
                "message": APPROVAL,
                "deletionPaths": [f"{PREFIX}/scarf_ingest.json"],
            },
            APPROVAL,
            id="approval",
        ),
        pytest.param(
            {"outcome": "needsInput", "message": "Counts need a decision"},
            "Counts need a decision",
            id="input",
        ),
    ],
)
def test_process_dataset_saves_unfinished_outcomes_as_errors(
    modal_harness, monkeypatch, outcome, error
):
    hub = modal_harness.hub
    _register(hub)
    _seed_runs(modal_harness.run_store, _run_file({PROCESS_KEY: _process_child()}))
    monkeypatch.setattr(app, "_run_dataset", lambda *args, **kwargs: dict(outcome))
    result = _process_worker(modal_harness)
    saved = hub.read_json(RECORD_PATH)
    assert result == outcome | {
        "cytebaseId": CYTEBASE_ID,
        "status": "registered",
        "record": saved,
    }
    assert (saved["stageOutcome"], saved["error"]) == (outcome["outcome"], error)


def test_process_dataset_saves_a_redacted_failure(modal_harness, monkeypatch, app_logs):
    hub = modal_harness.hub
    _register(hub)
    _seed_runs(modal_harness.run_store, _run_file({PROCESS_KEY: _process_child()}))

    def run_dataset(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("aria2c rejected hf_offlineTestToken")

    monkeypatch.setattr(app, "_run_dataset", run_dataset)
    result = _process_worker(modal_harness)

    message = "RuntimeError: aria2c rejected [redacted]"
    saved = hub.read_json(RECORD_PATH)
    assert result == {
        "outcome": "failed",
        "message": message,
        "cytebaseId": CYTEBASE_ID,
        "status": "failed",
        "record": saved,
    }
    assert (saved["status"], saved["stageOutcome"], saved["error"]) == (
        "failed",
        "failed",
        message,
    )
    [failure] = [
        record for record in app_logs.records if record.levelno == logging.ERROR
    ]
    assert failure.getMessage() == (
        f"Dataset worker failed: dataset={CYTEBASE_ID} {message}"
    )
    formatted = app._LogFormatter().format(failure)
    assert "Traceback" in formatted
    assert "hf_offlineTestToken" not in formatted


def test_process_dataset_refuses_work_it_cannot_claim(modal_harness, monkeypatch):
    hub = modal_harness.hub
    _register(hub)
    child = _process_child(call_id="fc-process-9")
    _seed_runs(modal_harness.run_store, _run_file({PROCESS_KEY: child}))
    before = hub.read(RECORD_PATH)
    monkeypatch.setattr(app, "_run_dataset", _unexpected)
    with pytest.raises(RuntimeError, match=UNOWNED):
        _process_worker(modal_harness)
    assert hub.read(RECORD_PATH) == before
    assert modal_harness.progress_store.puts == []


def test_process_dataset_does_not_save_results_after_losing_ownership(
    modal_harness, monkeypatch
):
    hub = modal_harness.hub
    _register(hub)
    _seed_runs(modal_harness.run_store, _run_file({PROCESS_KEY: _process_child()}))

    def run_dataset(*args: Any, **kwargs: Any) -> dict:
        # An operator resets the run while this worker is still busy.
        _seed_runs(
            modal_harness.run_store,
            _run_file({PROCESS_KEY: _process_child()}, state="reset"),
        )
        return {"outcome": "succeeded"}

    monkeypatch.setattr(app, "_run_dataset", run_dataset)
    with pytest.raises(RuntimeError, match=UNOWNED):
        _process_worker(modal_harness)
    assert hub.read_json(RECORD_PATH)["stageOutcome"] == "running"


def _catalog_worker(harness, request: dict, *, call_id: str = CATALOG_WORKER_ID):
    """Run the real ``build_catalog`` body as the given Modal call."""
    with _as_call(call_id):
        return harness.build_catalog.target(request, RUN_ID)


def test_build_catalog_runs_the_catalog_as_the_reserved_worker(
    modal_harness, monkeypatch, app_logs
):
    _seed_runs(modal_harness.run_store, _run_file({"catalog": _catalog_child()}))
    calls = []

    def run_catalog(request, storage, check):
        check()
        calls.append((request, storage.bucket_id, storage.token))
        return {"status": "done"}

    monkeypatch.setattr(pipeline_catalog, "run_catalog", run_catalog)
    assert _catalog_worker(modal_harness, {"updates": []}) == {"status": "done"}
    assert calls == [({"updates": []}, BUCKET_ID, "hf_offlineTestToken")]
    started, finished = app_logs.messages()
    assert started == f"Catalog worker started: run={RUN_ID}"
    assert finished.startswith(f"Catalog worker finished: run={RUN_ID} elapsed=")


def test_build_catalog_logs_and_reraises_failures(modal_harness, monkeypatch, app_logs):
    _seed_runs(modal_harness.run_store, _run_file({"catalog": _catalog_child()}))

    def run_catalog(request, storage, check):
        raise RuntimeError("catalog upload failed")

    monkeypatch.setattr(pipeline_catalog, "run_catalog", run_catalog)
    with pytest.raises(RuntimeError, match="catalog upload failed"):
        _catalog_worker(modal_harness, {})
    [failure] = [
        record for record in app_logs.records if record.levelno == logging.ERROR
    ]
    assert failure.getMessage() == f"Catalog worker failed: run={RUN_ID}"
    assert failure.exc_info is not None


def test_build_catalog_requires_the_catalog_reservation(modal_harness, monkeypatch):
    _seed_runs(
        modal_harness.run_store,
        _run_file({"catalog": _catalog_child(call_id="fc-catalog-2")}),
    )
    monkeypatch.setattr(pipeline_catalog, "run_catalog", _unexpected)
    with pytest.raises(RuntimeError, match=UNOWNED):
        _catalog_worker(modal_harness, {})


BLOCKED = _run_file({PROCESS_KEY: _process_child()}, state="blocked")
RESET = {"workersDrained": True, "expectedRunId": RUN_ID}


@pytest.mark.parametrize(
    ("run", "request_"),
    [
        pytest.param(BLOCKED, {"expectedRunId": RUN_ID}, id="unconfirmed"),
        pytest.param(BLOCKED, RESET | {"workersDrained": False}, id="not-drained"),
        pytest.param(BLOCKED, RESET | {"expectedRunId": "fc-run-0"}, id="other-run"),
        pytest.param(BLOCKED, {"workersDrained": True}, id="no-expected-run"),
        pytest.param(None, RESET, id="no-run-file"),
    ],
)
def test_reset_requires_drain_confirmation_for_the_current_run(
    fake_hub, monkeypatch, run, request_
):
    if run is not None:
        fake_hub.put(app.RUN_PATH, run)
    calls = Calls()
    monkeypatch.setattr(modal, "FunctionCall", calls)
    before = _snapshot(fake_hub)
    with pytest.raises(
        ValueError, match="exact expected run ID and explicit workersDrained"
    ):
        app._reset(fake_hub.bucket(), request_)
    assert calls.polled == []
    assert _snapshot(fake_hub) == before


def test_reset_accepts_finished_failed_and_timed_out_calls(
    fake_hub, monkeypatch, app_logs, run_store
):
    _register(fake_hub)
    _register(fake_hub, cytebaseId=OTHER_ID, datasetId=OTHER_DATASET_ID)
    state = _run_file(
        {
            "catalog": _catalog_child(state="succeeded"),
            PROCESS_KEY: _process_child(),
            f"{OTHER_ID}:process": _process_child(
                OTHER_ID, call_id=None, state="pending"
            ),
        },
        state="blocked",
    )
    fake_hub.put(app.RUN_PATH, state)
    _seed_runs(run_store, state)
    calls = Calls(
        {
            RUN_ID: modal.exception.FunctionTimeoutError("Function timed out"),
            CATALOG_WORKER_ID: {"status": "done"},
            WORKER_ID: RuntimeError("worker crashed with hf_leakedToken1"),
        }
    )
    monkeypatch.setattr(modal, "FunctionCall", calls)

    assert app._reset(fake_hub.bucket(), RESET) == {"runId": RUN_ID, "state": "reset"}

    # Finished children need no draining; the pending one was never spawned.
    assert calls.polled == [(RUN_ID, 0), (WORKER_ID, 0)]
    assert run_store.get(app.CURRENT_RUN) == {"runId": RUN_ID, "state": "reset"}
    assert app_logs.messages(logging.WARNING) == [
        f"Using explicit worker-drain confirmation for call {WORKER_ID}: "
        "RuntimeError: worker crashed with [redacted]"
    ]
    saved = fake_hub.read_json(app.RUN_PATH)
    assert saved == state | {"state": "reset", "updatedAt": saved["updatedAt"]}
    assert datetime.fromisoformat(saved["updatedAt"]) > datetime.fromisoformat(NOW)


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(modal.exception.OutputExpiredError("expired"), id="expired"),
        pytest.param(ConnectionError("control plane unavailable"), id="unavailable"),
    ],
)
def test_reset_relies_on_drain_confirmation_when_results_are_unavailable(
    fake_hub, monkeypatch, app_logs, run_store, error
):
    _register(fake_hub)
    fake_hub.put(app.RUN_PATH, BLOCKED)
    _seed_runs(run_store, BLOCKED)
    monkeypatch.setattr(modal, "FunctionCall", Calls({WORKER_ID: error}))
    assert app._reset(fake_hub.bucket(), RESET)["state"] == "reset"
    [warning] = app_logs.messages(logging.WARNING)
    assert warning.startswith(
        f"Using explicit worker-drain confirmation for call {WORKER_ID}"
    )


def test_reset_refuses_while_a_worker_is_still_running(
    fake_hub, monkeypatch, run_store
):
    _register(fake_hub, runId=RUN_ID, stageOutcome="running", status="processing")
    fake_hub.put(app.RUN_PATH, BLOCKED)
    _seed_runs(run_store, BLOCKED)
    monkeypatch.setattr(modal, "FunctionCall", Calls({WORKER_ID: TimeoutError()}))
    before = _snapshot(fake_hub)
    with pytest.raises(
        ValueError, match=f"Call {WORKER_ID} is still running; drain workers"
    ) as raised:
        app._reset(fake_hub.bucket(), RESET)
    assert isinstance(raised.value.__cause__, TimeoutError)
    assert _snapshot(fake_hub) == before


@pytest.mark.parametrize(
    ("overrides", "expected", "rewritten"),
    [
        pytest.param(
            {"runId": RUN_ID, "stageOutcome": "running", "status": "processing"},
            ("failed", "failed", INTERRUPTED),
            True,
            id="processing",
        ),
        pytest.param(
            {"runId": RUN_ID, "stageOutcome": "running", "status": "ready"},
            ("failed", "ready", INTERRUPTED),
            True,
            id="ready-store-kept",
        ),
        pytest.param(
            {"runId": "fc-run-0", "stageOutcome": "running", "status": "processing"},
            ("running", "processing", None),
            False,
            id="other-run",
        ),
        pytest.param(
            {"runId": RUN_ID, "stageOutcome": "succeeded", "status": "ready"},
            ("succeeded", "ready", None),
            False,
            id="finished",
        ),
    ],
)
def test_reset_fails_only_records_this_run_left_running(
    fake_hub, monkeypatch, run_store, overrides, expected, rewritten
):
    _register(fake_hub, **overrides)
    fake_hub.put(app.RUN_PATH, BLOCKED)
    _seed_runs(run_store, BLOCKED)
    monkeypatch.setattr(modal, "FunctionCall", Calls())
    app._reset(fake_hub.bucket(), RESET)
    saved = fake_hub.read_json(RECORD_PATH)
    assert (saved["stageOutcome"], saved["status"], saved.get("error")) == expected
    assert _writes(fake_hub) == [[RECORD_PATH]] * rewritten + [[app.RUN_PATH]]


def test_run_pipeline_rejects_unknown_actions(modal_harness):
    with pytest.raises(ValueError, match="Unknown pipeline action: publish"):
        modal_harness.run("publish", {}, run_id=RUN_ID)
    assert modal_harness.hub.files() == []


@pytest.mark.parametrize(
    ("request_", "match"),
    [
        pytest.param({}, "Supply exactly one of cytebaseIds", id="no-selector"),
        pytest.param({"cytebaseIds": ["Not-An-Id"]}, "lowercase letters", id="bad-id"),
    ],
)
def test_process_run_validates_its_selection_before_starting(
    modal_harness, request_, match
):
    with pytest.raises(ValueError, match=match):
        modal_harness.run("process", request_, run_id=RUN_ID)
    assert modal_harness.hub.files() == []


@pytest.mark.parametrize("previous", ["running", "blocked"])
def test_run_pipeline_waits_for_unresolved_runs_to_be_reset(modal_harness, previous):
    hub = modal_harness.hub
    hub.put(app.RUN_PATH, _run_file(run_id="fc-run-0", state=previous))
    before = hub.read(app.RUN_PATH)
    with pytest.raises(
        RuntimeError,
        match="Run fc-run-0 has unresolved workers; drain and explicitly reset it",
    ):
        modal_harness.run("catalog", {}, run_id=RUN_ID)
    assert hub.read(app.RUN_PATH) == before
    assert modal_harness.build_catalog.spawned == []


@pytest.mark.parametrize("previous", ["completed", "failed", "reset"])
def test_run_pipeline_starts_after_resolved_runs(modal_harness, previous):
    hub = modal_harness.hub
    hub.put(app.RUN_PATH, _run_file(run_id="fc-run-0", state=previous))
    assert modal_harness.run("catalog", {}, run_id=RUN_ID)["state"] == "completed"
    assert hub.read_json(app.RUN_PATH)["runId"] == RUN_ID


def test_catalog_run_publishes_every_registered_dataset(modal_harness):
    hub = modal_harness.hub
    _register(hub)
    result = modal_harness.run("catalog", {}, run_id=RUN_ID)
    assert result == {
        "callId": RUN_ID,
        "state": "completed",
        "datasets": [],
        "catalog": {
            "status": "done",
            "datasets": 1,
            "collections": 0,
            "catalogUri": f"{modal_harness.bucket.root}/{CATALOG_PATH}",
            "catalogSha256": hashlib.sha256(hub.read(CATALOG_PATH)).hexdigest(),
        },
        "error": None,
        "successes": [],
        "failures": [],
    }
    assert modal_harness.build_catalog.spawned == [({}, RUN_ID)]
    saved = hub.read_json(app.RUN_PATH)
    assert saved == {
        "runId": RUN_ID,
        "callId": RUN_ID,
        "action": "catalog",
        "state": "completed",
        "startedAt": saved["startedAt"],
        "updatedAt": saved["updatedAt"],
        "children": {"catalog": _catalog_child(state="succeeded")},
    }
    assert _catalog_rows(hub) == [(CYTEBASE_ID, "registered")]


def test_register_run_reports_registered_and_failed_collections(
    modal_harness, monkeypatch
):
    def fetch_collection(collection_id: str) -> tuple[bytes, dict]:
        if collection_id != COLLECTION_ID:
            raise ValueError("CELLxGENE returned a different collection identity")
        collection = cellxgene_collection()
        return json.dumps(collection).encode(), collection

    monkeypatch.setattr(pipeline_catalog, "fetch_collection", fetch_collection)
    request = {"collectionIds": [OTHER_COLLECTION_ID, COLLECTION_ID]}
    result = modal_harness.run("register", request, run_id=RUN_ID)

    assert (result["state"], result["error"]) == ("completed", None)
    assert result["datasets"] == [{"cytebaseId": CYTEBASE_ID, "status": "registered"}]
    assert result["successes"] == [COLLECTION_ID]
    assert result["failures"] == [OTHER_COLLECTION_ID]
    assert result["catalog"]["failedCollections"] == [
        {
            "collectionId": OTHER_COLLECTION_ID,
            "error": "ValueError: CELLxGENE returned a different collection identity",
        }
    ]
    assert modal_harness.build_catalog.spawned == [(request, RUN_ID)]
    assert modal_harness.hub.read_json(RECORD_PATH)["status"] == "registered"
    assert _catalog_rows(modal_harness.hub) == [(CYTEBASE_ID, "registered")]


def test_process_run_publishes_a_ready_store_and_catalog_row(
    modal_harness, monkeypatch
):
    hub = modal_harness.hub
    _register(hub)
    downloads = []

    def download_h5ad(
        url, destination, expected_bytes=None, *, progress=None, timings=None
    ):
        downloads.append((url, expected_bytes))
        write_h5ad(destination)
        size, checksum = source_details(destination)
        progress("downloading", completed=size, total=size, unit="bytes")
        return size, checksum

    monkeypatch.setattr(pipeline_download, "download_h5ad", download_h5ad)
    result = modal_harness.run("process", {"cytebaseIds": [CYTEBASE_ID]}, run_id=RUN_ID)

    saved = hub.read_json(RECORD_PATH)
    assert downloads == [(SOURCE_URL, 1234)]
    assert (result["state"], result["error"]) == ("completed", None)
    assert result["datasets"] == [
        {"outcome": "succeeded", "cytebaseId": CYTEBASE_ID, "status": "ready"}
    ]
    assert (result["successes"], result["failures"]) == ([CYTEBASE_ID], [])
    assert (saved["status"], saved["stageOutcome"], saved["processedVersionId"]) == (
        "ready",
        "succeeded",
        VERSION_ID,
    )
    assert (saved["runId"], saved["callId"], saved["attempt"]) == (
        RUN_ID,
        WORKER_ID,
        1,
    )
    assert saved["zarrUri"] == f"{modal_harness.bucket.root}/{PREFIX}/data.zarr"
    assert hub.path(f"{PREFIX}/data.zarr/zarr.json").is_file()
    assert {"downloadSeconds", "cleanupSeconds", "processSeconds"} <= set(
        saved["timings"]
    )
    assert _catalog_rows(hub) == [(CYTEBASE_ID, "ready")]
    stages = [value["stage"] for _, value in modal_harness.progress_store.puts]
    assert stages[:3] == ["process", "preflight", "downloading"]
    assert stages[-1] == "cleaning_local"
    assert hub.read_json(app.RUN_PATH)["children"] == {
        "catalog": _catalog_child(call_id="fc-catalog-2", state="succeeded"),
        PROCESS_KEY: _process_child(state="succeeded"),
    }


def _succeed(record, request, storage, progress, check, *, resources=None) -> dict:
    check()
    return {"outcome": "succeeded"}


@pytest.mark.parametrize("refusals", range(4))
def test_process_run_escalates_only_to_the_available_resource_tiers(
    modal_harness, worker, refusals
):
    hub = modal_harness.hub
    _register(hub, attempt=5)
    worker.build_errors = [
        ImportMemoryRefusal("count layout exceeds budget")
    ] * refusals

    result = modal_harness.run("process", {"cytebaseIds": [CYTEBASE_ID]}, run_id=RUN_ID)

    attempts = min(refusals + 1, 3)
    outcome = "failed" if refusals == 3 else "succeeded"
    workers = modal_harness.process_dataset
    assert [args[3] for args in workers.spawned] == list(range(attempts))
    assert workers.spawn_options == [
        {"cpu": resource.cpu, "memory": resource.memoryMiB}
        for resource in PROCESS_RESOURCES[:attempts]
    ]
    builds = [event for event in worker.events if event.step == "build"]
    downloads = [event for event in worker.events if event.step == "download"]
    assert [build.resources for build in builds] == list(PROCESS_RESOURCES[:attempts])
    assert len(downloads) == attempts
    assert len({event.destination.parent for event in downloads}) == attempts
    assert all(not event.destination.parent.exists() for event in downloads)
    assert worker.steps.count("publish") == (0 if refusals == 3 else 1)

    saved = hub.read_json(RECORD_PATH)
    assert (saved["attempt"], saved["callId"], saved["stageOutcome"]) == (
        5 + attempts,
        f"fc-process-{attempts}",
        outcome,
    )
    assert result["state"] == "completed"
    assert len(result["datasets"]) == 1
    assert result["datasets"][0]["outcome"] == outcome
    assert result["successes"] == ([] if refusals == 3 else [CYTEBASE_ID])
    assert result["failures"] == ([CYTEBASE_ID] if refusals == 3 else [])
    assert "_retryableMemory" not in json.dumps(result)
    assert "_retryableMemory" not in json.dumps(saved)
    # Neither a retryable failure nor its record reaches a catalog update.
    assert [args[0] for args in modal_harness.build_catalog.spawned] == [
        {"updates": []},
        {"updates": [saved]},
    ]
    children = [
        child
        for key, child in modal_harness.run_store.puts
        if key == f"{RUN_ID}:{PROCESS_KEY}"
    ]
    assert [child["state"] for child in children] == [
        state
        for attempt in range(attempts)
        for state in (
            "pending",
            "running",
            outcome if attempt == attempts - 1 else "failed",
        )
    ]
    assert [child["callId"] for child in children if child["state"] == "running"] == [
        f"fc-process-{attempt + 1}" for attempt in range(attempts)
    ]


@pytest.mark.parametrize(
    ("shape", "start"),
    [
        ({"meanGenesPerCell": None}, 0),
        ({"cellCount": 100_000, "nGenes": 36_601, "meanGenesPerCell": 5_000.0}, 1),
        ({"cellCount": 79_631, "nGenes": 18_736, "meanGenesPerCell": 5_519.0}, 2),
    ],
)
@pytest.mark.parametrize("refusals", range(3))
def test_process_run_starts_in_the_estimated_tier_and_escalates_from_it(
    modal_harness, worker, shape, start, refusals
):
    _register(modal_harness.hub, **shape)
    worker.build_errors = [
        ImportMemoryRefusal("count layout exceeds budget")
    ] * refusals

    result = modal_harness.run("process", {"cytebaseIds": [CYTEBASE_ID]}, run_id=RUN_ID)

    tiers = list(range(start, len(PROCESS_RESOURCES)))[: refusals + 1]
    workers = modal_harness.process_dataset
    assert [args[3] for args in workers.spawned] == tiers
    assert workers.spawn_options == [
        {
            "cpu": PROCESS_RESOURCES[tier].cpu,
            "memory": PROCESS_RESOURCES[tier].memoryMiB,
        }
        for tier in tiers
    ]
    builds = [event for event in worker.events if event.step == "build"]
    assert [build.resources for build in builds] == [
        PROCESS_RESOURCES[tier] for tier in tiers
    ]
    refused = refusals >= len(tiers)
    assert result["state"] == "completed"
    assert result["datasets"][0]["outcome"] == ("failed" if refused else "succeeded")


@pytest.mark.parametrize("step", ["download", "build", "publish"])
@pytest.mark.parametrize("error_type", [MemoryError, RuntimeError])
def test_process_run_does_not_retry_other_failures(
    modal_harness, worker, monkeypatch, step, error_type
):
    _register(modal_harness.hub)
    error = error_type("operation failed")
    if step == "download":
        worker.download_error = error
    elif step == "build":
        worker.build_errors = [error]
    else:

        def fail_publication(*args, **kwargs):
            raise error

        monkeypatch.setattr(pipeline_build, "publish_store", fail_publication)

    result = modal_harness.run("process", {"cytebaseIds": [CYTEBASE_ID]}, run_id=RUN_ID)

    assert len(modal_harness.process_dataset.spawned) == 1
    assert result["state"] == "completed"
    assert result["failures"] == [CYTEBASE_ID]
    assert (
        result["datasets"][0]["message"] == f"{error_type.__name__}: operation failed"
    )
    assert "_retryableMemory" not in result["datasets"][0]


@pytest.mark.parametrize(
    "outcome", ["succeeded", "skipped", "needsInput", "needsApproval"]
)
def test_process_run_requires_failed_outcome_to_retry(
    modal_harness, monkeypatch, outcome
):
    _register(modal_harness.hub)

    def run_dataset(*args, **kwargs):
        return {"outcome": outcome, "_retryableMemory": True}

    monkeypatch.setattr(app, "_run_dataset", run_dataset)
    result = modal_harness.run("process", {"cytebaseIds": [CYTEBASE_ID]}, run_id=RUN_ID)

    assert len(modal_harness.process_dataset.spawned) == 1
    assert result["datasets"][0]["outcome"] == outcome
    assert "_retryableMemory" not in json.dumps(result)


def test_process_run_does_not_retry_after_losing_memory_refusal_ownership(
    modal_harness, monkeypatch
):
    hub = modal_harness.hub
    _register(hub)

    def run_dataset(*args, **kwargs):
        modal_harness.run_store.put(
            app.CURRENT_RUN, {"runId": RUN_ID, "state": "reset"}
        )
        raise ImportMemoryRefusal("count layout exceeds budget")

    monkeypatch.setattr(app, "_run_dataset", run_dataset)
    result = modal_harness.run("process", {"cytebaseIds": [CYTEBASE_ID]}, run_id=RUN_ID)

    assert len(modal_harness.process_dataset.spawned) == 1
    assert result["state"] == "blocked"
    assert result["datasets"][0]["message"] == f"RuntimeError: {UNOWNED}"
    assert hub.read_json(RECORD_PATH)["stageOutcome"] == "running"
    assert [args[0] for args in modal_harness.build_catalog.spawned] == [
        {"updates": []}
    ]


def test_process_run_does_not_publish_a_refusal_when_the_retry_result_is_unknown(
    modal_harness, worker, monkeypatch
):
    _register(modal_harness.hub)
    worker.build_errors = [ImportMemoryRefusal("count layout exceeds budget")]
    workers = FakeFunction(
        modal_harness.process_dataset.target,
        "process",
        get_errors=[None, RuntimeError("retry result unavailable")],
    )
    monkeypatch.setattr(app, "process_dataset", workers)

    result = modal_harness.run("process", {"cytebaseIds": [CYTEBASE_ID]}, run_id=RUN_ID)

    assert [args[3] for args in workers.spawned] == [0, 1]
    assert result["state"] == "blocked"
    assert result["datasets"] == [
        {
            "cytebaseId": CYTEBASE_ID,
            "outcome": "failed",
            "message": "RuntimeError: retry result unavailable",
        }
    ]
    assert [args[0] for args in modal_harness.build_catalog.spawned] == [
        {"updates": []}
    ]


def test_a_pacing_error_before_a_retry_fails_the_run(
    modal_harness, worker, monkeypatch
):
    hub = modal_harness.hub
    _register(hub)
    worker.build_errors = [ImportMemoryRefusal("count layout exceeds budget")]

    class Pacer:
        def __init__(self, interval: float) -> None:
            self.waits = 0

        async def wait(self) -> None:
            self.waits += 1
            if self.waits > 1:
                raise RuntimeError("pacer unavailable")

    monkeypatch.setattr(app, "_StartPacer", Pacer)
    result = modal_harness.run("process", {"cytebaseIds": [CYTEBASE_ID]}, run_id=RUN_ID)

    # The refused attempt finished, so no outcome is unknown when the
    # orchestrator fails before the retry: the run fails instead of blocking.
    assert [args[3] for args in modal_harness.process_dataset.spawned] == [0]
    assert (result["state"], result["error"]) == (
        "failed",
        "RuntimeError: pacer unavailable",
    )
    assert hub.read_json(RECORD_PATH)["status"] == "registered"


def test_process_run_reports_unregistered_datasets_without_stopping_others(
    modal_harness, monkeypatch
):
    hub = modal_harness.hub
    _register(hub)
    monkeypatch.setattr(app, "_run_dataset", _succeed)
    result = modal_harness.run(
        "process", {"cytebaseIds": [CYTEBASE_ID, "missing_dataset"]}, run_id=RUN_ID
    )

    assert {row["cytebaseId"]: row for row in result["datasets"]} == {
        CYTEBASE_ID: {
            "outcome": "succeeded",
            "cytebaseId": CYTEBASE_ID,
            "status": "registered",
        },
        "missing_dataset": {
            "cytebaseId": "missing_dataset",
            "outcome": "failed",
            "message": "FileNotFoundError: Dataset missing_dataset is not registered",
        },
    }
    assert (result["state"], result["error"]) == ("completed", None)
    assert (result["successes"], result["failures"]) == (
        [CYTEBASE_ID],
        ["missing_dataset"],
    )
    saved = hub.read_json(app.RUN_PATH)
    assert saved["state"] == "completed"
    # Workers start concurrently, so call IDs are not in a fixed order.
    assert {key: child["state"] for key, child in saved["children"].items()} == {
        "catalog": "succeeded",
        PROCESS_KEY: "succeeded",
        "missing_dataset:process": "failed",
    }
    # The catalog is refreshed first, then receives only the processed record.
    assert [args[0] for args in modal_harness.build_catalog.spawned] == [
        {"updates": []},
        {"updates": [hub.read_json(RECORD_PATH)]},
    ]


@pytest.mark.parametrize(
    ("results", "elapsed", "summary"),
    [
        pytest.param(
            [],
            30.0,
            "done=0/4 running=1 queued=3 elapsed=0h00m",
            id="nothing-finished",
        ),
        pytest.param(
            [{"outcome": "succeeded"}, {"outcome": "failed"}],
            3_600.0,
            "done=2/4 running=1 queued=1 failed=1 succeeded=1 "
            "rate=2.0/h eta=1h00m elapsed=1h00m",
            id="half-done",
        ),
    ],
)
def test_run_progress_summarizes_outcomes_rate_and_eta(results, elapsed, summary):
    children = {
        "catalog": _catalog_child(state="running"),
        PROCESS_KEY: _process_child(state="running"),
        f"{OTHER_ID}:process": _process_child(OTHER_ID, state="succeeded"),
    }
    assert app._run_progress(4, children, results, elapsed) == summary


@pytest.mark.parametrize(("quota", "seconds"), [(1000, 15.0), (3000, 5.0), (6000, 2.5)])
def test_start_interval_keeps_dataset_starts_under_the_hub_quota(quota, seconds):
    assert app._start_interval(quota) == seconds


def test_start_pacer_spaces_starts_by_its_interval(monkeypatch):
    clock, slept = Clock(), []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)
        clock.advance(seconds)

    monkeypatch.setattr(app, "monotonic", clock)
    monkeypatch.setattr(app.asyncio, "sleep", sleep)

    async def three_starts() -> list[float]:
        pacer, started = app._StartPacer(15.0), []
        for _ in range(3):
            await pacer.wait()
            started.append(clock.now)
            clock.advance(4.0)
        return started

    assert app.asyncio.run(three_starts()) == [1_000.0, 1_015.0, 1_030.0]
    assert slept == [11.0, 11.0]


def _three_datasets(hub) -> list[str]:
    third = "roe_2022_kidney_atlas_66666666"
    _register(hub)
    _register(hub, cytebaseId=OTHER_ID, datasetId=OTHER_DATASET_ID)
    _register(hub, cytebaseId=third, datasetId="66666666-6666-4666-8666-666666666666")
    return [CYTEBASE_ID, OTHER_ID, third]


def test_process_run_keeps_at_most_the_container_limit_running(
    modal_harness, monkeypatch
):
    keys = _three_datasets(modal_harness.hub)
    lock, running, peak = threading.Lock(), [0], [0]

    def run_dataset(record, request, storage, progress, check, *, resources):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        try:
            if resources == PROCESS_RESOURCES[0]:
                raise ImportMemoryRefusal("count layout exceeds budget")
            return _succeed(record, request, storage, progress, check)
        finally:
            with lock:
                running[0] -= 1

    monkeypatch.setattr(app, "_run_dataset", run_dataset)
    monkeypatch.setattr(app, "PROCESS_CONTAINERS", 1)
    result = modal_harness.run("process", {"cytebaseIds": keys}, run_id=RUN_ID)

    assert (result["state"], sorted(result["successes"])) == ("completed", sorted(keys))
    assert peak[0] == 1
    spawned = modal_harness.process_dataset.spawned
    assert len(spawned) == 6
    # A dataset retains its slot between attempts, ahead of every queued dataset.
    assert all(spawned[index][0] == spawned[index + 1][0] for index in (0, 2, 4))
    assert [args[3] for args in spawned] == [0, 1] * 3


def test_process_run_paces_every_dataset_start(modal_harness, monkeypatch):
    keys = _three_datasets(modal_harness.hub)
    events: list[str] = []

    class Pacer:
        def __init__(self, interval: float) -> None:
            events.append(f"interval={interval}")

        async def wait(self) -> None:
            events.append("wait")

    workers = modal_harness.process_dataset
    with_options = workers.with_options

    def paced_options(**options):
        variant = with_options(**options)
        spawn = variant.spawn.aio

        async def spawn_after_pacing(*args):
            events.append("spawn")
            return await spawn(*args)

        variant.spawn = SimpleNamespace(aio=spawn_after_pacing)
        return variant

    def run_dataset(*args, resources):
        if resources == PROCESS_RESOURCES[0]:
            raise ImportMemoryRefusal("count layout exceeds budget")
        return _succeed(*args, resources=resources)

    monkeypatch.setattr(app, "_StartPacer", Pacer)
    monkeypatch.setattr(app, "DATASET_START_INTERVAL", 15.0)
    monkeypatch.setattr(workers, "with_options", paced_options)
    monkeypatch.setattr(app, "_run_dataset", run_dataset)
    modal_harness.run("process", {"cytebaseIds": keys}, run_id=RUN_ID)

    assert events[0] == "interval=15.0"
    assert events.count("wait") == events.count("spawn") == 6
    # Each start waits for the pacer before its worker is spawned.
    assert all(
        events[: index + 1].count("wait") > events[:index].count("spawn")
        for index, event in enumerate(events)
        if event == "spawn"
    )


def test_process_run_state_costs_a_fixed_number_of_bucket_calls(
    modal_harness, monkeypatch
):
    hub = modal_harness.hub
    keys = _three_datasets(hub)
    monkeypatch.setattr(app, "_run_dataset", _succeed)
    modal_harness.run("process", {"cytebaseIds": keys}, run_id=RUN_ID)

    reads = [
        call
        for call in hub.calls
        if call[0] == "download_bucket_files" and app.RUN_PATH in call[2]
    ]
    writes = [path for paths in _writes(hub) for path in paths if path == app.RUN_PATH]
    # One read to check the previous run, and one write each at start and end,
    # however many datasets run; claims and checks live in the run store.
    assert (len(reads), len(writes)) == (1, 2)
    assert {
        key: value["state"]
        for key, value in modal_harness.run_store.values.items()
        if key.startswith(f"{RUN_ID}:")
    } == {f"{RUN_ID}:{key}:process": "succeeded" for key in keys} | {
        f"{RUN_ID}:catalog": "succeeded"
    }
    assert modal_harness.run_store.get(app.CURRENT_RUN) == {
        "runId": RUN_ID,
        "state": "completed",
    }


def test_process_run_forwards_only_each_datasets_approved_paths(
    modal_harness, monkeypatch
):
    hub = modal_harness.hub
    _register(hub)
    for path in ("data.zarr/zarr.json", "scarf_ingest.json"):
        hub.put(f"{PREFIX}/{path}", "{}")
    monkeypatch.setattr(pipeline_download, "download_h5ad", _unexpected)
    approved = [
        f"{PREFIX}/data.zarr/zarr.json",
        f"{PREFIX}_v2/scarf_ingest.json",
        f"datasets/{OTHER_ID}/scarf_ingest.json",
    ]
    request = {"cytebaseIds": [CYTEBASE_ID], "approvedDeletionPaths": approved}
    result = modal_harness.run("process", request, run_id=RUN_ID)

    assert modal_harness.process_dataset.spawned == [
        (
            CYTEBASE_ID,
            RUN_ID,
            {"cytebaseIds": [CYTEBASE_ID], "approvedDeletionPaths": approved[:1]},
            0,
        )
    ]
    assert result["datasets"] == [
        {
            "outcome": "needsApproval",
            "message": APPROVAL,
            "deletionPaths": [f"{PREFIX}/scarf_ingest.json"],
            "cytebaseId": CYTEBASE_ID,
            "status": "registered",
        }
    ]
    assert (result["state"], result["failures"]) == ("completed", [CYTEBASE_ID])
    saved = hub.read_json(RECORD_PATH)
    assert (saved["stageOutcome"], saved["error"]) == ("needsApproval", APPROVAL)


@pytest.mark.parametrize(
    ("failure", "child"),
    [
        pytest.param(
            {"spawn_error": RuntimeError("worker unavailable")},
            _process_child(call_id=None, state="pending"),
            id="spawn",
        ),
        pytest.param(
            {"get_errors": [RuntimeError("worker unavailable")]},
            _process_child(state="running"),
            id="result",
        ),
    ],
)
def test_process_run_is_blocked_when_a_worker_outcome_is_unknown(
    modal_harness, monkeypatch, failure, child
):
    hub = modal_harness.hub
    _register(hub)
    workers = FakeFunction(modal_harness.process_dataset.target, "process", **failure)
    monkeypatch.setattr(app, "process_dataset", workers)
    before = hub.read(RECORD_PATH)
    result = modal_harness.run("process", {"cytebaseIds": [CYTEBASE_ID]}, run_id=RUN_ID)

    assert (result["state"], result["error"]) == ("blocked", None)
    assert result["datasets"] == [
        {
            "cytebaseId": CYTEBASE_ID,
            "outcome": "failed",
            "message": "RuntimeError: worker unavailable",
        }
    ]
    assert result["failures"] == [CYTEBASE_ID]
    saved = hub.read_json(app.RUN_PATH)
    assert (saved["state"], saved["children"][PROCESS_KEY]) == ("blocked", child)
    assert hub.read(RECORD_PATH) == before
    # Uncertain results are never merged into the catalog.
    assert [args[0] for args in modal_harness.build_catalog.spawned] == [
        {"updates": []}
    ]


def _periodic_run(modal_harness, monkeypatch, *, fail_update: bool):
    """Process two datasets; the first takes over a minute, the second waits.

    The second dataset finishes only after the orchestrator publishes the
    first one's record, so a periodic catalog update must happen in between.
    """
    hub = modal_harness.hub
    _register(hub)
    _register(hub, cytebaseId=OTHER_ID, datasetId=OTHER_DATASET_ID)
    clock, released, published = Clock(), threading.Event(), []
    monkeypatch.setattr(app, "monotonic", clock)

    def run_dataset(record, request, storage, progress, check, *, resources):
        if record.cytebaseId == CYTEBASE_ID:
            clock.advance(61)
        elif not released.wait(5):
            raise AssertionError("The periodic catalog update never happened")
        return {"outcome": "succeeded"}

    def run_catalog(request, storage, check):
        check()
        updates = [row["cytebaseId"] for row in request["updates"]]
        published.append(updates)
        if updates:
            released.set()
            if fail_update:
                raise RuntimeError("catalog unavailable")
        return {"status": "done", "updated": updates}

    monkeypatch.setattr(app, "_run_dataset", run_dataset)
    monkeypatch.setattr(pipeline_catalog, "run_catalog", run_catalog)
    result = modal_harness.run(
        "process", {"cytebaseIds": [CYTEBASE_ID, OTHER_ID]}, run_id=RUN_ID
    )
    return result, published


def test_process_run_publishes_finished_records_every_minute(
    modal_harness, monkeypatch, app_logs
):
    result, published = _periodic_run(modal_harness, monkeypatch, fail_update=False)
    assert published == [[], [CYTEBASE_ID], [OTHER_ID]]
    # The minute tick also logs how far the run has come.
    [progress] = [
        message
        for message in app_logs.messages()
        if message.startswith("Pipeline progress")
    ]
    assert progress == (
        f"Pipeline progress: run={RUN_ID} done=1/2 running=1 queued=0 "
        "succeeded=1 rate=59.0/h eta=0h01m elapsed=0h01m"
    )
    assert (result["state"], result["error"]) == ("completed", None)
    assert result["catalog"] == {"status": "done", "updated": [OTHER_ID]}
    assert result["successes"] == [CYTEBASE_ID, OTHER_ID]


def test_process_run_stops_publishing_after_a_periodic_update_fails(
    modal_harness, monkeypatch
):
    result, published = _periodic_run(modal_harness, monkeypatch, fail_update=True)
    # The second record is not published once a catalog result is uncertain.
    assert published == [[], [CYTEBASE_ID]]
    assert result["error"] == "RuntimeError: catalog unavailable"
    assert result["catalog"] == {"status": "done", "updated": []}
    assert result["successes"] == [CYTEBASE_ID, OTHER_ID]
    # A catalog worker that raised has an unknown outcome; see
    # test_a_raising_catalog_worker_blocks_the_run_until_reset.
    assert result["state"] == "blocked"


@pytest.mark.parametrize(
    ("action", "request_", "datasets"),
    [
        pytest.param("catalog", {}, [], id="catalog"),
        pytest.param("register", {"collectionIds": [COLLECTION_ID]}, [], id="register"),
        pytest.param(
            "process",
            {"cytebaseIds": [CYTEBASE_ID]},
            [
                {
                    "outcome": "succeeded",
                    "cytebaseId": CYTEBASE_ID,
                    "status": "registered",
                }
            ],
            id="final-process-update",
        ),
    ],
)
def test_run_pipeline_reports_a_failed_catalog_publication(
    modal_harness, monkeypatch, action, request_, datasets
):
    _register(modal_harness.hub)

    def run_catalog(request, storage, check):
        # Only the refresh that precedes processing succeeds.
        if request.get("updates") != []:
            raise RuntimeError("catalog unavailable")
        return {"status": "done"}

    monkeypatch.setattr(app, "_run_dataset", _succeed)
    monkeypatch.setattr(pipeline_catalog, "run_catalog", run_catalog)
    result = modal_harness.run(action, request_, run_id=RUN_ID)

    assert result["error"] == "RuntimeError: catalog unavailable"
    assert result["datasets"] == datasets
    assert result["failures"] == []
    # A catalog worker that raised has an unknown outcome; see
    # test_a_raising_catalog_worker_blocks_the_run_until_reset.
    assert result["state"] == "blocked"


def test_a_raising_catalog_worker_blocks_the_run_until_reset(
    modal_harness, monkeypatch
):
    hub = modal_harness.hub
    _register(hub)

    def run_catalog(request, storage, check):
        check()
        raise RuntimeError("catalog unavailable")

    monkeypatch.setattr(pipeline_catalog, "run_catalog", run_catalog)
    result = modal_harness.run("catalog", {}, run_id=RUN_ID)

    # Modal re-raises a worker's own exception as it reports an unavailable
    # result, so a raising worker's outcome is unknown and the run is blocked.
    assert (result["state"], result["error"]) == (
        "blocked",
        "RuntimeError: catalog unavailable",
    )
    saved = hub.read_json(app.RUN_PATH)
    assert saved["state"] == "blocked"
    assert saved["children"]["catalog"] == _catalog_child()
    with pytest.raises(RuntimeError, match=f"Run {RUN_ID} has unresolved workers"):
        modal_harness.run("catalog", {}, run_id="fc-run-2")

    calls = Calls({CATALOG_WORKER_ID: RuntimeError("catalog unavailable")})
    monkeypatch.setattr(modal, "FunctionCall", calls)
    reset = modal_harness.run("reset", RESET, run_id="fc-run-3")
    assert reset == {"runId": RUN_ID, "state": "reset"}
    assert calls.polled == [(RUN_ID, 0), (CATALOG_WORKER_ID, 0)]


def test_an_orchestration_error_fails_the_run_without_blocking_new_work(
    modal_harness, monkeypatch
):
    hub = modal_harness.hub
    _register(hub)

    class Pacer:
        def __init__(self, interval: float) -> None:
            pass

        async def wait(self) -> None:
            raise RuntimeError("pacer unavailable")

    monkeypatch.setattr(app, "_StartPacer", Pacer)
    process = {"cytebaseIds": [CYTEBASE_ID]}
    result = modal_harness.run("process", process, run_id=RUN_ID)

    # The error came from the orchestrator before any dataset worker started,
    # so no outcome is unknown: the run fails instead of blocking.
    assert (result["state"], result["error"]) == (
        "failed",
        "RuntimeError: pacer unavailable",
    )
    assert modal_harness.process_dataset.spawned == []
    assert hub.read_json(app.RUN_PATH)["state"] == "failed"
    assert modal_harness.run("catalog", {}, run_id="fc-run-2")["state"] == "completed"
    assert hub.read_json(app.RUN_PATH)["runId"] == "fc-run-2"


def test_blocked_run_must_be_reset_before_new_work(modal_harness, monkeypatch):
    hub = modal_harness.hub
    _register(hub)
    workers = FakeFunction(
        modal_harness.process_dataset.target,
        "process",
        get_errors=[RuntimeError("result unavailable")],
    )
    monkeypatch.setattr(app, "process_dataset", workers)
    process = {"cytebaseIds": [CYTEBASE_ID]}
    assert modal_harness.run("process", process, run_id=RUN_ID)["state"] == "blocked"

    with pytest.raises(RuntimeError, match=f"Run {RUN_ID} has unresolved workers"):
        modal_harness.run("catalog", {}, run_id="fc-run-2")

    calls = Calls({WORKER_ID: RuntimeError("worker lost")})
    monkeypatch.setattr(modal, "FunctionCall", calls)
    reset = modal_harness.run("reset", RESET, run_id="fc-run-3")
    assert reset == {"runId": RUN_ID, "state": "reset"}
    # Only the worker whose result was lost is still marked running.
    assert calls.polled == [(RUN_ID, 0), (WORKER_ID, 0)]

    assert modal_harness.run("catalog", {}, run_id="fc-run-4")["state"] == "completed"
    assert hub.read_json(app.RUN_PATH)["runId"] == "fc-run-4"
