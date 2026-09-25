"""One worker per dataset, a queued orchestrator, and the development HTTP API."""

import asyncio
import logging
import os
import re
import subprocess
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Lock, Thread
from time import monotonic
from typing import Any

import modal
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .._storage import Bucket, dataset_prefix, error_message
from .catalog import load_record, select_dataset_ids
from .download import download_connections
from .models import DatasetRecord, ProcessRequest, RegisterRequest


class _LogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        token = os.environ.get("HF_TOKEN")
        if token:
            message = message.replace(token, "[redacted]")
        message = re.sub(r"hf_[A-Za-z0-9]+", "[redacted]", message)
        return re.sub(r"(?:https?|hf)://[^\s\"'<>]+", "[remote URL]", message)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(_LogFormatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)

app = modal.App("cytebase")
secret = modal.Secret.from_name(
    "scarf-env", required_keys=["HF_TOKEN", "CYTEBASE_BUCKET"]
)
progress_store = modal.Dict.from_name("cytebase-progress", create_if_missing=True)
RUN_PATH = "_internal/pipeline.json"


def _limit(name: str) -> int:
    value = int(os.environ.get(name, "4"))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


PROCESS_CONTAINERS = _limit("CYTEBASE_PROCESS_CONTAINERS")
_SHA = (
    subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True
    ).strip()
    if modal.is_local()
    else "remote"
)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "aria2",
        "ca-certificates",
        "build-essential",
        "git",
        "libfftw3-dev",
        "libmetis-dev",
        "libtbb-dev",
    )
    .uv_sync(groups=["cytebase"], frozen=True, extra_options="--no-default-groups")
    .add_local_python_source(
        "scarf",
        copy=True,
        ignore=lambda path: "datasets" in path.parts or path.suffix != ".py",
    )
    .env(
        {
            "CYTEBASE_PIPELINE_VERSION": _SHA,
            "CYTEBASE_DOWNLOAD_CONNECTIONS": str(download_connections()),
            "CYTEBASE_PROCESS_CONTAINERS": str(PROCESS_CONTAINERS),
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        }
    )
)


def _storage() -> Bucket:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Missing required environment variable: HF_TOKEN")
    return Bucket(token=token)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _progress_summary(counters: dict | None) -> str:
    if not counters:
        return ""
    parts = []
    completed, total = counters.get("completed"), counters.get("total")
    if completed is not None:
        if counters.get("unit") == "bytes":
            expected = f"{total / 1024**3:.2f}" if total is not None else "?"
            label = "hashed" if counters.get("phase") == "checksum" else "downloaded"
            parts.append(f"{label}={completed / 1024**3:.2f}/{expected} GiB")
        else:
            expected = f"{total:,}" if total is not None else "?"
            parts.append(f"checked={completed:,}/{expected} {counters.get('unit', '')}")
    if counters.get("percent") is not None:
        parts.append(f"{counters['percent']:.1f}%")
    if (
        counters.get("phase") == "checksum"
        and counters.get("downloadedBytes") is not None
    ):
        parts.append(f"downloaded={counters['downloadedBytes'] / 1024**3:.2f} GiB")
    for key in ("activeConnections", "phase", "storedValuesChecked", "message"):
        if counters.get(key) is not None:
            parts.append(f"{key}={counters[key]}")
    if counters.get("downloadSpeedBytesPerSecond") is not None:
        parts.append(
            f"speed={counters['downloadSpeedBytesPerSecond'] / 1024**2:.1f} MiB/s"
        )
    return " ".join(parts)


def _owner(storage: Bucket, run_id: str, key: str, call_id: str | None) -> None:
    state = storage.read_json(RUN_PATH) or {}
    child = state.get("children", {}).get(key, {})
    if (
        state.get("runId") != run_id
        or state.get("state") != "running"
        or child.get("state") not in {"pending", "running"}
        or child.get("callId") not in {None, call_id}
    ):
        raise RuntimeError("This worker no longer owns the current pipeline stage")
    if not child:
        raise RuntimeError("Pipeline has no reservation for this worker")


@contextmanager
def _progress(record: DatasetRecord, stage: str) -> Iterator[Callable[..., None]]:
    """Progress is display-only; expiry never changes durable dataset readiness."""
    key = f"{record.runId}:{record.cytebaseId}:{stage}"
    lock, publish_lock, stopped = Lock(), Lock(), Event()
    started = stage_started = monotonic()
    value: dict[str, Any] = {
        "runId": record.runId,
        "callId": record.callId,
        "stage": stage,
        "attempt": record.attempt,
        "heartbeatAt": _now(),
        "progress": None,
    }

    def publish() -> None:
        # Serialize writes so a delayed heartbeat cannot replace a newer stage.
        with publish_lock:
            with lock:
                value["heartbeatAt"] = _now()
                snapshot = dict(value)
                stage_seconds = monotonic() - stage_started
            logger.info(
                "dataset=%s attempt=%s stage=%s elapsed=%.1fs stageElapsed=%.1fs %s",
                record.cytebaseId,
                record.attempt,
                snapshot["stage"],
                monotonic() - started,
                stage_seconds,
                _progress_summary(snapshot["progress"]),
            )
            try:
                progress_store.put(key, snapshot)
            except Exception as error:
                logger.warning("Progress update failed: %s", error_message(error))

    def update(name: str, **counters: Any) -> None:
        nonlocal stage_started
        completed, total = counters.get("completed"), counters.get("total")
        counters["percent"] = (
            min(100, 100 * completed / total)
            if completed is not None and total
            else None
        )
        with lock:
            changed = name != value["stage"]
            if changed:
                logger.info(
                    "dataset=%s stage transition %s -> %s after %.1fs; %s",
                    record.cytebaseId,
                    value["stage"],
                    name,
                    monotonic() - stage_started,
                    _progress_summary(value["progress"]),
                )
                stage_started = monotonic()
            value.update(stage=name, progress=counters)
        if changed:
            publish()

    def heartbeat() -> None:
        while not stopped.wait(15):
            publish()

    publish()
    thread = Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield update
    finally:
        stopped.set()
        thread.join()


def _cleanup_local(
    workspace: TemporaryDirectory,
    record: DatasetRecord,
    progress: Callable[..., None],
) -> None:
    """Release this worker's files without hiding the result of publication."""
    started = monotonic()
    try:
        progress("cleaning_local", message="Removing temporary source and store files")
    except Exception as error:
        logger.warning("Cleanup progress failed: %s", error_message(error))
    try:
        workspace.cleanup()
    except Exception as error:
        logger.warning("Local workspace cleanup failed: %s", error_message(error))
    finally:
        record.timings["cleanupSeconds"] = monotonic() - started


def _run_dataset(
    record: DatasetRecord,
    request: dict,
    storage: Bucket,
    progress: Callable[..., None],
    check: Callable[[], None],
) -> dict:
    from .build import build_local, publish_store, replacement_paths
    from .download import download_h5ad

    if (
        record.status == "ready"
        and record.processedVersionId == record.latestVersionId
        and record.zarrUri is not None
        and not request.get("force", False)
    ):
        return {
            "outcome": "skipped",
            "message": "The registered version already has a ready Scarf store",
        }
    progress("preflight", message="Checking source metadata and replacement approval")
    raw = storage.read_json(
        f"{dataset_prefix(record.cytebaseId)}/cellxgene/dataset.json"
    )
    if (
        raw is None
        or raw.get("dataset_id") != str(record.datasetId)
        or raw.get("dataset_version_id") != str(record.latestVersionId)
    ):
        raise ValueError("Registered source metadata is missing or has changed")
    missing = sorted(
        set(replacement_paths(record, storage))
        - set(request.get("approvedDeletionPaths", []))
    )
    if missing:
        return {
            "outcome": "needsApproval",
            "message": "Review these exact generated paths and resubmit with approvedDeletionPaths",
            "deletionPaths": missing,
        }
    check()
    if record.status != "ready":
        record.status = "processing"
    record.needsInput = None
    record.updatedAt = datetime.now(UTC)
    storage.write_json(
        f"{dataset_prefix(record.cytebaseId)}/dataset.json",
        record.model_dump(mode="json"),
    )
    workspace = TemporaryDirectory(prefix="cytebase-process-")
    try:
        source = Path(workspace.name) / "source.h5ad"
        store = Path(workspace.name) / "data.zarr"
        started = monotonic()
        try:
            size, checksum = download_h5ad(
                record.sourceUrl,
                source,
                record.sourceBytes,
                progress=progress,
                timings=record.timings,
            )
        finally:
            record.timings["downloadSeconds"] = monotonic() - started
        logger.info(
            "dataset=%s download completed: %.2f GiB in %.1fs; timings=%s",
            record.cytebaseId,
            size / 1024**3,
            record.timings["downloadSeconds"],
            record.timings,
        )
        check()
        manifest, converted = build_local(
            record, source, store, raw, size, checksum, progress
        )
        if converted["status"] == "done":
            try:
                source.unlink()
            except OSError as error:
                logger.warning("Local source cleanup failed: %s", error_message(error))
        return publish_store(
            record, request, storage, store, manifest, converted, progress, check
        )
    finally:
        _cleanup_local(workspace, record, progress)


def _execute(cytebase_id: str, run_id: str, request: dict) -> dict:
    storage = _storage()
    call_id = modal.current_function_call_id()
    key = f"{cytebase_id}:process"

    def check() -> None:
        _owner(storage, run_id, key, call_id)

    check()
    record = load_record(storage, cytebase_id)
    state = storage.read_json(RUN_PATH)
    if state is None:
        raise RuntimeError("Pipeline state is missing")
    if str(record.latestVersionId) != state["children"][key]["datasetVersionId"]:
        raise ValueError("Registered version changed after submission")
    record.attempt += 1
    record.runId, record.callId, record.stage = run_id, call_id, "process"
    record.pipelineVersion = os.environ["CYTEBASE_PIPELINE_VERSION"]
    record.stageOutcome, record.error = "running", None
    record.startedAt = datetime.now(UTC)
    record.timings = {}
    logger.info(
        "Dataset worker started: dataset=%s version=%s run=%s call=%s attempt=%s",
        cytebase_id,
        record.latestVersionId,
        run_id,
        call_id,
        record.attempt,
    )
    storage.write_json(
        f"{dataset_prefix(cytebase_id)}/dataset.json", record.model_dump(mode="json")
    )
    started = monotonic()
    try:
        with _progress(record, "process") as progress:
            storage.progress = progress
            result = _run_dataset(record, request, storage, progress, check)
    except Exception as error:
        message = error_message(error)
        logger.exception("Dataset worker failed: dataset=%s %s", cytebase_id, message)
        result = {"outcome": "failed", "message": message}
        record.status = "failed"
    finally:
        storage.progress = None
    record.stageOutcome = result["outcome"]
    if result["outcome"] not in {"succeeded", "skipped"}:
        record.error = result.get("message")
    record.timings["processSeconds"] = monotonic() - started
    record.updatedAt = datetime.now(UTC)
    check()
    storage.write_json(
        f"{dataset_prefix(cytebase_id)}/dataset.json", record.model_dump(mode="json")
    )
    logger.info(
        "Dataset worker finished: dataset=%s outcome=%s status=%s timings=%s message=%s",
        cytebase_id,
        result["outcome"],
        record.status,
        {key: round(value, 2) for key, value in record.timings.items()},
        result.get("message", ""),
    )
    return result | {
        "cytebaseId": cytebase_id,
        "status": record.status,
        "record": record.model_dump(mode="json"),
    }


@app.function(
    image=image,
    secrets=[secret],
    cpu=1,
    memory=4096,
    timeout=86400,
    retries=0,
    max_containers=1,
)
def build_catalog(request: dict, run_id: str) -> dict:
    from .catalog import run_catalog

    storage = _storage()

    def check() -> None:
        _owner(storage, run_id, "catalog", modal.current_function_call_id())

    check()
    started = monotonic()
    logger.info("Catalog worker started: run=%s", run_id)
    try:
        result = run_catalog(request, storage, check)
    except Exception:
        logger.exception("Catalog worker failed: run=%s", run_id)
        raise
    logger.info(
        "Catalog worker finished: run=%s elapsed=%.1fs", run_id, monotonic() - started
    )
    return result


@app.function(
    image=image,
    secrets=[secret],
    cpu=4,
    memory=16384,
    timeout=86400,
    retries=0,
    max_containers=PROCESS_CONTAINERS,
)
def process_dataset(cytebase_id: str, run_id: str, request: dict) -> dict:
    return _execute(cytebase_id, run_id, request)


def _reset(storage: Bucket, request: dict) -> dict:
    state = storage.read_json(RUN_PATH) or {}
    if not request.get("workersDrained") or state.get("runId") != request.get(
        "expectedRunId"
    ):
        raise ValueError(
            "Reset requires the exact expected run ID and explicit workersDrained confirmation"
        )
    ids = [
        state.get("callId"),
        *(child.get("callId") for child in state.get("children", {}).values()),
    ]
    for call_id in filter(None, ids):
        try:
            modal.FunctionCall.from_id(call_id).get(timeout=0)
        except modal.exception.FunctionTimeoutError:
            pass
        except TimeoutError as error:
            raise ValueError(
                f"Call {call_id} is still running; drain workers before reset"
            ) from error
        except Exception as error:
            # Modal re-raises original worker exceptions, not just RemoteError.
            # Failed or unavailable results rely on the required operator drain
            # confirmation; no elapsed-time inference authorizes this reset.
            logger.warning(
                "Using explicit worker-drain confirmation for call %s: %s",
                call_id,
                error_message(error),
            )
    affected = {
        child["cytebaseId"]
        for child in state.get("children", {}).values()
        if child.get("cytebaseId")
    }
    for key in affected:
        record = load_record(storage, key)
        if record.runId == state["runId"] and record.stageOutcome == "running":
            record.stageOutcome = "failed"
            record.error = (
                "Interrupted run reset after explicit worker-drain confirmation"
            )
            if record.status == "processing":
                record.status = "failed"
            record.updatedAt = datetime.now(UTC)
            storage.write_json(
                f"{dataset_prefix(key)}/dataset.json", record.model_dump(mode="json")
            )
    state.update(state="reset", updatedAt=_now())
    storage.write_json(RUN_PATH, state)
    return {"runId": state["runId"], "state": "reset"}


@app.function(
    image=image,
    secrets=[secret],
    cpu=1,
    memory=2048,
    timeout=86400,
    retries=0,
    max_containers=1,
)
async def run_pipeline(action: str, request: dict) -> dict:
    """Queue submissions, but advance datasets independently inside each submission."""
    storage = _storage()
    if action == "reset":
        return await asyncio.to_thread(_reset, storage, request)
    if action not in {"register", "catalog", "process"}:
        raise ValueError(f"Unknown pipeline action: {action}")
    previous = await asyncio.to_thread(storage.read_json, RUN_PATH) or {}
    if previous and previous.get("state") not in {"completed", "failed", "reset"}:
        raise RuntimeError(
            f"Run {previous.get('runId')} has unresolved workers; drain and explicitly reset it before resubmitting"
        )
    groups = {}
    if action == "process":
        job = ProcessRequest.model_validate(request)
        keys = await asyncio.to_thread(select_dataset_ids, job, storage)
        groups = {key: request for key in keys}
    state: dict[str, Any] = {
        "runId": modal.current_function_call_id(),
        "callId": modal.current_function_call_id(),
        "action": action,
        "state": "running",
        "startedAt": _now(),
        "updatedAt": _now(),
        "children": {},
    }
    await asyncio.to_thread(storage.write_json, RUN_PATH, state)
    logger.info(
        "Pipeline started: run=%s action=%s datasets=%s",
        state["runId"],
        action,
        len(groups),
    )
    lock = asyncio.Lock()
    uncertain = False

    async def invoke(
        function: modal.Function[..., dict, Any],
        key: str,
        args: tuple,
        reservation: dict,
    ) -> dict:
        nonlocal uncertain
        try:
            async with lock:
                state["children"][key] = reservation | {
                    "callId": None,
                    "state": "pending",
                }
                state["updatedAt"] = _now()
                await asyncio.to_thread(storage.write_json, RUN_PATH, state)
                call = await function.spawn.aio(*args)
                logger.info(
                    "Worker submitted: run=%s work=%s call=%s",
                    state["runId"],
                    key,
                    call.object_id,
                )
                state["children"][key].update(callId=call.object_id, state="running")
                await asyncio.to_thread(storage.write_json, RUN_PATH, state)
            result = await call.get.aio()
            async with lock:
                state["children"][key]["state"] = result.get("outcome", "succeeded")
                state["updatedAt"] = _now()
                await asyncio.to_thread(storage.write_json, RUN_PATH, state)
            return result
        except Exception:
            logger.exception(
                "Worker failed or result unavailable: run=%s work=%s",
                state["runId"],
                key,
            )
            uncertain = True
            raise

    async def publish(payload: dict) -> dict:
        return await invoke(
            build_catalog, "catalog", (payload, state["runId"]), {"stage": "catalog"}
        )

    results: list[dict] = []
    dirty: dict[str, dict] = {}
    catalog_result, catalog_error = None, None
    try:
        if action in {"register", "catalog"}:
            catalog_result = await publish(request if action == "register" else {})
        else:
            # Refresh before processing without scanning every remote dataset on
            # subsequent snapshots. Only completed stage records are merged.
            catalog_result = await publish({"updates": []})

            async def one(key: str, payload: dict) -> dict:
                try:
                    record = await asyncio.to_thread(load_record, storage, key)
                    arguments = payload | {
                        "approvedDeletionPaths": [
                            path
                            for path in payload.get("approvedDeletionPaths", [])
                            if path.startswith(f"{dataset_prefix(key)}/")
                        ],
                    }
                    result = await invoke(
                        process_dataset,
                        f"{key}:process",
                        (key, state["runId"], arguments),
                        {
                            "cytebaseId": key,
                            "datasetVersionId": str(record.latestVersionId),
                            "stage": "process",
                        },
                    )
                    dirty[key] = result.pop("record")
                    return result
                except Exception as error:
                    return {
                        "cytebaseId": key,
                        "outcome": "failed",
                        "message": error_message(error),
                    }

            pending = {
                asyncio.create_task(one(key, payload))
                for key, payload in groups.items()
            }
            last_publish = monotonic()
            while pending:
                done, pending = await asyncio.wait(
                    pending,
                    timeout=max(0.1, 60 - (monotonic() - last_publish)),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                results.extend(task.result() for task in done)
                if monotonic() - last_publish >= 60:
                    if dirty and not uncertain:
                        updates, dirty = list(dirty.values()), {}
                        try:
                            catalog_result = await publish({"updates": updates})
                        except Exception as error:
                            catalog_error = error_message(error)
                    last_publish = monotonic()
            if dirty and not uncertain:
                catalog_result = await publish({"updates": list(dirty.values())})
    except Exception as error:
        catalog_error = error_message(error)
    unfinished = any(
        child.get("state") in {"pending", "running", "unknown"}
        for child in state["children"].values()
    )
    state.update(
        state="blocked"
        if uncertain or unfinished
        else "failed"
        if catalog_error
        else "completed",
        updatedAt=_now(),
    )
    await asyncio.to_thread(storage.write_json, RUN_PATH, state)
    result: dict[str, Any] = {
        "callId": state["runId"],
        "state": state["state"],
        "datasets": results,
        "catalog": catalog_result,
        "error": catalog_error,
        "successes": [
            row["cytebaseId"]
            for row in results
            if row["outcome"] in {"succeeded", "skipped"}
        ],
        "failures": [
            row["cytebaseId"]
            for row in results
            if row["outcome"] not in {"succeeded", "skipped"}
        ],
    }
    if action == "register" and catalog_result:
        result["datasets"] = catalog_result.get("registeredDatasets", [])
        result["successes"] = [
            row["collection_id"] for row in catalog_result["registeredCollections"]
        ]
        result["failures"] = [
            row["collectionId"] for row in catalog_result["failedCollections"]
        ]
    logger.info(
        "Pipeline finished: run=%s action=%s state=%s successes=%s failures=%s error=%s",
        state["runId"],
        action,
        state["state"],
        len(result["successes"]),
        len(result["failures"]),
        catalog_error,
    )
    return result


def create_web_app() -> FastAPI:
    web = FastAPI(title="Cytebase pipeline (development)")

    @web.exception_handler(405)
    async def missing_endpoint(_request: Request, _error: Exception) -> JSONResponse:
        # A generic dataset GET route must not claim unsupported POST endpoints.
        return JSONResponse(status_code=404, content={"detail": "Not Found"})

    def submit(action: str, payload: dict) -> dict:
        call = run_pipeline.spawn(action, payload)
        return {"callId": call.object_id}

    @web.post("/collections/register", status_code=202)
    def register(job: RegisterRequest) -> dict:
        return submit("register", job.model_dump(mode="json"))

    @web.get("/collections")
    def collections() -> dict:
        from .catalog import list_collection_ids

        return {"collectionIds": list_collection_ids()}

    @web.post("/datasets/process", status_code=202)
    def process(job: ProcessRequest) -> dict:
        try:
            keys = select_dataset_ids(job, _storage())
        except ValueError as error:
            raise HTTPException(422, detail=error_message(error)) from error
        return submit("process", job.model_dump(mode="json")) | {
            "datasets": [
                {"cytebaseId": key, "statusUrl": f"/datasets/{key}"} for key in keys
            ]
        }

    @web.post("/catalog/build", status_code=202)
    def catalog() -> dict:
        return submit("catalog", {})

    @web.get("/jobs/{call_id}", response_model=None)
    def job(call_id: str) -> Any:
        try:
            return modal.FunctionCall.from_id(call_id).get(timeout=0)
        except (
            modal.exception.OutputExpiredError,
            modal.exception.NotFoundError,
            modal.exception.InvalidError,
        ) as error:
            raise HTTPException(
                404, detail="Job result is missing or expired"
            ) from error
        except modal.exception.FunctionTimeoutError as error:
            raise HTTPException(500, detail=error_message(error)) from error
        except TimeoutError:
            return JSONResponse(status_code=202, content={"status": "pending"})
        except Exception as error:
            raise HTTPException(500, detail=error_message(error)) from error

    @web.get("/datasets/{cytebase_id}")
    def dataset(cytebase_id: str, includeReplacementPaths: bool = False) -> dict:
        storage = _storage()
        try:
            record = load_record(storage, cytebase_id)
        except FileNotFoundError as error:
            raise HTTPException(404, detail=error_message(error)) from error
        except ValueError as error:
            raise HTTPException(422, detail=error_message(error)) from error
        status = {
            key: getattr(record, key)
            for key in (
                "status",
                "stage",
                "stageOutcome",
                "runId",
                "callId",
                "attempt",
                "updatedAt",
                "error",
                "needsInput",
            )
        }
        if record.stageOutcome == "running":
            status.update(
                progress_store.get(f"{record.runId}:{record.cytebaseId}:{record.stage}")
                or {}
            )
        result: dict[str, Any] = {
            "dataset": record.model_dump(mode="json"),
            "status": status,
        }
        if includeReplacementPaths:
            from .build import replacement_paths

            result["replacementPaths"] = replacement_paths(record, storage)
        return result

    @web.get("/health")
    def health() -> dict:
        return {"ok": True}

    return web


@app.function(image=image, secrets=[secret], max_containers=1)
@modal.asgi_app()
def web_app() -> FastAPI:
    return create_web_app()
