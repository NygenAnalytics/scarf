"""Run Leiden clustering in a child process so the parent container stays live.

Modal's runner heartbeat interval is about 900 seconds and is not configurable.
``leidenalg`` holds the GIL for the whole of a large partition, so a parent that
called it inline would emit no heartbeat or progress until it returned.
``profiling.stages`` polls this child every 30 seconds and warns at 1800 seconds
without killing it. Keep this indirection instead of trying to extend the
heartbeat threshold.
"""

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from scarf import configure_output

from profiling.config import StageResources, WorkflowParameters
from profiling.stages import _open_datastore

configure_output(progress=False, timestamps=True)


def _write_status(statusPath: Path, payload: dict[str, Any]) -> None:
    statusPath.write_text(json.dumps(payload), encoding="utf-8")


def run_leiden_worker(requestPath: Path) -> None:
    request = json.loads(requestPath.read_text(encoding="utf-8"))
    if not isinstance(request, dict):
        raise ValueError("Leiden worker request must be a JSON object")

    store_uri = str(request["storeUri"])
    workflow = WorkflowParameters.model_validate(request["workflow"])
    resources = StageResources.model_validate(request["resources"])
    status_path = Path(str(request["statusPath"]))

    print(
        f"[leiden_worker] START backend=leidenalg store={store_uri}",
        flush=True,
    )
    worker_started = time.perf_counter()
    setup_started = worker_started
    input_setup_seconds: float | None = None
    operation_started: float | None = None
    operation_seconds: float | None = None
    try:
        store = _open_datastore(
            store_uri,
            workflow,
            resources,
            initialize=False,
        )
        input_setup_seconds = time.perf_counter() - setup_started
        print(
            "[leiden_worker] datastore open; ENTER run_leiden_clustering",
            flush=True,
        )
        operation_started = time.perf_counter()
        store.run_leiden_clustering(
            from_assay=workflow.assayName,
            cell_key=workflow.cellKey,
            feat_key=workflow.hvgKey,
            resolution=workflow.leidenResolution,
            label=workflow.leidenLabel,
            random_seed=workflow.leidenSeed,
        )
        operation_seconds = time.perf_counter() - operation_started
        del store
    except BaseException as exc:
        now = time.perf_counter()
        if input_setup_seconds is None:
            input_setup_seconds = now - setup_started
        if operation_started is not None and operation_seconds is None:
            operation_seconds = now - operation_started
        error = f"{type(exc).__name__}: {exc}"
        _write_status(
            status_path,
            {
                "status": "error",
                "error": error,
                "inputSetupSeconds": input_setup_seconds,
                "operationSeconds": operation_seconds,
                "wholeWorkerSeconds": now - worker_started,
            },
        )
        print(f"[leiden_worker] ERROR {error}", flush=True)
        raise

    _write_status(
        status_path,
        {
            "status": "ok",
            "error": None,
            "inputSetupSeconds": input_setup_seconds,
            "operationSeconds": operation_seconds,
            "wholeWorkerSeconds": time.perf_counter() - worker_started,
        },
    )
    print("[leiden_worker] DONE run_leiden_clustering", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args(argv)
    run_leiden_worker(args.request)


if __name__ == "__main__":
    main()
