"""Run Paris clustering in a child process, matching the Leiden stage.

Modal's runner heartbeat interval is about 900 seconds and is not configurable,
and the earlier scikit-network implementation could block the parent for longer
than that. Scarf's Numba kernels release the GIL, so this isolation is no longer
strictly required, but profiling keeps it so both clustering stages report
progress and failures the same way. ``profiling.stages`` polls this child every
30 seconds and warns at 1800 seconds without killing it.
"""

import argparse
import hashlib
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


def run_paris_worker(requestPath: Path) -> None:
    request = json.loads(requestPath.read_text(encoding="utf-8"))
    if not isinstance(request, dict):
        raise ValueError("Paris worker request must be a JSON object")

    store_uri = str(request["storeUri"])
    workflow = WorkflowParameters.model_validate(request["workflow"])
    resources = StageResources.model_validate(request["resources"])
    status_path = Path(str(request["statusPath"]))

    print(f"[paris_worker] START store={store_uri}", flush=True)
    worker_started = time.perf_counter()
    cpu_started = time.process_time()
    setup_started = worker_started
    input_setup_seconds: float | None = None
    operation_started: float | None = None
    operation_seconds: float | None = None
    label_sha256: str | None = None
    cluster_count: int | None = None
    try:
        store = _open_datastore(
            store_uri,
            workflow,
            resources,
            initialize=False,
        )
        input_setup_seconds = time.perf_counter() - setup_started
        arguments: dict[str, Any] = {
            "from_assay": workflow.assayName,
            "cell_key": workflow.cellKey,
            "feat_key": workflow.hvgKey,
            "label": workflow.parisLabel,
            "n_clusters": workflow.parisNClusters,
        }
        if workflow.parisMinClusterSize is not None:
            arguments["min_cluster_size"] = workflow.parisMinClusterSize
        if request.get("invalidateCache") is True:
            arguments["force_recalc"] = True
        print(
            f"[paris_worker] cut nClusters={workflow.parisNClusters} "
            f"minClusterSize={workflow.parisMinClusterSize}",
            flush=True,
        )

        print("[paris_worker] datastore open; ENTER run_paris_clustering", flush=True)
        operation_started = time.perf_counter()
        result = store.run_paris_clustering(**arguments)
        operation_seconds = time.perf_counter() - operation_started
        labels = getattr(result, "labels", None)
        if labels is not None:
            label_sha256 = hashlib.sha256(labels.tobytes(order="C")).hexdigest()
        result_cluster_count = getattr(result, "n_clusters", None)
        if isinstance(result_cluster_count, int):
            cluster_count = result_cluster_count
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
                "processCpuSeconds": time.process_time() - cpu_started,
            },
        )
        print(f"[paris_worker] ERROR {error}", flush=True)
        raise

    _write_status(
        status_path,
        {
            "status": "ok",
            "error": None,
            "inputSetupSeconds": input_setup_seconds,
            "operationSeconds": operation_seconds,
            "wholeWorkerSeconds": time.perf_counter() - worker_started,
            "processCpuSeconds": time.process_time() - cpu_started,
            "labelSha256": label_sha256,
            "clusterCount": cluster_count,
        },
    )
    print("[paris_worker] DONE run_paris_clustering", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args(argv)
    run_paris_worker(args.request)


if __name__ == "__main__":
    main()
