import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from profiling.config import StageResources, WorkflowParameters
from profiling.stages import _open_datastore


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
    try:
        store = _open_datastore(
            store_uri,
            workflow,
            resources,
            initialize=False,
        )
        arguments: dict[str, Any] = {
            "from_assay": workflow.assayName,
            "cell_key": workflow.cellKey,
            "feat_key": workflow.hvgKey,
            "label": workflow.parisLabel,
            "n_clusters": workflow.parisNClusters,
        }
        if workflow.parisMinClusterSize is not None:
            arguments["min_cluster_size"] = workflow.parisMinClusterSize
        print(
            f"[paris_worker] cut nClusters={workflow.parisNClusters} "
            f"minClusterSize={workflow.parisMinClusterSize}",
            flush=True,
        )

        print("[paris_worker] datastore open; ENTER run_paris_clustering", flush=True)
        store.run_paris_clustering(**arguments)
        del store
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        _write_status(status_path, {"status": "error", "error": error})
        print(f"[paris_worker] ERROR {error}", flush=True)
        raise

    _write_status(status_path, {"status": "ok", "error": None})
    print("[paris_worker] DONE run_paris_clustering", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args(argv)
    run_paris_worker(args.request)


if __name__ == "__main__":
    main()
