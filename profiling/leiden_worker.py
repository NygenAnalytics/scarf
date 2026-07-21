import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from profiling.config import StageResources, WorkflowParameters
from profiling.stages import _open_datastore


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
    try:
        store = _open_datastore(
            store_uri,
            workflow,
            resources,
            initialize=False,
        )
        print(
            "[leiden_worker] datastore open; ENTER run_leiden_clustering",
            flush=True,
        )
        store.run_leiden_clustering(
            from_assay=workflow.assayName,
            cell_key=workflow.cellKey,
            feat_key=workflow.hvgKey,
            resolution=workflow.leidenResolution,
            label=workflow.leidenLabel,
            random_seed=workflow.leidenSeed,
        )
        del store
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        _write_status(status_path, {"status": "error", "error": error})
        print(f"[leiden_worker] ERROR {error}", flush=True)
        raise

    _write_status(status_path, {"status": "ok", "error": None})
    print("[leiden_worker] DONE run_leiden_clustering", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args(argv)
    run_leiden_worker(args.request)


if __name__ == "__main__":
    main()
