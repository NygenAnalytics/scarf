"""Local inspection, conversion, collection discovery, and explicit run reset."""

import argparse
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Inspect and validate a local H5AD")
    inspect.add_argument("source", type=Path)
    inspect.add_argument("--raw-data-location", choices=["X", "raw.X"])
    convert = commands.add_parser(
        "convert", help="Convert a local H5AD with its full inspection manifest"
    )
    convert.add_argument("source", type=Path)
    convert.add_argument("--manifest", type=Path, required=True)
    convert.add_argument("--output", type=Path, required=True)
    commands.add_parser("collection-ids", help="List public CELLxGENE collection IDs")
    inventory = commands.add_parser(
        "inventory",
        help="Save a public collection inventory with primary RNA selection",
    )
    inventory.add_argument("--output", type=Path, required=True)
    inventory.add_argument(
        "--bucket", help="Optionally join a verified Cytebase catalog snapshot"
    )
    reset = commands.add_parser(
        "reset-run", help="Reset an interrupted run only after draining its workers"
    )
    reset.add_argument("--expected-run-id", required=True)
    reset.add_argument("--workers-drained", action="store_true", required=True)
    reset.add_argument("--env", help="Modal environment containing the deployment")
    args = parser.parse_args()

    # Keep stdout machine-readable while preserving command diagnostics.
    with redirect_stdout(sys.stderr):
        result = _run_command(args)
    print(json.dumps(result, indent=2, allow_nan=False))


def _run_command(args: argparse.Namespace) -> dict:
    result: dict
    if args.command == "inspect":
        from .build import inspect_file

        result = inspect_file(args.source, args.raw_data_location)["manifest"]
    elif args.command == "convert":
        from .build import convert_local, verify_store

        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        store = args.output / "data.zarr"
        result = convert_local(args.source, store, manifest)
        if result["status"] == "done":
            result["verification"] = verify_store(str(store), manifest)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "scarf_ingest.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    elif args.command == "collection-ids":
        from .catalog import list_collection_ids

        result = {"collectionIds": list_collection_ids()}
    elif args.command == "inventory":
        from .inventory import build_inventory

        snapshot = build_inventory(args.output, bucket=args.bucket)
        result = {"output": str(args.output), **snapshot["summary"]}
        result["selectedDatasetsOverMillionCells"] = len(
            result["selectedDatasetsOverMillionCells"]
        )
    else:
        import modal

        function = modal.Function.from_name(
            "cytebase", "run_pipeline", environment_name=args.env
        )
        call = function.spawn(
            "reset",
            {
                "expectedRunId": args.expected_run_id,
                "workersDrained": args.workers_drained,
            },
        )
        result = {"callId": call.object_id}
    return result


if __name__ == "__main__":
    main()
