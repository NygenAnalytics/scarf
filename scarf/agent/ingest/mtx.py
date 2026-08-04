"""Matrix Market ingest handler."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .common import finish, require_zarr_path
from .result import IngestResult, needs_input


def ingest_mtx(
    path: Path,
    *,
    zarrPath: str | Path | None,
    directions: Mapping[str, Any],
    notes: list[str],
) -> IngestResult:
    from ...readers import MtxReader, inspect_mtx
    from ...writers import MtxToZarr

    candidates = inspect_mtx(path)
    if not candidates:
        return IngestResult(
            status="failed",
            format="mtx",
            notes=[*notes, f"No MTX matrix candidates found under {path}"],
        )
    if len(candidates) > 1 and directions.get("mtxIndex") is None:
        return needs_input(
            format_name="mtx",
            question="Multiple MTX layouts found. Which candidate index should be used?",
            options=[str(index) for index in range(len(candidates))],
            evidence_ids=[f"mtx:{index}" for index in range(len(candidates))],
            notes=[*notes, f"Found {len(candidates)} MTX candidates"],
        )
    index = int(directions.get("mtxIndex") or 0)
    reader = MtxReader(candidates[index])
    zarr_path = require_zarr_path(zarrPath, format_name="mtx")
    writer = MtxToZarr(reader, zarr_loc=zarr_path)
    writer.dump()
    return finish(
        format_name="mtx",
        zarr_path=zarr_path,
        notes=notes,
        convert_actions=[{"op": "MtxToZarr", "path": str(path), "zarrPath": zarr_path}],
        action_labels=["convert_mtx", "open_datastore"],
        default_assay=directions.get("defaultAssay"),
    )
