"""Loom ingest handler."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .common import finish, require_zarr_path
from .result import IngestResult


def ingest_loom(
    path: Path,
    *,
    zarrPath: str | Path | None,
    directions: Mapping[str, Any],
    notes: list[str],
) -> IngestResult:
    from ...readers import LoomReader
    from ...writers import LoomToZarr

    zarr_path = require_zarr_path(zarrPath, format_name="loom")
    reader_kwargs = {}
    if directions.get("cellNamesKey"):
        reader_kwargs["cell_names_key"] = directions["cellNamesKey"]
    if directions.get("featureNamesKey"):
        reader_kwargs["feature_names_key"] = directions["featureNamesKey"]
    reader = LoomReader(str(path), **reader_kwargs)
    try:
        writer = LoomToZarr(
            reader,
            zarr_loc=zarr_path,
            assay_name=directions.get("assayName") or "RNA",
        )
        writer.dump()
    finally:
        reader.h5.close()
    return finish(
        format_name="loom",
        zarr_path=zarr_path,
        notes=notes,
        convert_actions=[
            {"op": "LoomToZarr", "path": str(path), "zarrPath": zarr_path}
        ],
        action_labels=["convert_loom", "open_datastore"],
        default_assay=directions.get("defaultAssay"),
    )
