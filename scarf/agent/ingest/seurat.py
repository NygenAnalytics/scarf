"""Seurat ingest handler."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .common import finish, require_zarr_path
from .result import IngestResult


def ingest_seurat(
    path: Path,
    *,
    zarrPath: str | Path | None,
    directions: Mapping[str, Any],
    notes: list[str],
) -> IngestResult:
    from ...readers import SeuratReader
    from ...writers import SeuratToZarr

    zarr_path = require_zarr_path(zarrPath, format_name="seurat")
    reader = SeuratReader(str(path))
    try:
        writer = SeuratToZarr(reader, zarr_loc=zarr_path)
        writer.dump()
    finally:
        reader.close()
    return finish(
        format_name="seurat",
        zarr_path=zarr_path,
        notes=notes,
        convert_actions=[
            {"op": "SeuratToZarr", "path": str(path), "zarrPath": zarr_path}
        ],
        action_labels=["convert_seurat", "open_datastore"],
        default_assay=directions.get("defaultAssay"),
    )
