"""Seurat ingest handler."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .common import CONVERSION_DATA_ERRORS, finish
from .result import IngestResult, failed_from_exception


def ingest_seurat(
    path: Path,
    *,
    zarrPath: str | Path,
    directions: Mapping[str, Any],
    notes: list[str],
) -> IngestResult:
    from ...readers.seurat import SeuratReader
    from ...writers.seurat import SeuratToZarr

    zarr_path = str(zarrPath)
    overwrite = directions.get("overwrite") is True

    reader = None
    writer_started = False
    try:
        reader = SeuratReader(str(path))
        writer = SeuratToZarr(reader, zarr_loc=zarr_path)
        writer_started = True
        writer.dump()
    except CONVERSION_DATA_ERRORS as exc:
        return failed_from_exception(
            format_name="seurat",
            operation="convert seurat",
            exc=exc,
            zarr_path=zarr_path,
            notes=notes,
            partial_store=writer_started,
        )
    finally:
        if reader is not None:
            reader.close()

    convert_action: dict[str, Any] = {
        "op": "SeuratToZarr",
        "path": str(path),
        "zarrPath": zarr_path,
    }
    if overwrite:
        convert_action["overwrite"] = True

    return finish(
        format_name="seurat",
        zarr_path=zarr_path,
        notes=notes,
        convert_actions=[convert_action],
        action_labels=["convert_seurat", "open_datastore"],
        default_assay=directions.get("defaultAssay"),
    )
