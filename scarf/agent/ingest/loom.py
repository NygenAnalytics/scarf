"""Loom ingest handler."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .common import CONVERSION_DATA_ERRORS, finish
from .result import IngestResult, failed_from_exception


def ingest_loom(
    path: Path,
    *,
    zarrPath: str | Path,
    directions: Mapping[str, Any],
    notes: list[str],
) -> IngestResult:
    from ...readers.loom import LoomReader
    from ...writers.loom import LoomToZarr

    zarr_path = str(zarrPath)
    overwrite = directions.get("overwrite") is True

    reader_kwargs = {}
    if directions.get("cellNamesKey"):
        reader_kwargs["cell_names_key"] = directions["cellNamesKey"]
    if directions.get("featureNamesKey"):
        reader_kwargs["feature_names_key"] = directions["featureNamesKey"]

    reader = None
    writer_started = False
    try:
        reader = LoomReader(str(path), **reader_kwargs)
        writer = LoomToZarr(
            reader,
            zarr_loc=zarr_path,
            assay_name=directions.get("assayName") or "RNA",
        )
        writer_started = True
        writer.dump()
    except CONVERSION_DATA_ERRORS as exc:
        return failed_from_exception(
            format_name="loom",
            operation="convert loom",
            exc=exc,
            zarr_path=zarr_path,
            notes=notes,
            partial_store=writer_started,
        )
    finally:
        if reader is not None:
            reader.h5.close()

    convert_action: dict[str, Any] = {
        "op": "LoomToZarr",
        "path": str(path),
        "zarrPath": zarr_path,
    }
    if overwrite:
        convert_action["overwrite"] = True

    return finish(
        format_name="loom",
        zarr_path=zarr_path,
        notes=notes,
        convert_actions=[convert_action],
        action_labels=["convert_loom", "open_datastore"],
        default_assay=directions.get("defaultAssay"),
    )
