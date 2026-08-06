"""Cell Ranger H5 and directory ingest handlers."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .common import CONVERSION_DATA_ERRORS, finish, resolve_modality_choice
from .result import IngestResult, failed_from_exception


def ingest_cellranger(
    path: Path,
    *,
    format_name: str,
    reader_class_name: str,
    zarrPath: str | Path,
    model: Any | None,
    directions: Mapping[str, Any],
    notes: list[str],
) -> IngestResult:
    from ...readers.cellranger import CrDirReader, CrH5Reader
    from ...writers.cellranger import CrToZarr

    zarr_path = str(zarrPath)
    overwrite = directions.get("overwrite") is True

    reader_cls = CrH5Reader if reader_class_name == "CrH5Reader" else CrDirReader
    reader = None
    writer_started = False
    decision = None
    rename_assays: dict[str, str] = dict(directions.get("renameAssays") or {})
    try:
        reader = reader_cls(str(path))
        assay_columns = list(reader.assayFeats.columns)
        if "ADT" in assay_columns and "HTO" not in assay_columns:
            adt_names = [str(name) for name in reader.feature_names("ADT")]
            choice, decision, blocked = resolve_modality_choice(
                model=model,
                directions=directions,
                feature_names=adt_names,
                format_name=format_name,
            )
            if blocked is not None:
                blocked.notes = [*notes, *blocked.notes]
                return blocked
            if choice == "HTO":
                rename_assays.setdefault("ADT", "HTO")
                notes.append("Renamed ADT assay to HTO")

        if rename_assays:
            reader.rename_assays(rename_assays)

        writer = CrToZarr(reader, zarr_loc=zarr_path)
        writer_started = True
        writer.dump()
    except CONVERSION_DATA_ERRORS as exc:
        return failed_from_exception(
            format_name=format_name,
            operation="convert cellranger",
            exc=exc,
            zarr_path=zarr_path,
            notes=notes,
            partial_store=writer_started,
        )
    finally:
        if reader is not None:
            reader.close()

    convert_action: dict[str, Any] = {
        "op": "CrToZarr",
        "path": str(path),
        "zarrPath": zarr_path,
        "readerClass": reader_class_name,
        "renameAssays": rename_assays or None,
    }
    if overwrite:
        convert_action["overwrite"] = True

    return finish(
        format_name=format_name,
        zarr_path=zarr_path,
        notes=notes,
        convert_actions=[convert_action],
        action_labels=["convert_cellranger", "open_datastore"],
        default_assay=directions.get("defaultAssay"),
        decision=decision,
    )
