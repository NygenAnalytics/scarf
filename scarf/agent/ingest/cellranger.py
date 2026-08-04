"""Cell Ranger H5 and directory ingest handlers."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .common import finish, require_zarr_path, resolve_modality_choice
from .result import IngestResult


def ingest_cellranger(
    path: Path,
    *,
    format_name: str,
    reader_class_name: str,
    zarrPath: str | Path | None,
    model: Any | None,
    directions: Mapping[str, Any],
    notes: list[str],
) -> IngestResult:
    from ...readers import CrDirReader, CrH5Reader
    from ...writers import CrToZarr

    reader_cls = CrH5Reader if reader_class_name == "CrH5Reader" else CrDirReader
    reader = reader_cls(str(path))
    decision = None
    rename_assays: dict[str, str] = dict(directions.get("renameAssays") or {})

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

    zarr_path = require_zarr_path(zarrPath, format_name=format_name)
    writer = CrToZarr(reader, zarr_loc=zarr_path)
    writer.dump()
    return finish(
        format_name=format_name,
        zarr_path=zarr_path,
        notes=notes,
        convert_actions=[
            {
                "op": "CrToZarr",
                "path": str(path),
                "zarrPath": zarr_path,
                "readerClass": reader_class_name,
                "renameAssays": rename_assays or None,
            }
        ],
        action_labels=["convert_cellranger", "open_datastore"],
        default_assay=directions.get("defaultAssay"),
        decision=decision,
    )
