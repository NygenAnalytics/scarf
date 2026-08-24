"""H5AD ingest handler."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .common import (
    CONVERSION_DATA_ERRORS,
    DATA_LAYOUT_ERRORS,
    finish,
    resolve_modality_choice,
)
from .result import IngestResult, failed_from_exception, needs_input


def _antibody_names(inspection: Any) -> list[str]:
    if "ADT" not in inspection.suggestedAssays:
        return []
    import h5py

    from ...readers._h5ad_inspect import _as_text, _read_column

    with h5py.File(inspection.h5adFn, mode="r") as h5:
        feature_node = h5.get(inspection.featureAttrsKey)
        if feature_node is None:
            return []
        types = _read_column(feature_node, inspection.assaySplitKey)
        names = _read_column(feature_node, inspection.featureNameKey)
        if types is None or names is None:
            return []
        return [
            _as_text(name)
            for feature_type, name in zip(types, names, strict=True)
            if _as_text(feature_type) == "Antibody Capture"
        ]


def ingest_h5ad(
    path: Path,
    *,
    zarrPath: str | Path,
    model: Any | None,
    directions: Mapping[str, Any],
    notes: list[str],
) -> IngestResult:
    from ...readers._h5ad_inspect import inspect_h5ad
    from ...readers.h5ad import H5adReader
    from ...writers.h5ad import H5adToZarr

    zarr_path = str(zarrPath)
    overwrite = directions.get("overwrite") is True

    forced_matrix = directions.get("matrixKey")
    try:
        if forced_matrix is not None:
            forced_matrix = str(forced_matrix)
            inspection = inspect_h5ad(str(path), matrix_key=forced_matrix)
            notes.append(
                f"Forced matrix {inspection.matrixKey} via directions "
                f"(integerLike={inspection.integerLike})"
            )
        else:
            inspection = inspect_h5ad(str(path))
            notes.append(
                f"Selected matrix {inspection.matrixKey} "
                f"(integerLike={inspection.integerLike})"
            )
            if not inspection.integerLike:
                return needs_input(
                    format_name="h5ad",
                    question=(
                        "No integer-like count matrix matched this H5AD layout. "
                        "Provide a raw counts matrix, or retry with "
                        'directions={"matrixKey": "<key>"} to force a candidate.'
                    ),
                    options=list(inspection.matrixCandidates),
                    evidence_ids=[
                        f"matrix:{key}" for key in inspection.matrixCandidates
                    ],
                    notes=[
                        *notes,
                        "Prenormalized-only H5AD inputs are not imported silently",
                    ],
                )
    except DATA_LAYOUT_ERRORS as exc:
        return failed_from_exception(
            format_name="h5ad",
            operation="inspect_h5ad",
            exc=exc,
            zarr_path=zarr_path,
            notes=notes,
        )

    assay_name_map = dict(directions.get("assayNameMap") or {})
    decision = None
    if inspection.assaySplitKey and "ADT" in inspection.suggestedAssays:
        try:
            antibody_names = _antibody_names(inspection)
        except DATA_LAYOUT_ERRORS as exc:
            return failed_from_exception(
                format_name="h5ad",
                operation="read antibody names",
                exc=exc,
                zarr_path=zarr_path,
                notes=notes,
            )
        choice, decision, blocked = resolve_modality_choice(
            model=model,
            directions=directions,
            feature_names=antibody_names,
            format_name="h5ad",
        )
        if blocked is not None:
            blocked.notes = [*notes, *blocked.notes]
            return blocked
        if choice == "HTO":
            assay_name_map.setdefault("Antibody Capture", "HTO")
            notes.append("Mapped Antibody Capture to HTO")

    writer_started = False
    writer_kwargs: dict[str, Any] = {"zarr_loc": zarr_path}
    reader = None
    try:
        reader = H5adReader.from_inspect(inspection)
        if inspection.assaySplitKey is not None:
            writer_kwargs["assay_split_key"] = inspection.assaySplitKey
            if assay_name_map:
                writer_kwargs["assay_name_map"] = assay_name_map
        else:
            writer_kwargs["assay_name"] = directions.get("assayName") or "RNA"
        writer = H5adToZarr(reader, **writer_kwargs)
        writer_started = True
        writer.dump()
    except CONVERSION_DATA_ERRORS as exc:
        return failed_from_exception(
            format_name="h5ad",
            operation="convert h5ad",
            exc=exc,
            zarr_path=zarr_path,
            notes=notes,
            partial_store=writer_started,
        )
    finally:
        if reader is not None:
            reader.h5.close()

    convert_action: dict[str, Any] = {
        "op": "H5adToZarr",
        "path": str(path),
        "zarrPath": zarr_path,
        "assaySplitKey": inspection.assaySplitKey,
        "assayNameMap": assay_name_map or None,
        "assayName": None
        if inspection.assaySplitKey is not None
        else writer_kwargs.get("assay_name"),
    }
    if overwrite:
        convert_action["overwrite"] = True

    return finish(
        format_name="h5ad",
        zarr_path=zarr_path,
        notes=notes,
        convert_actions=[
            {
                "op": "inspect_h5ad",
                "path": str(path),
                "matrixKey": inspection.matrixKey,
            },
            convert_action,
        ],
        action_labels=["inspect_h5ad", "convert_h5ad", "open_datastore"],
        default_assay=directions.get("defaultAssay"),
        decision=decision,
    )
