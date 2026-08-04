"""H5AD ingest handler."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .common import finish, require_zarr_path, resolve_modality_choice
from .result import IngestResult, needs_input


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
    zarrPath: str | Path | None,
    model: Any | None,
    directions: Mapping[str, Any],
    notes: list[str],
) -> IngestResult:
    from ...readers import H5adReader, inspect_h5ad
    from ...writers import H5adToZarr

    forced_matrix = directions.get("matrixKey")
    if forced_matrix is not None:
        forced_matrix = str(forced_matrix)
        try:
            inspection = inspect_h5ad(str(path), matrix_key=forced_matrix)
        except ValueError as exc:
            return IngestResult(
                status="failed",
                format="h5ad",
                notes=[*notes, str(exc)],
            )
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
                evidence_ids=[f"matrix:{key}" for key in inspection.matrixCandidates],
                notes=[
                    *notes,
                    "Prenormalized-only H5AD inputs are not imported silently",
                ],
            )

    assay_name_map = dict(directions.get("assayNameMap") or {})
    decision = None
    if inspection.assaySplitKey and "ADT" in inspection.suggestedAssays:
        choice, decision, blocked = resolve_modality_choice(
            model=model,
            directions=directions,
            feature_names=_antibody_names(inspection),
            format_name="h5ad",
        )
        if blocked is not None:
            blocked.notes = [*notes, *blocked.notes]
            return blocked
        if choice == "HTO":
            assay_name_map.setdefault("Antibody Capture", "HTO")
            notes.append("Mapped Antibody Capture to HTO")

    zarr_path = require_zarr_path(zarrPath, format_name="h5ad")
    reader = H5adReader.from_inspect(inspection)
    try:
        writer_kwargs: dict[str, Any] = {"zarr_loc": zarr_path}
        if inspection.assaySplitKey is not None:
            writer_kwargs["assay_split_key"] = inspection.assaySplitKey
            if assay_name_map:
                writer_kwargs["assay_name_map"] = assay_name_map
        else:
            writer_kwargs["assay_name"] = directions.get("assayName") or "RNA"
        writer = H5adToZarr(reader, **writer_kwargs)
        writer.dump()
    finally:
        reader.h5.close()

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
            {
                "op": "H5adToZarr",
                "path": str(path),
                "zarrPath": zarr_path,
                "assaySplitKey": inspection.assaySplitKey,
                "assayNameMap": assay_name_map or None,
                "assayName": None
                if inspection.assaySplitKey is not None
                else writer_kwargs.get("assay_name"),
            },
        ],
        action_labels=["inspect_h5ad", "convert_h5ad", "open_datastore"],
        default_assay=directions.get("defaultAssay"),
        decision=decision,
    )
