"""Ingest Scarf-supported inputs into a typed Zarr store."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .cellranger import ingest_cellranger
from .common import CONVERT_FORMATS, ensure_convert_destination
from .detect import detect_format
from .h5ad import ingest_h5ad
from .loom import ingest_loom
from .mtx import ingest_mtx
from .result import IngestResult, needs_input
from .seurat import ingest_seurat
from .zarr_store import ingest_zarr

__all__ = [
    "IngestResult",
    "detect_format",
    "ingest",
]


def ingest(
    *,
    path: str | Path,
    zarrPath: str | Path | None = None,
    model: Any | None = None,
    directions: Mapping[str, Any] | None = None,
) -> IngestResult:
    """Detect, inspect, convert, and open a Scarf store, or return NeedsInput."""
    source = Path(path)
    if not source.exists():
        return IngestResult(
            status="failed",
            notes=[f"Input path does not exist: {source}"],
        )

    direction_map = dict(directions or {})
    notes: list[str] = []
    format_name = str(direction_map.get("format") or detect_format(source))
    notes.append(f"Detected format: {format_name}")

    destination: str | None = None
    if format_name in CONVERT_FORMATS:
        using_default_destination = zarrPath is None
        preflight = ensure_convert_destination(
            source,
            zarrPath,
            direction_map,
            format_name=format_name,
        )
        if isinstance(preflight, IngestResult):
            preflight.notes = [*notes, *preflight.notes]
            return preflight
        destination = preflight
        if using_default_destination:
            notes.append(f"Using derived Zarr destination: {destination}")
        if direction_map.get("overwrite") is True:
            notes.append(f"Overwrite authorized for destination {destination}")

    if format_name == "zarr":
        return ingest_zarr(
            source,
            notes,
            default_assay=direction_map.get("defaultAssay"),
        )
    if format_name == "h5ad":
        assert destination is not None
        return ingest_h5ad(
            source,
            zarrPath=destination,
            model=model,
            directions=direction_map,
            notes=notes,
        )
    if format_name == "10x_h5":
        assert destination is not None
        return ingest_cellranger(
            source,
            format_name=format_name,
            reader_class_name="CrH5Reader",
            zarrPath=destination,
            model=model,
            directions=direction_map,
            notes=notes,
        )
    if format_name == "10x_dir":
        assert destination is not None
        return ingest_cellranger(
            source,
            format_name=format_name,
            reader_class_name="CrDirReader",
            zarrPath=destination,
            model=model,
            directions=direction_map,
            notes=notes,
        )
    if format_name == "mtx":
        assert destination is not None
        return ingest_mtx(
            source,
            zarrPath=destination,
            directions=direction_map,
            notes=notes,
        )
    if format_name == "loom":
        assert destination is not None
        return ingest_loom(
            source,
            zarrPath=destination,
            directions=direction_map,
            notes=notes,
        )
    if format_name == "seurat":
        assert destination is not None
        return ingest_seurat(
            source,
            zarrPath=destination,
            directions=direction_map,
            notes=notes,
        )
    if format_name == "csv":
        return needs_input(
            format_name="csv",
            question=(
                "CSV/TSV import needs explicit parameters "
                "(assay name, cell/feature id columns). Provide directions to continue."
            ),
            options=[],
            evidence_ids=[],
            notes=notes,
        )
    return IngestResult(
        status="failed",
        format=format_name,
        notes=[*notes, f"Unsupported input format for path {source}"],
    )
