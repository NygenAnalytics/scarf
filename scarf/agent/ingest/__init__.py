"""Ingest Scarf-supported inputs into a typed Zarr store."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .cellranger import ingest_cellranger
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

    if format_name == "zarr":
        return ingest_zarr(
            source,
            notes,
            default_assay=direction_map.get("defaultAssay"),
        )
    if format_name == "h5ad":
        return ingest_h5ad(
            source,
            zarrPath=zarrPath,
            model=model,
            directions=direction_map,
            notes=notes,
        )
    if format_name == "10x_h5":
        return ingest_cellranger(
            source,
            format_name=format_name,
            reader_class_name="CrH5Reader",
            zarrPath=zarrPath,
            model=model,
            directions=direction_map,
            notes=notes,
        )
    if format_name == "10x_dir":
        return ingest_cellranger(
            source,
            format_name=format_name,
            reader_class_name="CrDirReader",
            zarrPath=zarrPath,
            model=model,
            directions=direction_map,
            notes=notes,
        )
    if format_name == "mtx":
        return ingest_mtx(
            source,
            zarrPath=zarrPath,
            directions=direction_map,
            notes=notes,
        )
    if format_name == "loom":
        return ingest_loom(
            source,
            zarrPath=zarrPath,
            directions=direction_map,
            notes=notes,
        )
    if format_name == "seurat":
        return ingest_seurat(
            source,
            zarrPath=zarrPath,
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
