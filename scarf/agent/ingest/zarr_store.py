"""Open an existing Scarf Zarr store."""

from pathlib import Path

from .common import finish
from .result import IngestResult


def ingest_zarr(
    path: Path,
    notes: list[str],
    *,
    default_assay: str | None,
) -> IngestResult:
    return finish(
        format_name="zarr",
        zarr_path=str(path),
        notes=notes,
        convert_actions=[],
        action_labels=["summarize_zarr"],
        default_assay=default_assay,
        summary_mode="r",
    )
