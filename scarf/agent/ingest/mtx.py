"""Matrix Market ingest handler."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .common import CONVERSION_DATA_ERRORS, DATA_LAYOUT_ERRORS, finish
from .result import IngestResult, failed, failed_from_exception, needs_input


def _resolve_mtx_index(
    raw: Any,
    *,
    n_candidates: int,
) -> int | IngestResult:
    if raw is None:
        return 0
    if type(raw) is bool or not isinstance(raw, int | float | str):
        return failed(
            format_name="mtx",
            notes=[f"mtxIndex must be an integer index; got {raw!r}"],
        )
    try:
        if isinstance(raw, float) and not raw.is_integer():
            raise ValueError("non-integral float")
        index = int(raw)
    except (TypeError, ValueError):
        return failed(
            format_name="mtx",
            notes=[f"mtxIndex must be an integer index; got {raw!r}"],
        )
    if index < 0 or index >= n_candidates:
        return failed(
            format_name="mtx",
            notes=[
                f"mtxIndex {index} is out of range for {n_candidates} MTX candidates"
            ],
        )
    return index


def ingest_mtx(
    path: Path,
    *,
    zarrPath: str | Path,
    directions: Mapping[str, Any],
    notes: list[str],
) -> IngestResult:
    from ...readers.mtx import MtxReader, inspect_mtx
    from ...writers.cellranger import MtxToZarr

    zarr_path = str(zarrPath)
    overwrite = directions.get("overwrite") is True

    try:
        candidates = inspect_mtx(path)
    except DATA_LAYOUT_ERRORS as exc:
        return failed_from_exception(
            format_name="mtx",
            operation="inspect_mtx",
            exc=exc,
            zarr_path=zarr_path,
            notes=notes,
        )
    if not candidates:
        return failed(
            format_name="mtx",
            zarr_path=zarr_path,
            notes=[*notes, f"No MTX matrix candidates found under {path}"],
        )
    if len(candidates) > 1 and directions.get("mtxIndex") is None:
        return needs_input(
            format_name="mtx",
            question="Multiple MTX layouts found. Which candidate index should be used?",
            options=[str(index) for index in range(len(candidates))],
            evidence_ids=[f"mtx:{index}" for index in range(len(candidates))],
            notes=[*notes, f"Found {len(candidates)} MTX candidates"],
        )

    resolved = _resolve_mtx_index(
        directions.get("mtxIndex"),
        n_candidates=len(candidates),
    )
    if isinstance(resolved, IngestResult):
        resolved.notes = [*notes, *resolved.notes]
        resolved.zarrPath = zarr_path
        return resolved

    reader = None
    writer_started = False
    try:
        reader = MtxReader(candidates[resolved])
        writer = MtxToZarr(reader, zarr_loc=zarr_path)
        writer_started = True
        writer.dump()
    except CONVERSION_DATA_ERRORS as exc:
        return failed_from_exception(
            format_name="mtx",
            operation="convert mtx",
            exc=exc,
            zarr_path=zarr_path,
            notes=notes,
            partial_store=writer_started,
        )
    finally:
        if reader is not None:
            reader.close()

    convert_action: dict[str, Any] = {
        "op": "MtxToZarr",
        "path": str(path),
        "zarrPath": zarr_path,
        "mtxIndex": resolved,
    }
    if overwrite:
        convert_action["overwrite"] = True

    return finish(
        format_name="mtx",
        zarr_path=zarr_path,
        notes=notes,
        convert_actions=[convert_action],
        action_labels=["convert_mtx", "open_datastore"],
        default_assay=directions.get("defaultAssay"),
    )
