from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import zarr

from .arrays import create_metadata_column
from .artifact_writer import (
    ArrayRequirement,
    finish_artifact,
    plan_artifact,
    reused_artifact_group,
    start_artifact,
)
from .artifacts import (
    ArtifactRef,
    ArtifactScope,
    ValueFingerprintBuilder,
    artifact_group,
    fingerprint_array,
    fingerprint_stored_arrays,
    fingerprint_stored_strings,
    fingerprint_strings,
    inspect_artifact,
)
from .errors import ArtifactErrorContextValue, ArtifactResolutionError
from .geometry import array_geometry
from .partition import row_band
from .types import as_zarr_array, as_zarr_group


_MISSING_COLUMN_PREFIX = "__scarf_missing__"


@dataclass(frozen=True, slots=True)
class ValidatedStoredSelection:
    """A complete selection artifact aligned with the current ordered row IDs."""

    ref: ArtifactRef
    values: zarr.Array
    row_ids: zarr.Array
    selected_count: int
    table_path: str


@dataclass(frozen=True, slots=True)
class StoredSelectionBlock:
    """One stored-mask block and its full-axis to compact-axis alignment."""

    start: int
    stop: int
    mask: np.ndarray
    selected_indices: np.ndarray
    compact_start: int
    compact_stop: int


@dataclass(frozen=True, slots=True)
class AlignedSelectionBlock:
    """One contiguous value block in its returned axis coordinates."""

    start: int
    stop: int
    values: np.ndarray


def _selection_context(
    ref: ArtifactRef,
    *,
    table_path: str,
    column: str | None = None,
) -> dict[str, ArtifactErrorContextValue]:
    context: dict[str, ArtifactErrorContextValue] = {
        "scope": ref.scope,
        "assay": ref.assay,
        "kind": ref.kind,
        "artifact_id": ref.artifact_id,
        "table": table_path,
    }
    if column is not None:
        context["column"] = column
    return context


def _stored_selection_summary(array: zarr.Array) -> tuple[str, int]:
    if array.ndim != 1 or np.dtype(array.dtype) != np.dtype(bool):
        raise TypeError("Stored selection columns must be one-dimensional booleans")
    builder = ValueFingerprintBuilder()
    builder.begin_array("values", array.shape, array.dtype)
    block_rows = row_band(array_geometry(array), unit="chunk", fallback=1)
    selected_count = 0
    for start in range(0, int(array.shape[0]), block_rows):
        stop = min(start + block_rows, int(array.shape[0]))
        block = np.asarray(array[start:stop], dtype=bool)
        builder.update_array_block("values", (start,), block)
        selected_count += int(np.count_nonzero(block))
    builder.end_array("values")
    return builder.hexdigest(), selected_count


def _stored_selection_fingerprint(array: zarr.Array) -> str:
    return _stored_selection_summary(array)[0]


def _selection_reuse_validator(
    expected_fingerprint: str,
) -> Callable[[ArtifactRef, zarr.Group], bool]:
    def validate(_ref: ArtifactRef, group: zarr.Group) -> bool:
        try:
            values = as_zarr_array(group["values"], name="values")
            return _stored_selection_fingerprint(values) == expected_fingerprint
        except (KeyError, TypeError, ValueError):
            return False

    return validate


def validate_stored_selection_integrity(
    root: zarr.Group,
    ref: ArtifactRef,
    *,
    kind: str,
    scope: ArtifactScope,
    assay: str | None,
    table_path: str,
    id_column: str = "ids",
) -> ValidatedStoredSelection:
    """Validate immutable selection payload and ordered row identity."""
    context = _selection_context(ref, table_path=table_path)
    if ref.kind != kind or ref.scope != scope or ref.assay != assay:
        raise ArtifactResolutionError(
            f"Expected {scope}-scoped {kind} artifact",
            code="artifact_reference_mismatch",
            context={
                **context,
                "expected_scope": scope,
                "expected_assay": assay,
                "expected_kind": kind,
                "actual_scope": ref.scope,
                "actual_assay": ref.assay,
                "actual_kind": ref.kind,
            },
        )
    try:
        status = inspect_artifact(root, ref)
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactResolutionError(
            f"{kind} artifact record is malformed",
            code="artifact_missing",
            context=context,
        ) from exc
    if not status.exists:
        raise ArtifactResolutionError(
            f"{kind} artifact does not exist",
            code="artifact_missing",
            context=context,
        )
    if not status.complete:
        raise ArtifactResolutionError(
            f"{kind} artifact is incomplete",
            code="artifact_incomplete",
            context=context,
        )
    if table_path not in root:
        raise ArtifactResolutionError(
            f"Selection table {table_path!r} is unavailable",
            code="selection_table_missing",
            context=context,
        )
    table = as_zarr_group(root[table_path], name=table_path)
    if id_column not in table:
        raise ArtifactResolutionError(
            f"Selection row identifier column {id_column!r} is unavailable",
            code="selection_row_ids_missing",
            context=context,
        )
    try:
        selection_group = artifact_group(root, ref)
    except (KeyError, TypeError) as exc:
        raise ArtifactResolutionError(
            f"{kind} artifact does not exist",
            code="artifact_missing",
            context=context,
        ) from exc
    if "values" not in selection_group:
        raise ArtifactResolutionError(
            f"{kind} artifact has no values",
            code="selection_values_missing",
            context=context,
        )
    try:
        stored_values = as_zarr_array(selection_group["values"], name="values")
        row_ids = as_zarr_array(table[id_column], name=id_column)
    except TypeError as exc:
        raise ArtifactResolutionError(
            f"{kind} selection payload is malformed",
            code="selection_values_changed",
            context=context,
        ) from exc
    expected_row_ids = (status.inputs or {}).get("ordered_row_ids_fingerprint")
    current_row_ids = (
        fingerprint_stored_strings(row_ids)
        if row_ids.ndim == 1 and row_ids.shape == stored_values.shape
        else None
    )
    if (
        row_ids.ndim != 1
        or row_ids.shape != stored_values.shape
        or not isinstance(expected_row_ids, str)
        or expected_row_ids != current_row_ids
    ):
        raise ArtifactResolutionError(
            f"{kind} row identity does not match its metadata table",
            code="row_identity_mismatch",
            context=context,
        )
    expected_values = (status.inputs or {}).get("values_fingerprint")
    try:
        stored_fingerprint, selected_count = _stored_selection_summary(stored_values)
    except TypeError as exc:
        raise ArtifactResolutionError(
            f"{kind} selection payload is malformed",
            code="selection_values_changed",
            context=context,
        ) from exc
    if not isinstance(expected_values, str) or expected_values != stored_fingerprint:
        raise ArtifactResolutionError(
            f"{kind} selection payload no longer matches its fingerprint",
            code="selection_values_changed",
            context=context,
        )
    if fingerprint_stored_strings(row_ids) != expected_row_ids:
        raise ArtifactResolutionError(
            f"{kind} row identity changed while the artifact was validated",
            code="row_identity_mismatch",
            context=context,
        )
    return ValidatedStoredSelection(
        ref=ref,
        values=stored_values,
        row_ids=row_ids,
        selected_count=selected_count,
        table_path=table_path,
    )


def validate_stored_selection_live_alias(
    root: zarr.Group,
    ref: ArtifactRef,
    *,
    kind: str,
    scope: ArtifactScope,
    assay: str | None,
    table_path: str,
    column: str,
    id_column: str = "ids",
) -> ValidatedStoredSelection:
    """Validate that a live boolean column still equals a selection artifact."""
    validated = validate_stored_selection_integrity(
        root,
        ref,
        kind=kind,
        scope=scope,
        assay=assay,
        table_path=table_path,
        id_column=id_column,
    )
    context = _selection_context(ref, table_path=table_path, column=column)
    table = as_zarr_group(root[table_path], name=table_path)
    if column not in table:
        raise ArtifactResolutionError(
            f"Selection source column {column!r} is unavailable",
            code="selection_column_missing",
            context=context,
        )
    try:
        current_values = as_zarr_array(table[column], name=column)
    except TypeError as exc:
        raise ArtifactResolutionError(
            f"Selection source column {column!r} is malformed",
            code="selection_values_changed",
            context=context,
        ) from exc
    stored_values = validated.values
    if (
        current_values.ndim != 1
        or np.dtype(current_values.dtype) != np.dtype(bool)
        or stored_values.shape != current_values.shape
    ):
        raise ArtifactResolutionError(
            f"Selection source column {column!r} no longer matches its artifact",
            code="selection_values_changed",
            context=context,
        )
    block_rows = min(
        row_band(array_geometry(stored_values), unit="chunk", fallback=1),
        row_band(array_geometry(current_values), unit="chunk", fallback=1),
    )
    for start in range(0, int(stored_values.shape[0]), block_rows):
        stop = min(start + block_rows, int(stored_values.shape[0]))
        if not np.array_equal(
            np.asarray(stored_values[start:stop], dtype=bool),
            np.asarray(current_values[start:stop], dtype=bool),
        ):
            raise ArtifactResolutionError(
                f"Selection source column {column!r} no longer matches its artifact",
                code="selection_values_changed",
                context=context,
            )
    return validated


def _selection_block_rows(
    selection: ValidatedStoredSelection,
    block_rows: int | None,
) -> int:
    chunk_rows = int(
        row_band(
            array_geometry(selection.values),
            unit="chunk",
            fallback=1,
        )
    )
    if block_rows is None:
        return chunk_rows
    requested = int(block_rows)
    if requested < 1:
        raise ValueError("block_rows must be >= 1")
    return min(requested, chunk_rows)


def _iter_validated_selection_blocks(
    selection: ValidatedStoredSelection,
    *,
    block_rows: int | None,
) -> Iterator[StoredSelectionBlock]:
    resolved_rows = _selection_block_rows(selection, block_rows)
    compact_start = 0
    length = int(selection.values.shape[0])
    for start in range(0, length, resolved_rows):
        stop = min(start + resolved_rows, length)
        mask = np.asarray(selection.values[start:stop], dtype=bool)
        selected_indices: np.ndarray = (
            np.flatnonzero(mask).astype(np.intp, copy=False) + start
        )
        compact_stop = compact_start + len(selected_indices)
        yield StoredSelectionBlock(
            start=start,
            stop=stop,
            mask=mask,
            selected_indices=selected_indices,
            compact_start=compact_start,
            compact_stop=compact_stop,
        )
        compact_start = compact_stop
    if compact_start != selection.selected_count:
        raise ArtifactResolutionError(
            "Selection values changed while they were being read",
            code="selection_values_changed",
            context=_selection_context(
                selection.ref,
                table_path=selection.table_path,
            ),
        )


def iter_stored_selection_blocks(
    root: zarr.Group,
    ref: ArtifactRef,
    *,
    kind: str,
    scope: ArtifactScope,
    assay: str | None,
    table_path: str,
    id_column: str = "ids",
    block_rows: int | None = None,
) -> Iterator[StoredSelectionBlock]:
    """Yield a validated stored mask with full and compact row coordinates."""
    selection = validate_stored_selection_integrity(
        root,
        ref,
        kind=kind,
        scope=scope,
        assay=assay,
        table_path=table_path,
        id_column=id_column,
    )
    yield from _iter_validated_selection_blocks(selection, block_rows=block_rows)


def read_stored_selection_mask(
    root: zarr.Group,
    ref: ArtifactRef,
    *,
    kind: str,
    scope: ArtifactScope,
    assay: str | None,
    table_path: str,
    id_column: str = "ids",
    block_rows: int | None = None,
) -> np.ndarray:
    """Read an explicitly requested selection mask after integrity validation."""
    selection = validate_stored_selection_integrity(
        root,
        ref,
        kind=kind,
        scope=scope,
        assay=assay,
        table_path=table_path,
        id_column=id_column,
    )
    output: np.ndarray = np.empty((int(selection.values.shape[0]),), dtype=bool)
    for block in _iter_validated_selection_blocks(selection, block_rows=block_rows):
        output[block.start : block.stop] = block.mask
    return output


def read_stored_selection_indices(
    root: zarr.Group,
    ref: ArtifactRef,
    *,
    kind: str,
    scope: ArtifactScope,
    assay: str | None,
    table_path: str,
    id_column: str = "ids",
    block_rows: int | None = None,
) -> np.ndarray:
    """Read selected full-axis indices after integrity validation."""
    selection = validate_stored_selection_integrity(
        root,
        ref,
        kind=kind,
        scope=scope,
        assay=assay,
        table_path=table_path,
        id_column=id_column,
    )
    output: np.ndarray = np.empty(selection.selected_count, dtype=np.intp)
    for block in _iter_validated_selection_blocks(selection, block_rows=block_rows):
        output[block.compact_start : block.compact_stop] = block.selected_indices
    return output


def iter_full_axis_selection_blocks(
    root: zarr.Group,
    ref: ArtifactRef,
    compact_values: zarr.Array | np.ndarray,
    *,
    fill_value: Any,
    kind: str,
    scope: ArtifactScope,
    assay: str | None,
    table_path: str,
    id_column: str = "ids",
    dtype: Any | None = None,
    block_rows: int | None = None,
) -> Iterator[AlignedSelectionBlock]:
    """Scatter compact selected-row values into bounded full-axis blocks."""
    selection = validate_stored_selection_integrity(
        root,
        ref,
        kind=kind,
        scope=scope,
        assay=assay,
        table_path=table_path,
        id_column=id_column,
    )
    source = (
        compact_values
        if isinstance(compact_values, zarr.Array)
        else np.asarray(compact_values)
    )
    if source.ndim < 1 or int(source.shape[0]) != selection.selected_count:
        raise ValueError(
            "Compact values must have one leading row per selected observation"
        )
    output_dtype = np.dtype(source.dtype if dtype is None else dtype)
    trailing_shape = tuple(int(size) for size in source.shape[1:])
    for block in _iter_validated_selection_blocks(selection, block_rows=block_rows):
        values = np.full(
            (block.stop - block.start, *trailing_shape),
            fill_value,
            dtype=output_dtype,
        )
        if block.compact_stop > block.compact_start:
            compact = np.asarray(
                source[block.compact_start : block.compact_stop],
                dtype=output_dtype,
            )
            values[block.mask] = compact
        yield AlignedSelectionBlock(
            start=block.start,
            stop=block.stop,
            values=values,
        )


def iter_selected_axis_selection_blocks(
    root: zarr.Group,
    ref: ArtifactRef,
    full_values: zarr.Array | np.ndarray,
    *,
    kind: str,
    scope: ArtifactScope,
    assay: str | None,
    table_path: str,
    id_column: str = "ids",
    block_rows: int | None = None,
) -> Iterator[AlignedSelectionBlock]:
    """Gather bounded full-axis values into compact selected-axis blocks."""
    selection = validate_stored_selection_integrity(
        root,
        ref,
        kind=kind,
        scope=scope,
        assay=assay,
        table_path=table_path,
        id_column=id_column,
    )
    source = (
        full_values if isinstance(full_values, zarr.Array) else np.asarray(full_values)
    )
    if source.ndim < 1 or int(source.shape[0]) != int(selection.values.shape[0]):
        raise ValueError("Full-axis values must align with the selection mask")
    for block in _iter_validated_selection_blocks(selection, block_rows=block_rows):
        values = np.asarray(source[block.start : block.stop])[block.mask]
        yield AlignedSelectionBlock(
            start=block.compact_start,
            stop=block.compact_stop,
            values=values,
        )


def fingerprint_selected_stored_strings(
    ids: zarr.Array,
    selection: zarr.Array,
) -> tuple[str, int]:
    """Fingerprint selected row IDs without materializing the selection."""
    if ids.ndim != 1 or selection.ndim != 1 or ids.shape != selection.shape:
        raise ValueError("Stored row IDs and selection must be aligned vectors")
    if np.dtype(selection.dtype) != np.dtype(bool):
        raise TypeError("Stored selection values must be booleans")
    source_dtype = np.dtype(ids.dtype)
    if source_dtype.kind not in {"O", "S", "T", "U"}:
        raise TypeError("Stored row IDs must contain strings")

    block_rows = min(
        row_band(array_geometry(ids), unit="chunk", fallback=1),
        row_band(array_geometry(selection), unit="chunk", fallback=1),
    )
    selected_count = 0
    if source_dtype.hasobject:
        max_length = 1
        for start in range(0, int(ids.shape[0]), block_rows):
            stop = min(start + block_rows, int(ids.shape[0]))
            mask = np.asarray(selection[start:stop], dtype=bool)
            values = np.asarray(ids[start:stop])[mask]
            selected_count += len(values)
            if values.size:
                max_length = max(max_length, max(len(str(value)) for value in values))
        string_dtype = np.dtype(f"U{max_length}")
    else:
        for start in range(0, int(ids.shape[0]), block_rows):
            stop = min(start + block_rows, int(ids.shape[0]))
            selected_count += int(
                np.count_nonzero(np.asarray(selection[start:stop], dtype=bool))
            )
        string_dtype = np.empty(0, dtype=source_dtype).astype(str).dtype

    builder = ValueFingerprintBuilder()
    builder.begin_array("values", (selected_count,), string_dtype)
    selected_start = 0
    for start in range(0, int(ids.shape[0]), block_rows):
        stop = min(start + block_rows, int(ids.shape[0]))
        mask = np.asarray(selection[start:stop], dtype=bool)
        values = np.asarray(ids[start:stop])[mask].astype(string_dtype)
        if values.size:
            builder.update_array_block("values", (selected_start,), values)
            selected_start += len(values)
    builder.end_array("values")
    return builder.hexdigest(), selected_count


def resolve_stored_selection_artifact(
    root: zarr.Group,
    *,
    table_path: str,
    id_column: str,
    source_column: str,
    scope: ArtifactScope,
    kind: str,
    operation: str,
    parameters: dict[str, Any],
    inputs: dict[str, Any],
    assay: str | None = None,
    invalidate_cache: bool = False,
) -> ArtifactRef:
    """Create a selection artifact by copying a stored column blockwise."""
    table = as_zarr_group(root[table_path], name=table_path)
    source = as_zarr_array(table[source_column], name=source_column)
    ids = as_zarr_array(table[id_column], name=id_column)
    if source.ndim != 1 or np.dtype(source.dtype) != np.dtype(bool):
        raise TypeError("Selection source column must be one-dimensional booleans")
    if ids.ndim != 1 or ids.shape != source.shape:
        raise ValueError("Selection row IDs must align with source values")
    values_fingerprint = _stored_selection_fingerprint(source)
    selection_inputs = dict(inputs)
    selection_inputs.update(
        {
            "ordered_row_ids_fingerprint": fingerprint_stored_strings(ids),
            "values_fingerprint": values_fingerprint,
        }
    )
    planned = plan_artifact(
        root,
        scope=scope,
        assay=assay,
        kind=kind,
        operation=operation,
        parameters=parameters,
        inputs=selection_inputs,
        execution_options={"source_column": source_column},
        invalidate_cache=invalidate_cache,
        required_arrays=(ArrayRequirement("values", shape=source.shape, dtype=bool),),
        reuse_validator=_selection_reuse_validator(values_fingerprint),
    )
    if planned.reused:
        return planned.ref
    group = start_artifact(root, planned)
    output = create_metadata_column(
        group,
        "values",
        dtype=bool,
        shape=int(source.shape[0]),
        chunkSize=row_band(array_geometry(source), unit="chunk", fallback=1),
        overwrite=True,
    )
    block_rows = row_band(array_geometry(source), unit="chunk", fallback=1)
    for start in range(0, int(source.shape[0]), block_rows):
        stop = min(start + block_rows, int(source.shape[0]))
        output[start:stop] = source[start:stop]
    if _stored_selection_fingerprint(output) != values_fingerprint:
        raise RuntimeError("Selection source changed while it was copied")
    finish_artifact(group, planned)
    return planned.ref


def resolve_selection_artifact(
    root: zarr.Group,
    *,
    scope: ArtifactScope,
    kind: str,
    values: np.ndarray,
    row_ids: np.ndarray,
    operation: str,
    parameters: dict[str, Any],
    inputs: dict[str, Any],
    source_column: str,
    assay: str | None = None,
    invalidate_cache: bool = False,
) -> ArtifactRef:
    mask = np.asarray(values)
    if mask.ndim != 1 or mask.dtype != bool:
        raise TypeError("Selection values must be a one-dimensional boolean array")
    rows = np.asarray(row_ids)
    if rows.ndim != 1 or len(rows) != len(mask):
        raise ValueError("Selection row IDs must align with selection values")
    values_fingerprint = fingerprint_array(mask)
    selection_inputs = dict(inputs)
    selection_inputs.update(
        {
            "ordered_row_ids_fingerprint": fingerprint_strings(rows),
            "values_fingerprint": values_fingerprint,
        }
    )
    planned = plan_artifact(
        root,
        scope=scope,
        assay=assay,
        kind=kind,
        operation=operation,
        parameters=parameters,
        inputs=selection_inputs,
        execution_options={"source_column": source_column},
        invalidate_cache=invalidate_cache,
        required_arrays=(ArrayRequirement("values", shape=mask.shape, dtype=bool),),
        reuse_validator=_selection_reuse_validator(values_fingerprint),
    )
    if planned.reused:
        return planned.ref
    group = start_artifact(root, planned)
    create_metadata_column(
        group,
        "values",
        data=mask,
        dtype=bool,
        overwrite=True,
    )
    finish_artifact(group, planned)
    return planned.ref


def resolve_generated_selection_artifact(
    root: zarr.Group,
    *,
    scope: ArtifactScope,
    kind: str,
    values: np.ndarray,
    row_ids: np.ndarray,
    operation: str,
    parameters: dict[str, Any],
    inputs: dict[str, Any],
    source_column: str,
    assay: str | None = None,
    invalidate_cache: bool = False,
) -> tuple[ArtifactRef, np.ndarray]:
    mask = np.asarray(values)
    if mask.ndim != 1 or mask.dtype != bool:
        raise TypeError("Selection values must be a one-dimensional boolean array")
    rows = np.asarray(row_ids)
    if rows.ndim != 1 or len(rows) != len(mask):
        raise ValueError("Selection row IDs must align with selection values")
    selection_inputs = dict(inputs)
    values_fingerprint = fingerprint_array(mask)
    selection_inputs.update(
        {
            "ordered_row_ids_fingerprint": fingerprint_strings(rows),
            "values_fingerprint": values_fingerprint,
        }
    )
    planned = plan_artifact(
        root,
        scope=scope,
        assay=assay,
        kind=kind,
        operation=operation,
        parameters=parameters,
        inputs=selection_inputs,
        execution_options={"source_column": source_column},
        invalidate_cache=invalidate_cache,
        required_arrays=(
            ArrayRequirement(
                "values",
                shape=mask.shape,
                dtype_kind="b",
            ),
        ),
        reuse_validator=_selection_reuse_validator(values_fingerprint),
    )
    if planned.reused:
        group = reused_artifact_group(root, planned)
        stored = as_zarr_array(group["values"], name="values")
        return planned.ref, np.asarray(stored[:], dtype=bool)
    group = start_artifact(root, planned)
    output = create_metadata_column(
        group,
        "values",
        data=mask,
        dtype=bool,
        overwrite=True,
    )
    if _stored_selection_fingerprint(output) != values_fingerprint:
        raise RuntimeError("Generated selection payload changed while it was stored")
    finish_artifact(group, planned)
    return planned.ref, mask


@dataclass(frozen=True, slots=True)
class _SnapshotColumn:
    name: str
    values: zarr.Array
    missing: zarr.Array | None
    dtype: np.dtype[Any]
    fingerprint: str


def _snapshot_block_rows(*arrays: zarr.Array) -> int:
    return min(
        int(row_band(array_geometry(array), unit="chunk", fallback=1))
        for array in arrays
    )


def _snapshot_text(value: Any) -> str:
    if isinstance(value, bytes | bytearray | np.bytes_):
        return bytes(value).decode("utf-8")
    if value is None:
        return ""
    return str(value)


def _snapshot_values_dtype(values: zarr.Array) -> np.dtype[Any]:
    source_dtype: np.dtype[Any] = np.dtype(values.dtype)
    if not source_dtype.hasobject:
        return source_dtype
    max_length = 1
    block_rows = row_band(array_geometry(values), unit="chunk", fallback=1)
    for start in range(0, int(values.shape[0]), block_rows):
        stop = min(start + block_rows, int(values.shape[0]))
        block = np.asarray(values[start:stop])
        if block.ndim != 1:
            raise ValueError("Snapshot metadata columns must be one-dimensional")
        if block.size:
            max_length = max(
                max_length,
                max(len(_snapshot_text(value)) for value in block),
            )
    return np.dtype(f"U{max_length}")


def _snapshot_values_block(
    values: zarr.Array,
    start: int,
    stop: int,
    dtype: np.dtype[Any],
) -> np.ndarray:
    block: np.ndarray = np.asarray(values[start:stop])
    if block.ndim != 1:
        raise ValueError("Snapshot metadata columns must be one-dimensional")
    if np.dtype(values.dtype).hasobject:
        text = [_snapshot_text(value) for value in block]
        width = dtype.itemsize // np.dtype("U1").itemsize
        if any(len(value) > width for value in text):
            raise RuntimeError("Snapshot string values changed while they were copied")
        return np.asarray(text, dtype=dtype)
    if block.dtype != dtype:
        return block.astype(dtype)
    return block


def _fingerprint_snapshot_column(
    values: zarr.Array,
    missing: zarr.Array | None,
    *,
    dtype: np.dtype[Any] | None = None,
) -> str:
    values_dtype = _snapshot_values_dtype(values) if dtype is None else np.dtype(dtype)
    builder = ValueFingerprintBuilder()
    builder.begin_array("values", values.shape, values_dtype)
    block_rows = row_band(array_geometry(values), unit="chunk", fallback=1)
    for start in range(0, int(values.shape[0]), block_rows):
        stop = min(start + block_rows, int(values.shape[0]))
        builder.update_array_block(
            "values",
            (start,),
            _snapshot_values_block(values, start, stop, values_dtype),
        )
    builder.end_array("values")
    if missing is not None:
        builder.begin_array("missing", missing.shape, missing.dtype)
        block_rows = row_band(array_geometry(missing), unit="chunk", fallback=1)
        for start in range(0, int(missing.shape[0]), block_rows):
            stop = min(start + block_rows, int(missing.shape[0]))
            builder.update_array_block(
                "missing",
                (start,),
                np.asarray(missing[start:stop], dtype=bool),
            )
        builder.end_array("missing")
    return str(builder.hexdigest())


def _snapshot_source_columns(
    table: zarr.Group,
    *,
    table_path: str,
    columns: Sequence[str],
    row_count: int,
) -> tuple[_SnapshotColumn, ...]:
    if isinstance(columns, str | bytes):
        raise TypeError("Snapshot columns must be a sequence of column names")
    names = tuple(columns)
    if not names:
        raise ValueError("Snapshot columns must not be empty")
    if any(not isinstance(name, str) or not name for name in names):
        raise TypeError("Snapshot column names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("Snapshot columns must be unique")
    invalid_names = [
        name for name in names if "/" in name or name.startswith(_MISSING_COLUMN_PREFIX)
    ]
    if invalid_names:
        raise ValueError(
            "Snapshot column names cannot be paths or internal missing-mask names"
        )

    resolved: list[_SnapshotColumn] = []
    for name in names:
        if name not in table:
            raise KeyError(f"Snapshot column {name!r} is unavailable in {table_path!r}")
        values = as_zarr_array(table[name], name=name)
        if values.ndim != 1 or int(values.shape[0]) != row_count:
            raise ValueError(
                f"Snapshot column {name!r} must align with the full metadata axis"
            )
        snapshot_dtype = _snapshot_values_dtype(values)
        missing_name = values.attrs.get("missing_mask")
        missing: zarr.Array | None = None
        if missing_name is not None:
            if (
                not isinstance(missing_name, str)
                or not missing_name
                or "/" in missing_name
                or missing_name not in table
            ):
                raise ValueError(
                    f"Snapshot column {name!r} has an invalid missing mask"
                )
            missing = as_zarr_array(table[missing_name], name=missing_name)
            if (
                missing.ndim != 1
                or missing.shape != values.shape
                or np.dtype(missing.dtype) != np.dtype(bool)
            ):
                raise ValueError(
                    f"Snapshot column {name!r} has a malformed missing mask"
                )
        resolved.append(
            _SnapshotColumn(
                name=name,
                values=values,
                missing=missing,
                dtype=snapshot_dtype,
                fingerprint=_fingerprint_snapshot_column(
                    values,
                    missing,
                    dtype=snapshot_dtype,
                ),
            )
        )
    return tuple(resolved)


def _snapshot_reuse_validator(
    columns: tuple[_SnapshotColumn, ...],
) -> Callable[[ArtifactRef, zarr.Group], bool]:
    expected_names = {column.name for column in columns}
    expected_names.update(
        f"{_MISSING_COLUMN_PREFIX}{column.name}"
        for column in columns
        if column.missing is not None
    )

    def validate(_ref: ArtifactRef, group: zarr.Group) -> bool:
        try:
            if set(group.array_keys()) != expected_names:
                return False
            for column in columns:
                values = as_zarr_array(group[column.name], name=column.name)
                missing_name = f"{_MISSING_COLUMN_PREFIX}{column.name}"
                if column.missing is None:
                    if "missing_mask" in values.attrs or set(values.attrs):
                        return False
                    missing = None
                else:
                    if values.attrs.get("missing_mask") != missing_name or set(
                        values.attrs
                    ) != {"missing_mask"}:
                        return False
                    missing = as_zarr_array(group[missing_name], name=missing_name)
                if _fingerprint_snapshot_column(values, missing) != column.fingerprint:
                    return False
        except (KeyError, TypeError, ValueError):
            return False
        return True

    return validate


def _snapshot_scope(axis: str, assay: str | None) -> ArtifactScope:
    if axis == "cell":
        if assay is not None:
            raise ValueError("Cell metadata snapshots cannot set an assay")
        return "datastore"
    if axis == "feature":
        if assay is None or not assay:
            raise ValueError("Feature metadata snapshots require an assay")
        return "assay"
    raise ValueError("Snapshot axis must be 'cell' or 'feature'")


def validate_run_metadata_snapshot(
    root: zarr.Group,
    ref: ArtifactRef,
    *,
    axis: str,
    assay: str | None,
    table_path: str,
    id_column: str = "ids",
    ordered_columns: Sequence[str] | None = None,
) -> zarr.Group:
    """Validate a run snapshot against its payload and current ordered row IDs."""
    scope = _snapshot_scope(axis, assay)
    context = _selection_context(ref, table_path=table_path)
    if ref.kind != "metadata_snapshot" or ref.scope != scope or ref.assay != assay:
        raise ArtifactResolutionError(
            "Metadata snapshot reference does not match its requested axis",
            code="artifact_reference_mismatch",
            context={
                **context,
                "expected_scope": scope,
                "expected_assay": assay,
                "expected_kind": "metadata_snapshot",
            },
        )
    try:
        status = inspect_artifact(root, ref)
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactResolutionError(
            "Metadata snapshot record is malformed",
            code="artifact_missing",
            context=context,
        ) from exc
    if not status.exists:
        raise ArtifactResolutionError(
            "Metadata snapshot does not exist",
            code="artifact_missing",
            context=context,
        )
    if not status.complete:
        raise ArtifactResolutionError(
            "Metadata snapshot is incomplete",
            code="artifact_incomplete",
            context=context,
        )
    parameters = status.parameters or {}
    raw_columns = parameters.get("ordered_columns")
    if (
        status.operation != "snapshot_run_metadata"
        or set(parameters) != {"axis", "assay", "ordered_columns"}
        or parameters.get("axis") != axis
        or parameters.get("assay") != assay
        or not isinstance(raw_columns, list)
        or not raw_columns
        or any(not isinstance(name, str) or not name for name in raw_columns)
        or len(set(raw_columns)) != len(raw_columns)
    ):
        raise ArtifactResolutionError(
            "Metadata snapshot axis or column contract is malformed",
            code="snapshot_contract_mismatch",
            context=context,
        )
    columns = tuple(raw_columns)
    if ordered_columns is not None:
        if (
            isinstance(ordered_columns, str | bytes)
            or tuple(ordered_columns) != columns
        ):
            raise ArtifactResolutionError(
                "Metadata snapshot columns do not match the requested order",
                code="snapshot_contract_mismatch",
                context=context,
            )
    if table_path not in root:
        raise ArtifactResolutionError(
            f"Snapshot metadata table {table_path!r} is unavailable",
            code="selection_table_missing",
            context=context,
        )
    table = as_zarr_group(root[table_path], name=table_path)
    if id_column not in table:
        raise ArtifactResolutionError(
            f"Snapshot row ID column {id_column!r} is unavailable",
            code="selection_row_ids_missing",
            context=context,
        )
    try:
        row_ids = as_zarr_array(table[id_column], name=id_column)
    except TypeError as exc:
        raise ArtifactResolutionError(
            "Snapshot row ID column is malformed",
            code="row_identity_mismatch",
            context=context,
        ) from exc
    inputs = status.inputs or {}
    expected_row_ids = inputs.get("ordered_row_ids_fingerprint")
    column_fingerprints = inputs.get("column_fingerprints")
    if (
        set(inputs) != {"ordered_row_ids_fingerprint", "column_fingerprints"}
        or row_ids.ndim != 1
        or np.dtype(row_ids.dtype).kind not in {"O", "S", "T", "U"}
        or not isinstance(expected_row_ids, str)
        or expected_row_ids != fingerprint_stored_strings(row_ids)
    ):
        raise ArtifactResolutionError(
            "Metadata snapshot row identity no longer matches its table",
            code="row_identity_mismatch",
            context=context,
        )
    if (
        not isinstance(column_fingerprints, dict)
        or set(column_fingerprints) != set(columns)
        or any(not isinstance(value, str) for value in column_fingerprints.values())
    ):
        raise ArtifactResolutionError(
            "Metadata snapshot column fingerprints are malformed",
            code="snapshot_contract_mismatch",
            context=context,
        )
    try:
        group = artifact_group(root, ref)
        expected_arrays: set[str] = set(columns)
        resolved: list[tuple[str, zarr.Array, zarr.Array | None]] = []
        for name in columns:
            values = as_zarr_array(group[name], name=name)
            if (
                values.ndim != 1
                or values.shape != row_ids.shape
                or np.dtype(values.dtype).hasobject
            ):
                raise ValueError("Snapshot values have invalid geometry")
            missing_name = values.attrs.get("missing_mask")
            if missing_name is None:
                if set(values.attrs):
                    raise ValueError("Snapshot values contain unexpected attributes")
                missing = None
            else:
                canonical_name = f"{_MISSING_COLUMN_PREFIX}{name}"
                if missing_name != canonical_name or set(values.attrs) != {
                    "missing_mask"
                }:
                    raise ValueError("Snapshot missing-mask link is malformed")
                missing = as_zarr_array(group[canonical_name], name=canonical_name)
                if (
                    missing.ndim != 1
                    or missing.shape != values.shape
                    or np.dtype(missing.dtype) != np.dtype(bool)
                ):
                    raise ValueError("Snapshot missing mask has invalid geometry")
                expected_arrays.add(canonical_name)
            resolved.append((name, values, missing))
        if set(group.array_keys()) != expected_arrays:
            raise ValueError("Snapshot contains unexpected arrays")
        for name, values, missing in resolved:
            if (
                _fingerprint_snapshot_column(values, missing)
                != column_fingerprints[name]
            ):
                raise ValueError("Snapshot column fingerprint changed")
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactResolutionError(
            "Metadata snapshot payload no longer matches its contract",
            code="snapshot_values_changed",
            context=context,
        ) from exc
    return group


def snapshot_run_metadata(
    root: zarr.Group,
    *,
    table_path: str,
    id_column: str,
    columns: Sequence[str],
    axis: str,
    assay: str | None = None,
    invalidate_cache: bool = False,
) -> ArtifactRef:
    """Copy full-axis run metadata into an immutable named-array artifact."""
    scope = _snapshot_scope(axis, assay)
    if table_path not in root:
        raise KeyError(f"Snapshot metadata table {table_path!r} is unavailable")
    table = as_zarr_group(root[table_path], name=table_path)
    if id_column not in table:
        raise KeyError(f"Snapshot row ID column {id_column!r} is unavailable")
    row_ids = as_zarr_array(table[id_column], name=id_column)
    if row_ids.ndim != 1 or np.dtype(row_ids.dtype).kind not in {"O", "S", "T", "U"}:
        raise TypeError("Snapshot row IDs must be a one-dimensional string column")
    ordered_row_ids_fingerprint = fingerprint_stored_strings(row_ids)
    sources = _snapshot_source_columns(
        table,
        table_path=table_path,
        columns=columns,
        row_count=int(row_ids.shape[0]),
    )
    names = tuple(column.name for column in sources)
    column_fingerprints = {column.name: column.fingerprint for column in sources}
    required_arrays: list[ArrayRequirement] = []
    for column in sources:
        required_arrays.append(
            ArrayRequirement(
                column.name,
                shape=column.values.shape,
                dtype=column.dtype,
            )
        )
        if column.missing is not None:
            required_arrays.append(
                ArrayRequirement(
                    f"{_MISSING_COLUMN_PREFIX}{column.name}",
                    shape=column.missing.shape,
                    dtype=bool,
                )
            )
    planned = plan_artifact(
        root,
        scope=scope,
        assay=assay,
        kind="metadata_snapshot",
        operation="snapshot_run_metadata",
        parameters={
            "axis": axis,
            "assay": assay,
            "ordered_columns": names,
        },
        inputs={
            "ordered_row_ids_fingerprint": ordered_row_ids_fingerprint,
            "column_fingerprints": column_fingerprints,
        },
        execution_options={},
        invalidate_cache=invalidate_cache,
        required_arrays=tuple(required_arrays),
        reuse_validator=_snapshot_reuse_validator(sources),
    )
    if planned.reused:
        return planned.ref

    group = start_artifact(root, planned)
    for column in sources:
        chunk_rows = _snapshot_block_rows(
            *(
                (column.values, column.missing)
                if column.missing is not None
                else (column.values,)
            )
        )
        output = create_metadata_column(
            group,
            column.name,
            dtype=column.dtype,
            shape=int(column.values.shape[0]),
            chunkSize=chunk_rows,
            overwrite=True,
        )
        missing_output: zarr.Array | None = None
        if column.missing is not None:
            missing_name = f"{_MISSING_COLUMN_PREFIX}{column.name}"
            missing_output = create_metadata_column(
                group,
                missing_name,
                dtype=bool,
                shape=int(column.missing.shape[0]),
                chunkSize=chunk_rows,
                overwrite=True,
            )
            output.attrs["missing_mask"] = missing_name
        for start in range(0, int(column.values.shape[0]), chunk_rows):
            stop = min(start + chunk_rows, int(column.values.shape[0]))
            output[start:stop] = _snapshot_values_block(
                column.values,
                start,
                stop,
                column.dtype,
            )
            if missing_output is not None and column.missing is not None:
                missing_output[start:stop] = column.missing[start:stop]
        if _fingerprint_snapshot_column(output, missing_output) != column.fingerprint:
            raise RuntimeError(
                f"Snapshot column {column.name!r} changed while it was copied"
            )
    if fingerprint_stored_strings(row_ids) != ordered_row_ids_fingerprint:
        raise RuntimeError("Snapshot row IDs changed while metadata was copied")
    finish_artifact(group, planned)
    return planned.ref


def resolve_metadata_snapshot(
    root: zarr.Group,
    *,
    values: np.ndarray,
    row_ids: np.ndarray,
    operation: str,
    parameters: dict[str, Any],
    inputs: dict[str, Any],
    source_columns: list[str],
    invalidate_cache: bool = False,
) -> ArtifactRef:
    array = np.asarray(values)
    rows = np.asarray(row_ids)
    if array.ndim < 1 or rows.ndim != 1 or array.shape[0] != len(rows):
        raise ValueError("Metadata values must align with row IDs")
    flattened = array.reshape(-1)
    stored_values = (
        flattened.astype(str) if flattened.dtype.kind in {"O", "S", "U"} else flattened
    )
    values_fingerprint = (
        fingerprint_strings(stored_values)
        if stored_values.dtype.kind in {"O", "S", "U"}
        else fingerprint_array(stored_values)
    )
    snapshot_inputs = dict(inputs)
    snapshot_inputs.update(
        {
            "ordered_row_ids_fingerprint": fingerprint_strings(rows),
            "values_fingerprint": values_fingerprint,
        }
    )
    snapshot_parameters = dict(parameters)
    snapshot_parameters["shape"] = list(array.shape)

    def reuse_validator(_ref: ArtifactRef, group: zarr.Group) -> bool:
        try:
            candidate = as_zarr_array(group["values"], name="values")
            if candidate.ndim != 1 or candidate.shape != stored_values.shape:
                return False
            if stored_values.dtype.kind in {"O", "S", "U"}:
                return fingerprint_stored_strings(candidate) == values_fingerprint
            return fingerprint_stored_arrays(group, ("values",)) == values_fingerprint
        except (KeyError, TypeError, ValueError):
            return False

    planned = plan_artifact(
        root,
        scope="datastore",
        kind="metadata_snapshot",
        operation=operation,
        parameters=snapshot_parameters,
        inputs=snapshot_inputs,
        execution_options={"source_columns": source_columns},
        invalidate_cache=invalidate_cache,
        required_arrays=(ArrayRequirement("values", shape=stored_values.shape),),
        reuse_validator=reuse_validator,
    )
    if planned.reused:
        return planned.ref
    group = start_artifact(root, planned)
    create_metadata_column(
        group,
        "values",
        data=stored_values,
        dtype=stored_values.dtype,
        overwrite=True,
    )
    finish_artifact(group, planned)
    return planned.ref
