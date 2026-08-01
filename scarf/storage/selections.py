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
    fingerprint_array,
    fingerprint_stored_strings,
    fingerprint_strings,
)
from .geometry import array_geometry
from .partition import row_band
from .types import as_zarr_array, as_zarr_group


def _stored_selection_fingerprint(array: zarr.Array) -> str:
    if array.ndim != 1 or np.dtype(array.dtype) != np.dtype(bool):
        raise TypeError("Stored selection columns must be one-dimensional booleans")
    builder = ValueFingerprintBuilder()
    builder.begin_array("values", array.shape, array.dtype)
    block_rows = row_band(array_geometry(array), unit="chunk", fallback=1)
    for start in range(0, int(array.shape[0]), block_rows):
        stop = min(start + block_rows, int(array.shape[0]))
        builder.update_array_block(
            "values",
            (start,),
            np.asarray(array[start:stop], dtype=bool),
        )
    builder.end_array("values")
    return builder.hexdigest()


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
    if source_dtype.kind not in {"O", "S", "U"}:
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
    selection_inputs = dict(inputs)
    selection_inputs.update(
        {
            "ordered_row_ids_fingerprint": fingerprint_stored_strings(ids),
            "values_fingerprint": _stored_selection_fingerprint(source),
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
    selection_inputs = dict(inputs)
    selection_inputs.update(
        {
            "ordered_row_ids_fingerprint": fingerprint_strings(rows),
            "values_fingerprint": fingerprint_array(mask),
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
    )
    if planned.reused:
        group = reused_artifact_group(root, planned)
        if "values" in group:
            try:
                stored_values = np.asarray(
                    as_zarr_array(group["values"], name="values")[:]
                )
            except TypeError:
                pass
            else:
                if (
                    stored_values.ndim == 1
                    and stored_values.dtype == np.dtype(bool)
                    and stored_values.shape == mask.shape
                    and np.array_equal(stored_values, mask)
                ):
                    return planned.ref
        planned = planned.invalidated(
            root,
            required_arrays=(
                ArrayRequirement(
                    "values",
                    shape=mask.shape,
                    dtype_kind="b",
                ),
            ),
        )
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
    selection_inputs["ordered_row_ids_fingerprint"] = fingerprint_strings(rows)
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
    )
    if planned.reused:
        group = reused_artifact_group(root, planned)
        stored_values = np.asarray(as_zarr_array(group["values"], name="values")[:])
        if (
            stored_values.ndim == 1
            and stored_values.dtype == np.dtype(bool)
            and stored_values.shape == mask.shape
        ):
            return planned.ref, stored_values
        planned = planned.invalidated(root)
    group = start_artifact(root, planned)
    create_metadata_column(
        group,
        "values",
        data=mask,
        dtype=bool,
        overwrite=True,
    )
    finish_artifact(group, planned)
    return planned.ref, mask


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
    values_fingerprint = (
        fingerprint_strings(array.reshape(-1))
        if array.dtype.kind in {"O", "S", "U"}
        else fingerprint_array(array)
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
    planned = plan_artifact(
        root,
        scope="datastore",
        kind="metadata_snapshot",
        operation=operation,
        parameters=snapshot_parameters,
        inputs=snapshot_inputs,
        execution_options={"source_columns": source_columns},
        invalidate_cache=invalidate_cache,
    )
    flattened = array.reshape(-1)
    if planned.reused:
        group = reused_artifact_group(root, planned)
        if "values" in group:
            try:
                stored_values = as_zarr_array(group["values"], name="values")[:]
            except TypeError:
                pass
            else:
                expected_values = (
                    flattened.astype(str)
                    if flattened.dtype.kind in {"O", "S", "U"}
                    else flattened
                )
                if np.array_equal(stored_values, expected_values):
                    return planned.ref
        planned = planned.invalidated(root)
    group = start_artifact(root, planned)
    create_metadata_column(
        group,
        "values",
        data=flattened.astype(str)
        if flattened.dtype.kind in {"O", "S", "U"}
        else flattened,
        overwrite=True,
    )
    finish_artifact(group, planned)
    return planned.ref
