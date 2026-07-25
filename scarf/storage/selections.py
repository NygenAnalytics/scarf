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
    fingerprint_array,
    fingerprint_strings,
)
from .types import as_zarr_array


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
        planned = plan_artifact(
            root,
            scope=scope,
            assay=assay,
            kind=kind,
            operation=operation,
            parameters=parameters,
            inputs=selection_inputs,
            execution_options={"source_column": source_column},
            invalidate_cache=True,
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
        planned = plan_artifact(
            root,
            scope=scope,
            assay=assay,
            kind=kind,
            operation=operation,
            parameters=parameters,
            inputs=selection_inputs,
            execution_options={"source_column": source_column},
            invalidate_cache=True,
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
        planned = plan_artifact(
            root,
            scope="datastore",
            kind="metadata_snapshot",
            operation=operation,
            parameters=snapshot_parameters,
            inputs=snapshot_inputs,
            execution_options={"source_columns": source_columns},
            invalidate_cache=True,
        )
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
