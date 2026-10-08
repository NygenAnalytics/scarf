"""Stored statistical-test results: slot layout, reuse check, writer, and reader.

A ``statistical_tests`` slot records the result's attributes and one child
group per key, named by the key's position. The child holds the table's
numeric columns in a ``stats`` array and each group-label column in its own
array; a post-hoc table uses the same layout under a ``posthoc_`` prefix.
:func:`statistical_slot_attributes` derives the attributes once: the writer
records them, and the reuse check compares a stored slot with them.
"""

from collections.abc import Callable, Sequence
from typing import Any, TypeGuard, cast

import numpy as np
import pandas as pd
import zarr

from ...features.statistical import (
    StatisticalTestResult,
    native_value,
    split_group_columns,
    statistical_posthoc_columns,
    statistical_storage_columns,
    statistical_summary_scope,
)
from ...metadata.arguments import StatisticalTestingArguments
from ...metadata.selection import CellField
from ...storage.artifacts import ArtifactRef
from ...storage.types import as_zarr_array, as_zarr_group

# The keys of statistical_slot_attributes, which every complete slot records.
_STATISTICAL_SLOT_ATTRIBUTES = frozenset(
    {
        "adjustment_method",
        "alternative",
        "cell_selection",
        "cell_selection_fingerprint",
        "equal_var",
        "expression_cutoff",
        "group_field",
        "group_fingerprint",
        "group_order",
        "grouping",
        "key_labels",
        "method",
        "n_cells",
        "n_groups",
        "normalization",
        "normalization_method",
        "p_value_method",
        "pair_by",
        "pair_fingerprint",
        "posthoc",
        "posthoc_stat_columns",
        "sample_by",
        "sample_fingerprint",
        "sample_stat",
        "size_factor",
        "source_assays",
        "source_dataset_fingerprint",
        "stat_columns",
        "subset_fingerprint",
        "summary_scope",
        "tested_features",
        "value_fingerprints",
    }
)


def statistical_slot_attributes(
    arguments: StatisticalTestingArguments,
    *,
    group_order: Sequence[Any],
    p_value_method: str | None,
    value_fingerprints: Sequence[str],
) -> dict[str, Any]:
    """Return the attributes that a statistical-test slot records."""
    return {
        "stat_columns": list(statistical_storage_columns(arguments.method)),
        "posthoc_stat_columns": list(statistical_posthoc_columns(arguments.posthoc)),
        "method": arguments.method,
        "p_value_method": p_value_method,
        "posthoc": arguments.posthoc,
        "adjustment_method": arguments.adjustment_method,
        "grouping": (
            arguments.grouping.to_dict() if arguments.grouping is not None else None
        ),
        "group_field": arguments.group_field,
        "sample_by": arguments.sample_by,
        "pair_by": arguments.pair_by,
        "sample_stat": arguments.sample_stat,
        "expression_cutoff": arguments.expression_cutoff,
        "alternative": arguments.alternative,
        "equal_var": arguments.equal_var,
        "normalization": dict(arguments.normalization),
        "normalization_method": arguments.normalization_method,
        "size_factor": arguments.size_factor,
        "group_fingerprint": arguments.group_fingerprint,
        "group_order": [native_value(value) for value in group_order],
        "subset_fingerprint": arguments.subset_fingerprint,
        "sample_fingerprint": arguments.sample_fingerprint,
        "pair_fingerprint": arguments.pair_fingerprint,
        "n_groups": arguments.n_groups,
        "n_cells": arguments.n_cells,
        "tested_features": list(arguments.tested_features),
        "source_assays": list(arguments.source_assays),
        "source_dataset_fingerprint": arguments.source_dataset_fingerprint,
        "value_fingerprints": list(value_fingerprints),
        "cell_selection": (
            arguments.cell_selection.to_dict()
            if arguments.cell_selection is not None
            else None
        ),
        "cell_selection_fingerprint": arguments.cell_selection_fingerprint,
        "summary_scope": statistical_summary_scope(arguments.sample_by),
        "key_labels": list(arguments.key_labels),
    }


def _p_value_methods(method: Any) -> tuple[str | None, ...]:
    """The p-value methods that a slot of ``method`` may record."""
    return ("exact", "asymptotic") if method == "mann_whitney" else (None,)


def _valid_value_fingerprints(value: Any, n_keys: int) -> TypeGuard[list[str]]:
    """Whether ``value`` holds one nonempty fingerprint per key."""
    return (
        isinstance(value, list)
        and len(value) == n_keys
        and all(isinstance(fingerprint, str) and fingerprint for fingerprint in value)
    )


def statistical_reuse_validator(
    arguments: StatisticalTestingArguments,
    *,
    group_order: Sequence[Any],
    value_fingerprints: Callable[[], Sequence[str]],
) -> Callable[[ArtifactRef, zarr.Group], bool]:
    """Return the check that a stored slot holds the planned result."""
    method = arguments.method
    posthoc_columns = statistical_posthoc_columns(arguments.posthoc)
    _, main_numeric = split_group_columns(statistical_storage_columns(method))
    _, posthoc_numeric = split_group_columns(posthoc_columns)
    n_keys = len(arguments.key_labels)

    def statistical_reuse_is_valid(
        _ref: ArtifactRef,
        candidate: zarr.Group,
    ) -> bool:
        try:
            stored: dict[str, Any] = dict(candidate.attrs)
            p_value_method = stored.get("p_value_method")
            if p_value_method not in _p_value_methods(method):
                return False
            stored_value_fingerprints = stored.get("value_fingerprints")
            if not _valid_value_fingerprints(stored_value_fingerprints, n_keys):
                return False
            # Attributes hold JSON values, which compare equal to the native
            # values that the request records.
            expected = statistical_slot_attributes(
                arguments,
                group_order=group_order,
                p_value_method=p_value_method,
                value_fingerprints=stored_value_fingerprints,
            )
            if any(stored.get(name) != value for name, value in expected.items()):
                return False
            if stored_value_fingerprints != list(value_fingerprints()):
                return False
            for idx in range(n_keys):
                key_group = as_zarr_group(
                    candidate[str(idx)],
                    name=str(idx),
                )
                stats = np.asarray(
                    as_zarr_array(
                        key_group["stats"],
                        name="stats",
                    )[:]
                )
                if stats.ndim != 2 or stats.shape[1] != len(main_numeric):
                    return False
                if set(cast(Any, key_group.attrs["stats_dtypes"])) != set(main_numeric):
                    return False
                if posthoc_columns:
                    posthoc_stats = np.asarray(
                        as_zarr_array(
                            key_group["posthoc_stats"],
                            name="posthoc_stats",
                        )[:]
                    )
                    if posthoc_stats.ndim != 2 or posthoc_stats.shape[1] != len(
                        posthoc_numeric
                    ):
                        return False
                    if set(cast(Any, key_group.attrs["posthoc_stats_dtypes"])) != set(
                        posthoc_numeric
                    ):
                        return False
        except (KeyError, TypeError, ValueError, IndexError):
            return False
        return True

    return statistical_reuse_is_valid


def write_statistical_slot(
    group: zarr.Group,
    result: StatisticalTestResult,
    *,
    arguments: StatisticalTestingArguments,
) -> None:
    """Write a result's attributes and per-key tables into its started slot."""
    posthoc_columns = statistical_posthoc_columns(arguments.posthoc)
    main_string_columns, main_numeric_columns = split_group_columns(
        statistical_storage_columns(arguments.method)
    )
    posthoc_string_columns, posthoc_numeric_columns = split_group_columns(
        posthoc_columns
    )
    group.attrs.update(
        statistical_slot_attributes(
            arguments,
            group_order=result.group_order,
            p_value_method=result.p_value_method,
            value_fingerprints=result.value_fingerprints,
        )
    )

    for idx, key_label in enumerate(arguments.key_labels):
        table = result.tables[key_label]
        key_group = group.create_group(str(idx))
        key_group.attrs["key_label"] = key_label
        key_group.attrs["key_index"] = idx
        _write_stats_array(key_group, "stats", table, main_numeric_columns)
        for column in main_string_columns:
            _write_group_column(key_group, column, table[column])
        if result.posthoc_tables and key_label in result.posthoc_tables:
            posthoc_table = result.posthoc_tables[key_label]
            _write_stats_array(
                key_group,
                "posthoc_stats",
                posthoc_table,
                posthoc_numeric_columns,
            )
            for column in posthoc_string_columns:
                _write_group_column(
                    key_group,
                    f"posthoc_{column}",
                    posthoc_table[column],
                )


def read_statistical_slot(
    slot_group: zarr.Group,
    *,
    artifact: ArtifactRef,
) -> StatisticalTestResult:
    """Read the result that a complete slot stores."""
    attrs: dict[str, Any] = dict(slot_group.attrs)
    method: Any = attrs.get("method")
    p_value_method = attrs.get("p_value_method")
    if method == "mann_whitney" and p_value_method not in _p_value_methods(method):
        raise ValueError(
            "Mann-Whitney artifact does not record its p-value method, "
            "so it predates exact small-sample p-values. Rerun "
            "run_statistical_testing to recompute it"
        )
    missing = sorted(_STATISTICAL_SLOT_ATTRIBUTES.difference(attrs))
    if missing:
        raise ValueError(
            "Statistical test artifact is missing metadata "
            f"({', '.join(missing)}); rerun run_statistical_testing"
        )
    if p_value_method not in _p_value_methods(method):
        raise ValueError("Statistical test p-value method metadata is invalid")
    posthoc = attrs["posthoc"]
    storage_columns = statistical_storage_columns(method)
    posthoc_columns = statistical_posthoc_columns(posthoc)
    main_string_columns, main_numeric_columns = split_group_columns(storage_columns)
    posthoc_string_columns, posthoc_numeric_columns = split_group_columns(
        posthoc_columns
    )
    key_labels = attrs["key_labels"]
    if (
        not isinstance(key_labels, list)
        or any(not isinstance(label, str) for label in key_labels)
        or len(set(key_labels)) != len(key_labels)
    ):
        raise ValueError("Statistical test key-label metadata is invalid")
    raw_value_fingerprints = attrs["value_fingerprints"]
    if not _valid_value_fingerprints(raw_value_fingerprints, len(key_labels)):
        raise ValueError("Statistical value-fingerprint metadata is invalid")
    value_fingerprints = tuple(raw_value_fingerprints)
    tables: dict[str, pd.DataFrame] = {}
    posthoc_tables: dict[str, pd.DataFrame] = {}
    for idx, key_label in enumerate(key_labels):
        key_group = as_zarr_group(slot_group[str(idx)], name=str(idx))
        stats = np.asarray(as_zarr_array(key_group["stats"], name="stats")[:])
        frame = pd.DataFrame(stats, columns=main_numeric_columns)
        stats_dtypes = key_group.attrs.get("stats_dtypes")
        if not isinstance(stats_dtypes, dict) or set(stats_dtypes) != set(
            main_numeric_columns
        ):
            raise ValueError("Statistical test dtype metadata is invalid")
        for column, dtype in stats_dtypes.items():
            frame[column] = frame[column].astype(dtype)
        for column in main_string_columns:
            frame[column] = _read_group_column(key_group, column)
        frame = frame.loc[:, list(storage_columns)]
        tables[str(key_label)] = frame
        if posthoc_columns:
            posthoc_stats = np.asarray(
                as_zarr_array(
                    key_group["posthoc_stats"],
                    name="posthoc_stats",
                )[:]
            )
            posthoc_frame = pd.DataFrame(
                posthoc_stats,
                columns=posthoc_numeric_columns,
            )
            posthoc_dtypes = key_group.attrs.get("posthoc_stats_dtypes")
            if not isinstance(posthoc_dtypes, dict) or set(posthoc_dtypes) != set(
                posthoc_numeric_columns
            ):
                raise ValueError("Statistical post-hoc dtype metadata is invalid")
            for column, dtype in posthoc_dtypes.items():
                posthoc_frame[column] = posthoc_frame[column].astype(dtype)
            for column in posthoc_string_columns:
                posthoc_frame[column] = _read_group_column(
                    key_group,
                    f"posthoc_{column}",
                )
            posthoc_frame = posthoc_frame.loc[:, list(posthoc_columns)]
            posthoc_tables[str(key_label)] = posthoc_frame
    raw_cell_selection = attrs["cell_selection"]
    cell_selection = (
        ArtifactRef.from_dict(raw_cell_selection)
        if raw_cell_selection is not None
        else None
    )
    raw_grouping = attrs["grouping"]
    raw_group_field = attrs["group_field"]
    if (raw_grouping is None) == (raw_group_field is None):
        raise ValueError(
            "Statistical test artifact lacks one explicit grouping source; "
            "rerun run_statistical_testing"
        )
    if raw_grouping is not None and not isinstance(raw_grouping, dict):
        raise ValueError("Statistical test grouping metadata is invalid")
    if raw_group_field is not None and (
        not isinstance(raw_group_field, str) or not raw_group_field
    ):
        raise ValueError("Statistical test group field is invalid")
    grouping = (
        ArtifactRef.from_dict(raw_grouping) if isinstance(raw_grouping, dict) else None
    )
    group_field = (
        CellField(raw_group_field) if isinstance(raw_group_field, str) else None
    )
    return StatisticalTestResult(
        method=str(method),
        posthoc=posthoc,
        adjustment_method=str(attrs["adjustment_method"]),
        grouping=grouping,
        group_field=group_field,
        sample_by=attrs["sample_by"],
        pair_by=attrs["pair_by"],
        sample_stat=str(attrs["sample_stat"]),
        expression_cutoff=float(attrs["expression_cutoff"]),
        alternative=str(attrs["alternative"]),
        equal_var=attrs["equal_var"],
        n_groups=int(attrs["n_groups"]),
        n_cells=int(attrs["n_cells"]),
        tested_features=tuple(attrs["tested_features"]),
        summary_scope=attrs["summary_scope"],
        artifact=artifact,
        cell_selection=cell_selection,
        cell_selection_fingerprint=attrs["cell_selection_fingerprint"],
        group_fingerprint=attrs["group_fingerprint"],
        group_order=tuple(attrs["group_order"]),
        normalization=dict(attrs["normalization"]),
        source_assays=tuple(attrs["source_assays"]),
        source_dataset_fingerprint=attrs["source_dataset_fingerprint"],
        value_fingerprints=value_fingerprints,
        sample_fingerprint=attrs["sample_fingerprint"],
        pair_fingerprint=attrs["pair_fingerprint"],
        normalization_method=attrs["normalization_method"],
        size_factor=attrs["size_factor"],
        tables=tables,
        posthoc_tables=posthoc_tables,
        p_value_method=p_value_method,
    )


def _write_stats_array(
    key_group: zarr.Group,
    name: str,
    table: pd.DataFrame,
    columns: Sequence[str],
) -> None:
    from ...storage.arrays import create_zarr_dataset

    key_group.attrs[f"{name}_dtypes"] = {
        column: str(table[column].dtype) for column in columns
    }
    # Selecting a list of columns keeps the table two-dimensional.
    numeric = np.asarray(table.loc[:, list(columns)].to_numpy(dtype=np.float64))
    if np.isnan(numeric).any():
        raise ValueError("Statistical test results must not contain NaN")
    finite_slots = [
        idx
        for idx, column in enumerate(columns)
        if column not in {"t_statistic", "f_statistic"}
    ]
    if finite_slots and not np.isfinite(numeric[:, finite_slots]).all():
        raise ValueError(
            "Only t_statistic and f_statistic may be infinite in statistical results"
        )
    stats = create_zarr_dataset(
        key_group,
        name,
        (int(numeric.shape[0]), int(numeric.shape[1])),
        "float64",
        (int(numeric.shape[0]), int(numeric.shape[1])),
    )
    stats[:] = numeric


def _write_group_column(
    key_group: zarr.Group,
    name: str,
    values: pd.Series,
) -> None:
    """Write a group-label column preserving its native dtype."""
    from ...storage.arrays import create_metadata_column, create_zarr_dataset

    array = np.asarray(values)
    kind = array.dtype.kind
    if kind == "b":
        dtype_tag = "bool"
        data = array.astype(bool)
    elif kind in {"i", "u"}:
        dtype_tag = "int"
        data = array.astype(np.int64)
    elif kind == "f":
        dtype_tag = "float"
        data = array.astype(np.float64)
    else:
        dtype_tag = "str"
        data = array
    key_group.attrs[f"{name}_dtype"] = dtype_tag
    if dtype_tag == "str":
        create_metadata_column(
            key_group,
            name,
            data=[str(value) for value in data],
        )
        return
    column = create_zarr_dataset(
        key_group,
        name,
        (int(len(data)),),
        dtype_tag,
        (int(len(data)),),
    )
    column[:] = data


def _read_group_column(key_group: zarr.Group, name: str) -> np.ndarray:
    dtype_tag = key_group.attrs.get(f"{name}_dtype")
    raw = np.asarray(as_zarr_array(key_group[name], name=name)[:])
    if dtype_tag == "bool":
        return np.asarray(raw, dtype=bool)
    if dtype_tag == "int":
        return np.asarray(raw, dtype=np.int64)
    if dtype_tag == "float":
        return np.asarray(raw, dtype=np.float64)
    if dtype_tag == "str":
        return np.asarray(raw, dtype=object)
    raise ValueError(f"Statistical test column {name!r} has invalid dtype metadata")
