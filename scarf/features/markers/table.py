"""Readers for persisted marker statistics."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import zarr

from .rank import sort_marker_results

LEGACY_STAT_COLUMNS = (
    "score",
    "mean",
    "mean_rest",
    "frac_exp",
    "frac_exp_rest",
    "fold_change",
    "p_value",
)
MARKER_STAT_COLUMNS = (
    *LEGACY_STAT_COLUMNS,
    "auc",
    "p_value_adjusted",
)
MARKER_METHOD = "mannwhitneyu"
MARKER_ALTERNATIVE = "two-sided"
MARKER_TIE_CORRECTION = True
MARKER_CONTINUITY_CORRECTION = True
MARKER_ADJUSTMENT_METHOD = "fdr_bh"
MARKER_ADJUSTMENT_SCOPE = "within_group_all_tested_features"
_MARKER_INDEX_COLUMN = "feature_index"
_MARKER_METADATA = {
    "method": MARKER_METHOD,
    "alternative": MARKER_ALTERNATIVE,
    "tie_correction": MARKER_TIE_CORRECTION,
    "continuity_correction": MARKER_CONTINUITY_CORRECTION,
    "adjustment_method": MARKER_ADJUSTMENT_METHOD,
    "adjustment_scope": MARKER_ADJUSTMENT_SCOPE,
}

__all__ = [
    "LEGACY_STAT_COLUMNS",
    "MARKER_STAT_COLUMNS",
    "load_marker_table",
    "read_legacy_marker_table",
]


def _empty_marker_frame(columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(dtype=object) for name in columns})


def _array_values(group: zarr.Group, name: str) -> np.ndarray:
    value = group[name]
    if not isinstance(value, zarr.Array):
        raise TypeError(f"Marker field {name!r} must be an array")
    return np.asarray(value[:])


def _stored_stat_columns(
    slot_group: zarr.Group,
    *,
    required: bool,
) -> list[str] | None:
    stored = slot_group.attrs.get("stat_columns")
    if stored is None:
        if required:
            raise ValueError("Canonical marker tables require stat_columns metadata")
        return None
    if isinstance(stored, str) or not isinstance(stored, Sequence):
        raise ValueError("Marker stat_columns metadata must be a sequence of names")
    columns: list[str] = []
    for name in stored:
        if not isinstance(name, str):
            raise ValueError("Marker stat_columns metadata must contain only strings")
        columns.append(name)
    if len(columns) != len(set(columns)):
        raise ValueError("Marker stat_columns metadata contains duplicate names")
    unknown = set(columns).difference(MARKER_STAT_COLUMNS)
    if unknown:
        raise ValueError(
            "Marker stat_columns metadata contains unknown columns: "
            + ", ".join(sorted(unknown))
        )
    return columns


def _validated_feature_indices(
    values: np.ndarray,
    *,
    n_features: int,
    require_integer_dtype: bool,
    require_unique: bool = False,
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError("Marker feature_index must be one-dimensional")
    if require_integer_dtype and raw.dtype.kind not in {"i", "u"}:
        raise ValueError("Canonical marker feature_index must use an integer dtype")
    try:
        numeric = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Marker feature_index must contain integer values") from exc
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError("Marker feature_index must contain finite integer values")
    indices = numeric.astype(np.int64)
    invalid = (indices < 0) | (indices >= n_features)
    if invalid.any():
        unresolved = ", ".join(str(value) for value in indices[invalid][:5])
        raise ValueError(
            "Marker feature_index contains unresolved or out-of-range values: "
            + unresolved
        )
    if require_unique and np.unique(indices).size != indices.size:
        raise ValueError("Canonical marker feature_index must contain unique values")
    return indices


def _display_frame(
    df: pd.DataFrame,
    *,
    group_id: Any,
    feature_names: np.ndarray,
    out_stat_columns: tuple[str, ...],
    require_integer_indices: bool = False,
    require_unique_indices: bool = False,
) -> pd.DataFrame:
    frame = df.copy()
    frame["group_id"] = group_id
    if "feature_index" in frame.columns:
        names = np.asarray(feature_names, dtype=object)
        if names.ndim != 1:
            raise ValueError("Feature names must be one-dimensional")
        indices = _validated_feature_indices(
            frame["feature_index"].to_numpy(copy=False),
            n_features=len(names),
            require_integer_dtype=require_integer_indices,
            require_unique=require_unique_indices,
        )
        frame["feature_index"] = indices
        frame["feature_name"] = names[indices]
    ordered = ["group_id", "feature_name", "feature_index", *out_stat_columns]
    for column in ordered:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[ordered]


def _resolve_compact_stat_columns(
    slot_group: zarr.Group,
    n_columns: int,
) -> list[str]:
    stored = _stored_stat_columns(slot_group, required=False)
    if stored is not None:
        if len(stored) != n_columns:
            raise ValueError(
                "Compact marker stats width does not match stored stat_columns"
            )
        return stored
    if n_columns == len(LEGACY_STAT_COLUMNS):
        return list(LEGACY_STAT_COLUMNS)
    raise ValueError(
        "Compact marker stats lack stat_columns and do not match a known layout"
    )


def _maybe_attach_adjusted_pvalues(frame: pd.DataFrame) -> pd.DataFrame:
    if "p_value_adjusted" in frame.columns and frame["p_value_adjusted"].notna().any():
        return frame
    if "p_value" not in frame.columns or not frame["p_value"].notna().any():
        if "p_value_adjusted" not in frame.columns:
            frame = frame.copy()
            frame["p_value_adjusted"] = np.nan
        return frame
    from .correction import _bh_adjusted_pvalues

    frame = frame.copy()
    frame["p_value_adjusted"] = _bh_adjusted_pvalues(
        frame["p_value"].to_numpy(dtype=np.float64, copy=False)
    )
    return frame


def read_legacy_marker_table(
    slot_group: zarr.Group,
    cluster_group: zarr.Group,
    feature_names: np.ndarray,
    *,
    group_id: Any,
    feature_ids: np.ndarray | None = None,
) -> pd.DataFrame:
    """Read master and unversioned marker layouts into a named table."""
    legacy_index_and_stats = (_MARKER_INDEX_COLUMN, *LEGACY_STAT_COLUMNS)
    out_cols = list(legacy_index_and_stats)
    if "feature_index" in slot_group and "stats" in cluster_group:
        feature_index = _array_values(slot_group, "feature_index")
        stats = _array_values(cluster_group, "stats")
        if stats.ndim != 2:
            raise ValueError("Compact marker stats must be two-dimensional")
        if stats.shape[0] != feature_index.shape[0]:
            raise ValueError("Compact marker stats do not align with feature_index")
        columns = _resolve_compact_stat_columns(slot_group, stats.shape[1])
        df = pd.DataFrame(stats, columns=columns)
        df["feature_index"] = feature_index
        for column in LEGACY_STAT_COLUMNS:
            if column not in df.columns:
                df[column] = np.nan
        display_columns = (
            MARKER_STAT_COLUMNS
            if {"auc", "p_value_adjusted"}.intersection(columns)
            else LEGACY_STAT_COLUMNS
        )
        displayed = _display_frame(
            df,
            group_id=group_id,
            feature_names=feature_names,
            out_stat_columns=display_columns,
        )
        return sort_marker_results(displayed)

    if "names" in cluster_group and "scores" in cluster_group:
        stored_names = _array_values(cluster_group, "names").astype(str)
        scores = np.asarray(_array_values(cluster_group, "scores"), dtype=np.float64)
        if stored_names.ndim != 1 or scores.ndim != 1:
            raise ValueError("Legacy marker names and scores must be one-dimensional")
        if stored_names.shape != scores.shape:
            raise ValueError("Legacy marker names and scores must align")
        lookup_values = feature_names if feature_ids is None else feature_ids
        lookup = {str(value): index for index, value in enumerate(lookup_values)}
        for index, value in enumerate(feature_names):
            lookup.setdefault(str(value), index)
        resolved_indices = [lookup.get(value) for value in stored_names]
        feature_index = pd.array(resolved_indices, dtype="Int64")
        names = np.asarray(feature_names, dtype=object)
        display_names = [
            names[index] if index is not None else value
            for value, index in zip(stored_names, resolved_indices, strict=True)
        ]
        frame = pd.DataFrame(
            {
                "group_id": group_id,
                "feature_name": display_names,
                "feature_index": feature_index,
                "score": scores,
            }
        )
        for column in LEGACY_STAT_COLUMNS:
            if column not in frame.columns:
                frame[column] = np.nan
        ordered = [
            "group_id",
            "feature_name",
            "feature_index",
            *LEGACY_STAT_COLUMNS,
        ]
        return (
            frame[ordered]
            .sort_values(
                ["score", "feature_name"],
                ascending=[False, True],
            )
            .reset_index(drop=True)
        )

    available_cols = [col for col in out_cols if col in cluster_group]
    if not available_cols:
        return _display_frame(
            _empty_marker_frame(tuple(out_cols)),
            group_id=group_id,
            feature_names=feature_names,
            out_stat_columns=LEGACY_STAT_COLUMNS,
        )
    cols = [_array_values(cluster_group, column) for column in available_cols]
    df = pd.DataFrame(dict(zip(available_cols, cols, strict=True)))
    for column in LEGACY_STAT_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
    if "feature_index" not in df.columns:
        raise ValueError("Legacy per-column marker tables require feature_index")
    return _display_frame(
        df,
        group_id=group_id,
        feature_names=feature_names,
        out_stat_columns=LEGACY_STAT_COLUMNS,
    )


def _read_canonical_marker_table(
    slot_group: zarr.Group,
    cluster_group: zarr.Group,
    feature_names: np.ndarray,
    *,
    group_id: Any,
) -> pd.DataFrame:
    """Read canonical compact marker tables with validated metadata."""
    if "feature_index" not in slot_group or "stats" not in cluster_group:
        raise ValueError("Canonical marker tables require feature_index and stats")
    for name, expected in _MARKER_METADATA.items():
        if name not in slot_group.attrs:
            raise ValueError(f"Canonical marker tables require {name} metadata")
        value = slot_group.attrs[name]
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"Canonical marker metadata {name!r} must be {expected!r}")
    columns = _stored_stat_columns(slot_group, required=True)
    assert columns is not None
    required = set(MARKER_STAT_COLUMNS)
    if set(columns) != required:
        raise ValueError(
            "Canonical marker tables must store the complete named stat_columns"
        )
    for name in ("n_group", "n_reference"):
        value = cluster_group.attrs.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | np.integer)
            or int(value) < 2
        ):
            raise ValueError(
                f"Canonical marker groups require integer {name} metadata >= 2"
            )
    feature_index = _array_values(slot_group, "feature_index")
    stats = _array_values(cluster_group, "stats")
    if stats.ndim != 2 or stats.shape[1] != len(columns):
        raise ValueError("Canonical marker stats do not match stat_columns")
    if stats.shape[0] != feature_index.shape[0]:
        raise ValueError("Canonical marker stats do not align with feature_index")
    if stats.shape[0] == 0:
        raise ValueError("Canonical marker groups must contain marker rows")
    if stats.dtype.kind != "f":
        raise ValueError("Canonical marker stats must use a floating dtype")
    adjusted = stats[:, columns.index("p_value_adjusted")]
    if not np.isfinite(adjusted).all():
        raise ValueError("Canonical marker p_value_adjusted values must all be finite")
    if not np.isfinite(stats).all():
        raise ValueError("Canonical marker statistics must all be finite")
    df = pd.DataFrame(stats, columns=columns)
    df["feature_index"] = feature_index
    displayed = _display_frame(
        df,
        group_id=group_id,
        feature_names=feature_names,
        out_stat_columns=MARKER_STAT_COLUMNS,
        require_integer_indices=True,
        require_unique_indices=True,
    )
    return sort_marker_results(displayed)


def _validate_marker_slot(
    slot_group: zarr.Group,
    feature_names: np.ndarray,
    *,
    expected_group_cell_counts: dict[str, tuple[int, int]] | None = None,
) -> None:
    group_names = sorted(slot_group.group_keys())
    if not group_names:
        raise ValueError("Canonical marker tables must contain populated groups")
    if expected_group_cell_counts is not None and set(group_names) != set(
        expected_group_cell_counts
    ):
        raise ValueError("Canonical marker groups do not match the requested groups")
    for group_name in group_names:
        cluster_group = slot_group[group_name]
        if not isinstance(cluster_group, zarr.Group):
            raise TypeError(f"Marker group {group_name!r} must be a group")
        _read_canonical_marker_table(
            slot_group,
            cluster_group,
            feature_names,
            group_id=group_name,
        )
        if expected_group_cell_counts is None:
            continue
        expected_group, expected_reference = expected_group_cell_counts[group_name]
        if (
            cluster_group.attrs.get("n_group") != expected_group
            or cluster_group.attrs.get("n_reference") != expected_reference
        ):
            raise ValueError(
                f"Canonical marker group {group_name!r} has stale cell counts"
            )


def load_marker_table(
    slot_group: zarr.Group,
    cluster_group: zarr.Group,
    feature_names: np.ndarray,
    *,
    group_id: Any,
    feature_ids: np.ndarray | None = None,
) -> pd.DataFrame:
    """Dispatch marker reading by structural layout."""
    canonical_metadata = any(
        name in slot_group.attrs for name in _MARKER_METADATA
    ) or any(name in cluster_group.attrs for name in ("n_group", "n_reference"))
    if canonical_metadata:
        return _read_canonical_marker_table(
            slot_group,
            cluster_group,
            feature_names,
            group_id=group_id,
        )

    frame = read_legacy_marker_table(
        slot_group,
        cluster_group,
        feature_names,
        group_id=group_id,
        feature_ids=feature_ids,
    )
    return _maybe_attach_adjusted_pvalues(frame)
