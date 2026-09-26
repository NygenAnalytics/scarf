"""Readers for persisted marker statistics."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import zarr

from .rank import sort_marker_results
from ...utils.arrays import has_duplicates

MARKER_STAT_COLUMNS = (
    "score",
    "mean",
    "mean_rest",
    "frac_exp",
    "frac_exp_rest",
    "fold_change",
    "p_value",
    "auc",
    "p_value_adjusted",
)
MARKER_METHOD = "mannwhitneyu"
MARKER_ALTERNATIVE = "two-sided"
MARKER_TIE_CORRECTION = True
MARKER_CONTINUITY_CORRECTION = True
MARKER_ADJUSTMENT_METHOD = "fdr_bh"
MARKER_ADJUSTMENT_SCOPE = "within_group_all_tested_features"
_MARKER_METADATA = {
    "method": MARKER_METHOD,
    "alternative": MARKER_ALTERNATIVE,
    "tie_correction": MARKER_TIE_CORRECTION,
    "continuity_correction": MARKER_CONTINUITY_CORRECTION,
    "adjustment_method": MARKER_ADJUSTMENT_METHOD,
    "adjustment_scope": MARKER_ADJUSTMENT_SCOPE,
}

__all__ = [
    "MARKER_STAT_COLUMNS",
    "load_marker_table",
]


def _array_values(group: zarr.Group, name: str) -> np.ndarray:
    value = group[name]
    if not isinstance(value, zarr.Array):
        raise TypeError(f"Marker field {name!r} must be an array")
    return np.asarray(value[:])


def _stored_stat_columns(slot_group: zarr.Group) -> list[str]:
    stored = slot_group.attrs.get("stat_columns")
    if stored is None:
        raise ValueError("Canonical marker tables require stat_columns metadata")
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


def _validated_feature_indices(values: np.ndarray, *, n_features: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError("Marker feature_index must be one-dimensional")
    if raw.dtype.kind not in {"i", "u"}:
        raise ValueError("Canonical marker feature_index must use an integer dtype")
    indices = raw.astype(np.int64)
    invalid = (indices < 0) | (indices >= n_features)
    if invalid.any():
        unresolved = ", ".join(str(value) for value in indices[invalid][:5])
        raise ValueError(
            "Marker feature_index contains unresolved or out-of-range values: "
            + unresolved
        )
    if has_duplicates(indices):
        raise ValueError("Canonical marker feature_index must contain unique values")
    return indices


def _canonical_slot(
    slot_group: zarr.Group, n_features: int
) -> tuple[list[str], np.ndarray]:
    """Validate slot metadata and return stat columns and feature indices."""
    if "feature_index" not in slot_group:
        raise ValueError("Canonical marker tables require feature_index and stats")
    for name, expected in _MARKER_METADATA.items():
        if name not in slot_group.attrs:
            raise ValueError(f"Canonical marker tables require {name} metadata")
        value = slot_group.attrs[name]
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"Canonical marker metadata {name!r} must be {expected!r}")
    columns = _stored_stat_columns(slot_group)
    if set(columns) != set(MARKER_STAT_COLUMNS):
        raise ValueError(
            "Canonical marker tables must store the complete named stat_columns"
        )
    feature_index = _validated_feature_indices(
        _array_values(slot_group, "feature_index"),
        n_features=n_features,
    )
    if feature_index.shape[0] == 0:
        raise ValueError("Canonical marker groups must contain marker rows")
    return columns, feature_index


def _canonical_cluster_stats(
    cluster_group: zarr.Group, columns: list[str], n_rows: int
) -> np.ndarray:
    """Validate one canonical marker group and return its stats."""
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
    if "stats" not in cluster_group:
        raise ValueError("Canonical marker tables require feature_index and stats")
    stats = _array_values(cluster_group, "stats")
    if stats.ndim != 2 or stats.shape[1] != len(columns):
        raise ValueError("Canonical marker stats do not match stat_columns")
    if stats.shape[0] != n_rows:
        raise ValueError("Canonical marker stats do not align with feature_index")
    if stats.dtype.kind != "f":
        raise ValueError("Canonical marker stats must use a floating dtype")
    if not np.isfinite(stats[:, columns.index("p_value_adjusted")]).all():
        raise ValueError("Canonical marker p_value_adjusted values must all be finite")
    if not np.isfinite(stats).all():
        raise ValueError("Canonical marker statistics must all be finite")
    return stats


def _validate_marker_slot(
    slot_group: zarr.Group,
    feature_names: np.ndarray,
    *,
    expected_group_cell_counts: dict[str, tuple[int, int]] | None = None,
    workers: int = 1,
) -> None:
    """Check every marker group without building display tables.

    ``workers`` reads groups concurrently, which hides object-store latency.
    """
    group_names = sorted(slot_group.group_keys())
    if not group_names:
        raise ValueError("Canonical marker tables must contain populated groups")
    if expected_group_cell_counts is not None and set(group_names) != set(
        expected_group_cell_counts
    ):
        raise ValueError("Canonical marker groups do not match the requested groups")
    columns, feature_index = _canonical_slot(slot_group, len(feature_names))

    def check(group_name: str) -> None:
        cluster_group = slot_group[group_name]
        if not isinstance(cluster_group, zarr.Group):
            raise TypeError(f"Marker group {group_name!r} must be a group")
        _canonical_cluster_stats(cluster_group, columns, feature_index.shape[0])
        if expected_group_cell_counts is None:
            return
        expected_group, expected_reference = expected_group_cell_counts[group_name]
        if (
            cluster_group.attrs.get("n_group") != expected_group
            or cluster_group.attrs.get("n_reference") != expected_reference
        ):
            raise ValueError(
                f"Canonical marker group {group_name!r} has stale cell counts"
            )

    if workers <= 1:
        for group_name in group_names:
            check(group_name)
        return
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(workers, len(group_names))) as pool:
        list(pool.map(check, group_names))


def load_marker_table(
    slot_group: zarr.Group,
    cluster_group: zarr.Group,
    feature_names: np.ndarray,
    *,
    group_id: Any,
) -> pd.DataFrame:
    """Read one canonical marker group into a named, ranked table."""
    columns, feature_index = _canonical_slot(slot_group, len(feature_names))
    stats = _canonical_cluster_stats(cluster_group, columns, feature_index.shape[0])
    names = np.asarray(feature_names, dtype=object)
    if names.ndim != 1:
        raise ValueError("Feature names must be one-dimensional")
    frame = pd.DataFrame(stats, columns=columns)
    frame["group_id"] = group_id
    frame["feature_name"] = names[feature_index]
    frame["feature_index"] = feature_index
    return sort_marker_results(
        frame[["group_id", "feature_name", "feature_index", *MARKER_STAT_COLUMNS]]
    )
