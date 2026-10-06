"""Rank marker results and the readers of their persisted statistics."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import zarr
from scipy.special import ndtr

from ..statistical import adjust_pvalues
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
# ``fold_change`` is the group mean over the rest mean when both are
# non-negative and the rest mean is positive, +inf when only the group mean is
# positive, and NaN when both means are 0 or either is negative. Marker slots
# record this policy as metadata, which the loader and the reuse validator
# require, so a table that uses the 100.1 and 0 sentinels of earlier releases
# is neither read nor reused. Operation revision 2 of run_marker_search, not a
# recorded parameter, identifies tables that follow the policy, so planning
# reports such an earlier table as a superseded match.
MARKER_FOLD_CHANGE_POLICY = "ratio_of_means_inf_when_rest_zero_nan_when_undefined"
_MARKER_METADATA = {
    "method": MARKER_METHOD,
    "alternative": MARKER_ALTERNATIVE,
    "tie_correction": MARKER_TIE_CORRECTION,
    "continuity_correction": MARKER_CONTINUITY_CORRECTION,
    "adjustment_method": MARKER_ADJUSTMENT_METHOD,
    "adjustment_scope": MARKER_ADJUSTMENT_SCOPE,
    "fold_change_policy": MARKER_FOLD_CHANGE_POLICY,
}

__all__ = [
    "MARKER_STAT_COLUMNS",
    "RankMarkerResult",
    "load_marker_table",
]

# The rank kernels compute every stored column except ``p_value_adjusted``,
# with the Mann-Whitney z statistic in place of ``p_value``.
_RANK_COLUMNS = len(MARKER_STAT_COLUMNS) - 1
_P_VALUE = MARKER_STAT_COLUMNS.index("p_value")


def _statistics_violation(stats: np.ndarray, columns: Sequence[str]) -> str | None:
    """Return how one group's stored marker statistics break their contract.

    Every statistic but ``fold_change`` is finite, and ``fold_change``
    follows ``MARKER_FOLD_CHANGE_POLICY``: it is NaN or lies in [0, +inf], it
    is +inf only where ``mean_rest`` is 0, and it is NaN only where both
    means are 0 or either is negative. Stored values are rounded to five
    decimals, so the checks hold in one direction only: rounding keeps a mean
    of 0 at 0 and the sign of a negative mean, which can round to -0.0, but
    it also rounds a small positive mean to 0.

    Returns:
        A description of the first broken rule, or None.
    """
    # One column at a time, so the checks hold no copy of the table.
    for position, name in enumerate(columns):
        if name != "fold_change" and not np.isfinite(stats[:, position]).all():
            return "statistics other than fold_change must all be finite"
    ratios = stats[:, columns.index("fold_change")]
    means = stats[:, columns.index("mean")]
    rest_means = stats[:, columns.index("mean_rest")]
    if np.any(ratios < 0.0):
        return "fold_change holds a negative value"
    if np.any(np.isposinf(ratios) & (rest_means != 0.0)):
        return "fold_change is +inf where mean_rest is not 0"
    # The sign bit marks a negative mean and the -0.0 that it can round to.
    undefined = (
        ((means == 0.0) & (rest_means == 0.0))
        | np.signbit(means)
        | np.signbit(rest_means)
    )
    if np.any(np.isnan(ratios) & ~undefined):
        return (
            "fold_change is NaN where both means are non-negative and one is positive"
        )
    return None


def stored_table_bytes(n_features: int) -> int:
    """Return a bound on what finishing and writing one group's table holds.

    While a table is finished, the intermediates of its p-values and their
    adjustment take at most one more table's size; while it is written, the
    bytes Zarr encodes and compresses from it take at most two.
    """
    table = max(0, int(n_features)) * len(MARKER_STAT_COLUMNS)
    return 3 * table * np.dtype(np.float64).itemsize


def _validate_rank_marker_groups(group_sizes: np.ndarray) -> None:
    """Refuse groups that a stored marker table cannot hold.

    A stored group and its one-versus-rest complement each need at least two
    cells, which at least two groups of at least two cells give.
    """
    if group_sizes.size < 2:
        raise ValueError("Rank markers require at least two populated groups")
    if np.any(group_sizes < 2):
        raise ValueError("Rank markers require at least two cells in every group")


def _ranked_table(
    stats: np.ndarray,
    columns: Sequence[str],
    feature_index: np.ndarray,
    feature_names: np.ndarray,
    group_id: Any,
) -> pd.DataFrame:
    """Name and rank one group's stored statistics for display."""
    names = np.asarray(feature_names, dtype=object)
    if names.ndim != 1:
        raise ValueError("Feature names must be one-dimensional")
    frame = pd.DataFrame(stats, columns=list(columns))
    frame["group_id"] = group_id
    frame["feature_name"] = names[feature_index]
    frame["feature_index"] = feature_index
    return sort_marker_results(
        frame[["group_id", "feature_name", "feature_index", *MARKER_STAT_COLUMNS]]
    )


@dataclass(frozen=True, slots=True, eq=False)
class RankMarkerResult:
    """Rank marker statistics of every group, as the rank kernels compute them.

    ``statistics`` holds, for each tested feature and group, the first eight
    ``MARKER_STAT_COLUMNS`` with the continuity- and tie-corrected
    Mann-Whitney z statistic in place of ``p_value``. ``stored_statistics``
    finishes one group into the table that a marker artifact stores, and
    ``table`` into the ranked table that such an artifact reads back.

    Attributes:
        group_ids: Sorted unique group labels, at least two.
        group_sizes: Number of cells in each group, at least two.
        feature_index: Ascending assay indices of the tested features, at
            least one.
        statistics: Features-by-groups-by-8 rank statistics.
    """

    group_ids: np.ndarray = field(repr=False)
    group_sizes: np.ndarray = field(repr=False)
    feature_index: np.ndarray = field(repr=False)
    statistics: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        group_ids = np.asarray(self.group_ids)
        group_sizes = np.asarray(self.group_sizes)
        feature_index = np.asarray(self.feature_index)
        statistics = np.asarray(self.statistics)
        if (
            group_ids.ndim != 1
            or has_duplicates(group_ids)
            or group_sizes.shape != group_ids.shape
            or group_sizes.dtype.kind not in {"i", "u"}
        ):
            raise ValueError(
                "Rank marker groups must be unique labels aligned with integer sizes"
            )
        _validate_rank_marker_groups(group_sizes)
        # Neighbor comparisons, unlike differences, do not wrap for unsigned
        # indices.
        if (
            feature_index.ndim != 1
            or feature_index.size == 0
            or feature_index.dtype.kind not in {"i", "u"}
            or np.any(feature_index[1:] <= feature_index[:-1])
            or feature_index[0] < 0
        ):
            raise ValueError(
                "Rank marker feature_index must hold one or more ascending "
                "unique non-negative indices"
            )
        if statistics.dtype != np.float64 or statistics.shape != (
            feature_index.size,
            group_ids.size,
            _RANK_COLUMNS,
        ):
            raise ValueError(
                "Rank marker statistics must be float64 with one row per feature "
                f"and {_RANK_COLUMNS} statistics per group"
            )
        for name, value in (
            ("group_ids", group_ids),
            ("group_sizes", group_sizes),
            ("feature_index", feature_index),
            ("statistics", statistics),
        ):
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def _position(self, group_id: Any) -> int:
        matches = np.flatnonzero(self.group_ids == group_id)
        if matches.size != 1:
            raise ValueError(f"Rank marker result has no group {group_id!r}")
        return int(matches[0])

    def stored_statistics(self, group_id: Any) -> np.ndarray:
        """Return one group's marker statistics as a marker artifact stores them.

        Rows follow ``feature_index`` and columns ``MARKER_STAT_COLUMNS``. The
        z statistic becomes its two-sided p-value, every other statistic is
        rounded to five decimals, and ``p_value_adjusted`` holds the
        Benjamini-Hochberg values of the unrounded p-values over every tested
        feature.

        Args:
            group_id: One of ``group_ids``.

        Raises:
            ValueError: If ``group_id`` is not a group of the result, a
                statistic other than ``fold_change`` is not finite, or
                ``fold_change`` breaks ``MARKER_FOLD_CHANGE_POLICY``.
        """
        rank = self.statistics[:, self._position(group_id)]
        stored = np.empty((rank.shape[0], len(MARKER_STAT_COLUMNS)), dtype=np.float64)
        np.round(rank, 5, out=stored[:, :_RANK_COLUMNS])
        p_values = 2.0 * ndtr(-np.abs(rank[:, _P_VALUE]))
        stored[:, _P_VALUE] = p_values
        stored[:, -1] = adjust_pvalues(p_values, "fdr_bh")
        violation = _statistics_violation(stored, MARKER_STAT_COLUMNS)
        if violation is not None:
            raise ValueError(f"Marker {violation}")
        return stored

    def table(self, group_id: Any, feature_names: np.ndarray) -> pd.DataFrame:
        """Return one group's ranked marker table.

        The table equals the one read back from a marker artifact written from
        this result, before ``get_markers`` filters it: ``group_id``,
        ``feature_name``, and ``feature_index`` columns followed by
        ``MARKER_STAT_COLUMNS``, by descending score, then p-value, then
        feature name.

        Args:
            group_id: One of ``group_ids``; the table's ``group_id`` column
                holds the string form of that label, which names the group in
                a marker artifact.
            feature_names: Names of every feature of the assay, indexed by
                ``feature_index``.
        """
        return _ranked_table(
            self.stored_statistics(group_id),
            MARKER_STAT_COLUMNS,
            self.feature_index.astype(np.int64, copy=False),
            feature_names,
            str(self.group_ids[self._position(group_id)]),
        )


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
    if "fold_change_policy" not in slot_group.attrs:
        raise ValueError(
            "Marker table records no fold_change_policy metadata: an earlier "
            "Scarf release wrote it, and its fold_change column holds that "
            "release's sentinels, 100.1 for a feature absent from the other "
            "cells and 0 for a feature absent from every cell. Recompute it "
            "from the same clustering and feature selection with "
            "DataStore.run_marker_search; for a pipeline run, "
            'ds.run_marker_search(run["clusters"], '
            'features=run["feature_universe"]). A saved agent result keeps '
            "the table that it read, so run that analysis again"
        )
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
    violation = _statistics_violation(stats, columns)
    if violation is not None:
        raise ValueError(f"Canonical marker {violation}")
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
    return _ranked_table(stats, columns, feature_index, feature_names, group_id)
