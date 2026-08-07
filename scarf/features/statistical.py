"""Statistical tests for group-wise distribution comparisons.

This module implements non-parametric tests on one or more value arrays
grouped by a categorical variable. It is the compute backend behind
``DataStore.run_statistical_testing`` and operates purely on in-memory
arrays so that both the datastore method and plotting can share one
implementation.

Tests are standard for zero-inflated, non-normal single-cell values:

- Two independent groups: two-sided Mann-Whitney U, reusing the rank-based
  implementation from :mod:`scarf.features.markers.rank` (the same
  continuity- and tie-corrected statistic the marker search reports).
- Three or more independent groups: Kruskal-Wallis with an optional Dunn's
  post-hoc test for pairwise significance.
- Paired samples (aggregated to biological samples): Wilcoxon signed-rank.

Multiple-testing correction is delegated to ``statsmodels.stats.multitest``
(the same backend the marker statistics use).
"""

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import kruskal, norm, wilcoxon

from .markers.rank import mannwhitneyu_from_ranks

TestMethod = Literal["auto", "mann_whitney", "kruskal_wallis", "wilcoxon"]
PosthocMethod = Literal["dunn"]
AdjustmentMethod = Literal["fdr_bh", "bonferroni", "holm", "none"]
SampleStatistic = Literal["mean", "median", "fraction"]

MANN_WHITNEY_COLUMNS = (
    "group_1",
    "group_2",
    "n_1",
    "n_2",
    "u_statistic",
    "mean_1",
    "mean_2",
    "mean_difference",
    "p_value",
)
KRUSKAL_WALLIS_COLUMNS = ("kruskal_statistic", "df", "p_value")
DUNN_COLUMNS = ("group_1", "group_2", "z", "p_value")
WILCOXON_COLUMNS = ("group_1", "group_2", "n_pairs", "statistic", "p_value")

_METHOD_COLUMNS: dict[str, tuple[str, ...]] = {
    "mann_whitney": MANN_WHITNEY_COLUMNS,
    "kruskal_wallis": KRUSKAL_WALLIS_COLUMNS,
    "wilcoxon": WILCOXON_COLUMNS,
}

__all__ = [
    "DUNN_COLUMNS",
    "KRUSKAL_WALLIS_COLUMNS",
    "MANN_WHITNEY_COLUMNS",
    "StatisticalTestResult",
    "WILCOXON_COLUMNS",
    "adjust_pvalues",
    "aggregate_samples",
    "compare_group_distributions",
]


@dataclass(frozen=True, slots=True, eq=False)
class StatisticalTestResult:
    """Statistical tests for one grouping across multiple value keys."""

    method: str
    posthoc: str | None
    adjustment_method: str
    group_key: str
    cell_key: str | None
    sample_by: str | None = None
    pair_by: str | None = None
    sample_stat: str = "mean"
    expression_cutoff: float = 0.0
    n_groups: int = 0
    n_cells: int = 0
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)


def adjust_pvalues(
    p_values: np.ndarray,
    method: AdjustmentMethod = "fdr_bh",
) -> np.ndarray:
    """Return multiple-testing corrected p-values.

    ``method`` may be ``"fdr_bh"`` (default), ``"bonferroni"``, ``"holm"``,
    or ``"none"``. Non-finite entries are left as ``NaN`` and excluded from
    the correction.
    """
    values = np.asarray(p_values, dtype=np.float64)
    if method not in ("fdr_bh", "bonferroni", "holm", "none"):
        raise ValueError("adjustment must be 'fdr_bh', 'bonferroni', 'holm', or 'none'")
    if method == "none":
        return values.copy()
    from statsmodels.stats.multitest import multipletests

    adjusted = np.full(values.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(values)
    if not np.any(mask):
        return adjusted
    _, corrected, _, _ = multipletests(values[mask], method=method)
    adjusted[mask] = corrected
    return adjusted


def _category_sort_key(value: Any) -> tuple[int, str]:
    numeric = isinstance(
        value, int | float | np.integer | np.floating
    ) and not isinstance(
        value,
        bool,
    )
    return (0 if numeric else 1, str(value))


def _natural_group_order(groups: np.ndarray) -> list[Any]:
    return sorted(list(pd.unique(groups)), key=_category_sort_key)


def _native(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _valid_group_mask(groups: np.ndarray) -> np.ndarray:
    as_str = np.asarray(groups, dtype=str)
    return ~np.isin(as_str, ["", "nan", "None"])


def aggregate_samples(
    values: np.ndarray,
    groups: np.ndarray,
    samples: np.ndarray,
    *,
    sample_stat: SampleStatistic = "mean",
    expression_cutoff: float = 0.0,
    pairs: np.ndarray | None = None,
) -> pd.DataFrame:
    """Aggregate per-cell values within biological samples.

    Returns a frame with one row per ``(sample, group)`` (and ``pair`` when
    supplied) holding the aggregated ``value``. ``sample_stat`` is ``"mean"``,
    ``"median"``, or ``"fraction"`` (fraction of cells above
    ``expression_cutoff``). Each sample must map to a single pair key.
    """
    frame = pd.DataFrame(
        {
            "value": np.asarray(values, dtype=np.float64),
            "group": np.asarray(groups, dtype=object),
            "sample": np.asarray(samples, dtype=object),
        }
    )
    valid = pd.notna(frame["sample"]) & (frame["sample"].astype(str) != "")
    frame = frame.loc[valid]
    if frame.empty:
        raise ValueError("No selected cells have a valid sample value")
    if pairs is not None:
        frame["pair"] = np.asarray(pairs, dtype=object)
        pair_counts = frame.groupby("sample", observed=False)["pair"].nunique()
        if (pair_counts > 1).any():
            raise ValueError("Each sample must map to exactly one pair key")
    grouped = frame.groupby(["sample", "group"], observed=False)["value"]
    if sample_stat == "mean":
        values_by_group = grouped.mean()
    elif sample_stat == "median":
        values_by_group = grouped.median()
    elif sample_stat == "fraction":
        values_by_group = grouped.apply(
            lambda value: float(
                np.mean(value.to_numpy(dtype=np.float64) > float(expression_cutoff))
            )
        )
    else:
        raise ValueError("sample_stat must be 'mean', 'median', or 'fraction'")
    out = values_by_group.rename("value").reset_index()
    if pairs is not None:
        pair_for_sample = frame.groupby("sample", observed=False)["pair"].first()
        out["pair"] = out["sample"].map(pair_for_sample)
    return out


def _mann_whitney(
    values: np.ndarray,
    groups: np.ndarray,
    present: list[Any],
    comparisons: Sequence[tuple[Any, Any]] | None,
) -> pd.DataFrame:
    if len(present) != 2:
        raise ValueError(
            "mann_whitney requires exactly two groups; use groups= to select "
            "two groups or kruskal_wallis for three or more"
        )
    g1, g2 = present
    if comparisons is not None:
        for left, right in comparisons:
            if {_native(left), _native(right)} != {g1, g2}:
                raise ValueError(
                    "mann_whitney comparisons must reference the two selected "
                    f"groups ({g1!r}, {g2!r})"
                )
    m1 = values[groups == g1]
    m2 = values[groups == g2]
    n1 = len(m1)
    n2 = len(m2)
    if n1 < 2 or n2 < 2:
        raise ValueError("mann_whitney requires at least two cells in every group")
    ranked = pd.DataFrame(
        {"feature": pd.Series(np.concatenate([m1, m2])).rank(method="average")}
    )
    group_vec = np.concatenate([np.repeat(g1, n1), np.repeat(g2, n2)]).astype(object)
    p_values = mannwhitneyu_from_ranks(
        ranked,
        group_vec,
        np.array([g1, g2], dtype=object),
    )
    u1 = float(ranked.iloc[:n1]["feature"].sum()) - n1 * (n1 + 1) / 2
    mean_1 = float(np.mean(m1))
    mean_2 = float(np.mean(m2))
    return pd.DataFrame(
        [
            {
                "group_1": g1,
                "group_2": g2,
                "n_1": n1,
                "n_2": n2,
                "u_statistic": u1,
                "mean_1": mean_1,
                "mean_2": mean_2,
                "mean_difference": mean_1 - mean_2,
                "p_value": float(p_values.loc[g1, "feature"]),
            }
        ],
        columns=list(MANN_WHITNEY_COLUMNS),
    )


def _kruskal_wallis(
    values: np.ndarray,
    groups: np.ndarray,
    present: list[Any],
) -> pd.DataFrame:
    if len(present) < 3:
        raise ValueError(
            "kruskal_wallis requires at least three groups; use mann_whitney "
            "for exactly two"
        )
    group_values = [values[groups == g] for g in present]
    if any(len(v) < 2 for v in group_values):
        raise ValueError("kruskal_wallis requires at least two cells in every group")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        statistic, p_value = kruskal(*group_values)
    if not np.isfinite(statistic) or not np.isfinite(p_value):
        statistic = 0.0
        p_value = 1.0
    return pd.DataFrame(
        [
            {
                "kruskal_statistic": float(statistic),
                "df": len(present) - 1,
                "p_value": float(p_value),
            }
        ],
        columns=list(KRUSKAL_WALLIS_COLUMNS),
    )


def _dunn_posthoc(
    values: np.ndarray,
    groups: np.ndarray,
    present: list[Any],
    comparisons: Sequence[tuple[Any, Any]] | None,
) -> pd.DataFrame:
    ranked = pd.Series(values).rank(method="average").to_numpy(dtype=np.float64)
    n_total = len(values)
    _, counts = np.unique(values, return_counts=True)
    tied = counts[counts > 1]
    tie_correction = float(np.sum(tied**3 - tied)) if tied.size else 0.0
    variance = (n_total * (n_total + 1)) / 12 - tie_correction / (12 * (n_total - 1))
    mean_rank = {g: float(np.mean(ranked[groups == g])) for g in present}
    sizes = {g: int((groups == g).sum()) for g in present}
    pairs = (
        list(comparisons) if comparisons is not None else list(combinations(present, 2))
    )
    rows: list[dict[str, Any]] = []
    for left, right in pairs:
        left, right = _native(left), _native(right)
        if left not in mean_rank or right not in mean_rank:
            raise ValueError(
                "comparisons references a group not present in the data: "
                f"{left!r} or {right!r}"
            )
        standard_error = np.sqrt(variance * (1 / sizes[left] + 1 / sizes[right]))
        z = (
            (mean_rank[left] - mean_rank[right]) / standard_error
            if standard_error > 0
            else 0.0
        )
        rows.append(
            {
                "group_1": left,
                "group_2": right,
                "z": float(z),
                "p_value": float(2 * norm.sf(abs(z))),
            }
        )
    return pd.DataFrame(rows, columns=list(DUNN_COLUMNS))


def _wilcoxon_signed_rank(
    aggregated: pd.DataFrame,
    present: list[Any],
    comparisons: Sequence[tuple[Any, Any]] | None,
) -> pd.DataFrame:
    if len(present) != 2:
        raise ValueError(
            "wilcoxon requires exactly two groups on aggregated sample data"
        )
    g1, g2 = present
    if comparisons is not None:
        for left, right in comparisons:
            if {_native(left), _native(right)} != {g1, g2}:
                raise ValueError(
                    "wilcoxon comparisons must reference the two selected "
                    f"groups ({g1!r}, {g2!r})"
                )
    left = aggregated[aggregated["group"] == g1]
    right = aggregated[aggregated["group"] == g2]
    merged = left.merge(right, on="pair", suffixes=("_1", "_2"))
    n_pairs = len(merged)
    if n_pairs == 0:
        raise ValueError(
            "wilcoxon found no samples measured in both groups; pair keys "
            "must be shared across the two groups"
        )
    if n_pairs < 2:
        raise ValueError("wilcoxon requires at least two matched pairs")
    differences = merged["value_1"].to_numpy(dtype=np.float64) - merged[
        "value_2"
    ].to_numpy(dtype=np.float64)
    if not np.any(differences):
        statistic = 0.0
        p_value = 1.0
    else:
        statistic, p_value = wilcoxon(
            merged["value_1"].to_numpy(dtype=np.float64),
            merged["value_2"].to_numpy(dtype=np.float64),
        )
    return pd.DataFrame(
        [
            {
                "group_1": g1,
                "group_2": g2,
                "n_pairs": n_pairs,
                "statistic": float(statistic),
                "p_value": float(p_value),
            }
        ],
        columns=list(WILCOXON_COLUMNS),
    )


def compare_group_distributions(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    test: TestMethod = "auto",
    posthoc: PosthocMethod | None = None,
    adjustment: AdjustmentMethod = "none",
    samples: np.ndarray | None = None,
    pairs: np.ndarray | None = None,
    comparisons: Sequence[tuple[Any, Any]] | None = None,
    sample_stat: SampleStatistic = "mean",
    expression_cutoff: float = 0.0,
) -> pd.DataFrame:
    """Compare one value array across groups with a non-parametric test.

    ``values`` and ``groups`` must be one-dimensional and equal in length.
    When ``samples`` is provided, cells are aggregated to biological samples
    first using ``sample_stat``. ``pairs`` (a subject or donor id per cell)
    enables the paired Wilcoxon signed-rank test, which is only meaningful on
    aggregated sample data and therefore requires ``samples`` too.

    ``test`` is ``"auto"`` (pick by design: paired -> Wilcoxon, two groups ->
    Mann-Whitney, three or more -> Kruskal-Wallis), ``"mann_whitney"``,
    ``"kruskal_wallis"``, or ``"wilcoxon"``. ``posthoc="dunn"`` adds pairwise
    Dunn's tests after Kruskal-Wallis. ``comparisons`` restricts the pairwise
    rows to the listed group pairs. ``adjustment`` corrects p-values within
    the returned table when it holds multiple comparisons; pass ``"none"`` to
    adjust across keys in the caller instead.

    Returns:
        A DataFrame with one row per comparison (Mann-Whitney, Dunn, and
        Wilcoxon) or a single row per key (Kruskal-Wallis).
    """
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups, dtype=object)
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if groups.shape != values.shape:
        raise ValueError("groups length must match values")
    if not np.isfinite(values).all():
        raise ValueError("values must contain only finite entries")
    if posthoc not in (None, "dunn"):
        raise ValueError("posthoc must be 'dunn' or None")
    if comparisons is not None and not comparisons:
        raise ValueError("comparisons must be non-empty when provided")

    valid = _valid_group_mask(groups)
    values = values[valid]
    groups = groups[valid]
    if samples is not None:
        samples = np.asarray(samples, dtype=object)[valid]
    if pairs is not None:
        pairs = np.asarray(pairs, dtype=object)[valid]
    if len(values) == 0:
        raise ValueError("No values remain after dropping missing groups")

    aggregated: pd.DataFrame | None = None
    if samples is not None or pairs is not None:
        if pairs is not None and samples is None:
            raise ValueError(
                "pairs requires samples: pairing is only meaningful on "
                "aggregated sample data"
            )
        assert samples is not None
        aggregated = aggregate_samples(
            values,
            groups,
            samples,
            sample_stat=sample_stat,
            expression_cutoff=expression_cutoff,
            pairs=pairs,
        )
        values = aggregated["value"].to_numpy(dtype=np.float64)
        groups = aggregated["group"].to_numpy(dtype=object)
        pairs = aggregated["pair"].to_numpy(dtype=object) if pairs is not None else None

    present = _natural_group_order(groups)
    if len(present) < 2:
        raise ValueError("At least two populated groups are required")

    if test == "auto":
        if pairs is not None:
            test = "wilcoxon"
        elif len(present) == 2:
            test = "mann_whitney"
        else:
            test = "kruskal_wallis"
    if test not in ("mann_whitney", "kruskal_wallis", "wilcoxon"):
        raise ValueError(
            "test must be 'auto', 'mann_whitney', 'kruskal_wallis', or "
            f"'wilcoxon'; got {test!r}"
        )
    if test == "wilcoxon":
        if pairs is None:
            raise ValueError(
                "wilcoxon requires sample aggregation with samples and pairs"
            )
        assert aggregated is not None
        table = _wilcoxon_signed_rank(aggregated, present, comparisons)
    elif test == "mann_whitney":
        table = _mann_whitney(values, groups, present, comparisons)
    else:
        if posthoc == "dunn":
            table = _dunn_posthoc(values, groups, present, comparisons)
        else:
            table = _kruskal_wallis(values, groups, present)

    if adjustment != "none" and len(table) > 1:
        table = table.copy()
        table["p_value_adjusted"] = adjust_pvalues(
            table["p_value"].to_numpy(dtype=np.float64, copy=False),
            adjustment,
        )
    return table
