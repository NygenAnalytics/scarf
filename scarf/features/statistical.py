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
- Explicitly requested parametric alternatives operate on raw cell-level
  values: Welch's t-test (``test="welch"``, aliased by ``"t_test"``) with a
  configurable ``alternative``, and one-way ANOVA (``test="one_way_anova"``).
  Both are descriptive distribution testing on single cells, which violates
  normality assumptions, so keep them beside the rank-based defaults.

Multiple-testing correction is delegated to ``statsmodels.stats.multitest``
(the same backend the marker statistics use).

Non-implemented parametric aliases still raise ``NotImplementedError``.
Cell-level testing (no ``samples``) is descriptive
distribution testing and emits a ``UserWarning``; sample-level aggregation
(``sample_stat`` over ``samples``) is "sample-level distribution summary
testing" and is not a replacement for replicate-aware differential
expression (for example DESeq2 or edgeR).
"""

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, kruskal, norm, ttest_ind, wilcoxon

from ..metadata.selection import CellField, valid_category_mask as _valid_group_mask
from ..storage.refs import ArtifactRef
from .markers.rank import mannwhitneyu_from_ranks

TestMethod = Literal[
    "auto",
    "mann_whitney",
    "kruskal_wallis",
    "wilcoxon",
    "welch",
    "t_test",
    "one_way_anova",
]
PosthocMethod = Literal["dunn"]
AdjustmentMethod = Literal["fdr_bh", "bonferroni", "holm", "none"]
SampleStatistic = Literal["mean", "median", "fraction"]
SummaryScope = Literal["cell", "sample"]
Alternative = Literal["two-sided", "less", "greater"]

_ALTERNATIVES = ("two-sided", "less", "greater")
_CELL_LEVEL_PARAMETRIC_TESTS = frozenset({"welch", "t_test", "one_way_anova"})
# Aliases that stay rejected until their dedicated implementations exist.
_PARAMETRIC_TESTS = frozenset(
    {
        "anova",
        "welch_t_test",
        "paired_t_test",
        "student_t_test",
        "f_test",
    }
)
_CELL_LEVEL_WARNING = (
    "Cell-level statistical testing treats each cell as an independent "
    "observation. Results are descriptive distribution testing, not "
    "population-level condition or disease inference; this applies equally "
    "to explicitly requested parametric tests on single cells (welch, "
    "one_way_anova), which must survive zero-inflated, non-normal data. "
    "Aggregate cells to biological samples with sample_by for sample-level "
    "summary testing."
)

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
WELCH_COLUMNS = (
    "group_1",
    "group_2",
    "n_1",
    "n_2",
    "t_statistic",
    "df",
    "mean_1",
    "mean_2",
    "mean_difference",
    "p_value",
)
ANOVA_COLUMNS = ("f_statistic", "df_between", "df_within", "p_value")

_METHOD_COLUMNS: dict[str, tuple[str, ...]] = {
    "mann_whitney": MANN_WHITNEY_COLUMNS,
    "kruskal_wallis": KRUSKAL_WALLIS_COLUMNS,
    "wilcoxon": WILCOXON_COLUMNS,
    "welch": WELCH_COLUMNS,
    "t_test": WELCH_COLUMNS,
    "one_way_anova": ANOVA_COLUMNS,
}

__all__ = [
    "ANOVA_COLUMNS",
    "DUNN_COLUMNS",
    "GroupComparisonResult",
    "KRUSKAL_WALLIS_COLUMNS",
    "MANN_WHITNEY_COLUMNS",
    "StatisticalTestResult",
    "WELCH_COLUMNS",
    "WILCOXON_COLUMNS",
    "adjust_pvalues",
    "aggregate_samples",
    "compare_group_distributions",
    "resolve_group_order",
]


@dataclass(frozen=True, slots=True, eq=False)
class GroupComparisonResult:
    """One key's statistical comparison outcome.

    ``table`` is the primary test table. When ``posthoc="dunn"`` was
    requested, ``table`` holds the omnibus Kruskal-Wallis result and
    ``posthoc_table`` holds the pairwise Dunn's table; otherwise
    ``posthoc_table`` is ``None``.
    """

    table: pd.DataFrame
    posthoc_table: pd.DataFrame | None = None


@dataclass(frozen=True, slots=True, eq=False)
class StatisticalTestResult:
    """Statistical tests for one grouping across multiple value keys."""

    method: str
    posthoc: str | None
    adjustment_method: str
    grouping: ArtifactRef | None
    group_field: CellField | None
    sample_by: str | None = None
    pair_by: str | None = None
    sample_stat: str = "mean"
    expression_cutoff: float = 0.0
    alternative: str = "two-sided"
    equal_var: bool | None = None
    n_groups: int = 0
    n_cells: int = 0
    tested_features: tuple[str, ...] = ()
    summary_scope: SummaryScope = "cell"
    artifact: ArtifactRef | None = None
    cell_selection: ArtifactRef | None = None
    cell_selection_fingerprint: str | None = None
    group_fingerprint: str | None = None
    group_order: tuple[Any, ...] = ()
    normalization: dict[str, str] = field(default_factory=dict)
    normalization_method: dict[str, str] | None = None
    size_factor: float | None = None
    source_assays: tuple[str | None, ...] = ()
    source_dataset_fingerprint: str | None = None
    value_fingerprints: tuple[str, ...] = ()
    sample_fingerprint: str | None = None
    pair_fingerprint: str | None = None
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    posthoc_tables: dict[str, pd.DataFrame] = field(default_factory=dict)


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


def _native(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def resolve_group_order(
    groups: np.ndarray,
    *,
    group_order: Sequence[Any] | None = None,
    full_groups: np.ndarray | None = None,
) -> list[Any]:
    """Return the ordered surviving group labels.

    When ``group_order`` is provided its order is preserved exactly, which
    fixes the contrast direction and row order of the reported tables. When
    omitted, groups appear in first-seen order; labels are never
    natural-sorted. Missing or empty labels are dropped.

    Requested groups are validated strictly:

    - Duplicate labels in ``group_order`` raise ``ValueError``.
    - A requested group that does not exist in the data at all raises
      ``ValueError`` (matching the cell-selection helper). Pass
      ``full_groups`` to distinguish "absent from the dataset" from "present
      but removed by subset or sample filters".
    - A requested group that exists but had every cell removed by filtering
      (``full_groups`` supplied) emits a ``UserWarning`` and is excluded from
      the returned order; the contrast design is otherwise left untouched.

    Plot alignment: this first-seen convention differs from
    :func:`scarf.plotting.distribution`, which sorts categories with
    ``sort_categories`` when no explicit order is given. Pass the same
    ``groups`` selection to both APIs to keep violin panel order and test
    contrasts aligned (the first selected group becomes ``group_1``); when no
    explicit order is used, annotation helpers must read the displayed
    category order from the plot instead of assuming it matches this order.
    """
    values = np.asarray(groups, dtype=object)
    valid = _valid_group_mask(values)
    surviving = [_native(value) for value in pd.unique(values[valid])]
    if group_order is None:
        return surviving
    ordered = [_native(value) for value in group_order]
    if len(set(ordered)) != len(ordered):
        raise ValueError("group_order must not contain duplicate labels")
    surviving_set = set(surviving)
    if full_groups is not None:
        full = np.asarray(full_groups, dtype=object)
        full_set = {_native(value) for value in pd.unique(full)}
        missing = [value for value in ordered if value not in full_set]
        if missing:
            raise ValueError(
                "groups contains labels not present in the data: "
                + ", ".join(map(str, missing[:10]))
            )
        dropped = [value for value in ordered if value not in surviving_set]
        for value in dropped:
            warnings.warn(
                f"Requested group {value!r} was removed because all of its "
                "cells were excluded by subset or sample filters; it is not "
                "part of the comparison design.",
                UserWarning,
                stacklevel=3,
            )
    else:
        missing = [value for value in ordered if value not in surviving_set]
        if missing:
            raise ValueError(
                "groups contains labels not present in the data: "
                + ", ".join(map(str, missing[:10]))
            )
    return [value for value in ordered if value in surviving_set]


def _maybe_adjust(
    table: pd.DataFrame,
    adjustment: AdjustmentMethod,
) -> pd.DataFrame:
    """Add a within-table ``p_value_adjusted`` column when it has multiple rows."""
    if adjustment == "none" or len(table) <= 1:
        return table
    frame = table.copy()
    frame["p_value_adjusted"] = adjust_pvalues(
        frame["p_value"].to_numpy(dtype=np.float64, copy=False),
        adjustment,
    )
    return frame


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

    This is sample-level distribution summary testing: each sample becomes
    one observation of its group. It is not a raw-count pseudobulk
    aggregation and must not be confused with replicate-aware differential
    expression tools such as DESeq2 or edgeR.

    Returns a frame with one row per ``(sample, group)`` (and ``pair`` when
    supplied) holding the aggregated ``value``. ``sample_stat`` is ``"mean"``,
    ``"median"``, or ``"fraction"`` (fraction of cells above
    ``expression_cutoff``). Each retained sample must belong to exactly one
    group and, when paired, map to exactly one valid pair key. Missing pair
    keys are rejected rather than treated as a shared pair.
    """
    value_array = np.asarray(values, dtype=np.float64)
    group_array = np.asarray(groups, dtype=object)
    sample_array = np.asarray(samples, dtype=object)
    if value_array.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if group_array.shape != value_array.shape:
        raise ValueError("groups length must match values")
    if sample_array.shape != value_array.shape:
        raise ValueError("samples length must match values")
    pair_array: np.ndarray | None = None
    if pairs is not None:
        pair_array = np.asarray(pairs, dtype=object)
        if pair_array.shape != value_array.shape:
            raise ValueError("pairs length must match values")

    columns: dict[str, np.ndarray] = {
        "value": value_array,
        "group": group_array,
        "sample": sample_array,
    }
    if pair_array is not None:
        columns["pair"] = pair_array
    frame = pd.DataFrame(columns)
    valid = _valid_group_mask(frame["sample"].to_numpy(dtype=object, copy=False))
    frame = frame.loc[valid].copy()
    if frame.empty:
        raise ValueError("No selected cells have a valid sample value")
    sample_group_counts = frame.groupby(
        "sample",
        observed=False,
        sort=False,
    )["group"].nunique(dropna=False)
    if (sample_group_counts != 1).any():
        samples_in_multiple_groups = sample_group_counts[
            sample_group_counts != 1
        ].index[:5]
        examples = ", ".join(repr(sample) for sample in samples_in_multiple_groups)
        raise ValueError(
            "Each sample must belong to exactly one group; samples observed in "
            f"multiple groups include {examples}. Use distinct sample ids per "
            "condition and pair_by to identify repeated subjects."
        )
    if pair_array is not None:
        valid_pairs = _valid_group_mask(
            frame["pair"].to_numpy(dtype=object, copy=False)
        )
        if not np.all(valid_pairs):
            raise ValueError(
                "pairs must contain a valid pair value for every cell with a "
                "valid sample"
            )
        pair_counts = frame.groupby(
            "sample",
            observed=False,
            sort=False,
        )["pair"].nunique(dropna=False)
        if (pair_counts != 1).any():
            raise ValueError("Each sample must map to exactly one pair key")
    grouped = frame.groupby(
        ["sample", "group"],
        observed=False,
        sort=False,
    )["value"]
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
    if pair_array is not None:
        pair_for_sample = frame.groupby(
            "sample",
            observed=False,
            sort=False,
        )["pair"].first()
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


def _welch_ttest(
    values: np.ndarray,
    groups: np.ndarray,
    present: list[Any],
    comparisons: Sequence[tuple[Any, Any]] | None,
    *,
    alternative: Alternative = "two-sided",
) -> pd.DataFrame:
    """Welch's t-test on raw cell-level values for exactly two groups."""
    if len(present) != 2:
        raise ValueError(
            "welch requires exactly two groups; use groups= to select two "
            "groups or one_way_anova for three or more"
        )
    g1, g2 = present
    if comparisons is not None:
        for left, right in comparisons:
            if {_native(left), _native(right)} != {g1, g2}:
                raise ValueError(
                    "welch comparisons must reference the two selected "
                    f"groups ({g1!r}, {g2!r})"
                )
    m1 = values[groups == g1]
    m2 = values[groups == g2]
    n1 = len(m1)
    n2 = len(m2)
    if n1 < 2 or n2 < 2:
        raise ValueError("welch requires at least two cells in every group")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = ttest_ind(
            m1,
            m2,
            equal_var=False,
            alternative=alternative,
            nan_policy="omit",
        )
    statistic = float(result.statistic)
    p_value = float(result.pvalue)
    df_attr = getattr(result, "df", None)
    if df_attr is not None and np.isfinite(df_attr):
        df_stat = float(df_attr)
    else:
        var_1 = float(np.var(m1, ddof=1))
        var_2 = float(np.var(m2, ddof=1))
        numerator = (var_1 / n1 + var_2 / n2) ** 2
        denominator = (var_1 / n1) ** 2 / (n1 - 1) + (var_2 / n2) ** 2 / (n2 - 1)
        df_stat = (
            float(numerator / denominator) if denominator > 0 else float(n1 + n2 - 2)
        )
    all_tied = bool(np.all(m1 == m1[0]) and np.all(m2 == m1[0]))
    if np.isnan(statistic) or np.isnan(p_value):
        if not all_tied:
            raise ValueError("welch returned an undefined statistic for these values")
        statistic = 0.0
        p_value = 1.0
    if not np.isfinite(df_stat):
        df_stat = float(n1 + n2 - 2)
    mean_1 = float(np.mean(m1))
    mean_2 = float(np.mean(m2))
    return pd.DataFrame(
        [
            {
                "group_1": g1,
                "group_2": g2,
                "n_1": n1,
                "n_2": n2,
                "t_statistic": statistic,
                "df": df_stat,
                "mean_1": mean_1,
                "mean_2": mean_2,
                "mean_difference": mean_1 - mean_2,
                "p_value": p_value,
            }
        ],
        columns=list(WELCH_COLUMNS),
    )


def _one_way_anova(
    values: np.ndarray,
    groups: np.ndarray,
    present: list[Any],
) -> pd.DataFrame:
    """Classic one-way ANOVA (equal-variance F-test) on raw cell values.

    Homoscedasticity is assumed; robust alternatives are deferred. With two
    groups this reduces to the square of a Student's t-test under equal
    variance, so prefer ``test="welch"`` for uneven group spreads.
    """
    if len(present) < 2:
        raise ValueError(
            "one_way_anova requires at least two groups; use groups= to "
            "select them explicitly"
        )
    group_values = [values[groups == g] for g in present]
    if any(len(v) < 2 for v in group_values):
        raise ValueError("one_way_anova requires at least two cells in every group")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        statistic, p_value = f_oneway(*group_values)
    all_tied = bool(np.all(values == values[0]))
    if np.isnan(statistic) or np.isnan(p_value):
        if not all_tied:
            raise ValueError(
                "one_way_anova returned an undefined statistic for these values"
            )
        statistic = 0.0
        p_value = 1.0
    return pd.DataFrame(
        [
            {
                "f_statistic": float(statistic),
                "df_between": len(present) - 1,
                "df_within": len(values) - len(present),
                "p_value": float(p_value),
            }
        ],
        columns=list(ANOVA_COLUMNS),
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
        try:
            statistic, p_value = kruskal(*group_values)
        except ValueError as exc:
            # scipy <1.15 raises for the degenerate all-tied case where
            # newer releases return NaN; both mean "no evidence of
            # differences", so normalize to statistic 0 and p-value 1.
            if "identical" not in str(exc):
                raise
            statistic, p_value = 0.0, 1.0
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
    pair_group_counts = aggregated.groupby(
        ["pair", "group"],
        observed=False,
    ).size()
    duplicates = pair_group_counts[pair_group_counts > 1]
    if not duplicates.empty:
        pairs = ", ".join(f"{pair!r}/{group!r}" for pair, group in duplicates.index[:5])
        raise ValueError(
            "Duplicate (pair, group) rows found for the Wilcoxon test "
            f"(e.g. {pairs}); each subject or pair must have exactly one "
            "aggregated value per group. Collapse technical replicates "
            "(for example by summing or averaging them) before running the "
            "test."
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
    group_order: Sequence[Any] | None = None,
    alternative: Alternative = "two-sided",
) -> GroupComparisonResult:
    """Compare one value array across groups with the configured test.

    ``values`` and ``groups`` must be one-dimensional and equal in length.
    When ``samples`` is provided, cells are aggregated to biological samples
    first using ``sample_stat``. ``pairs`` (a subject or donor id per cell)
    enables the paired Wilcoxon signed-rank test, which is only meaningful on
    aggregated sample data and therefore requires ``samples`` too. Pair keys
    are rejected for explicitly selected independent tests rather than ignored.

    ``test`` is ``"auto"`` (pick by design: paired -> Wilcoxon, two groups ->
    Mann-Whitney, three or more -> Kruskal-Wallis), one of the non-parametric
    methods, or an explicit cell-level parametric method. ``"auto"`` never
    selects a parametric test; request ``"welch"`` (or the ``"t_test"``
    alias) or ``"one_way_anova"`` to opt into them. The parametric options run
    on raw unaggregated values only; passing ``samples`` or ``pairs`` with
    them raises ``ValueError``.

    ``alternative`` sets the alternative hypothesis direction for Welch's
    t-test; other tests require ``"two-sided"``. ``posthoc="dunn"`` adds
    pairwise Dunn's tests after Kruskal-Wallis and preserves the omnibus result
    in the returned :class:`GroupComparisonResult`. ``comparisons`` restricts
    the pairwise rows to the listed group pairs. ``group_order`` selects and
    orders the groups (and therefore fixes the contrast direction) exactly;
    when omitted, first-seen order is used, including after sample aggregation.
    ``adjustment`` corrects p-values within the returned tables when they hold
    multiple comparisons; pass ``"none"`` to adjust across keys in the caller
    instead.

    Returns:
        A :class:`GroupComparisonResult` whose ``table`` is the primary test
        table (one row per comparison for Mann-Whitney and Wilcoxon, a single
        row for Kruskal-Wallis, Welch, and ANOVA) and whose ``posthoc_table``
        holds the pairwise Dunn's table when ``posthoc="dunn"``.
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
    if adjustment not in ("fdr_bh", "bonferroni", "holm", "none"):
        raise ValueError("adjustment must be 'fdr_bh', 'bonferroni', 'holm', or 'none'")
    if alternative not in _ALTERNATIVES:
        raise ValueError(
            f"alternative must be 'two-sided', 'less', or 'greater'; got {alternative!r}"
        )
    if comparisons is not None:
        comparisons = tuple(
            (_native(left), _native(right)) for left, right in comparisons
        )
        if len(comparisons) == 0:
            raise ValueError("comparisons must be non-empty when provided")
    if samples is not None:
        samples = np.asarray(samples, dtype=object)
        if samples.shape != values.shape:
            raise ValueError("samples length must match values")
    elif sample_stat != "mean" or expression_cutoff != 0.0:
        raise ValueError(
            "sample_stat and expression_cutoff require samples; cell-level "
            "tests do not use aggregation parameters"
        )
    if samples is not None and sample_stat != "fraction" and expression_cutoff != 0.0:
        raise ValueError("expression_cutoff is only used with sample_stat='fraction'")
    if pairs is not None:
        pairs = np.asarray(pairs, dtype=object)
        if pairs.shape != values.shape:
            raise ValueError("pairs length must match values")
    if pairs is not None and test not in ("auto", "wilcoxon"):
        raise ValueError(
            "pairs is only supported with test='auto' or test='wilcoxon'; "
            "independent tests do not model pairing"
        )
    if test in _CELL_LEVEL_PARAMETRIC_TESTS and (
        samples is not None or pairs is not None
    ):
        raise ValueError(
            "welch and one_way_anova run on cell-level values; "
            "sample_by/pair_by aggregation is not supported for them"
        )
    if test in _PARAMETRIC_TESTS:
        raise NotImplementedError(
            "Scarf implements mann_whitney, kruskal_wallis, wilcoxon plus "
            "the explicit cell-level parametric welch/t_test and "
            f"one_way_anova; {test!r} is a non-parametric-phase alias with "
            "no implementation."
        )
    if alternative != "two-sided" and test not in ("welch", "t_test"):
        raise ValueError(
            "alternative is only supported for test='welch' or test='t_test'; "
            "other tests are two-sided"
        )
    if group_order is not None:
        ordered_check = [_native(value) for value in group_order]
        if len(set(ordered_check)) != len(ordered_check):
            raise ValueError("group_order must not contain duplicate labels")
    if comparisons is not None:
        seen_pairs: set[tuple[Any, Any]] = set()
        for left, right in comparisons:
            pair = (_native(left), _native(right))
            if pair[0] == pair[1]:
                raise ValueError("comparisons must reference two distinct groups")
            if pair in seen_pairs or (pair[1], pair[0]) in seen_pairs:
                raise ValueError(
                    "comparisons must not contain duplicate or reversed-duplicate pairs"
                )
            seen_pairs.add(pair)

    if samples is None:
        warnings.warn(_CELL_LEVEL_WARNING, UserWarning, stacklevel=2)

    valid = _valid_group_mask(groups)
    values = values[valid]
    groups = groups[valid]
    if samples is not None:
        samples = samples[valid]
    if pairs is not None:
        pairs = pairs[valid]
    if len(values) == 0:
        raise ValueError("No values remain after dropping missing groups")

    pre_aggregation_order = resolve_group_order(groups, group_order=group_order)
    if group_order is not None:
        selected = (
            pd.Series(groups, dtype=object).isin(pre_aggregation_order).to_numpy()
        )
        values = values[selected]
        groups = groups[selected]
        if samples is not None:
            samples = samples[selected]
        if pairs is not None:
            pairs = pairs[selected]

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

    if group_order is None:
        surviving = set(resolve_group_order(groups))
        present = [group for group in pre_aggregation_order if group in surviving]
    else:
        present = resolve_group_order(groups, group_order=pre_aggregation_order)
    if comparisons is not None:
        present_set = set(present)
        for left, right in comparisons:
            if _native(left) not in present_set or _native(right) not in present_set:
                raise ValueError(
                    "comparisons references a group not present in "
                    f"group_order: {left!r} or {right!r}"
                )
    if len(present) < 2:
        raise ValueError("At least two populated groups are required")

    if test == "auto":
        if pairs is not None:
            test = "wilcoxon"
        elif len(present) == 2:
            test = "mann_whitney"
        else:
            test = "kruskal_wallis"
    implemented = (
        "mann_whitney",
        "kruskal_wallis",
        "wilcoxon",
        *_CELL_LEVEL_PARAMETRIC_TESTS,
    )
    if test not in implemented:
        raise ValueError(
            "test must be one of "
            + ", ".join(f"'{name}'" for name in ("auto", *implemented))
            + f"; got {test!r}"
        )
    if posthoc == "dunn" and test != "kruskal_wallis":
        raise ValueError("posthoc='dunn' requires test='kruskal_wallis'")
    if comparisons is not None and (
        test == "one_way_anova" or (test == "kruskal_wallis" and posthoc is None)
    ):
        raise ValueError(
            "comparisons is only supported by pairwise tests or "
            "kruskal_wallis with posthoc='dunn'"
        )

    if test == "wilcoxon":
        if pairs is None:
            raise ValueError(
                "wilcoxon requires sample aggregation with samples and pairs"
            )
        assert aggregated is not None
        table = _wilcoxon_signed_rank(aggregated, present, comparisons)
        return GroupComparisonResult(
            _maybe_adjust(table, adjustment),
        )
    if test in ("welch", "t_test"):
        table = _welch_ttest(
            values,
            groups,
            present,
            comparisons,
            alternative=alternative,
        )
        return GroupComparisonResult(
            _maybe_adjust(table, adjustment),
        )
    if test == "one_way_anova":
        table = _one_way_anova(values, groups, present)
        return GroupComparisonResult(
            _maybe_adjust(table, adjustment),
        )
    if test == "mann_whitney":
        table = _mann_whitney(values, groups, present, comparisons)
        return GroupComparisonResult(
            _maybe_adjust(table, adjustment),
        )
    if posthoc == "dunn":
        omnibus = _kruskal_wallis(values, groups, present)
        posthoc_table = _dunn_posthoc(values, groups, present, comparisons)
        return GroupComparisonResult(
            _maybe_adjust(omnibus, adjustment),
            _maybe_adjust(posthoc_table, adjustment),
        )
    table = _kruskal_wallis(values, groups, present)
    return GroupComparisonResult(
        _maybe_adjust(table, adjustment),
    )
