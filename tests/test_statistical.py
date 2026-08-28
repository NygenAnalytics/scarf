import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import f_oneway, kruskal
from scipy.stats import mannwhitneyu as scipy_mannwhitneyu
from scipy.stats import norm
from scipy.stats import rankdata
from scipy.stats import ttest_ind
from scipy.stats import wilcoxon as scipy_wilcoxon
from statsmodels.stats.multitest import multipletests

from scarf.features.markers import mannwhitneyu_from_ranks
from scarf.features.statistical import (
    GroupComparisonResult,
    StatisticalTestResult,
    adjust_pvalues,
    aggregate_samples,
    compare_group_distributions,
    resolve_group_order,
)
from scarf.metadata.selection import CellField

pytestmark = pytest.mark.filterwarnings("ignore:Cell-level statistical testing")


def _dunn_reference(values, groups):
    """Independent Dunn's test reference using scipy rankdata and norm."""
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups, dtype=object)
    present = sorted(pd.unique(groups), key=str)
    ranks = rankdata(values, method="average")
    n_total = len(values)
    _, counts = np.unique(values, return_counts=True)
    tied = counts[counts > 1]
    tie_correction = float(np.sum(tied**3 - tied)) if tied.size else 0.0
    variance = (n_total * (n_total + 1)) / 12 - tie_correction / (12 * (n_total - 1))
    rows = []
    for idx, g1 in enumerate(present):
        for g2 in present[idx + 1 :]:
            rank_1 = float(np.mean(ranks[groups == g1]))
            rank_2 = float(np.mean(ranks[groups == g2]))
            n_1 = int((groups == g1).sum())
            n_2 = int((groups == g2).sum())
            standard_error = np.sqrt(variance * (1 / n_1 + 1 / n_2))
            z = (rank_1 - rank_2) / standard_error if standard_error > 0 else 0.0
            rows.append((g1, g2, float(z), float(2 * norm.sf(abs(z)))))
    return pd.DataFrame(rows, columns=["group_1", "group_2", "z", "p_value"])


def _seeded_groups(rng, n, n_groups):
    return np.array([f"g{i % n_groups}" for i in range(n)], dtype=object)


def test_mann_whitney_matches_scipy_and_marker_ranks():
    rng = np.random.default_rng(11)
    values = np.concatenate([rng.poisson(2, 60), rng.poisson(4, 60), np.zeros(20)])
    groups = _seeded_groups(rng, 140, 2)
    result = compare_group_distributions(values, groups, test="mann_whitney")
    table = result.table
    assert isinstance(result, GroupComparisonResult)
    assert result.posthoc_table is None
    g1, g2 = table.loc[0, "group_1"], table.loc[0, "group_2"]
    left = values[groups == g1]
    right = values[groups == g2]
    u_scipy, p_scipy = scipy_mannwhitneyu(
        left,
        right,
        method="asymptotic",
        alternative="two-sided",
    )
    assert table.loc[0, "n_1"] == len(left)
    assert table.loc[0, "n_2"] == len(right)
    assert np.isclose(table.loc[0, "u_statistic"], float(u_scipy))
    assert np.isclose(table.loc[0, "p_value"], float(p_scipy))
    assert np.isclose(table.loc[0, "mean_1"], float(np.mean(left)))
    assert np.isclose(
        table.loc[0, "mean_difference"],
        float(np.mean(left) - np.mean(right)),
    )
    ranked = pd.DataFrame({"feature": pd.Series(values).rank(method="average")})
    reference = mannwhitneyu_from_ranks(
        ranked,
        groups,
        np.array([g1, g2], dtype=object),
    )
    assert np.isclose(table.loc[0, "p_value"], float(reference.loc[g1, "feature"]))


def test_mann_whitney_requires_two_groups():
    rng = np.random.default_rng(1)
    values = rng.normal(size=90)
    groups = _seeded_groups(rng, 90, 3)
    with pytest.raises(ValueError, match="exactly two groups"):
        compare_group_distributions(values, groups, test="mann_whitney")
    two_groups = np.array(["g0", "g1"], dtype=object)
    with pytest.raises(ValueError, match="at least two cells"):
        compare_group_distributions(np.arange(2, dtype=float), two_groups)


def test_kruskal_matches_scipy():
    rng = np.random.default_rng(2)
    values = np.concatenate(
        [rng.normal(0, 1, 40), rng.normal(1, 1, 40), rng.normal(2, 1, 40)]
    )
    groups = _seeded_groups(rng, 120, 3)
    table = compare_group_distributions(values, groups, test="kruskal_wallis").table
    expected_stat, expected_p = kruskal(
        values[groups == "g0"],
        values[groups == "g1"],
        values[groups == "g2"],
    )
    assert np.isclose(table.loc[0, "kruskal_statistic"], float(expected_stat))
    assert np.isclose(table.loc[0, "p_value"], float(expected_p))
    assert table.loc[0, "df"] == 2


def test_kruskal_handles_all_tied_values():
    values = np.zeros(60)
    groups = _seeded_groups(np.random.default_rng(0), 60, 3)
    table = compare_group_distributions(values, groups, test="kruskal_wallis").table
    assert table.loc[0, "kruskal_statistic"] == 0.0
    assert table.loc[0, "p_value"] == 1.0


def test_kruskal_requires_three_groups():
    rng = np.random.default_rng(3)
    values = rng.normal(size=40)
    groups = _seeded_groups(rng, 40, 2)
    with pytest.raises(ValueError, match="at least three groups"):
        compare_group_distributions(values, groups, test="kruskal_wallis")


def test_dunn_posthoc_matches_reference():
    rng = np.random.default_rng(4)
    values = np.concatenate(
        [
            rng.normal(0, 1, 30),
            rng.normal(0.8, 1, 30),
            rng.normal(1.6, 1, 30),
            rng.poisson(1, 15),
        ]
    )
    groups = _seeded_groups(rng, 105, 4)
    result = compare_group_distributions(
        values,
        groups,
        test="kruskal_wallis",
        posthoc="dunn",
    )
    assert result.posthoc_table is not None
    table = result.posthoc_table
    reference = _dunn_reference(values, groups)
    assert len(table) == 6
    merged = table.merge(reference, on=["group_1", "group_2"], suffixes=("", "_ref"))
    assert np.allclose(merged["z"], merged["z_ref"])
    assert np.allclose(merged["p_value"], merged["p_value_ref"])


def test_dunn_preserves_omnibus_and_posthoc():
    rng = np.random.default_rng(16)
    values = np.concatenate(
        [rng.normal(0, 1, 40), rng.normal(1, 1, 40), rng.normal(2, 1, 40)]
    )
    groups = _seeded_groups(rng, 120, 3)
    result = compare_group_distributions(
        values,
        groups,
        test="kruskal_wallis",
        posthoc="dunn",
    )
    assert set(result.table.columns) == set(["kruskal_statistic", "df", "p_value"])
    assert result.posthoc_table is not None
    assert {"group_1", "group_2", "z", "p_value"} <= set(result.posthoc_table.columns)
    omnibus = compare_group_distributions(
        values,
        groups,
        test="kruskal_wallis",
    ).table
    assert np.isclose(
        result.table.loc[0, "kruskal_statistic"],
        omnibus.loc[0, "kruskal_statistic"],
    )
    assert np.isclose(result.table.loc[0, "p_value"], omnibus.loc[0, "p_value"])


def test_group_order_determines_contrast_direction():
    rng = np.random.default_rng(17)
    n = 80
    values = np.concatenate([rng.normal(0, 1, n // 2), rng.normal(1.0, 1, n // 2)])
    groups = np.array(["control"] * (n // 2) + ["treated"] * (n // 2), dtype=object)
    forward = compare_group_distributions(
        values,
        groups,
        group_order=["treated", "control"],
    ).table
    reversed_order = compare_group_distributions(
        values,
        groups,
        group_order=["control", "treated"],
    ).table
    assert forward.loc[0, "group_1"] == "treated"
    assert forward.loc[0, "group_2"] == "control"
    assert reversed_order.loc[0, "group_1"] == "control"
    assert reversed_order.loc[0, "group_2"] == "treated"
    assert np.isclose(
        forward.loc[0, "mean_difference"],
        -reversed_order.loc[0, "mean_difference"],
    )
    assert forward.loc[0, "p_value"] == reversed_order.loc[0, "p_value"]


def test_group_order_is_first_seen_when_not_provided():
    rng = np.random.default_rng(18)
    values = rng.normal(size=40)
    groups = np.array(["b", "a"] * 20, dtype=object)
    table = compare_group_distributions(values, groups, test="mann_whitney").table
    assert table.loc[0, "group_1"] == "b"
    assert table.loc[0, "group_2"] == "a"
    assert resolve_group_order(groups) == ["b", "a"]


def test_group_order_filters_omitted_groups_before_dunn():
    values = np.arange(1, 17, dtype=np.float64)
    groups = np.repeat(np.array(["a", "b", "c", "omitted"], dtype=object), 4)
    selected_order = ["a", "b", "c"]

    from_full = compare_group_distributions(
        values,
        groups,
        test="kruskal_wallis",
        posthoc="dunn",
        group_order=selected_order,
    )
    selected = groups != "omitted"
    from_filtered = compare_group_distributions(
        values[selected],
        groups[selected],
        test="kruskal_wallis",
        posthoc="dunn",
        group_order=selected_order,
    )

    pd.testing.assert_frame_equal(from_full.table, from_filtered.table)
    pd.testing.assert_frame_equal(from_full.posthoc_table, from_filtered.posthoc_table)


def test_dunn_comparisons_restricts_pairs():
    rng = np.random.default_rng(5)
    values = rng.normal(size=90)
    groups = _seeded_groups(rng, 90, 3)
    table = compare_group_distributions(
        values,
        groups,
        test="kruskal_wallis",
        posthoc="dunn",
        comparisons=[("g0", "g2")],
    ).posthoc_table
    assert list(table["group_1"]) == ["g0"]
    assert list(table["group_2"]) == ["g2"]
    reference = _dunn_reference(values, groups)
    expected = reference[
        (reference["group_1"] == "g0") & (reference["group_2"] == "g2")
    ]
    assert np.isclose(table.loc[0, "z"], expected["z"].iloc[0])


def test_dunn_accepts_numpy_comparison_sequence():
    rng = np.random.default_rng(51)
    values = rng.normal(size=90)
    groups = _seeded_groups(rng, 90, 3)

    table = compare_group_distributions(
        values,
        groups,
        test="kruskal_wallis",
        posthoc="dunn",
        comparisons=np.array([["g0", "g2"]], dtype=object),
    ).posthoc_table

    assert table is not None
    assert table[["group_1", "group_2"]].to_dict("records") == [
        {"group_1": "g0", "group_2": "g2"}
    ]


def test_dunn_comparisons_missing_group_raises():
    rng = np.random.default_rng(6)
    values = rng.normal(size=90)
    groups = _seeded_groups(rng, 90, 3)
    with pytest.raises(ValueError, match="not present in group_order"):
        compare_group_distributions(
            values,
            groups,
            test="kruskal_wallis",
            posthoc="dunn",
            comparisons=[("g0", "missing")],
        )


def test_dunn_requires_kruskal_wallis():
    rng = np.random.default_rng(20)
    values = rng.normal(size=40)
    groups = _seeded_groups(rng, 40, 2)
    with pytest.raises(ValueError, match="requires test"):
        compare_group_distributions(
            values,
            groups,
            test="mann_whitney",
            posthoc="dunn",
        )


def test_wilcoxon_matches_scipy_on_aggregated_pairs():
    rng = np.random.default_rng(7)
    n = 120
    groups = np.array([f"g{i % 2}" for i in range(n)], dtype=object)
    values = np.where(
        groups == "g1",
        rng.normal(1.2, 1, n),
        rng.normal(0, 1, n),
    )
    samples = np.array([f"s{i // 8}_{groups[i]}" for i in range(n)], dtype=object)
    pairs = np.array([f"d{i // 8}" for i in range(n)], dtype=object)
    table = compare_group_distributions(
        values,
        groups,
        test="wilcoxon",
        samples=samples,
        pairs=pairs,
    ).table
    aggregated = aggregate_samples(
        values,
        groups,
        samples,
        pairs=pairs,
    )
    left = aggregated[aggregated["group"] == "g0"]
    right = aggregated[aggregated["group"] == "g1"]
    merged = left.merge(right, on="pair")
    stat, p = scipy_wilcoxon(
        merged["value_x"].to_numpy(),
        merged["value_y"].to_numpy(),
    )
    assert table.loc[0, "n_pairs"] == len(merged)
    assert np.isclose(table.loc[0, "statistic"], float(stat))
    assert np.isclose(table.loc[0, "p_value"], float(p))


def test_wilcoxon_rejects_duplicate_pair_groups():
    rng = np.random.default_rng(19)
    n = 40
    values = rng.normal(size=n)
    groups = np.array([f"g{i % 2}" for i in range(n)], dtype=object)
    samples = np.array([f"s{i // 2}_{groups[i]}" for i in range(n)], dtype=object)
    pairs = np.array([f"d{i // 4}" for i in range(n)], dtype=object)
    with pytest.raises(ValueError, match="Duplicate \\(pair, group\\)"):
        compare_group_distributions(
            values,
            groups,
            test="wilcoxon",
            samples=samples,
            pairs=pairs,
        )


def test_wilcoxon_requires_samples_and_pairs():
    rng = np.random.default_rng(8)
    values = rng.normal(size=40)
    groups = _seeded_groups(rng, 40, 2)
    with pytest.raises(ValueError, match="requires sample aggregation"):
        compare_group_distributions(values, groups, test="wilcoxon")
    samples = np.array([f"s{i % 4}" for i in range(40)], dtype=object)
    with pytest.raises(ValueError, match="requires sample aggregation"):
        compare_group_distributions(
            values,
            groups,
            test="wilcoxon",
            samples=samples,
        )


def test_pairs_require_samples():
    rng = np.random.default_rng(9)
    values = rng.normal(size=40)
    groups = _seeded_groups(rng, 40, 2)
    pairs = np.array([f"d{i % 4}" for i in range(40)], dtype=object)
    with pytest.raises(ValueError, match="pairs requires samples"):
        compare_group_distributions(values, groups, samples=None, pairs=pairs)


def test_aggregate_samples_semantics():
    rng = np.random.default_rng(10)
    n = 32
    values = rng.poisson(3, n).astype(float)
    groups = np.array([f"g{i % 2}" for i in range(n)], dtype=object)
    samples = np.array([f"s{i // 4}_{groups[i]}" for i in range(n)], dtype=object)
    mean_frame = aggregate_samples(values, groups, samples, sample_stat="mean")
    median_frame = aggregate_samples(values, groups, samples, sample_stat="median")
    frac_frame = aggregate_samples(
        values,
        groups,
        samples,
        sample_stat="fraction",
        expression_cutoff=1.0,
    )
    assert len(mean_frame) == 16
    for _, row in mean_frame.iterrows():
        cells = values[(groups == row["group"]) & (samples == row["sample"])]
        assert np.isclose(row["value"], float(np.mean(cells)))
    for _, row in median_frame.iterrows():
        cells = values[(groups == row["group"]) & (samples == row["sample"])]
        assert np.isclose(row["value"], float(np.median(cells)))
    for _, row in frac_frame.iterrows():
        cells = values[(groups == row["group"]) & (samples == row["sample"])]
        assert np.isclose(row["value"], float(np.mean(cells > 1.0)))


def test_aggregate_samples_rejects_sample_in_multiple_groups():
    values = np.arange(6, dtype=np.float64)
    groups = np.array(["a", "a", "b", "b", "a", "b"], dtype=object)
    samples = np.array(["a0", "shared", "b0", "shared", "a1", "b1"])

    with pytest.raises(ValueError, match="exactly one group"):
        aggregate_samples(values, groups, samples)


def test_aggregate_samples_rejects_multiple_pairs_per_sample():
    rng = np.random.default_rng(12)
    n = 40
    values = rng.normal(size=n)
    groups = _seeded_groups(rng, n, 2)
    samples = np.array([f"s{i % 4}" for i in range(n)], dtype=object)
    pairs = np.array([f"d{i % 8}" for i in range(n)], dtype=object)
    with pytest.raises(ValueError, match="exactly one pair key"):
        aggregate_samples(values, groups, samples, pairs=pairs)


@pytest.mark.parametrize("missing_pair", [None, np.nan, "", "   "])
def test_aggregate_samples_rejects_missing_pair_values(missing_pair):
    values = np.arange(8, dtype=np.float64)
    groups = np.array(["a", "b"] * 4, dtype=object)
    samples = np.array([f"s{i // 2}_{groups[i]}" for i in range(8)], dtype=object)
    pairs = np.array(["p0", "p0", "p1", "p1", "p2", "p2", "p3", "p3"], dtype=object)
    pairs[2] = missing_pair

    with pytest.raises(ValueError, match="valid pair value for every cell"):
        aggregate_samples(values, groups, samples, pairs=pairs)


def test_aggregate_samples_aligns_pairs_after_dropping_missing_samples():
    values = np.arange(6, dtype=np.float64)
    groups = np.array(["a", "a", "b", "b", "a", "b"], dtype=object)
    samples = np.array(
        [None, "s1", "s2", "s2", "s3_a", "s3_b"],
        dtype=object,
    )
    pairs = np.array(["ignored", "p1", "p2", "p2", "p3", "p3"], dtype=object)

    aggregated = aggregate_samples(values, groups, samples, pairs=pairs)

    assert aggregated["sample"].tolist() == ["s1", "s2", "s3_a", "s3_b"]
    assert aggregated["pair"].tolist() == ["p1", "p2", "p3", "p3"]


def test_sample_aggregation_preserves_first_seen_group_order():
    values = np.array([10.0, 1.0, 11.0, 2.0, 12.0, 3.0, 13.0, 4.0])
    groups = np.array(["b", "a"] * 4, dtype=object)
    samples = np.array(
        ["s2_b", "s2_a", "s2_b", "s2_a", "s1_b", "s1_a", "s1_b", "s1_a"],
        dtype=object,
    )

    table = compare_group_distributions(
        values,
        groups,
        test="mann_whitney",
        samples=samples,
    ).table

    assert table.loc[0, "group_1"] == "b"
    assert table.loc[0, "group_2"] == "a"
    assert table.loc[0, "mean_difference"] == 9.0


def test_auto_selects_test_by_design():
    rng = np.random.default_rng(13)
    values2 = rng.normal(size=80)
    groups2 = _seeded_groups(rng, 80, 2)
    two_group_table = compare_group_distributions(values2, groups2).table
    assert two_group_table.columns[0] == "group_1"
    values3 = rng.normal(size=120)
    groups3 = _seeded_groups(rng, 120, 3)
    assert (
        compare_group_distributions(values3, groups3).table.columns[0]
        == "kruskal_statistic"
    )
    samples = np.array(
        [f"s{i // 4}_{groups2[i]}" for i in range(80)],
        dtype=object,
    )
    pairs = np.array([f"d{i // 4}" for i in range(80)], dtype=object)
    paired_table = compare_group_distributions(
        values2,
        groups2,
        samples=samples,
        pairs=pairs,
    ).table
    assert "n_pairs" in paired_table.columns


def test_adjust_pvalues_matches_statsmodels():
    rng = np.random.default_rng(14)
    p_values = rng.uniform(0, 1, 50)
    for method in ("fdr_bh", "bonferroni", "holm"):
        expected = multipletests(p_values, method=method)[1]
        assert np.allclose(adjust_pvalues(p_values, method), expected)
    assert np.allclose(adjust_pvalues(p_values, "none"), p_values)
    with_nan = p_values.copy()
    with_nan[3] = np.nan
    adjusted = adjust_pvalues(with_nan, "fdr_bh")
    assert np.isnan(adjusted[3])
    assert np.isfinite(adjusted).sum() == 49


def test_compare_rejects_bad_inputs():
    rng = np.random.default_rng(15)
    values = rng.normal(size=40)
    groups = _seeded_groups(rng, 40, 2)
    with pytest.raises(ValueError, match="must match values"):
        compare_group_distributions(values[:20], groups)
    with pytest.raises(ValueError, match="one-dimensional"):
        compare_group_distributions(values.reshape(2, 20), groups)
    with pytest.raises(ValueError, match="finite"):
        values_with_nan = values.copy()
        values_with_nan[0] = np.nan
        compare_group_distributions(values_with_nan, groups)
    with pytest.raises(NotImplementedError, match="non-parametric"):
        compare_group_distributions(values, groups, test="anova")
    with pytest.raises(ValueError, match="test must be"):
        compare_group_distributions(values, groups, test="bogus")
    with pytest.raises(ValueError, match="posthoc must be"):
        compare_group_distributions(values, groups, posthoc="bonferroni")
    with pytest.raises(ValueError, match="adjustment must be"):
        compare_group_distributions(values, groups, adjustment="bogus")
    single = values[:20]
    single_groups = np.full(20, "g0", dtype=object)
    with pytest.raises(ValueError, match="two populated groups"):
        compare_group_distributions(single, single_groups)
    dropped_groups = groups[:20].copy()
    dropped_groups[::2] = ""
    with pytest.raises(ValueError, match="two populated groups"):
        compare_group_distributions(values[:20], dropped_groups)


def test_compare_validates_auxiliary_arrays():
    rng = np.random.default_rng(21)
    values = rng.normal(size=40)
    groups = _seeded_groups(rng, 40, 2)
    with pytest.raises(ValueError, match="samples length must match values"):
        compare_group_distributions(
            values,
            groups,
            samples=np.array(["s"] * 30, dtype=object),
        )
    with pytest.raises(ValueError, match="pairs length must match values"):
        compare_group_distributions(
            values,
            groups,
            samples=np.array([f"s_{groups[i]}" for i in range(40)], dtype=object),
            pairs=np.array(["d"] * 39, dtype=object),
        )


def test_compare_rejects_unused_aggregation_parameters():
    values = np.arange(6, dtype=float)
    groups = np.array(["a", "a", "a", "b", "b", "b"], dtype=object)
    samples = np.array(["s0", "s0", "s1", "s2", "s2", "s3"], dtype=object)

    with pytest.raises(ValueError, match="require samples"):
        compare_group_distributions(values, groups, sample_stat="median")
    with pytest.raises(ValueError, match="require samples"):
        compare_group_distributions(values, groups, expression_cutoff=1.0)
    with pytest.raises(ValueError, match="only used with sample_stat='fraction'"):
        compare_group_distributions(
            values,
            groups,
            samples=samples,
            sample_stat="mean",
            expression_cutoff=1.0,
        )


@pytest.mark.parametrize(
    ("test", "posthoc"),
    [("one_way_anova", None), ("kruskal_wallis", None)],
)
def test_compare_rejects_comparisons_for_omnibus_only_tests(test, posthoc):
    values = np.arange(9, dtype=float)
    groups = np.repeat(np.array(["a", "b", "c"], dtype=object), 3)

    with pytest.warns(UserWarning, match="Cell-level"):
        with pytest.raises(ValueError, match="comparisons is only supported"):
            compare_group_distributions(
                values,
                groups,
                test=test,
                posthoc=posthoc,
                comparisons=[("a", "b")],
            )


def test_compare_rejects_pairing_for_independent_tests():
    values = np.arange(8, dtype=np.float64)
    groups = np.array(["a", "b"] * 4, dtype=object)
    samples = np.array([f"s{i // 2}_{groups[i]}" for i in range(8)], dtype=object)
    pairs = np.array([f"p{i // 2}" for i in range(8)], dtype=object)

    with pytest.raises(ValueError, match="independent tests do not model pairing"):
        compare_group_distributions(
            values,
            groups,
            test="mann_whitney",
            samples=samples,
            pairs=pairs,
        )


def test_compare_validates_group_order_and_comparisons():
    rng = np.random.default_rng(22)
    values = rng.normal(size=60)
    groups = _seeded_groups(rng, 60, 3)
    with pytest.raises(ValueError, match="duplicate labels"):
        compare_group_distributions(
            values,
            groups,
            group_order=["g0", "g0", "g1"],
        )
    with pytest.raises(ValueError, match="reversed-duplicate"):
        compare_group_distributions(
            values,
            groups,
            test="kruskal_wallis",
            posthoc="dunn",
            comparisons=[("g0", "g1"), ("g1", "g0")],
        )
    with pytest.raises(ValueError, match="reversed-duplicate"):
        compare_group_distributions(
            values,
            groups,
            test="kruskal_wallis",
            posthoc="dunn",
            comparisons=[("g0", "g1"), ("g0", "g1")],
        )
    with pytest.raises(ValueError, match="two distinct groups"):
        compare_group_distributions(
            values,
            groups,
            test="kruskal_wallis",
            posthoc="dunn",
            comparisons=[("g0", "g0")],
        )
    with pytest.raises(ValueError, match="not present in group_order"):
        compare_group_distributions(
            values,
            groups,
            group_order=["g0", "g1"],
            test="kruskal_wallis",
            posthoc="dunn",
            comparisons=[("g1", "g2")],
        )


def test_resolve_group_order_rejects_missing_requested_group():
    rng = np.random.default_rng(23)
    groups = _seeded_groups(rng, 40, 2)
    with pytest.raises(ValueError, match="not present in the data"):
        resolve_group_order(groups, group_order=["g0", "missing"])
    with pytest.raises(ValueError, match="duplicate labels"):
        resolve_group_order(groups, group_order=["g0", "g0", "g1"])


def test_resolve_group_order_warns_and_drops_filtered_groups():
    rng = np.random.default_rng(24)
    groups = _seeded_groups(rng, 40, 2)
    surviving = np.array(["g0"] * 20, dtype=object)
    with pytest.warns(UserWarning, match="removed because all of its cells"):
        present = resolve_group_order(
            surviving,
            group_order=["g1", "g0"],
            full_groups=groups,
        )
    assert present == ["g0"]


def test_cell_level_testing_warns_user():
    rng = np.random.default_rng(25)
    values = rng.normal(size=40)
    groups = _seeded_groups(rng, 40, 2)
    with pytest.warns(UserWarning, match="descriptive distribution testing"):
        compare_group_distributions(values, groups, test="mann_whitney")
    samples = np.array(
        [f"s{i // 4}_{groups[i]}" for i in range(40)],
        dtype=object,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        compare_group_distributions(
            values,
            groups,
            test="mann_whitney",
            samples=samples,
        )


def test_welch_matches_scipy_two_sided():
    rng = np.random.default_rng(30)
    values = np.concatenate([rng.normal(0, 1, 40), rng.normal(1.0, 1.4, 40)])
    groups = _seeded_groups(rng, 80, 2)
    table = compare_group_distributions(values, groups, test="welch").table
    g1 = table.loc[0, "group_1"]
    g2 = table.loc[0, "group_2"]
    a = values[groups == g1]
    b = values[groups == g2]
    expected_stat, expected_p = ttest_ind(
        a,
        b,
        equal_var=False,
        alternative="two-sided",
    )
    assert table.loc[0, "n_1"] == len(a)
    assert table.loc[0, "n_2"] == len(b)
    assert np.isclose(table.loc[0, "t_statistic"], float(expected_stat))
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    expected_df = (var_a / len(a) + var_b / len(b)) ** 2 / (
        (var_a / len(a)) ** 2 / (len(a) - 1) + (var_b / len(b)) ** 2 / (len(b) - 1)
    )
    assert np.isclose(table.loc[0, "df"], expected_df)
    assert np.isclose(table.loc[0, "p_value"], float(expected_p))
    assert table.columns.tolist() == [
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
    ]


def test_welch_alternative_less_and_greater():
    rng = np.random.default_rng(31)
    values = rng.normal(size=80) + 0.8
    groups = _seeded_groups(rng, 80, 2)
    less_table = compare_group_distributions(
        values,
        groups,
        test="welch",
        alternative="less",
    ).table
    greater_table = compare_group_distributions(
        values,
        groups,
        test="welch",
        alternative="greater",
    ).table
    two_sided_table = compare_group_distributions(
        values,
        groups,
        test="welch",
        alternative="two-sided",
    ).table
    g1 = less_table.loc[0, "group_1"]
    g2 = less_table.loc[0, "group_2"]
    expected_less_p = ttest_ind(
        values[groups == g1],
        values[groups == g2],
        equal_var=False,
        alternative="less",
    ).pvalue
    expected_greater_p = ttest_ind(
        values[groups == g1],
        values[groups == g2],
        equal_var=False,
        alternative="greater",
    ).pvalue
    assert np.isclose(less_table.loc[0, "p_value"], float(expected_less_p))
    assert np.isclose(greater_table.loc[0, "p_value"], float(expected_greater_p))
    assert np.isclose(
        greater_table.loc[0, "p_value"] + less_table.loc[0, "p_value"],
        1.0,
    )
    assert two_sided_table.loc[0, "p_value"] <= max(
        less_table.loc[0, "p_value"],
        greater_table.loc[0, "p_value"],
    )


def test_welch_preserves_constant_separation():
    values = np.concatenate([np.zeros(4), np.ones(4)])
    groups = np.array(["a"] * 4 + ["b"] * 4, dtype=object)

    table = compare_group_distributions(values, groups, test="welch").table

    assert np.isneginf(table.loc[0, "t_statistic"])
    assert table.loc[0, "p_value"] == 0.0
    assert table.loc[0, "mean_difference"] == -1.0


def test_welch_requires_two_groups_and_cells():
    rng = np.random.default_rng(32)
    values = rng.normal(size=90)
    groups = _seeded_groups(rng, 90, 3)
    with pytest.raises(ValueError, match="exactly two groups"):
        compare_group_distributions(values, groups, test="welch")
    two_cells = values[:2]
    two_groups = np.array(["g0", "g1"], dtype=object)
    with pytest.raises(ValueError, match="at least two cells"):
        compare_group_distributions(two_cells, two_groups)
    lopsided_values = np.arange(3, dtype=float)
    lopsided_groups = np.array(["g0", "g0", "g1"], dtype=object)
    with pytest.raises(ValueError, match="at least two cells"):
        compare_group_distributions(lopsided_values, lopsided_groups)


def test_one_way_anova_matches_scipy():
    rng = np.random.default_rng(33)
    values = np.concatenate(
        [rng.normal(0, 1, 40), rng.normal(0.7, 1, 40), rng.normal(1.4, 1, 40)]
    )
    groups = _seeded_groups(rng, 120, 3)
    table = compare_group_distributions(values, groups, test="one_way_anova").table
    expected_stat, expected_p = f_oneway(
        values[groups == "g0"],
        values[groups == "g1"],
        values[groups == "g2"],
    )
    assert np.isclose(table.loc[0, "f_statistic"], float(expected_stat))
    assert np.isclose(table.loc[0, "p_value"], float(expected_p))
    assert table.loc[0, "df_between"] == 2
    assert table.loc[0, "df_within"] == 117


def test_one_way_anova_handles_all_tied_values():
    values = np.zeros(60)
    groups = _seeded_groups(np.random.default_rng(34), 60, 3)
    table = compare_group_distributions(values, groups, test="one_way_anova").table
    assert table.loc[0, "f_statistic"] == 0.0
    assert table.loc[0, "p_value"] == 1.0
    assert table.loc[0, "df_between"] == 2
    assert table.loc[0, "df_within"] == 57


def test_one_way_anova_preserves_constant_separation():
    values = np.concatenate([np.zeros(4), np.ones(4), np.full(4, 2.0)])
    groups = np.repeat(np.array(["a", "b", "c"], dtype=object), 4)

    table = compare_group_distributions(values, groups, test="one_way_anova").table

    assert np.isposinf(table.loc[0, "f_statistic"])
    assert table.loc[0, "p_value"] == 0.0


def test_one_way_anova_group_order_filters_values_and_degrees_of_freedom():
    values = np.arange(1, 17, dtype=np.float64)
    groups = np.repeat(np.array(["a", "b", "c", "omitted"], dtype=object), 4)

    table = compare_group_distributions(
        values,
        groups,
        test="one_way_anova",
        group_order=["a", "b", "c"],
    ).table
    expected_stat, expected_p = f_oneway(
        values[groups == "a"],
        values[groups == "b"],
        values[groups == "c"],
    )

    assert table.loc[0, "f_statistic"] == pytest.approx(float(expected_stat))
    assert table.loc[0, "p_value"] == pytest.approx(float(expected_p))
    assert table.loc[0, "df_between"] == 2
    assert table.loc[0, "df_within"] == 9


def test_compare_rejects_bad_alternative_and_mismatched_posthoc():
    rng = np.random.default_rng(35)
    values = rng.normal(size=60)
    groups = _seeded_groups(rng, 60, 2)
    with pytest.raises(ValueError, match="alternative must be"):
        compare_group_distributions(values, groups, test="welch", alternative="up")
    with pytest.raises(ValueError, match="requires test"):
        compare_group_distributions(
            values,
            groups,
            test="welch",
            posthoc="dunn",
        )
    with pytest.raises(ValueError, match="not supported for them"):
        compare_group_distributions(
            values,
            groups,
            test="one_way_anova",
            samples=np.array([f"s{i % 6}" for i in range(60)], dtype=object),
        )
    with pytest.raises(NotImplementedError, match="non-parametric-phase"):
        compare_group_distributions(values, groups, test="student_t_test")


@pytest.mark.parametrize(
    "test",
    ["auto", "mann_whitney", "kruskal_wallis", "wilcoxon", "one_way_anova"],
)
def test_directional_alternative_rejected_for_unsupported_tests(test):
    values = np.arange(60, dtype=np.float64)
    groups = _seeded_groups(np.random.default_rng(37), 60, 3)

    with pytest.raises(ValueError, match="alternative is only supported"):
        compare_group_distributions(
            values,
            groups,
            test=test,
            alternative="greater",
        )


def test_group_order_controls_welch_direction():
    rng = np.random.default_rng(36)
    n = 80
    values = np.concatenate([rng.normal(0, 1, n // 2), rng.normal(1.2, 1, n // 2)])
    groups = np.array(["A"] * (n // 2) + ["B"] * (n // 2), dtype=object)
    forward = compare_group_distributions(
        values,
        groups,
        test="welch",
        group_order=["B", "A"],
    ).table
    reversed_order = compare_group_distributions(
        values,
        groups,
        test="welch",
        group_order=["A", "B"],
    ).table
    assert forward.loc[0, "group_1"] == "B"
    assert forward.loc[0, "group_2"] == "A"
    assert reversed_order.loc[0, "group_1"] == "A"
    assert reversed_order.loc[0, "group_2"] == "B"
    assert forward.loc[0, "t_statistic"] == pytest.approx(
        -reversed_order.loc[0, "t_statistic"]
    )
    assert forward.loc[0, "mean_difference"] > 0
    assert reversed_order.loc[0, "mean_difference"] < 0
    assert forward.loc[0, "p_value"] == reversed_order.loc[0, "p_value"]


def test_summary_scope_tracks_sample_aggregation():
    rng = np.random.default_rng(26)
    values = rng.normal(size=40)
    groups = _seeded_groups(rng, 40, 2)
    result = StatisticalTestResult(
        method="mann_whitney",
        posthoc=None,
        adjustment_method="fdr_bh",
        grouping=None,
        group_field=CellField("grp"),
        sample_by="sample",
        summary_scope="sample",
        tables={
            "k": compare_group_distributions(
                values,
                groups,
                test="mann_whitney",
            ).table
        },
    )
    assert result.summary_scope == "sample"
    assert (
        StatisticalTestResult(
            method="mann_whitney",
            posthoc=None,
            adjustment_method="fdr_bh",
            grouping=None,
            group_field=CellField("grp"),
        ).summary_scope
        == "cell"
    )


def test_statistical_test_result_is_frozen():
    result = StatisticalTestResult(
        method="mann_whitney",
        posthoc=None,
        adjustment_method="fdr_bh",
        grouping=None,
        group_field=CellField("grp"),
    )
    with pytest.raises(AttributeError):
        result.method = "kruskal_wallis"


def test_statistical_test_result_identity_defaults_are_optional():
    result = StatisticalTestResult(
        method="mann_whitney",
        posthoc=None,
        adjustment_method="fdr_bh",
        grouping=None,
        group_field=CellField("grp"),
    )

    assert result.artifact is None
    assert result.cell_selection is None
    assert result.cell_selection_fingerprint is None
    assert result.group_fingerprint is None
    assert result.group_order == ()
    assert result.normalization == {}
    assert result.source_assays == ()
    assert result.source_dataset_fingerprint is None
    assert result.value_fingerprints == ()
