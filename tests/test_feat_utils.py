import numpy as np
import pandas as pd
import pytest

from scarf.features.genomic.intervals import (
    binary_search,
    create_bed_from_coord_ids,
    get_feature_mappings,
)
from scarf.features.scoring import binned_sampling
from scarf.features.variability import fit_lowess, select_highly_variable_features


def test_fit_lowess_returns_per_feature_corrections():
    rng = np.random.default_rng(0)
    n_genes = 80
    mean_expr = rng.uniform(0.5, 20.0, n_genes)
    variance = mean_expr**1.5 + rng.normal(0, 0.05, n_genes)
    variance = np.clip(variance, 0.1, None)

    corrected = fit_lowess(mean_expr, variance, n_bins=8, lowess_frac=0.6)
    explicit_adaptive = fit_lowess(
        mean_expr,
        variance,
        n_bins=8,
        lowess_frac=0.6,
        bin_strategy="adaptive",
    )
    fixed = fit_lowess(
        mean_expr,
        variance,
        n_bins=8,
        lowess_frac=0.6,
        bin_strategy="fixed",
    )

    assert corrected.shape == (n_genes,)
    assert np.all(np.isfinite(corrected))
    assert np.all(corrected > 0)
    np.testing.assert_array_equal(corrected, explicit_adaptive)
    assert np.all(np.isfinite(fixed))
    assert np.all(fixed > 0)


def test_fit_lowess_fixed_regression():
    mean_expr = np.array([0.5, 0.8, 1.2, 1.8, 2.7, 4.0, 6.0, 9.0, 13.0, 20.0])
    variance = np.array([0.4, 0.9, 1.1, 3.2, 2.8, 8.5, 7.0, 25.0, 22.0, 70.0])
    expected = np.array(
        [1.0, 2.25, 2.75, 8 / 7, 1.0, 17 / 14, 1.0, 25 / 22, 1.0, 35 / 11]
    )

    corrected = fit_lowess(
        mean_expr,
        variance,
        n_bins=4,
        lowess_frac=0.75,
        bin_strategy="fixed",
    )

    np.testing.assert_allclose(corrected, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"n_bins": 0}, "n_bins must be at least 1"),
        ({"lowess_frac": 1.5}, "lowess_frac must be between 0 and 1"),
    ],
)
def test_fit_lowess_checks_the_trend_arguments_of_the_fixed_strategy(
    arguments, message
):
    means = np.geomspace(1, 100, 30)
    variances = means**1.4

    with pytest.raises(ValueError, match=f"^{message}$"):
        fit_lowess(
            means,
            variances,
            **{"n_bins": 8, "lowess_frac": 0.6, **arguments},
            bin_strategy="fixed",
        )


def test_fit_lowess_rejects_unconverged_adaptive_fit(monkeypatch):
    import scipy.optimize

    minimize = scipy.optimize.minimize

    def stop_after_one_iteration(*args, **kwargs):
        kwargs["options"] = dict(kwargs["options"], maxiter=1)
        return minimize(*args, **kwargs)

    monkeypatch.setattr(scipy.optimize, "minimize", stop_after_one_iteration)
    means = np.geomspace(1, 100, 30)
    variances = means**1.4 * np.exp(np.random.default_rng(7).normal(0, 0.2, 30))
    with pytest.raises(ValueError, match="Adaptive variance trend fit failed"):
        fit_lowess(means, variances, n_bins=10, lowess_frac=0.5)


def test_fit_lowess_rejects_unrepresentable_adaptive_scores():
    variance = np.array([np.nextafter(0.0, 1.0), 1.0, np.finfo(float).max])
    with pytest.raises(ValueError, match="nonfinite or zero scores"):
        fit_lowess(np.ones(3), variance, n_bins=10, lowess_frac=0.5)


@pytest.mark.parametrize("trend", ["power", "curved", "steep_tail"])
@pytest.mark.parametrize(
    ("n_genes", "seed"), [(2_000, 17), (5_000, 23), (20_000, 17), (20_000, 23)]
)
def test_fit_lowess_adaptive_removes_trend_through_sparse_tails(trend, n_genes, seed):
    mean_expr = np.exp(np.random.default_rng(seed).normal(0, 2, n_genes))
    if trend == "power":
        variance = mean_expr**1.4
    elif trend == "curved":
        variance = mean_expr + mean_expr**2
    else:
        variance = mean_expr**1.4 * np.sqrt(1 + (mean_expr / 100) ** 2)

    corrected = fit_lowess(mean_expr, variance, n_bins=200, lowess_frac=0.1)

    np.testing.assert_allclose(corrected, 1.0, rtol=0.05)


def test_fit_lowess_adaptive_calibrates_the_lower_quartile():
    means = np.repeat(np.geomspace(0.01, 100, 40), 9)
    offsets = np.tile(np.arange(-4, 5) / 10, 40)
    variance = means**1.4 * np.exp(offsets)

    corrected = fit_lowess(means, variance, n_bins=200, lowess_frac=0.1)

    np.testing.assert_allclose(corrected, np.exp(offsets + 0.2), rtol=0.005)


@pytest.mark.parametrize("multiplier", [4.0, 1e-12])
def test_fit_lowess_adaptive_protects_curved_tails_from_isolated_outliers(
    multiplier,
):
    means = np.sort(np.exp(np.random.default_rng(41).normal(0, 2, 5_000)))
    variance = means**1.4 * np.sqrt(1 + (means / 100) ** 2)
    baseline = fit_lowess(means, variance, n_bins=200, lowess_frac=0.1)
    variance[-1] *= multiplier

    corrected = fit_lowess(means, variance, n_bins=200, lowess_frac=0.1)

    np.testing.assert_allclose(corrected[-1] / baseline[-1], multiplier, rtol=0.05)
    np.testing.assert_allclose(corrected[:-1], baseline[:-1], rtol=0.05)


@pytest.mark.parametrize("seed", [9, 12, 14])
def test_fit_lowess_adaptive_bounds_error_with_sparse_expression_support(seed):
    means = np.exp(np.random.default_rng(seed).normal(0, 2, 2_000))
    variance = means**1.4 * np.sqrt(1 + (means / 100) ** 2)

    corrected = fit_lowess(means, variance, n_bins=200, lowess_frac=0.1)

    np.testing.assert_allclose(corrected, 1.0, rtol=0.2)


@pytest.mark.parametrize("lowess_frac", [0.0, 0.1])
@pytest.mark.parametrize(
    "mean_expr",
    [
        np.ones(80),
        np.repeat([1.0, 2.0, 4.0], [30, 25, 25]),
        np.repeat([1.0, 10.0], [70, 70]),
        np.concatenate([np.ones(300), np.geomspace(2.0, 100.0, 70)]),
    ],
    ids=["all_tied", "three_means", "two_large_ties", "large_tie_and_tail"],
)
def test_fit_lowess_adaptive_preserves_variance_signal_with_tied_means(
    mean_expr, lowess_frac
):
    variance = mean_expr**1.4
    variance[-1] *= 4
    expected = np.ones(len(mean_expr))
    expected[-1] = 4

    corrected = fit_lowess(mean_expr, variance, n_bins=200, lowess_frac=lowess_frac)

    np.testing.assert_allclose(corrected, expected, rtol=0.01)


@pytest.mark.parametrize("position", [0, -1, -25])
def test_fit_lowess_adaptive_preserves_injected_variance_signal(position):
    mean_expr = np.sort(np.exp(np.random.default_rng(17).normal(0, 2, 20_000)))
    variance = mean_expr**1.4
    variance[position] *= 4
    expected = np.ones(len(mean_expr))
    expected[position] = 4

    corrected = fit_lowess(mean_expr, variance, n_bins=200, lowess_frac=0.1)

    np.testing.assert_allclose(corrected, expected, rtol=0.05)


def test_fit_lowess_adaptive_is_invariant_to_order_and_units():
    rng = np.random.default_rng(23)
    mean_expr = np.exp(rng.normal(0, 2, 2000))
    variance = (mean_expr + mean_expr**2) * np.exp(rng.normal(0, 0.1, 2000))
    order = rng.permutation(len(mean_expr))

    corrected = fit_lowess(mean_expr, variance, n_bins=200, lowess_frac=0.1)
    permuted = fit_lowess(
        mean_expr[order], variance[order], n_bins=200, lowess_frac=0.1
    )
    scaled = fit_lowess(mean_expr * 1000, variance * 1e6, n_bins=200, lowess_frac=0.1)

    np.testing.assert_allclose(permuted, corrected[order], rtol=1e-7)
    np.testing.assert_allclose(scaled, corrected, rtol=1e-7)


def test_fit_lowess_adaptive_resists_single_low_variance_outlier():
    rng = np.random.default_rng(4)
    mean_expr = np.geomspace(0.01, 100.0, 500)
    variance = mean_expr**1.4 * np.exp(rng.normal(0, 0.08, len(mean_expr)))
    baseline = fit_lowess(
        mean_expr,
        variance,
        n_bins=20,
        lowess_frac=0.4,
        bin_strategy="adaptive",
    )

    outlier_variance = variance.copy()
    outlier_variance[12] *= 1e-12
    with_outlier = fit_lowess(
        mean_expr,
        outlier_variance,
        n_bins=20,
        lowess_frac=0.4,
        bin_strategy="adaptive",
    )

    unaffected = np.arange(len(mean_expr)) != 12
    log_change = np.abs(np.log(baseline[unaffected] / with_outlier[unaffected]))
    assert log_change.max() < 0.05


@pytest.mark.parametrize(("gradient", "accepted"), [(1e-10, True), (1e-6, False)])
def test_fit_lowess_adaptive_accepts_a_stopped_line_search_only_at_the_optimum(
    monkeypatch, gradient, accepted
):
    # L-BFGS-B can stop its line search at a fit that is optimal to machine
    # precision, depending on the scipy version and BLAS threading.
    import scipy.optimize

    means = np.exp(np.random.default_rng(17).normal(0, 2, 2_000))
    variance = means**1.4
    expected = fit_lowess(means, variance, n_bins=200, lowess_frac=0.1)
    minimize = scipy.optimize.minimize

    def stopped_minimize(*args, **kwargs):
        result = minimize(*args, **kwargs)
        result.status = 2
        result.success = False
        result.message = "ABNORMAL: "
        result.jac = np.full_like(result.jac, gradient)
        return result

    monkeypatch.setattr(scipy.optimize, "minimize", stopped_minimize)
    if accepted:
        np.testing.assert_array_equal(
            fit_lowess(means, variance, n_bins=200, lowess_frac=0.1), expected
        )
    else:
        with pytest.raises(ValueError, match="trend fit failed: ABNORMAL"):
            fit_lowess(means, variance, n_bins=200, lowess_frac=0.1)


def test_fit_lowess_fixed_fits_only_positive_finite_genes():
    # A zero variance became log(0) = -inf, was chosen as its bin minimum, and
    # turned every fixed-mode correction into NaN.
    rng = np.random.default_rng(3)
    means = np.exp(rng.uniform(0.0, 5.0, 400))
    variances = means * (1 + 0.2 * means) * np.exp(rng.normal(0.0, 0.3, 400))
    invalid = np.array([3, 7, 50, 51])
    poisoned = variances.copy()
    poisoned[[3, 50]] = 0.0
    poisoned[7] = np.nan
    poisoned[51] = -1.0
    valid = np.ones(len(means), dtype=bool)
    valid[invalid] = False

    corrected = fit_lowess(means, poisoned, 20, 0.3, bin_strategy="fixed")
    expected = fit_lowess(means[valid], variances[valid], 20, 0.3, bin_strategy="fixed")

    np.testing.assert_array_equal(corrected[invalid], np.zeros(len(invalid)))
    np.testing.assert_allclose(corrected[valid], expected, rtol=1e-12)
    np.testing.assert_array_equal(
        fit_lowess(np.array([1.0, 2.0]), np.zeros(2), 4, 0.5, bin_strategy="fixed"),
        np.zeros(2),
    )


def test_fit_lowess_adaptive_handles_small_and_invalid_inputs():
    mean_expr = np.array([1.0, 2.0, 3.0, 4.0, np.nan, 0.0, 8.0])
    variance = np.array([1.0, 0.0, -1.0, 4.0, 2.0, 2.0, 16.0])

    corrected = fit_lowess(
        mean_expr,
        variance,
        n_bins=200,
        lowess_frac=0.1,
        bin_strategy="adaptive",
    )

    assert np.all(np.isfinite(corrected))
    assert np.all(corrected[[0, 3, 6]] > 0)
    np.testing.assert_array_equal(corrected[[1, 2, 4, 5]], np.zeros(4))
    np.testing.assert_array_equal(
        fit_lowess(
            np.array([0.0, np.nan]),
            np.array([0.0, 1.0]),
            n_bins=200,
            lowess_frac=0.1,
            bin_strategy="adaptive",
        ),
        np.zeros(2),
    )

    with pytest.raises(ValueError, match="n_bins"):
        fit_lowess(
            mean_expr,
            variance,
            n_bins=0,
            lowess_frac=0.1,
            bin_strategy="adaptive",
        )
    with pytest.raises(TypeError, match="n_bins"):
        fit_lowess(
            mean_expr,
            variance,
            n_bins=True,
            lowess_frac=0.1,
            bin_strategy="adaptive",
        )
    with pytest.raises(TypeError, match="n_bins"):
        fit_lowess(
            mean_expr,
            variance,
            n_bins=2.5,
            lowess_frac=0.1,
            bin_strategy="adaptive",
        )
    with pytest.raises(ValueError, match="lowess_frac"):
        fit_lowess(
            mean_expr,
            variance,
            n_bins=20,
            lowess_frac=np.nan,
            bin_strategy="adaptive",
        )
    with pytest.raises(ValueError, match="lowess_frac"):
        fit_lowess(
            mean_expr,
            variance,
            n_bins=20,
            lowess_frac=2.0,
            bin_strategy="adaptive",
        )
    # The shared float_argument validator names a real number.
    for not_numeric in ("0.1", True):
        with pytest.raises(TypeError, match="lowess_frac must be a real number"):
            fit_lowess(
                mean_expr,
                variance,
                n_bins=20,
                lowess_frac=not_numeric,
                bin_strategy="adaptive",
            )
    with pytest.raises(ValueError, match="one-dimensional arrays of equal length"):
        fit_lowess(mean_expr, variance[:-1], n_bins=20, lowess_frac=0.1)
    with pytest.raises(ValueError, match="bin_strategy"):
        fit_lowess(
            mean_expr,
            variance,
            n_bins=20,
            lowess_frac=0.1,
            bin_strategy="unknown",
        )


@pytest.mark.parametrize("n_genes", [1, 2])
def test_fit_lowess_adaptive_rejects_insufficient_valid_genes(n_genes):
    with pytest.raises(ValueError, match="At least three genes"):
        fit_lowess(
            np.r_[np.arange(1, n_genes + 1), np.nan, 0],
            np.ones(n_genes + 2),
            n_bins=200,
            lowess_frac=0.1,
        )


def test_highly_variable_feature_selection_applies_all_candidate_filters():
    selected = select_highly_variable_features(
        corrected_variance=np.array([10.0, 8.0, 6.0, 4.0, 2.0]),
        normalized_cell_counts=np.full(5, 5),
        mean_nonzero=np.full(5, 2.0),
        active_features=np.array([True, True, True, True, False]),
        feature_names=np.array(["A", "MT-X", "B", "C", "D"]),
        min_cells=0,
        max_cells=np.inf,
        top_n=2,
        min_var=-np.inf,
        max_var=np.inf,
        min_mean=-np.inf,
        max_mean=np.inf,
        blacklist="^MT-",
        keep_bounds=False,
    )

    np.testing.assert_array_equal(
        selected,
        np.array([True, False, True, False, False]),
    )


def test_hvg_cell_count_bounds_include_minimum_and_exclude_maximum():
    selected = select_highly_variable_features(
        corrected_variance=np.array([100.0, 10.0, 8.0, 100.0]),
        normalized_cell_counts=np.array([19, 20, 21, 80]),
        mean_nonzero=np.ones(4),
        active_features=np.ones(4, dtype=bool),
        feature_names=np.array(["below", "minimum", "inside", "maximum"]),
        min_cells=20,
        max_cells=80,
        top_n=1,
        min_var=-np.inf,
        max_var=np.inf,
        min_mean=-np.inf,
        max_mean=np.inf,
        blacklist="",
        keep_bounds=False,
    )

    np.testing.assert_array_equal(
        selected,
        np.array([False, True, False, False]),
    )


def _hvg_kwargs(**overrides):
    values = dict(
        min_cells=0,
        max_cells=np.inf,
        min_var=-np.inf,
        max_var=np.inf,
        min_mean=-np.inf,
        max_mean=np.inf,
        blacklist="",
        keep_bounds=False,
    )
    values.update(overrides)
    return values


@pytest.mark.parametrize(
    ("names", "blacklist", "expected"),
    [
        (["RPS3", "RPSX", "GENE"], r"^RPS\d+$", [False, True, True]),
        (["g_a", "g-", "GENE"], r"^g_\w+$", [False, True, True]),
        (["x y", "xy", "GENE"], r"^x\sy$", [False, True, True]),
        (["MT-CO1", "mt-Co1", "GENE"], r"(?-i:^MT-)", [False, True, True]),
        (["MT-CO1", "mt-Co1", "GENE"], r"(?i)^mt-", [False, False, True]),
    ],
)
def test_hvg_blacklist_preserves_regex_semantics(names, blacklist, expected):
    selected = select_highly_variable_features(
        corrected_variance=np.array([3.0, 2.0, 1.0]),
        normalized_cell_counts=np.full(3, 5),
        mean_nonzero=np.ones(3),
        active_features=np.ones(3, dtype=bool),
        feature_names=np.array(names),
        top_n=3,
        **_hvg_kwargs(blacklist=blacklist),
    )
    np.testing.assert_array_equal(selected, expected)


def test_hvg_selection_rejects_misaligned_inputs_and_empty_requests():
    inputs = dict(
        corrected_variance=np.array([3.0, 1.0, 2.0]),
        normalized_cell_counts=np.full(3, 5),
        mean_nonzero=np.ones(3),
        active_features=np.ones(3, dtype=bool),
        feature_names=np.array(["a", "b", "c"]),
    )

    with pytest.raises(ValueError, match="one-dimensional arrays of equal length"):
        select_highly_variable_features(
            **{**inputs, "mean_nonzero": np.ones(2)}, top_n=1, **_hvg_kwargs()
        )
    with pytest.raises(ValueError, match="value greater than 0 for `top_n`"):
        select_highly_variable_features(**inputs, top_n=0, **_hvg_kwargs())


def test_hvg_exact_top_n_selects_all_when_top_n_equals_valid_count():
    selected = select_highly_variable_features(
        corrected_variance=np.array([3.0, 1.0, 2.0]),
        normalized_cell_counts=np.full(3, 5),
        mean_nonzero=np.ones(3),
        active_features=np.ones(3, dtype=bool),
        feature_names=np.array(["a", "b", "c"]),
        top_n=3,
        **_hvg_kwargs(),
    )
    np.testing.assert_array_equal(selected, np.array([True, True, True]))


def test_hvg_exact_top_n_selects_the_sole_candidate():
    selected = select_highly_variable_features(
        corrected_variance=np.array([3.0, 1.0, 2.0]),
        normalized_cell_counts=np.full(3, 5),
        mean_nonzero=np.ones(3),
        active_features=np.array([False, True, False]),
        feature_names=np.array(["a", "b", "c"]),
        top_n=10,
        **_hvg_kwargs(),
    )
    np.testing.assert_array_equal(selected, np.array([False, True, False]))


def test_hvg_exact_top_n_tie_breaks_by_feature_index():
    selected = select_highly_variable_features(
        corrected_variance=np.array([5.0, 5.0, 1.0]),
        normalized_cell_counts=np.full(3, 5),
        mean_nonzero=np.ones(3),
        active_features=np.ones(3, dtype=bool),
        feature_names=np.array(["a", "b", "c"]),
        top_n=1,
        **_hvg_kwargs(),
    )
    np.testing.assert_array_equal(selected, np.array([True, False, False]))


def test_binned_sampling_excludes_query_genes():
    rng = np.random.default_rng(1)
    gene_names = [f"gene_{i}" for i in range(120)]
    values = pd.Series(rng.exponential(1.0, len(gene_names)), index=gene_names)
    query_genes = gene_names[10:25]

    controls = binned_sampling(
        values,
        feature_list=query_genes,
        ctrl_size=8,
        n_bins=6,
        rand_seed=42,
    )

    # Genes fall in bins of 120 / (6 - 1) = 24 by expression rank, and each
    # bin that holds a query gene gives up to eight control genes.
    bins = (values.rank(method="min") / 24).astype(int)
    query_bins = set(bins[query_genes])
    drawn = pd.Series(controls).map(bins)
    assert set(controls).isdisjoint(query_genes)
    assert set(drawn) == query_bins
    assert drawn.value_counts().max() <= 8
    # Controls keep the order of the expression table.
    assert controls == [name for name in gene_names if name in set(controls)]


def test_binned_sampling_advances_between_bins_without_changing_global_rng():
    values = pd.Series(np.arange(60), index=[f"g{i}" for i in range(60)])
    targets = ["g0", "g9", "g19", "g29", "g39", "g49", "g59"]
    before = np.random.get_state()
    controls = binned_sampling(values, targets, ctrl_size=3, n_bins=7, rand_seed=4466)
    after = np.random.get_state()

    offsets = [
        tuple(i for i in range(10) if f"g{start + i}" in controls)
        for start in (9, 19, 29, 39, 49)
    ]
    assert len(set(offsets)) == 5
    assert controls == binned_sampling(values, targets, 3, 7, 4466)
    assert set(controls).isdisjoint(targets)
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@pytest.mark.parametrize(
    ("ctrl_size", "n_bins", "error", "message"),
    [
        (True, 4, TypeError, "ctrl_size must be a positive integer"),
        (1.5, 4, TypeError, "ctrl_size must be a positive integer"),
        (0, 4, ValueError, "ctrl_size must be a positive integer"),
        (2, True, TypeError, "n_bins must be an integer greater than one"),
        (2, 1, ValueError, "n_bins must be greater than one"),
        (2, 20, ValueError, "n_bins is too large"),
    ],
)
def test_binned_sampling_rejects_invalid_controls_and_bins(
    ctrl_size, n_bins, error, message
):
    values = pd.Series(np.arange(6), index=[f"g{i}" for i in range(6)])

    with pytest.raises(error, match=message):
        binned_sampling(values, ["g0"], ctrl_size, n_bins, 4466)


def test_interval_search_uses_half_open_overlap_boundaries():
    ranges = np.array([[0, 10], [20, 30], [30, 40]], dtype=np.int64)
    queries = np.array(
        [
            [10, 20],
            [9, 21],
            [30, 30],
            [29, 31],
        ],
        dtype=np.int64,
    )

    np.testing.assert_array_equal(
        binary_search(ranges, queries),
        np.array(
            [
                [-1, -1],
                [0, 2],
                [-1, -1],
                [1, 3],
            ]
        ),
    )


def test_feature_mapping_preserves_half_open_interval_edges():
    peaks = create_bed_from_coord_ids(["chr1:100-200", "chr1:200-300"])
    features = pd.DataFrame(
        [
            ("chr1", 0, 100, "before", "Before", "+"),
            ("chr1", 100, 200, "first", "First", "+"),
            ("chr1", 199, 201, "bridge", "Bridge", "+"),
        ]
    )

    feature_ids, _, mapping = get_feature_mappings(peaks, features)

    assert feature_ids.tolist() == ["before", "first", "bridge"]
    np.testing.assert_array_equal(
        mapping.toarray(),
        np.array(
            [
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    )


def test_feature_mapping_rejects_empty_feature_table():
    peaks = create_bed_from_coord_ids(["chr1:100-200"])
    features = pd.DataFrame(columns=range(6))

    with pytest.raises(ValueError, match="None of the features were found"):
        get_feature_mappings(peaks, features)
