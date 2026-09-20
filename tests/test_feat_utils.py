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
from scarf.quality_control.hto import (
    _background_clusters,
    _classify_hto_identities,
    _cluster_labels,
    _clr_normalize,
    _fit_negative_binomial_parameters,
    _negative_binomial_cutoff,
    _positive_hto_calls,
    hto_demux,
)


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

    assert len(controls) > 0
    assert set(controls).isdisjoint(query_genes)
    assert all(name in gene_names for name in controls)


def test_hto_negative_binomial_cutoff_is_unshifted(monkeypatch):
    assert _negative_binomial_cutoff(mu=1, alpha=1) == 6

    counts = pd.DataFrame({"HTO_A": [1, 2, 6, 7]})
    monkeypatch.setattr(
        "scarf.quality_control.hto._fit_negative_binomial_parameters",
        lambda values, hto_name: (1, 1),
    )

    positive = _positive_hto_calls(counts, np.asarray([0, 0, 1, 1]))

    assert positive["HTO_A"].tolist() == [False, False, False, True]


def test_hto_background_cluster_uses_raw_means():
    counts = pd.DataFrame({"HTO_A": [0, 100, 40, 40]})
    cluster_labels = np.asarray([0, 0, 1, 1])

    background = _background_clusters(counts, cluster_labels)
    normalized_means = _clr_normalize(counts).groupby(cluster_labels).mean()

    assert background["HTO_A"] == 1
    assert normalized_means["HTO_A"].idxmin() == 0


def test_hto_classification_uses_clr_argmax_for_singlets():
    index = ["negative", "singlet", "doublet", "tie"]
    normalized = pd.DataFrame(
        {
            "HTO_A": [0.2, 0.2, 1.0, 0.5],
            "HTO_B": [0.1, 1.5, 0.9, 0.5],
        },
        index=index,
    )
    positive = pd.DataFrame(
        {
            "HTO_A": [False, True, True, True],
            "HTO_B": [False, False, True, False],
        },
        index=index,
    )

    identities = _classify_hto_identities(normalized, positive)

    assert identities.to_dict() == {
        "negative": "Negative",
        "singlet": "HTO_B",
        "doublet": "Doublet",
        "tie": "HTO_A",
    }


def test_hto_demux_assigns_singlet_and_negative_labels():
    rng = np.random.default_rng(2)
    n_cells = 60
    hto_names = ["cluster", "HTO_B", "HTO_C"]

    background = rng.poisson(2, size=(n_cells, len(hto_names)))
    counts = background.astype(float)
    for i in range(n_cells):
        dominant = i % len(hto_names)
        counts[i, dominant] += rng.integers(30, 80)

    hto_counts = pd.DataFrame(
        counts,
        columns=hto_names,
        index=[f"cell_{index}" for index in range(n_cells)],
    )
    original = hto_counts.copy(deep=True)
    assignments = hto_demux(hto_counts)
    repeated = hto_demux(hto_counts)

    assert len(assignments) == n_cells
    pd.testing.assert_series_equal(assignments, repeated)
    assert assignments.index.equals(hto_counts.index)
    pd.testing.assert_frame_equal(hto_counts, original)
    allowed = {"Negative", "Singlet", "Doublet", *hto_names}
    assert set(assignments.unique()).issubset(allowed)
    assert set(assignments.unique()) & set(hto_names)


@pytest.mark.parametrize(
    ("hto_counts", "error", "message"),
    [
        (
            np.asarray([[1], [2]]),
            TypeError,
            "must be a pandas DataFrame",
        ),
        (
            pd.DataFrame(index=range(2)),
            ValueError,
            "at least one HTO",
        ),
        (
            pd.DataFrame(np.ones((3, 2)), columns=["HTO_A", "HTO_A"]),
            ValueError,
            "HTO IDs must be unique",
        ),
        (
            pd.DataFrame({" ": [1, 2]}),
            ValueError,
            "non-empty strings",
        ),
        (
            pd.DataFrame({"Negative": [1, 2]}),
            ValueError,
            "reserved identity labels",
        ),
        (
            pd.DataFrame({"HTO_A": [1, 2]}, index=["cell", "cell"]),
            ValueError,
            "cell index must be unique",
        ),
        (
            pd.DataFrame({"HTO_A": ["1", "2"]}),
            TypeError,
            "only numeric raw counts",
        ),
        (
            pd.DataFrame({"HTO_A": [1, np.nan]}),
            ValueError,
            "only finite raw counts",
        ),
        (
            pd.DataFrame({"HTO_A": [1, -1]}),
            ValueError,
            "only nonnegative raw counts",
        ),
        (
            pd.DataFrame({"HTO_A": [1, 1.5]}),
            ValueError,
            "integer-valued raw counts",
        ),
        (
            pd.DataFrame({"HTO_A": [0, 0]}),
            ValueError,
            "no positive counts",
        ),
    ],
    ids=[
        "not-dataframe",
        "no-htos",
        "duplicate-htos",
        "empty-hto",
        "reserved-hto",
        "duplicate-cells",
        "nonnumeric",
        "nonfinite",
        "negative",
        "fractional",
        "all-zero-hto",
    ],
)
def test_hto_demux_rejects_invalid_input(hto_counts, error, message):
    with pytest.raises(error, match=message):
        hto_demux(hto_counts)


def test_hto_demux_rejects_insufficiently_distinct_profiles():
    hto_counts = pd.DataFrame(
        {
            "HTO_A": [1, 1, 1],
            "HTO_B": [2, 2, 2],
        }
    )

    with pytest.raises(ValueError, match="3 distinct normalized cell profiles"):
        hto_demux(hto_counts)


def test_hto_cluster_labels_rejects_collapsed_kmeans(monkeypatch):
    class CollapsedKMeans:
        def __init__(self, **kwargs):
            pass

        def fit_predict(self, values):
            return np.asarray([0, 0, 1])

    monkeypatch.setattr("sklearn.cluster.KMeans", CollapsedKMeans)
    normalized = pd.DataFrame(
        {
            "HTO_A": [0.0, 1.0, 2.0],
            "HTO_B": [2.0, 1.0, 0.0],
        }
    )

    with pytest.raises(ValueError, match="2 occupied clusters; expected 3"):
        _cluster_labels(normalized, random_seed=0)


@pytest.mark.parametrize(
    ("counts", "labels", "message"),
    [
        (
            pd.DataFrame({"HTO_A": [1, 10, 20]}),
            np.asarray([0, 1, 2]),
            "at least two cells",
        ),
        (
            pd.DataFrame({"HTO_A": [0, 0, 10, 20]}),
            np.asarray([0, 0, 1, 1]),
            "contains only zero counts",
        ),
    ],
    ids=["single-cell", "all-zero"],
)
def test_hto_positive_calls_rejects_invalid_backgrounds(counts, labels, message):
    with pytest.raises(ValueError, match=message):
        _positive_hto_calls(counts, labels)


def test_hto_negative_binomial_fit_rejects_nonconvergence(monkeypatch):
    class FitResult:
        mle_retvals = {"converged": False}
        params = np.asarray([0.0, 1.0])

    class NonconvergedModel:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, **kwargs):
            return FitResult()

    monkeypatch.setattr(
        "statsmodels.discrete.discrete_model.NegativeBinomial",
        NonconvergedModel,
    )

    with pytest.raises(ValueError, match="did not converge for HTO 'HTO_A'"):
        _fit_negative_binomial_parameters(np.asarray([1, 2]), "HTO_A")


def test_hto_negative_binomial_fit_wraps_optimizer_errors(monkeypatch):
    class FailingModel:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("optimizer failed")

    monkeypatch.setattr(
        "statsmodels.discrete.discrete_model.NegativeBinomial",
        FailingModel,
    )

    with pytest.raises(ValueError, match="fit failed for HTO 'HTO_A'"):
        _fit_negative_binomial_parameters(np.asarray([1, 2]), "HTO_A")


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        (np.asarray([np.inf, 1.0]), "invalid mean"),
        (np.asarray([0.0, 0.0]), "invalid dispersion"),
    ],
    ids=["mean", "dispersion"],
)
def test_hto_negative_binomial_fit_rejects_invalid_parameters(
    monkeypatch,
    parameters,
    message,
):
    class FitResult:
        mle_retvals = {"converged": True}
        params = parameters

    class InvalidModel:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, **kwargs):
            return FitResult()

    monkeypatch.setattr(
        "statsmodels.discrete.discrete_model.NegativeBinomial",
        InvalidModel,
    )

    with pytest.raises(ValueError, match=message):
        _fit_negative_binomial_parameters(np.asarray([1, 2]), "HTO_A")


def test_hto_negative_binomial_cutoff_rejects_nonfinite_ppf(monkeypatch):
    monkeypatch.setattr("scipy.stats.nbinom.ppf", lambda *args, **kwargs: np.inf)

    with pytest.raises(ValueError, match="cutoff must be a finite integer"):
        _negative_binomial_cutoff(mu=1, alpha=1)


def test_hto_demux_rejects_too_few_cells():
    hto_counts = pd.DataFrame(
        {
            "HTO_A": [1, 2],
            "HTO_B": [2, 1],
        }
    )
    with pytest.raises(ValueError, match="at least 3 selected cells"):
        hto_demux(hto_counts)


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
