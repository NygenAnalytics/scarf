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


def test_fit_lowess_adaptive_balances_bins_and_interpolates(monkeypatch):
    rng = np.random.default_rng(3)
    mean_expr = np.exp(rng.exponential(1.0, 101))
    variance = mean_expr**1.2 * np.exp(rng.normal(0, 0.1, len(mean_expr)))
    captured: dict[str, np.ndarray] = {}

    def fake_lowess(
        endog,
        exog,
        *,
        return_sorted,
        frac,
        it,
    ):
        captured["endog"] = np.asarray(endog)
        captured["exog"] = np.asarray(exog)
        return np.asarray(exog)

    monkeypatch.setattr(
        "statsmodels.nonparametric.smoothers_lowess.lowess",
        fake_lowess,
    )
    corrected = fit_lowess(
        mean_expr,
        variance,
        n_bins=4,
        lowess_frac=0.4,
        bin_strategy="adaptive",
    )

    order = np.argsort(np.log(mean_expr), kind="stable")
    sorted_means = np.log(mean_expr)[order]
    sorted_variances = np.log(variance)[order]
    bins = (slice(0, 26), slice(26, 51), slice(51, 76), slice(76, 101))
    expected_x = np.array([np.median(sorted_means[idx]) for idx in bins])
    expected_y = np.array([np.quantile(sorted_variances[idx], 0.1) for idx in bins])

    np.testing.assert_allclose(captured["exog"], expected_x)
    np.testing.assert_allclose(captured["endog"], expected_y)
    expected_correction = np.interp(np.log(mean_expr), expected_x, expected_x)
    np.testing.assert_allclose(
        corrected,
        np.exp(np.log(variance) - expected_correction),
    )


def test_fit_lowess_adaptive_keeps_equal_means_in_one_bin(monkeypatch):
    mean_expr = np.repeat([1.0, 2.0, 4.0], [30, 25, 25])
    variance = np.linspace(1.0, 3.0, len(mean_expr))
    captured: dict[str, np.ndarray] = {}

    def fake_lowess(
        endog,
        exog,
        *,
        return_sorted,
        frac,
        it,
    ):
        captured["exog"] = np.asarray(exog)
        return np.asarray(endog)

    monkeypatch.setattr(
        "statsmodels.nonparametric.smoothers_lowess.lowess",
        fake_lowess,
    )
    corrected = fit_lowess(
        mean_expr,
        variance,
        n_bins=3,
        lowess_frac=0.5,
        bin_strategy="adaptive",
    )

    np.testing.assert_allclose(captured["exog"], np.log([1.0, 2.0, 4.0]))
    assert np.all(np.isfinite(corrected))
    assert np.all(corrected > 0)


def test_fit_lowess_adaptive_reduces_bins_after_large_tie(monkeypatch):
    mean_expr = np.concatenate([np.ones(30), np.geomspace(2.0, 100.0, 70)])
    variance = mean_expr**1.2
    captured: dict[str, np.ndarray] = {}

    def fake_lowess(
        endog,
        exog,
        *,
        return_sorted,
        frac,
        it,
    ):
        captured["exog"] = np.asarray(exog)
        return np.asarray(endog)

    monkeypatch.setattr(
        "statsmodels.nonparametric.smoothers_lowess.lowess",
        fake_lowess,
    )
    corrected = fit_lowess(
        mean_expr,
        variance,
        n_bins=4,
        lowess_frac=0.5,
        bin_strategy="adaptive",
    )

    assert len(captured["exog"]) == 3
    assert captured["exog"][0] == 0
    assert np.all(np.isfinite(corrected))


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
    mean_expr = np.array([1.0, 2.0, 3.0, 4.0, np.nan, 0.0])
    variance = np.array([1.0, 0.0, -1.0, 4.0, 2.0, 2.0])

    corrected = fit_lowess(
        mean_expr,
        variance,
        n_bins=200,
        lowess_frac=0.1,
        bin_strategy="adaptive",
    )

    assert np.all(np.isfinite(corrected))
    assert np.all(corrected[[0, 3]] > 0)
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
