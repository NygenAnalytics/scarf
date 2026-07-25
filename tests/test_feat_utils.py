import numpy as np
import pandas as pd
import pytest

from scarf.features.scoring import binned_sampling
from scarf.features.variability import fit_lowess, select_highly_variable_features
from scarf.quality_control.hto import hto_demux


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


def test_hto_demux_assigns_singlet_and_negative_labels():
    rng = np.random.default_rng(2)
    n_cells = 60
    hto_names = ["HTO_A", "HTO_B", "HTO_C"]

    background = rng.poisson(2, size=(n_cells, len(hto_names)))
    counts = background.astype(float)
    for i in range(n_cells):
        dominant = i % len(hto_names)
        counts[i, dominant] += rng.integers(30, 80)

    hto_counts = pd.DataFrame(counts, columns=hto_names)
    assignments = hto_demux(hto_counts)
    repeated = hto_demux(hto_counts)

    assert len(assignments) == n_cells
    pd.testing.assert_series_equal(assignments, repeated)
    assert assignments.index.equals(hto_counts.index)
    allowed = {"Negative", "Singlet", "Doublet", *hto_names}
    assert set(assignments.unique()).issubset(allowed)
    assert set(assignments.unique()) & set(hto_names)


def test_hto_demux_rejects_empty_cluster_means():
    hto_counts = pd.DataFrame(
        {
            "HTO_A": [0, 0, 0],
            "HTO_B": [0, 0, 0],
        }
    )
    with pytest.raises(AssertionError):
        hto_demux(hto_counts)


def test_hto_demux_rejects_too_few_cells():
    hto_counts = pd.DataFrame(
        {
            "HTO_A": [1, 2],
            "HTO_B": [2, 1],
        }
    )
    with pytest.raises(ValueError, match="at least 3 selected cells"):
        hto_demux(hto_counts)
