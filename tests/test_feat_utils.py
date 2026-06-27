import numpy as np
import pandas as pd
import pytest

from scarf.feat_utils import binned_sampling, fit_lowess, hto_demux


def test_fit_lowess_returns_per_feature_corrections():
    rng = np.random.default_rng(0)
    n_genes = 80
    mean_expr = rng.uniform(0.5, 20.0, n_genes)
    variance = mean_expr**1.5 + rng.normal(0, 0.05, n_genes)
    variance = np.clip(variance, 0.1, None)

    corrected = fit_lowess(mean_expr, variance, n_bins=8, lowess_frac=0.6)

    assert corrected.shape == (n_genes,)
    assert np.all(np.isfinite(corrected))
    assert np.all(corrected > 0)


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

    assert len(assignments) == n_cells
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
