import numpy as np
import pandas as pd
import pytest
from scipy.sparse import coo_matrix

pytest.importorskip("umap")

from scarf.harmony import run_harmony
from scarf.umap import calc_dens_map_params, fit_transform, fuzzy_simplicial_set


def _ring_graph(n: int) -> coo_matrix:
    rows, cols, data = [], [], []
    for i in range(n):
        for j in (i - 1, i + 1):
            neighbor = j % n
            if neighbor != i:
                rows.append(i)
                cols.append(neighbor)
                data.append(1.0)
    return coo_matrix((data, (rows, cols)), shape=(n, n))


def test_calc_dens_map_params():
    graph = _ring_graph(8)
    dists = np.full((8, 8), 2.0, dtype=np.float32)
    np.fill_diagonal(dists, 0.0)
    for i in range(8):
        dists[i, (i + 1) % 8] = 1.0
        dists[i, (i - 1) % 8] = 1.0
    mu_sum, r_term = calc_dens_map_params(graph, dists)
    assert mu_sum.shape == (8,)
    assert r_term.shape == (8,)
    assert np.all(np.isfinite(mu_sum))
    assert np.all(np.isfinite(mu_sum[mu_sum > 0]))


def test_fuzzy_simplicial_set_produces_coo_graph():
    graph = coo_matrix(([1.0, 0.6], ([0, 1], [1, 2])), shape=(4, 4))
    merged = fuzzy_simplicial_set(graph, 1.0)
    assert merged.shape == (4, 4)
    assert merged.nnz > 0


def test_fit_transform_runs_short_embedding():
    n_cells = 24
    graph = _ring_graph(n_cells)
    ini_embed = np.random.default_rng(0).normal(size=(n_cells, 2)).astype(np.float32)

    embedding, a, b = fit_transform(
        graph,
        ini_embed,
        spread=1.0,
        min_dist=0.5,
        n_epochs=15,
        random_seed=42,
        repulsion_strength=1.0,
        initial_alpha=1.0,
        negative_sample_rate=5,
        densmap_kwds={},
        parallel=False,
        nthreads=1,
        verbose=False,
    )

    assert embedding.shape == (n_cells, 2)
    assert np.all(np.isfinite(embedding))
    assert a > 0
    assert b > 0


def test_run_harmony_corrects_batch_structure():
    rng = np.random.default_rng(0)
    n_cells = 180
    n_dims = 12
    batch = rng.integers(0, 3, n_cells)
    batch_effect = rng.normal(scale=2.0, size=(3, n_dims))
    data = rng.normal(size=(n_dims, n_cells))
    for cell_idx, batch_id in enumerate(batch):
        data[:, cell_idx] += batch_effect[batch_id]

    meta = pd.DataFrame({"batch": [f"batch_{x}" for x in batch]})
    corrected = run_harmony(
        data,
        meta,
        nclust=15,
        max_iter_harmony=4,
        max_iter_kmeans=5,
        random_state=0,
    )

    assert corrected.shape == data.shape
    assert np.all(np.isfinite(corrected))
    batch_means_before = [data[:, batch == b].mean(axis=1) for b in range(3)]
    batch_means_after = [corrected[:, batch == b].mean(axis=1) for b in range(3)]
    spread_before = np.std([m.mean() for m in batch_means_before])
    spread_after = np.std([m.mean() for m in batch_means_after])
    assert spread_after <= spread_before
