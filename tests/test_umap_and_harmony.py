import numpy as np
import pandas as pd
import pytest
from scipy.sparse import coo_matrix

from scarf.embeddings.initialization import initial_embedding
from scarf.embeddings.umap import (
    calc_dens_map_params,
    fit_transform,
    fuzzy_simplicial_set,
    simplicial_set_embedding,
)
from scarf.embeddings.harmony import run_harmony


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
    dists = np.full((8, 8), 3.0, dtype=np.float32)
    np.fill_diagonal(dists, 0.0)
    for i in range(8):
        for j in (i - 1, i + 1):
            neighbor = j % 8
            dists[i, neighbor] = (
                1.0 + ((min(i, neighbor) * 3 + max(i, neighbor)) % 5) / 5.0
            )
    mu_sum, r_term = calc_dens_map_params(graph, dists)
    np.testing.assert_array_equal(mu_sum, np.full(8, 4.0, dtype=np.float32))
    np.testing.assert_allclose(
        r_term,
        [
            -0.1043465,
            -1.270655,
            0.6717964,
            1.7731321,
            0.89659786,
            -0.10434671,
            -1.270655,
            -0.59152323,
        ],
        rtol=1e-6,
        atol=1e-6,
    )


def test_simplicial_embedding_restores_numba_threads_on_failure(monkeypatch):
    import numba
    import umap.layouts

    def fail(**_kwargs):
        raise RuntimeError("injected UMAP failure")

    monkeypatch.setattr(umap.layouts, "optimize_layout_euclidean", fail)
    graph = _ring_graph(4)
    previous_threads = numba.get_num_threads()
    with pytest.raises(RuntimeError, match="injected UMAP failure"):
        simplicial_set_embedding(
            graph,
            np.zeros((4, 2), dtype=np.float32),
            2,
            1.0,
            1.0,
            1,
            1.0,
            1.0,
            5,
            {},
            False,
            1,
            False,
        )
    assert numba.get_num_threads() == previous_threads


def test_fuzzy_simplicial_set_produces_coo_graph():
    graph = coo_matrix(([1.0, 0.6], ([0, 1], [1, 2])), shape=(4, 4))
    merged = fuzzy_simplicial_set(graph, 1.0)
    np.testing.assert_array_equal(
        merged.toarray(),
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.6, 0.0],
            [0.0, 0.6, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
    )


def test_fit_transform_runs_short_embedding():
    n_cells = 24
    graph = _ring_graph(n_cells)
    ini_embed = np.random.default_rng(0).normal(size=(n_cells, 2)).astype(np.float32)

    def run_embedding() -> tuple[np.ndarray, float, float]:
        return fit_transform(
            graph,
            ini_embed.copy(),
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

    embedding, a, b = run_embedding()
    repeated_embedding, repeated_a, repeated_b = run_embedding()

    assert embedding.shape == (n_cells, 2)
    assert np.all(np.isfinite(embedding))
    assert np.all(np.ptp(embedding, axis=0) > 0)
    assert not np.array_equal(embedding, ini_embed)
    np.testing.assert_array_equal(embedding, repeated_embedding)
    np.testing.assert_allclose(
        [a, b],
        [0.58303002, 1.33416699],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose([a, b], [repeated_a, repeated_b], rtol=0, atol=0)


def test_initial_embedding_matches_regression_values():
    centers = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 1.0],
            [0.0, 3.0, 2.0],
            [2.0, 3.0, 4.0],
        ]
    )
    labels = np.array([2, 0, 3, 1, 2])

    actual = initial_embedding(centers, labels, 2)

    np.testing.assert_allclose(
        actual,
        [
            [1.0399364, -1.3556585],
            [-2.435071, -0.58934784],
            [2.4441376, 0.7173242],
            [-1.3938057, 1.3570011],
            [1.0399364, -1.3556585],
        ],
        rtol=1e-6,
        atol=1e-6,
    )


def test_initial_embedding_accepts_integral_float_labels_and_rejects_invalid():
    centers = np.eye(3)
    integral = initial_embedding(centers, np.array([0.0, 1.0, 2.0]), 2)
    assert integral.shape == (3, 2)

    for labels in (
        np.array([0.0, 1.5]),
        np.array([0.0, -1.0]),
        np.array([0.0, np.nan]),
        np.array([0, 3]),
    ):
        with pytest.raises(ValueError):
            initial_embedding(centers, labels, 2)


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
