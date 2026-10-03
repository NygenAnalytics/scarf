import numpy as np
import pandas as pd
import pytest
from scipy.optimize import curve_fit
from scipy.sparse import coo_matrix
from scipy.stats import norm

from scarf.embeddings.initialization import initial_embedding
from scarf.embeddings.umap import (
    calc_dens_map_params,
    fit_transform,
    fuzzy_simplicial_set,
    simplicial_set_embedding,
)
from scarf.embeddings.harmony import fit_harmony


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
    assert not np.array_equal(embedding, ini_embed)
    np.testing.assert_array_equal(embedding, repeated_embedding)
    # The random start has no ring structure; the layout pulls ring neighbors
    # well inside the typical pairwise distance.
    distances = np.linalg.norm(embedding[:, None] - embedding[None], axis=2)
    ring_neighbors = distances[np.arange(n_cells), (np.arange(n_cells) + 1) % n_cells]
    assert ring_neighbors.mean() < 0.5 * distances[np.triu_indices(n_cells, 1)].mean()
    # UMAP fits 1 / (1 + a * d ** (2 * b)) to the target membership curve.
    spread, min_dist = 1.0, 0.5
    grid = np.linspace(0, spread * 3, 300)
    target = np.where(grid < min_dist, 1.0, np.exp(-(grid - min_dist) / spread))
    (expected_a, expected_b), _ = curve_fit(
        lambda d, a, b: 1.0 / (1.0 + a * d ** (2 * b)), grid, target
    )
    np.testing.assert_allclose([a, b], [expected_a, expected_b], rtol=1e-6)
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
    # Independently: principal-component scores of the centers, each clipped
    # to the 10th and 90th percentiles of a normal fitted around its median,
    # then read for each cell's label. Component signs are arbitrary.
    centered = centers - centers.mean(axis=0)
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    for component in range(2):
        scores = centered @ components[component]
        center, scale = np.median(scores), np.std(scores)
        clipped = np.clip(
            scores,
            norm.ppf(0.1, center, scale),
            norm.ppf(0.9, center, scale),
        )[labels]
        sign = np.sign(actual[0, component] * clipped[0])
        np.testing.assert_allclose(actual[:, component], sign * clipped, atol=1e-6)


def test_initial_embedding_is_repeatable_when_pca_uses_the_randomized_solver():
    rng = np.random.default_rng(0)
    # Nearly tied leading variances make an unseeded randomized SVD differ.
    centers = rng.normal(size=(1000, 150)) * np.linspace(3, 0.1, 150)
    labels = rng.integers(0, len(centers), 400)

    first = initial_embedding(centers, labels, 2)
    second = initial_embedding(centers, labels, 2)

    np.testing.assert_array_equal(first, second)


def test_initial_embedding_accepts_integral_float_labels_and_rejects_invalid():
    centers = np.eye(3)
    integral = initial_embedding(centers, np.array([0.0, 1.0, 2.0]), 2)
    np.testing.assert_array_equal(
        integral, initial_embedding(centers, np.array([0, 1, 2]), 2)
    )

    for labels, message in (
        (np.array([0.0, 1.5]), "must be finite integers"),
        (np.array([0.0, np.nan]), "must be finite integers"),
        (np.array([0.0, -1.0]), "outside the center range"),
        (np.array([0, 3]), "outside the center range"),
        (np.array([[0, 1], [1, 2]]), "must be one-dimensional"),
    ):
        with pytest.raises(ValueError, match=message):
            initial_embedding(centers, labels, 2)
    with pytest.raises(TypeError, match="must contain numeric integers"):
        initial_embedding(centers, np.array(["0", "1"]), 2)
    # Three centers in three dimensions support at most three components.
    for n_components in (0, 4):
        with pytest.raises(ValueError, match="cannot exceed the center count"):
            initial_embedding(centers, np.array([0, 1]), n_components)


def _same_label_neighbor_fraction(values: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of each cell's ten nearest neighbors that share its label."""
    distances = np.linalg.norm(values.T[:, None] - values.T[None], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = np.argsort(distances, axis=1)[:, :10]
    return float(np.mean(labels[nearest] == labels[:, None]))


def test_fit_harmony_corrects_batch_structure():
    rng = np.random.default_rng(0)
    # Two cell types present in every one of three shifted batches.
    batch = np.repeat(np.arange(3), 60)
    cell_type = np.tile(np.repeat(np.arange(2), 30), 3)
    type_centers = np.zeros((2, 6))
    type_centers[0, 0] = type_centers[1, 1] = 5.0
    batch_shift = rng.normal(scale=1.5, size=(3, 6))
    data = (
        rng.normal(scale=0.5, size=(180, 6))
        + type_centers[cell_type]
        + batch_shift[batch]
    ).T

    def batch_spread(values: np.ndarray) -> float:
        """Mean distance of batch centroids from their mean, within cell types."""
        spreads = []
        for kind in range(2):
            centroids = np.stack(
                [
                    values[:, (batch == level) & (cell_type == kind)].mean(axis=1)
                    for level in range(3)
                ]
            )
            spreads.append(
                np.linalg.norm(centroids - centroids.mean(axis=0), axis=1).mean()
            )
        return float(np.mean(spreads))

    corrected = fit_harmony(
        data,
        pd.DataFrame({"batch": [f"batch_{x}" for x in batch]}),
        nclust=4,
        max_iter_harmony=4,
        max_iter_kmeans=5,
        random_state=0,
    ).corrected

    assert corrected.shape == data.shape
    assert np.all(np.isfinite(corrected))
    assert batch_spread(corrected) < 0.2 * batch_spread(data)
    # Batches separate the raw neighborhoods; after correction a cell's
    # neighbors come from all three batches, about one third from its own.
    assert _same_label_neighbor_fraction(data, batch) > 0.9
    assert _same_label_neighbor_fraction(corrected, batch) < 0.45
    # Cell types stay apart. Every batch holds both types equally, so removing
    # batch deviations around each cluster keeps the gap between type centroids.
    assert _same_label_neighbor_fraction(corrected, cell_type) == 1.0

    def type_gap(values: np.ndarray) -> float:
        return float(
            np.linalg.norm(
                values[:, cell_type == 0].mean(axis=1)
                - values[:, cell_type == 1].mean(axis=1)
            )
        )

    np.testing.assert_allclose(type_gap(corrected), type_gap(data), rtol=1e-3)
