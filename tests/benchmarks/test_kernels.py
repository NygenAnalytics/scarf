"""Benchmarks of the compute kernels behind the slowest production stages.

Each benchmark calls the function its stage calls, on inputs shaped like the
recorded 1M-cell funnel, and checks the result, so it doubles as a test of
that kernel at the smoke size.
"""

import numpy as np
import pytest

from . import inputs
from .harness import Ladder

pytestmark = pytest.mark.benchmark

CELLS = Ladder(sizes=(4_000, 8_000, 16_000, 32_000), smoke=600)
GRAPH_CELLS = Ladder(sizes=(2_000, 4_000, 8_000, 16_000), smoke=600)


def test_library_size_normalization(bench) -> None:
    from scarf.assay.normalization import library_size_values

    def make(n_cells: int):
        counts, _labels = inputs.labelled_counts(n_cells)
        block = np.ascontiguousarray(counts[:, : inputs.N_HVGS])
        totals = counts.sum(axis=1, dtype=np.float64)
        totals[totals == 0] = 1
        return lambda: library_size_values(
            block,
            totals,
            inputs.SIZE_FACTOR,
            dtype=np.float64,
            log_transform=True,
        )

    def check(n_cells: int, values) -> None:
        counts, _labels = inputs.labelled_counts(n_cells)
        totals = counts.sum(axis=1, dtype=np.float64)
        totals[totals == 0] = 1
        expected = np.log1p(
            inputs.SIZE_FACTOR * counts[:, : inputs.N_HVGS] / totals[:, None]
        )
        np.testing.assert_allclose(values, expected, rtol=1e-12)

    bench("normalization.library_size_log", make, CELLS, check=check)


def test_hvg_gene_major_statistics(bench) -> None:
    from scarf.assay.rna import _hvg_stats_gene_major

    def make(n_cells: int):
        counts, _labels = inputs.labelled_counts(n_cells)
        gene_major = np.ascontiguousarray(counts.T)
        totals = counts.sum(axis=1, dtype=np.float64)
        totals[totals == 0] = 1
        inverse = 1.0 / totals
        destination = np.arange(inputs.N_GENES, dtype=np.int64)

        def call():
            outputs = [np.zeros(inputs.N_GENES) for _ in range(3)]
            _hvg_stats_gene_major(
                gene_major,
                inverse,
                inputs.SIZE_FACTOR,
                destination,
                *outputs,
            )
            return outputs

        return call

    def check(n_cells: int, outputs) -> None:
        counts, _labels = inputs.labelled_counts(n_cells)
        totals = counts.sum(axis=1, dtype=np.float64)
        totals[totals == 0] = 1
        values = inputs.SIZE_FACTOR * counts / totals[:, None]
        nonzero, total, squares = outputs
        np.testing.assert_array_equal(nonzero, (counts > 0).sum(axis=0))
        np.testing.assert_allclose(total, values.sum(axis=0), rtol=1e-10)
        np.testing.assert_allclose(squares, (values**2).sum(axis=0), rtol=1e-10)

    bench("hvg.gene_major_statistics", make, CELLS, check=check)


def _marker_inputs(n_cells: int):
    counts, labels = inputs.labelled_counts(n_cells)
    gene_major = np.ascontiguousarray(counts[:, :200].T)
    totals = counts.sum(axis=1, dtype=np.float64)
    totals[totals == 0] = 1
    codes = labels.astype(np.int64)
    group_counts = np.bincount(codes, minlength=inputs.N_GROUPS).astype(np.float64)
    return gene_major, totals, codes, group_counts


def test_marker_gene_major_ranking(bench) -> None:
    from scarf.features.markers.rank import _batch_stats, _marker_stats_gene_major

    def make(n_cells: int):
        gene_major, totals, codes, group_counts = _marker_inputs(n_cells)
        destination = np.arange(gene_major.shape[0], dtype=np.int64)

        def call():
            out = np.zeros((gene_major.shape[0], inputs.N_GROUPS, 8))
            invalid = _marker_stats_gene_major(
                gene_major,
                totals,
                inputs.SIZE_FACTOR,
                True,
                codes,
                group_counts,
                float(n_cells),
                destination,
                1,
                out,
            )
            assert invalid == -1
            return out

        return call

    def check(n_cells: int, out) -> None:
        # The sparse radix path must agree with the dense ranking kernel.
        gene_major, totals, codes, group_counts = _marker_inputs(n_cells)
        normalized = np.log1p(inputs.SIZE_FACTOR * gene_major.T / totals[:, None])
        expected = _batch_stats(normalized, codes, group_counts, n_cells)
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-9)

    bench("markers.gene_major_ranking", make, CELLS, check=check)


def test_connectivity_membership(bench) -> None:
    from scarf.neighbors.graph import build_connectivity_arrays

    def make(n_cells: int):
        indices, distances = inputs.neighbors(n_cells)
        return lambda: build_connectivity_arrays(
            indices, distances, local_connectivity=1.0, bandwidth=1.5
        )

    def check(n_cells: int, value) -> None:
        edges, weights = value
        indices, _distances = inputs.neighbors(n_cells)
        np.testing.assert_array_equal(edges[:, 0], np.repeat(np.arange(n_cells), 11))
        np.testing.assert_array_equal(edges[:, 1], indices.ravel())
        assert np.all((weights > 0) & (weights <= 1))
        # With local_connectivity=1 every cell's nearest neighbour has weight 1.
        np.testing.assert_allclose(weights.reshape(n_cells, -1)[:, 0], 1.0)

    bench("graph.connectivity_membership", make, CELLS, check=check)


def test_umap_layout(bench) -> None:
    from scarf.embeddings.umap import fit_transform

    def make(n_cells: int):
        graph = inputs.connectivity_graph(n_cells).tocoo()
        start = inputs.initial_layout(n_cells)

        def call():
            embedding, _a, _b = fit_transform(
                graph,
                start.copy(),
                spread=2.0,
                min_dist=1.0,
                n_epochs=30,
                random_seed=4444,
                repulsion_strength=1.0,
                initial_alpha=1.0,
                negative_sample_rate=5,
                densmap_kwds={},
                parallel=False,
                nthreads=1,
                verbose=False,
            )
            return embedding

        return call

    def check(n_cells: int, embedding) -> None:
        assert embedding.shape == (n_cells, 2)
        assert np.isfinite(embedding).all()
        # Neighbours in the graph end up closer than random pairs of cells.
        graph = inputs.connectivity_graph(n_cells).tocoo()
        linked = np.linalg.norm(embedding[graph.row] - embedding[graph.col], axis=1)
        rng = np.random.default_rng(0)
        random_pairs = rng.integers(0, n_cells, size=(graph.nnz, 2))
        unlinked = np.linalg.norm(
            embedding[random_pairs[:, 0]] - embedding[random_pairs[:, 1]], axis=1
        )
        assert np.median(linked) < 0.5 * np.median(unlinked)

    # The production funnel runs 300 epochs; layout time is linear in epochs.
    ladder = Ladder(sizes=GRAPH_CELLS.sizes, smoke=GRAPH_CELLS.smoke, work=10.0)
    bench("embedding.umap_layout", make, ladder, check=check)


def test_leiden_partition(bench) -> None:
    from scarf.clustering.leiden import leiden_membership

    def make(n_cells: int):
        graph = inputs.connectivity_graph(n_cells)
        return lambda: leiden_membership(graph, 1.0, 4444)

    def check(n_cells: int, labels) -> None:
        _coordinates, truth = inputs.labelled_coordinates(n_cells)
        assert labels.shape == (n_cells,)
        assert labels.min() == 1
        # Clusters may split a drawn group but must not mix two of them.
        assert inputs.cluster_purity(labels, truth) > 0.99

    bench("clustering.leiden", make, GRAPH_CELLS, check=check)


def test_paris_hierarchy(bench) -> None:
    from scarf.clustering._paris_core import fit_paris_hierarchy
    from scarf.clustering.paris import fixed_cut

    def make(n_cells: int):
        graph = inputs.connectivity_graph(n_cells)
        return lambda: fit_paris_hierarchy(graph, nthreads=1)

    def check(n_cells: int, hierarchy) -> None:
        _coordinates, truth = inputs.labelled_coordinates(n_cells)
        labels = fixed_cut(hierarchy, inputs.N_GROUPS)
        # Cutting at the drawn group count recovers the drawn groups.
        assert len(np.unique(labels)) == inputs.N_GROUPS
        assert inputs.cluster_purity(labels, truth) > 0.99

    bench("clustering.paris_hierarchy", make, GRAPH_CELLS, check=check)


def test_gram_pca(bench) -> None:
    from scarf.embeddings.reduction import fit_incremental_pca
    from scarf.matrix.chunked import ChunkedArray

    def normalized(n_cells: int) -> np.ndarray:
        counts, _labels = inputs.labelled_counts(n_cells)
        totals = counts.sum(axis=1, dtype=np.float64)
        totals[totals == 0] = 1
        return np.log1p(
            inputs.SIZE_FACTOR * counts[:, : inputs.N_HVGS] / totals[:, None]
        ).astype(np.float32)

    def make(n_cells: int):
        values = normalized(n_cells)
        data = ChunkedArray.from_numpy(values, block_size=2_000)
        mask = np.ones(n_cells, dtype=bool)
        return lambda: fit_incremental_pca(
            data,
            dims=inputs.N_DIMS,
            batch_size=2_000,
            use_for_pca=mask,
            scale=None,
            nthreads=1,
        )

    def check(n_cells: int, value) -> None:
        loadings, _model = value
        values = normalized(n_cells).astype(np.float64)
        centered = values - values.mean(axis=0)
        singular = np.linalg.svd(centered, compute_uv=False)
        assert loadings.shape == (inputs.N_HVGS, inputs.N_DIMS)
        # Projecting the centered data onto the loadings keeps the variance
        # of the exact top components, which only the top subspace does.
        kept = np.linalg.norm(centered @ loadings) ** 2
        np.testing.assert_allclose(
            kept, np.sum(singular[: inputs.N_DIMS] ** 2), rtol=1e-4
        )

    bench("reduction.gram_pca", make, CELLS, check=check)


def test_kmeans_initialization(bench) -> None:
    from scarf.matrix.chunked import ChunkedArray
    from scarf.neighbors.stages import (
        ChunkedCoordinateStream,
        KMeansInitializationStage,
    )

    def make(n_cells: int):
        coordinates, _labels = inputs.labelled_coordinates(n_cells)
        stream = ChunkedCoordinateStream(
            ChunkedArray.from_numpy(coordinates, block_size=2_000), nthreads=1
        )
        return lambda: KMeansInitializationStage.fit(
            stream=stream,
            n_rows=n_cells,
            batch_size=2_000,
            n_clusters=1_000,
            rand_state=4466,
            nthreads=1,
        )

    def check(n_cells: int, fitted) -> None:
        coordinates, _labels = inputs.labelled_coordinates(n_cells)
        points = coordinates.astype(np.float64)
        centers = np.asarray(fitted.model.cluster_centers_, dtype=np.float64)
        assert centers.shape == (1_000, inputs.N_DIMS)
        distances = (
            (points**2).sum(axis=1)[:, None]
            - 2 * points @ centers.T
            + (centers**2).sum(axis=1)[None, :]
        )
        labelled = distances[np.arange(n_cells), fitted.labels.astype(np.intp)]
        # Every cell is labelled with its nearest centroid, up to the float32
        # rounding that separates near-ties among 1,000 dense centroids. The
        # absolute term covers cells that are themselves a centroid, against
        # squared distances in the hundreds between drawn clusters.
        np.testing.assert_allclose(
            labelled, distances.min(axis=1), rtol=1e-4, atol=1e-6
        )

    # The production funnel fits 1,000 centroids, so the smoke size holds more.
    ladder = Ladder(sizes=CELLS.sizes, smoke=2_000)
    bench("initialization.kmeans", make, ladder, check=check)
