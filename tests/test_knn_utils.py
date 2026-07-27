import warnings

import numpy as np
import pytest
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.metrics import adjusted_rand_score

from scarf.clustering.leiden import leiden_membership
from scarf.neighbors.graph import (
    build_connectivity_arrays,
    calc_snn,
    merge_graphs,
    weight_sort_indices,
)
from scarf.neighbors.diffusion import diffusion_operator
from scarf.neighbors.integration import wnn_integration
from scarf.utils import logger


def _simple_knn_graph(n: int, k: int = 3) -> csr_matrix:
    rows, cols, data = [], [], []
    for i in range(n):
        for j in range(1, k + 1):
            neighbor = (i + j) % n
            rows.append(i)
            cols.append(neighbor)
            data.append(float(j))
    return csr_matrix((data, (rows, cols)), shape=(n, n))


def _grouped_knn_graph(groups: list[list[int]]) -> csr_matrix:
    n_cells = sum(len(group) for group in groups)
    rows = []
    cols = []
    for group in groups:
        for cell in group:
            neighbors = [neighbor for neighbor in group if neighbor != cell]
            rows.extend([cell] * len(neighbors))
            cols.extend(neighbors)
    return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_cells, n_cells))


def test_leiden_membership_preserves_disconnected_partitions():
    graph = _grouped_knn_graph([[0, 1, 2, 3], [4, 5, 6, 7]])

    actual = leiden_membership(graph, resolution=1.0, random_seed=4444)

    assert adjusted_rand_score([1, 1, 1, 1, 2, 2, 2, 2], actual) == pytest.approx(1.0)


def test_diffusion_operator_matches_powered_row_normalization():
    graph = csr_matrix(
        [
            [0.0, 2.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 3.0, 0.0],
        ]
    )

    actual = diffusion_operator(graph, power=2)

    np.testing.assert_allclose(
        actual.toarray(),
        [
            [0.5, 0.0, 0.5],
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 0.5],
        ],
        rtol=0,
        atol=1e-12,
    )


def _multimodal_wnn_inputs() -> tuple[csr_matrix, np.ndarray, csr_matrix, np.ndarray]:
    g1 = _grouped_knn_graph([[0, 1, 2, 3], [4, 5, 6, 7]])
    g2 = _grouped_knn_graph([[0, 2, 4, 6], [1, 3, 5, 7]])
    ld1 = np.array(
        [0, 101, -97, 2, 1e6, 1e6 + 101, 1e6 - 97, 1e6 + 2],
        dtype=np.float64,
    ).reshape(-1, 1)
    ld2 = np.array(
        [0, 1e6, 101, 1e6 + 101, -97, 1e6 - 97, 2, 1e6 + 2],
        dtype=np.float64,
    ).reshape(-1, 1)
    return g1, ld1, g2, ld2


def test_calc_snn_returns_normalized_overlap():
    graph = _simple_knn_graph(6, k=3)
    indices = graph.indices.reshape((6, 3))
    snn = calc_snn(indices)
    assert snn.shape == (6, 3)
    assert np.all(snn >= 0)
    assert np.all(snn <= 1)


def test_weight_sort_indices_keeps_top_neighbors():
    indices = np.array([4, 1, 2, 1, 3])
    weights = np.array([0.2, 0.5, 0.4, 0.6, 0.1])
    sort_weights = weights + np.array([0.0, 0.2, 0.1, 0.2, 0.0])
    kept_idx, kept_w = weight_sort_indices(indices, weights, sort_weights, n=3)
    assert len(kept_idx) == 3
    assert len(kept_w) == 3
    assert len(set(kept_idx)) == len(kept_idx)


def test_merge_graphs_preserves_shape_and_edge_count():
    g1 = _simple_knn_graph(8, k=3)
    g2 = _simple_knn_graph(8, k=3)
    merged = merge_graphs([g1, g2])
    assert isinstance(merged, coo_matrix)
    assert merged.shape == g1.shape
    assert merged.nnz == g1.nnz


def test_merge_graphs_rejects_mismatched_shapes():
    g1 = _simple_knn_graph(6, k=3)
    g2 = _simple_knn_graph(8, k=3)
    with pytest.raises(ValueError, match="same shape"):
        merge_graphs([g1, g2])


def test_merge_graphs_rejects_one_neighbor_snn_input():
    graph = _simple_knn_graph(6, k=1)

    with pytest.raises(ValueError, match="at least two neighbors"):
        merge_graphs([graph, graph.copy()])


def test_build_connectivity_arrays_runs_in_memory():
    n_cells, n_neighbors = 6, 5
    idx = np.array(
        [
            [(row + offset + 1) % n_cells for offset in range(n_neighbors)]
            for row in range(n_cells)
        ]
    )
    dist = np.array(
        [
            [0.10, 0.30, 0.80, 1.50, 3.00],
            [0.13, 0.35, 0.82, 1.60, 3.20],
            [0.16, 0.40, 0.84, 1.70, 3.40],
            [0.19, 0.45, 0.86, 1.80, 3.60],
            [0.22, 0.50, 0.88, 1.90, 3.80],
            [0.25, 0.55, 0.90, 2.00, 4.00],
        ]
    )
    expected_weights = np.array(
        [
            1.0,
            0.9779863,
            0.92504877,
            0.8557153,
            0.7241439,
            1.0,
            0.97687674,
            0.92925274,
            0.85528576,
            0.7214707,
            1.0,
            0.97586185,
            0.9331116,
            0.8548864,
            0.7190224,
            1.0,
            0.97493076,
            0.93666923,
            0.8545199,
            0.71678144,
            1.0,
            0.9740712,
            0.9399542,
            0.85416996,
            0.71470064,
            1.0,
            0.973276,
            0.9429993,
            0.8538406,
            0.7127714,
        ],
        dtype=np.float32,
    )

    edges, weights = build_connectivity_arrays(
        idx,
        dist,
        local_connectivity=1.0,
        bandwidth=1.5,
    )

    assert edges.shape == (n_cells * n_neighbors, 2)
    assert weights.shape == (n_cells * n_neighbors,)
    assert edges.dtype == np.uint32
    assert weights.dtype == np.float32
    assert np.all(weights > 0)
    np.testing.assert_array_equal(
        edges,
        np.column_stack(
            (
                np.repeat(np.arange(n_cells), n_neighbors),
                idx.reshape(-1),
            )
        ).astype(np.uint32),
    )
    np.testing.assert_allclose(weights, expected_weights, rtol=1e-6, atol=1e-7)


def test_connectivity_omits_zero_membership_edges():
    n_cells, n_neighbors = 10, 5
    indices = np.array(
        [
            [(row + offset + 1) % n_cells for offset in range(n_neighbors)]
            for row in range(n_cells)
        ]
    )
    distances = np.tile(
        np.array([0.0, 1.0, 1e20, 1e30, 1e35]),
        (n_cells, 1),
    )

    edges, weights = build_connectivity_arrays(
        indices,
        distances,
        local_connectivity=1.0,
        bandwidth=1.5,
    )

    expected = np.tile(
        np.array([1.0, 1.0, 1.0, 0.9512299], dtype=np.float32),
        n_cells,
    )
    assert len(edges) == n_cells * 4
    np.testing.assert_allclose(weights, expected, rtol=1e-6, atol=1e-7)


def test_wnn_integration_handles_extreme_affinities_without_runtime_warnings():
    g1, ld1, g2, ld2 = _multimodal_wnn_inputs()

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        merged = wnn_integration("RNA", g1, ld1, "ADT", g2, ld2, n_threads=1)

    assert isinstance(merged, coo_matrix)
    assert merged.shape == g1.shape
    assert merged.nnz == g1.nnz
    np.testing.assert_array_equal(
        np.bincount(merged.row, minlength=g1.shape[0]),
        np.repeat(g1.getnnz(axis=1)[0], g1.shape[0]),
    )
    assert np.all(np.isfinite(merged.data))
    assert np.all(merged.data > 0)
    assert np.all(merged.data <= 1)


def test_wnn_integration_is_invariant_to_cell_order():
    g1, _, g2, _ = _multimodal_wnn_inputs()
    rng = np.random.default_rng(42)
    ld1 = rng.normal(size=(g1.shape[0], 3))
    ld2 = rng.normal(size=(g2.shape[0], 4))
    expected = wnn_integration("RNA", g1, ld1, "ADT", g2, ld2, n_threads=1)

    permutation = np.array([5, 0, 7, 2, 6, 1, 4, 3])
    permuted = wnn_integration(
        "RNA",
        g1[permutation][:, permutation].tocsr(),
        ld1[permutation],
        "ADT",
        g2[permutation][:, permutation].tocsr(),
        ld2[permutation],
        n_threads=1,
    )
    inverse = np.argsort(permutation)
    restored = permuted.tocsr()[inverse][:, inverse]

    np.testing.assert_allclose(expected.toarray(), restored.toarray())


def test_wnn_integration_rejects_mismatched_graph_shapes():
    g1 = _simple_knn_graph(6, k=3)
    g2 = _simple_knn_graph(7, k=3)

    with pytest.raises(ValueError, match="same shape"):
        wnn_integration(
            "RNA",
            g1,
            np.zeros((6, 2)),
            "ADT",
            g2,
            np.zeros((7, 2)),
            n_threads=1,
        )


def test_wnn_integration_rejects_irregular_row_degree():
    g1 = _simple_knn_graph(6, k=3).tolil()
    g1[0, 1] = 0
    g1 = g1.tocsr()
    g1.eliminate_zeros()
    g2 = _simple_knn_graph(6, k=3)
    embeddings = np.arange(12, dtype=np.float64).reshape(6, 2)

    with pytest.raises(ValueError, match="regular row degree"):
        wnn_integration("RNA", g1, embeddings, "ADT", g2, embeddings, n_threads=1)


@pytest.mark.parametrize(
    ("embedding", "match"),
    [
        (np.zeros((5, 2)), "one row per graph cell"),
        (np.empty((6, 0)), "non-empty matrix"),
        (
            np.array(
                [[0.0, 0.0]] * 5 + [[np.nan, 0.0]],
                dtype=np.float64,
            ),
            "non-finite values",
        ),
    ],
)
def test_wnn_integration_rejects_invalid_embeddings(embedding, match):
    graph = _simple_knn_graph(6, k=3)
    valid_embedding = np.zeros((6, 2))

    with pytest.raises(ValueError, match=match):
        wnn_integration(
            "RNA",
            graph,
            embedding,
            "ADT",
            graph,
            valid_embedding,
            n_threads=1,
        )


def test_wnn_integration_uses_minimum_neighbor_count_for_mismatched_graphs():
    g1 = _simple_knn_graph(8, k=3)
    g2 = _simple_knn_graph(8, k=2)
    rng = np.random.default_rng(7)
    ld1 = rng.normal(size=(8, 3))
    ld2 = rng.normal(size=(8, 2))
    messages = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]), level="WARNING"
    )
    try:
        merged = wnn_integration("RNA", g1, ld1, "ADT", g2, ld2, n_threads=1)
        swapped = wnn_integration("ADT", g2, ld2, "RNA", g1, ld1, n_threads=1)
    finally:
        logger.remove(sink)

    assert any("different neighbor counts" in message for message in messages)
    assert merged.nnz == g1.shape[0] * 2
    assert np.all(np.isfinite(merged.data))
    np.testing.assert_allclose(merged.toarray(), swapped.toarray())
