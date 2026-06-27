import numpy as np
import pytest
from scipy.sparse import csr_matrix, coo_matrix

from scarf.knn_utils import calc_snn, merge_graphs, weight_sort_indices


def _simple_knn_graph(n: int, k: int = 3) -> csr_matrix:
    rows, cols, data = [], [], []
    for i in range(n):
        for j in range(1, k + 1):
            neighbor = (i + j) % n
            rows.append(i)
            cols.append(neighbor)
            data.append(float(j))
    return csr_matrix((data, (rows, cols)), shape=(n, n))


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
