import warnings

import numpy as np
import pytest
import zarr
from scipy.sparse import coo_matrix, csr_matrix
from zarr.storage import MemoryStore

from scarf.knn_utils import (
    _patch_null_weights,
    calc_snn,
    merge_graphs,
    self_query_knn,
    smoothen_dists,
    weight_sort_indices,
    wnn_integration,
)
from scarf.utils import logger
from scarf.writers import create_zarr_dataset


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


class _BlockSource:
    def __init__(self, blocks: list[np.ndarray]):
        self.blocks = blocks
        self.numblocks = (len(blocks), 1)
        self.shape = (sum(len(block) for block in blocks), blocks[0].shape[1])


class _SyntheticAnn:
    def __init__(
        self,
        data: _BlockSource,
        embeddings: np.ndarray | None,
        harmonized_data: _BlockSource | None,
    ):
        self.data = data
        self.embeddings = embeddings
        self.harmonizedData = harmonized_data
        self.nCells = data.shape[0]
        self.k = 2
        self.batchSize = 2
        self.queries: list[tuple[np.ndarray, np.ndarray]] = []

    @staticmethod
    def reducer(values: np.ndarray) -> np.ndarray:
        return np.asarray(values) + 10

    def transform_ann(
        self,
        values: np.ndarray,
        k: int,
        self_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        self.queries.append((np.asarray(values).copy(), self_indices.copy()))
        offsets = np.arange(1, k + 1)
        indices = (self_indices[:, None] + offsets) % self.nCells
        distances = np.broadcast_to(offsets, indices.shape).astype(np.float64)
        missed_self = int(np.count_nonzero(self_indices % 2))
        return indices, distances, missed_self


def test_self_query_knn_uses_cached_embeddings_and_writes_graph():
    raw = np.arange(10, dtype=np.float64).reshape(5, 2)
    data = _BlockSource([raw[:2], raw[2:4], raw[4:]])
    embeddings = raw + 100
    ann = _SyntheticAnn(data, embeddings=embeddings, harmonized_data=None)
    store = zarr.open_group(store=MemoryStore(), mode="w")

    recall = self_query_knn(ann, store, chunk_size=2, nthreads=1)

    expected_indices = (np.arange(5)[:, None] + np.array([1, 2], dtype=np.int64)) % 5
    assert recall == pytest.approx(60.0)
    np.testing.assert_array_equal(store["indices"][:], expected_indices)
    np.testing.assert_allclose(store["distances"][:], [[1.0, 2.0]] * 5)
    np.testing.assert_array_equal(
        np.vstack([values for values, _ in ann.queries]),
        embeddings,
    )
    np.testing.assert_array_equal(
        np.concatenate([indices for _, indices in ann.queries]),
        np.arange(5),
    )


@pytest.mark.parametrize("use_harmonized", [False, True])
def test_self_query_knn_streams_reduced_or_harmonized_blocks(use_harmonized):
    raw = np.arange(10, dtype=np.float64).reshape(5, 2)
    data = _BlockSource([raw[:2], raw[2:4], raw[4:]])
    harmonized_values = raw + 100
    harmonized_data = (
        _BlockSource(
            [
                harmonized_values[:2],
                harmonized_values[2:4],
                harmonized_values[4:],
            ]
        )
        if use_harmonized
        else None
    )
    ann = _SyntheticAnn(
        data,
        embeddings=None,
        harmonized_data=harmonized_data,
    )
    store = zarr.open_group(store=MemoryStore(), mode="w")

    recall = self_query_knn(ann, store, chunk_size=3, nthreads=1)

    expected_queries = harmonized_values if use_harmonized else raw + 10
    assert recall == pytest.approx(60.0)
    np.testing.assert_array_equal(
        np.vstack([values for values, _ in ann.queries]),
        expected_queries,
    )
    assert store["indices"].shape == (5, 2)
    assert store["distances"].shape == (5, 2)


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


def test_patch_null_weights_matches_full_rewrite(tmp_path):
    weights = np.array([0.0, 0.2, 0.0, 0.5, 0.0, 0.3], dtype=np.float64)
    null_positions = np.flatnonzero(weights == 0).tolist()
    fill = 0.15

    expected = weights.copy()
    expected[null_positions] = fill

    root = zarr.open_group(str(tmp_path / "weights.zarr"), mode="w")
    zgw = create_zarr_dataset(root, "weights", (2,), "f8", weights.shape)
    zgw[:] = weights
    _patch_null_weights(zgw, null_positions, fill, patch_chunk=2)
    np.testing.assert_allclose(zgw[:], expected)


def test_smoothen_dists_runs(tmp_path):
    pytest.importorskip("umap")
    n_cells, n_neighbors = 24, 5
    chunk_size = 8
    rng = np.random.default_rng(0)
    dist = rng.random((n_cells, n_neighbors)).astype(np.float64)
    dist[:, 0] = 0.0
    idx = np.tile(np.arange(n_cells), (n_cells, 1)) % n_cells

    root = zarr.open_group(str(tmp_path / "graph.zarr"), mode="w")
    knn = root.create_group("knn")
    z_idx = create_zarr_dataset(knn, "indices", (chunk_size,), "u8", idx.shape)
    z_dist = create_zarr_dataset(knn, "distances", (chunk_size,), "f8", dist.shape)
    z_idx[:] = idx
    z_dist[:] = dist
    graph = root.create_group("graph")
    smoothen_dists(graph, z_idx, z_dist, lc=1.0, bw=1.5, chunk_size=chunk_size)
    assert graph["weights"].shape[0] == graph["edges"].shape[0]
    assert graph["weights"].shape[0] > 0


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
