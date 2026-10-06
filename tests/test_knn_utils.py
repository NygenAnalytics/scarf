import functools
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.metrics import adjusted_rand_score

from scarf.clustering.leiden import leiden_membership
from scarf.neighbors.graph import (
    build_connectivity_arrays,
    calc_snn,
    merge_graphs,
    take_nearest_per_row,
    weight_sort_indices,
)
from scarf.neighbors.diffusion import bounded_diffusion_operator, transition_matrix
from scarf.neighbors.integration import _wnn_integration_many
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


def _simple_knn_indices(n: int, k: int = 3) -> np.ndarray:
    return np.asarray(
        [[(cell + offset + 1) % n for offset in range(k)] for cell in range(n)],
        dtype=np.int64,
    )


def _grouped_knn_indices(groups: list[list[int]]) -> np.ndarray:
    graph = _grouped_knn_graph(groups)
    degree = int(graph.getnnz(axis=1)[0])
    return graph.indices.reshape(graph.shape[0], degree)


def _wnn_pair(
    name1: str,
    indices1: np.ndarray,
    ld1: np.ndarray,
    name2: str,
    indices2: np.ndarray,
    ld2: np.ndarray,
    nthreads: int,
    *,
    l2_normalize: bool = True,
) -> tuple[coo_matrix, np.ndarray]:
    return _wnn_integration_many(
        [(name1, indices1, ld1), (name2, indices2, ld2)],
        nthreads,
        l2_normalize=l2_normalize,
    )


@pytest.mark.parametrize("backend", ["igraph", "leidenalg"])
def test_leiden_membership_preserves_disconnected_partitions(backend):
    graph = _grouped_knn_graph([[0, 1, 2, 3], [4, 5, 6, 7]])

    actual = leiden_membership(
        graph,
        resolution=1.0,
        random_seed=4444,
        backend=backend,
    )

    assert adjusted_rand_score([1, 1, 1, 1, 2, 2, 2, 2], actual) == pytest.approx(1.0)


@pytest.mark.parametrize("backend", ["igraph", "leidenalg"])
def test_leiden_membership_uses_edge_weights(backend):
    weights = np.full((8, 8), 0.001)
    weights[:4, :4] = 1.0
    weights[4:, 4:] = 1.0
    np.fill_diagonal(weights, 0)

    actual = leiden_membership(csr_matrix(weights), 1.0, 11, backend)

    assert adjusted_rand_score([0, 0, 0, 0, 1, 1, 1, 1], actual) == 1.0


def test_native_leiden_membership_is_seeded_and_repeatable():
    graph = _simple_knn_graph(100)

    first = leiden_membership(graph, resolution=1.0, random_seed=4444)
    second = leiden_membership(graph, resolution=1.0, random_seed=4444)

    np.testing.assert_array_equal(second, first)


def test_leiden_restores_the_seedable_igraph_generator():
    import random

    import igraph

    def seeded_edges() -> list[tuple[int, int]]:
        random.seed(7)
        return igraph.Graph.Erdos_Renyi(n=30, p=0.2).get_edgelist()

    igraph.set_random_number_generator(random)
    expected = seeded_edges()
    leiden_membership(_simple_knn_graph(12), resolution=1.0, random_seed=1)

    assert seeded_edges() == expected


def test_leiden_membership_rejects_unknown_backend():
    graph = _simple_knn_graph(10)

    with pytest.raises(ValueError, match="backend"):
        leiden_membership(
            graph,
            resolution=1.0,
            random_seed=4444,
            backend="unknown",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("backend", ["igraph", "leidenalg"])
def test_leiden_ignores_explicit_zero_weight_edges(backend):
    solid = _grouped_knn_graph([[0, 1, 2, 3], [4, 5, 6, 7]]).tocoo()
    padded = coo_matrix(
        (
            np.concatenate([solid.data, np.zeros(4)]),
            (
                np.concatenate([solid.row, np.array([0, 1, 2, 3])]),
                np.concatenate([solid.col, np.array([4, 5, 6, 7])]),
            ),
        ),
        shape=solid.shape,
    )

    assert np.count_nonzero(padded.data) != padded.nnz

    actual = leiden_membership(
        padded, resolution=1.0, random_seed=4444, backend=backend
    )
    expected = leiden_membership(
        solid, resolution=1.0, random_seed=4444, backend=backend
    )

    np.testing.assert_array_equal(actual, expected)


def test_igraph_membership_requires_igraph(monkeypatch):
    monkeypatch.setitem(sys.modules, "igraph", None)
    graph = _simple_knn_graph(10)

    with pytest.raises(ImportError, match="igraph"):
        leiden_membership(graph, resolution=1.0, random_seed=4444)


def test_leidenalg_membership_requires_leidenalg(monkeypatch):
    monkeypatch.setitem(sys.modules, "leidenalg", None)
    graph = _simple_knn_graph(10)

    with pytest.raises(ImportError, match="leidenalg"):
        leiden_membership(
            graph,
            resolution=1.0,
            random_seed=4444,
            backend="leidenalg",
        )


def _powered_transition(graph: csr_matrix, power: int) -> coo_matrix:
    """Unbounded reference: SciPy's power of the row-normalized graph."""
    inverse_degree = np.ravel(graph.sum(axis=1))
    inverse_degree[inverse_degree != 0] = 1 / inverse_degree[inverse_degree != 0]
    n_cells = graph.shape[0]
    diagonal = csr_matrix(
        (inverse_degree, (range(n_cells), range(n_cells))),
        shape=[n_cells, n_cells],
    )
    return (diagonal.dot(graph) ** power).tocoo()


def test_transition_matrix_and_its_power_row_normalize_the_graph():
    graph = csr_matrix(
        [
            [0.0, 2.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    transition = transition_matrix(graph)
    squared = bounded_diffusion_operator(graph, 2, memory_bytes=1024**2)

    assert isinstance(transition, csr_matrix)
    assert transition.nnz == graph.nnz
    np.testing.assert_allclose(
        transition.toarray(),
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.5, 0.0, 0.5, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        squared.toarray(),
        [
            [0.5, 0.0, 0.5, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.5, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        rtol=0,
        atol=1e-12,
    )


def test_bounded_diffusion_operator_matches_unbounded_power_and_checks_budget():
    graph = _simple_knn_graph(40, k=3)
    graph = (graph + graph.T).tocsr()
    for power in range(1, 7):
        expected = _powered_transition(graph, power)
        actual = bounded_diffusion_operator(graph, power, memory_bytes=1024**3).tocoo()
        # Entries match exactly and in storage order, so persisted payloads
        # and their fingerprints are unchanged.
        np.testing.assert_array_equal(actual.row, expected.row)
        np.testing.assert_array_equal(actual.col, expected.col)
        np.testing.assert_array_equal(actual.data, expected.data)

    graph_bytes = graph.data.nbytes + graph.indices.nbytes + graph.indptr.nbytes
    with pytest.raises(MemoryError, match="Diffusion step"):
        bounded_diffusion_operator(graph, 6, memory_bytes=4 * graph_bytes)
    with pytest.raises(ValueError, match="positive integer"):
        bounded_diffusion_operator(graph, 0, memory_bytes=1024**3)


def test_bounded_diffusion_uses_exact_count_when_upper_bound_exceeds_budget():
    graph = csr_matrix(
        [
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    # Repeated paths give an upper bound of twelve entries but only six
    # distinct outputs. The exact product and working arrays need 340 bytes.
    actual = bounded_diffusion_operator(graph, 2, memory_bytes=341)

    np.testing.assert_array_equal(
        actual.toarray(), _powered_transition(graph, 2).toarray()
    )
    np.testing.assert_array_equal(np.asarray(actual.sum(axis=1)).ravel(), [1, 1, 1, 0])
    with pytest.raises(MemoryError, match="6 operator entries.*340 bytes"):
        bounded_diffusion_operator(graph, 2, memory_bytes=340)


@pytest.mark.parametrize("power", [True, 1.5, "2"])
def test_bounded_diffusion_rejects_noninteger_powers(power):
    with pytest.raises(TypeError, match="positive integer"):
        bounded_diffusion_operator(_simple_knn_graph(4), power, memory_bytes=1024**2)


def _multimodal_wnn_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices1 = _grouped_knn_indices([[0, 1, 2, 3], [4, 5, 6, 7]])
    indices2 = _grouped_knn_indices([[0, 2, 4, 6], [1, 3, 5, 7]])
    ld1 = np.array(
        [
            [0.0, 1.0, 0.2],
            [0.1, 0.9, 0.3],
            [-0.1, 1.1, 0.1],
            [0.2, 0.8, 0.4],
            [3.0, -1.0, 0.0],
            [3.1, -0.9, 0.1],
            [2.9, -1.1, -0.1],
            [3.2, -0.8, 0.2],
        ],
        dtype=np.float64,
    )
    ld2 = np.array(
        [
            [1.0, 0.0],
            [-1.0, 3.0],
            [0.9, 0.1],
            [-0.9, 3.1],
            [1.1, -0.1],
            [-1.1, 2.9],
            [0.8, 0.2],
            [-0.8, 3.2],
        ],
        dtype=np.float64,
    )
    return indices1, ld1, indices2, ld2


def _reference_wnn(
    indices1: np.ndarray,
    ld1: np.ndarray,
    indices2: np.ndarray,
    ld2: np.ndarray,
    *,
    l2_normalize: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def normalize(values: np.ndarray) -> np.ndarray:
        output = np.asarray(values, dtype=np.float64).copy()
        if not l2_normalize:
            return output
        norms = np.linalg.norm(output, axis=1)
        np.divide(
            output,
            norms[:, np.newaxis],
            out=output,
            where=norms[:, np.newaxis] > 0,
        )
        return output

    def kernel(
        distances: np.ndarray,
        nearest: float,
        bandwidth: float,
    ) -> np.ndarray:
        adjusted = np.maximum(distances - nearest, 0)
        tolerance = 8.0 * np.finfo(np.float64).eps * nearest
        if bandwidth <= tolerance:
            return (adjusted <= tolerance).astype(np.float64)
        return np.exp(-(adjusted / bandwidth))

    embedding1 = normalize(ld1)
    embedding2 = normalize(ld2)
    n_cells = len(indices1)
    output_k = min(indices1.shape[1], indices2.shape[1])
    selected_indices = np.empty((n_cells, output_k), dtype=np.int64)
    selected_affinities = np.empty((n_cells, output_k), dtype=np.float64)
    modality_weights = np.empty((n_cells, 2), dtype=np.float64)

    for cell in range(n_cells):
        neighbors1 = indices1[cell]
        neighbors2 = indices2[cell]
        candidates = np.union1d(neighbors1, neighbors2)
        candidate1 = embedding1[candidates]
        candidate2 = embedding2[candidates]
        point1 = embedding1[cell]
        point2 = embedding2[cell]
        distances1 = np.linalg.norm(point1 - candidate1, axis=1)
        distances2 = np.linalg.norm(point2 - candidate2, axis=1)
        positions1 = np.searchsorted(candidates, neighbors1)
        positions2 = np.searchsorted(candidates, neighbors2)
        own1 = np.sort(distances1[positions1])
        own2 = np.sort(distances2[positions2])
        nearest1, nearest2 = float(own1[0]), float(own2[0])
        bandwidth1 = float(own1[-1] - nearest1)
        bandwidth2 = float(own2[-1] - nearest2)

        within1 = kernel(
            np.asarray([np.linalg.norm(point1 - candidate1[positions1].mean(axis=0))]),
            nearest1,
            bandwidth1,
        )[0]
        cross1 = kernel(
            np.asarray([np.linalg.norm(point1 - candidate1[positions2].mean(axis=0))]),
            nearest1,
            bandwidth1,
        )[0]
        within2 = kernel(
            np.asarray([np.linalg.norm(point2 - candidate2[positions2].mean(axis=0))]),
            nearest2,
            bandwidth2,
        )[0]
        cross2 = kernel(
            np.asarray([np.linalg.norm(point2 - candidate2[positions1].mean(axis=0))]),
            nearest2,
            bandwidth2,
        )[0]
        score1 = np.clip(within1 / (cross1 + 1e-4), 0, 200)
        score2 = np.clip(within2 / (cross2 + 1e-4), 0, 200)
        score_max = max(score1, score2)
        exp1, exp2 = np.exp(score1 - score_max), np.exp(score2 - score_max)
        weight1 = exp1 / (exp1 + exp2)
        weight2 = exp2 / (exp1 + exp2)
        modality_weights[cell] = (weight1, weight2)

        affinities = weight1 * kernel(
            distances1,
            nearest1,
            bandwidth1,
        ) + weight2 * kernel(
            distances2,
            nearest2,
            bandwidth2,
        )
        selected = np.lexsort((candidates, -affinities))[:output_k]
        selected_indices[cell] = candidates[selected]
        selected_affinities[cell] = affinities[selected]

    return selected_indices, selected_affinities, modality_weights


def _reference_wnn_many(
    modalities: list[tuple[str, np.ndarray, np.ndarray]],
    *,
    l2_normalize: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def normalize(values: np.ndarray) -> np.ndarray:
        output = np.asarray(values, dtype=np.float64).copy()
        if not l2_normalize:
            return output
        norms = np.linalg.norm(output, axis=1)
        np.divide(
            output,
            norms[:, np.newaxis],
            out=output,
            where=norms[:, np.newaxis] > 0,
        )
        return output

    def kernel(
        distances: np.ndarray,
        nearest: float,
        bandwidth: float,
    ) -> np.ndarray:
        adjusted = np.maximum(distances - nearest, 0)
        tolerance = 8.0 * np.finfo(np.float64).eps * nearest
        if bandwidth <= tolerance:
            return (adjusted <= tolerance).astype(np.float64)
        return np.exp(-(adjusted / bandwidth))

    indices = [np.asarray(modality[1]) for modality in modalities]
    embeddings = [normalize(modality[2]) for modality in modalities]
    n_cells = indices[0].shape[0]
    output_k = min(values.shape[1] for values in indices)
    selected_indices = np.empty((n_cells, output_k), dtype=np.int64)
    selected_affinities = np.empty((n_cells, output_k), dtype=np.float64)
    modality_weights = np.empty((n_cells, len(modalities)), dtype=np.float64)

    for cell in range(n_cells):
        neighbor_rows = [values[cell] for values in indices]
        candidates = np.unique(np.concatenate(neighbor_rows))
        positions = [
            np.searchsorted(candidates, neighbors) for neighbors in neighbor_rows
        ]
        candidate_embeddings = [values[candidates] for values in embeddings]
        points = [values[cell] for values in embeddings]
        distances = [
            np.linalg.norm(point - candidate, axis=1)
            for point, candidate in zip(points, candidate_embeddings, strict=True)
        ]
        own_distances = [
            np.sort(values[own_positions])
            for values, own_positions in zip(distances, positions, strict=True)
        ]
        nearest = [float(values[0]) for values in own_distances]
        bandwidths = [float(values[-1] - values[0]) for values in own_distances]

        within = [
            kernel(
                np.asarray(
                    [np.linalg.norm(point - candidate[own_positions].mean(axis=0))]
                ),
                nearest_distance,
                bandwidth,
            )[0]
            for point, candidate, own_positions, nearest_distance, bandwidth in zip(
                points,
                candidate_embeddings,
                positions,
                nearest,
                bandwidths,
                strict=True,
            )
        ]
        directed_scores = np.full(
            (len(modalities), len(modalities)),
            -np.inf,
            dtype=np.float64,
        )
        for target, (
            point,
            candidate,
            nearest_distance,
            bandwidth,
            within_affinity,
        ) in enumerate(
            zip(
                points,
                candidate_embeddings,
                nearest,
                bandwidths,
                within,
                strict=True,
            )
        ):
            for source, source_positions in enumerate(positions):
                if source == target:
                    continue
                cross = kernel(
                    np.asarray(
                        [
                            np.linalg.norm(
                                point - candidate[source_positions].mean(axis=0)
                            )
                        ]
                    ),
                    nearest_distance,
                    bandwidth,
                )[0]
                directed_scores[target, source] = np.clip(
                    within_affinity / (cross + 1e-4),
                    0,
                    200,
                )

        finite = np.isfinite(directed_scores)
        shifted = directed_scores[finite] - directed_scores[finite].max()
        strengths = np.zeros(directed_scores.shape, dtype=np.float64)
        strengths[finite] = np.exp(shifted)
        weights = strengths.sum(axis=1)
        weights /= weights.sum()
        modality_weights[cell] = weights

        affinity = weights[0] * kernel(distances[0], nearest[0], bandwidths[0])
        for weight, values, nearest_distance, bandwidth in zip(
            weights[1:],
            distances[1:],
            nearest[1:],
            bandwidths[1:],
            strict=True,
        ):
            affinity += weight * kernel(values, nearest_distance, bandwidth)
        selected = np.lexsort((candidates, -affinity))[:output_k]
        selected_indices[cell] = candidates[selected]
        selected_affinities[cell] = affinity[selected]

    return selected_indices, selected_affinities, modality_weights


def _three_way_wnn_inputs() -> list[tuple[str, np.ndarray, np.ndarray]]:
    indices1, ld1, indices2, ld2 = _multimodal_wnn_inputs()
    indices3 = _grouped_knn_indices([[0, 1, 4, 5], [2, 3, 6, 7]])
    ld3 = np.array(
        [
            [0.0, 0.1, 2.0, -0.2],
            [0.1, 0.0, 1.9, -0.1],
            [2.1, -0.2, 0.1, 0.0],
            [1.9, -0.1, 0.0, 0.1],
            [0.2, 0.2, 2.2, -0.3],
            [0.0, -0.1, 1.8, -0.2],
            [2.2, -0.3, 0.2, 0.0],
            [1.8, 0.0, -0.1, 0.2],
        ],
        dtype=np.float64,
    )
    return [
        ("RNA", indices1, ld1),
        ("ATAC", indices2, ld2),
        ("ADT", indices3, ld3),
    ]


@pytest.mark.parametrize("kernel", [calc_snn, calc_snn.py_func], ids=["jit", "python"])
def test_calc_snn_returns_normalized_overlap(kernel):
    # Cell i's neighbors are i+1, i+2, and i+3 (mod 6), so a neighbor at offset
    # d shares 3 - d of them: (2, 1, 0) shared over k - 1 = 2.
    indices = _simple_knn_indices(6, k=3)

    snn = kernel(indices)

    np.testing.assert_array_equal(snn, np.tile([1.0, 0.5, 0.0], (6, 1)))
    # The overlap does not depend on the order of a cell's neighbors.
    np.testing.assert_array_equal(kernel(indices[:, ::-1]), snn[:, ::-1])


def test_weight_sort_indices_keeps_top_neighbors():
    indices = np.array([4, 1, 2, 1, 3])
    weights = np.array([0.2, 0.5, 0.4, 0.6, 0.1])
    sort_weights = weights + np.array([0.0, 0.2, 0.1, 0.2, 0.0])
    kept_idx, kept_w = weight_sort_indices(indices, weights, sort_weights, n=3)
    # Sort weights rank the entries 1 (0.8), 1 (0.7), 2 (0.5), 4 (0.2), 3 (0.1).
    # The duplicate of neighbor 1 keeps its higher-ranked weight.
    np.testing.assert_array_equal(kept_idx, [1, 2, 4])
    np.testing.assert_allclose(kept_w, [0.6, 0.4, 0.2])


def test_merging_a_graph_with_itself_returns_the_graph():
    graph = _simple_knn_graph(8, k=3)

    merged = merge_graphs([graph, graph.copy()])

    assert isinstance(merged, coo_matrix)
    assert merged.shape == graph.shape
    assert merged.nnz == graph.nnz
    np.testing.assert_array_equal(merged.toarray(), graph.toarray())


def _reference_merged_rows(
    graphs: list[csr_matrix],
    n_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge graphs row by row with Python sets and dictionaries.

    Each candidate edge is ranked by its weight plus the fraction of the source
    cell's neighbors it shares in that graph. A candidate keeps the weight of
    its best-ranked copy, and each row keeps its ``n_neighbors`` best ranks.
    """
    neighbor_sets = [
        [set(graph[row].indices.tolist()) for row in range(graph.shape[0])]
        for graph in graphs
    ]
    columns: list[int] = []
    weights: list[float] = []
    for row in range(graphs[0].shape[0]):
        best: dict[int, tuple[float, float]] = {}
        for graph, sets in zip(graphs, neighbor_sets, strict=True):
            for neighbor, weight in zip(
                graph[row].indices, graph[row].data, strict=True
            ):
                shared = len(sets[row] & sets[neighbor]) / (n_neighbors - 1)
                rank = float(weight) + shared
                if neighbor not in best or rank > best[neighbor][0]:
                    best[int(neighbor)] = (rank, float(weight))
        ordered = sorted(best.items(), key=lambda item: -item[1][0])[:n_neighbors]
        columns.extend(neighbor for neighbor, _ in ordered)
        weights.extend(weight for _, (_, weight) in ordered)
    return np.asarray(columns), np.asarray(weights, dtype=np.float32)


def test_merge_graphs_matches_row_wise_reference_and_keeps_dtypes():
    rng = np.random.default_rng(5)
    n_cells, n_neighbors = 40, 4
    graphs = []
    for _ in range(2):
        columns = np.stack(
            [
                rng.choice(
                    np.delete(np.arange(n_cells), cell),
                    size=n_neighbors,
                    replace=False,
                )
                for cell in range(n_cells)
            ]
        )
        weights = rng.random((n_cells, n_neighbors)).astype(np.float32)
        graphs.append(
            csr_matrix(
                (
                    weights.ravel(),
                    (np.repeat(np.arange(n_cells), n_neighbors), columns.ravel()),
                ),
                shape=(n_cells, n_cells),
            )
        )
    expected_columns, expected_weights = _reference_merged_rows(graphs, n_neighbors)

    merged = merge_graphs(graphs)

    np.testing.assert_array_equal(
        merged.row,
        np.repeat(np.arange(n_cells), n_neighbors),
    )
    np.testing.assert_array_equal(merged.col, expected_columns)
    np.testing.assert_array_equal(merged.data, expected_weights)
    assert merged.data.dtype == np.float32


def test_merge_graphs_rejects_mismatched_shapes():
    g1 = _simple_knn_graph(6, k=3)
    g2 = _simple_knn_graph(8, k=3)
    with pytest.raises(ValueError, match="same shape"):
        merge_graphs([g1, g2])


def test_merge_graphs_rejects_one_neighbor_snn_input():
    graph = _simple_knn_graph(6, k=1)

    with pytest.raises(ValueError, match="at least two neighbors"):
        merge_graphs([graph, graph.copy()])


def test_merge_graphs_rejects_empty_and_irregular_inputs():
    regular = _simple_knn_graph(6, k=3)
    irregular = regular.tolil()
    irregular[0, 1] = 0
    irregular = irregular.tocsr()
    irregular.eliminate_zeros()

    with pytest.raises(ValueError, match="At least one graph is required"):
        merge_graphs([])
    with pytest.raises(ValueError, match="regular neighbor count"):
        merge_graphs([regular, irregular])
    with pytest.raises(ValueError, match="regular neighbor count"):
        merge_graphs([csr_matrix((0, 0)), csr_matrix((0, 0))])
    with pytest.raises(ValueError, match="same number of edges"):
        merge_graphs([regular, _simple_knn_graph(6, k=2)])


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
    # umap-learn's weights for these rows with each cell prepended at distance
    # zero, as fuzzy_simplicial_set computes them for n_neighbors=6.
    expected_weights = np.array(
        [
            1.0,
            0.9449202,
            0.8201305,
            0.6726141,
            0.43977392,
            1.0,
            0.9420736,
            0.8293171,
            0.67118084,
            0.4348761,
            1.0,
            0.93945545,
            0.8378171,
            0.669816,
            0.43035665,
            1.0,
            0.93703866,
            0.84570956,
            0.668519,
            0.42617577,
            1.0,
            0.93480074,
            0.85306203,
            0.6672895,
            0.422301,
            1.0,
            0.93272024,
            0.8599265,
            0.6661159,
            0.41868982,
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
    # The UMAP kernel over a row that starts with the cell itself: the nearest
    # neighbor (rho) gets weight one and the others exp(-(d - rho) / sigma),
    # with one sigma per cell chosen so that all k weights sum to
    # log2(k + 1) * bandwidth.
    rows = weights.reshape(n_cells, n_neighbors).astype(np.float64)
    np.testing.assert_array_equal(rows[:, 0], 1.0)
    np.testing.assert_allclose(
        rows.sum(axis=1), np.log2(n_neighbors + 1) * 1.5, rtol=1e-4
    )
    sigmas = -(dist[:, 1:] - dist[:, :1]) / np.log(rows[:, 1:])
    np.testing.assert_allclose(
        sigmas, np.broadcast_to(sigmas[:, :1], sigmas.shape), rtol=1e-4
    )
    # Distances may arrive in any memory layout.
    fortran_edges, fortran_weights = build_connectivity_arrays(
        idx,
        np.asfortranarray(dist),
        local_connectivity=1.0,
        bandwidth=1.5,
    )
    np.testing.assert_array_equal(fortran_edges, edges)
    np.testing.assert_array_equal(fortran_weights, weights)


@pytest.mark.parametrize(
    ("indices", "distances", "error", "message"),
    [
        (np.arange(4), np.ones(4), ValueError, "matching matrices"),
        (
            _simple_knn_indices(4, k=2),
            np.ones((4, 3)),
            ValueError,
            "matching matrices",
        ),
        (
            _simple_knn_indices(4, k=2).astype(np.float64),
            np.ones((4, 2)),
            TypeError,
            "must be integers",
        ),
        (
            _simple_knn_indices(4, k=2) - 1,
            np.ones((4, 2)),
            ValueError,
            "outside the cell range",
        ),
        (
            _simple_knn_indices(4, k=2) + 2,
            np.ones((4, 2)),
            ValueError,
            "outside the cell range",
        ),
        (
            _simple_knn_indices(4, k=2),
            np.array([[1.0, np.inf]] * 4),
            ValueError,
            "finite and non-negative",
        ),
        (
            _simple_knn_indices(4, k=2),
            np.array([[1.0, -0.5]] * 4),
            ValueError,
            "finite and non-negative",
        ),
    ],
    ids=[
        "one_dimensional",
        "shape_mismatch",
        "float_indices",
        "negative_index",
        "index_past_last_cell",
        "infinite_distance",
        "negative_distance",
    ],
)
def test_build_connectivity_arrays_rejects_invalid_neighbor_matrices(
    indices, distances, error, message
):
    with pytest.raises(error, match=message):
        build_connectivity_arrays(
            indices,
            distances,
            local_connectivity=1.0,
            bandwidth=1.5,
        )


@pytest.mark.parametrize(
    ("membership", "message"),
    [
        (
            (np.zeros(7, dtype=np.int64), np.ones(8, dtype=np.int64), np.ones(8)),
            "does not match the KNN matrix",
        ),
        (
            (np.zeros(8, dtype=np.int64), np.full(8, -1, dtype=np.int64), np.ones(8)),
            "exceed uint32 bounds",
        ),
        (
            (
                np.zeros(8, dtype=np.int64),
                np.full(8, 2**32, dtype=np.int64),
                np.ones(8),
            ),
            "exceed uint32 bounds",
        ),
        (
            (
                np.zeros(8, dtype=np.int64),
                np.ones(8, dtype=np.int64),
                np.full(8, np.nan),
            ),
            "weights must be finite",
        ),
    ],
    ids=["short_output", "negative_column", "column_past_uint32", "nan_weight"],
)
def test_build_connectivity_arrays_rejects_inconsistent_membership_output(
    monkeypatch, membership, message
):
    # Guards against a umap-learn release that changes the membership output.
    monkeypatch.setattr(
        "scarf.neighbors.graph.smooth_knn_chunk",
        lambda *_args, **_kwargs: membership,
    )

    with pytest.raises(ValueError, match=message):
        build_connectivity_arrays(
            _simple_knn_indices(4, k=2),
            np.ones((4, 2)),
            local_connectivity=1.0,
            bandwidth=1.5,
        )


def test_connectivity_preserves_zero_weight_neighbors():
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
        np.array([1.0, 1.0, 1.0, 0.94176507, 0.0], dtype=np.float32),
        n_cells,
    )
    np.testing.assert_array_equal(
        edges,
        np.column_stack((np.repeat(np.arange(n_cells), n_neighbors), indices.ravel())),
    )
    np.testing.assert_allclose(weights, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.slow
def test_connectivity_matches_umap_fuzzy_simplicial_set():
    from umap.umap_ import fuzzy_simplicial_set

    n_cells, k = 500, 11
    coordinates = np.random.default_rng(0).normal(size=(n_cells, 8))
    squared = ((coordinates[:, None, :] - coordinates[None, :, :]) ** 2).sum(axis=2)
    # umap-learn's KNN rows start with the cell itself at distance zero.
    with_self = np.argsort(squared, axis=1, kind="stable")[:, : k + 1]
    np.testing.assert_array_equal(with_self[:, 0], np.arange(n_cells))
    distances = np.sqrt(np.take_along_axis(squared, with_self, axis=1)).astype(
        np.float32
    )
    expected, _sigmas, _rhos = fuzzy_simplicial_set(
        coordinates,
        n_neighbors=k + 1,
        random_state=None,
        metric="euclidean",
        knn_indices=with_self,
        knn_dists=distances,
        local_connectivity=1.0,
        apply_set_operations=False,
    )

    # Scarf's rows hold the k other cells only.
    edges, weights = build_connectivity_arrays(
        with_self[:, 1:],
        distances[:, 1:],
        local_connectivity=1.0,
        bandwidth=1.0,
    )

    observed = coo_matrix(
        (weights, (edges[:, 0], edges[:, 1])), shape=(n_cells, n_cells)
    )
    np.testing.assert_array_equal(observed.toarray(), expected.toarray())


def test_take_nearest_per_row_handles_rows_that_lost_zero_weight_edges():
    # Row 1 kept two edges instead of three because a zero-weight edge was
    # dropped. Assuming a fixed row width here selects the wrong neighbors.
    edges = np.array(
        [[0, 5], [0, 6], [0, 7], [1, 8], [1, 9], [2, 1], [2, 2], [2, 3]],
        dtype=np.uint32,
    )
    weights = np.arange(1, 9, dtype=np.float32)

    kept_weights, kept_edges = take_nearest_per_row(weights, edges, 3, 2)

    np.testing.assert_array_equal(kept_edges[:, 1], [5, 6, 8, 9, 1, 2])
    np.testing.assert_allclose(kept_weights, [1.0, 2.0, 4.0, 5.0, 6.0, 7.0])

    full_weights, full_edges = take_nearest_per_row(weights, edges, 3, 3)
    np.testing.assert_array_equal(full_edges, edges)
    np.testing.assert_allclose(full_weights, weights)

    with pytest.raises(ValueError, match="grouped by source cell"):
        take_nearest_per_row(
            np.ones(2, dtype=np.float32),
            np.array([[1, 0], [0, 1]], dtype=np.uint32),
            2,
            1,
        )


def test_take_nearest_per_row_keeps_every_cell_when_a_row_is_empty():
    edges = np.array([[0, 1], [0, 2], [2, 0]], dtype=np.uint32)
    weights = np.array([0.5, 0.25, 0.75], dtype=np.float32)

    kept_weights, kept_edges = take_nearest_per_row(weights, edges, 3, 1)

    np.testing.assert_array_equal(kept_edges[:, 0], [0, 2])
    np.testing.assert_allclose(kept_weights, [0.5, 0.75])


def test_wnn_integration_is_invariant_to_cell_order():
    indices1, _, indices2, _ = _multimodal_wnn_inputs()
    rng = np.random.default_rng(42)
    ld1 = rng.normal(size=(len(indices1), 3))
    ld2 = rng.normal(size=(len(indices2), 4))
    expected, expected_weights = _wnn_pair(
        "RNA",
        indices1,
        ld1,
        "ADT",
        indices2,
        ld2,
        nthreads=1,
    )

    permutation = np.array([5, 0, 7, 2, 6, 1, 4, 3])
    old_to_new = np.argsort(permutation)
    permuted, permuted_weights = _wnn_pair(
        "RNA",
        old_to_new[indices1[permutation]],
        ld1[permutation],
        "ADT",
        old_to_new[indices2[permutation]],
        ld2[permutation],
        nthreads=1,
    )
    inverse = np.argsort(permutation)
    restored = permuted.tocsr()[inverse][:, inverse]

    np.testing.assert_allclose(expected.toarray(), restored.toarray())
    np.testing.assert_allclose(expected_weights, permuted_weights[inverse])


def test_wnn_integration_rejects_mismatched_neighbor_rows():
    indices1 = _simple_knn_indices(6, k=3)
    indices2 = _simple_knn_indices(7, k=3)

    with pytest.raises(ValueError, match="same number of cells"):
        _wnn_pair(
            "RNA",
            indices1,
            np.zeros((6, 2)),
            "ADT",
            indices2,
            np.zeros((7, 2)),
            nthreads=1,
        )


@pytest.mark.parametrize(
    ("indices", "error", "match"),
    [
        (np.arange(6), ValueError, "non-empty matrix"),
        (np.zeros((6, 3), dtype=np.float64), TypeError, "integer indices"),
        (
            np.array(
                [
                    [1, 1, 2],
                    [0, 2, 3],
                    [0, 1, 3],
                    [0, 1, 2],
                    [0, 1, 2],
                    [0, 1, 2],
                ]
            ),
            ValueError,
            "unique within each row",
        ),
        (
            np.array(
                [
                    [0, 1, 2],
                    [0, 2, 3],
                    [0, 1, 3],
                    [0, 1, 2],
                    [0, 1, 2],
                    [0, 1, 2],
                ]
            ),
            ValueError,
            "exclude self",
        ),
        (
            np.array(
                [
                    [1, 2, 6],
                    [0, 2, 3],
                    [0, 1, 3],
                    [0, 1, 2],
                    [0, 1, 2],
                    [0, 1, 2],
                ]
            ),
            ValueError,
            "outside cell range",
        ),
    ],
)
def test_wnn_integration_rejects_invalid_neighbor_matrices(indices, error, match):
    valid = _simple_knn_indices(6, k=3)
    embeddings = np.arange(12, dtype=np.float64).reshape(6, 2)

    with pytest.raises(error, match=match):
        _wnn_pair(
            "RNA",
            indices,
            embeddings,
            "ADT",
            valid,
            embeddings,
            nthreads=1,
        )


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
    indices = _simple_knn_indices(6, k=3)
    valid_embedding = np.zeros((6, 2))

    with pytest.raises(ValueError, match=match):
        _wnn_pair(
            "RNA",
            indices,
            embedding,
            "ADT",
            indices,
            valid_embedding,
            nthreads=1,
        )


def test_wnn_integration_uses_minimum_neighbor_count_for_mismatched_graphs():
    indices1 = _simple_knn_indices(8, k=3)
    indices2 = _simple_knn_indices(8, k=2)
    rng = np.random.default_rng(7)
    ld1 = rng.normal(size=(8, 3))
    ld2 = rng.normal(size=(8, 2))
    messages = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]), level="WARNING"
    )
    try:
        merged, modality_weights = _wnn_pair(
            "RNA",
            indices1,
            ld1,
            "ADT",
            indices2,
            ld2,
            nthreads=1,
        )
        swapped, swapped_weights = _wnn_pair(
            "ADT",
            indices2,
            ld2,
            "RNA",
            indices1,
            ld1,
            nthreads=1,
        )
    finally:
        logger.remove(sink)

    assert any("different neighbor counts" in message for message in messages)
    assert merged.nnz == len(indices1) * 2
    expected_indices, expected_affinities, expected_weights = _reference_wnn(
        indices1, ld1, indices2, ld2, l2_normalize=True
    )
    np.testing.assert_array_equal(
        merged.col.reshape(expected_indices.shape), expected_indices
    )
    np.testing.assert_allclose(
        merged.data.reshape(expected_affinities.shape),
        expected_affinities,
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(modality_weights, expected_weights, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(merged.toarray(), swapped.toarray())
    np.testing.assert_allclose(modality_weights, swapped_weights[:, ::-1])


@pytest.mark.parametrize("l2_normalize", [True, False])
@pytest.mark.parametrize(("modality", "scale"), [(1, 1e8), (2, 1e-7)])
def test_wnn_integration_is_invariant_to_per_modality_scale(
    l2_normalize,
    modality,
    scale,
):
    indices1, ld1, indices2, ld2 = _multimodal_wnn_inputs()
    expected, expected_weights = _wnn_pair(
        "RNA",
        indices1,
        ld1,
        "ADT",
        indices2,
        ld2,
        nthreads=1,
        l2_normalize=l2_normalize,
    )
    if modality == 1:
        ld1 = ld1 * scale
    else:
        ld2 = ld2 * scale

    actual, actual_weights = _wnn_pair(
        "RNA",
        indices1,
        ld1,
        "ADT",
        indices2,
        ld2,
        nthreads=1,
        l2_normalize=l2_normalize,
    )

    np.testing.assert_array_equal(actual.row, expected.row)
    np.testing.assert_array_equal(actual.col, expected.col)
    np.testing.assert_allclose(actual.data, expected.data, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(
        actual_weights,
        expected_weights,
        rtol=2e-6,
        atol=1e-7,
    )


def test_wnn_integration_uses_nearest_to_kth_distance_span_for_bandwidth():
    indices = _simple_knn_indices(5, k=2)
    embedding = np.arange(5, dtype=np.float64).reshape(-1, 1)

    graph, _ = _wnn_pair(
        "RNA",
        indices,
        embedding,
        "ADT",
        indices,
        embedding,
        nthreads=1,
        l2_normalize=False,
    )
    row_zero = graph.data[graph.row == 0]

    np.testing.assert_allclose(
        row_zero,
        np.array([1.0, np.exp(-1)], dtype=np.float32),
        rtol=1e-6,
        atol=1e-7,
    )


def test_wnn_integration_handles_degenerate_bandwidth_deterministically():
    indices1, _, indices2, _ = _multimodal_wnn_inputs()
    embedding1 = np.zeros((len(indices1), 3))
    embedding2 = np.zeros((len(indices2), 2))

    first, first_weights = _wnn_pair(
        "RNA",
        indices1,
        embedding1,
        "ADT",
        indices2,
        embedding2,
        nthreads=1,
    )
    second, second_weights = _wnn_pair(
        "RNA",
        indices1,
        embedding1,
        "ADT",
        indices2,
        embedding2,
        nthreads=1,
    )

    np.testing.assert_array_equal(first.row, second.row)
    np.testing.assert_array_equal(first.col, second.col)
    np.testing.assert_array_equal(first.data, np.ones(first.nnz, dtype=np.float32))
    np.testing.assert_array_equal(first.data, second.data)
    np.testing.assert_allclose(first_weights, 0.5)
    np.testing.assert_array_equal(first_weights, second_weights)


def test_wnn_integration_matches_scalar_affinity_reference():
    indices1, ld1, indices2, ld2 = _multimodal_wnn_inputs()
    expected_indices, expected_affinities, expected_weights = _reference_wnn(
        indices1,
        ld1,
        indices2,
        ld2,
        l2_normalize=True,
    )

    actual, actual_weights = _wnn_pair(
        "RNA",
        indices1,
        ld1,
        "ADT",
        indices2,
        ld2,
        nthreads=1,
    )

    np.testing.assert_array_equal(
        actual.col.reshape(expected_indices.shape),
        expected_indices,
    )
    np.testing.assert_allclose(
        actual.data.reshape(expected_affinities.shape),
        expected_affinities,
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        actual_weights,
        expected_weights,
        rtol=1e-6,
        atol=1e-7,
    )


def test_wnn_many_matches_independent_scalar_reference():
    modalities = _three_way_wnn_inputs()
    expected_indices, expected_affinities, expected_weights = _reference_wnn_many(
        modalities,
        l2_normalize=True,
    )

    actual, actual_weights = _wnn_integration_many(
        modalities,
        nthreads=1,
    )

    assert actual_weights.shape == (len(expected_indices), 3)
    np.testing.assert_array_equal(
        actual.col.reshape(expected_indices.shape),
        expected_indices,
    )
    np.testing.assert_allclose(
        actual.data.reshape(expected_affinities.shape),
        expected_affinities,
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        actual_weights,
        expected_weights,
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(actual_weights.sum(axis=1), 1, rtol=1e-6)


def test_wnn_many_is_equivariant_to_modality_permutation():
    modalities = _three_way_wnn_inputs()
    expected, expected_weights = _wnn_integration_many(modalities, nthreads=1)
    permutation = [2, 0, 1]

    actual, actual_weights = _wnn_integration_many(
        [modalities[index] for index in permutation],
        nthreads=1,
    )

    np.testing.assert_allclose(actual.toarray(), expected.toarray())
    np.testing.assert_allclose(
        actual_weights,
        expected_weights[:, permutation],
        rtol=1e-6,
        atol=1e-7,
    )


def test_wnn_many_is_invariant_to_cell_order():
    modalities = _three_way_wnn_inputs()
    expected, expected_weights = _wnn_integration_many(modalities, nthreads=1)
    permutation = np.array([5, 0, 7, 2, 6, 1, 4, 3])
    old_to_new = np.argsort(permutation)
    permuted_modalities = [
        (
            name,
            old_to_new[indices[permutation]],
            embedding[permutation],
        )
        for name, indices, embedding in modalities
    ]

    actual, actual_weights = _wnn_integration_many(
        permuted_modalities,
        nthreads=1,
    )
    inverse = np.argsort(permutation)

    np.testing.assert_allclose(
        actual.tocsr()[inverse][:, inverse].toarray(),
        expected.toarray(),
    )
    np.testing.assert_allclose(actual_weights[inverse], expected_weights)


def test_wnn_many_handles_degenerate_bandwidths():
    modalities = [
        (name, indices, np.zeros_like(embedding))
        for name, indices, embedding in _three_way_wnn_inputs()
    ]

    graph, weights = _wnn_integration_many(modalities, nthreads=1)

    np.testing.assert_array_equal(graph.data, np.ones(graph.nnz, dtype=np.float32))
    np.testing.assert_allclose(weights, 1 / 3, rtol=0, atol=1e-7)


def test_wnn_many_rejects_too_few_or_duplicate_modalities():
    modalities = _three_way_wnn_inputs()

    with pytest.raises(ValueError, match="at least two modalities"):
        _wnn_integration_many(modalities[:1], nthreads=1)
    with pytest.raises(ValueError, match="names must be unique"):
        _wnn_integration_many(
            [modalities[0], ("RNA", modalities[1][1], modalities[1][2])],
            nthreads=1,
        )


def test_wnn_integration_follows_informative_modality_across_numeric_scales():
    indices1 = _grouped_knn_indices([[0, 1, 2], [3, 4, 5]])
    indices2 = _grouped_knn_indices([[0, 3, 4], [1, 2, 5]])
    informative = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [1.1, -0.1],
            [-1.0, 0.0],
            [-0.9, 0.1],
            [-1.1, -0.1],
        ]
    )
    noisy = (
        np.array(
            [
                [0.0, 0.0],
                [4.0, 1.0],
                [-3.0, 2.0],
                [1.0, -4.0],
                [-2.0, -3.0],
                [3.0, 4.0],
            ]
        )
        * 1e9
    )

    graph, modality_weights = _wnn_pair(
        "RNA",
        indices1,
        informative,
        "ADT",
        indices2,
        noisy,
        nthreads=1,
        l2_normalize=False,
    )
    selected = graph.col.reshape(len(indices1), -1)
    informative_overlap = np.mean(
        [
            len(set(row).intersection(neighbors)) / len(row)
            for row, neighbors in zip(selected, indices1, strict=True)
        ]
    )
    noisy_overlap = np.mean(
        [
            len(set(row).intersection(neighbors)) / len(row)
            for row, neighbors in zip(selected, indices2, strict=True)
        ]
    )

    assert np.mean(modality_weights[:, 0]) > 0.5
    assert informative_overlap > noisy_overlap
    assert all(
        set(row).issubset(set(informative_neighbors))
        for row, informative_neighbors in zip(selected, indices1, strict=True)
    )


def test_wnn_integration_is_scale_invariant_at_near_degenerate_bandwidth():
    indices1 = _grouped_knn_indices([[0, 1, 2], [3, 4, 5]])
    indices2 = _grouped_knn_indices([[0, 3, 4], [1, 2, 5]])
    ld1 = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [1.1, -0.1],
            [-1.0, 0.0],
            [-0.9, 0.1],
            [-1.1, -0.1],
        ]
    )
    ld2 = (
        np.array(
            [
                [0.0, 0.0],
                [4.0, 1.0],
                [-3.0, 2.0],
                [1.0, -4.0],
                [-2.0, -3.0],
                [3.0, 4.0],
            ]
        )
        * 1e9
    )

    results = [
        _wnn_pair(
            "RNA",
            indices1,
            ld1 * scale,
            "ADT",
            indices2,
            ld2,
            nthreads=1,
            l2_normalize=False,
        )
        for scale in (1.0, 1e6, 1e9)
    ]
    baseline, baseline_weights = results[0]

    # Equidistant neighbours differ only by rounding noise in the distance
    # reduction, so neither may be demoted below an unrelated candidate.
    np.testing.assert_array_equal(
        baseline.col.reshape(len(indices1), -1),
        indices1,
    )
    for graph, weights in results[1:]:
        np.testing.assert_array_equal(graph.col, baseline.col)
        np.testing.assert_allclose(graph.data, baseline.data, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(weights, baseline_weights, rtol=1e-6, atol=1e-7)


@functools.cache
def _seurat_golden() -> dict:
    path = Path(__file__).parent / "seurat_wnn_5_5_1_golden.json"
    return json.loads(path.read_text())


def _seurat_golden_wnn() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run Scarf on the fixture inputs and reshape the graph to one row per cell."""
    fixture = _seurat_golden()
    inputs = fixture["inputs"]
    indices1 = np.asarray(inputs["rnaIndices"], dtype=np.uint32)
    indices2 = np.asarray(inputs["adtIndices"], dtype=np.uint32)
    graph, weights = _wnn_pair(
        "RNA",
        indices1,
        np.asarray(inputs["rnaEmbedding"], dtype=np.float64),
        "ADT",
        indices2,
        np.asarray(inputs["adtEmbedding"], dtype=np.float64),
        nthreads=1,
        l2_normalize=fixture["provenance"]["l2Normalize"],
    )
    shape = indices1.shape
    return (
        graph.col.reshape(shape),
        graph.data.reshape(shape).astype(np.float64),
        weights.astype(np.float64),
    )


@functools.cache
def _seurat_three_way_golden() -> dict:
    path = Path(__file__).parent / "seurat_wnn_3way_5_5_1_golden.json"
    return json.loads(path.read_text())


def _seurat_three_way_golden_wnn() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fixture = _seurat_three_way_golden()
    inputs = fixture["inputs"]
    modalities = [
        (
            name,
            np.asarray(inputs["neighborIndices"][name], dtype=np.uint32),
            np.asarray(inputs["embeddings"][name], dtype=np.float64),
        )
        for name in inputs["modalityNames"]
    ]
    graph, weights = _wnn_integration_many(
        modalities,
        nthreads=1,
        l2_normalize=fixture["provenance"]["l2Normalize"],
    )
    shape = modalities[0][1].shape
    return (
        graph.col.reshape(shape),
        graph.data.reshape(shape).astype(np.float64),
        weights.astype(np.float64),
    )


def test_seurat_wnn_golden_fixture_pins_its_provenance():
    provenance = _seurat_golden()["provenance"]

    assert provenance["package"] == "Seurat"
    assert provenance["packageVersion"] == "5.5.1"
    assert provenance["dataset"] == "tenx_8K_pbmc_citeseq"
    assert provenance["l2Normalize"] is True
    assert "Seurat:::PredictAssay" in provenance["matchedFunctions"]
    assert provenance["defaultFunction"] == "Seurat::FindMultiModalNeighbors"


def test_seurat_three_way_wnn_fixture_pins_its_provenance():
    provenance = _seurat_three_way_golden()["provenance"]

    assert provenance["package"] == "Seurat"
    assert provenance["packageVersion"] == "5.5.1"
    assert provenance["dataset"] == "synthetic_three_modality"
    assert provenance["modalityNames"] == ["RNA", "ATAC", "ADT"]
    assert provenance["l2Normalize"] is True


def test_wnn_many_matches_seurat_three_way_equations():
    expected = _seurat_three_way_golden()["matched"]
    selected, affinities, weights = _seurat_three_way_golden_wnn()

    np.testing.assert_array_equal(
        selected,
        np.asarray(expected["neighborIndices"], dtype=selected.dtype),
    )
    np.testing.assert_allclose(
        weights,
        np.asarray(expected["modalityWeights"]),
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        affinities,
        np.asarray(expected["neighborAffinities"]),
        rtol=0,
        atol=1e-6,
    )


def test_wnn_integration_matches_seurat_matched_equations():
    """Seurat's own routines on Scarf's candidate pool and bandwidth.

    Residual disagreement sits at the float32 resolution of the returned graph,
    so the tolerance is set just above it. Any error in the affinity kernel,
    the bandwidth index, the row normalization or the weight softmax moves the
    result by several orders of magnitude more than this.
    """
    expected = _seurat_golden()["matched"]
    selected, affinities, weights = _seurat_golden_wnn()

    np.testing.assert_array_equal(
        selected,
        np.asarray(expected["neighborIndices"], dtype=selected.dtype),
    )
    np.testing.assert_allclose(
        weights,
        np.asarray(expected["modalityWeights"]),
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        affinities,
        np.asarray(expected["neighborAffinities"]),
        rtol=0,
        atol=1e-6,
    )


def test_wnn_integration_stays_close_to_seurat_defaults():
    """Bound on how far Scarf's documented deviations move the result.

    Seurat's defaults search a wider pool and use an SNN-far bandwidth, so exact
    agreement is not expected. The floors sit below the values measured when the
    fixture was written (0.89 neighbour overlap, 0.76 weight correlation) and
    exist to catch the gap widening.
    """
    expected = _seurat_golden()["seuratDefault"]
    selected, _affinities, weights = _seurat_golden_wnn()
    k = selected.shape[1]
    seurat_indices = np.asarray(expected["neighborIndices"])
    seurat_weights = np.asarray(expected["modalityWeights"])

    overlap = np.mean(
        [
            len(set(row.tolist()).intersection(reference.tolist())) / k
            for row, reference in zip(selected, seurat_indices, strict=True)
        ]
    )
    correlation = np.corrcoef(weights[:, 0], seurat_weights[:, 0])[0, 1]

    assert overlap > 0.80
    assert correlation > 0.65
    assert abs(weights[:, 0].mean() - seurat_weights[:, 0].mean()) < 0.05


def test_wnn_integration_output_contract():
    indices1, ld1, indices2, ld2 = _multimodal_wnn_inputs()

    # The clusters are far apart, so cross-modality affinities nearly vanish
    # and the clipped scores must not overflow or divide by zero.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        graph, modality_weights = _wnn_pair(
            "RNA",
            indices1,
            ld1,
            "ADT",
            indices2,
            ld2,
            nthreads=1,
        )

    assert isinstance(graph, coo_matrix)
    assert graph.shape == (len(indices1), len(indices1))
    assert not np.any(graph.row == graph.col)
    np.testing.assert_array_equal(
        graph.row, np.repeat(np.arange(len(indices1)), indices1.shape[1])
    )
    assert graph.data.dtype == np.float32
    assert np.all(np.isfinite(graph.data))
    assert np.all((graph.data > 0) & (graph.data <= 1))
    assert modality_weights.dtype == np.float32
    assert modality_weights.shape == (len(indices1), 2)
    assert np.all(np.isfinite(modality_weights))
    assert np.all(modality_weights >= 0)
    np.testing.assert_allclose(modality_weights.sum(axis=1), 1, rtol=1e-6)


def test_wnn_integration_rejects_invalid_arguments():
    indices1, ld1, indices2, ld2 = _multimodal_wnn_inputs()

    with pytest.raises(TypeError, match="l2_normalize must be a boolean"):
        _wnn_integration_many(
            [("RNA", indices1, ld1), ("ADT", indices2, ld2)],
            1,
            l2_normalize="yes",
        )
    with pytest.raises(ValueError, match="at least two neighbors per cell"):
        _wnn_pair("RNA", indices1[:, :1], ld1, "ADT", indices2, ld2, nthreads=1)
    for embedding in (
        ld1.astype(complex),
        ld1.astype(str),
        ld1 > 0,
    ):
        with pytest.raises(TypeError, match="must contain real numeric values"):
            _wnn_pair("RNA", indices1, embedding, "ADT", indices2, ld2, nthreads=1)


def test_wnn_integration_reads_integer_embeddings_as_floats():
    indices1, _, indices2, ld2 = _multimodal_wnn_inputs()
    counts = np.arange(16).reshape(8, 2) % 5

    graph, weights = _wnn_pair(
        "RNA", indices1, counts, "ADT", indices2, ld2, nthreads=1
    )
    float_graph, float_weights = _wnn_pair(
        "RNA", indices1, counts.astype(np.float64), "ADT", indices2, ld2, nthreads=1
    )

    np.testing.assert_array_equal(graph.col, float_graph.col)
    np.testing.assert_array_equal(graph.data, float_graph.data)
    np.testing.assert_array_equal(weights, float_weights)


def test_wnn_integration_rejects_distances_that_overflow():
    indices1, ld1, indices2, ld2 = _multimodal_wnn_inputs()

    # Finite coordinates whose squared distances overflow give infinite
    # distances and so non-finite affinities, which must not be stored.
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(FloatingPointError, match="non-finite graph weights"):
            _wnn_pair(
                "RNA",
                indices1,
                ld1 * 1e200,
                "ADT",
                indices2,
                ld2,
                nthreads=1,
                l2_normalize=False,
            )


def test_wnn_integration_reports_overflow_in_every_modality():
    indices1, ld1, indices2, ld2 = _multimodal_wnn_inputs()

    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(
            FloatingPointError,
            match="^WNN integration produced non-finite modality weights$",
        ):
            _wnn_pair(
                "RNA",
                indices1,
                ld1 * 1e200,
                "ADT",
                indices2,
                ld2 * 1e200,
                nthreads=1,
                l2_normalize=False,
            )


def test_diffusion_product_bytes_count_index_widening():
    from scarf.neighbors.diffusion import _product_bytes

    narrow = csr_matrix(np.eye(2))
    wide = csr_matrix(np.eye(2))
    wide.indptr = wide.indptr.astype(np.int64)
    wide.indices = wide.indices.astype(np.int64)

    # Two rows and columns: int32 output (four 12-byte entries and three
    # pointers) plus one 16-byte value and link slot per output column.
    assert _product_bytes(narrow, narrow, 4) == 4 * 12 + 3 * 4 + 2 * 16
    # int64 operands make every index 8 bytes.
    assert _product_bytes(wide, wide, 4) == 4 * 16 + 3 * 8 + 2 * 24
    # An output past the int32 range widens the int32 operands, copying their
    # two-entry index arrays and three-entry pointers to int64 first.
    assert _product_bytes(narrow, narrow, 2**31) == (
        2**31 * 16 + 3 * 8 + 2 * 24 + 8 * (2 + 2 + 3 + 3)
    )
