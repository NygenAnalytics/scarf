import functools
import json
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
from scarf.neighbors.diffusion import diffusion_operator
from scarf.neighbors.integration import _wnn_integration_many, wnn_integration
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


def test_wnn_integration_handles_extreme_affinities_without_runtime_warnings():
    indices1, ld1, indices2, ld2 = _multimodal_wnn_inputs()

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        merged, modality_weights = wnn_integration(
            "RNA",
            indices1,
            ld1,
            "ADT",
            indices2,
            ld2,
            nthreads=1,
        )

    assert isinstance(merged, coo_matrix)
    assert merged.shape == (len(indices1), len(indices1))
    assert merged.nnz == indices1.size
    np.testing.assert_array_equal(
        np.bincount(merged.row, minlength=len(indices1)),
        np.repeat(indices1.shape[1], len(indices1)),
    )
    assert np.all(np.isfinite(merged.data))
    assert np.all(merged.data > 0)
    assert np.all(merged.data <= 1)
    assert modality_weights.dtype == np.float32
    assert np.all(np.isfinite(modality_weights))
    assert np.all(modality_weights >= 0)
    np.testing.assert_allclose(modality_weights.sum(axis=1), 1, rtol=1e-6)


def test_wnn_integration_is_invariant_to_cell_order():
    indices1, _, indices2, _ = _multimodal_wnn_inputs()
    rng = np.random.default_rng(42)
    ld1 = rng.normal(size=(len(indices1), 3))
    ld2 = rng.normal(size=(len(indices2), 4))
    expected, expected_weights = wnn_integration(
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
    permuted, permuted_weights = wnn_integration(
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
        wnn_integration(
            "RNA",
            indices1,
            np.zeros((6, 2)),
            "ADT",
            indices2,
            np.zeros((7, 2)),
            nthreads=1,
        )


def test_two_input_wnn_adapter_preserves_duplicate_diagnostic_names():
    indices1, ld1, indices2, ld2 = _multimodal_wnn_inputs()
    expected, expected_weights = wnn_integration(
        "first",
        indices1,
        ld1,
        "second",
        indices2,
        ld2,
        nthreads=1,
    )
    actual, actual_weights = wnn_integration(
        "same",
        indices1,
        ld1,
        "same",
        indices2,
        ld2,
        nthreads=1,
    )

    np.testing.assert_array_equal(actual.row, expected.row)
    np.testing.assert_array_equal(actual.col, expected.col)
    np.testing.assert_array_equal(actual.data, expected.data)
    np.testing.assert_array_equal(actual_weights, expected_weights)


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
        wnn_integration(
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
        wnn_integration(
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
        merged, modality_weights = wnn_integration(
            "RNA",
            indices1,
            ld1,
            "ADT",
            indices2,
            ld2,
            nthreads=1,
        )
        swapped, swapped_weights = wnn_integration(
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
    assert np.all(np.isfinite(merged.data))
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
    expected, expected_weights = wnn_integration(
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

    actual, actual_weights = wnn_integration(
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

    graph, _ = wnn_integration(
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

    first, first_weights = wnn_integration(
        "RNA",
        indices1,
        embedding1,
        "ADT",
        indices2,
        embedding2,
        nthreads=1,
    )
    second, second_weights = wnn_integration(
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

    actual, actual_weights = wnn_integration(
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


def test_wnn_grouped_pairwise_weights_differ_from_max_cross_shortcut():
    directed_scores = np.array(
        [
            [-np.inf, 4.0, 0.0],
            [2.0, -np.inf, 2.0],
            [1.0, 0.5, -np.inf],
        ]
    )
    finite = np.isfinite(directed_scores)
    pairwise = np.zeros_like(directed_scores)
    pairwise[finite] = np.exp(directed_scores[finite] - directed_scores[finite].max())
    grouped = pairwise.sum(axis=1)
    grouped /= grouped.sum()
    max_cross_scores = np.max(directed_scores, axis=1)
    max_cross = np.exp(max_cross_scores - max_cross_scores.max())
    max_cross /= max_cross.sum()

    assert not np.allclose(grouped, max_cross)
    assert grouped[1] > max_cross[1]


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

    graph, modality_weights = wnn_integration(
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
        wnn_integration(
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
    graph, weights = wnn_integration(
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

    graph, modality_weights = wnn_integration(
        "RNA",
        indices1,
        ld1,
        "ADT",
        indices2,
        ld2,
        nthreads=1,
    )

    assert not np.any(graph.row == graph.col)
    np.testing.assert_array_equal(
        np.bincount(graph.row, minlength=len(indices1)),
        np.full(len(indices1), min(indices1.shape[1], indices2.shape[1])),
    )
    assert graph.data.dtype == np.float32
    assert np.all(np.isfinite(graph.data))
    assert np.all((graph.data > 0) & (graph.data <= 1))
    assert modality_weights.shape == (len(indices1), 2)
    assert np.all(np.isfinite(modality_weights))
    assert np.all(modality_weights >= 0)
    np.testing.assert_allclose(modality_weights.sum(axis=1), 1, rtol=1e-6)
