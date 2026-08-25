import numpy as np
import pandas as pd
import pytest
import zarr
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from zarr.storage import MemoryStore

from scarf.metrics import (
    calculate_knn_cluster_similarity,
    calculate_top_k_neighbor_distances,
    calculate_weighted_cluster_similarity,
    clisi_knn,
    compute_lisi,
    compute_simpson,
    graph_connectivity,
    ilisi_knn,
    knn_to_csr_matrix,
    label_concordance_score,
    lisi_batch_mixing_score,
    silhouette_scoring,
)
from scarf.metrics.lisi import _effective_perplexity, _neighbor_probabilities
from scarf.storage.errors import ArtifactResolutionError


def _uniform_self_free_knn() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.array(
        [
            [1, 2, 3],
            [0, 2, 3],
            [0, 1, 3],
            [0, 1, 2],
        ]
    )
    distances = np.ones_like(indices, dtype=np.float64)
    labels = np.array([0, 0, 1, 1])
    return distances, indices, labels


def test_compute_lisi_single_category_is_one():
    metadata = pd.DataFrame({"batch": np.zeros(4, dtype=np.int8)})
    distances, indices, _labels = _uniform_self_free_knn()

    scores = compute_lisi(
        distances,
        indices,
        metadata,
        label_colnames=["batch"],
        perplexity=1,
    )

    assert np.allclose(scores[:, 0], 1.0)


def test_compute_lisi_uses_all_stored_neighbors():
    metadata = pd.DataFrame({"batch": [1, 0, 0, 0]})
    indices = np.tile(np.array([0, 1, 2]), (4, 1))
    distances = np.ones_like(indices, dtype=np.float64)

    scores = compute_lisi(
        distances,
        indices,
        metadata,
        label_colnames=["batch"],
        perplexity=1,
    )

    assert np.allclose(scores[:, 0], 1.8)


def test_compute_lisi_rejects_missing_labels():
    metadata = pd.DataFrame({"batch": [0, 1, np.nan, 1]})
    indices = np.tile(np.array([0, 1, 2]), (4, 1))
    distances = np.ones_like(indices, dtype=np.float64)

    with pytest.raises(ValueError, match="missing values"):
        compute_lisi(distances, indices, metadata, ["batch"])


def test_compute_lisi_returns_empty_matrix_when_no_labels_are_requested():
    metadata = pd.DataFrame(index=np.arange(2))
    distances = np.empty((2, 0))
    indices = np.empty((2, 0), dtype=np.int64)

    scores = compute_lisi(distances, indices, metadata, [])

    assert scores.shape == (2, 0)
    assert scores.dtype == np.float64


def test_compute_lisi_handles_more_categories_than_neighbors():
    metadata = pd.DataFrame({"batch": ["a", "b", "c", "d"]})
    indices = np.array(
        [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 0],
            [3, 0, 1],
        ]
    )
    distances = np.ones_like(indices, dtype=np.float64)

    scores = compute_lisi(
        distances,
        indices,
        metadata,
        label_colnames=["batch"],
        perplexity=1,
    )

    assert np.allclose(scores[:, 0], 3.0)


def test_compute_lisi_caps_perplexity_to_neighbor_capacity():
    metadata = pd.DataFrame({"batch": [0, 0, 1, 1]})
    indices = np.tile(np.array([0, 1, 2, 3, 0, 1]), (4, 1))
    distances = np.tile(np.arange(6, dtype=np.float64), (4, 1))

    capped = compute_lisi(distances, indices, metadata, ["batch"], perplexity=2)
    oversized = compute_lisi(distances, indices, metadata, ["batch"], perplexity=30)

    assert np.allclose(oversized, capped)


@pytest.mark.parametrize("perplexity", [0.5, np.inf, np.nan])
def test_effective_perplexity_rejects_invalid_values(perplexity):
    with pytest.raises(ValueError, match="finite value"):
        _effective_perplexity(perplexity, n_neighbors=3)


def test_effective_perplexity_requires_three_neighbors():
    with pytest.raises(ValueError, match="at least three"):
        _effective_perplexity(perplexity=1, n_neighbors=2)


def test_neighbor_probabilities_calibrate_diffuse_and_concentrated_rows():
    distances = np.array(
        [
            [0.0, 0.1, 0.2, 0.3],
            [0.0, 10.0, 20.0, 30.0],
        ]
    )

    probabilities = _neighbor_probabilities(
        distances,
        perplexity=2,
        tol=1e-8,
        max_iter=100,
    )
    log_probabilities = np.log(
        probabilities,
        out=np.zeros_like(probabilities),
        where=probabilities > 0,
    )
    calibrated_perplexity = np.exp(-np.sum(probabilities * log_probabilities, axis=1))

    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.allclose(calibrated_perplexity, 2.0, rtol=1e-7)


def test_compute_simpson_with_uniform_neighbor_weights():
    distances = np.ones((3, 2), dtype=np.float64)
    indices = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
        ]
    )
    labels = pd.Categorical([0, 0, 1, 1])

    simpson = compute_simpson(distances, indices, labels, perplexity=1)

    assert np.allclose(simpson, 5 / 9)


def test_compute_lisi_rejects_invalid_neighbor_distances():
    metadata = pd.DataFrame({"batch": [0, 0, 1, 1]})
    indices = np.tile(np.array([0, 1, 2]), (4, 1))
    distances = np.ones_like(indices, dtype=np.float64)
    distances[0, 0] = -1

    with pytest.raises(ValueError, match="finite, non-negative"):
        compute_lisi(distances, indices, metadata, ["batch"], perplexity=1)


def test_ilisi_and_clisi_match_analytic_self_free_values():
    distances, indices, labels = _uniform_self_free_knn()

    assert ilisi_knn(distances, indices, labels) == pytest.approx(0.8)
    assert clisi_knn(distances, indices, labels) == pytest.approx(0.2)
    assert ilisi_knn(
        distances,
        indices,
        labels,
        perplexity=1,
        scale=False,
    ) == pytest.approx(1.8)


def test_lisi_summary_default_perplexity_matches_floor_k_over_three():
    distances, indices, labels = _uniform_self_free_knn()

    assert ilisi_knn(distances, indices, labels) == pytest.approx(
        ilisi_knn(distances, indices, labels, perplexity=1)
    )
    assert clisi_knn(distances, indices, labels) == pytest.approx(
        clisi_knn(distances, indices, labels, perplexity=1)
    )


@pytest.mark.parametrize("metric", [ilisi_knn, clisi_knn])
def test_lisi_summary_validates_categories_and_alignment(metric):
    distances, indices, labels = _uniform_self_free_knn()

    with pytest.raises(ValueError, match="at least two categories"):
        metric(distances, indices, np.zeros(len(labels)), perplexity=1)
    with pytest.raises(ValueError, match="missing values"):
        metric(distances, indices, np.array([0, 0, 1, np.nan]), perplexity=1)
    with pytest.raises(ValueError, match="number of cells"):
        metric(distances, indices, labels[:-1], perplexity=1)
    with pytest.raises(ValueError, match="at least two categories"):
        metric(
            np.empty((0, 3)),
            np.empty((0, 3), dtype=np.int64),
            np.array([]),
            perplexity=1,
        )


def test_lisi_summaries_match_chunked_zarr_inputs():
    distances, indices, labels = _uniform_self_free_knn()
    root = zarr.open_group(store=MemoryStore(), mode="w")
    z_distances = root.create_array(
        "distances",
        data=distances,
        chunks=(2, 3),
    )
    z_indices = root.create_array(
        "indices",
        data=indices,
        chunks=(2, 3),
    )

    assert ilisi_knn(z_distances, z_indices, labels) == pytest.approx(
        ilisi_knn(distances, indices, labels)
    )
    assert clisi_knn(z_distances, z_indices, labels) == pytest.approx(
        clisi_knn(distances, indices, labels)
    )


def test_proportional_batch_mixing_differs_from_ilisi_for_imbalanced_batches():
    labels = np.array([0] * 8 + [1] * 2)
    indices = np.array(
        [[1, 8, 9]] + [[0, 8, 9] for _ in range(7)] + [[0, 1, 9], [0, 1, 8]]
    )
    distances = np.ones_like(indices, dtype=np.float64)
    metadata = pd.DataFrame({"batch": labels})
    per_cell = compute_lisi(
        distances,
        indices,
        metadata,
        ["batch"],
        perplexity=1,
    )[:, 0]

    assert lisi_batch_mixing_score(per_cell, labels) == pytest.approx(1)
    assert ilisi_knn(distances, indices, labels, perplexity=1) == pytest.approx(0.8)


def _materialized_graph_connectivity(
    edges: np.ndarray,
    labels: np.ndarray,
) -> float:
    n_cells = len(labels)
    graph = csr_matrix(
        (
            np.ones(len(edges)),
            (edges[:, 0], edges[:, 1]),
        ),
        shape=(n_cells, n_cells),
    )
    symmetric = graph + graph.T - graph.multiply(graph.T)
    scores = []
    for label in np.unique(labels):
        mask = labels == label
        subgraph = symmetric[mask][:, mask]
        n_components, component_labels = connected_components(
            subgraph,
            directed=False,
        )
        sizes = np.bincount(component_labels, minlength=n_components)
        scores.append(sizes.max() / sizes.sum())
    return float(np.mean(scores))


def test_graph_connectivity_matches_materialized_symmetric_reference():
    labels = np.array(["a", "a", "a", "b", "b", "c"])
    edges = np.array([[0, 1], [1, 0], [3, 4]], dtype=np.int64)

    expected = _materialized_graph_connectivity(edges, labels)

    assert expected == pytest.approx((2 / 3 + 1 + 1) / 3)
    assert graph_connectivity(edges, labels, batch_rows=1) == pytest.approx(expected)
    assert graph_connectivity(
        np.empty((0, 2), dtype=np.int64),
        np.array(["isolated", "isolated", "isolated"]),
    ) == pytest.approx(1 / 3)


def test_graph_connectivity_matches_chunked_zarr_and_chunk_sizes():
    labels = np.array([0, 0, 0, 1, 1])
    edges = np.array(
        [[0, 1], [1, 0], [1, 2], [3, 4], [4, 3]],
        dtype=np.uint64,
    )
    root = zarr.open_group(store=MemoryStore(), mode="w")
    z_edges = root.create_array("edges", data=edges, chunks=(2, 2))
    expected = graph_connectivity(edges, labels, batch_rows=len(edges))

    assert graph_connectivity(edges, labels, batch_rows=1) == pytest.approx(expected)
    assert graph_connectivity(z_edges, labels, batch_rows=3) == pytest.approx(expected)


def test_graph_connectivity_validates_inputs():
    labels = np.array([0, 0])

    with pytest.raises(TypeError, match="integers"):
        graph_connectivity(np.array([[0.0, 1.0]]), labels)
    with pytest.raises(IndexError, match="outside"):
        graph_connectivity(np.array([[0, 2]]), labels)
    with pytest.raises(ValueError, match="shape"):
        graph_connectivity(np.array([0, 1]), labels)
    with pytest.raises(ValueError, match="greater than zero"):
        graph_connectivity(np.array([[0, 1]]), labels, batch_rows=0)
    with pytest.raises(ValueError, match="at least one cell"):
        graph_connectivity(np.empty((0, 2), dtype=np.int64), np.array([]))
    with pytest.raises(ValueError, match="missing values"):
        graph_connectivity(
            np.array([[0, 1]]),
            np.array([0, np.nan]),
        )


def test_knn_without_affinities_preserves_distances():
    indices = np.array([[1, 2], [0, 2], [0, 1]])
    distances = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    graph = knn_to_csr_matrix(indices, distances)

    assert np.array_equal(
        graph.toarray(),
        np.array(
            [
                [0.0, 1.0, 2.0],
                [3.0, 0.0, 4.0],
                [5.0, 6.0, 0.0],
            ]
        ),
    )


@pytest.mark.parametrize(
    ("indices", "distances"),
    [
        (np.array([0, 1]), np.ones((2, 1))),
        (np.zeros((2, 1), dtype=np.int64), np.array([1.0, 2.0])),
    ],
)
def test_knn_to_csr_matrix_requires_two_dimensional_inputs(indices, distances):
    with pytest.raises(ValueError, match="two-dimensional"):
        knn_to_csr_matrix(indices, distances)


def test_knn_to_csr_matrix_requires_matching_shapes_and_integer_indices():
    with pytest.raises(ValueError, match="matching shapes"):
        knn_to_csr_matrix(
            np.array([[1], [0]]),
            np.ones((2, 2)),
        )
    with pytest.raises(TypeError, match="indices must contain integers"):
        knn_to_csr_matrix(
            np.array([[1.0], [0.0]]),
            np.ones((2, 1)),
        )


@pytest.mark.parametrize("invalid_distance", [-1.0, np.nan, np.inf])
def test_knn_to_csr_matrix_rejects_invalid_distances(invalid_distance):
    distances = np.ones((2, 1))
    distances[0, 0] = invalid_distance

    with pytest.raises(ValueError, match="finite and non-negative"):
        knn_to_csr_matrix(np.array([[1], [0]]), distances)


@pytest.mark.parametrize("shape", [(0, 1), (1, 0)])
def test_knn_to_csr_matrix_rejects_empty_axes(shape):
    with pytest.raises(ValueError, match="contain cells and neighbors"):
        knn_to_csr_matrix(
            np.empty(shape, dtype=np.int64),
            np.empty(shape, dtype=np.float64),
        )


@pytest.mark.parametrize("invalid_index", [-1, 2])
def test_knn_to_csr_matrix_rejects_out_of_range_indices(invalid_index):
    with pytest.raises(IndexError, match="outside the graph"):
        knn_to_csr_matrix(
            np.array([[invalid_index], [0]]),
            np.ones((2, 1)),
        )


def test_knn_affinities_decrease_with_distance():
    indices = np.array([[1, 2], [0, 2], [3, 0], [2, 1]])
    distances = np.array([[0.1, 10.0], [0.1, 3.0], [0.2, 4.0], [0.2, 5.0]])

    graph = knn_to_csr_matrix(indices, distances, use_affinities=True)

    assert graph[0, 1] > graph[0, 2]


def test_cluster_similarity_is_symmetric_with_unit_diagonal():
    graph = csr_matrix(
        np.array(
            [
                [0.0, 1.0, 0.2, 0.0],
                [1.0, 0.0, 0.3, 0.0],
                [0.2, 0.3, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
    )
    labels = np.array([0, 0, 1, 1])

    similarities = calculate_weighted_cluster_similarity(graph, labels)

    assert np.allclose(similarities, similarities.T)
    assert np.allclose(np.diag(similarities), 1)
    assert similarities[0, 1] > 0


def test_weighted_cluster_similarity_validates_graph_and_labels():
    with pytest.raises(ValueError, match="square"):
        calculate_weighted_cluster_similarity(
            csr_matrix((2, 3), dtype=np.float64),
            np.array([0, 1]),
        )

    for invalid_weight in (-1.0, np.nan, np.inf):
        graph = csr_matrix(np.array([[0.0, invalid_weight], [1.0, 0.0]]))
        with pytest.raises(ValueError, match="finite and non-negative"):
            calculate_weighted_cluster_similarity(graph, np.array([0, 1]))

    graph = csr_matrix(np.eye(2))
    with pytest.raises(ValueError, match="one value per graph node"):
        calculate_weighted_cluster_similarity(graph, np.array([0]))
    with pytest.raises(TypeError, match="labels must contain integers"):
        calculate_weighted_cluster_similarity(graph, np.array(["a", "b"]))
    with pytest.raises(ValueError, match="contiguous integers starting at 0"):
        calculate_weighted_cluster_similarity(graph, np.array([0, 2]))


def test_streamed_knn_similarity_matches_csr_similarity():
    indices = np.array([[1, 2], [0, 2], [3, 0], [2, 1]])
    distances = np.array([[0.1, 10.0], [0.1, 3.0], [0.2, 4.0], [0.2, 5.0]])
    labels = np.array([0, 0, 1, 1])
    graph = knn_to_csr_matrix(indices, distances, use_affinities=True)

    streamed = calculate_knn_cluster_similarity(
        indices,
        distances,
        labels,
        batch_rows=2,
    )
    materialized = calculate_weighted_cluster_similarity(graph, labels)

    assert np.allclose(streamed, materialized)


def test_streamed_knn_similarity_validates_structure():
    labels = np.array([0, 1])

    with pytest.raises(ValueError, match="two-dimensional"):
        calculate_knn_cluster_similarity(
            np.array([1, 0]),
            np.ones((2, 1)),
            labels,
        )
    with pytest.raises(ValueError, match="matching shapes"):
        calculate_knn_cluster_similarity(
            np.array([[1], [0]]),
            np.ones((2, 2)),
            labels,
        )
    with pytest.raises(ValueError, match="contain neighbors"):
        calculate_knn_cluster_similarity(
            np.empty((2, 0), dtype=np.int64),
            np.empty((2, 0), dtype=np.float64),
            labels,
        )
    with pytest.raises(ValueError, match="greater than zero"):
        calculate_knn_cluster_similarity(
            np.array([[1], [0]]),
            np.ones((2, 1)),
            labels,
            batch_rows=0,
        )


@pytest.mark.parametrize(
    ("indices", "distances", "error", "message"),
    [
        (
            np.array([[1.0], [0.0]]),
            np.ones((2, 1)),
            TypeError,
            "indices must contain integers",
        ),
        (
            np.array([[-1], [0]]),
            np.ones((2, 1)),
            IndexError,
            "outside the graph",
        ),
        (
            np.array([[2], [0]]),
            np.ones((2, 1)),
            IndexError,
            "outside the graph",
        ),
        (
            np.array([[1], [0]]),
            np.array([[-1.0], [1.0]]),
            ValueError,
            "finite and non-negative",
        ),
        (
            np.array([[1], [0]]),
            np.array([[np.nan], [1.0]]),
            ValueError,
            "finite and non-negative",
        ),
    ],
)
def test_streamed_knn_similarity_validates_blocks(
    indices,
    distances,
    error,
    message,
):
    with pytest.raises(error, match=message):
        calculate_knn_cluster_similarity(
            indices,
            distances,
            np.array([0, 1]),
            batch_rows=1,
        )


def test_top_k_distances_accept_all_candidates():
    distances = calculate_top_k_neighbor_distances(
        np.array([[0.0]]),
        np.array([[1.0], [2.0]]),
        k=2,
    )

    assert np.allclose(np.sort(distances[0]), [1.0, 2.0])


@pytest.mark.parametrize(
    ("matrix_a", "matrix_b"),
    [
        (np.array([0.0, 1.0]), np.ones((2, 2))),
        (np.ones((2, 2)), np.array([0.0, 1.0])),
    ],
)
def test_top_k_distances_require_two_dimensional_inputs(matrix_a, matrix_b):
    with pytest.raises(ValueError, match="two-dimensional"):
        calculate_top_k_neighbor_distances(matrix_a, matrix_b, k=1)


def test_top_k_distances_require_matching_feature_counts():
    with pytest.raises(ValueError, match="same number of features"):
        calculate_top_k_neighbor_distances(
            np.ones((2, 1)),
            np.ones((2, 2)),
            k=1,
        )


@pytest.mark.parametrize(
    ("matrix_a", "matrix_b"),
    [
        (np.empty((0, 2)), np.ones((1, 2))),
        (np.ones((1, 2)), np.empty((0, 2))),
    ],
)
def test_top_k_distances_reject_empty_point_sets(matrix_a, matrix_b):
    with pytest.raises(ValueError, match="at least one point"):
        calculate_top_k_neighbor_distances(matrix_a, matrix_b, k=1)


@pytest.mark.parametrize(
    ("matrix_a", "matrix_b"),
    [
        (np.array([[np.nan, 0.0]]), np.ones((1, 2))),
        (np.ones((1, 2)), np.array([[np.inf, 0.0]])),
    ],
)
def test_top_k_distances_reject_nonfinite_values(matrix_a, matrix_b):
    with pytest.raises(ValueError, match="finite values"):
        calculate_top_k_neighbor_distances(matrix_a, matrix_b, k=1)


@pytest.mark.parametrize("k", [0, -1])
def test_top_k_distances_require_positive_k(k):
    with pytest.raises(ValueError, match="greater than zero"):
        calculate_top_k_neighbor_distances(
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            k=k,
        )


def test_top_k_cosine_distances_handle_opposite_and_zero_vectors():
    distances = calculate_top_k_neighbor_distances(
        np.array([[1.0, 0.0], [0.0, 0.0]]),
        np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
        k=10,
        metric="cosine",
    )

    assert np.allclose(
        np.sort(distances, axis=1),
        np.array(
            [
                [0.0, 1.0, 2.0],
                [1.0, 1.0, 1.0],
            ]
        ),
    )


def test_top_k_inner_product_distances_are_clipped_at_zero():
    distances = calculate_top_k_neighbor_distances(
        np.array([[2.0, 0.0], [0.5, 0.0]]),
        np.array([[1.0, 0.0], [0.25, 0.0], [-1.0, 0.0]]),
        k=3,
        metric="ip",
    )

    assert np.allclose(
        np.sort(distances, axis=1),
        np.array(
            [
                [0.0, 0.5, 3.0],
                [0.5, 0.875, 1.5],
            ]
        ),
    )


def test_top_k_distances_reject_unsupported_metric():
    with pytest.raises(ValueError, match="Unsupported neighbor metric"):
        calculate_top_k_neighbor_distances(
            np.array([[0.0]]),
            np.array([[1.0]]),
            k=1,
            metric="manhattan",
        )


def test_metric_silhouette(datastore, graph_artifacts, leiden_clustering):
    scores = datastore.metric_graph_silhouette(random_seed=42)
    scores_from_full_label = datastore.metric_graph_silhouette(
        res_label="RNA_leiden_cluster",
        random_seed=42,
    )

    assert scores is not None
    assert scores_from_full_label is not None
    assert np.array_equal(scores, scores_from_full_label, equal_nan=True)
    assert np.isfinite(scores).any()
    assert np.all(np.abs(scores[np.isfinite(scores)]) <= 1)


def test_small_cluster_does_not_invalidate_other_silhouette_scores():
    class Cells:
        columns = ["RNA_subset_cluster"]

        @staticmethod
        def fetch(column, key="I"):
            assert column == "RNA_subset_cluster"
            assert key == "subset"
            return np.array([1, 2, 2, 2, 2, 3, 3, 3, 3])

    class Store:
        cells = Cells()

    class Ann:
        annMetric = "l2"

        @staticmethod
        def reducer(values):
            return values

    data = np.array(
        [
            [20.0, 20.0],
            [0.0, 0.0],
            [0.0, 0.1],
            [0.1, 0.0],
            [0.1, 0.1],
            [10.0, 10.0],
            [10.0, 10.1],
            [10.1, 10.0],
            [10.1, 10.1],
        ]
    )
    graph = csr_matrix(np.ones((len(data), len(data))) - np.eye(len(data)))

    scores = silhouette_scoring(
        Store(),
        Ann(),
        graph,
        data,
        "RNA",
        "cluster",
        cell_key="subset",
        sample_size=2,
        random_seed=42,
    )

    assert scores is not None
    assert np.isnan(scores[0])
    assert np.isfinite(scores[1:]).all()


@pytest.mark.parametrize("metric", ["ari", "nmi"])
def test_label_concordance_identical_partitions(metric):
    labels = np.array([0, 0, 1, 1])

    assert label_concordance_score([labels, labels], metric) == pytest.approx(1)


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("ari", 0.125),
        ("nmi", 0.18872187554086706),
    ],
)
def test_label_concordance_partial_agreement(metric, expected):
    first = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    second = np.array([0, 0, 0, 1, 1, 1, 1, 0])

    assert label_concordance_score([first, second], metric) == pytest.approx(expected)


def test_label_concordance_requires_exactly_two_partitions():
    with pytest.raises(ValueError, match="Exactly two"):
        label_concordance_score([np.array([0, 1])])


def test_lisi_batch_mixing_score():
    labels = np.array([0, 0, 1, 1])

    assert lisi_batch_mixing_score(np.ones(4), labels) == pytest.approx(0)
    assert lisi_batch_mixing_score(np.full(4, 2.0), labels) == pytest.approx(1)


def test_metric_label_concordance(datastore, graph_artifacts, leiden_clustering):
    rng = np.random.default_rng(42)
    labels1 = rng.integers(0, 2, datastore.cells.N)
    labels2 = labels1.copy()
    selected = np.zeros(datastore.cells.N, dtype=bool)
    selected[: datastore.cells.N // 2] = True
    labels2[~selected] = 1 - labels2[~selected]
    datastore.cells.insert(
        column_name="labels1",
        values=labels1,
        overwrite=True,
    )
    datastore.cells.insert(
        column_name="labels2",
        values=labels2,
        overwrite=True,
    )
    datastore.cells.insert(
        column_name="metric_subset",
        values=selected,
        overwrite=True,
    )

    assert datastore.metric_label_concordance(
        ["labels1", "labels2"],
        cell_key="metric_subset",
    ) == pytest.approx(1)
    assert (
        datastore.metric_label_concordance(
            ["labels1", "labels2"],
            cell_key="I",
        )
        < 1
    )
    mixing_score = datastore.metric_proportional_batch_mixing("labels1")
    assert 0 <= mixing_score <= 1


def test_datastore_scib_metrics(datastore, graph_artifacts, leiden_clustering):
    label_colname = "RNA_leiden_cluster"
    state = datastore.get_assay_state("RNA")
    assert state is not None
    assert state.connectivity_map is not None

    labels = np.zeros(datastore.cells.N, dtype=np.int8)
    datastore.cells.insert(
        column_name="single_batch",
        values=labels,
        overwrite=True,
    )
    lisi = datastore.metric_lisi(label_columns=["single_batch"])
    assert np.allclose(lisi["single_batch"], 1)

    ilisi = datastore.metric_ilisi(label_colname)
    clisi = datastore.metric_clisi(label_colname)
    latest_connectivity = datastore.metric_graph_connectivity(
        label_colname,
        from_assay="RNA",
        cell_key="I",
    )
    explicit_connectivity = datastore.metric_graph_connectivity(
        label_colname,
        state.connectivity_map,
    )

    assert 0 <= ilisi <= 1
    assert 0 <= clisi <= 1
    assert 0 <= latest_connectivity <= 1
    assert explicit_connectivity == pytest.approx(latest_connectivity)
    assert state.neighbors is not None
    with pytest.raises(ValueError, match="connectivity map or an integrated graph"):
        datastore.metric_graph_connectivity(
            label_colname,
            state.neighbors,
        )


def test_metric_lisi_rejects_invalid_inputs(datastore, graph_artifacts):
    with pytest.raises(TypeError, match="sequence of column names"):
        datastore.metric_lisi("names")
    with pytest.raises(ValueError, match="non-empty"):
        datastore.metric_lisi([])
    with pytest.raises(TypeError, match="only strings"):
        datastore.metric_lisi(["names", 1])
    with pytest.raises(ValueError, match="duplicate names"):
        datastore.metric_lisi(["names", "names"])
    with pytest.raises(KeyError, match="__missing_lisi_column__"):
        datastore.metric_lisi(["__missing_lisi_column__"])
    with pytest.raises(TypeError, match="artifact reference"):
        datastore.metric_lisi(
            ["names"],
            neighbors="not-a-ref",
        )
    state = datastore.get_assay_state("RNA")
    assert state is not None
    assert state.connectivity_map is not None
    with pytest.raises(ValueError, match="neighbors artifact"):
        datastore.metric_lisi(
            ["names"],
            neighbors=state.connectivity_map,
        )
    assert state.neighbors is not None
    with pytest.raises(ValueError, match="different assay"):
        datastore.metric_lisi(
            ["names"],
            neighbors=state.neighbors,
            from_assay="missing",
        )


def test_metric_lisi_revalidates_explicit_neighbor_cell_selection(
    datastore,
    graph_artifacts,
    monkeypatch,
):
    del graph_artifacts
    state = datastore.get_assay_state("RNA")
    assert state is not None
    assert state.neighbors is not None
    column = datastore.zw["cellData/I"]
    original = np.asarray(column[:], dtype=bool)
    selected = np.flatnonzero(original)
    excluded = np.flatnonzero(~original)
    assert selected.size and excluded.size
    changed = original.copy()
    changed[selected[0]] = False
    changed[excluded[0]] = True
    column[:] = changed

    def reject_current_state_read(*_args, **_kwargs):
        raise AssertionError("explicit neighbors resolved the current state")

    monkeypatch.setattr(
        "scarf.datastore._operations.integration_metrics.read_assay_state",
        reject_current_state_read,
    )
    try:
        with pytest.raises(ArtifactResolutionError) as error:
            datastore.metric_lisi(
                ["names"],
                neighbors=state.neighbors,
                perplexity=1,
            )
        assert error.value.code == "selection_values_changed"
    finally:
        column[:] = original


def test_silhouette_scoring_missing_cluster_labels(datastore):
    result = silhouette_scoring(
        datastore,
        ann_obj=None,
        graph=None,
        hvg_data=None,
        assay_type="RNA",
        res_label="missing_resolution_label",
    )
    assert result is None
