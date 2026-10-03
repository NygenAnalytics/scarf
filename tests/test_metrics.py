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
    graph_connectivity,
    ilisi_knn,
    label_concordance_score,
    lisi_batch_mixing_score,
    silhouette_scoring,
)
from scarf.metrics.lisi import _effective_perplexity, _neighbor_probabilities
from scarf.metadata.artifacts import plan_cell_data_artifact, write_cell_data_artifact
from scarf.storage.artifacts import ArtifactRef, ArtifactScope
from scarf.storage.selections import resolve_stored_selection_artifact


def _graph_neighbors(datastore, graph: ArtifactRef) -> ArtifactRef:
    raw = datastore.inspect_artifact(graph).inputs["neighbors"]
    return ArtifactRef.from_dict(raw)


def _clustering_artifact(
    datastore,
    labels: np.ndarray,
    *,
    selection: ArtifactRef,
    scope: ArtifactScope = "assay",
    assay: str | None = "RNA",
    kind: str = "cluster_labels",
) -> ArtifactRef:
    if scope == "datastore":
        assay = None
    values = np.asarray(labels)
    value_name = "values" if kind == "cluster_labels" else "labels"
    planned = plan_cell_data_artifact(
        datastore.zw,
        scope=scope,
        assay=assay,
        kind=kind,
        operation="test_metric_clustering",
        parameters={},
        inputs={},
        execution_options={},
        cell_selection=selection,
        arrays={value_name: (values.shape, None)},
        invalidate_cache=True,
    )
    write_cell_data_artifact(datastore.zw, planned, {value_name: values})
    return planned.ref


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


@pytest.mark.parametrize("n_neighbors", [11, 90, 300])
def test_lisi_entry_points_share_graph_width_default(n_neighbors):
    rng = np.random.default_rng(18)
    n_cells = n_neighbors + 1
    indices = np.array([np.delete(np.arange(n_cells), i) for i in range(n_cells)])
    distances = np.sort(rng.uniform(0.1, 3, indices.shape), axis=1)
    labels = np.arange(n_cells) % 3
    metadata = pd.DataFrame({"labels": labels})
    actual = compute_lisi(distances, indices, metadata, ["labels"])[:, 0]
    explicit = compute_lisi(
        distances, indices, metadata, ["labels"], perplexity=n_neighbors // 3
    )[:, 0]
    np.testing.assert_array_equal(actual, explicit)
    assert ilisi_knn(distances, indices, labels, scale=False) == np.median(actual)
    assert clisi_knn(distances, indices, labels, scale=False) == np.median(actual)


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


@pytest.mark.parametrize("batch_rows", [1, 3, 10])
@pytest.mark.parametrize("stored", [False, True])
def test_graph_connectivity_ignores_zero_weight_bridges(batch_rows, stored):
    labels = np.zeros(4, dtype=int)
    edges = np.array(
        [[0, 1], [1, 0], [2, 3], [3, 2], [1, 2]],
        dtype=np.uint64,
    )
    weights = np.array([1.0, 0.5, 0.25, 1.0, 0.0], dtype=np.float32)
    if stored:
        root = zarr.open_group(store=MemoryStore(), mode="w")
        edges = root.create_array("edges", data=edges, chunks=(2, 2))
        weights = root.create_array("weights", data=weights, chunks=(3,))

    assert (
        graph_connectivity(edges, labels, batch_rows=batch_rows, weights=weights) == 0.5
    )
    assert graph_connectivity(edges, labels, batch_rows=batch_rows) == 1.0


@pytest.mark.parametrize(
    ("weights", "error", "message"),
    [
        (np.ones((2, 1)), ValueError, "one value per edge"),
        (np.ones(1), ValueError, "one value per edge"),
        (np.array(["1", "0"]), TypeError, "real numbers"),
        (np.array([1.0, np.nan]), ValueError, "finite and non-negative"),
        (np.array([1.0, -1.0]), ValueError, "finite and non-negative"),
    ],
)
def test_graph_connectivity_rejects_invalid_weights(weights, error, message):
    with pytest.raises(error, match=message):
        graph_connectivity(
            np.array([[0, 1], [1, 0]]),
            np.zeros(2),
            batch_rows=1,
            weights=weights,
        )


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
    graph = csr_matrix(
        (
            (1 / (np.log1p(distances) + 1)).ravel(),
            (np.repeat(np.arange(len(indices)), indices.shape[1]), indices.ravel()),
        ),
        shape=(len(indices), len(indices)),
    )

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


def test_metric_silhouette(datastore, connectivity_graph, leiden_clustering):
    neighbors = _graph_neighbors(datastore, connectivity_graph)
    scores = datastore.metric_graph_silhouette(
        neighbors,
        leiden_clustering,
        random_seed=42,
    )

    assert scores is not None
    n_clusters = np.unique(datastore.load_artifact(leiden_clustering)["values"][:]).size
    # One score per Leiden cluster; each has at least two cells.
    assert scores.shape == (n_clusters,)
    assert np.isfinite(scores).all()
    assert np.all(np.abs(scores) <= 1)
    # Graph clusters are on average closer within than to their nearest cluster.
    assert scores.mean() > 0
    np.testing.assert_array_equal(
        datastore.metric_graph_silhouette(
            neighbors,
            leiden_clustering,
            random_seed=42,
        ),
        scores,
    )


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
        graph,
        data,
        "RNA",
        "cluster",
        cell_key="subset",
        sample_size=2,
        random_seed=42,
        distance_metric="l2",
    )

    assert scores is not None
    assert np.isnan(scores[0])
    # Tight squares about 14 apart: within distances near 0.1 give scores
    # of about 1 - 0.1 / 14.
    assert np.all(scores[1:] > 0.98)


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


def test_metric_label_concordance_uses_frozen_clustering_refs_after_alias_drift(
    datastore_ephemeral,
):
    datastore = datastore_ephemeral
    selection = datastore.snapshot_cell_selection("I")
    selection_mask = np.asarray(datastore.load_artifact(selection)["values"][:])
    first_labels = np.arange(int(selection_mask.sum())) % 3
    second_labels = first_labels.copy()
    second_labels[::5] = (second_labels[::5] + 1) % 3
    first = _clustering_artifact(
        datastore,
        first_labels,
        selection=selection,
    )
    second = _clustering_artifact(
        datastore,
        second_labels,
        selection=selection,
        assay="assay2",
        kind="cluster_cut",
    )
    expected_ari = label_concordance_score([first_labels, second_labels], "ari")
    expected_nmi = label_concordance_score([first_labels, second_labels], "nmi")
    columns_before = set(datastore.cells.columns)
    artifacts_before = set(datastore.list_artifacts())
    live_selection = datastore.zw["cellData/I"]
    original = np.asarray(live_selection[:], dtype=bool)
    live_selection[:] = ~original
    try:
        assert datastore.metric_label_concordance(first, second) == pytest.approx(
            expected_ari
        )
        assert datastore.metric_label_concordance(
            first,
            second,
            metric="nmi",
        ) == pytest.approx(expected_nmi)
    finally:
        live_selection[:] = original

    assert set(datastore.cells.columns) == columns_before
    assert set(datastore.list_artifacts()) == artifacts_before


def test_metric_label_concordance_requires_the_exact_cell_selection_ref(
    datastore_ephemeral,
):
    datastore = datastore_ephemeral
    first_selection = datastore.snapshot_cell_selection("I")
    second_selection = resolve_stored_selection_artifact(
        datastore.zw,
        table_path="cellData",
        id_column="ids",
        source_column="I",
        scope="datastore",
        kind="cell_selection",
        operation="test_metric_selection",
        parameters={},
        inputs={},
        invalidate_cache=True,
    )
    assert second_selection != first_selection
    selected_count = int(
        np.asarray(datastore.load_artifact(first_selection)["values"][:]).sum()
    )
    labels = np.arange(selected_count) % 2
    first = _clustering_artifact(
        datastore,
        labels,
        selection=first_selection,
    )
    second = _clustering_artifact(
        datastore,
        labels,
        selection=second_selection,
    )

    with pytest.raises(ValueError, match="different cell selections"):
        datastore.metric_label_concordance(first, second)
    with pytest.raises(TypeError, match="first must be an ArtifactRef"):
        datastore.metric_label_concordance("labels1", second)


def test_metric_label_concordance_fails_closed_on_invalid_clustering_contracts(
    datastore_ephemeral,
):
    datastore = datastore_ephemeral
    selection = datastore.snapshot_cell_selection("I")
    selected_count = int(
        np.asarray(datastore.load_artifact(selection)["values"][:]).sum()
    )
    labels = np.arange(selected_count) % 2
    valid = _clustering_artifact(datastore, labels, selection=selection)

    with pytest.raises(ValueError, match="clustering artifact"):
        datastore.metric_label_concordance(selection, valid)

    group = datastore.zw[datastore.inspect_artifact(valid).path]
    original_provenance = dict(group.attrs["provenance"])
    try:
        group.attrs["provenance"] = {**original_provenance, "inputs": {}}
        with pytest.raises(ValueError, match="no cell-selection input"):
            datastore.metric_label_concordance(valid, valid)

        malformed = {**original_provenance, "inputs": {"cell_selection": {}}}
        group.attrs["provenance"] = malformed
        with pytest.raises(ValueError, match="malformed cell-selection input"):
            datastore.metric_label_concordance(valid, valid)
    finally:
        group.attrs["provenance"] = original_provenance

    missing_values = _clustering_artifact(datastore, labels, selection=selection)
    missing_group = datastore.zw[datastore.inspect_artifact(missing_values).path]
    del missing_group["values"]
    with pytest.raises(ValueError, match="no canonical 'values' label array"):
        datastore.metric_label_concordance(missing_values, valid)

    misaligned = _clustering_artifact(datastore, labels, selection=selection)
    misaligned_values = datastore.zw[datastore.inspect_artifact(misaligned).path][
        "values"
    ]
    misaligned_values.resize((selected_count - 1,))
    with pytest.raises(ValueError, match="one label per selected cell"):
        datastore.metric_label_concordance(misaligned, valid)


def test_metric_label_concordance_accepts_datastore_scoped_clusterings(
    datastore_ephemeral,
):
    datastore = datastore_ephemeral
    selection = datastore.snapshot_cell_selection("I")
    selected_count = int(
        np.asarray(datastore.load_artifact(selection)["values"][:]).sum()
    )
    labels = np.arange(selected_count) % 2
    assay_scoped = _clustering_artifact(datastore, labels, selection=selection)
    datastore_scoped = _clustering_artifact(
        datastore,
        labels,
        selection=selection,
        scope="datastore",
    )
    assert datastore.metric_label_concordance(
        datastore_scoped,
        assay_scoped,
    ) == pytest.approx(1.0)


def test_metric_label_concordance_rejects_missing_cluster_labels(
    datastore_ephemeral,
):
    datastore = datastore_ephemeral
    selection = datastore.snapshot_cell_selection("I")
    selected_count = int(
        np.asarray(datastore.load_artifact(selection)["values"][:]).sum()
    )
    labels = np.arange(selected_count) % 2
    complete = _clustering_artifact(datastore, labels, selection=selection)
    masked = _clustering_artifact(datastore, labels, selection=selection)
    group = datastore.zw[datastore.inspect_artifact(masked).path]
    missing_name = "__scarf_missing__values"
    missing = np.zeros(selected_count, dtype=bool)
    missing[0] = True
    group.create_array(missing_name, data=missing)
    group["values"].attrs["missing_mask"] = missing_name

    with pytest.raises(ValueError, match="contains missing cluster labels"):
        datastore.metric_label_concordance(masked, complete)


def test_metric_label_concordance_rejects_malformed_cluster_missing_masks(
    datastore_ephemeral,
):
    datastore = datastore_ephemeral
    selection = datastore.snapshot_cell_selection("I")
    selected_count = int(
        np.asarray(datastore.load_artifact(selection)["values"][:]).sum()
    )
    labels = np.arange(selected_count) % 2
    complete = _clustering_artifact(datastore, labels, selection=selection)
    for mask_case in (
        "non_string_link",
        "missing_array",
        "wrong_dtype",
        "wrong_shape",
    ):
        malformed = _clustering_artifact(datastore, labels, selection=selection)
        group = datastore.zw[datastore.inspect_artifact(malformed).path]
        missing_name = "__scarf_missing__values"
        if mask_case == "wrong_dtype":
            group.create_array(
                missing_name,
                data=np.zeros(selected_count, dtype=np.int8),
            )
        elif mask_case == "wrong_shape":
            group.create_array(
                missing_name,
                data=np.zeros(selected_count + 1, dtype=bool),
            )
        group["values"].attrs["missing_mask"] = (
            None if mask_case == "non_string_link" else missing_name
        )

        with pytest.raises(ValueError, match="malformed missing-label mask"):
            datastore.metric_label_concordance(malformed, complete)


def test_datastore_scib_metrics(datastore, connectivity_graph):
    neighbors = _graph_neighbors(datastore, connectivity_graph)
    annotations = np.arange(datastore.cells.N) % 3
    datastore.cells.insert(
        column_name="metric_annotations",
        values=annotations,
        overwrite=True,
    )
    batches = np.arange(datastore.cells.N) % 2
    datastore.cells.insert(
        column_name="metric_batches",
        values=batches,
        overwrite=True,
    )
    ilisi = datastore.metric_ilisi("metric_batches", neighbors)
    clisi = datastore.metric_clisi(
        annotation_column="metric_annotations",
        neighbors=neighbors,
    )
    graph_connectivity_score = datastore.metric_graph_connectivity(
        annotation_column="metric_annotations",
        graph=connectivity_graph,
    )
    mixing_score = datastore.metric_proportional_batch_mixing(
        "metric_batches",
        neighbors,
    )

    assert 0 <= ilisi <= 1
    assert 0 <= clisi <= 1
    assert 0 <= graph_connectivity_score <= 1
    assert 0 <= mixing_score <= 1
    with pytest.raises(ValueError, match="connectivity_map or integrated_graph"):
        datastore.metric_graph_connectivity(
            "metric_annotations",
            neighbors,
        )
    with pytest.raises(TypeError, match="unexpected keyword argument 'label_colname'"):
        datastore.metric_clisi(
            label_colname="metric_annotations",
            neighbors=neighbors,
        )
    with pytest.raises(TypeError, match="unexpected keyword argument 'label_colname'"):
        datastore.metric_graph_connectivity(
            label_colname="metric_annotations",
            graph=connectivity_graph,
        )


def test_datastore_metrics_reject_missing_metadata_labels(
    datastore,
    connectivity_graph,
):
    neighbors = _graph_neighbors(datastore, connectivity_graph)
    column = "masked_metric_labels"
    datastore.cells.insert(
        column_name=column,
        values=np.zeros(datastore.cells.N, dtype=np.int8),
        overwrite=True,
    )
    cell_data = datastore.zw["cellData"]
    missing_name = f"__scarf_missing__{column}"
    cell_data.create_array(
        missing_name,
        data=np.ones(datastore.cells.N, dtype=bool),
    )
    cell_data[column].attrs["missing_mask"] = missing_name

    with pytest.raises(ValueError, match="contains missing values"):
        datastore.metric_ilisi(column, neighbors)
    with pytest.raises(ValueError, match="contains missing values"):
        datastore.metric_clisi(column, neighbors)
    with pytest.raises(ValueError, match="contains missing values"):
        datastore.metric_graph_connectivity(column, connectivity_graph)
    with pytest.raises(ValueError, match="contains missing values"):
        datastore.metric_proportional_batch_mixing(column, neighbors)


def test_datastore_metrics_reject_blank_labels_and_name_missing_columns(
    datastore,
    connectivity_graph,
):
    neighbors = _graph_neighbors(datastore, connectivity_graph)
    column = "blank_metric_labels"
    labels = np.resize(np.asarray(["", "a", "b"], dtype=object), datastore.cells.N)
    datastore.cells.insert(column_name=column, values=labels, overwrite=True)
    try:
        with pytest.raises(ValueError, match="contains missing values"):
            datastore.metric_ilisi(column, neighbors)
        with pytest.raises(ValueError, match="contains missing values"):
            datastore.metric_clisi(column, neighbors)
        with pytest.raises(ValueError, match="contains missing values"):
            datastore.metric_graph_connectivity(column, connectivity_graph)
        with pytest.raises(ValueError, match="contains missing values"):
            datastore.metric_proportional_batch_mixing(column, neighbors)
    finally:
        datastore.cells.drop(column)
    with pytest.raises(KeyError, match="'absent_metric_labels' was not found"):
        datastore.metric_ilisi("absent_metric_labels", neighbors)


@pytest.mark.parametrize(
    "mask_case",
    ("missing_array", "wrong_dtype", "wrong_shape", "non_string_link"),
)
def test_datastore_metrics_reject_malformed_metadata_missing_masks(
    datastore,
    connectivity_graph,
    mask_case,
):
    neighbors = _graph_neighbors(datastore, connectivity_graph)
    column = f"malformed_metric_mask_{mask_case}"
    datastore.cells.insert(
        column_name=column,
        values=np.zeros(datastore.cells.N, dtype=np.int8),
        overwrite=True,
    )
    cell_data = datastore.zw["cellData"]
    missing_name = f"__scarf_missing__{column}"
    if mask_case == "wrong_dtype":
        cell_data.create_array(
            missing_name,
            data=np.zeros(datastore.cells.N, dtype=np.int8),
        )
    elif mask_case == "wrong_shape":
        cell_data.create_array(
            missing_name,
            data=np.zeros(datastore.cells.N - 1, dtype=bool),
        )
    cell_data[column].attrs["missing_mask"] = (
        None if mask_case == "non_string_link" else missing_name
    )

    try:
        with pytest.raises(ValueError, match="missing-mask"):
            datastore.metric_ilisi(column, neighbors)
    finally:
        # The session datastore is shared, and copying rejects this column, so
        # remove it directly; drop() refuses a malformed missing-value link.
        for name in (column, missing_name):
            if name in cell_data:
                del cell_data[name]


def test_silhouette_scoring_missing_cluster_labels(datastore):
    result = silhouette_scoring(
        datastore,
        graph=None,
        hvg_data=None,
        assay_type="RNA",
        res_label="missing_resolution_label",
        distance_metric="l2",
    )
    assert result is None


class _LabelledCells:
    """Cell metadata holding one cluster column for silhouette scoring."""

    def __init__(self, labels: np.ndarray) -> None:
        self.labels = np.asarray(labels)
        self.columns = ["RNA_cluster"]

    def fetch(self, column, key="I"):
        assert (column, key) == ("RNA_cluster", "I")
        return self.labels


def _silhouette(labels, data, *, graph=None, **options):
    from types import SimpleNamespace

    labels = np.asarray(labels)
    if graph is None and "neighbor_indices" not in options:
        graph = csr_matrix(np.ones((len(labels), len(labels))) - np.eye(len(labels)))
    return silhouette_scoring(
        SimpleNamespace(cells=_LabelledCells(labels)),
        graph,
        np.asarray(data, dtype=np.float64),
        "RNA",
        "cluster",
        **({"distance_metric": "l2", "random_seed": 3} | options),
    )


def test_silhouette_scores_follow_cluster_separation():
    rng = np.random.default_rng(8)
    labels = np.repeat([0, 1], 12)
    # Two clusters of unit spread, far apart and then overlapping.
    separated = rng.normal(size=(24, 3)) + np.repeat([[0.0] * 3, [50.0] * 3], 12, 0)
    overlapping = rng.normal(size=(24, 3))

    far = _silhouette(labels, separated, sample_size=4)
    near = _silhouette(labels, overlapping, sample_size=4)

    assert far.shape == near.shape == (2,)
    assert np.all(far > 0.9)
    assert np.all(np.abs(near) < 0.5)
    # Identical cells have zero distance to both clusters, which scores zero.
    np.testing.assert_array_equal(
        _silhouette(labels, np.zeros((24, 3)), sample_size=4), [0.0, 0.0]
    )


def test_silhouette_accepts_streamed_neighbors_instead_of_a_graph():
    labels = np.repeat([0, 1], 6)
    data = (
        np.vstack([np.zeros((6, 2)), np.full((6, 2), 10.0)])
        + np.arange(12)[:, None] * 0.01
    )
    indices = np.array([[(cell + 1) % 12, (cell + 2) % 12] for cell in range(12)])

    streamed = _silhouette(
        labels,
        data,
        neighbor_indices=indices,
        neighbor_distances=np.ones(indices.shape),
        sample_size=3,
    )

    assert streamed.shape == (2,)
    assert np.all(streamed > 0.9)


def test_silhouette_scoring_validates_inputs():
    labels = np.repeat([0, 1], 4)
    data = np.arange(16, dtype=np.float64).reshape(8, 2)

    with pytest.raises(ValueError, match="sample_size must be greater than zero"):
        _silhouette(labels, data, sample_size=0)
    with pytest.raises(ValueError, match="Embedding data and cluster labels"):
        _silhouette(labels, data[:-1])
    with pytest.raises(ValueError, match="KNN graph and cluster labels"):
        _silhouette(labels, data, graph=csr_matrix((7, 7)))
    with pytest.raises(ValueError, match="Provide a KNN graph or neighbor indices"):
        _silhouette(labels, data, neighbor_indices=np.zeros((8, 2), dtype=int))
    with pytest.raises(ValueError, match="Unsupported neighbor metric: manhattan"):
        _silhouette(labels, data, distance_metric="manhattan")


def test_silhouette_needs_two_clusters():
    from scarf.utils import logger

    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="WARNING",
    )
    try:
        scores = _silhouette(
            np.zeros(6, dtype=int), np.random.default_rng(1).random((6, 2))
        )
    finally:
        logger.remove(sink)

    assert scores.shape == (1,)
    assert np.isnan(scores).all()
    assert messages == ["Silhouette scoring requires at least two clusters"]


def test_cluster_sampling_draws_disjoint_seeded_groups():
    from scarf.metrics.silhouette import _sample_cluster_rows, process_cluster

    data = np.arange(20, dtype=np.float64).reshape(10, 2)
    cells = np.array([1, 3, 4, 6, 8, 9])

    first, second = process_cluster(cells, data, 3)
    repeated = process_cluster(cells, data, 3, rng=np.random.default_rng(4444))

    # Row i of data is [2i, 2i + 1], so each sampled row names its cell. The
    # groups are read in row order and split the cluster between them.
    first_cells = (first[:, 0] // 2).astype(int).tolist()
    second_cells = (second[:, 0] // 2).astype(int).tolist()
    assert first_cells == sorted(first_cells)
    assert second_cells == sorted(second_cells)
    assert sorted(first_cells + second_cells) == cells.tolist()
    np.testing.assert_array_equal(first, repeated[0])
    np.testing.assert_array_equal(second, repeated[1])
    for k in (0, 4):
        with pytest.raises(ValueError, match="at least 2 \\* k cells"):
            process_cluster(cells, data, k)
    for count in (0, 7):
        with pytest.raises(ValueError, match="Sample count must fit"):
            _sample_cluster_rows(cells, data, count, np.random.default_rng(0))


def test_read_matrix_rows_reads_numpy_zarr_and_chunked_rows():
    from scarf.matrix import ChunkedArray
    from scarf.metrics._rows import read_matrix_rows

    values = np.arange(20.0).reshape(5, 4)
    stored = zarr.open_group(store=MemoryStore(), mode="w").create_array(
        "values", data=values, chunks=(2, 4)
    )
    rows = np.array([3, 0, 4])

    for source in (
        values,
        stored,
        ChunkedArray.from_numpy(values, block_size=2, nthreads=1),
    ):
        np.testing.assert_array_equal(read_matrix_rows(source, rows), values[rows])


@pytest.fixture(scope="module")
def metric_store_template(tmp_path_factory):
    """Forty cells in two expression groups with a PCA graph and clusterings."""
    from scarf import DataStore
    from tests.storage_helpers import write_count_store

    rng = np.random.default_rng(17)
    rates = np.ones((2, 30))
    rates[0, :10] = rates[1, 10:20] = 8.0
    counts = np.vstack([rng.poisson(rates[group], size=(20, 30)) for group in (0, 1)])
    zarr_loc = tmp_path_factory.mktemp("metric_store") / "store.zarr"
    write_count_store(str(zarr_loc), {"RNA": counts + 1}, "uint16")
    store = DataStore(
        str(zarr_loc), default_assay="RNA", min_features_per_cell=0, nthreads=1
    )
    cells = store.snapshot_cell_selection()
    normalized = store.run_normalization(
        cells, store.select_all_features(from_assay="RNA")
    )
    pca = store.run_pca(normalized, dims=3)
    neighbors = store.query_neighbors(store.build_ann_index(pca), k=5)
    graph = store.build_connectivity_map(neighbors)
    store.cells.insert("batch", np.arange(40) % 2)
    store.cells.insert("first_half", np.arange(40) < 20)
    groups = np.repeat([0, 1], 20)
    refs = {
        "cells": cells,
        "normalized": normalized,
        "pca": pca,
        "lsi": store.run_lsi(normalized, dims=2),
        "neighbors": neighbors,
        "graph": graph,
        "clusters": _clustering_artifact(store, groups, selection=cells),
        "subset_clusters": _clustering_artifact(
            store,
            groups[:20] + np.arange(20) % 2,
            selection=store.snapshot_cell_selection("first_half"),
        ),
        "datastore_clusters": _clustering_artifact(
            store, groups, selection=cells, scope="datastore"
        ),
    }
    return zarr_loc, refs


@pytest.fixture
def metric_store(metric_store_template, tmp_path):
    """A writable copy of the metric template."""
    import shutil

    from scarf import DataStore

    zarr_loc, refs = metric_store_template
    target = tmp_path / "store.zarr"
    shutil.copytree(zarr_loc, target)
    return (
        DataStore(
            str(target), default_assay="RNA", min_features_per_cell=0, nthreads=1
        ),
        refs,
    )


def _datastore_neighbors() -> ArtifactRef:
    from scarf.storage.artifacts import new_artifact_id

    return ArtifactRef(
        scope="datastore", kind="neighbors", artifact_id=new_artifact_id()
    )


def test_neighbor_metrics_require_an_assay_neighbors_reference(metric_store):
    from scarf.storage.errors import ArtifactResolutionError

    store, refs = metric_store

    for metric in (store.metric_ilisi, store.metric_clisi):
        with pytest.raises(TypeError, match="neighbors must be an artifact reference"):
            metric("batch", "neighbors")
        with pytest.raises(ArtifactResolutionError, match="neighbors artifact") as kind:
            metric("batch", refs["graph"])
        assert kind.value.code == "wrong_kind"
        with pytest.raises(ArtifactResolutionError, match="assay-scoped") as scope:
            metric("batch", _datastore_neighbors())
        assert scope.value.code == "wrong_scope"


def test_graph_connectivity_metric_checks_the_graph_reference_and_rows(metric_store):
    store, refs = metric_store
    payload = store.load_artifact(refs["graph"])
    weights = payload["weights"][:]
    # The stored labels are read for the graph's cells; zero weights are no edge.
    assert store.metric_graph_connectivity("batch", refs["graph"]) == pytest.approx(
        _materialized_graph_connectivity(
            payload["edges"][:][weights > 0], np.arange(40) % 2
        )
    )

    with pytest.raises(TypeError, match="graph must be an ArtifactRef"):
        store.metric_graph_connectivity("batch", "graph")
    store.zw[store.inspect_artifact(refs["graph"]).path].attrs["n_cells"] = 39
    with pytest.raises(ValueError, match="Graph labels must match the number of cells"):
        store.metric_graph_connectivity("batch", refs["graph"])


def test_graph_silhouette_requires_clusters_over_the_neighbor_cells(metric_store):
    store, refs = metric_store
    scores = store.metric_graph_silhouette(refs["neighbors"], refs["clusters"])
    # The two expression groups are the clusters, so both score positively.
    assert scores.shape == (2,)
    assert np.all(scores > 0)

    with pytest.raises(ValueError, match="use different cell selections"):
        store.metric_graph_silhouette(refs["neighbors"], refs["subset_clusters"])


def test_cluster_separability_metric_validates_its_arguments(metric_store):
    store, refs = metric_store
    pca, clusters = refs["pca"], refs["clusters"]

    for invalid, message in (
        ([clusters], "non-empty mapping of names to refs"),
        ({}, "non-empty mapping of names to refs"),
        ({"": clusters}, "cluster names must be non-empty strings"),
        ({"groups": "clusters"}, "cluster values must be ArtifactRefs"),
    ):
        with pytest.raises(TypeError, match=message):
            store.metric_cluster_separability(pca, invalid)
    with pytest.raises(ValueError, match="must reference a PCA reduction artifact"):
        store.metric_cluster_separability(refs["lsi"], {"groups": clusters})
    with pytest.raises(
        ValueError, match="assay-scoped clustering artifact for the PCA"
    ):
        store.metric_cluster_separability(pca, {"groups": refs["datastore_clusters"]})
    with pytest.raises(ValueError, match="does not use the PCA cell selection"):
        store.metric_cluster_separability(pca, {"groups": refs["subset_clusters"]})


def _replace_provenance_input(store, ref: ArtifactRef, name: str, value) -> None:
    group = store.zw[store.inspect_artifact(ref).path]
    provenance = dict(group.attrs["provenance"])
    inputs = dict(provenance["inputs"])
    if value is None:
        del inputs[name]
    else:
        inputs[name] = value
    provenance["inputs"] = inputs
    group.attrs["provenance"] = provenance


def _datastore_scoped_normalized(store, normalized: ArtifactRef) -> ArtifactRef:
    """Copy a normalized record's selection inputs into a datastore-scoped record."""
    from scarf.storage.artifact_writer import artifact_transaction, plan_artifact

    inputs = store.inspect_artifact(normalized).inputs
    planned = plan_artifact(
        store.zw,
        scope="datastore",
        kind="normalized",
        operation="run_normalization",
        parameters={},
        inputs={
            "cell_selection": ArtifactRef.from_dict(inputs["cell_selection"]),
            "feature_selection": ArtifactRef.from_dict(inputs["feature_selection"]),
        },
        execution_options={},
    )
    with artifact_transaction(store.zw, planned):
        pass
    return planned.ref


@pytest.mark.parametrize(
    "damage",
    [
        "no_coordinates",
        "short_coordinates",
        "no_feature_selection",
        "datastore_normalized",
    ],
)
def test_cluster_separability_metric_rejects_damaged_pca_records(metric_store, damage):
    from scarf.storage.errors import ArtifactResolutionError

    store, refs = metric_store
    pca_group = store.zw[store.inspect_artifact(refs["pca"]).path]
    error, message = ValueError, "PCA reduction coordinates are missing"
    if damage == "no_coordinates":
        del pca_group["data"]
    elif damage == "short_coordinates":
        pca_group.create_array(
            "data", data=np.ones((39, 3), np.float32), overwrite=True
        )
        message = "does not align with PCA rows"
    elif damage == "no_feature_selection":
        _replace_provenance_input(store, refs["normalized"], "feature_selection", None)
        error, message = ArtifactResolutionError, "missing its feature_selection input"
    else:
        # A normalized input outside every assay has no feature space.
        detached = _datastore_scoped_normalized(store, refs["normalized"])
        _replace_provenance_input(store, refs["pca"], "normalized", detached.to_dict())
        error, message = ArtifactResolutionError, "Normalized feature selection has no"

    with pytest.raises(error, match=message):
        store.metric_cluster_separability(refs["pca"], {"groups": refs["clusters"]})
