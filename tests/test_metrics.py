import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from scarf.metrics import (
    calculate_knn_cluster_similarity,
    calculate_top_k_neighbor_distances,
    calculate_weighted_cluster_similarity,
    compute_lisi,
    integration_score,
    knn_to_csr_matrix,
    label_concordance_score,
    lisi_batch_mixing_score,
    silhouette_scoring,
)


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


def test_metric_lisi_single_category_is_one(datastore, make_graph):
    labels = np.zeros(datastore.cells.N, dtype=np.int8)
    datastore.cells.insert(
        column_name="single_batch",
        values=labels,
        overwrite=True,
    )
    lisi = datastore.metric_lisi(
        label_colnames=(name for name in ["single_batch"]),
        save_result=False,
        return_lisi=True,
    )

    assert lisi is not None
    assert np.allclose(lisi[0][1], 1)


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


def test_top_k_distances_accept_all_candidates():
    distances = calculate_top_k_neighbor_distances(
        np.array([[0.0]]),
        np.array([[1.0], [2.0]]),
        k=2,
    )

    assert np.allclose(np.sort(distances[0]), [1.0, 2.0])


def test_metric_silhouette(datastore, make_graph, leiden_clustering):
    scores = datastore.metric_silhouette(random_seed=42)
    scores_from_full_label = datastore.metric_silhouette(
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
    assert integration_score([labels, labels], metric) == pytest.approx(1)


def test_label_concordance_requires_exactly_two_partitions():
    with pytest.raises(ValueError, match="Exactly two"):
        label_concordance_score([np.array([0, 1])])


def test_lisi_batch_mixing_score():
    labels = np.array([0, 0, 1, 1])

    assert lisi_batch_mixing_score(np.ones(4), labels) == pytest.approx(0)
    assert lisi_batch_mixing_score(np.full(4, 2.0), labels) == pytest.approx(1)


def test_metric_label_concordance(datastore, make_graph, leiden_clustering):
    rng = np.random.default_rng(42)
    labels1 = rng.integers(0, 2, datastore.cells.N)
    labels2 = labels1.copy()
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

    assert datastore.metric_label_concordance(["labels1", "labels2"]) == pytest.approx(
        1
    )
    assert datastore.metric_integration(["labels1", "labels2"]) == pytest.approx(1)
    mixing_score = datastore.metric_batch_mixing("labels1")
    assert 0 <= mixing_score <= 1


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
