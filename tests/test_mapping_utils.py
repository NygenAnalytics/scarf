import numpy as np
import pytest

from scarf.mapping.confidence import (
    _distance_quantile_summary,
    conformal_prediction_sets,
    distance_weights,
    mapping_score_weights,
)
from scarf.mapping.features import _feature_ids
from scarf.mapping.hashing import array_hash, array_store_hash


def test_mapping_array_hashes_match_golden_values():
    numeric = np.array([[1.5, -2.0], [0.0, 4.25]], dtype=np.float64)
    identifiers = np.array(["gene_a", "gene_b"], dtype="<U6")

    assert (
        array_hash(numeric)
        == "ab342d324ef6a1e409055ac301654f1472817f569ab4c0df2c10464bad546a49"
    )
    assert (
        array_store_hash(numeric)
        == "ab342d324ef6a1e409055ac301654f1472817f569ab4c0df2c10464bad546a49"
    )
    assert (
        array_hash(identifiers)
        == "7b628da091bfd6a5c5f8c6dd644c3fefc482779799a9e51018f9f0c290215ad2"
    )
    assert (
        array_store_hash(identifiers)
        == "7b628da091bfd6a5c5f8c6dd644c3fefc482779799a9e51018f9f0c290215ad2"
    )


def test_mapping_score_weights_stay_absolute_across_query_cells():
    near = np.full((1, 6), 1.0)
    far = np.full((1, 6), 100.0)

    near_weights = mapping_score_weights(near)
    far_weights = mapping_score_weights(far)

    np.testing.assert_allclose(near_weights, 1.0 / (np.log1p(1.0) + 1.0))
    np.testing.assert_allclose(far_weights, 1.0 / (np.log1p(100.0) + 1.0))
    # A query cell far from the reference must deposit less total weight than one
    # that lands on it. Row normalization would make both sums equal and reduce
    # the mapping score to a neighbor count.
    assert far_weights.sum() < 0.4 * near_weights.sum()
    assert mapping_score_weights(np.zeros((1, 3))).tolist() == [[1.0, 1.0, 1.0]]

    with pytest.raises(ValueError, match="non-negative"):
        mapping_score_weights(np.array([[-1.0, 1.0]]))
    with pytest.raises(ValueError, match="finite"):
        mapping_score_weights(np.array([[np.nan, 1.0]]))
    with pytest.raises(ValueError, match="two-dimensional"):
        mapping_score_weights(np.array([1.0, 2.0]))


def test_distance_weights_use_metric_distances_and_split_zero_ties():
    weights = distance_weights(
        np.array(
            [
                [1.0, 9.0, 9.0],
                [0.0, 0.0, 4.0],
                [1.0, 1.0, 1.0],
            ]
        )
    )

    np.testing.assert_allclose(weights[0], [0.8181818181818182, 1 / 11, 1 / 11])
    np.testing.assert_allclose(weights[1], [0.5, 0.5, 0.0])
    np.testing.assert_allclose(weights[2], [1 / 3, 1 / 3, 1 / 3])
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)


def test_distance_weights_keep_subnormal_positive_rows_finite():
    smallest = np.nextafter(0.0, 1.0)
    next_smallest = np.nextafter(smallest, 1.0)

    weights = distance_weights(
        np.array(
            [
                [1.0, 9.0],
                [smallest, next_smallest],
                [smallest, np.finfo(np.float64).tiny],
            ]
        )
    )

    np.testing.assert_allclose(weights[0], [0.9, 0.1])
    np.testing.assert_allclose(weights[1], [2 / 3, 1 / 3])
    assert np.all(np.isfinite(weights))
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)


def test_distance_weights_reject_invalid_metric_distances():
    with pytest.raises(ValueError, match="non-negative"):
        distance_weights(np.array([[-1.0, 1.0]]))
    with pytest.raises(ValueError, match="finite"):
        distance_weights(np.array([[np.nan, 1.0]]))


def test_distance_quantile_summary_handles_vectors_and_neighbor_matrices():
    first_neighbors = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
    neighbor_matrix = np.column_stack((first_neighbors, first_neighbors + 1))

    vector_summary = _distance_quantile_summary(
        first_neighbors,
        max_samples=3,
        n_quantiles=3,
    )
    matrix_summary = _distance_quantile_summary(
        neighbor_matrix,
        max_samples=3,
        n_quantiles=3,
    )

    np.testing.assert_allclose(vector_summary[0], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(vector_summary[1], [0.0, 4.0, 16.0])
    np.testing.assert_allclose(matrix_summary[0], vector_summary[0])
    np.testing.assert_allclose(matrix_summary[1], vector_summary[1])


def test_conformal_prediction_sets_include_high_score_labels():
    sets = conformal_prediction_sets(
        np.array([[0.95, 0.1], [0.7, 0.7]]),
        np.array([0.05, 0.1, 0.2, 0.25]),
        alpha=0.2,
    )

    assert sets.shape == (2, 2)
    assert sets[0, 0]
    assert not sets[0, 1]


def test_feature_identifier_order_is_preserved():
    identifiers = _feature_ids(
        np.array(["gene_b", "gene_a", "gene_c"]),
        name="Reference feature identifiers",
    )

    np.testing.assert_array_equal(identifiers, ["gene_b", "gene_a", "gene_c"])
    assert not identifiers.flags.writeable


def test_feature_alignment_rejects_duplicate_identifiers():
    with pytest.raises(ValueError, match="unique"):
        _feature_ids(
            np.array(["gene_a", "gene_a"]),
            name="Reference feature identifiers",
        )
