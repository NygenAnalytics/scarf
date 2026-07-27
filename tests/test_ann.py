import numpy as np

from scarf.neighbors.index import fix_knn_query


def test_fix_knn_query_removes_leading_self_matches():
    indices = np.array([[4, 1, 2], [7, 5, 6]])
    distances = np.array([[0.0, 0.2, 0.4], [0.0, 0.3, 0.6]])

    fixed_indices, fixed_distances, missed_self_hits = fix_knn_query(
        indices,
        distances,
        ref_idx=np.array([4, 7]),
    )

    assert missed_self_hits == 0
    assert np.array_equal(fixed_indices, [[1, 2], [5, 6]])
    assert np.allclose(fixed_distances, [[0.2, 0.4], [0.3, 0.6]])


def test_fix_knn_query_handles_each_self_neighbor_case():
    indices = np.array(
        [
            [0, 1, 2, 3],
            [2, 3, 1, 4],
            [0, 2, 3, 4],
        ]
    )
    distances = np.array(
        [
            [0.0, 0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
        ]
    )
    original_indices = indices.copy()
    original_distances = distances.copy()

    fixed_indices, fixed_distances, missed_self_hits = fix_knn_query(
        indices,
        distances,
        ref_idx=np.array([0, 1, 5]),
    )

    assert missed_self_hits == 1
    assert np.array_equal(
        fixed_indices,
        np.array(
            [
                [1, 2, 3],
                [2, 3, 4],
                [0, 2, 3],
            ]
        ),
    )
    assert np.allclose(
        fixed_distances,
        np.array(
            [
                [0.1, 0.2, 0.3],
                [0.1, 0.2, 0.4],
                [0.1, 0.2, 0.3],
            ]
        ),
    )
    assert np.array_equal(indices, original_indices)
    assert np.array_equal(distances, original_distances)


def test_fix_knn_query_removes_only_the_first_repeated_self_label():
    indices = np.array([[3, 8, 3, 9]])
    distances = np.array([[0.0, 0.1, 0.2, 0.3]])

    fixed_indices, fixed_distances, missed_self_hits = fix_knn_query(
        indices,
        distances,
        ref_idx=np.array([3]),
    )

    assert missed_self_hits == 0
    np.testing.assert_array_equal(fixed_indices, [[8, 3, 9]])
    np.testing.assert_allclose(fixed_distances, [[0.1, 0.2, 0.3]])
