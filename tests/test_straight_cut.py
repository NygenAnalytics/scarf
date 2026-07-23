import numpy as np
import pytest
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sknetwork.hierarchy import cut_straight as reference_straight_cut

from scarf.clustering.paris import straight_cut


@pytest.mark.parametrize("n_leaves", [3, 10, 40])
def test_straight_cut_matches_scikit_network(n_leaves: int) -> None:
    rng = np.random.default_rng(100 + n_leaves)
    points = rng.normal(size=(n_leaves, 4))
    dendrogram = linkage(pdist(points), method="average")
    dendrogram[:, 2] = np.round(dendrogram[:, 2], 1)

    for n_clusters in range(2, n_leaves + 1):
        expected = reference_straight_cut(
            dendrogram,
            n_clusters=n_clusters,
        )
        assert np.array_equal(
            straight_cut(dendrogram, n_clusters),
            expected + 1,
        )


def test_straight_cut_matches_non_monotonic_dendrogram() -> None:
    dendrogram = np.asarray(
        [
            [0, 1, 0.5, 2],
            [2, 3, 0.5, 2],
            [4, 5, 1.5, 2],
            [6, 7, 2.0, 4],
            [8, 9, 1.0, 6],
        ],
        dtype=np.float64,
    )

    for n_clusters in range(2, 7):
        expected = reference_straight_cut(
            dendrogram,
            n_clusters=n_clusters,
        )
        assert np.array_equal(
            straight_cut(dendrogram, n_clusters),
            expected + 1,
        )


def test_equal_heights_can_return_more_clusters_than_requested() -> None:
    dendrogram = np.asarray(
        [
            [0, 1, 1, 2],
            [2, 3, 1, 2],
            [4, 5, 2, 4],
        ],
        dtype=np.float64,
    )

    labels = straight_cut(dendrogram, n_clusters=3)

    assert labels.tolist() == [1, 2, 3, 4]


def test_one_cluster_and_single_leaf_are_supported() -> None:
    dendrogram = np.asarray(
        [
            [0, 1, 1, 2],
            [2, 3, 2, 3],
        ],
        dtype=np.float64,
    )
    assert straight_cut(dendrogram, n_clusters=1).tolist() == [1, 1, 1]
    assert straight_cut(np.empty((0, 4)), n_clusters=1).tolist() == [1]


@pytest.mark.parametrize("n_clusters", [0, 4])
def test_straight_cut_rejects_out_of_range_cluster_counts(
    n_clusters: int,
) -> None:
    dendrogram = np.asarray(
        [
            [0, 1, 1, 2],
            [2, 3, 2, 3],
        ],
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match="between 1"):
        straight_cut(dendrogram, n_clusters)


def test_straight_cut_validates_types_and_shape() -> None:
    dendrogram = np.asarray(
        [
            [0, 1, 1, 2],
            [2, 3, 2, 3],
        ],
        dtype=np.float64,
    )
    with pytest.raises(TypeError, match="integer"):
        straight_cut(dendrogram, True)
    with pytest.raises(ValueError, match="shape"):
        straight_cut(np.ones((2, 3)), 2)


@pytest.mark.parametrize(
    ("children", "message"),
    [
        ([[np.nan, 1], [2, 3]], "finite integers"),
        ([[np.inf, 1], [2, 3]], "finite integers"),
        ([[0.5, 1], [2, 3]], "finite integers"),
        ([[-1, 1], [2, 3]], "non-negative"),
        ([[0, 0], [2, 3]], "two distinct children"),
        ([[0, 3], [1, 2]], "only earlier nodes"),
        ([[0, 1], [0, 3]], "consumed more than once"),
    ],
)
def test_straight_cut_rejects_malformed_children_before_one_cluster_return(
    children: list[list[float]],
    message: str,
) -> None:
    dendrogram = np.asarray(
        [
            [*children[0], 1, 2],
            [*children[1], 2, 3],
        ],
        dtype=np.float64,
    )

    with pytest.raises(ValueError, match=message):
        straight_cut(dendrogram, n_clusters=1)
