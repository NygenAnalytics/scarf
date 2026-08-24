import numpy as np

from scarf.clustering.balanced_cut import BalancedCut


def _balanced_linkage() -> np.ndarray:
    return np.asarray(
        [
            [0, 1, 1, 2],
            [2, 3, 1, 2],
            [4, 5, 1, 2],
            [6, 7, 1, 2],
            [8, 9, 2, 4],
            [10, 11, 2, 4],
            [12, 13, 10, 8],
        ],
        dtype=np.float64,
    )


def test_balanced_cut_respects_maximum_cluster_size() -> None:
    cutter = BalancedCut(
        _balanced_linkage(),
        max_size=4,
        min_size=1,
        max_distance_fc=2.0,
    )
    labels = cutter.get_clusters()

    assert np.array_equal(labels[:4], np.full(4, labels[0]))
    assert np.array_equal(labels[4:], np.full(4, labels[4]))
    assert labels[0] != labels[4]
    assert np.bincount(labels)[1:].tolist() == [4, 4]


def test_balanced_cut_accepts_root_sized_cluster() -> None:
    cutter = BalancedCut(
        _balanced_linkage(),
        max_size=8,
        min_size=1,
        max_distance_fc=2.0,
    )

    assert cutter.get_clusters().tolist() == [1] * 8
