import numpy as np
import pytest

from scarf.clustering.cluster_tree import CoalesceTree, make_digraph


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


def test_make_digraph_preserves_linkage_topology_and_leaf_clusters() -> None:
    dendrogram = _balanced_linkage()
    clusters = np.asarray([0, 0, 1, 1, 2, 2, 3, 3])

    graph = make_digraph(dendrogram, clust_info=clusters)

    assert set(graph.nodes) == set(range(15))
    assert set(graph.edges) == {
        (8, 0),
        (8, 1),
        (9, 2),
        (9, 3),
        (10, 4),
        (10, 5),
        (11, 6),
        (11, 7),
        (12, 8),
        (12, 9),
        (13, 10),
        (13, 11),
        (14, 12),
        (14, 13),
    }
    assert [graph.nodes[leaf]["cluster"] for leaf in range(8)] == clusters.tolist()
    assert graph.nodes[14]["nleaves"] == 8
    assert graph.nodes[14]["dist"] == 10


def test_make_digraph_rejects_mismatched_cluster_info() -> None:
    dendrogram = _balanced_linkage()
    with pytest.raises(ValueError, match="cluster information"):
        make_digraph(dendrogram, clust_info=np.zeros(3))


def test_coalesce_tree_retains_cluster_holding_nodes_and_ancestors() -> None:
    clusters = np.asarray([0, 0, 1, 1, 2, 2, 3, 3])
    graph = make_digraph(_balanced_linkage(), clust_info=clusters)

    coalesced = CoalesceTree(graph, clusters)

    assert set(coalesced.nodes) == set(range(8, 15))
    assert set(coalesced.edges) == {
        (12, 8),
        (12, 9),
        (13, 10),
        (13, 11),
        (14, 12),
        (14, 13),
    }
    assert {
        int(node): int(attributes["partition_id"])
        for node, attributes in coalesced.nodes(data=True)
        if "partition_id" in attributes
    } == {8: 0, 9: 1, 10: 2, 11: 3}


def test_coalesce_tree_rejects_non_monophyletic_clusters() -> None:
    clusters = np.asarray([0, 1, 0, 1, 2, 2, 3, 3])
    graph = make_digraph(_balanced_linkage(), clust_info=clusters)

    with pytest.raises(ValueError, match="not monophyletic"):
        CoalesceTree(graph, clusters)
