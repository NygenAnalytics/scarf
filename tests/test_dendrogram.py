import numpy as np
import pytest
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from scarf.dendrogram import BalancedCut, CoalesceTree, make_digraph


def test_make_digraph_builds_expected_node_count():
    rng = np.random.default_rng(0)
    points = rng.normal(size=(6, 4))
    dendrogram = linkage(pdist(points), method="average")
    graph = make_digraph(dendrogram)
    assert graph.number_of_edges() == dendrogram.shape[0] * 2
    assert graph.number_of_nodes() >= len(points)


def test_make_digraph_rejects_mismatched_cluster_info():
    rng = np.random.default_rng(1)
    points = rng.normal(size=(5, 3))
    dendrogram = linkage(pdist(points), method="single")
    with pytest.raises(ValueError, match="cluster information"):
        make_digraph(dendrogram, clust_info=np.zeros(3))


def test_coalesce_tree_reduces_hierarchy():
    rng = np.random.default_rng(2)
    points = rng.normal(size=(8, 3))
    dendrogram = linkage(pdist(points), method="average")
    clusters = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    graph = make_digraph(dendrogram, clust_info=clusters)
    coalesced = CoalesceTree(graph, clusters)
    assert coalesced.number_of_nodes() <= graph.number_of_nodes()


def test_balanced_cut_assigns_all_cells():
    rng = np.random.default_rng(3)
    points = rng.normal(size=(8, 3))
    dendrogram = linkage(pdist(points), method="average")
    cutter = BalancedCut(
        dendrogram,
        max_size=4,
        min_size=1,
        max_distance_fc=2.0,
    )
    labels = cutter.get_clusters()
    assert len(labels) == len(points)
    assert labels.min() >= -1
