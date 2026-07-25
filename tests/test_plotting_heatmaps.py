import matplotlib
import networkx as nx
import pytest

matplotlib.use("Agg")

import scarf.plotting as splt
from scarf.plotting.heatmaps import _hierarchy_positions


def test_hierarchy_positions_are_pure_and_complete():
    graph = nx.DiGraph([(4, 2), (4, 3), (2, 0), (2, 1)])
    before = graph.copy()

    positions = _hierarchy_positions(graph, width=2.0)

    assert set(positions) == set(graph)
    assert nx.utils.graphs_equal(graph, before)
    assert positions[4][1] == 0.0
    assert positions[0][1] < positions[2][1]


def test_marker_heatmap_returns_owned_result(
    marker_search,
    datastore,
):
    result = splt.marker_heatmap(
        datastore,
        group_key="RNA_cluster",
        topn=3,
        figsize=(4, 6),
        show=False,
    )

    assert isinstance(result, splt.PlotResult)
    assert result.owns_figure is True
    assert tuple(result.figure.get_size_inches()) == (4, 6)
    assert {"heatmap", "row_dendrogram", "column_dendrogram", "colorbar"} <= set(
        result.axes
    )
    assert not result.tables["matrix"].empty
    assert not result.tables["markers"].empty
    assert result.legends
    assert result.scales
    assert result.provenance.notes[0] == "marker_heatmap"
    result.close()


def test_cluster_tree_prepares_cache_and_returns_tables(
    paris_clustering,
    datastore,
):
    prepared = datastore._prepare_cluster_tree(cluster_key="RNA_cluster")
    assert prepared["coalesced_location"] in datastore.zw
    assert nx.is_tree(prepared["graph"])

    cached = datastore._prepare_cluster_tree(cluster_key="RNA_cluster")
    cached_labels = {
        data["partition_id"]
        for _, data in cached["graph"].nodes(data=True)
        if "partition_id" in data
    }
    assert cached_labels == set(paris_clustering)

    result = splt.cluster_tree(
        datastore,
        cluster_key="RNA_cluster",
        figsize=(4, 4),
        show=False,
    )

    assert isinstance(result, splt.PlotResult)
    assert result.owns_figure is True
    assert tuple(result.figure.get_size_inches()) == (4, 4)
    assert {"nodes", "edges", "positions", "cluster_summary"} <= set(result.tables)
    assert len(result.tables["cluster_summary"]) == len(set(paris_clustering))
    assert result.provenance.extras["coalesced_location"] in datastore.zw
    result.close()


def test_pseudotime_heatmap_returns_aligned_tables(
    pseudotime_aggregation,
    datastore,
):
    result = splt.pseudotime_heatmap(
        datastore,
        cell_key="I",
        feat_key="I",
        feature_cluster_key="pseudotime_clusters",
        pseudotime_key="RNA_pseudotime",
        show_features=["Wsb1", "Rest"],
        figsize=(4, 6),
        show=False,
    )

    assert isinstance(result, splt.PlotResult)
    assert result.owns_figure is True
    assert tuple(result.figure.get_size_inches()) == (4, 6)
    assert set(result.axes) == {
        "heatmap",
        "feature_clusters",
        "colorbar",
        "pseudotime",
    }
    assert result.tables["matrix"].shape[0] == len(result.tables["features"])
    assert result.tables["matrix"].shape[1] == len(result.tables["pseudotime_bins"])
    assert len(result.tables["pseudotime"]) == result.provenance.n_cells
    assert result.legends
    assert result.scales
    result.close()


def test_pseudotime_heatmap_rejects_malformed_artifact_link(
    pseudotime_aggregation,
    datastore,
):
    column = datastore.RNA.z["featureData/pseudotime_clusters"]
    original_ref = dict(column.attrs["source_artifact"])
    column.attrs["source_artifact"] = "broken"
    try:
        with pytest.raises(ValueError, match="malformed source artifact"):
            splt.pseudotime_heatmap(
                datastore,
                cell_key="I",
                feat_key="I",
                feature_cluster_key="pseudotime_clusters",
                pseudotime_key="RNA_pseudotime",
                show=False,
            )
    finally:
        column.attrs["source_artifact"] = original_ref
