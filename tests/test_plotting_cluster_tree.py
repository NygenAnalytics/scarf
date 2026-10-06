"""Cluster tree drawing from a prepared hierarchy."""

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest
from matplotlib.colors import to_hex

import scarf.plotting as splt
from scarf.storage import ArtifactRef

_GRAPH = ArtifactRef(
    scope="assay", assay="RNA", kind="connectivity_map", artifact_id="1" * 64
)
_CLUSTERS = ArtifactRef(
    scope="assay", assay="RNA", kind="cluster_cut", artifact_id="2" * 64
)


def _tree_store(color_values, color_missing=None):
    """A root over clusters 1 and 2, each holding three of six cells."""
    graph = nx.DiGraph([(0, 1), (0, 2)])
    nx.set_node_attributes(graph, {0: 6, 1: 3, 2: 3}, "nleaves")
    nx.set_node_attributes(graph, {1: 1, 2: 2}, "partition_id")
    prepared = {
        "graph": graph,
        "clusters": np.array([1, 1, 1, 2, 2, 2]),
        "color_values": None if color_values is None else np.asarray(color_values),
        "color_missing": None if color_missing is None else np.asarray(color_missing),
        "coalesced_location": "tree",
        "from_assay": "RNA",
        "graph_ref": _GRAPH,
        "clusters_ref": _CLUSTERS,
        "cell_selection": ArtifactRef(
            scope="datastore", kind="cell_selection", artifact_id="3" * 64
        ),
    }
    return SimpleNamespace(_prepare_cluster_tree=lambda **_kwargs: prepared)


def _node_colors(result):
    """Face colors of the hierarchy nodes in graph node order (0, 1, 2)."""
    (nodes,) = [
        collection
        for collection in result.axes["tree"].collections
        if len(collection.get_offsets()) == 3
    ]
    return [to_hex(color) for color in nodes.get_facecolors()]


def test_cluster_tree_draws_value_pies_and_skips_clusters_without_values():
    result = splt.cluster_tree(
        _tree_store(
            ["A", "B", "A", "C", "C", "C"],
            color_missing=[False, False, False, True, True, True],
        ),
        graph=_GRAPH,
        clusters=_CLUSTERS,
        fill_by_value="phase",
        color_key={"A": "#ff0000", "B": "#0000ff"},
        show_labels=False,
        show=False,
    )

    positions = result.tables["positions"].set_index("node")
    wedges = [
        collection
        for collection in result.axes["tree"].collections
        if len(collection.get_offsets()) == 1
    ]
    # Cluster 1 holds two A cells and one B cell; every value of cluster 2 is
    # missing, so it draws no wedge.
    assert [to_hex(wedge.get_facecolors()[0]) for wedge in wedges] == [
        "#ff0000",
        "#0000ff",
    ]
    for wedge in wedges:
        np.testing.assert_allclose(
            wedge.get_offsets()[0], positions.loc[1, ["x", "y"]].to_numpy(float)
        )
    # Value pies replace the cluster fills, which stay white and hidden.
    assert _node_colors(result)[1:] == ["#ffffff", "#ffffff"]
    result.close()


def test_cluster_tree_shows_owned_results_by_default(monkeypatch):
    shown = []
    monkeypatch.setattr(splt.PlotResult, "show", lambda result: shown.append(result))

    result = splt.cluster_tree(
        _tree_store(None), graph=_GRAPH, clusters=_CLUSTERS, show_labels=False
    )

    assert shown == [result]
    assert result.tables["cluster_summary"]["n_cells"].tolist() == [3, 3]
    result.close()
    assert not plt.fignum_exists(result.figure.number)


def test_cluster_tree_keeps_missing_clusters_grey_under_a_uniform_fill():
    def node_colors(values, missing):
        result = splt.cluster_tree(
            _tree_store(values, color_missing=missing),
            graph=_GRAPH,
            clusters=_CLUSTERS,
            fill_by_value="score",
            force_ints_as_cats=False,
            show_labels=False,
            show=False,
        )
        try:
            return _node_colors(result)
        finally:
            result.close()

    uniform = node_colors([5.0] * 6, [False] * 6)
    partly_missing = node_colors(
        [5.0, 5.0, 5.0, 0.0, 0.0, 0.0], [False, False, False, True, True, True]
    )

    # Both clusters share the one observed value's color when nothing is
    # missing; a cluster whose values are all missing shows as missing, and
    # the observed cluster keeps its color.
    assert uniform[1] == uniform[2] != splt.ColorScale().missing_color
    assert partly_missing[1] == uniform[1]
    assert partly_missing[2] == splt.ColorScale().missing_color


def test_cluster_tree_keeps_a_constant_fill_value():
    result = splt.cluster_tree(
        _tree_store([5.0] * 6),
        graph=_GRAPH,
        clusters=_CLUSTERS,
        fill_by_value="score",
        force_ints_as_cats=False,
        cmap="viridis",
        show_labels=False,
        show=False,
    )
    try:
        (scale, _sizes) = result.scales
        assert isinstance(scale, splt.ColorScale)
        assert (scale.vmin, scale.vmax) == (5.0, 6.0)
        (legend,) = result.legends
        assert legend.kind == "colorbar"
        assert (legend.extras["vmin"], legend.extras["vmax"]) == (5.0, 6.0)
        assert result.axes["colorbar"].get_ylim() == pytest.approx((5.0, 6.0))
        # The shared policy places a constant value at the low end of the map.
        low = to_hex(matplotlib.colormaps["viridis"](0.0))
        assert _node_colors(result)[1:] == [low, low]
    finally:
        result.close()


def test_cluster_tree_keeps_a_single_category_categorical():
    result = splt.cluster_tree(
        _tree_store(np.asarray(["A"] * 6)),
        graph=_GRAPH,
        clusters=_CLUSTERS,
        fill_by_value="phase",
        show_labels=False,
        show=False,
    )
    try:
        (scale, _sizes) = result.scales
        assert isinstance(scale, splt.CategoricalScale)
        assert scale.order == ("A",)
        assert list(scale.palette) == ["A"]
        (legend,) = result.legends
        assert legend.kind == "categorical"
        assert "colorbar" not in result.axes
        wedges = [
            collection
            for collection in result.axes["tree"].collections
            if len(collection.get_offsets()) == 1
        ]
        # Each cluster draws one full wedge in the category's color.
        assert [to_hex(wedge.get_facecolors()[0]) for wedge in wedges] == [
            scale.palette["A"]
        ] * 2
    finally:
        result.close()
