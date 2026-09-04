import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection
from scipy import sparse

from scarf.plotting._contracts import CategoricalScale, SizeScale
from scarf.plotting._figure import PlotResult
from scarf.plotting.cluster_connectivity import cluster_connectivity
from scarf.storage.artifacts import ArtifactRef


class _FakeCells:
    def __init__(self, values):
        self.values = values
        self.fetch_calls = []

    def fetch(self, column, *, key):
        self.fetch_calls.append((column, key))
        return self.values[column]


class _FakeStore:
    _defaultAssay = "RNA"

    def __init__(self, values, graph):
        self.cells = _FakeCells(values)
        self.graph = graph
        self.graph_calls = []
        self.zw = object()

    def load_graph(self, graph=None, **kwargs):
        self.graph_calls.append((graph, kwargs))
        return self.graph


def _symmetric_graph(n_cells, weighted_edges):
    rows = []
    columns = []
    data = []
    for source, target, weight in weighted_edges:
        rows.extend((source, target))
        columns.extend((target, source))
        data.extend((weight, weight))
    return sparse.csr_matrix((data, (rows, columns)), shape=(n_cells, n_cells))


def _store():
    values = {
        "layout1": np.array([0.0, 0.0, 9.0, 2.0, 2.0, 8.0, 4.0, 4.0, 10.0]),
        "layout2": np.array([0.0, 0.0, 3.0, 1.0, 1.0, 4.0, 0.0, 0.0, -3.0]),
        "cluster": np.array(["A", "A", "A", "B", "B", "B", "C", "C", "C"]),
    }
    graph = _symmetric_graph(
        9,
        [
            (0, 1, 1.0),
            (3, 4, 1.0),
            (6, 7, 1.0),
            (0, 3, 0.6),
            (1, 4, 0.4),
            (0, 6, 0.2),
            (3, 6, 0.8),
        ],
    )
    return _FakeStore(values, graph)


def _plot(store, **kwargs):
    from unittest.mock import patch

    graph_ref = kwargs.pop(
        "graph",
        ArtifactRef(
            scope="assay",
            assay="RNA",
            kind="connectivity_map",
            artifact_id="1" * 64,
        ),
    )
    cell_key = kwargs.pop("cell_key", "I")
    categorical_scale = kwargs.pop("categorical_scale", CategoricalScale())
    with (
        patch(
            "scarf.plotting.cluster_connectivity.graph_cell_selection",
            return_value=ArtifactRef(
                scope="datastore",
                kind="cell_selection",
                artifact_id="2" * 64,
            ),
        ),
        patch(
            "scarf.plotting.cluster_connectivity.validate_stored_selection_live_alias"
        ),
    ):
        return cluster_connectivity(
            store,
            group_by="cluster",
            layout_key="layout",
            graph=graph_ref,
            cell_key=cell_key,
            categorical_scale=categorical_scale,
            show=False,
            **kwargs,
        )


def test_cluster_connectivity_aggregates_reciprocals_once():
    store = _store()
    result = _plot(
        store,
        cell_key="selected",
        minimum_edge_weight=0.0,
    )

    assert isinstance(result, PlotResult)
    assert result.owns_figure is True
    assert list(result.axes) == ["cluster_connectivity"]
    assert store.cells.fetch_calls == [
        ("layout1", "selected"),
        ("layout2", "selected"),
        ("cluster", "selected"),
    ]
    assert store.graph_calls == [
        (
            ArtifactRef(
                scope="assay",
                assay="RNA",
                kind="connectivity_map",
                artifact_id="1" * 64,
            ),
            {"symmetric": True},
        )
    ]

    nodes = result.tables["nodes"]
    assert list(nodes.columns) == [
        "category",
        "x",
        "y",
        "nCells",
        "proportion",
        "size",
        "displayLabel",
    ]
    assert nodes["category"].tolist() == ["A", "B", "C"]
    np.testing.assert_allclose(nodes["x"], [0.0, 2.0, 4.0])
    np.testing.assert_allclose(nodes["y"], [0.0, 1.0, 0.0])
    assert nodes["nCells"].tolist() == [3, 3, 3]
    np.testing.assert_allclose(nodes["proportion"], [1 / 3] * 3)

    edges = result.tables["edges"]
    assert list(edges.columns) == [
        "source",
        "target",
        "rawWeight",
        "normalizedWeight",
    ]
    assert list(zip(edges["source"], edges["target"], strict=True)) == [
        ("A", "B"),
        ("A", "C"),
        ("B", "C"),
    ]
    np.testing.assert_allclose(edges["rawWeight"], [1.0, 0.2, 0.8])
    np.testing.assert_allclose(
        edges["normalizedWeight"],
        [
            1.0 / np.sqrt(3.2 * 3.8),
            0.2 / np.sqrt(3.2 * 3.0),
            0.8 / np.sqrt(3.8 * 3.0),
        ],
    )
    pairs = [frozenset(pair) for pair in zip(edges["source"], edges["target"])]
    assert len(pairs) == len(set(pairs))
    assert not (edges["source"] == edges["target"]).any()

    assert isinstance(result.scales[0], CategoricalScale)
    assert isinstance(result.scales[1], SizeScale)
    assert result.provenance.n_cells == 9
    assert result.provenance.extras["n_aggregated_edges"] == 3
    assert result.provenance.extras["n_edges"] == 3
    assert "sqrt" in result.provenance.extras["normalization"]
    ax = result.axes["cluster_connectivity"]
    assert isinstance(ax.collections[0], LineCollection)
    assert ax.collections[0].get_zorder() < ax.collections[1].get_zorder()
    assert len(ax.get_xticks()) == 0
    assert len(ax.get_yticks()) == 0
    assert ax.get_box_aspect() == pytest.approx(1.0)
    result.close()


def test_cluster_connectivity_threshold_and_degree_cap_are_deterministic():
    thresholded = _plot(_store(), minimum_edge_weight=0.25)
    assert thresholded.tables["edges"][["source", "target"]].values.tolist() == [
        ["A", "B"]
    ]
    thresholded.close()

    capped = _plot(
        _store(),
        minimum_edge_weight=0.0,
        max_edges_per_node=1,
    )
    assert capped.tables["edges"][["source", "target"]].values.tolist() == [["A", "B"]]
    capped.close()


def test_cluster_connectivity_explicit_positions_scales_and_labels():
    categorical_scale = CategoricalScale(
        order=("C", "A", "B"),
        palette={"A": "#aa0000", "B": "#00aa00", "C": "#0000aa"},
        labels={"A": "Alpha", "C": "Gamma"},
    )
    size_scale = SizeScale(
        vmin=0.0,
        vmax=0.5,
        size_min=20.0,
        size_max=80.0,
    )
    positions = {
        "A": (10.0, 11.0),
        "B": (20.0, 21.0),
        "C": (30.0, 31.0),
    }

    result = _plot(
        _store(),
        positions=positions,
        categorical_scale=categorical_scale,
        size_scale=size_scale,
        minimum_edge_weight=0.0,
    )

    nodes = result.tables["nodes"]
    assert nodes["category"].tolist() == ["C", "A", "B"]
    np.testing.assert_allclose(nodes["x"], [30.0, 10.0, 20.0])
    np.testing.assert_allclose(nodes["y"], [31.0, 11.0, 21.0])
    assert nodes["displayLabel"].tolist() == ["Gamma", "Alpha", "B"]
    np.testing.assert_allclose(nodes["size"], [60.0, 60.0, 60.0])
    assert result.scales[0].order == ("C", "A", "B")
    assert result.scales[0].labels == {
        "C": "Gamma",
        "A": "Alpha",
        "B": "B",
    }
    assert result.scales[1] is size_scale
    assert [text.get_text() for text in result.axes["cluster_connectivity"].texts] == [
        "Gamma",
        "Alpha",
        "B",
    ]
    font_sizes = [
        text.get_fontsize() for text in result.axes["cluster_connectivity"].texts
    ]
    assert font_sizes[0] < font_sizes[2]
    result.close()


def test_cluster_connectivity_mean_positions():
    result = _plot(_store(), position="mean")
    nodes = result.tables["nodes"]
    np.testing.assert_allclose(nodes["x"], [3.0, 4.0, 6.0])
    np.testing.assert_allclose(nodes["y"], [1.0, 2.0, -1.0])
    result.close()


def test_cluster_connectivity_uses_caller_owned_target_and_optional_cells():
    figure, ax = plt.subplots()
    result = _plot(
        _store(),
        target=ax,
        show_cells=True,
        labels=False,
    )

    assert result.figure is figure
    assert result.axes == {"cluster_connectivity": ax}
    assert result.owns_figure is False
    assert [artist.get_zorder() for artist in ax.collections] == [0, 1, 2]
    result.close()
    assert plt.fignum_exists(figure.number)
    plt.close(figure)


@pytest.mark.parametrize(
    ("column", "replacement", "message"),
    [
        ("layout1", np.array([0.0] * 8), "matching lengths"),
        (
            "layout2",
            np.array([0.0, 0.0, np.nan, 1.0, 1.0, 4.0, 0.0, 0.0, -3.0]),
            "non-finite coordinates",
        ),
        (
            "cluster",
            np.array(["A", "A", None, "B", "B", "B", "C", "C", "C"]),
            "missing values",
        ),
    ],
)
def test_cluster_connectivity_validates_cell_arrays(column, replacement, message):
    store = _store()
    store.cells.values[column] = replacement
    with pytest.raises(ValueError, match=message):
        _plot(store)


def test_cluster_connectivity_validates_graph_shape_and_weights():
    wrong_shape = _store()
    wrong_shape.graph = sparse.csr_matrix((8, 8))
    with pytest.raises(ValueError, match="Graph shape"):
        _plot(wrong_shape)

    dense = _store()
    dense.graph = dense.graph.toarray()
    with pytest.raises(TypeError, match="sparse matrix"):
        _plot(dense)

    negative = _store()
    negative.graph = negative.graph.copy()
    negative.graph.data[0] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        _plot(negative)


def test_cluster_connectivity_requires_exact_explicit_position_coverage():
    with pytest.raises(ValueError, match="missing: C"):
        _plot(
            _store(),
            positions={"A": (0.0, 0.0), "B": (1.0, 1.0)},
        )
    with pytest.raises(ValueError, match="unexpected: D"):
        _plot(
            _store(),
            positions={
                "A": (0.0, 0.0),
                "B": (1.0, 1.0),
                "C": (2.0, 2.0),
                "D": (3.0, 3.0),
            },
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"position": "mode"}, "position"),
        ({"minimum_edge_weight": -0.1}, "minimum_edge_weight"),
        ({"max_edges_per_node": -1}, "max_edges_per_node"),
        ({"edge_alpha": 1.1}, "edge_alpha"),
        ({"edge_width_range": (2.0, 1.0)}, "edge_width_range"),
    ],
)
def test_cluster_connectivity_validates_plot_options(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _plot(_store(), **kwargs)
