import matplotlib

matplotlib.use("Agg")

from contextlib import contextmanager
from importlib import import_module
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from scipy import sparse

from scarf.plotting._contracts import CategoricalScale, SizeScale
from scarf.plotting._figure import PlotResult
from scarf.plotting.cluster_connectivity import cluster_connectivity
from scarf.storage.artifacts import ArtifactRef

_SELECTION = ArtifactRef(
    scope="datastore",
    kind="cell_selection",
    artifact_id="2" * 64,
)
_GRAPH = ArtifactRef(
    scope="assay",
    assay="RNA",
    kind="connectivity_map",
    artifact_id="1" * 64,
)
_LAYOUT = ArtifactRef(
    scope="assay",
    assay="RNA",
    kind="embedding",
    artifact_id="3" * 64,
)
_GROUPS = ArtifactRef(
    scope="assay",
    assay="RNA",
    kind="cluster_cut",
    artifact_id="4" * 64,
)


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


@contextmanager
def _stored_graph_selection():
    with (
        patch(
            "scarf.plotting.cluster_connectivity.graph_cell_selection",
            return_value=_SELECTION,
        ),
        patch(
            "scarf.plotting.cluster_connectivity.validate_stored_selection_live_alias"
        ),
    ):
        yield


def _plot(store, **kwargs):
    graph_ref = kwargs.pop("graph", _GRAPH)
    cell_key = kwargs.pop("cell_key", "I")
    categorical_scale = kwargs.pop("categorical_scale", CategoricalScale())
    show = kwargs.pop("show", False)
    with _stored_graph_selection():
        return cluster_connectivity(
            store,
            group_by="cluster",
            layout_key="layout",
            graph=graph_ref,
            cell_key=cell_key,
            categorical_scale=categorical_scale,
            show=show,
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

    # The default size scale maps each one-third share onto [70, 650].
    np.testing.assert_allclose(nodes["size"], [70.0 + 580.0 / 3.0] * 3)
    assert nodes["displayLabel"].tolist() == ["A", "B", "C"]

    assert result.scales[0].order == ("A", "B", "C")
    assert result.scales[1] == SizeScale(size_min=70.0, size_max=650.0)
    assert result.provenance.n_cells == 9
    assert result.provenance.cell_key == "selected"
    assert result.provenance.notes == ("cluster_connectivity", "live_metadata")
    assert result.provenance.extras["n_nodes"] == 3
    assert result.provenance.extras["n_aggregated_edges"] == 3
    assert result.provenance.extras["n_edges"] == 3
    assert result.provenance.extras["position"] == "median"
    assert result.provenance.extras["cell_size_source"] == "hidden"
    assert result.provenance.extras["normalization"] == (
        "rawWeight / sqrt(incidentWeight[source] * incidentWeight[target])"
    )
    assert result.legends[0].label == "cluster"
    assert result.legends[0].extras == {
        "categories": ["A", "B", "C"],
        "placement": "nodes",
    }

    ax = result.axes["cluster_connectivity"]
    lines, node_points = ax.collections
    assert isinstance(lines, LineCollection)
    assert lines.get_zorder() < node_points.get_zorder()
    # One segment joins the two node centres of every retained edge.
    assert [segment.tolist() for segment in lines.get_segments()] == [
        [[0.0, 0.0], [2.0, 1.0]],
        [[0.0, 0.0], [4.0, 0.0]],
        [[2.0, 1.0], [4.0, 0.0]],
    ]
    # Widths rescale normalized weights linearly onto edge_width_range (0.4, 5).
    weights = edges["normalizedWeight"].to_numpy()
    expected_widths = 0.4 + (weights - weights.min()) / np.ptp(weights) * 4.6
    np.testing.assert_allclose(lines.get_linewidths(), expected_widths)
    assert lines.get_linewidths()[0] == pytest.approx(5.0)
    assert lines.get_alpha() == pytest.approx(0.45)
    np.testing.assert_allclose(node_points.get_offsets(), nodes[["x", "y"]])
    np.testing.assert_allclose(node_points.get_sizes(), nodes["size"])
    palette = result.scales[0].palette
    np.testing.assert_allclose(
        node_points.get_facecolors(),
        [to_rgba(palette[category]) for category in ("A", "B", "C")],
    )
    assert [(text.get_text(), text.get_position()) for text in ax.texts] == [
        ("A", (0.0, 0.0)),
        ("B", (2.0, 1.0)),
        ("C", (4.0, 0.0)),
    ]
    # Short labels in large markers use the largest label size.
    assert [text.get_fontsize() for text in ax.texts] == [8.0, 8.0, 8.0]
    # Cells span x in [0, 10] and y in [-3, 4]; 5% padding then squaring.
    assert ax.get_xlim() == pytest.approx((-0.5, 10.5))
    assert ax.get_ylim() == pytest.approx((-5.0, 6.0))
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
    assert result.provenance.extras["position"] == "explicit"
    ax = result.axes["cluster_connectivity"]
    node_points = ax.collections[1]
    np.testing.assert_allclose(
        node_points.get_offsets(),
        [[30.0, 31.0], [10.0, 11.0], [20.0, 21.0]],
    )
    np.testing.assert_allclose(
        node_points.get_facecolors(),
        [to_rgba(color) for color in ("#0000aa", "#aa0000", "#00aa00")],
    )
    # Edges keep category-code order (C, A, B) and join the explicit positions.
    assert [segment.tolist() for segment in ax.collections[0].get_segments()] == [
        [[30.0, 31.0], [10.0, 11.0]],
        [[30.0, 31.0], [20.0, 21.0]],
        [[10.0, 11.0], [20.0, 21.0]],
    ]
    # Without cells, the window frames only the explicit node positions.
    assert ax.get_xlim() == pytest.approx((9.0, 31.0))
    assert ax.get_ylim() == pytest.approx((10.0, 32.0))
    assert [text.get_text() for text in ax.texts] == [
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
    cells = ax.collections[0]
    store_values = _store().cells.values
    np.testing.assert_allclose(
        cells.get_offsets(),
        np.column_stack((store_values["layout1"], store_values["layout2"])),
    )
    palette = result.scales[0].palette
    np.testing.assert_allclose(
        cells.get_facecolors(),
        [to_rgba(palette[group], 0.3) for group in store_values["cluster"]],
    )
    assert cells.get_alpha() == pytest.approx(0.3)
    assert len(ax.texts) == 0
    assert result.legends[0].extras["placement"] == "none"
    assert result.provenance.extras["show_cells"] is True
    assert result.provenance.extras["cell_size_source"] == "panel"
    assert result.provenance.extras["cell_size"] == cells.get_sizes()[0]
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


def test_cluster_connectivity_without_intercluster_edges_draws_no_segments():
    store = _store()
    store.graph = _symmetric_graph(9, [(0, 1, 1.0), (3, 4, 1.0), (6, 8, 0.5)])

    result = _plot(store, minimum_edge_weight=0.0)

    assert result.tables["edges"].empty
    assert list(result.tables["edges"].columns) == [
        "source",
        "target",
        "rawWeight",
        "normalizedWeight",
    ]
    assert result.tables["nodes"]["category"].tolist() == ["A", "B", "C"]
    assert result.provenance.extras["n_aggregated_edges"] == 0
    assert result.provenance.extras["n_edges"] == 0
    lines = result.axes["cluster_connectivity"].collections[0]
    assert len(lines.get_segments()) == 0
    result.close()


def test_cluster_connectivity_shows_owned_results_by_default(monkeypatch):
    shown = []
    monkeypatch.setattr(PlotResult, "show", lambda result: shown.append(result))

    result = _plot(_store(), show=True)
    _plot(_store()).close()

    assert shown == [result]
    result.close()


@pytest.mark.parametrize(
    ("column", "replacement", "error", "message"),
    [
        (
            "layout1",
            np.array(["east"] * 9),
            TypeError,
            "Layout 'layout' coordinates must be numeric",
        ),
        (
            "layout2",
            np.zeros((9, 2)),
            ValueError,
            "Cell data 'layout2' must be one-dimensional",
        ),
    ],
)
def test_cluster_connectivity_rejects_malformed_layout_columns(
    column, replacement, error, message
):
    store = _store()
    store.cells.values[column] = replacement

    with pytest.raises(error) as raised:
        _plot(store)

    assert raised.value.args == (message,)


def test_cluster_connectivity_rejects_an_empty_cell_selection():
    store = _store()
    store.cells.values = {
        "layout1": np.array([], dtype=float),
        "layout2": np.array([], dtype=float),
        "cluster": np.array([], dtype=object),
    }

    with pytest.raises(ValueError) as raised:
        _plot(store, cell_key="empty")

    assert raised.value.args == ("No cells selected by cell_key 'empty'",)


def test_cluster_connectivity_rejects_unhashable_group_values():
    store = _store()
    groups = np.empty(9, dtype=object)
    for index in range(9):
        groups[index] = ["A"] if index < 5 else ["B"]
    store.cells.values["cluster"] = groups

    with pytest.raises(TypeError) as raised:
        _plot(store)

    assert raised.value.args == ("group_by values must be hashable categories",)
    assert isinstance(raised.value.__cause__, TypeError)


def test_category_codes_rejects_values_outside_the_resolved_order():
    from scarf.plotting.cluster_connectivity import _category_codes

    groups = np.array(["B", "A", "B"], dtype=object)
    np.testing.assert_array_equal(_category_codes(groups, ["A", "B"]), [1, 0, 1])
    with pytest.raises(ValueError) as raised:
        _category_codes(groups, ["A"])
    assert raised.value.args == ("Could not map every group value to a category",)
    assert isinstance(raised.value.__cause__, KeyError)


@pytest.mark.parametrize(
    ("position", "error", "message"),
    [
        (("left", "up"), TypeError, "Position for category 'A' must be a numeric pair"),
        ((1.0, np.inf), ValueError, "Position for category 'A' must be a finite pair"),
        (
            (1.0, 2.0, 3.0),
            ValueError,
            "Position for category 'A' must be a finite pair",
        ),
    ],
)
def test_cluster_connectivity_rejects_malformed_explicit_positions(
    position, error, message
):
    positions = {"A": position, "B": (1.0, 1.0), "C": (2.0, 2.0)}

    with pytest.raises(error) as raised:
        _plot(_store(), positions=positions)

    assert raised.value.args == (message,)


def test_cluster_connectivity_rejects_non_finite_graph_weights():
    store = _store()
    store.graph = store.graph.copy()
    store.graph.data[0] = np.inf

    with pytest.raises(ValueError) as raised:
        _plot(store)

    assert raised.value.args == ("Graph contains non-finite edge weights",)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"cell_size": -1.0}, "cell_size must be finite and non-negative"),
        ({"cell_size": np.nan}, "cell_size must be finite and non-negative"),
        ({"cell_alpha": 1.5}, "cell_alpha must be between 0 and 1"),
        ({"node_linewidth": -0.1}, "node_linewidth must be finite and non-negative"),
        ({"edge_width_range": (1.0,)}, "edge_width_range must contain two values"),
        (
            {"size_scale": SizeScale(size_max=np.inf)},
            "size_scale produced invalid marker areas",
        ),
    ],
)
def test_cluster_connectivity_rejects_invalid_marker_options(kwargs, message):
    with pytest.raises(ValueError) as raised:
        _plot(_store(), **kwargs)

    assert raised.value.args == (message,)


def test_cluster_connectivity_requires_artifact_graph_and_complete_live_inputs():
    store = _store()
    with _stored_graph_selection():
        with pytest.raises(TypeError) as raised:
            cluster_connectivity(
                store,
                group_by="cluster",
                layout_key="layout",
                graph=_GRAPH.to_dict(),
                show=False,
            )
        assert raised.value.args == ("graph must be an ArtifactRef",)
        with pytest.raises(ValueError) as raised:
            cluster_connectivity(store, group_by="cluster", graph=_GRAPH, show=False)
        assert raised.value.args == (
            "group_by and layout_key must be provided together",
        )
    assert store.cells.fetch_calls == []
    assert store.graph_calls == []


def _artifact_plot(
    monkeypatch,
    *,
    layout_selection=_SELECTION,
    group_selection=_SELECTION,
    layout_indices=(10, 11, 12, 13),
    group_indices=(10, 11, 12, 13),
    group_missing=None,
    **kwargs,
):
    module = import_module("scarf.plotting.cluster_connectivity")
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 5.0], [6.0, 5.0]])
    labels = np.array(["x", "x", "y", "y"], dtype=object)
    monkeypatch.setattr(
        module,
        "_resolve_layout",
        lambda _store, layout: (
            coordinates,
            np.asarray(layout_indices),
            layout_selection,
        ),
    )
    monkeypatch.setattr(
        module,
        "_resolve_grouping",
        lambda *_args, **_kwargs: (
            ("groups",),
            np.asarray(group_indices),
            [labels],
            group_missing,
        ),
    )
    monkeypatch.setattr(
        module,
        "_artifact_cell_selection",
        lambda _store, _ref: group_selection,
    )
    store = _FakeStore({}, _symmetric_graph(4, [(0, 1, 1.0), (1, 2, 0.5), (2, 3, 1.0)]))
    arguments = {"groups": _GROUPS, "layout": _LAYOUT, "graph": _GRAPH}
    arguments.update(kwargs)
    with _stored_graph_selection():
        return cluster_connectivity(store, show=False, **arguments)


def test_cluster_connectivity_reads_aligned_group_and_layout_artifacts(monkeypatch):
    result = _artifact_plot(monkeypatch, minimum_edge_weight=0.0)

    nodes = result.tables["nodes"]
    assert nodes["category"].tolist() == ["x", "y"]
    np.testing.assert_allclose(nodes[["x", "y"]], [[0.5, 0.0], [5.5, 5.0]])
    edges = result.tables["edges"]
    assert edges[["source", "target"]].values.tolist() == [["x", "y"]]
    # Incident weights: x = 1 + (1 + 0.5) = 2.5 and y = (0.5 + 1) + 1 = 2.5.
    np.testing.assert_allclose(edges["rawWeight"], [0.5])
    np.testing.assert_allclose(edges["normalizedWeight"], [0.5 / 2.5])
    assert result.legends[0].label == "groups"
    assert result.provenance.cell_key is None
    assert result.provenance.notes == ("cluster_connectivity", "artifact")
    assert result.provenance.extras["groups"] == _GROUPS.to_dict()
    assert result.provenance.extras["layout"] == _LAYOUT.to_dict()
    result.close()


_OTHER_SELECTION = ArtifactRef(
    scope="datastore",
    kind="cell_selection",
    artifact_id="9" * 64,
)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"group_by": "cluster"},
            "Use groups and layout together, or group_by and layout_key together",
        ),
        ({"layout": None}, "groups and layout must be provided together"),
        (
            {"cell_key": "filtered"},
            "cell_key cannot override artifacts' stored cell selection",
        ),
        (
            {
                "layout": ArtifactRef(
                    scope="assay",
                    assay="ADT",
                    kind="embedding",
                    artifact_id="3" * 64,
                )
            },
            "layout and graph must belong to the same assay",
        ),
        (
            {
                "groups": ArtifactRef(
                    scope="assay",
                    assay="ADT",
                    kind="cluster_cut",
                    artifact_id="4" * 64,
                )
            },
            "groups and graph must belong to the same assay",
        ),
        (
            {"group_missing": np.array([False, True, False, False])},
            "groups contains missing labels",
        ),
        (
            {"layout_selection": _OTHER_SELECTION},
            "graph, groups, and layout must share the same cell selection",
        ),
        (
            {"group_selection": _OTHER_SELECTION},
            "graph, groups, and layout must share the same cell selection",
        ),
        (
            {"group_indices": (10, 12, 11, 13)},
            "groups and layout select cells in a different order",
        ),
    ],
)
def test_cluster_connectivity_rejects_inconsistent_artifact_inputs(
    monkeypatch, kwargs, message
):
    with pytest.raises(ValueError) as raised:
        _artifact_plot(monkeypatch, **kwargs)

    assert raised.value.args == (message,)


_STORED_CLUSTER_DISPLAY = {
    "kind": "categorical",
    "categories": [
        {"value": "C", "label": "Gamma", "color": "#0000aa"},
        {"value": "A", "label": "Alpha", "color": "#aa0000"},
        {"value": "B", "label": "Beta", "color": "#00aa00"},
    ],
    "missing_label": "unassigned",
    "missing_color": "#cccccc",
}


def _store_with_display(display):
    store = _store()
    store.display_requests = []

    def stored_display(column):
        store.display_requests.append(column)
        return display

    store._stored_display_metadata = stored_display
    return store


def test_cluster_connectivity_uses_stored_categorical_display_metadata():
    store = _store_with_display(_STORED_CLUSTER_DISPLAY)

    result = _plot(store, categorical_scale=None, minimum_edge_weight=0.0)

    assert store.display_requests == ["cluster"]
    nodes = result.tables["nodes"]
    assert nodes["category"].tolist() == ["C", "A", "B"]
    assert nodes["displayLabel"].tolist() == ["Gamma", "Alpha", "Beta"]
    scale = result.scales[0]
    assert scale.order == ("C", "A", "B")
    assert scale.palette == {"C": "#0000aa", "A": "#aa0000", "B": "#00aa00"}
    assert (scale.missing_label, scale.missing_color) == ("unassigned", "#cccccc")
    ax = result.axes["cluster_connectivity"]
    np.testing.assert_allclose(
        ax.collections[1].get_facecolors(),
        [to_rgba(color) for color in ("#0000aa", "#aa0000", "#00aa00")],
    )
    assert [text.get_text() for text in ax.texts] == ["Gamma", "Alpha", "Beta"]
    result.close()


def test_cluster_connectivity_prefers_explicit_scale_over_stored_display():
    store = _store_with_display(_STORED_CLUSTER_DISPLAY)

    result = _plot(
        store,
        categorical_scale=CategoricalScale(order=("A", "B", "C")),
    )

    assert store.display_requests == []
    assert result.tables["nodes"]["category"].tolist() == ["A", "B", "C"]
    assert result.tables["nodes"]["displayLabel"].tolist() == ["A", "B", "C"]
    result.close()


def test_cluster_connectivity_ignores_non_categorical_stored_display():
    store = _store_with_display(
        {
            "kind": "continuous",
            "colormap": "magma",
            "minimum": 0.0,
            "maximum": 1.0,
            "scale": "linear",
        }
    )

    result = _plot(store, categorical_scale=None)

    assert store.display_requests == ["cluster"]
    # Without a categorical contract, categories sort naturally.
    assert result.scales[0].order == ("A", "B", "C")
    assert result.scales[0].labels is None
    result.close()
