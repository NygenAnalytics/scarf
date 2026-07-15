import matplotlib
import networkx as nx
import pytest

matplotlib.use("Agg")

import scarf.plots as plots  # noqa: E402


def test_shade_scatter_continuous_panels_share_linear_limits():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    x = np.linspace(-1.0, 1.0, 30)
    dfs = [
        pd.DataFrame({"x": x, "y": x, "value": np.linspace(0.0, 1.0, 30)}),
        pd.DataFrame({"x": x, "y": -x, "value": np.linspace(10.0, 20.0, 30)}),
    ]
    axes = plots.shade_scatter(
        dfs,
        pixels=32,
        n_columns=2,
        legend_ondata=False,
        legend_onside=False,
        show_fig=False,
    )
    assert axes is not None
    artists = [
        next(child for child in ax.get_children() if hasattr(child, "get_clim"))
        for ax in axes.flat
    ]
    assert artists[0].get_clim() == pytest.approx(artists[1].get_clim())
    assert artists[0].norm.__class__.__name__ == "Normalize"
    plt.close(axes.flat[0].figure)


def _assert_expected_tree_layout(
    positions: dict[str, tuple[float, float]],
) -> None:
    expected = {
        "root": (1.2, 1.0),
        "left": (0.4, 0.5),
        "leaf": (0.4, 0.0),
        "right": (2.0, 0.5),
    }
    assert positions.keys() == expected.keys()
    for node, coordinates in expected.items():
        assert positions[node] == pytest.approx(coordinates)


def test_hierarchy_pos_directed_tree_uses_source_as_root():
    graph = nx.DiGraph(
        [
            ("root", "left"),
            ("root", "right"),
            ("left", "leaf"),
        ]
    )

    positions = plots.hierarchy_pos(
        graph,
        width=2.0,
        vert_gap=0.5,
        vert_loc=1.0,
    )

    _assert_expected_tree_layout(positions)

    branch_positions = plots.hierarchy_pos(graph, root="left")
    assert branch_positions.keys() == {"left", "leaf"}
    assert branch_positions["left"][1] == 0
    assert branch_positions["leaf"][1] == pytest.approx(-0.2)


def test_hierarchy_pos_undirected_tree_selects_root_deterministically(monkeypatch):
    graph = nx.Graph(
        [
            ("root", "left"),
            ("root", "right"),
            ("left", "leaf"),
        ]
    )
    choices = []

    def choose_root(nodes):
        choices.append(nodes)
        return "root"

    monkeypatch.setattr(plots.np.random, "choice", choose_root)

    positions = plots.hierarchy_pos(
        graph,
        width=2.0,
        vert_gap=0.5,
        vert_loc=1.0,
    )

    assert len(choices) == 1
    assert set(choices[0]) == set(graph)
    _assert_expected_tree_layout(positions)


@pytest.mark.parametrize(
    "graph",
    [
        pytest.param(nx.cycle_graph(3), id="cycle"),
        pytest.param(nx.Graph([(0, 1), (2, 3)]), id="disconnected"),
    ],
)
def test_hierarchy_pos_rejects_non_trees(graph):
    with pytest.raises(
        TypeError,
        match="cannot use hierarchy_pos on a graph that is not a tree",
    ):
        plots.hierarchy_pos(graph, root=0)


@pytest.mark.parametrize(
    ("titles", "panel_count", "expected"),
    [
        (None, 3, None),
        ("Only panel", 1, ["Only panel"]),
        (["First", "Second"], 2, ["First", "Second"]),
        (["Only panel"], 1, ["Only panel"]),
    ],
)
def test_handle_titles_type_accepts_matching_titles(titles, panel_count, expected):
    assert plots._handle_titles_type(titles, panel_count) == expected


@pytest.mark.parametrize(
    ("titles", "panel_count"),
    [
        pytest.param(["Only one"], 2, id="wrong-count"),
        pytest.param("ab", 2, id="wrong-type"),
    ],
)
def test_handle_titles_type_rejects_invalid_multi_panel_titles(titles, panel_count):
    assert plots._handle_titles_type(titles, panel_count) is None
