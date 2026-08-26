"""Native cluster-tree plotting."""

import math
from typing import Any, cast

import numpy as np
import pandas as pd

from ..storage.artifacts import ArtifactRef
from ._contracts import CategoricalScale, ColorScale, PlotProvenance, SizeScale
from ._deps import require_matplotlib
from ._figure import LegendSpec, PlotResult, normalize_axes_target
from ._style import (
    apply_figure_chrome,
    continuous_norm,
    sort_categories,
    theme_context,
)


def _hierarchy_positions(
    graph: Any,
    *,
    root: Any | None = None,
    width: float = 1.0,
    vert_gap: float = 0.2,
    vert_loc: float = 0.0,
    leaf_vs_root_factor: float = 0.5,
) -> dict[Any, tuple[float, float]]:
    import networkx as nx

    if not nx.is_tree(graph):
        raise TypeError("cannot position a graph that is not a tree")
    if root is None:
        root = (
            next(iter(nx.topological_sort(graph)))
            if isinstance(graph, nx.DiGraph)
            else next(iter(graph.nodes))
        )

    def place(
        node: Any,
        *,
        leftmost: float,
        branch_width: float,
        leaf_dx: float,
        y: float,
        x_center: float,
        root_positions: dict[Any, tuple[float, float]],
        leaf_positions: dict[Any, tuple[float, float]],
        parent: Any | None,
    ) -> int:
        root_positions[node] = (x_center, y)
        children = list(graph.neighbors(node))
        if not isinstance(graph, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if not children:
            leaf_positions[node] = (leftmost, y)
            return 1

        root_dx = branch_width / len(children)
        next_x = x_center - branch_width / 2 - root_dx / 2
        leaf_count = 0
        for child in children:
            next_x += root_dx
            leaf_count += place(
                child,
                leftmost=leftmost + leaf_count * leaf_dx,
                branch_width=root_dx,
                leaf_dx=leaf_dx,
                y=y - vert_gap,
                x_center=next_x,
                root_positions=root_positions,
                leaf_positions=leaf_positions,
                parent=node,
            )
        child_x = [leaf_positions[child][0] for child in children]
        leaf_positions[node] = (
            (min(child_x) + max(child_x)) / 2,
            y,
        )
        return leaf_count

    if isinstance(graph, nx.DiGraph):
        leaf_count = len(
            [
                node
                for node in nx.descendants(graph, root)
                if graph.out_degree(node) == 0
            ]
        )
    else:
        leaf_count = len(
            [
                node
                for node in nx.node_connected_component(graph, root)
                if graph.degree(node) == 1 and node != root
            ]
        )
    leaf_count = max(leaf_count, 1)
    root_positions: dict[Any, tuple[float, float]] = {}
    leaf_positions: dict[Any, tuple[float, float]] = {}
    place(
        root,
        leftmost=0.0,
        branch_width=width,
        leaf_dx=width / leaf_count,
        y=vert_loc,
        x_center=width / 2,
        root_positions=root_positions,
        leaf_positions=leaf_positions,
        parent=None,
    )
    positions = {
        node: (
            leaf_vs_root_factor * leaf_positions[node][0]
            + (1 - leaf_vs_root_factor) * root_positions[node][0],
            leaf_positions[node][1],
        )
        for node in root_positions
    }
    x_max = max((x for x, _ in positions.values()), default=0.0)
    if x_max != 0:
        positions = {node: (x * width / x_max, y) for node, (x, y) in positions.items()}
    return positions


def _tree_color_series(
    values: np.ndarray,
    *,
    force_ints_as_cats: bool,
) -> tuple[pd.Series, bool]:
    series = pd.Series(values)
    if series.nunique() == 1:
        return pd.Series(np.ones(len(series)), index=series.index), False
    categorical = (
        pd.api.types.is_bool_dtype(series.dtype)
        or pd.api.types.is_string_dtype(series.dtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or (pd.api.types.is_integer_dtype(series.dtype) and force_ints_as_cats)
    )
    if categorical:
        return series.astype("category"), True
    return series.astype(np.float64), False


def _tree_palette(
    mpl: Any,
    categories: list[Any],
    *,
    cmap: str,
    color_key: dict[Any, str] | None,
) -> dict[Any, str]:
    if color_key is not None:
        palette = dict(color_key)
        missing = [category for category in categories if category not in palette]
        if missing:
            raise KeyError(f"ERROR: key {missing[0]} missing in `color_key`")
        return palette
    colormap = mpl.colormaps.get_cmap(cmap).resampled(max(len(categories), 1))
    return {
        category: mpl.colors.to_hex(colormap(index))
        for index, category in enumerate(categories)
    }


def _draw_tree_pie(
    ax: Any,
    values: np.ndarray,
    colors: list[str],
    *,
    x: float,
    y: float,
    size: float,
) -> None:
    cumulative = np.cumsum(values, dtype=np.float64)
    cumulative /= cumulative[-1]
    bounds = [0.0, *cumulative.tolist()]
    for start, stop, color in zip(bounds[:-1], bounds[1:], colors):
        angles = np.linspace(2 * np.pi * start, 2 * np.pi * stop)
        marker_x = [0.0, *np.cos(angles).tolist()]
        marker_y = [0.0, *np.sin(angles).tolist()]
        marker = np.column_stack([marker_x, marker_y])
        ax.scatter([x], [y], marker=marker, s=size, c=color, zorder=3)


def cluster_tree(
    store: Any,
    *,
    graph: ArtifactRef | None = None,
    from_assay: str | None = None,
    cell_key: str | None = None,
    cluster_key: str | None = None,
    fill_by_value: str | None = None,
    force_ints_as_cats: bool = True,
    width: float = 1,
    lvr_factor: float = 0.5,
    vert_gap: float = 0.2,
    min_node_size: float = 10,
    node_size_multiplier: float = 10_000.0,
    node_power: float = 1.2,
    root_size: float = 100,
    non_leaf_size: float = 10,
    show_labels: bool = True,
    fontsize: float = 10,
    root_color: str = "#C0C0C0",
    non_leaf_color: str = "k",
    cmap: str = "tab20",
    color_key: dict[Any, str] | None = None,
    edgecolors: str = "k",
    edgewidth: float = 1,
    alpha: float = 0.7,
    figsize: tuple[float, float] = (5, 5),
    ax: Any = None,
    theme: str = "notebook",
    show: bool = True,
) -> PlotResult:
    """Plot the coalesced hierarchy for a clustering result."""
    import networkx as nx

    prepared = store._prepare_cluster_tree(
        graph=graph,
        from_assay=from_assay,
        cell_key=cell_key,
        cluster_key=cluster_key,
        fill_by_value=fill_by_value,
    )
    tree_graph: Any = prepared["graph"]
    clusters = np.asarray(prepared["clusters"])
    raw_color_values = (
        clusters
        if prepared["color_values"] is None
        else np.asarray(prepared["color_values"])
    )
    if raw_color_values.shape[0] != clusters.shape[0]:
        raise ValueError("Cluster colors and cluster assignments are misaligned")
    using_clusters = prepared["color_values"] is None
    color_values, categorical = _tree_color_series(
        raw_color_values,
        force_ints_as_cats=force_ints_as_cats,
    )

    angular_positions = _hierarchy_positions(
        tree_graph,
        width=width * math.pi,
        leaf_vs_root_factor=lvr_factor,
        vert_gap=vert_gap,
    )
    positions = {
        node: (radius * math.cos(theta), radius * math.sin(theta))
        for node, (theta, radius) in angular_positions.items()
    }
    cluster_counts = pd.Series(clusters).value_counts()
    cluster_sizes = (
        node_size_multiplier * ((cluster_counts / cluster_counts.sum()) ** node_power)
    ).to_dict()

    plt, mpl = require_matplotlib()
    categories: list[Any] = []
    palette: dict[Any, str] | None = None
    cluster_means: pd.Series | None = None
    norm = None
    colormap = mpl.colormaps.get_cmap(cmap)
    if categorical:
        categories = sort_categories(list(color_values.dropna().unique()))
        palette = _tree_palette(
            mpl,
            categories,
            cmap=cmap,
            color_key=color_key,
        )
    else:
        cluster_means = (
            pd.DataFrame({"cluster": clusters, "value": color_values})
            .groupby("cluster", observed=False)["value"]
            .mean()
        )
        color_min = float(cluster_means.min())
        color_max = float(cluster_means.max())
        norm = continuous_norm(
            mpl,
            vmin=color_min,
            vmax=color_max,
            vcenter=None,
        )

    node_order = list(tree_graph.nodes())
    node_colors: list[str] = []
    node_sizes: list[float] = []
    for node in node_order:
        node_data = tree_graph.nodes[node]
        if "partition_id" in node_data:
            cluster_id = node_data["partition_id"]
            if categorical and using_clusters:
                assert palette is not None
                node_colors.append(palette[cluster_id])
                node_sizes.append(max(float(cluster_sizes[cluster_id]), min_node_size))
            elif not categorical:
                assert cluster_means is not None and norm is not None
                node_colors.append(
                    mpl.colors.to_hex(colormap(norm(cluster_means.loc[cluster_id])))
                )
                node_sizes.append(max(float(cluster_sizes[cluster_id]), min_node_size))
            else:
                node_colors.append("white")
                node_sizes.append(0.0)
        elif node_data["nleaves"] == len(clusters):
            node_colors.append(root_color)
            node_sizes.append(root_size)
        else:
            node_colors.append(non_leaf_color)
            node_sizes.append(non_leaf_size)

    with theme_context(theme):
        fig, axes, owns_figure = normalize_axes_target(
            ax,
            panel_keys=["tree"],
            figsize=figsize if ax is None else None,
        )
        tree_ax = axes["tree"]
        nx.draw_networkx_edges(
            tree_graph,
            pos=positions,
            ax=tree_ax,
            edge_color=edgecolors,
            width=edgewidth,
            alpha=alpha,
        )
        nx.draw_networkx_nodes(
            tree_graph,
            pos=positions,
            ax=tree_ax,
            nodelist=node_order,
            node_size=node_sizes,
            node_color=node_colors,
            edgecolors=edgecolors,
            linewidths=edgewidth,
            alpha=alpha,
        )

        if categorical and not using_clusters:
            assert palette is not None
            for node in node_order:
                node_data = tree_graph.nodes[node]
                if "partition_id" not in node_data:
                    continue
                cluster_id = node_data["partition_id"]
                counts = color_values[clusters == cluster_id].value_counts()
                _draw_tree_pie(
                    tree_ax,
                    counts.to_numpy(),
                    [palette[value] for value in counts.index],
                    x=positions[node][0],
                    y=positions[node][1],
                    size=max(float(cluster_sizes[cluster_id]), min_node_size),
                )

        if show_labels:
            for node in node_order:
                node_data = tree_graph.nodes[node]
                if "partition_id" not in node_data:
                    continue
                cluster_id = node_data["partition_id"]
                tree_ax.text(
                    positions[node][0],
                    positions[node][1],
                    str(cluster_id),
                    fontsize=fontsize,
                    ha="center",
                    va="center",
                )
        tree_ax.set_axis_off()
        tree_ax.set_aspect("equal", adjustable="datalim")

        if not categorical:
            assert norm is not None
            scalar_mappable = mpl.cm.ScalarMappable(norm=norm, cmap=colormap)
            colorbar = fig.colorbar(
                scalar_mappable,
                ax=tree_ax,
                location="right",
                shrink=0.65,
                pad=0.02,
            )
            colorbar.set_label(fill_by_value or cast(str, prepared["cluster_key"]))
            axes["colorbar"] = colorbar.ax
        apply_figure_chrome(fig, theme)

    nodes = pd.DataFrame(
        [
            {
                "node": node,
                "nleaves": tree_graph.nodes[node]["nleaves"],
                "partition_id": tree_graph.nodes[node].get("partition_id"),
            }
            for node in node_order
        ]
    )
    edges = pd.DataFrame(tree_graph.edges(), columns=["source", "target"])
    position_table = pd.DataFrame(
        [{"node": node, "x": xy[0], "y": xy[1]} for node, xy in positions.items()]
    )
    cluster_summary = (
        cluster_counts.rename_axis("cluster").rename("n_cells").reset_index()
    )

    if categorical:
        assert palette is not None
        color_scale: Any = CategoricalScale(
            order=tuple(categories),
            palette=palette,
        )
        legend = LegendSpec(
            kind="categorical",
            label=fill_by_value or cast(str, prepared["cluster_key"]),
        )
    else:
        assert cluster_means is not None
        color_min = float(cluster_means.min())
        color_max = float(cluster_means.max())
        if color_max <= color_min:
            color_max = color_min + 1.0
        color_scale = ColorScale(
            cmap=cmap,
            vmin=color_min,
            vmax=color_max,
        )
        legend = LegendSpec(
            kind="colorbar",
            label=fill_by_value or cast(str, prepared["cluster_key"]),
            extras={"vmin": color_min, "vmax": color_max},
        )
    size_scale = SizeScale(
        vmin=0.0,
        vmax=1.0,
        size_min=min_node_size,
        size_max=node_size_multiplier,
    )
    result = PlotResult(
        figure=fig,
        axes=axes,
        tables={
            "nodes": nodes,
            "edges": edges,
            "positions": position_table,
            "cluster_summary": cluster_summary,
        },
        legends=(legend,),
        scales=(color_scale, size_scale),
        provenance=PlotProvenance(
            assay=cast(str, prepared["from_assay"]),
            cell_key=cast(str, prepared["cell_key"]),
            n_cells=len(clusters),
            renderer="matplotlib",
            notes=("cluster_tree", "coalesced"),
            extras={
                "graph": cast(ArtifactRef, prepared["graph_ref"]).to_dict(),
                "cluster_key": prepared["cluster_key"],
                "fill_by_value": fill_by_value,
                "coalesced_location": prepared["coalesced_location"],
                "force_ints_as_cats": force_ints_as_cats,
                "node_power": node_power,
            },
        ),
        owns_figure=owns_figure,
        theme=theme,
    )
    if show:
        result.show()
    return result
