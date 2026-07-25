"""Native heatmap and cluster-tree plotting."""

import math
from collections.abc import Hashable
from importlib.metadata import version
from typing import Any, cast

import numpy as np
import pandas as pd

from ..storage.artifacts import ArtifactRef, inspect_artifact
from ..storage.types import as_zarr_array, as_zarr_group
from ..matrix import ChunkedArray
from ..utils.arrays import array_digest
from ..utils.compute import controlled_compute
from ..utils.logging import logger
from ..utils.progress import tqdmbar
from ._contracts import CategoricalScale, ColorScale, PlotProvenance, SizeScale
from ._deps import require_matplotlib, require_seaborn
from ._figure import LegendSpec, PlotResult, normalize_axes_target
from ._style import (
    apply_figure_chrome,
    continuous_norm,
    sort_categories,
    theme_context,
)


def _scarf_version() -> str:
    try:
        return version("scarf")
    except Exception:
        return "unknown"


def _prepare_marker_heatmap(
    store: Any,
    *,
    from_assay: str | None,
    group_key: str | None,
    cell_key: str | None,
    topn: int,
    log_transform: bool,
    vmin: float,
    vmax: float,
) -> dict[str, Any]:
    assay = store._get_assay(from_assay)
    if group_key is None:
        raise ValueError("ERROR: Please provide a value for `group_key`")
    if cell_key is None:
        cell_key = "I"

    assay_group = as_zarr_group(store.zw[assay.name], name=assay.name)
    if "markers" not in assay_group:
        raise KeyError("ERROR: Please run `run_marker_search` first")
    try:
        marker_slot = store._resolve_marker_group(
            assay.name,
            cell_key,
            group_key,
        )
    except KeyError:
        raise KeyError(
            "ERROR: Please run `run_marker_search` first with "
            f"{group_key} as `group_key` and {cell_key} as `cell_key`"
        ) from None

    feature_indices: list[int] = []
    marker_rows: list[dict[str, Any]] = []
    if "feature_index" in marker_slot:
        shared_index = np.asarray(
            as_zarr_array(marker_slot["feature_index"], name="feature_index")[:]
        )
        for group_name in marker_slot.group_keys():
            marker_group = as_zarr_group(marker_slot[group_name], name=group_name)
            if "stats" not in marker_group:
                continue
            stats = np.asarray(as_zarr_array(marker_group["stats"], name="stats")[:])
            top = np.argsort(-stats[:, 0])[:topn]
            selected = shared_index[top].astype(int)
            feature_indices.extend(selected.tolist())
            marker_rows.extend(
                {
                    "group": group_name,
                    "rank": rank,
                    "feature_index": int(feature_index),
                    "score": float(stats[stat_index, 0]),
                }
                for rank, (feature_index, stat_index) in enumerate(
                    zip(selected, top), start=1
                )
            )
    else:
        for group_name in marker_slot.group_keys():
            marker_group = as_zarr_group(marker_slot[group_name], name=group_name)
            if "feature_index" not in marker_group:
                continue
            selected = np.asarray(
                as_zarr_array(marker_group["feature_index"], name="feature_index")[
                    :topn
                ],
                dtype=int,
            )
            feature_indices.extend(selected.tolist())
            marker_rows.extend(
                {
                    "group": group_name,
                    "rank": rank,
                    "feature_index": int(feature_index),
                    "score": np.nan,
                }
                for rank, feature_index in enumerate(selected, start=1)
            )

    if not feature_indices:
        raise ValueError("ERROR: Marker list is empty for all the groups")
    feature_index = np.asarray(sorted(set(feature_indices)), dtype=int)
    cell_index = np.asarray(assay.cells.active_index(cell_key))
    normalized = assay.normed(
        cell_idx=cell_index,
        feat_idx=feature_index,
        log_transform=log_transform,
    )
    groups = np.asarray(assay.cells.fetch(group_key, cell_key))

    group_sums: dict[Any, np.ndarray] = {}
    group_counts: dict[Any, int] = {}
    row_start = 0
    for block in tqdmbar(
        normalized.blocks,
        total=normalized.numblocks[0],
        desc="Aggregating marker values per group",
    ):
        values = controlled_compute(block, store.nthreads)
        block_groups = groups[row_start : row_start + values.shape[0]]
        row_start += values.shape[0]
        block_frame = pd.DataFrame(values)
        block_frame["__group__"] = block_groups
        grouped = block_frame.groupby("__group__")
        block_sum = grouped.sum()
        block_count = grouped.size()
        for label in block_sum.index:
            summed = block_sum.loc[label].to_numpy(dtype=np.float64)
            if label not in group_sums:
                group_sums[label] = summed
                group_counts[label] = int(block_count.loc[label])
            else:
                group_sums[label] += summed
                group_counts[label] += int(block_count.loc[label])

    labels = sorted(group_sums)
    group_means = pd.DataFrame(
        np.vstack([group_sums[label] / group_counts[label] for label in labels]),
        index=labels,
    )
    group_means = group_means.apply(
        lambda values: (values - values.mean()) / values.std(),
        axis=0,
    )
    feature_names = np.asarray(assay.feats.fetch_all("names"))
    group_means.columns = feature_names[feature_index]
    matrix = group_means.T
    matrix[matrix < vmin] = vmin
    matrix[matrix > vmax] = vmax

    marker_table = pd.DataFrame(marker_rows)
    marker_table["feature"] = feature_names[
        marker_table["feature_index"].to_numpy(dtype=int)
    ]
    return {
        "matrix": matrix,
        "markers": marker_table,
        "assay": assay.name,
        "cell_key": cell_key,
        "group_key": group_key,
        "n_cells": len(cell_index),
    }


def marker_heatmap(
    store: Any,
    *,
    from_assay: str | None = None,
    group_key: str | None = None,
    cell_key: str | None = None,
    topn: int = 5,
    log_transform: bool = True,
    vmin: float = -1,
    vmax: float = 2,
    figsize: tuple[float, float] | None = None,
    fontsize: float = 10,
    width_factor: float = 0.03,
    height_factor: float = 0.02,
    cmap: Any = "magma_r",
    theme: str = "notebook",
    show: bool = True,
    **heatmap_kwargs: Any,
) -> PlotResult:
    """Plot standardized expression of the top marker features."""
    if vmax <= vmin:
        raise ValueError("vmax must be greater than vmin")
    prepared = _prepare_marker_heatmap(
        store,
        from_assay=from_assay,
        group_key=group_key,
        cell_key=cell_key,
        topn=topn,
        log_transform=log_transform,
        vmin=vmin,
        vmax=vmax,
    )
    matrix = cast(pd.DataFrame, prepared["matrix"])
    if figsize is None:
        figsize = (
            matrix.shape[1] * fontsize * width_factor,
            fontsize * matrix.shape[0] * height_factor,
        )

    sns = require_seaborn()
    clustermap_kwargs = {
        "yticklabels": matrix.index,
        "xticklabels": matrix.columns,
        "method": "ward",
        "figsize": figsize,
        "cmap": cmap,
        "rasterized": True,
    }
    clustermap_kwargs.update(heatmap_kwargs)
    with theme_context(theme):
        cluster_grid = sns.clustermap(matrix, **clustermap_kwargs)
        row_order = (
            cluster_grid.dendrogram_row.reordered_ind
            if cluster_grid.dendrogram_row is not None
            else list(range(matrix.shape[0]))
        )
        column_order = (
            cluster_grid.dendrogram_col.reordered_ind
            if cluster_grid.dendrogram_col is not None
            else list(range(matrix.shape[1]))
        )
        if clustermap_kwargs["yticklabels"] is not False:
            cluster_grid.ax_heatmap.set_yticklabels(
                matrix.index[row_order],
                fontsize=fontsize,
            )
        if clustermap_kwargs["xticklabels"] is not False:
            cluster_grid.ax_heatmap.set_xticklabels(
                matrix.columns[column_order],
                fontsize=fontsize,
            )
        apply_figure_chrome(cluster_grid.fig, theme)

    axes: dict[Hashable, Any] = {
        "heatmap": cluster_grid.ax_heatmap,
        "row_dendrogram": cluster_grid.ax_row_dendrogram,
        "column_dendrogram": cluster_grid.ax_col_dendrogram,
        "colorbar": cluster_grid.cax,
    }
    cmap_name = cmap if isinstance(cmap, str) else getattr(cmap, "name", None)
    result = PlotResult(
        figure=cluster_grid.fig,
        axes=axes,
        tables={
            "matrix": matrix.copy(),
            "markers": cast(pd.DataFrame, prepared["markers"]).copy(),
        },
        legends=(
            LegendSpec(
                kind="colorbar",
                label="standardized expression",
                extras={"vmin": vmin, "vmax": vmax},
            ),
        ),
        scales=(
            ColorScale(cmap=cmap_name, vmin=vmin, vmax=vmax),
            CategoricalScale(order=tuple(matrix.columns)),
        ),
        provenance=PlotProvenance(
            scarf_version=_scarf_version(),
            assay=cast(str, prepared["assay"]),
            cell_key=cast(str, prepared["cell_key"]),
            n_cells=cast(int, prepared["n_cells"]),
            renderer="matplotlib",
            notes=("marker_heatmap", "clustered"),
            extras={
                "group_key": prepared["group_key"],
                "topn": topn,
                "log_transform": log_transform,
                "vmin": vmin,
                "vmax": vmax,
            },
        ),
        owns_figure=True,
        theme=theme,
    )
    if show:
        result.show()
    return result


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
    from_assay: str | None = None,
    cell_key: str | None = None,
    feat_key: str | None = None,
    integrated_graph: str | None = None,
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
        from_assay=from_assay,
        cell_key=cell_key,
        feat_key=feat_key,
        integrated_graph=integrated_graph,
        cluster_key=cluster_key,
        fill_by_value=fill_by_value,
    )
    graph = prepared["graph"]
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
        graph,
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

    node_order = list(graph.nodes())
    node_colors: list[str] = []
    node_sizes: list[float] = []
    for node in node_order:
        node_data = graph.nodes[node]
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
            graph,
            pos=positions,
            ax=tree_ax,
            edge_color=edgecolors,
            width=edgewidth,
            alpha=alpha,
        )
        nx.draw_networkx_nodes(
            graph,
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
                node_data = graph.nodes[node]
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
                node_data = graph.nodes[node]
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
                "nleaves": graph.nodes[node]["nleaves"],
                "partition_id": graph.nodes[node].get("partition_id"),
            }
            for node in node_order
        ]
    )
    edges = pd.DataFrame(graph.edges(), columns=["source", "target"])
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
            scarf_version=_scarf_version(),
            assay=cast(str, prepared["from_assay"]),
            cell_key=cast(str, prepared["cell_key"]),
            n_cells=len(clusters),
            renderer="matplotlib",
            notes=("cluster_tree", "coalesced"),
            extras={
                "feat_key": prepared["feat_key"],
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


def _prepare_pseudotime_heatmap(
    store: Any,
    *,
    from_assay: str | None,
    cell_key: str | None,
    feat_key: str | None,
    feature_cluster_key: str | None,
    pseudotime_key: str | None,
) -> dict[str, Any]:
    assay = store._get_assay(from_assay)
    if cell_key is None:
        raise ValueError("ERROR: Please provide a value for parameter `cell_key`")
    if feat_key is None:
        raise ValueError("ERROR: Please provide a value for parameter `feat_key`")
    if feature_cluster_key is None:
        raise ValueError(
            "ERROR: Please provide a value for parameter `feature_cluster_key`"
        )
    if pseudotime_key is None:
        raise ValueError("ERROR: Please provide a value for parameter `pseudotime_key`")

    cell_ordering = np.asarray(
        assay.cells.fetch(pseudotime_key, key=cell_key),
        dtype=float,
    )
    cell_index, feature_index = assay._get_cell_feat_idx(cell_key, feat_key)
    hashes = [
        array_digest(np.asarray(values))
        for values in (cell_index, feature_index, cell_ordering)
    ]
    location = f"aggregated_{cell_key}_{feat_key}_{pseudotime_key}"
    aggregation_group = None
    artifact_backed = False
    feature_data = as_zarr_group(assay.z["featureData"], name="featureData")
    if feature_cluster_key in feature_data:
        cluster_column = as_zarr_array(
            feature_data[feature_cluster_key],
            name=feature_cluster_key,
        )
        raw_ref = cluster_column.attrs.get("source_artifact")
        if isinstance(raw_ref, dict):
            try:
                aggregation_ref = ArtifactRef.from_dict(raw_ref)
                status = inspect_artifact(store.zw, aggregation_ref)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "Feature cluster column has an invalid source artifact. "
                    "Rerun run_pseudotime_aggregation"
                ) from exc
            source_value = cluster_column.attrs.get("source_value")
            if (
                aggregation_ref.kind != "pseudotime_aggregation"
                or aggregation_ref.scope != "assay"
                or aggregation_ref.assay != assay.name
                or status.operation != "run_pseudotime_aggregation"
                or not status.complete
                or source_value != "cluster_values"
                or "value_index" in cluster_column.attrs
            ):
                raise ValueError(
                    "Feature cluster column is not linked to a complete "
                    "pseudotime aggregation artifact for this assay"
                )
            aggregation_group = as_zarr_group(
                store.zw[status.path],
                name=status.path,
            )
            location = status.path
            artifact_backed = True
        elif raw_ref is not None:
            raise ValueError(
                "Feature cluster column has a malformed source artifact. "
                "Rerun run_pseudotime_aggregation"
            )
    if aggregation_group is None:
        if location not in assay.z:
            raise KeyError(
                "ERROR: Could not find aggregated feature values. "
                "Please run `run_pseudotime_aggregation` with the same "
                "`cell_key`, `feat_key`, and `pseudotime_key`"
            )
        aggregation_group = as_zarr_group(assay.z[location], name=location)
    if (
        ("input_fingerprints" if artifact_backed else "hashes")
        not in aggregation_group.attrs
        or "data" not in aggregation_group
        or "feature_indices" not in aggregation_group
        or "valid_features" not in aggregation_group
    ):
        raise ValueError(
            f"Aggregated data at '{location}' is incomplete. "
            "Rerun run_pseudotime_aggregation before plotting"
        )
    fingerprint_attr = "input_fingerprints" if artifact_backed else "hashes"
    if hashes != cast(list[str], aggregation_group.attrs[fingerprint_attr]):
        raise ValueError(
            "ERROR: The values under one or more of these columns: `cell_key`, "
            "`feat_key` or/and `pseudotime_key have been updated after running "
            "`run_pseudotime_aggregation`"
        )

    data = ChunkedArray(
        as_zarr_array(aggregation_group["data"], name="data"),
        nthreads=store.nthreads,
    )
    feature_indices = np.asarray(
        as_zarr_array(aggregation_group["feature_indices"], name="feature_indices")[:]
    )
    if "valid_features" not in aggregation_group:
        raise ValueError(
            f"Aggregated data at '{location}' has no valid_features mask. "
            "Rerun run_pseudotime_aggregation"
        )
    valid_features = np.asarray(
        as_zarr_array(aggregation_group["valid_features"], name="valid_features")[:],
        dtype=bool,
    )
    if valid_features.shape[0] != feature_indices.shape[0]:
        raise ValueError("Aggregated feature indices and validity mask are misaligned")
    matrix = np.asarray(data[: feature_indices.shape[0]])
    if matrix.shape[0] != feature_indices.shape[0]:
        raise ValueError("Aggregated feature matrix and feature indices are misaligned")
    matrix = matrix[valid_features]
    feature_indices = feature_indices[valid_features]
    if not np.isfinite(matrix).all():
        raise ValueError("Aggregated feature matrix contains non-finite values")

    all_feature_clusters = assay.feats.fetch_all(feature_cluster_key)
    if artifact_backed:
        if "cluster_values" not in aggregation_group:
            raise ValueError(
                "Aggregation artifact has no stored feature-cluster values"
            )
        stored_cluster_values = np.asarray(
            as_zarr_array(
                aggregation_group["cluster_values"],
                name="cluster_values",
            )[:]
        )
        if not np.array_equal(stored_cluster_values, all_feature_clusters):
            logger.warning(
                f"Feature cluster column '{feature_cluster_key}' changed after "
                "aggregation and may be stale"
            )
    else:
        cached_cluster_label = aggregation_group.attrs.get("cluster_label")
        cached_cluster_digest = aggregation_group.attrs.get("cluster_digest")
        current_cluster_digest = array_digest(
            np.asarray(all_feature_clusters).astype(str)
        )
        if cached_cluster_label is None or cached_cluster_digest is None:
            raise ValueError(
                "Aggregated data has no completed feature-clustering provenance. "
                "Rerun run_pseudotime_aggregation"
            )
        if cached_cluster_label != feature_cluster_key:
            logger.warning(
                f"Heatmap requested feature clusters '{feature_cluster_key}', but "
                f"the aggregation cache was clustered as '{cached_cluster_label}'"
            )
        if cached_cluster_digest != current_cluster_digest:
            logger.warning(
                f"Feature cluster column '{feature_cluster_key}' changed after "
                "aggregation and may be stale"
            )

    feature_clusters = np.asarray(all_feature_clusters)[feature_indices]
    feature_labels = np.asarray(assay.feats.fetch_all("names"))[feature_indices]
    order = np.argsort(feature_clusters)
    return {
        "matrix": matrix[order],
        "feature_indices": feature_indices[order],
        "feature_clusters": feature_clusters[order],
        "feature_labels": feature_labels[order],
        "pseudotime": np.asarray(
            assay.cells.fetch(pseudotime_key, key=cell_key),
            dtype=float,
        ),
        "assay": assay.name,
        "cell_key": cell_key,
        "feat_key": feat_key,
        "feature_cluster_key": feature_cluster_key,
        "pseudotime_key": pseudotime_key,
        "aggregation_location": location,
    }


def pseudotime_heatmap(
    store: Any,
    *,
    from_assay: str | None = None,
    cell_key: str | None = None,
    feat_key: str | None = None,
    feature_cluster_key: str | None = None,
    pseudotime_key: str | None = None,
    show_features: list[str] | None = None,
    figsize: tuple[float, float] = (5, 10),
    vmin: float = -2.0,
    vmax: float = 2.0,
    heatmap_cmap: str | None = None,
    pseudotime_cmap: str | None = None,
    clusterbar_cmap: str | None = None,
    tick_fontsize: int = 10,
    axis_fontsize: int = 12,
    feature_label_fontsize: int = 12,
    theme: str = "notebook",
    show: bool = True,
) -> PlotResult:
    """Plot binned feature dynamics ordered by pseudotime."""
    if vmax <= vmin:
        raise ValueError("vmax must be greater than vmin")
    prepared = _prepare_pseudotime_heatmap(
        store,
        from_assay=from_assay,
        cell_key=cell_key,
        feat_key=feat_key,
        feature_cluster_key=feature_cluster_key,
        pseudotime_key=pseudotime_key,
    )
    matrix = np.asarray(prepared["matrix"])
    feature_clusters = np.asarray(prepared["feature_clusters"])
    feature_labels = np.asarray(prepared["feature_labels"])
    pseudotime = np.asarray(prepared["pseudotime"], dtype=float)
    show_features = [] if show_features is None else list(show_features)
    heatmap_cmap = heatmap_cmap or "coolwarm"
    pseudotime_cmap = pseudotime_cmap or "viridis"
    clusterbar_cmap = clusterbar_cmap or "tab20"

    cluster_order = sort_categories(list(pd.unique(feature_clusters)))
    cluster_codes = {cluster: index for index, cluster in enumerate(cluster_order)}
    encoded_clusters = np.asarray(
        [cluster_codes[value] for value in feature_clusters],
        dtype=float,
    )
    binned_pseudotime = np.asarray(
        [
            values.mean()
            for values in np.array_split(np.sort(pseudotime), matrix.shape[1])
        ],
        dtype=float,
    )

    plt, _ = require_matplotlib()
    with theme_context(theme):
        fig = plt.figure(constrained_layout=False, figsize=figsize)
        grid = fig.add_gridspec(
            nrows=20,
            ncols=20,
            wspace=0,
            hspace=0,
        )
        heatmap_ax = fig.add_subplot(grid[:-2, 1:16])
        cluster_ax = fig.add_subplot(grid[:-2, 17:18])
        colorbar_ax = fig.add_subplot(grid[7:12, -1:])
        pseudotime_ax = fig.add_subplot(grid[-1:, 1:16])

        image = heatmap_ax.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap=heatmap_cmap,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
        heatmap_ax.set_xticks([])
        heatmap_ax.set_yticks([])
        if show_features:
            row_index = {
                str(label).lower(): index for index, label in enumerate(feature_labels)
            }
            visible_features = [
                feature for feature in show_features if feature.lower() in row_index
            ]
            heatmap_ax.set_yticks(
                [row_index[feature.lower()] for feature in visible_features]
            )
            heatmap_ax.set_yticklabels(
                visible_features,
                fontsize=feature_label_fontsize,
            )
        heatmap_ax.set_title(
            f"{matrix.shape[0]} features",
            fontsize=axis_fontsize,
        )
        colorbar = fig.colorbar(image, cax=colorbar_ax)
        colorbar.ax.tick_params(labelsize=tick_fontsize)
        colorbar.set_label("standardized expression", fontsize=axis_fontsize)

        cluster_ax.imshow(
            encoded_clusters.reshape(-1, 1),
            aspect="auto",
            interpolation="nearest",
            cmap=clusterbar_cmap,
            vmin=-0.5,
            vmax=max(len(cluster_order) - 0.5, 0.5),
        )
        cluster_ax.set_xticks([])
        cluster_ax.set_yticks([])
        for cluster in cluster_order:
            row = float(np.where(feature_clusters == cluster)[0].mean())
            cluster_ax.text(
                0,
                row,
                str(cluster),
                fontsize=axis_fontsize,
                ha="center",
                va="center",
            )

        pseudotime_ax.imshow(
            binned_pseudotime.reshape(1, -1),
            aspect="auto",
            interpolation="nearest",
            cmap=pseudotime_cmap,
        )
        pseudotime_ax.set_xticks([])
        pseudotime_ax.set_yticks([])
        pseudotime_ax.set_xlabel(
            "Pseudotime",
            fontsize=axis_fontsize,
        )
        apply_figure_chrome(fig, theme)

    axes: dict[Hashable, Any] = {
        "heatmap": heatmap_ax,
        "feature_clusters": cluster_ax,
        "colorbar": colorbar_ax,
        "pseudotime": pseudotime_ax,
    }
    feature_table = pd.DataFrame(
        {
            "feature_index": np.asarray(prepared["feature_indices"]),
            "feature": feature_labels,
            "cluster": feature_clusters,
        }
    )
    matrix_table = pd.DataFrame(
        matrix,
        index=pd.Index(feature_labels, name="feature"),
    )
    pseudotime_table = pd.DataFrame(
        {
            "cell_order": np.arange(len(pseudotime)),
            "pseudotime": pseudotime,
        }
    )
    pseudotime_bins = pd.DataFrame(
        {
            "bin": np.arange(len(binned_pseudotime)),
            "pseudotime": binned_pseudotime,
        }
    )
    result = PlotResult(
        figure=fig,
        axes=axes,
        tables={
            "matrix": matrix_table,
            "features": feature_table,
            "pseudotime": pseudotime_table,
            "pseudotime_bins": pseudotime_bins,
        },
        legends=(
            LegendSpec(
                kind="colorbar",
                label="standardized expression",
                extras={"vmin": vmin, "vmax": vmax},
            ),
            LegendSpec(kind="categorical", label=feature_cluster_key),
            LegendSpec(kind="colorbar", label=pseudotime_key),
        ),
        scales=(
            ColorScale(cmap=heatmap_cmap, vmin=vmin, vmax=vmax),
            CategoricalScale(order=tuple(cluster_order)),
            ColorScale(
                cmap=pseudotime_cmap,
                vmin=float(np.min(pseudotime)),
                vmax=float(np.max(pseudotime)),
            ),
        ),
        provenance=PlotProvenance(
            scarf_version=_scarf_version(),
            assay=cast(str, prepared["assay"]),
            cell_key=cast(str, prepared["cell_key"]),
            n_cells=len(pseudotime),
            renderer="matplotlib",
            notes=("pseudotime_heatmap", "aggregated"),
            extras={
                "feat_key": prepared["feat_key"],
                "feature_cluster_key": prepared["feature_cluster_key"],
                "pseudotime_key": prepared["pseudotime_key"],
                "aggregation_location": prepared["aggregation_location"],
                "n_features": matrix.shape[0],
                "n_bins": matrix.shape[1],
                "show_features": show_features,
            },
        ),
        owns_figure=True,
        theme=theme,
    )
    if show:
        result.show()
    return result
