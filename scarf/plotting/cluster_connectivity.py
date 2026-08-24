"""Cluster connectivity plots derived from a sparse cell graph."""

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import sparse

from ._contracts import CategoricalScale, PlotProvenance, SizeScale
from ._deps import require_matplotlib
from ._display import resolve_categorical_scale
from ._figure import LegendSpec, PlotResult, normalize_axes_target
from ._style import (
    apply_figure_chrome,
    categorical_color_map,
    finish_embedding_axes,
    refresh_layout_point_sizes,
    register_layout_point_size,
    scatter_edgecolor,
    sort_categories,
    square_axis_limits,
    theme_context,
)


def _fetch_inputs(
    store: Any,
    *,
    group_by: str,
    layout_key: str,
    cell_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        x = np.asarray(
            store.cells.fetch(f"{layout_key}1", key=cell_key),
            dtype=np.float64,
        )
        y = np.asarray(
            store.cells.fetch(f"{layout_key}2", key=cell_key),
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Layout {layout_key!r} coordinates must be numeric") from exc
    groups = np.asarray(store.cells.fetch(group_by, key=cell_key), dtype=object)

    for name, values in (
        (f"{layout_key}1", x),
        (f"{layout_key}2", y),
        (group_by, groups),
    ):
        if values.ndim != 1:
            raise ValueError(f"Cell data {name!r} must be one-dimensional")
    if len(x) == 0:
        raise ValueError(f"No cells selected by cell_key {cell_key!r}")
    if len(y) != len(x) or len(groups) != len(x):
        raise ValueError(
            "Layout coordinates and group values must have matching lengths "
            f"for cell_key {cell_key!r}"
        )
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"Layout {layout_key!r} contains non-finite coordinates")
    if pd.Series(groups, copy=False).isna().any():
        raise ValueError(f"group_by {group_by!r} contains missing values")
    return x, y, groups


def _resolve_categories(
    groups: np.ndarray,
    categorical_scale: CategoricalScale | None,
) -> tuple[list[Any], dict[Any, str], dict[Any, str], CategoricalScale]:
    try:
        observed = list(pd.unique(groups))
        observed_set = set(observed)
    except TypeError as exc:
        raise TypeError("group_by values must be hashable categories") from exc

    if categorical_scale is not None and categorical_scale.order is not None:
        requested_order = list(categorical_scale.order)
        if len(set(requested_order)) != len(requested_order):
            raise ValueError("categorical_scale.order cannot contain duplicates")
        missing = [category for category in observed if category not in requested_order]
        if missing:
            raise ValueError(
                "categorical_scale.order is missing observed values: "
                + ", ".join(map(str, missing[:10]))
            )
        order = [category for category in requested_order if category in observed_set]
    else:
        order = sort_categories(observed)

    palette = categorical_color_map(
        order,
        palette=(categorical_scale.palette if categorical_scale is not None else None),
        palette_name=(
            categorical_scale.palette_name
            if categorical_scale is not None
            else "default"
        ),
    )
    explicit_labels = (
        categorical_scale.labels if categorical_scale is not None else None
    )
    display_labels = {
        category: str(
            explicit_labels.get(category, category)
            if explicit_labels is not None
            else category
        )
        for category in order
    }
    resolved_labels = dict(display_labels) if explicit_labels is not None else None
    resolved_scale = CategoricalScale(
        order=tuple(order),
        palette=dict(palette),
        labels=resolved_labels,
        missing_color=(
            categorical_scale.missing_color
            if categorical_scale is not None
            else "#bdbdbd"
        ),
        missing_label=(
            categorical_scale.missing_label if categorical_scale is not None else "NA"
        ),
        palette_name=(
            categorical_scale.palette_name
            if categorical_scale is not None
            else "default"
        ),
    )
    return order, palette, display_labels, resolved_scale


def _category_codes(groups: np.ndarray, order: list[Any]) -> np.ndarray:
    category_index = {category: index for index, category in enumerate(order)}
    try:
        return np.fromiter(
            (category_index[value] for value in groups),
            dtype=np.intp,
            count=len(groups),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("Could not map every group value to a category") from exc


def _resolve_node_positions(
    x: np.ndarray,
    y: np.ndarray,
    codes: np.ndarray,
    order: list[Any],
    *,
    position: Literal["median", "mean"],
    positions: Mapping[Any, tuple[float, float]] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if position not in ("median", "mean"):
        raise ValueError("position must be 'median' or 'mean'")

    if positions is not None:
        expected = set(order)
        missing = [category for category in order if category not in positions]
        unexpected = [category for category in positions if category not in expected]
        if missing or unexpected:
            details: list[str] = []
            if missing:
                details.append("missing: " + ", ".join(map(str, missing)))
            if unexpected:
                details.append("unexpected: " + ", ".join(map(str, unexpected)))
            raise ValueError(
                "positions must contain exactly the observed categories ("
                + "; ".join(details)
                + ")"
            )
        resolved = np.empty((len(order), 2), dtype=np.float64)
        for index, category in enumerate(order):
            try:
                coordinate = np.asarray(positions[category], dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"Position for category {category!r} must be a numeric pair"
                ) from exc
            if coordinate.shape != (2,) or not np.isfinite(coordinate).all():
                raise ValueError(
                    f"Position for category {category!r} must be a finite pair"
                )
            resolved[index] = coordinate
        return resolved[:, 0], resolved[:, 1]

    node_x = np.empty(len(order), dtype=np.float64)
    node_y = np.empty(len(order), dtype=np.float64)
    reducer = np.median if position == "median" else np.mean
    for index in range(len(order)):
        mask = codes == index
        node_x[index] = float(reducer(x[mask]))
        node_y[index] = float(reducer(y[mask]))
    return node_x, node_y


def _load_graph(
    store: Any,
    *,
    n_cells: int,
    from_assay: str | None,
    cell_key: str,
    feat_key: str | None,
    graph_loc: str | None,
) -> sparse.csr_matrix:
    graph = store.load_graph(
        from_assay=from_assay,
        cell_key=cell_key,
        feat_key=feat_key,
        graph_loc=graph_loc,
        symmetric=True,
    )
    if not sparse.issparse(graph):
        raise TypeError("load_graph must return a scipy sparse matrix")
    if graph.shape != (n_cells, n_cells):
        raise ValueError(
            "Graph shape must match the selected cells; "
            f"expected {(n_cells, n_cells)}, got {graph.shape}"
        )

    graph = graph.tocsr(copy=True).astype(np.float64, copy=False)
    graph.sum_duplicates()
    graph.eliminate_zeros()
    if not np.isfinite(graph.data).all():
        raise ValueError("Graph contains non-finite edge weights")
    if (graph.data < 0).any():
        raise ValueError("Graph edge weights must be non-negative")
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    return graph


def _aggregate_intercluster_edges(
    graph: sparse.csr_matrix,
    codes: np.ndarray,
    n_categories: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate each undirected cell edge once and normalize symmetrically."""
    incident_by_cell = np.asarray(graph.sum(axis=1), dtype=np.float64).ravel()
    incident_by_category = np.bincount(
        codes,
        weights=incident_by_cell,
        minlength=n_categories,
    )

    upper = sparse.triu(graph, k=1, format="coo")
    source_codes = codes[upper.row]
    target_codes = codes[upper.col]
    intercluster = source_codes != target_codes
    if not intercluster.any():
        empty_int = np.empty(0, dtype=np.intp)
        empty_float = np.empty(0, dtype=np.float64)
        return empty_int, empty_int.copy(), empty_float, empty_float.copy()

    first = np.minimum(source_codes[intercluster], target_codes[intercluster])
    second = np.maximum(source_codes[intercluster], target_codes[intercluster])
    pair_keys = first.astype(np.int64) * n_categories + second
    unique_keys, inverse = np.unique(pair_keys, return_inverse=True)
    raw_weights = np.bincount(
        inverse,
        weights=np.asarray(upper.data[intercluster], dtype=np.float64),
    )
    source = (unique_keys // n_categories).astype(np.intp, copy=False)
    target = (unique_keys % n_categories).astype(np.intp, copy=False)

    denominator = np.sqrt(incident_by_category[source] * incident_by_category[target])
    normalized = np.divide(
        raw_weights,
        denominator,
        out=np.zeros_like(raw_weights),
        where=denominator > 0,
    )
    return source, target, raw_weights, normalized


def _filter_edges(
    source: np.ndarray,
    target: np.ndarray,
    raw_weights: np.ndarray,
    normalized_weights: np.ndarray,
    *,
    minimum_edge_weight: float,
    max_edges_per_node: int | None,
    n_categories: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    retained = normalized_weights >= minimum_edge_weight
    source = source[retained]
    target = target[retained]
    raw_weights = raw_weights[retained]
    normalized_weights = normalized_weights[retained]

    if max_edges_per_node is not None and len(source):
        ranked = np.lexsort((target, source, -raw_weights, -normalized_weights))
        degree = np.zeros(n_categories, dtype=np.intp)
        keep = np.zeros(len(source), dtype=bool)
        for edge_index in ranked:
            left = source[edge_index]
            right = target[edge_index]
            if (
                degree[left] >= max_edges_per_node
                or degree[right] >= max_edges_per_node
            ):
                continue
            keep[edge_index] = True
            degree[left] += 1
            degree[right] += 1
        source = source[keep]
        target = target[keep]
        raw_weights = raw_weights[keep]
        normalized_weights = normalized_weights[keep]

    ordered = np.lexsort((target, source))
    return (
        source[ordered],
        target[ordered],
        raw_weights[ordered],
        normalized_weights[ordered],
    )


def _edge_widths(
    weights: np.ndarray,
    edge_width_range: tuple[float, float],
) -> np.ndarray:
    if len(weights) == 0:
        return np.empty(0, dtype=np.float64)
    minimum, maximum = edge_width_range
    weight_min = float(weights.min())
    weight_max = float(weights.max())
    if weight_max <= weight_min:
        return np.full(len(weights), maximum, dtype=np.float64)
    scaled = (weights - weight_min) / (weight_max - weight_min)
    return minimum + scaled * (maximum - minimum)


def _axis_limits(
    x: np.ndarray,
    y: np.ndarray,
    node_x: np.ndarray,
    node_y: np.ndarray,
    *,
    include_cells: bool,
) -> tuple[tuple[float, float], tuple[float, float]]:
    limit_x = np.concatenate((x, node_x)) if include_cells else node_x
    limit_y = np.concatenate((y, node_y)) if include_cells else node_y
    x_span = float(limit_x.max() - limit_x.min())
    y_span = float(limit_y.max() - limit_y.min())
    x_pad = 0.05 * (x_span if x_span > 0 else 1.0)
    y_pad = 0.05 * (y_span if y_span > 0 else 1.0)
    return square_axis_limits(
        (float(limit_x.min() - x_pad), float(limit_x.max() + x_pad)),
        (float(limit_y.min() - y_pad), float(limit_y.max() + y_pad)),
    )


def cluster_connectivity(
    store: Any,
    *,
    group_by: str,
    layout_key: str,
    cell_key: str = "I",
    from_assay: str | None = None,
    feat_key: str | None = None,
    graph_loc: str | None = None,
    position: Literal["median", "mean"] = "median",
    positions: Mapping[Any, tuple[float, float]] | None = None,
    categorical_scale: CategoricalScale | None = None,
    size_scale: SizeScale | None = None,
    minimum_edge_weight: float = 0.02,
    max_edges_per_node: int | None = None,
    show_cells: bool = False,
    cell_size: float | None = None,
    cell_alpha: float = 0.3,
    cell_color: str | None = None,
    node_edgecolor: str | None = None,
    node_linewidth: float = 0.8,
    edge_color: str | None = None,
    edge_alpha: float = 0.45,
    edge_width_range: tuple[float, float] = (0.4, 5.0),
    labels: bool = True,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show: bool = True,
) -> PlotResult:
    """Plot cluster connectivity on an existing two-dimensional cell layout.

    ``rawWeight`` is the sum of intercluster weights after reciprocal sparse
    graph entries are counted once. ``normalizedWeight`` divides that value by
    ``sqrt(incidentWeight[source] * incidentWeight[target])``. Incident weight
    is the sum of weighted cell degrees in each cluster, including internal
    cluster edges. Clusters with zero incident weight receive a normalized
    value of zero.
    """
    if not np.isfinite(minimum_edge_weight) or minimum_edge_weight < 0:
        raise ValueError("minimum_edge_weight must be finite and non-negative")
    if max_edges_per_node is not None and (
        isinstance(max_edges_per_node, bool)
        or not isinstance(max_edges_per_node, int)
        or max_edges_per_node < 0
    ):
        raise ValueError("max_edges_per_node must be a non-negative integer or None")
    if cell_size is not None and (not np.isfinite(cell_size) or cell_size < 0):
        raise ValueError("cell_size must be finite and non-negative")
    if not np.isfinite(cell_alpha) or not 0 <= cell_alpha <= 1:
        raise ValueError("cell_alpha must be between 0 and 1")
    if not np.isfinite(node_linewidth) or node_linewidth < 0:
        raise ValueError("node_linewidth must be finite and non-negative")
    if not np.isfinite(edge_alpha) or not 0 <= edge_alpha <= 1:
        raise ValueError("edge_alpha must be between 0 and 1")
    if len(edge_width_range) != 2:
        raise ValueError("edge_width_range must contain two values")
    edge_width_min, edge_width_max = edge_width_range
    if (
        not np.isfinite(edge_width_min)
        or not np.isfinite(edge_width_max)
        or edge_width_min < 0
        or edge_width_max < edge_width_min
    ):
        raise ValueError(
            "edge_width_range must be finite and satisfy 0 <= minimum <= maximum"
        )

    x, y, groups = _fetch_inputs(
        store,
        group_by=group_by,
        layout_key=layout_key,
        cell_key=cell_key,
    )
    categorical_scale = resolve_categorical_scale(
        store,
        group_by,
        categorical_scale,
    )
    order, palette, display_labels, resolved_categorical_scale = _resolve_categories(
        groups, categorical_scale
    )
    codes = _category_codes(groups, order)
    node_x, node_y = _resolve_node_positions(
        x,
        y,
        codes,
        order,
        position=position,
        positions=positions,
    )

    graph = _load_graph(
        store,
        n_cells=len(x),
        from_assay=from_assay,
        cell_key=cell_key,
        feat_key=feat_key,
        graph_loc=graph_loc,
    )
    source_index, target_index, raw_weights, normalized_weights = (
        _aggregate_intercluster_edges(graph, codes, len(order))
    )
    aggregated_edge_count = len(source_index)
    source_index, target_index, raw_weights, normalized_weights = _filter_edges(
        source_index,
        target_index,
        raw_weights,
        normalized_weights,
        minimum_edge_weight=minimum_edge_weight,
        max_edges_per_node=max_edges_per_node,
        n_categories=len(order),
    )

    counts = np.bincount(codes, minlength=len(order))
    proportions = counts.astype(np.float64) / len(groups)
    resolved_size_scale = size_scale or SizeScale(size_min=70.0, size_max=650.0)
    node_sizes = resolved_size_scale.areas(proportions)
    if not np.isfinite(node_sizes).all() or (node_sizes < 0).any():
        raise ValueError("size_scale produced invalid marker areas")

    category_values = np.empty(len(order), dtype=object)
    category_values[:] = order
    nodes = pd.DataFrame(
        {
            "category": category_values,
            "x": node_x,
            "y": node_y,
            "nCells": counts,
            "proportion": proportions,
            "size": node_sizes,
            "displayLabel": [display_labels[category] for category in order],
        }
    )
    edges = pd.DataFrame(
        {
            "source": category_values[source_index],
            "target": category_values[target_index],
            "rawWeight": raw_weights,
            "normalizedWeight": normalized_weights,
        }
    )

    require_matplotlib()
    from matplotlib import patheffects
    from matplotlib.collections import LineCollection

    panel_key = "cluster_connectivity"
    with theme_context(theme):
        resolved_figsize = (5.0, 5.0) if target is None and figsize is None else figsize
        figure, axes, owns_figure = normalize_axes_target(
            target,
            panel_keys=[panel_key],
            figsize=resolved_figsize,
        )
        ax = axes[panel_key]
        resolved_node_edgecolor = node_edgecolor or scatter_edgecolor(theme)
        resolved_edge_color = edge_color or scatter_edgecolor(theme)

        cell_artist = None
        if show_cells:
            resolved_cell_size = 6.0 if cell_size is None else cell_size
            cell_artist = ax.scatter(
                x,
                y,
                s=resolved_cell_size,
                c=(
                    [palette[group] for group in groups]
                    if cell_color is None
                    else cell_color
                ),
                alpha=cell_alpha,
                linewidths=0,
                rasterized=len(x) >= 50_000,
                zorder=0,
            )
            if cell_size is None:
                register_layout_point_size(
                    cell_artist,
                    n_points=len(x),
                    size_min=4.0,
                    size_max=12.0,
                )

        segments = np.empty((len(edges), 2, 2), dtype=np.float64)
        if len(edges):
            segments[:, 0, 0] = node_x[source_index]
            segments[:, 0, 1] = node_y[source_index]
            segments[:, 1, 0] = node_x[target_index]
            segments[:, 1, 1] = node_y[target_index]
        line_collection = LineCollection(
            segments.tolist(),
            colors=resolved_edge_color,
            linewidths=_edge_widths(normalized_weights, edge_width_range),
            alpha=edge_alpha,
            zorder=1,
        )
        ax.add_collection(line_collection)
        ax.scatter(
            node_x,
            node_y,
            s=node_sizes,
            c=[palette[category] for category in order],
            edgecolors=resolved_node_edgecolor,
            linewidths=node_linewidth,
            zorder=2,
        )

        if labels:
            text_color = "#f5f5f5" if theme == "dark" else "#222222"
            stroke_color = "#222222" if theme == "dark" else "#ffffff"
            diameters = np.sqrt(np.asarray(node_sizes, dtype=np.float64))
            for node_index, node in enumerate(nodes.itertuples(index=False)):
                text = str(node.displayLabel)
                # Shrink long labels so they stay inside their marker.
                fitted = 0.85 * diameters[node_index] / (0.62 * max(len(text), 1))
                ax.text(
                    node.x,
                    node.y,
                    text,
                    ha="center",
                    va="center",
                    fontsize=float(np.clip(fitted, 5.0, 8.0)),
                    color=text_color,
                    path_effects=[
                        patheffects.withStroke(
                            linewidth=2.0,
                            foreground=stroke_color,
                        )
                    ],
                    zorder=3,
                )

        xlim, ylim = _axis_limits(
            x,
            y,
            node_x,
            node_y,
            include_cells=positions is None or show_cells,
        )
        finish_embedding_axes(
            ax,
            xlim=xlim,
            ylim=ylim,
            frame="none",
        )
        apply_figure_chrome(figure, theme)
        if cell_artist is not None and cell_size is None:
            refresh_layout_point_sizes(figure)

    result = PlotResult(
        figure=figure,
        axes=axes,
        tables={"nodes": nodes, "edges": edges},
        legends=(
            LegendSpec(
                kind="categorical",
                label=group_by,
                extras={
                    "categories": list(order),
                    "placement": "nodes" if labels else "none",
                },
            ),
        ),
        scales=(resolved_categorical_scale, resolved_size_scale),
        provenance=PlotProvenance(
            assay=from_assay or getattr(store, "_defaultAssay", None),
            cell_key=cell_key,
            n_cells=len(groups),
            renderer="matplotlib",
            notes=("cluster_connectivity", "materialized", f"layout={layout_key}"),
            extras={
                "group_by": group_by,
                "layout_key": layout_key,
                "feat_key": feat_key,
                "graph_loc": graph_loc,
                "position": "explicit" if positions is not None else position,
                "minimum_edge_weight": minimum_edge_weight,
                "max_edges_per_node": max_edges_per_node,
                "show_cells": show_cells,
                "cell_size": (
                    None if cell_artist is None else float(cell_artist.get_sizes()[0])
                ),
                "cell_size_source": (
                    "hidden"
                    if not show_cells
                    else "panel"
                    if cell_size is None
                    else "explicit"
                ),
                "cell_alpha": cell_alpha,
                "cell_color": "category" if cell_color is None else cell_color,
                "n_nodes": len(nodes),
                "n_aggregated_edges": aggregated_edge_count,
                "n_edges": len(edges),
                "normalization": (
                    "rawWeight / sqrt(incidentWeight[source] * incidentWeight[target])"
                ),
            },
        ),
        owns_figure=owns_figure,
        theme=theme,
    )
    if show:
        result.show()
    return result
