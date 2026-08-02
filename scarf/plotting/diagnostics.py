"""Native diagnostic plots."""

from typing import Any

import numpy as np
import pandas as pd

from ._contracts import CategoricalScale, PlotProvenance
from ._deps import require_kneed, require_matplotlib, require_seaborn
from ._figure import LegendSpec, PlotResult, normalize_axes_target
from ._style import apply_figure_chrome, sort_categories, theme_context


def _clean_axis(ax: Any, *, tick_size: float = 10.0) -> None:
    ax.tick_params(axis="both", labelsize=tick_size)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(which="major", linestyle="--", alpha=0.4)


def _return_result(result: PlotResult, *, show: bool) -> PlotResult:
    if show:
        result.show()
    return result


def qc(
    data: pd.DataFrame,
    color: str = "steelblue",
    cmap: str = "tab20",
    figsize: tuple[float, float] | None = None,
    label_size: float = 10.0,
    title_size: float = 10,
    sup_title: str | None = None,
    sup_title_size: float = 12,
    scatter_size: float = 1.0,
    max_points: int = 10_000,
    seed: int = 0,
    show_on_single_row: bool = True,
    show: bool = True,
    theme: str = "notebook",
) -> PlotResult:
    """Plot grouped distributions for cell-level QC metrics."""
    require_matplotlib()
    sns = require_seaborn()
    if "groups" not in data.columns:
        raise KeyError("data must contain a 'groups' column")
    if data.empty:
        raise ValueError("data must contain at least one row")
    if max_points < 0:
        raise ValueError("max_points must be non-negative")

    metric_columns = [column for column in data.columns if column != "groups"]
    if not metric_columns:
        raise ValueError("data must contain at least one metric column")
    if len(set(metric_columns)) != len(metric_columns):
        raise ValueError("metric column names must be unique")

    numeric_values: dict[Any, np.ndarray] = {}
    for metric in metric_columns:
        try:
            numeric_values[metric] = pd.to_numeric(
                data[metric], errors="raise"
            ).to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"QC metric {metric!r} must be numeric") from exc

    group_values = data["groups"].astype(object).where(data["groups"].notna(), "NA")
    group_order = sort_categories(list(pd.unique(group_values)))
    n_groups = len(group_order)
    n_panels = len(metric_columns)
    n_columns = n_panels if show_on_single_row else 1
    n_rows = 1 if show_on_single_row else n_panels
    resolved_figsize = figsize
    if resolved_figsize is None:
        resolved_figsize = (
            min(15.0, n_groups + (2.0 * n_columns)),
            1.0 + (2.5 * n_rows),
        )

    palette_colors = sns.color_palette(cmap, n_colors=n_groups).as_hex()
    palette = dict(zip(group_order, palette_colors, strict=True))
    if n_groups == 1:
        palette[group_order[0]] = color

    long_frames: list[pd.DataFrame] = []
    displayed_points: dict[str, int] = {}
    with theme_context(theme):
        figure, axes, owns_figure = normalize_axes_target(
            None,
            panel_keys=metric_columns,
            figsize=resolved_figsize,
            n_columns=n_columns,
        )
        for metric in metric_columns:
            ax = axes[metric]
            metric_frame = pd.DataFrame(
                {
                    "group": group_values.to_numpy(),
                    "metric": str(metric),
                    "value": numeric_values[metric],
                }
            )
            long_frames.append(metric_frame)
            collections_before = len(ax.collections)
            if n_groups == 1:
                sns.violinplot(
                    data=metric_frame,
                    x="group",
                    y="value",
                    order=group_order,
                    ax=ax,
                    color=color,
                    linewidth=1,
                    inner=None,
                    cut=0,
                    saturation=1,
                )
            else:
                sns.violinplot(
                    data=metric_frame,
                    x="group",
                    y="value",
                    order=group_order,
                    hue="group",
                    hue_order=group_order,
                    palette=palette,
                    legend=False,
                    dodge=False,
                    ax=ax,
                    linewidth=1,
                    inner=None,
                    cut=0,
                    saturation=1,
                )
            for collection in ax.collections[collections_before:]:
                collection.set_alpha(0.6)

            if max_points == 0:
                display_frame = metric_frame.iloc[:0]
            elif len(metric_frame) > max_points:
                display_frame = metric_frame.sample(
                    n=max_points,
                    random_state=seed,
                )
            else:
                display_frame = metric_frame
            displayed_points[str(metric)] = len(display_frame)
            if not display_frame.empty:
                sns.stripplot(
                    data=display_frame,
                    x="group",
                    y="value",
                    order=group_order,
                    jitter=0.4,
                    ax=ax,
                    size=scatter_size,
                    color="black",
                    alpha=0.4,
                )

            ax.set_ylabel(str(metric), fontsize=label_size)
            ax.set_xlabel("")
            if n_groups == 1:
                ax.set_xticks([])
                finite = numeric_values[metric][np.isfinite(numeric_values[metric])]
                if len(finite):
                    ax.set_title(
                        f"Median: {np.median(finite):.1f}",
                        fontsize=title_size,
                    )
            _clean_axis(ax, tick_size=label_size)

        if sup_title is not None:
            figure.suptitle(sup_title, fontsize=sup_title_size)
        apply_figure_chrome(figure, theme)

    long_data = pd.concat(long_frames, ignore_index=True)
    summary = (
        long_data.groupby(["group", "metric"], observed=False, dropna=False, sort=False)
        .agg(count=("value", "count"), median=("value", "median"))
        .reset_index()
    )
    result = PlotResult(
        figure=figure,
        axes=axes,
        tables={
            "data": data.copy(),
            "summary": summary,
        },
        legends=(
            LegendSpec(
                kind="categorical",
                label="groups",
                extras={"categories": list(group_order)},
            ),
        ),
        scales=(
            CategoricalScale(
                order=tuple(group_order),
                palette=palette,
            ),
        ),
        provenance=PlotProvenance(
            n_cells=len(data),
            renderer="matplotlib",
            notes=("qc",),
            extras={
                "metrics": [str(metric) for metric in metric_columns],
                "groups": list(group_order),
                "max_points": max_points,
                "seed": seed,
                "displayed_points": displayed_points,
                "show_on_single_row": show_on_single_row,
            },
        ),
        owns_figure=owns_figure,
        theme=theme,
    )
    return _return_result(result, show=show)


def elbow(
    variance_explained: np.ndarray | list[float],
    figsize: tuple[float | None, float] = (None, 2),
    theme: str = "notebook",
    show: bool = True,
) -> PlotResult:
    """Plot explained variance and mark the detected elbow."""
    require_matplotlib()
    values = np.asarray(variance_explained, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("variance_explained must be a non-empty one-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("variance_explained must contain only finite values")

    components = np.arange(len(values), dtype=np.int64)
    knee_locator = require_kneed()
    locator = knee_locator(
        components,
        values,
        S=1.0,
        curve="convex",
        direction="decreasing",
    )
    detected = getattr(locator, "elbow", None)
    elbow_component = None if detected is None else int(detected)
    width, height = figsize
    resolved_figsize = (
        0.25 * len(values) if width is None else float(width),
        float(height),
    )

    with theme_context(theme):
        figure, axes, owns_figure = normalize_axes_target(
            None,
            panel_keys=["elbow"],
            figsize=resolved_figsize,
        )
        ax = axes["elbow"]
        ax.plot(components, values, linewidth=1)
        ax.set_xticks(components)
        if elbow_component is not None:
            ax.axvline(
                elbow_component,
                linewidth=1,
                color="red",
                label="Elbow",
            )
            ax.legend(frameon=False, fontsize=9)
        ax.set_ylabel("% Variance explained", fontsize=9)
        ax.set_xlabel("Principal components", fontsize=9)
        _clean_axis(ax, tick_size=8)
        apply_figure_chrome(figure, theme)

    table = pd.DataFrame(
        {
            "component": components,
            "variance_explained": values,
            "is_elbow": components == elbow_component,
        }
    )
    result = PlotResult(
        figure=figure,
        axes=axes,
        tables={"variance_explained": table},
        legends=(
            ()
            if elbow_component is None
            else (
                LegendSpec(
                    kind="line",
                    label="Elbow",
                    scale_key="component",
                    extras={"component": elbow_component},
                ),
            )
        ),
        scales=(),
        provenance=PlotProvenance(
            n_cells=0,
            renderer="matplotlib",
            notes=("elbow",),
            extras={
                "elbow": elbow_component,
                "n_components": len(values),
                "sensitivity": 1.0,
            },
        ),
        owns_figure=owns_figure,
        theme=theme,
    )
    return _return_result(result, show=show)


def graph_qc(
    graph: Any,
    theme: str = "notebook",
    show: bool = True,
) -> PlotResult:
    """Plot node degree and edge weight distributions for a sparse graph."""
    require_matplotlib()
    shape = getattr(graph, "shape", None)
    if shape is None or len(shape) != 2:
        raise TypeError("graph must be a two-dimensional sparse matrix")
    if not hasattr(graph, "data"):
        raise TypeError("graph must expose sparse edge weights through .data")

    try:
        degrees = np.asarray((graph != 0).sum(axis=0)).ravel().astype(np.int64)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError(
            "graph must support sparse non-zero degree calculation"
        ) from exc
    edge_weights = np.asarray(graph.data, dtype=np.float64).ravel()
    node_degrees = pd.DataFrame(
        {
            "node": np.arange(len(degrees), dtype=np.int64),
            "degree": degrees,
        }
    )
    degree_frequencies = (
        pd.Series(degrees, name="degree")
        .value_counts(sort=False)
        .sort_index()
        .rename_axis("degree")
        .reset_index(name="frequency")
    )
    degree_clip_limit = (
        float(np.percentile(degrees, 99.5) + 5.0) if len(degrees) else 1.0
    )

    with theme_context(theme):
        figure, axes, owns_figure = normalize_axes_target(
            None,
            panel_keys=["node_degree", "edge_weight"],
            figsize=(12.0, 4.0),
            n_columns=2,
        )
        degree_ax = axes["node_degree"]
        degree_ax.bar(
            degree_frequencies["degree"],
            degree_frequencies["frequency"],
            width=0.5,
        )
        degree_ax.set_xlim((0.0, max(1.0, degree_clip_limit)))
        degree_ax.set_xlabel("Node degree")
        degree_ax.set_ylabel("Frequency")
        if len(degrees) and degrees.max() > degree_clip_limit:
            degree_ax.text(
                max(1.0, degree_clip_limit),
                float(degree_frequencies["frequency"].max()),
                f"plot is clipped (max degree: {degrees.max()})",
                ha="right",
                fontsize=9,
            )
        _clean_axis(degree_ax)

        weight_ax = axes["edge_weight"]
        weight_ax.hist(edge_weights, bins=30)
        weight_ax.set_xlabel("Edge weight")
        weight_ax.set_ylabel("Frequency")
        _clean_axis(weight_ax)
        apply_figure_chrome(figure, theme)

    result = PlotResult(
        figure=figure,
        axes=axes,
        tables={
            "node_degrees": node_degrees,
            "degree_frequencies": degree_frequencies,
            "edge_weights": pd.DataFrame({"edge_weight": edge_weights}),
        },
        legends=(
            LegendSpec(kind="bar", label="Node degree frequency"),
            LegendSpec(
                kind="histogram",
                label="Edge weight",
                extras={"bins": 30},
            ),
        ),
        scales=(),
        provenance=PlotProvenance(
            n_cells=len(degrees),
            renderer="matplotlib",
            notes=("graph_qc",),
            extras={
                "graph_shape": [int(shape[0]), int(shape[1])],
                "n_edges": len(edge_weights),
                "degree_clip_limit": degree_clip_limit,
            },
        ),
        owns_figure=owns_figure,
        theme=theme,
    )
    return _return_result(result, show=show)


def highly_variable_features(
    mean_nonzero: np.ndarray,
    corrected_variance: np.ndarray,
    n_cells: np.ndarray,
    selected: np.ndarray,
    *,
    label_size: float = 12,
    figsize: tuple[float, float] = (4.5, 4.0),
    point_sizes: tuple[float, float] = (3, 30),
    colormaps: tuple[str, str] = ("winter", "magma_r"),
    theme: str = "notebook",
    show: bool = True,
) -> PlotResult:
    """Plot mean-variance diagnostics for feature selection."""
    require_matplotlib()
    means = np.asarray(mean_nonzero, dtype=np.float64)
    variances = np.asarray(corrected_variance, dtype=np.float64)
    cell_counts = np.asarray(n_cells, dtype=np.float64)
    selected_mask = np.asarray(selected)
    arrays = (means, variances, cell_counts, selected_mask)
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("feature diagnostic inputs must be one-dimensional")
    if len({len(array) for array in arrays}) != 1:
        raise ValueError("feature diagnostic inputs must have matching lengths")
    selected_mask = selected_mask.astype(bool, copy=False)
    if len(point_sizes) != 2:
        raise ValueError("point_sizes must contain two values")
    if len(colormaps) != 2:
        raise ValueError("colormaps must contain two values")

    with np.errstate(divide="ignore", invalid="ignore"):
        log_means = np.log2(means)
        log_variances = np.log2(variances)

    with theme_context(theme):
        figure, axes, owns_figure = normalize_axes_target(
            None,
            panel_keys=["highly_variable_features"],
            figsize=figsize,
        )
        ax = axes["highly_variable_features"]
        ax.scatter(
            log_means[~selected_mask],
            log_variances[~selected_mask],
            alpha=0.6,
            c=cell_counts[~selected_mask],
            cmap=colormaps[0],
            s=point_sizes[0],
            label="Not selected",
        )
        ax.scatter(
            log_means[selected_mask],
            log_variances[selected_mask],
            alpha=0.8,
            c=cell_counts[selected_mask],
            cmap=colormaps[1],
            s=point_sizes[1],
            edgecolor="black",
            linewidth=0.5,
            label="Selected",
        )
        ax.set_xlabel("Log mean non-zero expression", fontsize=label_size)
        ax.set_ylabel("Log corrected variance", fontsize=label_size)
        ax.legend(frameon=False, fontsize=max(8.0, label_size - 2.0))
        _clean_axis(ax, tick_size=label_size)
        apply_figure_chrome(figure, theme)

    feature_table = pd.DataFrame(
        {
            "mean_nonzero": means,
            "corrected_variance": variances,
            "n_cells": cell_counts,
            "selected": selected_mask,
            "log2_mean_nonzero": log_means,
            "log2_corrected_variance": log_variances,
        }
    )
    result = PlotResult(
        figure=figure,
        axes=axes,
        tables={"features": feature_table},
        legends=(
            LegendSpec(
                kind="categorical",
                label="Highly variable",
                extras={"categories": ["Not selected", "Selected"]},
            ),
        ),
        scales=(),
        provenance=PlotProvenance(
            n_cells=0,
            renderer="matplotlib",
            notes=("highly_variable_features",),
            extras={
                "n_features": len(means),
                "n_selected": int(selected_mask.sum()),
                "max_expressing_cells": (
                    float(np.nanmax(cell_counts)) if len(cell_counts) else 0.0
                ),
                "point_sizes": list(point_sizes),
                "colormaps": list(colormaps),
            },
        ),
        owns_figure=owns_figure,
        theme=theme,
    )
    return _return_result(result, show=show)
