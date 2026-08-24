"""Dotplot and matrixplot."""

from collections.abc import Mapping, Sequence
import textwrap
from typing import Any, Hashable

import numpy as np
import pandas as pd

from ._contracts import (
    CategoricalScale,
    ColorScale,
    FeatureRef,
    NormalizationSpec,
    PlotProvenance,
    SizeScale,
    StudyDesign,
)
from ._data import coerce_feature_list, resolve_feature, summarize_features_by_group
from ._deps import require_matplotlib
from ._display import resolve_categorical_scale
from ._figure import LegendSpec, PlotResult, normalize_axes_target
from ._heatmap_utils import (
    annotation_colors,
    draw_annotation_strips,
    normalize_annotations,
    order_heatmap,
)
from ._style import (
    apply_figure_chrome,
    capped_figsize,
    continuous_norm,
    scatter_edgecolor,
    sort_categories,
    theme_context,
)


def _wrap_tick_labels(values: Sequence[Any], width: int | None) -> list[str]:
    labels = [str(value) for value in values]
    if width is None:
        return labels
    if width < 1:
        raise ValueError("label_wrap must be positive or None")
    return [textwrap.fill(label, width=width) for label in labels]


def _default_dot_size_scale(ax: Any, *, n_x: int, n_y: int) -> SizeScale:
    """Fit default marker areas to the physical dot-grid cells."""
    bounds = ax.get_position()
    width_points = max(float(bounds.width * ax.figure.get_figwidth() * 72), 1.0)
    height_points = max(float(bounds.height * ax.figure.get_figheight() * 72), 1.0)
    slot_points = min(
        width_points / max(n_x, 1),
        height_points / max(n_y, 1),
    )
    maximum_diameter = float(np.clip(0.72 * slot_points, 2.5, 26.0))
    minimum_diameter = float(np.clip(0.18 * maximum_diameter, 1.0, 3.0))
    return SizeScale(
        size_min=minimum_diameter**2,
        size_max=maximum_diameter**2,
    )


def _draw_feature_group_brackets(
    ax: Any,
    feature_order: list[str],
    feature_groups: Mapping[str, Any],
    *,
    swap_axes: bool,
) -> int:
    line_color = ax.xaxis.label.get_color()
    grouped_ranges: list[tuple[str, int, int]] = []
    start = 0
    while start < len(feature_order):
        group = feature_groups.get(feature_order[start])
        end = start
        while (
            end + 1 < len(feature_order)
            and feature_groups.get(feature_order[end + 1]) == group
        ):
            end += 1
        if group is not None and not pd.isna(group):
            grouped_ranges.append((str(group), start, end))
        start = end + 1
    left_geometry: tuple[Any, Any, float] | None = None
    if grouped_ranges and not swap_axes:
        from matplotlib.transforms import ScaledTranslation

        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()
        axes_width = max(float(ax.get_window_extent(renderer).width), 1.0)
        label_width = max(
            (
                float(tick.get_window_extent(renderer).width)
                for tick in ax.get_yticklabels()
                if tick.get_visible() and tick.get_text()
            ),
            default=0.0,
        )
        bracket_transform = ax.get_yaxis_transform() + ScaledTranslation(
            -(label_width / ax.figure.dpi + 16.0 / 72.0),
            0,
            ax.figure.dpi_scale_trans,
        )
        text_transform = bracket_transform + ScaledTranslation(
            -7.0 / 72.0,
            0,
            ax.figure.dpi_scale_trans,
        )
        left_geometry = (
            bracket_transform,
            text_transform,
            4.0 * ax.figure.dpi / (72.0 * axes_width),
        )
    for label, start, end in grouped_ranges:
        centre = (start + end) / 2
        if swap_axes:
            transform = ax.get_xaxis_transform()
            line = ax.plot(
                [start - 0.35, end + 0.35],
                [1.03, 1.03],
                transform=transform,
                color=line_color,
                linewidth=0.8,
                clip_on=False,
            )[0]
            ax.plot(
                [start - 0.35, start - 0.35, end + 0.35, end + 0.35],
                [1.00, 1.03, 1.03, 1.00],
                transform=transform,
                color=line_color,
                linewidth=0.8,
                clip_on=False,
            )
            ax.text(
                centre,
                1.07,
                label,
                transform=transform,
                ha="center",
                va="bottom",
                fontsize=7,
                clip_on=False,
            )
        else:
            assert left_geometry is not None
            bracket_transform, text_transform, cap_width = left_geometry
            line = ax.plot(
                [0, 0],
                [start - 0.35, end + 0.35],
                transform=bracket_transform,
                color=line_color,
                linewidth=0.8,
                clip_on=False,
            )[0]
            ax.plot(
                [
                    cap_width,
                    0,
                    0,
                    cap_width,
                ],
                [start - 0.35, start - 0.35, end + 0.35, end + 0.35],
                transform=bracket_transform,
                color=line_color,
                linewidth=0.8,
                clip_on=False,
            )
            group_label = ax.text(
                0,
                centre,
                label,
                transform=text_transform,
                ha="right",
                va="center",
                fontsize=7,
                clip_on=False,
            )
            group_label.set_gid("feature-group-label")
        line.set_gid("feature-group-bracket")
    return len(grouped_ranges)


def _standardize_feature(df: pd.DataFrame, value_col: str = "mean") -> pd.DataFrame:
    out = df.copy()
    means = out.groupby("feature", observed=False)[value_col].transform("mean")
    stds = out.groupby("feature", observed=False)[value_col].transform("std")
    stds = stds.replace(0, np.nan)
    out[value_col] = (out[value_col] - means) / stds
    return out


def _group_axis_labels(df: pd.DataFrame, group_keys: tuple[str, ...]) -> pd.Series:
    if len(group_keys) == 1:
        return df[group_keys[0]].astype(str)
    return df[list(group_keys)].astype(str).agg(" | ".join, axis=1)


def _color_limits(values: np.ndarray, scale: ColorScale) -> tuple[float, float]:
    finite = np.isfinite(values)
    if finite.any():
        if scale.quantiles is not None:
            low, high = scale.quantiles
            vmin = float(np.nanquantile(values[finite], low))
            vmax = float(np.nanquantile(values[finite], high))
        else:
            vmin = float(np.nanmin(values[finite]))
            vmax = float(np.nanmax(values[finite]))
    else:
        vmin, vmax = 0.0, 1.0
    if scale.vmin is not None:
        vmin = scale.vmin
    if scale.vmax is not None:
        vmax = scale.vmax
    if vmax <= vmin:
        if scale.vmin is not None or scale.vmax is not None:
            raise ValueError("Color limits must satisfy vmin < vmax")
        vmax = vmin + 1.0
    return vmin, vmax


def _sample_counts(
    store: Any,
    *,
    cell_key: str,
    sample_by: str | None,
    study_design: StudyDesign | None,
) -> tuple[int | None, int]:
    sample_key = study_design.sample_by if study_design is not None else sample_by
    if sample_key is None:
        return None, 0
    values = np.asarray(store.cells.fetch(sample_key, key=cell_key), dtype=object)
    valid = pd.notna(values) & (values != "")
    return int(pd.Series(values[valid]).nunique()), int((~valid).sum())


def _feature_assays(
    store: Any,
    features: Sequence[str | FeatureRef] | Mapping[str, Sequence[str | FeatureRef]],
    from_assay: str | None,
) -> set[str]:
    assays: set[str] = set()
    for _, feature in coerce_feature_list(features):
        assay = (
            feature.assay
            if isinstance(feature, FeatureRef) and feature.assay is not None
            else from_assay or store._defaultAssay
        )
        assays.add(assay)
    return assays


def dotplot(
    store: Any,
    *,
    features: Sequence[str | FeatureRef] | Mapping[str, Sequence[str | FeatureRef]],
    group_by: str | tuple[str, ...],
    cell_key: str = "I",
    from_assay: str | None = None,
    sample_by: str | None = None,
    study_design: StudyDesign | None = None,
    normalization: NormalizationSpec | None = None,
    expression_cutoff: float = 0.0,
    standardize: str = "none",
    color_scale: ColorScale | None = None,
    size_scale: SizeScale | None = None,
    categorical_scale: CategoricalScale | None = None,
    group_order: Sequence[Any] | None = None,
    feature_order: Sequence[str] | None = None,
    swap_axes: bool = False,
    marker_edgecolor: str | None = None,
    marker_linewidth: float = 0.3,
    label_wrap: int | None = None,
    italicize_features: bool = False,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    max_figure_width: float | None = 7.5,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Dotplot of expression by group.

    Color is the mean value in the group. Dot size is the fraction of cells
    above ``expression_cutoff``. Pass ``features`` as a list of genes, or as a
    mapping of group name to gene list when you want gene-set brackets.

    With ``sample_by`` (or ``study_design.sample_by``), each sample contributes
    equally: Scarf first summarizes within sample, then averages across samples.
    Without it, every cell contributes equally.
    """
    _, mpl = require_matplotlib()
    color_scale = color_scale or ColorScale(cmap="viridis")
    if color_scale.scale != "linear":
        raise NotImplementedError("dotplot currently supports only linear color scales")
    size_scale_is_explicit = size_scale is not None
    normalization = normalization or NormalizationSpec()
    if marker_linewidth < 0:
        raise ValueError("marker_linewidth must be non-negative")
    if isinstance(group_by, str):
        categorical_scale = resolve_categorical_scale(
            store,
            group_by,
            categorical_scale,
        )
    requested_feature_order = list(
        dict.fromkeys(
            resolve_feature(store, feature, from_assay=from_assay).label
            for _, feature in coerce_feature_list(features)
        )
    )

    aggregate, per_sample = summarize_features_by_group(
        store,
        features=features,
        group_by=group_by,
        cell_key=cell_key,
        from_assay=from_assay,
        sample_by=sample_by,
        study_design=study_design,
        normalization=normalization,
        expression_cutoff=expression_cutoff,
    )
    if standardize == "feature":
        aggregate = _standardize_feature(aggregate, "mean")
    elif standardize not in ("none", "feature"):
        raise ValueError("standardize must be 'none' or 'feature'")

    group_keys = (group_by,) if isinstance(group_by, str) else tuple(group_by)
    plot_df = aggregate.copy()
    plot_df["group_label"] = _group_axis_labels(plot_df, group_keys)
    # Preserve feature order from input
    observed_features = [
        value
        for value in requested_feature_order
        if value in set(plot_df["feature"].tolist())
    ]
    observed_groups = list(dict.fromkeys(plot_df["group_label"].tolist()))
    if feature_order is None:
        resolved_feature_order = observed_features
    else:
        resolved_feature_order = [str(value) for value in feature_order]
        missing = [
            value for value in observed_features if value not in resolved_feature_order
        ]
        if missing:
            raise ValueError(
                "feature_order is missing observed features: "
                + ", ".join(map(str, missing[:10]))
            )
    if group_order is not None:
        resolved_group_order = [str(value) for value in group_order]
    elif categorical_scale is not None and categorical_scale.order is not None:
        resolved_group_order = [str(value) for value in categorical_scale.order]
    else:
        resolved_group_order = sort_categories(observed_groups)
    missing_groups = [
        value for value in observed_groups if value not in resolved_group_order
    ]
    if missing_groups:
        raise ValueError(
            "group order is missing observed groups: "
            + ", ".join(map(str, missing_groups[:10]))
        )
    feature_group_map = (
        plot_df[["feature", "feature_group"]]
        .drop_duplicates("feature")
        .set_index("feature")["feature_group"]
        .to_dict()
    )
    plot_df["feature"] = pd.Categorical(
        plot_df["feature"], categories=resolved_feature_order, ordered=True
    )
    plot_df["group_label"] = pd.Categorical(
        plot_df["group_label"], categories=resolved_group_order, ordered=True
    )

    panel_key: Hashable = "dotplot"
    resolved_figsize = figsize
    if resolved_figsize is None and target is None:
        x_count = (
            len(resolved_feature_order) if swap_axes else len(resolved_group_order)
        )
        y_count = (
            len(resolved_group_order) if swap_axes else len(resolved_feature_order)
        )
        resolved_figsize = (
            max(4.6, 0.42 * x_count + 2.6),
            max(3.4, 0.34 * y_count + 2.4),
        )
        resolved_figsize = capped_figsize(
            *resolved_figsize,
            max_width=max_figure_width,
        )
    with theme_context(theme):
        fig, axes, owns = normalize_axes_target(
            target,
            panel_keys=[panel_key],
            figsize=resolved_figsize,
        )
        ax = axes[panel_key]
        if size_scale is None:
            size_scale = _default_dot_size_scale(
                ax,
                n_x=(
                    len(resolved_feature_order)
                    if swap_axes
                    else len(resolved_group_order)
                ),
                n_y=(
                    len(resolved_group_order)
                    if swap_axes
                    else len(resolved_feature_order)
                ),
            )

        vals = plot_df["mean"].to_numpy(dtype=np.float64)
        vmin, vmax = _color_limits(vals, color_scale)
        norm = continuous_norm(
            mpl,
            vmin=vmin,
            vmax=vmax,
            vcenter=color_scale.vcenter,
        )

        areas = size_scale.areas(plot_df["fraction"].to_numpy(dtype=np.float64))
        x = plot_df["group_label"].cat.codes.to_numpy()
        y = plot_df["feature"].cat.codes.to_numpy()

        edgecolor = marker_edgecolor or scatter_edgecolor(theme)
        x_values = x if not swap_axes else y
        y_values = y if not swap_axes else x
        sc = ax.scatter(
            x_values,
            y_values,
            c=vals,
            s=areas,
            cmap=color_scale.cmap or "viridis",
            norm=norm,
            edgecolors=edgecolor,
            linewidths=marker_linewidth,
            rasterized=False,
        )
        if swap_axes:
            ax.set_xticks(range(len(resolved_feature_order)))
            ax.set_xticklabels(
                _wrap_tick_labels(resolved_feature_order, label_wrap),
                rotation=45,
                ha="right",
                fontstyle=("italic" if italicize_features else "normal"),
            )
            ax.set_yticks(range(len(resolved_group_order)))
            ax.set_yticklabels(_wrap_tick_labels(resolved_group_order, label_wrap))
            ax.set_xlim(-0.5, len(resolved_feature_order) - 0.5)
            ax.set_ylim(-0.5, len(resolved_group_order) - 0.5)
            ax.set_xlabel("")
            ax.set_ylabel(" / ".join(group_keys))
        else:
            ax.set_xticks(range(len(resolved_group_order)))
            ax.set_xticklabels(
                _wrap_tick_labels(resolved_group_order, label_wrap),
                rotation=45,
                ha="right",
            )
            ax.set_yticks(range(len(resolved_feature_order)))
            ax.set_yticklabels(
                _wrap_tick_labels(resolved_feature_order, label_wrap),
                fontstyle=("italic" if italicize_features else "normal"),
            )
            ax.set_xlim(-0.5, len(resolved_group_order) - 0.5)
            ax.set_ylim(-0.5, len(resolved_feature_order) - 0.5)
            ax.invert_yaxis()
            ax.set_xlabel(" / ".join(group_keys))
            ax.set_ylabel("")
        bracket_count = _draw_feature_group_brackets(
            ax,
            resolved_feature_order,
            feature_group_map,
            swap_axes=swap_axes,
        )
        colorbar_label = (
            "Mean expression"
            if standardize == "none"
            else "Standardized mean expression"
        )
        if show_legend:
            cb = fig.colorbar(
                sc,
                ax=ax,
                location=("bottom" if swap_axes else "top"),
                orientation="horizontal",
                shrink=0.55,
                pad=(0.08 if swap_axes else 0.04),
                fraction=0.05,
            )
            cb.set_label(colorbar_label)
        legend_values = np.array([0.25, 0.5, 0.75, 1.0])
        legend_areas = size_scale.areas(legend_values)
        legend_area_factor = min(
            1.0,
            180.0 / max(float(legend_areas.max()), 1.0),
        )
        handles = [
            ax.scatter(
                [],
                [],
                s=area * legend_area_factor,
                facecolor="#bdbdbd",
                edgecolor=edgecolor,
                linewidth=marker_linewidth,
            )
            for area in legend_areas
        ]
        # Keep fraction sizes beside the colorbar, away from x tick labels.
        if show_legend:
            legend_kwargs = {
                "handles": handles,
                "labels": [f"{value:.0%}" for value in legend_values],
                "title": "Detected cells",
                "frameon": False,
                "borderaxespad": 0.4,
                "handletextpad": 0.4,
                "labelspacing": 1.1,
            }
            if owns:
                fig.legend(loc="outside right center", **legend_kwargs)
            else:
                ax.legend(
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1),
                    **legend_kwargs,
                )
        apply_figure_chrome(fig, theme)
        if not size_scale_is_explicit:
            fig.canvas.draw()
            size_scale = _default_dot_size_scale(
                ax,
                n_x=(
                    len(resolved_feature_order)
                    if swap_axes
                    else len(resolved_group_order)
                ),
                n_y=(
                    len(resolved_group_order)
                    if swap_axes
                    else len(resolved_feature_order)
                ),
            )
            areas = size_scale.areas(plot_df["fraction"].to_numpy(dtype=np.float64))
            sc.set_sizes(areas)
            legend_areas = size_scale.areas(legend_values)
            legend_area_factor = min(
                1.0,
                180.0 / max(float(legend_areas.max()), 1.0),
            )
            for handle, area in zip(handles, legend_areas, strict=True):
                handle.set_sizes([area * legend_area_factor])

    tables = {"aggregate": aggregate}
    if per_sample is not None:
        tables["per_sample"] = per_sample
    n_samples, dropped_sample_cells = _sample_counts(
        store,
        cell_key=cell_key,
        sample_by=sample_by,
        study_design=study_design,
    )
    assays = _feature_assays(store, features, from_assay)

    result = PlotResult(
        figure=fig,
        axes=axes,
        tables=tables,
        legends=(
            LegendSpec(
                kind="colorbar",
                label=colorbar_label,
                extras={"vmin": vmin, "vmax": vmax},
            ),
            LegendSpec(
                kind="size",
                label="Detected cells",
                extras={"domain": [0.0, 1.0]},
            ),
        ),
        scales=(
            color_scale,
            size_scale,
            CategoricalScale(
                order=tuple(resolved_group_order),
                palette=(
                    {
                        str(value): color
                        for value, color in categorical_scale.palette.items()
                    }
                    if categorical_scale is not None
                    and categorical_scale.palette is not None
                    else None
                ),
                labels=(
                    {
                        str(value): label
                        for value, label in categorical_scale.labels.items()
                    }
                    if categorical_scale is not None
                    and categorical_scale.labels is not None
                    else None
                ),
                palette_name=(
                    categorical_scale.palette_name
                    if categorical_scale is not None
                    else "default"
                ),
            ),
        ),
        provenance=PlotProvenance(
            assay=next(iter(assays)) if len(assays) == 1 else None,
            cell_key=cell_key,
            n_cells=len(store.cells.active_index(cell_key)),
            n_samples=n_samples,
            renderer="matplotlib",
            notes=("dotplot",),
            extras={
                "group_by": list(group_keys),
                "sample_by": (
                    study_design.sample_by if study_design is not None else sample_by
                ),
                "expression_cutoff": expression_cutoff,
                "dropped_sample_cells": dropped_sample_cells,
                "normalization": {
                    "source": normalization.source,
                    "transform": normalization.transform,
                },
                "assays": sorted(assays),
                "feature_group_brackets": bracket_count,
                "swap_axes": swap_axes,
                "group_order": resolved_group_order,
                "feature_order": resolved_feature_order,
                "size_scale_source": (
                    "explicit" if size_scale_is_explicit else "panel"
                ),
                "size_range": [size_scale.size_min, size_scale.size_max],
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        result.show()
    return result


def matrixplot(
    store: Any,
    *,
    features: Sequence[str | FeatureRef] | Mapping[str, Sequence[str | FeatureRef]],
    group_by: str | tuple[str, ...],
    cell_key: str = "I",
    from_assay: str | None = None,
    sample_by: str | None = None,
    study_design: StudyDesign | None = None,
    normalization: NormalizationSpec | None = None,
    expression_cutoff: float = 0.0,
    value: str = "mean",
    standardize: str = "none",
    color_scale: ColorScale | None = None,
    feature_order: Sequence[Any] | None = None,
    group_order: Sequence[Any] | None = None,
    cluster_features: bool = False,
    cluster_groups: bool = False,
    cluster_method: str = "average",
    cluster_metric: str = "euclidean",
    row_annotations: Mapping[
        str,
        Mapping[Any, Any] | Sequence[Any],
    ]
    | None = None,
    column_annotations: Mapping[
        str,
        Mapping[Any, Any] | Sequence[Any],
    ]
    | None = None,
    annotation_scales: Mapping[str, CategoricalScale] | None = None,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Heatmap of mean expression or detection fraction by group.

    Rows and columns preserve input order by default. Set explicit orders or
    enable hierarchical clustering independently for either axis.
    ``value="mean"`` colors by average expression; ``value="fraction"`` colors
    by the share of cells above ``expression_cutoff``. ``sample_by`` has the
    same equal-sample weighting behavior as :func:`dotplot`.
    """
    _, mpl = require_matplotlib()
    if value not in ("mean", "fraction"):
        raise ValueError("value must be 'mean' or 'fraction'")
    color_scale = color_scale or ColorScale(cmap="viridis")
    if color_scale.scale != "linear":
        raise NotImplementedError(
            "matrixplot currently supports only linear color scales"
        )
    normalization = normalization or NormalizationSpec()

    aggregate, per_sample = summarize_features_by_group(
        store,
        features=features,
        group_by=group_by,
        cell_key=cell_key,
        from_assay=from_assay,
        sample_by=sample_by,
        study_design=study_design,
        normalization=normalization,
        expression_cutoff=expression_cutoff,
    )
    plot_df = aggregate.copy()
    if value == "mean" and standardize == "feature":
        plot_df = _standardize_feature(plot_df, "mean")
    elif standardize not in ("none", "feature"):
        raise ValueError("standardize must be 'none' or 'feature'")

    group_keys = (group_by,) if isinstance(group_by, str) else tuple(group_by)
    plot_df["group_label"] = _group_axis_labels(plot_df, group_keys)
    requested_feature_order = list(
        dict.fromkeys(
            resolve_feature(store, feature, from_assay=from_assay).label
            for _, feature in coerce_feature_list(features)
        )
    )
    summarized_features = list(dict.fromkeys(plot_df["feature"].tolist()))
    observed_feature_order = [
        feature
        for feature in requested_feature_order
        if feature in set(summarized_features)
    ]
    observed_feature_order += [
        feature
        for feature in summarized_features
        if feature not in set(observed_feature_order)
    ]
    observed_group_order = list(dict.fromkeys(plot_df["group_label"].tolist()))
    mat = plot_df.pivot_table(
        index="feature",
        columns="group_label",
        values=value,
        observed=False,
    ).reindex(index=observed_feature_order, columns=observed_group_order)
    row_annotation_values = normalize_annotations(
        list(mat.index),
        row_annotations,
        axis_name="row",
    )
    column_annotation_values = normalize_annotations(
        list(mat.columns),
        column_annotations,
        axis_name="column",
    )
    mat, row_linkage, column_linkage = order_heatmap(
        mat,
        row_order=feature_order,
        column_order=group_order,
        cluster_rows=cluster_features,
        cluster_columns=cluster_groups,
        method=cluster_method,
        metric=cluster_metric,
    )
    resolved_feature_order = list(mat.index)
    resolved_group_order = list(mat.columns)
    row_annotation_values = row_annotation_values.reindex(resolved_feature_order)
    column_annotation_values = column_annotation_values.reindex(resolved_group_order)
    row_colors, row_annotation_scales = annotation_colors(
        row_annotation_values,
        annotation_scales,
    )
    column_colors, column_annotation_scales = annotation_colors(
        column_annotation_values,
        annotation_scales,
    )
    resolved_annotation_scales = row_annotation_scales + column_annotation_scales

    panel_key: Hashable = "matrixplot"
    resolved_figsize = figsize
    if resolved_figsize is None and target is None:
        # Extra height keeps rotated group labels from clipping.
        resolved_figsize = (
            max(4.5, 0.45 * len(resolved_group_order) + 2.2),
            max(4.2, 0.45 * len(resolved_feature_order) + 2.8),
        )
    data = mat.to_numpy(dtype=np.float64)
    vmin, vmax = _color_limits(data, color_scale)
    norm = continuous_norm(
        mpl,
        vmin=vmin,
        vmax=vmax,
        vcenter=color_scale.vcenter,
    )

    with theme_context(theme):
        fig, axes, owns = normalize_axes_target(
            target,
            panel_keys=[panel_key],
            figsize=resolved_figsize,
        )
        ax = axes[panel_key]
        im = ax.imshow(
            data,
            aspect="auto",
            cmap=color_scale.cmap or "viridis",
            norm=norm,
            interpolation="nearest",
        )
        ax.set_xticks(range(len(resolved_group_order)))
        ax.set_xticklabels(resolved_group_order, rotation=45, ha="right")
        ax.set_yticks(range(len(resolved_feature_order)))
        ax.set_yticklabels(resolved_feature_order)
        annotation_xlim, annotation_ylim = draw_annotation_strips(
            ax,
            row_colors=row_colors,
            column_colors=column_colors,
            n_rows=len(resolved_feature_order),
            n_columns=len(resolved_group_order),
        )
        ax.set_xlim(annotation_xlim)
        ax.set_ylim(annotation_ylim)
        colorbar_label = (
            "Mean expression"
            if value == "mean" and standardize == "none"
            else "Standardized mean expression"
            if value == "mean"
            else "Detected cells"
        )
        if show_legend:
            cb = fig.colorbar(
                im,
                ax=ax,
                location="top",
                orientation="horizontal",
                shrink=0.8,
                fraction=0.06,
                pad=0.04,
            )
            cb.set_label(colorbar_label)
            annotation_handles: list[Any] = []
            for name, scale in zip(
                list(row_annotation_values.columns)
                + list(column_annotation_values.columns),
                resolved_annotation_scales,
                strict=True,
            ):
                if scale.order is None or scale.palette is None:
                    continue
                annotation_handles.extend(
                    mpl.lines.Line2D(
                        [],
                        [],
                        marker="s",
                        linestyle="",
                        markerfacecolor=scale.palette[item],
                        markeredgecolor="none",
                        markersize=5,
                        label=(
                            f"{name}: "
                            + (
                                scale.labels.get(item, str(item))
                                if scale.labels is not None
                                else str(item)
                            )
                        ),
                    )
                    for item in scale.order
                )
            if annotation_handles:
                legend_kwargs = {
                    "handles": annotation_handles,
                    "title": "Annotations",
                    "frameon": False,
                }
                if owns:
                    fig.legend(loc="outside right center", **legend_kwargs)
                else:
                    ax.legend(
                        loc="upper left",
                        bbox_to_anchor=(1.02, 1),
                        borderaxespad=0,
                        **legend_kwargs,
                    )
        apply_figure_chrome(fig, theme)

    tables = {"aggregate": aggregate, "matrix": mat.reset_index()}
    if per_sample is not None:
        tables["per_sample"] = per_sample
    if not row_annotation_values.empty:
        tables["row_annotations"] = row_annotation_values.reset_index()
    if not column_annotation_values.empty:
        tables["column_annotations"] = column_annotation_values.reset_index()
    n_samples, dropped_sample_cells = _sample_counts(
        store,
        cell_key=cell_key,
        sample_by=sample_by,
        study_design=study_design,
    )
    assays = _feature_assays(store, features, from_assay)

    result = PlotResult(
        figure=fig,
        axes=axes,
        tables=tables,
        legends=(
            LegendSpec(
                kind="colorbar",
                label=colorbar_label,
                extras={"vmin": vmin, "vmax": vmax},
            ),
            *(
                LegendSpec(kind="categorical", label=name)
                for name in list(row_annotation_values.columns)
                + list(column_annotation_values.columns)
            ),
        ),
        scales=(color_scale, *resolved_annotation_scales),
        provenance=PlotProvenance(
            assay=next(iter(assays)) if len(assays) == 1 else None,
            cell_key=cell_key,
            n_cells=len(store.cells.active_index(cell_key)),
            n_samples=n_samples,
            renderer="matplotlib",
            notes=("matrixplot", value),
            extras={
                "group_by": list(group_keys),
                "sample_by": (
                    study_design.sample_by if study_design is not None else sample_by
                ),
                "expression_cutoff": expression_cutoff,
                "dropped_sample_cells": dropped_sample_cells,
                "normalization": {
                    "source": normalization.source,
                    "transform": normalization.transform,
                },
                "assays": sorted(assays),
                "feature_order": resolved_feature_order,
                "group_order": resolved_group_order,
                "cluster_features": row_linkage is not None,
                "cluster_groups": column_linkage is not None,
                "cluster_method": cluster_method,
                "cluster_metric": cluster_metric,
                "row_annotations": list(row_annotation_values.columns),
                "column_annotations": list(column_annotation_values.columns),
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        result.show()
    return result
