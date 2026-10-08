"""Embedding scatter plots."""

from collections.abc import Collection, Sequence
from dataclasses import replace
from typing import Any, Hashable

import numpy as np
import pandas as pd

from ..metadata.rows import apply_missing_mask
from ..metadata.selection import CellFieldKind, cell_value_spec
from ..storage.artifacts import ArtifactRef
from ._contracts import (
    CategoricalScale,
    CellField,
    ColorScale,
    DensityOverlay,
    FeatureRef,
    Highlight,
    NormalizationSpec,
    PlotProvenance,
)
from ._data import (
    _artifact_cell_selection,
    _cell_column_values,
    _cell_metadata_columns,
    _fetch_cell_column,
    _resolve_grouping,
    _resolve_layout,
    fetch_normalized_feature_matrix,
    has_finite_values,
    resolve_cell_selection,
    resolve_feature,
    unmeasured_cell_counts,
    unmeasured_extras,
)
from ._deps import require_matplotlib
from ._display import (
    categorical_display_scale,
    continuous_display_scale,
    stored_display_metadata,
)
from ._figure import (
    LegendSpec,
    PlotResult,
    _place_legend_blocks,
    close_figures_on_error,
    normalize_axes_target,
)
from ..utils.arguments import clip_fraction_argument
from ..utils.arrays import sort_categories
from ._style import (
    DEFAULT_PANEL_INCHES,
    DEFAULT_RASTERIZE_THRESHOLD,
    LEGEND_SIDE_MAX_ENTRIES,
    FrameStyle,
    LegendLoc,
    apply_figure_chrome,
    category_label,
    continuous_norm,
    default_point_edgewidth,
    default_point_size,
    finish_embedding_axes,
    legend_side_columns,
    padded_square_limits,
    panel_area_inches,
    refresh_layout_point_sizes,
    register_layout_point_size,
    resolve_category_scale,
    resolve_color_limits,
    resolve_legend_loc,
    scatter_edgecolor,
    scatter_edges,
    theme_context,
)


def _is_categorical(values: pd.Series, kind: str) -> bool:
    if kind == "categorical":
        return True
    if kind == "continuous":
        return False
    if values.dtype.name in (
        "category",
        "object",
        "bool",
        "string",
    ) or pd.api.types.is_string_dtype(values):
        return True
    if pd.api.types.is_integer_dtype(values) and values.nunique(dropna=True) <= 100:
        return True
    return False


def _coerce_color_items(
    color_by: str
    | ArtifactRef
    | FeatureRef
    | CellField
    | Sequence[str | ArtifactRef | FeatureRef | CellField]
    | None,
) -> list[str | ArtifactRef | FeatureRef | CellField | None]:
    if color_by is None:
        return [None]
    if isinstance(color_by, (str, ArtifactRef, FeatureRef, CellField)):
        return [color_by]
    return list(color_by)


def _coerce_layout_items(layout_key: str | Sequence[str]) -> list[str]:
    if isinstance(layout_key, str):
        return [layout_key]
    layouts = list(layout_key)
    if not layouts:
        raise ValueError("layout_key must contain at least one layout")
    if any(not isinstance(layout, str) for layout in layouts):
        raise TypeError("Every layout_key entry must be a string")
    if len(set(layouts)) != len(layouts):
        raise ValueError("layout_key entries must be unique")
    return layouts


def _color_labels(
    store: Any,
    color_items: Sequence[str | ArtifactRef | FeatureRef | CellField | None],
    *,
    from_assay: str | None,
) -> list[str]:
    metadata_columns = _cell_metadata_columns(store, color_items)
    labels: list[str] = []
    for item in color_items:
        if item is None:
            labels.append("cells")
        elif isinstance(item, ArtifactRef):
            labels.append(item.kind)
        elif isinstance(item, CellField):
            labels.append(item.label or item.key)
        elif isinstance(item, str) and item in metadata_columns:
            labels.append(item)
        else:
            labels.append(resolve_feature(store, item, from_assay=from_assay).label)
    return labels


def _embedding_panel_keys(
    labels: Sequence[str],
    facets: Sequence[Any],
) -> list[Hashable]:
    panel_keys: list[Hashable] = []
    for label in labels:
        for facet in facets:
            panel_keys.append(label if facet is None else (label, facet))
    if len(set(panel_keys)) != len(panel_keys):
        return [(index, key) for index, key in enumerate(panel_keys)]
    return panel_keys


def _layout_panel_key(layout: str, panel_key: Hashable) -> Hashable:
    if isinstance(panel_key, tuple):
        return (layout, *panel_key)
    return (layout, panel_key)


def _multi_layout_facets(
    store: Any,
    *,
    facet_by: str | None,
    facet_order: Sequence[Any] | None,
    groups: Sequence[Any] | None,
    subset_by: str | None,
    cell_key: str,
) -> list[Any]:
    if facet_by is None:
        return [None]
    if groups is not None:
        return list(groups)
    if facet_order is not None:
        return list(facet_order)

    values = _fetch_cell_column(store, facet_by, cell_key=cell_key, labels=True)
    subset = (
        _fetch_cell_column(store, subset_by, cell_key=cell_key)
        if subset_by is not None
        else None
    )
    selection, _ = resolve_cell_selection(
        len(values),
        subset=subset,
        subset_name=subset_by,
    )
    return sort_categories(list(pd.unique(values[selection])))


def _embedding_multiple_layouts(
    store: Any,
    *,
    layout_keys: Sequence[str],
    color_by: str
    | ArtifactRef
    | FeatureRef
    | CellField
    | Sequence[str | ArtifactRef | FeatureRef | CellField]
    | None,
    facet_by: str | None,
    facet_order: Sequence[Any] | None,
    cell_key: str,
    from_assay: str | None,
    normalization: NormalizationSpec | None,
    groups: Sequence[Any] | None,
    subset_by: str | None,
    n_columns: int | None,
    target: Any | None,
    figsize: tuple[float, float] | None,
    theme: str,
    show_titles: bool,
    show: bool,
    child_kwargs: dict[str, Any],
) -> PlotResult:
    color_items = _coerce_color_items(color_by)
    if not color_items:
        raise ValueError("color_by must contain at least one item")
    labels = _color_labels(store, color_items, from_assay=from_assay)
    facets = _multi_layout_facets(
        store,
        facet_by=facet_by,
        facet_order=facet_order,
        groups=groups,
        subset_by=subset_by,
        cell_key=cell_key,
    )
    child_panel_keys = _embedding_panel_keys(labels, facets)
    panel_keys = [
        _layout_panel_key(layout, panel_key)
        for layout in layout_keys
        for panel_key in child_panel_keys
    ]
    resolved_columns = n_columns if n_columns is not None else len(child_panel_keys)
    resolved_columns = max(1, min(resolved_columns, len(panel_keys)))
    if figsize is None and target is None:
        nrows = int(np.ceil(len(panel_keys) / resolved_columns))
        figsize = (
            DEFAULT_PANEL_INCHES * resolved_columns + 0.4,
            DEFAULT_PANEL_INCHES * nrows + 0.2,
        )

    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target,
            panel_keys=panel_keys,
            figsize=figsize,
            n_columns=resolved_columns,
        )

    children: list[tuple[str, PlotResult]] = []
    for layout in layout_keys:
        child_target = {
            panel_key: axes[_layout_panel_key(layout, panel_key)]
            for panel_key in child_panel_keys
        }
        child = embedding(
            store,
            layout_key=layout,
            color_by=color_by,
            facet_by=facet_by,
            facet_order=facets if facet_by is not None else facet_order,
            cell_key=cell_key,
            from_assay=from_assay,
            normalization=normalization,
            groups=groups,
            subset_by=subset_by,
            n_columns=len(child_panel_keys),
            target=child_target,
            figsize=None,
            theme=theme,
            show_titles=show_titles,
            show=False,
            **child_kwargs,
        )
        if show_titles:
            # Each row repeats the color panels, so name its layout.
            for ax in child_target.values():
                title = ax.get_title()
                ax.set_title(f"{layout} | {title}" if title else layout)
        children.append((layout, child))

    tables: dict[str, pd.DataFrame] = {}
    legends: list[LegendSpec] = []
    scales: list[Any] = []
    for layout, child in children:
        tables.update({f"{layout}:{key}": value for key, value in child.tables.items()})
        for legend in child.legends:
            if legend not in legends:
                legends.append(legend)
        for scale in child.scales:
            if not any(
                type(existing) is type(scale) and existing == scale
                for existing in scales
            ):
                scales.append(scale)

    provenances = {layout: child.provenance for layout, child in children}
    first_provenance = children[0][1].provenance
    assays = sorted(
        {
            assay
            for provenance in provenances.values()
            for assay in provenance.extras.get("assays", [])
        }
    )
    assay_values = {
        provenance.assay
        for provenance in provenances.values()
        if provenance.assay is not None
    }
    extras = dict(first_provenance.extras)
    extras.update(
        {
            "layouts": list(layout_keys),
            "n_layouts": len(layout_keys),
            "panel_keys": [str(key) for key in panel_keys],
            "layout_provenance": provenances,
            "color_limits_by_layout": {
                layout: provenance.extras.get("color_limits")
                for layout, provenance in provenances.items()
            },
            "invalid_coordinate_cells": {
                layout: provenance.extras.get("invalid_coordinate_cells", 0)
                for layout, provenance in provenances.items()
            },
            "assays": assays,
        }
    )
    notes = tuple(
        dict.fromkeys(
            (
                "embedding",
                "materialized",
                "multi_layout",
                *(
                    note
                    for provenance in provenances.values()
                    for note in provenance.notes
                    if note not in ("embedding", "materialized")
                ),
            )
        )
    )
    result = PlotResult(
        figure=figure,
        axes=axes,
        tables=tables,
        legends=tuple(legends),
        scales=tuple(scales),
        provenance=PlotProvenance(
            scarf_version=first_provenance.scarf_version,
            assay=next(iter(assay_values)) if len(assay_values) == 1 else None,
            cell_key=cell_key,
            n_cells=max(provenance.n_cells for provenance in provenances.values()),
            renderer=first_provenance.renderer,
            notes=notes,
            extras=extras,
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        result.show()
    return result


def _selected_metadata_column(
    store: Any,
    column: str,
    *,
    cell_key: str,
    cell_indices: np.ndarray | None,
    kind: CellFieldKind | None = None,
) -> np.ndarray:
    """Read one metadata column for the plotted cells with masked rows missing.

    A color column that ``kind`` classifies as categorical from its stored
    values keeps its labels and shows masked rows as None. Other columns, and
    columns read without ``kind``, follow the raster representation.
    """
    values, missing = _cell_column_values(
        store,
        column,
        cell_key=cell_key,
        cell_idx=cell_indices,
    )
    if missing is None or not missing.any():
        return values
    labels = kind is not None and _is_categorical(pd.Series(values), kind)
    return apply_missing_mask(values, missing, labels=labels)


def _prefetch_colors(
    store: Any,
    color_items: Sequence[str | ArtifactRef | FeatureRef | CellField | None],
    *,
    metadata_columns: Collection[str],
    from_assay: str | None,
    cell_key: str,
    n_cells: int,
    normalization: NormalizationSpec,
    cell_indices: np.ndarray | None = None,
    unmeasured: dict[str, np.ndarray] | None = None,
) -> list[tuple[np.ndarray, str, bool, bool]]:
    """Return list of (values, label, is_categorical, is_uniform).

    ``metadata_columns`` lists the cell-metadata columns that a plain string
    item may name; any other string item is resolved as a feature.
    ``unmeasured`` receives the cells that each feature assay did not
    measure, as the value layer records them.
    """
    out: list[tuple[np.ndarray, str, bool, bool]] = []

    # Batch RNA-like feature refs / gene strings for one matrix read.
    feature_slots: list[tuple[int, Any]] = []
    for i, item in enumerate(color_items):
        if item is None:
            out.append((np.ones(n_cells), "cells", False, True))
            continue
        if isinstance(item, ArtifactRef):
            if cell_indices is None:
                raise ValueError(
                    "ArtifactRef color_by requires an explicit layout ArtifactRef"
                )
            _, grouping_indices, grouping_values, _ = _resolve_grouping(
                store,
                group_by=None,
                groups=item,
                cell_key="I",
            )
            if not np.array_equal(grouping_indices, cell_indices):
                raise ValueError(
                    "color_by and layout artifacts select cells in a different order"
                )
            values = np.asarray(grouping_values[0])
            out.append(
                (
                    values,
                    item.kind,
                    # Label artifacts are categorical whatever their dtype.
                    cell_value_spec(item.kind).categorical
                    or _is_categorical(pd.Series(values), "auto"),
                    False,
                )
            )
            continue
        if isinstance(item, CellField):
            vals = _selected_metadata_column(
                store,
                item.key,
                cell_key=cell_key,
                cell_indices=cell_indices,
                kind=item.kind,
            )
            series = pd.Series(vals)
            out.append(
                (
                    np.asarray(vals),
                    item.label or item.key,
                    _is_categorical(series, item.kind),
                    False,
                )
            )
            continue
        if isinstance(item, str) and item in metadata_columns:
            vals = _selected_metadata_column(
                store,
                item,
                cell_key=cell_key,
                cell_indices=cell_indices,
                kind="auto",
            )
            series = pd.Series(vals)
            out.append((np.asarray(vals), item, _is_categorical(series, "auto"), False))
            continue
        # Feature path: placeholder, fill after batch resolve
        out.append((np.zeros(n_cells), "", False, False))
        feature_slots.append((i, item))

    if feature_slots:
        resolved = [
            resolve_feature(store, item, from_assay=from_assay)
            for _, item in feature_slots
        ]
        cell_idx = (
            store.cells.active_index(cell_key) if cell_indices is None else cell_indices
        )
        mat = fetch_normalized_feature_matrix(
            store,
            resolved,
            cell_idx,
            normalization=normalization,
            unmeasured=unmeasured,
        )
        for col_i, (slot_i, _) in enumerate(feature_slots):
            feat = resolved[col_i]
            out[slot_i] = (mat[:, col_i], feat.label, False, False)
    return out


def _resolve_highlight_mask(
    store: Any,
    highlight: Highlight | None,
    *,
    cell_key: str,
    n_cells: int,
    cell_indices: np.ndarray | None = None,
) -> np.ndarray | None:
    if highlight is None:
        return None
    if highlight.indices is not None:
        indices = np.asarray(highlight.indices, dtype=np.int64)
        if len(indices) and int(indices.max()) >= n_cells:
            raise IndexError("highlight index is outside the selected cell range")
        mask = np.zeros(n_cells, dtype=bool)
        mask[indices] = True
        return mask
    assert highlight.by is not None
    values = _selected_metadata_column(
        store,
        highlight.by,
        cell_key=cell_key,
        cell_indices=cell_indices,
    )
    if len(values) != n_cells:
        raise ValueError("highlight metadata length does not match selected cells")
    if highlight.groups is None:
        if values.dtype != bool:
            raise TypeError(
                "Highlight without groups requires a boolean metadata column"
            )
        return values.astype(bool, copy=False)
    return np.isin(values, np.asarray(highlight.groups, dtype=object))


def _density_selection_mask(
    store: Any,
    overlay: DensityOverlay | None,
    *,
    cell_key: str,
    n_cells: int,
    cell_indices: np.ndarray | None = None,
) -> np.ndarray | None:
    if overlay is None or overlay.group_by is None:
        return None
    values = _selected_metadata_column(
        store,
        overlay.group_by,
        cell_key=cell_key,
        cell_indices=cell_indices,
    )
    if len(values) != n_cells:
        raise ValueError("density metadata length does not match selected cells")
    if overlay.groups is None:
        return np.asarray(pd.notna(values), dtype=bool)
    return np.isin(values, np.asarray(overlay.groups, dtype=object))


def _weighted_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: np.ndarray,
) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    if cumulative[-1] <= 0:
        return np.asarray(
            np.quantile(sorted_values, quantiles),
            dtype=np.float64,
        )
    return np.asarray(
        np.interp(quantiles * cumulative[-1], cumulative, sorted_values),
        dtype=np.float64,
    )


def _smoothed_local_mean(
    xx: np.ndarray,
    yy: np.ndarray,
    values: np.ndarray,
    *,
    grid_pixels: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    sigma: float,
    min_support: float,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.ndimage import gaussian_filter

    numeric_values = np.asarray(values, dtype=np.float64)
    if len(numeric_values) != len(xx):
        raise ValueError("contour values must match the selected coordinates")
    finite = np.isfinite(xx) & np.isfinite(yy) & np.isfinite(numeric_values)
    sums_xy, _, _ = np.histogram2d(
        xx[finite],
        yy[finite],
        bins=grid_pixels,
        range=(x_range, y_range),
        weights=numeric_values[finite],
    )
    counts_xy, _, _ = np.histogram2d(
        xx[finite],
        yy[finite],
        bins=grid_pixels,
        range=(x_range, y_range),
    )
    support = gaussian_filter(
        counts_xy.T,
        sigma=sigma,
        mode="constant",
    )
    smoothed_sum = gaussian_filter(
        sums_xy.T,
        sigma=sigma,
        mode="constant",
    )
    local_mean = np.zeros_like(smoothed_sum)
    np.divide(
        smoothed_sum,
        support,
        out=local_mean,
        where=(support > 0) & np.isfinite(smoothed_sum),
    )
    support_taper = np.clip(support / min_support, 0.0, 1.0)
    return local_mean * support_taper**2, support


def _retain_strongest_hotspots(
    surface: np.ndarray,
    support: np.ndarray,
    *,
    level: float,
    max_hotspots: int | None,
) -> np.ndarray:
    if max_hotspots is None:
        return surface
    from scipy.ndimage import binary_fill_holes
    from scipy.ndimage import label as label_components

    above_level = np.isfinite(surface) & (surface >= level)
    component_ids, n_components = label_components(
        above_level,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    if n_components <= max_hotspots:
        retained = set(range(1, n_components + 1))
    else:
        component_scores: list[tuple[float, int]] = []
        for component_id in range(1, n_components + 1):
            component_mask = component_ids == component_id
            excess = np.maximum(surface[component_mask] - level, 0)
            score = float(
                np.sum(support[component_mask] * (excess + np.finfo(np.float64).eps))
            )
            component_scores.append((score, component_id))
        component_scores.sort(reverse=True)
        retained = {component_id for _, component_id in component_scores[:max_hotspots]}
    remove = above_level & ~np.isin(component_ids, list(retained))
    filtered = surface.copy()
    filtered[remove] = np.nextafter(level, -np.inf)
    filled_retained = np.zeros_like(above_level)
    for component_id in retained:
        filled_retained |= binary_fill_holes(component_ids == component_id)
    holes = filled_retained & ~above_level
    filtered[holes] = np.nextafter(level, np.inf)
    return filtered


def _draw_density_overlay(
    ax: Any,
    xx: np.ndarray,
    yy: np.ndarray,
    *,
    overlay: DensityOverlay,
    values: np.ndarray | None,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    theme: str,
) -> None:
    if len(xx) < 3:
        return
    from scipy.ndimage import gaussian_filter

    padding_pixels = max(2, int(np.ceil(4 * overlay.sigma)))
    grid_pixels = overlay.pixels + 2 * padding_pixels
    x_step = (xlim[1] - xlim[0]) / overlay.pixels
    y_step = (ylim[1] - ylim[0]) / overlay.pixels
    padded_xlim = (
        xlim[0] - padding_pixels * x_step,
        xlim[1] + padding_pixels * x_step,
    )
    padded_ylim = (
        ylim[0] - padding_pixels * y_step,
        ylim[1] + padding_pixels * y_step,
    )
    level_weights: np.ndarray | None = None
    if overlay.statistic == "mean":
        if values is None:
            raise ValueError(
                "DensityOverlay(statistic='mean') requires a continuous color panel"
            )
        surface, support = _smoothed_local_mean(
            xx,
            yy,
            values,
            grid_pixels=grid_pixels,
            x_range=padded_xlim,
            y_range=padded_ylim,
            sigma=overlay.sigma,
            min_support=overlay.min_support,
        )
        level_mask = (support >= overlay.min_support) & np.isfinite(surface)
        if level_mask.any() and np.all(surface[level_mask] >= 0):
            # Non-negative means contour only where the mean is above zero.
            level_mask &= surface > max(
                float(np.max(surface[level_mask])) * 1e-6,
                1e-12,
            )
        finite_surface = surface[level_mask]
        level_weights = support[level_mask]
    else:
        from ._raster import density_canvas_from_points

        canvas = density_canvas_from_points(
            xx,
            yy,
            extent=(padded_xlim[0], padded_xlim[1], padded_ylim[0], padded_ylim[1]),
            pixels=grid_pixels,
        )
        support = gaussian_filter(
            np.flipud(canvas.counts).astype(np.float64),
            sigma=overlay.sigma,
            mode="constant",
        )
        surface = support
        finite_surface = surface[np.isfinite(surface) & (surface > 0)]
    if len(finite_surface) == 0:
        return
    if isinstance(overlay.levels, int):
        quantiles = np.linspace(0.55, 0.97, overlay.levels)
        levels = np.unique(
            _weighted_quantiles(finite_surface, level_weights, quantiles)
            if level_weights is not None
            else np.quantile(finite_surface, quantiles)
        )
    else:
        requested = np.asarray(overlay.levels, dtype=np.float64)
        if np.all((0 < requested) & (requested < 1)):
            levels = np.unique(
                _weighted_quantiles(finite_surface, level_weights, requested)
                if level_weights is not None
                else np.quantile(finite_surface, requested)
            )
        else:
            levels = np.unique(requested)
    surface_min = float(np.nanmin(surface))
    surface_max = float(np.nanmax(surface))
    levels = levels[(levels > surface_min) & (levels < surface_max)]
    if len(levels) == 0:
        return
    surface = _retain_strongest_hotspots(
        surface,
        support,
        level=float(levels[0]),
        max_hotspots=overlay.max_hotspots,
    )
    surface_max = float(np.nanmax(surface))
    xcentres = np.linspace(
        padded_xlim[0],
        padded_xlim[1],
        grid_pixels,
        endpoint=False,
    )
    ycentres = np.linspace(
        padded_ylim[0],
        padded_ylim[1],
        grid_pixels,
        endpoint=False,
    )
    xcentres += (padded_xlim[1] - padded_xlim[0]) / (2 * grid_pixels)
    ycentres += (padded_ylim[1] - padded_ylim[0]) / (2 * grid_pixels)
    color = overlay.color or ("#f0f0f0" if theme == "dark" else "#202020")
    contour_surface = np.ma.masked_invalid(surface)
    if overlay.kind == "filled":
        upper = np.nextafter(surface_max, np.inf)
        fill_levels = np.unique(np.append(levels, upper))
        if len(fill_levels) >= 2:
            ax.contourf(
                xcentres,
                ycentres,
                contour_surface,
                levels=fill_levels,
                cmap=overlay.cmap,
                colors=None if overlay.cmap else [color],
                alpha=overlay.alpha,
                antialiased=True,
                zorder=overlay.zorder,
            )
    else:
        contours = ax.contour(
            xcentres,
            ycentres,
            contour_surface,
            levels=levels,
            cmap=overlay.cmap,
            colors=None if overlay.cmap else color,
            alpha=overlay.alpha,
            linewidths=overlay.linewidth,
            zorder=overlay.zorder,
        )
        if overlay.halo_width > 0:
            from matplotlib import patheffects

            halo_color = overlay.halo_color or (
                "#202020" if theme == "dark" else "#ffffff"
            )
            contours.set_path_effects(
                [
                    patheffects.withStroke(
                        linewidth=overlay.linewidth + 2 * overlay.halo_width,
                        foreground=halo_color,
                    ),
                    patheffects.Normal(),
                ]
            )


def _draw_highlight(
    ax: Any,
    xx: np.ndarray,
    yy: np.ndarray,
    sizes: np.ndarray,
    *,
    highlight: Highlight,
    edgecolor: str,
    rasterized: bool,
) -> Any:
    if len(xx) == 0:
        return
    halo = highlight.halo_color or edgecolor
    return ax.scatter(
        xx,
        yy,
        s=sizes * highlight.size_multiplier,
        c=highlight.color,
        alpha=highlight.alpha,
        edgecolors=halo,
        linewidths=highlight.halo_width,
        rasterized=rasterized,
        zorder=5,
    )


def _category_codes(values: np.ndarray) -> tuple[np.ndarray, dict[Any, int]]:
    """Integer code per value (missing is -1) and the code of each category."""
    codes, uniques = pd.factorize(pd.Series(np.asarray(values, dtype=object)))
    return codes, {value: code for code, value in enumerate(uniques)}


def _draw_categorical(
    ax: Any,
    xx: np.ndarray,
    yy: np.ndarray,
    vv: np.ndarray,
    ss: np.ndarray,
    *,
    scale: CategoricalScale,
    edgecolor: str,
    edgewidth: float,
    alpha: float,
    rasterized: bool,
    mpl: Any,
) -> Any:
    codes, categories = _category_codes(vv)
    palette = scale.palette or {}
    # The last row colors missing values and categories outside the scale.
    colors = np.empty((len(categories) + 1, 4), dtype=np.float64)
    for value, code in categories.items():
        colors[code] = mpl.colors.to_rgba(palette.get(value, scale.missing_color))
    colors[-1] = mpl.colors.to_rgba(scale.missing_color)
    edges, lw = scatter_edges(edgecolor, edgewidth)
    return ax.scatter(
        xx,
        yy,
        c=colors[codes],
        s=ss,
        linewidths=lw,
        edgecolors=edges,
        alpha=alpha,
        rasterized=rasterized,
    )


def _categorical_legend_block(
    mpl: Any,
    *,
    scale: CategoricalScale,
    label: str,
    missing: bool,
    edgecolor: str,
    values: np.ndarray,
    max_entries: int = LEGEND_SIDE_MAX_ENTRIES,
) -> tuple[str | None, list[Any], list[Any]]:
    """Return the legend title, handles, and categories left out of the legend.

    Beyond ``max_entries`` categories, the most frequent ones are listed.
    """
    order = list(scale.order or ())
    palette = scale.palette or {}
    if len(order) > max_entries:
        counts = pd.Series(np.asarray(values, dtype=object)).value_counts(dropna=True)
        ranked = sorted(
            enumerate(order),
            key=lambda item: (-int(counts.get(item[1], 0)), item[0]),
        )
        selected = {index for index, _ in ranked[:max_entries]}
        shown = [value for index, value in enumerate(order) if index in selected]
        omitted = [value for index, value in enumerate(order) if index not in selected]
    else:
        shown = order
        omitted = []
    entries = [(palette[value], category_label(scale, value)) for value in shown]
    if missing:
        entries.append((scale.missing_color, scale.missing_label))
    handles = [
        mpl.lines.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor=edgecolor,
            markeredgewidth=0.3,
            markersize=5,
            label=text,
        )
        for color, text in entries
    ]
    title = f"{label} ({len(shown)} of {len(order)})" if omitted else label or None
    return title, handles, omitted


def _side_legend_kwargs(title: str | None, handles: list[Any]) -> dict[str, Any]:
    return {
        "handles": handles,
        "title": title,
        "frameon": False,
        "borderaxespad": 0,
        "ncols": legend_side_columns(len(handles)),
        "columnspacing": 0.8,
    }


def _add_on_data_labels(
    ax: Any,
    xx: np.ndarray,
    yy: np.ndarray,
    vv: np.ndarray,
    *,
    scale: CategoricalScale,
    theme: str,
    max_labels: int,
) -> list[Any]:
    from matplotlib import patheffects

    text_color = "#f5f5f5" if theme == "dark" else "#222222"
    stroke = "#222222" if theme == "dark" else "#ffffff"
    codes, categories = _category_codes(vv)
    candidates: list[tuple[Any, float, float, int]] = []
    for value in scale.order or ():
        code = categories.get(value)
        if code is None:
            continue
        mask = codes == code
        candidates.append(
            (
                value,
                float(np.median(xx[mask])),
                float(np.median(yy[mask])),
                int(mask.sum()),
            )
        )
    candidates.sort(key=lambda item: (-item[3], str(item[0])))
    omitted = [value for value, _, _, _ in candidates[max_labels:]]
    chosen = candidates[:max_labels]
    if chosen:
        yrange = max(float(np.ptp(yy)), 1e-12)
        minimum_gap = 0.035 * yrange
        ordered = sorted(chosen, key=lambda item: item[2])
        adjusted: list[tuple[Any, float, float, int]] = []
        previous_y = -np.inf
        for value, xpos, ypos, count in ordered:
            adjusted_y = max(ypos, previous_y + minimum_gap)
            adjusted.append((value, xpos, adjusted_y, count))
            previous_y = adjusted_y
        chosen = adjusted
    for value, xpos, ypos, _ in chosen:
        ax.text(
            xpos,
            ypos,
            category_label(scale, value),
            ha="center",
            va="center",
            fontsize=8,
            color=text_color,
            path_effects=[patheffects.withStroke(linewidth=2.0, foreground=stroke)],
            zorder=5,
        )
    return omitted


def _no_automatic_scale(values: np.ndarray, scale: ColorScale) -> bool:
    """Whether a continuous color has neither a value nor an explicit limit.

    Such a color, such as a feature whose assay measured none of the drawn
    cells, gets no colorbar; its points take the scale's missing color.
    """
    return scale.vmin is None and scale.vmax is None and not has_finite_values(values)


def _placeholder_limits(scale: ColorScale) -> tuple[float, float]:
    """Limits that build a valid norm for a color without values."""
    if scale.scale == "log":
        return (1.0, 10.0)
    return resolve_color_limits(np.empty(0), scale)


def _draw_continuous(
    ax: Any,
    fig: Any,
    xx: np.ndarray,
    yy: np.ndarray,
    vnum: np.ndarray,
    ss: np.ndarray,
    *,
    limits: tuple[float, float],
    cmap_name: str | None,
    vcenter: float | None,
    scale: str,
    missing_color: str,
    default_color: str,
    edgecolor: str,
    edgewidth: float,
    alpha: float,
    label: str,
    is_uniform: bool,
    sort_values: bool,
    rng: np.random.Generator | None,
    add_colorbar: bool,
    rasterized: bool,
    plt: Any,
    mpl: Any,
) -> Any:
    finite = np.isfinite(vnum)
    order_idx = np.arange(len(vnum))
    if sort_values and finite.any():
        order_idx = np.argsort(np.where(finite, vnum, -np.inf))
    elif rng is not None:
        order_idx = rng.permutation(len(vnum))
    xx = xx[order_idx]
    yy = yy[order_idx]
    vnum = vnum[order_idx]
    ss = ss[order_idx]
    finite = finite[order_idx]
    edges, lw = scatter_edges(edgecolor, edgewidth)

    if is_uniform or len(xx) == 0:
        return ax.scatter(
            xx,
            yy,
            c=[default_color] * len(xx),
            s=ss if len(ss) else 10,
            linewidths=lw,
            edgecolors=edges,
            alpha=alpha,
            rasterized=rasterized,
        )

    norm = continuous_norm(
        mpl,
        vmin=limits[0],
        vmax=limits[1],
        vcenter=vcenter,
        scale=scale,
    )
    cmap = plt.get_cmap(cmap_name or "viridis")
    face = np.empty((len(vnum), 4))
    face[:] = mpl.colors.to_rgba(missing_color)
    if finite.any():
        face[finite] = cmap(norm(vnum[finite]))
    collection = ax.scatter(
        xx,
        yy,
        c=face,
        s=ss,
        linewidths=lw,
        edgecolors=edges,
        alpha=alpha,
        rasterized=rasterized,
    )
    if add_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        # Top colorbars keep the embedding panel square and avoid colliding
        # with neighboring y-axis labels in multi-panel figures.
        cb = fig.colorbar(
            sm,
            ax=ax,
            location="top",
            orientation="horizontal",
            shrink=0.8,
            fraction=0.06,
            pad=0.02,
        )
        cb.set_label(label)
    return collection


def _soft_clip(values: np.ndarray, clip_fraction: float) -> np.ndarray:
    """Clip finite values to their quantiles; ``embedding`` validated the fraction."""
    if clip_fraction <= 0:
        return values
    v = np.asarray(values, dtype=np.float64).copy()
    finite = np.isfinite(v)
    if not finite.any():
        return v
    lo = float(np.percentile(v[finite], 100 * clip_fraction))
    hi = float(np.percentile(v[finite], 100 - 100 * clip_fraction))
    v[finite & (v < lo)] = lo
    v[finite & (v > hi)] = hi
    return v


@close_figures_on_error
def embedding(
    store: Any,
    *,
    layout_key: str | Sequence[str] | None = None,
    layout: ArtifactRef | None = None,
    color_by: str
    | ArtifactRef
    | FeatureRef
    | CellField
    | Sequence[str | ArtifactRef | FeatureRef | CellField]
    | None = None,
    facet_by: str | None = None,
    facet_order: Sequence[Any] | None = None,
    cell_key: str = "I",
    from_assay: str | None = None,
    normalization: NormalizationSpec | None = None,
    point_size: float | None = None,
    point_sizes: np.ndarray | Sequence[float] | None = None,
    point_size_range: tuple[float, float] = (1.0, 28.0),
    point_edgecolor: str | None = None,
    point_edgewidth: float | None = None,
    point_alpha: float = 1.0,
    sort_values: bool = False,
    color_scale: ColorScale | None = None,
    categorical_scale: CategoricalScale | None = None,
    default_color: str = "steelblue",
    missing_color: str | None = None,
    clip_fraction: float = 0.0,
    subset_by: str | None = None,
    groups: Sequence[Any] | None = None,
    n_columns: int | None = None,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    legend_loc: LegendLoc = "auto",
    max_on_data_labels: int = 40,
    show_legend: bool = True,
    show_titles: bool = True,
    frame: FrameStyle = "minimal",
    density_overlay: DensityOverlay | None = None,
    highlight: Highlight | None = None,
    seed: int | None = None,
    rasterize_threshold: int = DEFAULT_RASTERIZE_THRESHOLD,
    show: bool = True,
) -> PlotResult:
    """Scatter cells on one or more 2D layouts (UMAP, t-SNE, and similar).

    ``color_by`` accepts a cell-metadata column, a gene name, or a list of
    either. With several genes, Scarf draws one panel per gene. With
    ``facet_by``, each gene becomes a row and each facet level a column.
    Color limits for a gene are shared across facets so panels stay comparable.
    Multiple ``layout_key`` values produce the cartesian product of layouts and
    colors in one figure.

    Common choices:

    - ``subset_by``: keep cells marked ``True`` in a boolean metadata column.
    - ``groups``: keep only these categories. Applies to ``facet_by`` when set,
      otherwise to the first categorical ``color_by`` column. Also sets legend
      order when ``categorical_scale.order`` is omitted.
    - ``legend_loc``: ``"auto"`` puts a side legend when there are few
      categories, labels on the clusters for medium category counts, and a
      wrapped side legend for larger sets. Use ``"right"``, ``"on_data"``, or
      ``"none"`` to force a placement.
    - ``frame``: ``"minimal"`` (default) keeps an L-shaped axes edge without
      UMAP tick labels. ``"none"`` removes the box. ``"axes"`` keeps UMAP1 /
      UMAP2 labels on the outer panels.
    - ``point_size``: leave as ``None`` to size markers from the cell count.
      Pass a number (matplotlib marker area) or ``point_sizes`` for per-cell
      sizes such as mapping confidence.
    - ``sort_values=True``: for continuous colors, draw high values last so
      expressing cells sit on top of the cloud.
    - ``theme``: ``"notebook"`` or ``"paper"`` for white figures; ``"dark"``
      for dark notebook themes.

    Returns a :class:`PlotResult`. Access ``.figure`` in notebooks, or call
    ``.save(...)`` / ``.close()`` when you own the figure.
    """
    clip_fraction = clip_fraction_argument(clip_fraction)
    if (layout_key is None) == (layout is None):
        raise ValueError("Provide exactly one of layout_key or layout")
    if layout is None:
        assert layout_key is not None
        layout_keys = _coerce_layout_items(layout_key)
    else:
        if cell_key != "I":
            raise ValueError(
                "cell_key cannot override an artifact's stored cell selection"
            )
        layout_keys = []
    if len(layout_keys) > 1:
        return _embedding_multiple_layouts(
            store,
            layout_keys=layout_keys,
            color_by=color_by,
            facet_by=facet_by,
            facet_order=facet_order,
            cell_key=cell_key,
            from_assay=from_assay,
            normalization=normalization,
            groups=groups,
            subset_by=subset_by,
            n_columns=n_columns,
            target=target,
            figsize=figsize,
            theme=theme,
            show_titles=show_titles,
            show=show,
            child_kwargs={
                "point_size": point_size,
                "point_sizes": point_sizes,
                "point_size_range": point_size_range,
                "point_edgecolor": point_edgecolor,
                "point_edgewidth": point_edgewidth,
                "point_alpha": point_alpha,
                "sort_values": sort_values,
                "color_scale": color_scale,
                "categorical_scale": categorical_scale,
                "default_color": default_color,
                "missing_color": missing_color,
                "clip_fraction": clip_fraction,
                "legend_loc": legend_loc,
                "max_on_data_labels": max_on_data_labels,
                "show_legend": show_legend,
                "frame": frame,
                "density_overlay": density_overlay,
                "highlight": highlight,
                "seed": seed,
                "rasterize_threshold": rasterize_threshold,
            },
        )
    artifact_cell_indices: np.ndarray | None = None
    layout_selection: ArtifactRef | None = None
    if layout is None:
        layout_name = layout_keys[0]
        x = np.asarray(
            _fetch_cell_column(store, f"{layout_name}1", cell_key=cell_key),
            dtype=np.float64,
        )
        y = np.asarray(
            _fetch_cell_column(store, f"{layout_name}2", cell_key=cell_key),
            dtype=np.float64,
        )
    else:
        coordinates, artifact_cell_indices, layout_selection = _resolve_layout(
            store,
            layout,
        )
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        layout_name = "Embedding"
        for item in _coerce_color_items(color_by):
            if not isinstance(item, ArtifactRef):
                continue
            color_selection = _artifact_cell_selection(store, item)
            if color_selection != layout_selection:
                raise ValueError(
                    "color_by and layout artifacts must share the same cell selection"
                )
    resolved_from_assay = (
        from_assay
        if from_assay is not None
        else layout.assay
        if layout is not None
        else None
    )

    plt, mpl = require_matplotlib()
    if rasterize_threshold < 0:
        raise ValueError("rasterize_threshold must be >= 0")
    if point_size_range[0] <= 0 or point_size_range[1] < point_size_range[0]:
        raise ValueError("point_size_range must satisfy 0 < minimum <= maximum")
    if point_edgewidth is not None and point_edgewidth < 0:
        raise ValueError("point_edgewidth must be non-negative")
    if not 0 <= point_alpha <= 1:
        raise ValueError("point_alpha must be between 0 and 1")
    if max_on_data_labels < 1:
        raise ValueError("max_on_data_labels must be positive")
    normalization = normalization or NormalizationSpec()
    color_scale_was_explicit = color_scale is not None
    categorical_scale_was_explicit = categorical_scale is not None
    color_scale = color_scale or ColorScale(cmap="viridis", scope="feature")
    resolved_missing_color = missing_color or "#bdbdbd"

    n = len(x)
    if n == 0:
        raise ValueError(f"No cells selected by cell_key {cell_key!r}")
    finite_coordinates = np.isfinite(x) & np.isfinite(y)
    if not finite_coordinates.any():
        raise ValueError(f"Layout {layout_name!r} has no finite coordinates")
    if point_sizes is not None and len(point_sizes) != n:
        raise ValueError("point_sizes length must match number of selected cells")
    if point_size is not None and (not np.isfinite(point_size) or point_size <= 0):
        raise ValueError("point_size must be finite and positive")
    auto_point_size = point_size is None and point_sizes is None
    resolved_point_size = (
        float(point_size)
        if point_size is not None
        else default_point_size(
            n,
            size_min=point_size_range[0],
            size_max=point_size_range[1],
        )
    )
    size_arr = (
        np.asarray(point_sizes, dtype=np.float64)
        if point_sizes is not None
        else np.full(n, resolved_point_size, dtype=np.float64)
    )
    if not np.all(np.isfinite(size_arr)) or np.any(size_arr <= 0):
        raise ValueError("point_sizes must contain finite positive values")

    color_items = _coerce_color_items(color_by)
    stored_displays: list[dict[str, Any] | None] = []
    for item in color_items:
        column = (
            item
            if isinstance(item, str)
            else item.key
            if isinstance(item, CellField)
            else None
        )
        stored_displays.append(
            None if column is None else stored_display_metadata(store, column)
        )
    metadata_columns = _cell_metadata_columns(store, color_items)
    # The cells of the layout that each feature assay did not measure.
    unmeasured: dict[str, np.ndarray] = {}
    color_cache = _prefetch_colors(
        store,
        color_items,
        metadata_columns=metadata_columns,
        from_assay=resolved_from_assay,
        cell_key=cell_key,
        n_cells=n,
        normalization=normalization,
        cell_indices=artifact_cell_indices,
        unmeasured=unmeasured,
    )
    classified_cache: list[tuple[np.ndarray, str, bool, bool]] = []
    for index, entry in enumerate(color_cache):
        values, label, is_categorical, is_uniform = entry
        item = color_items[index]
        display = stored_displays[index]
        cell_kind = item.kind if isinstance(item, CellField) else "auto"
        if display is not None and cell_kind == "auto":
            is_categorical = display["kind"] == "categorical"
        classified_cache.append((values, label, is_categorical, is_uniform))
    color_cache = classified_cache
    if density_overlay is not None and density_overlay.statistic == "mean":
        if any(is_cat or is_uniform for _, _, is_cat, is_uniform in color_cache):
            raise ValueError("Mean contours require a continuous color_by panel")
    stored_color_scales: dict[int, ColorScale] = {}
    stored_categorical_scales: dict[int, CategoricalScale] = {}
    for index, display in enumerate(stored_displays):
        if display is None:
            continue
        if display["kind"] == "continuous" and not color_scale_was_explicit:
            stored_scale = continuous_display_scale(display)
            # Clipping redefines the value range, so stored limits no longer apply.
            stored_color_scales[index] = (
                replace(stored_scale, vmin=None, vmax=None)
                if clip_fraction > 0
                else stored_scale
            )
        elif display["kind"] == "categorical" and not categorical_scale_was_explicit:
            stored_categorical = categorical_display_scale(display)
            stored_categorical_scales[index] = (
                stored_categorical
                if missing_color is None
                else replace(stored_categorical, missing_color=missing_color)
            )
    if clip_fraction > 0:
        clipped: list[tuple[np.ndarray, str, bool, bool]] = []
        for vals, label, is_cat, is_uniform in color_cache:
            if is_cat or is_uniform:
                clipped.append((vals, label, is_cat, is_uniform))
            else:
                clipped.append(
                    (_soft_clip(vals, clip_fraction), label, is_cat, is_uniform)
                )
        color_cache = clipped

    subset_vals = (
        _selected_metadata_column(
            store,
            subset_by,
            cell_key=cell_key,
            cell_indices=artifact_cell_indices,
        )
        if subset_by is not None
        else None
    )
    highlight_mask = _resolve_highlight_mask(
        store,
        highlight,
        cell_key=cell_key,
        n_cells=n,
        cell_indices=artifact_cell_indices,
    )
    density_filter = _density_selection_mask(
        store,
        density_overlay,
        cell_key=cell_key,
        n_cells=n,
        cell_indices=artifact_cell_indices,
    )
    facet_values: np.ndarray | None = None
    groups_category: np.ndarray | None = None
    groups_color_index: int | None = None
    if facet_by is not None:
        facet_values = _selected_metadata_column(
            store,
            facet_by,
            cell_key=cell_key,
            cell_indices=artifact_cell_indices,
            kind="categorical",
        )
        if groups is not None:
            groups_category = facet_values
    elif groups is not None:
        cat_columns = [
            (index, vals)
            for index, (vals, _, is_cat, is_uniform) in enumerate(color_cache)
            if is_cat and not is_uniform
        ]
        if not cat_columns:
            raise ValueError(
                "groups requires a categorical color_by column, or facet_by"
            )
        groups_color_index, group_values = cat_columns[0]
        groups_category = np.asarray(group_values)

    selection_mask, group_order = resolve_cell_selection(
        n,
        subset=subset_vals,
        subset_name=subset_by,
        category_values=groups_category,
        groups=groups,
    )
    base_mask = finite_coordinates & selection_mask

    if facet_by is not None:
        assert facet_values is not None
        if groups is not None and group_order is not None:
            facets = list(group_order)
        elif facet_order is not None:
            facets = list(facet_order)
        else:
            facets = sort_categories(list(pd.unique(facet_values[base_mask])))
    else:
        facets = [None]

    labels = [label for _, label, _, _ in color_cache]
    panel_keys = _embedding_panel_keys(labels, facets)

    n_colors = len(color_cache)
    n_facets = len(facets)
    if n_columns is None:
        n_columns = n_facets if facet_by is not None else min(n_colors, 4)
    n_columns = max(1, min(n_columns, len(panel_keys)))

    xlim, ylim = padded_square_limits(x[base_mask], y[base_mask])
    edgecolor = point_edgecolor or scatter_edgecolor(theme)

    label_counts = pd.Series(labels).value_counts()

    def display_key(index: int, label: str) -> str:
        return label if label_counts[label] == 1 else f"{index}:{label}"

    limit_map: dict[int, tuple[float, float]] = {}
    # Continuous colors without a value or an explicit limit: no colorbar.
    unscaled: set[int] = set()
    categorical_maps: dict[int, CategoricalScale] = {}
    resolved_color_scales: dict[int, ColorScale] = {}
    shared_limits: tuple[float, float] | None = None
    if color_scale.scope == "shared":
        shared_values = [
            np.asarray(values, dtype=np.float64)[base_mask]
            for values, _, is_categorical, is_uniform in color_cache
            if not is_categorical and not is_uniform
        ]
        if shared_values and not _no_automatic_scale(
            np.concatenate(shared_values), color_scale
        ):
            shared_limits = resolve_color_limits(
                np.concatenate(shared_values),
                color_scale,
            )

    for color_index, (vals, _, is_cat, is_uniform) in enumerate(color_cache):
        if is_uniform:
            continue
        if is_cat:
            source = (
                categorical_scale
                if categorical_scale_was_explicit
                else stored_categorical_scales.get(color_index)
            ) or CategoricalScale(missing_color=resolved_missing_color)
            if (
                groups is not None
                and group_order is not None
                and facet_by is None
                and color_index == groups_color_index
                and (source.order is None or not categorical_scale_was_explicit)
            ):
                # Requested groups order the legend unless a caller's scale does.
                source = replace(source, order=tuple(group_order))
            # Categories come from every selected cell, so layouts that cannot
            # place some cells still share one order and palette.
            categorical_maps[color_index] = resolve_category_scale(
                np.asarray(vals)[selection_mask],
                source,
            )
            continue
        active_color_scale = stored_color_scales.get(color_index, color_scale)
        resolved_color_scales[color_index] = active_color_scale
        if active_color_scale.scope != "panel":
            if shared_limits is not None and color_scale_was_explicit:
                limit_map[color_index] = shared_limits
            elif _no_automatic_scale(np.asarray(vals)[base_mask], active_color_scale):
                limit_map[color_index] = _placeholder_limits(active_color_scale)
                unscaled.add(color_index)
            else:
                limit_map[color_index] = resolve_color_limits(
                    np.asarray(vals)[base_mask],
                    active_color_scale,
                )

    legend_locs: dict[int, LegendLoc] = {
        color_index: (
            resolve_legend_loc(len(scale.order or ()), legend_loc)
            if show_legend
            else "none"
        )
        for color_index, scale in categorical_maps.items()
    }
    needs_side_legend = any(loc == "right" for loc in legend_locs.values())
    if figsize is None and target is None:
        nrows = int(np.ceil(len(panel_keys) / n_columns))
        needs_top_cbar = any(
            (not is_cat) and (not is_uniform)
            for _, _, is_cat, is_uniform in color_cache
        )
        panel = DEFAULT_PANEL_INCHES
        side_columns = max(
            (
                legend_side_columns(len(scale.order or ()))
                for color_index, scale in categorical_maps.items()
                if legend_locs[color_index] == "right"
            ),
            default=0,
        )
        width = panel * n_columns + (
            0.95 * side_columns + 0.4 if needs_side_legend else 0.25
        )
        height = panel * nrows + (0.55 if needs_top_cbar else 0.2)
        figsize = (width, height)

    legends: list[LegendSpec] = []
    categorical_scales: list[Any] = []
    for color_index, scale in categorical_maps.items():
        categorical_scales.append(scale)
        legends.append(
            LegendSpec(
                kind="categorical",
                label=labels[color_index],
                extras={"legend_loc": legend_locs[color_index]},
            )
        )
    panel_limit_specs: dict[int, dict[str, list[float]]] = {}
    colorbar_colors: list[int] = []
    for color_index, (_, label, is_categorical, is_uniform) in enumerate(color_cache):
        if is_categorical or is_uniform or color_index in unscaled:
            continue
        colorbar_colors.append(color_index)
        limits = limit_map.get(color_index)
        if limits is None:
            panel_limit_specs[color_index] = {}
            extras: dict[str, Any] = {
                "scope": "panel",
                "panel_limits": panel_limit_specs[color_index],
            }
        else:
            extras = {"vmin": limits[0], "vmax": limits[1]}
        legends.append(LegendSpec(kind="colorbar", label=label, extras=extras))

    rng = np.random.default_rng(seed) if seed is not None else None
    # None for a panel without a finite value to scale.
    panel_limit_map: dict[str, tuple[float, float] | None] = {}
    panel_point_sizes: dict[str, float] = {}
    panel_edgewidths: dict[str, float] = {}
    omitted_labels: dict[str, list[Any]] = {}
    omitted_legend_entries: dict[str, list[str]] = {}
    sized_artists: list[tuple[str, Any]] = []
    legend_blocks: list[tuple[str | None, list[Any]]] = []

    with theme_context(theme):
        fig, axes, owns = normalize_axes_target(
            target, panel_keys=panel_keys, figsize=figsize, n_columns=n_columns
        )
        panel_i = 0
        for color_index, (vals, label, is_cat, is_uniform) in enumerate(color_cache):
            for fac_i, fac in enumerate(facets):
                panel_key = panel_keys[panel_i]
                ax = axes[panel_key]
                if facet_values is None:
                    mask = base_mask.copy()
                elif pd.isna(fac):
                    mask = base_mask & pd.isna(facet_values)
                else:
                    mask = base_mask & (facet_values == fac)

                xx = x[mask]
                yy = y[mask]
                vv = np.asarray(vals)[mask]
                if auto_point_size:
                    panel_size = default_point_size(
                        int(mask.sum()),
                        panel_area=panel_area_inches(ax),
                        size_min=point_size_range[0],
                        size_max=point_size_range[1],
                    )
                    ss = np.full(int(mask.sum()), panel_size, dtype=np.float64)
                else:
                    ss = size_arr[mask]
                    panel_size = (
                        float(np.nanmedian(ss)) if len(ss) else resolved_point_size
                    )
                active_edgewidth = (
                    float(point_edgewidth)
                    if point_edgewidth is not None
                    else default_point_edgewidth(
                        int(mask.sum()),
                        point_size=panel_size,
                    )
                )
                panel_point_sizes[str(panel_key)] = panel_size
                panel_edgewidths[str(panel_key)] = active_edgewidth
                rasterized = len(xx) >= rasterize_threshold
                base_alpha = (
                    highlight.dim_alpha if highlight is not None else point_alpha
                )

                if mask.sum() == 0:
                    ax.set_axis_off()
                    ax.set_title(
                        f"{label} | {facet_by}={fac} (empty)"
                        if fac is not None
                        else f"{label} (empty)"
                    )
                    panel_i += 1
                    continue

                if is_cat:
                    category_scale = categorical_maps[color_index]
                    base_artist = _draw_categorical(
                        ax,
                        xx,
                        yy,
                        vv,
                        ss,
                        scale=category_scale,
                        edgecolor=edgecolor,
                        edgewidth=active_edgewidth,
                        alpha=base_alpha,
                        rasterized=rasterized,
                        mpl=mpl,
                    )
                    panel_legend = legend_locs[color_index]
                    if panel_legend == "on_data":
                        omitted = _add_on_data_labels(
                            ax,
                            xx,
                            yy,
                            vv,
                            scale=category_scale,
                            theme=theme,
                            max_labels=max_on_data_labels,
                        )
                        if omitted:
                            omitted_labels[str(panel_key)] = omitted
                    elif panel_legend == "right" and fac_i == n_facets - 1:
                        selected_values = np.asarray(vals)[selection_mask]
                        legend_title, handles, omitted = _categorical_legend_block(
                            mpl,
                            scale=category_scale,
                            label=label,
                            missing=bool(pd.isna(selected_values).any()),
                            edgecolor=edgecolor,
                            values=selected_values,
                        )
                        if owns:
                            # Figure legends are placed together after drawing.
                            legend_blocks.append((legend_title, handles))
                        else:
                            ax.legend(
                                **_side_legend_kwargs(legend_title, handles),
                                loc="upper left",
                                bbox_to_anchor=(1.02, 1.0),
                            )
                        if omitted:
                            omitted_legend_entries[str(panel_key)] = [
                                str(value) for value in omitted
                            ]
                else:
                    active_color_scale = resolved_color_scales.get(
                        color_index,
                        color_scale,
                    )
                    vnum = pd.to_numeric(pd.Series(vv), errors="coerce").to_numpy(
                        dtype=np.float64
                    )
                    panel_unscaled = False
                    if is_uniform:
                        limits = (0.0, 1.0)
                    elif active_color_scale.scope == "panel":
                        if _no_automatic_scale(vnum, active_color_scale):
                            limits = _placeholder_limits(active_color_scale)
                            panel_limit_map[str(panel_key)] = None
                            panel_unscaled = True
                        else:
                            limits = resolve_color_limits(vnum, active_color_scale)
                            panel_limit_map[str(panel_key)] = limits
                            panel_limit_specs[color_index][str(panel_key)] = list(
                                limits
                            )
                    else:
                        limits = limit_map[color_index]
                    add_cb = (
                        show_legend
                        and (not is_uniform)
                        and not panel_unscaled
                        and color_index not in unscaled
                        and (
                            active_color_scale.scope == "panel"
                            or facet_by is None
                            or fac_i == n_facets - 1
                        )
                    )
                    base_artist = _draw_continuous(
                        ax,
                        fig,
                        xx,
                        yy,
                        vnum,
                        ss,
                        limits=limits,
                        cmap_name=active_color_scale.cmap,
                        vcenter=active_color_scale.vcenter,
                        scale=active_color_scale.scale,
                        missing_color=active_color_scale.missing_color,
                        default_color=default_color,
                        edgecolor=edgecolor,
                        edgewidth=active_edgewidth,
                        alpha=base_alpha,
                        label=label,
                        is_uniform=is_uniform,
                        sort_values=sort_values,
                        rng=rng,
                        add_colorbar=add_cb,
                        rasterized=rasterized,
                        plt=plt,
                        mpl=mpl,
                    )

                if density_overlay is not None:
                    density_mask = mask.copy()
                    if density_filter is not None:
                        density_mask &= density_filter
                    contour_values = (
                        pd.to_numeric(
                            pd.Series(np.asarray(vals)[density_mask]),
                            errors="coerce",
                        ).to_numpy(dtype=np.float64)
                        if density_overlay.statistic == "mean"
                        else None
                    )
                    _draw_density_overlay(
                        ax,
                        x[density_mask],
                        y[density_mask],
                        overlay=density_overlay,
                        values=contour_values,
                        xlim=xlim,
                        ylim=ylim,
                        theme=theme,
                    )
                highlight_artist = None
                if highlight is not None and highlight_mask is not None:
                    panel_highlight = highlight_mask[mask]
                    highlight_artist = _draw_highlight(
                        ax,
                        xx[panel_highlight],
                        yy[panel_highlight],
                        ss[panel_highlight],
                        highlight=highlight,
                        edgecolor=edgecolor,
                        rasterized=rasterized,
                    )
                if auto_point_size:
                    # Marker areas follow the final panel size after layout.
                    register_layout_point_size(
                        base_artist,
                        n_points=int(mask.sum()),
                        size_min=point_size_range[0],
                        size_max=point_size_range[1],
                        edgecolor=edgecolor,
                        edgewidth=point_edgewidth,
                    )
                    if highlight_artist is not None and highlight is not None:
                        register_layout_point_size(
                            highlight_artist,
                            n_points=int(mask.sum()),
                            size_min=point_size_range[0],
                            size_max=point_size_range[1],
                            multiplier=highlight.size_multiplier,
                        )
                    sized_artists.append((str(panel_key), base_artist))

                if not show_titles:
                    title = None
                elif fac is None:
                    # A labelled colorbar already names the panel.
                    title = None if (not is_cat and add_cb) else label
                else:
                    title = (
                        f"{label} | {facet_by}={fac}" if label else f"{facet_by}={fac}"
                    )
                col = panel_i % n_columns
                row = panel_i // n_columns
                nrows = int(np.ceil(len(panel_keys) / n_columns))
                finish_embedding_axes(
                    ax,
                    xlim=xlim,
                    ylim=ylim,
                    xlabel=f"{layout_name}1" if row == nrows - 1 else "",
                    ylabel=f"{layout_name}2" if col == 0 else "",
                    title=title or None,
                    frame=frame,
                )
                panel_i += 1
        if len(legend_blocks) == 1:
            legend_title, handles = legend_blocks[0]
            fig.legend(
                **_side_legend_kwargs(legend_title, handles),
                loc="outside right center",
            )
        elif legend_blocks:
            _place_legend_blocks(
                fig,
                [
                    (legend_title, handles, [str(h.get_label()) for h in handles])
                    for legend_title, handles in legend_blocks
                ],
            )
        apply_figure_chrome(fig, theme)
        if sized_artists:
            # Constrained layout settles over two draws before sizes are final.
            fig.canvas.draw()
            refresh_layout_point_sizes(fig)
            for key, base_artist in sized_artists:
                panel_point_sizes[key] = float(base_artist.get_sizes()[0])
                panel_edgewidths[key] = float(
                    np.atleast_1d(base_artist.get_linewidths())[0]
                )

    if color_scale.scope == "panel":
        color_limits = panel_limit_map
    else:
        # A color without a value among the drawn cells, such as a feature
        # whose assay measured none of them, has no limits.
        color_limits = {
            display_key(index, labels[index]): (
                limits
                if has_finite_values(np.asarray(color_cache[index][0])[base_mask])
                else None
            )
            for index, limits in limit_map.items()
        }
    feature_assays: set[str] = set()
    for item in color_items:
        if not (
            isinstance(item, FeatureRef)
            or (isinstance(item, str) and item not in metadata_columns)
        ):
            continue
        assay_name = (
            item.assay
            if isinstance(item, FeatureRef) and item.assay is not None
            else resolved_from_assay or getattr(store, "_defaultAssay", None)
        )
        if assay_name is not None:
            feature_assays.add(assay_name)

    # A panel-scope colorbar that no panel gave limits has no scale to show.
    legends = [
        legend
        for legend in legends
        if not (
            legend.kind == "colorbar"
            and legend.extras.get("scope") == "panel"
            and not legend.extras.get("panel_limits")
        )
    ]
    # Each shown colorbar keeps its color's scale, in order, so composed
    # figures pair them; a color without a colorbar keeps no scale.
    shown = [
        index
        for index in colorbar_colors
        if index not in panel_limit_specs or panel_limit_specs[index]
    ]
    scales_out: list[Any] = [resolved_color_scales[index] for index in shown] or [
        color_scale
    ]
    scales_out.extend(categorical_scales)
    result = PlotResult(
        figure=fig,
        axes=axes,
        tables={},
        legends=tuple(legends),
        scales=tuple(scales_out),
        provenance=PlotProvenance(
            assay=(
                layout.assay
                if layout is not None
                else next(iter(feature_assays))
                if len(feature_assays) == 1
                else None
            ),
            cell_key=None if layout is not None else cell_key,
            n_cells=int(base_mask.sum()),
            renderer="matplotlib",
            notes=(
                "embedding",
                "artifact" if layout is not None else "live_metadata",
            ),
            extras={
                "layout": None if layout is None else layout.to_dict(),
                "color_artifacts": [
                    item.to_dict()
                    for item in color_items
                    if isinstance(item, ArtifactRef)
                ],
                "cell_selection": (
                    None if layout_selection is None else layout_selection.to_dict()
                ),
                "sort_values": sort_values,
                "color_scale_scope": color_scale.scope,
                "color_limits": color_limits,
                "facet_by": facet_by,
                "n_colors": n_colors,
                "n_facets": n_facets,
                "panel_keys": [str(k) for k in panel_keys],
                "clip_fraction": clip_fraction,
                "subset_by": subset_by,
                "groups": None if groups is None else list(groups),
                "input_n_cells": n,
                "invalid_coordinate_cells": int((~finite_coordinates).sum()),
                "rasterize_threshold": rasterize_threshold,
                "point_size_by_panel": panel_point_sizes,
                "point_edgewidth_by_panel": panel_edgewidths,
                "point_size_range": list(point_size_range),
                "show_titles": show_titles,
                "omitted_labels": omitted_labels,
                "omitted_legend_entries": omitted_legend_entries,
                "density_overlay": (
                    None
                    if density_overlay is None
                    else {
                        "kind": density_overlay.kind,
                        "statistic": density_overlay.statistic,
                        "pixels": density_overlay.pixels,
                        "sigma": density_overlay.sigma,
                        "min_support": density_overlay.min_support,
                        "levels": density_overlay.levels,
                        "max_hotspots": density_overlay.max_hotspots,
                        "group_by": density_overlay.group_by,
                        "groups": density_overlay.groups,
                        "halo_width": density_overlay.halo_width,
                    }
                ),
                "highlight": (
                    None
                    if highlight is None
                    else {
                        "by": highlight.by,
                        "groups": highlight.groups,
                        "indices": highlight.indices,
                        "n_highlighted": int(
                            (highlight_mask & base_mask).sum()
                            if highlight_mask is not None
                            else 0
                        ),
                    }
                ),
                "normalization": {
                    "source": normalization.source,
                    "transform": normalization.transform,
                },
                "assays": sorted(feature_assays),
                **unmeasured_extras(unmeasured_cell_counts(unmeasured, base_mask)),
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        result.show()
    return result
