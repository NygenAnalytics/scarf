"""Unified reference+target embedding plots from projection slots."""

from collections.abc import Sequence
from importlib.metadata import version
from typing import Any, cast

import numpy as np
import pandas as pd

from ._contracts import CategoricalScale, PlotProvenance
from ._deps import require_matplotlib
from ._figure import LegendSpec, PlotResult, normalize_axes_target
from ._style import (
    DEFAULT_PANEL_INCHES,
    DEFAULT_POINT_EDGEWIDTH,
    DEFAULT_RASTERIZE_THRESHOLD,
    FrameStyle,
    LegendLoc,
    apply_figure_chrome,
    categorical_color_map,
    default_point_edgewidth,
    default_point_size,
    finish_embedding_axes,
    resolve_legend_loc,
    scatter_edgecolor,
    square_axis_limits,
    theme_context,
)
from .embedding import _add_categorical_legend, _add_on_data_labels, _draw_categorical


def _scarf_version() -> str:
    try:
        return version("scarf")
    except Exception:
        return "unknown"


def _load_unified_layout(
    store: Any,
    *,
    layout_key: str,
    from_assay: str | None,
) -> tuple[np.ndarray, np.ndarray, int, list[int], list[str]]:
    loader = getattr(store, "_load_unified_layout_data", None)
    if not callable(loader):
        raise TypeError("store does not provide unified layout data")
    return cast(
        tuple[np.ndarray, np.ndarray, int, list[int], list[str]],
        loader(layout_key=layout_key, from_assay=from_assay),
    )


def _build_unified_groups(
    *,
    ref_n_cells: int,
    target_n_cells: list[int],
    target_names: list[str],
    ref_name: str,
    target_groups: Sequence[Any] | None,
) -> tuple[np.ndarray, list[Any]]:
    n_target = int(sum(target_n_cells))
    if target_groups is None:
        labels: list[Any] = [ref_name] * ref_n_cells
        for name, count in zip(target_names, target_n_cells):
            labels.extend([name] * int(count))
        return np.asarray(labels, dtype=object), [ref_name, *target_names]

    group_labels = list(target_groups)
    if len(group_labels) == len(target_names):
        flattened: list[Any] = []
        for entry in group_labels:
            flattened.extend(list(entry))
        group_labels = flattened
    if len(group_labels) != n_target:
        raise ValueError(
            "target_groups length must equal the number of target cells "
            f"({n_target}); got {len(group_labels)}"
        )
    if any(pd.isna(value) or str(value) == "nan" for value in group_labels):
        raise ValueError("target_groups cannot contain missing values")
    labels = [ref_name] * ref_n_cells + list(group_labels)
    order = [ref_name, *list(dict.fromkeys(group_labels))]
    return np.asarray(labels, dtype=object), order


def unified_embedding(
    store: Any,
    *,
    layout_key: str,
    from_assay: str | None = None,
    show_target_only: bool = False,
    ref_name: str = "reference",
    target_groups: Sequence[Any] | None = None,
    point_size: float | None = None,
    categorical_scale: CategoricalScale | None = None,
    missing_color: str = "#bdbdbd",
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    legend_loc: LegendLoc = "auto",
    frame: FrameStyle = "minimal",
    seed: int | None = None,
    rasterize_threshold: int = DEFAULT_RASTERIZE_THRESHOLD,
    target: Any | None = None,
    show: bool = True,
) -> PlotResult:
    """Plot reference and query cells on a unified projection.

    Use this after ``run_unified_umap`` or ``run_unified_tsne``. Coordinates
    come from ``{assay}/projections/{layout_key}``, not from cell-metadata
    UMAP columns, so ``embedding`` is the wrong entrypoint here.

    By default every cell is drawn and colored as reference vs each target
    batch. ``show_target_only=True`` hides the reference cloud.
    ``target_groups`` replaces batch names with your own labels (one value per
    target cell). ``legend_loc`` and ``frame`` follow the same rules as
    :func:`embedding`.
    """
    plt, mpl = require_matplotlib()
    _ = plt
    x, y, ref_n_cells, target_n_cells, target_names = _load_unified_layout(
        store,
        layout_key=layout_key,
        from_assay=from_assay,
    )
    groups, default_order = _build_unified_groups(
        ref_n_cells=ref_n_cells,
        target_n_cells=target_n_cells,
        target_names=target_names,
        ref_name=ref_name,
        target_groups=target_groups,
    )
    if show_target_only:
        x = x[ref_n_cells:]
        y = y[ref_n_cells:]
        groups = groups[ref_n_cells:]
        default_order = [value for value in default_order if value != ref_name]

    if categorical_scale is not None and categorical_scale.order is not None:
        order = list(categorical_scale.order)
        missing = [value for value in pd.unique(groups) if value not in order]
        if missing:
            raise ValueError(
                "categorical_scale.order is missing observed values: "
                + ", ".join(map(str, missing[:10]))
            )
    else:
        order = list(default_order)
    palette = categorical_color_map(
        order,
        palette=categorical_scale.palette if categorical_scale is not None else None,
    )
    display_labels = categorical_scale.labels if categorical_scale is not None else None
    resolved_missing_color = (
        categorical_scale.missing_color
        if categorical_scale is not None
        else missing_color
    )
    resolved_missing_label = (
        categorical_scale.missing_label if categorical_scale is not None else "NA"
    )

    if seed is not None:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(x))
        x = x[perm]
        y = y[perm]
        groups = groups[perm]

    panel_key = layout_key
    resolved_legend = resolve_legend_loc(len(order), legend_loc)
    resolved_point_size = (
        float(point_size) if point_size is not None else default_point_size(len(x))
    )
    edgewidth = (
        DEFAULT_POINT_EDGEWIDTH
        if point_size is not None
        else default_point_edgewidth(len(x))
    )
    resolved_figsize = figsize
    if resolved_figsize is None and target is None:
        gutter = 1.35 if resolved_legend == "right" else 0.25
        resolved_figsize = (DEFAULT_PANEL_INCHES + gutter, DEFAULT_PANEL_INCHES)

    with theme_context(theme):
        fig, axes, owns = normalize_axes_target(
            target,
            panel_keys=[panel_key],
            figsize=resolved_figsize,
        )
        ax = axes[panel_key]
        edgecolor = scatter_edgecolor(theme)
        sizes = np.full(len(x), resolved_point_size, dtype=np.float64)
        _draw_categorical(
            ax,
            x,
            y,
            groups,
            sizes,
            order=order,
            palette=palette,
            missing_color=resolved_missing_color,
            edgecolor=edgecolor,
            edgewidth=edgewidth,
            rasterized=len(x) >= rasterize_threshold,
        )
        xpad = 0.05 * (float(np.nanmax(x) - np.nanmin(x)) or 1.0)
        ypad = 0.05 * (float(np.nanmax(y) - np.nanmin(y)) or 1.0)
        xlim, ylim = square_axis_limits(
            (float(np.nanmin(x) - xpad), float(np.nanmax(x) + xpad)),
            (float(np.nanmin(y) - ypad), float(np.nanmax(y) + ypad)),
        )
        finish_embedding_axes(
            ax,
            xlim=xlim,
            ylim=ylim,
            xlabel=f"{layout_key}1",
            ylabel=f"{layout_key}2",
            title=layout_key,
            frame=frame,
        )
        if resolved_legend == "on_data":
            _add_on_data_labels(
                ax,
                x,
                y,
                groups,
                order=order,
                labels=display_labels,
                theme=theme,
            )
        elif resolved_legend == "right":
            _add_categorical_legend(
                ax,
                fig,
                mpl,
                order=order,
                palette=palette,
                labels=display_labels,
                label="cells",
                missing=False,
                missing_color=resolved_missing_color,
                missing_label=resolved_missing_label,
                edgecolor=edgecolor,
                figure_level=owns,
            )
        apply_figure_chrome(fig, theme)

    result = PlotResult(
        figure=fig,
        axes=axes,
        tables={
            "cells": pd.DataFrame(
                {
                    f"{layout_key}1": x,
                    f"{layout_key}2": y,
                    "group": groups,
                }
            )
        },
        legends=(LegendSpec(kind="categorical", label="cells"),),
        scales=(
            CategoricalScale(
                order=tuple(order),
                palette=dict(palette),
                labels=(dict(display_labels) if display_labels is not None else None),
                missing_color=resolved_missing_color,
                missing_label=resolved_missing_label,
            ),
        ),
        provenance=PlotProvenance(
            scarf_version=_scarf_version(),
            assay=from_assay or getattr(store, "_defaultAssay", None),
            n_cells=int(len(x)),
            renderer="matplotlib",
            notes=("unified_embedding", f"layout={layout_key}"),
            extras={
                "show_target_only": show_target_only,
                "ref_name": ref_name,
                "target_names": target_names,
                "ref_n_cells": ref_n_cells,
                "target_n_cells": target_n_cells,
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        result.show()
    return result
