"""Embedding scatter plots."""

from collections.abc import Sequence
from typing import Any, Hashable

import numpy as np
import pandas as pd

from ._contracts import (
    CategoricalScale,
    CellField,
    ColorScale,
    FeatureRef,
    NormalizationSpec,
    PlotProvenance,
)
from ._data import fetch_normalized_feature_matrix, resolve_feature
from ._deps import require_matplotlib
from ._figure import LegendSpec, PlotResult, normalize_axes_target
from ._style import categorical_color_map, continuous_norm, theme_context


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


def _scarf_version() -> str:
    try:
        from importlib.metadata import version

        return version("scarf")
    except Exception:
        return "unknown"


def _coerce_color_items(
    color_by: str
    | FeatureRef
    | CellField
    | Sequence[str | FeatureRef | CellField]
    | None,
) -> list[str | FeatureRef | CellField | None]:
    if color_by is None:
        return [None]
    if isinstance(color_by, (str, FeatureRef, CellField)):
        return [color_by]
    return list(color_by)


def _prefetch_colors(
    store: Any,
    color_items: Sequence[str | FeatureRef | CellField | None],
    *,
    from_assay: str | None,
    cell_key: str,
    n_cells: int,
    normalization: NormalizationSpec,
) -> list[tuple[np.ndarray, str, bool, bool]]:
    """Return list of (values, label, is_categorical, is_uniform)."""
    out: list[tuple[np.ndarray, str, bool, bool]] = []

    # Batch RNA-like feature refs / gene strings for one matrix read.
    feature_slots: list[tuple[int, Any]] = []
    for i, item in enumerate(color_items):
        if item is None:
            out.append((np.ones(n_cells), "cells", False, True))
            continue
        if isinstance(item, CellField):
            vals = store.cells.fetch(item.key, key=cell_key)
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
        if isinstance(item, str) and item in store.cells.columns:
            vals = store.cells.fetch(item, key=cell_key)
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
        cell_idx = store.cells.active_index(cell_key)
        mat = fetch_normalized_feature_matrix(
            store,
            resolved,
            cell_idx,
            normalization=normalization,
        )
        for col_i, (slot_i, _) in enumerate(feature_slots):
            feat = resolved[col_i]
            out[slot_i] = (mat[:, col_i], feat.label, False, False)
    return out


def _continuous_limits(
    values: np.ndarray,
    color_scale: ColorScale,
) -> tuple[float, float]:
    v = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(v)
    if not finite.any():
        return 0.0, 1.0
    if color_scale.quantiles is not None:
        q0, q1 = color_scale.quantiles
        vmin = float(np.nanquantile(v[finite], q0))
        vmax = float(np.nanquantile(v[finite], q1))
    else:
        vmin = float(np.nanmin(v[finite]))
        vmax = float(np.nanmax(v[finite]))
    if color_scale.vmin is not None:
        vmin = color_scale.vmin
    if color_scale.vmax is not None:
        vmax = color_scale.vmax
    if vmax <= vmin:
        if color_scale.vmin is not None or color_scale.vmax is not None:
            raise ValueError("Color limits must satisfy vmin < vmax")
        vmax = vmin + 1.0
    return vmin, vmax


def _draw_categorical(
    ax: Any,
    xx: np.ndarray,
    yy: np.ndarray,
    vv: np.ndarray,
    ss: np.ndarray,
    *,
    order: list[Any],
    palette: dict[Any, str],
    missing_color: str,
    rasterized: bool,
) -> None:
    colors = [
        missing_color if pd.isna(val) or val not in palette else palette[val]
        for val in vv
    ]
    ax.scatter(
        xx,
        yy,
        c=colors,
        s=ss,
        linewidths=0.1,
        edgecolors="k",
        rasterized=rasterized,
    )


def _add_categorical_legend(
    ax: Any,
    mpl: Any,
    *,
    order: list[Any],
    palette: dict[Any, str],
    label: str,
    missing: bool,
    missing_color: str,
    missing_label: str,
) -> None:
    handles = [
        mpl.lines.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markerfacecolor=palette[value],
            markeredgecolor="k",
            markeredgewidth=0.3,
            markersize=5,
            label=str(value),
        )
        for value in order
    ]
    if missing:
        handles.append(
            mpl.lines.Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markerfacecolor=missing_color,
                markeredgecolor="k",
                markeredgewidth=0.3,
                markersize=5,
                label=missing_label,
            )
        )
    ax.legend(
        handles=handles,
        title=label or None,
        frameon=False,
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0,
    )


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
    missing_color: str,
    default_color: str,
    label: str,
    is_uniform: bool,
    sort_values: bool,
    rng: np.random.Generator | None,
    add_colorbar: bool,
    rasterized: bool,
    plt: Any,
    mpl: Any,
) -> None:
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

    if is_uniform or len(xx) == 0:
        ax.scatter(
            xx,
            yy,
            c=[default_color] * len(xx),
            s=ss if len(ss) else 10,
            linewidths=0.1,
            edgecolors="k",
            rasterized=rasterized,
        )
        return

    vmin, vmax = limits
    if vmax == vmin:
        vmax = vmin + 1.0
    norm = continuous_norm(mpl, vmin=vmin, vmax=vmax, vcenter=vcenter)
    cmap = plt.get_cmap(cmap_name or "viridis")
    face = np.empty((len(vnum), 4))
    face[:] = mpl.colors.to_rgba(missing_color)
    if finite.any():
        face[finite] = cmap(norm(vnum[finite]))
    ax.scatter(
        xx,
        yy,
        c=face,
        s=ss,
        linewidths=0.1,
        edgecolors="k",
        rasterized=rasterized,
    )
    if add_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, shrink=0.6, fraction=0.05, pad=0.02)
        if label:
            cb.set_label(label)


def _soft_clip(values: np.ndarray, clip_fraction: float) -> np.ndarray:
    if clip_fraction <= 0:
        return values
    if clip_fraction >= 0.5:
        raise ValueError("clip_fraction must be in [0, 0.5)")
    v = np.asarray(values, dtype=np.float64).copy()
    finite = np.isfinite(v)
    if not finite.any():
        return v
    lo = float(np.percentile(v[finite], 100 * clip_fraction))
    hi = float(np.percentile(v[finite], 100 - 100 * clip_fraction))
    v[finite & (v < lo)] = lo
    v[finite & (v > hi)] = hi
    return v


def embedding(
    store: Any,
    *,
    layout_key: str,
    color_by: str
    | FeatureRef
    | CellField
    | Sequence[str | FeatureRef | CellField]
    | None = None,
    facet_by: str | None = None,
    facet_order: Sequence[Any] | None = None,
    cell_key: str = "I",
    from_assay: str | None = None,
    normalization: NormalizationSpec | None = None,
    point_size: float = 10,
    point_sizes: np.ndarray | Sequence[float] | None = None,
    sort_values: bool = False,
    color_scale: ColorScale | None = None,
    categorical_scale: CategoricalScale | None = None,
    default_color: str = "steelblue",
    missing_color: str = "#bdbdbd",
    clip_fraction: float = 0.0,
    subset_by: str | None = None,
    n_columns: int | None = None,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    seed: int | None = None,
    rasterize_threshold: int = 50_000,
    show: bool = False,
) -> PlotResult:
    """2D embedding scatter with shared per-feature color scales across facets.

    Multi-gene × condition layouts use one row per color value and one column
    per facet by default. Color limits and categorical palettes are computed on
    the full selected population so panels stay comparable.

    Preserves useful ``DataStore.plot_layout`` behaviors:
    - ``sort_values=True`` draws higher continuous values on top
    - ``point_sizes`` sets per-point marker areas (e.g. mapping confidence)
    """
    plt, mpl = require_matplotlib()
    if rasterize_threshold < 0:
        raise ValueError("rasterize_threshold must be >= 0")
    normalization = normalization or NormalizationSpec()
    color_scale = color_scale or ColorScale(cmap="viridis", scope="feature")
    missing = (
        categorical_scale.missing_color
        if categorical_scale is not None
        else missing_color
    )

    x = np.asarray(store.cells.fetch(f"{layout_key}1", key=cell_key), dtype=np.float64)
    y = np.asarray(store.cells.fetch(f"{layout_key}2", key=cell_key), dtype=np.float64)
    n = len(x)
    if n == 0:
        raise ValueError(f"No cells selected by cell_key {cell_key!r}")
    finite_coordinates = np.isfinite(x) & np.isfinite(y)
    if not finite_coordinates.any():
        raise ValueError(f"Layout {layout_key!r} has no finite coordinates")
    if point_sizes is not None and len(point_sizes) != n:
        raise ValueError("point_sizes length must match number of selected cells")
    size_arr = (
        np.asarray(point_sizes, dtype=np.float64)
        if point_sizes is not None
        else np.full(n, point_size, dtype=np.float64)
    )

    color_items = _coerce_color_items(color_by)
    color_cache = _prefetch_colors(
        store,
        color_items,
        from_assay=from_assay,
        cell_key=cell_key,
        n_cells=n,
        normalization=normalization,
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

    # Optional boolean cell subselection (legacy plot_layout behavior)
    base_mask = finite_coordinates.copy()
    if subset_by is not None:
        sub = np.asarray(store.cells.fetch(subset_by, key=cell_key))
        if sub.dtype != bool:
            raise TypeError(f"subset_by {subset_by!r} must be boolean; got {sub.dtype}")
        if len(sub) != n:
            raise ValueError("subset_by length must match selected cells")
        base_mask &= sub
    if not base_mask.any():
        raise ValueError("No cells remain after applying layout/subset filters")

    if facet_by is not None:
        facet_values = np.asarray(store.cells.fetch(facet_by, key=cell_key))
        if facet_order is not None:
            facets = list(facet_order)
        else:
            facets = sorted(pd.unique(facet_values), key=lambda v: (pd.isna(v), str(v)))
    else:
        facet_values = None
        facets = [None]

    # Stable panel keys: (color_label, facet) when faceting, else color_label
    panel_keys: list[Hashable] = []
    for _, label, _, _ in color_cache:
        for fac in facets:
            if fac is None:
                panel_keys.append(label)
            else:
                panel_keys.append((label, fac))
    if len(set(panel_keys)) != len(panel_keys):
        panel_keys = [(i, k) for i, k in enumerate(panel_keys)]

    n_colors = len(color_cache)
    n_facets = len(facets)
    if n_columns is None:
        n_columns = n_facets if facet_by is not None else min(n_colors, 4)
    n_columns = max(1, min(n_columns, len(panel_keys)))

    if figsize is None and target is None:
        nrows = int(np.ceil(len(panel_keys) / n_columns))
        figsize = (3.6 * n_columns, 3.6 * nrows)

    fig, axes, owns = normalize_axes_target(
        target, panel_keys=panel_keys, figsize=figsize, n_columns=n_columns
    )

    selected_x = x[base_mask]
    selected_y = y[base_mask]
    xpad = 0.05 * (float(selected_x.max() - selected_x.min()) or 1.0)
    ypad = 0.05 * (float(selected_y.max() - selected_y.min()) or 1.0)
    xlim = (float(selected_x.min() - xpad), float(selected_x.max() + xpad))
    ylim = (float(selected_y.min() - ypad), float(selected_y.max() + ypad))

    labels = [label for _, label, _, _ in color_cache]
    label_counts = pd.Series(labels).value_counts()

    def display_key(index: int, label: str) -> str:
        return label if label_counts[label] == 1 else f"{index}:{label}"

    limit_map: dict[int, tuple[float, float]] = {}
    categorical_maps: dict[int, tuple[list[Any], dict[Any, str]]] = {}
    shared_limits: tuple[float, float] | None = None
    if color_scale.scope == "shared":
        shared_values = [
            np.asarray(values, dtype=np.float64)[base_mask]
            for values, _, is_categorical, is_uniform in color_cache
            if not is_categorical and not is_uniform
        ]
        if shared_values:
            shared_limits = _continuous_limits(
                np.concatenate(shared_values),
                color_scale,
            )

    for color_index, (vals, _, is_cat, is_uniform) in enumerate(color_cache):
        if is_uniform:
            continue
        vals_sel = np.asarray(vals)[base_mask]
        if is_cat:
            observed = list(pd.Series(vals_sel).dropna().unique())
            if categorical_scale and categorical_scale.order is not None:
                order = list(categorical_scale.order)
                unlisted = [value for value in observed if value not in order]
                if unlisted:
                    raise ValueError(
                        "categorical_scale.order is missing observed values: "
                        + ", ".join(map(str, unlisted[:10]))
                    )
            else:
                order = sorted(observed, key=lambda v: str(v))
            palette = categorical_color_map(
                order,
                palette=categorical_scale.palette if categorical_scale else None,
                missing_label=None,
            )
            categorical_maps[color_index] = (order, palette)
        elif color_scale.scope != "panel":
            limit_map[color_index] = (
                shared_limits
                if shared_limits is not None
                else _continuous_limits(vals_sel, color_scale)
            )

    legends: list[LegendSpec] = []
    scales_out: list[Any] = [color_scale]
    for color_index, (order, palette) in categorical_maps.items():
        label = labels[color_index]
        scales_out.append(CategoricalScale(order=tuple(order), palette=dict(palette)))
        legends.append(LegendSpec(kind="categorical", label=label))
    for color_index, (_, label, is_categorical, is_uniform) in enumerate(color_cache):
        if is_categorical or is_uniform:
            continue
        limits = limit_map.get(color_index)
        legends.append(
            LegendSpec(
                kind="colorbar",
                label=label,
                extras=(
                    {"vmin": limits[0], "vmax": limits[1]}
                    if limits is not None
                    else {"scope": "panel"}
                ),
            )
        )

    rng = np.random.default_rng(seed) if seed is not None else None
    panel_limit_map: dict[str, tuple[float, float]] = {}

    with theme_context(theme):
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
                ss = size_arr[mask]
                rasterized = len(xx) >= rasterize_threshold

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
                    order, palette = categorical_maps[color_index]
                    _draw_categorical(
                        ax,
                        xx,
                        yy,
                        vv,
                        ss,
                        order=order,
                        palette=palette,
                        missing_color=missing,
                        rasterized=rasterized,
                    )
                    if fac_i == n_facets - 1:
                        _add_categorical_legend(
                            ax,
                            mpl,
                            order=order,
                            palette=palette,
                            label=label,
                            missing=bool(pd.isna(np.asarray(vals)[base_mask]).any()),
                            missing_color=missing,
                            missing_label=(
                                categorical_scale.missing_label
                                if categorical_scale is not None
                                else "NA"
                            ),
                        )
                else:
                    vnum = pd.to_numeric(pd.Series(vv), errors="coerce").to_numpy(
                        dtype=np.float64
                    )
                    if is_uniform:
                        limits = (0.0, 1.0)
                    elif color_scale.scope == "panel":
                        limits = _continuous_limits(vnum, color_scale)
                        panel_limit_map[str(panel_key)] = limits
                    else:
                        limits = limit_map[color_index]
                    add_cb = (not is_uniform) and (
                        color_scale.scope == "panel"
                        or facet_by is None
                        or fac_i == n_facets - 1
                    )
                    _draw_continuous(
                        ax,
                        fig,
                        xx,
                        yy,
                        vnum,
                        ss,
                        limits=limits,
                        cmap_name=color_scale.cmap,
                        vcenter=color_scale.vcenter,
                        missing_color=color_scale.missing_color,
                        default_color=default_color,
                        label=label,
                        is_uniform=is_uniform,
                        sort_values=sort_values,
                        rng=rng,
                        add_colorbar=add_cb,
                        rasterized=rasterized,
                        plt=plt,
                        mpl=mpl,
                    )

                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xticks([])
                ax.set_yticks([])
                if fac is None:
                    title = label
                else:
                    title = (
                        f"{label} | {facet_by}={fac}" if label else f"{facet_by}={fac}"
                    )
                if title:
                    ax.set_title(title)
                ax.set_xlabel(f"{layout_key}1")
                ax.set_ylabel(f"{layout_key}2")
                panel_i += 1

    if color_scale.scope == "panel":
        color_limits = panel_limit_map
    else:
        color_limits = {
            display_key(index, labels[index]): limits
            for index, limits in limit_map.items()
        }
    feature_assays: set[str] = set()
    for item in color_items:
        if not (
            isinstance(item, FeatureRef)
            or (isinstance(item, str) and item not in store.cells.columns)
        ):
            continue
        assay_name = (
            item.assay
            if isinstance(item, FeatureRef) and item.assay is not None
            else from_assay or getattr(store, "_defaultAssay", None)
        )
        if assay_name is not None:
            feature_assays.add(assay_name)

    result = PlotResult(
        figure=fig,
        axes=axes,
        tables={},
        legends=tuple(legends),
        scales=tuple(scales_out),
        provenance=PlotProvenance(
            scarf_version=_scarf_version(),
            assay=next(iter(feature_assays)) if len(feature_assays) == 1 else None,
            cell_key=cell_key,
            n_cells=int(base_mask.sum()),
            renderer="matplotlib",
            notes=("embedding", "materialized", f"layout={layout_key}"),
            extras={
                "sort_values": sort_values,
                "color_scale_scope": color_scale.scope,
                "color_limits": color_limits,
                "facet_by": facet_by,
                "n_colors": n_colors,
                "n_facets": n_facets,
                "panel_keys": [str(k) for k in panel_keys],
                "clip_fraction": clip_fraction,
                "subset_by": subset_by,
                "input_n_cells": n,
                "invalid_coordinate_cells": int((~finite_coordinates).sum()),
                "rasterize_threshold": rasterize_threshold,
                "normalization": {
                    "source": normalization.source,
                    "transform": normalization.transform,
                },
                "assays": sorted(feature_assays),
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        result.show()
    return result
