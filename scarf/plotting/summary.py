"""Dotplot and matrixplot."""

from collections.abc import Mapping, Sequence
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
from ._data import coerce_feature_list, summarize_features_by_group
from ._deps import require_matplotlib
from ._figure import LegendSpec, PlotResult, normalize_axes_target
from ._style import continuous_norm, theme_context


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
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show: bool = False,
) -> PlotResult:
    """Mean expression (color) and fraction expressing (size) by group."""
    _, mpl = require_matplotlib()
    color_scale = color_scale or ColorScale(cmap="viridis")
    size_scale = size_scale or SizeScale()
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
    if standardize == "feature":
        aggregate = _standardize_feature(aggregate, "mean")
    elif standardize not in ("none", "feature"):
        raise ValueError("standardize must be 'none' or 'feature'")

    group_keys = (group_by,) if isinstance(group_by, str) else tuple(group_by)
    plot_df = aggregate.copy()
    plot_df["group_label"] = _group_axis_labels(plot_df, group_keys)
    # Preserve feature order from input
    feature_order = list(dict.fromkeys(plot_df["feature"].tolist()))
    group_order = list(dict.fromkeys(plot_df["group_label"].tolist()))
    plot_df["feature"] = pd.Categorical(
        plot_df["feature"], categories=feature_order, ordered=True
    )
    plot_df["group_label"] = pd.Categorical(
        plot_df["group_label"], categories=group_order, ordered=True
    )

    panel_key: Hashable = "dotplot"
    resolved_figsize = figsize
    if resolved_figsize is None and target is None:
        resolved_figsize = (
            max(4.0, 0.35 * len(group_order) + 2),
            max(3.0, 0.3 * len(feature_order) + 1.5),
        )
    fig, axes, owns = normalize_axes_target(
        target,
        panel_keys=[panel_key],
        figsize=resolved_figsize,
    )
    ax = axes[panel_key]

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

    with theme_context(theme):
        sc = ax.scatter(
            x,
            y,
            c=vals,
            s=areas,
            cmap=color_scale.cmap or "viridis",
            norm=norm,
            edgecolors="k",
            linewidths=0.2,
            rasterized=False,
        )
        ax.set_xticks(range(len(group_order)))
        ax.set_xticklabels(group_order, rotation=45, ha="right")
        ax.set_yticks(range(len(feature_order)))
        ax.set_yticklabels(feature_order)
        ax.set_xlim(-0.5, len(group_order) - 0.5)
        ax.set_ylim(-0.5, len(feature_order) - 0.5)
        ax.invert_yaxis()
        cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
        cb.set_label("mean" if standardize == "none" else "standardized mean")
        legend_values = np.array([0.25, 0.5, 0.75, 1.0])
        legend_areas = size_scale.areas(legend_values)
        handles = [
            ax.scatter(
                [],
                [],
                s=area,
                facecolor="#bdbdbd",
                edgecolor="k",
                linewidth=0.2,
            )
            for area in legend_areas
        ]
        ax.legend(
            handles,
            [f"{value:g}" for value in legend_values],
            title="fraction",
            frameon=False,
            bbox_to_anchor=(1.02, 0),
            loc="lower left",
            borderaxespad=0,
        )
        ax.set_xlabel(" / ".join(group_keys))
        ax.set_ylabel("feature")

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

    from importlib.metadata import version

    try:
        scarf_version = version("scarf")
    except Exception:
        scarf_version = "unknown"

    result = PlotResult(
        figure=fig,
        axes=axes,
        tables=tables,
        legends=(
            LegendSpec(kind="colorbar", label="mean"),
            LegendSpec(kind="size", label="fraction", extras={"domain": [0.0, 1.0]}),
        ),
        scales=(color_scale, size_scale, CategoricalScale(order=tuple(group_order))),
        provenance=PlotProvenance(
            scarf_version=scarf_version,
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
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show: bool = False,
) -> PlotResult:
    """Feature-by-group matrix of mean or fraction (no forced clustering)."""
    _, mpl = require_matplotlib()
    if value not in ("mean", "fraction"):
        raise ValueError("value must be 'mean' or 'fraction'")
    color_scale = color_scale or ColorScale(cmap="viridis")
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
    feature_order = list(dict.fromkeys(plot_df["feature"].tolist()))
    group_order = list(dict.fromkeys(plot_df["group_label"].tolist()))
    mat = plot_df.pivot_table(
        index="feature",
        columns="group_label",
        values=value,
        observed=False,
    ).reindex(index=feature_order, columns=group_order)

    panel_key: Hashable = "matrixplot"
    resolved_figsize = figsize
    if resolved_figsize is None and target is None:
        resolved_figsize = (
            max(4.0, 0.4 * len(group_order) + 2),
            max(3.0, 0.35 * len(feature_order) + 1.5),
        )
    fig, axes, owns = normalize_axes_target(
        target,
        panel_keys=[panel_key],
        figsize=resolved_figsize,
    )
    ax = axes[panel_key]
    data = mat.to_numpy(dtype=np.float64)
    vmin, vmax = _color_limits(data, color_scale)
    norm = continuous_norm(
        mpl,
        vmin=vmin,
        vmax=vmax,
        vcenter=color_scale.vcenter,
    )

    with theme_context(theme):
        im = ax.imshow(
            data,
            aspect="auto",
            cmap=color_scale.cmap or "viridis",
            norm=norm,
            interpolation="nearest",
        )
        ax.set_xticks(range(len(group_order)))
        ax.set_xticklabels(group_order, rotation=45, ha="right")
        ax.set_yticks(range(len(feature_order)))
        ax.set_yticklabels(feature_order)
        cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
        cb.set_label(value)

    tables = {"aggregate": aggregate, "matrix": mat.reset_index()}
    if per_sample is not None:
        tables["per_sample"] = per_sample
    n_samples, dropped_sample_cells = _sample_counts(
        store,
        cell_key=cell_key,
        sample_by=sample_by,
        study_design=study_design,
    )
    assays = _feature_assays(store, features, from_assay)

    from importlib.metadata import version

    try:
        scarf_version = version("scarf")
    except Exception:
        scarf_version = "unknown"

    result = PlotResult(
        figure=fig,
        axes=axes,
        tables=tables,
        legends=(LegendSpec(kind="colorbar", label=value),),
        scales=(color_scale,),
        provenance=PlotProvenance(
            scarf_version=scarf_version,
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
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        result.show()
    return result
