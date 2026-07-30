"""Diagnostics for reference mapping and label transfer."""

from collections.abc import Sequence
from importlib.metadata import version
from typing import Any, Hashable, Literal
import warnings

import numpy as np
import pandas as pd

from ._contracts import (
    CategoricalScale,
    ColorScale,
    PlotProvenance,
)
from ._deps import require_matplotlib
from ._figure import LegendSpec, PlotResult, normalize_axes_target
from ._style import (
    apply_figure_chrome,
    categorical_color_map,
    continuous_norm,
    default_point_edgewidth,
    default_point_size,
    finish_embedding_axes,
    scatter_edgecolor,
    sort_categories,
    square_axis_limits,
    theme_context,
)


def _scarf_version() -> str:
    try:
        return version("scarf")
    except Exception:
        return "unknown"


def _resolved_mapping_keys(
    store: Any,
    from_assay: str | None,
    cell_key: str | None,
) -> tuple[str, str]:
    assay = from_assay or getattr(store, "_defaultAssay", None)
    if assay is None:
        raise ValueError("No default assay is configured")
    if cell_key is None:
        resolver = getattr(store, "_get_latest_cell_key", None)
        cell_key = resolver(assay) if callable(resolver) else "I"
    return str(assay), str(cell_key)


def _mapping_result(
    store: Any,
    *,
    target_name: str,
    from_assay: str | None,
    cell_key: str | None,
) -> Any:
    result = store.get_mapping_result(
        target_name,
        from_assay=from_assay,
        cell_key=cell_key,
        load_arrays=True,
    )
    if result.indices is None or result.distances is None:
        raise ValueError(f"Mapping {target_name!r} has no saved neighbor arrays")
    return result


def _projected_mapping_coordinates(
    store: Any,
    result: Any,
    *,
    reference_layout_key: str,
    cell_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    reference_x = np.asarray(
        store.cells.fetch(f"{reference_layout_key}1", key=cell_key),
        dtype=np.float64,
    )
    reference_y = np.asarray(
        store.cells.fetch(f"{reference_layout_key}2", key=cell_key),
        dtype=np.float64,
    )
    reference_layout = np.column_stack((reference_x, reference_y))
    indices = np.asarray(result.indices)
    distances = np.asarray(result.distances)
    if indices.ndim != 2 or distances.shape != indices.shape:
        raise ValueError("Mapping neighbor arrays have incompatible shapes")
    if len(indices) and int(indices.max()) >= len(reference_layout):
        raise ValueError("Mapping neighbor indices exceed the reference layout")
    from ..mapping.confidence import distance_weights

    weights = distance_weights(distances)
    projected = np.einsum("nk,nkd->nd", weights, reference_layout[indices])
    return reference_layout, projected


def _external_groups(
    values: Sequence[Any] | np.ndarray | None,
    n_rows: int,
    *,
    default: str,
    argument_name: str,
) -> np.ndarray:
    if values is None:
        return np.full(n_rows, default, dtype=object)
    groups = np.asarray(values, dtype=object)
    if groups.ndim != 1 or len(groups) != n_rows:
        raise ValueError(f"{argument_name} must have one value per mapped cell")
    if pd.isna(groups).any():
        raise ValueError(f"{argument_name} cannot contain missing values")
    return groups


def _categorical_contract(
    values: np.ndarray,
    scale: CategoricalScale | None,
) -> tuple[list[Any], dict[Any, str], CategoricalScale]:
    observed = list(pd.unique(values))
    if scale is not None and scale.order is not None:
        missing = [value for value in observed if value not in scale.order]
        if missing:
            raise ValueError(
                "categorical_scale.order is missing values: "
                + ", ".join(map(str, missing[:10]))
            )
        order = [value for value in scale.order if value in set(observed)]
    else:
        order = sort_categories(observed)
    palette = categorical_color_map(
        order,
        palette=scale.palette if scale is not None else None,
        palette_name=scale.palette_name if scale is not None else "default",
    )
    return (
        order,
        palette,
        CategoricalScale(
            order=tuple(order),
            palette=palette,
            labels=scale.labels if scale is not None else None,
            missing_color=scale.missing_color if scale is not None else "#bdbdbd",
            missing_label=scale.missing_label if scale is not None else "NA",
            palette_name=scale.palette_name if scale is not None else "default",
        ),
    )


def _axis_limits(x: np.ndarray, y: np.ndarray) -> tuple[Any, Any]:
    xpad = 0.05 * (float(np.ptp(x)) or 1.0)
    ypad = 0.05 * (float(np.ptp(y)) or 1.0)
    return square_axis_limits(
        (float(np.min(x) - xpad), float(np.max(x) + xpad)),
        (float(np.min(y) - ypad), float(np.max(y) + ypad)),
    )


def mapping_score(
    store: Any,
    *,
    target_name: str,
    target_groups: Sequence[Any] | np.ndarray | None = None,
    layout_key: str | None = None,
    kind: Literal["embedding", "histogram"] = "embedding",
    from_assay: str | None = None,
    cell_key: str | None = None,
    log_transform: bool = True,
    multiplier: float = 1000,
    weighted: bool = True,
    fixed_weight: float = 0.1,
    bins: int = 40,
    point_size: float | None = None,
    color_scale: ColorScale | None = None,
    categorical_scale: CategoricalScale | None = None,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot reference-cell mapping scores for one or more query groups."""
    if kind not in ("embedding", "histogram"):
        raise ValueError("kind must be 'embedding' or 'histogram'")
    if kind == "embedding" and layout_key is None:
        raise ValueError("layout_key is required for an embedding mapping score")
    if bins < 1:
        raise ValueError("bins must be positive")
    score_rows = list(
        store.get_mapping_score(
            target_name,
            target_groups=(
                None if target_groups is None else np.asarray(target_groups)
            ),
            from_assay=from_assay,
            cell_key=cell_key,
            log_transform=log_transform,
            multiplier=multiplier,
            weighted=weighted,
            fixed_weight=fixed_weight,
        )
    )
    if not score_rows:
        raise ValueError(f"Mapping {target_name!r} produced no score groups")
    labels = [row[0] for row in score_rows]
    display_labels = {label: str(label) for label in labels}
    if target_groups is None and len(labels) == 1:
        display_labels[labels[0]] = target_name
    score_arrays = [np.asarray(row[1], dtype=np.float64) for row in score_rows]
    n_reference = len(score_arrays[0])
    if any(values.shape != (n_reference,) for values in score_arrays):
        raise ValueError("Mapping score groups have incompatible lengths")
    score_table = pd.concat(
        (
            pd.DataFrame(
                {
                    "group": label,
                    "referenceIndex": np.arange(n_reference),
                    "score": values,
                }
            )
            for label, values in zip(labels, score_arrays, strict=True)
        ),
        ignore_index=True,
    )
    _, mpl = require_matplotlib()
    color_scale = color_scale or ColorScale(cmap="viridis")
    panel_keys: list[Hashable] = (
        list(labels) if kind == "embedding" else ["mapping_score"]
    )
    if figsize is None and target is None:
        figsize = (3.4 * len(panel_keys), 3.2) if kind == "embedding" else (5.0, 3.6)
    legends: list[LegendSpec] = []
    scales: list[Any] = []
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target,
            panel_keys=panel_keys,
            figsize=figsize,
        )
        if kind == "embedding":
            assert layout_key is not None
            _, resolved_cell_key = _resolved_mapping_keys(
                store,
                from_assay,
                cell_key,
            )
            x = np.asarray(
                store.cells.fetch(f"{layout_key}1", key=resolved_cell_key),
                dtype=np.float64,
            )
            y = np.asarray(
                store.cells.fetch(f"{layout_key}2", key=resolved_cell_key),
                dtype=np.float64,
            )
            if len(x) != n_reference or len(y) != n_reference:
                raise ValueError("Reference layout does not match mapping scores")
            xlim, ylim = _axis_limits(x, y)
            shared_values = np.concatenate(score_arrays)
            shared_limits = (
                _continuous_limits(shared_values, color_scale)
                if color_scale.scope == "shared"
                else None
            )
            resolved_point_size = (
                float(point_size)
                if point_size is not None
                else default_point_size(n_reference)
            )
            edgewidth = default_point_edgewidth(
                n_reference,
                point_size=resolved_point_size,
            )
            edgecolor = scatter_edgecolor(theme)
            for label, values in zip(labels, score_arrays, strict=True):
                ax = axes[label]
                limits = shared_limits or _continuous_limits(values, color_scale)
                norm = continuous_norm(
                    mpl,
                    vmin=limits[0],
                    vmax=limits[1],
                    vcenter=color_scale.vcenter,
                )
                order_index = np.argsort(values)
                artist = ax.scatter(
                    x[order_index],
                    y[order_index],
                    c=values[order_index],
                    s=resolved_point_size,
                    cmap=color_scale.cmap or "viridis",
                    norm=norm,
                    edgecolors=edgecolor if edgewidth > 0 else "none",
                    linewidths=edgewidth,
                    rasterized=n_reference >= 50_000,
                )
                finish_embedding_axes(
                    ax,
                    xlim=xlim,
                    ylim=ylim,
                    title=display_labels[label],
                    frame="minimal",
                )
                if show_legend:
                    colorbar = figure.colorbar(
                        artist,
                        ax=ax,
                        location="top",
                        orientation="horizontal",
                        shrink=0.8,
                        fraction=0.06,
                        pad=0.03,
                    )
                    colorbar.set_label("Mapping score")
                legends.append(
                    LegendSpec(
                        kind="colorbar",
                        label=f"Mapping score: {display_labels[label]}",
                        extras={"vmin": limits[0], "vmax": limits[1]},
                    )
                )
            scales.append(color_scale)
        else:
            group_values = np.asarray(labels, dtype=object)
            order, palette, resolved_categorical = _categorical_contract(
                group_values,
                categorical_scale,
            )
            ax = axes["mapping_score"]
            finite = np.concatenate(
                [values[np.isfinite(values)] for values in score_arrays]
            )
            shared_bins: int | np.ndarray = (
                np.histogram_bin_edges(finite, bins=bins) if len(finite) else bins
            )
            for label in order:
                values = score_arrays[labels.index(label)]
                ax.hist(
                    values[np.isfinite(values)],
                    bins=shared_bins,
                    histtype="step",
                    linewidth=1.5,
                    color=palette[label],
                    label=display_labels[label],
                )
            ax.set_xlabel("Mapping score")
            ax.set_ylabel("Reference cells")
            if show_legend:
                ax.legend(frameon=False, title="Query group")
            legends.append(LegendSpec(kind="categorical", label="Query group"))
            scales.append(resolved_categorical)
        apply_figure_chrome(figure, theme)
    result = PlotResult(
        figure=figure,
        axes=axes,
        tables={"scores": score_table},
        legends=tuple(legends),
        scales=tuple(scales),
        provenance=PlotProvenance(
            scarf_version=_scarf_version(),
            assay=from_assay or getattr(store, "_defaultAssay", None),
            cell_key=cell_key,
            n_cells=n_reference,
            renderer="matplotlib",
            notes=("mapping_score", kind),
            extras={
                "target_name": target_name,
                "layout_key": layout_key,
                "groups": list(labels),
                "log_transform": log_transform,
                "multiplier": multiplier,
                "weighted": weighted,
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        result.show()
    return result


def _continuous_limits(
    values: np.ndarray,
    scale: ColorScale,
) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return (0.0, 1.0)
    if scale.quantiles is not None:
        low, high = np.quantile(finite, scale.quantiles)
    else:
        low, high = float(np.min(finite)), float(np.max(finite))
    if scale.vmin is not None:
        low = scale.vmin
    if scale.vmax is not None:
        high = scale.vmax
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def _label_evidence(
    store: Any,
    *,
    target_name: str,
    reference_class_group: str,
    from_assay: str | None,
    cell_key: str | None,
    threshold_fraction: float,
    na_val: str,
    max_distance: float | None,
) -> pd.DataFrame:
    return store.get_target_label_evidence(
        target_name,
        reference_class_group=reference_class_group,
        from_assay=from_assay,
        cell_key=cell_key,
        threshold_fraction=threshold_fraction,
        na_val=na_val,
        max_distance=max_distance,
    ).copy()


def mapping_evidence(
    store: Any,
    *,
    target_name: str,
    reference_class_group: str,
    target_groups: Sequence[Any] | np.ndarray | None = None,
    metrics: Sequence[str] = (
        "voteFraction",
        "topTwoMargin",
        "voteEntropy",
        "referenceDistancePercentile",
    ),
    kind: Literal["histogram", "box", "embedding"] = "histogram",
    reference_layout_key: str | None = None,
    bins: int = 30,
    from_assay: str | None = None,
    cell_key: str | None = None,
    threshold_fraction: float = 0.5,
    na_val: str = "NA",
    max_distance: float | None = None,
    categorical_scale: CategoricalScale | None = None,
    color_scale: ColorScale | None = None,
    point_size: float | None = None,
    show_unknown: bool = True,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot query-level label-transfer evidence."""
    if kind not in ("histogram", "box", "embedding"):
        raise ValueError("kind must be 'histogram', 'box', or 'embedding'")
    if kind == "embedding" and reference_layout_key is None:
        raise ValueError("reference_layout_key is required for an evidence embedding")
    if bins < 1:
        raise ValueError("bins must be positive")
    evidence = _label_evidence(
        store,
        target_name=target_name,
        reference_class_group=reference_class_group,
        from_assay=from_assay,
        cell_key=cell_key,
        threshold_fraction=threshold_fraction,
        na_val=na_val,
        max_distance=max_distance,
    )
    groups = _external_groups(
        target_groups,
        len(evidence),
        default=target_name,
        argument_name="target_groups",
    )
    evidence["group"] = groups
    requested_metrics = list(metrics)
    if not requested_metrics:
        raise ValueError("metrics must be non-empty")
    missing = [metric for metric in requested_metrics if metric not in evidence]
    if missing:
        raise KeyError("Unknown evidence metrics: " + ", ".join(missing))
    for metric in requested_metrics:
        evidence[metric] = pd.to_numeric(evidence[metric], errors="coerce")
    order: list[Any] = []
    palette: dict[Any, str] = {}
    resolved_categorical: CategoricalScale | None = None
    reference_layout: np.ndarray | None = None
    projected: np.ndarray | None = None
    resolved_assay, resolved_cell_key = _resolved_mapping_keys(
        store,
        from_assay,
        cell_key,
    )
    if kind == "embedding":
        assert reference_layout_key is not None
        mapping_result = _mapping_result(
            store,
            target_name=target_name,
            from_assay=resolved_assay,
            cell_key=resolved_cell_key,
        )
        reference_layout, projected = _projected_mapping_coordinates(
            store,
            mapping_result,
            reference_layout_key=reference_layout_key,
            cell_key=resolved_cell_key,
        )
        if len(projected) != len(evidence):
            raise ValueError("Projected coordinates do not match label evidence")
        color_scale = color_scale or ColorScale(
            cmap="viridis",
            quantiles=(0.01, 0.99),
        )
    else:
        order, palette, resolved_categorical = _categorical_contract(
            groups,
            categorical_scale,
        )
    if figsize is None and target is None:
        panel_width = 3.5 if kind == "embedding" else 4.0
        figsize = (panel_width * min(len(requested_metrics), 3), 3.3)
    _, mpl = require_matplotlib()
    legend_specs: list[LegendSpec] = []
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target,
            panel_keys=requested_metrics,
            figsize=figsize,
            n_columns=min(len(requested_metrics), 3),
        )
        if kind == "embedding":
            assert reference_layout is not None and projected is not None
            assert color_scale is not None
            combined = np.vstack((reference_layout, projected))
            xlim, ylim = _axis_limits(combined[:, 0], combined[:, 1])
            shared_limits = (
                _continuous_limits(
                    evidence[requested_metrics].to_numpy(dtype=np.float64),
                    color_scale,
                )
                if color_scale.scope == "shared"
                else None
            )
            resolved_point_size = (
                float(point_size)
                if point_size is not None
                else default_point_size(len(projected))
            )
            unknown = evidence["isUnknown"].to_numpy(dtype=bool)
        for metric_index, metric in enumerate(requested_metrics):
            ax = axes[metric]
            if kind == "embedding":
                assert reference_layout is not None and projected is not None
                assert color_scale is not None
                values = evidence[metric].to_numpy(dtype=np.float64)
                limits = shared_limits or _continuous_limits(values, color_scale)
                norm = continuous_norm(
                    mpl,
                    vmin=limits[0],
                    vmax=limits[1],
                    vcenter=color_scale.vcenter,
                )
                ax.scatter(
                    reference_layout[:, 0],
                    reference_layout[:, 1],
                    s=resolved_point_size * 0.55,
                    c="#bdbdbd",
                    alpha=0.2,
                    linewidths=0,
                    rasterized=len(reference_layout) >= 50_000,
                    zorder=0,
                )
                draw_order = np.argsort(
                    np.nan_to_num(values, nan=-np.inf),
                )
                artist = ax.scatter(
                    projected[draw_order, 0],
                    projected[draw_order, 1],
                    c=values[draw_order],
                    s=resolved_point_size,
                    cmap=color_scale.cmap or "viridis",
                    norm=norm,
                    edgecolors="none",
                    rasterized=len(projected) >= 50_000,
                    zorder=1,
                )
                if show_unknown and unknown.any():
                    ax.scatter(
                        projected[unknown, 0],
                        projected[unknown, 1],
                        s=resolved_point_size * 1.6,
                        marker="x",
                        c=scatter_edgecolor(theme),
                        linewidths=0.7,
                        rasterized=len(projected) >= 50_000,
                        zorder=2,
                    )
                finish_embedding_axes(
                    ax,
                    xlim=xlim,
                    ylim=ylim,
                    title=metric,
                    frame="minimal",
                )
                if show_legend:
                    colorbar = figure.colorbar(
                        artist,
                        ax=ax,
                        location="top",
                        orientation="horizontal",
                        shrink=0.8,
                        fraction=0.06,
                        pad=0.03,
                    )
                    colorbar.set_label(metric)
                legend_specs.append(
                    LegendSpec(
                        kind="colorbar",
                        label=metric,
                        extras={"vmin": limits[0], "vmax": limits[1]},
                    )
                )
            elif kind == "histogram":
                all_values = evidence[metric].to_numpy(dtype=np.float64)
                all_values = all_values[np.isfinite(all_values)]
                shared_bins: int | np.ndarray = (
                    np.histogram_bin_edges(all_values, bins=bins)
                    if len(all_values)
                    else bins
                )
                for group in order:
                    values = evidence.loc[
                        evidence["group"] == group,
                        metric,
                    ].to_numpy(dtype=np.float64)
                    ax.hist(
                        values[np.isfinite(values)],
                        bins=shared_bins,
                        histtype="step",
                        linewidth=1.4,
                        color=palette[group],
                        label=str(group),
                    )
                ax.set_ylabel("Mapped cells")
            else:
                grouped_values = [
                    evidence.loc[
                        evidence["group"] == group,
                        metric,
                    ].dropna()
                    for group in order
                ]
                boxes = ax.boxplot(
                    grouped_values,
                    tick_labels=[str(group) for group in order],
                    patch_artist=True,
                    showfliers=False,
                )
                for patch, group in zip(
                    boxes["boxes"],
                    order,
                    strict=True,
                ):
                    patch.set_facecolor(palette[group])
                    patch.set_alpha(0.8)
                ax.tick_params(axis="x", labelrotation=45)
            if kind != "embedding":
                ax.set_xlabel(metric)
            if show_legend and kind == "histogram" and metric_index == 0:
                ax.legend(frameon=False, title="Query group")
        apply_figure_chrome(figure, theme)
    if kind != "embedding":
        legend_specs.append(LegendSpec(kind="categorical", label="Query group"))
    elif show_unknown and evidence["isUnknown"].any():
        legend_specs.append(
            LegendSpec(
                kind="marker",
                label="Transfer status",
                extras={"values": ["Unknown"], "markers": ["x"]},
            )
        )
    if kind == "embedding":
        assert color_scale is not None
        result_scales: tuple[Any, ...] = (color_scale,)
    else:
        assert resolved_categorical is not None
        result_scales = (resolved_categorical,)
    result = PlotResult(
        figure=figure,
        axes=axes,
        tables={"evidence": evidence},
        legends=tuple(legend_specs),
        scales=result_scales,
        provenance=PlotProvenance(
            scarf_version=_scarf_version(),
            assay=from_assay or getattr(store, "_defaultAssay", None),
            cell_key=cell_key,
            n_cells=len(evidence),
            renderer="matplotlib",
            notes=("mapping_evidence", kind),
            extras={
                "target_name": target_name,
                "reference_class_group": reference_class_group,
                "metrics": requested_metrics,
                "reference_layout_key": reference_layout_key,
                "threshold_fraction": threshold_fraction,
                "max_distance": max_distance,
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        result.show()
    return result


def mapping_confusion(
    store: Any,
    *,
    target_name: str,
    reference_class_group: str,
    known_labels: Sequence[Any] | np.ndarray,
    normalize: Literal["none", "true", "predicted", "all"] = "true",
    known_order: Sequence[Any] | None = None,
    predicted_order: Sequence[Any] | None = None,
    from_assay: str | None = None,
    cell_key: str | None = None,
    threshold_fraction: float = 0.5,
    na_val: str = "NA",
    max_distance: float | None = None,
    color_scale: ColorScale | None = None,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot known query labels against transferred labels."""
    if normalize not in ("none", "true", "predicted", "all"):
        raise ValueError("normalize must be 'none', 'true', 'predicted', or 'all'")
    evidence = _label_evidence(
        store,
        target_name=target_name,
        reference_class_group=reference_class_group,
        from_assay=from_assay,
        cell_key=cell_key,
        threshold_fraction=threshold_fraction,
        na_val=na_val,
        max_distance=max_distance,
    )
    known = np.asarray(known_labels, dtype=object)
    if known.ndim != 1 or len(known) != len(evidence):
        raise ValueError("known_labels must have one value per mapped cell")
    valid = pd.notna(known)
    truth = known[valid]
    predicted = evidence.loc[valid, "label"].to_numpy(dtype=object)
    observed_known = sort_categories(list(pd.unique(truth)))
    observed_predicted = sort_categories(list(pd.unique(predicted)))
    rows = list(known_order) if known_order is not None else observed_known
    columns = (
        list(predicted_order) if predicted_order is not None else observed_predicted
    )
    if any(value not in rows for value in observed_known):
        raise ValueError("known_order is missing observed labels")
    if any(value not in columns for value in observed_predicted):
        raise ValueError("predicted_order is missing observed labels")
    counts = pd.crosstab(
        pd.Series(truth, name="known"),
        pd.Series(predicted, name="predicted"),
    ).reindex(index=rows, columns=columns, fill_value=0)
    class_rows = []
    for label in dict.fromkeys([*rows, *columns]):
        true_positive = (
            int(counts.loc[label, label])
            if label in counts.index and label in counts.columns
            else 0
        )
        support = int(counts.loc[label].sum()) if label in counts.index else 0
        predicted_count = int(counts[label].sum()) if label in counts.columns else 0
        class_rows.append(
            {
                "label": label,
                "precision": (
                    true_positive / predicted_count if predicted_count else np.nan
                ),
                "recall": true_positive / support if support else np.nan,
                "support": support,
                "predicted": predicted_count,
            }
        )
    per_class = pd.DataFrame(class_rows)
    display = counts.astype(np.float64)
    if normalize == "true":
        display = display.div(display.sum(axis=1).replace(0, np.nan), axis=0)
    elif normalize == "predicted":
        display = display.div(display.sum(axis=0).replace(0, np.nan), axis=1)
    elif normalize == "all":
        display = display / max(float(display.to_numpy().sum()), 1.0)
    display = display.fillna(0)
    color_scale = color_scale or ColorScale(cmap="Blues", vmin=0)
    limits = _continuous_limits(display.to_numpy(), color_scale)
    _, mpl = require_matplotlib()
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target,
            panel_keys=["mapping_confusion"],
            figsize=figsize or ((5.2, 4.6) if target is None else None),
        )
        ax = axes["mapping_confusion"]
        norm = continuous_norm(
            mpl,
            vmin=limits[0],
            vmax=limits[1],
            vcenter=color_scale.vcenter,
        )
        image = ax.imshow(
            display.to_numpy(),
            cmap=color_scale.cmap or "Blues",
            norm=norm,
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_xticks(range(len(columns)))
        ax.set_xticklabels(columns, rotation=45, ha="right")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Known label")
        midpoint = 0.5 * sum(limits)
        for row_index in range(len(rows)):
            for column_index in range(len(columns)):
                value = float(display.iloc[row_index, column_index])
                label = f"{value:.0f}" if normalize == "none" else f"{value:.0%}"
                ax.text(
                    column_index,
                    row_index,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#ffffff" if value > midpoint else "#222222",
                )
        if show_legend:
            colorbar = figure.colorbar(
                image,
                ax=ax,
                location="top",
                orientation="horizontal",
                shrink=0.75,
                fraction=0.06,
                pad=0.04,
            )
            colorbar.set_label("Cells" if normalize == "none" else "Fraction of cells")
        apply_figure_chrome(figure, theme)
    result = PlotResult(
        figure=figure,
        axes=axes,
        tables={
            "counts": counts.reset_index(),
            "matrix": display.reset_index(),
            "perClass": per_class,
            "evidence": evidence.assign(knownLabel=known),
        },
        legends=(
            LegendSpec(
                kind="colorbar",
                label="Cells" if normalize == "none" else "Fraction of cells",
                extras={"vmin": limits[0], "vmax": limits[1]},
            ),
        ),
        scales=(color_scale,),
        provenance=PlotProvenance(
            scarf_version=_scarf_version(),
            assay=from_assay or getattr(store, "_defaultAssay", None),
            cell_key=cell_key,
            n_cells=int(valid.sum()),
            renderer="matplotlib",
            notes=("mapping_confusion",),
            extras={
                "target_name": target_name,
                "reference_class_group": reference_class_group,
                "normalize": normalize,
                "threshold_fraction": threshold_fraction,
                "dropped_missing_truth": int((~valid).sum()),
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        result.show()
    return result


def mapping_calibration(
    store: Any,
    *,
    target_name: str,
    reference_class_group: str,
    known_labels: Sequence[Any] | np.ndarray,
    metric: str = "voteFraction",
    direction: Literal["auto", "higher", "lower"] = "auto",
    thresholds: Sequence[float] | np.ndarray | None = None,
    n_thresholds: int = 50,
    chosen_threshold: float | None = None,
    from_assay: str | None = None,
    cell_key: str | None = None,
    na_val: str = "NA",
    max_distance: float | None = None,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show: bool = True,
) -> PlotResult:
    """Plot held-out label accuracy against retained mapping coverage."""
    if n_thresholds < 2:
        raise ValueError("n_thresholds must be at least 2")
    if direction not in ("auto", "higher", "lower"):
        raise ValueError("direction must be 'auto', 'higher', or 'lower'")
    if direction == "auto":
        if metric in {"voteFraction", "topTwoMargin"}:
            resolved_direction = "higher"
        elif metric in {
            "voteEntropy",
            "referenceDistancePercentile",
            "meanNeighborDistance",
        }:
            resolved_direction = "lower"
        else:
            raise ValueError(
                f"Cannot infer threshold direction for metric {metric!r}; "
                "pass direction='higher' or direction='lower'"
            )
    else:
        resolved_direction = direction
    evidence = _label_evidence(
        store,
        target_name=target_name,
        reference_class_group=reference_class_group,
        from_assay=from_assay,
        cell_key=cell_key,
        threshold_fraction=0.0,
        na_val=na_val,
        max_distance=max_distance,
    )
    if metric not in evidence:
        raise KeyError(f"Evidence has no threshold metric {metric!r}")
    known = np.asarray(known_labels, dtype=object)
    if known.ndim != 1 or len(known) != len(evidence):
        raise ValueError("known_labels must have one value per mapped cell")
    metric_values = pd.to_numeric(
        evidence[metric],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    predicted = evidence["label"].to_numpy(dtype=object)
    valid = pd.notna(known) & np.isfinite(metric_values)
    values = metric_values[valid]
    correct = predicted[valid] == known[valid]
    correct_all = np.zeros(len(evidence), dtype=bool)
    correct_all[valid] = correct
    informative = ~evidence.loc[valid, "isUnknown"].to_numpy(dtype=bool)
    if len(values) == 0:
        raise ValueError("No finite metric values with known labels")
    textual_matches = np.fromiter(
        (
            str(known_value) == str(predicted_value)
            for known_value, predicted_value in zip(
                known[valid],
                predicted[valid],
                strict=True,
            )
        ),
        dtype=bool,
        count=len(values),
    )
    if np.any(textual_matches & ~correct):
        raise ValueError(
            "Some known_labels only equal their transferred labels after text "
            "conversion. Convert known_labels to the value type stored "
            f"in {reference_class_group!r}"
        )
    if thresholds is None:
        resolved_thresholds = np.unique(
            np.quantile(values, np.linspace(0, 1, n_thresholds))
        )
    else:
        resolved_thresholds = np.unique(np.asarray(thresholds, dtype=np.float64))
        if (
            resolved_thresholds.ndim != 1
            or len(resolved_thresholds) == 0
            or not np.isfinite(resolved_thresholds).all()
        ):
            raise ValueError("thresholds must contain finite numeric values")
    if chosen_threshold is not None:
        if not np.isfinite(chosen_threshold):
            raise ValueError("chosen_threshold must be finite")
        resolved_thresholds = np.unique(
            np.append(resolved_thresholds, chosen_threshold)
        )
    rows: list[dict[str, float | int]] = []
    z_value = 1.959963984540054
    for threshold in resolved_thresholds:
        passes = (
            values >= threshold
            if resolved_direction == "higher"
            else values <= threshold
        )
        accepted = informative & passes
        n_accepted = int(accepted.sum())
        if n_accepted == 0:
            continue
        accuracy = float(np.mean(correct[accepted]))
        denominator = 1 + z_value**2 / n_accepted
        center = (accuracy + z_value**2 / (2 * n_accepted)) / denominator
        margin = (
            z_value
            * np.sqrt(
                accuracy * (1 - accuracy) / n_accepted
                + z_value**2 / (4 * n_accepted**2)
            )
            / denominator
        )
        rows.append(
            {
                "threshold": float(threshold),
                "coverage": n_accepted / len(values),
                "accuracy": accuracy,
                "accuracyLower": max(0.0, center - margin),
                "accuracyUpper": min(1.0, center + margin),
                "nAccepted": n_accepted,
                "nEvaluated": len(values),
            }
        )
    calibration = pd.DataFrame(rows)
    if calibration.empty:
        raise ValueError("No threshold retained any mapped cells")
    calibration = calibration.sort_values(
        ["coverage", "threshold"],
        kind="stable",
    ).reset_index(drop=True)
    if (
        chosen_threshold is not None
        and not np.isclose(
            calibration["threshold"],
            chosen_threshold,
        ).any()
    ):
        warnings.warn(
            f"chosen_threshold={chosen_threshold:g} retained no mapped cells; "
            "the threshold marker was omitted",
            RuntimeWarning,
            stacklevel=2,
        )
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target,
            panel_keys=["mapping_calibration"],
            figsize=figsize or ((4.5, 4.0) if target is None else None),
        )
        ax = axes["mapping_calibration"]
        coverage = calibration["coverage"].to_numpy(dtype=np.float64)
        accuracy = calibration["accuracy"].to_numpy(dtype=np.float64)
        lower = calibration["accuracyLower"].to_numpy(dtype=np.float64)
        upper = calibration["accuracyUpper"].to_numpy(dtype=np.float64)
        ax.fill_between(
            coverage,
            lower,
            upper,
            color="#2b6cb0",
            alpha=0.16,
            linewidth=0,
        )
        ax.plot(
            coverage,
            accuracy,
            color="#2b6cb0",
            linewidth=1.4,
        )
        if chosen_threshold is not None:
            chosen = calibration.loc[
                np.isclose(calibration["threshold"], chosen_threshold)
            ]
            if not chosen.empty:
                selected = chosen.iloc[0]
                ax.scatter(
                    [selected["coverage"]],
                    [selected["accuracy"]],
                    s=48,
                    color="#d1495b",
                    edgecolor=scatter_edgecolor(theme),
                    linewidth=0.5,
                    zorder=3,
                )
                ax.annotate(
                    f"{metric} = {chosen_threshold:g}",
                    (
                        float(selected["coverage"]),
                        float(selected["accuracy"]),
                    ),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )
        # Pad the unit square so curves sitting at zero or one stay visible.
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Retained coverage")
        ax.set_ylabel("Held-out label accuracy")
        ax.set_title(f"Threshold trade-off: {metric}")
        apply_figure_chrome(figure, theme)
    result = PlotResult(
        figure=figure,
        axes=axes,
        tables={
            "calibration": calibration,
            "evidence": evidence.assign(
                knownLabel=known,
                correct=correct_all,
            ),
        },
        legends=(),
        scales=(),
        provenance=PlotProvenance(
            scarf_version=_scarf_version(),
            assay=from_assay or getattr(store, "_defaultAssay", None),
            cell_key=cell_key,
            n_cells=int(valid.sum()),
            renderer="matplotlib",
            notes=("mapping_calibration",),
            extras={
                "target_name": target_name,
                "reference_class_group": reference_class_group,
                "metric": metric,
                "direction": resolved_direction,
                "n_thresholds": len(resolved_thresholds),
                "chosen_threshold": chosen_threshold,
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        result.show()
    return result


def mapping_correction(
    store: Any,
    *,
    target_name: str,
    batch_labels: Sequence[Any] | np.ndarray | None = None,
    dimensions: tuple[int, int] = (0, 1),
    from_assay: str | None = None,
    cell_key: str | None = None,
    categorical_scale: CategoricalScale | None = None,
    point_size: float | None = None,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Compare query latent coordinates before and after mapping correction."""
    result_data = _mapping_result(
        store,
        target_name=target_name,
        from_assay=from_assay,
        cell_key=cell_key,
    )
    before = result_data.uncorrected_latent
    after = result_data.corrected_latent
    if before is None or after is None:
        raise ValueError(
            f"Mapping {target_name!r} has no saved before and after latent arrays"
        )
    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    if before.shape != after.shape or before.ndim != 2:
        raise ValueError("Correction latent arrays have incompatible shapes")
    first, second = dimensions
    if first == second or min(dimensions) < 0 or max(dimensions) >= before.shape[1]:
        raise ValueError("dimensions must select two distinct latent dimensions")
    groups = _external_groups(
        batch_labels,
        len(before),
        default="query",
        argument_name="batch_labels",
    )
    order, palette, resolved_scale = _categorical_contract(
        groups,
        categorical_scale,
    )
    panel_keys: list[Hashable] = ["before", "after", "displacement"]
    if figsize is None and target is None:
        figsize = (10.0, 3.2)
    combined_x = np.concatenate((before[:, first], after[:, first]))
    combined_y = np.concatenate((before[:, second], after[:, second]))
    xlim, ylim = _axis_limits(combined_x, combined_y)
    displacement = np.linalg.norm(after - before, axis=1)
    size = (
        float(point_size) if point_size is not None else default_point_size(len(before))
    )
    edgewidth = default_point_edgewidth(len(before), point_size=size)
    edgecolor = scatter_edgecolor(theme)
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target,
            panel_keys=panel_keys,
            figsize=figsize,
            n_columns=3,
        )
        for panel, coordinates in (("before", before), ("after", after)):
            ax = axes[panel]
            colors = [palette[value] for value in groups]
            ax.scatter(
                coordinates[:, first],
                coordinates[:, second],
                c=colors,
                s=size,
                edgecolors=edgecolor if edgewidth > 0 else "none",
                linewidths=edgewidth,
                rasterized=len(before) >= 50_000,
            )
            finish_embedding_axes(
                ax,
                xlim=xlim,
                ylim=ylim,
                xlabel=f"latent {first + 1}",
                ylabel=f"latent {second + 1}",
                title=panel.capitalize(),
                frame="axes",
            )
        displacement_ax = axes["displacement"]
        for group in order:
            values = displacement[groups == group]
            displacement_ax.hist(
                values,
                bins=30,
                histtype="step",
                linewidth=1.4,
                color=palette[group],
                label=str(group),
            )
        displacement_ax.set_xlabel("Correction displacement")
        displacement_ax.set_ylabel("Mapped cells")
        if show_legend:
            displacement_ax.legend(frameon=False, title="Query batch")
        apply_figure_chrome(figure, theme)
    table = pd.DataFrame(
        {
            "cellIndex": np.arange(len(before)),
            "batch": groups,
            "beforeX": before[:, first],
            "beforeY": before[:, second],
            "afterX": after[:, first],
            "afterY": after[:, second],
            "displacement": displacement,
        }
    )
    result = PlotResult(
        figure=figure,
        axes=axes,
        tables={"cells": table},
        legends=(LegendSpec(kind="categorical", label="Query batch"),),
        scales=(resolved_scale,),
        provenance=PlotProvenance(
            scarf_version=_scarf_version(),
            assay=from_assay or getattr(store, "_defaultAssay", None),
            cell_key=cell_key,
            n_cells=len(before),
            renderer="matplotlib",
            notes=("mapping_correction", result_data.correction_method),
            extras={
                "target_name": target_name,
                "dimensions": list(dimensions),
                "mean_displacement": float(np.mean(displacement)),
                "median_displacement": float(np.median(displacement)),
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        result.show()
    return result


def mapping_projection(
    store: Any,
    *,
    target_name: str,
    reference_layout_key: str,
    reference_groups: str | Sequence[Any] | np.ndarray | None = None,
    target_groups: Sequence[Any] | np.ndarray | None = None,
    reference_class_group: str | None = None,
    threshold_fraction: float = 0.5,
    na_val: str = "NA",
    from_assay: str | None = None,
    cell_key: str | None = None,
    ref_name: str = "reference",
    categorical_scale: CategoricalScale | None = None,
    point_size: float | None = None,
    reference_alpha: float = 0.35,
    target_alpha: float = 0.9,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot query cells projected into an unchanged reference layout."""
    if not 0 <= reference_alpha <= 1 or not 0 <= target_alpha <= 1:
        raise ValueError("point alpha values must be between zero and one")
    resolved_assay, resolved_cell_key = _resolved_mapping_keys(
        store,
        from_assay,
        cell_key,
    )
    result_data = _mapping_result(
        store,
        target_name=target_name,
        from_assay=resolved_assay,
        cell_key=resolved_cell_key,
    )
    reference_layout, projected = _projected_mapping_coordinates(
        store,
        result_data,
        reference_layout_key=reference_layout_key,
        cell_key=resolved_cell_key,
    )
    reference_x = reference_layout[:, 0]
    reference_y = reference_layout[:, 1]
    if reference_groups is None and reference_class_group is None:
        reference_values = np.full(len(reference_layout), ref_name, dtype=object)
        query_values = _external_groups(
            target_groups,
            len(projected),
            default=target_name,
            argument_name="target_groups",
        )
    else:
        group_source = (
            reference_class_group
            if reference_class_group is not None
            else reference_groups
        )
        if isinstance(group_source, str):
            reference_values = np.asarray(
                store.cells.fetch(group_source, key=resolved_cell_key),
                dtype=object,
            )
            if target_groups is None:
                evidence = _label_evidence(
                    store,
                    target_name=target_name,
                    reference_class_group=group_source,
                    from_assay=resolved_assay,
                    cell_key=resolved_cell_key,
                    threshold_fraction=threshold_fraction,
                    na_val=na_val,
                    max_distance=None,
                )
                query_values = evidence["label"].to_numpy(dtype=object)
            else:
                query_values = _external_groups(
                    target_groups,
                    len(projected),
                    default=target_name,
                    argument_name="target_groups",
                )
        else:
            reference_values = np.asarray(group_source, dtype=object)
            if reference_values.shape != (len(reference_layout),):
                raise ValueError(
                    "reference_groups must have one value per reference cell"
                )
            query_values = _external_groups(
                target_groups,
                len(projected),
                default=target_name,
                argument_name="target_groups",
            )
    missing_label = (
        categorical_scale.missing_label if categorical_scale is not None else "NA"
    )
    reference_values = np.asarray(
        [missing_label if pd.isna(value) else value for value in reference_values],
        dtype=object,
    )
    query_values = np.asarray(
        [missing_label if pd.isna(value) else value for value in query_values],
        dtype=object,
    )
    all_groups = np.concatenate((reference_values, query_values))
    order, palette, resolved_scale = _categorical_contract(
        all_groups,
        categorical_scale,
    )
    combined_x = np.concatenate((reference_x, projected[:, 0]))
    combined_y = np.concatenate((reference_y, projected[:, 1]))
    xlim, ylim = _axis_limits(combined_x, combined_y)
    size = (
        float(point_size)
        if point_size is not None
        else default_point_size(len(all_groups))
    )
    edgecolor = scatter_edgecolor(theme)
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target,
            panel_keys=["mapping_projection"],
            figsize=figsize or ((5.0, 4.4) if target is None else None),
        )
        ax = axes["mapping_projection"]
        ax.scatter(
            reference_x,
            reference_y,
            c=[palette[value] for value in reference_values],
            s=size,
            alpha=reference_alpha,
            edgecolors="none",
            rasterized=len(reference_x) >= 50_000,
            zorder=0,
        )
        ax.scatter(
            projected[:, 0],
            projected[:, 1],
            c=[palette[value] for value in query_values],
            s=size * 1.25,
            alpha=target_alpha,
            edgecolors=edgecolor,
            linewidths=0.25,
            rasterized=len(projected) >= 50_000,
            zorder=1,
        )
        finish_embedding_axes(
            ax,
            xlim=xlim,
            ylim=ylim,
            title=f"{target_name} on {reference_layout_key}",
            frame="minimal",
        )
        if show_legend:
            handles = [
                require_matplotlib()[1].lines.Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="",
                    markerfacecolor=palette[value],
                    markeredgecolor="none",
                    markersize=5,
                    label=(
                        resolved_scale.labels.get(value, str(value))
                        if resolved_scale.labels is not None
                        else str(value)
                    ),
                )
                for value in order
            ]
            if owns:
                figure.legend(
                    handles=handles,
                    frameon=False,
                    loc="outside right center",
                    title="Cells",
                )
            else:
                ax.legend(
                    handles=handles,
                    frameon=False,
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1),
                    title="Cells",
                )
        apply_figure_chrome(figure, theme)
    cells = pd.DataFrame(
        {
            "source": np.concatenate(
                (
                    np.full(len(reference_layout), "reference", dtype=object),
                    np.full(len(projected), "query", dtype=object),
                )
            ),
            "group": all_groups,
            "x": combined_x,
            "y": combined_y,
        }
    )
    result = PlotResult(
        figure=figure,
        axes=axes,
        tables={"cells": cells},
        legends=(LegendSpec(kind="categorical", label="Cells"),),
        scales=(resolved_scale,),
        provenance=PlotProvenance(
            scarf_version=_scarf_version(),
            assay=resolved_assay,
            cell_key=resolved_cell_key,
            n_cells=len(all_groups),
            renderer="matplotlib",
            notes=("mapping_projection",),
            extras={
                "target_name": target_name,
                "reference_layout_key": reference_layout_key,
                "n_reference": len(reference_layout),
                "n_query": len(projected),
                "reference_class_group": reference_class_group,
                "threshold_fraction": threshold_fraction,
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        result.show()
    return result
