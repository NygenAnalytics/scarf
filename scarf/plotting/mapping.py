"""Diagnostics for reference mapping and label transfer."""

from collections.abc import Sequence
from typing import Any, Hashable, Literal
import warnings

import numpy as np
import pandas as pd

from ..mapping.models import MappingResult
from ..mapping.reference import MappingReference
from ..storage.refs import ArtifactRef
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
    default_point_size,
    finish_embedding_axes,
    scatter_edgecolor,
    sort_categories,
    square_axis_limits,
    theme_context,
)


def _mapping_result(
    store: Any,
    result: ArtifactRef,
    *,
    reference: MappingReference,
) -> MappingResult:
    loader = getattr(store, "get_mapping_result", None)
    if not callable(loader):
        raise TypeError("store does not provide mapping result data")
    loaded = loader(
        result,
        reference=reference,
        load_arrays=False,
    )
    if not isinstance(loaded, MappingResult):
        raise TypeError("store returned an invalid mapping result")
    return loaded


def _reference_layout(
    reference: MappingReference,
    layout: ArtifactRef,
) -> np.ndarray:
    values = np.asarray(reference.fetch_layout(layout), dtype=np.float64)
    if values.shape != (reference.selected_cell_count, 2):
        raise ValueError(
            "Reference layout must have two columns and one row per selected "
            "reference cell"
        )
    if not np.all(np.isfinite(values) | np.isnan(values)):
        raise ValueError("Reference layout contains infinite coordinates")
    return values


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
    if x.shape != y.shape:
        raise ValueError("Coordinate columns must have matching shapes")
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        raise ValueError("No finite coordinates are available to plot")
    finite_x = x[finite]
    finite_y = y[finite]
    xpad = 0.05 * (float(np.ptp(finite_x)) or 1.0)
    ypad = 0.05 * (float(np.ptp(finite_y)) or 1.0)
    return square_axis_limits(
        (float(np.min(finite_x) - xpad), float(np.max(finite_x) + xpad)),
        (float(np.min(finite_y) - ypad), float(np.max(finite_y) + ypad)),
    )


def mapping_score(
    store: Any,
    result: ArtifactRef,
    *,
    reference: MappingReference,
    target_groups: Sequence[Any] | np.ndarray | None = None,
    layout: ArtifactRef | None = None,
    kind: Literal["embedding", "histogram", "box"] = "embedding",
    reference_class_group: str | None = None,
    size_by_score: bool = False,
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
    if kind not in ("embedding", "histogram", "box"):
        raise ValueError("kind must be 'embedding', 'histogram', or 'box'")
    if kind == "embedding" and layout is None:
        raise ValueError("layout is required for an embedding mapping score")
    if kind == "box" and reference_class_group is None:
        raise ValueError("reference_class_group is required for a box mapping score")
    if size_by_score and kind != "embedding":
        raise ValueError("size_by_score is only supported for kind='embedding'")
    if bins < 1:
        raise ValueError("bins must be positive")
    mapping = _mapping_result(
        store,
        result,
        reference=reference,
    )
    score_loader = getattr(store, "get_mapping_score", None)
    if not callable(score_loader):
        raise TypeError("store does not provide mapping score data")
    score_rows = list(
        score_loader(
            result,
            target_groups=(
                None if target_groups is None else np.asarray(target_groups)
            ),
            reference=reference,
            log_transform=log_transform,
            multiplier=multiplier,
            weighted=weighted,
            fixed_weight=fixed_weight,
        )
    )
    if not score_rows:
        raise ValueError("Mapping produced no score groups")
    labels = [row[0] for row in score_rows]
    display_labels = {label: str(label) for label in labels}
    if target_groups is None and len(labels) == 1:
        display_labels[labels[0]] = "all query cells"
    score_arrays = [np.asarray(row[1], dtype=np.float64) for row in score_rows]
    n_reference = len(score_arrays[0])
    if any(values.shape != (n_reference,) for values in score_arrays):
        raise ValueError("Mapping score groups have incompatible lengths")
    if n_reference != mapping.reference.selected_cell_count:
        raise ValueError("Mapping scores do not match the selected reference cells")
    reference_classes: np.ndarray | None = None
    if reference_class_group is not None:
        if not isinstance(reference_class_group, str) or not reference_class_group:
            raise TypeError("reference_class_group must be a non-empty string")
        reference_classes = np.asarray(
            mapping.reference.fetch_cell_column(reference_class_group),
            dtype=object,
        )
        if reference_classes.shape != (n_reference,):
            raise ValueError(
                "Reference class labels must contain one value per reference cell"
            )
    score_frames = []
    for label, values in zip(labels, score_arrays, strict=True):
        frame = {
            "group": label,
            "referenceIndex": np.arange(n_reference),
            "score": values,
        }
        if reference_classes is not None:
            frame["referenceClass"] = reference_classes
        score_frames.append(pd.DataFrame(frame))
    score_table = pd.concat(score_frames, ignore_index=True)
    plt, mpl = require_matplotlib()
    # Sparse mapping scores are mostly zero. Clip the color ceiling to the upper
    # tail so the few reference cells that received query weight remain visible.
    color_scale = color_scale or ColorScale(
        cmap="magma",
        vmin=0.0,
        quantiles=(0.0, 0.995),
    )
    panel_keys: list[Hashable] = (
        list(labels) if kind in {"embedding", "box"} else ["mapping_score"]
    )
    if figsize is None and target is None:
        if kind == "embedding":
            figsize = (3.6 * len(panel_keys), 3.4)
        elif kind == "box":
            figsize = (4.2 * len(panel_keys), 3.6)
        else:
            figsize = (5.0, 3.6)
    legends: list[LegendSpec] = []
    scales: list[Any] = []
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target,
            panel_keys=panel_keys,
            figsize=figsize,
        )
        if kind == "embedding":
            assert layout is not None
            layout_values = _reference_layout(mapping.reference, layout)
            x = layout_values[:, 0]
            y = layout_values[:, 1]
            xlim, ylim = _axis_limits(x, y)
            shared_values = np.concatenate(
                [values[np.isfinite(values) & (values > 0)] for values in score_arrays]
            )
            if shared_values.size == 0:
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
            for label, values in zip(labels, score_arrays, strict=True):
                ax = axes[label]
                positive = values[np.isfinite(values) & (values > 0)]
                limits = shared_limits or _continuous_limits(
                    positive if positive.size else values,
                    color_scale,
                )
                norm = continuous_norm(
                    mpl,
                    vmin=limits[0],
                    vmax=limits[1],
                    vcenter=color_scale.vcenter,
                )
                visible = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
                # Draw the full reference cloud first so zero-score cells stay
                # readable as context, then overlay the cells that received weight.
                ax.scatter(
                    x[visible],
                    y[visible],
                    c=color_scale.missing_color,
                    s=resolved_point_size * (0.45 if size_by_score else 1.0),
                    edgecolors="none",
                    rasterized=n_reference >= 50_000,
                    zorder=0,
                )
                highlighted = visible & (values > 0)
                order_index = np.flatnonzero(highlighted)
                order_index = order_index[np.argsort(values[order_index])]
                if size_by_score:
                    score_span = max(limits[1] - limits[0], 1e-12)
                    sizes: float | np.ndarray = resolved_point_size * (
                        0.8
                        + 1.6
                        * np.clip(
                            (values[order_index] - limits[0]) / score_span,
                            0.0,
                            1.0,
                        )
                    )
                else:
                    sizes = resolved_point_size * 1.35
                artist = ax.scatter(
                    x[order_index],
                    y[order_index],
                    c=values[order_index],
                    s=sizes,
                    cmap=color_scale.cmap or "magma",
                    norm=norm,
                    edgecolors="none",
                    rasterized=n_reference >= 50_000,
                    zorder=1,
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
        elif kind == "box":
            assert reference_classes is not None
            class_order, palette, resolved_categorical = _categorical_contract(
                reference_classes,
                categorical_scale,
            )
            for label, values in zip(labels, score_arrays, strict=True):
                ax = axes[label]
                grouped_values = [
                    values[reference_classes == class_label]
                    for class_label in class_order
                ]
                boxes = ax.boxplot(
                    grouped_values,
                    tick_labels=[str(class_label) for class_label in class_order],
                    patch_artist=True,
                    showfliers=False,
                )
                for patch, class_label in zip(
                    boxes["boxes"],
                    class_order,
                    strict=True,
                ):
                    patch.set_facecolor(palette[class_label])
                    patch.set_alpha(0.8)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
                ax.set_ylabel("Mapping score")
                ax.set_title(display_labels[label])
            legends.append(LegendSpec(kind="categorical", label="Reference class"))
            scales.append(resolved_categorical)
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
    plot_result = PlotResult(
        figure=figure,
        axes=axes,
        tables={"scores": score_table},
        legends=tuple(legends),
        scales=tuple(scales),
        provenance=PlotProvenance(
            assay=mapping.ref.assay,
            cell_key=None,
            n_cells=n_reference,
            renderer="matplotlib",
            notes=("mapping_score", kind),
            extras={
                "layout": layout.to_dict() if layout is not None else None,
                "groups": list(labels),
                "reference_class_group": reference_class_group,
                "size_by_score": size_by_score,
                "log_transform": log_transform,
                "multiplier": multiplier,
                "weighted": weighted,
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        plot_result.show()
    return plot_result


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
    result: ArtifactRef,
    *,
    reference: MappingReference,
    reference_class_group: str,
    threshold_fraction: float,
    na_val: str,
    max_distance: float | None,
) -> pd.DataFrame:
    loader = getattr(store, "get_target_label_evidence", None)
    if not callable(loader):
        raise TypeError("store does not provide mapping evidence data")
    return loader(
        result,
        reference_class_group=reference_class_group,
        reference=reference,
        threshold_fraction=threshold_fraction,
        na_val=na_val,
        max_distance=max_distance,
    ).copy()


def mapping_evidence(
    store: Any,
    result: ArtifactRef,
    *,
    reference: MappingReference,
    reference_class_group: str,
    target_groups: Sequence[Any] | np.ndarray | None = None,
    metrics: Sequence[str] = (
        "voteFraction",
        "topTwoMargin",
        "voteEntropy",
        "referenceDistancePercentile",
    ),
    kind: Literal["histogram", "box"] = "histogram",
    bins: int = 30,
    threshold_fraction: float = 0.5,
    na_val: str = "NA",
    max_distance: float | None = None,
    categorical_scale: CategoricalScale | None = None,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot query-level label-transfer evidence."""
    if kind not in ("histogram", "box"):
        raise ValueError("kind must be 'histogram' or 'box'")
    if bins < 1:
        raise ValueError("bins must be positive")
    mapping = _mapping_result(
        store,
        result,
        reference=reference,
    )
    evidence = _label_evidence(
        store,
        result,
        reference=reference,
        reference_class_group=reference_class_group,
        threshold_fraction=threshold_fraction,
        na_val=na_val,
        max_distance=max_distance,
    )
    groups = _external_groups(
        target_groups,
        len(evidence),
        default="all query cells",
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
    order, palette, resolved_categorical = _categorical_contract(
        groups,
        categorical_scale,
    )
    if figsize is None and target is None:
        figsize = (4.0 * min(len(requested_metrics), 3), 3.3)
    plt, _ = require_matplotlib()
    legend_specs: list[LegendSpec] = []
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target,
            panel_keys=requested_metrics,
            figsize=figsize,
            n_columns=min(len(requested_metrics), 3),
        )
        for metric_index, metric in enumerate(requested_metrics):
            ax = axes[metric]
            if kind == "histogram":
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
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            ax.set_xlabel(metric)
            if show_legend and kind == "histogram" and metric_index == 0:
                ax.legend(frameon=False, title="Query group")
        apply_figure_chrome(figure, theme)
    legend_specs.append(LegendSpec(kind="categorical", label="Query group"))
    assert resolved_categorical is not None
    plot_result = PlotResult(
        figure=figure,
        axes=axes,
        tables={"evidence": evidence},
        legends=tuple(legend_specs),
        scales=(resolved_categorical,),
        provenance=PlotProvenance(
            assay=mapping.ref.assay,
            cell_key=None,
            n_cells=len(evidence),
            renderer="matplotlib",
            notes=("mapping_evidence", kind),
            extras={
                "reference_class_group": reference_class_group,
                "metrics": requested_metrics,
                "threshold_fraction": threshold_fraction,
                "max_distance": max_distance,
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        plot_result.show()
    return plot_result


def mapping_confusion(
    store: Any,
    result: ArtifactRef,
    *,
    reference: MappingReference,
    reference_class_group: str,
    known_labels: Sequence[Any] | np.ndarray,
    normalize: Literal["none", "true", "predicted", "all"] = "true",
    known_order: Sequence[Any] | None = None,
    predicted_order: Sequence[Any] | None = None,
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
    mapping = _mapping_result(
        store,
        result,
        reference=reference,
    )
    evidence = _label_evidence(
        store,
        result,
        reference=reference,
        reference_class_group=reference_class_group,
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
    plot_result = PlotResult(
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
            assay=mapping.ref.assay,
            cell_key=None,
            n_cells=int(valid.sum()),
            renderer="matplotlib",
            notes=("mapping_confusion",),
            extras={
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
        plot_result.show()
    return plot_result


def mapping_calibration(
    store: Any,
    result: ArtifactRef,
    *,
    reference: MappingReference,
    reference_class_group: str,
    known_labels: Sequence[Any] | np.ndarray,
    metric: str = "voteFraction",
    direction: Literal["auto", "higher", "lower"] = "auto",
    thresholds: Sequence[float] | np.ndarray | None = None,
    n_thresholds: int = 50,
    chosen_threshold: float | None = None,
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
    mapping = _mapping_result(
        store,
        result,
        reference=reference,
    )
    evidence = _label_evidence(
        store,
        result,
        reference=reference,
        reference_class_group=reference_class_group,
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
    plot_result = PlotResult(
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
            assay=mapping.ref.assay,
            cell_key=None,
            n_cells=int(valid.sum()),
            renderer="matplotlib",
            notes=("mapping_calibration",),
            extras={
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
        plot_result.show()
    return plot_result
