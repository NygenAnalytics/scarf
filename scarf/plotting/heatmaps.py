"""Native heatmap and cluster-tree plotting."""

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pandas as pd

from ..assay.normalization import resolve_normalization_params
from ..features.markers.table import load_marker_table
from ..metadata.membership import measured_rows
from ..metadata.selection import grouping_value_name
from ..storage.artifacts import ArtifactRef, artifact_path
from ..storage.selections import read_stored_selection_indices
from ..storage.types import as_zarr_array, as_zarr_group
from ._contracts import CategoricalScale, ColorScale, PlotProvenance
from ._data import _artifact_input, unmeasured_extras
from ._deps import require_matplotlib, require_seaborn
from ._figure import (
    LegendSpec,
    PlotResult,
    close_figures_on_error,
    normalize_axes_target,
)
from ._heatmap_utils import (
    annotation_colors,
    annotation_legend_handles,
    draw_annotation_strips,
    normalize_annotations,
    order_heatmap,
    validate_linkage,
)
from ..utils.arguments import integer_argument
from ..utils.arrays import sort_categories
from ._style import (
    apply_figure_chrome,
    colormap_palette,
    continuous_norm,
    resolve_category_scale,
    resolve_color_limits,
    theme_context,
)


def _marker_log_transform(assay: Any, value: bool | None) -> bool:
    """Resolve ``log_transform`` for the assay's configured normalizer.

    None logs the values of ``norm_lib_size``, whose scale Scarf knows, and
    leaves every other normalizer's values as they are. True also logs the
    values of ``norm_dummy`` and of custom normalizers, and raises
    ``ValueError`` for a normalizer that cannot be logged.
    """
    if value is not None and not isinstance(value, (bool, np.bool_)):
        raise TypeError("log_transform must be a boolean or None")
    return resolve_normalization_params(
        assay,
        {"log_transform": value},
        caller="marker_heatmap",
        default=True,
    )["log_transform"]


def _place_clustermap_annotation_legend(
    figure: Any,
    axes: Mapping[Hashable, Any],
    handles: Sequence[Any],
    *,
    has_column_dendrogram: bool,
) -> None:
    header = axes["column_dendrogram"].get_position()
    longest_label = max(len(str(handle.get_label())) for handle in handles)
    estimated_column_width = 0.45 + 0.09 * longest_label
    available_width = max(0.5, (1.0 - header.x0) * figure.get_figwidth() - 0.15)
    n_columns = max(
        1,
        min(
            4,
            len(handles),
            int(available_width // estimated_column_width),
        ),
    )
    if has_column_dendrogram:
        anchor = (header.x0, 0.99)
    else:
        anchor = (header.x0, header.y1)
    legend = figure.legend(
        handles=handles,
        title="Annotations",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=anchor,
        bbox_transform=figure.transFigure,
        ncols=n_columns,
    )
    if has_column_dendrogram:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        legend_bottom = (
            legend.get_window_extent(renderer)
            .transformed(figure.transFigure.inverted())
            .y0
        )
        top = max(axis.get_position().y1 for axis in dict.fromkeys(axes.values()))
        scale = min(1.0, (legend_bottom - 0.01) / top)
        if scale < 1.0:
            for axis in dict.fromkeys(axes.values()):
                position = axis.get_position()
                axis.set_position(
                    [
                        position.x0,
                        position.y0 * scale,
                        position.width,
                        position.height * scale,
                    ],
                    which="both",
                )


def _writable_float64(values: np.ndarray) -> np.ndarray:
    """Copy values so later in-place adds do not write through a view."""
    return np.array(values, dtype=np.float64, copy=True)


def _marker_row_labels(
    marker_slot: Any,
    feature_names: np.ndarray,
    feature_index: np.ndarray,
) -> dict[int, str]:
    """Label each selected feature by name, and a repeated name also by ID.

    Feature names can repeat across features of one assay, but heatmap rows
    must stay distinct, so every feature whose name repeats among the
    selected ones is shown as ``name (ID)`` with the ID the marker search
    recorded.
    """
    names = [str(feature_names[int(index)]) for index in feature_index]
    counts = pd.Series(names).value_counts()
    repeated = set(counts.index[counts > 1])
    if not repeated:
        return dict(zip((int(index) for index in feature_index), names, strict=True))
    feature_ids = np.asarray(marker_slot["feature_ids"][:]).astype(str)
    return {
        int(index): (
            f"{name} ({feature_ids[int(index)]})" if name in repeated else name
        )
        for index, name in zip(feature_index, names, strict=True)
    }


def _prepare_marker_heatmap(
    store: Any,
    *,
    marker: ArtifactRef,
    topn: int,
    log_transform: bool | None,
) -> dict[str, Any]:
    if not isinstance(marker, ArtifactRef):
        raise TypeError("marker must be an ArtifactRef")
    # A negative count would drop markers from the end of every group, as
    # DataFrame.head does. Zero selects no markers and fails below.
    topn = integer_argument(topn, "topn", minimum=0)
    assay, marker_slot = store._resolve_marker_group(marker)
    selection_ref = _artifact_input(store, marker, "cell_selection")
    clusters_ref = _artifact_input(store, marker, "clusters")
    cell_index = read_stored_selection_indices(
        store.zw,
        selection_ref,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    cluster_group = as_zarr_group(
        store.zw[artifact_path(clusters_ref)],
        name=clusters_ref.artifact_id,
    )
    cluster_value_name = grouping_value_name(clusters_ref.kind)
    groups = np.asarray(
        as_zarr_array(
            cluster_group[cluster_value_name],
            name=cluster_value_name,
        )[:]
    )
    if groups.shape != (len(cell_index),):
        raise ValueError("Marker clusters do not align with their cell selection")

    feature_indices: list[int] = []
    marker_rows: list[dict[str, Any]] = []
    feature_names = np.asarray(marker_slot["feature_names"][:]).astype(str)
    for group_name in marker_slot.group_keys():
        marker_group = as_zarr_group(marker_slot[group_name], name=group_name)
        markers = load_marker_table(
            marker_slot,
            marker_group,
            feature_names,
            group_id=group_name,
        )
        ranked = markers.sort_values(
            ["score", "feature_name"],
            ascending=[False, True],
            kind="mergesort",
        ).head(topn)
        selected = ranked["feature_index"].to_numpy(dtype=int)
        feature_indices.extend(selected.tolist())
        marker_rows.extend(
            {
                "group": group_name,
                "rank": rank,
                "feature_index": int(feature_index),
                "score": float(score),
            }
            for rank, (feature_index, score) in enumerate(
                zip(selected, ranked["score"].to_numpy(dtype=float), strict=True),
                start=1,
            )
        )

    if not feature_indices:
        raise ValueError("ERROR: Marker list is empty for all the groups")
    feature_index = np.asarray(sorted(set(feature_indices)), dtype=int)
    resolved_log_transform = _marker_log_transform(assay, log_transform)
    # A cell that the assay did not measure, such as one of a marker search
    # that an earlier release ran over a merged store, has no value: it is
    # left out of the normalization and of its group's mean, as display reads
    # leave it out. Membership comes from the assay's own cell table.
    measured = measured_rows(assay.cells, assay.name, cell_index)
    unmeasured: dict[str, int] = {}
    if measured is not None:
        unmeasured[assay.name] = int(np.count_nonzero(~measured))
        cell_index = cell_index[measured]
        groups = groups[measured]
    normalized = assay.normed(
        cell_idx=cell_index,
        feat_idx=feature_index,
        log_transform=resolved_log_transform,
    )
    group_sums: dict[Any, np.ndarray] = {}
    group_counts: dict[Any, int] = {}
    row_start = 0
    for values in normalized.stream_blocks(
        nthreads=store.nthreads,
        msg="Aggregating marker values per group",
    ):
        block_groups = groups[row_start : row_start + values.shape[0]]
        row_start += values.shape[0]
        block_frame = pd.DataFrame(values)
        block_frame["__group__"] = block_groups
        grouped = block_frame.groupby("__group__")
        block_sum = grouped.sum()
        block_count = grouped.size()
        for label in block_sum.index:
            summed = _writable_float64(block_sum.loc[label].to_numpy())
            if label not in group_sums:
                group_sums[label] = summed
                group_counts[label] = int(block_count.loc[label])
            else:
                group_sums[label] += summed
                group_counts[label] += int(block_count.loc[label])

    labels = sort_categories(list(group_sums))
    group_means = pd.DataFrame(
        np.vstack([group_sums[label] / group_counts[label] for label in labels]),
        index=labels,
    )
    group_means = group_means.apply(
        lambda values: (values - values.mean()) / values.std(),
        axis=0,
    )
    row_labels = _marker_row_labels(marker_slot, feature_names, feature_index)
    group_means.columns = [row_labels[int(index)] for index in feature_index]
    # Features become rows. Display limits clip colors only, so they cannot
    # change the clustering or the returned values.
    matrix = pd.DataFrame(
        group_means.to_numpy(dtype=np.float64, copy=True).T,
        index=group_means.columns,
        columns=group_means.index,
    )

    marker_table = pd.DataFrame(marker_rows)
    marker_table["feature"] = [
        row_labels[int(index)]
        for index in marker_table["feature_index"].to_numpy(dtype=int)
    ]
    return {
        "matrix": matrix,
        "markers": marker_table,
        "assay": assay.name,
        "marker_ref": marker,
        "clusters_ref": clusters_ref,
        "cell_selection": selection_ref,
        "n_cells": len(cell_index),
        "unmeasured_cells": unmeasured,
    }


@close_figures_on_error
def marker_heatmap(
    store: Any,
    *,
    marker: ArtifactRef,
    topn: int = 5,
    log_transform: bool | None = None,
    vmin: float = -1,
    vmax: float = 2,
    figsize: tuple[float, float] | None = None,
    fontsize: float = 10,
    width_factor: float = 0.03,
    height_factor: float = 0.02,
    cmap: Any = "magma_r",
    color_scale: ColorScale | None = None,
    row_order: Sequence[Any] | None = None,
    column_order: Sequence[Any] | None = None,
    cluster_rows: bool = True,
    cluster_columns: bool = True,
    cluster_method: str = "ward",
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
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
    **heatmap_kwargs: Any,
) -> PlotResult:
    """Plot standardized expression of the top marker features."""
    cluster_rows = bool(heatmap_kwargs.pop("row_cluster", cluster_rows))
    cluster_columns = bool(heatmap_kwargs.pop("col_cluster", cluster_columns))
    cluster_method = str(heatmap_kwargs.pop("method", cluster_method))
    cluster_metric = str(heatmap_kwargs.pop("metric", cluster_metric))
    validate_linkage(cluster_method, cluster_metric)
    if "row_linkage" in heatmap_kwargs or "col_linkage" in heatmap_kwargs:
        raise ValueError(
            "Pass clustering controls through cluster_rows, cluster_columns, "
            "cluster_method, and cluster_metric"
        )
    if "z_score" in heatmap_kwargs or "standard_scale" in heatmap_kwargs:
        raise ValueError("marker_heatmap already standardizes features before plotting")
    resolved_color_scale = color_scale or ColorScale(
        cmap=(cmap if isinstance(cmap, str) else getattr(cmap, "name", None)),
        vmin=vmin,
        vmax=vmax,
    )
    if resolved_color_scale.scale != "linear":
        raise NotImplementedError("marker_heatmap supports only linear color scales")
    resolved_vmin = (
        float(resolved_color_scale.vmin)
        if resolved_color_scale.vmin is not None
        else float(vmin)
    )
    resolved_vmax = (
        float(resolved_color_scale.vmax)
        if resolved_color_scale.vmax is not None
        else float(vmax)
    )
    resolved_cmap = resolved_color_scale.cmap or cmap
    if resolved_vmax <= resolved_vmin:
        raise ValueError("vmax must be greater than vmin")
    prepared = _prepare_marker_heatmap(
        store,
        marker=marker,
        topn=topn,
        log_transform=log_transform,
    )
    matrix = cast(pd.DataFrame, prepared["matrix"])
    row_annotation_values = normalize_annotations(
        list(matrix.index),
        row_annotations,
        axis_name="row",
    )
    column_annotation_values = normalize_annotations(
        list(matrix.columns),
        column_annotations,
        axis_name="column",
    )
    ordered_matrix, row_linkage, column_linkage = order_heatmap(
        matrix,
        row_order=row_order,
        column_order=column_order,
        cluster_rows=cluster_rows,
        cluster_columns=cluster_columns,
        method=cluster_method,
        metric=cluster_metric,
    )
    row_colors, row_annotation_scales = annotation_colors(
        row_annotation_values,
        annotation_scales,
    )
    column_colors, column_annotation_scales = annotation_colors(
        column_annotation_values,
        annotation_scales,
    )
    resolved_annotation_scales = row_annotation_scales + column_annotation_scales
    annotation_names = list(row_annotation_values.columns) + list(
        column_annotation_values.columns
    )
    if figsize is None and target is None:
        figsize = (
            ordered_matrix.shape[1] * fontsize * width_factor,
            fontsize * ordered_matrix.shape[0] * height_factor,
        )

    _, mpl = require_matplotlib()
    if target is None:
        sns = require_seaborn()
        plot_index = (
            list(matrix.index)
            if row_linkage is not None
            else list(ordered_matrix.index)
        )
        plot_columns = (
            list(matrix.columns)
            if column_linkage is not None
            else list(ordered_matrix.columns)
        )
        plot_matrix = matrix.reindex(index=plot_index, columns=plot_columns)
        clustermap_kwargs = {
            "yticklabels": True,
            "xticklabels": True,
            "figsize": figsize,
            "cmap": resolved_cmap,
            "vmin": resolved_vmin,
            "vmax": resolved_vmax,
            "center": resolved_color_scale.vcenter,
            "rasterized": True,
            "row_cluster": row_linkage is not None,
            "col_cluster": column_linkage is not None,
            "row_linkage": row_linkage,
            "col_linkage": column_linkage,
            "row_colors": None if row_colors.empty else row_colors,
            "col_colors": None if column_colors.empty else column_colors,
            "cbar_pos": (0.02, 0.8, 0.03, 0.15) if show_legend else None,
        }
        clustermap_kwargs.update(heatmap_kwargs)
        with theme_context(theme):
            cluster_grid = sns.clustermap(plot_matrix, **clustermap_kwargs)
            displayed_matrix = cluster_grid.data2d.copy()
            if clustermap_kwargs["yticklabels"] is not False:
                cluster_grid.ax_heatmap.set_yticklabels(
                    displayed_matrix.index,
                    fontsize=fontsize,
                )
            if clustermap_kwargs["xticklabels"] is not False:
                cluster_grid.ax_heatmap.set_xticklabels(
                    displayed_matrix.columns,
                    fontsize=fontsize,
                )
            if show_legend and cluster_grid.cax is not None:
                cluster_grid.cax.set_ylabel("standardized expression")
            apply_figure_chrome(cluster_grid.fig, theme)
        figure = cluster_grid.fig
        axes: dict[Hashable, Any] = {
            "heatmap": cluster_grid.ax_heatmap,
            "row_dendrogram": cluster_grid.ax_row_dendrogram,
            "column_dendrogram": cluster_grid.ax_col_dendrogram,
        }
        if cluster_grid.cax is not None:
            axes["colorbar"] = cluster_grid.cax
        owns_figure = True
    else:
        allowed_native_kwargs = {
            "aspect",
            "interpolation",
            "rasterized",
        }
        unknown = set(heatmap_kwargs) - allowed_native_kwargs
        if unknown:
            raise TypeError(
                "Unsupported heatmap keyword(s) with target: "
                + ", ".join(sorted(unknown))
            )
        with theme_context(theme):
            figure, axes, owns_figure = normalize_axes_target(
                target,
                panel_keys=["heatmap"],
                figsize=figsize,
            )
            heatmap_ax = axes["heatmap"]
            norm = continuous_norm(
                mpl,
                vmin=resolved_vmin,
                vmax=resolved_vmax,
                vcenter=resolved_color_scale.vcenter,
            )
            image = heatmap_ax.imshow(
                ordered_matrix.to_numpy(dtype=np.float64),
                cmap=resolved_cmap,
                norm=norm,
                aspect=heatmap_kwargs.get("aspect", "auto"),
                interpolation=heatmap_kwargs.get("interpolation", "nearest"),
                rasterized=heatmap_kwargs.get("rasterized", True),
            )
            heatmap_ax.set_xticks(range(ordered_matrix.shape[1]))
            heatmap_ax.set_xticklabels(
                ordered_matrix.columns,
                fontsize=fontsize,
            )
            heatmap_ax.set_yticks(range(ordered_matrix.shape[0]))
            heatmap_ax.set_yticklabels(
                ordered_matrix.index,
                fontsize=fontsize,
            )
            annotation_xlim, annotation_ylim = draw_annotation_strips(
                heatmap_ax,
                row_colors=row_colors.reindex(ordered_matrix.index),
                column_colors=column_colors.reindex(ordered_matrix.columns),
                n_rows=ordered_matrix.shape[0],
                n_columns=ordered_matrix.shape[1],
            )
            heatmap_ax.set_xlim(annotation_xlim)
            heatmap_ax.set_ylim(annotation_ylim)
            if show_legend:
                colorbar = figure.colorbar(
                    image,
                    ax=heatmap_ax,
                    location="top",
                    orientation="horizontal",
                    shrink=0.8,
                    fraction=0.06,
                    pad=0.04,
                )
                colorbar.set_label("standardized expression")
                axes["colorbar"] = colorbar.ax
            apply_figure_chrome(figure, theme)
        displayed_matrix = ordered_matrix

    if show_legend and resolved_annotation_scales:
        annotation_handles = annotation_legend_handles(
            mpl,
            annotation_names,
            resolved_annotation_scales,
        )
        if annotation_handles:
            if owns_figure:
                _place_clustermap_annotation_legend(
                    figure,
                    axes,
                    annotation_handles,
                    has_column_dendrogram=column_linkage is not None,
                )
            else:
                axes["heatmap"].legend(
                    handles=annotation_handles,
                    title="Annotations",
                    frameon=False,
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1),
                )

    resolved_scale = ColorScale(
        cmap=(
            resolved_cmap
            if isinstance(resolved_cmap, str)
            else getattr(resolved_cmap, "name", None)
        ),
        vmin=resolved_vmin,
        vmax=resolved_vmax,
        vcenter=resolved_color_scale.vcenter,
        missing_color=resolved_color_scale.missing_color,
    )
    # The tables name their axes by role, as matrixplot's do. The drawn
    # matrix keeps unnamed axes, because seaborn labels the heatmap's axes
    # with their names.
    tables = {
        "matrix": displayed_matrix.rename_axis(index="feature", columns="group"),
        "markers": cast(pd.DataFrame, prepared["markers"]).copy(),
    }
    if not row_annotation_values.empty:
        tables["row_annotations"] = row_annotation_values.reindex(
            list(displayed_matrix.index)
        ).rename_axis("feature")
    if not column_annotation_values.empty:
        tables["column_annotations"] = column_annotation_values.reindex(
            list(displayed_matrix.columns)
        ).rename_axis("group")
    result = PlotResult(
        figure=figure,
        axes=axes,
        tables=tables,
        legends=(
            LegendSpec(
                kind="colorbar",
                label="standardized expression",
                extras={"vmin": resolved_vmin, "vmax": resolved_vmax},
            ),
            *(LegendSpec(kind="categorical", label=name) for name in annotation_names),
        ),
        scales=(
            resolved_scale,
            *resolved_annotation_scales,
            CategoricalScale(order=tuple(displayed_matrix.columns)),
        ),
        provenance=PlotProvenance(
            assay=cast(str, prepared["assay"]),
            cell_key=None,
            n_cells=cast(int, prepared["n_cells"]),
            renderer="matplotlib",
            notes=(
                "marker_heatmap",
                "clustered"
                if row_linkage is not None or column_linkage is not None
                else "ordered",
            ),
            extras={
                "marker": cast(ArtifactRef, prepared["marker_ref"]).to_dict(),
                "clusters": cast(
                    ArtifactRef,
                    prepared["clusters_ref"],
                ).to_dict(),
                "cell_selection": cast(
                    ArtifactRef,
                    prepared["cell_selection"],
                ).to_dict(),
                "topn": topn,
                "log_transform": log_transform,
                "vmin": resolved_vmin,
                "vmax": resolved_vmax,
                "row_order": list(displayed_matrix.index),
                "column_order": list(displayed_matrix.columns),
                "cluster_rows": row_linkage is not None,
                "cluster_columns": column_linkage is not None,
                "cluster_method": cluster_method,
                "cluster_metric": cluster_metric,
                "row_annotations": list(row_annotation_values.columns),
                "column_annotations": list(column_annotation_values.columns),
                **unmeasured_extras(cast(dict[str, int], prepared["unmeasured_cells"])),
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
    aggregation: ArtifactRef,
) -> dict[str, Any]:
    if not isinstance(aggregation, ArtifactRef):
        raise TypeError("aggregation must be an ArtifactRef")
    loaded = store.load_pseudotime_aggregation(aggregation)
    if loaded.ref != aggregation:
        raise ValueError("Loaded pseudotime aggregation does not match the request")
    pseudotime_result = store.load_pseudotime_scoring(loaded.pseudotime)
    if pseudotime_result.cell_selection != loaded.cell_selection:
        raise ValueError(
            "Pseudotime aggregation and scoring use different cell selections"
        )
    matrix = np.asarray(loaded.data[:])
    feature_indices = np.asarray(loaded.feature_indices, dtype=np.int64)
    feature_clusters = np.asarray(loaded.feature_clusters)
    if matrix.ndim != 2 or matrix.shape[0] != len(feature_indices):
        raise ValueError("Aggregated feature matrix and indices are misaligned")
    if feature_clusters.shape != feature_indices.shape:
        raise ValueError("Aggregated feature clusters and indices are misaligned")
    if not np.isfinite(matrix).all():
        raise ValueError("Aggregated feature matrix contains non-finite values")
    feature_labels = np.asarray(loaded.feature_names).astype(str)
    if feature_labels.shape != feature_indices.shape:
        raise ValueError("Frozen feature labels do not align with aggregation rows")
    order = np.argsort(feature_clusters)
    pseudotime = np.asarray(
        pseudotime_result.values[pseudotime_result.valid],
        dtype=np.float64,
    )
    if len(pseudotime) == 0 or not np.isfinite(pseudotime).all():
        raise ValueError("Pseudotime artifact has no finite scored cells")
    return {
        "matrix": matrix[order],
        "feature_indices": feature_indices[order],
        "feature_clusters": feature_clusters[order],
        "feature_labels": feature_labels[order],
        "pseudotime": pseudotime,
        "assay": loaded.assay,
        "cell_selection": loaded.cell_selection,
        "feature_selection": loaded.feature_selection,
        "pseudotime_ref": loaded.pseudotime,
        "aggregation_ref": aggregation,
    }


@close_figures_on_error
def pseudotime_heatmap(
    store: Any,
    *,
    aggregation: ArtifactRef,
    show_features: list[str] | None = None,
    feature_order: Sequence[str] | None = None,
    feature_cluster_order: Sequence[Any] | None = None,
    figsize: tuple[float, float] = (5, 10),
    vmin: float = -2.0,
    vmax: float = 2.0,
    heatmap_cmap: str | None = None,
    pseudotime_cmap: str | None = None,
    clusterbar_cmap: str | None = None,
    color_scale: ColorScale | None = None,
    feature_cluster_scale: CategoricalScale | None = None,
    pseudotime_scale: ColorScale | None = None,
    tick_fontsize: int = 10,
    axis_fontsize: int = 12,
    feature_label_fontsize: int = 12,
    target: Mapping[Hashable, Any] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot binned feature dynamics ordered by pseudotime."""
    resolved_color_scale = color_scale or ColorScale(
        cmap=heatmap_cmap or "coolwarm",
        vmin=vmin,
        vmax=vmax,
    )
    if resolved_color_scale.scale != "linear":
        raise NotImplementedError("pseudotime_heatmap supports linear scales")
    resolved_vmin = (
        float(resolved_color_scale.vmin)
        if resolved_color_scale.vmin is not None
        else float(vmin)
    )
    resolved_vmax = (
        float(resolved_color_scale.vmax)
        if resolved_color_scale.vmax is not None
        else float(vmax)
    )
    if resolved_vmax <= resolved_vmin:
        raise ValueError("vmax must be greater than vmin")
    prepared = _prepare_pseudotime_heatmap(
        store,
        aggregation=aggregation,
    )
    matrix = np.asarray(prepared["matrix"])
    feature_clusters = np.asarray(prepared["feature_clusters"])
    feature_labels = np.asarray(prepared["feature_labels"])
    pseudotime = np.asarray(prepared["pseudotime"], dtype=float)
    show_features = [] if show_features is None else list(show_features)
    resolved_heatmap_cmap = resolved_color_scale.cmap or heatmap_cmap or "coolwarm"
    pseudotime_color_scale = pseudotime_scale or ColorScale(
        cmap=pseudotime_cmap or "viridis",
    )
    if pseudotime_color_scale.scale != "linear":
        raise NotImplementedError("pseudotime annotations support linear scales")
    resolved_pseudotime_cmap = (
        pseudotime_color_scale.cmap or pseudotime_cmap or "viridis"
    )
    pseudotime_vmin, pseudotime_vmax = resolve_color_limits(
        pseudotime,
        pseudotime_color_scale,
    )
    cluster_source = feature_cluster_scale
    if feature_cluster_order is not None:
        requested_cluster_order = list(feature_cluster_order)
        if len(requested_cluster_order) != len(set(requested_cluster_order)):
            raise ValueError("feature_cluster_order cannot contain duplicates")
        if set(requested_cluster_order) != set(pd.unique(feature_clusters)):
            raise ValueError(
                "feature_cluster_order must contain every observed feature cluster"
            )
        cluster_source = replace(
            feature_cluster_scale or CategoricalScale(),
            order=tuple(requested_cluster_order),
        )
    resolved_feature_cluster_scale = resolve_category_scale(
        feature_clusters,
        cluster_source,
        context="feature_cluster_scale",
    )
    cluster_order = list(resolved_feature_cluster_scale.order or ())
    if feature_cluster_scale is None:
        resolved_feature_cluster_scale = replace(
            resolved_feature_cluster_scale,
            palette=colormap_palette(cluster_order, clusterbar_cmap or "tab20"),
        )
    cluster_palette = resolved_feature_cluster_scale.palette or {}
    if feature_order is not None:
        requested_features = list(feature_order)
        if len(requested_features) != len(set(requested_features)):
            raise ValueError("feature_order cannot contain duplicates")
        observed_features = list(feature_labels)
        if set(requested_features) != set(observed_features):
            raise ValueError("feature_order must contain every plotted feature")
        # An order by name cannot place two rows that share a name, and
        # mapping names to rows would silently drop all but one of them.
        label_counts = pd.Series(observed_features).value_counts()
        repeated = sorted(map(str, label_counts.index[label_counts > 1]))
        if repeated:
            raise ValueError(
                "feature_order cannot order features whose names repeat: "
                + ", ".join(repr(label) for label in repeated)
            )
        feature_index = {label: index for index, label in enumerate(feature_labels)}
        row_order = np.asarray(
            [feature_index[label] for label in requested_features],
            dtype=np.intp,
        )
    else:
        cluster_index = {cluster: index for index, cluster in enumerate(cluster_order)}
        row_order = np.argsort(
            np.asarray([cluster_index[value] for value in feature_clusters]),
            kind="stable",
        )
    matrix = matrix[row_order]
    feature_clusters = feature_clusters[row_order]
    feature_labels = feature_labels[row_order]
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

    plt, mpl = require_matplotlib()
    cluster_cmap = mpl.colors.ListedColormap(
        [cluster_palette[cluster] for cluster in cluster_order]
    )
    with theme_context(theme):
        if target is None:
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
            axes: dict[Hashable, Any] = {
                "heatmap": heatmap_ax,
                "feature_clusters": cluster_ax,
                "colorbar": colorbar_ax,
                "pseudotime": pseudotime_ax,
            }
            owns_figure = True
        else:
            required_targets = {
                "heatmap",
                "feature_clusters",
                "pseudotime",
            }
            missing_targets = required_targets - set(target)
            if missing_targets:
                raise ValueError(
                    "pseudotime_heatmap target is missing axes: "
                    + ", ".join(sorted(map(str, missing_targets)))
                )
            axes = dict(target)
            heatmap_ax = axes["heatmap"]
            cluster_ax = axes["feature_clusters"]
            pseudotime_ax = axes["pseudotime"]
            fig = heatmap_ax.figure
            if any(axis.figure is not fig for axis in axes.values()):
                raise ValueError(
                    "All pseudotime_heatmap target axes must share a figure"
                )
            colorbar_ax = axes.get("colorbar")
            owns_figure = False

        heatmap_norm = continuous_norm(
            mpl,
            vmin=resolved_vmin,
            vmax=resolved_vmax,
            vcenter=resolved_color_scale.vcenter,
        )
        image = heatmap_ax.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap=resolved_heatmap_cmap,
            norm=heatmap_norm,
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
        if show_legend:
            colorbar = (
                fig.colorbar(image, cax=colorbar_ax)
                if colorbar_ax is not None
                else fig.colorbar(
                    image,
                    ax=heatmap_ax,
                    location="right",
                    fraction=0.06,
                    pad=0.04,
                )
            )
            colorbar.ax.tick_params(labelsize=tick_fontsize)
            colorbar.set_label("standardized expression", fontsize=axis_fontsize)
            axes["colorbar"] = colorbar.ax
        elif colorbar_ax is not None:
            colorbar_ax.set_axis_off()

        cluster_ax.imshow(
            encoded_clusters.reshape(-1, 1),
            aspect="auto",
            interpolation="nearest",
            cmap=cluster_cmap,
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
            cmap=resolved_pseudotime_cmap,
            vmin=pseudotime_vmin,
            vmax=pseudotime_vmax,
        )
        pseudotime_ax.set_xticks([0, len(binned_pseudotime) - 1])
        pseudotime_ax.set_xticklabels(
            [f"{binned_pseudotime[0]:.2g}", f"{binned_pseudotime[-1]:.2g}"],
            fontsize=tick_fontsize,
        )
        pseudotime_ax.set_yticks([])
        pseudotime_ax.set_xlabel(
            "Pseudotime",
            fontsize=axis_fontsize,
        )
        apply_figure_chrome(fig, theme)

    feature_indices = np.asarray(prepared["feature_indices"])[row_order]
    feature_table = pd.DataFrame(
        {
            "feature_index": feature_indices,
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
                extras={"vmin": resolved_vmin, "vmax": resolved_vmax},
            ),
            LegendSpec(kind="categorical", label="feature clusters"),
            LegendSpec(
                kind="colorbar",
                label="pseudotime",
                extras={"vmin": pseudotime_vmin, "vmax": pseudotime_vmax},
            ),
        ),
        scales=(
            ColorScale(
                cmap=resolved_heatmap_cmap,
                vmin=resolved_vmin,
                vmax=resolved_vmax,
                vcenter=resolved_color_scale.vcenter,
            ),
            resolved_feature_cluster_scale,
            ColorScale(
                cmap=resolved_pseudotime_cmap,
                vmin=pseudotime_vmin,
                vmax=pseudotime_vmax,
            ),
        ),
        provenance=PlotProvenance(
            assay=cast(str, prepared["assay"]),
            cell_key=None,
            n_cells=len(pseudotime),
            renderer="matplotlib",
            notes=("pseudotime_heatmap", "aggregated"),
            extras={
                "feature_selection": cast(
                    ArtifactRef,
                    prepared["feature_selection"],
                ).to_dict(),
                "cell_selection": cast(
                    ArtifactRef,
                    prepared["cell_selection"],
                ).to_dict(),
                "pseudotime": cast(
                    ArtifactRef,
                    prepared["pseudotime_ref"],
                ).to_dict(),
                "aggregation": cast(
                    ArtifactRef,
                    prepared["aggregation_ref"],
                ).to_dict(),
                "n_features": matrix.shape[0],
                "n_bins": matrix.shape[1],
                "show_features": show_features,
                "feature_order": list(feature_labels),
                "feature_cluster_order": list(cluster_order),
                "show_legend": show_legend,
            },
        ),
        owns_figure=owns_figure,
        theme=theme,
    )
    if show:
        result.show()
    return result
