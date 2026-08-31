"""Plotting functions bound to a datastore instance."""

from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from ..mapping.reference import MappingReference
from ..plotting._contracts import (
    CategoricalScale,
    CellField,
    ColorScale,
    DensityOverlay,
    DistKind,
    FeatureRef,
    FrameStyle,
    Highlight,
    LegendLoc,
    NormalizationSpec,
    SizeScale,
    StudyDesign,
)
from ..plotting._figure import PlotResult
from ..plotting.recipes import PlotRecipe, PlotRecipeResult
from ..storage.refs import ArtifactRef
from .pipeline_run import PipelineRun

if TYPE_CHECKING:
    from .datastore import DataStore


class _FrozenRunPlotCells:
    """Adapt a frozen run axis to the small metadata surface plotting uses."""

    __slots__ = ("_cells",)

    def __init__(self, cells: Any) -> None:
        self._cells = cells

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self._cells.columns)

    @property
    def _selection_ref(self) -> ArtifactRef:
        return cast(ArtifactRef, self._cells._selection_ref)

    def fetch_all(self, column: str) -> np.ndarray:
        return np.asarray(self._cells._plot_fetch_all(column))

    def fetch(self, column: str, *, key: str = "I") -> np.ndarray:
        if key != "I":
            raise ValueError("Run plots use the frozen pipeline cell selection")
        plot_selected = getattr(self._cells, "_plot_fetch_selected", None)
        if callable(plot_selected):
            return np.asarray(plot_selected(column))
        values = self.fetch_all(column)
        return np.asarray(values[self._cells.fetch_all("I")])

    def get_dtype(self, column: str) -> np.dtype[Any]:
        return cast(np.dtype[Any], self._cells._field_dtype(column))

    def _field_dtype(self, column: str) -> np.dtype[Any]:
        return cast(np.dtype[Any], self._cells._field_dtype(column))

    def _field_display(self, column: str) -> dict[str, Any] | None:
        return cast(dict[str, Any] | None, self._cells._field_display(column))

    def _iter_selected_blocks(
        self,
        columns: Sequence[str],
        block_rows: int | None = None,
    ) -> Any:
        return self._cells._iter_selected_blocks(columns, block_rows)


class _FrozenRunPlotStore:
    """Expose only the frozen run cells needed by artifact embedding plots."""

    __slots__ = ("_defaultAssay", "cells", "zw")

    def __init__(self, store: "DataStore", *, assay: str, cells: Any) -> None:
        self._defaultAssay = assay
        self.cells = _FrozenRunPlotCells(cells)
        self.zw = store.zw

    def _stored_display_metadata(self, column: str) -> dict[str, Any] | None:
        return self.cells._field_display(column)


class DataStorePlotAccessor:
    """Datastore-bound facade for store-first plotting functions."""

    __slots__ = ("_store",)

    def __init__(self, store: "DataStore") -> None:
        self._store = store

    def embedding(
        self,
        *,
        layout_key: str | Sequence[str] | None = None,
        layout: str | ArtifactRef | None = None,
        run: PipelineRun | None = None,
        color_by: (
            "str"
            " | ArtifactRef"
            " | FeatureRef"
            " | CellField"
            " | Sequence[str | ArtifactRef | FeatureRef | CellField]"
            " | None"
        ) = None,
        facet_by: str | None = None,
        facet_order: Sequence[Any] | None = None,
        cell_key: str = "I",
        from_assay: str | None = None,
        normalization: "NormalizationSpec | None" = None,
        point_size: float | None = None,
        point_sizes: "np.ndarray | Sequence[float] | None" = None,
        point_size_range: tuple[float, float] = (1.0, 28.0),
        point_edgecolor: str | None = None,
        point_edgewidth: float | None = None,
        point_alpha: float = 1.0,
        sort_values: bool = False,
        color_scale: "ColorScale | None" = None,
        categorical_scale: "CategoricalScale | None" = None,
        default_color: str = "steelblue",
        missing_color: str | None = None,
        clip_fraction: float = 0.0,
        subset_by: str | None = None,
        groups: Sequence[Any] | None = None,
        n_columns: int | None = None,
        target: Any | None = None,
        figsize: tuple[float, float] | None = None,
        theme: str = "notebook",
        legend_loc: "LegendLoc" = "auto",
        max_on_data_labels: int = 40,
        show_legend: bool = True,
        show_titles: bool = True,
        frame: "FrameStyle" = "minimal",
        density_overlay: "DensityOverlay | None" = None,
        highlight: "Highlight | None" = None,
        seed: int | None = None,
        rasterize_threshold: int = 50_000,
        show: bool = True,
    ) -> "PlotResult":
        """Plot cells in a stored two-dimensional embedding."""
        if run is not None:
            if not isinstance(run, PipelineRun):
                raise TypeError("run must be a PipelineRun")
            if run._owner is not self._store:
                raise ValueError("run must be opened from this datastore")
            if layout_key is not None or isinstance(layout, ArtifactRef):
                raise ValueError(
                    "run is mutually exclusive with layout_key or an ArtifactRef layout"
                )
            if layout is not None and not isinstance(layout, str):
                raise TypeError("layout must name a pipeline output")
            if not isinstance(color_by, str | type(None)):
                raise TypeError("color_by must name a frozen cell field or be None")
            if (
                cell_key != "I"
                or from_assay is not None
                or normalization is not None
                or point_sizes is not None
                or facet_by is not None
                or facet_order is not None
                or subset_by is not None
            ):
                raise ValueError(
                    "Run embedding uses frozen layout and color outputs; live "
                    "selection, feature, facet, and subset inputs are unavailable"
                )
            if density_overlay is not None and density_overlay.group_by is not None:
                raise ValueError(
                    "Run embedding density filters cannot use live metadata"
                )
            if highlight is not None and highlight.by is not None:
                raise ValueError("Run embedding highlights cannot use live metadata")
            cells = run.cells
            layout_ref = run["umap" if layout is None else layout]
            if color_by is None:
                resolved_color: str | None = None
            elif color_by in cells.columns:
                resolved_color = color_by
            else:
                raise KeyError(f"Pipeline run has no frozen cell field {color_by!r}")
            from ..plotting import embedding

            return embedding(
                _FrozenRunPlotStore(self._store, assay=run.assay, cells=cells),
                layout=layout_ref,
                color_by=resolved_color,
                point_size=point_size,
                point_size_range=point_size_range,
                point_edgecolor=point_edgecolor,
                point_edgewidth=point_edgewidth,
                point_alpha=point_alpha,
                sort_values=sort_values,
                color_scale=color_scale,
                categorical_scale=categorical_scale,
                default_color=default_color,
                missing_color=missing_color,
                clip_fraction=clip_fraction,
                groups=groups,
                n_columns=n_columns,
                target=target,
                figsize=figsize,
                theme=theme,
                legend_loc=legend_loc,
                max_on_data_labels=max_on_data_labels,
                show_legend=show_legend,
                show_titles=show_titles,
                frame=frame,
                density_overlay=density_overlay,
                highlight=highlight,
                seed=seed,
                rasterize_threshold=rasterize_threshold,
                show=show,
            )
        if isinstance(layout, str):
            raise TypeError("String layout names require a pipeline run")
        from ..plotting import embedding

        return embedding(
            self._store,
            layout_key=layout_key,
            layout=layout,
            color_by=color_by,
            facet_by=facet_by,
            facet_order=facet_order,
            cell_key=cell_key,
            from_assay=from_assay,
            normalization=normalization,
            point_size=point_size,
            point_sizes=point_sizes,
            point_size_range=point_size_range,
            point_edgecolor=point_edgecolor,
            point_edgewidth=point_edgewidth,
            point_alpha=point_alpha,
            sort_values=sort_values,
            color_scale=color_scale,
            categorical_scale=categorical_scale,
            default_color=default_color,
            missing_color=missing_color,
            clip_fraction=clip_fraction,
            subset_by=subset_by,
            groups=groups,
            n_columns=n_columns,
            target=target,
            figsize=figsize,
            theme=theme,
            legend_loc=legend_loc,
            max_on_data_labels=max_on_data_labels,
            show_legend=show_legend,
            show_titles=show_titles,
            frame=frame,
            density_overlay=density_overlay,
            highlight=highlight,
            seed=seed,
            rasterize_threshold=rasterize_threshold,
            show=show,
        )

    def embedding_raster(
        self,
        *,
        layout_key: str | None = None,
        layout: str | ArtifactRef | None = None,
        run: PipelineRun | None = None,
        color_by: "str | CellField | None" = None,
        cell_key: str = "I",
        pixels: int = 400,
        block_rows: int | None = None,
        color_scale: "ColorScale | None" = None,
        missing_color: str | None = None,
        subset_by: str | None = None,
        target: Any | None = None,
        figsize: tuple[float, float] | None = None,
        theme: str = "notebook",
        seed: int = 0,
        show: bool = True,
    ) -> "PlotResult":
        """Rasterize continuous cell metadata over a stored embedding."""
        from ..plotting import embedding_raster

        if run is not None:
            if not isinstance(run, PipelineRun):
                raise TypeError("run must be a PipelineRun")
            if run._owner is not self._store:
                raise ValueError("run must be opened from this datastore")
            if layout_key is not None or isinstance(layout, ArtifactRef):
                raise ValueError(
                    "run is mutually exclusive with layout_key or an ArtifactRef layout"
                )
            if layout is not None and not isinstance(layout, str):
                raise TypeError("layout must name a pipeline output")
            if cell_key != "I":
                raise ValueError("Run raster uses the frozen pipeline cell selection")
            if not isinstance(color_by, str | type(None)):
                raise TypeError("color_by must name a frozen cell field or be None")
            color_key = color_by
            cells = run.cells
            plot_store = _FrozenRunPlotStore(
                self._store,
                assay=run.assay,
                cells=cells,
            )
            if color_key is not None and color_key not in cells.columns:
                raise KeyError(f"Pipeline run has no frozen cell field {color_key!r}")
            if subset_by is not None and subset_by not in cells.columns:
                raise KeyError(f"Pipeline run has no frozen cell field {subset_by!r}")
            return embedding_raster(
                plot_store,
                layout=run["umap" if layout is None else layout],
                color_by=color_by,
                pixels=pixels,
                block_rows=block_rows,
                color_scale=color_scale,
                missing_color=missing_color,
                subset_by=subset_by,
                target=target,
                figsize=figsize,
                theme=theme,
                seed=seed,
                show=show,
            )
        if isinstance(layout, str):
            raise TypeError("String layout names require a pipeline run")
        return embedding_raster(
            self._store,
            layout_key=layout_key,
            layout=layout,
            color_by=color_by,
            cell_key=cell_key,
            pixels=pixels,
            block_rows=block_rows,
            color_scale=color_scale,
            missing_color=missing_color,
            subset_by=subset_by,
            target=target,
            figsize=figsize,
            theme=theme,
            seed=seed,
            show=show,
        )

    def mapping_score(
        self,
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
        from ..plotting import mapping_score

        return mapping_score(
            self._store,
            result,
            reference=reference,
            target_groups=target_groups,
            layout=layout,
            kind=kind,
            reference_class_group=reference_class_group,
            size_by_score=size_by_score,
            log_transform=log_transform,
            multiplier=multiplier,
            weighted=weighted,
            fixed_weight=fixed_weight,
            bins=bins,
            point_size=point_size,
            color_scale=color_scale,
            categorical_scale=categorical_scale,
            target=target,
            figsize=figsize,
            theme=theme,
            show_legend=show_legend,
            show=show,
        )

    def mapping_evidence(
        self,
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
        from ..plotting import mapping_evidence

        return mapping_evidence(
            self._store,
            result,
            reference=reference,
            reference_class_group=reference_class_group,
            target_groups=target_groups,
            metrics=metrics,
            kind=kind,
            bins=bins,
            threshold_fraction=threshold_fraction,
            na_val=na_val,
            max_distance=max_distance,
            categorical_scale=categorical_scale,
            target=target,
            figsize=figsize,
            theme=theme,
            show_legend=show_legend,
            show=show,
        )

    def mapping_confusion(
        self,
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
        from ..plotting import mapping_confusion

        return mapping_confusion(
            self._store,
            result,
            reference=reference,
            reference_class_group=reference_class_group,
            known_labels=known_labels,
            normalize=normalize,
            known_order=known_order,
            predicted_order=predicted_order,
            threshold_fraction=threshold_fraction,
            na_val=na_val,
            max_distance=max_distance,
            color_scale=color_scale,
            target=target,
            figsize=figsize,
            theme=theme,
            show_legend=show_legend,
            show=show,
        )

    def mapping_calibration(
        self,
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
        from ..plotting import mapping_calibration

        return mapping_calibration(
            self._store,
            result,
            reference=reference,
            reference_class_group=reference_class_group,
            known_labels=known_labels,
            metric=metric,
            direction=direction,
            thresholds=thresholds,
            n_thresholds=n_thresholds,
            chosen_threshold=chosen_threshold,
            na_val=na_val,
            max_distance=max_distance,
            target=target,
            figsize=figsize,
            theme=theme,
            show=show,
        )

    def dotplot(
        self,
        *,
        features: (
            "Sequence[str | FeatureRef] | Mapping[str, Sequence[str | FeatureRef]]"
        ),
        group_by: str | tuple[str, ...] | None = None,
        groups: ArtifactRef | None = None,
        cell_key: str = "I",
        from_assay: str | None = None,
        sample_by: str | None = None,
        study_design: "StudyDesign | None" = None,
        normalization: "NormalizationSpec | None" = None,
        expression_cutoff: float = 0.0,
        standardize: str = "none",
        color_scale: "ColorScale | None" = None,
        size_scale: "SizeScale | None" = None,
        categorical_scale: "CategoricalScale | None" = None,
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
    ) -> "PlotResult":
        """Summarize feature expression as a dot plot."""
        from ..plotting import dotplot

        return dotplot(
            self._store,
            features=features,
            group_by=group_by,
            groups=groups,
            cell_key=cell_key,
            from_assay=from_assay,
            sample_by=sample_by,
            study_design=study_design,
            normalization=normalization,
            expression_cutoff=expression_cutoff,
            standardize=standardize,
            color_scale=color_scale,
            size_scale=size_scale,
            categorical_scale=categorical_scale,
            group_order=group_order,
            feature_order=feature_order,
            swap_axes=swap_axes,
            marker_edgecolor=marker_edgecolor,
            marker_linewidth=marker_linewidth,
            label_wrap=label_wrap,
            italicize_features=italicize_features,
            target=target,
            figsize=figsize,
            max_figure_width=max_figure_width,
            theme=theme,
            show_legend=show_legend,
            show=show,
        )

    def matrixplot(
        self,
        *,
        features: (
            "Sequence[str | FeatureRef] | Mapping[str, Sequence[str | FeatureRef]]"
        ),
        group_by: str | tuple[str, ...] | None = None,
        groups: ArtifactRef | None = None,
        cell_key: str = "I",
        from_assay: str | None = None,
        sample_by: str | None = None,
        study_design: "StudyDesign | None" = None,
        normalization: "NormalizationSpec | None" = None,
        expression_cutoff: float = 0.0,
        value: str = "mean",
        standardize: str = "none",
        color_scale: "ColorScale | None" = None,
        feature_order: Sequence[Any] | None = None,
        group_order: Sequence[Any] | None = None,
        cluster_features: bool = False,
        cluster_groups: bool = False,
        cluster_method: str = "average",
        cluster_metric: str = "euclidean",
        row_annotations: (
            Mapping[str, Mapping[Any, Any] | Sequence[Any]] | None
        ) = None,
        column_annotations: (
            Mapping[str, Mapping[Any, Any] | Sequence[Any]] | None
        ) = None,
        annotation_scales: Mapping[str, CategoricalScale] | None = None,
        target: Any | None = None,
        figsize: tuple[float, float] | None = None,
        theme: str = "notebook",
        show_legend: bool = True,
        show: bool = True,
    ) -> "PlotResult":
        """Summarize feature expression as a matrix plot."""
        from ..plotting import matrixplot

        return matrixplot(
            self._store,
            features=features,
            group_by=group_by,
            groups=groups,
            cell_key=cell_key,
            from_assay=from_assay,
            sample_by=sample_by,
            study_design=study_design,
            normalization=normalization,
            expression_cutoff=expression_cutoff,
            value=value,
            standardize=standardize,
            color_scale=color_scale,
            feature_order=feature_order,
            group_order=group_order,
            cluster_features=cluster_features,
            cluster_groups=cluster_groups,
            cluster_method=cluster_method,
            cluster_metric=cluster_metric,
            row_annotations=row_annotations,
            column_annotations=column_annotations,
            annotation_scales=annotation_scales,
            target=target,
            figsize=figsize,
            theme=theme,
            show_legend=show_legend,
            show=show,
        )

    def composition(
        self,
        *,
        category_by: str | None = None,
        categories: ArtifactRef | None = None,
        cell_key: str = "I",
        sample_by: str | None = None,
        grouping: ArtifactRef | None = None,
        subject_by: str | None = None,
        pair_by: str | None = None,
        condition_by: str | None = None,
        study_design: "StudyDesign | None" = None,
        kind: Literal["stacked", "per_sample"] = "stacked",
        show_summary: bool = True,
        uncertainty: Literal["none", "sd", "se", "ci95"] | None = None,
        categorical_scale: "CategoricalScale | None" = None,
        bar_width: float = 0.82,
        bar_gap: float = 0.12,
        segment_edgecolor: str | None = None,
        segment_linewidth: float = 0.5,
        show_percent_labels: bool = False,
        label_min_fraction: float = 0.08,
        percent_format: str = "{:.0%}",
        target: Any | None = None,
        figsize: tuple[float, float] | None = None,
        max_figure_width: float | None = 7.5,
        theme: str = "notebook",
        show_legend: bool = True,
        show: bool = True,
    ) -> "PlotResult":
        """Plot category composition for the selected cells."""
        from ..plotting import composition

        return composition(
            self._store,
            category_by=category_by,
            categories=categories,
            cell_key=cell_key,
            sample_by=sample_by,
            grouping=grouping,
            subject_by=subject_by,
            pair_by=pair_by,
            condition_by=condition_by,
            study_design=study_design,
            kind=kind,
            show_summary=show_summary,
            uncertainty=uncertainty,
            categorical_scale=categorical_scale,
            bar_width=bar_width,
            bar_gap=bar_gap,
            segment_edgecolor=segment_edgecolor,
            segment_linewidth=segment_linewidth,
            show_percent_labels=show_percent_labels,
            label_min_fraction=label_min_fraction,
            percent_format=percent_format,
            target=target,
            figsize=figsize,
            max_figure_width=max_figure_width,
            theme=theme,
            show_legend=show_legend,
            show=show,
        )

    def distribution(
        self,
        keys: (
            "str | CellField | FeatureRef | ArtifactRef"
            " | Sequence[str | CellField | FeatureRef]"
        ),
        *,
        grouping: ArtifactRef | CellField | None = None,
        cell_selection: ArtifactRef | None = None,
        groups: Sequence[Any] | None = None,
        split_by: str | None = None,
        sample_by: str | None = None,
        study_design: "StudyDesign | None" = None,
        sample_stat: Literal["mean", "median", "fraction"] = "mean",
        expression_cutoff: float = 0.0,
        subset_by: str | None = None,
        from_assay: str | None = None,
        normalization: "NormalizationSpec | None" = None,
        categorical_scale: "CategoricalScale | None" = None,
        split_scale: "CategoricalScale | None" = None,
        kind: "DistKind" = "violin",
        bins: int = 40,
        max_points: int | None = 10000,
        point_size: float = 0.8,
        point_alpha: float = 0.28,
        seed: int = 0,
        color: str = "steelblue",
        color_by: Literal["group", "mean"] = "group",
        color_scale: "ColorScale | None" = None,
        orientation: Literal["vertical", "horizontal"] = "vertical",
        row_standardize: bool = False,
        share_y: bool | None = None,
        violin_inner: str | None = "quartile",
        violin_linewidth: float = 0.8,
        violin_alpha: float = 0.9,
        italicize_features: bool = False,
        target: Any | None = None,
        figsize: tuple[float, float] | None = None,
        max_figure_width: float | None = 7.5,
        title: str | None = None,
        theme: str = "notebook",
        show_legend: bool = True,
        stats_results: Any = None,
        stats_keys: Sequence[str] | None = None,
        stats_bracket_height: float | None = None,
        stats_show_p: bool = True,
        show: bool = True,
    ) -> "PlotResult":
        """Plot distributions of cell metadata or feature values.

        ``stats_results`` overlays significance brackets from
        ``run_statistical_testing`` results onto the drawn violins or
        boxes; see :func:`scarf.plotting.distribution` for the full
        behaviour. ``max_points`` defaults to ``10000``; explicit ``None``
        disables the point overlay for stacked violins and otherwise uses
        ``10000``.
        """
        from ..plotting import distribution

        resolved_stats = stats_results
        if isinstance(stats_results, ArtifactRef):
            resolved_stats = self._store.get_statistical_tests(stats_results)
        elif isinstance(stats_results, Mapping):
            resolved_stats = {
                key: (
                    self._store.get_statistical_tests(value)
                    if isinstance(value, ArtifactRef)
                    else value
                )
                for key, value in stats_results.items()
            }
        return distribution(
            self._store,
            keys,
            grouping=grouping,
            cell_selection=cell_selection,
            groups=groups,
            split_by=split_by,
            sample_by=sample_by,
            study_design=study_design,
            sample_stat=sample_stat,
            expression_cutoff=expression_cutoff,
            subset_by=subset_by,
            from_assay=from_assay,
            normalization=normalization,
            categorical_scale=categorical_scale,
            split_scale=split_scale,
            kind=kind,
            bins=bins,
            max_points=max_points,
            point_size=point_size,
            point_alpha=point_alpha,
            seed=seed,
            color=color,
            color_by=color_by,
            color_scale=color_scale,
            orientation=orientation,
            row_standardize=row_standardize,
            share_y=share_y,
            violin_inner=violin_inner,
            violin_linewidth=violin_linewidth,
            violin_alpha=violin_alpha,
            italicize_features=italicize_features,
            target=target,
            figsize=figsize,
            max_figure_width=max_figure_width,
            title=title,
            theme=theme,
            show_legend=show_legend,
            stats_results=resolved_stats,
            stats_keys=stats_keys,
            stats_bracket_height=stats_bracket_height,
            stats_show_p=stats_show_p,
            show=show,
        )

    def marker_heatmap(
        self,
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
        color_scale: "ColorScale | None" = None,
        row_order: Sequence[Any] | None = None,
        column_order: Sequence[Any] | None = None,
        cluster_rows: bool = True,
        cluster_columns: bool = True,
        cluster_method: str = "ward",
        cluster_metric: str = "euclidean",
        row_annotations: (
            Mapping[str, Mapping[Any, Any] | Sequence[Any]] | None
        ) = None,
        column_annotations: (
            Mapping[str, Mapping[Any, Any] | Sequence[Any]] | None
        ) = None,
        annotation_scales: Mapping[str, CategoricalScale] | None = None,
        target: Any | None = None,
        theme: str = "notebook",
        show_legend: bool = True,
        show: bool = True,
        **heatmap_kwargs: Any,
    ) -> "PlotResult":
        """Plot the stored marker table as a heatmap."""
        from ..plotting import marker_heatmap

        return marker_heatmap(
            self._store,
            marker=marker,
            topn=topn,
            log_transform=log_transform,
            vmin=vmin,
            vmax=vmax,
            figsize=figsize,
            fontsize=fontsize,
            width_factor=width_factor,
            height_factor=height_factor,
            cmap=cmap,
            color_scale=color_scale,
            row_order=row_order,
            column_order=column_order,
            cluster_rows=cluster_rows,
            cluster_columns=cluster_columns,
            cluster_method=cluster_method,
            cluster_metric=cluster_metric,
            row_annotations=row_annotations,
            column_annotations=column_annotations,
            annotation_scales=annotation_scales,
            target=target,
            theme=theme,
            show_legend=show_legend,
            show=show,
            **heatmap_kwargs,
        )

    def run_recipe(
        self,
        recipe: "PlotRecipe | str | Path",
        *,
        artifacts: Mapping[str, Any] | None = None,
        targets: Mapping[str, Any] | None = None,
        output_dir: str | Path | None = None,
        show: bool = False,
        continue_on_error: bool = False,
    ) -> "PlotRecipeResult":
        """Run a declarative plotting recipe against this datastore."""
        from ..plotting import run_recipe

        return run_recipe(
            self._store,
            recipe,
            artifacts=artifacts,
            targets=targets,
            output_dir=output_dir,
            show=show,
            continue_on_error=continue_on_error,
        )

    def cluster_connectivity(
        self,
        *,
        group_by: str | None = None,
        layout_key: str | None = None,
        groups: ArtifactRef | None = None,
        layout: ArtifactRef | None = None,
        graph: ArtifactRef,
        cell_key: str = "I",
        position: Literal["median", "mean"] = "median",
        positions: Mapping[Any, tuple[float, float]] | None = None,
        categorical_scale: "CategoricalScale | None" = None,
        size_scale: "SizeScale | None" = None,
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
    ) -> "PlotResult":
        """Summarize cell graph connectivity between embedding clusters."""
        from ..plotting import cluster_connectivity

        return cluster_connectivity(
            self._store,
            group_by=group_by,
            layout_key=layout_key,
            groups=groups,
            layout=layout,
            graph=graph,
            cell_key=cell_key,
            position=position,
            positions=positions,
            categorical_scale=categorical_scale,
            size_scale=size_scale,
            minimum_edge_weight=minimum_edge_weight,
            max_edges_per_node=max_edges_per_node,
            show_cells=show_cells,
            cell_size=cell_size,
            cell_alpha=cell_alpha,
            cell_color=cell_color,
            node_edgecolor=node_edgecolor,
            node_linewidth=node_linewidth,
            edge_color=edge_color,
            edge_alpha=edge_alpha,
            edge_width_range=edge_width_range,
            labels=labels,
            target=target,
            figsize=figsize,
            theme=theme,
            show=show,
        )

    def modality_weights(
        self,
        *,
        graph: ArtifactRef,
        layout: ArtifactRef,
        point_size: float | None = None,
        point_alpha: float = 1.0,
        cmap: str = "viridis",
        n_columns: int | None = None,
        target: Any | None = None,
        figsize: tuple[float, float] | None = None,
        theme: str = "notebook",
        frame: FrameStyle = "minimal",
        rasterize_threshold: int = 50_000,
        show: bool = True,
    ) -> "PlotResult":
        """Plot each assay's WNN contribution over an explicit embedding."""
        from ..plotting import modality_weights

        return modality_weights(
            self._store,
            graph=graph,
            layout=layout,
            point_size=point_size,
            point_alpha=point_alpha,
            cmap=cmap,
            n_columns=n_columns,
            target=target,
            figsize=figsize,
            theme=theme,
            frame=frame,
            rasterize_threshold=rasterize_threshold,
            show=show,
        )

    def cluster_tree(
        self,
        *,
        graph: ArtifactRef,
        clusters: ArtifactRef,
        from_assay: str | None = None,
        fill_by_value: str | None = None,
        force_ints_as_cats: bool = True,
        width: float = 1,
        lvr_factor: float = 0.5,
        vert_gap: float = 0.2,
        min_node_size: float = 10,
        node_size_multiplier: float = 10_000.0,
        node_power: float = 1.2,
        root_size: float = 100,
        non_leaf_size: float = 10,
        show_labels: bool = True,
        fontsize: float = 10,
        root_color: str = "#C0C0C0",
        non_leaf_color: str = "k",
        cmap: str = "tab20",
        color_key: dict[Any, str] | None = None,
        edgecolors: str = "k",
        edgewidth: float = 1,
        alpha: float = 0.7,
        figsize: tuple[float, float] = (5, 5),
        ax: Any = None,
        theme: str = "notebook",
        show: bool = True,
    ) -> "PlotResult":
        """Plot a stored hierarchical clustering tree."""
        from ..plotting import cluster_tree

        return cluster_tree(
            self._store,
            graph=graph,
            clusters=clusters,
            from_assay=from_assay,
            fill_by_value=fill_by_value,
            force_ints_as_cats=force_ints_as_cats,
            width=width,
            lvr_factor=lvr_factor,
            vert_gap=vert_gap,
            min_node_size=min_node_size,
            node_size_multiplier=node_size_multiplier,
            node_power=node_power,
            root_size=root_size,
            non_leaf_size=non_leaf_size,
            show_labels=show_labels,
            fontsize=fontsize,
            root_color=root_color,
            non_leaf_color=non_leaf_color,
            cmap=cmap,
            color_key=color_key,
            edgecolors=edgecolors,
            edgewidth=edgewidth,
            alpha=alpha,
            figsize=figsize,
            ax=ax,
            theme=theme,
            show=show,
        )

    def pseudotime_heatmap(
        self,
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
        color_scale: "ColorScale | None" = None,
        feature_cluster_scale: "CategoricalScale | None" = None,
        pseudotime_scale: "ColorScale | None" = None,
        tick_fontsize: int = 10,
        axis_fontsize: int = 12,
        feature_label_fontsize: int = 12,
        target: Mapping[Hashable, Any] | None = None,
        theme: str = "notebook",
        show_legend: bool = True,
        show: bool = True,
    ) -> "PlotResult":
        """Plot feature profiles ordered by stored pseudotime."""
        from ..plotting import pseudotime_heatmap

        return pseudotime_heatmap(
            self._store,
            aggregation=aggregation,
            show_features=show_features,
            feature_order=feature_order,
            feature_cluster_order=feature_cluster_order,
            figsize=figsize,
            vmin=vmin,
            vmax=vmax,
            heatmap_cmap=heatmap_cmap,
            pseudotime_cmap=pseudotime_cmap,
            clusterbar_cmap=clusterbar_cmap,
            color_scale=color_scale,
            feature_cluster_scale=feature_cluster_scale,
            pseudotime_scale=pseudotime_scale,
            tick_fontsize=tick_fontsize,
            axis_fontsize=axis_fontsize,
            feature_label_fontsize=feature_label_fontsize,
            target=target,
            theme=theme,
            show_legend=show_legend,
            show=show,
        )
