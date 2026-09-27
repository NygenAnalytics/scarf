"""Plotting functions bound to a datastore instance."""

from collections.abc import Hashable, Mapping, Sequence
from functools import cache
from inspect import Parameter, signature
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


@cache
def _forwarding_layout(
    name: str,
) -> tuple[tuple[str, ...], tuple[str, ...], str | None]:
    """Return an accessor method's positional, keyword, and ``**`` parameters.

    Parameters after ``self`` keep their call style: positional-or-keyword
    parameters are passed positionally, keyword-only parameters other than
    ``run`` by keyword, and a ``**`` parameter is flattened into the keywords.
    """
    positional: list[str] = []
    keywords: list[str] = []
    var_keyword: str | None = None
    method = getattr(DataStorePlotAccessor, name)
    for parameter in list(signature(method).parameters.values())[1:]:
        if parameter.kind is Parameter.POSITIONAL_OR_KEYWORD:
            positional.append(parameter.name)
        elif parameter.kind is Parameter.KEYWORD_ONLY:
            if parameter.name != "run":
                keywords.append(parameter.name)
        elif parameter.kind is Parameter.VAR_KEYWORD:
            var_keyword = parameter.name
        else:
            raise TypeError(
                f"{name}() has an unsupported {parameter.kind.description} parameter"
            )
    return tuple(positional), tuple(keywords), var_keyword


class DataStorePlotAccessor:
    """Datastore-bound facade for store-first plotting functions."""

    __slots__ = ("_store",)

    def __init__(self, store: "DataStore") -> None:
        self._store = store

    def _forward[R](
        self,
        result_type: type[R],
        name: str,
        arguments: Mapping[str, Any],
        /,
        *,
        store: Any = None,
        **overrides: Any,
    ) -> R:
        """Call ``scarf.plotting.<name>`` with a method's declared arguments.

        ``arguments`` is the calling method's ``locals()``. Only that method's
        declared parameters are read from it, and ``run`` is never forwarded.
        ``store`` replaces the bound datastore, and ``overrides`` replace
        keyword arguments.
        """
        from .. import plotting

        positional, keywords, var_keyword = _forwarding_layout(name)
        call_kwargs = {key: arguments[key] for key in keywords}
        call_kwargs.update(overrides)
        if var_keyword is not None:
            call_kwargs.update(arguments[var_keyword])
        function = getattr(plotting, name)
        return cast(
            R,
            function(
                self._store if store is None else store,
                *(arguments[key] for key in positional),
                **call_kwargs,
            ),
        )

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
            # The live-only inputs validated above equal their canonical defaults.
            return self._forward(
                PlotResult,
                "embedding",
                locals(),
                store=_FrozenRunPlotStore(self._store, assay=run.assay, cells=cells),
                layout=layout_ref,
                color_by=resolved_color,
            )
        if isinstance(layout, str):
            raise TypeError("String layout names require a pipeline run")
        return self._forward(PlotResult, "embedding", locals())

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
            # layout_key and cell_key were validated to their canonical defaults.
            return self._forward(
                PlotResult,
                "embedding_raster",
                locals(),
                store=plot_store,
                layout=run["umap" if layout is None else layout],
            )
        if isinstance(layout, str):
            raise TypeError("String layout names require a pipeline run")
        return self._forward(PlotResult, "embedding_raster", locals())

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
        return self._forward(PlotResult, "mapping_score", locals())

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
        return self._forward(PlotResult, "mapping_evidence", locals())

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
        return self._forward(PlotResult, "mapping_confusion", locals())

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
        return self._forward(PlotResult, "mapping_calibration", locals())

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
        return self._forward(PlotResult, "dotplot", locals())

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
        return self._forward(PlotResult, "matrixplot", locals())

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
        return self._forward(PlotResult, "composition", locals())

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
        return self._forward(PlotResult, "distribution", locals())

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
        return self._forward(PlotResult, "marker_heatmap", locals())

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
        return self._forward(PlotRecipeResult, "run_recipe", locals())

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
        return self._forward(PlotResult, "cluster_connectivity", locals())

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
        return self._forward(PlotResult, "modality_weights", locals())

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
        return self._forward(PlotResult, "cluster_tree", locals())

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
        return self._forward(PlotResult, "pseudotime_heatmap", locals())
