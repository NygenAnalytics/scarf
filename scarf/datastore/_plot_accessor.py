"""Plotting functions bound to a datastore instance."""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ..plotting._contracts import (
    CategoricalScale,
    CellField,
    ColorScale,
    DistKind,
    FeatureRef,
    FrameStyle,
    LegendLoc,
    NormalizationSpec,
    SizeScale,
    StudyDesign,
)
from ..plotting._figure import PlotResult

if TYPE_CHECKING:
    from .datastore import DataStore


class DataStorePlotAccessor:
    """Datastore-bound facade for store-first plotting functions."""

    __slots__ = ("_store",)

    def __init__(self, store: "DataStore") -> None:
        self._store = store

    def embedding(
        self,
        *,
        layout_key: str | Sequence[str],
        color_by: (
            "str"
            " | FeatureRef"
            " | CellField"
            " | Sequence[str | FeatureRef | CellField]"
            " | None"
        ) = None,
        facet_by: str | None = None,
        facet_order: Sequence[Any] | None = None,
        cell_key: str = "I",
        from_assay: str | None = None,
        normalization: "NormalizationSpec | None" = None,
        point_size: float | None = None,
        point_sizes: "np.ndarray | Sequence[float] | None" = None,
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
        frame: "FrameStyle" = "minimal",
        seed: int | None = None,
        rasterize_threshold: int = 50_000,
        show: bool = True,
    ) -> "PlotResult":
        """Plot cells in a stored two-dimensional embedding."""
        from ..plotting import embedding

        return embedding(
            self._store,
            layout_key=layout_key,
            color_by=color_by,
            facet_by=facet_by,
            facet_order=facet_order,
            cell_key=cell_key,
            from_assay=from_assay,
            normalization=normalization,
            point_size=point_size,
            point_sizes=point_sizes,
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
            frame=frame,
            seed=seed,
            rasterize_threshold=rasterize_threshold,
            show=show,
        )

    def embedding_raster(
        self,
        *,
        layout_key: str,
        color_by: "str | CellField | None" = None,
        cell_key: str = "I",
        pixels: int = 400,
        block_rows: int | None = None,
        color_scale: "ColorScale | None" = None,
        missing_color: str = "white",
        subset_by: str | None = None,
        target: Any | None = None,
        figsize: tuple[float, float] | None = None,
        theme: str = "notebook",
        seed: int = 0,
        show: bool = True,
    ) -> "PlotResult":
        """Rasterize continuous cell metadata over a stored embedding."""
        from ..plotting import embedding_raster

        return embedding_raster(
            self._store,
            layout_key=layout_key,
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

    def unified_embedding(
        self,
        *,
        layout_key: str,
        from_assay: str | None = None,
        show_target_only: bool = False,
        ref_name: str = "reference",
        target_groups: Sequence[Any] | None = None,
        point_size: float | None = None,
        categorical_scale: "CategoricalScale | None" = None,
        missing_color: str = "#bdbdbd",
        figsize: tuple[float, float] | None = None,
        theme: str = "notebook",
        legend_loc: "LegendLoc" = "auto",
        frame: "FrameStyle" = "minimal",
        seed: int | None = None,
        rasterize_threshold: int = 50_000,
        target: Any | None = None,
        show: bool = True,
    ) -> "PlotResult":
        """Plot a unified reference and query embedding."""
        from ..plotting import unified_embedding

        return unified_embedding(
            self._store,
            layout_key=layout_key,
            from_assay=from_assay,
            show_target_only=show_target_only,
            ref_name=ref_name,
            target_groups=target_groups,
            point_size=point_size,
            categorical_scale=categorical_scale,
            missing_color=missing_color,
            figsize=figsize,
            theme=theme,
            legend_loc=legend_loc,
            frame=frame,
            seed=seed,
            rasterize_threshold=rasterize_threshold,
            target=target,
            show=show,
        )

    def dotplot(
        self,
        *,
        features: (
            "Sequence[str | FeatureRef] | Mapping[str, Sequence[str | FeatureRef]]"
        ),
        group_by: str | tuple[str, ...],
        cell_key: str = "I",
        from_assay: str | None = None,
        sample_by: str | None = None,
        study_design: "StudyDesign | None" = None,
        normalization: "NormalizationSpec | None" = None,
        expression_cutoff: float = 0.0,
        standardize: str = "none",
        color_scale: "ColorScale | None" = None,
        size_scale: "SizeScale | None" = None,
        target: Any | None = None,
        figsize: tuple[float, float] | None = None,
        theme: str = "notebook",
        show: bool = True,
    ) -> "PlotResult":
        """Summarize feature expression as a dot plot."""
        from ..plotting import dotplot

        return dotplot(
            self._store,
            features=features,
            group_by=group_by,
            cell_key=cell_key,
            from_assay=from_assay,
            sample_by=sample_by,
            study_design=study_design,
            normalization=normalization,
            expression_cutoff=expression_cutoff,
            standardize=standardize,
            color_scale=color_scale,
            size_scale=size_scale,
            target=target,
            figsize=figsize,
            theme=theme,
            show=show,
        )

    def matrixplot(
        self,
        *,
        features: (
            "Sequence[str | FeatureRef] | Mapping[str, Sequence[str | FeatureRef]]"
        ),
        group_by: str | tuple[str, ...],
        cell_key: str = "I",
        from_assay: str | None = None,
        sample_by: str | None = None,
        study_design: "StudyDesign | None" = None,
        normalization: "NormalizationSpec | None" = None,
        expression_cutoff: float = 0.0,
        value: str = "mean",
        standardize: str = "none",
        color_scale: "ColorScale | None" = None,
        target: Any | None = None,
        figsize: tuple[float, float] | None = None,
        theme: str = "notebook",
        show: bool = True,
    ) -> "PlotResult":
        """Summarize feature expression as a matrix plot."""
        from ..plotting import matrixplot

        return matrixplot(
            self._store,
            features=features,
            group_by=group_by,
            cell_key=cell_key,
            from_assay=from_assay,
            sample_by=sample_by,
            study_design=study_design,
            normalization=normalization,
            expression_cutoff=expression_cutoff,
            value=value,
            standardize=standardize,
            color_scale=color_scale,
            target=target,
            figsize=figsize,
            theme=theme,
            show=show,
        )

    def composition(
        self,
        *,
        category_by: str,
        cell_key: str = "I",
        sample_by: str | None = None,
        subject_by: str | None = None,
        pair_by: str | None = None,
        condition_by: str | None = None,
        study_design: "StudyDesign | None" = None,
        kind: Literal["stacked", "per_sample"] = "stacked",
        categorical_scale: "CategoricalScale | None" = None,
        target: Any | None = None,
        figsize: tuple[float, float] | None = None,
        theme: str = "notebook",
        show: bool = True,
    ) -> "PlotResult":
        """Plot category composition for the selected cells."""
        from ..plotting import composition

        return composition(
            self._store,
            category_by=category_by,
            cell_key=cell_key,
            sample_by=sample_by,
            subject_by=subject_by,
            pair_by=pair_by,
            condition_by=condition_by,
            study_design=study_design,
            kind=kind,
            categorical_scale=categorical_scale,
            target=target,
            figsize=figsize,
            theme=theme,
            show=show,
        )

    def distribution(
        self,
        keys: ("str | CellField | FeatureRef | Sequence[str | CellField | FeatureRef]"),
        *,
        group_by: str | None = None,
        groups: Sequence[Any] | None = None,
        subset_by: str | None = None,
        cell_key: str | None = "I",
        from_assay: str | None = None,
        normalization: "NormalizationSpec | None" = None,
        categorical_scale: "CategoricalScale | None" = None,
        kind: "DistKind" = "violin",
        bins: int = 40,
        max_points: int = 10000,
        point_size: float = 0.8,
        seed: int = 0,
        color: str = "steelblue",
        target: Any | None = None,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
        theme: str = "notebook",
        show: bool = True,
    ) -> "PlotResult":
        """Plot distributions of cell metadata or feature values."""
        from ..plotting import distribution

        return distribution(
            self._store,
            keys,
            group_by=group_by,
            groups=groups,
            subset_by=subset_by,
            cell_key=cell_key,
            from_assay=from_assay,
            normalization=normalization,
            categorical_scale=categorical_scale,
            kind=kind,
            bins=bins,
            max_points=max_points,
            point_size=point_size,
            seed=seed,
            color=color,
            target=target,
            figsize=figsize,
            title=title,
            theme=theme,
            show=show,
        )

    def marker_heatmap(
        self,
        *,
        from_assay: str | None = None,
        group_key: str | None = None,
        cell_key: str | None = None,
        topn: int = 5,
        log_transform: bool = True,
        vmin: float = -1,
        vmax: float = 2,
        figsize: tuple[float, float] | None = None,
        fontsize: float = 10,
        width_factor: float = 0.03,
        height_factor: float = 0.02,
        cmap: Any = "magma_r",
        theme: str = "notebook",
        show: bool = True,
        **heatmap_kwargs: Any,
    ) -> "PlotResult":
        """Plot the stored marker table as a heatmap."""
        from ..plotting import marker_heatmap

        return marker_heatmap(
            self._store,
            from_assay=from_assay,
            group_key=group_key,
            cell_key=cell_key,
            topn=topn,
            log_transform=log_transform,
            vmin=vmin,
            vmax=vmax,
            figsize=figsize,
            fontsize=fontsize,
            width_factor=width_factor,
            height_factor=height_factor,
            cmap=cmap,
            theme=theme,
            show=show,
            **heatmap_kwargs,
        )

    def cluster_tree(
        self,
        *,
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        integrated_graph: str | None = None,
        cluster_key: str | None = None,
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
            from_assay=from_assay,
            cell_key=cell_key,
            feat_key=feat_key,
            integrated_graph=integrated_graph,
            cluster_key=cluster_key,
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
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        feature_cluster_key: str | None = None,
        pseudotime_key: str | None = None,
        show_features: list[str] | None = None,
        figsize: tuple[float, float] = (5, 10),
        vmin: float = -2.0,
        vmax: float = 2.0,
        heatmap_cmap: str | None = None,
        pseudotime_cmap: str | None = None,
        clusterbar_cmap: str | None = None,
        tick_fontsize: int = 10,
        axis_fontsize: int = 12,
        feature_label_fontsize: int = 12,
        theme: str = "notebook",
        show: bool = True,
    ) -> "PlotResult":
        """Plot feature profiles ordered by stored pseudotime."""
        from ..plotting import pseudotime_heatmap

        return pseudotime_heatmap(
            self._store,
            from_assay=from_assay,
            cell_key=cell_key,
            feat_key=feat_key,
            feature_cluster_key=feature_cluster_key,
            pseudotime_key=pseudotime_key,
            show_features=show_features,
            figsize=figsize,
            vmin=vmin,
            vmax=vmax,
            heatmap_cmap=heatmap_cmap,
            pseudotime_cmap=pseudotime_cmap,
            clusterbar_cmap=clusterbar_cmap,
            tick_fontsize=tick_fontsize,
            axis_fontsize=axis_fontsize,
            feature_label_fontsize=feature_label_fontsize,
            theme=theme,
            show=show,
        )
