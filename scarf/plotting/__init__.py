from typing import TYPE_CHECKING

from .._facade import lazy_facade as _lazy_facade

if TYPE_CHECKING:
    from ._contracts import (
        CategoricalScale as CategoricalScale,
        CellField as CellField,
        ColorScale as ColorScale,
        DensityOverlay as DensityOverlay,
        FeatureRef as FeatureRef,
        Highlight as Highlight,
        NormalizationSpec as NormalizationSpec,
        PlotProvenance as PlotProvenance,
        SizeScale as SizeScale,
        StudyDesign as StudyDesign,
    )
    from ._figure import (
        LegendSpec as LegendSpec,
        PlotResult as PlotResult,
        compose_results as compose_results,
        label_panels as label_panels,
    )
    from ._style import THEMES as THEMES
    from ._style import theme_context as theme_context
    from .recipes import (
        PlotOutput as PlotOutput,
        PlotOutputSettings as PlotOutputSettings,
        PlotPanelTarget as PlotPanelTarget,
        PlotRecipe as PlotRecipe,
        PlotRecipeResult as PlotRecipeResult,
        PlotStep as PlotStep,
        run_recipe as run_recipe,
    )
    from .composition import composition as composition
    from .cluster_connectivity import cluster_connectivity as cluster_connectivity
    from .diagnostics import (
        elbow as elbow,
        graph_qc as graph_qc,
        highly_variable_features as highly_variable_features,
        qc as qc,
    )
    from .distribution import distribution as distribution
    from .embedding import embedding as embedding
    from .embedding_raster import embedding_raster as embedding_raster
    from .cluster_tree import cluster_tree as cluster_tree
    from .heatmaps import (
        marker_heatmap as marker_heatmap,
        pseudotime_heatmap as pseudotime_heatmap,
    )
    from .mapping import (
        mapping_calibration as mapping_calibration,
        mapping_confusion as mapping_confusion,
        mapping_evidence as mapping_evidence,
        mapping_score as mapping_score,
    )
    from .modality_weights import modality_weights as modality_weights
    from .summary import dotplot as dotplot
    from .summary import matrixplot as matrixplot

__all__ = [
    "CategoricalScale",
    "CellField",
    "ColorScale",
    "DensityOverlay",
    "FeatureRef",
    "Highlight",
    "LegendSpec",
    "NormalizationSpec",
    "PlotProvenance",
    "PlotOutput",
    "PlotOutputSettings",
    "PlotPanelTarget",
    "PlotRecipe",
    "PlotRecipeResult",
    "PlotResult",
    "PlotStep",
    "SizeScale",
    "StudyDesign",
    "THEMES",
    "cluster_tree",
    "cluster_connectivity",
    "compose_results",
    "composition",
    "distribution",
    "dotplot",
    "elbow",
    "embedding",
    "embedding_raster",
    "graph_qc",
    "highly_variable_features",
    "label_panels",
    "marker_heatmap",
    "mapping_calibration",
    "mapping_confusion",
    "mapping_evidence",
    "mapping_score",
    "matrixplot",
    "modality_weights",
    "pseudotime_heatmap",
    "qc",
    "run_recipe",
    "theme_context",
]

__getattr__, __dir__ = _lazy_facade(
    __name__,
    {
        "CategoricalScale": "._contracts",
        "CellField": "._contracts",
        "ColorScale": "._contracts",
        "DensityOverlay": "._contracts",
        "FeatureRef": "._contracts",
        "Highlight": "._contracts",
        "LegendSpec": "._figure",
        "NormalizationSpec": "._contracts",
        "PlotProvenance": "._contracts",
        "PlotOutput": ".recipes",
        "PlotOutputSettings": ".recipes",
        "PlotPanelTarget": ".recipes",
        "PlotRecipe": ".recipes",
        "PlotRecipeResult": ".recipes",
        "PlotResult": "._figure",
        "PlotStep": ".recipes",
        "SizeScale": "._contracts",
        "StudyDesign": "._contracts",
        "THEMES": "._style",
        "cluster_tree": ".cluster_tree",
        "cluster_connectivity": ".cluster_connectivity",
        "compose_results": "._figure",
        "composition": ".composition",
        "distribution": ".distribution",
        "dotplot": ".summary",
        "elbow": ".diagnostics",
        "embedding": ".embedding",
        "embedding_raster": ".embedding_raster",
        "graph_qc": ".diagnostics",
        "highly_variable_features": ".diagnostics",
        "label_panels": "._figure",
        "marker_heatmap": ".heatmaps",
        "mapping_calibration": ".mapping",
        "mapping_confusion": ".mapping",
        "mapping_evidence": ".mapping",
        "mapping_score": ".mapping",
        "matrixplot": ".summary",
        "modality_weights": ".modality_weights",
        "pseudotime_heatmap": ".heatmaps",
        "qc": ".diagnostics",
        "run_recipe": ".recipes",
        "theme_context": "._style",
    },
)
