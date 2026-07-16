from ._contracts import (
    CategoricalScale,
    CellField,
    ColorScale,
    FeatureRef,
    FeatureSummary,
    NormalizationSpec,
    PlotProvenance,
    SizeScale,
    StudyDesign,
)
from ._figure import LegendSpec, PlotResult, collect_legends, label_panels
from ._style import THEMES, theme_context
from .composition import composition
from .diagnostics import elbow, graph_qc, highly_variable_features, qc
from .distribution import distribution
from .embedding import embedding
from .embedding_raster import embedding_raster
from .heatmaps import cluster_tree, marker_heatmap, pseudotime_heatmap
from .summary import dotplot, matrixplot
from .unified import unified_embedding

__all__ = [
    "CategoricalScale",
    "CellField",
    "ColorScale",
    "FeatureRef",
    "FeatureSummary",
    "LegendSpec",
    "NormalizationSpec",
    "PlotProvenance",
    "PlotResult",
    "SizeScale",
    "StudyDesign",
    "THEMES",
    "cluster_tree",
    "collect_legends",
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
    "matrixplot",
    "pseudotime_heatmap",
    "qc",
    "theme_context",
    "unified_embedding",
]
