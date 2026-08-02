from importlib import import_module as _import_module
from types import ModuleType as _ModuleType
from typing import TYPE_CHECKING, Any
import sys as _sys

if TYPE_CHECKING:
    from ._contracts import (
        CategoricalScale as CategoricalScale,
        CellField as CellField,
        ColorScale as ColorScale,
        DensityOverlay as DensityOverlay,
        FeatureRef as FeatureRef,
        FeatureSummary as FeatureSummary,
        Highlight as Highlight,
        NormalizationSpec as NormalizationSpec,
        PlotProvenance as PlotProvenance,
        SizeScale as SizeScale,
        StudyDesign as StudyDesign,
    )
    from ._figure import (
        LegendSpec as LegendSpec,
        PlotResult as PlotResult,
        collect_legends as collect_legends,
        compose_results as compose_results,
        label_panels as label_panels,
    )
    from ._style import THEMES as THEMES
    from ._style import register_theme as register_theme
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
        mapping_projection as mapping_projection,
        mapping_score as mapping_score,
    )
    from .summary import dotplot as dotplot
    from .summary import matrixplot as matrixplot

__all__ = [
    "CategoricalScale",
    "CellField",
    "ColorScale",
    "DensityOverlay",
    "FeatureRef",
    "FeatureSummary",
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
    "collect_legends",
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
    "mapping_projection",
    "mapping_score",
    "matrixplot",
    "pseudotime_heatmap",
    "qc",
    "register_theme",
    "run_recipe",
    "theme_context",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "CategoricalScale": ("._contracts", "CategoricalScale"),
    "CellField": ("._contracts", "CellField"),
    "ColorScale": ("._contracts", "ColorScale"),
    "DensityOverlay": ("._contracts", "DensityOverlay"),
    "FeatureRef": ("._contracts", "FeatureRef"),
    "FeatureSummary": ("._contracts", "FeatureSummary"),
    "Highlight": ("._contracts", "Highlight"),
    "LegendSpec": ("._figure", "LegendSpec"),
    "NormalizationSpec": ("._contracts", "NormalizationSpec"),
    "PlotProvenance": ("._contracts", "PlotProvenance"),
    "PlotOutput": (".recipes", "PlotOutput"),
    "PlotOutputSettings": (".recipes", "PlotOutputSettings"),
    "PlotPanelTarget": (".recipes", "PlotPanelTarget"),
    "PlotRecipe": (".recipes", "PlotRecipe"),
    "PlotRecipeResult": (".recipes", "PlotRecipeResult"),
    "PlotResult": ("._figure", "PlotResult"),
    "PlotStep": (".recipes", "PlotStep"),
    "SizeScale": ("._contracts", "SizeScale"),
    "StudyDesign": ("._contracts", "StudyDesign"),
    "THEMES": ("._style", "THEMES"),
    "cluster_tree": (".cluster_tree", "cluster_tree"),
    "cluster_connectivity": (".cluster_connectivity", "cluster_connectivity"),
    "collect_legends": ("._figure", "collect_legends"),
    "compose_results": ("._figure", "compose_results"),
    "composition": (".composition", "composition"),
    "distribution": (".distribution", "distribution"),
    "dotplot": (".summary", "dotplot"),
    "elbow": (".diagnostics", "elbow"),
    "embedding": (".embedding", "embedding"),
    "embedding_raster": (".embedding_raster", "embedding_raster"),
    "graph_qc": (".diagnostics", "graph_qc"),
    "highly_variable_features": (".diagnostics", "highly_variable_features"),
    "label_panels": ("._figure", "label_panels"),
    "marker_heatmap": (".heatmaps", "marker_heatmap"),
    "mapping_calibration": (".mapping", "mapping_calibration"),
    "mapping_confusion": (".mapping", "mapping_confusion"),
    "mapping_evidence": (".mapping", "mapping_evidence"),
    "mapping_projection": (".mapping", "mapping_projection"),
    "mapping_score": (".mapping", "mapping_score"),
    "matrixplot": (".summary", "matrixplot"),
    "pseudotime_heatmap": (".heatmaps", "pseudotime_heatmap"),
    "qc": (".diagnostics", "qc"),
    "register_theme": ("._style", "register_theme"),
    "run_recipe": (".recipes", "run_recipe"),
    "theme_context": ("._style", "theme_context"),
}

for _export_name in _LAZY_EXPORTS:
    globals().pop(_export_name, None)
del _export_name


def _load_export(name: str) -> Any:
    export = _LAZY_EXPORTS.get(name)
    if export is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = export
    value = getattr(_import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __getattr__(name: str) -> Any:
    return _load_export(name)


def __dir__() -> list[str]:
    return sorted(set(globals()).union(_LAZY_EXPORTS))


class _PlottingFacadeModule(_ModuleType):
    def __getattribute__(self, name: str) -> Any:
        namespace = _ModuleType.__getattribute__(self, "__dict__")
        if name in namespace.get("_LAZY_EXPORTS", {}):
            current = namespace.get(name)
            if isinstance(current, _ModuleType):
                return namespace["_load_export"](name)
        return _ModuleType.__getattribute__(self, name)


_sys.modules[__name__].__class__ = _PlottingFacadeModule
