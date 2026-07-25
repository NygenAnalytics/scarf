from importlib import import_module as _import_module
from types import ModuleType as _ModuleType
from typing import TYPE_CHECKING, Any
import sys as _sys

if TYPE_CHECKING:
    from ._contracts import (
        CategoricalScale as CategoricalScale,
        CellField as CellField,
        ColorScale as ColorScale,
        FeatureRef as FeatureRef,
        FeatureSummary as FeatureSummary,
        NormalizationSpec as NormalizationSpec,
        PlotProvenance as PlotProvenance,
        SizeScale as SizeScale,
        StudyDesign as StudyDesign,
    )
    from ._figure import (
        LegendSpec as LegendSpec,
        PlotResult as PlotResult,
        collect_legends as collect_legends,
        label_panels as label_panels,
    )
    from ._style import THEMES as THEMES
    from ._style import theme_context as theme_context
    from .composition import composition as composition
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
    from .summary import dotplot as dotplot
    from .summary import matrixplot as matrixplot
    from .unified import unified_embedding as unified_embedding

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

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "CategoricalScale": ("._contracts", "CategoricalScale"),
    "CellField": ("._contracts", "CellField"),
    "ColorScale": ("._contracts", "ColorScale"),
    "FeatureRef": ("._contracts", "FeatureRef"),
    "FeatureSummary": ("._contracts", "FeatureSummary"),
    "LegendSpec": ("._figure", "LegendSpec"),
    "NormalizationSpec": ("._contracts", "NormalizationSpec"),
    "PlotProvenance": ("._contracts", "PlotProvenance"),
    "PlotResult": ("._figure", "PlotResult"),
    "SizeScale": ("._contracts", "SizeScale"),
    "StudyDesign": ("._contracts", "StudyDesign"),
    "THEMES": ("._style", "THEMES"),
    "cluster_tree": (".cluster_tree", "cluster_tree"),
    "collect_legends": ("._figure", "collect_legends"),
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
    "matrixplot": (".summary", "matrixplot"),
    "pseudotime_heatmap": (".heatmaps", "pseudotime_heatmap"),
    "qc": (".diagnostics", "qc"),
    "theme_context": ("._style", "theme_context"),
    "unified_embedding": (".unified", "unified_embedding"),
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
