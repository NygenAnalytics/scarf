from importlib import import_module as _import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .feature_dynamics import (
        aggregate_feature_profiles as aggregate_feature_profiles,
        scatter_feature_clusters as scatter_feature_clusters,
        validate_pseudotime_regressor as validate_pseudotime_regressor,
    )
    from .pseudotime import (
        make_source_sink_vector as make_source_sink_vector,
        random_walk_laplacian_transpose as random_walk_laplacian_transpose,
        select_pseudotime_component as select_pseudotime_component,
        truncated_pba_potential as truncated_pba_potential,
        validate_source_sink_labels as validate_source_sink_labels,
        validate_source_sink_vector as validate_source_sink_vector,
    )
    from .results import (
        FateMappingResult as FateMappingResult,
        PseudotimeAggregationResult as PseudotimeAggregationResult,
        PseudotimeMarkerResult as PseudotimeMarkerResult,
        PseudotimeScoreResult as PseudotimeScoreResult,
    )

__all__ = [
    "FateMappingResult",
    "PseudotimeAggregationResult",
    "PseudotimeMarkerResult",
    "PseudotimeScoreResult",
    "aggregate_feature_profiles",
    "make_source_sink_vector",
    "random_walk_laplacian_transpose",
    "select_pseudotime_component",
    "scatter_feature_clusters",
    "truncated_pba_potential",
    "validate_source_sink_labels",
    "validate_source_sink_vector",
    "validate_pseudotime_regressor",
]

_LAZY_EXPORTS = {
    "aggregate_feature_profiles": (
        ".feature_dynamics",
        "aggregate_feature_profiles",
    ),
    "make_source_sink_vector": (
        ".pseudotime",
        "make_source_sink_vector",
    ),
    "random_walk_laplacian_transpose": (
        ".pseudotime",
        "random_walk_laplacian_transpose",
    ),
    "select_pseudotime_component": (
        ".pseudotime",
        "select_pseudotime_component",
    ),
    "scatter_feature_clusters": (
        ".feature_dynamics",
        "scatter_feature_clusters",
    ),
    "truncated_pba_potential": (
        ".pseudotime",
        "truncated_pba_potential",
    ),
    "validate_source_sink_labels": (
        ".pseudotime",
        "validate_source_sink_labels",
    ),
    "validate_source_sink_vector": (
        ".pseudotime",
        "validate_source_sink_vector",
    ),
    "validate_pseudotime_regressor": (
        ".feature_dynamics",
        "validate_pseudotime_regressor",
    ),
    "FateMappingResult": (
        ".results",
        "FateMappingResult",
    ),
    "PseudotimeAggregationResult": (
        ".results",
        "PseudotimeAggregationResult",
    ),
    "PseudotimeMarkerResult": (".results", "PseudotimeMarkerResult"),
    "PseudotimeScoreResult": (".results", "PseudotimeScoreResult"),
}

for _export_name in _LAZY_EXPORTS:
    globals().pop(_export_name, None)
del _export_name


def __getattr__(name: str) -> Any:
    export = _LAZY_EXPORTS.get(name)
    if export is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = export
    value = getattr(_import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(_LAZY_EXPORTS))
