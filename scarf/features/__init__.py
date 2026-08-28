from importlib import import_module as _import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .enrichment import (
        EnrichmentResult as EnrichmentResult,
        read_gmt as read_gmt,
    )
    from .genomic import (
        GffReader as GffReader,
        binary_search as binary_search,
        coordinate_melding as coordinate_melding,
        create_bed_from_coord_ids as create_bed_from_coord_ids,
        create_counts_mat as create_counts_mat,
        get_feature_mappings as get_feature_mappings,
        get_ranges as get_ranges,
    )
    from .markers import (
        find_markers_by_rank as find_markers_by_rank,
        find_markers_by_regression as find_markers_by_regression,
        mannwhitneyu_from_ranks as mannwhitneyu_from_ranks,
        resolve_marker_gene_batch_size as resolve_marker_gene_batch_size,
        sort_marker_results as sort_marker_results,
    )
    from .scoring import binned_sampling as binned_sampling
    from .statistical import (
        GroupComparisonResult as GroupComparisonResult,
        StatisticalTestResult as StatisticalTestResult,
        adjust_pvalues as adjust_pvalues,
        aggregate_samples as aggregate_samples,
        compare_group_distributions as compare_group_distributions,
        resolve_group_order as resolve_group_order,
    )
    from .variability import (
        fit_lowess as fit_lowess,
        select_highly_variable_features as select_highly_variable_features,
    )

__all__ = [
    "EnrichmentResult",
    "GffReader",
    "GroupComparisonResult",
    "StatisticalTestResult",
    "adjust_pvalues",
    "aggregate_samples",
    "binary_search",
    "binned_sampling",
    "coordinate_melding",
    "compare_group_distributions",
    "create_bed_from_coord_ids",
    "create_counts_mat",
    "find_markers_by_rank",
    "find_markers_by_regression",
    "fit_lowess",
    "get_feature_mappings",
    "get_ranges",
    "mannwhitneyu_from_ranks",
    "read_gmt",
    "resolve_group_order",
    "resolve_marker_gene_batch_size",
    "select_highly_variable_features",
    "sort_marker_results",
]

_LAZY_EXPORTS = {
    "EnrichmentResult": (".enrichment", "EnrichmentResult"),
    "GffReader": (".genomic", "GffReader"),
    "GroupComparisonResult": (".statistical", "GroupComparisonResult"),
    "StatisticalTestResult": (".statistical", "StatisticalTestResult"),
    "adjust_pvalues": (".statistical", "adjust_pvalues"),
    "aggregate_samples": (".statistical", "aggregate_samples"),
    "binary_search": (".genomic", "binary_search"),
    "binned_sampling": (".scoring", "binned_sampling"),
    "coordinate_melding": (".genomic", "coordinate_melding"),
    "compare_group_distributions": (
        ".statistical",
        "compare_group_distributions",
    ),
    "create_bed_from_coord_ids": (".genomic", "create_bed_from_coord_ids"),
    "create_counts_mat": (".genomic", "create_counts_mat"),
    "find_markers_by_rank": (".markers", "find_markers_by_rank"),
    "find_markers_by_regression": (".markers", "find_markers_by_regression"),
    "fit_lowess": (".variability", "fit_lowess"),
    "get_feature_mappings": (".genomic", "get_feature_mappings"),
    "get_ranges": (".genomic", "get_ranges"),
    "mannwhitneyu_from_ranks": (".markers", "mannwhitneyu_from_ranks"),
    "read_gmt": (".enrichment", "read_gmt"),
    "resolve_group_order": (".statistical", "resolve_group_order"),
    "resolve_marker_gene_batch_size": (".markers", "resolve_marker_gene_batch_size"),
    "select_highly_variable_features": (
        ".variability",
        "select_highly_variable_features",
    ),
    "sort_marker_results": (".markers", "sort_marker_results"),
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
