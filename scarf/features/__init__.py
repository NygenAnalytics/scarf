from typing import TYPE_CHECKING

from .._facade import lazy_facade as _lazy_facade

if TYPE_CHECKING:
    from .enrichment import (
        EnrichmentResult as EnrichmentResult,
        read_gmt as read_gmt,
    )
    from .genomic import (
        binary_search as binary_search,
        create_bed_from_coord_ids as create_bed_from_coord_ids,
        create_counts_mat as create_counts_mat,
        get_feature_mappings as get_feature_mappings,
        get_ranges as get_ranges,
    )
    from .markers import (
        RankMarkerResult as RankMarkerResult,
        find_markers_by_rank as find_markers_by_rank,
        find_markers_by_regression as find_markers_by_regression,
        mannwhitneyu_from_ranks as mannwhitneyu_from_ranks,
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
    "GroupComparisonResult",
    "RankMarkerResult",
    "StatisticalTestResult",
    "adjust_pvalues",
    "aggregate_samples",
    "binary_search",
    "binned_sampling",
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
    "select_highly_variable_features",
    "sort_marker_results",
]

__getattr__, __dir__ = _lazy_facade(
    __name__,
    {
        "EnrichmentResult": ".enrichment",
        "GroupComparisonResult": ".statistical",
        "RankMarkerResult": ".markers",
        "StatisticalTestResult": ".statistical",
        "adjust_pvalues": ".statistical",
        "aggregate_samples": ".statistical",
        "binary_search": ".genomic",
        "binned_sampling": ".scoring",
        "compare_group_distributions": ".statistical",
        "create_bed_from_coord_ids": ".genomic",
        "create_counts_mat": ".genomic",
        "find_markers_by_rank": ".markers",
        "find_markers_by_regression": ".markers",
        "fit_lowess": ".variability",
        "get_feature_mappings": ".genomic",
        "get_ranges": ".genomic",
        "mannwhitneyu_from_ranks": ".markers",
        "read_gmt": ".enrichment",
        "resolve_group_order": ".statistical",
        "select_highly_variable_features": ".variability",
        "sort_marker_results": ".markers",
    },
)
