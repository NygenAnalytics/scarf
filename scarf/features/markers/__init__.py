from typing import TYPE_CHECKING

from ..._facade import lazy_facade as _lazy_facade

if TYPE_CHECKING:
    from .rank import (
        mannwhitneyu_from_ranks as mannwhitneyu_from_ranks,
        sort_marker_results as sort_marker_results,
    )
    from .search import (
        find_markers_by_rank as find_markers_by_rank,
        find_markers_by_regression as find_markers_by_regression,
    )

__all__ = [
    "find_markers_by_rank",
    "find_markers_by_regression",
    "mannwhitneyu_from_ranks",
    "sort_marker_results",
]

__getattr__, __dir__ = _lazy_facade(
    __name__,
    {
        "find_markers_by_rank": ".search",
        "find_markers_by_regression": ".search",
        "mannwhitneyu_from_ranks": ".rank",
        "sort_marker_results": ".rank",
    },
)
