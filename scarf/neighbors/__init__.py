from typing import TYPE_CHECKING

from .._facade import lazy_facade as _lazy_facade

if TYPE_CHECKING:
    from .graph import (
        calc_snn as calc_snn,
        merge_graphs as merge_graphs,
        smooth_knn_chunk as smooth_knn_chunk,
        weight_sort_indices as weight_sort_indices,
    )
    from .index import (
        fix_knn_query as fix_knn_query,
        instantiate_knn_index as instantiate_knn_index,
    )

__all__ = [
    "calc_snn",
    "fix_knn_query",
    "instantiate_knn_index",
    "merge_graphs",
    "smooth_knn_chunk",
    "weight_sort_indices",
]

__getattr__, __dir__ = _lazy_facade(
    __name__,
    {
        "calc_snn": ".graph",
        "fix_knn_query": ".index",
        "instantiate_knn_index": ".index",
        "merge_graphs": ".graph",
        "smooth_knn_chunk": ".graph",
        "weight_sort_indices": ".graph",
    },
)
