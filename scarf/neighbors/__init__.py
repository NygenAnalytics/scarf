from importlib import import_module as _import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .diffusion import diffusion_operator as diffusion_operator
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
    from .integration import wnn_integration as wnn_integration
    from .stream import AnnStream as AnnStream

__all__ = [
    "AnnStream",
    "calc_snn",
    "diffusion_operator",
    "fix_knn_query",
    "instantiate_knn_index",
    "merge_graphs",
    "smooth_knn_chunk",
    "weight_sort_indices",
    "wnn_integration",
]

_LAZY_EXPORTS = {
    "AnnStream": (".stream", "AnnStream"),
    "calc_snn": (".graph", "calc_snn"),
    "diffusion_operator": (".diffusion", "diffusion_operator"),
    "fix_knn_query": (".index", "fix_knn_query"),
    "instantiate_knn_index": (".index", "instantiate_knn_index"),
    "merge_graphs": (".graph", "merge_graphs"),
    "smooth_knn_chunk": (".graph", "smooth_knn_chunk"),
    "weight_sort_indices": (".graph", "weight_sort_indices"),
    "wnn_integration": (".integration", "wnn_integration"),
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
