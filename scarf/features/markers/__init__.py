from importlib import import_module as _import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .batching import (
        resolve_marker_gene_batch_size as resolve_marker_gene_batch_size,
    )
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
    "resolve_marker_gene_batch_size",
    "sort_marker_results",
]

_LAZY_EXPORTS = {
    "find_markers_by_rank": (".search", "find_markers_by_rank"),
    "find_markers_by_regression": (".search", "find_markers_by_regression"),
    "mannwhitneyu_from_ranks": (".rank", "mannwhitneyu_from_ranks"),
    "resolve_marker_gene_batch_size": (".batching", "resolve_marker_gene_batch_size"),
    "sort_marker_results": (".rank", "sort_marker_results"),
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
