from importlib import import_module as _import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .balanced_cut import BalancedCut as BalancedCut
    from .cluster_tree import (
        CoalesceTree as CoalesceTree,
        make_digraph as make_digraph,
    )
    from .leiden import leiden_membership as leiden_membership
    from .paris_multiscale import (
        ParisClusterDiagnostic as ParisClusterDiagnostic,
        ParisClusteringResult as ParisClusteringResult,
        adaptive_cut as adaptive_cut,
    )
    from .paris import (
        balanced_cut as balanced_cut,
        paris_dendrogram as paris_dendrogram,
        straight_cut as straight_cut,
    )

__all__ = [
    "BalancedCut",
    "CoalesceTree",
    "ParisClusterDiagnostic",
    "ParisClusteringResult",
    "adaptive_cut",
    "balanced_cut",
    "leiden_membership",
    "make_digraph",
    "paris_dendrogram",
    "straight_cut",
]

_LAZY_EXPORTS = {
    "BalancedCut": (".balanced_cut", "BalancedCut"),
    "CoalesceTree": (".cluster_tree", "CoalesceTree"),
    "ParisClusterDiagnostic": (".paris_multiscale", "ParisClusterDiagnostic"),
    "ParisClusteringResult": (".paris_multiscale", "ParisClusteringResult"),
    "adaptive_cut": (".paris_multiscale", "adaptive_cut"),
    "balanced_cut": (".paris", "balanced_cut"),
    "leiden_membership": (".leiden", "leiden_membership"),
    "make_digraph": (".cluster_tree", "make_digraph"),
    "paris_dendrogram": (".paris", "paris_dendrogram"),
    "straight_cut": (".paris", "straight_cut"),
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
