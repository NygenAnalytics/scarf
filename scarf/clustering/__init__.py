from typing import TYPE_CHECKING

from .._facade import lazy_facade as _lazy_facade

if TYPE_CHECKING:
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
    from .paris import straight_cut as straight_cut

__all__ = [
    "CoalesceTree",
    "ParisClusterDiagnostic",
    "ParisClusteringResult",
    "adaptive_cut",
    "leiden_membership",
    "make_digraph",
    "straight_cut",
]

__getattr__, __dir__ = _lazy_facade(
    __name__,
    {
        "CoalesceTree": ".cluster_tree",
        "ParisClusterDiagnostic": ".paris_multiscale",
        "ParisClusteringResult": ".paris_multiscale",
        "adaptive_cut": ".paris_multiscale",
        "leiden_membership": ".leiden",
        "make_digraph": ".cluster_tree",
        "straight_cut": ".paris",
    },
)
