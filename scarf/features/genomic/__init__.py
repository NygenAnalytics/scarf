from typing import TYPE_CHECKING

from ..._facade import lazy_facade as _lazy_facade

if TYPE_CHECKING:
    from .intervals import (
        binary_search as binary_search,
        create_bed_from_coord_ids as create_bed_from_coord_ids,
        get_feature_mappings as get_feature_mappings,
        get_ranges as get_ranges,
    )
    from .melding import create_counts_mat as create_counts_mat

__all__ = [
    "binary_search",
    "create_bed_from_coord_ids",
    "create_counts_mat",
    "get_feature_mappings",
    "get_ranges",
]

__getattr__, __dir__ = _lazy_facade(
    __name__,
    {
        "binary_search": ".intervals",
        "create_bed_from_coord_ids": ".intervals",
        "create_counts_mat": ".melding",
        "get_feature_mappings": ".intervals",
        "get_ranges": ".intervals",
    },
)
