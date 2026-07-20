from importlib import import_module as _import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .gff import GffReader as GffReader
    from .intervals import (
        binary_search as binary_search,
        create_bed_from_coord_ids as create_bed_from_coord_ids,
        get_feature_mappings as get_feature_mappings,
        get_ranges as get_ranges,
    )
    from .melding import (
        create_counts_mat as create_counts_mat,
        coordinate_melding as coordinate_melding,
    )

__all__ = [
    "GffReader",
    "binary_search",
    "coordinate_melding",
    "create_bed_from_coord_ids",
    "create_counts_mat",
    "get_feature_mappings",
    "get_ranges",
]

_LAZY_EXPORTS = {
    "GffReader": (".gff", "GffReader"),
    "binary_search": (".intervals", "binary_search"),
    "coordinate_melding": (".melding", "coordinate_melding"),
    "create_bed_from_coord_ids": (".intervals", "create_bed_from_coord_ids"),
    "create_counts_mat": (".melding", "create_counts_mat"),
    "get_feature_mappings": (".intervals", "get_feature_mappings"),
    "get_ranges": (".intervals", "get_ranges"),
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
