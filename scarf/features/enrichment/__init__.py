from importlib import import_module as _import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .net import read_gmt as read_gmt
    from .results import EnrichmentResult as EnrichmentResult

__all__ = ["EnrichmentResult", "read_gmt"]

_LAZY_EXPORTS = {
    "EnrichmentResult": (".results", "EnrichmentResult"),
    "read_gmt": (".net", "read_gmt"),
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
