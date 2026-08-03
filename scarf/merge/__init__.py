"""Methods and classes for merging datasets."""

import sys
from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .datasets import DataStoreMerge as DataStoreMerge
    from .models import (
        AssayMergePlan as AssayMergePlan,
        ComponentResult as ComponentResult,
        MergePlan as MergePlan,
        MergeResult as MergeResult,
    )

__all__ = [
    "DataStoreMerge",
    "MergePlan",
    "MergeResult",
    "AssayMergePlan",
    "ComponentResult",
]

_LAZY_EXPORTS = {
    "DataStoreMerge": "datasets",
    "MergePlan": "models",
    "MergeResult": "models",
    "AssayMergePlan": "models",
    "ComponentResult": "models",
}

for _export_name in _LAZY_EXPORTS:
    globals().pop(_export_name, None)
del _export_name

_CLASS_EXPORTS = frozenset(
    {
        "DataStoreMerge",
        "MergePlan",
        "MergeResult",
        "AssayMergePlan",
        "ComponentResult",
    }
)


def _normalize_class_metadata(value: type[Any]) -> None:
    value.__module__ = __name__
    for descriptor in value.__dict__.values():
        if isinstance(descriptor, (classmethod, staticmethod)):
            method = descriptor.__func__
        else:
            method = descriptor
        if callable(method) and hasattr(method, "__module__"):
            method.__module__ = __name__


def _cache_loaded_exports() -> None:
    for export_name, implementation_name in _LAZY_EXPORTS.items():
        module = sys.modules.get(f"{__name__}.{implementation_name}")
        if module is None or not hasattr(module, export_name):
            continue
        value = getattr(module, export_name)
        if export_name in _CLASS_EXPORTS:
            _normalize_class_metadata(value)
        globals().setdefault(export_name, value)


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import_module(f".{module_name}", __name__)
    _cache_loaded_exports()
    return globals()[name]


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
