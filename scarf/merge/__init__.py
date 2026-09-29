"""Methods and classes for merging datasets."""

from typing import TYPE_CHECKING

from .._facade import lazy_facade as _lazy_facade

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

__getattr__, __dir__ = _lazy_facade(
    __name__,
    {
        "DataStoreMerge": ".datasets",
        "MergePlan": ".models",
        "MergeResult": ".models",
        "AssayMergePlan": ".models",
        "ComponentResult": ".models",
    },
    set_module=True,
)
