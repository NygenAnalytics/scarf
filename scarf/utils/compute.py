from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import numpy as np

from .logging import progress_enabled

T = TypeVar("T")


def controlled_compute(arr: Any, nthreads: int) -> np.ndarray:
    """Materialize a deferred array with a bounded thread count."""
    if hasattr(arr, "compute"):
        return np.asarray(arr.compute(nthreads))
    return np.asarray(arr)


def show_dask_progress(
    arr: Any,
    msg: str | None = None,
    nthreads: int = 1,
) -> np.ndarray:
    """Materialize a deferred array while reporting progress."""
    if hasattr(arr, "compute"):
        progress_msg = msg if progress_enabled() else None
        return np.asarray(arr.compute(nthreads, progress_msg))
    return np.asarray(arr)


def pairwise_merge_tree(values: Sequence[T], merge: Callable[[T, T], T]) -> T:
    """Reduce values with a fixed pairwise tree independent of completion order.

    Position in ``values`` is the unit index. Adjacent pairs merge first, then
    the next level, so the association does not depend on which unit finished
    first. An odd leftover at a level is promoted unchanged.
    """
    if not values:
        raise ValueError("merge tree needs at least one value")
    items = list(values)
    while len(items) > 1:
        nxt: list[T] = []
        for index in range(0, len(items), 2):
            if index + 1 < len(items):
                nxt.append(merge(items[index], items[index + 1]))
            else:
                nxt.append(items[index])
        items = nxt
    return items[0]


def add_stat_arrays(
    left: tuple[np.ndarray, ...],
    right: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, ...]:
    """Add aligned accumulator arrays for a merge tree."""
    if len(left) != len(right):
        raise ValueError("stat tuples must have the same length")
    return tuple(a + b for a, b in zip(left, right, strict=True))
