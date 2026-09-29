import itertools
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, TypeVar

import numpy as np

from .logging import progress_enabled

T = TypeVar("T")

_thread_limit_lock = threading.Lock()
_thread_limit_tokens = itertools.count()
_active_thread_limits: dict[int, int | None] = {}
_original_thread_limits: Any | None = None


def enter_thread_limit(limits: int | None) -> int:
    """Start a process-wide native thread-pool limit and return its token.

    Native thread-pool limits apply to the whole process, so scopes held by
    generators or entered on other threads can end in any order. The first
    active scope records the limits in force, the most recent remaining request
    applies while scopes overlap, and the last scope to end restores the
    recorded limits. ``None`` changes nothing but still takes part.
    """
    from threadpoolctl import ThreadpoolController

    global _original_thread_limits
    with _thread_limit_lock:
        controller = ThreadpoolController()
        if not _active_thread_limits:
            _original_thread_limits = controller.limit(limits=None)
        token = next(_thread_limit_tokens)
        _active_thread_limits[token] = limits
        if limits is not None:
            controller.limit(limits=limits)
    return token


def exit_thread_limit(token: int) -> None:
    """End a limit started by ``enter_thread_limit``."""
    from threadpoolctl import ThreadpoolController

    global _original_thread_limits
    with _thread_limit_lock:
        del _active_thread_limits[token]
        original = _original_thread_limits
        if original is None:
            return
        requested = [
            limits for limits in _active_thread_limits.values() if limits is not None
        ]
        if not _active_thread_limits:
            _original_thread_limits = None
        if requested:
            ThreadpoolController().limit(limits=requested[-1])
        else:
            original.restore_original_limits()


@contextmanager
def process_thread_limit(limits: int | None) -> Iterator[None]:
    """Hold a limit from ``enter_thread_limit`` for the duration of a block."""
    token = enter_thread_limit(limits)
    try:
        yield
    finally:
        exit_thread_limit(token)


def controlled_compute(arr: Any, nthreads: int) -> np.ndarray:
    """Materialize a deferred array with a bounded thread count."""
    if hasattr(arr, "compute"):
        return np.asarray(arr.compute(nthreads))
    return np.asarray(arr)


def compute_with_progress(
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
