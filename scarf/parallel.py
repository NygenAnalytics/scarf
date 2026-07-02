"""Generic shard-parallel processing over row-banded arrays.

Both primitives process shards in order with a shallow read-ahead so the next
band downloads while the current one is worked on; the worker budget is spent
parallelising inner-chunk IO *within* a shard (BLAS stays single-threaded for
reproducibility) rather than fanning out over many shards at once (which only
inflates peak memory).

- ``map_shards``: budget-aware ordered map over ``(start, end)`` row ranges,
  used by ChunkedArray reductions/compute and the write pipeline. It sets Zarr
  ``async.concurrency`` and per-shard BLAS threads from the plan and preserves
  result order.
- ``stream_shards``: an ordered, bounded read-ahead generator used by
  sequential consumers (PCA / k-means / ANN fitting) that must see blocks in
  order while the next one is fetched in the background.

Threads are the default backend: the per-shard hot paths (numpy ufuncs, BLAS
``dot``, scipy sparse, hnswlib) release the GIL, so a thread pool already uses
multiple cores without process pickling or memory blow-up. A ``serial`` backend
exists for tiny inputs and tests, and as the seam for a future process backend.

A thread-local ``active`` flag guards against nested fan-out: a ``map_shards``
invoked from inside another shard worker runs serially so total in-flight
requests stay bounded.
"""

import threading
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from typing import Any, Literal

from threadpoolctl import threadpool_limits

from .storage.budget import ShardPlan, shard_parallelism

__all__ = ["map_shards", "stream_shards", "in_shard_context"]

type Backend = Literal["thread", "serial"]
type RangeProduce = Callable[[int, int, int], Any]

_shard_ctx = threading.local()


def in_shard_context() -> bool:
    """True when the caller is already running inside a shard worker."""
    return bool(getattr(_shard_ctx, "active", False))


@contextmanager
def _shard_context() -> Iterator[None]:
    prev = getattr(_shard_ctx, "active", False)
    _shard_ctx.active = True
    try:
        yield
    finally:
        _shard_ctx.active = prev


@contextmanager
def _io_concurrency(io: int | None) -> Iterator[None]:
    if io is None:
        yield
        return
    import zarr

    with zarr.config.set({"async.concurrency": max(1, int(io))}):
        yield


def _blas_limit(within: int | None) -> Any:
    """Process-global BLAS/OpenMP cap for the whole pool (set once, not per task).

    BLAS thread counts are process-wide, so entering ``threadpool_limits`` once
    in the driving thread bounds every worker without the per-task overhead and
    concurrent enter/exit races of wrapping each task.
    """
    if within is None or within < 1:
        return nullcontext()
    return threadpool_limits(limits=within)


def _imap_ordered(
    items: Iterable[Any],
    fn: Callable[[Any], Any],
    *,
    workers: int,
    within_block_threads: int | None,
) -> Iterator[Any]:
    """Ordered map with a bounded rolling window of ``workers`` in-flight tasks."""

    def worker(item: Any) -> Any:
        _shard_ctx.active = True
        return fn(item)

    iterator = iter(items)
    with (
        _blas_limit(within_block_threads),
        ThreadPoolExecutor(max_workers=workers) as ex,
    ):
        pending: deque[Future[Any]] = deque()

        def enqueue() -> bool:
            try:
                item = next(iterator)
            except StopIteration:
                return False
            pending.append(ex.submit(worker, item))
            return True

        for _ in range(workers):
            if not enqueue():
                break
        while pending:
            result = pending.popleft().result()
            enqueue()
            yield result


def _resolve_plan(
    workers: int | None, n_shards: int | None, backend: Backend
) -> ShardPlan:
    if backend == "serial" or in_shard_context():
        return ShardPlan(readAhead=1, ioConcurrency=1, withinBlockThreads=1)
    return shard_parallelism(workers=workers, n_shards=n_shards)


def _progress(it: Iterator[Any], msg: str | None, total: int | None) -> Iterator[Any]:
    if msg is None:
        yield from it
        return
    from .utils import tqdmbar

    yield from tqdmbar(it, desc=msg, total=total)


def stream_shards(
    items: Iterable[Any],
    fn: Callable[[Any], Any],
    *,
    workers: int,
    within_block_threads: int | None = None,
    io_concurrency: int | None = None,
    msg: str | None = None,
    total: int | None = None,
    backend: Backend = "thread",
) -> Iterator[Any]:
    """Yield ``fn(item)`` in order with bounded read-ahead.

    Meant for ordered streaming into sequential consumers. Falls back to a
    serial pass when ``workers <= 1``, the backend is ``serial``, or already
    inside a shard context. ``io_concurrency`` optionally caps Zarr
    ``async.concurrency`` for the lifetime of the stream so that
    ``workers * io_concurrency`` in-flight requests stays bounded on a remote
    store; leave it ``None`` to be config-neutral.
    """
    workers = max(1, int(workers))
    if backend == "serial" or workers <= 1 or in_shard_context():
        with _io_concurrency(io_concurrency), _blas_limit(within_block_threads):
            yield from _progress((fn(item) for item in items), msg, total)
        return
    with _io_concurrency(io_concurrency):
        base = _imap_ordered(
            items, fn, workers=workers, within_block_threads=within_block_threads
        )
        yield from _progress(base, msg, total)


def map_shards(
    ranges: list[tuple[int, int]],
    produce: RangeProduce,
    *,
    workers: int | None = None,
    msg: str | None = None,
    backend: Backend = "thread",
) -> list[Any]:
    """Map ``produce(block_idx, start, end)`` over row ranges, preserving order.

    Budget-aware: derives a :class:`ShardPlan` and, for the duration of the
    run, sets Zarr ``async.concurrency`` to the plan's IO concurrency and bounds
    per-shard BLAS threads. Returns results in ``ranges`` order so downstream
    reductions keep their current float-summation order.
    """
    n = len(ranges)
    if n == 0:
        return []
    plan = _resolve_plan(workers, n, backend)
    indexed = list(enumerate(ranges))

    def call(item: tuple[int, tuple[int, int]]) -> Any:
        idx, (start, end) = item
        return produce(idx, start, end)

    if plan.readAhead <= 1:
        with _io_concurrency(plan.ioConcurrency), _blas_limit(plan.withinBlockThreads):
            return list(_progress((call(it) for it in indexed), msg, n))

    with (
        _shard_context(),
        _io_concurrency(plan.ioConcurrency),
    ):
        stream = _imap_ordered(
            indexed,
            call,
            workers=plan.readAhead,
            within_block_threads=plan.withinBlockThreads,
        )
        return list(_progress(stream, msg, n))
