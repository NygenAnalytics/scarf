import threading
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from typing import Any, Literal

from threadpoolctl import threadpool_limits

from .budget import ShardPlan, shard_parallelism

__all__ = ["map_shards", "stream_shards", "in_shard_context"]

type Backend = Literal["thread", "serial"]
type RangeProduce = Callable[[int, int, int], Any]

_shard_ctx = threading.local()


def in_shard_context() -> bool:
    """Return whether the caller is already inside a shard worker."""
    return bool(getattr(_shard_ctx, "active", False))


@contextmanager
def _shard_context() -> Iterator[None]:
    previous = getattr(_shard_ctx, "active", False)
    _shard_ctx.active = True
    try:
        yield
    finally:
        _shard_ctx.active = previous


@contextmanager
def _io_concurrency(io: int | None) -> Iterator[None]:
    if io is None:
        yield
        return
    import zarr

    with zarr.config.set({"async.concurrency": max(1, int(io))}):
        yield


def _blas_limit(within: int | None) -> Any:
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
    def worker(item: Any) -> Any:
        _shard_ctx.active = True
        return fn(item)

    iterator = iter(items)
    with (
        _blas_limit(within_block_threads),
        ThreadPoolExecutor(max_workers=workers) as executor,
    ):
        pending: deque[Future[Any]] = deque()

        def enqueue() -> bool:
            try:
                item = next(iterator)
            except StopIteration:
                return False
            pending.append(executor.submit(worker, item))
            return True

        for _ in range(workers):
            if not enqueue():
                break
        while pending:
            result = pending.popleft().result()
            enqueue()
            yield result


def _resolve_plan(
    workers: int | None,
    n_shards: int | None,
    backend: Backend,
) -> ShardPlan:
    if backend == "serial" or in_shard_context():
        return ShardPlan(readAhead=1, ioConcurrency=1, withinBlockThreads=1)
    return shard_parallelism(workers=workers, n_shards=n_shards)


def _progress(
    iterator: Iterator[Any],
    msg: str | None,
    total: int | None,
) -> Iterator[Any]:
    if msg is None:
        yield from iterator
        return
    from ..utils.progress import tqdmbar

    yield from tqdmbar(iterator, desc=msg, total=total)


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
    """Yield transformed items in order with bounded read-ahead."""
    workers = max(1, int(workers))
    if backend == "serial" or workers <= 1 or in_shard_context():
        with _io_concurrency(io_concurrency), _blas_limit(within_block_threads):
            yield from _progress((fn(item) for item in items), msg, total)
        return
    with _io_concurrency(io_concurrency):
        base = _imap_ordered(
            items,
            fn,
            workers=workers,
            within_block_threads=within_block_threads,
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
    """Map row ranges in parallel while preserving input order."""
    n_ranges = len(ranges)
    if n_ranges == 0:
        return []
    plan = _resolve_plan(workers, n_ranges, backend)
    indexed = list(enumerate(ranges))

    def call(item: tuple[int, tuple[int, int]]) -> Any:
        index, (start, end) = item
        return produce(index, start, end)

    if plan.readAhead <= 1:
        with _io_concurrency(plan.ioConcurrency), _blas_limit(plan.withinBlockThreads):
            return list(_progress((call(item) for item in indexed), msg, n_ranges))

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
        return list(_progress(stream, msg, n_ranges))
