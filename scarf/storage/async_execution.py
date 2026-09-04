"""One async coordinator for Scarf-owned Zarr array operations."""

import asyncio
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, TypeVar, cast

import zarr

from threadpoolctl import threadpool_limits

from ..utils.shutdown import shutdown_checkpoint

from .budget import ResourceBudget, detect_workers
from .execution import OperationPlan

T = TypeVar("T")

# Zarr's sync ThreadPoolExecutor is created on first use and never resized.
# This is the process thread ceiling, not an operation plan.
_HOST_THREAD_CEILING: int | None = None


_NUMBA_THREAD_LOCK = threading.Lock()
_WORKER_NUMBA_CAP = threading.local()


def _install_numba_thread_cap(threads: int) -> Callable[[], None] | None:
    """Cap Numba threads. Concurrent set_num_threads can deadlock.

    The setter is serialized. Each compute worker also applies the cap because
    Numba's thread mask is thread-local on some builds. Profiled runs can already
    look capped when Numba shares a process-wide thread count.
    """
    try:
        import numba as numba_mod
    except ImportError:
        return None
    getter = getattr(numba_mod, "get_num_threads")
    setter = getattr(numba_mod, "set_num_threads")
    cap = max(1, int(getattr(numba_mod.config, "NUMBA_NUM_THREADS")))
    target = min(max(1, int(threads)), cap)
    with _NUMBA_THREAD_LOCK:
        previous = int(getter())
        setter(target)

    def _restore() -> None:
        with _NUMBA_THREAD_LOCK:
            setter(previous)

    return _restore


def _ensure_worker_numba_cap(threads: int) -> None:
    if getattr(_WORKER_NUMBA_CAP, "applied", None) == threads:
        return
    _install_numba_thread_cap(threads)
    _WORKER_NUMBA_CAP.applied = threads


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    codecWorkerLimit: int
    zarrAsyncConcurrency: int
    computeWorkerLimit: int
    readGroupsInFlight: int
    destinationCommitsInFlight: int
    chunksPerShard: int
    threadsPerComputeWorker: int = 1


def resolve_execution_plan(
    resources: ResourceBudget,
    *,
    chunksPerShard: int = 10,
    readGroupsInFlight: int = 1,
    destinationCommitsInFlight: int = 1,
    computeWorkerLimit: int = 1,
    threadsPerComputeWorker: int = 1,
    operation: OperationPlan | None = None,
) -> ExecutionPlan:
    if operation is not None:
        chunksPerShard = operation.chunksPerShard
        readGroupsInFlight = operation.readWorkers * operation.innerReads
        destinationCommitsInFlight = operation.writeWorkers
        computeWorkerLimit = operation.computeWorkers
        threadsPerComputeWorker = operation.threadsPerComputeWorker
    workers = max(1, int(resources.workers))
    host_cores = max(1, detect_workers())
    codec_workers = host_cores
    compute_workers = min(workers, max(1, int(computeWorkerLimit)))
    threads = max(1, int(threadsPerComputeWorker))
    if compute_workers * threads > workers:
        threads = max(1, workers // compute_workers)
    lanes = max(1, int(readGroupsInFlight))
    return ExecutionPlan(
        codecWorkerLimit=codec_workers,
        zarrAsyncConcurrency=max(
            lanes, min(codec_workers, max(1, int(chunksPerShard)))
        ),
        computeWorkerLimit=compute_workers,
        readGroupsInFlight=lanes,
        destinationCommitsInFlight=max(1, int(destinationCommitsInFlight)),
        chunksPerShard=max(1, int(chunksPerShard)),
        threadsPerComputeWorker=threads,
    )


def ensure_zarr_host_ceiling(maxWorkers: int | None = None) -> int:
    """Install the host Zarr thread ceiling once. Later calls do not shrink it."""
    global _HOST_THREAD_CEILING
    host = max(1, detect_workers())
    requested = host if maxWorkers is None else max(1, int(maxWorkers))
    if _HOST_THREAD_CEILING is None:
        ceiling = max(host, requested)
        zarr.config.set({"threading.max_workers": ceiling})
        _HOST_THREAD_CEILING = ceiling
    return _HOST_THREAD_CEILING


def zarr_runtime_installed() -> bool:
    """Return True when this process already has a Zarr thread ceiling."""
    return _HOST_THREAD_CEILING is not None


def configure_zarr_runtime(
    *,
    codecWorkers: int,
    asyncConcurrency: int,
) -> None:
    """Set the process Zarr thread ceiling before opening remote arrays.

    ``asyncConcurrency`` becomes the restored default after each runner
    scopes its own cap. The first ceiling wins; Zarr does not resize a
    live thread pool.
    """
    codec_workers = int(codecWorkers)
    async_concurrency = int(asyncConcurrency)
    if codec_workers < 1 or async_concurrency < 1:
        raise ValueError("Zarr runtime limits must be positive")
    ensure_zarr_host_ceiling(codec_workers)
    zarr.config.set({"async.concurrency": async_concurrency})


def reset_zarr_runtime_for_tests() -> None:
    global _HOST_THREAD_CEILING
    zarr.config.set(
        {
            "threading.max_workers": None,
            "async.concurrency": 10,
        }
    )
    _HOST_THREAD_CEILING = None


class ByteLedger:
    """Admit Scarf-owned buffers before async tasks are created."""

    def __init__(self, limitBytes: int):
        if limitBytes < 1:
            raise ValueError("byte ledger limit must be positive")
        self.limitBytes = int(limitBytes)
        self._held = 0
        self._peak = 0
        self._condition = asyncio.Condition()

    async def acquire(self, nbytes: int) -> None:
        size = int(nbytes)
        if size < 1:
            return
        if size > self.limitBytes:
            raise MemoryError(
                f"One buffer needs {size} bytes, but the operation limit is "
                f"{self.limitBytes} bytes"
            )
        async with self._condition:
            while self._held + size > self.limitBytes:
                await self._condition.wait()
            self._held += size
            self._peak = max(self._peak, self._held)

    async def release(self, nbytes: int) -> None:
        size = int(nbytes)
        if size < 0:
            raise ValueError("released byte count must not be negative")
        async with self._condition:
            if size > self._held:
                raise RuntimeError(
                    f"cannot release {size} bytes from a ledger holding {self._held}"
                )
            self._held -= size
            self._condition.notify_all()

    def held_bytes(self) -> int:
        return self._held

    def peak_bytes(self) -> int:
        return self._peak

    def is_empty(self) -> bool:
        return self._held == 0


class AsyncStorageRunner:
    """Own one event loop, codec executor, compute pool, and byte ledger."""

    def __init__(
        self,
        resources: ResourceBudget,
        *,
        chunksPerShard: int = 10,
        readGroupsInFlight: int = 1,
        destinationCommitsInFlight: int = 1,
        computeWorkerLimit: int = 1,
        threadsPerComputeWorker: int = 1,
        operation: OperationPlan | None = None,
    ):
        self.resources = resources
        self.plan = resolve_execution_plan(
            resources,
            chunksPerShard=chunksPerShard,
            readGroupsInFlight=readGroupsInFlight,
            destinationCommitsInFlight=destinationCommitsInFlight,
            computeWorkerLimit=computeWorkerLimit,
            threadsPerComputeWorker=threadsPerComputeWorker,
            operation=operation,
        )
        self.ledger = ByteLedger(resources.memoryBytes)
        self.readerWaitSeconds = 0.0
        self._compute_pool: ThreadPoolExecutor | None = None
        self._codec_pool: ThreadPoolExecutor | None = None
        self._read_slots: asyncio.Semaphore | None = None
        self._commit_slots: asyncio.Semaphore | None = None

    def run(self, operation: Callable[["AsyncStorageRunner"], Awaitable[T]]) -> T:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._run(operation))
        error: list[BaseException] = []
        result: list[T] = []

        def _in_thread() -> None:
            try:
                result.append(asyncio.run(self._run(operation)))
            except BaseException as exc:
                error.append(exc)

        thread = threading.Thread(target=_in_thread)
        thread.start()
        thread.join()
        if error:
            raise error[0]
        return result[0]

    async def _run(
        self,
        operation: Callable[["AsyncStorageRunner"], Awaitable[T]],
    ) -> T:
        loop = asyncio.get_running_loop()
        ensure_zarr_host_ceiling()
        self._codec_pool = ThreadPoolExecutor(
            max_workers=self.plan.codecWorkerLimit,
            thread_name_prefix="scarf-zarr-codec",
        )
        loop.set_default_executor(self._codec_pool)
        self._compute_pool = ThreadPoolExecutor(
            max_workers=self.plan.computeWorkerLimit,
            thread_name_prefix="scarf-compute",
        )
        self._read_slots = asyncio.Semaphore(self.plan.readGroupsInFlight)
        self._commit_slots = asyncio.Semaphore(self.plan.destinationCommitsInFlight)
        result: Any = None
        operation_error: BaseException | None = None
        restore_numba = _install_numba_thread_cap(self.plan.threadsPerComputeWorker)
        try:
            shutdown_checkpoint()
            with zarr.config.set({"async.concurrency": self.plan.zarrAsyncConcurrency}):
                result = await operation(self)
            shutdown_checkpoint()
        except BaseException as exc:
            operation_error = exc
        finally:
            leftover = self.ledger.held_bytes()
            if restore_numba is not None:
                restore_numba()
            if self._compute_pool is not None:
                self._compute_pool.shutdown(wait=True, cancel_futures=True)
            if self._codec_pool is not None:
                self._codec_pool.shutdown(wait=True, cancel_futures=True)
        if leftover:
            ledger_error = RuntimeError(
                f"byte ledger still holds {leftover} bytes after the operation"
            )
            if operation_error is not None:
                raise BaseExceptionGroup(
                    "async storage operation failed and leaked admitted bytes",
                    [operation_error, ledger_error],
                )
            raise ledger_error
        if operation_error is not None:
            raise operation_error
        return cast(T, result)

    async def compute(self, fn: Callable[[], T]) -> T:
        if self._compute_pool is None:
            raise RuntimeError("compute pool is not installed")
        loop = asyncio.get_running_loop()
        threads = max(1, int(self.plan.threadsPerComputeWorker))

        def _limited() -> T:
            _ensure_worker_numba_cap(threads)
            with threadpool_limits(limits=threads):
                return fn()

        shutdown_checkpoint()
        result = await loop.run_in_executor(self._compute_pool, _limited)
        shutdown_checkpoint()
        return result

    async def read_slot(self) -> asyncio.Semaphore:
        if self._read_slots is None:
            raise RuntimeError("read slots are not installed")
        return self._read_slots

    async def commit_slot(self) -> asyncio.Semaphore:
        if self._commit_slots is None:
            raise RuntimeError("commit slots are not installed")
        return self._commit_slots

    @asynccontextmanager
    async def reserve_bytes(self, nbytes: int) -> AsyncIterator[None]:
        """Hold a ledger charge for the complete lifetime of an owned buffer."""
        started = time.perf_counter()
        shutdown_checkpoint()
        await self.ledger.acquire(nbytes)
        shutdown_checkpoint()
        self.readerWaitSeconds += time.perf_counter() - started
        try:
            yield
        finally:
            await self.ledger.release(nbytes)

    @asynccontextmanager
    async def read_lane(self) -> AsyncIterator[None]:
        """Enter one bounded outer read lane without changing byte ownership."""
        slot = await self.read_slot()
        started = time.perf_counter()
        await slot.acquire()
        self.readerWaitSeconds += time.perf_counter() - started
        try:
            yield
        finally:
            slot.release()

    @asynccontextmanager
    async def commit_lane(self) -> AsyncIterator[None]:
        """Enter one bounded destination commit lane without changing ownership."""
        slot = await self.commit_slot()
        async with slot:
            yield

    async def bounded_read(
        self,
        nbytes: int,
        factory: Callable[[], Coroutine[Any, Any, T]],
    ) -> T:
        slot = await self.read_slot()
        started = time.perf_counter()
        await self.ledger.acquire(nbytes)
        await slot.acquire()
        self.readerWaitSeconds += time.perf_counter() - started
        try:
            return await factory()
        finally:
            slot.release()
            await self.ledger.release(nbytes)

    async def bounded_commit(
        self,
        nbytes: int,
        factory: Callable[[], Coroutine[Any, Any, T]],
    ) -> T:
        slot = await self.commit_slot()
        await self.ledger.acquire(nbytes)
        try:
            async with slot:
                return await factory()
        finally:
            await self.ledger.release(nbytes)
