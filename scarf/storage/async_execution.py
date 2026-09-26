"""One async coordinator for Scarf-owned Zarr array operations."""

import asyncio
import contextvars
import importlib
import itertools
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from typing import Any, TypeVar, cast

import zarr

from threadpoolctl import ThreadpoolController

from ..utils.shutdown import shutdown_checkpoint

from .budget import detect_workers
from .execution import OperationPlan

T = TypeVar("T")

# Zarr's sync ThreadPoolExecutor is created on first use and never resized.
# This is the process thread ceiling, not an operation plan.
_HOST_THREAD_CEILING: int | None = None
_ZARR_CONFIG_LOCK = threading.RLock()
_ACTIVE_IO_LIMITS: dict[object, int] = {}
_IDLE_IO_LIMIT: int | None = None


_NUMBA_THREAD_LOCK = threading.Lock()
_WORKER_NUMBA_CAP = threading.local()
_IO_TASKS: contextvars.ContextVar[set[asyncio.Future[Any]] | None] = (
    contextvars.ContextVar("storage_io_tasks", default=None)
)


@contextmanager
def zarr_io_concurrency(limit: int) -> Iterator[None]:
    global _IDLE_IO_LIMIT
    token = object()
    with _ZARR_CONFIG_LOCK:
        if not _ACTIVE_IO_LIMITS:
            _IDLE_IO_LIMIT = zarr.config.get("async.concurrency")
        _ACTIVE_IO_LIMITS[token] = max(1, int(limit))
        zarr.config.set({"async.concurrency": min(_ACTIVE_IO_LIMITS.values())})
    try:
        yield
    finally:
        with _ZARR_CONFIG_LOCK:
            del _ACTIVE_IO_LIMITS[token]
            restored = (
                min(_ACTIVE_IO_LIMITS.values()) if _ACTIVE_IO_LIMITS else _IDLE_IO_LIMIT
            )
            zarr.config.set({"async.concurrency": restored})


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


async def _await_completion(future: asyncio.Future[T]) -> T:
    cancellation: asyncio.CancelledError | None = None
    while not future.done():
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError as exc:
            cancellation = exc
        except BaseException:
            break
    if cancellation is not None:
        try:
            future.result()
        except BaseException as exc:
            if not isinstance(exc, asyncio.CancelledError):
                raise BaseExceptionGroup(
                    "Storage work failed during cancellation", [cancellation, exc]
                ) from None
        raise cancellation
    return future.result()


def _scoped_task(
    loop: asyncio.AbstractEventLoop,
    coro: Coroutine[Any, Any, Any],
    **kwargs: Any,
) -> asyncio.Future[Any]:
    task = asyncio.Task(coro, loop=loop, **kwargs)
    context = kwargs.get("context")
    scope = _IO_TASKS.get() if context is None else context.get(_IO_TASKS)
    if scope is not None:
        scope.add(task)
    return task


async def _drained(coroutine: Coroutine[Any, Any, T]) -> T:
    """Run one storage coroutine and wait for every task it started."""
    tasks: set[asyncio.Future[Any]] = set()
    token = _IO_TASKS.set(tasks)
    try:
        root = asyncio.create_task(coroutine)
    finally:
        _IO_TASKS.reset(token)
    errors: list[BaseException] = []
    result: Any = None
    try:
        result = await _await_completion(root)
    except BaseException as exc:
        errors.append(exc)
    observed: set[asyncio.Future[Any]] = set()
    while outstanding := tasks - observed:
        observed.update(outstanding)
        try:
            outcomes = await _await_completion(
                asyncio.gather(*outstanding, return_exceptions=True)
            )
            for outcome in outcomes:
                if isinstance(outcome, BaseException) and not any(
                    outcome is error for error in errors
                ):
                    errors.append(outcome)
        except BaseException as exc:
            errors.append(exc)
    tasks.clear()
    if len(errors) > 1:
        raise BaseExceptionGroup("Storage I/O failed while draining work", errors)
    if errors:
        raise errors[0]
    return cast(T, result)


class _IoLoops:
    """Event-loop threads that run storage coroutines.

    Zarr copies decoded chunks into their destination on the thread that runs
    its event loop. Spreading operations over several loops runs those copies
    in parallel, while codec work still uses the shared codec pool.
    """

    def __init__(self, count: int, executor: ThreadPoolExecutor) -> None:
        self._loops: list[asyncio.AbstractEventLoop] = []
        self._threads: list[threading.Thread] = []
        self._turn = itertools.count()
        for index in range(max(1, count)):
            loop = asyncio.new_event_loop()
            loop.set_default_executor(executor)
            loop.set_task_factory(_scoped_task)
            thread = threading.Thread(
                target=loop.run_forever, name=f"scarf-zarr-io_{index}", daemon=True
            )
            thread.start()
            self._loops.append(loop)
            self._threads.append(thread)

    def submit(self, coroutine: Coroutine[Any, Any, T]) -> "Future[T]":
        loop = self._loops[next(self._turn) % len(self._loops)]
        return asyncio.run_coroutine_threadsafe(_drained(coroutine), loop)

    def close(self) -> None:
        for loop in self._loops:
            loop.call_soon_threadsafe(loop.stop)
        for loop, thread in zip(self._loops, self._threads, strict=True):
            thread.join()
            loop.close()


def _active_zarr_workers() -> int | None:
    sync = importlib.import_module("zarr.core.sync")
    pool = sync._executor
    if pool is None and sync.loop[0] is not None:
        pool = sync.loop[0]._default_executor
    return None if pool is None else int(pool._max_workers)


def ensure_zarr_host_ceiling(maxWorkers: int | None = None) -> int:
    """Keep the existing process pool, or configure it before its first use."""
    global _HOST_THREAD_CEILING
    host = max(1, detect_workers())
    if maxWorkers is not None and maxWorkers < 1:
        raise ValueError("Zarr worker ceiling must be positive")
    with _ZARR_CONFIG_LOCK:
        active = _active_zarr_workers()
        configured = zarr.config.get("threading.max_workers", None)
        existing = active or _HOST_THREAD_CEILING or configured
        if maxWorkers is not None and existing is not None and maxWorkers != existing:
            raise RuntimeError(
                f"Zarr already uses a ceiling of {existing}; configure "
                f"{maxWorkers} workers before opening storage in a fresh process"
            )
        if active is not None:
            _HOST_THREAD_CEILING = active
        if _HOST_THREAD_CEILING is None:
            ceiling = int(existing or maxWorkers or host)
            if active is None:
                zarr.config.set({"threading.max_workers": ceiling})
            _HOST_THREAD_CEILING = ceiling
    return _HOST_THREAD_CEILING


def configure_zarr_runtime(
    *,
    codecWorkers: int,
    asyncConcurrency: int,
) -> None:
    """Set the process Zarr thread ceiling before opening remote arrays.

    ``asyncConcurrency`` becomes the restored default after each runner.
    Conflicting explicit ceilings require a fresh process.
    """
    codec_workers = int(codecWorkers)
    async_concurrency = int(asyncConcurrency)
    if codec_workers < 1 or async_concurrency < 1:
        raise ValueError("Zarr runtime limits must be positive")
    with _ZARR_CONFIG_LOCK:
        if _ACTIVE_IO_LIMITS:
            raise RuntimeError(
                "Cannot reconfigure Zarr during active storage operations"
            )
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
        *,
        operation: OperationPlan,
    ):
        self.plan = operation
        self.ledger = ByteLedger(
            operation.requestedMemoryBytes - operation.residentBytes
        )
        self.readerWaitSeconds = 0.0
        self._compute_pool: ThreadPoolExecutor | None = None
        self._codec_pool: ThreadPoolExecutor | None = None
        self._read_slots: asyncio.Semaphore | None = None
        self._commit_slots: asyncio.Semaphore | None = None
        self._io_loops: _IoLoops | None = None
        self._blas: ThreadpoolController | None = None

    def run(self, operation: Callable[["AsyncStorageRunner"], Awaitable[T]]) -> T:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._run(operation))
        error: list[BaseException] = []
        result: list[T] = []
        context = contextvars.copy_context()

        def _in_thread() -> None:
            try:
                result.append(context.run(lambda: asyncio.run(self._run(operation))))
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
        result: Any = None
        errors: list[BaseException] = []
        restore_numba = None
        blas_state: Any = None
        io_context = zarr_io_concurrency(self.plan.ioConcurrency)
        io_entered = False
        try:
            ensure_zarr_host_ceiling()
            self._codec_pool = ThreadPoolExecutor(
                max_workers=self.plan.codecWorkers,
                thread_name_prefix="scarf-zarr-codec",
            )
            loop.set_default_executor(self._codec_pool)
            self._io_loops = _IoLoops(self.plan.codecWorkers, self._codec_pool)
            # Scanning loaded libraries takes milliseconds under the GIL; scan
            # once per operation instead of once per compute call.
            self._blas = ThreadpoolController()
            # Compute tasks limit BLAS threads process-wide, and concurrent
            # limit scopes restore each other's values; restore the limits in
            # force before the operation once every task has finished.
            blas_state = self._blas.limit(limits=None)
            self._compute_pool = ThreadPoolExecutor(
                max_workers=self.plan.computeWorkers,
                thread_name_prefix="scarf-compute",
            )
            self._read_slots = asyncio.Semaphore(
                self.plan.readWorkers * self.plan.innerReads
            )
            self._commit_slots = asyncio.Semaphore(self.plan.writeWorkers)
            restore_numba = _install_numba_thread_cap(self.plan.threadsPerComputeWorker)
            io_context.__enter__()
            io_entered = True
            shutdown_checkpoint()
            result = await operation(self)
            shutdown_checkpoint()
        except BaseException as exc:
            errors.append(exc)
        finally:
            pending = asyncio.all_tasks(loop) - {asyncio.current_task()}
            for task in pending:
                task.cancel()
            if pending:
                try:
                    outcomes = await _await_completion(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                    errors.extend(
                        item
                        for item in outcomes
                        if isinstance(item, BaseException)
                        and not isinstance(item, asyncio.CancelledError)
                    )
                except BaseException as exc:
                    errors.append(exc)
            for pool in (self._compute_pool, self._codec_pool):
                if pool is not None:
                    try:
                        pool.shutdown(wait=True, cancel_futures=True)
                    except BaseException as exc:
                        errors.append(exc)
            if self._io_loops is not None:
                try:
                    self._io_loops.close()
                except BaseException as exc:
                    errors.append(exc)
                self._io_loops = None
            if blas_state is not None:
                blas_state.restore_original_limits()
            if restore_numba is not None:
                try:
                    restore_numba()
                except BaseException as exc:
                    errors.append(exc)
            if io_entered:
                try:
                    io_context.__exit__(None, None, None)
                except BaseException as exc:
                    errors.append(exc)
            leftover = self.ledger.held_bytes()
        if leftover:
            errors.append(
                RuntimeError(
                    f"byte ledger still holds {leftover} bytes after the operation"
                )
            )
        if len(errors) > 1:
            raise BaseExceptionGroup(
                "Async storage failed during execution or cleanup",
                errors,
            )
        if errors:
            raise errors[0]
        return cast(T, result)

    async def io(self, coroutine: Coroutine[Any, Any, T]) -> T:
        """Run a self-contained storage coroutine on one of the I/O loops."""
        if self._io_loops is None:
            raise RuntimeError("I/O loops are not installed")
        return await _await_completion(
            asyncio.wrap_future(self._io_loops.submit(coroutine))
        )

    async def compute(self, fn: Callable[[], T]) -> T:
        if self._compute_pool is None or self._blas is None:
            raise RuntimeError("compute pool is not installed")
        loop = asyncio.get_running_loop()
        threads = max(1, int(self.plan.threadsPerComputeWorker))
        blas = self._blas

        def _limited() -> T:
            from .parallel import _shard_context

            _ensure_worker_numba_cap(threads)
            with _shard_context(), blas.limit(limits=threads):
                return fn()

        shutdown_checkpoint()
        result = await _await_completion(
            loop.run_in_executor(self._compute_pool, _limited)
        )
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
        try:
            shutdown_checkpoint()
            self.readerWaitSeconds += time.perf_counter() - started
            yield
        finally:
            await _await_completion(asyncio.create_task(self.ledger.release(nbytes)))

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
