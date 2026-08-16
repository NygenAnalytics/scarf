"""One async coordinator for Scarf-owned Zarr array operations."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from typing import Any, TypeVar, cast

import zarr

from .budget import ResourceBudget

T = TypeVar("T")

_INSTALLED_RUNTIME: tuple[int, int] | None = None
_EXPLICIT_RUNTIME = False


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    codecWorkerLimit: int
    zarrAsyncConcurrency: int
    computeWorkerLimit: int
    readGroupsInFlight: int
    destinationCommitsInFlight: int
    chunksPerShard: int


def resolve_execution_plan(
    resources: ResourceBudget,
    *,
    chunksPerShard: int = 10,
    readGroupsInFlight: int = 1,
    destinationCommitsInFlight: int = 1,
) -> ExecutionPlan:
    workers = max(1, int(resources.workers))
    codec_workers = max(1, workers - 1) if workers > 1 else 1
    return ExecutionPlan(
        codecWorkerLimit=codec_workers,
        zarrAsyncConcurrency=min(codec_workers, max(1, int(chunksPerShard))),
        computeWorkerLimit=1,
        readGroupsInFlight=max(1, int(readGroupsInFlight)),
        destinationCommitsInFlight=max(1, int(destinationCommitsInFlight)),
        chunksPerShard=max(1, int(chunksPerShard)),
    )


def install_zarr_runtime(plan: ExecutionPlan, *, _explicit: bool = False) -> None:
    global _EXPLICIT_RUNTIME, _INSTALLED_RUNTIME
    desired = (plan.codecWorkerLimit, plan.zarrAsyncConcurrency)
    if _INSTALLED_RUNTIME is not None and _INSTALLED_RUNTIME != desired:
        raise RuntimeError(
            "Zarr already has a process runtime of "
            f"codecWorkers={_INSTALLED_RUNTIME[0]}, "
            f"async.concurrency={_INSTALLED_RUNTIME[1]}; "
            f"this operation requested {desired}. Start a new process instead of "
            "reconfiguring a live executor."
        )
    if _INSTALLED_RUNTIME is None:
        zarr.config.set(
            {
                "threading.max_workers": plan.codecWorkerLimit,
                "async.concurrency": plan.zarrAsyncConcurrency,
            }
        )
        _INSTALLED_RUNTIME = desired
    if _explicit:
        _EXPLICIT_RUNTIME = True


def ensure_zarr_runtime(plan: ExecutionPlan) -> None:
    install_zarr_runtime(plan)


def configure_zarr_runtime(
    *,
    codecWorkers: int,
    asyncConcurrency: int,
) -> None:
    """Install one explicit process runtime before opening remote arrays."""
    codec_workers = int(codecWorkers)
    async_concurrency = int(asyncConcurrency)
    if codec_workers < 1 or async_concurrency < 1:
        raise ValueError("Zarr runtime limits must be positive")
    install_zarr_runtime(
        ExecutionPlan(
            codecWorkerLimit=codec_workers,
            zarrAsyncConcurrency=async_concurrency,
            computeWorkerLimit=1,
            readGroupsInFlight=1,
            destinationCommitsInFlight=1,
            chunksPerShard=max(1, async_concurrency),
        ),
        _explicit=True,
    )


def reset_zarr_runtime_for_tests() -> None:
    global _EXPLICIT_RUNTIME, _INSTALLED_RUNTIME
    zarr.config.set(
        {
            "threading.max_workers": None,
            "async.concurrency": 10,
        }
    )
    _INSTALLED_RUNTIME = None
    _EXPLICIT_RUNTIME = False


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
    ):
        self.resources = resources
        self.plan = resolve_execution_plan(
            resources,
            chunksPerShard=chunksPerShard,
            readGroupsInFlight=readGroupsInFlight,
            destinationCommitsInFlight=destinationCommitsInFlight,
        )
        if (
            _EXPLICIT_RUNTIME
            and _INSTALLED_RUNTIME is not None
            and self.plan.codecWorkerLimit == _INSTALLED_RUNTIME[0]
        ):
            self.plan = replace(
                self.plan,
                zarrAsyncConcurrency=_INSTALLED_RUNTIME[1],
            )
        self.ledger = ByteLedger(resources.memoryBytes)
        self._compute_pool: ThreadPoolExecutor | None = None
        self._codec_pool: ThreadPoolExecutor | None = None
        self._read_slots: asyncio.Semaphore | None = None
        self._commit_slots: asyncio.Semaphore | None = None

    def run(self, operation: Callable[["AsyncStorageRunner"], Awaitable[T]]) -> T:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._run(operation))
        raise RuntimeError(
            "AsyncStorageRunner.run cannot nest inside a running event loop; "
            "await the operation on the existing runner instead"
        )

    async def _run(
        self,
        operation: Callable[["AsyncStorageRunner"], Awaitable[T]],
    ) -> T:
        loop = asyncio.get_running_loop()
        ensure_zarr_runtime(self.plan)
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
        try:
            result = await operation(self)
        except BaseException as exc:
            operation_error = exc
        finally:
            leftover = self.ledger.held_bytes()
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
        return await loop.run_in_executor(self._compute_pool, fn)

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
        await self.ledger.acquire(nbytes)
        try:
            yield
        finally:
            await self.ledger.release(nbytes)

    @asynccontextmanager
    async def read_lane(self) -> AsyncIterator[None]:
        """Enter one bounded outer read lane without changing byte ownership."""
        slot = await self.read_slot()
        async with slot:
            yield

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
        await self.ledger.acquire(nbytes)
        try:
            async with slot:
                return await factory()
        finally:
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
