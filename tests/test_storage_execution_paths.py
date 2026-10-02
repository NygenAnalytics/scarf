"""Failure, shutdown, and budget paths of storage execution and row-band writers."""

import asyncio
import queue
import threading
import tracemalloc
from collections import deque
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import zarr
from scipy.sparse import coo_matrix
from zarr.storage import LocalStore, MemoryStore

import scarf.storage.async_execution as async_execution
import scarf.storage.feature_stream as feature_stream
from scarf.matrix.chunked import ChunkedArray
from scarf.storage.async_execution import AsyncStorageRunner
from scarf.storage.budget import ResourceBudget
from scarf.storage.count_matrix import (
    CountMatrixPolicy,
    persist_count_matrix_plan,
    plan_count_matrix_pair,
)
from scarf.storage.execution import WorkShape, execution_report_scope, plan_operation
from scarf.storage.feature_stream import _iter_bounded_handoff
from scarf.storage.io_policy import DEFAULT_STORAGE_IO_POLICY, StorageIoPolicy
from scarf.storage.sharding import (
    SparseRowBand,
    SparseShardBuffer,
    SparseWriteBand,
    _sparse_batch_plan,
    _sparse_task_working_bytes,
    write_counts_t,
    write_dense_from_row_batches,
    write_sparse_bands,
)
from tests.storage_helpers import finalize_test_counts


def _runner(resources: ResourceBudget) -> AsyncStorageRunner:
    operation = plan_operation(
        resources,
        WorkShape(nUnits=resources.workers, unitBytes=1),
        policy=StorageIoPolicy(readWorkers=resources.workers),
    )
    return AsyncStorageRunner(operation=operation)


def test_runner_failures_are_not_chained_to_the_loop_lookup() -> None:
    async def operation(_active: AsyncStorageRunner) -> None:
        raise ValueError("operation failed")

    with pytest.raises(ValueError, match="operation failed") as raised:
        _runner(ResourceBudget(1024, 1)).run(operation)
    # It used to carry the RuntimeError that found no running loop.
    assert raised.value.__context__ is None


def test_compute_failure_after_cancellation_reports_both() -> None:
    started = threading.Event()
    release = threading.Event()

    def native() -> None:
        started.set()
        assert release.wait(5)
        raise ValueError("native work failed")

    async def operation(active: AsyncStorageRunner) -> list[type[BaseException]]:
        task = asyncio.create_task(active.compute(native))
        while not started.is_set():
            await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0.01)
        release.set()
        # The cancelled call waits for the native work, which then fails.
        with pytest.raises(BaseExceptionGroup, match="during cancellation") as raised:
            await task
        return [type(error) for error in raised.value.exceptions]

    assert _runner(ResourceBudget(1024, 1)).run(operation) == [
        asyncio.CancelledError,
        ValueError,
    ]


def test_io_cancelled_on_its_loop_reports_one_cancellation() -> None:
    async def cancelled_read() -> None:
        task = asyncio.current_task()
        assert task is not None
        task.cancel()
        await asyncio.sleep(0)

    async def operation(active: AsyncStorageRunner) -> bool:
        # A group of two cancellations of the same coroutine used to arrive.
        with pytest.raises(asyncio.CancelledError):
            await active.io(cancelled_read())
        return True

    assert _runner(ResourceBudget(1024, 1)).run(operation)


def test_io_reports_root_and_child_failures_together() -> None:
    async def write() -> None:
        async def child() -> None:
            raise ValueError("child write failed")

        asyncio.get_running_loop().create_task(child())
        await asyncio.sleep(0)
        raise RuntimeError("root write failed")

    async def operation(active: AsyncStorageRunner) -> None:
        await active.io(write())

    with pytest.raises(ExceptionGroup, match="draining work") as raised:
        _runner(ResourceBudget(1024, 1)).run(operation)
    assert {str(error) for error in raised.value.exceptions} == {
        "root write failed",
        "child write failed",
    }


def test_runner_cancels_tasks_an_operation_leaves_behind() -> None:
    cancelled: list[str] = []
    left: list[asyncio.Future[None]] = []

    async def linger() -> None:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.append("linger")
            raise

    async def fail_on_cancel() -> None:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            raise ValueError("leftover task failed") from None

    async def leave(_active: AsyncStorageRunner) -> int:
        left.append(asyncio.ensure_future(linger()))
        await asyncio.sleep(0)
        return 7

    assert _runner(ResourceBudget(1024, 1)).run(leave) == 7
    assert cancelled == ["linger"]

    async def leave_failing(_active: AsyncStorageRunner) -> int:
        left.append(asyncio.ensure_future(fail_on_cancel()))
        await asyncio.sleep(0)
        return 7

    with pytest.raises(ValueError, match="leftover task failed"):
        _runner(ResourceBudget(1024, 1)).run(leave_failing)


def test_runner_reports_a_cancellation_while_it_cancels_leftover_tasks() -> None:
    left: list[asyncio.Future[None]] = []

    async def operation(_active: AsyncStorageRunner) -> int:
        main = asyncio.current_task()
        assert main is not None

        async def cancel_the_runner() -> None:
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                main.cancel()
                raise

        left.append(asyncio.ensure_future(cancel_the_runner()))
        await asyncio.sleep(0)
        return 1

    runner = _runner(ResourceBudget(1024, 1))
    with pytest.raises(asyncio.CancelledError):
        runner.run(operation)
    # Cleanup continued after the cancellation.
    assert runner._compute_pool is not None and runner._compute_pool._shutdown
    assert runner._io_loops is None
    assert zarr.config.get("async.concurrency") == 10


def test_runner_setup_failure_leaves_nothing_to_clean_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail() -> int:
        raise RuntimeError("host ceiling failed")

    monkeypatch.setattr(async_execution, "ensure_zarr_host_ceiling", fail)
    runner = _runner(ResourceBudget(1024, 1))

    async def operation(_active: AsyncStorageRunner) -> None:
        pytest.fail("the operation must not start")

    with pytest.raises(RuntimeError, match="host ceiling failed"):
        runner.run(operation)
    assert runner._compute_pool is None
    assert runner._codec_pool is None
    assert runner._io_loops is None
    assert zarr.config.get("async.concurrency") == 10


def test_runner_reports_every_cleanup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingShutdownPool(ThreadPoolExecutor):
        def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
            super().shutdown(wait, cancel_futures=cancel_futures)
            # Only the runner's own shutdown fails, not the event loop's.
            if cancel_futures:
                raise RuntimeError(f"{self._thread_name_prefix} shutdown failed")

    close = async_execution._IoLoops.close

    def failing_close(loops: Any) -> None:
        close(loops)
        raise RuntimeError("I/O loop close failed")

    io_limit = async_execution.zarr_io_concurrency

    @contextmanager
    def failing_io_limit(limit: int) -> Any:
        with io_limit(limit):
            yield
        raise RuntimeError("I/O limit restore failed")

    monkeypatch.setattr(async_execution, "ThreadPoolExecutor", FailingShutdownPool)
    monkeypatch.setattr(async_execution._IoLoops, "close", failing_close)
    monkeypatch.setattr(async_execution, "zarr_io_concurrency", failing_io_limit)

    async def operation(_active: AsyncStorageRunner) -> int:
        return 1

    with pytest.raises(ExceptionGroup, match="execution or cleanup") as raised:
        _runner(ResourceBudget(1024, 1)).run(operation)
    assert [str(error) for error in raised.value.exceptions] == [
        "scarf-compute shutdown failed",
        "scarf-zarr-codec shutdown failed",
        "I/O loop close failed",
        "I/O limit restore failed",
    ]
    assert zarr.config.get("async.concurrency") == 10


class _ReportingQueue(queue.Queue):
    """A handoff queue that reports what its blocking puts did.

    Only the producer's end marker uses a blocking put.
    """

    def __init__(self, maxsize: int = 0, *, refuseFirstPut: bool = False) -> None:
        super().__init__(maxsize)
        self.refuseFirstPut = refuseFirstPut
        self.full = threading.Event()
        self.ended = threading.Event()

    def put(self, item: Any, block: bool = True, timeout: float | None = None) -> None:
        if not block:
            super().put(item, block, timeout)
            return
        if self.refuseFirstPut:
            self.refuseFirstPut = False
            self.full.set()
            raise queue.Full
        try:
            super().put(item, block, timeout)
        except queue.Full:
            self.full.set()
            raise
        self.ended.set()


def _reporting_queues(
    monkeypatch: pytest.MonkeyPatch, *, refuseFirstPut: bool = False
) -> list[_ReportingQueue]:
    created: list[_ReportingQueue] = []

    def make(maxsize: int = 0) -> _ReportingQueue:
        created.append(_ReportingQueue(maxsize, refuseFirstPut=refuseFirstPut))
        return created[-1]

    monkeypatch.setattr(
        feature_stream,
        "queue",
        SimpleNamespace(Queue=make, Full=queue.Full, Empty=queue.Empty),
    )
    return created


def _queue_behind_held_item(
    deliver: Any, holding: threading.Event
) -> "asyncio.Future[tuple[asyncio.Future[None], asyncio.Future[None]]]":
    """Deliver two items; the second is queued while the consumer holds the first."""

    async def deliveries() -> tuple[asyncio.Future[None], asyncio.Future[None]]:
        first = asyncio.ensure_future(deliver("first"))
        assert await asyncio.to_thread(holding.wait, 5)
        second = asyncio.ensure_future(deliver("second"))
        # The second delivery puts its item before this coroutine resumes.
        await asyncio.sleep(0)
        return first, second

    return asyncio.ensure_future(deliveries())


def test_handoff_retries_its_end_marker_while_the_consumer_holds_an_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queues = _reporting_queues(monkeypatch)
    holding = threading.Event()
    deliveries: list[asyncio.Future[None]] = []

    def run(deliver: Any, _stop: threading.Event) -> None:
        async def main() -> None:
            # The producer returns with the second item queued, and
            # asyncio.run cancels both deliveries.
            deliveries.extend(await _queue_behind_held_item(deliver, holding))

        asyncio.run(main())

    stream = _iter_bounded_handoff(in_flight=1, run=run)
    assert next(stream) == "first"
    holding.set()
    assert queues[0].full.wait(5)
    assert list(stream) == ["second"]


def test_stopped_handoff_producer_drops_items_nobody_took(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queues = _reporting_queues(monkeypatch)
    holding = threading.Event()
    returned = threading.Event()

    def run(deliver: Any, stop: threading.Event) -> None:
        async def main() -> None:
            first, second = await _queue_behind_held_item(deliver, holding)
            # Deliveries stop waiting for acknowledgments once the stream stops.
            stop.set()
            await asyncio.gather(first, second)
            returned.set()

        asyncio.run(main())

    stream = _iter_bounded_handoff(in_flight=1, run=run)
    assert next(stream) == "first"
    holding.set()
    assert queues[0].ended.wait(5)
    assert returned.is_set() and queues[0].full.is_set()
    assert list(stream) == []


def test_closed_handoff_releases_the_items_still_queued(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queues = _reporting_queues(monkeypatch)
    holding = threading.Event()
    queued = threading.Event()
    finished: list[bool] = []

    def run(deliver: Any, _stop: threading.Event) -> None:
        async def main() -> None:
            first, second = await _queue_behind_held_item(deliver, holding)
            queued.set()
            # Return only after the closing consumer has taken the queued item.
            while queues[0].qsize():
                await asyncio.sleep(0.005)
            await asyncio.gather(first, second)
            finished.append(True)

        asyncio.run(main())

    stream = _iter_bounded_handoff(in_flight=1, run=run)
    assert next(stream) == "first"
    holding.set()
    assert queued.wait(5)
    stream.close()
    assert finished == [True]


def test_handoff_end_marker_retries_when_the_queue_empties_meanwhile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queues = _reporting_queues(monkeypatch, refuseFirstPut=True)

    def run(_deliver: Any, stop: threading.Event) -> None:
        stop.set()

    assert list(_iter_bounded_handoff(in_flight=1, run=run)) == []
    assert queues[0].full.is_set() and queues[0].ended.is_set()


def _band(destination: zarr.Array, values: np.ndarray) -> SparseWriteBand:
    coo = coo_matrix(values)
    return SparseWriteBand(
        destination=destination,
        band=SparseRowBand(
            start=0,
            end=values.shape[0],
            nColumns=values.shape[1],
            row=coo.row.astype(np.int64),
            column=coo.col.astype(np.int64),
            data=coo.data,
            dtype=values.dtype,
        ),
    )


def test_sparse_bands_of_a_wider_destination_wait_for_the_narrower_ones() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    narrow = root.create_array(
        "narrow", shape=(4, 3), chunks=(4, 3), shards=(4, 3), dtype=np.uint16
    )
    wide = root.create_array(
        "wide", shape=(4, 300), chunks=(4, 300), shards=(4, 300), dtype=np.uint8
    )
    narrow_values = np.arange(1, 13, dtype=np.uint16).reshape(4, 3)
    wide_values = (np.arange(1200) % 50 + 1).astype(np.uint8).reshape(4, 300)
    first, second = _band(narrow, narrow_values), _band(wide, wide_values)
    # The wide band's write just fits beside its own values, and the producer
    # reserve leaves room to write the narrow band while it is pulled.
    memory = second.band.sparseBytes + _sparse_task_working_bytes(second, 1)
    reserve = memory - first.band.sparseBytes - _sparse_task_working_bytes(first, 1)
    resources = ResourceBudget(memory, 4)
    with pytest.raises(MemoryError):
        _sparse_batch_plan(
            deque([first, second]), resources, 0, 2, DEFAULT_STORAGE_IO_POLICY
        )

    with execution_report_scope() as reports:
        write_sparse_bands(
            iter([first, second]), resources=resources, producerReserveBytes=reserve
        )
    np.testing.assert_array_equal(narrow[:], narrow_values)
    np.testing.assert_array_equal(wide[:], wide_values)
    (report,) = [r for r in reports if r.unitKind == "countsImportBand"]
    assert (report.unitsCompleted, report.extra["batches"]) == (2, 2)


def test_sparse_bands_cast_float_values_into_integer_destinations() -> None:
    destination = zarr.open_group(store=MemoryStore(), mode="w").create_array(
        "counts", shape=(4, 3), chunks=(2, 3), shards=(4, 3), dtype=np.uint16
    )
    values = np.arange(1, 13, dtype=np.uint16).reshape(4, 3)
    write = _band(destination, values)
    floats = replace(write.band, data=write.band.data.astype(np.float64))
    resources = ResourceBudget(1 << 20, 2)
    write_sparse_bands(iter([replace(write, band=floats)]), resources=resources)
    np.testing.assert_array_equal(destination[:], values)

    fractional = replace(floats, data=floats.data + 0.5)
    with pytest.raises(OverflowError, match="cannot be represented"):
        write_sparse_bands(iter([replace(write, band=fractional)]), resources=resources)


def test_sparse_writers_need_a_chunked_two_dimensional_destination() -> None:
    vector = zarr.open_group(store=MemoryStore(), mode="w").create_array(
        "vector", shape=(4,), chunks=(2,), dtype=np.uint16
    )
    for destination in (vector, np.zeros((4, 2))):
        with pytest.raises(ValueError, match="two-dimensional arrays"):
            SparseShardBuffer(destination)


def test_dense_row_batches_are_checked_against_their_destination() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    resources = ResourceBudget(1 << 24, 2)

    def destination(name: str) -> zarr.Array:
        return root.create_array(
            name, shape=(6, 2), chunks=(2, 2), shards=(4, 2), dtype=np.float64
        )

    empty = root.create_array("empty", shape=(0, 2), chunks=(1, 2), dtype=np.float64)
    assert (
        write_dense_from_row_batches(
            empty, iter([np.empty((0, 2))]), resources=resources
        )
        == 0
    )

    values = np.arange(12, dtype=np.float64).reshape(6, 2)
    target = destination("skips_empty_batches")
    batches = [np.empty((0, 2)), values[:3], np.empty((0, 2)), values[3:]]
    assert write_dense_from_row_batches(target, iter(batches), resources=resources) == 6
    np.testing.assert_array_equal(target[:], values)

    failures: list[tuple[list[np.ndarray], type[Exception], str]] = [
        ([], ValueError, "contains 0 rows, expected 6"),
        ([np.empty((0, 2))], ValueError, "contains 0 rows, expected 6"),
        ([np.ones((2, 3))], ValueError, "invalid shape"),
        ([values[:2], np.ones((2, 3))], ValueError, "invalid shape"),
        # The producer reserves twice the first batch's allocation by default.
        ([np.ones((1, 2)), np.ones((5, 2))], MemoryError, "producer reservation"),
        ([values, values[:1]], ValueError, "more rows than its destination"),
        ([values[:4]], ValueError, "contains 4 rows, expected 6"),
    ]
    for index, (stream, error, message) in enumerate(failures):
        with pytest.raises(error, match=message):
            write_dense_from_row_batches(
                destination(f"failure_{index}"), iter(stream), resources=resources
            )


def test_chunked_to_zarr_writes_every_band_to_its_mirror() -> None:
    from scarf.storage.arrays import create_numeric_array
    from scarf.storage.layout import normed_array_spec
    from scarf.storage.profiles import resolve_storage_profile
    from scarf.writers import chunked_to_zarr

    values = np.arange(40 * 3, dtype=np.float32).reshape(40, 3)
    root = zarr.open_group(store=MemoryStore(), mode="w")
    mirror = create_numeric_array(
        zarr.open_group(store=MemoryStore(), mode="w"),
        "mirror",
        normed_array_spec(40, 3, profile=resolve_storage_profile(root.store)),
    )
    chunked_to_zarr(ChunkedArray.from_numpy(values), root, "data", 1, mirror=mirror)
    np.testing.assert_array_equal(root["data"][:], values)
    np.testing.assert_array_equal(mirror[:], values)


def _paired_counts(values: np.ndarray) -> tuple[zarr.Group, zarr.Array]:
    plan = plan_count_matrix_pair(
        values.shape[0],
        values.shape[1],
        values.dtype,
        policy=CountMatrixPolicy(unitBytes=2_000, chunkBytes=200),
    )
    group = zarr.open_group(store=MemoryStore(), mode="w").create_group("RNA")
    counts = group.create_array(
        "counts",
        shape=plan.counts.shape,
        chunks=plan.counts.chunks,
        shards=plan.counts.shards,
        dtype=values.dtype,
    )
    counts[:] = values
    persist_count_matrix_plan(group, plan)
    persist_count_matrix_plan(counts, plan)
    finalize_test_counts(counts)
    return group, counts


def test_failed_counts_t_write_stays_incomplete_without_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # One source chunk and one destination shard, so one transpose fails.
    group, counts = _paired_counts(np.arange(16, dtype=np.uint16).reshape(4, 4))

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("transpose failed")

    monkeypatch.setattr("scarf.utils.strided.transpose_into", fail)
    with pytest.raises(RuntimeError, match="transpose failed"):
        write_counts_t(counts, group, resources=ResourceBudget(32 * 1024 * 1024, 2))
    assert group["countsT"].attrs["complete"] is False


def test_counts_t_requires_a_zarr_format_3_group() -> None:
    group = zarr.open_group(store=MemoryStore(), mode="w", zarr_format=2)
    counts = group.create_array("counts", shape=(2, 2), chunks=(2, 2), dtype=np.uint16)
    with pytest.raises(ValueError, match="Zarr format 3"):
        write_counts_t(counts, group)


def test_row_block_operation_chains_reserve_their_largest_step(tmp_path: Any) -> None:
    values = np.random.default_rng(0).poisson(0.5, size=(20_000, 32)).astype(np.uint8)
    root = zarr.open_group(store=LocalStore(str(tmp_path)), mode="w")
    stored = root.create_array(
        "counts",
        shape=values.shape,
        chunks=(10_000, 8),
        shards=(10_000, 32),
        dtype=np.uint8,
    )
    stored[:] = values
    counts = ChunkedArray(stored, nthreads=1, resources=ResourceBudget(1 << 30, 1))
    elements = 10_000 * 32
    # One block of uint8 counts as read: the block and Zarr's copy of it.
    read = 2 * elements
    scaled = counts * 2.0
    logged = np.log1p(scaled)
    # Scaling holds the read beside its float64 output; the logarithm holds
    # that output beside its own, as library-size log normalization does.
    assert scaled._block_owned_bytes() == read + 8 * elements
    assert logged._block_owned_bytes() == 2 * 8 * elements

    def stream() -> None:
        for block in logged.stream_blocks():
            del block

    stream()
    with execution_report_scope() as reports:
        tracemalloc.start()
        try:
            base, _ = tracemalloc.get_traced_memory()
            stream()
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    (plan,) = [report.plan for report in reports]
    assert plan.unitBytes == logged._block_task_bytes()
    # One worker holds one block's chain at a time.
    assert peak - base <= plan.unitBytes
