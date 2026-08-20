import asyncio

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.storage.async_execution import (
    AsyncStorageRunner,
    ByteLedger,
    configure_zarr_runtime,
    ensure_zarr_host_ceiling,
    reset_zarr_runtime_for_tests,
)
from scarf.storage.budget import ResourceBudget, detect_workers
from scarf.storage.count_matrix import (
    CountMatrixPolicy,
    persist_count_matrix_plan,
    plan_count_matrix_pair,
)
from scarf.storage.io_policy import StorageIoPolicy
from scarf.storage.sharding import write_counts_t


@pytest.fixture(autouse=True)
def _reset_runtime() -> None:
    reset_zarr_runtime_for_tests()
    yield
    reset_zarr_runtime_for_tests()


def _root() -> zarr.Group:
    return zarr.open_group(store=MemoryStore(), mode="w")


def _scaled_policy() -> CountMatrixPolicy:
    return CountMatrixPolicy(unitBytes=2_000, chunkBytes=200)


def _write_counts(values: np.ndarray) -> tuple[zarr.Group, zarr.Array]:
    plan = plan_count_matrix_pair(
        values.shape[0],
        values.shape[1],
        values.dtype,
        policy=_scaled_policy(),
    )
    root = _root()
    group = root.create_group("RNA")
    counts = group.create_array(
        "counts",
        shape=plan.counts.shape,
        chunks=plan.counts.chunks,
        shards=plan.counts.shards,
        dtype=values.dtype,
        overwrite=True,
    )
    counts[:] = values
    persist_count_matrix_plan(group, plan)
    persist_count_matrix_plan(counts, plan)
    return group, counts


def test_writer_transposes_multiple_chunks_and_edges() -> None:
    values = (
        np.arange(17 * 41, dtype=np.uint16).reshape(17, 41) % np.iinfo(np.uint16).max
    )
    group, counts = _write_counts(values)
    counts_t = write_counts_t(
        counts,
        group,
        policy=_scaled_policy(),
        resources=ResourceBudget(64 * 1024 * 1024, 2),
    )
    assert counts_t.attrs["complete"] is True
    np.testing.assert_array_equal(np.asarray(counts_t[:]), values.T)
    assert counts_t.shape == (41, 17)


def test_writer_pipelines_destination_shards_and_commits() -> None:
    values = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    group, counts = _write_counts(values)
    metrics: dict[str, object] = {}
    counts_t = write_counts_t(
        counts,
        group,
        policy=_scaled_policy(),
        resources=ResourceBudget(32 * 1024 * 1024, 4),
        io=StorageIoPolicy(
            readWorkers=2,
            writeWorkers=2,
            computeWorkers=2,
        ),
        metrics=metrics,
    )
    assert counts_t.attrs["complete"] is True
    np.testing.assert_array_equal(np.asarray(counts_t[:]), values.T)
    plan = plan_count_matrix_pair(64, 64, values.dtype, policy=_scaled_policy())
    assert plan.countsT.shards is not None
    feat_shards = -(-64 // int(plan.countsT.shards[0]))
    cell_shards = -(-64 // int(plan.countsT.shards[1]))
    expected_owners = feat_shards * cell_shards
    assert expected_owners > 1
    assert int(metrics["destinationOwners"]) == expected_owners
    assert int(metrics["destinationCommits"]) == expected_owners
    assert int(metrics["requestedDestShardsInFlight"]) == 2
    assert int(metrics["effectiveDestShardsInFlight"]) == 2
    assert int(metrics["requestedDestCommitsInFlight"]) == 2
    assert int(metrics["requestedComputeWorkers"]) == 2
    assert int(metrics["effectiveComputeWorkers"]) == 2


def test_writer_auto_width_matches_compute_workers() -> None:
    values = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    group, counts = _write_counts(values)
    metrics: dict[str, object] = {}
    counts_t = write_counts_t(
        counts,
        group,
        policy=_scaled_policy(),
        resources=ResourceBudget(64 * 1024 * 1024, 4),
        metrics=metrics,
    )

    assert int(metrics["effectiveDestShardsInFlight"]) == 4
    assert int(metrics["effectiveComputeWorkers"]) == 4
    assert int(metrics["sourceRepeatedDecodeCount"]) == 0
    assert int(metrics["reservedBytes"]) <= 64 * 1024 * 1024
    assert int(metrics["peakLedgerBytes"]) <= 64 * 1024 * 1024
    np.testing.assert_array_equal(np.asarray(counts_t[:]), values.T)


def test_writer_honors_explicit_read_width_above_compute_workers() -> None:
    values = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    group, counts = _write_counts(values)
    metrics: dict[str, object] = {}
    counts_t = write_counts_t(
        counts,
        group,
        policy=_scaled_policy(),
        resources=ResourceBudget(64 * 1024 * 1024, 4),
        io=StorageIoPolicy(readWorkers=8),
        metrics=metrics,
    )

    assert int(metrics["effectiveDestShardsInFlight"]) == 8
    assert int(metrics["effectiveComputeWorkers"]) == 4
    assert int(metrics["sourceRepeatedDecodeCount"]) == 0
    assert int(metrics["reservedBytes"]) <= 64 * 1024 * 1024
    assert int(metrics["peakLedgerBytes"]) <= 64 * 1024 * 1024
    np.testing.assert_array_equal(np.asarray(counts_t[:]), values.T)


def test_writer_reuses_complete_matching_destination() -> None:
    values = np.arange(12, dtype=np.uint16).reshape(3, 4)
    group, counts = _write_counts(values)
    first = write_counts_t(
        counts,
        group,
        policy=_scaled_policy(),
        resources=ResourceBudget(32 * 1024 * 1024, 2),
    )
    first.attrs["reuseSentinel"] = "keep"
    second = write_counts_t(
        counts,
        group,
        policy=_scaled_policy(),
        resources=ResourceBudget(32 * 1024 * 1024, 2),
    )
    assert second.attrs.get("reuseSentinel") == "keep"
    assert second.attrs["complete"] is True
    np.testing.assert_array_equal(np.asarray(second[:]), values.T)


def test_writer_resident_bytes_reduce_destination_width() -> None:
    values = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    group, counts = _write_counts(values)
    plan = plan_count_matrix_pair(
        values.shape[0],
        values.shape[1],
        values.dtype,
        policy=_scaled_policy(),
    )
    budget = 32 * 1024 * 1024
    assert plan.countsT.shards is not None
    dest_unit = (
        int(plan.countsT.shards[0])
        * int(plan.countsT.shards[1])
        * int(values.dtype.itemsize)
    )
    per_destination_set = dest_unit + int(plan.sourceBufferBytes)
    resident = budget - per_destination_set
    metrics: dict[str, object] = {}
    write_counts_t(
        counts,
        group,
        policy=_scaled_policy(),
        resources=ResourceBudget(budget, 8),
        residentBytes=resident,
        io=StorageIoPolicy(
            readWorkers=4,
        ),
        metrics=metrics,
    )
    assert int(metrics["requestedDestShardsInFlight"]) == 4
    assert int(metrics["effectiveDestShardsInFlight"]) == 1
    assert int(metrics["plannedDestinationSetBytes"]) == dest_unit
    assert int(metrics["peakLedgerBytes"]) + resident <= budget


def test_writer_bounds_grouped_source_reads_by_inner_chunk_plan() -> None:
    values = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    group, counts = _write_counts(values)
    metrics: dict[str, object] = {}
    write_counts_t(
        counts,
        group,
        policy=_scaled_policy(),
        resources=ResourceBudget(64 * 1024 * 1024, 8),
        io=StorageIoPolicy(
            readWorkers=2,
        ),
        metrics=metrics,
    )
    assert int(metrics["effectiveDestShardsInFlight"]) == 2
    assert int(metrics["readGroupChunks"]) == 1
    assert int(metrics["sourceReadsPerDestination"]) * int(
        metrics["readGroupChunks"]
    ) <= int(metrics["innerReads"])
    assert int(metrics["effectiveSourceReadsInFlight"]) == (
        int(metrics["effectiveDestShardsInFlight"])
        * int(metrics["sourceReadsPerDestination"])
    )


def test_retry_replaces_incomplete_destination() -> None:
    values = np.arange(12, dtype=np.uint16).reshape(3, 4)
    group, counts = _write_counts(values)
    partial = group.create_array(
        "countsT",
        shape=(4, 3),
        chunks=(2, 3),
        shards=(2, 3),
        dtype=values.dtype,
        overwrite=True,
    )
    partial.attrs["complete"] = False
    partial[:] = 0
    counts_t = write_counts_t(
        counts,
        group,
        policy=_scaled_policy(),
        resources=ResourceBudget(32 * 1024 * 1024, 2),
    )
    assert counts_t.attrs["complete"] is True
    np.testing.assert_array_equal(np.asarray(counts_t[:]), values.T)


def test_writer_rejects_mismatched_persisted_plan() -> None:
    values = np.arange(20, dtype=np.uint16).reshape(4, 5)
    group, counts = _write_counts(values)
    recorded = dict(counts.attrs["scarf:countMatrixLayout"])
    recorded["fingerprint"] = "wrong"
    counts.attrs["scarf:countMatrixLayout"] = recorded

    with pytest.raises(ValueError, match="metadata does not match"):
        write_counts_t(
            counts,
            group,
            policy=_scaled_policy(),
            resources=ResourceBudget(32 * 1024 * 1024, 2),
        )


def test_byte_ledger_is_empty_after_success() -> None:
    values = np.arange(20, dtype=np.uint16).reshape(4, 5)
    group, counts = _write_counts(values)
    runner_holder: list[AsyncStorageRunner] = []
    original = AsyncStorageRunner.run

    def wrapped(self, operation):  # type: ignore[no-untyped-def]
        runner_holder.append(self)
        return original(self, operation)

    AsyncStorageRunner.run = wrapped  # type: ignore[method-assign]
    try:
        write_counts_t(
            counts,
            group,
            policy=_scaled_policy(),
            resources=ResourceBudget(32 * 1024 * 1024, 2),
        )
    finally:
        AsyncStorageRunner.run = original  # type: ignore[method-assign]
    assert runner_holder[0].ledger.is_empty()


async def _current_async_concurrency(_active: AsyncStorageRunner) -> int:
    return int(zarr.config.get("async.concurrency"))


def test_host_ceiling_is_set_once_and_does_not_shrink() -> None:
    first = ensure_zarr_host_ceiling(2)
    assert first >= 2
    assert zarr.config.get("threading.max_workers") == first
    assert ensure_zarr_host_ceiling(8) == first
    assert zarr.config.get("threading.max_workers") == first


def test_sequential_runners_keep_their_own_plans() -> None:
    host = detect_workers()
    first = AsyncStorageRunner(ResourceBudget(1024, 2), chunksPerShard=10)
    second = AsyncStorageRunner(ResourceBudget(1024, 4), chunksPerShard=1)
    third = AsyncStorageRunner(ResourceBudget(1024, 4), chunksPerShard=10)

    assert first.plan.codecWorkerLimit == host
    assert first.plan.zarrAsyncConcurrency == min(host, 10)
    assert second.plan.codecWorkerLimit == host
    assert second.plan.zarrAsyncConcurrency == 1
    assert third.plan.codecWorkerLimit == host
    assert third.plan.zarrAsyncConcurrency == min(host, 10)

    assert first.run(_current_async_concurrency) == first.plan.zarrAsyncConcurrency
    assert second.run(_current_async_concurrency) == 1
    assert third.run(_current_async_concurrency) == third.plan.zarrAsyncConcurrency
    assert zarr.config.get("async.concurrency") == 10


def test_runner_scopes_async_concurrency_and_restores_configured_default() -> None:
    configure_zarr_runtime(codecWorkers=3, asyncConcurrency=3)
    runner = AsyncStorageRunner(
        ResourceBudget(1024, 4),
        chunksPerShard=1,
    )

    assert runner.plan.zarrAsyncConcurrency == 1
    assert runner.run(_current_async_concurrency) == 1
    assert zarr.config.get("async.concurrency") == 3
    assert int(zarr.config.get("threading.max_workers")) >= 3


def test_runner_restores_async_concurrency_after_failure() -> None:
    runner = AsyncStorageRunner(ResourceBudget(1024, 4), chunksPerShard=10)

    async def boom(_active: AsyncStorageRunner) -> None:
        assert (
            int(zarr.config.get("async.concurrency"))
            == runner.plan.zarrAsyncConcurrency
        )
        raise ValueError("operation failed")

    with pytest.raises(ValueError, match="operation failed"):
        runner.run(boom)
    assert zarr.config.get("async.concurrency") == 10


def test_write_counts_t_accepts_a_later_larger_worker_budget() -> None:
    values = np.arange(12, dtype=np.uint16).reshape(3, 4)
    group, counts = _write_counts(values)
    first = write_counts_t(
        counts,
        group,
        policy=_scaled_policy(),
        resources=ResourceBudget(32 * 1024 * 1024, 2),
    )
    del group["countsT"]
    second = write_counts_t(
        counts,
        group,
        policy=_scaled_policy(),
        resources=ResourceBudget(32 * 1024 * 1024, 4),
    )
    assert first.attrs["complete"] is True
    assert second.attrs["complete"] is True
    np.testing.assert_array_equal(np.asarray(second[:]), values.T)


def test_byte_ledger_rejects_over_release() -> None:
    async def exercise() -> None:
        ledger = ByteLedger(100)
        await ledger.acquire(40)
        with pytest.raises(RuntimeError, match="ledger holding 40"):
            await ledger.release(41)
        await ledger.release(40)
        assert ledger.is_empty()

    asyncio.run(exercise())


def test_runner_splits_ledger_wait_from_held_time() -> None:
    runner = AsyncStorageRunner(ResourceBudget(100, 2))

    async def contend(active: AsyncStorageRunner) -> None:
        async def hold() -> None:
            async with active.reserve_bytes(80):
                await asyncio.sleep(0.05)

        async def wait_for_bytes() -> None:
            async with active.reserve_bytes(80):
                pass

        await asyncio.gather(hold(), wait_for_bytes())

    runner.run(contend)
    assert runner.readerWaitSeconds >= 0.04


def test_compute_workers_apply_local_numba_cap() -> None:
    import numba

    observed: list[int] = []
    runner = AsyncStorageRunner(
        ResourceBudget(1024 * 1024, 4),
        computeWorkerLimit=4,
        threadsPerComputeWorker=1,
    )

    async def operation(active: AsyncStorageRunner) -> None:
        def probe() -> int:
            observed.append(int(numba.get_num_threads()))
            return 0

        await asyncio.gather(*[active.compute(probe) for _ in range(4)])

    runner.run(operation)
    assert observed
    assert all(threads == 1 for threads in observed)


def test_writer_completes_when_numba_and_many_compute_workers() -> None:
    """Concurrent compute must not deadlock on Numba thread caps."""
    import numba

    assert numba.get_num_threads() >= 2
    values = np.array(
        [
            [5, 0, 1, 0, 0, 2],
            [0, 3, 0, 4, 0, 0],
            [1, 2, 0, 0, 0, 0],
            [0, 0, 0, 5, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [2, 1, 3, 1, 0, 0],
        ],
        dtype=np.uint32,
    )
    from scarf.storage.count_matrix import CountMatrixPolicy
    from scarf.writers import create_cell_data, create_zarr_count_assay
    from scarf.writers.counts_t import finalize_writer_counts_t

    root = zarr.open_group(store=MemoryStore(), mode="w")
    create_cell_data(
        root,
        None,
        ids=np.array([f"c{i}" for i in range(values.shape[0])]),
        names=np.array([f"c{i}" for i in range(values.shape[0])]),
        profile="fast_local",
    )
    counts = create_zarr_count_assay(
        root,
        "RNA",
        None,
        values.shape[0],
        feat_ids=np.array([f"f{i}" for i in range(values.shape[1])]),
        feat_names=np.array(["MT-CO1", "RPS3", "GENE_A", "RPL5", "ZERO", "GENE_B"]),
        dtype="uint32",
        profile="fast_local",
        policy=CountMatrixPolicy(unitBytes=48, chunkBytes=16),
    )
    counts[:] = values
    finalize_writer_counts_t(
        root,
        "RNA",
        None,
        profile="fast_local",
        resources=ResourceBudget(64 * 1024 * 1024, 8),
    )
    np.testing.assert_array_equal(np.asarray(root["RNA/countsT"][:]), values.T)


def test_writer_source_aligned_destinations_do_not_repeat_decodes() -> None:
    values = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    group, counts = _write_counts(values)
    metrics: dict[str, object] = {}
    counts_t = write_counts_t(
        counts,
        group,
        policy=_scaled_policy(),
        resources=ResourceBudget(32 * 1024 * 1024, 4),
        io=StorageIoPolicy(readWorkers=6, writeWorkers=2, computeWorkers=2),
        metrics=metrics,
    )
    np.testing.assert_array_equal(np.asarray(counts_t[:]), values.T)
    assert int(metrics["sourceRepeatedDecodeCount"]) == 0
    assert int(metrics["fusedDestinationStrips"]) == 0
    assert int(metrics["destinationSets"]) == int(metrics["destinationOwners"])


def test_async_runner_reports_failure_ledger_leaks() -> None:
    runner = AsyncStorageRunner(ResourceBudget(100, 1))

    async def fail_with_leak(active: AsyncStorageRunner) -> None:
        await active.ledger.acquire(40)
        raise ValueError("operation failed")

    with pytest.raises(BaseExceptionGroup, match="leaked admitted bytes") as raised:
        runner.run(fail_with_leak)
    assert any(
        isinstance(error, RuntimeError) and "still holds 40 bytes" in str(error)
        for error in raised.value.exceptions
    )


def test_async_runner_nested_loop_bounded_io_and_leaks() -> None:
    from scarf.storage.async_execution import (
        _install_numba_thread_cap,
        resolve_execution_plan,
    )

    with pytest.raises(ValueError, match="must be positive"):
        ByteLedger(0)
    with pytest.raises(ValueError, match="must be positive"):
        configure_zarr_runtime(codecWorkers=0, asyncConcurrency=1)
    with pytest.raises(ValueError, match="must be positive"):
        StorageIoPolicy(readWorkers=0)

    runner = AsyncStorageRunner(ResourceBudget(64, 1))

    async def leak(active: AsyncStorageRunner) -> None:
        await active.ledger.acquire(8)

    with pytest.raises(RuntimeError, match="still holds"):
        runner.run(leak)

    runner = AsyncStorageRunner(ResourceBudget(256, 1))

    async def io_ops(active: AsyncStorageRunner) -> int:
        async def factory() -> int:
            return 7

        first = await active.bounded_read(8, factory)
        second = await active.bounded_commit(8, factory)
        return first + second

    assert runner.run(io_ops) == 14

    nested = AsyncStorageRunner(ResourceBudget(64, 1))

    async def outer() -> int:
        async def inner(active: AsyncStorageRunner) -> int:
            return 3

        return nested.run(inner)

    assert asyncio.run(outer()) == 3

    with pytest.raises(RuntimeError, match="compute pool is not installed"):
        asyncio.run(AsyncStorageRunner(ResourceBudget(64, 1)).compute(lambda: 1))

    with pytest.raises(MemoryError, match="One buffer needs"):
        asyncio.run(ByteLedger(8).acquire(32))

    asyncio.run(ByteLedger(16).acquire(0))
    with pytest.raises(ValueError, match="must not be negative"):
        asyncio.run(ByteLedger(16).release(-1))
    runner = AsyncStorageRunner(ResourceBudget(64, 1))
    with pytest.raises(RuntimeError, match="read slots"):
        asyncio.run(runner.read_slot())
    with pytest.raises(RuntimeError, match="commit slots"):
        asyncio.run(runner.commit_slot())

    nested = AsyncStorageRunner(ResourceBudget(64, 1))

    async def nested_outer() -> None:
        async def boom(_active: AsyncStorageRunner) -> None:
            raise ValueError("nested failure")

        nested.run(boom)

    with pytest.raises(ValueError, match="nested failure"):
        asyncio.run(nested_outer())

    import sys

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setitem(sys.modules, "numba", None)
    try:
        assert _install_numba_thread_cap(2) is None
    finally:
        monkeypatch.undo()

    shrunk = resolve_execution_plan(
        ResourceBudget(1024, 4),
        computeWorkerLimit=4,
        threadsPerComputeWorker=4,
    )
    assert shrunk.threadsPerComputeWorker == 1


def test_shared_source_decode_cache_errors_and_sharing() -> None:
    from scarf.storage.sharding import _SharedSourceDecode

    cache = _SharedSourceDecode()
    runner = AsyncStorageRunner(ResourceBudget(1024, 2))

    async def decode_ops(active: AsyncStorageRunner) -> None:
        owner_started = asyncio.Event()
        allow_failure = asyncio.Event()

        async def boom() -> np.ndarray:
            owner_started.set()
            await allow_failure.wait()
            raise RuntimeError("decode fail")

        failed_key = (0, 1, 0, 1)
        owner = asyncio.create_task(cache.get(failed_key, 8, active, boom))
        await owner_started.wait()
        waiter = asyncio.create_task(cache.get(failed_key, 8, active, boom))
        await asyncio.sleep(0)
        assert cache._users[failed_key] == 2
        allow_failure.set()
        failures = await asyncio.gather(owner, waiter, return_exceptions=True)
        assert all(
            isinstance(failure, RuntimeError) and str(failure) == "decode fail"
            for failure in failures
        )

        async def load() -> np.ndarray:
            await asyncio.sleep(0.01)
            return np.ones(4, dtype=np.uint16)

        first, second = await asyncio.gather(
            cache.get((1, 2, 3, 4), 8, active, load),
            cache.get((1, 2, 3, 4), 8, active, load),
        )
        assert {first[1], second[1]} == {True, False}
        await cache.release((1, 2, 3, 4), active)
        await cache.release((1, 2, 3, 4), active)

    runner.run(decode_ops)
