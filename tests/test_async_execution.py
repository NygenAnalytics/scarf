import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

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
    zarr_io_concurrency,
)
from scarf.storage.budget import ResourceBudget, detect_workers
from scarf.storage.count_matrix import (
    CountMatrixPolicy,
    persist_count_matrix_plan,
    plan_count_matrix_pair,
)
from scarf.storage.execution import WorkShape, plan_operation
from scarf.storage.io_policy import StorageIoPolicy
from scarf.storage.sharding import write_counts_t


@pytest.fixture(autouse=True)
def _reset_runtime() -> None:
    reset_zarr_runtime_for_tests()
    yield
    reset_zarr_runtime_for_tests()


def _runner(
    resources: ResourceBudget, *, chunksPerShard: int = 1
) -> AsyncStorageRunner:
    operation = plan_operation(
        resources,
        WorkShape(nUnits=resources.workers, unitBytes=1, chunksPerShard=chunksPerShard),
        policy=StorageIoPolicy(readWorkers=resources.workers),
    )
    return AsyncStorageRunner(operation=operation)


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
    from scarf.storage.identity import finalize_counts

    finalize_counts(counts)
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

    second.attrs["source_fingerprint"] = "stale"
    with pytest.raises(ValueError, match="use overwrite=True"):
        write_counts_t(counts, group, policy=_scaled_policy())
    rewritten = write_counts_t(counts, group, policy=_scaled_policy(), overwrite=True)
    assert "reuseSentinel" not in rewritten.attrs
    assert rewritten.attrs["source_fingerprint"] == counts.attrs["content_fingerprint"]
    np.testing.assert_array_equal(np.asarray(rewritten[:]), values.T)


def test_writer_resident_bytes_reduce_destination_width() -> None:
    from scarf.storage.sharding import _counts_t_read_peak, _counts_t_write_peak

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
    per_destination_set = _counts_t_write_peak(plan.countsT) + _counts_t_read_peak(
        plan.counts
    )
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


def test_writer_rejects_a_budget_that_only_fits_uncompressed_buffers() -> None:
    values = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    group, counts = _write_counts(values)
    plan = plan_count_matrix_pair(64, 64, values.dtype, policy=_scaled_policy())

    with pytest.raises(MemoryError, match="countsT write needs"):
        write_counts_t(
            counts,
            group,
            policy=_scaled_policy(),
            resources=ResourceBudget(
                plan.destinationBufferBytes + plan.sourceBufferBytes, 2
            ),
        )

    assert "countsT" not in group


def test_writer_holds_encoding_reservation_until_store_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from zarr.core.array import AsyncArray

    values = np.random.default_rng(0).integers(1, 65535, (64, 64), dtype=np.uint16)
    group, counts = _write_counts(values)
    active: list[AsyncStorageRunner] = []
    original_run = AsyncStorageRunner.run
    original_setitem = AsyncArray.setitem
    admitted: list[tuple[int, int]] = []

    def run(self, operation):
        active.append(self)
        return original_run(self, operation)

    async def setitem(self, selection, value, **kwargs):
        if self.path.endswith("countsT"):
            admitted.append((active[-1].ledger.held_bytes(), value.nbytes))
        return await original_setitem(self, selection, value, **kwargs)

    monkeypatch.setattr(AsyncStorageRunner, "run", run)
    monkeypatch.setattr(AsyncArray, "setitem", setitem)
    result = write_counts_t(
        counts,
        group,
        policy=_scaled_policy(),
        resources=ResourceBudget(64 * 1024, 1),
    )

    np.testing.assert_array_equal(result[:], values.T)
    assert admitted
    assert all(held > 3 * raw for held, raw in admitted)
    assert active[-1].ledger.peak_bytes() <= 64 * 1024
    assert active[-1].ledger.is_empty()


def test_writer_source_reads_share_the_destination_encoding_workspace() -> None:
    from scarf.storage.sharding import _counts_t_read_peak

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
    plan = plan_count_matrix_pair(64, 64, values.dtype, policy=_scaled_policy())
    read_peak = _counts_t_read_peak(plan.counts)
    reads = int(metrics["sourceReadsPerDestination"])
    working = int(metrics["destinationWorkingBytes"])
    # Reads finish before the shard encodes, so they fit in its workspace.
    assert reads > 1
    assert working == max(int(metrics["destinationEncodingBytes"]), reads * read_peak)
    assert reads * read_peak <= working
    assert int(metrics["effectiveDestShardsInFlight"]) == 2
    assert int(metrics["effectiveSourceReadsInFlight"]) == 2 * reads
    assert int(metrics["peakLedgerBytes"]) <= 64 * 1024 * 1024


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
        overwrite=True,
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


def test_host_ceiling_rejects_conflicting_explicit_configuration(monkeypatch) -> None:
    monkeypatch.setattr(
        "scarf.storage.async_execution._active_zarr_workers", lambda: None
    )
    first = ensure_zarr_host_ceiling(2)
    assert first >= 2
    assert zarr.config.get("threading.max_workers") == first
    with pytest.raises(RuntimeError, match="fresh process"):
        ensure_zarr_host_ceiling(8)
    assert zarr.config.get("threading.max_workers") == first


def test_sequential_runners_keep_their_own_plans() -> None:
    host = detect_workers()
    first = _runner(ResourceBudget(1024, 2), chunksPerShard=10)
    second = _runner(ResourceBudget(1024, 4), chunksPerShard=1)
    third = _runner(ResourceBudget(1024, 4), chunksPerShard=10)

    assert first.plan.codecWorkers == min(host, 2)
    assert first.plan.ioConcurrency == min(host, 2)
    assert second.plan.codecWorkers == min(host, 4)
    assert second.plan.ioConcurrency == 1
    assert third.plan.codecWorkers == min(host, 4)
    assert third.plan.ioConcurrency == min(host, 4)

    assert first.run(_current_async_concurrency) == first.plan.ioConcurrency
    assert second.run(_current_async_concurrency) == 1
    assert third.run(_current_async_concurrency) == third.plan.ioConcurrency
    assert zarr.config.get("async.concurrency") == 10


def test_codec_tasks_use_the_worker_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scarf.storage.async_execution.detect_workers", lambda: 64)
    runner = _runner(ResourceBudget(1024, 2))
    barrier = threading.Barrier(2, timeout=5)

    def codec_task() -> int:
        barrier.wait()
        return threading.get_ident()

    async def operation(_active: AsyncStorageRunner) -> list[int]:
        return await asyncio.gather(*(asyncio.to_thread(codec_task) for _ in range(8)))

    assert len(set(runner.run(operation))) == 2
    assert ensure_zarr_host_ceiling() != 2


def test_runner_scopes_async_concurrency_and_restores_configured_default(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "scarf.storage.async_execution._active_zarr_workers", lambda: None
    )
    configure_zarr_runtime(codecWorkers=3, asyncConcurrency=3)
    runner = _runner(
        ResourceBudget(1024, 4),
        chunksPerShard=1,
    )

    assert runner.plan.ioConcurrency == 1
    assert runner.run(_current_async_concurrency) == 1
    assert zarr.config.get("async.concurrency") == 3
    assert int(zarr.config.get("threading.max_workers")) >= 3


def test_overlapping_io_limits_restore_the_remaining_operation() -> None:
    entered = threading.Event()
    release = threading.Event()

    def limited() -> None:
        with zarr_io_concurrency(2):
            entered.set()
            assert release.wait(5)

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(limited)
        assert entered.wait(5)
        try:
            with zarr_io_concurrency(7):
                assert zarr.config.get("async.concurrency") == 2
                release.set()
                pending.result(timeout=5)
                assert zarr.config.get("async.concurrency") == 7
        finally:
            release.set()
    assert zarr.config.get("async.concurrency") == 10


def test_runtime_reconfiguration_cannot_override_an_active_io_limit(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "scarf.storage.async_execution._active_zarr_workers", lambda: None
    )
    configure_zarr_runtime(codecWorkers=3, asyncConcurrency=6)
    ceiling = zarr.config.get("threading.max_workers")
    with zarr_io_concurrency(2):
        with pytest.raises(RuntimeError, match="during active storage operations"):
            configure_zarr_runtime(codecWorkers=5, asyncConcurrency=8)
        assert zarr.config.get("async.concurrency") == 2
        assert zarr.config.get("threading.max_workers") == ceiling
    assert zarr.config.get("async.concurrency") == 6
    with pytest.raises(RuntimeError, match="fresh process"):
        configure_zarr_runtime(codecWorkers=5, asyncConcurrency=8)
    assert zarr.config.get("async.concurrency") == 6


def test_runner_restores_async_concurrency_after_failure() -> None:
    runner = _runner(ResourceBudget(1024, 4), chunksPerShard=10)

    async def boom(_active: AsyncStorageRunner) -> None:
        assert int(zarr.config.get("async.concurrency")) == runner.plan.ioConcurrency
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
    runner = _runner(ResourceBudget(100, 2))

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
    runner = _runner(
        ResourceBudget(1024 * 1024, 4),
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
    from scarf.storage.identity import finalize_counts

    finalize_counts(counts)
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
    assert int(metrics["destinationCommits"]) == int(metrics["destinationOwners"])


def test_async_runner_reports_failure_ledger_leaks() -> None:
    runner = _runner(ResourceBudget(100, 1))

    async def fail_with_leak(active: AsyncStorageRunner) -> None:
        await active.ledger.acquire(40)
        raise ValueError("operation failed")

    with pytest.raises(BaseExceptionGroup, match="execution or cleanup") as raised:
        runner.run(fail_with_leak)
    assert any(
        isinstance(error, RuntimeError) and "still holds 40 bytes" in str(error)
        for error in raised.value.exceptions
    )


def test_async_runner_nested_loop_bounded_io_and_leaks() -> None:
    from scarf.storage.async_execution import (
        _install_numba_thread_cap,
    )

    with pytest.raises(ValueError, match="must be positive"):
        ByteLedger(0)
    with pytest.raises(ValueError, match="must be positive"):
        configure_zarr_runtime(codecWorkers=0, asyncConcurrency=1)
    with pytest.raises(ValueError, match="must be positive"):
        StorageIoPolicy(readWorkers=0)

    runner = _runner(ResourceBudget(64, 1))

    async def leak(active: AsyncStorageRunner) -> None:
        await active.ledger.acquire(8)

    with pytest.raises(RuntimeError, match="still holds"):
        runner.run(leak)

    runner = _runner(ResourceBudget(256, 1))

    async def io_ops(active: AsyncStorageRunner) -> int:
        async def factory() -> int:
            return 7

        async with active.reserve_bytes(8), active.read_lane():
            first = await active.io(factory())
        async with active.reserve_bytes(8), active.commit_lane():
            second = await active.io(factory())
        return first + second

    assert runner.run(io_ops) == 14

    nested = _runner(ResourceBudget(64, 1))

    async def outer() -> int:
        async def inner(active: AsyncStorageRunner) -> int:
            return 3

        return nested.run(inner)

    assert asyncio.run(outer()) == 3

    with pytest.raises(RuntimeError, match="compute pool is not installed"):
        asyncio.run(_runner(ResourceBudget(64, 1)).compute(lambda: 1))

    with pytest.raises(MemoryError, match="One buffer needs"):
        asyncio.run(ByteLedger(8).acquire(32))

    asyncio.run(ByteLedger(16).acquire(0))
    with pytest.raises(ValueError, match="must not be negative"):
        asyncio.run(ByteLedger(16).release(-1))
    runner = _runner(ResourceBudget(64, 1))
    with pytest.raises(RuntimeError, match="read slots"):
        asyncio.run(runner.read_slot())
    with pytest.raises(RuntimeError, match="commit slots"):
        asyncio.run(runner.commit_slot())

    nested = _runner(ResourceBudget(64, 1))

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

    shrunk = plan_operation(ResourceBudget(1024, 4), WorkShape(nUnits=4, unitBytes=1))
    assert shrunk.threadsPerComputeWorker == 1


@pytest.mark.parametrize("kind", ["compute", "io"])
def test_cancelled_native_work_keeps_its_reservation_until_finished(kind):
    runner = _runner(ResourceBudget(100, 2))
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def native():
        started.set()
        assert release.wait(5)
        finished.set()

    async def operation(active):
        async def work():
            async with active.reserve_bytes(40):
                if kind == "compute":
                    await active.compute(native)
                else:
                    await active.io(asyncio.to_thread(native))

        task = asyncio.create_task(work())
        try:
            while not started.is_set():
                await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0.01)
            assert not task.done()
            assert active.ledger.held_bytes() == 40
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
        assert active.ledger.is_empty()

    runner.run(operation)


def test_io_failure_drains_sibling_encoders_before_releasing_buffers():
    runner = _runner(ResourceBudget(100, 2))
    started = threading.Event()
    release = threading.Event()
    failed = threading.Event()

    def native():
        started.set()
        assert release.wait(5)

    async def operation(active):
        async def fail():
            while not started.is_set():
                await asyncio.sleep(0)
            failed.set()
            raise ValueError("encoding failed")

        async def encode():
            await asyncio.gather(asyncio.to_thread(native), fail())

        async def work():
            async with active.reserve_bytes(40):
                await active.io(encode())

        task = asyncio.create_task(work())
        try:
            while not failed.is_set():
                await asyncio.sleep(0.001)
            await asyncio.sleep(0.01)
            assert not task.done()
            assert active.ledger.held_bytes() == 40
        finally:
            release.set()
        with pytest.raises(ValueError, match="encoding failed"):
            await task
        assert active.ledger.is_empty()

    runner.run(operation)


def test_cleanup_failure_does_not_skip_pool_shutdown(monkeypatch):
    def install(_):
        def restore():
            raise RuntimeError("restore failed")

        return restore

    monkeypatch.setattr(
        "scarf.storage.async_execution._install_numba_thread_cap", install
    )
    runner = _runner(ResourceBudget(100, 1))

    async def operation(_):
        raise ValueError("operation failed")

    with pytest.raises(ExceptionGroup) as raised:
        runner.run(operation)
    assert {str(error) for error in raised.value.exceptions} == {
        "operation failed",
        "restore failed",
    }
    assert runner._compute_pool._shutdown
    assert runner._codec_pool._shutdown
    assert zarr.config.get("async.concurrency") == 10


def test_reservation_released_when_post_acquisition_checkpoint_fails(monkeypatch):
    runner = _runner(ResourceBudget(100, 1))

    def checkpoint():
        if runner.ledger.held_bytes():
            raise ValueError("stop after acquisition")

    monkeypatch.setattr("scarf.storage.async_execution.shutdown_checkpoint", checkpoint)

    async def operation(active):
        async with active.reserve_bytes(40):
            pytest.fail("checkpoint must fail before entering")

    with pytest.raises(ValueError, match="stop after acquisition"):
        runner.run(operation)
    assert runner.ledger.is_empty()


def test_completed_io_releases_payloads_without_cyclic_collection():
    import gc
    import weakref

    references = []

    async def produce():
        values = np.ones(1024)
        references.append(weakref.ref(values))
        return values

    async def operation(runner):
        for _ in range(6):
            values = await runner.io(produce())
            del values
            await asyncio.sleep(0)
        # The I/O loop thread drops its finished task just after handing over
        # the result. Give it that moment; cyclic collection stays disabled, so
        # a payload held by a reference cycle is still never freed.
        for _ in range(200):
            if all(reference() is None for reference in references):
                break
            await asyncio.sleep(0.01)
        assert all(reference() is None for reference in references)

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        _runner(ResourceBudget(1024**2, 1)).run(operation)
    finally:
        if was_enabled:
            gc.enable()


def test_task_factory_forwards_keywords_that_newer_asyncio_passes():
    async def operation(active):
        async def value():
            return 7

        async def work():
            loop = asyncio.get_running_loop()
            task = loop.get_task_factory()(loop, value(), name="named")
            return await task, task.get_name()

        return await active.io(work())

    assert _runner(ResourceBudget(1024, 1)).run(operation) == (7, "named")


def test_operation_restores_blas_limits_left_by_concurrent_compute_tasks():
    import time

    from threadpoolctl import ThreadpoolController, threadpool_limits

    controller = ThreadpoolController()

    def blas_threads() -> list[int]:
        return [
            lib.num_threads
            for lib in controller.lib_controllers
            if lib.user_api == "blas"
        ]

    async def operation(runner):
        async def task() -> None:
            await runner.compute(lambda: time.sleep(0.001))

        # Per-task limit scopes on several workers restore each other's values.
        await asyncio.gather(*(task() for _ in range(64)))

    runner = _runner(ResourceBudget(1024**2, 8))
    assert runner.plan.computeWorkers > 1
    # The test session pins BLAS to one thread; raise it so a change shows.
    with threadpool_limits(limits=2, user_api="blas"):
        before = blas_threads()
        if not before or max(before) < 2:
            pytest.skip("needs a multi-threaded BLAS")
        runner.run(operation)
        assert blas_threads() == before
