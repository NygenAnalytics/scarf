import asyncio

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.storage.async_execution import (
    AsyncStorageRunner,
    ByteLedger,
    configure_zarr_runtime,
    reset_zarr_runtime_for_tests,
)
from scarf.storage.budget import ResourceBudget
from scarf.storage.count_matrix import (
    CountMatrixLayoutPolicy,
    persist_count_matrix_plan,
    plan_count_matrix_pair,
)
from scarf.storage.sharding import write_counts_t_experimental


@pytest.fixture(autouse=True)
def _reset_runtime() -> None:
    reset_zarr_runtime_for_tests()
    yield
    reset_zarr_runtime_for_tests()


def _root() -> zarr.Group:
    return zarr.open_group(store=MemoryStore(), mode="w")


def _scaled_policy() -> CountMatrixLayoutPolicy:
    return CountMatrixLayoutPolicy(
        targetReadUnitBytes=2_000,
        targetChunkBytes=200,
    )


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


def test_experimental_writer_transposes_multiple_chunks_and_edges() -> None:
    values = (
        np.arange(17 * 41, dtype=np.uint16).reshape(17, 41) % np.iinfo(np.uint16).max
    )
    group, counts = _write_counts(values)
    counts_t = write_counts_t_experimental(
        counts,
        group,
        policy=_scaled_policy(),
        resources=ResourceBudget(64 * 1024 * 1024, 2),
    )
    assert counts_t.attrs["complete"] is True
    np.testing.assert_array_equal(np.asarray(counts_t[:]), values.T)
    assert counts_t.shape == (41, 17)


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
    counts_t = write_counts_t_experimental(
        counts,
        group,
        policy=_scaled_policy(),
        resources=ResourceBudget(32 * 1024 * 1024, 2),
    )
    assert counts_t.attrs["complete"] is True
    np.testing.assert_array_equal(np.asarray(counts_t[:]), values.T)


def test_experimental_writer_rejects_mismatched_persisted_plan() -> None:
    values = np.arange(20, dtype=np.uint16).reshape(4, 5)
    group, counts = _write_counts(values)
    recorded = dict(counts.attrs["scarf:countMatrixLayout"])
    recorded["fingerprint"] = "wrong"
    counts.attrs["scarf:countMatrixLayout"] = recorded

    with pytest.raises(ValueError, match="metadata does not match"):
        write_counts_t_experimental(
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
        write_counts_t_experimental(
            counts,
            group,
            policy=_scaled_policy(),
            resources=ResourceBudget(32 * 1024 * 1024, 2),
        )
    finally:
        AsyncStorageRunner.run = original  # type: ignore[method-assign]
    assert runner_holder[0].ledger.is_empty()


def test_incompatible_zarr_runtime_fails_closed() -> None:
    from scarf.storage.async_execution import ExecutionPlan, install_zarr_runtime

    install_zarr_runtime(
        ExecutionPlan(
            codecWorkerLimit=1,
            zarrAsyncConcurrency=1,
            computeWorkerLimit=1,
            readGroupsInFlight=1,
            destinationCommitsInFlight=1,
            chunksPerShard=10,
        )
    )
    with pytest.raises(RuntimeError, match="already has a process runtime"):
        install_zarr_runtime(
            ExecutionPlan(
                codecWorkerLimit=3,
                zarrAsyncConcurrency=3,
                computeWorkerLimit=1,
                readGroupsInFlight=1,
                destinationCommitsInFlight=1,
                chunksPerShard=10,
            )
        )


def test_runner_uses_explicit_process_concurrency_across_array_geometries() -> None:
    configure_zarr_runtime(codecWorkers=3, asyncConcurrency=3)
    runner = AsyncStorageRunner(
        ResourceBudget(1024, 4),
        chunksPerShard=1,
    )

    async def read_nothing(_active: AsyncStorageRunner) -> str:
        return "ok"

    assert runner.run(read_nothing) == "ok"
    assert runner.plan.zarrAsyncConcurrency == 3


def test_byte_ledger_rejects_over_release() -> None:
    async def exercise() -> None:
        ledger = ByteLedger(100)
        await ledger.acquire(40)
        with pytest.raises(RuntimeError, match="ledger holding 40"):
            await ledger.release(41)
        await ledger.release(40)
        assert ledger.is_empty()

    asyncio.run(exercise())


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
