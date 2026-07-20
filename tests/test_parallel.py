import threading
import time

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.matrix import ChunkedArray
from scarf.storage.budget import (
    READ_AHEAD,
    ResourceBudget,
    set_resource_budget,
    shard_parallelism,
)
from scarf.storage.parallel import in_shard_context, map_shards, stream_shards


@pytest.fixture
def budget():
    set_resource_budget(
        ResourceBudget(memoryBytes=32 * 1024**3, workers=8, workingCopies=8)
    )
    yield
    set_resource_budget(None)


def test_shard_parallelism_spends_budget_within_shard(budget):
    plan = shard_parallelism(workers=8, n_shards=20)
    assert plan.readAhead == READ_AHEAD
    assert plan.ioConcurrency == 8
    assert plan.withinBlockThreads == 1


def test_shard_parallelism_read_ahead_capped_by_shard_count(budget):
    plan = shard_parallelism(workers=8, n_shards=1)
    assert plan.readAhead == 1


def test_shard_parallelism_read_ahead_capped_by_working_copies():
    tight = ResourceBudget(memoryBytes=32 * 1024**3, workers=8, workingCopies=1)
    plan = shard_parallelism(workers=8, n_shards=100, budget=tight)
    assert plan.readAhead == 1
    assert plan.ioConcurrency == 8
    assert plan.withinBlockThreads == 1


def test_map_shards_preserves_order(budget):
    ranges = [(i * 10, i * 10 + 10) for i in range(6)]
    out = map_shards(ranges, lambda idx, s, e: (idx, s, e), workers=8)
    assert out == [(i, i * 10, i * 10 + 10) for i in range(6)]


def test_map_shards_empty(budget):
    assert map_shards([], lambda i, s, e: i, workers=8) == []


def test_map_shards_bounds_in_flight(budget):
    lock = threading.Lock()
    in_flight = 0
    max_seen = 0
    ranges = [(i, i + 1) for i in range(16)]

    def produce(idx, s, e):
        nonlocal in_flight, max_seen
        with lock:
            in_flight += 1
            max_seen = max(max_seen, in_flight)
        time.sleep(0.01)
        with lock:
            in_flight -= 1
        return idx

    map_shards(ranges, produce, workers=8)
    assert max_seen <= READ_AHEAD


def test_map_shards_serial_backend_runs_inline(budget):
    seen_context = []

    def produce(idx, s, e):
        seen_context.append(in_shard_context())
        return idx

    out = map_shards([(0, 1), (1, 2)], produce, workers=8, backend="serial")
    assert out == [0, 1]


def test_nested_map_shards_runs_serial(budget):
    inner_context = []

    def outer(idx, s, e):
        assert in_shard_context() is True

        def inner(j, a, b):
            inner_context.append(in_shard_context())
            return j

        return map_shards([(0, 1), (1, 2), (2, 3)], inner, workers=8)

    map_shards([(0, 1), (1, 2)], outer, workers=8)
    assert inner_context and all(inner_context)


def test_stream_shards_preserves_order(budget):
    out = list(stream_shards(range(5), lambda x: x * 2, workers=4))
    assert out == [0, 2, 4, 6, 8]


def test_stream_shards_bounds_and_restores_io_concurrency():
    with zarr.config.set({"async.concurrency": 7}):
        seen = []

        def fn(x):
            seen.append(zarr.config.get("async.concurrency"))
            return x

        out = list(stream_shards(range(4), fn, workers=2, io_concurrency=3))
        assert out == [0, 1, 2, 3]
        assert seen and all(s == 3 for s in seen)
        assert zarr.config.get("async.concurrency") == 7


def test_stream_shards_config_neutral_without_io():
    with zarr.config.set({"async.concurrency": 5}):
        seen = []

        def fn(x):
            seen.append(zarr.config.get("async.concurrency"))
            return x

        list(stream_shards(range(4), fn, workers=2))
        assert seen and all(s == 5 for s in seen)


def test_map_shards_sets_io_concurrency_from_plan(budget):
    with zarr.config.set({"async.concurrency": 99}):
        seen = []

        def produce(idx, s, e):
            seen.append(zarr.config.get("async.concurrency"))
            return idx

        # budget: workers=8; the whole worker budget is spent within a shard, so
        # async.concurrency is set to 8 for the duration of the op.
        map_shards([(i, i + 1) for i in range(4)], produce, workers=8)
        assert seen and all(s == 8 for s in seen)
        assert zarr.config.get("async.concurrency") == 99


def _toy_chunked(data, chunk_rows, nthreads):
    root = zarr.open_group(store=MemoryStore(), mode="w")
    arr = root.create_array(
        "d", shape=data.shape, chunks=(chunk_rows, data.shape[1]), dtype=data.dtype
    )
    arr[:] = data
    return ChunkedArray(arr, nthreads=nthreads)


def test_stream_blocks_matches_serial_materialization():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((35, 6)).astype(np.float32)
    serial = np.vstack(list(_toy_chunked(data, 10, 1).stream_blocks(nthreads=1)))
    parallel = np.vstack(list(_toy_chunked(data, 10, 4).stream_blocks(nthreads=4)))
    assert np.array_equal(serial, data)
    assert np.array_equal(serial, parallel)


def test_reductions_bit_identical_across_threads():
    rng = np.random.default_rng(1)
    data = rng.standard_normal((37, 5)).astype(np.float32)
    ca1 = _toy_chunked(data, 10, 1)
    ca4 = _toy_chunked(data, 10, 4)
    assert np.array_equal(
        np.asarray(ca1.sum(axis=0).compute(1)),
        np.asarray(ca4.sum(axis=0).compute(4)),
    )
    assert np.array_equal(
        np.asarray(ca1.var(axis=0).compute(1)),
        np.asarray(ca4.var(axis=0).compute(4)),
    )
    m1, s1 = ca1.mean_and_std(nthreads=1)
    m4, s4 = ca4.mean_and_std(nthreads=4)
    assert np.array_equal(m1, m4)
    assert np.array_equal(s1, s4)


def test_compute_matches_source_across_threads():
    rng = np.random.default_rng(2)
    data = rng.standard_normal((40, 4)).astype(np.float32)
    assert np.array_equal(_toy_chunked(data, 8, 1).compute(1), data)
    assert np.array_equal(_toy_chunked(data, 8, 5).compute(5), data)
