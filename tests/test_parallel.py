import threading
import time

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.matrix import ChunkedArray
from scarf.storage.parallel import in_shard_context, map_shards, stream_shards


def test_map_shards_preserves_order():
    ranges = [(i * 10, i * 10 + 10) for i in range(6)]
    out = map_shards(ranges, lambda idx, s, e: (idx, s, e), workers=8)
    assert out == [(i, i * 10, i * 10 + 10) for i in range(6)]


def test_map_shards_empty():
    assert map_shards([], lambda i, s, e: i, workers=8) == []


def test_map_shards_bounds_in_flight():
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
    assert 1 < max_seen <= 8


def test_map_shards_serial_backend_runs_inline():
    seen_context = []

    def produce(idx, s, e):
        seen_context.append(in_shard_context())
        return idx

    out = map_shards([(0, 1), (1, 2)], produce, workers=8, backend="serial")
    assert out == [0, 1]


def test_nested_map_shards_runs_serial():
    inner_context = []

    def outer(idx, s, e):
        assert in_shard_context() is True

        def inner(j, a, b):
            inner_context.append(in_shard_context())
            return j

        return map_shards([(0, 1), (1, 2), (2, 3)], inner, workers=8)

    map_shards([(0, 1), (1, 2)], outer, workers=8)
    assert inner_context and all(inner_context)


def test_stream_shards_preserves_order():
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


def test_stream_shards_cancels_unconsumed_work():
    started = []
    lock = threading.Lock()

    def work(value):
        with lock:
            started.append(value)
        time.sleep(0.02)
        return value

    stream = stream_shards(range(100), work, workers=2)
    assert next(stream) == 0
    stream.close()
    assert set(started).issubset({0, 1, 2})


@pytest.mark.parametrize("workers", [1, 2])
def test_stream_shards_closes_source_iterator(workers):
    closed = False

    def source():
        nonlocal closed
        try:
            yield from range(100)
        finally:
            closed = True

    stream = stream_shards(source(), lambda value: value, workers=workers)
    assert next(stream) == 0
    stream.close()
    assert closed


def test_stream_shards_cancels_pending_work_after_failure():
    started = []
    lock = threading.Lock()

    def work(value):
        with lock:
            started.append(value)
        if value == 0:
            raise RuntimeError("injected worker failure")
        time.sleep(0.05)
        return value

    with pytest.raises(RuntimeError, match="injected worker failure"):
        list(stream_shards(range(100), work, workers=2))
    assert set(started).issubset({0, 1})


def test_io_concurrency_isolated_across_parallel_runtimes():
    before = zarr.config.get("async.concurrency")
    barrier = threading.Barrier(2)
    seen = {3: [], 7: []}

    def run(io_concurrency):
        def inspect_config(value):
            barrier.wait()
            seen[io_concurrency].append(zarr.config.get("async.concurrency"))
            return value

        list(
            stream_shards(
                [0],
                inspect_config,
                workers=1,
                io_concurrency=io_concurrency,
            )
        )

    first = threading.Thread(target=run, args=(3,))
    second = threading.Thread(target=run, args=(7,))
    first.start()
    second.start()
    first.join()
    second.join()
    assert seen == {3: [3], 7: [7]}
    assert zarr.config.get("async.concurrency") == before


def test_map_shards_uses_worker_budget_once_across_tasks_and_io():
    with zarr.config.set({"async.concurrency": 99}):
        seen = []
        lock = threading.Lock()
        in_flight = 0
        max_in_flight = 0

        def produce(idx, s, e):
            nonlocal in_flight, max_in_flight
            with lock:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
            seen.append(zarr.config.get("async.concurrency"))
            time.sleep(0.01)
            with lock:
                in_flight -= 1
            return idx

        map_shards([(i, i + 1) for i in range(16)], produce, workers=8)
        assert seen and all(s == 1 for s in seen)
        assert max_in_flight > 1
        assert max_in_flight * seen[0] <= 8
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
