import threading
import weakref

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.matrix import ChunkedArray
from scarf.storage.parallel import in_shard_context, stream_shards


def test_progress_closes_display_after_producer_cleanup_fails(monkeypatch):
    from scarf.utils import progress

    closed = []

    class Display:
        def update(self):
            pass

        def close(self):
            closed.append("display")
            raise OSError("display cleanup")

    def blocks():
        try:
            yield 1
        finally:
            closed.append("producer")
            raise RuntimeError("producer cleanup")

    monkeypatch.setattr(progress, "tqdmbar", lambda **kwargs: Display())
    stream = progress.iter_progress(blocks())
    assert next(stream) == 1
    with pytest.raises(ExceptionGroup) as caught:
        stream.close()
    assert closed == ["producer", "display"]
    assert [str(error) for error in caught.value.exceptions] == [
        "producer cleanup",
        "display cleanup",
    ]


def test_array_explicit_thread_limit_is_respected(monkeypatch):
    from scarf.storage.budget import ResourceBudget

    array = ChunkedArray.from_numpy(
        np.ones((32, 4)), block_size=4, nthreads=4, resources=ResourceBudget(1024**2, 4)
    )
    caller = threading.get_ident()
    threads = []
    original = ChunkedArray._materialize_range

    def materialize(self, start, end):
        threads.append(threading.get_ident())
        return original(self, start, end)

    monkeypatch.setattr(ChunkedArray, "_materialize_range", materialize)
    np.testing.assert_array_equal(array.sum(axis=0).compute(nthreads=1), [32.0] * 4)
    assert threads == [caller] * 8


def test_array_budget_includes_storage_retained_by_operand_views():
    from scarf.storage.budget import ResourceBudget

    backing = np.ones((4, 8))
    large = np.ones((10_000, 8))
    array = ChunkedArray.from_numpy(
        backing, block_size=2, resources=ResourceBudget(16_384, 1)
    )
    with pytest.raises(MemoryError, match="Resident data"):
        (array + large[:1]).compute()
    np.testing.assert_array_equal((array + large[:1].copy()).compute(), backing * 2)


def test_stream_shards_bounds_in_flight():
    lock = threading.Lock()
    in_flight = 0
    max_seen = 0
    # Each item waits for seven others, so the stream finishes only if eight
    # items run at once.
    barrier = threading.Barrier(8, timeout=10)

    def produce(value):
        nonlocal in_flight, max_seen
        with lock:
            in_flight += 1
            max_seen = max(max_seen, in_flight)
        barrier.wait()
        with lock:
            in_flight -= 1
        return value

    assert list(stream_shards(range(16), produce, workers=8)) == list(range(16))
    assert max_seen == 8


def test_paused_serial_stream_does_not_mark_its_consumer_as_a_worker():
    contexts = []

    def produce(value):
        contexts.append(in_shard_context())
        return value

    stream = stream_shards([1, 2], produce, workers=1)
    try:
        assert next(stream) == 1
        assert not in_shard_context()
        assert contexts == [True]
        worker_threads = list(
            stream_shards([0, 1], lambda _: threading.get_ident(), workers=2)
        )
        assert all(worker != threading.get_ident() for worker in worker_threads)
    finally:
        stream.close()
    assert not in_shard_context()


def test_stream_shards_serial_backend_runs_inline():
    caller = threading.get_ident()
    threads = []

    def produce(value):
        threads.append(threading.get_ident())
        return value

    out = list(stream_shards([0, 1], produce, workers=8, backend="serial"))
    assert out == [0, 1]
    assert threads == [caller, caller]


def test_nested_stream_shards_run_serial():
    inner_context = []

    def outer(value):
        assert in_shard_context() is True

        def inner(item):
            inner_context.append(in_shard_context())
            return item

        return list(stream_shards([0, 1, 2], inner, workers=8))

    list(stream_shards([0, 1], outer, workers=8))
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
        return value

    stream = stream_shards(range(100), work, workers=2)
    assert next(stream) == 0
    stream.close()
    # Two items are read ahead, and the next is submitted only when the
    # consumer asks for it, so closing leaves the rest of the source unread.
    assert set(started) <= {0, 1}


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
        return value

    with pytest.raises(RuntimeError, match="injected worker failure"):
        list(stream_shards(range(100), work, workers=2))
    assert set(started).issubset({0, 1})


def test_overlapping_parallel_runtimes_share_the_lower_io_limit():
    before = zarr.config.get("async.concurrency")
    barrier = threading.Barrier(2)
    seen = {3: [], 7: []}

    def run(io_concurrency):
        def inspect_config(value):
            barrier.wait()
            seen[io_concurrency].append(zarr.config.get("async.concurrency"))
            barrier.wait()
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
    assert seen == {3: [3], 7: [3]}
    assert zarr.config.get("async.concurrency") == before


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


def test_stream_counts_the_consumed_block_toward_its_limit():
    alive = []
    maximum = 0
    lock = threading.Lock()
    produced = [threading.Event() for _ in range(12)]

    def produce(index):
        nonlocal maximum
        block = np.full(512, index)
        with lock:
            alive.append(weakref.ref(block))
            maximum = max(maximum, sum(ref() is not None for ref in alive))
        produced[index].set()
        return block

    stream = stream_shards(range(12), produce, workers=2)
    try:
        for index in range(12):
            block = next(stream)
            np.testing.assert_array_equal(block, index)
            # Hold the block until the next one exists beside it.
            if index + 1 < 12:
                assert produced[index + 1].wait(10)
            del block
    finally:
        stream.close()
    assert maximum == 2


def test_serial_source_observes_previous_consumption():
    consumed = []

    def source():
        for _ in range(4):
            yield len(consumed)

    for item in stream_shards(source(), lambda value: value, workers=1):
        consumed.append(item)
    assert consumed == [0, 1, 2, 3]


def test_stream_cleanup_reports_the_failure_with_every_cleanup_error():
    closed = []

    def source():
        try:
            yield from range(10)
        finally:
            closed.append("source")
            raise OSError("source cleanup failed")

    def work(value):
        if value == 0:
            raise ValueError("worker failed")
        return value

    with pytest.raises(
        BaseExceptionGroup, match="Block stream failed during cleanup"
    ) as caught:
        list(stream_shards(source(), work, workers=2))
    assert [(type(error), str(error)) for error in caught.value.exceptions] == [
        (ValueError, "worker failed"),
        (OSError, "source cleanup failed"),
    ]
    assert closed == ["source"]


def test_failed_pool_shutdown_still_closes_the_source(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    import scarf.storage.parallel as parallel

    class FailingShutdown(ThreadPoolExecutor):
        def shutdown(self, wait=True, *, cancel_futures=False):
            super().shutdown(wait, cancel_futures=cancel_futures)
            raise RuntimeError("pool shutdown interrupted")

    monkeypatch.setattr(parallel, "ThreadPoolExecutor", FailingShutdown)
    closed = []

    def source():
        try:
            yield from range(4)
        finally:
            closed.append("source")

    stream = stream_shards(source(), lambda value: value, workers=2)
    assert next(stream) == 0
    # Closing early is no failure of its own, so the cleanup error stands alone.
    with pytest.raises(RuntimeError, match="pool shutdown interrupted"):
        stream.close()
    assert closed == ["source"]


def test_progress_stream_reports_a_failure_with_its_cleanup_errors(monkeypatch):
    from scarf.utils import progress

    class Display:
        def update(self):
            pass

        def close(self):
            raise OSError("display cleanup")

    def failing():
        yield 1
        raise ValueError("producer failed")

    monkeypatch.setattr(progress, "tqdmbar", lambda **kwargs: Display())
    with pytest.raises(BaseExceptionGroup, match="Progress stream cleanup") as caught:
        list(progress.iter_progress(failing()))
    assert [(type(error), str(error)) for error in caught.value.exceptions] == [
        (ValueError, "producer failed"),
        (OSError, "display cleanup"),
    ]

    stream = progress.iter_progress(iter([1, 2]))
    assert next(stream) == 1
    with pytest.raises(OSError, match="display cleanup"):
        stream.close()


def test_progress_bars_let_explicit_settings_replace_scarf_defaults(monkeypatch):
    import tqdm.auto

    from scarf.utils.progress import tqdm_params, tqdmbar

    captured: list[dict] = []
    monkeypatch.setattr(
        tqdm.auto, "tqdm", lambda *args, **kwargs: captured.append(kwargs)
    )
    tqdmbar(total=3, bar_format="{n}/{total}")

    assert captured[0]["bar_format"] == "{n}/{total}"
    assert captured[0]["total"] == 3
    assert captured[0]["colour"] == tqdm_params["colour"]
    assert captured[0]["dynamic_ncols"] is True


def test_early_close_surfaces_worker_failure():
    failed = threading.Event()

    def produce(index):
        if index:
            failed.set()
            raise ValueError("producer failed")
        return index

    stream = stream_shards(range(2), produce, workers=2)
    assert next(stream) == 0
    assert failed.wait(5)
    with pytest.raises(ValueError, match="producer failed"):
        stream.close()
