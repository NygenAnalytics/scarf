import threading
import time

import numpy as np
import zarr

from scarf.utils import iter_column_blocks


def test_iter_column_blocks_preserves_order_and_bounds_reads():
    lock = threading.Lock()
    in_flight = 0
    max_seen = 0

    def read_block(block_idx: int) -> np.ndarray:
        nonlocal in_flight, max_seen
        with lock:
            in_flight += 1
            max_seen = max(max_seen, in_flight)
        time.sleep(0.02)
        with lock:
            in_flight -= 1
        return np.full((4, 2), block_idx, dtype=np.float64)

    results = list(iter_column_blocks(8, read_block, workers=2))
    seen = [int(arr[0, 0]) for _, arr, _, _ in results]
    assert seen == list(range(8))
    assert {source for _, _, _, source in results} == {"direct"}
    assert max_seen <= 2


def test_iter_column_blocks_respects_worker_limit():
    lock = threading.Lock()
    in_flight = 0
    max_seen = 0

    def read_block(block_idx: int) -> np.ndarray:
        nonlocal in_flight, max_seen
        with lock:
            in_flight += 1
            max_seen = max(max_seen, in_flight)
        time.sleep(0.01)
        with lock:
            in_flight -= 1
        return np.array([[block_idx]])

    list(iter_column_blocks(4, read_block, workers=1))
    assert max_seen == 1


def test_iter_column_blocks_splits_workers_between_reads_and_object_io():
    lock = threading.Lock()
    in_flight = 0
    max_seen = 0
    io_concurrency = []

    def read_block(block_idx: int) -> np.ndarray:
        nonlocal in_flight, max_seen
        with lock:
            in_flight += 1
            max_seen = max(max_seen, in_flight)
        io_concurrency.append(zarr.config.get("async.concurrency"))
        time.sleep(0.01)
        with lock:
            in_flight -= 1
        return np.array([[block_idx]])

    with zarr.config.set({"async.concurrency": 99}):
        list(
            iter_column_blocks(
                4,
                read_block,
                workers=2,
                io_concurrency=4,
            )
        )
        assert zarr.config.get("async.concurrency") == 99

    assert max_seen > 1
    assert io_concurrency and all(value == 4 for value in io_concurrency)
    assert max_seen <= 2


def test_iter_column_blocks_empty():
    assert list(iter_column_blocks(0, lambda _: np.empty((0, 0)))) == []
