import threading
import time
from pathlib import Path

import numpy as np
import pytest

from scarf.storage.budget import (
    READ_AHEAD,
    ResourceBudget,
    set_resource_budget,
    worker_prefetch_depth,
)
from scarf.storage.profiles import set_storage_profile
from scarf.utils import (
    ColumnBlockPipeline,
    iter_column_blocks,
    prefetch_blocks,
    remote_column_disk_ahead,
    remote_column_ram_ahead,
)


@pytest.fixture(autouse=True)
def reset_profile():
    set_storage_profile(None)
    yield
    set_storage_profile(None)


def test_prefetch_blocks_preserves_order():
    results = list(prefetch_blocks(range(5), lambda x: x * 2, max_ahead=2))
    assert results == [0, 2, 4, 6, 8]


def test_prefetch_blocks_empty():
    assert list(prefetch_blocks(iter([]), lambda x: x)) == []


def test_prefetch_blocks_single_item():
    assert list(prefetch_blocks([42], lambda x: x + 1, max_ahead=4)) == [43]


def test_prefetch_blocks_respects_max_ahead():
    lock = threading.Lock()
    in_flight = 0
    max_seen = 0

    def slow_fn(x):
        nonlocal in_flight, max_seen
        with lock:
            in_flight += 1
            max_seen = max(max_seen, in_flight)
        time.sleep(0.02)
        with lock:
            in_flight -= 1
        return x

    list(prefetch_blocks(range(8), slow_fn, max_ahead=3))
    assert max_seen <= 3


def test_worker_prefetch_depth_from_budget():
    set_resource_budget(
        ResourceBudget(memoryBytes=8 * 1024**3, workers=4, workingCopies=8)
    )
    try:
        assert worker_prefetch_depth() == READ_AHEAD
        assert worker_prefetch_depth(requested=1) == 1
    finally:
        set_resource_budget(None)


def test_iter_column_blocks_preserves_order(tmp_path: Path):
    def read_block(block_idx: int) -> np.ndarray:
        return np.full((4, 2), block_idx, dtype=np.float64)

    seen = [
        int(arr[0, 0])
        for _, arr, _, _ in iter_column_blocks(
            3, read_block, disk_ahead=1, scratch_dir=str(tmp_path)
        )
    ]
    assert seen == [0, 1, 2]


def test_remote_column_disk_ahead_short_pipelines():
    assert remote_column_disk_ahead(remote=True, n_blocks=4) == 0
    assert remote_column_disk_ahead(remote=True, n_blocks=5) == 5
    assert remote_column_disk_ahead(remote=False, n_blocks=10) == 0


def test_remote_column_ram_ahead():
    assert remote_column_ram_ahead(remote=False, n_blocks=4) == 1
    assert remote_column_ram_ahead(remote=True, n_blocks=4) == 2
    assert remote_column_ram_ahead(remote=True, n_blocks=2) == 1


def test_column_block_pipeline_preserves_order(tmp_path: Path):
    def read_block(block_idx: int) -> np.ndarray:
        return np.full((4, 2), block_idx, dtype=np.float64)

    with ColumnBlockPipeline(
        3, read_block, disk_ahead=1, scratch_dir=str(tmp_path)
    ) as pipeline:
        seen = []
        for block_idx in range(3):
            arr, _, _ = pipeline.take(block_idx)
            seen.append(int(arr[0, 0]))
        assert seen == [0, 1, 2]


def test_column_block_pipeline_uses_disk_staging(tmp_path: Path):
    reads: list[int] = []

    def read_block(block_idx: int) -> np.ndarray:
        reads.append(block_idx)
        time.sleep(0.05)
        return np.full((2, 1), block_idx, dtype=np.float64)

    with ColumnBlockPipeline(
        4, read_block, disk_ahead=2, scratch_dir=str(tmp_path)
    ) as pipeline:
        for block_idx in range(4):
            arr, wait, source = pipeline.take(block_idx)
            assert int(arr[0, 0]) == block_idx
            if block_idx >= 2:
                assert source in {"ram", "disk"}
    assert 0 in reads and 3 in reads


def test_column_block_pipeline_stages_disk_serially(tmp_path: Path):
    inflight = 0
    max_inflight = 0
    lock = threading.Lock()

    def read_block(block_idx: int) -> np.ndarray:
        nonlocal inflight, max_inflight
        with lock:
            inflight += 1
            max_inflight = max(max_inflight, inflight)
        time.sleep(0.05)
        with lock:
            inflight -= 1
        return np.full((2, 1), block_idx, dtype=np.float64)

    with ColumnBlockPipeline(
        5, read_block, disk_ahead=3, scratch_dir=str(tmp_path)
    ) as pipeline:
        for block_idx in range(5):
            pipeline.take(block_idx)

    assert max_inflight <= 2
