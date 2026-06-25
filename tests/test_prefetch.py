import threading
import time

import pytest

from scarf.storage.zarr_store import (
    PROFILE_PREFETCH_DEPTH,
    profile_prefetch_depth,
    set_storage_profile,
)
from scarf.utils import prefetch_blocks


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


def test_profile_prefetch_depth():
    set_storage_profile("fast_local")
    assert profile_prefetch_depth() == PROFILE_PREFETCH_DEPTH["fast_local"]
    set_storage_profile("cloud")
    assert profile_prefetch_depth() == PROFILE_PREFETCH_DEPTH["cloud"]
