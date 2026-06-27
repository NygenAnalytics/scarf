import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf._types import as_zarr_array, as_zarr_group, array_metadata_shards
from scarf.utils import (
    clean_array,
    permute_into_chunks,
    prefetch_blocks,
    rescale_array,
    rolling_window,
    set_verbosity,
)


def test_as_zarr_array_accepts_array():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    arr = root.create_array("data", shape=(3,), dtype="f8")
    assert as_zarr_array(arr) is arr


def test_as_zarr_array_rejects_group():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    group = root.create_group("nested")
    with pytest.raises(TypeError, match="Expected Zarr array"):
        as_zarr_array(group, name="nested")


def test_as_zarr_group_accepts_group():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    group = root.create_group("nested")
    assert as_zarr_group(group) is group


def test_as_zarr_group_rejects_array():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    arr = root.create_array("data", shape=(2,), dtype="f8")
    with pytest.raises(TypeError, match="Expected Zarr group"):
        as_zarr_group(arr, name="data")


def test_array_metadata_shards_returns_none_without_sharding():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    arr = root.create_array("data", shape=(4,), dtype="f8")
    assert array_metadata_shards(arr) is None


def test_clean_array_replaces_nan_inf_and_zero():
    raw = np.array([1.0, np.nan, np.inf, -np.inf, 0.0])
    cleaned = clean_array(raw, fill_val=-1.0)
    assert cleaned[0] == 1.0
    assert cleaned[-1] == -1.0
    assert np.all(np.isfinite(cleaned))


def test_rescale_array_trims_extreme_values():
    values = np.concatenate([np.linspace(-2, 2, 499), np.array([100.0])])
    trimmed = rescale_array(values, frac=0.9)
    assert trimmed.max() < 100.0
    assert trimmed.min() > -100.0


def test_set_verbosity_rejects_invalid_level():
    with pytest.raises(ValueError, match="Please provide a value for level"):
        set_verbosity("NOT_A_REAL_LEVEL")


def test_set_verbosity_accepts_valid_level():
    set_verbosity("ERROR")
    set_verbosity("INFO")


def test_rolling_window_smoothes_along_rows():
    data = np.arange(20, dtype=float).reshape(10, 2)
    smoothed = rolling_window(data, w=3)
    assert smoothed.shape == data.shape
    assert np.all(np.isfinite(smoothed))


def test_permute_into_chunks_preserves_all_indices():
    chunks = permute_into_chunks(10, 3, seed=7)
    merged = np.concatenate(chunks)
    assert np.array_equal(np.sort(merged), np.arange(10))


def test_prefetch_blocks_preserves_order():
    blocks = list(range(8))

    def double(x: int) -> int:
        return x * 2

    results = list(prefetch_blocks(blocks, double, max_ahead=3))
    assert results == [x * 2 for x in blocks]
