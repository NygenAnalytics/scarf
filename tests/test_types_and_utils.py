import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.storage.types import (
    array_metadata_shards,
    as_zarr_array,
    as_zarr_group,
)
import scarf.utils.progress as progress_module
from scarf.utils import (
    array_digest,
    clean_array,
    permute_into_chunks,
    rescale_array,
    rolling_window,
    set_verbosity,
    tqdmbar,
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


def test_tqdmbar_enabled_for_tty_or_notebook(monkeypatch):
    import tqdm
    import tqdm.auto as tqdm_auto

    captured: dict[str, bool] = {}

    class FakeTqdm:
        def __init__(self, *args, **kwargs):
            captured["disable"] = bool(kwargs.get("disable"))

        def __iter__(self):
            return iter(())

    monkeypatch.setattr(progress_module, "get_log_level", lambda: 20)
    monkeypatch.setattr(progress_module, "stdout_is_interactive", lambda: False)
    monkeypatch.setattr(progress_module, "is_notebook", lambda: True)
    monkeypatch.setattr(tqdm, "tqdm_notebook", FakeTqdm)
    list(tqdmbar(range(1), desc="test"))
    assert captured["disable"] is False

    monkeypatch.setattr(progress_module, "is_notebook", lambda: False)
    monkeypatch.setattr(progress_module, "stdout_is_interactive", lambda: True)
    monkeypatch.setattr(tqdm_auto, "tqdm", FakeTqdm)
    list(tqdmbar(range(1), desc="test"))
    assert captured["disable"] is False


def test_tqdmbar_disabled_when_redirected_or_quiet(monkeypatch):
    import tqdm.auto as tqdm_auto

    captured: dict[str, bool] = {}

    class FakeTqdm:
        def __init__(self, *args, **kwargs):
            captured["disable"] = bool(kwargs.get("disable"))

        def __iter__(self):
            return iter(())

    monkeypatch.setattr(tqdm_auto, "tqdm", FakeTqdm)
    monkeypatch.setattr(progress_module, "is_notebook", lambda: False)
    monkeypatch.setattr(progress_module, "stdout_is_interactive", lambda: False)
    monkeypatch.setattr(progress_module, "get_log_level", lambda: 20)
    list(tqdmbar(range(1), desc="test"))
    assert captured["disable"] is True

    monkeypatch.setattr(progress_module, "stdout_is_interactive", lambda: True)
    monkeypatch.setattr(progress_module, "get_log_level", lambda: 30)
    list(tqdmbar(range(1), desc="test"))
    assert captured["disable"] is True


def test_rolling_window_smoothes_along_rows():
    data = np.arange(10, dtype=float).reshape(5, 2)
    smoothed = rolling_window(data, w=3)
    expected = np.array(
        [
            [1.0, 2.0],
            [2.0, 3.0],
            [4.0, 5.0],
            [6.0, 7.0],
            [7.0, 8.0],
        ]
    )
    assert np.array_equal(smoothed, expected)


def test_rolling_window_even_and_oversized_windows():
    data = np.arange(5, dtype=float).reshape(-1, 1)

    assert np.array_equal(rolling_window(np.array([[7.0]]), w=1), [[7.0]])
    assert np.array_equal(
        rolling_window(data, w=2).ravel(),
        [0.5, 1.5, 2.5, 3.5, 4.0],
    )
    assert np.array_equal(
        rolling_window(data, w=20).ravel(),
        [1.0, 1.5, 2.0, 2.5, 3.0],
    )
    for window_size in (0, -1):
        with pytest.raises(ValueError, match="greater than zero"):
            rolling_window(data, w=window_size)


def test_array_digest_is_deterministic_and_shape_sensitive():
    values = np.arange(6, dtype=np.int64)

    assert array_digest(values) == array_digest(values.copy())
    assert array_digest(values) != array_digest(values.reshape(2, 3))


def test_permute_into_chunks_preserves_all_indices():
    chunks = permute_into_chunks(10, 3, seed=7)
    merged = np.concatenate(chunks)
    assert np.array_equal(np.sort(merged), np.arange(10))
