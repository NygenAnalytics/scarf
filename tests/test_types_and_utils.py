import gc
import weakref

import numpy as np
import pytest
import zarr
from scipy.sparse import coo_matrix
from zarr.storage import MemoryStore

from scarf.storage.types import (
    array_metadata_shards,
    as_zarr_array,
    as_zarr_group,
)
from scarf.utils import (
    array_digest,
    clean_array,
    configure_output,
    permute_into_chunks,
    rescale_array,
    rolling_window,
    set_verbosity,
    compute_with_progress,
    tqdmbar,
)
from scarf.utils.arrays import canonicalize_sparse, checked_sparse_cast
from scarf.utils.progress import iter_progress


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


def test_tqdmbar_uses_explicit_progress_independently_of_severity(monkeypatch):
    import tqdm.auto as tqdm_auto

    captured: list[bool] = []

    class FakeTqdm:
        def __init__(self, *args, **kwargs):
            captured.append(bool(kwargs.get("disable")))

        def __iter__(self):
            return iter(())

    monkeypatch.setattr(tqdm_auto, "tqdm", FakeTqdm)
    try:
        configure_output(level="ERROR", progress=True)
        list(tqdmbar(range(1), desc="test"))
        set_verbosity("WARNING")
        list(tqdmbar(range(1), desc="test"))
        configure_output(level="DEBUG", progress=False)
        list(tqdmbar(range(1), desc="test"))
        list(tqdmbar(range(1), desc="test", disable=False))
        configure_output(progress=True)
        list(tqdmbar(range(1), desc="test", disable=True))
    finally:
        configure_output(level="INFO", progress=False, timestamps=False)

    assert captured == [False, False, True, True, True]


def test_compute_with_progress_uses_explicit_progress_setting():
    calls: list[tuple[int, str | None]] = []

    class Deferred:
        def compute(self, nthreads, msg):
            calls.append((nthreads, msg))
            return np.array([1])

    try:
        configure_output(progress=False)
        compute_with_progress(Deferred(), "Computing", 2)
        configure_output(progress=True)
        compute_with_progress(Deferred(), "Computing", 3)
    finally:
        configure_output(progress=False)

    assert calls == [(2, None), (3, "Computing")]
    np.testing.assert_array_equal(
        compute_with_progress(np.array([2, 3])),
        np.array([2, 3]),
    )


def test_progress_iterator_releases_consumed_values():
    references: list[weakref.ReferenceType[object]] = []

    class Chunk:
        pass

    def source():
        for _ in range(2):
            chunk = Chunk()
            references.append(weakref.ref(chunk))
            yield chunk

    stream = iter_progress(source(), total=2, disable=True)
    first = next(stream)
    second = next(stream)
    del first
    gc.collect()

    assert references[0]() is None
    assert references[1]() is second
    stream.close()


def test_progress_iterator_closes_source_on_early_exit():
    closed = False

    def source():
        nonlocal closed
        try:
            yield object()
            yield object()
        finally:
            closed = True

    stream = iter_progress(source(), total=2, disable=True)
    next(stream)
    stream.close()

    assert closed


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


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (
            np.array([1 + 0j]),
            "Complex sparse values cannot use an integer destination",
        ),
        (
            np.array([1.5]),
            "cannot be represented by the destination dtype",
        ),
        (
            np.array([np.nan]),
            "cannot be represented by the destination dtype",
        ),
    ],
)
def test_checked_sparse_cast_rejects_lossy_integer_conversions(values, message):
    with pytest.raises(OverflowError, match=message):
        checked_sparse_cast(values, np.int32)


def test_canonicalize_sparse_handles_empty_and_already_canonical_inputs():
    empty = coo_matrix(
        (
            np.array([], dtype=np.int64),
            (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
            ),
        ),
        shape=(2, 2),
    )
    empty.has_canonical_format = False

    canonical_empty = canonicalize_sparse(empty)

    assert canonical_empty.shape == (2, 2)
    assert canonical_empty.nnz == 0
    assert canonical_empty.has_canonical_format

    canonical = coo_matrix(
        (
            np.array([1.0, 2.0]),
            (
                np.array([0, 1]),
                np.array([0, 1]),
            ),
        ),
        shape=(2, 2),
    )
    canonical.sum_duplicates()

    returned = canonicalize_sparse(canonical, dtype=np.int16)

    assert returned is canonical
    assert returned.dtype == np.dtype(np.int16)
    np.testing.assert_array_equal(returned.data, [1, 2])


def test_canonicalize_sparse_detects_int64_duplicate_overflow():
    maximum = np.iinfo(np.int64).max
    duplicated = coo_matrix(
        (
            np.array([maximum, 1], dtype=np.int64),
            (
                np.array([0, 0]),
                np.array([0, 0]),
            ),
        ),
        shape=(1, 1),
    )

    with pytest.raises(OverflowError, match="Duplicate sparse values exceed"):
        canonicalize_sparse(duplicated)


def test_array_digest_rejects_object_values():
    with pytest.raises(TypeError, match="object arrays"):
        array_digest(np.array([object()], dtype=object))
