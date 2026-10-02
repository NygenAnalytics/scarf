"""The count storage dtype policy, its scanners, and the finite-count rule."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import zarr
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from zarr.storage import MemoryStore

from scarf.readers import H5adReader
from scarf.storage.count_dtype import count_storage_dtype
from scarf.utils import count_values
from scarf.utils.arrays import canonicalize_sparse
from scarf.utils.count_values import (
    CountValueRange,
    compressed_count_ranges,
    dense_count_ranges,
    new_count_ranges,
)
from scarf.writers import H5adToZarr


def compressed_count_range(*args, **kwargs) -> CountValueRange:
    """Return the one range of a scan without groups."""
    (value_range,) = compressed_count_ranges(*args, **kwargs)
    return value_range


def dense_count_range(*args, **kwargs) -> CountValueRange:
    """Return the one range of a scan without groups."""
    (value_range,) = dense_count_ranges(*args, **kwargs)
    return value_range


def _dtype_of(values: np.ndarray, source: object | None = None) -> np.dtype:
    value_range = CountValueRange()
    value_range.update(values)
    return count_storage_dtype(values.dtype if source is None else source, value_range)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (np.array([0, 255], dtype=np.float32), np.uint8),
        (np.array([256], dtype=np.float32), np.uint16),
        (np.array([65_535], dtype=np.float64), np.uint16),
        (np.array([65_536], dtype=np.float64), np.uint32),
        (np.array([2**32 - 1], dtype=np.int64), np.uint32),
        (np.array([2**32], dtype=np.int64), np.uint64),
        (np.array([2**64 - 1], dtype=np.uint64), np.uint64),
        (np.array([2.0**65], dtype=np.float64), np.float64),
        (np.array([-0.0, 3.0], dtype=np.float32), np.uint8),
        (np.array([True, False]), np.uint8),
        (np.array([1.5, 2.0], dtype=np.float32), np.float32),
        (np.array([-1, 5], dtype=np.int16), np.int16),
        (np.array([-2.0, 5.0], dtype=np.float64), np.float64),
        (np.array([7], dtype=np.uint32), np.uint8),
    ],
    ids=[
        "uint8-maximum",
        "past-uint8",
        "uint16-maximum",
        "past-uint16",
        "uint32-maximum",
        "past-uint32",
        "uint64-maximum",
        "float-past-uint64",
        "negative-zero",
        "bool",
        "fractional",
        "negative-integer",
        "negative-float",
        "narrowed-unsigned",
    ],
)
def test_storage_dtype_is_the_narrowest_unsigned_dtype_of_integral_counts(
    values, expected
):
    assert _dtype_of(values) == np.dtype(expected)


def test_storage_dtype_of_other_values_keeps_the_native_source_dtype():
    assert count_storage_dtype(np.float16, CountValueRange(integral=False)) == (
        np.dtype(np.float32)
    )
    big_endian = np.array([0.5], dtype=">f4")
    assert _dtype_of(big_endian) == np.dtype(np.float32)
    assert _dtype_of(big_endian).isnative
    # A range that never saw a value stores in the narrowest dtype.
    assert count_storage_dtype(np.float64, CountValueRange()) == np.dtype(np.uint8)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_value_range_rejects_non_finite_counts(value):
    value_range = CountValueRange()
    with pytest.raises(ValueError, match="finite values"):
        value_range.update(np.array([1.0, value], dtype=np.float32))


def test_value_range_rejects_values_that_are_not_real_numbers():
    with pytest.raises(TypeError, match="real numbers"):
        CountValueRange().update(np.array([1 + 2j]))
    with pytest.raises(TypeError, match="real numbers"):
        CountValueRange().update(np.array(["1"]))
    with pytest.raises(TypeError, match="real numbers"):
        compressed_count_range(
            np.array([0, 1]),
            np.array([0]),
            np.array([1 + 2j]),
            minorSize=1,
            maxBytes=1 << 20,
        )


def test_value_range_settles_at_the_first_fractional_or_negative_block():
    value_range = CountValueRange()
    value_range.update(np.array([3, 9], dtype=np.int32))
    value_range.update(np.array([-1], dtype=np.int32))
    # Later blocks no longer change the range, but are still checked for NaN
    # and infinity.
    value_range.update(np.array([0.5, 1e9]))
    assert value_range == CountValueRange(integral=False, maximum=9)
    value_range.update(np.array([], dtype=np.float32))
    assert value_range == CountValueRange(integral=False, maximum=9)
    for value in (np.nan, np.inf):
        with pytest.raises(ValueError, match="finite values"):
            value_range.update(np.array([1.0, value]))


class _CountingSlices:
    """Slice a NumPy array and count the reads."""

    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        self.shape = values.shape
        self.reads = 0

    def __getitem__(self, selection):
        self.reads += 1
        return self.values[selection]


def _compressed(matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return matrix.indptr, matrix.indices, matrix.data


@pytest.mark.parametrize("dtype", [np.float32, np.int8, np.uint8])
def test_compressed_range_sums_duplicates_whatever_the_encoding(dtype):
    # One 3 x 4 matrix with duplicate coordinates (0, 1) and (2, 3), stored as
    # unsorted CSR and as CSC.
    encodings = {
        "csr": (
            np.array([0, 2, 3, 6]),
            np.array([1, 1, 0, 3, 2, 3]),
            np.array([100, 100, 7, 60, 1, 60]),
            4,
        ),
        "csc": (
            np.array([0, 1, 3, 4, 6]),
            np.array([1, 0, 0, 2, 2, 2]),
            np.array([7, 100, 100, 1, 60, 60]),
            3,
        ),
    }
    for indptr, indices, data, minor in encodings.values():
        values = data.astype(dtype)
        originals = (indptr.copy(), indices.copy(), values.copy())
        value_range = compressed_count_range(
            indptr, indices, values, minorSize=minor, maxBytes=1 << 20
        )
        # Narrow integers widen before they are summed, so 200 cannot wrap.
        assert value_range == CountValueRange(integral=True, maximum=200)
        assert count_storage_dtype(dtype, value_range) == np.uint8
        # The windows are views of the caller's arrays, which stay unsorted.
        for array, original in zip((indptr, indices, values), originals, strict=True):
            np.testing.assert_array_equal(array, original)


# Scan memory per value, so tests can size budgets in values.
_PER_VALUE = count_values._SCAN_BYTES_PER_VALUE


def test_compressed_range_reads_a_long_vector_whole_and_every_value(monkeypatch):
    monkeypatch.setattr(count_values, "_SCAN_WINDOW_VALUES", 4)
    matrix = csr_matrix(
        np.array(
            [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [0, 9, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
    )
    # The first row holds more values than a window and is read whole.
    assert compressed_count_range(
        *_compressed(matrix), minorSize=8, maxBytes=_PER_VALUE * 8
    ) == CountValueRange(integral=True, maximum=9)

    for dtype, first in ((np.float32, 0.5), (np.int32, -1)):
        settling = matrix.astype(dtype)
        settling.data[0] = first
        data = _CountingSlices(settling.data)
        value_range = compressed_count_range(
            settling.indptr, settling.indices, data, minorSize=8, maxBytes=1 << 20
        )
        assert value_range == CountValueRange(integral=False, maximum=0)
        # The scan reads past the value that settles the range, so a later
        # NaN or infinity is still found.
        assert data.reads == 2
    late = matrix.copy()
    late.data[0] = 0.5
    late.data[-1] = np.inf
    with pytest.raises(ValueError, match="finite values"):
        compressed_count_range(*_compressed(late), minorSize=8, maxBytes=1 << 20)

    with pytest.raises(MemoryError, match="holds 8 values"):
        compressed_count_range(
            *_compressed(matrix), minorSize=8, maxBytes=_PER_VALUE * 8 - 1
        )
    # A budget that leaves nothing for the scan reports no negative limit.
    with pytest.raises(MemoryError, match="more than the 0-byte limit"):
        compressed_count_range(*_compressed(matrix), minorSize=8, maxBytes=-5)


def test_compressed_range_reads_each_pointer_once(monkeypatch):
    monkeypatch.setattr(count_values, "_SCAN_WINDOW_VALUES", 8)
    # Forty rows of two values: a pointer block covers eight rows and two windows.
    values = np.zeros((40, 5), dtype=np.float32)
    values[:, [1, 3]] = np.arange(1, 81, dtype=np.float32).reshape(40, 2)
    matrix = csr_matrix(values)
    indptr = _CountingSlices(matrix.indptr)
    data = _CountingSlices(matrix.data)
    value_range = compressed_count_range(
        indptr, matrix.indices, data, minorSize=5, maxBytes=1 << 20
    )
    assert value_range == CountValueRange(integral=True, maximum=80)
    assert indptr.reads == 5
    assert data.reads == 10


def test_compressed_range_does_not_depend_on_the_budget():
    rng = np.random.default_rng(3)
    values = rng.poisson(1.0, size=(200, 30)).astype(np.float32)
    values[17, 4] = 70_000
    matrix = csr_matrix(values)
    ranges = {
        compressed_count_range(*_compressed(matrix), minorSize=30, maxBytes=budget)
        == CountValueRange(integral=True, maximum=70_000)
        for budget in (_PER_VALUE * 30, _PER_VALUE * 1_000, 1 << 30)
    }
    assert ranges == {True}


def test_dense_range_reads_row_blocks_within_the_budget():
    values = np.arange(12, dtype=np.float64).reshape(4, 3)
    blocks: list[tuple[int, int]] = []

    def read(start: int, stop: int) -> np.ndarray:
        blocks.append((start, stop))
        return values[start:stop]

    value_range = dense_count_range(read, 4, 3, maxBytes=2 * 3 * _PER_VALUE)
    assert value_range == CountValueRange(integral=True, maximum=11)
    assert blocks == [(0, 2), (2, 4)]

    blocks.clear()
    values[0, 0] = 0.5
    assert dense_count_range(read, 4, 3, maxBytes=3 * _PER_VALUE).integral is False
    # Every row is read after the range settles, to find NaN and infinity.
    assert blocks == [(0, 1), (1, 2), (2, 3), (3, 4)]
    values[3, 2] = np.nan
    with pytest.raises(ValueError, match="finite values"):
        dense_count_range(read, 4, 3, maxBytes=3 * _PER_VALUE)

    with pytest.raises(MemoryError, match="the 0-byte limit"):
        dense_count_range(read, 4, 3, maxBytes=-1)
    with pytest.raises(MemoryError, match="One dense row"):
        dense_count_range(read, 4, 3, maxBytes=3 * _PER_VALUE - 1)
    assert dense_count_range(read, 0, 3, maxBytes=0) == CountValueRange()


def test_scans_resolve_one_range_for_each_group_of_features():
    values = np.array(
        [[300, 0, 2, 0], [0, 5, 0, 1], [7, 0, 0.5, 0]],
        dtype=np.float32,
    )
    # Features 0 and 1 form group 0, and features 2 and 3 group 1, so only
    # group 1 holds a fraction.
    groups = np.array([0, 0, 1, 1])
    expected = [np.dtype(np.uint16), np.dtype(np.float32)]

    def dtypes(ranges: list[CountValueRange]) -> list[np.dtype]:
        return [count_storage_dtype(np.float32, value_range) for value_range in ranges]

    # Features are the minor axis of CSR and the vectors of CSC.
    for matrix, axis, minor in (
        (csr_matrix(values), 1, 4),
        (csc_matrix(values), 0, 3),
    ):
        ranges = compressed_count_ranges(
            *_compressed(matrix),
            minorSize=minor,
            maxBytes=1 << 20,
            groups=groups,
            groupAxis=axis,
        )
        assert dtypes(ranges) == expected
    ranges = dense_count_ranges(
        lambda start, stop: values[start:stop],
        3,
        4,
        maxBytes=1 << 20,
        groups=groups,
    )
    assert dtypes(ranges) == expected
    # A scan in parts extends the ranges it is given.
    matrix = csr_matrix(values)
    ranges = new_count_ranges(groups)
    for start, stop in ((0, 2), (2, 3)):
        compressed_count_ranges(
            matrix.indptr[start : stop + 1],
            matrix.indices,
            matrix.data,
            minorSize=4,
            maxBytes=1 << 20,
            groups=groups,
            valueRanges=ranges,
        )
    assert dtypes(ranges) == expected
    assert new_count_ranges(None) == [CountValueRange()]


@pytest.mark.parametrize(
    ("duplicates", "integral"),
    [([0.1, 0.2, 0.7], False), ([0.7, 0.2, 0.1], True)],
)
def test_compressed_range_sums_float_duplicates_as_writers_store_them(
    duplicates, integral
):
    # Fractional duplicates sum to 1 in exact arithmetic, but their float sum
    # depends on the order of the terms.
    columns = np.array([2, 0, 2, 5, 2])
    data = np.array([duplicates[0], 3.0, duplicates[1], 2.0, duplicates[2]])
    indptr = np.array([0, 5])
    stored = canonicalize_sparse(
        coo_matrix((data, (np.zeros(5, dtype=np.int64), columns)), shape=(1, 6))
    ).data
    expected = CountValueRange()
    expected.update(stored)
    value_range = compressed_count_range(
        indptr, columns, data, minorSize=6, maxBytes=1 << 20
    )
    assert value_range == expected
    assert value_range.integral is integral


def test_count_summary_rejects_non_finite_counts_and_keeps_negative_values():
    from scarf.storage.identity import CountSummary

    root = zarr.open_group(store=MemoryStore(), mode="w")
    floats = root.create_array("floats", shape=(2, 3), dtype=np.float32)
    summary = CountSummary(floats)
    summary.update(0, np.array([[1.5, -2.0, 0.0]], dtype=np.float32))
    assert summary.rowSums[0] == -0.5
    for value in (np.nan, np.inf):
        with pytest.raises(ValueError, match="finite values"):
            summary.update(1, np.array([[1.0, value, 0.0]], dtype=np.float32))

    integers = root.create_array("integers", shape=(1, 2), dtype=np.int16)
    signed = CountSummary(integers)
    signed.update(0, np.array([[-3, 4]], dtype=np.int16))
    assert signed.rowSums[0] == 1


_VALUES = np.array(
    [
        [0, 3, 0, 1, 0],
        [2500, 0, 0, 0, 4],
        [0, 0, 0, 0, 0],
        [1, 1, 7, 0, 2],
    ],
    dtype=np.int64,
)


def _write_h5ad(
    path: Path,
    values: np.ndarray,
    *,
    encoding: str = "csr",
    dtype: object = np.float32,
    order: str = "sorted",
) -> Path:
    with h5py.File(path, "w") as h5:
        if encoding == "dense":
            h5.create_dataset("X", data=values.astype(dtype))
        else:
            matrix = (csr_matrix if encoding == "csr" else csc_matrix)(values)
            data = matrix.data.astype(dtype)
            indices = matrix.indices.copy()
            for start, stop in zip(matrix.indptr[:-1], matrix.indptr[1:], strict=True):
                if order == "descending":
                    permutation = np.arange(stop - start)[::-1]
                elif order == "random":
                    permutation = np.random.default_rng(start).permutation(stop - start)
                else:
                    continue
                data[start:stop] = data[start:stop][permutation]
                indices[start:stop] = indices[start:stop][permutation]
            group = h5.create_group("X")
            group.attrs["encoding-type"] = f"{encoding}_matrix"
            group.attrs["shape"] = values.shape
            group.create_dataset("data", data=data)
            group.create_dataset("indices", data=indices)
            group.create_dataset("indptr", data=matrix.indptr)
        h5.create_group("obs").create_dataset(
            "_index",
            data=np.array([f"c{i}".encode() for i in range(values.shape[0])]),
        )
        var = h5.create_group("var")
        names = np.array([f"g{i}".encode() for i in range(values.shape[1])])
        var.create_dataset("_index", data=names)
        var.create_dataset("feature_name", data=names)
        h5.create_group("obsm")
    return path


def _import(path: Path, **options: object) -> zarr.Array:
    reader = H5adReader(str(path), feature_name_key="feature_name")
    store = MemoryStore()
    try:
        H5adToZarr(reader, zarr_loc=store, nthreads=1, **options).dump()
    finally:
        reader.close()
    return zarr.open_group(store=store, mode="r")["RNA/counts"]


def test_h5ad_counts_import_identically_from_every_encoding(tmp_path):
    encodings = [
        ("csr", np.float32, "sorted"),
        ("csr", np.float64, "sorted"),
        ("csr", np.float16, "sorted"),
        ("csr", np.int32, "sorted"),
        ("csr", np.int64, "sorted"),
        ("csr", np.uint32, "sorted"),
        ("csr", np.float32, "descending"),
        ("csr", np.float32, "random"),
        ("csc", np.float32, "sorted"),
        ("csc", np.int32, "random"),
        ("dense", np.float32, "sorted"),
        ("dense", np.int32, "sorted"),
    ]
    imported = set()
    for encoding, dtype, order in encodings:
        path = _write_h5ad(
            tmp_path / f"{encoding}-{np.dtype(dtype).name}-{order}.h5ad",
            _VALUES,
            encoding=encoding,
            dtype=dtype,
            order=order,
        )
        counts = _import(path)
        np.testing.assert_array_equal(counts[:], _VALUES)
        imported.add((np.dtype(counts.dtype), counts.attrs["content_fingerprint"]))
    assert len(imported) == 1
    assert next(iter(imported))[0] == np.dtype(np.uint16)


@pytest.mark.parametrize(
    ("dtype", "maximum", "expected"),
    [
        (np.float32, 70_000, np.uint32),
        (np.float64, 5e9, np.uint64),
        (np.int32, 300, np.uint16),
    ],
)
def test_h5ad_integral_counts_import_unsigned(tmp_path, dtype, maximum, expected):
    values = _VALUES.astype(np.float64)
    values[1, 0] = maximum
    path = _write_h5ad(tmp_path / "counts.h5ad", values, dtype=dtype)
    counts = _import(path)
    assert counts.dtype == np.dtype(expected)
    np.testing.assert_array_equal(counts[:], values.astype(expected))


@pytest.mark.parametrize(
    ("dtype", "value"),
    [(np.float32, 1.5), (np.float32, -3), (np.int32, -3), (np.float64, 0.25)],
)
def test_h5ad_counts_that_are_not_non_negative_integers_keep_their_dtype(
    tmp_path, dtype, value
):
    values = _VALUES.astype(np.float64)
    values[3, 4] = value
    path = _write_h5ad(tmp_path / "counts.h5ad", values, dtype=dtype)
    counts = _import(path)
    assert counts.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(counts[:], values.astype(dtype))


@pytest.mark.parametrize("encoding", ["csr", "csc", "dense"])
@pytest.mark.parametrize("value", [np.nan, np.inf])
@pytest.mark.parametrize("after_a_fraction", [False, True])
def test_h5ad_non_finite_counts_fail_before_the_destination_exists(
    tmp_path, monkeypatch, encoding, value, after_a_fraction
):
    values = _VALUES.astype(np.float32)
    values[1, 4] = value
    if after_a_fraction:
        # The fraction settles the range in an earlier scan window than the
        # one that holds the NaN or infinity.
        monkeypatch.setattr(count_values, "_SCAN_WINDOW_VALUES", 2)
        values[0, 1] = 0.5
    path = _write_h5ad(tmp_path / "counts.h5ad", values, encoding=encoding)
    destination = tmp_path / "counts.zarr"
    reader = H5adReader(str(path), feature_name_key="feature_name")
    try:
        with pytest.raises(ValueError, match="finite values"):
            H5adToZarr(reader, zarr_loc=str(destination), nthreads=1)
    finally:
        reader.close()
    assert not destination.exists()


def test_h5ad_storage_dtype_does_not_depend_on_the_budget(tmp_path, monkeypatch):
    # One gene is detected in every cell, so its CSC column is longer than a
    # scan window and is read whole.
    monkeypatch.setattr(count_values, "_SCAN_WINDOW_VALUES", 1_000)
    values = np.zeros((3_000, 4), dtype=np.float32)
    values[:, 1] = np.arange(3_000) % 300 + 1
    values[::7, 3] = 2
    path = _write_h5ad(tmp_path / "column.h5ad", values, encoding="csc")
    imported = {
        (np.dtype(counts.dtype), counts.attrs["content_fingerprint"])
        for counts in (_import(path, mem_budget=budget) for budget in ("8M", "1G"))
    }
    assert len(imported) == 1
    assert next(iter(imported))[0] == np.dtype(np.uint16)

    reader = H5adReader(str(path), feature_name_key="feature_name")
    try:
        # Scanning the dense column needs more than a 64 KiB budget allows.
        with pytest.raises(MemoryError, match="compressed vector holds 3000"):
            reader.count_value_ranges(64 * 1024)
    finally:
        reader.close()


def test_h5ad_reader_yields_source_values_for_every_encoding(tmp_path):
    for encoding in ("csr", "csc", "dense"):
        path = _write_h5ad(tmp_path / f"{encoding}.h5ad", _VALUES, encoding=encoding)
        reader = H5adReader(str(path), feature_name_key="feature_name")
        try:
            assert reader.sourceMatrixDtype == np.float32
            batches = list(reader.consume(3))
            assert {batch.dtype for batch in batches} == {np.dtype(np.float32)}
            np.testing.assert_array_equal(
                np.vstack([batch.toarray() for batch in batches]), _VALUES
            )
        finally:
            reader.close()
    with pytest.raises(TypeError, match="dtype"):
        H5adReader(str(path), dtype="uint16")  # type: ignore[call-arg]
