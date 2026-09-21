from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import coo_matrix, csr_matrix

from scarf.readers._sparse import SparseRowStore


@pytest.mark.parametrize("first_nnz", [2**31 - 1, 2**31, 2**31 + 1])
def test_sparse_row_store_preserves_offsets_beyond_int32(
    tmp_path, monkeypatch, first_nnz
):
    source = coo_matrix(np.ones((2, 1), dtype=np.uint16))
    first = SimpleNamespace(
        data=np.array([1], dtype=np.uint16),
        indices=np.array([0], dtype=np.int32),
        indptr=np.array([0, first_nnz], dtype=np.int64),
        nnz=first_nnz,
    )
    # A logical first bucket avoids allocating billions of entries.
    buckets = iter((first, csr_matrix([[1]], dtype=np.uint16)))

    def canonicalize(*_):
        matrix = next(buckets)
        return SimpleNamespace(tocsr=lambda: matrix)

    monkeypatch.setattr("scarf.readers._sparse.canonicalize_sparse", canonicalize)
    store = SparseRowStore(
        lambda: iter((source,)),
        source.shape,
        source.dtype,
        max_bytes=288,
        temp_dir=tmp_path,
    )
    try:
        assert store.indptr.dtype == np.dtype(np.int64)
        np.testing.assert_array_equal(store.indptr, [0, first_nnz, first_nnz + 1])
    finally:
        store.close()
    assert list(tmp_path.iterdir()) == []


def test_sparse_row_store_combines_duplicates_across_chunks_and_cleans_up(tmp_path):
    shape = (4, 5)
    chunks = (
        coo_matrix(([1.5, 6.0], ([2, 0], [3, 4])), shape=shape),
        coo_matrix(([2.5, 0.0], ([2, 1], [3, 2])), shape=shape),
        coo_matrix(shape),
    )
    store = SparseRowStore(
        lambda: iter(chunks),
        shape,
        np.uint16,
        source_dtype=np.float64,
        max_bytes=544,
        temp_dir=tmp_path,
    )
    try:
        expected = np.zeros(shape, dtype=np.uint16)
        expected[0, 4] = 6
        expected[2, 3] = 4
        for start, stop in ((0, 4), (1, 3), (2, 2)):
            result = store.read(start, stop)
            assert result.dtype == expected.dtype
            assert result.has_canonical_format
            np.testing.assert_array_equal(result.toarray(), expected[start:stop])
        assert sorted(path.name for path in next(tmp_path.iterdir()).iterdir()) == [
            "data",
            "indices",
        ]
    finally:
        store.close()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("shape", [(0, 4), (4, 0), (4, 5)])
def test_sparse_row_store_round_trips_empty_axes_and_rows(tmp_path, shape):
    store = SparseRowStore(
        lambda: iter((coo_matrix(shape),)),
        shape,
        np.float32,
        max_bytes=1024,
        temp_dir=tmp_path,
    )
    try:
        np.testing.assert_array_equal(
            store.read(0, shape[0]).toarray(), np.zeros(shape)
        )
    finally:
        store.close()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("max_bytes", "max_nnz", "message"),
    [
        (100, 10, "conversion exceeds the memory limit"),
        (300, 10, "One sparse row"),
        (1024, 1, "exceeds maxNnz"),
    ],
)
def test_sparse_row_store_enforces_limits_and_removes_partial_files(
    tmp_path, max_bytes, max_nnz, message
):
    matrix = coo_matrix([[1, 2], [0, 0]])
    with pytest.raises(MemoryError, match=message):
        SparseRowStore(
            lambda: iter((matrix,)),
            matrix.shape,
            matrix.dtype,
            max_bytes=max_bytes,
            max_nnz=max_nnz,
            temp_dir=tmp_path,
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("invalid_pass", [0, 1])
def test_sparse_row_store_rejects_wrong_shapes_on_either_pass(tmp_path, invalid_pass):
    correct = coo_matrix([[1, 0], [0, 2]])
    wrong = coo_matrix([[1, 0, 0]])
    passes = iter(
        (
            wrong if invalid_pass == 0 else correct,
            wrong if invalid_pass == 1 else correct,
        )
    )
    with pytest.raises(ValueError, match="wrong shape"):
        SparseRowStore(
            lambda: iter((next(passes),)),
            correct.shape,
            correct.dtype,
            max_bytes=1024,
            temp_dir=tmp_path,
        )
    assert list(tmp_path.iterdir()) == []


def test_sparse_row_store_rejects_a_source_that_changes_between_passes(tmp_path):
    passes = iter((coo_matrix([[1, 2]]), coo_matrix([[1, 0]])))
    with pytest.raises(RuntimeError, match="source changed"):
        SparseRowStore(
            lambda: iter((next(passes),)),
            (1, 2),
            np.int64,
            max_bytes=1024,
            temp_dir=tmp_path,
        )
    assert list(tmp_path.iterdir()) == []


def test_sparse_row_store_cleans_up_when_disk_space_is_insufficient(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "scarf.readers._sparse.shutil.disk_usage", lambda _: SimpleNamespace(free=0)
    )
    matrix = coo_matrix([[1]])
    with pytest.raises(OSError, match="temporary bytes"):
        SparseRowStore(
            lambda: iter((matrix,)),
            matrix.shape,
            matrix.dtype,
            max_bytes=1024,
            temp_dir=tmp_path,
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("filename", ["data", "indices"])
def test_sparse_row_store_rejects_truncated_files(tmp_path, filename):
    matrix = coo_matrix([[1, 2]])
    store = SparseRowStore(
        lambda: iter((matrix,)),
        matrix.shape,
        matrix.dtype,
        max_bytes=1024,
        temp_dir=tmp_path,
    )
    try:
        next(tmp_path.glob(f"scarf-sparse-*/{filename}")).write_bytes(b"")
        with pytest.raises(RuntimeError, match="truncated"):
            store.read(0, 1)
    finally:
        store.close()


@pytest.mark.parametrize(("start", "stop"), [(-1, 1), (1, 0), (0, 2)])
def test_sparse_row_store_rejects_windows_outside_the_matrix(tmp_path, start, stop):
    matrix = coo_matrix([[1]])
    store = SparseRowStore(
        lambda: iter((matrix,)),
        matrix.shape,
        matrix.dtype,
        max_bytes=1024,
        temp_dir=tmp_path,
    )
    try:
        with pytest.raises(IndexError, match="outside the matrix"):
            store.read(start, stop)
    finally:
        store.close()
