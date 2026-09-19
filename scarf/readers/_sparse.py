import shutil
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from ..utils.arrays import canonicalize_sparse


class SparseRowStore:
    def __init__(
        self,
        chunks: Callable[[], Iterator[coo_matrix]],
        shape: tuple[int, int],
        dtype: Any,
        *,
        max_bytes: int,
        max_nnz: int = np.iinfo(np.int64).max,
        source_dtype: Any | None = None,
        temp_dir: str | Path | None = None,
    ) -> None:
        self.shape = shape
        self.dtype = np.dtype(dtype)
        metadata_bytes = (shape[0] + 1) * 32
        available = int(max_bytes) - metadata_bytes
        if available < 192:
            raise MemoryError("Sparse row conversion exceeds the memory limit")
        entries_per_bucket = max(1, available // 192)
        counts = np.zeros(shape[0], dtype=np.int64)
        for chunk in chunks():
            if chunk.shape != shape:
                raise ValueError("Sparse conversion chunk has the wrong shape")
            counts += np.bincount(chunk.row, minlength=shape[0])
            del chunk
        nnz = int(counts.sum())
        if counts.size and int(counts.max()) > entries_per_bucket:
            raise MemoryError("One sparse row exceeds the conversion memory limit")
        cumulative = np.empty(shape[0] + 1, dtype=np.int64)
        cumulative[0] = 0
        np.cumsum(counts, out=cumulative[1:])
        boundaries = np.empty(shape[0] + 1, dtype=np.int64)
        boundaries[0] = 0
        bucket_count = 0
        while boundaries[bucket_count] < shape[0]:
            start = int(boundaries[bucket_count])
            end = int(
                np.searchsorted(
                    cumulative,
                    int(cumulative[start]) + entries_per_bucket,
                    side="right",
                )
                - 1
            )
            bucket_count += 1
            boundaries[bucket_count] = min(shape[0], max(start + 1, end))
        boundaries = boundaries[: bucket_count + 1]
        del counts
        self._directory = tempfile.TemporaryDirectory(
            prefix="scarf-sparse-", dir=temp_dir
        )
        directory = Path(self._directory.name)
        record_dtype = np.dtype(
            [
                ("row", np.int64),
                ("column", np.int64),
                ("value", self.dtype if source_dtype is None else source_dtype),
            ]
        )
        required = nnz * (record_dtype.itemsize + self.dtype.itemsize + 8)
        try:
            if required > shutil.disk_usage(directory).free:
                raise OSError(f"Sparse row conversion needs {required} temporary bytes")
            written = np.zeros(bucket_count, dtype=np.int64)
            for chunk in chunks():
                if chunk.shape != shape:
                    raise ValueError("Sparse conversion chunk has the wrong shape")
                buckets = np.searchsorted(boundaries[1:], chunk.row, side="right")
                order = np.argsort(buckets, kind="stable")
                sorted_buckets = buckets[order]
                cuts = np.r_[
                    0,
                    np.flatnonzero(sorted_buckets[1:] != sorted_buckets[:-1]) + 1,
                    order.size,
                ]
                for left, right in zip(cuts[:-1], cuts[1:], strict=True):
                    if left == right:
                        continue
                    bucket = int(sorted_buckets[left])
                    selected = order[left:right]
                    records = np.empty(right - left, dtype=record_dtype)
                    records["row"] = chunk.row[selected] - boundaries[bucket]
                    records["column"] = chunk.col[selected]
                    records["value"] = chunk.data[selected]
                    with (directory / f"bucket-{bucket}").open("ab") as stream:
                        records.tofile(stream)
                    written[bucket] += right - left
                    del records
                del chunk, buckets, order, sorted_buckets, cuts
            expected = np.diff(cumulative[boundaries])
            if not np.array_equal(written, expected):
                raise RuntimeError("Sparse source changed during row conversion")
            del cumulative
            self.indptr = np.zeros(shape[0] + 1, dtype=np.int64)
            self._data_path = directory / "data"
            self._indices_path = directory / "indices"
            offset = 0
            with (
                self._data_path.open("wb") as data_stream,
                self._indices_path.open("wb") as index_stream,
            ):
                for bucket, (start, stop) in enumerate(
                    zip(boundaries[:-1], boundaries[1:], strict=True)
                ):
                    path = directory / f"bucket-{bucket}"
                    if written[bucket]:
                        records = np.fromfile(path, dtype=record_dtype)
                        matrix = canonicalize_sparse(
                            coo_matrix(
                                (
                                    records["value"],
                                    (records["row"], records["column"]),
                                ),
                                shape=(int(stop - start), shape[1]),
                            ),
                            self.dtype,
                        ).tocsr()
                        matrix.data.tofile(data_stream)
                        matrix.indices.astype(np.int64, copy=False).tofile(index_stream)
                        self.indptr[start + 1 : stop + 1] = matrix.indptr[1:] + offset
                        offset += int(matrix.nnz)
                        if offset > max_nnz:
                            raise MemoryError(
                                f"Sparse row conversion exceeds maxNnz={max_nnz}"
                            )
                        path.unlink()
                        del records, matrix
                    else:
                        self.indptr[start + 1 : stop + 1] = offset
        except BaseException:
            self._directory.cleanup()
            raise

    def read(self, start: int, stop: int) -> csr_matrix:
        if start < 0 or stop < start or stop > self.shape[0]:
            raise IndexError("Sparse row window is outside the matrix")
        first, last = int(self.indptr[start]), int(self.indptr[stop])
        data = np.fromfile(
            self._data_path,
            dtype=self.dtype,
            count=last - first,
            offset=first * self.dtype.itemsize,
        )
        indices = np.fromfile(
            self._indices_path,
            dtype=np.int64,
            count=last - first,
            offset=first * np.dtype(np.int64).itemsize,
        )
        if data.size != last - first or indices.size != last - first:
            raise RuntimeError("Sparse row store is truncated")
        return csr_matrix(
            (data, indices, self.indptr[start : stop + 1] - first),
            shape=(stop - start, self.shape[1]),
        )

    def close(self) -> None:
        self._directory.cleanup()
