"""Lazy blockwise matrix operations over NumPy and Zarr arrays."""

from collections.abc import Callable, Iterator
from typing import Any, cast

import numpy as np
import zarr
from numpy.typing import NDArray

from ..storage.budget import (
    DEFAULT_READ_AHEAD_BLOCKS,
    ResourceBudget,
    admit_stream,
    admitted_worker_count,
)
from ..storage.geometry import ArrayGeometry, array_geometry
from ._indexing import is_contiguous, local_positions
from ._operations import (
    _Op,
    _binary_op,
    _classify_operand,
    _matmul_op,
    _unary_op,
)
from ._reductions import ReductionOp, _Reduction
from .blocks import Block

__all__ = ["ChunkedArray"]

type Backing = np.ndarray | zarr.Array
type BlockFn = Callable[[int, int, int], NDArray[Any]]


class ChunkedArray:
    """A lazy, row-chunked 2D matrix backed by Zarr or NumPy."""

    __array_priority__ = 1000.0

    def __init__(
        self,
        backing: Backing,
        rows: np.ndarray | None = None,
        cols: np.ndarray | None = None,
        ops: list[_Op] | None = None,
        out_cols: int | None = None,
        block_size: int | None = None,
        nthreads: int = 1,
        resources: ResourceBudget | None = None,
        is_numpy: bool | None = None,
    ) -> None:
        self._backing = backing
        self._rows = None if rows is None else np.asarray(rows)
        self._cols = None if cols is None else np.asarray(cols)
        self._ops: list[_Op] = list(ops) if ops else []
        self._resources = resources
        self._nthreads = (
            max(1, min(int(nthreads), resources.workers))
            if resources is not None
            else max(1, int(nthreads))
        )
        if is_numpy is None:
            is_numpy = isinstance(backing, np.ndarray)
        self._is_numpy = is_numpy
        backing_rows, backing_cols = backing.shape
        self._n_rows = backing_rows if self._rows is None else int(self._rows.size)
        base_cols = backing_cols if self._cols is None else int(self._cols.size)
        self._out_cols = base_cols if out_cols is None else int(out_cols)
        if block_size is None:
            if self._is_numpy:
                block_size = self._n_rows if self._n_rows > 0 else 1
            else:
                from ..storage.partition import row_band

                block_size = row_band(
                    self._geometry(),
                    fallback=int(self._backing.shape[0]),
                )
        self._block_size = max(int(block_size), 1)

    @classmethod
    def from_numpy(
        cls,
        arr: np.ndarray,
        block_size: int | None = None,
        nthreads: int = 1,
        resources: ResourceBudget | None = None,
    ) -> "ChunkedArray":
        arr = np.asarray(arr)
        return cls(
            arr,
            block_size=block_size,
            nthreads=nthreads,
            resources=resources,
            is_numpy=True,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return (self._n_rows, self._out_cols)

    @property
    def out_cols(self) -> int:
        return self._out_cols

    @property
    def dtype(self) -> np.dtype[Any]:
        if not self._ops:
            return self._backing.dtype
        base_columns = (
            int(self._backing.shape[1]) if self._cols is None else int(self._cols.size)
        )
        sample = np.empty((0, base_columns), dtype=self._backing.dtype)
        for operation in self._ops:
            sample = operation.apply(sample, 0, 0)
        return sample.dtype

    @property
    def chunksize(self) -> tuple[int, int]:
        return (
            min(self._block_size, self._n_rows) if self._n_rows else self._block_size,
            self._out_cols,
        )

    @property
    def numblocks(self) -> tuple[int, int]:
        return (self._n_block_count(), 1)

    @property
    def chunks(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        sizes = tuple(end - start for start, end in self._ranges())
        return (sizes if sizes else (0,), (self._out_cols,))

    @property
    def nthreads(self) -> int:
        return self._nthreads

    def __len__(self) -> int:
        return self._n_rows

    def _n_block_count(self) -> int:
        if self._n_rows == 0:
            return 0
        return int(np.ceil(self._n_rows / self._block_size))

    def _ranges(self) -> list[tuple[int, int]]:
        from ..storage.partition import contiguous_ranges

        return contiguous_ranges(self._n_rows, self._block_size)

    def _read(self, start: int, end: int) -> np.ndarray:
        if self._rows is None:
            row_selection: slice | np.ndarray = slice(start, end)
            rows_contiguous = True
        else:
            row_selection = self._rows[start:end]
            rows_contiguous = is_contiguous(row_selection)
        if self._is_numpy:
            numpy_backing = cast(np.ndarray, self._backing)
            if self._cols is None:
                return np.asarray(numpy_backing[row_selection])
            return np.asarray(numpy_backing[row_selection][:, self._cols])

        zarr_backing = cast(zarr.Array, self._backing)
        if self._cols is None:
            if isinstance(row_selection, slice):
                return np.asarray(zarr_backing[row_selection, :])
            if rows_contiguous:
                return np.asarray(
                    zarr_backing[
                        int(row_selection[0]) : int(row_selection[-1]) + 1,
                        :,
                    ]
                )
            return np.asarray(
                zarr_backing.get_orthogonal_selection(
                    (row_selection, slice(None)),
                )
            )
        if isinstance(row_selection, slice):
            return np.asarray(
                zarr_backing.get_orthogonal_selection(
                    (row_selection, self._cols),
                )
            )
        if rows_contiguous:
            return np.asarray(
                zarr_backing.get_orthogonal_selection(
                    (
                        slice(
                            int(row_selection[0]),
                            int(row_selection[-1]) + 1,
                        ),
                        self._cols,
                    )
                )
            )
        return np.asarray(
            zarr_backing.get_orthogonal_selection(
                (row_selection, self._cols),
            )
        )

    def _materialize_range(self, start: int, end: int) -> np.ndarray:
        array = self._read(start, end)
        for operation in self._ops:
            array = operation.apply(array, start, end)
        return array

    def _geometry(self) -> ArrayGeometry | None:
        return array_geometry(self._backing)

    def _max_decode_bytes(self) -> int:
        geometry = self._geometry()
        return 0 if geometry is None else geometry.nominalChunkBytes()

    def _block_owned_bytes(self) -> int:
        """Bytes one materialized row block owns, excluding its chunk decode."""
        rows = min(self._block_size, max(1, self._n_rows))
        elements = rows * max(1, self._out_cols)
        input_bytes = elements * max(1, int(self._backing.dtype.itemsize))
        output_bytes = elements * max(1, int(self.dtype.itemsize))
        return input_bytes + (output_bytes if self._ops else 0)

    def _block_task_bytes(self) -> int:
        """Bytes one row block owns where its reader decodes one chunk at a time."""
        return self._block_owned_bytes() + self._max_decode_bytes()

    def _map_blocks(
        self,
        fn: BlockFn,
        nthreads: int | None,
        msg: str | None,
    ) -> list[NDArray[Any]]:
        from ..storage.parallel import map_shards

        ranges = self._ranges()
        requested = self._nthreads if nthreads is None else max(1, int(nthreads))
        workers = requested
        if self._resources is not None:
            # map_shards pins Zarr to one decode per worker, so one chunk is exact.
            workers = admitted_worker_count(
                self._resources,
                taskBytes=self._block_task_bytes(),
                requested=requested,
            )
        results = map_shards(ranges, fn, workers=workers, msg=msg)
        return [np.asarray(result) for result in results]

    def stream_blocks(
        self,
        nthreads: int | None = None,
        msg: str | None = None,
        prefetch: int | None = None,
    ) -> Iterator[np.ndarray]:
        """Yield materialized row blocks with bounded read-ahead."""
        yield from self._stream_blocks(
            nthreads=nthreads,
            msg=msg,
            prefetch=prefetch,
            row_mask=None,
            resident_bytes=0,
        )

    def _stream_blocks(
        self,
        *,
        nthreads: int | None,
        msg: str | None,
        prefetch: int | None,
        row_mask: np.ndarray | None,
        resident_bytes: int = 0,
    ) -> Iterator[np.ndarray]:
        from ..storage.parallel import stream_shards

        threads = self._nthreads if nthreads is None else max(1, int(nthreads))
        requested = (
            DEFAULT_READ_AHEAD_BLOCKS if prefetch is None else max(1, int(prefetch))
        )
        depth = min(threads, requested)
        ranges = self._ranges()
        mask = None if row_mask is None else np.asarray(row_mask)
        if mask is not None:
            if mask.dtype != bool or mask.shape != (self._n_rows,):
                raise ValueError(
                    "row_mask must be a boolean vector matching array rows"
                )
            ranges = [
                (start, end) for start, end in ranges if bool(mask[start:end].any())
            ]

        io_concurrency: int | None = None
        if self._resources is not None:
            admission = admit_stream(
                self._resources,
                nBlocks=min(depth, max(1, len(ranges))),
                blockBytes=self._block_owned_bytes(),
                decodeBytes=self._max_decode_bytes(),
                residentBytes=max(0, int(resident_bytes)),
            )
            depth = admission.outerWorkers
            io_concurrency = admission.ioConcurrency
        within = max(1, threads // depth)

        def materialize(interval: tuple[int, int]) -> np.ndarray:
            start, end = interval
            values = self._materialize_range(start, end)
            return values if mask is None else values[mask[start:end]]

        yield from stream_shards(
            ranges,
            materialize,
            workers=depth,
            within_block_threads=within,
            io_concurrency=io_concurrency,
            msg=msg,
            total=len(ranges),
        )

    def map_blocks(
        self,
        fn: BlockFn,
        nthreads: int | None = None,
        msg: str | None = None,
    ) -> list[NDArray[Any]]:
        """Map a function over row blocks in order."""
        return self._map_blocks(fn, nthreads, msg)

    def compute(
        self,
        nthreads: int | None = None,
        msg: str | None = None,
    ) -> np.ndarray:
        if self._n_rows == 0:
            return np.empty((0, self._out_cols), dtype=self.dtype)

        def materialize(_: int, start: int, end: int) -> NDArray[Any]:
            return self._materialize_range(start, end)

        parts = self._map_blocks(materialize, nthreads, msg)
        return np.vstack(parts) if len(parts) > 1 else parts[0]

    def __array__(self, dtype: np.dtype[Any] | None = None) -> np.ndarray:
        array = self.compute()
        return array.astype(dtype) if dtype is not None else array

    @property
    def blocks(self) -> Iterator[Block]:
        for start, end in self._ranges():
            yield Block(self, start, end)

    def _with_op(
        self,
        operation: _Op,
        out_cols: int | None = None,
    ) -> "ChunkedArray":
        return ChunkedArray(
            self._backing,
            rows=self._rows,
            cols=self._cols,
            ops=self._ops + [operation],
            out_cols=self._out_cols if out_cols is None else out_cols,
            block_size=self._block_size,
            nthreads=self._nthreads,
            resources=self._resources,
            is_numpy=self._is_numpy,
        )

    def _with_block_size(self, block_size: int) -> "ChunkedArray":
        if block_size < 1:
            raise ValueError("block_size must be greater than zero")
        return ChunkedArray(
            self._backing,
            rows=self._rows,
            cols=self._cols,
            ops=self._ops,
            out_cols=self._out_cols,
            block_size=block_size,
            nthreads=self._nthreads,
            resources=self._resources,
            is_numpy=self._is_numpy,
        )

    def _unary(self, func: Callable[..., NDArray[Any]]) -> "ChunkedArray":
        return self._with_op(_unary_op(func))

    def _binary(
        self,
        func: Callable[..., NDArray[Any]],
        other: object,
        side: str,
    ) -> "ChunkedArray":
        if isinstance(other, _Reduction):
            other = other._arr
        kind, operand = _classify_operand(
            other,
            self._n_rows,
            self._out_cols,
        )
        return self._with_op(_binary_op(func, operand, side, kind))

    def __array_ufunc__(
        self,
        ufunc: Any,
        method: str,
        *inputs: Any,
        **kwargs: Any,
    ) -> Any:
        if method != "__call__" or kwargs.get("out") is not None:
            return NotImplemented
        if len(inputs) == 1:
            return self._unary(ufunc)
        if len(inputs) == 2:
            left, right = inputs
            if left is self:
                return self._binary(ufunc, right, "left")
            return self._binary(ufunc, left, "right")
        return NotImplemented

    def __mul__(self, o: object) -> "ChunkedArray":
        return self._binary(np.multiply, o, "left")

    def __rmul__(self, o: object) -> "ChunkedArray":
        return self._binary(np.multiply, o, "right")

    def __truediv__(self, o: object) -> "ChunkedArray":
        return self._binary(np.true_divide, o, "left")

    def __rtruediv__(self, o: object) -> "ChunkedArray":
        return self._binary(np.true_divide, o, "right")

    def __add__(self, o: object) -> "ChunkedArray":
        return self._binary(np.add, o, "left")

    def __radd__(self, o: object) -> "ChunkedArray":
        return self._binary(np.add, o, "right")

    def __sub__(self, o: object) -> "ChunkedArray":
        return self._binary(np.subtract, o, "left")

    def __rsub__(self, o: object) -> "ChunkedArray":
        return self._binary(np.subtract, o, "right")

    def __gt__(self, o: object) -> "ChunkedArray":
        return self._binary(np.greater, o, "left")

    def __lt__(self, o: object) -> "ChunkedArray":
        return self._binary(np.less, o, "left")

    def __ge__(self, o: object) -> "ChunkedArray":
        return self._binary(np.greater_equal, o, "left")

    def __le__(self, o: object) -> "ChunkedArray":
        return self._binary(np.less_equal, o, "left")

    def dot(self, b: np.ndarray | NDArray[Any]) -> "ChunkedArray":
        operand_array = np.asarray(b)
        return self._with_op(
            _matmul_op(operand_array),
            out_cols=operand_array.shape[1],
        )

    @staticmethod
    def _local_positions(key: object, length: int) -> np.ndarray | None:
        return local_positions(key, length)

    def __getitem__(self, key: object) -> "ChunkedArray":
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("ChunkedArray supports at most 2D indexing")
            row_key, col_key = key
        else:
            row_key, col_key = key, slice(None)

        rows = self._rows
        cols = self._cols
        operations = self._ops
        out_cols = self._out_cols
        n_rows = self._n_rows

        col_positions = self._local_positions(col_key, out_cols)
        if col_positions is not None:
            operations = [
                operation.subset_cols(col_positions) for operation in operations
            ]
            base_cols = cols if cols is not None else np.arange(self._backing.shape[1])
            cols = base_cols[col_positions]
            out_cols = int(col_positions.size)

        row_positions = self._local_positions(row_key, n_rows)
        if row_positions is not None:
            operations = [
                operation.subset_rows(row_positions) for operation in operations
            ]
            base_rows = rows if rows is not None else np.arange(self._backing.shape[0])
            rows = base_rows[row_positions]
            n_rows = int(row_positions.size)

        block_size = n_rows if self._is_numpy and n_rows > 0 else self._block_size
        return ChunkedArray(
            self._backing,
            rows=rows,
            cols=cols,
            ops=operations,
            out_cols=out_cols,
            block_size=block_size,
            nthreads=self._nthreads,
            resources=self._resources,
            is_numpy=self._is_numpy,
        )

    def sum(self, axis: int | None = None) -> _Reduction:
        return _Reduction(self, "sum", axis)

    def mean(self, axis: int | None = None) -> _Reduction:
        return _Reduction(self, "mean", axis)

    def var(self, axis: int | None = None) -> _Reduction:
        return _Reduction(self, "var", axis)

    def std(self, axis: int | None = None) -> _Reduction:
        return _Reduction(self, "std", axis)

    def mean_and_std(
        self,
        axis: int = 0,
        nthreads: int | None = None,
        msg: str | None = None,
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        """Compute column mean and standard deviation in one pass."""
        if axis != 0:
            raise NotImplementedError("mean_and_std only supports axis=0")

        def summarize(_: int, start: int, end: int) -> NDArray[Any]:
            array = self._materialize_range(start, end).astype(
                np.float64,
                copy=False,
            )
            return np.array(
                [
                    array.sum(axis=0),
                    np.square(array).sum(axis=0),
                ]
            )

        parts = self._map_blocks(summarize, nthreads, msg)
        stacked = np.sum(parts, axis=0)
        total, squared_total = stacked[0], stacked[1]
        mean = total / self._n_rows
        variance = squared_total / self._n_rows - np.square(mean)
        return np.asarray(mean), np.asarray(np.sqrt(np.clip(variance, 0, None)))

    def count_nonzero(self, axis: int | None = None) -> _Reduction:
        return _Reduction(self, "count_nonzero", axis)

    def argmax(self, axis: int | None = None) -> _Reduction:
        return _Reduction(self, "argmax", axis)

    def _reduce(
        self,
        op: ReductionOp,
        axis: int | None,
        nthreads: int | None,
        msg: str | None,
    ) -> np.ndarray:
        if axis == 1 or axis is None:
            return self._reduce_axis1(op, axis, nthreads, msg)
        return self._reduce_axis0(op, nthreads, msg)

    def _reduce_axis1(
        self,
        op: ReductionOp,
        axis: int | None,
        nthreads: int | None,
        msg: str | None,
    ) -> np.ndarray:
        def reduce_block(_: int, start: int, end: int) -> NDArray[Any]:
            array = self._materialize_range(start, end)
            if op == "sum":
                return np.asarray(array.sum(axis=axis))
            if op == "mean":
                return np.asarray(
                    array.sum() if axis is None else array.mean(axis=axis)
                )
            if op == "var":
                if axis is None:
                    values = array.astype(np.float64, copy=False)
                    return np.asarray([values.sum(), np.square(values).sum()])
                return np.asarray(array.var(axis=axis))
            if op == "std":
                if axis is None:
                    values = array.astype(np.float64, copy=False)
                    return np.asarray([values.sum(), np.square(values).sum()])
                return np.asarray(array.std(axis=axis))
            if op == "count_nonzero":
                return np.asarray(np.count_nonzero(array, axis=axis))
            if op == "argmax":
                return np.asarray(array.argmax(axis=axis))
            raise ValueError(f"Unknown reduction {op}")

        parts = self._map_blocks(reduce_block, nthreads, msg)
        if axis is None:
            array = np.asarray(parts)
            if op == "sum":
                return np.asarray(array.sum())
            if op == "mean":
                return np.asarray(array.sum() / (self._n_rows * self._out_cols))
            if op in ("var", "std"):
                total, squared_total = np.sum(parts, axis=0)
                count = self._n_rows * self._out_cols
                mean = total / count
                variance = squared_total / count - np.square(mean)
                if op == "std":
                    return np.asarray(np.sqrt(max(float(variance), 0.0)))
                return np.asarray(variance)
            if op == "count_nonzero":
                return np.asarray(array.sum())
            raise ValueError(f"Reduction {op} with axis=None is not supported")
        return np.concatenate(parts)

    def _reduce_axis0(
        self,
        op: ReductionOp,
        nthreads: int | None,
        msg: str | None,
    ) -> np.ndarray:
        if op in ("sum", "mean"):

            def sum_block(_: int, start: int, end: int) -> NDArray[Any]:
                array = self._materialize_range(start, end)
                return np.asarray(array.sum(axis=0))

            parts = self._map_blocks(sum_block, nthreads, msg)
            total = np.sum(parts, axis=0)
            if op == "mean":
                return np.asarray(total / self._n_rows)
            return np.asarray(total)
        if op == "count_nonzero":

            def count_block(_: int, start: int, end: int) -> NDArray[Any]:
                array = self._materialize_range(start, end)
                return np.asarray(np.count_nonzero(array, axis=0))

            parts = self._map_blocks(count_block, nthreads, msg)
            return np.asarray(np.sum(parts, axis=0))
        if op in ("var", "std"):

            def variance_block(_: int, start: int, end: int) -> NDArray[Any]:
                array = self._materialize_range(start, end).astype(
                    np.float64,
                    copy=False,
                )
                return np.array(
                    [
                        array.sum(axis=0),
                        np.square(array).sum(axis=0),
                    ]
                )

            parts = self._map_blocks(variance_block, nthreads, msg)
            stacked = np.sum(parts, axis=0)
            total, squared_total = stacked[0], stacked[1]
            mean = total / self._n_rows
            variance = squared_total / self._n_rows - np.square(mean)
            if op == "std":
                return np.asarray(np.sqrt(np.clip(variance, 0, None)))
            return np.asarray(variance)
        if op == "argmax":
            raise NotImplementedError("argmax(axis=0) is not supported")
        raise ValueError(f"Unknown reduction {op}")

    def __repr__(self) -> str:
        return (
            f"ChunkedArray(shape={self.shape}, dtype={self.dtype}, "
            f"chunksize={self.chunksize}, numblocks={self.numblocks[0]})"
        )
