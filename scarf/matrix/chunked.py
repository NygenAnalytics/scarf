"""Lazy blockwise matrix operations over NumPy and Zarr arrays."""

import operator
import warnings
from collections.abc import Callable, Iterator
from functools import partial
from typing import Any, Literal, cast

import numpy as np
import zarr
from numpy.typing import DTypeLike, NDArray

from ..utils.moments import ColumnMoments, column_moments
from ..storage.budget import (
    DEFAULT_READ_AHEAD_BLOCKS,
    ResourceBudget,
)
from ..storage.io_policy import StorageIoPolicy
from ..storage.geometry import ArrayGeometry, array_geometry
from ..storage.partition import is_contiguous
from ._indexing import local_positions
from ._operations import (
    _Op,
    _binary_op,
    _classify_operand,
    _unary_op,
)
from ._reductions import ReductionOp, _Reduction

__all__ = ["ChunkedArray"]

type Backing = np.ndarray | zarr.Array
type BlockFn[T] = Callable[[int, int, int], T]
type Axis = Literal[0, 1] | None

# Ufunc keywords a lazy operation can honour; others would be silently dropped.
_UFUNC_KEYWORDS = frozenset({"dtype", "out"})


def _mean_accumulator(dtype: np.dtype[Any]) -> np.dtype[Any]:
    """Return the dtype that means and variances accumulate and return in.

    Float64 for bool, integer, and floating-point values up to float64, so
    integer sums cannot overflow and float32 sums keep float64 precision.
    Promotion with float64 keeps a wider floating-point dtype and widens
    complex values to at least complex128.
    """
    return np.promote_types(dtype, np.float64)


def _reduction_axis(axis: int | None) -> Axis:
    """Normalize a reduction axis of a two-dimensional matrix."""
    if axis is None:
        return None
    if isinstance(axis, bool) or not isinstance(axis, int | np.integer):
        raise TypeError("axis must be an integer or None")
    if axis in (0, -2):
        return 0
    if axis in (1, -1):
        return 1
    raise np.exceptions.AxisError(int(axis), 2)


class ChunkedArray:
    """A lazy, row-chunked 2D matrix backed by Zarr or NumPy."""

    __array_priority__ = 1000.0

    def __init__(
        self,
        backing: Backing,
        rows: np.ndarray | None = None,
        cols: np.ndarray | None = None,
        ops: list[_Op] | None = None,
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
        self._io: StorageIoPolicy | None = None
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
        self._n_cols = backing_cols if self._cols is None else int(self._cols.size)
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
        return (self._n_rows, self._n_cols)

    @property
    def dtype(self) -> np.dtype[Any]:
        dtypes = self._op_dtypes()
        return dtypes[-1] if dtypes else self._backing.dtype

    def _op_dtypes(self) -> list[np.dtype[Any]]:
        """Return the dtype of each operation's output, in order."""
        sample = np.empty((0, self._n_cols), dtype=self._backing.dtype)
        dtypes: list[np.dtype[Any]] = []
        for operation in self._ops:
            sample = operation.apply(sample, 0, 0)
            dtypes.append(sample.dtype)
        return dtypes

    @property
    def chunksize(self) -> tuple[int, int]:
        return (
            min(self._block_size, self._n_rows) if self._n_rows else self._block_size,
            self._n_cols,
        )

    @property
    def numblocks(self) -> tuple[int, int]:
        return (self._n_block_count(), 1)

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

    def _read_bytes(self, input_bytes: int) -> int:
        """Bytes the Zarr read of a row block with ``input_bytes`` holds.

        Row-block streams run Zarr with one decode at a time, and a block's
        read touches, in each shard, the inner chunks of every row chunk and
        of the selected columns.
        """
        geometry = self._geometry()
        if geometry is None:
            return input_bytes
        row_chunks = -(-geometry.axisShard(0) // geometry.axisChunk(0))
        column_chunks = -(-geometry.axisShard(1) // geometry.axisChunk(1))
        if self._cols is not None:
            column_chunks = min(
                column_chunks, int(np.unique(geometry.binOf(1, self._cols)).size)
            )
        return geometry.readBytes(input_bytes, row_chunks * column_chunks, decodes=1)

    def _max_decode_bytes(self) -> int:
        """Bytes a row block's read holds for its inner chunks, whatever its rows."""
        return self._read_bytes(0)

    def _block_owned_bytes(self) -> int:
        """Bytes of one row block that grow with its rows.

        Its input and what Zarr's read holds in proportion to it, beside the
        output of its first operation. Each later operation holds the previous
        output beside its own, and the largest of these pairs is charged.
        """
        rows = min(self._block_size, max(1, self._n_rows))
        elements = rows * max(1, self._n_cols)
        input_bytes = elements * max(1, int(self._backing.dtype.itemsize))
        held = self._read_bytes(input_bytes) - self._max_decode_bytes()
        peak = held
        for dtype in self._op_dtypes():
            output_bytes = elements * max(1, int(dtype.itemsize))
            peak = max(peak, held + output_bytes)
            held = output_bytes
        return peak

    def _block_task_bytes(self) -> int:
        """Bytes one row block holds while it is read and transformed."""
        return self._block_owned_bytes() + self._max_decode_bytes()

    def _block_results[T](
        self,
        fn: BlockFn[T],
        nthreads: int | None,
        msg: str | None,
        result_bytes: int,
    ) -> Iterator[T]:
        """Yield ``fn(index, start, end)`` for every row block in row order.

        ``result_bytes`` bounds what the caller retains from the results.
        Callers reduce a matrix without rows themselves, so there is at least
        one block.
        """
        from ..storage.execution import (
            ExecutionReport,
            WorkShape,
            plan_operation,
            record_execution_report,
        )
        from ..storage.parallel import in_shard_context, stream_shards

        ranges = self._ranges()
        workers = self._nthreads if nthreads is None else max(1, int(nthreads))
        within = 1
        io_concurrency: int | None = None
        planned = None
        if self._resources is not None:
            planned = plan_operation(
                ResourceBudget(
                    self._resources.memoryBytes, min(workers, self._resources.workers)
                ),
                WorkShape(
                    nUnits=len(ranges),
                    unitBytes=self._block_task_bytes(),
                    residentBytes=2 * result_bytes + self._resident_bytes(),
                    ordered=False,
                ),
                policy=self._io,
            )
            workers = planned.computeWorkers
            within = planned.threadsPerComputeWorker
            io_concurrency = planned.ioConcurrency
        if in_shard_context():
            workers = 1
        workers = min(workers, len(ranges))

        def produce(item: tuple[int, tuple[int, int]]) -> T:
            index, (start, end) = item
            return fn(index, start, end)

        completed = 0
        for result in stream_shards(
            list(enumerate(ranges)),
            produce,
            workers=workers,
            within_block_threads=within,
            io_concurrency=workers if io_concurrency is None else io_concurrency,
            msg=msg,
            total=len(ranges),
        ):
            completed += 1
            yield result
        if planned is not None:
            record_execution_report(
                ExecutionReport(
                    plan=planned,
                    unitKind="countsRowBlock",
                    actualReadWorkers=workers,
                    actualComputeWorkers=workers,
                    actualWriteWorkers=1,
                    unitsCompleted=completed,
                )
            )

    def _accumulate[T](
        self,
        fn: BlockFn[T],
        nthreads: int | None,
        msg: str | None,
        result_bytes: int,
        merge: Callable[[T, T], T] = operator.add,
    ) -> T:
        """Merge block results in row order, holding one running result.

        Each block's result is merged into the running result as the next
        rows, ``merge(running, block)``; the default adds them.
        """
        total: T | None = None
        for part in self._block_results(fn, nthreads, msg, result_bytes):
            total = part if total is None else merge(total, part)
        assert total is not None
        return total

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
        from ..storage.execution import WorkShape, plan_operation
        from ..storage.parallel import stream_shards, in_shard_context

        threads = self._nthreads if nthreads is None else max(1, int(nthreads))
        ranges = self._ranges()
        # Callers pass a boolean vector over the rows as ``row_mask``.
        mask = None if row_mask is None else np.asarray(row_mask)
        if mask is not None:
            ranges = [
                (start, end) for start, end in ranges if bool(mask[start:end].any())
            ]

        io_concurrency: int | None = None
        planned = None
        if self._resources is not None:
            planned = plan_operation(
                ResourceBudget(
                    self._resources.memoryBytes, min(threads, self._resources.workers)
                ),
                WorkShape(
                    nUnits=max(1, len(ranges)),
                    unitBytes=self._block_owned_bytes(),
                    decodeBytes=self._max_decode_bytes(),
                    residentBytes=max(0, int(resident_bytes)) + self._resident_bytes(),
                    ordered=False,
                ),
                policy=self._io,
            )
            depth = min(planned.readWorkers, planned.computeWorkers)
            if prefetch is not None:
                depth = min(depth, max(1, int(prefetch)))
            within = planned.threadsPerComputeWorker
            io_concurrency = planned.ioConcurrency
        else:
            requested = (
                max(1, int(prefetch))
                if prefetch is not None
                else DEFAULT_READ_AHEAD_BLOCKS
            )
            depth = min(threads, requested, max(1, len(ranges)))
            within = 1

        def materialize(interval: tuple[int, int]) -> list[np.ndarray]:
            start, end = interval
            values = self._materialize_range(start, end)
            # A holder this stream empties, so that neither the reader that
            # produced a block nor the read-ahead keeps it once it is yielded.
            return [values if mask is None else values[mask[start:end]]]

        if in_shard_context():
            depth = 1
        completed = 0
        for held in stream_shards(
            ranges,
            materialize,
            workers=depth,
            within_block_threads=within,
            io_concurrency=io_concurrency,
            msg=msg,
            total=len(ranges),
        ):
            completed += 1
            block = held.pop()
            yield block
            del block
        if planned is not None:
            from ..storage.execution import ExecutionReport, record_execution_report

            record_execution_report(
                ExecutionReport(
                    plan=planned,
                    unitKind="countsRowBlock",
                    actualReadWorkers=depth,
                    actualComputeWorkers=min(depth, planned.computeWorkers),
                    actualWriteWorkers=1,
                    unitsCompleted=completed,
                )
            )

    def compute(
        self,
        nthreads: int | None = None,
        msg: str | None = None,
    ) -> np.ndarray:
        if self._n_rows == 0:
            return np.empty((0, self._n_cols), dtype=self.dtype)

        blocks = self._stream_blocks(
            nthreads=nthreads,
            msg=msg,
            prefetch=None,
            row_mask=None,
            resident_bytes=self._n_rows * self._n_cols * self.dtype.itemsize,
        )
        try:
            first = next(blocks)
            result = np.empty(self.shape, dtype=self.dtype)
            offset = len(first)
            result[:offset] = first
            del first
            for block in blocks:
                result[offset : offset + len(block)] = block
                offset += len(block)
                # The plan holds no block for the caller while the next is read.
                del block
            return result
        finally:
            from ..storage.parallel import _close_iterator

            _close_iterator(blocks)

    def _resident_bytes(self) -> int:
        arrays = [
            self._backing,
            self._rows,
            self._cols,
            *(op.operand for op in self._ops),
        ]
        retained: dict[int, int] = {}
        for array in arrays:
            if not isinstance(array, np.ndarray):
                continue
            while isinstance(array.base, np.ndarray):
                array = array.base
            retained[id(array)] = array.nbytes
        return sum(retained.values())

    def __array__(
        self,
        dtype: np.dtype[Any] | None = None,
        copy: bool | None = None,
    ) -> np.ndarray:
        if copy is False:
            raise ValueError(
                "A ChunkedArray is computed from its blocks, so converting it "
                "always creates a new array"
            )
        return np.asarray(self.compute(), dtype=dtype)

    def _with_io(self, array: "ChunkedArray") -> "ChunkedArray":
        array._io = self._io
        return array

    def _with_op(self, operation: _Op) -> "ChunkedArray":
        return self._with_io(
            ChunkedArray(
                self._backing,
                rows=self._rows,
                cols=self._cols,
                ops=self._ops + [operation],
                block_size=self._block_size,
                nthreads=self._nthreads,
                resources=self._resources,
                is_numpy=self._is_numpy,
            )
        )

    def _with_block_size(self, block_size: int) -> "ChunkedArray":
        return self._with_io(
            ChunkedArray(
                self._backing,
                rows=self._rows,
                cols=self._cols,
                ops=self._ops,
                block_size=block_size,
                nthreads=self._nthreads,
                resources=self._resources,
                is_numpy=self._is_numpy,
            )
        )

    def _unary(self, func: Callable[..., NDArray[Any]]) -> "ChunkedArray":
        return self._with_op(_unary_op(func))

    def _binary(
        self,
        func: Callable[..., NDArray[Any]],
        other: object,
        side: str,
    ) -> "ChunkedArray":
        if isinstance(other, ChunkedArray):
            raise TypeError(
                "Operations between two ChunkedArrays are not lazy; combine "
                "their row blocks explicitly"
            )
        if isinstance(other, _Reduction):
            other = other._arr
        kind, operand = _classify_operand(
            other,
            self._n_rows,
            self._n_cols,
        )
        return self._with_op(_binary_op(func, operand, side, kind))

    def __array_ufunc__(
        self,
        ufunc: Any,
        method: str,
        *inputs: Any,
        **kwargs: Any,
    ) -> Any:
        if (
            method != "__call__"
            or kwargs.get("out") is not None
            or not _UFUNC_KEYWORDS.issuperset(kwargs)
        ):
            return NotImplemented
        # A requested dtype sets the ufunc's computation dtype for each block,
        # so integer blocks can be scaled or logged without a cast copy.
        dtype = kwargs.get("dtype")
        func = ufunc if dtype is None else partial(ufunc, dtype=np.dtype(dtype))
        if len(inputs) == 1:
            return self._unary(func)
        if len(inputs) == 2:
            left, right = inputs
            if left is self:
                return self._binary(func, right, "left")
            return self._binary(func, left, "right")
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
        n_rows = self._n_rows

        col_positions = local_positions(col_key, self._n_cols)
        if col_positions is not None:
            operations = [
                operation.subset_cols(col_positions) for operation in operations
            ]
            base_cols = cols if cols is not None else np.arange(self._backing.shape[1])
            cols = base_cols[col_positions]

        row_positions = local_positions(row_key, n_rows)
        if row_positions is not None:
            operations = [
                operation.subset_rows(row_positions) for operation in operations
            ]
            base_rows = rows if rows is not None else np.arange(self._backing.shape[0])
            rows = base_rows[row_positions]
            n_rows = int(row_positions.size)

        block_size = n_rows if self._is_numpy and n_rows > 0 else self._block_size
        return self._with_io(
            ChunkedArray(
                self._backing,
                rows=rows,
                cols=cols,
                ops=operations,
                block_size=block_size,
                nthreads=self._nthreads,
                resources=self._resources,
                is_numpy=self._is_numpy,
            )
        )

    def sum(
        self,
        axis: int | None = None,
        dtype: DTypeLike | None = None,
    ) -> _Reduction:
        """Return the deferred sum; ``dtype`` sets the accumulator as in NumPy."""
        return _Reduction(
            self,
            "sum",
            _reduction_axis(axis),
            None if dtype is None else np.dtype(dtype),
        )

    def mean(self, axis: int | None = None) -> _Reduction:
        """Return the deferred mean, accumulated and returned in float64."""
        return _Reduction(self, "mean", _reduction_axis(axis))

    def var(self, axis: int | None = None) -> _Reduction:
        """Return the deferred population variance in float64."""
        return _Reduction(self, "var", _reduction_axis(axis))

    def mean_and_std(
        self,
        axis: int = 0,
        nthreads: int | None = None,
        msg: str | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute column mean and standard deviation in one pass."""
        if _reduction_axis(axis) != 0:
            raise NotImplementedError("mean_and_std only supports axis=0")
        if self._n_rows == 0:
            missing = np.full(self._n_cols, np.nan)
            return missing, missing.copy()
        moments = self._column_moments(nthreads, msg, collapse=False)
        return moments.mean, np.sqrt(moments.variance())

    def _column_moments(
        self,
        nthreads: int | None,
        msg: str | None,
        *,
        collapse: bool,
    ) -> ColumnMoments:
        """Merge the column moments of the row blocks in row order.

        With ``collapse``, each block's values pool into one column, giving the
        moments of every value of the matrix without per-column arrays.
        """

        def summarize(_: int, start: int, end: int) -> ColumnMoments:
            block = self._materialize_range(start, end)
            return column_moments(np.reshape(block, (-1, 1)) if collapse else block)

        width = 1 if collapse else self._n_cols
        return self._accumulate(
            summarize,
            nthreads,
            msg,
            result_bytes=2 * 2 * width * 8,
            merge=ColumnMoments.merge,
        )

    def count_nonzero(self, axis: int | None = None) -> _Reduction:
        return _Reduction(self, "count_nonzero", _reduction_axis(axis))

    def argmax(self, axis: int | None = None) -> _Reduction:
        return _Reduction(self, "argmax", _reduction_axis(axis))

    def _reduce(
        self,
        op: ReductionOp,
        axis: int | None,
        nthreads: int | None,
        msg: str | None,
        *,
        dtype: np.dtype[Any] | None = None,
    ) -> np.ndarray:
        if op == "argmax" and axis == 0:
            raise NotImplementedError("argmax(axis=0) is not supported")
        if op == "argmax" and axis is None:
            raise ValueError("Reduction argmax with axis=None is not supported")
        if self._n_rows == 0:
            return self._reduce_empty(op, axis, dtype)
        if axis == 1:
            return self._reduce_rows(op, nthreads, msg, dtype)
        if axis is None:
            return self._reduce_all(op, nthreads, msg, dtype)
        return self._reduce_columns(op, nthreads, msg, dtype)

    def _reduce_empty(
        self,
        op: ReductionOp,
        axis: int | None,
        dtype: np.dtype[Any] | None,
    ) -> np.ndarray:
        """Reduce a matrix without rows exactly as NumPy reduces one.

        Means and variances take the dtype they accumulate in on rows.
        """
        empty = np.empty((0, self._n_cols), dtype=self.dtype)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if op == "count_nonzero":
                return np.asarray(np.count_nonzero(empty, axis=axis))
            if op == "sum":
                return np.asarray(empty.sum(axis=axis, dtype=dtype))
            if op in ("mean", "var"):
                accumulator = _mean_accumulator(empty.dtype)
                return np.asarray(getattr(empty, op)(axis=axis, dtype=accumulator))
            return np.asarray(getattr(empty, op)(axis=axis))

    def _reduce_rows(
        self,
        op: ReductionOp,
        nthreads: int | None,
        msg: str | None,
        dtype: np.dtype[Any] | None,
    ) -> np.ndarray:
        def reduce_block(_: int, start: int, end: int) -> NDArray[Any]:
            array = self._materialize_range(start, end)
            if op == "count_nonzero":
                return np.asarray(np.count_nonzero(array, axis=1))
            if op == "sum":
                return np.asarray(array.sum(axis=1, dtype=dtype))
            if op in ("mean", "var"):
                accumulator = _mean_accumulator(array.dtype)
                return np.asarray(getattr(array, op)(axis=1, dtype=accumulator))
            return np.asarray(getattr(array, op)(axis=1))

        result: np.ndarray | None = None
        offset = 0
        for part in self._block_results(
            reduce_block,
            nthreads,
            msg,
            result_bytes=self._n_rows * 8,
        ):
            if result is None:
                result = np.empty(self._n_rows, dtype=part.dtype)
            result[offset : offset + len(part)] = part
            offset += len(part)
        assert result is not None
        return result

    def _reduce_all(
        self,
        op: ReductionOp,
        nthreads: int | None,
        msg: str | None,
        dtype: np.dtype[Any] | None,
    ) -> np.ndarray:
        if op == "var":
            moments = self._column_moments(nthreads, msg, collapse=True)
            return np.asarray(moments.variance()[0])
        if op == "mean":
            dtype = _mean_accumulator(self.dtype)

        def reduce_block(_: int, start: int, end: int) -> NDArray[Any]:
            array = self._materialize_range(start, end)
            if op == "count_nonzero":
                return np.asarray(np.count_nonzero(array))
            return np.asarray(array.sum(dtype=dtype))

        total = self._accumulate(reduce_block, nthreads, msg, result_bytes=2 * 16)
        if op == "mean":
            return np.asarray(total / (self._n_rows * self._n_cols))
        return np.asarray(total)

    def _reduce_columns(
        self,
        op: ReductionOp,
        nthreads: int | None,
        msg: str | None,
        dtype: np.dtype[Any] | None,
    ) -> np.ndarray:
        if op == "var":
            return self._column_moments(nthreads, msg, collapse=False).variance()
        if op == "mean":
            dtype = _mean_accumulator(self.dtype)

        def reduce_block(_: int, start: int, end: int) -> NDArray[Any]:
            array = self._materialize_range(start, end)
            if op == "count_nonzero":
                return np.asarray(np.count_nonzero(array, axis=0))
            return np.asarray(array.sum(axis=0, dtype=dtype))

        total = self._accumulate(
            reduce_block,
            nthreads,
            msg,
            result_bytes=2 * self._n_cols * 8,
        )
        if op == "mean":
            return np.asarray(total / self._n_rows)
        return np.asarray(total)

    def __repr__(self) -> str:
        return (
            f"ChunkedArray(shape={self.shape}, dtype={self.dtype}, "
            f"chunksize={self.chunksize}, numblocks={self.numblocks[0]})"
        )
