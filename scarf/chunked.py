"""A minimal, Zarr-backed chunked array used in place of Dask.

This module provides ``ChunkedArray``, a lazy, row-chunked array over a Zarr
(or in-memory NumPy) 2D matrix. It mirrors the small slice of the Dask Array
API that Scarf relies on:

- block iteration via ``.blocks`` (each block is one row-chunk)
- lazy element-wise ufuncs and broadcast arithmetic
- ``.dot`` against a small loadings matrix
- streaming, thread-parallel reductions (``sum``/``mean``/``var``/
  ``count_nonzero``/``argmax``) that run eagerly but lazily, returning a
  deferred result so a progress bar with a message can still be attached.

Element-wise operations stay lazy and are applied per row-block at compute
time. Reductions are evaluated by streaming over row-blocks and combining the
partial results, which keeps peak memory bounded and independent of the task
graph topology.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

__all__ = ["ChunkedArray", "Block"]


def _is_contiguous(idx: np.ndarray) -> bool:
    """True if idx is a strictly increasing run of consecutive integers."""
    if idx.size == 0:
        return True
    return bool(idx[0] >= 0 and np.array_equal(idx, np.arange(idx[0], idx[0] + idx.size)))


class _Op:
    """A structured element-wise (or matmul) operation on a ChunkedArray.

    Operations are recorded lazily and replayed per row-block at compute time.
    Storing them structurally (rather than as opaque closures) lets indexing
    re-subset per-feature and per-row operands so column/row selection can be
    pushed through already-applied normalization.
    """

    __slots__ = ("kind", "func", "operand", "side", "btype")

    def __init__(self, kind, func=None, operand=None, side="left", btype=None):
        self.kind = kind  # 'unary' | 'binary' | 'matmul'
        self.func = func
        self.operand = operand
        self.side = side
        self.btype = btype  # for 'binary': 'scalar' | 'col' | 'row' | 'full'

    def apply(self, a, start, end):
        if self.kind == "unary":
            return self.func(a)
        if self.kind == "matmul":
            return a @ self.operand
        o = self.operand
        if self.btype == "col":
            o = np.asarray(o)[start:end]
            if o.ndim == 1:
                o = o.reshape(-1, 1)
        elif self.btype == "full":
            o = np.asarray(o)[start:end]
        return self.func(a, o) if self.side == "left" else self.func(o, a)

    def subset_cols(self, col_idx: np.ndarray) -> "_Op":
        if self.kind == "matmul":
            raise NotImplementedError("Column-indexing after .dot is not supported")
        if self.kind == "binary" and self.btype == "row":
            return _Op("binary", self.func, np.asarray(self.operand)[col_idx], self.side, "row")
        if self.kind == "binary" and self.btype == "full":
            return _Op("binary", self.func, np.asarray(self.operand)[:, col_idx], self.side, "full")
        return self

    def subset_rows(self, row_idx: np.ndarray) -> "_Op":
        if self.kind == "binary" and self.btype == "col":
            return _Op("binary", self.func, np.asarray(self.operand)[row_idx], self.side, "col")
        if self.kind == "binary" and self.btype == "full":
            return _Op("binary", self.func, np.asarray(self.operand)[row_idx], self.side, "full")
        return self


def _unary_op(func) -> _Op:
    return _Op("unary", func=func)


def _matmul_op(b: np.ndarray) -> _Op:
    return _Op("matmul", operand=b)


def _classify_operand(other, n_rows: int, n_cols: int) -> tuple[str, object]:
    """Classify a binary operand for per-block broadcasting.

    Returns one of 'scalar', 'col' (per-row vector), 'row' (per-feature
    vector) or 'full' (per-element matrix), along with the coerced operand.
    """
    if np.isscalar(other):
        return "scalar", other
    arr = np.asarray(other)
    if arr.ndim == 0:
        return "scalar", arr
    shape = arr.shape
    if (arr.ndim == 1 and shape[0] == n_rows) or (arr.ndim == 2 and shape == (n_rows, 1)):
        return "col", arr
    if (arr.ndim == 1 and shape[0] == n_cols) or (arr.ndim == 2 and shape == (1, n_cols)):
        return "row", arr
    if arr.ndim == 2 and shape[0] == n_rows:
        return "full", arr
    if arr.size == 1:
        return "scalar", arr
    return "row", arr


def _binary_op(func, other, side: str, kind: str) -> _Op:
    return _Op("binary", func=func, operand=other, side=side, btype=kind)


class Block:
    """A single row-chunk view of a ChunkedArray."""

    __slots__ = ("_parent", "_start", "_end", "_row_perm")

    def __init__(self, parent: "ChunkedArray", start: int, end: int, row_perm=None):
        self._parent = parent
        self._start = start
        self._end = end
        self._row_perm = row_perm

    @property
    def shape(self) -> tuple[int, int]:
        n = (self._end - self._start) if self._row_perm is None else len(self._row_perm)
        return (n, self._parent.out_cols)

    @property
    def dtype(self):
        return self._parent.dtype

    def compute(self, nthreads: int | None = None, msg: str | None = None) -> np.ndarray:
        a = self._parent._materialize_range(self._start, self._end)
        if self._row_perm is not None:
            a = a[self._row_perm]
        return a

    def __array__(self, dtype=None):
        a = self.compute()
        return a.astype(dtype) if dtype is not None else a

    def __getitem__(self, key):
        # Supports block[row_index, :] used during merge row permutation.
        if isinstance(key, tuple):
            row_key = key[0]
        else:
            row_key = key
        return Block(self._parent, self._start, self._end, row_perm=np.asarray(row_key))


class _Reduction:
    """A deferred reduction result that behaves like a NumPy array.

    It computes lazily and caches the result. ``compute`` accepts an optional
    progress message; any other array-like use (arithmetic, ufuncs, attribute
    access) forces evaluation transparently without a progress bar.
    """

    __slots__ = ("_parent", "_op", "_axis", "_cached")

    def __init__(self, parent: "ChunkedArray", op: str, axis: int | None):
        self._parent = parent
        self._op = op
        self._axis = axis
        self._cached: np.ndarray | None = None

    def compute(self, nthreads: int | None = None, msg: str | None = None) -> np.ndarray:
        if self._cached is None:
            self._cached = self._parent._reduce(self._op, self._axis, nthreads, msg)
        return self._cached

    @property
    def _arr(self) -> np.ndarray:
        return self.compute()

    def __array__(self, dtype=None):
        a = self._arr
        return a.astype(dtype) if dtype is not None else a

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            return NotImplemented
        inputs = [i._arr if isinstance(i, _Reduction) else i for i in inputs]
        return ufunc(*inputs, **kwargs)

    def __getattr__(self, item):
        return getattr(self._arr, item)

    def __len__(self):
        return len(self._arr)

    def __iter__(self):
        return iter(self._arr)

    def __getitem__(self, key):
        return self._arr[key]

    def _bin(self, other, func, side):
        o = other._arr if isinstance(other, _Reduction) else other
        return func(self._arr, o) if side == "left" else func(o, self._arr)

    def __mul__(self, o):
        return self._bin(o, np.multiply, "left")

    def __rmul__(self, o):
        return self._bin(o, np.multiply, "right")

    def __truediv__(self, o):
        return self._bin(o, np.true_divide, "left")

    def __rtruediv__(self, o):
        return self._bin(o, np.true_divide, "right")

    def __add__(self, o):
        return self._bin(o, np.add, "left")

    def __radd__(self, o):
        return self._bin(o, np.add, "right")

    def __sub__(self, o):
        return self._bin(o, np.subtract, "left")

    def __rsub__(self, o):
        return self._bin(o, np.subtract, "right")

    def __gt__(self, o):
        return self._bin(o, np.greater, "left")

    def __lt__(self, o):
        return self._bin(o, np.less, "left")

    def __ge__(self, o):
        return self._bin(o, np.greater_equal, "left")

    def __le__(self, o):
        return self._bin(o, np.less_equal, "left")

    def __repr__(self):
        return f"<deferred {self._op}(axis={self._axis})>"


class ChunkedArray:
    """A lazy, row-chunked 2D array backed by Zarr or NumPy."""

    __array_priority__ = 1000.0

    def __init__(
        self,
        backing,
        rows: np.ndarray | None = None,
        cols: np.ndarray | None = None,
        ops: list[_Op] | None = None,
        out_cols: int | None = None,
        block_size: int | None = None,
        nthreads: int = 1,
        is_numpy: bool | None = None,
    ):
        self._backing = backing
        self._rows = None if rows is None else np.asarray(rows)
        self._cols = None if cols is None else np.asarray(cols)
        self._ops: list[_Op] = list(ops) if ops else []
        self._nthreads = nthreads
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
                from .storage.zarr_store import streaming_block_size

                block_size = streaming_block_size(self._backing)
        self._block_size = max(int(block_size), 1)

    @classmethod
    def from_numpy(cls, arr: np.ndarray, block_size: int | None = None, nthreads: int = 1) -> "ChunkedArray":
        arr = np.asarray(arr)
        return cls(arr, block_size=block_size, nthreads=nthreads, is_numpy=True)

    @property
    def shape(self) -> tuple[int, int]:
        return (self._n_rows, self._out_cols)

    @property
    def out_cols(self) -> int:
        return self._out_cols

    @property
    def dtype(self):
        if not self._ops:
            return self._backing.dtype
        return np.dtype(np.float64)

    @property
    def chunksize(self) -> tuple[int, int]:
        return (min(self._block_size, self._n_rows) if self._n_rows else self._block_size, self._out_cols)

    @property
    def numblocks(self) -> tuple[int, int]:
        return (self._n_block_count(), 1)

    @property
    def chunks(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        sizes = tuple(e - s for s, e in self._ranges())
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
        return [
            (i, min(i + self._block_size, self._n_rows))
            for i in range(0, self._n_rows, self._block_size)
        ]

    # -- materialization -------------------------------------------------
    def _read(self, start: int, end: int) -> np.ndarray:
        if self._rows is None:
            row_sel: slice | np.ndarray = slice(start, end)
            rows_contiguous = True
        else:
            row_sel = self._rows[start:end]
            rows_contiguous = _is_contiguous(row_sel)
        if self._is_numpy:
            if self._cols is None:
                return np.asarray(self._backing[row_sel])
            return np.asarray(self._backing[row_sel][:, self._cols])
        # Zarr backing
        if self._cols is None:
            if isinstance(row_sel, slice):
                return self._backing[row_sel, :]
            if rows_contiguous:
                return self._backing[int(row_sel[0]):int(row_sel[-1]) + 1, :]
            return self._backing.get_orthogonal_selection((row_sel, slice(None)))
        if isinstance(row_sel, slice):
            return self._backing.get_orthogonal_selection((row_sel, self._cols))
        if rows_contiguous:
            return self._backing.get_orthogonal_selection(
                (slice(int(row_sel[0]), int(row_sel[-1]) + 1), self._cols)
            )
        return self._backing.get_orthogonal_selection((row_sel, self._cols))

    def _materialize_range(self, start: int, end: int) -> np.ndarray:
        a = self._read(start, end)
        for op in self._ops:
            a = op.apply(a, start, end)
        return a

    # -- block parallelism ----------------------------------------------
    def _map_blocks(self, fn, nthreads: int | None, msg: str | None) -> list:
        from .utils import tqdmbar

        ranges = self._ranges()
        nthreads = self._nthreads if nthreads is None else nthreads
        bar = tqdmbar(total=len(ranges), desc=msg) if msg is not None else None
        results: list = [None] * len(ranges)
        if nthreads is not None and nthreads > 1 and len(ranges) > 1:
            with ThreadPoolExecutor(max_workers=nthreads) as ex:
                futures = {ex.submit(fn, i, r[0], r[1]): i for i, r in enumerate(ranges)}
                for fut, i in futures.items():
                    results[i] = fut.result()
                    if bar is not None:
                        bar.update(1)
        else:
            for i, r in enumerate(ranges):
                results[i] = fn(i, r[0], r[1])
                if bar is not None:
                    bar.update(1)
        if bar is not None:
            bar.close()
        return results

    def compute(self, nthreads: int | None = None, msg: str | None = None) -> np.ndarray:
        if self._n_rows == 0:
            return np.empty((0, self._out_cols), dtype=self.dtype)

        def fn(i, s, e):
            return self._materialize_range(s, e)

        parts = self._map_blocks(fn, nthreads, msg)
        return np.vstack(parts) if len(parts) > 1 else parts[0]

    def __array__(self, dtype=None):
        a = self.compute()
        return a.astype(dtype) if dtype is not None else a

    # -- blocks ----------------------------------------------------------
    @property
    def blocks(self):
        for s, e in self._ranges():
            yield Block(self, s, e)

    # -- lazy elementwise ------------------------------------------------
    def _with_op(self, op: _Op, out_cols: int | None = None) -> "ChunkedArray":
        return ChunkedArray(
            self._backing,
            rows=self._rows,
            cols=self._cols,
            ops=self._ops + [op],
            out_cols=self._out_cols if out_cols is None else out_cols,
            block_size=self._block_size,
            nthreads=self._nthreads,
            is_numpy=self._is_numpy,
        )

    def _unary(self, func) -> "ChunkedArray":
        return self._with_op(_unary_op(func))

    def _binary(self, func, other, side: str) -> "ChunkedArray":
        if isinstance(other, _Reduction):
            other = other._arr
        kind, operand = _classify_operand(other, self._n_rows, self._out_cols)
        return self._with_op(_binary_op(func, operand, side, kind))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__" or kwargs.get("out") is not None:
            return NotImplemented
        if len(inputs) == 1:
            return self._unary(ufunc)
        if len(inputs) == 2:
            a, b = inputs
            if a is self:
                return self._binary(ufunc, b, "left")
            return self._binary(ufunc, a, "right")
        return NotImplemented

    def __mul__(self, o):
        return self._binary(np.multiply, o, "left")

    def __rmul__(self, o):
        return self._binary(np.multiply, o, "right")

    def __truediv__(self, o):
        return self._binary(np.true_divide, o, "left")

    def __rtruediv__(self, o):
        return self._binary(np.true_divide, o, "right")

    def __add__(self, o):
        return self._binary(np.add, o, "left")

    def __radd__(self, o):
        return self._binary(np.add, o, "right")

    def __sub__(self, o):
        return self._binary(np.subtract, o, "left")

    def __rsub__(self, o):
        return self._binary(np.subtract, o, "right")

    def __gt__(self, o):
        return self._binary(np.greater, o, "left")

    def __lt__(self, o):
        return self._binary(np.less, o, "left")

    def __ge__(self, o):
        return self._binary(np.greater_equal, o, "left")

    def __le__(self, o):
        return self._binary(np.less_equal, o, "left")

    def dot(self, b) -> "ChunkedArray":
        b = np.asarray(b)
        return self._with_op(_matmul_op(b), out_cols=b.shape[1])

    # -- indexing --------------------------------------------------------
    @staticmethod
    def _local_positions(key, length: int) -> np.ndarray | None:
        """Resolve an index key into integer positions within ``length``.

        Returns None for a full slice (no-op).
        """
        if isinstance(key, slice):
            if key == slice(None):
                return None
            return np.arange(length)[key]
        key = np.asarray(key)
        if key.dtype == bool:
            return np.arange(length)[key]
        return key.astype(int)

    def __getitem__(self, key) -> "ChunkedArray":
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("ChunkedArray supports at most 2D indexing")
            row_key, col_key = key
        else:
            row_key, col_key = key, slice(None)

        rows = self._rows
        cols = self._cols
        ops = self._ops
        out_cols = self._out_cols
        n_rows = self._n_rows

        col_pos = self._local_positions(col_key, out_cols)
        if col_pos is not None:
            ops = [op.subset_cols(col_pos) for op in ops]
            base_cols = cols if cols is not None else np.arange(self._backing.shape[1])
            cols = base_cols[col_pos]
            out_cols = int(col_pos.size)

        row_pos = self._local_positions(row_key, n_rows)
        if row_pos is not None:
            ops = [op.subset_rows(row_pos) for op in ops]
            base_rows = rows if rows is not None else np.arange(self._backing.shape[0])
            rows = base_rows[row_pos]
            n_rows = int(row_pos.size)

        block_size = n_rows if (self._is_numpy and n_rows > 0) else self._block_size
        new = ChunkedArray(
            self._backing,
            rows=rows,
            cols=cols,
            ops=ops,
            out_cols=out_cols,
            block_size=block_size,
            nthreads=self._nthreads,
            is_numpy=self._is_numpy,
        )
        return new

    # -- reductions (deferred, evaluated by streaming over blocks) -------
    def sum(self, axis: int | None = None) -> _Reduction:
        return _Reduction(self, "sum", axis)

    def mean(self, axis: int | None = None) -> _Reduction:
        return _Reduction(self, "mean", axis)

    def var(self, axis: int | None = None) -> _Reduction:
        return _Reduction(self, "var", axis)

    def std(self, axis: int | None = None) -> _Reduction:
        return _Reduction(self, "std", axis)

    def count_nonzero(self, axis: int | None = None) -> _Reduction:
        return _Reduction(self, "count_nonzero", axis)

    def argmax(self, axis: int | None = None) -> _Reduction:
        return _Reduction(self, "argmax", axis)

    def _reduce(self, op: str, axis: int | None, nthreads: int | None, msg: str | None) -> np.ndarray:
        if axis == 1 or axis is None:
            return self._reduce_axis1(op, axis, nthreads, msg)
        return self._reduce_axis0(op, nthreads, msg)

    def _reduce_axis1(self, op: str, axis, nthreads, msg) -> np.ndarray:
        def fn(i, s, e):
            a = self._materialize_range(s, e)
            if op == "sum":
                return a.sum(axis=axis)
            if op == "mean":
                return a.mean(axis=axis)
            if op == "var":
                return a.var(axis=axis)
            if op == "std":
                return a.std(axis=axis)
            if op == "count_nonzero":
                return np.count_nonzero(a, axis=axis)
            if op == "argmax":
                return a.argmax(axis=axis)
            raise ValueError(f"Unknown reduction {op}")

        parts = self._map_blocks(fn, nthreads, msg)
        if axis is None:
            arr = np.array(parts)
            if op == "sum":
                return arr.sum()
            if op == "mean":
                return arr.mean()
            raise ValueError(f"Reduction {op} with axis=None is not supported")
        return np.concatenate(parts)

    def _reduce_axis0(self, op: str, nthreads, msg) -> np.ndarray:
        if op in ("sum", "mean"):
            def fn(i, s, e):
                a = self._materialize_range(s, e)
                return a.sum(axis=0)

            parts = self._map_blocks(fn, nthreads, msg)
            total = np.sum(parts, axis=0)
            if op == "mean":
                return total / self._n_rows
            return total
        if op == "count_nonzero":
            def fn(i, s, e):
                a = self._materialize_range(s, e)
                return np.count_nonzero(a, axis=0)

            parts = self._map_blocks(fn, nthreads, msg)
            return np.sum(parts, axis=0)
        if op in ("var", "std"):
            def fn(i, s, e):
                a = self._materialize_range(s, e).astype(np.float64, copy=False)
                return np.array([a.sum(axis=0), np.square(a).sum(axis=0)])

            parts = self._map_blocks(fn, nthreads, msg)
            stacked = np.sum(parts, axis=0)
            s1, s2 = stacked[0], stacked[1]
            n = self._n_rows
            mean = s1 / n
            variance = s2 / n - np.square(mean)
            return np.sqrt(np.clip(variance, 0, None)) if op == "std" else variance
        if op == "argmax":
            raise NotImplementedError("argmax(axis=0) is not supported")
        raise ValueError(f"Unknown reduction {op}")

    def __repr__(self) -> str:
        return (
            f"ChunkedArray(shape={self.shape}, dtype={self.dtype}, "
            f"chunksize={self.chunksize}, numblocks={self.numblocks[0]})"
        )
