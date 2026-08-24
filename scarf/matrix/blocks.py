from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .chunked import ChunkedArray


class Block:
    """A single row-chunk view of a ChunkedArray."""

    __slots__ = ("_parent", "_start", "_end", "_row_perm")

    def __init__(
        self,
        parent: "ChunkedArray",
        start: int,
        end: int,
        row_perm: np.ndarray | None = None,
    ) -> None:
        self._parent = parent
        self._start = start
        self._end = end
        self._row_perm = row_perm

    @property
    def shape(self) -> tuple[int, int]:
        n_rows = (
            self._end - self._start if self._row_perm is None else len(self._row_perm)
        )
        return (n_rows, self._parent.out_cols)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._parent.dtype

    def compute(
        self,
        nthreads: int | None = None,
        msg: str | None = None,
    ) -> np.ndarray:
        array = self._parent._materialize_range(self._start, self._end)
        if self._row_perm is not None:
            array = array[self._row_perm]
        return array

    def __array__(self, dtype: np.dtype[Any] | None = None) -> np.ndarray:
        array = self.compute()
        return array.astype(dtype) if dtype is not None else array

    def __getitem__(self, key: object) -> "Block":
        if isinstance(key, tuple):
            row_key = key[0]
        else:
            row_key = key
        return Block(
            self._parent,
            self._start,
            self._end,
            row_perm=np.asarray(row_key),
        )
