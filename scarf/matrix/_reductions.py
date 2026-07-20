from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .chunked import ChunkedArray


type ReductionOp = Literal["sum", "mean", "var", "std", "count_nonzero", "argmax"]
type UfuncSide = Literal["left", "right"]


class _Reduction:
    """A deferred and cached reduction over a ChunkedArray."""

    __slots__ = ("_parent", "_op", "_axis", "_cached")

    def __init__(
        self,
        parent: "ChunkedArray",
        op: ReductionOp,
        axis: int | None,
    ) -> None:
        self._parent = parent
        self._op = op
        self._axis = axis
        self._cached: np.ndarray | None = None

    def compute(
        self,
        nthreads: int | None = None,
        msg: str | None = None,
    ) -> np.ndarray:
        if self._cached is None:
            self._cached = self._parent._reduce(
                self._op,
                self._axis,
                nthreads,
                msg,
            )
        return self._cached

    @property
    def _arr(self) -> np.ndarray:
        return self.compute()

    def __array__(self, dtype: np.dtype[Any] | None = None) -> np.ndarray:
        array = self._arr
        return array.astype(dtype) if dtype is not None else array

    def __array_ufunc__(
        self,
        ufunc: Any,
        method: str,
        *inputs: Any,
        **kwargs: Any,
    ) -> Any:
        if method != "__call__":
            return NotImplemented
        resolved = tuple(
            item._arr if isinstance(item, _Reduction) else item for item in inputs
        )
        return ufunc(*resolved, **kwargs)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._arr, item)

    def __len__(self) -> int:
        return len(self._arr)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._arr)

    def __getitem__(self, key: object) -> Any:
        return self._arr[cast(Any, key)]

    def _bin(
        self,
        other: object,
        func: Callable[[NDArray[Any], NDArray[Any]], NDArray[Any]],
        side: UfuncSide,
    ) -> NDArray[Any]:
        other_array = other._arr if isinstance(other, _Reduction) else np.asarray(other)
        if side == "left":
            return func(self._arr, other_array)
        return func(other_array, self._arr)

    def __mul__(self, other: object) -> NDArray[Any]:
        return self._bin(other, np.multiply, "left")

    def __rmul__(self, other: object) -> NDArray[Any]:
        return self._bin(other, np.multiply, "right")

    def __truediv__(self, other: object) -> NDArray[Any]:
        return self._bin(other, np.true_divide, "left")

    def __rtruediv__(self, other: object) -> NDArray[Any]:
        return self._bin(other, np.true_divide, "right")

    def __add__(self, other: object) -> NDArray[Any]:
        return self._bin(other, np.add, "left")

    def __radd__(self, other: object) -> NDArray[Any]:
        return self._bin(other, np.add, "right")

    def __sub__(self, other: object) -> NDArray[Any]:
        return self._bin(other, np.subtract, "left")

    def __rsub__(self, other: object) -> NDArray[Any]:
        return self._bin(other, np.subtract, "right")

    def __gt__(self, other: object) -> NDArray[Any]:
        return self._bin(other, np.greater, "left")

    def __lt__(self, other: object) -> NDArray[Any]:
        return self._bin(other, np.less, "left")

    def __ge__(self, other: object) -> NDArray[Any]:
        return self._bin(other, np.greater_equal, "left")

    def __le__(self, other: object) -> NDArray[Any]:
        return self._bin(other, np.less_equal, "left")

    def __repr__(self) -> str:
        return f"<deferred {self._op}(axis={self._axis})>"
