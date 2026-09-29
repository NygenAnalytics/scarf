from collections.abc import Callable
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray


type OpKind = Literal["unary", "binary"]
type BType = Literal["scalar", "col", "row", "full"]
type UfuncSide = Literal["left", "right"]


class _Op:
    """One deferred element-wise operation."""

    __slots__ = ("kind", "func", "operand", "side", "btype")

    def __init__(
        self,
        kind: OpKind,
        func: Callable[..., NDArray[Any]],
        operand: object | None = None,
        side: UfuncSide = "left",
        btype: BType | None = None,
    ) -> None:
        self.kind = kind
        self.func = func
        self.operand = operand
        self.side = side
        self.btype = btype

    def apply(self, a: NDArray[Any], start: int, end: int) -> NDArray[Any]:
        if self.kind == "unary":
            return np.asarray(self.func(a))
        operand = self.operand
        if self.btype in ("col", "full"):
            operand = np.asarray(operand)[start:end]
        return (
            np.asarray(self.func(a, operand))
            if self.side == "left"
            else np.asarray(self.func(operand, a))
        )

    def subset_cols(self, col_idx: np.ndarray) -> "_Op":
        if self.kind == "binary" and self.btype == "row":
            return _Op(
                "binary",
                self.func,
                np.asarray(self.operand)[..., col_idx],
                self.side,
                "row",
            )
        if self.kind == "binary" and self.btype == "full":
            return _Op(
                "binary",
                self.func,
                np.asarray(self.operand)[:, col_idx],
                self.side,
                "full",
            )
        return self

    def subset_rows(self, row_idx: np.ndarray) -> "_Op":
        if self.kind == "binary" and self.btype in ("col", "full"):
            return _Op(
                "binary",
                self.func,
                np.asarray(self.operand)[row_idx],
                self.side,
                self.btype,
            )
        return self


def _unary_op(func: Callable[..., NDArray[Any]]) -> _Op:
    return _Op("unary", func=func)


def _classify_operand(
    other: object,
    n_rows: int,
    n_cols: int,
) -> tuple[BType, object]:
    """Classify an operand by how NumPy broadcasts it against the matrix.

    A one-dimensional operand aligns with the columns, as in NumPy. Rows are
    scaled only by an explicit ``(n_rows, 1)`` operand.
    """
    if np.isscalar(other):
        return "scalar", other
    array = np.asarray(other)
    if array.ndim <= 2 and array.size == 1:
        return "scalar", array.reshape(())
    if array.shape == (n_cols,):
        return "row", array
    if array.ndim == 2:
        if array.shape == (n_rows, n_cols):
            return "full", array
        if array.shape == (1, n_cols):
            return "row", array
        if array.shape == (n_rows, 1):
            return "col", array
    raise ValueError(
        f"An operand of shape {array.shape} does not broadcast against a "
        f"ChunkedArray of shape {(n_rows, n_cols)}; scale rows with a "
        f"({n_rows}, 1) array"
    )


def _binary_op(
    func: Callable[..., NDArray[Any]],
    other: object,
    side: str,
    kind: BType,
) -> _Op:
    return _Op(
        "binary",
        func=func,
        operand=other,
        side=cast(UfuncSide, side),
        btype=kind,
    )
