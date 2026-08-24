from collections.abc import Callable
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray


type OpKind = Literal["unary", "binary", "matmul"]
type BType = Literal["scalar", "col", "row", "full"]
type UfuncSide = Literal["left", "right"]


class _Op:
    """One deferred element-wise or matrix multiplication operation."""

    __slots__ = ("kind", "func", "operand", "side", "btype")

    def __init__(
        self,
        kind: OpKind,
        func: Callable[..., NDArray[Any]] | None = None,
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
            assert self.func is not None
            return np.asarray(self.func(a))
        if self.kind == "matmul":
            return np.asarray(a @ self.operand)
        assert self.func is not None
        operand = self.operand
        if self.btype == "col":
            operand = np.asarray(operand)[start:end]
            if np.asarray(operand).ndim == 1:
                operand = np.asarray(operand).reshape(-1, 1)
        elif self.btype == "full":
            operand = np.asarray(operand)[start:end]
        return (
            np.asarray(self.func(a, operand))
            if self.side == "left"
            else np.asarray(self.func(operand, a))
        )

    def subset_cols(self, col_idx: np.ndarray) -> "_Op":
        if self.kind == "matmul":
            raise NotImplementedError("Column-indexing after .dot is not supported")
        if self.kind == "binary" and self.btype == "row":
            operand = np.asarray(self.operand)
            return _Op(
                "binary",
                self.func,
                operand[col_idx] if operand.ndim == 1 else operand[:, col_idx],
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
        if self.kind == "binary" and self.btype == "col":
            return _Op(
                "binary",
                self.func,
                np.asarray(self.operand)[row_idx],
                self.side,
                "col",
            )
        if self.kind == "binary" and self.btype == "full":
            return _Op(
                "binary",
                self.func,
                np.asarray(self.operand)[row_idx],
                self.side,
                "full",
            )
        return self


def _unary_op(func: Callable[..., NDArray[Any]]) -> _Op:
    return _Op("unary", func=func)


def _matmul_op(operand: np.ndarray) -> _Op:
    return _Op("matmul", operand=operand)


def _classify_operand(
    other: object,
    n_rows: int,
    n_cols: int,
) -> tuple[BType, object]:
    if np.isscalar(other):
        return "scalar", other
    array = np.asarray(other)
    if array.ndim == 0:
        return "scalar", array
    shape = array.shape
    if (array.ndim == 1 and shape[0] == n_rows) or (
        array.ndim == 2 and shape == (n_rows, 1)
    ):
        return "col", array
    if (array.ndim == 1 and shape[0] == n_cols) or (
        array.ndim == 2 and shape == (1, n_cols)
    ):
        return "row", array
    if array.ndim == 2 and shape[0] == n_rows:
        return "full", array
    if array.size == 1:
        return "scalar", array
    return "row", array


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
