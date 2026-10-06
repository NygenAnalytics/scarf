"""Column moments that combine exactly across row blocks.

A streamed variance must not subtract the squared mean from the mean of the
squares. Both terms are large when a column's mean is large beside its
spread, so their difference loses its significant digits or turns negative.
These helpers keep instead, for each column, the count, the sum, and ``m2``:
the sum of squared deviations from the column mean. A two-pass kernel
computes them for a block, and blocks combine with the update of Chan,
Golub, and LeVeque, which adds the squared difference of the block means
weighted by the block counts. A block's sums are the row-order float64 sums
that NumPy computes for a C-ordered block, and merges add them in block
order, so streamed sums keep the bits they had before ``m2`` replaced the
squared sums.

Kernels over sparse values update a running mean and ``m2`` over the stored
values with ``welford_add`` and then add the implicit zeros with
``add_implicit_zeros``. Numba caches a kernel together with the helpers it
calls and does not notice a change to a helper defined in another file, so
after changing these helpers, delete the Numba caches of the kernels that
call them (the ``*.nbi`` and ``*.nbc`` files under ``__pycache__``).
"""

from dataclasses import dataclass

import numpy as np
from numba import njit
from numpy.typing import NDArray

__all__ = [
    "ColumnMoments",
    "add_implicit_zeros",
    "column_moments",
    "welford_add",
]


@dataclass(frozen=True, slots=True)
class ColumnMoments:
    """The count, sum, and centered sum of squares of each column."""

    count: int
    total: NDArray[np.float64]
    m2: NDArray[np.float64]

    def merge(self, other: "ColumnMoments") -> "ColumnMoments":
        """Return the moments of these values followed by those of ``other``."""
        if self.total.shape != other.total.shape:
            raise ValueError(
                f"Cannot merge moments of {self.total.size} and "
                f"{other.total.size} columns"
            )
        count = self.count + other.count
        total = self.total + other.total
        if self.count == 0 or other.count == 0:
            return ColumnMoments(count, total, self.m2 + other.m2)
        # The merged m2 starts as the difference of the means, so a merge
        # holds its two results and one temporary of the columns.
        m2 = np.divide(other.total, other.count)
        m2 -= self.total / self.count
        np.square(m2, out=m2)
        m2 *= self.count * other.count / count
        m2 += self.m2
        m2 += other.m2
        return ColumnMoments(count, total, m2)

    @property
    def mean(self) -> NDArray[np.float64]:
        """The mean of each column; NaN when there are no values."""
        if self.count == 0:
            return np.full(self.total.shape, np.nan)
        return self.total / self.count

    def variance(self, ddof: int = 0) -> NDArray[np.float64]:
        """Return the variance of each column with ``ddof`` degrees removed."""
        divisor = self.count - int(ddof)
        if divisor <= 0:
            return np.full(self.total.shape, np.nan)
        return self.m2 / divisor


@njit(cache=True, nogil=True)
def _column_moments_kernel(
    values: np.ndarray,
    total: np.ndarray,
    mean: np.ndarray,
    m2: np.ndarray,
) -> None:
    """Add the column sums of ``values`` to ``total`` and its ``m2`` to ``m2``.

    Both passes read the block in row order, so each sum adds its column's
    values one row after another. ``mean`` is scratch for the column means.
    """
    n_rows, n_columns = values.shape
    for row in range(n_rows):
        for column in range(n_columns):
            total[column] += np.float64(values[row, column])
    for column in range(n_columns):
        mean[column] = total[column] / n_rows
    for row in range(n_rows):
        for column in range(n_columns):
            deviation = np.float64(values[row, column]) - mean[column]
            m2[column] += deviation * deviation


def column_moments(block: np.ndarray) -> ColumnMoments:
    """Return the moments of each column of a two-dimensional block."""
    values = np.asarray(block)
    if values.ndim != 2:
        raise ValueError("Column moments need a two-dimensional block")
    if values.dtype.kind == "b":
        values = values.view(np.uint8)
    elif values.dtype.kind not in "iuf" or values.dtype.itemsize > 8:
        raise TypeError(f"Column moments do not support {values.dtype} values")
    elif values.dtype.kind == "f" and values.dtype.itemsize < 4:
        values = values.astype(np.float32)
    elif not values.dtype.isnative:
        values = values.astype(values.dtype.newbyteorder("="))
    n_rows, n_columns = values.shape
    total = np.zeros(n_columns, dtype=np.float64)
    m2 = np.zeros(n_columns, dtype=np.float64)
    if n_rows:
        mean = np.empty(n_columns, dtype=np.float64)
        _column_moments_kernel(values, total, mean, m2)
    return ColumnMoments(int(n_rows), total, m2)


@njit(cache=True, nogil=True, inline="always")
def welford_add(
    count: int, mean: float, m2: float, value: float
) -> tuple[float, float]:
    """Return the mean and ``m2`` updated with ``value`` as the ``count``-th value."""
    delta = value - mean
    mean += delta / count
    return mean, m2 + delta * (value - mean)


@njit(cache=True, nogil=True, inline="always")
def add_implicit_zeros(m2: float, mean: float, count: int, total_count: int) -> float:
    """Return the ``m2`` of ``count`` values joined by ``total_count - count`` zeros."""
    zeros = total_count - count
    if count == 0 or zeros == 0:
        return m2
    return m2 + mean * mean * (np.float64(count) * np.float64(zeros) / total_count)
