"""Compiled per-row digests and totals for dense count blocks."""

import numpy as np
from numba import njit

_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_ODD = np.uint64(0xD6E8FEB86659FD93)


@njit(cache=True, inline="always")
def _mix(value: np.uint64) -> np.uint64:
    value = (value ^ (value >> np.uint64(30))) * _M1
    value = (value ^ (value >> np.uint64(27))) * _M2
    return value ^ (value >> np.uint64(31))


@njit(cache=True, nogil=True)
def summarize_rows(
    values: np.ndarray,
    bits: np.ndarray,
    digests: np.ndarray,
    row_sums: np.ndarray,
    row_positive: np.ndarray,
    column_positive: np.ndarray,
) -> None:
    """Digest each row's nonzero column-value pairs and total its values.

    ``bits`` is the unsigned-integer view of ``values``. Zero entries are
    skipped, so digests depend only on each row's sparse content. Positive
    entries are counted per row and added to ``column_positive``.
    """
    for row in range(values.shape[0]):
        first = np.uint64(0x243F6A8885A308D3)
        second = np.uint64(0x13198A2E03707344)
        nonzero = np.uint64(0)
        positive = 0
        total = 0.0
        for column in range(values.shape[1]):
            value = values[row, column]
            if value != 0:
                key = _mix(
                    ((np.uint64(column) + np.uint64(1)) * _GOLDEN)
                    ^ np.uint64(bits[row, column])
                )
                first = _mix(first ^ key)
                second = (second + key) * _ODD
                nonzero += np.uint64(1)
                total += value
                if value > 0:
                    positive += 1
                    column_positive[column] += 1
        digests[row, 0] = _mix(first ^ nonzero)
        digests[row, 1] = _mix(second ^ (nonzero * _GOLDEN))
        row_sums[row] = total
        row_positive[row] = positive
