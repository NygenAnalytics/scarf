"""Compiled copies between strided arrays."""

import numpy as np
from numba import njit

_TILE = 32


@njit(cache=True, nogil=True)
def transpose_into(source: np.ndarray, target: np.ndarray) -> None:
    """Copy ``source.T`` into ``target`` one cache-sized tile at a time.

    ``target`` may be a strided view. A plain strided copy touches a new cache
    line for almost every element; tiles keep both sides cache resident.
    """
    rows, columns = source.shape
    for row_start in range(0, rows, _TILE):
        row_stop = min(row_start + _TILE, rows)
        for column_start in range(0, columns, _TILE):
            column_stop = min(column_start + _TILE, columns)
            for row in range(row_start, row_stop):
                for column in range(column_start, column_stop):
                    target[column, row] = source[row, column]


@njit(cache=True, nogil=True)
def copy_columns(
    source: np.ndarray,
    columns: np.ndarray,
    target: np.ndarray,
    positions: np.ndarray,
) -> None:
    """Copy ``source[:, columns[j]]`` into ``target[:, positions[j]]`` row by row."""
    for row in range(source.shape[0]):
        for index in range(columns.shape[0]):
            target[row, positions[index]] = source[row, columns[index]]
