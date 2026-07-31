import numpy as np
import zarr

from ..matrix import ChunkedArray
from ._types import MatrixData


def read_matrix_rows(data: MatrixData, row_indices: np.ndarray) -> np.ndarray:
    row_indices = np.asarray(row_indices, dtype=np.int64)
    if isinstance(data, ChunkedArray):
        return data[row_indices].compute()
    if isinstance(data, zarr.Array):
        return np.asarray(data.get_orthogonal_selection((row_indices, slice(None))))
    return np.asarray(data[row_indices])
