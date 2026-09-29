import numpy as np


def local_positions(key: object, length: int) -> np.ndarray | None:
    """Resolve one axis key into integer positions, or None for a full slice.

    A ChunkedArray always stays two-dimensional, so scalar keys are rejected.
    Select one row or column with ``[i : i + 1]`` or ``[[i]]``.
    """
    if isinstance(key, slice):
        if key == slice(None):
            return None
        return np.asarray(np.arange(length)[key])
    key_array = np.asarray(key)
    if key_array.ndim != 1:
        raise IndexError(
            "ChunkedArray keys must be slices or one-dimensional index arrays; "
            "select one row or column with [i : i + 1] or [[i]]"
        )
    if key_array.dtype == bool:
        return np.asarray(np.arange(length)[key_array])
    if key_array.size == 0:
        return np.empty(0, dtype=np.intp)
    if not np.issubdtype(key_array.dtype, np.integer):
        raise IndexError("ChunkedArray index arrays must hold integers or booleans")
    return key_array.astype(np.intp, copy=False)
