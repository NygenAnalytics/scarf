import numpy as np


def is_contiguous(indices: np.ndarray) -> bool:
    """Return whether indices form a strictly increasing consecutive run."""
    if indices.size == 0:
        return True
    return bool(
        indices[0] >= 0
        and np.array_equal(
            indices,
            np.arange(indices[0], indices[0] + indices.size),
        )
    )


def local_positions(key: object, length: int) -> np.ndarray | None:
    """Resolve an index key into integer positions."""
    if isinstance(key, slice):
        if key == slice(None):
            return None
        return np.asarray(np.arange(length)[key])
    key_array = np.asarray(key)
    if key_array.dtype == bool:
        return np.asarray(np.arange(length)[key_array])
    return np.asarray(key_array.astype(int))
