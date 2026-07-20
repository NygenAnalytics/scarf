import hashlib
from typing import Any

import numpy as np
from numpy.typing import NDArray


def rescale_array(a: np.ndarray, frac: float = 0.9) -> np.ndarray:
    """Trim extreme values using a fitted normal distribution."""
    from scipy.stats import norm

    location = (np.median(a) + np.median(a[::-1])) / 2
    distribution = norm(location, np.std(a))
    minimum, maximum = distribution.ppf(1 - frac), distribution.ppf(frac)
    a[a < minimum] = minimum
    a[a > maximum] = maximum
    return a


def clean_array(
    x: NDArray[Any] | list[Any],
    fill_val: int | float = 0,
) -> NDArray[Any]:
    """Replace non-finite and zero values in a numeric array."""
    array = np.asarray(x, dtype=np.float64)
    array = np.nan_to_num(array, copy=True)
    array[(array == np.inf) | (array == -np.inf)] = 0
    array[array == 0] = fill_val
    return array


def array_digest(values: np.ndarray) -> str:
    """Return a deterministic digest for a non-object NumPy array."""
    array = np.ascontiguousarray(values)
    if array.dtype.hasobject:
        raise TypeError("Cannot create a deterministic digest for object arrays")
    digest = hashlib.blake2b(digest_size=16)
    digest.update(array.dtype.str.encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.view(np.uint8).tobytes())
    return digest.hexdigest()


def permute_into_chunks(
    size: int,
    chunk_size: int,
    seed: int = 42,
) -> list[np.ndarray]:
    """Split sequential indices into independently permuted chunks."""
    rng = np.random.default_rng(seed=seed)
    array = np.arange(size)
    end = len(array) - len(array) % chunk_size
    chunks = [array[index : index + chunk_size] for index in range(0, end, chunk_size)]
    permuted = [rng.permutation(chunk) for chunk in chunks]
    if end < len(array):
        permuted.append(rng.permutation(array[end:]))
    return permuted


def _rolling_window_kernel(a: np.ndarray, w: int) -> np.ndarray:
    if a.ndim != 2:
        raise ValueError("a must be a two-dimensional array")
    if w <= 0:
        raise ValueError("w must be greater than zero")

    n, m = a.shape
    if n == 0:
        raise ValueError("a must contain at least one row")
    w = min(w, n)
    left = (w - 1) // 2
    right = w // 2
    result = np.empty((n, m), dtype=np.float64)

    for column in range(m):
        cumulative = np.empty(n + 1, dtype=np.float64)
        cumulative[0] = 0.0
        for row in range(n):
            cumulative[row + 1] = cumulative[row] + a[row, column]
        for row in range(n):
            start = max(0, row - left)
            stop = min(n, row + right + 1)
            result[row, column] = (cumulative[stop] - cumulative[start]) / (
                stop - start
            )
    return result


_rollingWindowImpl: Any = None


def rolling_window(a: np.ndarray, w: int) -> np.ndarray:
    """Apply a centered rolling mean along the first axis."""
    global _rollingWindowImpl
    if _rollingWindowImpl is None:
        from numba import jit

        _rollingWindowImpl = jit(nopython=True)(_rolling_window_kernel)
    return np.asarray(_rollingWindowImpl(a, w))
