import hashlib
from typing import Any

import numpy as np
from numpy.typing import NDArray


def checked_sparse_cast(values: np.ndarray, dtype: Any) -> np.ndarray:
    destination_dtype = np.dtype(dtype)
    if destination_dtype.kind in "biu" and values.size:
        if values.dtype.kind == "c":
            raise OverflowError(
                "Complex sparse values cannot use an integer destination dtype"
            )
        if values.dtype.kind == "f":
            if (
                not np.isfinite(values).all()
                or not np.equal(values, np.trunc(values)).all()
            ):
                raise OverflowError(
                    "Sparse values cannot be represented by the destination dtype"
                )
        if destination_dtype.kind == "b":
            lower, upper = 0, 1
        else:
            limits = np.iinfo(destination_dtype)
            lower, upper = limits.min, limits.max
        if values.min() < lower or values.max() > upper:
            raise OverflowError("Sparse values exceed the destination dtype")
    return values.astype(destination_dtype, copy=False)


def _canonical_64bit_integer_sparse(coo: Any) -> Any:
    from scipy.sparse import coo_matrix

    row = np.asarray(coo.row)
    column = np.asarray(coo.col)
    data = np.asarray(coo.data)
    if data.size == 0:
        canonical = coo_matrix(coo.shape, dtype=data.dtype)
        canonical.has_canonical_format = True
        return canonical
    order = np.lexsort((column, row))
    row = row[order]
    column = column[order]
    data = data[order]
    starts = np.flatnonzero(
        np.concatenate(
            (
                np.array([True]),
                (row[1:] != row[:-1]) | (column[1:] != column[:-1]),
            )
        )
    )
    ends = np.append(starts[1:], data.size)
    if starts.size == data.size:
        summed = data
    else:
        limits = np.iinfo(data.dtype)
        summed = np.empty(starts.size, dtype=data.dtype)
        for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
            total = sum(int(value) for value in data[start:end])
            if total < limits.min or total > limits.max:
                raise OverflowError(
                    "Duplicate sparse values exceed supported integer dtypes"
                )
            summed[index] = total
    canonical = coo_matrix(
        (summed, (row[starts], column[starts])),
        shape=coo.shape,
    )
    canonical.has_canonical_format = True
    return canonical


def canonicalize_sparse(coo: Any, dtype: Any | None = None) -> Any:
    from scipy.sparse import coo_matrix

    if bool(getattr(coo, "has_canonical_format", False)):
        if dtype is not None:
            coo.data = checked_sparse_cast(np.asarray(coo.data), dtype)
        return coo
    data = np.asarray(coo.data)
    if data.dtype.kind in "biu":
        if data.dtype.itemsize >= 8:
            canonical = _canonical_64bit_integer_sparse(coo)
        else:
            accumulator_dtype = np.uint64 if data.dtype.kind in "bu" else np.int64
            data = data.astype(accumulator_dtype)
            canonical = coo_matrix(
                (data, (coo.row, coo.col)),
                shape=coo.shape,
            )
            canonical.sum_duplicates()
    else:
        canonical = coo_matrix(
            (data, (coo.row, coo.col)),
            shape=coo.shape,
        )
        canonical.sum_duplicates()
    if dtype is not None:
        canonical.data = checked_sparse_cast(canonical.data, dtype)
    return canonical


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
