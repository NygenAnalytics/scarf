import hashlib
import math
import re
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import DTypeLike, NDArray

if TYPE_CHECKING:
    import pandas as pd


def regex_match_mask(values: Sequence[str] | np.ndarray, pattern: str) -> np.ndarray:
    expression = re.compile(pattern, re.IGNORECASE)
    return np.fromiter(
        (expression.match(str(value)) is not None for value in values),
        dtype=bool,
        count=len(values),
    )


def has_duplicates(values: Any) -> bool:
    """Return whether a one-dimensional array repeats a value.

    Sorting and comparing neighbours stays fast where ``np.unique`` is slow on
    many distinct integers; increasing values skip the sort.
    """
    array = np.asarray(values)
    if array.size < 2:
        return False
    ordered = array if np.all(array[1:] > array[:-1]) else np.sort(array)
    return bool(np.any(ordered[1:] == ordered[:-1]))


def read_only_copy(values: Any, dtype: DTypeLike | None = None) -> np.ndarray:
    """Return a C-ordered copy of ``values`` that cannot be made writable.

    The copy is a view of a read-only owner, so setting its ``writeable`` flag
    raises instead of exposing the data to changes.
    """
    owner = np.array(values, dtype=dtype, order="C", copy=True)
    owner.setflags(write=False)
    return owner.view()


def within_bounds(
    values: np.ndarray,
    lower: Any,
    upper: Any,
    *,
    keep_bounds: bool,
) -> np.ndarray:
    """Return where values lie between two bounds.

    ``keep_bounds`` retains values equal to a bound. ``None`` leaves a side
    open. Numeric values are compared with infinite open sides, so NaN never
    lies within bounds. Text values are compared lexically with text bounds.
    """
    resolved = np.asarray(values)
    if resolved.dtype.kind in "biufc":
        lower = -np.inf if lower is None else lower
        upper = np.inf if upper is None else upper
    keep = np.ones(resolved.shape, dtype=bool)
    if lower is not None:
        keep &= resolved >= lower if keep_bounds else resolved > lower
    if upper is not None:
        keep &= resolved <= upper if keep_bounds else resolved < upper
    return keep


# Only plain decimal notation is a numeric label, so "1_2" and "inf" are text.
_NUMERIC_LABEL = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")


def _category_sort_key(value: Any) -> tuple[Any, ...]:
    """Sort key: numbers in numeric order, then natural text, then missing.

    Every token of a text label records whether it is a digit run, so digit
    runs never compare with text and precede it where two labels differ in
    kind. The exact text breaks ties between labels that differ only in case,
    so the order never depends on input order.
    """
    import pandas as pd

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return (2, (), "")
    try:
        if pd.isna(value):
            return (2, (), "")
    except (TypeError, ValueError):
        pass

    if isinstance(value, (bool, np.bool_)):
        text = str(bool(value))
    elif isinstance(value, (int, np.integer, float, np.floating)):
        return (0, (float(value),), str(value))
    else:
        text = str(value)
    if _NUMERIC_LABEL.fullmatch(text):
        return (0, (float(text),), text)
    # Splitting on a captured pattern puts the digit runs at odd positions.
    tokens = tuple(
        (0, int(part)) if position % 2 else (1, part.casefold())
        for position, part in enumerate(re.split(r"(\d+)", text))
        if part
    )
    return (1, tokens, text)


def sort_categories(values: Iterable[Any]) -> list[Any]:
    """Order categories naturally, as plots and marker tables show them.

    Numbers and decimal labels come first by value, so ``"2"`` precedes
    ``"10"``. Other labels follow in natural text order (``"A2"`` before
    ``"A10"``, ``"2_T"`` before ``"B"``), and missing values come last.
    """
    return sorted(values, key=_category_sort_key)


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


def sparse_matrix_bytes(*matrices: Any) -> int:
    """Return the bytes of the arrays held by SciPy sparse matrices.

    An array shared by several matrices, or a matrix passed more than once, is
    counted once.
    """
    arrays = (
        getattr(matrix, name, None)
        for matrix in matrices
        for name in ("data", "row", "col", "indices", "indptr")
    )
    unique = {id(array): array for array in arrays if isinstance(array, np.ndarray)}
    return int(sum(array.nbytes for array in unique.values()))


def cumulative_nnz(row_nnz: np.ndarray) -> np.ndarray:
    """Return row-count prefix sums with a leading zero."""
    counts = np.asarray(row_nnz)
    cumulative = np.empty(counts.size + 1, dtype=np.int64)
    cumulative[0] = 0
    np.cumsum(counts, dtype=np.int64, out=cumulative[1:])
    return cumulative


def max_window_nnz(cumulative: np.ndarray, window_rows: int) -> int:
    """Return the largest stored-value count of any contiguous row window.

    Args:
        cumulative: Row-count prefix sums, such as a CSR ``indptr`` or the
            result of :func:`cumulative_nnz`.
        window_rows: Rows per window; windows are clipped to the row count.
    """
    if window_rows <= 0:
        raise ValueError("window_rows must be positive")
    width = min(int(window_rows), cumulative.size - 1)
    if width <= 0:
        return 0
    return int(np.max(cumulative[width:] - cumulative[:-width]))


def assay_feature_ranges(
    assay_feats: "pd.DataFrame",
) -> dict[str, tuple[tuple[int, int], ...]]:
    """Return the source feature ranges of each assay in first-seen order.

    Args:
        assay_feats: Assay table with one column per contiguous feature-type
            span and ``start`` and ``end`` rows, as readers expose it in
            ``assayFeats``. Spans that share an assay name belong to the same
            assay.

    Returns:
        Half-open ``(start, end)`` feature index ranges keyed by assay name.
    """
    ranges: dict[str, list[tuple[int, int]]] = {}
    for name, start, end in zip(
        assay_feats.columns,
        assay_feats.loc["start"],
        assay_feats.loc["end"],
        strict=True,
    ):
        ranges.setdefault(str(name), []).append((int(start), int(end)))
    return {name: tuple(spans) for name, spans in ranges.items()}


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
    array = np.nan_to_num(
        array, copy=True, nan=fill_val, posinf=fill_val, neginf=fill_val
    )
    array[array == 0] = fill_val
    return array


def sum_and_squared_sum(
    array: np.ndarray, axis: int | None = 0
) -> tuple[np.ndarray, np.ndarray]:
    expressions = {None: "ij,ij->", 0: "ij,ij->j", 1: "ij,ij->i"}
    return (
        np.asarray(array.sum(axis=axis, dtype=np.float64)),
        np.asarray(
            np.einsum(expressions[axis], array, array, dtype=np.float64, optimize=False)
        ),
    )


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
