"""Column moments: exact merges, never-negative variances, bounded memory."""

import operator
import tracemalloc
from functools import reduce

import numpy as np
import pytest
from numba import njit

from scarf.utils.moments import (
    add_implicit_zeros,
    column_moments,
    welford_add,
)

MOMENT_DTYPES = ["bool", "uint64", "int64", "float16", "float32", "float64"]


def _values(dtype: str, shape: tuple[int, int], seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if dtype == "bool":
        return rng.random(shape) > 0.6
    if np.dtype(dtype).kind == "f":
        return (rng.standard_normal(shape) * 7 + 3).astype(dtype)
    return rng.integers(0, 100, size=shape).astype(dtype)


def _two_pass_m2(values: np.ndarray) -> np.ndarray:
    widened = values.astype(np.float64)
    return np.square(widened - widened.mean(axis=0)).sum(axis=0)


@pytest.mark.parametrize("dtype", MOMENT_DTYPES)
@pytest.mark.parametrize("shape", [(1, 3), (37, 5)])
def test_column_moments_match_numpy_two_pass_statistics(dtype, shape):
    values = _values(dtype, shape)
    widened = values.astype(np.float64)
    moments = column_moments(values)

    assert moments.count == shape[0]
    assert moments.total.dtype == moments.m2.dtype == np.float64
    np.testing.assert_allclose(moments.total, widened.sum(axis=0), rtol=1e-14)
    np.testing.assert_allclose(moments.m2, _two_pass_m2(values), rtol=1e-12)
    np.testing.assert_allclose(moments.mean, widened.mean(axis=0), rtol=1e-14)
    np.testing.assert_allclose(moments.variance(), widened.var(axis=0), rtol=1e-12)
    if shape[0] > 1:
        np.testing.assert_allclose(
            moments.variance(ddof=1), widened.var(axis=0, ddof=1), rtol=1e-12
        )
    else:
        assert np.isnan(moments.variance(ddof=1)).all()


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_column_sums_keep_the_bits_of_a_row_order_float64_sum(dtype):
    values = _values(dtype, (513, 7), seed=3)
    sequential = np.zeros(values.shape[1])
    for row in values:
        sequential += row.astype(np.float64)

    total = column_moments(values).total

    np.testing.assert_array_equal(total, sequential)
    # NumPy also adds the rows of a C-ordered block one after another.
    np.testing.assert_array_equal(total, values.sum(axis=0, dtype=np.float64))


@pytest.mark.parametrize("dtype", ["float32", "uint64"])
def test_merged_blocks_match_the_whole_matrix(dtype):
    values = _values(dtype, (101, 6), seed=5)
    whole = column_moments(values)
    for bounds in ([0, 1, 101], [0, 50, 51, 101], [0, 33, 34, 35, 100, 101]):
        blocks = [
            column_moments(values[start:end])
            for start, end in zip(bounds[:-1], bounds[1:], strict=True)
        ]
        merged = reduce(lambda left, right: left.merge(right), blocks)

        assert merged.count == whole.count
        # Totals add as plain float64 sums in block order.
        np.testing.assert_array_equal(
            merged.total, reduce(operator.add, [block.total for block in blocks])
        )
        np.testing.assert_allclose(merged.m2, whole.m2, rtol=1e-12)


def test_moments_without_values():
    empty = column_moments(np.empty((0, 3), dtype=np.float32))
    assert empty.count == 0
    np.testing.assert_array_equal(empty.total, np.zeros(3))
    np.testing.assert_array_equal(empty.m2, np.zeros(3))
    assert np.isnan(empty.mean).all() and np.isnan(empty.variance()).all()

    values = _values("float64", (10, 3))
    filled = column_moments(values)
    for merged in (empty.merge(filled), filled.merge(empty)):
        assert merged.count == 10
        np.testing.assert_array_equal(merged.total, filled.total)
        np.testing.assert_array_equal(merged.m2, filled.m2)
    one = column_moments(values[:1])
    assert np.isnan(one.variance(ddof=1)).all()
    np.testing.assert_array_equal(one.variance(), np.zeros(3))


def test_column_moments_reject_other_inputs():
    with pytest.raises(ValueError, match="two-dimensional"):
        column_moments(np.ones(3))
    with pytest.raises(TypeError, match="complex"):
        column_moments(np.ones((2, 2), dtype=np.complex128))
    if np.dtype(np.longdouble).itemsize > 8:
        with pytest.raises(TypeError, match="do not support"):
            column_moments(np.ones((2, 2), dtype=np.longdouble))
    with pytest.raises(ValueError, match="merge moments of 2 and 3 columns"):
        column_moments(np.ones((2, 2))).merge(column_moments(np.ones((2, 3))))


@pytest.mark.parametrize("dtype", ["float64", "float16"])
def test_non_native_byte_order_is_read_by_value(dtype):
    values = _values(dtype, (31, 4), seed=9)
    swapped = values.astype(values.dtype.newbyteorder(">"))
    assert not swapped.dtype.isnative
    expected = column_moments(values)
    actual = column_moments(swapped)
    np.testing.assert_array_equal(actual.total, expected.total)
    np.testing.assert_array_equal(actual.m2, expected.m2)


@pytest.mark.parametrize("dtype", ["float32", "bool"])
def test_column_moments_avoid_matrix_sized_temporaries(dtype):
    values = (np.arange(1024 * 512).reshape(1024, 512) % 251).astype(dtype)
    reference = values.astype(np.float64)
    column_moments(values[:2])
    tracemalloc.start()
    try:
        moments = column_moments(values)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # Three float64 arrays of 512 columns; the block holds at least 512 KiB.
    assert peak < 64 * 1024
    np.testing.assert_allclose(moments.total, reference.sum(axis=0))
    np.testing.assert_allclose(moments.variance(), reference.var(axis=0))


@njit(cache=False)
def _sparse_moments(values: np.ndarray) -> tuple[float, float]:
    stored = 0
    mean = 0.0
    m2 = 0.0
    total = 0.0
    for value in values:
        if value == 0:
            continue
        total += value
        stored += 1
        mean, m2 = welford_add(stored, mean, m2, value)
    return total, add_implicit_zeros(m2, mean, stored, values.shape[0])


@pytest.mark.parametrize(
    "values",
    [
        [0.0, 0.0, 0.0],
        [0.0, 2.5, 0.0, 2.5],
        [-1.0, 0.0, -2.0],
        [3.0, 1e9, 0.0, 1e9 + 1, 0.0],
        [3000 / 7] * 7,
    ],
)
def test_sparse_kernels_add_implicit_zeros_exactly(values):
    array = np.asarray(values)
    total, m2 = _sparse_moments(array)

    assert total == reduce(operator.add, values, 0.0)
    if len(set(values)) == 1:
        # NumPy's two-pass m2 of these equal values is about 2e-26, from the
        # rounding of their mean; the running mean of equal values is exact.
        assert m2 == 0.0
    else:
        np.testing.assert_allclose(m2, _two_pass_m2(array[:, None])[0], rtol=1e-13)


def test_merging_with_zeros_matches_add_implicit_zeros():
    rng = np.random.default_rng(4)
    stored = rng.random((6, 3)) + 1
    merged = column_moments(stored).merge(column_moments(np.zeros((4, 3))))
    for column in range(3):
        moments = column_moments(stored[:, [column]])
        expected = add_implicit_zeros(
            float(moments.m2[0]), float(moments.mean[0]), 6, 10
        )
        assert merged.m2[column] == expected
