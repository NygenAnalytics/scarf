"""Tests for the ChunkedArray abstraction and the public rawData/normed API.

These verify that ChunkedArray reproduces NumPy semantics for the operations
Scarf relies on, and that the documented datastore access patterns from the
vignettes keep working.
"""

import numpy as np
import pytest
import zarr

from scarf.matrix import ChunkedArray


@pytest.fixture
def backed_pair(tmp_path):
    rng = np.random.default_rng(0)
    n, m = 230, 70
    dense = rng.integers(0, 6, size=(n, m)).astype(np.uint32)
    root = zarr.open_group(str(tmp_path / "ca.zarr"), mode="w")
    z = root.create_array("counts", shape=(n, m), chunks=(64, 32), dtype="uint32")
    z[:, :] = dense
    return ChunkedArray(root["counts"], nthreads=4), dense


class TestChunkedArrayParity:
    def test_shape_and_blocks(self, backed_pair):
        ca, dense = backed_pair
        from scarf.storage.layout import array_shard_rows

        assert ca.shape == dense.shape
        stream_rows = array_shard_rows(ca._backing)
        assert ca.numblocks[0] == int(np.ceil(dense.shape[0] / stream_rows))
        blocks = list(ca.stream_blocks(nthreads=1))
        assert len(blocks) == ca.numblocks[0]
        assert np.array_equal(np.vstack(blocks), dense)

    def test_compute(self, backed_pair):
        ca, dense = backed_pair
        assert np.array_equal(ca.compute(), dense)

    @pytest.mark.parametrize("axis", [0, 1])
    def test_reductions(self, backed_pair, axis):
        ca, dense = backed_pair
        assert np.allclose(ca.sum(axis=axis).compute(), dense.sum(axis))
        assert np.allclose(ca.mean(axis=axis).compute(), dense.mean(axis))
        assert np.allclose(ca.var(axis=axis).compute(), dense.var(axis))
        assert np.allclose(ca.sum(axis=axis - 2).compute(), dense.sum(axis))

    def test_count_nonzero_and_argmax(self, backed_pair):
        ca, dense = backed_pair
        assert np.array_equal(
            np.asarray(ca.count_nonzero(axis=1)), np.count_nonzero(dense, axis=1)
        )
        assert np.array_equal(np.asarray(ca.argmax(axis=1)), dense.argmax(1))
        with pytest.raises(NotImplementedError, match="argmax\\(axis=0\\)"):
            ca.argmax(axis=0).compute()

    def test_mean_and_std_matches_numpy_and_rejects_other_axes(self, backed_pair):
        ca, dense = backed_pair
        mean, std = ca.mean_and_std(axis=0)
        np.testing.assert_allclose(mean, dense.mean(axis=0))
        np.testing.assert_allclose(std, dense.std(axis=0))
        with pytest.raises(
            NotImplementedError, match="mean_and_std only supports axis=0"
        ):
            ca.mean_and_std(axis=1)

    def test_boolean_comparison_reduction(self, backed_pair):
        ca, dense = backed_pair
        assert (ca > 0).dtype == np.dtype(bool)
        assert np.array_equal((ca > 0).sum(axis=0).compute(), (dense > 0).sum(0))

    def test_scalar_reductions_weight_the_short_final_block(self, backed_pair):
        ca, dense = backed_pair
        assert np.allclose(ca.mean().compute(), dense.mean())
        assert np.allclose(ca.var().compute(), dense.var())
        assert ca.count_nonzero().compute() == np.count_nonzero(dense)

    def test_fancy_subset(self, backed_pair):
        ca, dense = backed_pair
        rng = np.random.default_rng(1)
        fidx = np.sort(rng.choice(dense.shape[1], 25, replace=False))
        cidx = np.sort(rng.choice(dense.shape[0], 90, replace=False))
        sub = ca[:, fidx][cidx, :]
        ref = dense[:, fidx][cidx, :]
        assert sub.shape == ref.shape
        assert np.array_equal(sub.compute(), ref)
        assert np.allclose(sub.sum(axis=1).compute(), ref.sum(1))

    def test_lib_size_normalization(self, backed_pair):
        ca, dense = backed_pair
        sub = ca[:, np.arange(dense.shape[1])]
        scalar = dense.sum(1).astype(float)
        scalar[scalar == 0] = 1
        normed = 1e4 * sub / scalar.reshape(-1, 1)
        ref = 1e4 * dense / scalar.reshape(-1, 1)
        assert np.allclose(normed.compute(), ref)
        assert np.allclose(np.log1p(normed).compute(), np.log1p(ref))

    def test_clr_with_axis0_inside_expression(self, backed_pair):
        ca, dense = backed_pair
        rng = np.random.default_rng(2)
        fidx = np.sort(rng.choice(dense.shape[1], 20, replace=False))
        cidx = np.sort(rng.choice(dense.shape[0], 80, replace=False))
        sub = ca[:, fidx][cidx, :]
        ref = dense[:, fidx][cidx, :]
        f = np.exp(np.log1p(sub).sum(axis=0) / len(sub))
        clr = np.log1p(sub / f)
        ref_f = np.exp(np.log1p(ref).sum(0) / ref.shape[0])
        assert np.allclose(clr.compute(), np.log1p(ref / ref_f))

    def test_column_subset_after_ops(self, backed_pair):
        ca, dense = backed_pair
        scalar = dense.sum(1).astype(float)
        scalar[scalar == 0] = 1
        normed = np.log1p(1e4 * ca / scalar.reshape(-1, 1))
        ref = np.log1p(1e4 * dense / scalar.reshape(-1, 1))
        cols = np.array([3, 10, 25, 40])
        assert np.allclose(normed[:, cols].compute(), ref[:, cols])

    def test_column_subset_after_two_dimensional_row_broadcast(self, backed_pair):
        ca, dense = backed_pair
        row_values = np.linspace(1.0, 2.0, dense.shape[1]).reshape(1, -1)
        transformed = ca * row_values
        cols = np.array([3, 10, 25, 40])

        np.testing.assert_allclose(
            transformed[:, cols].compute(),
            (dense * row_values)[:, cols],
        )

    def test_from_numpy(self, backed_pair):
        _, dense = backed_pair
        ca = ChunkedArray.from_numpy(dense.astype(float), block_size=50, nthreads=2)
        assert np.array_equal(ca.compute(), dense.astype(float))
        assert np.allclose(ca.sum(axis=0).compute(), dense.sum(0))


class TestNumpySemantics:
    def test_one_dimensional_operands_align_with_columns_on_square_matrices(self):
        values = np.arange(1, 10, dtype=np.float64).reshape(3, 3)
        ca = ChunkedArray.from_numpy(values, block_size=2)
        column_means = values.mean(axis=0)

        np.testing.assert_allclose((ca - column_means).compute(), values - column_means)
        np.testing.assert_allclose(
            (ca / ca.sum(axis=0)).compute(), values / values.sum(axis=0)
        )
        f = np.exp(np.log1p(ca).sum(axis=0) / len(ca))
        np.testing.assert_allclose(
            np.log1p(ca / f).compute(),
            np.log1p(values / np.exp(np.log1p(values).mean(axis=0))),
        )
        row_totals = values.sum(axis=1).reshape(-1, 1)
        np.testing.assert_allclose(
            (ca[np.array([2, 0]), :] / row_totals[[2, 0]]).compute(),
            values[[2, 0]] / row_totals[[2, 0]],
        )

    def test_operands_that_numpy_cannot_broadcast_are_rejected(self):
        ca = ChunkedArray.from_numpy(np.ones((2, 3)))
        with pytest.raises(ValueError, match=r"scale rows with a \(2, 1\) array"):
            ca * np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="does not broadcast"):
            ca + np.ones((3, 3))
        np.testing.assert_allclose(
            (ca * np.array([2.0])).compute(), np.full((2, 3), 2.0)
        )

    def test_reductions_of_zero_rows_match_numpy(self):
        values = np.arange(12, dtype=np.float64).reshape(4, 3)
        empty = ChunkedArray.from_numpy(values, block_size=2)[
            np.array([], dtype=np.int64), :
        ]
        reference = values[:0]
        with np.errstate(all="ignore"), pytest.warns(RuntimeWarning):
            expected_mean = reference.mean(axis=0)
        for axis in (0, 1):
            np.testing.assert_array_equal(
                empty.sum(axis=axis).compute(), reference.sum(axis=axis)
            )
            np.testing.assert_array_equal(
                empty.count_nonzero(axis=axis).compute(),
                np.count_nonzero(reference, axis=axis),
            )
        np.testing.assert_array_equal(empty.mean(axis=0).compute(), expected_mean)
        assert empty.mean(axis=1).compute().shape == (0,)
        assert empty.var(axis=1).compute().shape == (0,)
        assert empty.argmax(axis=1).compute().shape == (0,)
        assert empty.sum().compute() == 0
        mean, std = empty.mean_and_std()
        assert mean.shape == std.shape == (3,)
        assert np.isnan(mean).all() and np.isnan(std).all()

    def test_sum_accumulates_in_a_requested_dtype(self):
        # Above 2**24 float32 holds only even integers, so its own sums round
        # the odd totals that a float64 accumulator keeps.
        values = np.array([[2.0**24, 1.0], [1.0, 1.0], [1.0, 2.0]], dtype=np.float32)
        ca = ChunkedArray.from_numpy(values, block_size=1)
        for axis in (None, 0, 1):
            actual = ca.sum(axis=axis, dtype=np.float64).compute()
            assert actual.dtype == np.float64
            np.testing.assert_array_equal(
                actual, values.sum(axis=axis, dtype=np.float64)
            )
        assert ca.sum(axis=0).compute().dtype == np.float32
        empty = ca[np.array([], dtype=np.int64), :]
        for axis in (0, 1):
            expected = values[:0].sum(axis=axis, dtype=np.int64)
            actual = empty.sum(axis=axis, dtype=np.int64).compute()
            assert actual.dtype == expected.dtype
            np.testing.assert_array_equal(actual, expected)

    def test_reduction_axes_are_validated(self):
        values = np.arange(6, dtype=np.float64).reshape(2, 3)
        ca = ChunkedArray.from_numpy(values)
        np.testing.assert_allclose(ca.sum(axis=-1).compute(), values.sum(axis=-1))
        np.testing.assert_allclose(ca.mean(axis=-2).compute(), values.mean(axis=-2))
        with pytest.raises(np.exceptions.AxisError):
            ca.sum(axis=2)
        with pytest.raises(TypeError, match="axis"):
            ca.sum(axis=True)
        with pytest.raises(ValueError, match="axis=None"):
            ca.argmax().compute()

    def test_scalar_and_non_integer_keys_are_rejected(self):
        values = np.arange(6, dtype=np.float64).reshape(2, 3)
        ca = ChunkedArray.from_numpy(values)
        for key in (1, (slice(None), 1), np.int64(0), [[0]], np.array([0.5])):
            with pytest.raises(IndexError):
                ca[key]
        np.testing.assert_array_equal(ca[[1]].compute(), values[[1]])
        np.testing.assert_array_equal(ca[1:2, [2, 0]].compute(), values[1:2, [2, 0]])
        np.testing.assert_array_equal(ca[[]].compute(), values[[]])

    def test_unsupported_ufunc_keywords_are_rejected(self):
        ca = ChunkedArray.from_numpy(np.ones((2, 2)))
        with pytest.raises(TypeError):
            np.log1p(ca, where=np.zeros((2, 2), dtype=bool))
        with pytest.raises(TypeError):
            np.add(ca, 1, casting="unsafe")
        assert np.log1p(ca, dtype=np.float32).compute().dtype == np.float32

    def test_mixed_reduction_operands_stay_lazy(self, monkeypatch):
        values = np.arange(1, 7, dtype=np.float64).reshape(2, 3)
        ca = ChunkedArray.from_numpy(values)
        column_sums = ca.sum(axis=0)
        column_sums.compute()
        computed = []
        original = ChunkedArray.compute

        def compute(self, *args, **kwargs):
            computed.append(self)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(ChunkedArray, "compute", compute)
        scaled = column_sums * ca
        assert isinstance(scaled, ChunkedArray)
        assert computed == []
        np.testing.assert_allclose(scaled.compute(), values.sum(axis=0) * values)
        assert isinstance(column_sums > ca, ChunkedArray)
        with pytest.raises(TypeError, match="two ChunkedArrays"):
            ca + ca

    def test_array_conversion_honours_copy(self):
        values = np.arange(6, dtype=np.float64).reshape(2, 3)
        ca = ChunkedArray.from_numpy(values)
        np.testing.assert_array_equal(np.asarray(ca, copy=True), values)
        assert np.asarray(ca, dtype=np.float32).dtype == np.float32
        with pytest.raises(ValueError, match="new array"):
            np.asarray(ca, copy=False)
        reduction = ca.sum(axis=0)
        copied = np.asarray(reduction, copy=True)
        copied[0] = -1
        assert reduction.compute()[0] == values[:, 0].sum()
        converted = np.asarray(reduction, dtype=np.float32)
        assert converted.dtype == np.float32
        np.testing.assert_array_equal(converted, values.sum(axis=0))
        with pytest.raises(ValueError, match="avoid copy"):
            np.asarray(reduction, dtype=np.float32, copy=False)
        assert np.shares_memory(np.asarray(reduction), reduction.compute())


class TestPublicApiCompat:
    """Mirror the rawData/normed usage documented in the vignettes."""

    def test_rawdata_is_chunked(self, datastore):
        raw = datastore.RNA.rawData
        assert isinstance(raw, ChunkedArray)
        assert len(raw.chunksize) == 2
        assert raw.shape[0] == datastore.RNA.cells.N

    def test_rawdata_mean_axis0_compute_reshape(self, datastore):
        # Pattern from the MNIST vignette.
        fidx = np.arange(20)
        cidx = datastore.RNA.cells.active_index("I")[:50]
        out = (
            datastore.RNA.rawData[:, fidx][cidx, :]
            .mean(axis=0)
            .compute()
            .reshape(1, -1)
        )
        assert out.shape == (1, 20)

    def test_normed_mean_axis1_compute(self, datastore):
        # Pattern from the pseudotime dynamics vignette.
        vals = datastore.RNA.normed().mean(axis=1).compute()
        assert vals.shape[0] == datastore.RNA.cells.active_index("I").shape[0]
        assert np.all(np.isfinite(vals))

    def test_custom_normmethod_numpy_semantics(self, datastore):
        # User-overridable normMethod must accept NumPy-like array semantics.
        def custom(_, counts):
            lib = counts.sum(axis=1).reshape(-1, 1)
            return np.log2(counts / lib * 1000 + 1)

        assay = datastore.RNA
        original = assay.normMethod
        try:
            assay.normMethod = custom
            out = assay.normed().compute()
            assert np.all(np.isfinite(out))
        finally:
            assay.normMethod = original
