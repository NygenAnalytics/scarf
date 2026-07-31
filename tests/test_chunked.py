"""Tests for the ChunkedArray abstraction and the public rawData/normed API.

These verify that ChunkedArray reproduces NumPy semantics for the operations
Scarf relies on, and that the documented datastore access patterns from the
vignettes keep working after Dask was removed.
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
        assert sum(b.shape[0] for b in ca.blocks) == dense.shape[0]
        recon = np.vstack([b.compute() for b in ca.blocks])
        assert np.array_equal(recon, dense)

    def test_compute(self, backed_pair):
        ca, dense = backed_pair
        assert np.array_equal(ca.compute(), dense)

    @pytest.mark.parametrize("axis", [0, 1])
    def test_reductions(self, backed_pair, axis):
        ca, dense = backed_pair
        assert np.allclose(ca.sum(axis=axis).compute(), dense.sum(axis))
        assert np.allclose(ca.mean(axis=axis).compute(), dense.mean(axis))
        assert np.allclose(ca.var(axis=axis).compute(), dense.var(axis))
        assert np.allclose(ca.std(axis=axis).compute(), dense.std(axis))

    def test_count_nonzero_and_argmax(self, backed_pair):
        ca, dense = backed_pair
        assert np.array_equal(
            np.asarray(ca.count_nonzero(axis=1)), np.count_nonzero(dense, axis=1)
        )
        assert np.array_equal(np.asarray(ca.argmax(axis=1)), dense.argmax(1))

    def test_boolean_comparison_reduction(self, backed_pair):
        ca, dense = backed_pair
        assert (ca > 0).dtype == np.dtype(bool)
        assert np.array_equal((ca > 0).sum(axis=0).compute(), (dense > 0).sum(0))

    def test_scalar_reductions_weight_the_short_final_block(self, backed_pair):
        ca, dense = backed_pair
        assert np.allclose(ca.mean().compute(), dense.mean())
        assert np.allclose(ca.var().compute(), dense.var())
        assert np.allclose(ca.std().compute(), dense.std())
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

    def test_dot(self, backed_pair):
        ca, dense = backed_pair
        rng = np.random.default_rng(3)
        loadings = rng.standard_normal((dense.shape[1], 4))
        assert np.allclose(ca.dot(loadings).compute(), dense @ loadings)

    def test_from_numpy(self, backed_pair):
        _, dense = backed_pair
        ca = ChunkedArray.from_numpy(dense.astype(float), block_size=50, nthreads=2)
        assert np.array_equal(ca.compute(), dense.astype(float))
        assert np.allclose(ca.sum(axis=0).compute(), dense.sum(0))


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
