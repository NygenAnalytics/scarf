import numpy as np
import pytest
from sklearn.decomposition import IncrementalPCA

from scarf.embeddings.reduction import _mutable_fit_block, fit_incremental_pca, fit_lsi
from scarf.matrix import ChunkedArray


class _ReadTrackingArray(ChunkedArray):
    def __init__(self, values: np.ndarray, block_size: int) -> None:
        super().__init__(values, block_size=block_size, nthreads=1, is_numpy=True)
        self.read_ranges: list[tuple[int, int]] = []

    def _materialize_range(self, start: int, end: int) -> np.ndarray:
        self.read_ranges.append((start, end))
        return super()._materialize_range(start, end)


def test_mutable_fit_block_reuses_owned_values_and_copies_shared_views():
    owned = np.arange(24, dtype=np.float64).reshape(6, 4).copy()
    assert _mutable_fit_block(owned) is owned

    shared = owned[:, :2]
    copied = _mutable_fit_block(shared)
    assert copied is not shared
    assert copied.flags.owndata
    assert copied.flags.c_contiguous
    np.testing.assert_array_equal(copied, shared)


def test_incremental_pca_is_deterministic_across_chunked_input():
    values = np.random.default_rng(42).normal(size=(24, 6))
    original_values = values.copy()
    data = ChunkedArray.from_numpy(values, block_size=8)
    selected = np.ones(values.shape[0], dtype=bool)

    first, first_model = fit_incremental_pca(
        data,
        dims=3,
        batch_size=8,
        use_for_pca=selected,
        scale=None,
        nthreads=1,
    )
    second, second_model = fit_incremental_pca(
        data,
        dims=3,
        batch_size=8,
        use_for_pca=selected,
        scale=None,
        nthreads=1,
    )

    assert first.shape == (6, 3)
    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(
        first_model.explained_variance_ratio_,
        second_model.explained_variance_ratio_,
    )
    np.testing.assert_array_equal(values, original_values)


def test_incremental_pca_skips_blocks_without_selected_rows():
    values = np.random.default_rng(3).normal(size=(24, 6))
    data = _ReadTrackingArray(values, block_size=6)
    selected = np.zeros(values.shape[0], dtype=bool)
    selected[:6] = True
    selected[-6:] = True

    fit_incremental_pca(
        data,
        dims=2,
        batch_size=6,
        use_for_pca=selected,
        scale=None,
        nthreads=1,
    )

    assert data.read_ranges == [(0, 6), (18, 24)]


def test_incremental_pca_propagates_final_fit_failure(monkeypatch):
    values = np.random.default_rng(8).normal(size=(24, 6))
    data = ChunkedArray.from_numpy(values, block_size=8)
    selected = np.ones(values.shape[0], dtype=bool)
    original_partial_fit = IncrementalPCA.partial_fit
    calls = 0

    def fail_final_fit(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise np.linalg.LinAlgError("final fit failed")
        return original_partial_fit(self, *args, **kwargs)

    monkeypatch.setattr(IncrementalPCA, "partial_fit", fail_final_fit)

    with pytest.raises(np.linalg.LinAlgError, match="final fit failed"):
        fit_incremental_pca(
            data,
            dims=2,
            batch_size=8,
            use_for_pca=selected,
            scale=None,
            nthreads=1,
        )


def test_lsi_returns_requested_components_and_mutates_reserved_params():
    values = np.random.default_rng(7).uniform(size=(18, 5))
    data = ChunkedArray.from_numpy(values, block_size=6)
    params = {"n_iter": 4, "n_components": 99, "random_state": 99}

    loadings = fit_lsi(
        data,
        dims=2,
        skip_first=True,
        params=params,
        random_state=3,
        nthreads=1,
    )

    assert loadings.shape == (5, 2)
    assert params == {"n_iter": 4}
    assert np.all(np.isfinite(loadings))
