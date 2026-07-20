import numpy as np

from scarf.embeddings.reduction import fit_incremental_pca, fit_lsi
from scarf.matrix import ChunkedArray


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
