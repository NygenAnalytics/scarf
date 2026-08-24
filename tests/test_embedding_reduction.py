import numpy as np
import pytest
from sklearn.decomposition import IncrementalPCA, TruncatedSVD

import scarf.embeddings.reduction as reduction_module
from scarf.embeddings.reduction import (
    _gram_pca_dispatch,
    _mutable_fit_block,
    _streaming_lsi_resident_bytes,
    fit_incremental_pca,
    fit_lsi,
)
from scarf.matrix import ChunkedArray
from scarf.storage.budget import ResourceBudget


class _ReadTrackingArray(ChunkedArray):
    def __init__(self, values: np.ndarray, block_size: int) -> None:
        super().__init__(values, block_size=block_size, nthreads=1, is_numpy=True)
        self.read_ranges: list[tuple[int, int]] = []

    def _materialize_range(self, start: int, end: int) -> np.ndarray:
        self.read_ranges.append((start, end))
        return super()._materialize_range(start, end)


class _ResidentTrackingArray(ChunkedArray):
    def __init__(
        self,
        values: np.ndarray,
        *,
        block_size: int,
        resources: ResourceBudget,
    ) -> None:
        super().__init__(
            values,
            block_size=block_size,
            nthreads=resources.workers,
            resources=resources,
            is_numpy=True,
        )
        self.resident_bytes: list[int] = []

    def _stream_blocks(
        self,
        *,
        nthreads,
        msg,
        prefetch,
        row_mask,
        resident_bytes=0,
    ):
        self.resident_bytes.append(resident_bytes)
        yield from super()._stream_blocks(
            nthreads=nthreads,
            msg=msg,
            prefetch=prefetch,
            row_mask=row_mask,
            resident_bytes=resident_bytes,
        )


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


def test_gram_pca_matches_full_svd_and_preserves_model_contract():
    values = np.random.default_rng(11).normal(size=(40, 7))
    data = ChunkedArray.from_numpy(values, block_size=10)
    selected = np.ones(values.shape[0], dtype=bool)

    loadings, model = fit_incremental_pca(
        data,
        dims=3,
        batch_size=10,
        use_for_pca=selected,
        scale=None,
        nthreads=1,
    )

    centered = values - values.mean(axis=0)
    _, singular_values, expected_components = np.linalg.svd(
        centered,
        full_matrices=False,
    )
    expected_components = expected_components[:4]
    expected_variance = np.square(singular_values[:4]) / (len(values) - 1)
    total_variance = values.var(axis=0, ddof=1).sum()

    assert loadings.shape == (values.shape[1], 3)
    assert model.components_.shape == (4, values.shape[1])
    assert model.explained_variance_ratio_.shape == (4,)
    np.testing.assert_allclose(loadings, model.components_[:-1].T)
    np.testing.assert_allclose(
        np.abs(model.components_ @ expected_components.T),
        np.eye(4),
        atol=1e-10,
    )
    np.testing.assert_allclose(model.explained_variance_, expected_variance)
    np.testing.assert_allclose(
        model.explained_variance_ratio_,
        expected_variance / total_variance,
    )
    np.testing.assert_allclose(model.singular_values_, singular_values[:4])
    np.testing.assert_allclose(model.mean_, values.mean(axis=0))
    assert model.n_components_ == 4
    assert model.n_features_in_ == values.shape[1]
    assert model.n_samples_seen_ == values.shape[0]


def test_gram_pca_matches_selected_scaled_rows():
    values = np.random.default_rng(19).normal(size=(36, 5))
    data = ChunkedArray.from_numpy(values, block_size=6)
    selected = np.zeros(values.shape[0], dtype=bool)
    selected[::2] = True
    offset = np.linspace(-0.5, 0.5, values.shape[1])
    divisor = np.linspace(1.0, 2.0, values.shape[1])

    def scale(block: np.ndarray) -> np.ndarray:
        return (block - offset) / divisor

    loadings, model = fit_incremental_pca(
        data,
        dims=2,
        batch_size=6,
        use_for_pca=selected,
        scale=scale,
        nthreads=1,
    )

    expected_values = scale(values[selected])
    centered = expected_values - expected_values.mean(axis=0)
    _, _, expected_components = np.linalg.svd(centered, full_matrices=False)
    expected_components = expected_components[:3]

    np.testing.assert_allclose(
        np.abs(model.components_ @ expected_components.T),
        np.eye(3),
        atol=1e-10,
    )
    np.testing.assert_allclose(model.mean_, expected_values.mean(axis=0))
    np.testing.assert_allclose(loadings, model.components_[:-1].T)


def test_gram_pca_dispatch_covers_both_fallback_reasons():
    assert _gram_pca_dispatch(100, 100, 2) == (True, None)

    use_gram, reason = _gram_pca_dispatch(101, 100, 2)
    assert not use_gram
    assert reason is not None and "exceed 100 rows per block" in reason

    use_gram, reason = _gram_pca_dispatch(4097, 5000, 2)
    assert not use_gram
    assert reason is not None and "4096-feature limit" in reason

    use_gram, reason = _gram_pca_dispatch(100, 100, 1)
    assert not use_gram
    assert reason == "the input has only one row block"


def test_pca_logs_selected_solver(monkeypatch):
    messages: list[str] = []

    class RecordingLogger:
        @staticmethod
        def debug(message: str) -> None:
            messages.append(message)

    monkeypatch.setattr(reduction_module, "logger", RecordingLogger())
    selected = np.ones(24, dtype=bool)
    fit_incremental_pca(
        ChunkedArray.from_numpy(
            np.random.default_rng(21).normal(size=(24, 6)),
            block_size=8,
        ),
        dims=2,
        batch_size=8,
        use_for_pca=selected,
        scale=None,
        nthreads=1,
    )
    assert "Fitting PCA with the Gram covariance solver" in messages.pop()

    fit_incremental_pca(
        ChunkedArray.from_numpy(
            np.random.default_rng(22).normal(size=(24, 10)),
            block_size=8,
        ),
        dims=2,
        batch_size=8,
        use_for_pca=selected,
        scale=None,
        nthreads=1,
    )
    captured = messages.pop()
    assert "Fitting PCA with the IncrementalPCA solver" in captured
    assert "10 features exceed 8 rows per block" in captured


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
    values = np.random.default_rng(8).normal(size=(24, 10))
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


def test_streaming_lsi_matches_exact_singular_subspace_and_is_deterministic():
    rng = np.random.default_rng(19)
    left, _ = np.linalg.qr(rng.normal(size=(60, 5)))
    right, _ = np.linalg.qr(rng.normal(size=(20, 5)))
    values = left @ np.diag([20.0, 12.0, 7.0, 3.0, 1.0]) @ right.T
    values += 0.05 * rng.normal(size=values.shape)
    data = ChunkedArray.from_numpy(values, block_size=7)
    params = {
        "solver": "streaming",
        "n_iter": 3,
        "n_oversamples": 4,
    }

    observed = fit_lsi(
        data,
        dims=4,
        skip_first=False,
        params=params,
        random_state=11,
        nthreads=1,
    )
    repeated = fit_lsi(
        ChunkedArray.from_numpy(values, block_size=9),
        dims=4,
        skip_first=False,
        params=params.copy(),
        random_state=11,
        nthreads=1,
    )
    _, _, right_vectors = np.linalg.svd(values, full_matrices=False)
    expected = right_vectors[:4].T

    np.testing.assert_allclose(
        observed @ observed.T,
        expected @ expected.T,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(observed, repeated, rtol=1e-10, atol=1e-10)
    assert data.numblocks[0] > 1


def test_streaming_lsi_never_stacks_all_input_blocks(monkeypatch):
    values = np.random.default_rng(23).normal(size=(30, 12))

    def fail_vstack(*_args, **_kwargs):
        pytest.fail("Streaming LSI must not stack the full input matrix")

    monkeypatch.setattr(np, "vstack", fail_vstack)
    loadings = fit_lsi(
        ChunkedArray.from_numpy(values, block_size=4),
        dims=3,
        skip_first=True,
        params={"n_iter": 1},
        random_state=3,
        nthreads=1,
    )

    assert loadings.shape == (values.shape[1], 3)


def test_streaming_lsi_reserves_accumulators_and_block_temporaries():
    values = np.random.default_rng(31).normal(size=(30, 12))
    resources = ResourceBudget(memoryBytes=10_000, workers=4)
    data = _ResidentTrackingArray(
        values,
        block_size=4,
        resources=resources,
    )

    fit_lsi(
        data,
        dims=3,
        skip_first=True,
        params={
            "solver": "streaming",
            "n_iter": 1,
            "n_oversamples": 2,
        },
        random_state=5,
        nthreads=4,
    )

    expected = _streaming_lsi_resident_bytes(data, width=6)
    assert data.resident_bytes == [expected, expected, expected]
    assert expected + data._block_owned_bytes() <= resources.memoryBytes


def test_materialized_lsi_preserves_sklearn_compatibility():
    values = np.random.default_rng(29).normal(size=(28, 9))
    expected_model = TruncatedSVD(
        n_components=4,
        n_iter=3,
        n_oversamples=2,
        random_state=7,
    )
    expected_model.fit(values)

    observed = fit_lsi(
        ChunkedArray.from_numpy(values, block_size=5),
        dims=3,
        skip_first=True,
        params={
            "solver": "materialized",
            "n_iter": 3,
            "n_oversamples": 2,
        },
        random_state=7,
        nthreads=1,
    )

    np.testing.assert_allclose(
        observed,
        expected_model.components_.T[:, 1:],
        rtol=1e-12,
        atol=1e-12,
    )
