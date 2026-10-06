import tracemalloc
import weakref

import numpy as np
import pandas as pd
import pytest

from scarf.embeddings.harmony import HarmonyResult
from scarf.matrix import ChunkedArray
from scarf.neighbors.stages import (
    AnnIndexStage,
    BatchCorrectionStage,
    ChunkedCoordinateStream,
    KMeansInitializationStage,
    NeighborQueryStage,
    ReductionTransform,
    kmeans_fit_memory,
)


class _CountingChunkedArray(ChunkedArray):
    def __init__(self, values: np.ndarray, block_size: int) -> None:
        super().__init__(
            values,
            block_size=block_size,
            nthreads=1,
            is_numpy=True,
        )
        self.read_count = 0

    def _materialize_range(self, start: int, end: int) -> np.ndarray:
        self.read_count += 1
        return super()._materialize_range(start, end)


def _custom_inputs() -> tuple[np.ndarray, np.ndarray]:
    data = np.arange(32, dtype=np.float64).reshape(8, 4) / 10
    loadings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, -0.5],
            [-0.25, 0.75],
        ]
    )
    return data, loadings


def _coordinate_stream(
    coordinates: np.ndarray,
    block_size: int,
) -> tuple[ChunkedCoordinateStream, _CountingChunkedArray]:
    data = _CountingChunkedArray(coordinates, block_size=block_size)
    return ChunkedCoordinateStream(data, 1), data


def test_reduction_transform_with_loadings_reads_no_cells() -> None:
    values, loadings = _custom_inputs()
    data = _CountingChunkedArray(values, block_size=3)
    reduction = ReductionTransform(
        data=data,
        method="custom",
        dims=2,
        loadings=loadings,
        use_for_pca=np.ones(values.shape[0], dtype=bool),
        mu=np.zeros(values.shape[1]),
        sigma=np.ones(values.shape[1]),
        batch_size=3,
        nthreads=1,
        rand_state=4466,
        disable_scaling=True,
        lsi_skip_first=False,
        lsi_params={},
    )

    np.testing.assert_allclose(
        reduction.transform(values[:3]), values[:3].dot(loadings)
    )
    assert data.read_count == 0


@pytest.mark.parametrize("disable_scaling", [False, True])
@pytest.mark.parametrize("fit_subset", [False, True])
@pytest.mark.parametrize("block_size", [10, 30])
def test_pca_projection_preserves_fitted_center_on_reload(
    disable_scaling: bool, fit_subset: bool, block_size: int
) -> None:
    values = np.random.default_rng(31).normal(size=(30, 5))
    values[15:] += np.array([4, 8, 2, 10, 6])
    use_for_pca = np.arange(30) < (15 if fit_subset else 30)
    arguments = dict(
        data=ChunkedArray.from_numpy(values, block_size=block_size, nthreads=1),
        method="pca",
        dims=2,
        use_for_pca=use_for_pca,
        mu=values.mean(axis=0),
        sigma=values.std(axis=0),
        batch_size=block_size,
        nthreads=1,
        rand_state=4466,
        disable_scaling=disable_scaling,
        lsi_skip_first=False,
        lsi_params={},
    )
    fitted = ReductionTransform(loadings=None, **arguments)
    fit_values = values if disable_scaling else fitted.transform_z(values)
    expected = (fit_values - fit_values[use_for_pca].mean(axis=0)) @ fitted.loadings

    np.testing.assert_allclose(fitted.transform(values), expected, atol=1e-12)
    np.testing.assert_allclose(
        fitted.transform(values)[use_for_pca].mean(axis=0), 0, atol=1e-12
    )
    reloaded = ReductionTransform(
        loadings=fitted.loadings, center=fitted.center, **arguments
    )
    np.testing.assert_allclose(reloaded.transform(values), expected, atol=1e-12)


@pytest.mark.parametrize(
    "center", [None, np.zeros(3), np.full(4, np.nan), np.full(4, np.inf)]
)
def test_reloading_pca_loadings_requires_valid_center(center) -> None:
    values, loadings = _custom_inputs()
    with pytest.raises(ValueError, match="PCA.*center"):
        ReductionTransform(
            data=ChunkedArray.from_numpy(values, block_size=4, nthreads=1),
            method="pca",
            dims=2,
            loadings=loadings,
            center=center,
            use_for_pca=np.ones(values.shape[0], dtype=bool),
            mu=values.mean(axis=0),
            sigma=values.std(axis=0),
            batch_size=4,
            nthreads=1,
            rand_state=4466,
            disable_scaling=False,
            lsi_skip_first=False,
            lsi_params={},
        )


@pytest.mark.parametrize("disable_scaling", [False, True])
def test_pca_without_reduction_preserves_features(disable_scaling):
    values, _ = _custom_inputs()
    reduction = ReductionTransform(
        data=ChunkedArray.from_numpy(values, nthreads=1),
        method="pca",
        dims=0,
        loadings=None,
        use_for_pca=np.ones(len(values), dtype=bool),
        mu=values.mean(axis=0),
        sigma=values.std(axis=0),
        batch_size=4,
        nthreads=1,
        rand_state=4466,
        disable_scaling=disable_scaling,
        lsi_skip_first=False,
        lsi_params={},
    )
    expected = (
        values
        if disable_scaling
        else (values - values.mean(axis=0)) / values.std(axis=0)
    )
    np.testing.assert_allclose(reduction.transform(values), expected)
    assert reduction.center is None


def test_lsi_persisted_loadings_are_not_sliced_again() -> None:
    values, loadings = _custom_inputs()
    reduction = ReductionTransform(
        data=ChunkedArray.from_numpy(values, block_size=4, nthreads=1),
        method="lsi",
        dims=2,
        loadings=loadings,
        use_for_pca=np.ones(values.shape[0], dtype=bool),
        mu=np.array([], dtype=np.float64),
        sigma=np.array([], dtype=np.float64),
        batch_size=4,
        nthreads=1,
        rand_state=4466,
        disable_scaling=True,
        lsi_skip_first=True,
        lsi_params={},
    )

    assert reduction.dims == 2
    np.testing.assert_allclose(reduction.transform(values), values.dot(loadings))


@pytest.mark.parametrize("solver", ["streaming", "materialized"])
@pytest.mark.parametrize("skip_first", [False, True])
def test_lsi_dims_are_final_output_dimensions(skip_first: bool, solver: str) -> None:
    from scarf.embeddings.reduction import fit_lsi

    # Full-rank values with distinct singular values, so each component is
    # unique up to sign.
    values = np.random.default_rng(12).normal(size=(8, 4))
    loadings = fit_lsi(
        ChunkedArray.from_numpy(values, block_size=3, nthreads=1),
        dims=2,
        skip_first=skip_first,
        params={"solver": solver},
        random_state=4466,
        nthreads=1,
    )

    assert loadings.shape == (values.shape[1], 2)
    # Uncentered LSI keeps the leading right singular vectors, after the
    # skipped first one.
    _, _, right_vectors = np.linalg.svd(values, full_matrices=False)
    expected = right_vectors[int(skip_first) : int(skip_first) + 2].T
    np.testing.assert_allclose(np.abs(loadings.T @ expected), np.eye(2), atol=1e-8)


def test_lazy_coordinate_stages_do_not_hide_a_cross_stage_cache() -> None:
    values, loadings = _custom_inputs()
    stream, data = _coordinate_stream(values.dot(loadings), block_size=3)

    index = AnnIndexStage.fit(
        coordinates=stream,
        metric="l2",
        dims=2,
        n_cells=values.shape[0],
        ef_construction=50,
        ef=50,
        m=16,
        rand_state=4466,
        nthreads=1,
    )
    initialization = KMeansInitializationStage.fit(
        stream=stream,
        n_rows=values.shape[0],
        batch_size=3,
        n_clusters=3,
        rand_state=4466,
        nthreads=1,
    )

    assert data.read_count == 12
    assert index.get_current_count() == values.shape[0]
    query = NeighborQueryStage(index, k=3, metric="l2")
    coordinates = values.dot(loadings)
    indices, distances, missed = query.query(
        coordinates, self_indices=np.arange(values.shape[0])
    )
    # Eight points are few enough for the index search to be exact. The points
    # are evenly spaced, so tied neighbors are compared by distance.
    exact = np.linalg.norm(coordinates[:, None] - coordinates[None], axis=2)
    np.fill_diagonal(exact, np.inf)
    assert missed == 0
    np.testing.assert_allclose(
        distances, np.take_along_axis(exact, indices, axis=1), rtol=1e-5
    )
    np.testing.assert_allclose(
        np.sort(distances, axis=1), np.sort(exact, axis=1)[:, :3], rtol=1e-5
    )
    # Without self indices each cell is its own nearest neighbor.
    with_self, with_self_distances = query.query(coordinates)
    np.testing.assert_array_equal(with_self[:, 0], np.arange(values.shape[0]))
    np.testing.assert_allclose(with_self_distances[:, 0], 0, atol=1e-6)
    assert initialization.model is not None
    assert initialization.labels.shape == (values.shape[0],)


def test_streamed_kmeans_keeps_centroids_for_groups_stored_last() -> None:
    rng = np.random.default_rng(0)
    # Cells stored grouped by sample, as when samples are concatenated.
    values = np.vstack(
        [rng.normal(loc=center, size=(700, 4)) for center in (0.0, 6.0, 12.0)]
    )
    stream, _data = _coordinate_stream(values, block_size=500)

    result = KMeansInitializationStage.fit(
        stream=stream,
        n_rows=values.shape[0],
        batch_size=500,
        n_clusters=3,
        rand_state=4466,
        nthreads=1,
        kmeans_sampling=0.2,
        kmeans_batch_size=256,
    )

    np.testing.assert_allclose(
        np.sort(result.model.cluster_centers_[:, 0]),
        [0.0, 6.0, 12.0],
        atol=0.2,
    )
    for group in range(3):
        group_labels = result.labels[group * 700 : (group + 1) * 700]
        assert np.unique(group_labels).size == 1


def test_custom_reduction_requires_loadings() -> None:
    values, _loadings = _custom_inputs()
    with pytest.raises(ValueError, match="Custom reduction requires loadings"):
        ReductionTransform(
            data=ChunkedArray.from_numpy(values, block_size=4, nthreads=1),
            method="custom",
            dims=2,
            loadings=None,
            use_for_pca=np.ones(values.shape[0], dtype=bool),
            mu=np.zeros(values.shape[1]),
            sigma=np.ones(values.shape[1]),
            batch_size=4,
            nthreads=1,
            rand_state=4466,
            disable_scaling=True,
            lsi_skip_first=False,
            lsi_params={},
        )


def test_neighbor_query_validates_and_converts_metric_distances() -> None:
    stage = NeighborQueryStage(index=None, k=2, metric="l2")
    distances = np.array([[4.0, -1e-7]], dtype=np.float32)
    np.testing.assert_allclose(
        stage._metric_distances(distances),
        [[2.0, 0.0]],
    )

    cosine = NeighborQueryStage(index=None, k=2, metric="cosine")
    np.testing.assert_allclose(
        cosine._metric_distances(np.array([[0.25, -1e-7]], dtype=np.float32)),
        [[0.25, 0.0]],
    )
    with pytest.raises(ValueError, match="negative"):
        cosine._metric_distances(np.array([[0.1, -0.01]], dtype=np.float32))
    with pytest.raises(ValueError, match="non-finite"):
        cosine._metric_distances(np.array([[0.1, np.nan]], dtype=np.float32))

    # Inner-product scores may be negative and are returned unchanged.
    inner_product = NeighborQueryStage(index=None, k=2, metric="ip")
    np.testing.assert_array_equal(
        inner_product._metric_distances(np.array([[0.5, -2.0]], dtype=np.float32)),
        [[0.5, -2.0]],
    )
    # Any other metric is a distance, so a negative value is an error.
    other = NeighborQueryStage(index=None, k=2, metric="manhattan")
    np.testing.assert_array_equal(
        other._metric_distances(np.array([[4.0, 0.0]], dtype=np.float32)),
        [[4.0, 0.0]],
    )
    with pytest.raises(ValueError, match="negative neighbor distances"):
        other._metric_distances(np.array([[4.0, -1e-9]], dtype=np.float32))


def test_kmeans_initialization_reads_each_block_once_per_pass() -> None:
    values, loadings = _custom_inputs()
    stream, data = _coordinate_stream(values.dot(loadings), block_size=3)

    result = KMeansInitializationStage.fit(
        stream=stream,
        n_rows=values.shape[0],
        batch_size=3,
        n_clusters=3,
        rand_state=4466,
        nthreads=1,
    )
    assert result.model is not None
    assert result.labels.shape == (values.shape[0],)
    assert data.read_count == 9


def test_kmeans_initialization_uses_true_minibatches_for_one_full_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values, loadings = _custom_inputs()
    stream, data = _coordinate_stream(
        values.dot(loadings),
        block_size=values.shape[0],
    )
    closed_progress: list[tuple[int, int]] = []

    class Progress:
        def __init__(self, total: int) -> None:
            self.total = total
            self.value = 0

        def update(self) -> None:
            self.value += 1

        def close(self) -> None:
            closed_progress.append((self.value, self.total))

    monkeypatch.setattr(
        "scarf.utils.progress.tqdmbar",
        lambda *args, total, **kwargs: Progress(total),
    )

    result = KMeansInitializationStage.fit(
        stream=stream,
        n_rows=values.shape[0],
        batch_size=values.shape[0],
        n_clusters=3,
        rand_state=4466,
        nthreads=1,
        kmeans_sampling=0.5,
        kmeans_batch_size=3,
    )

    assert result.model is not None
    assert result.model.n_init == 1
    assert result.model.init_size == 4
    assert result.model.batch_size == 3
    assert result.model.n_steps_ > 1
    assert result.labels.shape == (values.shape[0],)
    assert data.read_count == 1
    assert closed_progress == [(1, 1)]


def test_kmeans_streaming_samples_all_blocks_and_coalesces_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sklearn.cluster import kmeans_plusplus as sklearn_kmeans_plusplus
    from sklearn.utils.random import sample_without_replacement

    values, loadings = _custom_inputs()
    transformed = values.dot(loadings)
    stream, data = _coordinate_stream(transformed, block_size=2)
    sampled: list[np.ndarray] = []

    def capture_kmeans_plusplus(
        values: np.ndarray,
        *,
        n_clusters: int,
        random_state: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        sampled.append(values.copy())
        return sklearn_kmeans_plusplus(
            values,
            n_clusters=n_clusters,
            random_state=random_state,
        )

    monkeypatch.setattr("sklearn.cluster.kmeans_plusplus", capture_kmeans_plusplus)
    result = KMeansInitializationStage.fit(
        stream=stream,
        n_rows=values.shape[0],
        batch_size=2,
        n_clusters=3,
        rand_state=4466,
        nthreads=1,
        kmeans_sampling=0.5,
        kmeans_batch_size=5,
    )

    expected_indices = np.sort(
        sample_without_replacement(
            values.shape[0],
            4,
            method="reservoir_sampling",
            random_state=4466,
        )
    )
    np.testing.assert_allclose(sampled[0], transformed[expected_indices])
    assert result.model is not None
    assert result.model.n_clusters == 3
    assert result.model.batch_size == 5
    assert result.model.n_steps_ == 2
    assert result.labels.shape == (values.shape[0],)
    assert data.read_count == 12

    other_stream, other_data = _coordinate_stream(transformed, block_size=3)
    other_result = KMeansInitializationStage.fit(
        stream=other_stream,
        n_rows=values.shape[0],
        batch_size=3,
        n_clusters=3,
        rand_state=4466,
        nthreads=1,
        kmeans_sampling=0.5,
        kmeans_batch_size=5,
    )
    assert other_result.model is not None
    np.testing.assert_allclose(
        result.model.cluster_centers_,
        other_result.model.cluster_centers_,
    )
    np.testing.assert_array_equal(result.labels, other_result.labels)
    assert other_data.read_count == 9


def test_kmeans_initialization_rejects_single_row_and_empty_inputs() -> None:
    values, loadings = _custom_inputs()
    single_stream, _ = _coordinate_stream(values[:1].dot(loadings), block_size=1)
    with pytest.raises(ValueError, match="at least two rows"):
        KMeansInitializationStage.fit(
            stream=single_stream,
            n_rows=1,
            batch_size=1,
            n_clusters=5,
            rand_state=4466,
            nthreads=1,
        )

    empty_stream, _ = _coordinate_stream(
        np.empty((0, loadings.shape[1])),
        block_size=1,
    )
    with pytest.raises(ValueError, match="at least one row"):
        KMeansInitializationStage.fit(
            stream=empty_stream,
            n_rows=0,
            batch_size=1,
            n_clusters=2,
            rand_state=4466,
            nthreads=1,
        )


def test_harmony_stage_materializes_uncorrected_coordinates_once(
    monkeypatch,
) -> None:
    values, loadings = _custom_inputs()
    stream, data = _coordinate_stream(values.dot(loadings), block_size=3)
    corrected_values = values.dot(loadings) + 1.0

    def fake_harmony(
        uncorrected: np.ndarray,
        batches: pd.DataFrame,
        **_parameters,
    ) -> HarmonyResult:
        np.testing.assert_allclose(uncorrected, values.dot(loadings).T)
        assert list(batches.columns) == ["batch"]
        return HarmonyResult(
            original=uncorrected,
            corrected=corrected_values.T,
            assignments=np.ones((1, values.shape[0])),
            centroids=np.zeros((1, 2)),
            sigma=np.ones(1),
            ridge=np.eye(1),
            batch_columns=("batch",),
            batch_levels=(("a", "b"),),
            parameters={},
        )

    monkeypatch.setattr("scarf.neighbors.stages.fit_harmony", fake_harmony)
    stage = BatchCorrectionStage(
        stream=stream,
        n_cells=values.shape[0],
        dims=loadings.shape[1],
        batch_size=3,
        batches=pd.DataFrame({"batch": ["a", "b"] * 4}),
        parameters={},
        corrected_data=None,
        nthreads=1,
    )

    first = stage.ensure_corrected()
    second = stage.ensure_corrected()

    np.testing.assert_allclose(first.compute(), corrected_values)
    assert second is first
    assert data.read_count == 3


def _transform_arguments(values: np.ndarray, **overrides) -> dict:
    arguments = dict(
        data=ChunkedArray.from_numpy(values, block_size=4, nthreads=1),
        method="pca",
        dims=2,
        loadings=None,
        use_for_pca=np.ones(values.shape[0], dtype=bool),
        mu=values.mean(axis=0),
        sigma=values.std(axis=0),
        batch_size=4,
        nthreads=1,
        rand_state=4466,
        disable_scaling=True,
        lsi_skip_first=False,
        lsi_params={},
    )
    return arguments | overrides


def test_reduction_transform_rejects_unknown_methods_and_short_fit_masks() -> None:
    values, _loadings = _custom_inputs()

    with pytest.raises(ValueError, match="Unknown reduction method: tsne"):
        ReductionTransform(**_transform_arguments(values, method="tsne"))
    with pytest.raises(ValueError, match="does not have sample length as nCells"):
        ReductionTransform(
            **_transform_arguments(values, use_for_pca=np.ones(7, dtype=bool))
        )


def test_lsi_without_reduction_returns_the_values_unchanged() -> None:
    values, _loadings = _custom_inputs()

    reduction = ReductionTransform(**_transform_arguments(values, method="lsi", dims=0))

    assert reduction.loadings is None
    np.testing.assert_array_equal(reduction.transform(values), values)


class _ListStream:
    """A coordinate source that yields scripted blocks, one list per pass.

    The last list repeats for every later pass.
    """

    def __init__(self, *passes: list[np.ndarray]) -> None:
        self.passes = passes
        self.calls = 0

    def iter_coordinate_blocks(self, message: str):
        blocks = self.passes[min(self.calls, len(self.passes) - 1)]
        self.calls += 1
        yield from blocks


def _batch_stage(stream, **overrides) -> BatchCorrectionStage:
    arguments = dict(
        stream=stream,
        n_cells=8,
        dims=2,
        batch_size=4,
        batches=pd.DataFrame({"batch": ["a", "b"] * 4}),
        parameters={},
        corrected_data=None,
        nthreads=1,
    )
    return BatchCorrectionStage(**(arguments | overrides))


def test_harmony_stage_rejects_missing_batches_and_malformed_coordinates() -> None:
    values = np.arange(16, dtype=np.float64).reshape(8, 2)

    with pytest.raises(ValueError, match="Harmony requires batch metadata"):
        _batch_stage(_ListStream([values]), batches=None).ensure_corrected()
    with pytest.raises(ValueError, match="Coordinate block has an invalid shape"):
        _batch_stage(_ListStream([values[:4], values[4:, :1]])).ensure_corrected()
    with pytest.raises(ValueError, match="Coordinate block has an invalid shape"):
        _batch_stage(_ListStream([values, values[:1]])).ensure_corrected()
    with pytest.raises(ValueError, match="contains 6 rows, expected 8"):
        _batch_stage(_ListStream([values[:6]])).ensure_corrected()


def test_harmony_stage_returns_supplied_corrections_without_reading() -> None:
    corrected = ChunkedArray.from_numpy(np.ones((8, 2)), block_size=4, nthreads=1)
    stream = _ListStream([])

    assert _batch_stage(stream, corrected_data=corrected).ensure_corrected() is (
        corrected
    )
    assert stream.calls == 0


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"kmeans_sampling": True}, TypeError, "kmeans_sampling must be a number"),
        ({"kmeans_sampling": "half"}, TypeError, "kmeans_sampling must be a number"),
        ({"kmeans_sampling": None}, TypeError, "kmeans_sampling must be a number"),
        ({"kmeans_sampling": 0.0}, ValueError, "greater than 0 and at most 1"),
        ({"kmeans_sampling": 1.5}, ValueError, "greater than 0 and at most 1"),
        ({"kmeans_sampling": np.nan}, ValueError, "greater than 0 and at most 1"),
        ({"kmeans_batch_size": True}, TypeError, "must be a positive integer"),
        ({"kmeans_batch_size": 2.5}, TypeError, "must be a positive integer"),
        ({"kmeans_batch_size": 0}, ValueError, "must be a positive integer"),
        ({"batch_size": True}, TypeError, "^batch_size must be a positive integer"),
        ({"batch_size": 2.5}, TypeError, "^batch_size must be a positive integer"),
        ({"batch_size": 0}, ValueError, "^batch_size must be a positive integer"),
    ],
)
def test_kmeans_initialization_rejects_invalid_options(options, error, message) -> None:
    stream = _ListStream([np.ones((8, 2))])

    with pytest.raises(error, match=message):
        KMeansInitializationStage.fit(
            stream=stream,
            n_rows=8,
            n_clusters=2,
            rand_state=4466,
            nthreads=1,
            **{"batch_size": 4, **options},
        )
    assert stream.calls == 0


# Two tight groups of four points, far apart.
_KMEANS_VALUES = np.array(
    [[0, 0], [0, 1], [1, 0], [1, 1], [10, 10], [10, 11], [11, 10], [11, 11]],
    dtype=np.float64,
)
_KMEANS_BLOCKS = [_KMEANS_VALUES[:4], _KMEANS_VALUES[4:]]


@pytest.mark.parametrize(
    ("passes", "message"),
    [
        ([[]], "coordinate source is empty"),
        ([[np.arange(4.0)]], "blocks must be two-dimensional"),
        # A batch size below the row count streams even one complete block.
        ([[_KMEANS_VALUES, _KMEANS_VALUES[:1]]], "has too many rows"),
        # Sampling pass.
        ([[_KMEANS_VALUES[:4], _KMEANS_VALUES[4:, :1]]], "dimensions changed"),
        (
            [[_KMEANS_VALUES[:4], _KMEANS_VALUES[4:].astype(np.float32)]],
            "dimensions changed",
        ),
        ([_KMEANS_BLOCKS + [_KMEANS_VALUES[:2]]], "has too many rows"),
        ([[_KMEANS_VALUES[:4], _KMEANS_VALUES[4:6]]], "contains 6 rows, expected 8"),
        # Fitting pass.
        ([_KMEANS_BLOCKS, [_KMEANS_VALUES[:4, :1]]], "dimensions changed"),
        ([_KMEANS_BLOCKS, _KMEANS_BLOCKS + [_KMEANS_VALUES[:2]]], "too many rows"),
        ([_KMEANS_BLOCKS, [_KMEANS_VALUES[:4]]], "contains 4 rows, expected 8"),
        # Prediction pass.
        (
            [_KMEANS_BLOCKS, _KMEANS_BLOCKS, [_KMEANS_VALUES[:4, :1]]],
            "dimensions changed",
        ),
        (
            [_KMEANS_BLOCKS, _KMEANS_BLOCKS, _KMEANS_BLOCKS + [_KMEANS_VALUES[:2]]],
            "too many rows",
        ),
        (
            [_KMEANS_BLOCKS, _KMEANS_BLOCKS, [_KMEANS_VALUES[4:]]],
            "contains 4 rows, expected 8",
        ),
    ],
    ids=[
        "empty",
        "one_dimensional",
        "streamed_rows_after_complete_block",
        "sampling_columns_changed",
        "sampling_dtype_changed",
        "sampling_too_many_rows",
        "sampling_too_few_rows",
        "fitting_columns_changed",
        "fitting_too_many_rows",
        "fitting_too_few_rows",
        "prediction_columns_changed",
        "prediction_too_many_rows",
        "prediction_too_few_rows",
    ],
)
def test_kmeans_initialization_rejects_inconsistent_coordinate_passes(
    passes, message
) -> None:
    with pytest.raises(ValueError, match=message):
        KMeansInitializationStage.fit(
            stream=_ListStream(*passes),
            n_rows=8,
            batch_size=4,
            n_clusters=2,
            rand_state=4466,
            nthreads=1,
        )


@pytest.mark.parametrize(
    ("blocks", "message"),
    [
        ([_KMEANS_VALUES, _KMEANS_VALUES[:1]], "rows after a complete block"),
        ([_KMEANS_VALUES[:4], _KMEANS_VALUES[4:, :1]], "dimensions changed"),
        (
            [_KMEANS_VALUES[:4], _KMEANS_VALUES[4:].astype(np.float32)],
            "dimensions changed",
        ),
        (_KMEANS_BLOCKS + [_KMEANS_VALUES[:2]], "has too many rows"),
        ([_KMEANS_VALUES[:4], _KMEANS_VALUES[4:6]], "contains 6 rows, expected 8"),
    ],
    ids=[
        "rows_after_complete_block",
        "columns_changed",
        "dtype_changed",
        "too_many_rows",
        "too_few_rows",
    ],
)
def test_in_memory_kmeans_rejects_inconsistent_blocks(blocks, message) -> None:
    with pytest.raises(ValueError, match=message):
        KMeansInitializationStage.fit(
            stream=_ListStream(blocks),
            n_rows=8,
            batch_size=8,
            n_clusters=2,
            rand_state=4466,
            nthreads=1,
        )


def test_kmeans_fit_mode_follows_the_batch_size_not_the_stream_blocks() -> None:
    rng = np.random.default_rng(1)
    values = np.vstack(
        [rng.normal(loc=center, size=(40, 3)) for center in (0.0, 5.0, 10.0)]
    )
    n_rows = values.shape[0]

    def fit(block_size: int, batch_size: int):
        stream, data = _coordinate_stream(values, block_size=block_size)
        result = KMeansInitializationStage.fit(
            stream=stream,
            n_rows=n_rows,
            batch_size=batch_size,
            n_clusters=3,
            rand_state=4466,
            nthreads=1,
            kmeans_sampling=0.5,
            kmeans_batch_size=16,
        )
        return result, data.read_count

    # A batch size that covers every row fits them in memory, in one pass,
    # however the stream splits them into blocks.
    in_memory, in_memory_reads = fit(n_rows, n_rows)
    gathered, gathered_reads = fit(25, n_rows)
    assert (in_memory_reads, gathered_reads) == (1, 5)
    assert in_memory.model.n_steps_ > 1
    np.testing.assert_array_equal(
        gathered.model.cluster_centers_, in_memory.model.cluster_centers_
    )
    np.testing.assert_array_equal(gathered.labels, in_memory.labels)
    # A smaller batch size streams sampling, fitting, and labelling passes,
    # even over one block that holds every row.
    streamed, streamed_reads = fit(25, 60)
    one_block, one_block_reads = fit(n_rows, 60)
    assert (streamed_reads, one_block_reads) == (15, 3)
    np.testing.assert_array_equal(
        one_block.model.cluster_centers_, streamed.model.cluster_centers_
    )
    np.testing.assert_array_equal(one_block.labels, streamed.labels)


def test_kmeans_initialization_streams_three_consistent_passes() -> None:
    stream = _ListStream(_KMEANS_BLOCKS)

    result = KMeansInitializationStage.fit(
        stream=stream,
        n_rows=8,
        batch_size=4,
        n_clusters=2,
        rand_state=4466,
        nthreads=1,
    )

    assert stream.calls == 3
    # The two groups are the two clusters.
    np.testing.assert_array_equal(
        result.labels, result.labels[[0, 0, 0, 0, 7, 7, 7, 7]]
    )
    assert result.labels[0] != result.labels[7]
    # The prediction pass labels each cell with its nearest fitted centroid.
    distances = np.linalg.norm(
        _KMEANS_VALUES[:, None] - result.model.cluster_centers_[None], axis=2
    )
    np.testing.assert_array_equal(result.labels, distances.argmin(axis=1))


class _ReleaseCheckingStream:
    """A coordinate source that yields a fresh copy of each row block.

    Before each read after the first of a pass, it records whether the block
    it yielded before is still alive. A read plan reserves only the blocks in
    flight, so a consumer that still holds the previous block exceeds it.
    """

    def __init__(self, values: np.ndarray, block_rows: int) -> None:
        self.values = values
        self.block_rows = block_rows
        self.held: list[bool] = []

    def iter_coordinate_blocks(self, message: str):
        previous = None
        for start in range(0, self.values.shape[0], self.block_rows):
            if previous is not None:
                self.held.append(previous() is not None)
            block = self.values[start : start + self.block_rows].copy()
            previous = weakref.ref(block)
            yield block
            del block


@pytest.mark.parametrize(
    ("batch_size", "reads"),
    [(120, 2), (40, 6)],
    ids=["in_memory", "streamed"],
)
def test_kmeans_fit_holds_no_block_while_its_stream_reads_the_next(
    batch_size: int, reads: int
) -> None:
    values = np.random.default_rng(2).normal(size=(120, 3))
    stream = _ReleaseCheckingStream(values, block_rows=40)

    KMeansInitializationStage.fit(
        stream=stream,
        n_rows=120,
        batch_size=batch_size,
        n_clusters=3,
        rand_state=4466,
        nthreads=1,
    )

    # Three blocks per pass: the in-memory fit gathers them in one pass, and
    # the streamed fit reads them in three.
    assert stream.held == [False] * reads


def test_ann_index_holds_no_block_while_its_stream_reads_the_next() -> None:
    values = np.random.default_rng(3).normal(size=(90, 4)).astype(np.float32)
    stream = _ReleaseCheckingStream(values, block_rows=30)

    AnnIndexStage.fit(
        coordinates=stream,
        metric="l2",
        dims=4,
        n_cells=90,
        ef_construction=20,
        ef=20,
        m=4,
        rand_state=1,
        nthreads=1,
    )

    assert stream.held == [False, False]


def test_harmony_stage_holds_no_block_while_its_stream_reads_the_next(
    monkeypatch,
) -> None:
    values = np.random.default_rng(4).normal(size=(90, 3))
    stream = _ReleaseCheckingStream(values, block_rows=30)

    def fit(uncorrected: np.ndarray, batches: pd.DataFrame, **_options):
        return HarmonyResult(
            original=uncorrected,
            corrected=uncorrected,
            assignments=np.ones((1, uncorrected.shape[1])),
            centroids=np.zeros((3, 1)),
            sigma=np.ones(1),
            ridge=np.eye(2),
            batch_columns=("batch",),
            batch_levels=(("a", "b"),),
            parameters={},
        )

    monkeypatch.setattr("scarf.neighbors.stages.fit_harmony", fit)
    stage = BatchCorrectionStage(
        stream=stream,
        n_cells=90,
        dims=3,
        batch_size=30,
        batches=pd.DataFrame({"batch": ["a", "b"] * 45}),
        parameters={},
        corrected_data=None,
        nthreads=1,
    )

    np.testing.assert_array_equal(stage.ensure_corrected().compute(), values)
    assert stream.held == [False, False]


def test_kmeans_fit_memory_counts_what_each_mode_holds() -> None:
    options = {
        "n_rows": 200_000,
        "dims": 2,
        "dtype": np.float32,
        "n_clusters": 50,
        "block_rows": 2_000,
        "kmeans_sampling": 0.001,
        "kmeans_batch_size": 1_000,
    }
    streamed = kmeans_fit_memory(batch_size=2_000, nthreads=1, **options)
    # The labelling pass holds one uint32 label per cell while it reads.
    assert streamed.streamResidentBytes >= 200_000 * 4
    # Each thread holds its own distances and centroid sums.
    threads = kmeans_fit_memory(batch_size=2_000, nthreads=2, **options)
    assert threads.peakBytes > streamed.peakBytes
    # An in-memory fit gathers every row while its stream reads the blocks.
    in_memory = kmeans_fit_memory(batch_size=200_000, nthreads=1, **options)
    assert in_memory.streamResidentBytes >= 200_000 * 2 * 4


def _fit_kmeans(values: np.ndarray, *, batch_size: int, **options):
    # Blocks of a numpy-backed stream are views of the values, so a trace of
    # the fit holds only what the fit allocates, as a stream's plan reserves
    # its own blocks.
    stream = ChunkedCoordinateStream(
        ChunkedArray.from_numpy(values, block_size=2_000, nthreads=1), 1
    )
    return KMeansInitializationStage.fit(
        stream=stream,
        n_rows=values.shape[0],
        batch_size=batch_size,
        n_clusters=50,
        rand_state=4466,
        nthreads=1,
        kmeans_batch_size=1_000,
        **options,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("dtype", "dims", "batch_size", "sampling"),
    [
        (np.float32, 10, 20_000, 0.1),
        (np.float64, 20, 20_000, 0.1),
        (np.float16, 10, 20_000, 0.1),
        (np.float32, 10, 2_000, 0.5),
        (np.float64, 20, 2_000, 0.5),
        (np.float16, 10, 2_000, 0.5),
    ],
    ids=[
        "in_memory_float32",
        "in_memory_float64",
        "in_memory_float16",
        "streamed_float32",
        "streamed_float64",
        "streamed_float16",
    ],
)
def test_kmeans_fit_memory_bounds_the_traced_peak(
    dtype, dims: int, batch_size: int, sampling: float
) -> None:
    n_rows = 20_000
    values = np.random.default_rng(5).normal(size=(n_rows, dims)).astype(dtype)
    memory = kmeans_fit_memory(
        n_rows=n_rows,
        dims=dims,
        dtype=values.dtype,
        batch_size=batch_size,
        n_clusters=50,
        block_rows=2_000,
        nthreads=1,
        kmeans_sampling=sampling,
        kmeans_batch_size=1_000,
    )
    # Import scikit-learn and create its thread-pool controller before the
    # trace, which would otherwise count them.
    small = values[:2_000]
    for warm_batch_size in (2_000, 500):
        _fit_kmeans(small, batch_size=warm_batch_size, kmeans_sampling=sampling)

    tracemalloc.start()
    try:
        _fit_kmeans(values, batch_size=batch_size, kmeans_sampling=sampling)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # The estimate also counts scikit-learn's per-thread buffers, which the
    # trace misses, and an allowance for the fit's Python objects. It leaves
    # out the transient with which each read pass sets thread limits, about
    # 0.6 MB here, which these shapes outgrow.
    assert peak <= memory.peakBytes <= 2 * peak
