import numpy as np
import pandas as pd
import pytest

from scarf.embeddings.harmony import HarmonyResult
from scarf.matrix import ChunkedArray
from scarf.neighbors.stages import (
    AnnIndexStage,
    BatchCorrectionStage,
    KMeansInitializationStage,
    LazyTransformStream,
    NeighborQueryStage,
    ReductionTransform,
)
from scarf.neighbors.stream import AnnStream


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


def _ann_stream() -> AnnStream:
    values, loadings = _custom_inputs()
    return AnnStream(
        data=ChunkedArray.from_numpy(values, block_size=4, nthreads=1),
        k=3,
        n_cluster=3,
        reduction_method="custom",
        dims=2,
        loadings=loadings,
        use_for_pca=np.ones(values.shape[0], dtype=bool),
        mu=np.zeros(values.shape[1]),
        sigma=np.ones(values.shape[1]),
        ann_metric="l2",
        ann_efc=50,
        ann_ef=50,
        ann_m=16,
        nthreads=1,
        ann_parallel=False,
        rand_state=4466,
        do_kmeans_fit=True,
        disable_scaling=True,
        ann_idx=None,
        lsi_skip_first=False,
        lsi_params={},
        harmonize=False,
    )


def test_reduction_transform_keeps_cell_coordinates_lazy() -> None:
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
    stream = LazyTransformStream(
        data=data,
        transform=reduction.transform,
        nthreads=1,
        batch_size=3,
    )

    assert data.read_count == 0
    first = next(stream.iter_transformed())
    np.testing.assert_allclose(first, values[:3].dot(loadings))
    assert 1 <= data.read_count <= 3


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


@pytest.mark.parametrize("skip_first", [False, True])
def test_lsi_dims_are_final_output_dimensions(skip_first: bool) -> None:
    from scarf.embeddings.reduction import fit_lsi

    values, _loadings = _custom_inputs()
    loadings = fit_lsi(
        ChunkedArray.from_numpy(values, block_size=4, nthreads=1),
        dims=2,
        skip_first=skip_first,
        params={},
        random_state=4466,
        nthreads=1,
    )

    assert loadings.shape == (values.shape[1], 2)


def test_ann_stream_adapter_preserves_reduction_numerics() -> None:
    values, loadings = _custom_inputs()
    stream = _ann_stream()
    expected = values.dot(loadings)

    np.testing.assert_allclose(stream.transform_query(values), expected)
    assert stream.clusterLabels.shape == (values.shape[0],)
    indices, distances = stream.transform_ann(expected, k=3)
    assert indices.shape == distances.shape == (values.shape[0], 3)


def test_lazy_coordinate_stages_do_not_hide_a_cross_stage_cache() -> None:
    values, loadings = _custom_inputs()
    data = _CountingChunkedArray(values, block_size=3)
    stream = LazyTransformStream(
        data=data,
        transform=lambda block: block.dot(loadings),
        nthreads=1,
        batch_size=3,
    )

    index = AnnIndexStage.fit(
        coordinates=stream,
        metric="l2",
        dims=2,
        n_cells=values.shape[0],
        ef_construction=50,
        ef=50,
        m=16,
        rand_state=4466,
        ann_threads=1,
    )
    initialization = KMeansInitializationStage.fit(
        stream=stream,
        n_clusters=3,
        rand_state=4466,
        nthreads=1,
        enabled=True,
    )

    assert data.read_count == 12
    assert index.get_current_count() == values.shape[0]
    query = NeighborQueryStage(index, k=3, metric="l2")
    indices, distances = query.query(values.dot(loadings), k=3)
    assert indices.shape == distances.shape == (values.shape[0], 3)
    assert initialization.model is not None
    assert initialization.labels.shape == (values.shape[0],)


def test_neighbor_query_validates_and_converts_metric_distances() -> None:
    stage = NeighborQueryStage(index=None, k=2, metric="l2")
    distances = np.array([[4.0, -1e-7]], dtype=np.float32)
    np.testing.assert_allclose(
        stage._metric_distances(distances),
        [[2.0, 0.0]],
    )

    cosine = NeighborQueryStage(index=None, k=2, metric="cosine")
    with pytest.raises(ValueError, match="negative"):
        cosine._metric_distances(np.array([[0.1, -0.01]], dtype=np.float32))
    with pytest.raises(ValueError, match="non-finite"):
        cosine._metric_distances(np.array([[0.1, np.nan]], dtype=np.float32))


def test_kmeans_initialization_runs_on_demand_without_ann() -> None:
    values, loadings = _custom_inputs()
    data = _CountingChunkedArray(values, block_size=3)
    stream = LazyTransformStream(
        data=data,
        transform=lambda block: block.dot(loadings),
        nthreads=1,
        batch_size=3,
    )

    disabled = KMeansInitializationStage.fit(
        stream=stream,
        n_clusters=3,
        rand_state=4466,
        nthreads=1,
        enabled=False,
    )
    assert disabled.model is None
    assert data.read_count == 0

    enabled = KMeansInitializationStage.fit(
        stream=stream,
        n_clusters=3,
        rand_state=4466,
        nthreads=1,
        enabled=True,
    )
    assert enabled.model is not None
    assert enabled.labels.shape == (values.shape[0],)
    assert data.read_count == 9


def test_kmeans_initialization_uses_true_minibatches_for_one_full_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values, loadings = _custom_inputs()
    data = _CountingChunkedArray(values, block_size=values.shape[0])
    stream = LazyTransformStream(
        data=data,
        transform=lambda block: block.dot(loadings),
        nthreads=1,
        batch_size=values.shape[0],
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
        n_clusters=3,
        rand_state=4466,
        nthreads=1,
        enabled=True,
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
    data = _CountingChunkedArray(values, block_size=2)
    stream = LazyTransformStream(
        data=data,
        transform=lambda block: block.dot(loadings),
        nthreads=1,
        batch_size=2,
    )
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
        n_clusters=3,
        rand_state=4466,
        nthreads=1,
        enabled=True,
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

    other_data = _CountingChunkedArray(values, block_size=3)
    other_result = KMeansInitializationStage.fit(
        stream=LazyTransformStream(
            data=other_data,
            transform=lambda block: block.dot(loadings),
            nthreads=1,
            batch_size=3,
        ),
        n_clusters=3,
        rand_state=4466,
        nthreads=1,
        enabled=True,
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
    single_data = ChunkedArray.from_numpy(values[:1], block_size=1, nthreads=1)
    single_stream = LazyTransformStream(
        data=single_data,
        transform=lambda block: block.dot(loadings),
        nthreads=1,
        batch_size=1,
    )
    with pytest.raises(ValueError, match="at least two rows"):
        KMeansInitializationStage.fit(
            stream=single_stream,
            n_clusters=5,
            rand_state=4466,
            nthreads=1,
            enabled=True,
        )

    empty_stream = LazyTransformStream(
        data=ChunkedArray.from_numpy(
            np.empty((0, values.shape[1])),
            block_size=1,
            nthreads=1,
        ),
        transform=lambda block: block.dot(loadings),
        nthreads=1,
        batch_size=1,
    )
    with pytest.raises(ValueError, match="at least one row"):
        KMeansInitializationStage.fit(
            stream=empty_stream,
            n_clusters=2,
            rand_state=4466,
            nthreads=1,
            enabled=True,
        )


def test_harmony_stage_materializes_uncorrected_coordinates_once(
    monkeypatch,
) -> None:
    values, loadings = _custom_inputs()
    data = _CountingChunkedArray(values, block_size=3)
    stream = LazyTransformStream(
        data=data,
        transform=lambda block: block.dot(loadings),
        nthreads=1,
        batch_size=3,
    )
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


def test_invalid_ann_configuration_fails_before_harmony_reads(
    monkeypatch,
) -> None:
    values, loadings = _custom_inputs()
    data = _CountingChunkedArray(values, block_size=3)
    harmony_called = False

    def fail_if_called(*_args, **_kwargs):
        nonlocal harmony_called
        harmony_called = True
        raise AssertionError("Harmony should not run")

    monkeypatch.setattr("scarf.neighbors.stages.fit_harmony", fail_if_called)
    with pytest.raises(RuntimeError):
        AnnStream(
            data=data,
            k=3,
            n_cluster=3,
            reduction_method="custom",
            dims=2,
            loadings=loadings,
            use_for_pca=np.ones(values.shape[0], dtype=bool),
            mu=np.zeros(values.shape[1]),
            sigma=np.ones(values.shape[1]),
            ann_metric="not_a_metric",
            ann_efc=50,
            ann_ef=50,
            ann_m=16,
            nthreads=1,
            ann_parallel=False,
            rand_state=4466,
            do_kmeans_fit=False,
            disable_scaling=True,
            ann_idx=None,
            lsi_skip_first=False,
            lsi_params={},
            harmonize=True,
            batches=pd.DataFrame({"batch": ["a", "b"] * 4}),
        )

    assert not harmony_called
    assert data.read_count == 0
