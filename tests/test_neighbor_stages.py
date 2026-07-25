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


def _ann_stream(*, cache_embeddings: bool) -> AnnStream:
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
        cache_embeddings=cache_embeddings,
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


def test_ann_stream_adapter_preserves_cached_and_lazy_numerics() -> None:
    values, loadings = _custom_inputs()
    cached = _ann_stream(cache_embeddings=True)
    lazy = _ann_stream(cache_embeddings=False)
    expected = values.dot(loadings)

    np.testing.assert_allclose(cached.embeddings, expected)
    assert lazy.embeddings is None
    np.testing.assert_allclose(lazy.transform_query(values), expected)
    np.testing.assert_array_equal(cached.clusterLabels, lazy.clusterLabels)
    np.testing.assert_allclose(
        cached.kmeans.cluster_centers_,
        lazy.kmeans.cluster_centers_,
    )

    cached_indices, cached_distances = cached.transform_ann(expected, k=3)
    lazy_indices, lazy_distances = lazy.transform_ann(expected, k=3)
    np.testing.assert_array_equal(cached_indices, lazy_indices)
    np.testing.assert_allclose(cached_distances, lazy_distances)


def test_lazy_transform_cache_is_shared_by_ann_and_kmeans() -> None:
    values, loadings = _custom_inputs()
    data = _CountingChunkedArray(values, block_size=3)
    stream = LazyTransformStream(
        data=data,
        transform=lambda block: block.dot(loadings),
        nthreads=1,
        batch_size=3,
    )

    assert data.read_count == 0
    cached = stream.cache("cache")
    assert data.read_count == 3
    assert stream.cache("cache again") is cached
    assert data.read_count == 3

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

    assert data.read_count == 3
    assert index.get_current_count() == values.shape[0]
    query = NeighborQueryStage(index, k=3)
    indices, distances = query.query(cached, k=3)
    assert indices.shape == distances.shape == (values.shape[0], 3)
    assert initialization.model is not None
    assert initialization.labels.shape == (values.shape[0],)


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
    assert data.read_count == 6


def test_kmeans_initialization_caps_small_inputs_and_rejects_empty() -> None:
    values, loadings = _custom_inputs()
    single_data = ChunkedArray.from_numpy(values[:1], block_size=1, nthreads=1)
    single_stream = LazyTransformStream(
        data=single_data,
        transform=lambda block: block.dot(loadings),
        nthreads=1,
        batch_size=1,
    )
    single = KMeansInitializationStage.fit(
        stream=single_stream,
        n_clusters=5,
        rand_state=4466,
        nthreads=1,
        enabled=True,
    )
    assert single.model is not None
    assert single.model.n_clusters == 1
    assert single.labels.shape == (1,)

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
            cache_embeddings=False,
        )

    assert not harmony_called
    assert data.read_count == 0
