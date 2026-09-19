import numpy as np
import pytest
from scipy.sparse import csr_matrix

from scarf import DataStore
from scarf.matrix import ChunkedArray
from scarf.neighbors.diffusion import diffusion_operator
from scarf.quality_control import doublets
from scarf.storage.budget import ResourceBudget
from scarf.storage.count_matrix import CountMatrixPolicy


@pytest.mark.parametrize(
    "save_k,error",
    [(0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError)],
)
def test_doublet_scoring_rejects_invalid_k_before_reading_counts(save_k, error):
    with pytest.raises(error, match="save_k must be a positive integer"):
        doublets.score_synthetic_doublets(
            None,
            None,
            np.arange(2),
            np.array([0, 1]),
            np.arange(3),
            cluster_sample_fraction=1,
            max_cells_per_cluster=2,
            simulation_ratio=1,
            heterotypic_fraction=0.8,
            save_k=save_k,
            random_seed=91,
            resources=ResourceBudget(1_000_000, 1),
        )


@pytest.mark.parametrize("detected", [[0, 9, 10, 11], [9, 10, 10, 11], [10, 10]])
def test_doublet_statistics_preserve_feature_filter_and_batch_results(detected):
    counts = -np.ones((len(detected), 14), dtype=np.int16)
    for i, n in enumerate(detected):
        counts[i, :n] = 1
    pool = csr_matrix(counts)
    pairs = np.arange(len(counts))
    if max(detected) == min(detected) == 10:
        with pytest.raises(ValueError, match="minimum-feature filter"):
            doublets._doublet_statistics(
                pool,
                pairs,
                pairs,
                resources=ResourceBudget(1_000_000, 1),
                preferred_rows=1,
                resident_bytes=0,
            )
        return
    expected_keep = (
        np.ones(len(counts), dtype=bool)
        if np.median(detected) < 10
        else np.asarray(detected) > 10
    )
    for rows in (1, 3, 100):
        totals, keep = doublets._doublet_statistics(
            pool,
            pairs,
            pairs,
            resources=ResourceBudget(1_000_000, 1),
            preferred_rows=rows,
            resident_bytes=0,
        )
        np.testing.assert_array_equal(totals, (counts * 2).sum(axis=1))
        np.testing.assert_array_equal(keep, expected_keep)


def test_doublet_parent_pool_is_read_in_admitted_blocks(monkeypatch):
    counts = np.arange(20 * 19, dtype=np.uint16).reshape(20, 19)
    raw = ChunkedArray.from_numpy(counts)
    rows = np.array([1, 3, 4, 7, 10, 11, 12, 16])
    sizes = []
    original = doublets.csr_matrix

    def observe(values):
        sizes.append(len(values))
        return original(values)

    monkeypatch.setattr(doublets, "csr_matrix", observe)
    result = doublets._load_parent_counts(raw, rows, ResourceBudget(4_096, 1), 0)
    np.testing.assert_array_equal(result.toarray(), counts[rows])
    assert len(sizes) > 1
    assert max(sizes) < len(rows)
    with pytest.raises(MemoryError):
        doublets._load_parent_counts(raw, rows, ResourceBudget(100, 1), 0)


@pytest.mark.parametrize("power", [1, 2, 5])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_doublet_smoothing_matches_powered_diffusion_with_hub_and_isolate(power, dtype):
    dense = np.zeros((101, 101), dtype=dtype)
    dense[0, 1:100] = dense[1:100, 0] = 0.5
    graph = csr_matrix(dense)
    scores = np.linspace(0, 1, len(dense))
    expected = diffusion_operator(graph, power=power).dot(scores)
    actual = doublets.smooth_doublet_scores(graph, scores, power=power, normalize=False)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)
    assert actual[-1] == 0
    constant = doublets.smooth_doublet_scores(
        graph,
        np.zeros(len(dense)),
        power=power,
        normalize=True,
    )
    np.testing.assert_array_equal(constant, 0)


@pytest.mark.parametrize(
    "power,error", [(0, ValueError), (True, TypeError), (1.5, TypeError)]
)
def test_doublet_smoothing_rejects_invalid_power(power, error):
    with pytest.raises(error, match="positive integer"):
        doublets.smooth_doublet_scores(
            csr_matrix(np.eye(2)), np.ones(2), power=power, normalize=True
        )


@pytest.mark.parametrize(
    "log,subset,constant,filtered",
    [
        (False, False, False, False),
        (True, False, False, False),
        (True, True, False, False),
        (False, False, True, False),
        (False, False, False, True),
    ],
)
def test_streamed_doublets_match_materialized_mapping(
    tmp_path, monkeypatch, log, subset, constant, filtered
):
    rng = np.random.default_rng(73)
    counts = (
        np.full((30, 32), 5, dtype=np.uint16)
        if constant
        else rng.integers(0, 200, (30, 24), dtype=np.uint16)
    )
    if filtered:
        counts[:15, 4:] = 0
    ids = np.array([f"g{i}" for i in range(counts.shape[1])])
    path = str(tmp_path / "reference.zarr")
    doublets.write_doublet_target_zarr(
        path,
        "RNA",
        csr_matrix(counts),
        ids,
        ids,
        dtype=str(counts.dtype),
        nthreads=1,
        policy=CountMatrixPolicy(unitBytes=16_384, chunkBytes=1024),
    )
    store = DataStore(path, default_assay="RNA", min_features_per_cell=0, nthreads=1)
    cells = store.snapshot_cell_selection("I")
    feature_indices = np.arange(0, counts.shape[1], 2)
    features = store.set_feature_selection(feature_indexes=feature_indices)
    normalized = store.run_normalization(
        cells, features, log_transform=log, renormalize_subset=subset
    )
    pca = store.run_pca(normalized, dims=3, local_cache=False)
    ann = store.build_ann_index(pca)
    neighbors = store.query_neighbors(ann, coordinates=pca, k=4)
    reference = store.get_mapping_reference(store.build_mapping_reference(neighbors))
    labels = np.arange(len(counts)) % 3
    options = dict(
        cluster_sample_fraction=0.5,
        max_cells_per_cluster=3,
        simulation_ratio=2.0,
        heterotypic_fraction=0.8,
        save_k=100,
        random_seed=91,
        resources=ResourceBudget(128 * 1024**2, 1),
    )
    rng = np.random.default_rng(91)
    parents = doublets.sample_cluster_pool(labels, 0.5, 3, rng)
    left, right = doublets.simulate_doublet_pairs(labels[parents], 60, 0.8, rng)
    simulated = doublets.sum_doublet_pairs(csr_matrix(counts[parents]), left, right)
    if filtered:
        detected = np.asarray((simulated > 0).sum(axis=1)).ravel()
        assert np.median(detected) > 10
        assert np.any(detected <= 10)
    query_path = str(tmp_path / "query.zarr")
    doublets.write_doublet_target_zarr(
        query_path,
        "RNA",
        simulated,
        ids,
        ids,
        dtype=str(simulated.dtype),
        nthreads=1,
        policy=CountMatrixPolicy(unitBytes=16_384, chunkBytes=1024),
    )
    query = DataStore(query_path, default_assay="RNA", nthreads=1)
    result = query.run_mapping(
        reference, query.snapshot_cell_selection("I"), save_k=100
    )
    _, expected = next(
        query.get_mapping_score(result, reference=reference, log_transform=True)
    )
    actual = doublets.score_synthetic_doublets(
        store.RNA,
        reference,
        np.arange(len(counts)),
        labels,
        feature_indices,
        **options,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-9)
    if constant:
        np.testing.assert_array_equal(actual, 0)
    original = doublets._doublet_batch_rows

    def smaller_batches(*args, **kwargs):
        return min(1 if filtered else 2, original(*args, **kwargs))

    monkeypatch.setattr(doublets, "_doublet_batch_rows", smaller_batches)
    repeated = doublets.score_synthetic_doublets(
        store.RNA,
        reference,
        np.arange(len(counts)),
        labels,
        feature_indices,
        **options,
    )
    np.testing.assert_allclose(repeated, expected, rtol=1e-8, atol=1e-9)
    with pytest.raises(MemoryError):
        doublets.score_synthetic_doublets(
            store.RNA,
            reference,
            np.arange(len(counts)),
            labels,
            feature_indices,
            **(options | {"resources": ResourceBudget(1_000_000, 1)}),
        )
