"""RNA assay, PCA and LSI fitting, and pseudotime paths on small inputs.

The RNA tests reach ``scarf.assay.rna`` through the DataStore and its public
assay methods. The fitting and pseudotime tests call the domain functions of
``scarf.embeddings.reduction`` and ``scarf.trajectory`` directly with small
deterministic arrays.
"""

from typing import Any

import numpy as np
import pytest
import zarr
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import ArpackNoConvergence
from sklearn.decomposition import TruncatedSVD

import scarf.trajectory.pseudotime as pseudotime_module
from scarf.assay.normalization import norm_dummy
from scarf.assay.rna import _hvg_stats_gene_major_kernel
from scarf.datastore.datastore import DataStore
from scarf.embeddings.reduction import fit_incremental_pca, fit_lsi
from scarf.matrix import ChunkedArray
from scarf.trajectory import (
    random_walk_laplacian_transpose,
    select_pseudotime_component,
    truncated_pba_potential,
    validate_source_sink_vector,
)
from tests.storage_helpers import write_count_store

N_CELLS = 24
N_GENES = 12


def _counts() -> np.ndarray:
    counts = np.random.default_rng(23).poisson(2.0, size=(N_CELLS, N_GENES))
    # Four cells without RNA counts.
    counts[:4] = 0
    return counts


def _write(directory: Any, counts: np.ndarray) -> str:
    zarr_loc = str(directory / "store.zarr")
    write_count_store(zarr_loc, {"RNA": counts}, "uint16")
    return zarr_loc


def _open(zarr_loc: str, **options: Any) -> DataStore:
    return DataStore(
        zarr_loc,
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
        **options,
    )


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> DataStore:
    store = _open(_write(tmp_path_factory.mktemp("rna_paths"), _counts()))
    store.cells.insert("everyone", np.ones(N_CELLS, dtype=bool), overwrite=True)
    store.cells.insert("nobody", np.zeros(N_CELLS, dtype=bool), overwrite=True)
    without_counts = np.zeros(N_CELLS, dtype=bool)
    without_counts[:4] = True
    store.cells.insert("without_counts", without_counts, overwrite=True)
    return store


def test_rna_feature_batches_require_one_dimensional_indices(
    store: DataStore,
) -> None:
    batches = store.RNA.iter_normed_feature_wise(
        np.zeros((2, 2), dtype=np.int64), np.arange(3), None, None
    )

    with pytest.raises(ValueError, match="must be one-dimensional"):
        next(batches)


def test_rna_feature_batches_of_no_features_are_empty(store: DataStore) -> None:
    batches = store.RNA.iter_normed_feature_wise(
        np.arange(N_CELLS), np.array([], dtype=np.int64), None, None
    )

    assert list(batches) == []


def test_rna_feature_statistics_count_only_positive_values_as_detected() -> None:
    values = np.array([[2, -3, 1], [-1, 0, -2]], dtype=np.int32)
    detected = np.zeros(2)
    totals = np.zeros(2)
    deviations = np.zeros(2)

    _hvg_stats_gene_major_kernel.py_func(
        values,
        np.ones(3),
        1.0,
        np.array([0, 1], dtype=np.int64),
        np.arange(3, dtype=np.int64),
        detected,
        totals,
        deviations,
    )

    # Negative stored values add to the sums and deviations but are not
    # detections. The second gene's mean of -1 includes its zero.
    np.testing.assert_array_equal(detected, [2.0, 0.0])
    np.testing.assert_array_equal(totals, [0.0, -3.0])
    np.testing.assert_array_equal(deviations, [14.0, 2.0])


def test_rna_gene_major_kernel_gives_equal_values_zero_deviation() -> None:
    # 3,000 / 7 is not a float64 integer, so squared sums of these equal
    # values do not cancel exactly.
    values = np.array([[3] * 9, [3, 0, 3, 3, 0, 3, 3, 3, 0]], dtype=np.uint16)
    deviations = np.zeros(2)

    _hvg_stats_gene_major_kernel(
        values,
        np.full(9, 1 / 7),
        1000.0,
        np.array([0, 1], dtype=np.int64),
        np.arange(9, dtype=np.int64),
        np.zeros(2),
        np.zeros(2),
        deviations,
    )

    assert deviations[0] == 0.0
    normalized = 1000.0 * values[1] / 7
    np.testing.assert_allclose(
        deviations[1], np.square(normalized - normalized.mean()).sum(), rtol=1e-14
    )


def test_constant_features_have_zero_variance_and_unit_pca_scale(
    tmp_path, monkeypatch
) -> None:
    from scarf.storage.artifacts import artifact_group, inspect_artifact

    rng = np.random.default_rng(3)
    n_cells = 10_007
    first, second = rng.integers(0, 5, size=(2, n_cells))
    # Every cell holds 11 counts, 3 of them in the first feature, so that
    # feature normalizes to the same value in every cell. Squared sums of
    # these values leave a variance of about 3e-8 before the log transform,
    # and of 2e-12 after it, which made the PCA scale about 1.5e-6.
    counts = np.column_stack(
        [np.full(n_cells, 3), first, 4 - first, second, 4 - second]
    )
    store = _open(_write(tmp_path, counts))

    stats = store.RNA._streaming_feature_stats(np.arange(n_cells), np.arange(5))
    # NumPy's own two-pass variance of the constant feature is about 1e-21.
    assert stats["sigmas"][0] == 0.0
    expected = 1000.0 * counts[:, 1:] / 11
    np.testing.assert_allclose(stats["sigmas"][1:], expected.var(axis=0), rtol=1e-12)

    normalized = store.run_normalization(
        store.snapshot_cell_selection(), store.select_all_features(from_assay="RNA")
    )
    group = artifact_group(store.zw, normalized)
    data = np.asarray(group["data"][:], dtype=np.float64)
    assert np.unique(data[:, 0]).size == 1
    assert group["feature_m2"][0] == 0.0
    np.testing.assert_allclose(
        group["feature_m2"][:],
        np.square(data - data.mean(axis=0)).sum(axis=0),
        rtol=1e-12,
    )
    np.testing.assert_allclose(group["feature_sum"][:], data.sum(axis=0), rtol=1e-12)

    def stream(*_args, **_kwargs):
        raise AssertionError("scaling must read the stored moments")

    monkeypatch.setattr(ChunkedArray, "mean_and_std", stream)
    pca = store.run_pca(normalized, dims=2, local_cache=False)
    scaling = artifact_group(
        store.zw, inspect_artifact(store.zw, pca).input_ref("feature_scaling")
    )
    assert scaling["scale"][0] == 1.0
    assert scaling["mean"][0] == data[0, 0]
    np.testing.assert_allclose(
        scaling["scale"][1:], data[:, 1:].std(axis=0), rtol=1e-12
    )


def test_read_only_store_without_a_saved_size_factor_uses_the_default(
    tmp_path,
) -> None:
    zarr_loc = _write(tmp_path, _counts())
    _open(zarr_loc)
    del zarr.open_group(zarr_loc, mode="r+")["RNA"].attrs["size_factor"]

    store = _open(zarr_loc, zarr_mode="r")

    assert store.RNA.sf == 1000
    assert "size_factor" not in store.RNA.attrs


def test_rna_feature_statistics_that_exceed_the_budget_raise_memory_error(
    tmp_path,
) -> None:
    counts = np.random.default_rng(5).poisson(3.0, size=(200, 40))
    store = _open(_write(tmp_path, counts), mem_budget="20K")

    with pytest.raises(MemoryError, match="operation limit"):
        store.select_detected_features(store.snapshot_cell_selection(), min_cells=1)


def test_rna_summary_of_an_empty_cell_selection_detects_no_feature(
    store: DataStore,
) -> None:
    nobody = store.snapshot_cell_selection("nobody")

    with pytest.raises(ValueError, match="Detected-feature selection contains no"):
        store.select_detected_features(nobody, min_cells=1)


def test_rna_summary_follows_a_replaced_normalization(
    store: DataStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(store.RNA, "normMethod", norm_dummy)
    cells = store.snapshot_cell_selection("everyone")

    detected = store.select_detected_features(cells, min_cells=19)

    expected = (_counts() > 0).sum(axis=0) >= 19
    assert expected.any() and not expected.all()
    np.testing.assert_array_equal(store.load_artifact(detected)["values"][:], expected)


def test_hvgs_of_cells_without_counts_have_no_corrected_variance(
    store: DataStore,
) -> None:
    cells = store.snapshot_cell_selection("without_counts")

    ref = store.select_hvgs(
        cells, min_cells=0, max_cells=np.inf, top_n=3, n_bins=4, show_plot=False
    )

    saved = store.load_artifact(ref)
    # Every feature ties at zero, so the lowest feature indices are kept.
    np.testing.assert_array_equal(np.flatnonzero(saved["values"][:]), [0, 1, 2])
    np.testing.assert_array_equal(saved["corrected_variance"][:], np.zeros(N_GENES))


@pytest.mark.parametrize(
    ("n_rows", "selected"),
    [
        # 25 rows in blocks of 8 leave a one-row block at the end.
        (25, None),
        # The first block keeps one selected row, fewer than dims + 1.
        (24, np.r_[[True], np.zeros(7, dtype=bool), np.ones(16, dtype=bool)]),
    ],
)
def test_incremental_pca_fits_rows_of_short_blocks_once(
    n_rows: int, selected: np.ndarray | None
) -> None:
    values = np.random.default_rng(13).normal(size=(n_rows, 10))
    mask = np.ones(n_rows, dtype=bool) if selected is None else selected

    _loadings, model = fit_incremental_pca(
        ChunkedArray.from_numpy(values, block_size=8),
        dims=2,
        batch_size=8,
        use_for_pca=mask,
        scale=None,
        nthreads=1,
    )

    # Ten features exceed eight rows per block, so IncrementalPCA fits.
    assert model.n_samples_seen_ == int(mask.sum())
    np.testing.assert_allclose(model.mean_, values[mask].mean(axis=0))


@pytest.mark.parametrize("value", [1.0, np.nan])
def test_gram_pca_rejects_input_without_positive_finite_variance(
    value: float,
) -> None:
    values = np.ones((24, 6))
    values[5, 2] = value

    with pytest.raises(ValueError, match="positive finite variance"):
        fit_incremental_pca(
            ChunkedArray.from_numpy(values, block_size=8),
            dims=2,
            batch_size=8,
            use_for_pca=np.ones(24, dtype=bool),
            scale=None,
            nthreads=1,
        )


@pytest.mark.parametrize(
    ("params", "message"),
    [
        (
            {"n_iter": 2, "algorithm": "arpack"},
            "does not support parameters: algorithm",
        ),
        ({"solver": "exact"}, "solver must be 'streaming' or 'materialized'"),
    ],
)
def test_lsi_rejects_unknown_solver_parameters(
    params: dict[str, Any], message: str
) -> None:
    data = ChunkedArray.from_numpy(np.ones((6, 4)), block_size=3)

    with pytest.raises(ValueError, match=message):
        fit_lsi(
            data, dims=1, skip_first=False, params=params, random_state=0, nthreads=1
        )


def test_materialized_lsi_keeps_the_first_component_when_asked() -> None:
    values = np.random.default_rng(3).uniform(size=(20, 8))
    expected = TruncatedSVD(n_components=3, n_iter=4, random_state=2).fit(values)

    loadings = fit_lsi(
        ChunkedArray.from_numpy(values, block_size=6),
        dims=3,
        skip_first=False,
        params={"solver": "materialized", "n_iter": 4},
        random_state=2,
        nthreads=1,
    )

    np.testing.assert_allclose(loadings, expected.components_.T, atol=1e-12)


def test_streaming_lsi_rejects_non_finite_input() -> None:
    values = np.random.default_rng(4).uniform(size=(12, 5))
    values[7, 1] = np.inf

    with pytest.raises(ValueError, match="only finite values"):
        fit_lsi(
            ChunkedArray.from_numpy(values, block_size=4),
            dims=2,
            skip_first=True,
            params={"n_iter": 1},
            random_state=0,
            nthreads=1,
        )


def _path_graph(n_cells: int) -> csr_matrix:
    adjacency = np.zeros((n_cells, n_cells))
    for index in range(n_cells - 1):
        adjacency[index, index + 1] = adjacency[index + 1, index] = 1.0
    return csr_matrix(adjacency)


def test_source_sink_vector_must_be_numeric() -> None:
    with pytest.raises(TypeError, match="ss_vec must contain numeric values"):
        validate_source_sink_vector(np.array(["source", "sink"]), 2, "ss_vec")


def test_component_selection_rejects_unknown_policies() -> None:
    with pytest.raises(ValueError, match="'largest' or 'error'"):
        select_pseudotime_component(_path_graph(3), np.arange(3), "smallest")  # type: ignore[arg-type]


def test_random_walk_laplacian_rejects_isolated_cells() -> None:
    graph = _path_graph(4).tolil()
    graph[2, 3] = graph[3, 2] = 0.0

    with pytest.raises(ValueError, match="contains isolated cells"):
        random_walk_laplacian_transpose(graph.tocsr())


def test_pseudotime_potential_reports_an_unconverged_svd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unconverged(*_args: Any, **_kwargs: Any) -> None:
        raise ArpackNoConvergence("no convergence", np.zeros(0), np.zeros((6, 0)))

    monkeypatch.setattr(pseudotime_module, "svds", unconverged)
    laplacian = random_walk_laplacian_transpose(_path_graph(6))

    with pytest.raises(RuntimeError, match="did not converge"):
        truncated_pba_potential(laplacian, 3, 0, np.r_[-1.0, np.zeros(4), 1.0])


def test_pseudotime_potential_requires_one_null_mode() -> None:
    source_sink = np.r_[-1.0, np.zeros(4), 1.0]
    two_paths = csr_matrix(
        np.block(
            [
                [_path_graph(3).toarray(), np.zeros((3, 3))],
                [np.zeros((3, 3)), _path_graph(3).toarray()],
            ]
        )
    )

    with pytest.raises(ValueError, match="does not contain the expected null mode"):
        truncated_pba_potential(csr_matrix(identity(6)), 3, 0, source_sink)
    with pytest.raises(ValueError, match="additional near-zero singular modes"):
        truncated_pba_potential(
            random_walk_laplacian_transpose(two_paths), 3, 0, source_sink
        )
