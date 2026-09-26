"""Datastore graph, reduction, and embedding operation contracts."""

import dataclasses
import shutil

import numpy as np
import pytest
from scipy.sparse import coo_matrix

from scarf.datastore.datastore import DataStore
from scarf.embeddings.umap import (
    DENSMAP_ALGORITHM_VERSION,
    calc_dens_map_params,
    densmap_distance_graph,
)
from scarf.metadata.arguments import UmapArguments
from scarf.storage.artifacts import ArtifactRef, fingerprint_array
from scarf.storage.budget import ResourceBudget

_STANDARD_UMAP_PARAMETERS = {
    "symmetric_graph": False,
    "graph_upper_only": False,
    "umap_dims": 2,
    "spread": 2.0,
    "min_dist": 1,
    "n_epochs": 300,
    "repulsion_strength": 1.0,
    "initial_alpha": 1.0,
    "negative_sample_rate": 5,
    "use_density_map": False,
    "dens_lambda": 2.0,
    "dens_frac": 0.3,
    "dens_var_shift": 0.1,
    "random_seed": 4444,
    "parallel": False,
    "parallel_threads": None,
}


@pytest.fixture(scope="module")
def analyzed_store(analyzed_datastore_zarr_root, tmp_path_factory) -> DataStore:
    root = tmp_path_factory.mktemp("graph_embedding_operations") / "data.zarr"
    shutil.copytree(analyzed_datastore_zarr_root, root)
    return DataStore(str(root), default_assay="RNA", nthreads=1)


@pytest.fixture
def store(analyzed_store: DataStore):
    resources = analyzed_store.resources
    yield analyzed_store
    analyzed_store.resources = resources


def _input(store: DataStore, ref: ArtifactRef, name: str) -> ArtifactRef:
    return ArtifactRef.from_dict(store.inspect_artifact(ref).inputs[name])


def _graph(store: DataStore) -> ArtifactRef:
    graphs = store.list_artifacts(
        kind="connectivity_map",
        from_assay="RNA",
        complete_only=True,
    )
    assert len(graphs) == 1
    return graphs[0]


def _reduction(store: DataStore) -> ArtifactRef:
    neighbors = _input(store, _graph(store), "neighbors")
    return _input(store, neighbors, "coordinates")


def _normalized(store: DataStore) -> ArtifactRef:
    return _input(store, _reduction(store), "normalized")


def _initialization(store: DataStore) -> ArtifactRef:
    initializations = store.list_artifacts(
        kind="embedding_initialization",
        from_assay="RNA",
        complete_only=True,
    )
    assert len(initializations) == 1
    return initializations[0]


def _incomplete(store: DataStore, kind: str, scope: str = "assay") -> list:
    return [
        ref
        for ref in store.list_artifacts(kind=kind, from_assay="RNA", scope=scope)
        if not store.inspect_artifact(ref).complete
    ]


def test_run_umap_copies_and_canonicalizes_numpy_initialization(store) -> None:
    graph = _graph(store)
    n_cells = store.load_graph(graph).shape[0]
    initial = np.random.default_rng(3).normal(size=(n_cells, 2)).astype(np.float32)
    original = initial.copy()

    ref = store.run_umap(graph, initial, n_epochs=5)

    np.testing.assert_array_equal(initial, original)
    assert store.inspect_artifact(ref).inputs["initialization"] == {
        "value_fingerprint": fingerprint_array(original)
    }
    wide = np.zeros((n_cells, 4), dtype=np.float32)
    wide[:, ::2] = original
    variants = (
        original.astype(np.float64),
        np.asfortranarray(original),
        wide[:, ::2],
    )
    for n_epochs, variant in enumerate(variants, start=6):
        unchanged = variant.copy()
        assert store.run_umap(graph, variant, n_epochs=5) == ref
        fitted = store.run_umap(graph, variant, n_epochs=n_epochs)
        np.testing.assert_array_equal(variant, unchanged)
        values = store.load_artifact(fitted)["values"][:]
        assert values.shape == (n_cells, 2)
        assert np.all(np.isfinite(values))


def test_run_umap_rejects_invalid_numpy_initialization(store) -> None:
    graph = _graph(store)
    n_cells = store.load_graph(graph).shape[0]
    initial = np.zeros((n_cells, 2), dtype=np.float32)
    initial[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        store.run_umap(graph, initial, n_epochs=5)
    with pytest.raises(ValueError, match="finite"):
        store.run_umap(graph, np.full((n_cells, 2), 1e300), n_epochs=5)
    with pytest.raises(TypeError, match="real numbers"):
        store.run_umap(graph, np.zeros((n_cells, 2), dtype=bool), n_epochs=5)
    with pytest.raises(ValueError, match="invalid shape"):
        store.run_umap(graph, np.zeros((n_cells, 3)), n_epochs=5)


def test_run_tsne_rejects_invalid_numpy_initialization(store) -> None:
    graph = _graph(store)
    n_cells = store.load_graph(graph).shape[0]

    with pytest.raises(ValueError, match="finite"):
        store.run_tsne(graph, np.full((n_cells, 2), np.nan))
    with pytest.raises(TypeError, match="real numbers"):
        store.run_tsne(graph, np.zeros((n_cells, 2), dtype=bool))


def test_standard_umap_arguments_keep_their_recorded_identity() -> None:
    arguments = UmapArguments(
        graph=ArtifactRef(
            scope="assay",
            assay="RNA",
            kind="connectivity_map",
            artifact_id="1" * 64,
        ),
        initialization=ArtifactRef(
            scope="assay",
            assay="RNA",
            kind="embedding_initialization",
            artifact_id="2" * 64,
        ),
        invalidate_cache=False,
        **_STANDARD_UMAP_PARAMETERS,
    )

    assert arguments.to_record().parameters == _STANDARD_UMAP_PARAMETERS
    assert arguments.provenance_hash() == (
        "56a2875652e4dfb859ad7e93d67b38ac422ede7405297d6fb0fa6a44deef16ec"
    )
    with pytest.raises(ValueError, match="densmap_algorithm_version"):
        dataclasses.replace(arguments, use_density_map=True)


def test_densmap_records_its_revision_and_reuses_without_reading_neighbors(
    store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _graph(store)
    initialization = _initialization(store)

    standard = store.run_umap(graph, initialization, n_epochs=5)
    densmap = store.run_umap(
        graph,
        initialization,
        n_epochs=5,
        use_density_map=True,
    )

    assert set(store.inspect_artifact(standard).parameters) == set(
        _STANDARD_UMAP_PARAMETERS
    )
    parameters = store.inspect_artifact(densmap).parameters
    assert parameters["use_density_map"] is True
    assert parameters["densmap_algorithm_version"] == DENSMAP_ALGORITHM_VERSION
    values = store.load_artifact(densmap)["values"][:]
    assert np.all(np.isfinite(values))

    def fail(*_args, **_kwargs):
        raise AssertionError("reused densMAP must not load neighbor distances")

    monkeypatch.setattr("scarf.embeddings.umap.densmap_distance_graph", fail)
    assert (
        store.run_umap(graph, initialization, n_epochs=5, use_density_map=True)
        == densmap
    )


def test_densmap_distances_are_symmetric_with_the_larger_direction() -> None:
    indices = np.array([[1], [2], [1]])
    distances = np.array([[0.5], [0.7], [0.9]], dtype=np.float32)

    symmetric = densmap_distance_graph(indices, distances)

    np.testing.assert_allclose(
        symmetric.toarray(),
        np.array(
            [
                [0.0, 0.5, 0.0],
                [0.5, 0.0, 0.9],
                [0.0, 0.9, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def test_densmap_parameters_read_reverse_only_edges_and_match_a_loop() -> None:
    rng = np.random.default_rng(11)
    n_cells, n_neighbors = 40, 4
    indices = np.array(
        [
            rng.choice(np.delete(np.arange(n_cells), cell), n_neighbors, replace=False)
            for cell in range(n_cells)
        ]
    )
    distances = rng.uniform(0.5, 2.0, size=(n_cells, n_neighbors)).astype(np.float32)
    rows = np.repeat(np.arange(n_cells), n_neighbors)
    directed = coo_matrix(
        (rng.uniform(0.1, 1.0, size=rows.size), (rows, indices.ravel())),
        shape=(n_cells, n_cells),
    ).tocsr()
    graph = (directed + directed.T).tocoo()
    symmetric = densmap_distance_graph(indices, distances)
    dense = symmetric.toarray()

    mu_sum, standardized = calc_dens_map_params(graph, symmetric)

    assert np.all(dense[graph.row, graph.col] > 0)
    expected_ro = np.zeros(n_cells)
    expected_mu = np.zeros(n_cells)
    for head, tail, weight in zip(graph.row, graph.col, graph.data):
        distance = float(dense[head, tail]) ** 2
        for cell in (head, tail):
            expected_ro[cell] += weight * distance
            expected_mu[cell] += weight
    expected_log = np.log(1e-8 + expected_ro / expected_mu)
    np.testing.assert_allclose(mu_sum, expected_mu, rtol=1e-6)
    np.testing.assert_allclose(
        standardized,
        (expected_log - expected_log.mean()) / expected_log.std(),
        rtol=1e-5,
        atol=1e-5,
    )
    dense_mu_sum, dense_standardized = calc_dens_map_params(graph, dense)
    np.testing.assert_array_equal(dense_mu_sum, mu_sum)
    np.testing.assert_array_equal(dense_standardized, standardized)
    assert mu_sum.dtype == standardized.dtype == np.float32


def test_load_graph_validates_use_k(store) -> None:
    graph = _graph(store)
    k = int(store.zw[store.inspect_artifact(graph).path].attrs["n_neighbors"])

    for value in (0, -1, k + 1):
        with pytest.raises(ValueError, match="use_k must be between 1"):
            store.load_graph(graph, use_k=value)
    for value in (True, 1.5, "2"):
        with pytest.raises(TypeError, match="use_k must be an integer or None"):
            store.load_graph(graph, use_k=value)  # type: ignore[arg-type]

    nearest = store.load_graph(graph, use_k=np.int64(1))
    assert np.diff(nearest.indptr).max() == 1
    assert (store.load_graph(graph) != store.load_graph(graph, use_k=k)).nnz == 0


def test_run_lsi_validates_skip_first_and_rand_state(store) -> None:
    normalized = _normalized(store)

    with pytest.raises(TypeError, match="skip_first must be a boolean"):
        store.run_lsi(normalized, dims=3, skip_first=2)  # type: ignore[arg-type]
    for value in (None, True, 1.5):
        with pytest.raises(TypeError, match="rand_state must be an integer"):
            store.run_lsi(normalized, dims=3, rand_state=value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rand_state must be non-negative"):
        store.run_lsi(normalized, dims=3, rand_state=-1)
    with pytest.raises(ValueError, match="loadings have shape"):
        store._run_reduction_artifact(
            method="lsi",
            normalized=normalized,
            dims=3,
            pca_cell_selection=None,
            feat_scaling=False,
            lsi_skip_first=2,  # type: ignore[arg-type]
            custom_loadings=None,
            rand_state=4466,
            batch_size=None,
            local_cache=False,
            show_elbow_plot=False,
            invalidate_cache=False,
        )
    assert _incomplete(store, "reduction") == []


def test_run_custom_reduction_requires_finite_real_loadings(store) -> None:
    normalized = _normalized(store)
    n_features = int(store.zw[store.inspect_artifact(normalized).path]["data"].shape[1])
    loadings = np.random.default_rng(5).normal(size=(n_features, 3))

    for bad_value in (np.nan, np.inf):
        invalid = loadings.copy()
        invalid[0, 0] = bad_value
        with pytest.raises(ValueError, match="finite"):
            store.run_custom_reduction(invalid, normalized, local_cache=False)
    for invalid in (loadings.astype(str), loadings > 0):
        with pytest.raises(TypeError, match="real numbers"):
            store.run_custom_reduction(invalid, normalized, local_cache=False)
    assert _incomplete(store, "reduction") == []

    ref = store.run_custom_reduction(loadings, normalized, local_cache=False)
    assert store.inspect_artifact(ref).complete
    assert store.load_artifact(ref)["data"].shape[1] == 3


def test_run_harmony_validates_arguments_before_snapshotting_metadata(store) -> None:
    reduction = _reduction(store)

    def snapshots() -> list:
        return store.list_artifacts(kind="metadata_snapshot", scope="datastore")

    before = snapshots()
    invalid_calls = (
        ({"batch_size": 0}, ValueError, "batch_size"),
        ({"batch_size": True}, TypeError, "batch_size"),
        ({"harmony_params": {"bogus": 1}}, ValueError, "Unsupported Harmony"),
    )
    for kwargs, error, message in invalid_calls:
        with pytest.raises(error, match=message):
            store.run_harmony(reduction, ["ids"], **kwargs)

    assert snapshots() == before


def test_integrate_assays_validates_chunk_size_first(store) -> None:
    graph = _graph(store)

    for value in (0, -5):
        with pytest.raises(ValueError, match="chunk_size"):
            store.integrate_assays([graph, graph], method="snn", chunk_size=value)
    for value in (True, 2.5):
        with pytest.raises(TypeError, match="chunk_size"):
            store.integrate_assays(
                [graph, graph],
                method="snn",
                chunk_size=value,  # type: ignore[arg-type]
            )


def test_materialized_lsi_checks_its_memory_budget_first(store) -> None:
    normalized = _normalized(store)
    store.resources = ResourceBudget(200_000, 1)

    with pytest.raises(MemoryError, match="solver='streaming'"):
        store.run_lsi(normalized, dims=3, solver="materialized", local_cache=False)

    assert _incomplete(store, "reduction") == []


def test_reduction_write_budget_fails_before_fitting(
    store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scarf.embeddings.reduction as reduction_module

    normalized = _normalized(store)
    fits: list[object] = []
    fit_lsi = reduction_module.fit_lsi

    def recorded_fit(*args, **kwargs):
        fits.append(args)
        return fit_lsi(*args, **kwargs)

    monkeypatch.setattr(reduction_module, "fit_lsi", recorded_fit)
    store.resources = ResourceBudget(2_000_000, 1)

    # The streaming solver fits this budget; writing its coordinates does not.
    with pytest.raises(MemoryError, match="One unit needs|Resident data needs"):
        store.run_lsi(normalized, dims=5, local_cache=False, invalidate_cache=True)

    assert fits == []
    assert _incomplete(store, "reduction") == []
