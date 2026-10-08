"""Datastore graph, reduction, and embedding operation contracts."""

import dataclasses
import shutil

import numpy as np
import pytest
from scipy.sparse import coo_matrix

import scarf.embeddings.sgtsne as sgtsne_module
from scarf.datastore.datastore import DataStore
from scarf.embeddings.umap import (
    calc_dens_map_params,
    densmap_distance_graph,
)
from scarf.graph.arguments import OperationArguments
from scarf.metadata.arguments import TsneArguments, UmapArguments
from scarf.storage.artifacts import (
    ArtifactRef,
    fingerprint_array,
    make_provenance,
    provenance_hash,
)
from scarf.storage.budget import ResourceBudget


def _identity(arguments: OperationArguments) -> str:
    """Return the provenance hash an artifact plan records for ``arguments``."""
    record = arguments.to_record()
    return provenance_hash(
        make_provenance(
            operation=arguments.operation,
            parameters=record.parameters,
            inputs=record.inputs,
        )
    )


_STANDARD_UMAP_PARAMETERS = {
    "symmetric_graph": False,
    "graph_upper_only": False,
    "umap_dims": 2,
    "spread": 2.0,
    "min_dist": 1.0,
    "n_epochs": 300,
    "repulsion_strength": 1.0,
    "initial_alpha": 1.0,
    "negative_sample_rate": 5.0,
    "use_density_map": False,
    "dens_lambda": 2.0,
    "dens_frac": 0.3,
    "dens_var_shift": 0.1,
    "random_seed": 4444,
    "parallel": False,
}


_STANDARD_TSNE_PARAMETERS = {
    "symmetric_graph": False,
    "graph_upper_only": False,
    "tsne_dims": 2,
    "lambda_scale": 1.0,
    "max_iter": 500,
    "early_iter": 200,
    "alpha": 10,
    "box_h": 0.7,
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


def _fake_tsne_backend(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Replace sgtsnepi with a recorder, so the tests run without the package."""
    calls: list[dict[str, object]] = []

    def run_sgtsne(graph, initial, **kwargs):
        calls.append(kwargs)
        return np.asarray(initial, dtype=np.float64).T + len(calls)

    monkeypatch.setattr(sgtsne_module, "require_sgtsnepi", lambda: run_sgtsne)
    monkeypatch.setattr(sgtsne_module, "run_sgtsne", run_sgtsne)
    return calls


def test_run_tsne_validates_settings_before_reading_or_planning(
    store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Unchecked settings were recorded as given, including infinity and NaN.
    # test_sgtsne.py checks each setting; run_tsne checks them all at once.
    graph = _graph(store)
    initialization = _initialization(store)

    def fail(*_args, **_kwargs):
        raise AssertionError("invalid settings must fail before any store access")

    monkeypatch.setattr(store, "_embedding_inputs", fail)
    monkeypatch.setattr(
        "scarf.datastore._operations.embeddings.plan_cell_data_artifact", fail
    )
    with pytest.raises(ValueError, match="lambda_scale must be finite"):
        store.run_tsne(graph, initialization, lambda_scale=float("inf"))


def test_tsne_identity_holds_canonical_settings_only(
    store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _graph(store)
    initialization = _initialization(store)
    calls = _fake_tsne_backend(monkeypatch)

    ref = store.run_tsne(
        graph,
        initialization,
        lambda_scale=1,
        max_iter=np.int64(30),
        box_h=np.float32(0.5),
        verbose=False,
    )

    status = store.inspect_artifact(ref)
    # Thread counts and temporary-file locations never identify an embedding.
    assert status.parameters == {
        **_STANDARD_TSNE_PARAMETERS,
        "max_iter": 30,
        "box_h": 0.5,
    }
    assert type(status.parameters["lambda_scale"]) is float
    assert status.execution_options == {"verbose": False, "invalidate_cache": False}
    assert calls == [
        {
            "tsne_dims": 2,
            "max_iter": 30,
            "early_iter": 200,
            "alpha": 10,
            "lambda_scale": 1.0,
            "box_h": 0.5,
            "verbose": False,
        }
    ]
    assert (
        store.run_tsne(graph, initialization, max_iter=30, box_h=0.5, lambda_scale=1.0)
        == ref
    )
    assert len(calls) == 1


def test_standard_tsne_arguments_keep_their_recorded_identity() -> None:
    arguments = TsneArguments(
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
        verbose=True,
        invalidate_cache=False,
        **_STANDARD_TSNE_PARAMETERS,
    )

    record = arguments.to_record()
    assert record.parameters == _STANDARD_TSNE_PARAMETERS
    assert record.execution_options == {"verbose": True, "invalidate_cache": False}
    assert _identity(arguments) == (
        "3b3310238140ed212e02eda7328eb7db43bccb5f4304e6ef06c813b625c1c307"
    )


def test_embedding_graph_flags_reach_the_graph_loader_as_booleans(
    store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _graph(store)
    initialization = _initialization(store)
    loads: list[tuple[object, object]] = []
    original = store._load_graph_artifact

    def recording(graph_ref, *, symmetric, upper_only, use_k):
        loads.append((symmetric, upper_only))
        return original(
            graph_ref, symmetric=symmetric, upper_only=upper_only, use_k=use_k
        )

    monkeypatch.setattr(store, "_load_graph_artifact", recording)
    ref = store.run_umap(
        graph,
        initialization,
        n_epochs=5,
        symmetric_graph=np.True_,
        nthreads=3,
        invalidate_cache=True,
    )

    assert loads == [(True, False)]
    assert all(type(flag) is bool for flag in loads[0])
    parameters = store.inspect_artifact(ref).parameters
    assert parameters["symmetric_graph"] is True
    # The omitted flag records False: None loaded the graph as False under a
    # second identity.
    assert parameters["graph_upper_only"] is False
    # A serial layout runs on one thread whatever the request.
    assert store.inspect_artifact(ref).execution_options["layout_threads"] == 1
    for method, flag in (
        (store.run_umap, "symmetric_graph"),
        (store.run_umap, "graph_upper_only"),
        (store.run_tsne, "graph_upper_only"),
    ):
        for value in (1, None):
            with pytest.raises(TypeError, match=f"{flag} must be a boolean"):
                method(graph, initialization, **{flag: value})


def test_reused_umap_neither_loads_the_graph_nor_expands_initialization(
    store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _graph(store)
    initialization = _initialization(store)
    ref = store.run_umap(graph, initialization, n_epochs=5)

    def fail(*_args, **_kwargs):
        raise AssertionError("a reused embedding must not read its inputs")

    monkeypatch.setattr(store, "_load_graph_artifact", fail)
    monkeypatch.setattr(store, "_get_ini_embed", fail)
    assert store.run_umap(graph, initialization, n_epochs=5) == ref


def test_standard_umap_arguments_record_no_thread_count_in_their_identity() -> None:
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
        nthreads=8,
        layout_threads=1,
        invalidate_cache=False,
        **_STANDARD_UMAP_PARAMETERS,
    )

    record = arguments.to_record()
    assert record.parameters == _STANDARD_UMAP_PARAMETERS
    assert record.execution_options == {
        "nthreads": 8,
        "layout_threads": 1,
        "invalidate_cache": False,
    }
    # Releases before 1.0.0 also recorded parallel_threads, so their standard
    # embeddings had other identities.
    assert _identity(arguments) == (
        "dcb87841b695f253f8efa2564f0e6b8d6a33370b29d01272e71da9d36af6291b"
    )
    # The requested and the resolved thread counts depend on the machine, so
    # they never identify an embedding; the parallel flag does.
    for threads in (1, 2, 64):
        assert _identity(
            dataclasses.replace(
                arguments,
                parallel=True,
                nthreads=threads,
                layout_threads=min(threads, 4),
            )
        ) == _identity(dataclasses.replace(arguments, parallel=True))
    assert _identity(dataclasses.replace(arguments, parallel=True)) != _identity(
        arguments
    )
    densmap = dataclasses.replace(arguments, use_density_map=True)
    assert densmap.to_record().parameters == {
        **_STANDARD_UMAP_PARAMETERS,
        "use_density_map": True,
    }


def test_umap_parameter_spellings_share_one_canonical_identity(store) -> None:
    graph = _graph(store)
    initialization = _initialization(store)

    first = store.run_umap(
        graph, initialization, n_epochs=5, spread=2, min_dist=1, dens_frac=0.3
    )
    parameters = store.inspect_artifact(first).parameters
    assert parameters is not None
    for name in ("spread", "min_dist", "negative_sample_rate", "initial_alpha"):
        assert type(parameters[name]) is float
    assert type(parameters["n_epochs"]) is int
    assert (
        store.run_umap(
            graph,
            initialization,
            n_epochs=np.int64(5),
            spread=np.float32(2.0),
            min_dist=1.0,
            negative_sample_rate=np.int32(5),
            random_seed=np.uint16(4444),
        )
        == first
    )
    for kwargs, error, message in (
        ({"min_dist": True}, TypeError, "min_dist must be a real number"),
        ({"spread": float("nan")}, ValueError, "spread must be finite"),
        ({"n_epochs": 5.0}, TypeError, "n_epochs must be an integer"),
        ({"parallel": 1}, TypeError, "parallel must be a boolean"),
    ):
        with pytest.raises(error, match=message):
            store.run_umap(graph, initialization, **kwargs)


@pytest.mark.slow
def test_densmap_identity_differs_only_by_its_flag_and_reuses_without_neighbors(
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

    standard_parameters = store.inspect_artifact(standard).parameters
    assert set(standard_parameters) == set(_STANDARD_UMAP_PARAMETERS)
    parameters = store.inspect_artifact(densmap).parameters
    # Releases before 1.0.0 also recorded densmap_algorithm_version.
    assert parameters == {**standard_parameters, "use_density_map": True}
    values = store.load_artifact(densmap)["values"][:]
    assert np.all(np.isfinite(values))
    # The density term changes the layout from the same start and seed.
    assert not np.allclose(values, store.load_artifact(standard)["values"][:])

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


def test_ann_thread_counts_are_execution_options(
    store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scarf.neighbors.stages import AnnIndexStage

    reduction = _reduction(store)
    monkeypatch.setattr(store, "nthreads", 3)
    serial = store.build_ann_index(reduction, ann_efc=41)

    status = store.inspect_artifact(serial)
    assert status.parameters["ann_parallel"] is False
    assert status.parameters["parallel_threads"] is None
    # A serial index is built on one thread whatever the datastore's count.
    assert status.execution_options["nthreads"] == 1

    parallel = store.build_ann_index(reduction, ann_efc=41, ann_parallel=True)

    parallel_status = store.inspect_artifact(parallel)
    # Only the flag identifies a parallel index; earlier releases also
    # recorded the build's thread count as parallel_threads.
    assert parallel_status.parameters == {**status.parameters, "ann_parallel": True}
    assert parallel_status.execution_options["nthreads"] == 3
    # Another machine's thread count requests the same index.
    monkeypatch.setattr(store, "nthreads", 2)
    assert store.build_ann_index(reduction, ann_efc=41, ann_parallel=True) == parallel

    configured: list[int] = []
    configure = AnnIndexStage.configure

    def recording(index, *, ef, threads):
        configured.append(threads)
        return configure(index, ef=ef, threads=threads)

    monkeypatch.setattr(AnnIndexStage, "configure", staticmethod(recording))
    parallel_neighbors = store.query_neighbors(parallel, k=4)
    serial_neighbors = store.query_neighbors(serial, k=4)

    # Queries of a parallel index use the querying datastore's threads.
    assert configured == [2, 1]
    assert store.inspect_artifact(parallel_neighbors).execution_options["nthreads"] == 2
    assert store.inspect_artifact(serial_neighbors).execution_options["nthreads"] == 1
    assert "nthreads" not in store.inspect_artifact(parallel_neighbors).parameters


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
    with pytest.raises(ValueError, match="rand_state must be at least 0"):
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


def test_pca_identity_records_incremental_block_rows(store) -> None:
    normalized = _normalized(store)
    n_cells, n_features = store.load_artifact(normalized)["data"].shape
    block_rows = n_features // 2
    assert 6 < block_rows < n_cells

    # Blocks as wide as the features give an exact fit; narrower blocks use
    # IncrementalPCA, whose result depends on the block size.
    exact = store.run_pca(normalized, dims=5, batch_size=n_features)
    incremental = store.run_pca(normalized, dims=5, batch_size=block_rows)

    assert incremental != exact
    assert "incremental_block_rows" not in store.inspect_artifact(exact).parameters
    assert (
        store.inspect_artifact(incremental).parameters["incremental_block_rows"]
        == block_rows
    )
    assert store.run_pca(normalized, dims=5, batch_size=block_rows) == incremental
    wider = store.run_pca(normalized, dims=5, batch_size=block_rows + 1)
    assert wider not in {exact, incremental}


def test_graph_producers_refuse_read_only_stores_before_computing(
    store,
    monkeypatch,
) -> None:
    read_only = DataStore(store.zarr_loc, default_assay="RNA", zarr_mode="r")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a read-only store must refuse before computing")

    monkeypatch.setattr(
        "scarf.datastore._operations.graph.AnnIndexStage.fit",
        forbidden,
    )
    monkeypatch.setattr("scarf.neighbors.graph.build_connectivity_arrays", forbidden)
    with pytest.raises(PermissionError, match="build_ann_index"):
        read_only.build_ann_index(_reduction(store), ann_efc=37)
    neighbors = _input(store, _graph(store), "neighbors")
    with pytest.raises(PermissionError, match="build_connectivity_map"):
        read_only.build_connectivity_map(neighbors, bandwidth=1.25)
    with pytest.raises(PermissionError, match="run_pca"):
        read_only.run_pca(_normalized(store), dims=4)


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
    store.resources = ResourceBudget(1_000_000, 1)

    # The streaming solver fits smaller blocks within this budget, but the
    # coordinate write holds three decoded chunks, which leave no room for a
    # block of any size.
    with pytest.raises(MemoryError, match="One unit needs|Resident data needs"):
        store.run_lsi(normalized, dims=5, local_cache=False, invalidate_cache=True)

    assert fits == []
    assert _incomplete(store, "reduction") == []
