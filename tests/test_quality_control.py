from pathlib import Path

import numpy as np
import pytest

from scarf.quality_control.cell_cycle import assign_cell_cycle_phase
from scarf.quality_control.doublets import sample_cluster_pool, simulate_doublet_pairs
from scarf.quality_control.filtering import gaussian_quantile_bounds
from scarf.graph.feature_projection import resolve_native_graph_inputs
from scarf.metadata.artifacts import (
    artifact_values,
    plan_cell_data_artifact,
    write_cell_data_artifact,
)
from scarf.storage.artifacts import (
    ArtifactRef,
    artifact_group,
    fingerprint_array,
    fingerprint_strings,
)
from scarf.storage.selections import (
    read_stored_selection_mask,
    resolve_selection_artifact,
)


def test_simulate_doublet_pairs_is_seeded_and_heterotypic():
    clusters = np.array([0, 0, 1, 1])
    left, right = simulate_doublet_pairs(
        clusters,
        n_sim=12,
        heterotypic_fraction=1.0,
        rng=np.random.default_rng(11),
        max_tries=100,
    )

    np.testing.assert_array_equal(left, [0, 0, 3, 1, 2, 2, 2, 0, 1, 0, 1, 3])
    np.testing.assert_array_equal(right, [2, 3, 1, 3, 1, 1, 0, 2, 3, 2, 3, 0])
    assert np.all(clusters[left] != clusters[right])


def test_simulate_doublet_pairs_allows_homotypic_when_fraction_is_zero():
    clusters = np.array([0, 0, 1, 1])
    left, right = simulate_doublet_pairs(
        clusters,
        n_sim=40,
        heterotypic_fraction=0.0,
        rng=np.random.default_rng(3),
    )

    assert left.shape == right.shape == (40,)
    assert np.any(clusters[left] == clusters[right])


def test_sample_cluster_pool_respects_fraction_and_cap():
    clusters = np.array([0, 0, 0, 0, 1, 1, 2])
    rng = np.random.default_rng(7)

    pool = sample_cluster_pool(
        clusters,
        fraction=0.5,
        max_per_cluster=2,
        rng=rng,
    )

    np.testing.assert_array_equal(pool, np.sort(pool))
    assert set(pool).issubset(set(range(len(clusters))))
    counts = {int(c): int((clusters[pool] == c).sum()) for c in np.unique(clusters)}
    assert counts == {0: 2, 1: 1, 2: 1}

    repeated = sample_cluster_pool(
        clusters,
        fraction=0.5,
        max_per_cluster=2,
        rng=np.random.default_rng(7),
    )
    np.testing.assert_array_equal(pool, repeated)

    with pytest.raises(ValueError, match="No cells could be sampled"):
        sample_cluster_pool(
            clusters,
            fraction=0.0,
            max_per_cluster=2,
            rng=np.random.default_rng(0),
        )


def test_assign_cell_cycle_phase_preserves_rule_precedence():
    phases = assign_cell_cycle_phase(
        s_score=np.array([1.0, 0.1, -2.0, 0.0, -1.0]),
        g2m_score=np.array([0.5, 0.2, -1.0, 0.0, 0.5]),
    )

    np.testing.assert_array_equal(phases, ["S", "G2M", "G1", "S", "G2M"])


def test_gaussian_quantile_bounds_uses_median_and_population_deviation():
    bounds = gaussian_quantile_bounds(
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        min_p=0.1,
        max_p=0.9,
    )

    np.testing.assert_allclose(
        bounds,
        (1.1876123951263535, 4.8123876048736465),
        rtol=0,
        atol=1e-12,
    )


def _snapshot_store(path: str) -> dict[str, bytes]:
    root = Path(path)
    return {
        str(file.relative_to(root)): file.read_bytes()
        for file in root.rglob("*")
        if file.is_file()
    }


def _doublet_clusters(datastore, graph: ArtifactRef) -> ArtifactRef:
    return datastore.run_leiden_clustering(graph, resolution=0.5)


def _fixture_graph(datastore) -> ArtifactRef:
    graphs = datastore.list_artifacts(
        kind="connectivity_map",
        from_assay="RNA",
        scope="assay",
        complete_only=True,
    )
    assert len(graphs) == 1
    return graphs[0]


def _fixture_quality_metric(
    datastore,
    cell_selection: ArtifactRef,
    values: np.ndarray,
) -> ArtifactRef:
    metric_values = np.asarray(values, dtype=np.float64)
    planned = plan_cell_data_artifact(
        datastore.zw,
        scope="assay",
        assay="RNA",
        kind="quality_metric",
        operation="fixture_quality_metric",
        parameters={},
        inputs={"values_fingerprint": fingerprint_array(metric_values)},
        execution_options={},
        cell_selection=cell_selection,
        arrays={"values": (metric_values.shape, "f")},
    )
    write_cell_data_artifact(
        datastore.zw,
        planned,
        {"values": metric_values},
    )
    return planned.ref


def _fixture_categorical_values(
    datastore,
    cell_selection: ArtifactRef,
    values: np.ndarray,
) -> ArtifactRef:
    labels = np.asarray(values)
    fingerprint = (
        fingerprint_strings(labels.astype(str))
        if labels.dtype.kind in {"O", "S", "U"}
        else fingerprint_array(labels)
    )
    planned = plan_cell_data_artifact(
        datastore.zw,
        scope="assay",
        assay="RNA",
        kind="hto_identity",
        operation="fixture_categorical_values",
        parameters={},
        inputs={"values_fingerprint": fingerprint},
        execution_options={},
        cell_selection=cell_selection,
        arrays={"values": (labels.shape, None)},
    )
    write_cell_data_artifact(datastore.zw, planned, {"values": labels})
    return planned.ref


def _selection_values(datastore, ref: ArtifactRef) -> np.ndarray:
    return read_stored_selection_mask(
        datastore.zw,
        ref,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )


def test_select_cells_thresholds_artifact_values_without_live_metadata_writes(
    datastore_ephemeral,
) -> None:
    store = datastore_ephemeral
    store.cells.insert(
        "I",
        np.ones(store.cells.N, dtype=bool),
        overwrite=True,
        force=True,
    )
    source = store.snapshot_cell_selection()
    values = np.linspace(-1.0, 1.0, store.cells.N)
    metric = _fixture_quality_metric(store, source, values)

    drifted = np.ones(store.cells.N, dtype=bool)
    drifted[::3] = False
    store.cells.insert("I", drifted, overwrite=True, force=True)
    metadata_before = _snapshot_store(str(Path(store.zarr_loc) / "cellData"))

    selected = store.select_cells(
        metric,
        low=-0.25,
        high=0.75,
        keep_bounds=True,
    )

    np.testing.assert_array_equal(
        _selection_values(store, selected),
        (values >= -0.25) & (values <= 0.75),
    )
    status = store.inspect_artifact(selected)
    assert status.operation == "select_cells"
    assert ArtifactRef.from_dict(status.inputs["values"]) == metric
    assert ArtifactRef.from_dict(status.inputs["source_cell_selection"]) == source
    assert ArtifactRef.from_dict(status.inputs["prior_cell_selection"]) == source
    assert (
        store.select_cells(
            metric,
            low=-0.25,
            high=0.75,
            keep_bounds=True,
        )
        == selected
    )
    assert _snapshot_store(str(Path(store.zarr_loc) / "cellData")) == metadata_before


def test_select_cells_composes_only_with_a_source_subset(datastore_ephemeral) -> None:
    store = datastore_ephemeral
    source_mask = np.arange(store.cells.N) % 2 == 0
    store.cells.insert("I", source_mask, overwrite=True, force=True)
    source = store.snapshot_cell_selection()
    metric = _fixture_quality_metric(
        store,
        source,
        np.arange(int(source_mask.sum()), dtype=np.float64),
    )

    prior_mask = source_mask & (np.arange(store.cells.N) % 4 == 0)
    store.cells.insert("I", prior_mask, overwrite=True, force=True)
    prior = store.snapshot_cell_selection()
    selected = store.select_cells(metric, low=None, high=None, cell_selection=prior)
    np.testing.assert_array_equal(_selection_values(store, selected), prior_mask)

    store.cells.insert(
        "I",
        np.ones(store.cells.N, dtype=bool),
        overwrite=True,
        force=True,
    )
    superset = store.snapshot_cell_selection()
    with pytest.raises(ValueError, match="must be a subset"):
        store.select_cells(metric, cell_selection=superset)

    with pytest.raises(ValueError, match="low cannot exceed high"):
        store.select_cells(metric, low=2.0, high=1.0)
    with pytest.raises(TypeError, match="low must be a finite number"):
        store.select_cells(metric, low=True)


def test_select_cells_includes_categorical_artifact_values(
    datastore_ephemeral,
) -> None:
    store = datastore_ephemeral
    store.cells.insert(
        "I",
        np.ones(store.cells.N, dtype=bool),
        overwrite=True,
        force=True,
    )
    source = store.snapshot_cell_selection()
    labels = np.resize(
        np.asarray(["tag-b", "Negative", "tag-a", "Doublet"]),
        store.cells.N,
    )
    identities = _fixture_categorical_values(store, source, labels)

    selected = store.select_cells(identities, include=["tag-b", "tag-a"])

    np.testing.assert_array_equal(
        _selection_values(store, selected),
        np.isin(labels, ["tag-a", "tag-b"]),
    )
    assert store.inspect_artifact(selected).parameters["include"] == [
        "tag-a",
        "tag-b",
    ]
    assert (
        store.select_cells(
            identities,
            include=["tag-a", "tag-b"],
        )
        == selected
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        store.select_cells(identities, include=["tag-a"], low=0)
    with pytest.raises(TypeError, match="numeric unless include"):
        store.select_cells(identities)


def test_select_cells_rejects_lossy_categorical_include_values(
    datastore_ephemeral,
) -> None:
    store = datastore_ephemeral
    store.cells.insert(
        "I",
        np.ones(store.cells.N, dtype=bool),
        overwrite=True,
        force=True,
    )
    source = store.snapshot_cell_selection()
    integer_labels = np.resize(np.asarray([1, 2], dtype=np.int16), store.cells.N)
    values = _fixture_categorical_values(store, source, integer_labels)

    selected = store.select_cells(values, include=[1])

    np.testing.assert_array_equal(
        _selection_values(store, selected),
        integer_labels == 1,
    )
    with pytest.raises(TypeError, match="integers for an integer artifact"):
        store.select_cells(values, include=[True])
    with pytest.raises(TypeError, match="integers for an integer artifact"):
        store.select_cells(values, include=[1, "1"])


def test_selection_equality_uses_validated_immutable_fingerprints(
    datastore_ephemeral,
) -> None:
    store = datastore_ephemeral
    values = np.asarray(store.cells.fetch_all("I"), dtype=bool)
    row_ids = np.asarray(store.cells.fetch_all("ids"))
    first = store.snapshot_cell_selection()
    same_values = resolve_selection_artifact(
        store.zw,
        scope="datastore",
        kind="cell_selection",
        values=values,
        row_ids=row_ids,
        operation="fixture_equal_selection",
        parameters={},
        inputs={},
        source_column="artifact",
        invalidate_cache=True,
    )
    changed_values = values.copy()
    changed_values[0] = ~changed_values[0]
    different = resolve_selection_artifact(
        store.zw,
        scope="datastore",
        kind="cell_selection",
        values=changed_values,
        row_ids=row_ids,
        operation="fixture_changed_selection",
        parameters={},
        inputs={},
        source_column="artifact",
    )

    assert first != same_values
    assert store._selection_artifacts_match(first, same_values)
    assert not store._selection_artifacts_match(first, different)

    artifact_group(store.zw, same_values)["values"][0] = ~values[0]
    assert not store._selection_artifacts_match(first, same_values)


def test_doublet_mapping_is_query_owned_and_leaves_reference_unprojected(
    analyzed_datastore_ephemeral,
    monkeypatch,
) -> None:
    datastore = analyzed_datastore_ephemeral
    selected_connectivity = _fixture_graph(datastore)
    clusters = _doublet_clusters(datastore, selected_connectivity)
    metadata_before = _snapshot_store(str(Path(datastore.zarr_loc) / "cellData"))
    reference_projections = set(
        datastore.list_artifacts(
            kind="projection",
            from_assay="RNA",
        )
    )
    temporary_paths: list[Path] = []
    mapping_calls = 0
    original_run_mapping = type(datastore).run_mapping

    def observe_mapping(query, reference, cell_selection, **kwargs):
        nonlocal mapping_calls
        mapping_calls += 1
        assert query is not datastore
        assert isinstance(cell_selection, ArtifactRef)
        assert cell_selection.kind == "cell_selection"
        assert kwargs == {"query_assay": "RNA", "save_k": 3}
        temporary_path = Path(query.zarr_loc)
        temporary_paths.append(temporary_path)
        assert temporary_path.exists()
        before = _snapshot_store(datastore.zarr_loc)
        result = original_run_mapping(
            query,
            reference,
            cell_selection,
            **kwargs,
        )
        assert _snapshot_store(datastore.zarr_loc) == before
        return result

    monkeypatch.setattr(type(datastore), "run_mapping", observe_mapping)

    score_ref = datastore.run_doublet_detection(
        clusters,
        selected_connectivity,
        cluster_sample_fraction=0.01,
        max_cells_per_cluster=2,
        simulation_ratio=0.01,
        save_k=3,
        smoothing_t=1,
        random_seed=19,
    )

    assert mapping_calls == 1
    assert temporary_paths and all(not path.exists() for path in temporary_paths)
    assert (
        set(
            datastore.list_artifacts(
                kind="projection",
                from_assay="RNA",
            )
        )
        == reference_projections
    )
    scores = artifact_values(
        artifact_group(datastore.zw, score_ref),
        "values",
    )
    assert scores.ndim == 1
    assert (
        scores.shape
        == artifact_values(
            artifact_group(datastore.zw, clusters),
            "values",
        ).shape
    )
    assert np.all(np.isfinite(scores))
    assert "RNA_doublet_score__raw" not in datastore.cells.columns
    assert (
        _snapshot_store(str(Path(datastore.zarr_loc) / "cellData")) == metadata_before
    )

    score_status = datastore.inspect_artifact(score_ref)
    neighbors = ArtifactRef.from_dict(score_status.inputs["neighbors"])
    reference_refs = datastore.list_artifacts(
        kind="mapping_reference",
        from_assay="RNA",
        scope="assay",
        complete_only=True,
    )
    references = [datastore.get_mapping_reference(ref) for ref in reference_refs]
    matching = [
        reference for reference in references if reference.neighbors == neighbors
    ]
    assert matching
    reference = matching[-1]
    assert reference.method == "pca"
    assert reference.symphony_state is None

    assert (
        datastore.run_doublet_detection(
            clusters,
            selected_connectivity,
            cluster_sample_fraction=0.01,
            max_cells_per_cluster=2,
            simulation_ratio=0.01,
            save_k=3,
            smoothing_t=1,
            random_seed=19,
        )
        == score_ref
    )
    assert mapping_calls == 1
    assert (
        _snapshot_store(str(Path(datastore.zarr_loc) / "cellData")) == metadata_before
    )


def test_doublet_mapping_failure_removes_temporary_query_store(
    analyzed_datastore_ephemeral,
    monkeypatch,
) -> None:
    datastore = analyzed_datastore_ephemeral
    graph = _fixture_graph(datastore)
    clusters = _doublet_clusters(datastore, graph)
    selected_count = len(
        artifact_values(artifact_group(datastore.zw, clusters), "values")
    )
    metadata_before = _snapshot_store(str(Path(datastore.zarr_loc) / "cellData"))
    temporary_paths: list[Path] = []
    original_run_mapping = type(datastore).run_mapping

    def capture_mapping(query, reference, cell_selection, **kwargs):
        temporary_paths.append(Path(query.zarr_loc))
        return original_run_mapping(
            query,
            reference,
            cell_selection,
            **kwargs,
        )

    def wrong_length_score(_query, _result, *_args, **_kwargs):
        yield 0, np.zeros(selected_count - 1)

    monkeypatch.setattr(type(datastore), "run_mapping", capture_mapping)
    monkeypatch.setattr(type(datastore), "get_mapping_score", wrong_length_score)

    with pytest.raises(RuntimeError, match="selected cells"):
        datastore.run_doublet_detection(
            clusters,
            graph,
            cluster_sample_fraction=0.01,
            max_cells_per_cluster=2,
            simulation_ratio=0.01,
            save_k=3,
            smoothing_t=1,
            random_seed=23,
        )

    assert temporary_paths and all(not path.exists() for path in temporary_paths)
    assert not datastore.list_artifacts(
        kind="projection",
        from_assay="RNA",
    )
    assert "RNA_doublet_score__raw" not in datastore.cells.columns
    assert (
        _snapshot_store(str(Path(datastore.zarr_loc) / "cellData")) == metadata_before
    )


def test_doublet_detection_rejects_symphony_connectivity_chain(
    analyzed_datastore_ephemeral,
    monkeypatch,
) -> None:
    datastore = analyzed_datastore_ephemeral
    native_graph = _fixture_graph(datastore)
    clusters = _doublet_clusters(datastore, native_graph)
    reduction = resolve_native_graph_inputs(datastore.zw, native_graph).coordinates
    datastore.cells.insert(
        "doublet_batch",
        np.where(np.arange(datastore.cells.N) % 2, "a", "b"),
        overwrite=True,
    )
    correction = datastore.run_harmony(
        reduction,
        ["doublet_batch"],
        harmony_params={"nclust": 5},
    )
    ann_index = datastore.build_ann_index(
        correction,
    )
    neighbors = datastore.query_neighbors(
        ann_index,
        coordinates=correction,
        k=3,
    )
    connectivity = datastore.build_connectivity_map(neighbors)
    references_before = set(
        datastore.list_artifacts(
            kind="mapping_reference",
            from_assay="RNA",
        )
    )

    def fail_temporary_store(*_args, **_kwargs):
        raise AssertionError("Temporary query store must not be created")

    monkeypatch.setattr(
        datastore,
        "_create_temporary_datastore",
        fail_temporary_store,
    )

    with pytest.raises(
        ValueError,
        match="uncorrected PCA graph",
    ):
        datastore.run_doublet_detection(
            clusters,
            connectivity,
            simulation_ratio=0.01,
        )

    assert datastore.inspect_artifact(connectivity).complete
    assert (
        set(
            datastore.list_artifacts(
                kind="mapping_reference",
                from_assay="RNA",
            )
        )
        == references_before
    )
