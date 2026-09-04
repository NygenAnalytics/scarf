from pathlib import Path
import shutil
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import scarf.datastore._operations.mapping as mapping_operations
from scarf.datastore.datastore import DataStore, mount_datastore
from scarf.graph.feature_projection import resolve_native_graph_inputs
from scarf.mapping.features import AlignedFeatureStream
from scarf.mapping.projection import load_projection
from scarf.storage.artifacts import (
    ArtifactRef,
    ExternalArtifactRef,
    artifact_group,
    list_artifacts,
)
from scarf.storage.selections import (
    read_stored_selection_indices,
    resolve_selection_artifact,
)


def _snapshot_store(path: str) -> dict[str, bytes]:
    root = Path(path)
    return {
        str(file.relative_to(root)): file.read_bytes()
        for file in root.rglob("*")
        if file.is_file()
    }


def _fixture_graph(datastore) -> ArtifactRef:
    refs = datastore.list_artifacts(
        kind="connectivity_map",
        from_assay="RNA",
        scope="assay",
        complete_only=True,
    )
    assert len(refs) == 1
    return refs[0]


def _plain_reference(datastore):
    graph = _fixture_graph(datastore)
    raw_neighbors = datastore.inspect_artifact(graph).inputs["neighbors"]
    neighbors = ArtifactRef.from_dict(raw_neighbors)
    reference_ref = datastore.build_mapping_reference(neighbors)
    return datastore.get_mapping_reference(reference_ref)


def _symphony_reference(datastore):
    reduction = resolve_native_graph_inputs(
        datastore.zw,
        _fixture_graph(datastore),
    ).coordinates
    datastore.cells.insert(
        "mapping_batch",
        np.where(np.arange(datastore.cells.N) % 2, "a", "b"),
        overwrite=True,
    )
    correction = datastore.run_harmony(
        reduction,
        ["mapping_batch"],
        harmony_params={"nclust": 5},
    )
    ann_index = datastore.build_ann_index(correction)
    neighbors = datastore.query_neighbors(
        ann_index,
        coordinates=correction,
        k=3,
    )
    reference_ref = datastore.build_mapping_reference(neighbors)
    return datastore.get_mapping_reference(reference_ref)


def _copied_query(datastore, path: Path, *, zarr_mode: str = "r+") -> DataStore:
    shutil.copytree(datastore.zarr_loc, path)
    return DataStore(
        str(path),
        default_assay="RNA",
        zarr_mode=zarr_mode,
    )


def _query_selection_matching_reference(query, reference) -> ArtifactRef:
    values = np.asarray(
        artifact_group(reference.datastore.zw, reference.cell_selection)["values"][:],
        dtype=bool,
    )
    return resolve_selection_artifact(
        query.zw,
        scope="datastore",
        kind="cell_selection",
        values=values,
        row_ids=np.asarray(query.cells.fetch_all("ids")),
        operation="select_mapping_query",
        parameters={},
        inputs={"mapping_reference": reference.external_ref},
        source_column="mapping_reference",
    )


def _changed_files(
    before: dict[str, bytes],
    after: dict[str, bytes],
) -> set[str]:
    return {
        path
        for path in before.keys() | after.keys()
        if before.get(path) != after.get(path)
    }


def test_plain_mapping_is_query_owned_and_reuses_exact_projection(
    analyzed_datastore_ephemeral,
    tmp_path,
    monkeypatch,
):
    reference_store = analyzed_datastore_ephemeral
    reference = _plain_reference(reference_store)
    query = _copied_query(reference_store, tmp_path / "query.zarr")
    reference_before = _snapshot_store(reference_store.zarr_loc)
    query_before = _snapshot_store(query.zarr_loc)
    query_sf = query.RNA.sf
    query_scalar = query.RNA.scalar
    query_attrs = dict(query.RNA.attrs)
    query_ncounts = np.array(query.cells.fetch_all("RNA_nCounts"), copy=True)
    query_feature_columns = set(query.RNA.feats.columns)
    available_k = int(
        artifact_group(reference_store.zw, reference.neighbors)["indices"].shape[1]
    )
    project_pca = mapping_operations.project_pca

    def project_with_zero_row(values, model):
        projected = project_pca(values, model)
        projected[0] = 0
        return projected

    monkeypatch.setattr(mapping_operations, "project_pca", project_with_zero_row)
    warning_messages = []
    monkeypatch.setattr(
        "scarf.datastore._operations.mapping.logger.warning",
        warning_messages.append,
    )

    projection_ref = query.run_mapping(
        reference,
        reference.cell_selection,
        save_k=available_k + 10,
    )
    assert isinstance(projection_ref, ArtifactRef)
    result = query.get_mapping_result(projection_ref, reference=reference)

    assert _snapshot_store(reference_store.zarr_loc) == reference_before
    assert len(warning_messages) == 1
    assert "save_k" in warning_messages[0]
    assert result.reference is reference
    assert result.correction_method == "none"
    assert result.indices is None
    assert result.distances is None
    assert result.uninformative is None
    assert set(result.diagnostics) == {
        "featureCoverage",
        "queryBatchCount",
        "algorithmVariant",
        "zeroNormCellCount",
        "queryScaledDispersion",
    }
    assert result.diagnostics["algorithmVariant"] == "scaled_pca"
    assert result.diagnostics["queryBatchCount"] == 1
    assert result.diagnostics["zeroNormCellCount"] > 0
    assert query.RNA.sf == query_sf
    assert query.RNA.scalar is query_scalar
    assert dict(query.RNA.attrs) == query_attrs
    np.testing.assert_array_equal(
        query.cells.fetch_all("RNA_nCounts"),
        query_ncounts,
    )
    assert set(query.RNA.feats.columns) == query_feature_columns

    status = query.inspect_artifact(projection_ref)
    assert status.parameters == {
        "save_k": available_k,
        "missing_feature_policy": "reference_mean",
        "correction_method": "none",
    }
    assert set(status.inputs or {}) == {
        "cell_selection",
        "feature_selection",
        "selected_expression_fingerprint",
        "query_batch_fingerprint",
        "query_batch_count",
        "mapping_reference",
    }
    assert (
        ExternalArtifactRef.from_dict((status.inputs or {})["mapping_reference"])
        == reference.external_ref
    )
    lineage = query.lineage(projection_ref, references=reference)
    assert reference.external_ref in lineage.graph
    assert all(
        node["status"] is not None and node["status"].complete
        for _, node in lineage.graph.nodes(data=True)
    )
    cell_selection = ArtifactRef.from_dict((status.inputs or {})["cell_selection"])
    feature_selection = ArtifactRef.from_dict(
        (status.inputs or {})["feature_selection"]
    )
    np.testing.assert_array_equal(
        artifact_group(query.zw, cell_selection)["values"][:],
        artifact_group(query.zw, reference.cell_selection)["values"][:],
    )
    query_feature_ids = np.asarray(query.RNA.feats.fetch_all("ids")).astype(str)
    np.testing.assert_array_equal(
        artifact_group(query.zw, feature_selection)["values"][:],
        np.isin(query_feature_ids, reference.feature_ids),
    )
    feature_status = query.inspect_artifact(feature_selection)
    assert feature_status.operation == "select_mapping_overlap"
    assert feature_status.parameters == {}
    assert set(feature_status.inputs or {}) == {
        "mapping_reference",
        "all_features",
    }
    assert (
        ExternalArtifactRef.from_dict(feature_status.inputs["mapping_reference"])
        == reference.external_ref
    )
    all_features = ArtifactRef.from_dict(feature_status.inputs["all_features"])
    all_features_status = query.inspect_artifact(all_features)
    assert all_features_status.operation == "create_all_features"
    assert all_features_status.inputs == {}
    group = artifact_group(query.zw, projection_ref)
    assert set(group.array_keys()) == {"indices", "distances", "uninformative"}
    assert set(group.group_keys()) == set()
    assert set(group.attrs) == {
        "artifact_id",
        "kind",
        "provenance",
        "execution_options",
        "created_at_ns",
        "scarf_version",
        "complete",
        "diagnostics",
        "payload_fingerprint",
    }
    loaded = load_projection(
        query.zw,
        projection_ref,
        load_arrays=True,
        reference=reference,
    )
    assert loaded.indices is not None
    assert loaded.indices.shape == (result.n_cells, available_k)
    assert loaded.distances is not None
    assert loaded.distances.shape == loaded.indices.shape
    assert loaded.uninformative is not None
    assert (
        int(np.count_nonzero(loaded.uninformative))
        == result.diagnostics["zeroNormCellCount"]
    )

    changed = _changed_files(query_before, _snapshot_store(query.zarr_loc))
    assert not any("aligned" in path or "normed" in path for path in changed)

    reuse_before = _snapshot_store(query.zarr_loc)

    def fail_ann_load(*args, **kwargs):
        raise AssertionError("A reused mapping must not load or query the ANN")

    monkeypatch.setattr(
        "scarf.datastore._operations.mapping._load_reference_neighbor_query",
        fail_ann_load,
    )
    reused = query.run_mapping(
        reference,
        reference.cell_selection,
        save_k=available_k + 10,
    )
    assert reused == projection_ref
    assert _snapshot_store(query.zarr_loc) == reuse_before


def test_mapping_failure_leaves_projection_incomplete(
    analyzed_datastore_ephemeral,
    tmp_path,
    monkeypatch,
):
    reference_store = analyzed_datastore_ephemeral
    reference = _plain_reference(reference_store)
    query = _copied_query(reference_store, tmp_path / "query.zarr")
    before = set(
        list_artifacts(
            query.zw,
            scope="assay",
            assay="RNA",
            kind="projection",
        )
    )

    class FailingNeighborQuery:
        @staticmethod
        def query(_values):
            raise RuntimeError("injected ANN failure")

    monkeypatch.setattr(
        mapping_operations,
        "_load_reference_neighbor_query",
        lambda *_args, **_kwargs: FailingNeighborQuery(),
    )
    with pytest.raises(RuntimeError, match="injected ANN failure"):
        query.run_mapping(reference, reference.cell_selection)

    created = (
        set(
            list_artifacts(
                query.zw,
                scope="assay",
                assay="RNA",
                kind="projection",
            )
        )
        - before
    )
    assert len(created) == 1
    failed = query.inspect_artifact(created.pop())
    assert failed.exists
    assert not failed.complete


def test_mapping_rejects_expression_changes_before_projection_finish(
    analyzed_datastore_ephemeral,
    tmp_path,
    monkeypatch,
):
    reference_store = analyzed_datastore_ephemeral
    reference = _plain_reference(reference_store)
    query = _copied_query(reference_store, tmp_path / "query-expression-change.zarr")
    before = set(
        query.list_artifacts(
            kind="projection",
            from_assay="RNA",
        )
    )
    monkeypatch.setattr(
        AlignedFeatureStream,
        "fingerprint_live_raw_expression",
        lambda _stream: "changed-during-mapping",
    )

    with pytest.raises(ValueError, match="expression changed during mapping"):
        query.run_mapping(reference, reference.cell_selection)

    created = (
        set(
            query.list_artifacts(
                kind="projection",
                from_assay="RNA",
            )
        )
        - before
    )
    assert len(created) == 1
    assert not query.inspect_artifact(created.pop()).complete


def test_mapping_rejects_reference_handles_forged_from_a_stored_reference(
    analyzed_datastore_ephemeral,
    tmp_path,
):
    reference_store = analyzed_datastore_ephemeral
    reference = _plain_reference(reference_store)
    query = _copied_query(reference_store, tmp_path / "query-forged-reference.zarr")
    projection = query.run_mapping(reference, reference.cell_selection)
    forged = replace(
        reference,
        model=object(),  # type: ignore[arg-type]
        cell_selection=ArtifactRef(
            scope="datastore",
            assay=None,
            kind="cell_selection",
            artifact_id="f" * 64,
        ),
    )

    with pytest.raises(ValueError, match="does not match its stored artifact"):
        query.run_mapping(forged, reference.cell_selection)
    with pytest.raises(ValueError, match="does not match its stored artifact"):
        query.get_mapping_result(projection, reference=forged)
    with pytest.raises(ValueError, match="does not match its stored artifact"):
        forged.fetch_cell_column("ids")
    with pytest.raises(ValueError, match="does not match its stored artifact"):
        query.lineage(projection, references=forged)


def test_mapping_rejects_feature_axis_changes_during_stream_setup(
    analyzed_datastore_ephemeral,
    tmp_path,
    monkeypatch,
):
    reference_store = analyzed_datastore_ephemeral
    reference = _plain_reference(reference_store)
    query = _copied_query(reference_store, tmp_path / "query-axis-change.zarr")
    before = set(
        query.list_artifacts(
            kind="projection",
            from_assay="RNA",
        )
    )
    live_ids = query.RNA.feats._get_array("ids")
    original_ids = np.asarray(live_ids[:]).copy()
    changed_ids = original_ids.copy()
    changed_ids[[0, 1]] = changed_ids[[1, 0]]
    original_init = AlignedFeatureStream.__init__

    def initialize_then_change_ids(stream, *args, **kwargs):
        original_init(stream, *args, **kwargs)
        live_ids[:] = changed_ids

    monkeypatch.setattr(
        AlignedFeatureStream,
        "__init__",
        initialize_then_change_ids,
    )
    try:
        with pytest.raises(ValueError, match="identities changed during mapping setup"):
            query.run_mapping(reference, reference.cell_selection)
    finally:
        live_ids[:] = original_ids

    assert (
        set(
            query.list_artifacts(
                kind="projection",
                from_assay="RNA",
            )
        )
        == before
    )


def test_mapping_producer_returns_ref_without_loading_projection(
    analyzed_datastore_ephemeral,
    tmp_path,
    monkeypatch,
):
    reference_store = analyzed_datastore_ephemeral
    reference = _plain_reference(reference_store)
    query = _copied_query(reference_store, tmp_path / "query-load-fail.zarr")
    before = set(
        list_artifacts(
            query.zw,
            scope="assay",
            assay="RNA",
            kind="projection",
        )
    )

    def fail_load(*_args, **_kwargs):
        raise RuntimeError("injected load_projection failure")

    monkeypatch.setattr(mapping_operations, "load_projection", fail_load)
    finished = query.run_mapping(reference, reference.cell_selection)

    created = (
        set(
            list_artifacts(
                query.zw,
                scope="assay",
                assay="RNA",
                kind="projection",
            )
        )
        - before
    )
    assert created == {finished}
    status = query.inspect_artifact(finished)
    assert status.exists
    assert status.complete

    monkeypatch.undo()
    loaded = query.get_mapping_result(finished, reference=reference)
    assert loaded.ref == finished
    reused = query.run_mapping(reference, reference.cell_selection)
    assert reused == finished


def test_mapping_guards_precede_query_writes(
    analyzed_datastore_ephemeral,
    tmp_path,
):
    reference_store = analyzed_datastore_ephemeral
    reference = _plain_reference(reference_store)
    reference_before = _snapshot_store(reference_store.zarr_loc)

    with pytest.raises(ValueError, match="same physical Zarr store"):
        reference_store.run_mapping(reference, reference.cell_selection)
    assert _snapshot_store(reference_store.zarr_loc) == reference_before

    reopened = DataStore(reference_store.zarr_loc, default_assay="RNA")
    with pytest.raises(ValueError, match="same physical Zarr store"):
        reopened.run_mapping(reference, reference.cell_selection)
    assert _snapshot_store(reference_store.zarr_loc) == reference_before

    read_only = _copied_query(
        reference_store,
        tmp_path / "read-only.zarr",
        zarr_mode="r",
    )
    read_only_before = _snapshot_store(read_only.zarr_loc)
    with pytest.raises(ValueError, match="read-write query datastore"):
        read_only.run_mapping(reference, reference.cell_selection)
    assert _snapshot_store(read_only.zarr_loc) == read_only_before

    writable = _copied_query(reference_store, tmp_path / "writable.zarr")
    writable_before = _snapshot_store(writable.zarr_loc)
    with pytest.raises(ValueError, match="only supported by Symphony"):
        writable.run_mapping(
            reference,
            reference.cell_selection,
            query_batches=pd.DataFrame({"batch": ["a"]}),
        )
    assert _snapshot_store(writable.zarr_loc) == writable_before

    with pytest.raises(TypeError, match="RNA query assays"):
        writable.run_mapping(
            reference,
            reference.cell_selection,
            query_assay="assay2",
        )
    assert _snapshot_store(writable.zarr_loc) == writable_before

    writable.cells.insert(
        "empty_mapping_selection",
        np.zeros(writable.cells.N, dtype=bool),
        overwrite=True,
    )
    empty_selection = writable.snapshot_cell_selection("empty_mapping_selection")
    empty_before = _snapshot_store(writable.zarr_loc)
    with pytest.raises(ValueError, match="at least one query cell"):
        writable.run_mapping(reference, empty_selection)
    assert _snapshot_store(writable.zarr_loc) == empty_before

    had_stored_fingerprint = "dataset_fingerprint" in reference_store.RNA.attrs
    original_fingerprint = reference_store.RNA.attrs.get("dataset_fingerprint")
    reference_store.RNA.attrs["dataset_fingerprint"] = "changed"
    fingerprint_before = _snapshot_store(writable.zarr_loc)
    try:
        with pytest.raises(ValueError, match="dataset fingerprint does not match"):
            writable.run_mapping(reference, reference.cell_selection)
    finally:
        if had_stored_fingerprint:
            reference_store.RNA.attrs["dataset_fingerprint"] = original_fingerprint
        else:
            del reference_store.RNA.attrs["dataset_fingerprint"]
    assert _snapshot_store(writable.zarr_loc) == fingerprint_before


def test_separately_mounted_query_can_map_its_source_reference(
    analyzed_datastore_ephemeral,
    tmp_path,
):
    reference_store = analyzed_datastore_ephemeral
    reference = _plain_reference(reference_store)
    reference_before = _snapshot_store(reference_store.zarr_loc)
    read_only_reference_store = DataStore(
        reference_store.zarr_loc,
        default_assay="RNA",
        zarr_mode="r",
    )
    read_only_reference = read_only_reference_store.get_mapping_reference(reference.ref)
    query = mount_datastore(
        reference_store.zarr_loc,
        at=str(tmp_path / "mounted-query.zarr"),
        default_assay="RNA",
    )
    query_selection = _query_selection_matching_reference(
        query,
        read_only_reference,
    )

    result_ref = query.run_mapping(
        read_only_reference,
        query_selection,
    )
    result = query.get_mapping_result(
        result_ref,
        reference=read_only_reference,
    )

    assert result.reference is read_only_reference
    assert result.n_cells == read_only_reference.selected_cell_count
    assert _snapshot_store(reference_store.zarr_loc) == reference_before


def test_query_projection_reproduces_stored_reference_coordinates(
    analyzed_datastore_ephemeral,
):
    reference_store = analyzed_datastore_ephemeral
    reference = _plain_reference(reference_store)
    stored = np.asarray(
        artifact_group(reference_store.zw, reference.reduction)["data"][:],
        dtype=np.float64,
    )

    stream = AlignedFeatureStream(
        query_assay=reference_store.RNA,
        query_cell_indices=read_stored_selection_indices(
            reference_store.zw,
            reference.cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ),
        reference_feature_ids=reference.feature_ids,
        reference_normalized_means=reference.model.feature_means,
        reference_normalization_parameters=reference.normalization_parameters,
        missing_feature_policy="error",
        resources=reference_store.resources,
    )
    projected = np.vstack(
        [
            mapping_operations.project_pca(block.values, reference.model)
            for block in stream
        ]
    )

    assert stream.feature_coverage == 1.0
    assert projected.shape == stored.shape
    np.testing.assert_allclose(
        projected,
        stored,
        rtol=0,
        atol=1e-4 * float(np.abs(stored).max()),
    )


def test_self_mapping_recovers_the_reference_graph_and_labels(
    analyzed_datastore_ephemeral,
    tmp_path,
):
    reference_store = analyzed_datastore_ephemeral
    reference_store.cells.insert(
        "mapping_labels",
        np.array([f"c{index % 7}" for index in range(reference_store.cells.N)]),
        overwrite=True,
    )
    reference = _plain_reference(reference_store)
    graph = artifact_group(reference_store.zw, reference.neighbors)
    reference_indices = np.asarray(graph["indices"][:])
    reference_distances = np.asarray(graph["distances"][:], dtype=np.float64)
    available_k = int(reference_indices.shape[1])
    assert not (reference_indices[:, 0] == np.arange(len(reference_indices))).all()

    query = mount_datastore(
        reference_store.zarr_loc,
        at=str(tmp_path / "self-query.zarr"),
        default_assay="RNA",
    )
    query_selection = _query_selection_matching_reference(query, reference)
    result = query.run_mapping(
        reference,
        query_selection,
        save_k=available_k,
    )
    loaded = query.get_mapping_result(result, reference=reference, load_arrays=True)
    indices = loaded.indices
    distances = loaded.distances
    assert indices is not None and distances is not None

    positions = np.arange(loaded.n_cells)
    assert loaded.n_cells == reference.selected_cell_count
    assert (indices[:, 0] == positions).mean() > 0.95
    assert float(np.median(distances[:, 0])) < 1e-4
    # The reference PCA is fitted on z-scored features, so a query that is the
    # reference has to disperse exactly like it. This anchors the diagnostic:
    # values well below 1 mean the query occupies a narrower region.
    assert loaded.diagnostics["queryScaledDispersion"] == pytest.approx(
        1.0,
        abs=0.02,
    )

    agreement = indices[:, 1:] == reference_indices[:, : available_k - 1]
    assert agreement.mean() > 0.9
    np.testing.assert_allclose(
        distances[:, 1:][agreement],
        reference_distances[:, : available_k - 1][agreement],
        rtol=1e-4,
        atol=1e-4,
    )
    assert len(np.unique(indices)) > 0.9 * reference.selected_cell_count

    transferred = query.get_target_classes(
        result,
        reference_class_group="mapping_labels",
        reference=reference,
        threshold_fraction=0.6,
    ).to_numpy()
    query_rows = read_stored_selection_indices(
        query.zw,
        query_selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    known = np.asarray(query.cells.fetch_all("mapping_labels"))[query_rows]
    assert (transferred == known).mean() > 0.95

    evidence = query.get_target_label_evidence(
        result,
        reference_class_group="mapping_labels",
        reference=reference,
    )
    assert float(np.median(evidence["referenceDistancePercentile"])) == 0.0
    assert float(np.median(evidence["voteFraction"])) > 0.99
    assert not evidence["isUnknown"].mean() > 0.05


def test_symphony_mapping_validates_batches_and_persists_diagnostics(
    analyzed_datastore_ephemeral,
    tmp_path,
    monkeypatch,
):
    reference_store = analyzed_datastore_ephemeral
    reference = _symphony_reference(reference_store)
    query = _copied_query(reference_store, tmp_path / "query.zarr")
    n_cells = reference.selected_cell_count

    invalid_frames = (
        pd.DataFrame(index=np.arange(n_cells)),
        pd.DataFrame({"batch": ["a"] * (n_cells - 1)}),
        pd.DataFrame(
            np.column_stack((np.zeros(n_cells), np.ones(n_cells))),
            columns=["batch", "batch"],
        ),
        pd.DataFrame({"batch": [None] + ["a"] * (n_cells - 1)}),
    )
    for batches in invalid_frames:
        before = _snapshot_store(query.zarr_loc)
        with pytest.raises(ValueError):
            query.run_mapping(
                reference,
                reference.cell_selection,
                query_batches=batches,
            )
        assert _snapshot_store(query.zarr_loc) == before

    batches = pd.DataFrame(
        {
            "donor": np.where(np.arange(n_cells) % 2, "a", "b"),
            "library": np.arange(n_cells) % 3,
        }
    )
    accumulated_rows = 0
    original_accumulate = mapping_operations.accumulate_sufficient_statistics

    def observe_accumulation(
        counts,
        sums,
        coordinates,
        assignments,
        batch_codes,
    ):
        nonlocal accumulated_rows
        assert not mapping_operations.zero_norm_rows(coordinates).any()
        accumulated_rows += len(coordinates)
        return original_accumulate(
            counts,
            sums,
            coordinates,
            assignments,
            batch_codes,
        )

    monkeypatch.setattr(
        mapping_operations,
        "accumulate_sufficient_statistics",
        observe_accumulation,
    )
    reference_before = _snapshot_store(reference_store.zarr_loc)
    result_ref = query.run_mapping(
        reference,
        reference.cell_selection,
        query_batches=batches,
    )
    result = query.get_mapping_result(result_ref, reference=reference)

    assert _snapshot_store(reference_store.zarr_loc) == reference_before
    assert result.reference is reference
    assert result.correction_method == "symphony"
    assert result.diagnostics == {
        "featureCoverage": 1.0,
        "queryBatchCount": 6,
        "algorithmVariant": "symphony",
        "zeroNormCellCount": result.diagnostics["zeroNormCellCount"],
        "queryScaledDispersion": result.diagnostics["queryScaledDispersion"],
    }
    assert accumulated_rows == n_cells - result.diagnostics["zeroNormCellCount"]
    loaded = load_projection(query.zw, result_ref, reference=reference)
    assert loaded.diagnostics == result.diagnostics
    reused = query.run_mapping(
        reference,
        reference.cell_selection,
        query_batches=batches.copy(),
    )
    assert reused == result_ref

    omitted_ref = query.run_mapping(reference, reference.cell_selection)
    omitted = query.get_mapping_result(omitted_ref, reference=reference)
    assert omitted.diagnostics["queryBatchCount"] == 1


def test_projection_cache_tracks_counts_and_exact_references(
    analyzed_datastore_ephemeral,
    tmp_path,
):
    reference_store = analyzed_datastore_ephemeral
    first_reference = _plain_reference(reference_store)
    second_reference = reference_store.get_mapping_reference(
        reference_store.build_mapping_reference(
            first_reference.neighbors,
            invalidate_cache=True,
        )
    )
    query = _copied_query(reference_store, tmp_path / "query.zarr")

    first = query.run_mapping(first_reference, first_reference.cell_selection)
    second = query.run_mapping(second_reference, second_reference.cell_selection)
    assert first != second
    assert load_projection(query.zw, first, reference=first_reference).ref == first
    assert load_projection(query.zw, second, reference=second_reference).ref == second

    newest = query.run_mapping(
        first_reference,
        first_reference.cell_selection,
        invalidate_cache=True,
    )
    assert newest != first
    assert load_projection(query.zw, newest, reference=first_reference).ref == (newest)
    assert load_projection(query.zw, first, reference=first_reference).ref == first

    backing = query.RNA.rawData._backing
    selected_row = int(
        read_stored_selection_indices(
            query.zw,
            first_reference.cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )[0]
    )
    original = int(backing[selected_row, 0])
    backing[selected_row, 0] = original + 1
    changed_counts = query.run_mapping(
        first_reference,
        first_reference.cell_selection,
    )
    backing[selected_row, 0] = original + 2
    changed_again = query.run_mapping(
        first_reference,
        first_reference.cell_selection,
    )
    assert changed_again != changed_counts
