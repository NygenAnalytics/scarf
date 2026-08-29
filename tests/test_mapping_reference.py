import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

import scarf.mapping as mapping
from scarf.datastore.datastore import DataStore
from scarf.graph.feature_projection import resolve_native_graph_inputs
from scarf.metadata.artifacts import plan_cell_data_artifact, write_cell_data_artifact
from scarf.storage.artifacts import ArtifactRef, artifact_group, artifact_path
from scarf.storage.selections import read_stored_selection_indices


_COMMON_ARRAYS = {
    "feature_ids",
    "feature_means",
    "feature_scales",
    "loadings",
    "reference_distance_quantiles",
    "reference_distance_values",
}
_SYMPHONY_ARRAYS = {
    "centroids",
    "raw_centroids",
    "corrected_centroids",
    "cluster_mass",
    "sigma",
}


def test_embedded_mapping_reference_helpers_are_not_public():
    for name in (
        "LATEST_MAPPING_REFERENCE_ATTRIBUTE",
        "MAPPING_REFERENCE_GROUP",
        "MAPPING_REFERENCES_GROUP",
        "load_mapping_reference",
        "mapping_reference_hash",
        "persist_mapping_reference",
        "resolve_mapping_reference_group",
        "validate_mapping_reference_artifact",
    ):
        assert name not in mapping.__all__
        assert not hasattr(mapping, name)
    assert not hasattr(mapping.MappingReference, "map_query")


def _selected_neighbors(datastore):
    graphs = datastore.list_artifacts(
        kind="connectivity_map",
        from_assay="RNA",
        scope="assay",
        complete_only=True,
    )
    assert len(graphs) == 1
    raw_neighbors = datastore.inspect_artifact(graphs[0]).inputs["neighbors"]
    return ArtifactRef.from_dict(raw_neighbors)


def _symphony_neighbors(datastore):
    graphs = datastore.list_artifacts(
        kind="connectivity_map",
        from_assay="RNA",
        scope="assay",
        complete_only=True,
    )
    assert len(graphs) == 1
    reduction = resolve_native_graph_inputs(datastore.zw, graphs[0]).coordinates
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
    ann_index = datastore.build_ann_index(
        correction,
    )
    return datastore.query_neighbors(
        ann_index,
        coordinates=correction,
        k=3,
    )


def _mapping_reference_source_paths(datastore, neighbors):
    neighbor_inputs = datastore.inspect_artifact(neighbors).inputs
    coordinates = ArtifactRef.from_dict(neighbor_inputs["coordinates"])
    ann_index = ArtifactRef.from_dict(neighbor_inputs["ann_index"])
    paths = {
        f"{artifact_path(neighbors)}/indices",
        f"{artifact_path(neighbors)}/distances",
        f"{artifact_path(ann_index)}/ann_idx_bytes",
        f"{artifact_path(coordinates)}/data",
    }
    if coordinates.kind == "batch_correction":
        reduction = ArtifactRef.from_dict(
            datastore.inspect_artifact(coordinates).inputs["reduction"]
        )
        paths.update(
            {
                f"{artifact_path(coordinates)}/{name}"
                for name in (
                    "centroids",
                    "raw_centroids",
                    "corrected_centroids",
                    "cluster_mass",
                    "sigma",
                )
            }
        )
    else:
        reduction = coordinates
    reduction_inputs = datastore.inspect_artifact(reduction).inputs
    scaling = ArtifactRef.from_dict(reduction_inputs["feature_scaling"])
    normalized = ArtifactRef.from_dict(reduction_inputs["normalized"])
    normalized_inputs = datastore.inspect_artifact(normalized).inputs
    cell_selection = ArtifactRef.from_dict(normalized_inputs["cell_selection"])
    feature_selection = ArtifactRef.from_dict(normalized_inputs["feature_selection"])
    paths.update(
        {
            f"{artifact_path(reduction)}/data",
            f"{artifact_path(reduction)}/loadings",
            f"{artifact_path(scaling)}/mean",
            f"{artifact_path(scaling)}/scale",
            f"{artifact_path(cell_selection)}/values",
            f"{artifact_path(feature_selection)}/values",
            f"{neighbors.assay}/featureData/ids",
        }
    )
    return paths


def _reject_full_array_reads(monkeypatch, paths):
    original_getitem = zarr.Array.__getitem__
    original_array = zarr.Array.__array__
    row_spans = {path: [] for path in paths}

    def axis_span(item, length):
        if isinstance(item, slice):
            start, stop, step = item.indices(length)
            return len(range(start, stop, step))
        if isinstance(item, (int, np.integer)):
            return 1
        return length

    def guarded_getitem(array, key):
        if array.path in paths:
            if key is Ellipsis:
                raise AssertionError(
                    f"mapping-reference publication materialized {array.path}"
                )
            first_axis = key[0] if isinstance(key, tuple) else key
            span = axis_span(first_axis, int(array.shape[0]))
            row_spans[array.path].append(span)
            if array.ndim == 2 and int(array.shape[0]) > 1_000 and span > 1_000:
                raise AssertionError(
                    f"mapping-reference publication read {span} rows from "
                    f"{array.path} in one block"
                )
        return original_getitem(array, key)

    def guarded_array(array, dtype=None, copy=None):
        if array.path in paths:
            raise AssertionError(
                f"mapping-reference publication implicitly materialized {array.path}"
            )
        return original_array(array, dtype=dtype, copy=copy)

    monkeypatch.setattr(zarr.Array, "__getitem__", guarded_getitem)
    monkeypatch.setattr(zarr.Array, "__array__", guarded_array)
    return row_spans


def test_plain_mapping_reference_packages_and_loads_existing_chain(
    analyzed_datastore_ephemeral,
):
    datastore = analyzed_datastore_ephemeral
    neighbors = _selected_neighbors(datastore)
    before = set(datastore.list_artifacts(from_assay="RNA"))
    reference_ref = datastore.build_mapping_reference(neighbors)
    assert isinstance(reference_ref, ArtifactRef)
    reference = datastore.get_mapping_reference(reference_ref)

    assert reference.method == "pca"
    assert reference.symphony_state is None
    assert reference.neighbors == neighbors
    assert not hasattr(reference, "feature_key")
    assert reference.dataset_fingerprint == datastore._calculate_dataset_fingerprint(
        "RNA"
    )
    selected_cells = read_stored_selection_indices(
        datastore.zw,
        reference.cell_selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    assert reference.selected_cell_count == len(selected_cells)
    assert reference.size_factor > 0
    assert reference.ann_metric in {"l2", "cosine"}

    group = artifact_group(datastore.zw, reference.ref)
    assert "feature_key" not in group.attrs["reference_metadata"]
    assert set(group.array_keys()) == _COMMON_ARRAYS
    status = datastore.inspect_artifact(reference.ref)
    assert status.parameters == {"method": "pca"}
    assert set((status.inputs or {})) == {
        "reduction",
        "ann_index",
        "neighbors",
        "cell_selection",
        "feature_selection",
    }
    created = set(datastore.list_artifacts(from_assay="RNA")) - before
    assert created == {reference.ref}

    assert datastore.get_mapping_reference(reference.ref).ref == reference.ref

    expected_layout = np.column_stack(
        (
            np.arange(len(selected_cells), dtype=np.float64),
            -np.arange(len(selected_cells), dtype=np.float64),
        )
    )
    planned_layout = plan_cell_data_artifact(
        datastore.zw,
        scope="assay",
        assay="RNA",
        kind="embedding",
        operation="manual_reference_embedding",
        parameters={},
        inputs={},
        execution_options={},
        cell_selection=reference.cell_selection,
        arrays={"values": (expected_layout.shape, "f")},
    )
    write_cell_data_artifact(
        datastore.zw,
        planned_layout,
        {"values": expected_layout},
    )
    np.testing.assert_array_equal(
        reference.fetch_cell_column("ids"),
        np.asarray(datastore.cells.fetch_all("ids"))[selected_cells],
    )
    np.testing.assert_array_equal(
        reference.fetch_layout(planned_layout.ref),
        expected_layout,
    )


def test_symphony_mapping_reference_has_conditional_state_and_read_only_reload(
    analyzed_datastore_ephemeral,
    monkeypatch,
):
    datastore = analyzed_datastore_ephemeral
    neighbors = _symphony_neighbors(datastore)
    _reject_full_array_reads(
        monkeypatch,
        _mapping_reference_source_paths(datastore, neighbors),
    )
    reference_ref = datastore.build_mapping_reference(neighbors)
    reference = datastore.get_mapping_reference(reference_ref)

    assert reference.method == "symphony"
    assert reference.symphony_state is not None
    assert reference.batch_correction is not None
    assert reference.symphony_state.n_dims == reference.model.n_dims
    assert reference.metadata["batch_columns"] == ["mapping_batch"]
    assert reference.metadata["harmony_parameters"]["nclust"] == 5

    group = artifact_group(datastore.zw, reference.ref)
    assert set(group.array_keys()) == _COMMON_ARRAYS | _SYMPHONY_ARRAYS
    assert set(group.attrs) == {
        "artifact_id",
        "kind",
        "provenance",
        "execution_options",
        "created_at_ns",
        "scarf_version",
        "complete",
        "reference_metadata",
        "payload_fingerprint",
    }
    status = datastore.inspect_artifact(reference.ref)
    assert status.parameters == {"method": "symphony"}
    assert set((status.inputs or {})) == {
        "reduction",
        "batch_correction",
        "ann_index",
        "neighbors",
        "cell_selection",
        "feature_selection",
    }

    read_only = DataStore(
        datastore.zarr_loc,
        default_assay="RNA",
        zarr_mode="r",
    )
    loaded = read_only.get_mapping_reference(reference.ref)
    assert loaded.ref == reference.ref
    assert loaded.symphony_state is not None
    np.testing.assert_array_equal(loaded.feature_ids, reference.feature_ids)
    np.testing.assert_allclose(
        loaded.symphony_state.corrected_centroids,
        reference.symphony_state.corrected_centroids,
    )


def test_loaded_mapping_reference_is_deeply_immutable(
    analyzed_datastore_ephemeral,
):
    datastore = analyzed_datastore_ephemeral
    reference = datastore.get_mapping_reference(
        datastore.build_mapping_reference(_selected_neighbors(datastore))
    )

    arrays = (
        reference.model.feature_means,
        reference.model.feature_scales,
        reference.model.loadings,
        reference.feature_ids,
        reference.reference_distance_quantiles,
        reference.reference_distance_values,
    )
    for values in arrays:
        assert not values.flags.writeable
        with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
            values.flags.writeable = True

    with pytest.raises(TypeError):
        reference.metadata["method"] = "symphony"  # type: ignore[index]
    batch_columns = reference.metadata.get("batch_columns")
    if batch_columns is not None:
        assert batch_columns == ["mapping_batch"]
        assert not batch_columns != ["mapping_batch"]
        assert ["mapping_batch"] == batch_columns
        assert not ["mapping_batch"] != batch_columns
    normalization = reference.normalization_parameters
    normalization["size_factor"] = -1
    assert reference.size_factor > 0


def test_mapping_reference_binding_and_publication_validation_are_blockwise(
    analyzed_datastore_ephemeral,
    monkeypatch,
):
    import scarf.mapping.artifact as mapping_artifact

    datastore = analyzed_datastore_ephemeral
    neighbors = _selected_neighbors(datastore)
    reference_ref = datastore.build_mapping_reference(neighbors)
    reference = datastore.get_mapping_reference(reference_ref)
    _reject_full_array_reads(
        monkeypatch,
        _mapping_reference_source_paths(datastore, neighbors),
    )

    def fail_materialization(*args, **kwargs):
        raise AssertionError("mapping-reference validation materialized an array")

    monkeypatch.setattr(mapping_artifact, "_values", fail_materialization)
    assert mapping_artifact.validate_mapping_reference_binding(reference) is reference

    replacement = datastore.build_mapping_reference(
        neighbors,
        invalidate_cache=True,
    )
    assert replacement != reference_ref
    assert datastore.inspect_artifact(replacement).complete


def test_mapping_reference_source_streaming_uses_bounded_explicit_slices(
    monkeypatch,
):
    import scarf.mapping.artifact as mapping_artifact

    root = zarr.open_group(store=MemoryStore(), mode="w")
    source_group = root.create_group("sources")
    target_group = root.create_group("reference")
    n_features = 10_001
    n_dims = 2
    feature_means = source_group.create_array(
        "feature_means",
        data=np.linspace(0.0, 1.0, n_features, dtype=np.float64),
        chunks=(1_000,),
    )
    feature_scales = source_group.create_array(
        "feature_scales",
        data=np.ones(n_features, dtype=np.float64),
        chunks=(1_000,),
    )
    loadings = source_group.create_array(
        "loadings",
        data=np.ones((n_features, n_dims), dtype=np.float64),
        chunks=(1_000, n_dims),
    )
    sources = {
        feature_means.path,
        feature_scales.path,
        loadings.path,
    }
    row_spans = _reject_full_array_reads(monkeypatch, sources)
    feature_ids = np.arange(n_features).astype(str)
    metadata = {"method": "pca"}
    quantiles = np.asarray([0.0, 1.0], dtype=np.float64)
    distance_values = np.asarray([0.5, 1.5], dtype=np.float64)

    assert mapping_artifact.validate_mapping_reference_sources(
        feature_means=feature_means,
        feature_scales=feature_scales,
        loadings=loadings,
        symphony_sources=None,
    ) == (n_features, n_dims)
    source_fingerprint = mapping_artifact.mapping_reference_source_fingerprint(
        feature_means=feature_means,
        feature_scales=feature_scales,
        loadings=loadings,
        symphony_sources=None,
    )
    mapping_artifact.write_artifact_mapping_reference_from_sources(
        target_group,
        feature_means=feature_means,
        feature_scales=feature_scales,
        loadings=loadings,
        symphony_sources=None,
        feature_ids=feature_ids,
        metadata=metadata,
        reference_distance_quantiles=quantiles,
        reference_distance_values=distance_values,
    )
    assert mapping_artifact.mapping_reference_payload_matches_sources(
        target_group,
        feature_means=feature_means,
        feature_scales=feature_scales,
        loadings=loadings,
        symphony_sources=None,
        feature_ids=feature_ids,
        metadata=metadata,
        reference_distance_quantiles=quantiles,
        reference_distance_values=distance_values,
        expected_source_fingerprint=source_fingerprint,
    )
    assert row_spans[feature_means.path]
    assert max(row_spans[feature_means.path]) <= 10_000
    assert row_spans[feature_scales.path]
    assert max(row_spans[feature_scales.path]) <= 10_000
    assert row_spans[loadings.path]
    assert max(row_spans[loadings.path]) <= 1_000


def test_mapping_reference_rejects_valid_shaped_payload_tampering(
    analyzed_datastore_ephemeral,
):
    datastore = analyzed_datastore_ephemeral
    reference = datastore.get_mapping_reference(
        datastore.build_mapping_reference(_selected_neighbors(datastore))
    )
    group = artifact_group(datastore.zw, reference.ref)
    loadings = group["loadings"]
    loadings[0, 0] = float(loadings[0, 0]) + 0.25

    with pytest.raises(ValueError, match="PCA model changed from its inputs"):
        datastore.get_mapping_reference(reference.ref)

    replacement = datastore.build_mapping_reference(reference.neighbors)
    assert replacement != reference.ref


def test_mapping_reference_binds_distance_summary_to_neighbor_input(
    analyzed_datastore_ephemeral,
):
    import scarf.mapping.artifact as mapping_artifact

    datastore = analyzed_datastore_ephemeral
    reference_ref = datastore.build_mapping_reference(_selected_neighbors(datastore))
    group = artifact_group(datastore.zw, reference_ref)
    values = group["reference_distance_values"]
    values[:] = np.asarray(values[:], dtype=np.float64) + 0.25
    metadata = dict(group.attrs["reference_metadata"])
    group.attrs["payload_fingerprint"] = mapping_artifact._payload_fingerprint(
        group,
        metadata["method"],
        metadata,
    )

    with pytest.raises(ValueError, match="changed from its neighbor input"):
        datastore.get_mapping_reference(reference_ref)


def test_mapping_reference_rejects_duplicate_selected_feature_ids(
    analyzed_datastore_ephemeral,
    monkeypatch,
):
    import scarf.datastore._operations.mapping_reference as reference_operations

    datastore = analyzed_datastore_ephemeral
    original = reference_operations._selected_feature_ids

    def duplicate_feature_ids(*args, **kwargs):
        feature_ids = np.array(original(*args, **kwargs), copy=True)
        feature_ids[1] = feature_ids[0]
        return feature_ids

    monkeypatch.setattr(
        reference_operations,
        "_selected_feature_ids",
        duplicate_feature_ids,
    )
    with pytest.raises(ValueError, match="feature IDs must be unique"):
        datastore.build_mapping_reference(_selected_neighbors(datastore))


def test_mapping_reference_validates_payload_before_finish(
    analyzed_datastore_ephemeral,
    monkeypatch,
):
    import scarf.datastore._operations.mapping_reference as reference_operations

    datastore = analyzed_datastore_ephemeral
    neighbors = _selected_neighbors(datastore)
    before = set(
        datastore.list_artifacts(
            kind="mapping_reference",
            from_assay="RNA",
        )
    )
    original_writer = reference_operations.write_artifact_mapping_reference_from_sources

    def corrupt_written_reference(group, *args, **kwargs):
        original_writer(group, *args, **kwargs)
        scales = group["feature_scales"]
        scales[0] = float(scales[0]) + 0.25

    monkeypatch.setattr(
        reference_operations,
        "write_artifact_mapping_reference_from_sources",
        corrupt_written_reference,
    )
    with pytest.raises(ValueError, match="reuse contract"):
        datastore.build_mapping_reference(neighbors, invalidate_cache=True)

    created = (
        set(
            datastore.list_artifacts(
                kind="mapping_reference",
                from_assay="RNA",
            )
        )
        - before
    )
    assert len(created) == 1
    assert not datastore.inspect_artifact(created.pop()).complete


def test_mapping_reference_rejects_source_mutation_during_publication(
    analyzed_datastore_ephemeral,
    monkeypatch,
):
    import scarf.datastore._operations.mapping_reference as reference_operations

    datastore = analyzed_datastore_ephemeral
    neighbors = _selected_neighbors(datastore)
    before = set(
        datastore.list_artifacts(
            kind="mapping_reference",
            from_assay="RNA",
        )
    )
    original_writer = reference_operations.write_artifact_mapping_reference_from_sources

    def mutate_source_then_write(group, *args, **kwargs):
        source = kwargs["loadings"]
        source[0, 0] = float(source[0, 0]) + 0.25
        original_writer(group, *args, **kwargs)

    monkeypatch.setattr(
        reference_operations,
        "write_artifact_mapping_reference_from_sources",
        mutate_source_then_write,
    )

    with pytest.raises(ValueError, match="reuse contract"):
        datastore.build_mapping_reference(neighbors, invalidate_cache=True)

    created = (
        set(
            datastore.list_artifacts(
                kind="mapping_reference",
                from_assay="RNA",
            )
        )
        - before
    )
    assert len(created) == 1
    assert not datastore.inspect_artifact(created.pop()).complete


def test_mapping_reference_rejects_corrupt_neighbor_payload_on_build_and_load(
    analyzed_datastore_ephemeral,
):
    datastore = analyzed_datastore_ephemeral
    neighbors = _selected_neighbors(datastore)
    reference_ref = datastore.build_mapping_reference(neighbors)
    neighbor_group = artifact_group(datastore.zw, neighbors)
    n_cells = int(neighbor_group["indices"].shape[0])
    del neighbor_group["distances"]
    neighbor_group.create_array(
        "distances",
        data=np.zeros(n_cells, dtype=np.float32),
        chunks=(max(1, min(n_cells, 10)),),
    )

    with pytest.raises(ValueError, match="neighbor distances are invalid"):
        datastore.get_mapping_reference(reference_ref)
    with pytest.raises(ValueError, match="stored dimensions"):
        datastore.build_mapping_reference(neighbors, invalidate_cache=True)


def test_mapping_reference_rejects_corrupt_ann_payload_on_build_and_load(
    analyzed_datastore_ephemeral,
):
    datastore = analyzed_datastore_ephemeral
    neighbors = _selected_neighbors(datastore)
    reference = datastore.get_mapping_reference(
        datastore.build_mapping_reference(neighbors)
    )
    ann_group = artifact_group(datastore.zw, reference.ann_index)
    payload = ann_group["ann_idx_bytes"]
    payload[0] = np.uint8(int(payload[0]) ^ 1)

    with pytest.raises(ValueError, match="ANN index payload is invalid"):
        datastore.get_mapping_reference(reference.ref)
    with pytest.raises(ValueError, match="payload digest"):
        datastore.build_mapping_reference(neighbors, invalidate_cache=True)


def test_mapping_reference_rejects_invalid_chain_contract(
    analyzed_datastore_ephemeral,
):
    datastore = analyzed_datastore_ephemeral
    neighbors = _selected_neighbors(datastore)
    neighbors_status = datastore.inspect_artifact(neighbors)
    assert neighbors_status.inputs is not None
    ann_index = neighbors_status.inputs["ann_index"]
    ann_ref = type(neighbors).from_dict(ann_index)
    ann_group = artifact_group(datastore.zw, ann_ref)
    provenance = dict(ann_group.attrs["provenance"])
    parameters = dict(provenance["parameters"])
    parameters["ann_metric"] = "ip"
    provenance["parameters"] = parameters
    ann_group.attrs["provenance"] = provenance

    with pytest.raises(ValueError, match="only l2 and cosine"):
        datastore.build_mapping_reference(neighbors)


def test_mapping_reference_rejects_array_attributes(
    analyzed_datastore_ephemeral,
):
    datastore = analyzed_datastore_ephemeral
    reference_ref = datastore.build_mapping_reference(_selected_neighbors(datastore))
    artifact_group(datastore.zw, reference_ref)["loadings"].attrs["schema_version"] = 1

    with pytest.raises(ValueError, match="array attributes"):
        datastore.get_mapping_reference(reference_ref)


def test_mapping_reference_rejects_unsupported_normalization_at_build_time(
    analyzed_datastore_ephemeral,
):
    datastore = analyzed_datastore_ephemeral
    neighbors = _selected_neighbors(datastore)
    coordinates = ArtifactRef.from_dict(
        datastore.inspect_artifact(neighbors).inputs["coordinates"]
    )
    normalized = ArtifactRef.from_dict(
        datastore.inspect_artifact(coordinates).inputs["normalized"]
    )
    group = artifact_group(datastore.zw, normalized)
    provenance = dict(group.attrs["provenance"])
    parameters = dict(provenance["parameters"])
    parameters["normalization_method"] = "unsupported"
    provenance["parameters"] = parameters
    group.attrs["provenance"] = provenance

    with pytest.raises(ValueError, match="Unsupported reference normalization"):
        datastore.build_mapping_reference(neighbors)


def test_mapping_reference_rejects_incomplete_contract(
    analyzed_datastore_ephemeral,
):
    datastore = analyzed_datastore_ephemeral
    neighbors = _selected_neighbors(datastore)
    reference_ref = datastore.build_mapping_reference(neighbors)
    group = artifact_group(datastore.zw, reference_ref)
    del group["feature_scales"]
    with pytest.raises(ValueError, match="build_mapping_reference\\(neighbors\\)"):
        datastore.get_mapping_reference(reference_ref)


def test_mapping_reference_validates_live_dataset_fingerprint(
    analyzed_datastore_ephemeral,
):
    datastore = analyzed_datastore_ephemeral
    reference_ref = datastore.build_mapping_reference(_selected_neighbors(datastore))
    datastore.RNA.attrs["dataset_fingerprint"] = "changed"

    with pytest.raises(ValueError, match="dataset fingerprint"):
        datastore.get_mapping_reference(reference_ref)


def test_mapping_reference_rejects_nonmonotonic_distance_summary(
    analyzed_datastore_ephemeral,
):
    datastore = analyzed_datastore_ephemeral
    reference_ref = datastore.build_mapping_reference(_selected_neighbors(datastore))
    values = artifact_group(datastore.zw, reference_ref)["reference_distance_values"]
    values[:] = np.linspace(1.0, 0.0, values.shape[0])

    with pytest.raises(ValueError, match="distance summary"):
        datastore.get_mapping_reference(reference_ref)
