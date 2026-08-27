import numpy as np
import pytest

import scarf.mapping as mapping
from scarf.datastore.datastore import DataStore
from scarf.graph.feature_projection import resolve_native_graph_inputs
from scarf.metadata.artifacts import plan_cell_data_artifact, write_cell_data_artifact
from scarf.storage.artifacts import ArtifactRef, artifact_group
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
):
    datastore = analyzed_datastore_ephemeral
    neighbors = _symphony_neighbors(datastore)
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
