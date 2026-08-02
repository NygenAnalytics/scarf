import numpy as np
import pytest

import scarf.mapping as mapping
from scarf.datastore.datastore import DataStore
from scarf.storage.artifacts import artifact_group


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
    state = datastore.get_assay_state("RNA")
    assert state is not None
    assert state.neighbors is not None
    return state.neighbors


def _symphony_neighbors(datastore):
    state = datastore.get_assay_state("RNA")
    assert state is not None
    assert state.reduction is not None
    datastore.cells.insert(
        "mapping_batch",
        np.where(np.arange(datastore.cells.N) % 2, "a", "b"),
        overwrite=True,
    )
    correction = datastore.run_harmony(
        ["mapping_batch"],
        state.reduction,
        harmony_params={"nclust": 5},
        update_state=False,
    )
    ann_index = datastore.build_ann_index(
        correction,
        update_state=False,
    )
    return datastore.query_neighbors(
        ann_index,
        coordinates=correction,
        k=3,
        update_state=False,
    )


def test_plain_mapping_reference_packages_and_loads_existing_chain(
    analyzed_datastore_ephemeral,
):
    datastore = analyzed_datastore_ephemeral
    neighbors = _selected_neighbors(datastore)
    before = set(datastore.list_artifacts(from_assay="RNA"))
    reference = datastore.build_mapping_reference(neighbors)

    assert reference.method == "pca"
    assert reference.symphony_state is None
    assert reference.neighbors == neighbors
    assert reference.dataset_fingerprint == datastore.RNA.attrs["dataset_fingerprint"]
    assert reference.selected_cell_count == len(
        datastore.cells.fetch("ids", key=reference.cell_key)
    )
    assert reference.size_factor > 0
    assert reference.ann_metric in {"l2", "cosine"}

    group = artifact_group(datastore.zw, reference.ref)
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

    state = datastore.get_assay_state("RNA")
    assert state is not None
    assert state.neighbors == neighbors
    assert state.named_results["mapping_reference"] == reference.ref
    assert datastore.get_mapping_reference().ref == reference.ref
    assert datastore.get_mapping_reference(reference.ref).ref == reference.ref

    datastore.cells.insert(
        "reference_layout1",
        np.arange(datastore.cells.N, dtype=np.float64),
        overwrite=True,
    )
    datastore.cells.insert(
        "reference_layout2",
        -np.arange(datastore.cells.N, dtype=np.float64),
        overwrite=True,
    )
    np.testing.assert_array_equal(
        reference.fetch_cell_column("ids"),
        datastore.cells.fetch("ids", key=reference.cell_key),
    )
    np.testing.assert_array_equal(
        reference.fetch_layout("reference_layout"),
        np.column_stack(
            (
                datastore.cells.fetch(
                    "reference_layout1",
                    key=reference.cell_key,
                ),
                datastore.cells.fetch(
                    "reference_layout2",
                    key=reference.cell_key,
                ),
            )
        ),
    )


def test_symphony_mapping_reference_has_conditional_state_and_read_only_reload(
    analyzed_datastore_ephemeral,
):
    datastore = analyzed_datastore_ephemeral
    neighbors = _symphony_neighbors(datastore)
    reference = datastore.build_mapping_reference(neighbors)

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


def test_mapping_reference_rejects_old_embedded_and_incomplete_contracts(
    analyzed_datastore_ephemeral,
):
    datastore = analyzed_datastore_ephemeral
    neighbors = _selected_neighbors(datastore)
    state = datastore.get_assay_state("RNA")
    assert state is not None
    assert state.reduction is not None
    artifact_group(datastore.zw, state.reduction).create_group("mappingReference")

    with pytest.raises(ValueError, match="build_mapping_reference\\(neighbors\\)"):
        datastore.get_mapping_reference()

    reference = datastore.build_mapping_reference(neighbors)
    group = artifact_group(datastore.zw, reference.ref)
    del group["feature_scales"]
    with pytest.raises(ValueError, match="build_mapping_reference\\(neighbors\\)"):
        datastore.get_mapping_reference(reference.ref)


def test_mapping_reference_validates_live_dataset_fingerprint(
    analyzed_datastore_ephemeral,
):
    datastore = analyzed_datastore_ephemeral
    reference = datastore.build_mapping_reference(_selected_neighbors(datastore))
    datastore.RNA.attrs["dataset_fingerprint"] = "changed"

    with pytest.raises(ValueError, match="dataset fingerprint"):
        datastore.get_mapping_reference(reference.ref)


def test_mapping_reference_rejects_nonmonotonic_distance_summary(
    analyzed_datastore_ephemeral,
):
    datastore = analyzed_datastore_ephemeral
    reference = datastore.build_mapping_reference(_selected_neighbors(datastore))
    values = artifact_group(datastore.zw, reference.ref)["reference_distance_values"]
    values[:] = np.linspace(1.0, 0.0, values.shape[0])

    with pytest.raises(ValueError, match="distance summary"):
        datastore.get_mapping_reference(reference.ref)
