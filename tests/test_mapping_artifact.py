"""Regression tests for mapping reference artifact load contracts."""

from types import SimpleNamespace

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.mapping.artifact import (
    load_artifact_mapping_reference,
    write_artifact_mapping_reference,
)
from scarf.mapping.models import (
    ScaledPCAProjectionModel,
    SymphonyCorrectionModel,
)
from scarf.storage.artifact_writer import finish_artifact, plan_artifact, start_artifact
from scarf.storage.artifacts import ArtifactRef, artifact_group


def _ref(
    *,
    kind: str = "mapping_reference",
    assay: str | None = "RNA",
    token: str = "a",
) -> ArtifactRef:
    return ArtifactRef(
        scope="datastore" if assay is None else "assay",
        assay=assay,
        kind=kind,
        artifact_id=token * 64,
    )


def _plain_reference(datastore):
    state = datastore.get_assay_state("RNA")
    assert state is not None and state.neighbors is not None
    return datastore.build_mapping_reference(state.neighbors)


def test_load_rejects_non_mapping_reference_refs() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    datastore = SimpleNamespace(zw=root)

    with pytest.raises(ValueError, match="assay-scoped mapping reference"):
        load_artifact_mapping_reference(datastore, _ref(kind="reduction"))
    with pytest.raises(ValueError, match="assay-scoped mapping reference"):
        load_artifact_mapping_reference(datastore, _ref(assay=None))
    with pytest.raises(ValueError, match="assay-scoped mapping reference"):
        load_artifact_mapping_reference(datastore, "not-a-ref")  # type: ignore[arg-type]


def test_load_rejects_missing_and_incomplete_artifacts() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    datastore = SimpleNamespace(zw=root)
    missing = _ref(token="b")

    with pytest.raises(ValueError, match="missing or incomplete"):
        load_artifact_mapping_reference(datastore, missing)

    planned = plan_artifact(
        root,
        scope="assay",
        assay="RNA",
        kind="mapping_reference",
        operation="build_mapping_reference",
        parameters={"method": "pca"},
        inputs={},
        execution_options={},
    )
    start_artifact(root, planned)
    with pytest.raises(ValueError, match="missing or incomplete"):
        load_artifact_mapping_reference(datastore, planned.ref)


def test_load_rejects_wrong_operation_and_unsupported_method() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    datastore = SimpleNamespace(zw=root)

    planned = plan_artifact(
        root,
        scope="assay",
        assay="RNA",
        kind="mapping_reference",
        operation="legacy_mapping_reference",
        parameters={"method": "pca"},
        inputs={
            "reduction": _ref(kind="reduction", token="1"),
            "ann_index": _ref(kind="ann_index", token="2"),
            "neighbors": _ref(kind="neighbors", token="3"),
            "cell_selection": _ref(kind="cell_selection", assay=None, token="4"),
            "feature_selection": _ref(kind="feature_selection", token="5"),
        },
        execution_options={},
    )
    group = start_artifact(root, planned)
    finish_artifact(group, planned)
    with pytest.raises(ValueError, match="old operation"):
        load_artifact_mapping_reference(datastore, planned.ref)

    planned_method = plan_artifact(
        root,
        scope="assay",
        assay="RNA",
        kind="mapping_reference",
        operation="build_mapping_reference",
        parameters={"method": "umap"},
        inputs={
            "reduction": _ref(kind="reduction", token="6"),
            "ann_index": _ref(kind="ann_index", token="7"),
            "neighbors": _ref(kind="neighbors", token="8"),
            "cell_selection": _ref(kind="cell_selection", assay=None, token="9"),
            "feature_selection": _ref(kind="feature_selection", token="c"),
        },
        execution_options={},
    )
    group = start_artifact(root, planned_method)
    finish_artifact(group, planned_method)
    with pytest.raises(ValueError, match="missing or unsupported"):
        load_artifact_mapping_reference(datastore, planned_method.ref)


def test_load_rejects_incomplete_input_set_and_pca_with_batch_correction() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    datastore = SimpleNamespace(zw=root)

    incomplete = plan_artifact(
        root,
        scope="assay",
        assay="RNA",
        kind="mapping_reference",
        operation="build_mapping_reference",
        parameters={"method": "pca"},
        inputs={"reduction": _ref(kind="reduction", token="1")},
        execution_options={},
    )
    group = start_artifact(root, incomplete)
    finish_artifact(group, incomplete)
    with pytest.raises(ValueError, match="inputs do not match"):
        load_artifact_mapping_reference(datastore, incomplete.ref)

    with_batch = plan_artifact(
        root,
        scope="assay",
        assay="RNA",
        kind="mapping_reference",
        operation="build_mapping_reference",
        parameters={"method": "pca"},
        inputs={
            "reduction": _ref(kind="reduction", token="2"),
            "ann_index": _ref(kind="ann_index", token="3"),
            "neighbors": _ref(kind="neighbors", token="4"),
            "cell_selection": _ref(kind="cell_selection", assay=None, token="5"),
            "feature_selection": _ref(kind="feature_selection", token="6"),
            "batch_correction": _ref(kind="batch_correction", token="7"),
        },
        execution_options={},
    )
    group = start_artifact(root, with_batch)
    finish_artifact(group, with_batch)
    with pytest.raises(ValueError, match="inputs do not match"):
        load_artifact_mapping_reference(datastore, with_batch.ref)


def test_write_and_load_reject_missing_payload_arrays_after_corruption(
    analyzed_datastore_ephemeral,
) -> None:
    datastore = analyzed_datastore_ephemeral
    reference = _plain_reference(datastore)
    group = artifact_group(datastore.zw, reference.ref)
    del group["loadings"]

    with pytest.raises(ValueError, match="build_mapping_reference\\(neighbors\\)"):
        load_artifact_mapping_reference(datastore, reference.ref)


def test_load_rejects_versioned_metadata_and_bad_distance_summary(
    analyzed_datastore_ephemeral,
) -> None:
    datastore = analyzed_datastore_ephemeral
    state = datastore.get_assay_state("RNA")
    assert state is not None and state.neighbors is not None
    reference = _plain_reference(datastore)
    group = artifact_group(datastore.zw, reference.ref)

    metadata = dict(group.attrs["reference_metadata"])
    metadata["schemaVersion"] = 1
    group.attrs["reference_metadata"] = metadata
    with pytest.raises(ValueError, match="versioned contract"):
        load_artifact_mapping_reference(datastore, reference.ref)

    reference = datastore.build_mapping_reference(state.neighbors)
    group = artifact_group(datastore.zw, reference.ref)
    metadata = dict(group.attrs["reference_metadata"])
    metadata.pop("schemaVersion", None)
    group.attrs["reference_metadata"] = metadata
    quantiles = group["reference_distance_quantiles"]
    quantiles[:] = np.linspace(1.0, 0.0, quantiles.shape[0])
    with pytest.raises(ValueError, match="distance summary"):
        load_artifact_mapping_reference(datastore, reference.ref)


def test_load_rejects_malformed_scoped_and_missing_input_refs(
    analyzed_datastore_ephemeral,
) -> None:
    datastore = analyzed_datastore_ephemeral
    reference = _plain_reference(datastore)
    group = artifact_group(datastore.zw, reference.ref)
    original = dict(group.attrs["provenance"])
    original_inputs = dict(original["inputs"])
    malformed_reduction = dict(original_inputs["reduction"])
    malformed_reduction["artifact_id"] = "invalid"
    corruptions = (
        ("not-a-ref", "input 'reduction' is missing"),
        (malformed_reduction, "input 'reduction' is malformed"),
        (
            _ref(kind="reduction", assay=None, token="d").to_dict(),
            "wrong artifact kind or scope",
        ),
        (
            _ref(kind="reduction", token="e").to_dict(),
            "input 'reduction' is missing or incomplete",
        ),
    )

    for value, message in corruptions:
        provenance = dict(original)
        inputs = dict(original_inputs)
        inputs["reduction"] = value
        provenance["inputs"] = inputs
        group.attrs["provenance"] = provenance
        with pytest.raises(ValueError, match=message):
            load_artifact_mapping_reference(datastore, reference.ref)

    group.attrs["provenance"] = original


def test_load_rejects_coordinate_chain_and_live_fingerprint_mismatches(
    analyzed_datastore_ephemeral,
) -> None:
    datastore = analyzed_datastore_ephemeral
    reference = _plain_reference(datastore)
    ann_group = artifact_group(datastore.zw, reference.ann_index)
    original_ann_provenance = dict(ann_group.attrs["provenance"])
    ann_provenance = dict(original_ann_provenance)
    ann_inputs = dict(ann_provenance["inputs"])
    ann_inputs["coordinates"] = reference.feature_selection.to_dict()
    ann_provenance["inputs"] = ann_inputs
    ann_group.attrs["provenance"] = ann_provenance

    with pytest.raises(ValueError, match="ANN index uses different coordinates"):
        load_artifact_mapping_reference(datastore, reference.ref)

    ann_group.attrs["provenance"] = original_ann_provenance
    original_fingerprint = datastore.RNA.attrs["dataset_fingerprint"]
    datastore.RNA.attrs["dataset_fingerprint"] = "changed"
    with pytest.raises(ValueError, match="dataset fingerprint"):
        load_artifact_mapping_reference(datastore, reference.ref)
    datastore.RNA.attrs["dataset_fingerprint"] = original_fingerprint


def test_load_rejects_metadata_model_and_payload_tampering(
    analyzed_datastore_ephemeral,
) -> None:
    datastore = analyzed_datastore_ephemeral
    reference = _plain_reference(datastore)
    group = artifact_group(datastore.zw, reference.ref)
    original_metadata = dict(group.attrs["reference_metadata"])

    metadata = dict(original_metadata)
    metadata["feature_key"] = "I"
    group.attrs["reference_metadata"] = metadata
    with pytest.raises(ValueError, match="removed feature-key contract"):
        load_artifact_mapping_reference(datastore, reference.ref)

    metadata = dict(original_metadata)
    metadata["assay"] = "other"
    group.attrs["reference_metadata"] = metadata
    with pytest.raises(ValueError, match="metadata does not match"):
        load_artifact_mapping_reference(datastore, reference.ref)

    metadata = dict(original_metadata)
    metadata["normalization_parameters"] = []
    group.attrs["reference_metadata"] = metadata
    with pytest.raises(ValueError, match="normalization parameters are missing"):
        load_artifact_mapping_reference(datastore, reference.ref)

    metadata = dict(original_metadata)
    metadata["selected_cell_count"] = reference.selected_cell_count + 1
    group.attrs["reference_metadata"] = metadata
    with pytest.raises(ValueError, match="cell count does not match"):
        load_artifact_mapping_reference(datastore, reference.ref)

    group.attrs["reference_metadata"] = original_metadata
    scales = group["feature_scales"]
    original_scale = float(scales[0])
    scales[0] = 0.0
    with pytest.raises(ValueError, match="PCA model is invalid"):
        load_artifact_mapping_reference(datastore, reference.ref)
    scales[0] = original_scale

    feature_ids = group["feature_ids"]
    original_feature_id = feature_ids[0]
    feature_ids[0] = "__tampered_feature__"
    with pytest.raises(ValueError, match="feature IDs do not match"):
        load_artifact_mapping_reference(datastore, reference.ref)
    feature_ids[0] = original_feature_id

    group.create_group("extra")
    with pytest.raises(ValueError, match="groups outside"):
        load_artifact_mapping_reference(datastore, reference.ref)
    del group["extra"]

    group.create_array("extra", data=np.ones(1), chunks=(1,))
    with pytest.raises(ValueError, match="arrays outside"):
        load_artifact_mapping_reference(datastore, reference.ref)
    del group["extra"]

    del group.attrs["reference_metadata"]
    with pytest.raises(ValueError, match="metadata is missing"):
        load_artifact_mapping_reference(datastore, reference.ref)
    group.attrs["reference_metadata"] = original_metadata


def test_write_artifact_mapping_reference_persists_required_pca_arrays() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    planned = plan_artifact(
        root,
        scope="assay",
        assay="RNA",
        kind="mapping_reference",
        operation="build_mapping_reference",
        parameters={"method": "pca"},
        inputs={},
        execution_options={},
    )
    group = start_artifact(root, planned)
    write_artifact_mapping_reference(
        group,
        ScaledPCAProjectionModel(
            feature_means=np.zeros(2),
            feature_scales=np.ones(2),
            loadings=np.eye(2),
        ),
        None,
        np.array(["g0", "g1"], dtype=object),
        {
            "assay": "RNA",
            "method": "pca",
            "cell_key": "I",
            "ann_metric": "l2",
            "dataset_fingerprint": "fp",
            "selected_cell_count": 2,
            "normalization_parameters": {"size_factor": 1000.0},
        },
        np.array([0.0, 1.0]),
        np.array([0.1, 0.2]),
    )
    finish_artifact(group, planned)

    assert set(group.array_keys()) == {
        "feature_ids",
        "feature_means",
        "feature_scales",
        "loadings",
        "reference_distance_quantiles",
        "reference_distance_values",
    }
    assert group.attrs["reference_metadata"]["method"] == "pca"


def test_write_artifact_mapping_reference_persists_symphony_state() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    planned = plan_artifact(
        root,
        scope="assay",
        assay="RNA",
        kind="mapping_reference",
        operation="build_mapping_reference",
        parameters={"method": "symphony"},
        inputs={},
        execution_options={},
    )
    group = start_artifact(root, planned)
    write_artifact_mapping_reference(
        group,
        ScaledPCAProjectionModel(
            feature_means=np.zeros(2),
            feature_scales=np.ones(2),
            loadings=np.eye(2),
        ),
        SymphonyCorrectionModel(
            centroids=np.eye(2),
            raw_centroids=np.eye(2),
            corrected_centroids=np.eye(2) * 2,
            cluster_mass=np.array([1.0, 2.0]),
            sigma=np.array([0.5, 1.0]),
        ),
        np.array(["g0", "g1"], dtype=object),
        {"method": "symphony"},
        np.array([0.0, 1.0]),
        np.array([0.1, 0.2]),
    )
    finish_artifact(group, planned)

    assert set(group.array_keys()) == {
        "feature_ids",
        "feature_means",
        "feature_scales",
        "loadings",
        "reference_distance_quantiles",
        "reference_distance_values",
        "centroids",
        "raw_centroids",
        "corrected_centroids",
        "cluster_mass",
        "sigma",
    }


def test_write_artifact_mapping_reference_rejects_high_rank_arrays() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    planned = plan_artifact(
        root,
        scope="assay",
        assay="RNA",
        kind="mapping_reference",
        operation="build_mapping_reference",
        parameters={"method": "pca"},
        inputs={},
        execution_options={},
    )
    group = start_artifact(root, planned)

    with pytest.raises(ValueError, match="one or two axes"):
        write_artifact_mapping_reference(
            group,
            ScaledPCAProjectionModel(
                feature_means=np.zeros(2),
                feature_scales=np.ones(2),
                loadings=np.eye(2),
            ),
            None,
            np.array(["g0", "g1"], dtype=object),
            {"method": "pca"},
            np.ones((1, 1, 1)),
            np.ones(1),
        )

    assert not group.attrs["complete"]
