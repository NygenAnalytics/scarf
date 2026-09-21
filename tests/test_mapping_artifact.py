"""Regression tests for mapping reference artifact load contracts."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.mapping.artifact import (
    _reference_available_k,
    load_artifact_mapping_reference,
    validate_mapping_reference_sources,
    write_artifact_mapping_reference_from_sources,
    mapping_reference_payload_matches_sources,
    mapping_reference_source_fingerprint,
)
from scarf.storage.artifact_writer import finish_artifact, plan_artifact, start_artifact
from scarf.storage.ann_index import ANN_INDEX_ARRAY
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
    graphs = datastore.list_artifacts(
        kind="connectivity_map",
        from_assay="RNA",
        scope="assay",
        complete_only=True,
    )
    assert len(graphs) == 1
    neighbors = ArtifactRef.from_dict(
        datastore.inspect_artifact(graphs[0]).inputs["neighbors"]
    )
    reference_ref = datastore.build_mapping_reference(neighbors)
    return datastore.get_mapping_reference(reference_ref)


def test_reference_query_rejects_inconsistent_handles(analyzed_datastore_ephemeral):
    reference = _plain_reference(analyzed_datastore_ephemeral)
    assert _reference_available_k(reference) > 0
    for changes, message in (
        ({"ref": replace(reference.ref, assay="other")}, "assay identity"),
        ({"feature_ids": reference.feature_ids[:-1]}, "feature dimensions"),
        ({"symphony_state": object()}, "Plain mapping reference has Symphony state"),
        (
            {"metadata": dict(reference.metadata) | {"method": "symphony"}},
            "no correction state",
        ),
        (
            {"metadata": dict(reference.metadata) | {"method": "unknown"}},
            "method is unsupported",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            _reference_available_k(replace(reference, **changes))


@pytest.mark.parametrize(
    ("source", "section", "key", "value", "message"),
    [
        (
            "ann_index",
            "parameters",
            "ann_metric",
            "cosine",
            "ANN metric is inconsistent",
        ),
        ("ann_index", "parameters", "ann_ef", 0, "search depth is invalid"),
        ("ann_index", "parameters", "ann_ef", True, "search depth is invalid"),
        ("neighbors", "inputs", "ann_index", None, "another ANN index"),
        (
            "neighbors",
            "parameters",
            "distance_metric",
            "cosine",
            "neighbor metric is inconsistent",
        ),
    ],
)
def test_reference_query_rejects_corrupted_provenance(
    analyzed_datastore_ephemeral, source, section, key, value, message
):
    reference = _plain_reference(analyzed_datastore_ephemeral)
    group = artifact_group(reference.datastore.zw, getattr(reference, source))
    provenance = dict(group.attrs["provenance"])
    provenance[section] = dict(provenance[section]) | {key: value}
    group.attrs["provenance"] = provenance
    with pytest.raises(ValueError, match=message):
        _reference_available_k(reference)


@pytest.mark.parametrize("source", ["ref", "reduction", "ann_index", "neighbors"])
def test_reference_query_rejects_incomplete_graph_chain(
    analyzed_datastore_ephemeral, source
):
    reference = _plain_reference(analyzed_datastore_ephemeral)
    artifact_group(reference.datastore.zw, getattr(reference, source)).attrs[
        "complete"
    ] = False
    with pytest.raises(ValueError, match="graph chain is incomplete"):
        _reference_available_k(reference)


@pytest.mark.parametrize("payload", ["neighbors", "ann_index"])
def test_reference_query_rejects_corrupted_payload(
    analyzed_datastore_ephemeral, payload
):
    reference = _plain_reference(analyzed_datastore_ephemeral)
    group = artifact_group(reference.datastore.zw, getattr(reference, payload))
    if payload == "neighbors":
        group["distances"].resize((reference.selected_cell_count, 1))
        message = "neighbor payload is invalid"
    else:
        del group[ANN_INDEX_ARRAY]
        message = "ANN index is missing"
    with pytest.raises(ValueError, match=message):
        _reference_available_k(reference)


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


@pytest.mark.parametrize("array_name", ["loadings", "center"])
def test_write_and_load_reject_missing_payload_arrays_after_corruption(
    analyzed_datastore_ephemeral,
    array_name,
) -> None:
    datastore = analyzed_datastore_ephemeral
    reference = _plain_reference(datastore)
    group = artifact_group(datastore.zw, reference.ref)
    del group[array_name]

    message = (
        "center.*Recompute PCA"
        if array_name == "center"
        else "build_mapping_reference\\(neighbors\\)"
    )
    with pytest.raises(ValueError, match=message):
        load_artifact_mapping_reference(datastore, reference.ref)

    replacement = datastore.build_mapping_reference(reference.neighbors)
    assert replacement != reference.ref
    restored = datastore.get_mapping_reference(replacement)
    np.testing.assert_array_equal(restored.model.center, reference.model.center)


def test_mapping_reference_rejects_source_pca_without_fitted_center(
    analyzed_datastore_ephemeral,
) -> None:
    datastore = analyzed_datastore_ephemeral
    reference = _plain_reference(datastore)
    del artifact_group(datastore.zw, reference.reduction)["center"]

    with pytest.raises(ValueError, match="no fitted center.*recompute PCA"):
        datastore.get_mapping_reference(reference.ref)
    with pytest.raises(ValueError, match="no fitted center.*Recompute PCA"):
        datastore.build_mapping_reference(reference.neighbors)


def test_load_rejects_versioned_metadata_and_bad_distance_summary(
    analyzed_datastore_ephemeral,
) -> None:
    datastore = analyzed_datastore_ephemeral
    reference = _plain_reference(datastore)
    group = artifact_group(datastore.zw, reference.ref)

    metadata = dict(group.attrs["reference_metadata"])
    metadata["schemaVersion"] = 1
    group.attrs["reference_metadata"] = metadata
    with pytest.raises(ValueError, match="versioned contract"):
        load_artifact_mapping_reference(datastore, reference.ref)

    reference = datastore.get_mapping_reference(
        datastore.build_mapping_reference(
            reference.neighbors,
            invalidate_cache=True,
        )
    )
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
    had_stored_fingerprint = "dataset_fingerprint" in datastore.RNA.attrs
    original_fingerprint = datastore.RNA.attrs.get("dataset_fingerprint")
    datastore.RNA.attrs["dataset_fingerprint"] = "changed"
    with pytest.raises(ValueError, match="dataset fingerprint"):
        load_artifact_mapping_reference(datastore, reference.ref)
    if had_stored_fingerprint:
        datastore.RNA.attrs["dataset_fingerprint"] = original_fingerprint
    else:
        del datastore.RNA.attrs["dataset_fingerprint"]


def test_load_rejects_metadata_model_and_payload_tampering(
    analyzed_datastore_ephemeral,
) -> None:
    datastore = analyzed_datastore_ephemeral
    reference = _plain_reference(datastore)
    group = artifact_group(datastore.zw, reference.ref)
    original_metadata = dict(group.attrs["reference_metadata"])

    metadata = dict(original_metadata)
    metadata["unexpected"] = "value"
    group.attrs["reference_metadata"] = metadata
    with pytest.raises(ValueError, match="current contract"):
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


def _reference_source_arrays(root, *, symphony=False):
    source_group = root.create_group("sources")
    arrays = {
        name: source_group.create_array(name, data=values)
        for name, values in {
            "feature_means": np.zeros(2),
            "feature_scales": np.ones(2),
            "center": np.zeros(2),
            "loadings": np.eye(2),
        }.items()
    }
    arrays["symphony_sources"] = None
    if symphony:
        arrays["symphony_sources"] = {
            name: source_group.create_array(name, data=values)
            for name, values in {
                "centroids": np.array([[1.0, 2.0], [3.0, 4.0]]),
                "raw_centroids": np.eye(2),
                "corrected_centroids": np.eye(2) * 2,
                "cluster_mass": np.array([1.0, 2.0]),
                "sigma": np.array([0.5, 1.0]),
            }.items()
        }
    return arrays


@pytest.mark.parametrize("symphony", [False, True])
def test_mapping_reference_stream_writer_preserves_model_arrays(symphony):
    root = zarr.open_group(store=MemoryStore(), mode="w")
    sources = _reference_source_arrays(root, symphony=symphony)
    method = "symphony" if symphony else "pca"
    planned = plan_artifact(
        root,
        scope="assay",
        assay="RNA",
        kind="mapping_reference",
        operation="build_mapping_reference",
        parameters={"method": method},
        inputs={},
        execution_options={},
    )
    group = start_artifact(root, planned)
    payload = {
        "feature_ids": np.array(["g0", "g1"]),
        "metadata": {"method": method},
        "reference_distance_quantiles": np.array([0.0, 1.0]),
        "reference_distance_values": np.array([0.1, 0.2]),
    }
    write_artifact_mapping_reference_from_sources(group, **sources, **payload)
    finish_artifact(group, planned)
    expected_arrays = {
        "feature_ids",
        "feature_means",
        "feature_scales",
        "center",
        "loadings",
        "reference_distance_quantiles",
        "reference_distance_values",
    }
    if symphony:
        expected_arrays.update(sources["symphony_sources"])
        np.testing.assert_array_equal(
            group["centroids"][:], sources["symphony_sources"]["centroids"][:].T
        )
    assert set(group.array_keys()) == expected_arrays
    assert group.attrs["reference_metadata"]["method"] == method
    assert isinstance(group.attrs["payload_fingerprint"], str)
    fingerprint = mapping_reference_source_fingerprint(**sources)
    assert mapping_reference_payload_matches_sources(
        group, **sources, **payload, expected_source_fingerprint=fingerprint
    )
    if symphony:
        incomplete_sources = dict(
            sources,
            symphony_sources={"centroids": sources["symphony_sources"]["centroids"]},
        )
        assert not mapping_reference_payload_matches_sources(
            group,
            **incomplete_sources,
            **payload,
            expected_source_fingerprint=fingerprint,
        )
    group["loadings"][0, 0] += 1
    assert not mapping_reference_payload_matches_sources(
        group, **sources, **payload, expected_source_fingerprint=fingerprint
    )
    del group["loadings"]
    assert not mapping_reference_payload_matches_sources(
        group, **sources, **payload, expected_source_fingerprint=fingerprint
    )


def test_mapping_reference_stream_writer_rejects_high_rank_summaries():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    sources = _reference_source_arrays(root)
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
        write_artifact_mapping_reference_from_sources(
            group,
            **sources,
            feature_ids=np.array(["g0", "g1"]),
            metadata={"method": "pca"},
            reference_distance_quantiles=np.ones((1, 1, 1)),
            reference_distance_values=np.ones(1),
        )
    assert not group.attrs["complete"]


def test_mapping_reference_source_validation_is_strict_and_semantic() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    means = root.create_array(
        "means",
        data=np.zeros(3, dtype=np.float64),
        chunks=(2,),
    )
    scales = root.create_array(
        "scales",
        data=np.ones(3, dtype=np.float64),
        chunks=(2,),
    )
    center = root.create_array(
        "center",
        data=np.zeros(3, dtype=np.float64),
        chunks=(2,),
    )
    loadings = root.create_array(
        "loadings",
        data=np.ones((3, 2), dtype=np.float64),
        chunks=(2, 2),
    )

    assert validate_mapping_reference_sources(
        feature_means=means,
        feature_scales=scales,
        center=center,
        loadings=loadings,
        symphony_sources=None,
    ) == (3, 2)

    scales[1] = 0.0
    with pytest.raises(ValueError, match="PCA model arrays are invalid"):
        validate_mapping_reference_sources(
            feature_means=means,
            feature_scales=scales,
            center=center,
            loadings=loadings,
            symphony_sources=None,
        )

    float32_loadings = root.create_array(
        "float32_loadings",
        data=np.ones((3, 2), dtype=np.float32),
        chunks=(2, 2),
    )
    scales[1] = 1.0
    with pytest.raises(ValueError, match="PCA model arrays are invalid"):
        validate_mapping_reference_sources(
            feature_means=means,
            feature_scales=scales,
            center=center,
            loadings=float32_loadings,
            symphony_sources=None,
        )
