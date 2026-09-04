import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.storage.artifact_writer import finish_artifact, plan_artifact, start_artifact
from scarf.storage.artifacts import (
    ArtifactRef,
    artifact_path,
    fingerprint_array,
    fingerprint_stored_arrays,
    fingerprint_stored_strings,
    inspect_artifact,
)
from scarf.storage.errors import ArtifactResolutionError
from scarf.storage.feature_selection import resolve_feature_selection


def _attach_integrity(root: zarr.Group, group: zarr.Group) -> None:
    group.attrs["ordered_feature_ids_fingerprint"] = fingerprint_stored_strings(
        root["RNA/featureData/ids"]
    )
    group.attrs["payload_fingerprint"] = fingerprint_stored_arrays(
        group,
        ("values",),
    )


def _selection_store(
    values: np.ndarray | None = None,
) -> tuple[zarr.Group, MemoryStore, ArtifactRef]:
    store = MemoryStore()
    root = zarr.open_group(store=store, mode="w")
    feature_data = root.create_group("RNA/featureData")
    ids = np.asarray(["g0", "g1", "g2", "g3"])
    feature_data.create_array("ids", data=ids)
    feature_data.create_array("names", data=ids)
    feature_data.create_array("I", data=np.ones(4, dtype=bool))
    selected = (
        np.asarray([True, False, True, False])
        if values is None
        else np.asarray(values, dtype=bool)
    )
    row_fingerprint = fingerprint_stored_strings(feature_data["ids"])
    all_plan = plan_artifact(
        root,
        scope="assay",
        assay="RNA",
        kind="feature_selection",
        operation="create_all_features",
        parameters={
            "dataset_fingerprint": "dataset",
            "ordered_feature_ids_fingerprint": row_fingerprint,
        },
        inputs={},
        execution_options={},
    )
    all_group = start_artifact(root, all_plan)
    all_group.create_array("values", data=np.ones(len(ids), dtype=bool))
    _attach_integrity(root, all_group)
    finish_artifact(all_group, all_plan)

    selected_plan = plan_artifact(
        root,
        scope="assay",
        assay="RNA",
        kind="feature_selection",
        operation="set_feature_selection",
        parameters={"values_fingerprint": fingerprint_array(selected)},
        inputs={"all_features": all_plan.ref},
        execution_options={},
    )
    selected_group = start_artifact(root, selected_plan)
    selected_group.create_array("values", data=selected)
    _attach_integrity(root, selected_group)
    finish_artifact(selected_group, selected_plan)
    return root, store, selected_plan.ref


def test_feature_selection_resolves_exact_ref_read_only() -> None:
    root, store, ref = _selection_store()

    assert resolve_feature_selection(root, "RNA", ref) == ref
    read_only = zarr.open_group(store=store.with_read_only(True), mode="r")
    assert resolve_feature_selection(read_only, "RNA", ref) == ref


@pytest.mark.parametrize("value", ["selected", "all_features", 1, None])
def test_feature_selection_rejects_aliases_and_non_refs(value: object) -> None:
    root, _store, _ref = _selection_store()

    with pytest.raises(TypeError, match="features must be an ArtifactRef"):
        resolve_feature_selection(root, "RNA", value)  # type: ignore[arg-type]


def test_feature_selection_resolution_reports_structured_ref_failures() -> None:
    root, _store, ref = _selection_store()
    wrong_kind = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="normalized",
        artifact_id=ref.artifact_id,
    )
    with pytest.raises(ArtifactResolutionError) as caught:
        resolve_feature_selection(root, "RNA", wrong_kind)
    assert caught.value.code == "wrong_kind"

    missing = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="feature_selection",
        artifact_id="f" * 64,
    )
    with pytest.raises(ArtifactResolutionError) as caught:
        resolve_feature_selection(root, "RNA", missing)
    assert caught.value.code == "missing_artifact"

    root[artifact_path(ref)].attrs["complete"] = False
    with pytest.raises(ArtifactResolutionError) as caught:
        resolve_feature_selection(root, "RNA", ref)
    assert caught.value.code == "incomplete_artifact"


def test_feature_selection_resolution_detects_payload_and_row_drift() -> None:
    root, _store, ref = _selection_store()
    group = root[artifact_path(ref)]
    group["values"][0] = False

    with pytest.raises(ArtifactResolutionError) as caught:
        resolve_feature_selection(root, "RNA", ref)
    assert caught.value.code == "corrupt_payload"

    group["values"][0] = True
    root["RNA/featureData/ids"][0] = "changed"
    with pytest.raises(ArtifactResolutionError) as caught:
        resolve_feature_selection(root, "RNA", ref)
    assert caught.value.code == "row_mismatch"


def test_feature_selection_inputs_are_exact_artifact_references() -> None:
    root, _store, ref = _selection_store()
    group = root[artifact_path(ref)]
    provenance = dict(group.attrs["provenance"])
    inputs = dict(provenance["inputs"])
    all_features = dict(inputs["all_features"])
    all_features["value_fingerprint"] = "embedded-scalar"
    inputs["all_features"] = all_features
    provenance["inputs"] = inputs
    group.attrs["provenance"] = provenance

    with pytest.raises(ArtifactResolutionError) as caught:
        resolve_feature_selection(root, "RNA", ref)
    assert caught.value.code == "corrupt_payload"


def test_corrupt_all_features_is_replaced_without_metadata_alias(
    datastore_ephemeral,
) -> None:
    store = datastore_ephemeral
    columns_before = set(store.RNA.feats.columns)
    first = store.select_all_features(from_assay="RNA")
    first_status = inspect_artifact(store.zw, first)
    assert first_status.inputs == {}
    assert set(first_status.parameters or {}) == {
        "dataset_fingerprint",
        "ordered_feature_ids_fingerprint",
    }
    store.zw[artifact_path(first)].create_array(
        "unexpected",
        data=np.zeros(store.RNA.feats.N, dtype=np.float64),
    )

    replacement = store.select_all_features(from_assay="RNA")

    assert replacement != first
    assert store.resolve_features("RNA", replacement) == replacement
    assert set(store.RNA.feats.columns) == columns_before
    with pytest.raises(ArtifactResolutionError) as caught:
        store.resolve_features("RNA", first)
    assert caught.value.code == "corrupt_payload"


def test_detected_feature_producer_rejects_empty_result_without_metadata_mutation(
    datastore_ephemeral,
) -> None:
    store = datastore_ephemeral
    store.select_all_features(from_assay="RNA")
    cell_selection = store.snapshot_cell_selection()
    before = set(store.list_artifacts(kind="feature_selection", from_assay="RNA"))
    columns_before = set(store.RNA.feats.columns)

    with pytest.raises(ValueError, match="contains no features"):
        store.select_detected_features(
            cell_selection,
            from_assay="RNA",
            min_cells=store.cells.N + 1,
        )

    after = set(store.list_artifacts(kind="feature_selection", from_assay="RNA"))
    assert after == before
    assert set(store.RNA.feats.columns) == columns_before


def test_manual_mask_and_indexes_share_identity_without_aliases(
    datastore_ephemeral,
) -> None:
    store = datastore_ephemeral
    mask = np.zeros(store.RNA.feats.N, dtype=bool)
    mask[[0, 2, 4]] = True
    columns_before = set(store.RNA.feats.columns)

    from_mask = store.set_feature_selection(mask=mask)
    from_indexes = store.set_feature_selection(feature_indexes=[4, 0, 2])

    assert from_indexes == from_mask
    assert set(store.RNA.feats.columns) == columns_before
    status = inspect_artifact(store.zw, from_mask)
    assert status.parameters == {"values_fingerprint": fingerprint_array(mask)}
    assert set(status.inputs or {}) == {"all_features"}
    all_features = ArtifactRef.from_dict(status.inputs["all_features"])
    all_status = inspect_artifact(store.zw, all_features)
    assert all_status.operation == "create_all_features"
    assert all_status.inputs == {}
    assert set(all_status.parameters or {}) == {
        "dataset_fingerprint",
        "ordered_feature_ids_fingerprint",
    }
