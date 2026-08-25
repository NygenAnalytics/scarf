from collections.abc import Callable

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.graph.errors import IncompatibleAnalysisStateError
from scarf.graph.state import AssayState
from scarf.storage.artifacts import (
    ArtifactRef,
    artifact_path,
    fingerprint_array,
    fingerprint_stored_arrays,
    fingerprint_stored_strings,
    inspect_artifact,
    new_artifact_id,
)
from scarf.storage.artifact_writer import finish_artifact, plan_artifact, start_artifact
from scarf.storage.errors import ArtifactResolutionError
from scarf.storage.feature_selection import (
    publish_feature_selection_alias,
    resolve_feature_selection,
)


class _Crash(RuntimeError):
    pass


def _attach_integrity(
    root: zarr.Group,
    group: zarr.Group,
    ids: np.ndarray,
) -> None:
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
    _attach_integrity(root, all_group, ids)
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
    _attach_integrity(root, selected_group, ids)
    finish_artifact(selected_group, selected_plan)
    return root, store, selected_plan.ref


def _crash_at(step: str) -> Callable[[str], None]:
    def checkpoint(current: str) -> None:
        if current == step:
            raise _Crash(step)

    return checkpoint


def _new_selection(
    root: zarr.Group,
    parent: ArtifactRef,
    values: np.ndarray,
) -> ArtifactRef:
    raw_universe = (inspect_artifact(root, parent).inputs or {})["all_features"]
    all_features = ArtifactRef.from_dict(raw_universe)
    selected = np.asarray(values, dtype=bool)
    planned = plan_artifact(
        root,
        scope="assay",
        assay="RNA",
        kind="feature_selection",
        operation="set_feature_selection",
        parameters={"values_fingerprint": fingerprint_array(selected)},
        inputs={"all_features": all_features},
        execution_options={},
    )
    group = start_artifact(root, planned)
    group.create_array("values", data=selected)
    _attach_integrity(
        root,
        group,
        np.asarray(root["RNA/featureData/ids"][:]),
    )
    finish_artifact(group, planned)
    return planned.ref


def _forge_committed_alias(
    root: zarr.Group,
    label: str,
    ref: ArtifactRef,
) -> None:
    values = np.asarray(root[f"{artifact_path(ref)}/values"][:], dtype=bool)
    column = root["RNA/featureData"].create_array(label, data=values)
    column.attrs["source_value"] = "values"
    column.attrs["source_artifact"] = ref.to_dict()


def test_feature_selection_resolves_exact_ref_and_published_label() -> None:
    root, store, ref = _selection_store()

    publish_feature_selection_alias(root, "RNA", "selected", ref)

    assert resolve_feature_selection(root, "RNA", ref) == ref
    assert resolve_feature_selection(root, "RNA", "selected") == ref
    column = root["RNA/featureData/selected"]
    np.testing.assert_array_equal(column[:], [True, False, True, False])
    assert column.attrs["source_artifact"] == ref.to_dict()
    assert column.attrs["source_value"] == "values"
    assert "value_index" not in column.attrs
    assert "pending_feature_selection_aliases" not in root["RNA/featureData"].attrs

    read_only = zarr.open_group(store=store.with_read_only(True), mode="r")
    assert resolve_feature_selection(read_only, "RNA", "selected") == ref


def test_read_only_resolver_does_not_create_a_missing_feature_universe() -> None:
    root, store, _ref = _selection_store()
    before_feature_keys = set(root["RNA/featureData"].keys())
    before_feature_attrs = dict(root["RNA/featureData"].attrs)
    before_artifacts = set(root["RNA/artifacts/feature_selection"].keys())
    read_only = zarr.open_group(store=store.with_read_only(True), mode="r")

    with pytest.raises(ArtifactResolutionError) as caught:
        resolve_feature_selection(read_only, "RNA", "all_features")

    assert caught.value.code == "missing_universe"
    assert set(root["RNA/featureData"].keys()) == before_feature_keys
    assert dict(root["RNA/featureData"].attrs) == before_feature_attrs
    assert set(root["RNA/artifacts/feature_selection"].keys()) == before_artifacts


@pytest.mark.parametrize(
    "step",
    ["journaled", "column_ready", "source_metadata_written", "values_written"],
)
def test_precommit_crashes_remain_pending_and_same_target_retry_repairs(
    step: str,
) -> None:
    root, _store, ref = _selection_store()

    with pytest.raises(_Crash, match=step):
        publish_feature_selection_alias(
            root,
            "RNA",
            "selected",
            ref,
            _checkpoint=_crash_at(step),
        )

    with pytest.raises(ArtifactResolutionError) as caught:
        resolve_feature_selection(root, "RNA", "selected")
    assert caught.value.code == "pending_alias"
    assert resolve_feature_selection(root, "RNA", ref) == ref

    publish_feature_selection_alias(root, "RNA", "selected", ref)

    assert resolve_feature_selection(root, "RNA", "selected") == ref
    assert "pending_feature_selection_aliases" not in root["RNA/featureData"].attrs


def test_commit_then_crash_is_readable_and_retry_only_cleans_stale_journal() -> None:
    root, store, ref = _selection_store()

    with pytest.raises(_Crash, match="committed"):
        publish_feature_selection_alias(
            root,
            "RNA",
            "selected",
            ref,
            _checkpoint=_crash_at("committed"),
        )

    column = root["RNA/featureData/selected"]
    before_values = np.asarray(column[:]).copy()
    before_attrs = dict(column.attrs)
    assert "pending_feature_selection_aliases" in root["RNA/featureData"].attrs

    read_only = zarr.open_group(store=store.with_read_only(True), mode="r")
    assert resolve_feature_selection(read_only, "RNA", "selected") == ref

    publish_feature_selection_alias(root, "RNA", "selected", ref)

    np.testing.assert_array_equal(column[:], before_values)
    assert dict(column.attrs) == before_attrs
    assert "pending_feature_selection_aliases" not in root["RNA/featureData"].attrs


@pytest.mark.parametrize(
    ("with_stale_journal", "expected_code"),
    [(False, "stale_label"), (True, "pending_alias")],
)
def test_all_features_label_requires_the_feature_universe_operation(
    with_stale_journal: bool,
    expected_code: str,
) -> None:
    root, _store, ref = _selection_store()
    _forge_committed_alias(root, "all_features", ref)
    if with_stale_journal:
        root["RNA/featureData"].attrs["pending_feature_selection_aliases"] = {
            "all_features": ref.to_dict()
        }

    with pytest.raises(ArtifactResolutionError) as caught:
        resolve_feature_selection(root, "RNA", "all_features")

    assert caught.value.code == expected_code
    assert caught.value.context["assay"] == "RNA"
    assert caught.value.context["label"] == "all_features"
    assert caught.value.context["artifact_id"] == ref.artifact_id
    assert caught.value.context["operation"] == "set_feature_selection"
    if with_stale_journal:
        assert caught.value.context["journal_label"] == "all_features"


def test_genuine_pending_target_blocks_a_different_publisher() -> None:
    root, _store, first = _selection_store()
    second = _new_selection(root, first, np.asarray([False, True, False, True]))
    third = _new_selection(root, first, np.ones(4, dtype=bool))
    publish_feature_selection_alias(root, "RNA", "selected", first)

    with pytest.raises(_Crash):
        publish_feature_selection_alias(
            root,
            "RNA",
            "selected",
            second,
            _checkpoint=_crash_at("values_written"),
        )

    with pytest.raises(ArtifactResolutionError) as caught:
        publish_feature_selection_alias(root, "RNA", "selected", third)
    assert caught.value.code == "pending_alias"

    publish_feature_selection_alias(root, "RNA", "selected", second)
    assert resolve_feature_selection(root, "RNA", "selected") == second
    assert resolve_feature_selection(root, "RNA", first) == first


def test_stale_committed_journal_does_not_block_repointing() -> None:
    root, _store, first = _selection_store()
    second = _new_selection(root, first, np.asarray([False, True, False, True]))
    with pytest.raises(_Crash):
        publish_feature_selection_alias(
            root,
            "RNA",
            "selected",
            first,
            _checkpoint=_crash_at("committed"),
        )

    publish_feature_selection_alias(root, "RNA", "selected", second)

    assert resolve_feature_selection(root, "RNA", "selected") == second


@pytest.mark.parametrize(
    "journal",
    [
        {"bad__label": {}},
        {"other": "not-a-reference"},
        {
            "other": ArtifactRef(
                scope="assay",
                assay="ADT",
                kind="feature_selection",
                artifact_id="e" * 64,
            ).to_dict()
        },
    ],
)
def test_resolver_validates_the_complete_publication_journal(
    journal: dict[str, object],
) -> None:
    root, _store, ref = _selection_store()
    publish_feature_selection_alias(root, "RNA", "selected", ref)
    root["RNA/featureData"].attrs["pending_feature_selection_aliases"] = journal

    with pytest.raises(ArtifactResolutionError) as caught:
        resolve_feature_selection(root, "RNA", "selected")

    assert caught.value.code == "pending_alias"


@pytest.mark.parametrize(
    ("failure", "target_error_code"),
    [
        ("missing", "missing_artifact"),
        ("incomplete", "incomplete_artifact"),
        ("corrupt", "corrupt_payload"),
    ],
)
def test_resolver_validates_every_unrelated_journal_target(
    failure: str,
    target_error_code: str,
) -> None:
    root, _store, selected = _selection_store()
    publish_feature_selection_alias(root, "RNA", "selected", selected)
    if failure == "missing":
        journal_ref = ArtifactRef(
            scope="assay",
            assay="RNA",
            kind="feature_selection",
            artifact_id="f" * 64,
        )
    else:
        journal_ref = _new_selection(
            root,
            selected,
            np.asarray([False, True, False, True]),
        )
        group = root[artifact_path(journal_ref)]
        if failure == "incomplete":
            group.attrs["complete"] = False
        else:
            group["values"][0] = True
    root["RNA/featureData"].attrs["pending_feature_selection_aliases"] = {
        "other": journal_ref.to_dict()
    }

    with pytest.raises(ArtifactResolutionError) as caught:
        resolve_feature_selection(root, "RNA", "selected")

    assert caught.value.code == "pending_alias"
    assert caught.value.context == {
        "assay": "RNA",
        "label": "selected",
        "artifact_id": journal_ref.artifact_id,
        "journal_label": "other",
        "journal_error_code": target_error_code,
    }


def test_foreign_columns_and_invalid_labels_are_never_overwritten() -> None:
    root, _store, ref = _selection_store()
    feature_data = root["RNA/featureData"]
    feature_data.create_array("foreign", data=np.ones(4, dtype=bool))

    with pytest.raises(ArtifactResolutionError) as caught:
        publish_feature_selection_alias(root, "RNA", "foreign", ref)
    assert caught.value.code == "label_collision"
    np.testing.assert_array_equal(feature_data["foreign"][:], np.ones(4, dtype=bool))

    for label in ("I", "bad__label", "Bad", ""):
        with pytest.raises(ArtifactResolutionError) as caught:
            publish_feature_selection_alias(root, "RNA", label, ref)
        assert caught.value.code == "invalid_label"


def test_feature_selection_resolution_reports_structured_failures() -> None:
    root, _store, ref = _selection_store()

    with pytest.raises(ArtifactResolutionError) as caught:
        resolve_feature_selection(root, "RNA", "missing")
    assert caught.value.code == "missing_label"

    with pytest.raises(ArtifactResolutionError) as caught:
        resolve_feature_selection(root, "RNA", "I")
    assert caught.value.code == "invalid_label"

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


def test_published_label_requires_an_exact_artifact_reference() -> None:
    root, _store, ref = _selection_store()
    publish_feature_selection_alias(root, "RNA", "selected", ref)
    column = root["RNA/featureData/selected"]
    source = dict(column.attrs["source_artifact"])
    source["path"] = artifact_path(ref)
    column.attrs["source_artifact"] = source

    with pytest.raises(ArtifactResolutionError) as caught:
        resolve_feature_selection(root, "RNA", "selected")

    assert caught.value.code == "stale_label"


def test_feature_publication_rejects_legacy_state_before_mutation(
    datastore_ephemeral,
) -> None:
    store = datastore_ephemeral
    assay_group = store.zw["RNA"]
    state_group = (
        store.zw["RNA/state"]
        if "RNA/state" in store.zw
        else assay_group.create_group("state")
    )
    state_group.attrs["state"] = {
        "assay": "RNA",
        "cell_key": "I",
        "feat_key": "I",
    }
    feature_data = store.zw["RNA/featureData"]
    feature_columns = set(feature_data.keys())
    feature_attrs = dict(feature_data.attrs)
    assay_groups = set(assay_group.group_keys())
    assay_attrs = dict(assay_group.attrs)
    state_attrs = dict(state_group.attrs)

    with pytest.raises(IncompatibleAnalysisStateError) as caught:
        store.set_feature_selection(
            from_assay="RNA",
            mask=np.ones(store.RNA.feats.N, dtype=bool),
            label="new_selection",
        )

    assert caught.value.code == "legacy_feature_contract"
    assert set(feature_data.keys()) == feature_columns
    assert dict(feature_data.attrs) == feature_attrs
    assert set(assay_group.group_keys()) == assay_groups
    assert dict(assay_group.attrs) == assay_attrs
    assert dict(state_group.attrs) == state_attrs


def test_new_normalized_chain_recovers_from_unavailable_current_graph(
    datastore_ephemeral,
) -> None:
    store = datastore_ephemeral
    missing_graph = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="connectivity_map",
        artifact_id=new_artifact_id(),
    )
    state_group = (
        store.zw["RNA/state"]
        if "RNA/state" in store.zw
        else store.zw["RNA"].create_group("state")
    )
    state_group.attrs["state"] = AssayState(
        assay="RNA",
        cell_key="I",
        connectivity_map=missing_graph,
    ).to_dict()

    with pytest.raises(ArtifactResolutionError) as caught:
        store.get_assay_state("RNA")
    assert caught.value.code == "missing_artifact"
    assert store._get_latest_cell_key("RNA") == "I"

    mask = np.zeros(store.RNA.feats.N, dtype=bool)
    mask[: min(32, store.RNA.feats.N)] = True
    selection = store.set_feature_selection(
        from_assay="RNA",
        mask=mask,
        label="recovery_features",
    )
    normalized = store.run_normalization(
        from_assay="RNA",
        cell_key="I",
        features=selection,
        log_transform=False,
        renormalize_subset=True,
    )

    recovered = store.get_assay_state("RNA")
    assert recovered is not None
    assert recovered.normalized == normalized
    assert recovered.feature_scaling is None
    assert recovered.reduction is None
    assert recovered.neighbors is None
    assert recovered.connectivity_map is None


def test_corrupt_all_features_is_replaced_and_alias_is_repointed(
    datastore_ephemeral,
) -> None:
    store = datastore_ephemeral
    first = store._ensure_all_features(store.RNA)
    first_status = inspect_artifact(store.zw, first)
    assert first_status.inputs == {}
    assert set(first_status.parameters or {}) == {
        "dataset_fingerprint",
        "ordered_feature_ids_fingerprint",
    }
    first_group = store.zw[artifact_path(first)]
    first_group.create_array(
        "unexpected",
        data=np.zeros(store.RNA.feats.N, dtype=np.float64),
    )

    replacement = store._ensure_all_features(store.RNA)

    assert replacement != first
    assert store.resolve_features("RNA", "all_features") == replacement
    with pytest.raises(ArtifactResolutionError) as caught:
        store.resolve_features("RNA", first)
    assert caught.value.code == "corrupt_payload"


def test_detected_feature_producer_rejects_empty_result_before_publication(
    datastore_ephemeral,
) -> None:
    store = datastore_ephemeral
    store._ensure_all_features(store.RNA)
    before = set(store.list_artifacts(kind="feature_selection", from_assay="RNA"))

    with pytest.raises(ValueError, match="contains no features"):
        store.select_detected_features(
            from_assay="RNA",
            min_cells=store.cells.N + 1,
            label="empty_detected",
        )

    after = set(store.list_artifacts(kind="feature_selection", from_assay="RNA"))
    assert after == before
    assert "empty_detected" not in store.RNA.feats.columns


def test_manual_mask_and_indexes_share_exact_value_identity(
    datastore_ephemeral,
) -> None:
    store = datastore_ephemeral
    mask = np.zeros(store.RNA.feats.N, dtype=bool)
    mask[[0, 2, 4]] = True

    from_mask = store.set_feature_selection(mask=mask, label="manual_mask")
    from_indexes = store.set_feature_selection(
        feature_indexes=[4, 0, 2],
        label="manual_indexes",
    )

    assert from_indexes == from_mask
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
    assert store.resolve_features("RNA", "manual_mask") == from_mask
    assert store.resolve_features("RNA", "manual_indexes") == from_indexes
