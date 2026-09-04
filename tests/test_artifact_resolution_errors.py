import json
import pickle

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

import scarf
from scarf.storage.artifacts import artifact_path
from scarf.storage.errors import ArtifactResolutionError
from scarf.storage.refs import ArtifactRef
from scarf.storage.selections import (
    resolve_selection_artifact,
    validate_stored_selection_integrity,
    validate_stored_selection_live_alias,
)


def _root_with_selection() -> tuple[zarr.Group, MemoryStore, ArtifactRef]:
    store = MemoryStore()
    root = zarr.open_group(store=store, mode="w")
    cells = root.create_group("cellData")
    cell_ids = np.array(["a", "b", "c"])
    selection = np.array([True, False, True])
    cells.create_array("ids", data=cell_ids)
    cells.create_array("I", data=selection)
    ref = resolve_selection_artifact(
        root,
        scope="datastore",
        kind="cell_selection",
        values=selection,
        row_ids=cell_ids,
        operation="manual_selection",
        parameters={},
        inputs={},
        source_column="I",
    )
    return root, store, ref


def _read_only_root(store: MemoryStore) -> zarr.Group:
    return zarr.open_group(store=store.with_read_only(True), mode="r")


def _assert_resolution_error(
    store: MemoryStore,
    ref: ArtifactRef,
    code: str,
    *,
    live_alias: bool = False,
) -> ArtifactResolutionError:
    with pytest.raises(ArtifactResolutionError) as caught:
        root = _read_only_root(store)
        if live_alias:
            validate_stored_selection_live_alias(
                root,
                ref,
                kind="cell_selection",
                scope="datastore",
                assay=None,
                table_path="cellData",
                column="I",
            )
        else:
            validate_stored_selection_integrity(
                root,
                ref,
                kind="cell_selection",
                scope="datastore",
                assay=None,
                table_path="cellData",
            )

    error = caught.value
    assert isinstance(error, ValueError)
    assert error.code == code
    assert json.loads(json.dumps(error.context, allow_nan=False)) == error.context
    assert "zarr_loc" not in error.context
    return error


def test_selection_validation_is_read_only() -> None:
    _, store, ref = _root_with_selection()

    validate_stored_selection_integrity(
        _read_only_root(store),
        ref,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )


def test_selection_error_for_reference_mismatch() -> None:
    _, store, ref = _root_with_selection()
    wrong_ref = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="feature_selection",
        artifact_id=ref.artifact_id,
    )

    error = _assert_resolution_error(
        store,
        wrong_ref,
        "artifact_reference_mismatch",
    )

    assert error.context["expected_scope"] == "datastore"
    assert error.context["actual_scope"] == "assay"
    assert error.context["actual_kind"] == "feature_selection"


def test_selection_error_for_missing_artifact() -> None:
    _, store, ref = _root_with_selection()
    missing_ref = ArtifactRef(
        scope=ref.scope,
        kind=ref.kind,
        artifact_id="f" * 64,
    )

    error = _assert_resolution_error(store, missing_ref, "artifact_missing")

    assert "does not exist" in str(error)
    assert error.context["artifact_id"] == "f" * 64


def test_selection_error_for_incomplete_artifact() -> None:
    root, store, ref = _root_with_selection()
    root[artifact_path(ref)].attrs["complete"] = False

    error = _assert_resolution_error(store, ref, "artifact_incomplete")

    assert "incomplete" in str(error)


def test_selection_error_for_missing_table() -> None:
    root, store, ref = _root_with_selection()
    del root["cellData"]

    error = _assert_resolution_error(store, ref, "selection_table_missing")

    assert error.context["table"] == "cellData"


def test_selection_error_for_missing_column() -> None:
    root, store, ref = _root_with_selection()
    del root["cellData/I"]

    error = _assert_resolution_error(
        store,
        ref,
        "selection_column_missing",
        live_alias=True,
    )

    assert error.context["column"] == "I"


def test_selection_error_for_missing_row_ids() -> None:
    root, store, ref = _root_with_selection()
    del root["cellData/ids"]

    error = _assert_resolution_error(store, ref, "selection_row_ids_missing")

    assert error.context["table"] == "cellData"


def test_selection_error_for_missing_artifact_values() -> None:
    root, store, ref = _root_with_selection()
    del root[f"{artifact_path(ref)}/values"]

    error = _assert_resolution_error(store, ref, "selection_values_missing")

    assert error.context["kind"] == "cell_selection"


def test_selection_error_for_changed_row_identity() -> None:
    root, store, ref = _root_with_selection()
    root["cellData/ids"][:] = np.array(["x", "y", "z"])

    error = _assert_resolution_error(store, ref, "row_identity_mismatch")

    assert "row identity" in str(error)


def test_selection_error_for_changed_values() -> None:
    root, store, ref = _root_with_selection()
    root["cellData/I"][:] = np.array([False, True, False])

    error = _assert_resolution_error(
        store,
        ref,
        "selection_values_changed",
        live_alias=True,
    )

    assert "no longer matches" in str(error)


def test_artifact_resolution_error_has_public_facades() -> None:
    assert scarf.ArtifactResolutionError is ArtifactResolutionError
    assert scarf.storage.ArtifactResolutionError is ArtifactResolutionError
    assert not hasattr(scarf, "ArtifactSelectionError")
    assert not hasattr(scarf.graph, "ArtifactSelectionError")


def test_artifact_resolution_error_round_trips_through_pickle() -> None:
    error = ArtifactResolutionError(
        "Stored selection is unavailable",
        code="artifact_missing",
        context={
            "kind": "cell_selection",
            "artifact_id": "f" * 64,
        },
    )

    restored = pickle.loads(pickle.dumps(error))

    assert type(restored) is ArtifactResolutionError
    assert restored.args == error.args
    assert str(restored) == str(error)
    assert restored.code == error.code
    assert restored.context == error.context
