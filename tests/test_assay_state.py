import hashlib
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.datastore.datastore import DataStore
from scarf.datastore.graph_datastore import GraphDataStore
from scarf.embeddings.imported import write_imported_coordinates
from scarf.graph.errors import IncompatibleAnalysisStateError
from scarf.graph.feature_projection import resolve_native_graph_inputs
from scarf.graph.state import (
    AssayState,
    embedding_initialization_path_from_state,
    normalized_path_from_state,
    read_assay_state,
    resolve_graph_selection,
    write_assay_state,
)
from scarf.storage.errors import ArtifactResolutionError
from scarf.storage.artifacts import (
    ArtifactRef,
    ArtifactScope,
    artifact_path,
    fingerprint_array,
    fingerprint_stored_arrays,
    fingerprint_stored_strings,
    inspect_artifact,
    make_provenance,
    new_artifact_id,
)


class _StateGraphStore(GraphDataStore):
    @property
    def assay_names(self) -> list[str]:
        return ["RNA"]


def _ref(kind: str, token: str) -> ArtifactRef:
    return ArtifactRef(
        scope="assay",
        assay="RNA",
        kind=kind,
        artifact_id=token * 64,
    )


def _add_artifact(
    root: zarr.Group,
    *,
    kind: str,
    operation: str,
    parameters: dict[str, Any] | None = None,
    inputs: dict[str, Any] | None = None,
    execution_options: dict[str, Any] | None = None,
    scope: ArtifactScope = "assay",
) -> tuple[ArtifactRef, zarr.Group]:
    parameters = parameters or {}
    inputs = inputs or {}
    execution_options = execution_options or {}
    ref = ArtifactRef(
        scope=scope,
        assay="RNA" if scope == "assay" else None,
        kind=kind,
        artifact_id=new_artifact_id(),
    )
    provenance = make_provenance(
        operation=operation,
        parameters=parameters,
        inputs=inputs,
    )
    group = root.create_group(artifact_path(ref))
    group.attrs.update(
        {
            "artifact_id": ref.artifact_id,
            "kind": ref.kind,
            "provenance": provenance,
            "execution_options": execution_options,
            "complete": True,
        }
    )
    return ref, group


def _state_store() -> tuple[_StateGraphStore, AssayState]:
    datastore = _StateGraphStore.__new__(_StateGraphStore)
    datastore.z = zarr.open_group(store=MemoryStore(), mode="w")
    datastore.workspace = None
    datastore.zarr_mode = "r+"
    datastore._defaultAssay = "RNA"
    datastore._integratedGraphsLoc = "integratedGraphs"
    datastore._cachedMagicOperator = None
    datastore._cachedMagicOperatorLoc = None
    datastore.nthreads = 1
    cell_ids = np.asarray(["c0", "c1", "c2"])
    feature_ids = np.asarray(["f0", "f1", "f2", "f3"])
    cell_data = datastore.z.create_group("cellData")
    cell_data.create_array("ids", data=cell_ids)
    cell_data.create_array("I", data=np.ones(3, dtype=bool))
    feature_data = datastore.z.create_group("RNA/featureData")
    feature_data.create_array("ids", data=feature_ids)
    feature_data.create_array("I", data=np.ones(4, dtype=bool))

    cell_selection, cell_selection_group = _add_artifact(
        datastore.z,
        kind="cell_selection",
        operation="manual_selection",
        inputs={
            "ordered_row_ids_fingerprint": fingerprint_stored_strings(cell_data["ids"]),
            "values_fingerprint": fingerprint_array(np.ones(3, dtype=bool)),
        },
        execution_options={"source_column": "I"},
        scope="datastore",
    )
    cell_selection_group.create_array(
        "values",
        data=np.ones(3, dtype=bool),
    )
    all_features, all_features_group = _add_artifact(
        datastore.z,
        kind="feature_selection",
        operation="create_all_features",
        parameters={
            "dataset_fingerprint": "test-dataset",
            "ordered_feature_ids_fingerprint": fingerprint_stored_strings(
                feature_data["ids"]
            ),
        },
    )
    all_features_group.create_array(
        "values",
        data=np.ones(4, dtype=bool),
    )
    all_features_group.attrs["ordered_feature_ids_fingerprint"] = (
        fingerprint_stored_strings(feature_data["ids"])
    )
    all_features_group.attrs["payload_fingerprint"] = fingerprint_stored_arrays(
        all_features_group,
        ("values",),
    )
    feature_selection, feature_selection_group = _add_artifact(
        datastore.z,
        kind="feature_selection",
        operation="set_feature_selection",
        parameters={
            "values_fingerprint": fingerprint_array(np.ones(4, dtype=bool)),
        },
        inputs={"all_features": all_features},
        execution_options={"label": "hvgs"},
    )
    feature_selection_group.create_array(
        "values",
        data=np.ones(4, dtype=bool),
    )
    feature_selection_group.attrs["ordered_feature_ids_fingerprint"] = (
        fingerprint_stored_strings(feature_data["ids"])
    )
    feature_selection_group.attrs["payload_fingerprint"] = fingerprint_stored_arrays(
        feature_selection_group,
        ("values",),
    )
    feature_label = feature_data.create_array(
        "hvgs",
        data=np.ones(4, dtype=bool),
    )
    feature_label.attrs["source_value"] = "values"
    feature_label.attrs["source_artifact"] = feature_selection.to_dict()
    normalized, normalized_group = _add_artifact(
        datastore.z,
        kind="normalized",
        operation="run_normalization",
        inputs={
            "cell_selection": cell_selection,
            "feature_selection": feature_selection,
        },
        execution_options={"cell_key": "I"},
    )
    normalized_group.create_array(
        "data",
        data=np.arange(12, dtype=np.float64).reshape(3, 4),
    )
    scaling, scaling_group = _add_artifact(
        datastore.z,
        kind="feature_scaling",
        operation="calculate_feature_scaling",
        inputs={"normalized": normalized},
    )
    scaling_group.create_array("mean", data=np.arange(4, dtype=np.float64))
    scaling_group.create_array("scale", data=np.ones(4, dtype=np.float64))
    reduction, reduction_group = _add_artifact(
        datastore.z,
        kind="reduction",
        operation="run_reduction",
        parameters={
            "reduction_method": "pca",
            "dims": 2,
            "pca_cell_key": "I",
            "feat_scaling": True,
            "lsi_skip_first": True,
        },
        inputs={
            "normalized": normalized,
            "feature_scaling": scaling,
        },
    )
    reduction_group.create_array(
        "loadings",
        data=np.arange(8, dtype=np.float64).reshape(4, 2),
    )
    correction, correction_group = _add_artifact(
        datastore.z,
        kind="batch_correction",
        operation="run_harmony",
        inputs={"reduction": reduction},
    )
    correction_group.create_array(
        "data",
        data=np.arange(6, dtype=np.float64).reshape(3, 2),
    )
    ann_index, _ann_index_group = _add_artifact(
        datastore.z,
        kind="ann_index",
        operation="build_ann_index",
        parameters={
            "ann_metric": "l2",
            "ann_efc": 50,
            "ann_ef": 50,
            "ann_m": 16,
            "rand_state": 4466,
        },
        inputs={"coordinates": correction},
    )
    initialization, initialization_group = _add_artifact(
        datastore.z,
        kind="embedding_initialization",
        operation="build_embedding_initialization",
        parameters={"n_centroids": 3, "rand_state": 4466},
        inputs={"reduction": reduction},
    )
    initialization_group.create_array(
        "cluster_centers",
        data=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
    )
    initialization_group.create_array(
        "cluster_labels",
        data=np.array([0, 1, 2], dtype=np.int64),
    )
    neighbors, neighbors_group = _add_artifact(
        datastore.z,
        kind="neighbors",
        operation="query_neighbors",
        parameters={"k": 2},
        inputs={
            "ann_index": ann_index,
            "coordinates": correction,
        },
    )
    neighbors_group.create_array(
        "indices",
        data=np.array([[1, 2], [0, 2], [0, 1]], dtype=np.uint64),
    )
    neighbors_group.create_array(
        "distances",
        data=np.ones((3, 2), dtype=np.float64),
    )
    connectivity, connectivity_group = _add_artifact(
        datastore.z,
        kind="connectivity_map",
        operation="build_connectivity_map",
        parameters={"local_connectivity": 1.0, "bandwidth": 1.5},
        inputs={"neighbors": neighbors},
    )
    connectivity_group.attrs["n_cells"] = 3
    connectivity_group.attrs["n_neighbors"] = 2
    connectivity_group.create_array(
        "edges",
        data=np.array(
            [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]],
            dtype=np.uint64,
        ),
    )
    connectivity_group.create_array(
        "weights",
        data=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
    )
    embedding, _embedding_group = _add_artifact(
        datastore.z,
        kind="embedding",
        operation="run_umap",
        inputs={"connectivity_map": connectivity},
    )

    state = AssayState(
        assay="RNA",
        cell_key="I",
        normalized=normalized,
        feature_scaling=scaling,
        reduction=reduction,
        batch_correction=correction,
        ann_index=ann_index,
        embedding_initialization=initialization,
        neighbors=neighbors,
        connectivity_map=connectivity,
        named_results={"umap": embedding},
    )
    state_group = datastore.z.create_group("RNA/state")
    state_group.attrs["state"] = state.to_dict()
    legacy_normalized_path = "RNA/normed__I__legacy"
    legacy_reduction_path = f"{legacy_normalized_path}/reduction__pca__5__I"
    legacy_ann_path = f"{legacy_reduction_path}/ann__l2__50__50__16__4466"
    legacy_neighbors_path = f"{legacy_ann_path}/knn__3"
    legacy_graph_path = f"{legacy_neighbors_path}/graph__1.0__1.5"
    legacy_normalized = datastore.z.create_group(legacy_normalized_path)
    legacy_reduction = datastore.z.create_group(legacy_reduction_path)
    legacy_ann = datastore.z.create_group(legacy_ann_path)
    legacy_neighbors = datastore.z.create_group(legacy_neighbors_path)
    legacy_graph = datastore.z.create_group(legacy_graph_path)
    legacy_normalized.attrs["latest_reduction"] = legacy_reduction_path
    legacy_reduction.attrs["latest_ann"] = legacy_ann_path
    legacy_ann.attrs["latest_knn"] = legacy_neighbors_path
    legacy_neighbors.attrs["latest_graph"] = legacy_graph_path
    legacy_neighbors.create_array(
        "indices",
        data=np.array([[1, 2, 0], [0, 2, 1], [0, 1, 2]], dtype=np.uint64),
    )
    legacy_graph.create_array(
        "edges",
        data=np.array(
            [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]],
            dtype=np.uint64,
        ),
    )
    legacy_graph.create_array(
        "weights",
        data=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
    )
    return datastore, state


def _store_digest(root: zarr.Group) -> str:
    digest = hashlib.blake2b(digest_size=32)

    def visit(group: zarr.Group, prefix: str = "") -> None:
        digest.update(prefix.encode())
        digest.update(repr(dict(group.attrs)).encode())
        for key in sorted(group.keys()):
            node = group[key]
            path = f"{prefix}/{key}" if prefix else key
            digest.update(path.encode())
            digest.update(repr(dict(node.attrs)).encode())
            if isinstance(node, zarr.Array):
                digest.update(str(node.dtype).encode())
                digest.update(np.asarray(node[:]).tobytes())
            else:
                visit(node, path)

    visit(root)
    return digest.hexdigest()


def test_assay_state_round_trip_and_validation() -> None:
    datastore, state = _state_store()
    assert AssayState.from_dict(state.to_dict()) == state
    assert set(state.to_dict()) == {
        "assay",
        "cell_key",
        "normalized",
        "feature_scaling",
        "reduction",
        "batch_correction",
        "ann_index",
        "embedding_initialization",
        "neighbors",
        "connectivity_map",
        "named_results",
    }
    public_datastore = DataStore.__new__(DataStore)
    public_datastore.z = datastore.z
    public_datastore.workspace = None
    public_datastore._defaultAssay = "RNA"
    assert public_datastore.get_assay_state() == state

    with pytest.raises(ValueError, match="requires kind"):
        AssayState(
            assay="RNA",
            cell_key="I",
            normalized=_ref("neighbors", "a"),
        )
    with pytest.raises(ValueError, match="snake_case"):
        AssayState(
            assay="RNA",
            cell_key="I",
            named_results={"UMAP 1": _ref("embedding", "a")},
        )
    malformed = state.to_dict()
    malformed["normalized"] = "not-a-reference"
    with pytest.raises(IncompatibleAnalysisStateError) as caught:
        AssayState.from_dict(malformed)
    assert caught.value.code == "invalid_analysis_state"
    malformed = state.to_dict()
    malformed["cell_key"] = None
    with pytest.raises(IncompatibleAnalysisStateError) as caught:
        AssayState.from_dict(malformed)
    assert caught.value.code == "invalid_analysis_state"


@pytest.mark.parametrize("legacy_field", ["feat_key", "feature_selection"])
def test_assay_state_rejects_legacy_feature_fields_without_mutation(
    legacy_field: str,
) -> None:
    datastore, state = _state_store()
    malformed = state.to_dict()
    malformed[legacy_field] = "hvgs"
    datastore.z["RNA/state"].attrs["state"] = malformed
    before = _store_digest(datastore.z)

    with pytest.raises(IncompatibleAnalysisStateError) as caught:
        read_assay_state(datastore.zw, "RNA")

    assert caught.value.code == "legacy_feature_contract"
    assert caught.value.context["keys"] == legacy_field
    assert _store_digest(datastore.z) == before


def test_assay_state_rejects_unknown_fields() -> None:
    _datastore, state = _state_store()
    malformed = state.to_dict()
    malformed["latest_graph"] = "path"

    with pytest.raises(IncompatibleAnalysisStateError) as caught:
        AssayState.from_dict(malformed)

    assert caught.value.code == "invalid_analysis_state"
    assert caught.value.context["keys"] == "latest_graph"

    malformed_ref = state.to_dict()
    raw_normalized = malformed_ref["normalized"]
    assert isinstance(raw_normalized, dict)
    raw_normalized["path"] = "RNA/legacy"
    with pytest.raises(IncompatibleAnalysisStateError) as caught:
        AssayState.from_dict(malformed_ref)
    assert caught.value.code == "invalid_analysis_state"


def test_state_first_graph_lookup_uses_artifacts() -> None:
    datastore, state = _state_store()

    assert read_assay_state(datastore.zw, "RNA") == state
    assert normalized_path_from_state(datastore.zw, "RNA", "I") == artifact_path(
        state.normalized
    )
    assert embedding_initialization_path_from_state(
        datastore.zw,
        "RNA",
        "I",
    ) == artifact_path(state.embedding_initialization)
    selected = resolve_graph_selection(
        datastore,
        state.connectivity_map,
        from_assay="RNA",
        cell_key="I",
    )
    assert selected.graph_ref == state.connectivity_map
    assert selected.graph_input == state.connectivity_map
    assert selected.graph_loc == artifact_path(state.connectivity_map)
    assert selected.from_assay == "RNA"
    assert selected.cell_key == "I"
    assert selected.integrated_label is None


def test_state_classifies_missing_normalized_feature_selection_as_corruption() -> None:
    datastore, state = _state_store()
    assert state.normalized is not None
    normalized = datastore.zw[artifact_path(state.normalized)]
    provenance = dict(normalized.attrs["provenance"])
    inputs = dict(provenance["inputs"])
    del inputs["feature_selection"]
    provenance["inputs"] = inputs
    normalized.attrs["provenance"] = provenance

    with pytest.raises(ArtifactResolutionError) as caught:
        read_assay_state(datastore.zw, "RNA")

    assert caught.value.code == "corrupt_payload"
    assert caught.value.context["field"] == "normalized.feature_selection"


def test_state_reserves_legacy_code_for_removed_normalized_feature_keys() -> None:
    datastore, state = _state_store()
    assert state.normalized is not None
    normalized = datastore.zw[artifact_path(state.normalized)]
    provenance = dict(normalized.attrs["provenance"])
    inputs = dict(provenance["inputs"])
    del inputs["feature_selection"]
    inputs["feat_key"] = "I__hvgs"
    provenance["inputs"] = inputs
    normalized.attrs["provenance"] = provenance

    with pytest.raises(IncompatibleAnalysisStateError) as caught:
        read_assay_state(datastore.zw, "RNA")

    assert caught.value.code == "legacy_feature_contract"


def test_state_validates_normalized_feature_selection_payload() -> None:
    datastore, state = _state_store()
    assert state.normalized is not None
    inputs = inspect_artifact(datastore.zw, state.normalized).inputs or {}
    raw_selection = inputs["feature_selection"]
    assert isinstance(raw_selection, dict)
    selection = ArtifactRef.from_dict(raw_selection)
    datastore.zw[artifact_path(selection)]["values"][0] = False

    with pytest.raises(ArtifactResolutionError) as caught:
        read_assay_state(datastore.zw, "RNA")

    assert caught.value.code == "corrupt_payload"


def test_imported_coordinate_state_does_not_require_normalized() -> None:
    datastore, native_state = _state_store()
    assert native_state.normalized is not None
    normalized_inputs = (
        inspect_artifact(
            datastore.zw,
            native_state.normalized,
        ).inputs
        or {}
    )
    raw_cells = normalized_inputs["cell_selection"]
    assert isinstance(raw_cells, dict)
    cell_selection = ArtifactRef.from_dict(raw_cells)
    cell_ids = np.asarray(datastore.z["cellData/ids"][:])
    coordinate_values = np.arange(6, dtype=np.float32).reshape(3, 2)
    coordinates = write_imported_coordinates(
        datastore.z,
        assay="RNA",
        dimreduc_key="pca",
        role="pca",
        coordinates=coordinate_values,
        source_digest=hashlib.sha256(b"state-import").digest(),
        payload_fingerprints={"data": fingerprint_array(coordinate_values)},
        source_cell_ids=cell_ids,
        cell_selection=cell_selection,
        cell_key="I",
        block_rows=2,
    )
    ann_index, _ann_group = _add_artifact(
        datastore.z,
        kind="ann_index",
        operation="build_ann_index",
        inputs={"coordinates": coordinates},
    )
    neighbors, _neighbors_group = _add_artifact(
        datastore.z,
        kind="neighbors",
        operation="query_neighbors",
        inputs={
            "ann_index": ann_index,
            "coordinates": coordinates,
        },
    )
    connectivity, _connectivity_group = _add_artifact(
        datastore.z,
        kind="connectivity_map",
        operation="build_connectivity_map",
        inputs={"neighbors": neighbors},
    )
    imported_state = AssayState(
        assay="RNA",
        cell_key="I",
        ann_index=ann_index,
        neighbors=neighbors,
        connectivity_map=connectivity,
    )

    write_assay_state(datastore.zw, imported_state)

    assert read_assay_state(datastore.zw, "RNA") == imported_state
    assert imported_state.normalized is None


def test_state_validates_ann_index_coordinates_without_a_graph() -> None:
    datastore, native_state = _state_store()
    assert native_state.ann_index is not None
    ann_only = AssayState(
        assay="RNA",
        cell_key="I",
        ann_index=native_state.ann_index,
    )
    datastore.z["RNA/state"].attrs["state"] = ann_only.to_dict()

    with pytest.raises(IncompatibleAnalysisStateError) as caught:
        read_assay_state(datastore.zw, "RNA")

    assert caught.value.code == "invalid_analysis_state"
    assert caught.value.context["field"] == "ann_index.coordinates"


def test_native_graph_resolution_fully_validates_imported_coordinates() -> None:
    datastore, native_state = _state_store()
    assert native_state.normalized is not None
    normalized_inputs = (
        inspect_artifact(
            datastore.zw,
            native_state.normalized,
        ).inputs
        or {}
    )
    raw_cells = normalized_inputs["cell_selection"]
    assert isinstance(raw_cells, dict)
    cell_selection = ArtifactRef.from_dict(raw_cells)
    cell_ids = np.asarray(datastore.z["cellData/ids"][:])
    coordinate_values = np.arange(6, dtype=np.float32).reshape(3, 2)
    coordinates = write_imported_coordinates(
        datastore.z,
        assay="RNA",
        dimreduc_key="pca",
        role="pca",
        coordinates=coordinate_values,
        source_digest=hashlib.sha256(b"projection-import").digest(),
        payload_fingerprints={"data": fingerprint_array(coordinate_values)},
        source_cell_ids=cell_ids,
        cell_selection=cell_selection,
        cell_key="I",
        block_rows=2,
    )
    ann_index, _ann_group = _add_artifact(
        datastore.z,
        kind="ann_index",
        operation="build_ann_index",
        inputs={"coordinates": coordinates},
    )
    neighbors, _neighbors_group = _add_artifact(
        datastore.z,
        kind="neighbors",
        operation="query_neighbors",
        inputs={"ann_index": ann_index, "coordinates": coordinates},
    )
    datastore.z[artifact_path(coordinates)]["data"][0, 0] = -1.0

    with pytest.raises(ArtifactResolutionError) as caught:
        resolve_native_graph_inputs(datastore.zw, neighbors)

    assert caught.value.code == "corrupt_payload"


def test_read_assay_state_wraps_malformed_state_node() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    assay_group = root.create_group("RNA")
    assay_group.create_array("state", data=np.asarray([1], dtype=np.int8))

    with pytest.raises(IncompatibleAnalysisStateError) as caught:
        read_assay_state(root, "RNA")

    assert caught.value.code == "invalid_analysis_state"


def test_read_assay_state_rejects_present_group_without_state_attribute() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    root.create_group("RNA/state")

    with pytest.raises(IncompatibleAnalysisStateError) as caught:
        read_assay_state(root, "RNA")

    assert caught.value.code == "invalid_analysis_state"


def test_state_reads_do_not_mutate_artifacts_or_state() -> None:
    datastore, state = _state_store()
    before = _store_digest(datastore.z)

    read_assay_state(datastore.zw, "RNA")
    normalized_path_from_state(datastore.zw, "RNA", "I")
    embedding_initialization_path_from_state(datastore.zw, "RNA", "I")
    resolve_graph_selection(
        datastore,
        state.connectivity_map,
        from_assay="RNA",
        cell_key="I",
    )

    assert _store_digest(datastore.z) == before


def test_incomplete_selected_artifact_is_not_silently_used() -> None:
    datastore, state = _state_store()
    datastore.z[artifact_path(state.normalized)].attrs["complete"] = False

    with pytest.raises(ArtifactResolutionError) as caught:
        read_assay_state(datastore.zw, "RNA")
    assert caught.value.code == "incomplete_artifact"

    datastore.z[artifact_path(state.normalized)].attrs["complete"] = True
    datastore.z[artifact_path(state.connectivity_map)].attrs["complete"] = False
    with pytest.raises(ArtifactResolutionError) as caught:
        read_assay_state(datastore.zw, "RNA")
    assert caught.value.code == "incomplete_artifact"


@pytest.mark.parametrize(
    ("failure", "expected_code"),
    [
        ("missing", "missing_artifact"),
        ("incomplete", "incomplete_artifact"),
    ],
)
def test_state_read_tolerates_unavailable_optional_named_results_but_write_rejects(
    failure: str,
    expected_code: str,
) -> None:
    datastore, state = _state_store()
    result = state.named_results["umap"]
    if failure == "missing":
        del datastore.z[artifact_path(result)]
    else:
        datastore.z[artifact_path(result)].attrs["complete"] = False
    before = _store_digest(datastore.z)

    assert read_assay_state(datastore.zw, "RNA") == state
    with pytest.raises(ArtifactResolutionError) as caught:
        write_assay_state(datastore.zw, state)

    assert caught.value.code == expected_code
    assert caught.value.context["field"] == "named_results['umap']"
    assert _store_digest(datastore.z) == before


@pytest.mark.parametrize(
    ("failure", "expected_code"),
    [
        ("missing", "artifact_missing"),
        ("incomplete", "artifact_incomplete"),
    ],
)
def test_selection_artifact_failures_are_structured_through_state_lookup(
    failure: str,
    expected_code: str,
) -> None:
    datastore, state = _state_store()
    assert state.normalized is not None
    inputs = inspect_artifact(datastore.zw, state.normalized).inputs or {}
    raw_selection = inputs.get("cell_selection")
    assert isinstance(raw_selection, dict)
    selection = ArtifactRef.from_dict(raw_selection)
    if failure == "missing":
        del datastore.z[artifact_path(selection)]
    else:
        datastore.z[artifact_path(selection)].attrs["complete"] = False
    before = _store_digest(datastore.z)

    with pytest.raises(ArtifactResolutionError) as caught:
        read_assay_state(datastore.zw, "RNA")

    assert caught.value.code == expected_code
    assert caught.value.context["artifact_id"] == selection.artifact_id
    assert _store_digest(datastore.z) == before


def test_matching_state_does_not_fall_back_to_stale_legacy_paths() -> None:
    datastore, state = _state_store()
    datastore.z.create_group("RNA/normed__I__hvgs")
    state_without_normalized = replace(state, normalized=None)
    datastore.z["RNA/state"].attrs["state"] = state_without_normalized.to_dict()

    with pytest.raises(IncompatibleAnalysisStateError) as caught:
        read_assay_state(datastore.zw, "RNA")
    assert caught.value.code == "invalid_analysis_state"


def test_state_rejects_unrelated_complete_artifact_chains() -> None:
    datastore, state = _state_store()
    wrong_neighbors, _group = _add_artifact(
        datastore.z,
        kind="neighbors",
        operation="query_neighbors",
        parameters={"k": 3},
        inputs={
            "ann_index": state.ann_index,
            "coordinates": state.batch_correction,
        },
    )
    mismatched = replace(state, neighbors=wrong_neighbors)
    datastore.z["RNA/state"].attrs["state"] = mismatched.to_dict()

    with pytest.raises(IncompatibleAnalysisStateError) as caught:
        read_assay_state(datastore.zw, "RNA")
    assert caught.value.code == "invalid_analysis_state"


def test_state_rejects_incomplete_or_missing_graph_inputs() -> None:
    datastore, state = _state_store()
    datastore.z[artifact_path(state.feature_scaling)].attrs["complete"] = False
    with pytest.raises(ArtifactResolutionError) as caught:
        read_assay_state(datastore.zw, "RNA")
    assert caught.value.code == "incomplete_artifact"

    datastore.z[artifact_path(state.batch_correction)].attrs["complete"] = True
    without_scaling = replace(state, feature_scaling=None)
    datastore.z["RNA/state"].attrs["state"] = without_scaling.to_dict()
    with pytest.raises(IncompatibleAnalysisStateError) as caught:
        read_assay_state(datastore.zw, "RNA")
    assert caught.value.code == "invalid_analysis_state"

    datastore.z["RNA/state"].attrs["state"] = state.to_dict()
    datastore.z[artifact_path(state.feature_scaling)].attrs["complete"] = True
    datastore.z[artifact_path(state.batch_correction)].attrs["complete"] = False
    with pytest.raises(ArtifactResolutionError) as caught:
        read_assay_state(datastore.zw, "RNA")
    assert caught.value.code == "incomplete_artifact"


def test_republishing_same_normalized_ref_drops_unavailable_current_graph() -> None:
    datastore, state = _state_store()
    assert state.normalized is not None
    assert state.connectivity_map is not None
    del datastore.z[artifact_path(state.connectivity_map)]

    datastore._publish_current_artifact(state.normalized, update_state=True)

    recovered = read_assay_state(datastore.zw, "RNA")
    assert recovered is not None
    assert recovered.normalized == state.normalized
    assert recovered.feature_scaling is None
    assert recovered.reduction is None
    assert recovered.batch_correction is None
    assert recovered.ann_index is None
    assert recovered.embedding_initialization is None
    assert recovered.neighbors is None
    assert recovered.connectivity_map is None
