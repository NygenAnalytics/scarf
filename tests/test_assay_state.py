import hashlib
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.datastore.datastore import DataStore
from scarf.datastore.graph_datastore import GraphDataStore
from scarf.graph.encoded_paths import (
    make_cell_graph_group_path,
    make_kmeans_initialization_group_path,
    make_nearest_neighbors_group_path,
    make_neighbor_index_group_path,
    make_normalized_group_path,
    make_reduction_group_path,
)
from scarf.graph.paths import AssayGraphPaths, StoredAssayGraph, StoredIntegratedGraph
from scarf.graph.state import (
    ArtifactSelectionError,
    AssayState,
    _legacy_subset_hash,
    read_assay_state,
    stored_assay_graph_from_state,
    validate_legacy_graph_selection,
    write_assay_state,
)
from scarf.storage.artifacts import (
    ArtifactRef,
    ArtifactScope,
    artifact_path,
    fingerprint_strings,
    inspect_artifact,
    list_artifacts,
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


def _compose_assay_graph_paths(
    *,
    from_assay: str,
    cell_key: str,
    feat_key: str,
    reduction_method: str,
    dims: int,
    pca_cell_key: str,
    ann_metric: str,
    ann_efc: int,
    ann_ef: int,
    ann_m: int,
    rand_state: int,
    k: int,
    local_connectivity: float,
    bandwidth: float,
    n_centroids: int | None = None,
    feat_scaling: bool = True,
    harmony_contract_hash: str | None = None,
) -> AssayGraphPaths:
    normalized = make_normalized_group_path(from_assay, cell_key, feat_key)
    reduction = make_reduction_group_path(
        normalized, reduction_method, dims, pca_cell_key
    )
    neighbor_index = make_neighbor_index_group_path(
        reduction,
        ann_metric,
        ann_efc,
        ann_ef,
        ann_m,
        rand_state,
        feat_scaling=feat_scaling,
        harmony_contract_hash=harmony_contract_hash,
    )
    nearest_neighbors = make_nearest_neighbors_group_path(neighbor_index, k)
    cell_graph = make_cell_graph_group_path(
        nearest_neighbors, local_connectivity, bandwidth
    )
    kmeans = None
    if n_centroids is not None:
        kmeans = make_kmeans_initialization_group_path(
            reduction, n_centroids, rand_state
        )
    return AssayGraphPaths(
        normalized_group_path=normalized,
        reduction_group_path=reduction,
        neighbor_index_group_path=neighbor_index,
        nearest_neighbors_group_path=nearest_neighbors,
        cell_graph_group_path=cell_graph,
        kmeans_initialization_group_path=kmeans,
    )


def test_legacy_graph_without_selection_provenance_fails_closed(
    datastore_ephemeral,
) -> None:
    paths = _compose_assay_graph_paths(
        from_assay="RNA",
        cell_key="I",
        feat_key="I",
        reduction_method="pca",
        dims=5,
        pca_cell_key="I",
        ann_metric="l2",
        ann_efc=50,
        ann_ef=50,
        ann_m=16,
        rand_state=1,
        k=3,
        local_connectivity=1.0,
        bandwidth=1.0,
    )
    datastore_ephemeral.zw.require_group(paths.normalized_group_path)

    with pytest.raises(ValueError, match="selection provenance is missing"):
        validate_legacy_graph_selection(
            datastore_ephemeral,
            paths.nearest_neighbors_group_path,
            "RNA",
            "I",
            "I",
        )


@pytest.mark.parametrize("hash_format", ["current", "legacy"])
def test_legacy_graph_accepts_supported_selection_hashes(
    datastore_ephemeral,
    hash_format: str,
) -> None:
    paths = _compose_assay_graph_paths(
        from_assay="RNA",
        cell_key="I",
        feat_key="I",
        reduction_method="pca",
        dims=5,
        pca_cell_key="I",
        ann_metric="l2",
        ann_efc=50,
        ann_ef=50,
        ann_m=16,
        rand_state=1,
        k=3,
        local_connectivity=1.0,
        bandwidth=1.0,
    )
    normalized = datastore_ephemeral.zw.require_group(paths.normalized_group_path)
    assay = datastore_ephemeral._get_assay("RNA")
    cell_indices = datastore_ephemeral.cells.active_index("I")
    feature_indices = assay.feats.active_index("I")
    normalized.attrs["subset_hash"] = (
        assay._create_subset_hash(cell_indices, feature_indices)
        if hash_format == "current"
        else _legacy_subset_hash(cell_indices, feature_indices)
    )

    validate_legacy_graph_selection(
        datastore_ephemeral,
        paths.nearest_neighbors_group_path,
        "RNA",
        "I",
        "I",
    )


def test_legacy_graph_rejects_mismatched_or_invalid_selection_hash(
    datastore_ephemeral,
) -> None:
    paths = _compose_assay_graph_paths(
        from_assay="RNA",
        cell_key="I",
        feat_key="I",
        reduction_method="pca",
        dims=5,
        pca_cell_key="I",
        ann_metric="l2",
        ann_efc=50,
        ann_ef=50,
        ann_m=16,
        rand_state=1,
        k=3,
        local_connectivity=1.0,
        bandwidth=1.0,
    )
    normalized = datastore_ephemeral.zw.require_group(paths.normalized_group_path)
    normalized.attrs["subset_hash"] = 0
    with pytest.raises(ValueError, match="no longer matches"):
        validate_legacy_graph_selection(
            datastore_ephemeral,
            paths.nearest_neighbors_group_path,
            "RNA",
            "I",
            "I",
        )

    normalized.attrs["subset_hash"] = True
    with pytest.raises(ValueError, match="invalid type"):
        validate_legacy_graph_selection(
            datastore_ephemeral,
            paths.nearest_neighbors_group_path,
            "RNA",
            "I",
            "I",
        )


def test_legacy_graph_qualifies_non_default_feature_key_exactly_once(
    datastore_ephemeral,
) -> None:
    assay = datastore_ephemeral._get_assay("RNA")
    qualified_key = "I__I__qualified"
    assay.feats.insert(
        qualified_key,
        np.asarray(assay.feats.fetch_all("I"), dtype=bool),
        overwrite=True,
    )
    paths = _compose_assay_graph_paths(
        from_assay="RNA",
        cell_key="I",
        feat_key="I__qualified",
        reduction_method="pca",
        dims=5,
        pca_cell_key="I",
        ann_metric="l2",
        ann_efc=50,
        ann_ef=50,
        ann_m=16,
        rand_state=1,
        k=3,
        local_connectivity=1.0,
        bandwidth=1.0,
    )
    normalized = datastore_ephemeral.zw.require_group(paths.normalized_group_path)
    normalized.attrs["subset_hash"] = _legacy_subset_hash(
        datastore_ephemeral.cells.active_index("I"),
        assay.feats.active_index(qualified_key),
    )

    validate_legacy_graph_selection(
        datastore_ephemeral,
        paths.nearest_neighbors_group_path,
        "RNA",
        "I",
        "I__qualified",
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
    feature_data.create_array("I__hvgs", data=np.ones(4, dtype=bool))

    cell_selection, cell_selection_group = _add_artifact(
        datastore.z,
        kind="cell_selection",
        operation="manual_selection",
        inputs={
            "ordered_row_ids_fingerprint": fingerprint_strings(cell_ids),
            "values_fingerprint": "cells",
        },
        execution_options={"source_column": "I"},
        scope="datastore",
    )
    cell_selection_group.create_array(
        "values",
        data=np.ones(3, dtype=bool),
    )
    feature_selection, feature_selection_group = _add_artifact(
        datastore.z,
        kind="feature_selection",
        operation="manual_selection",
        inputs={
            "ordered_row_ids_fingerprint": fingerprint_strings(feature_ids),
            "values_fingerprint": "features",
        },
        execution_options={"source_column": "I__hvgs"},
    )
    feature_selection_group.create_array(
        "values",
        data=np.ones(4, dtype=bool),
    )
    normalized, normalized_group = _add_artifact(
        datastore.z,
        kind="normalized",
        operation="run_normalization",
        inputs={
            "cell_selection": cell_selection,
            "feature_selection": feature_selection,
        },
        execution_options={"cell_key": "I", "feat_key": "hvgs"},
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
        feat_key="hvgs",
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
    public_datastore = DataStore.__new__(DataStore)
    public_datastore.z = datastore.z
    public_datastore.workspace = None
    public_datastore._defaultAssay = "RNA"
    assert public_datastore.get_assay_state() == state

    with pytest.raises(ValueError, match="requires kind"):
        AssayState(
            assay="RNA",
            cell_key="I",
            feat_key="hvgs",
            normalized=_ref("neighbors", "a"),
        )
    with pytest.raises(ValueError, match="snake_case"):
        AssayState(
            assay="RNA",
            cell_key="I",
            feat_key="hvgs",
            named_results={"UMAP 1": _ref("embedding", "a")},
        )
    malformed = state.to_dict()
    malformed["normalized"] = "not-a-reference"
    with pytest.raises(TypeError, match="normalized must be"):
        AssayState.from_dict(malformed)
    malformed = state.to_dict()
    malformed["cell_key"] = None
    with pytest.raises(TypeError, match="must be strings"):
        AssayState.from_dict(malformed)


def test_state_first_graph_lookup_uses_artifacts() -> None:
    datastore, state = _state_store()

    assert read_assay_state(datastore.zw, "RNA") == state
    assert datastore.get_normalized_group_path("RNA", "I", "hvgs") == artifact_path(
        state.normalized
    )
    assert datastore.get_latest_graph_loc("RNA", "I", "hvgs") == artifact_path(
        state.connectivity_map
    )
    assert datastore.load_graph().shape == (3, 3)

    stored = datastore._lookup_stored_graph("RNA", "I", "hvgs")
    assert isinstance(stored, StoredAssayGraph)
    assert stored.dims == 2
    assert stored.k == 2
    assert stored.ann_metric == "l2"
    assert stored.local_connectivity == 1.0
    assert stored.paths.kmeans_initialization_group_path == artifact_path(
        state.embedding_initialization
    )


def test_state_graph_summary_rejects_missing_required_provenance() -> None:
    datastore, state = _state_store()
    assert state.ann_index is not None
    ann_group = datastore.zw[artifact_path(state.ann_index)]
    provenance = dict(ann_group.attrs["provenance"])
    parameters = dict(provenance["parameters"])
    del parameters["ann_efc"]
    provenance["parameters"] = parameters
    ann_group.attrs["provenance"] = provenance

    with pytest.raises(ValueError, match="ann_index provenance is missing 'ann_efc'"):
        datastore._lookup_stored_graph("RNA", "I", "hvgs")


def test_state_graph_summary_rejects_invalid_optional_provenance_type() -> None:
    datastore, state = _state_store()
    assert state.reduction is not None
    reduction_group = datastore.zw[artifact_path(state.reduction)]
    provenance = dict(reduction_group.attrs["provenance"])
    parameters = dict(provenance["parameters"])
    parameters["feat_scaling"] = "false"
    provenance["parameters"] = parameters
    reduction_group.attrs["provenance"] = provenance

    with pytest.raises(TypeError, match="must be boolean"):
        datastore._lookup_stored_graph("RNA", "I", "hvgs")


def test_explicit_artifact_graph_path_loads_without_latest_attrs() -> None:
    datastore, state = _state_store()
    graph_path = artifact_path(state.connectivity_map)

    graph = datastore.load_graph(graph_loc=graph_path)
    stored = datastore._lookup_stored_graph(graph_loc=graph_path)

    assert graph.shape == (3, 3)
    assert graph.nnz == 6
    assert isinstance(stored, StoredAssayGraph)
    assert stored.paths.cell_graph_group_path == graph_path

    older_ref, older = _add_artifact(
        datastore.z,
        kind="connectivity_map",
        operation="build_connectivity_map",
        parameters={"local_connectivity": 0.5, "bandwidth": 1.0},
        inputs={"neighbors": state.neighbors},
    )
    older.attrs["n_cells"] = 3
    older.attrs["n_neighbors"] = 2
    selected = datastore.z[graph_path]
    older.create_array("edges", data=np.asarray(selected["edges"][:]))
    older.create_array("weights", data=np.asarray(selected["weights"][:]) * 0.5)

    explicit_older = datastore.load_graph(graph_loc=artifact_path(older_ref))
    stored_older = datastore._lookup_stored_graph(graph_loc=artifact_path(older_ref))
    assert explicit_older.shape == (3, 3)
    assert explicit_older.nnz == 6
    assert isinstance(stored_older, StoredAssayGraph)
    assert stored_older.local_connectivity == 0.5


def test_explicit_integrated_artifact_loads_without_legacy_slot() -> None:
    datastore, state = _state_store()
    assert state.normalized is not None
    normalized_inputs = inspect_artifact(datastore.zw, state.normalized).inputs or {}
    raw_selection = normalized_inputs.get("cell_selection")
    assert isinstance(raw_selection, dict)
    cell_selection = ArtifactRef.from_dict(raw_selection)
    ref, group = _add_artifact(
        datastore.z,
        kind="integrated_graph",
        operation="integrate_assays",
        parameters={"method": "wnn"},
        inputs={
            "rna": state.connectivity_map,
            "cell_selection": cell_selection,
        },
        scope="datastore",
    )
    group.attrs["n_cells"] = 3
    group.attrs["n_neighbors"] = 2
    group.create_array(
        "edges",
        data=np.array(
            [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]],
            dtype=np.uint64,
        ),
    )
    group.create_array(
        "weights",
        data=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
    )

    graph = datastore.load_graph(graph_loc=artifact_path(ref))
    stored = datastore._lookup_stored_graph(graph_loc=artifact_path(ref))

    assert graph.shape == (3, 3)
    assert graph.nnz == 6
    assert isinstance(stored, StoredIntegratedGraph)
    assert stored.cell_graph_group_path == artifact_path(ref)
    assert stored.n_cells == 3
    assert stored.n_neighbors == 2


@pytest.mark.parametrize(
    ("failure", "expected_code"),
    [
        ("missing", "artifact_missing"),
        ("incomplete", "artifact_incomplete"),
    ],
)
def test_integrated_graph_selection_failures_are_structured(
    failure: str,
    expected_code: str,
) -> None:
    datastore, state = _state_store()
    assert state.normalized is not None
    normalized_inputs = inspect_artifact(datastore.zw, state.normalized).inputs or {}
    raw_selection = normalized_inputs.get("cell_selection")
    assert isinstance(raw_selection, dict)
    selection = ArtifactRef.from_dict(raw_selection)
    graph_ref, _group = _add_artifact(
        datastore.z,
        kind="integrated_graph",
        operation="integrate_assays",
        inputs={"cell_selection": selection},
        scope="datastore",
    )
    if failure == "missing":
        del datastore.z[artifact_path(selection)]
    else:
        datastore.z[artifact_path(selection)].attrs["complete"] = False
    before = _store_digest(datastore.z)

    with pytest.raises(ArtifactSelectionError) as caught:
        datastore.load_graph(graph_loc=artifact_path(graph_ref))

    assert caught.value.code == expected_code
    assert caught.value.context["artifact_id"] == selection.artifact_id
    assert _store_digest(datastore.z) == before


def test_artifact_embedding_initialization_is_state_resolved() -> None:
    datastore, _state = _state_store()
    embedding = datastore._get_ini_embed("RNA", "I", "hvgs", 2)

    assert embedding.shape == (3, 2)
    assert np.isfinite(embedding).all()


def test_state_mismatch_keeps_legacy_normalized_path_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datastore, _state = _state_store()
    monkeypatch.setattr(
        "scarf.datastore._operations.graph.validate_legacy_graph_selection",
        lambda *_args, **_kwargs: None,
    )
    assert datastore.get_normalized_group_path("RNA", "I", "legacy") == (
        "RNA/normed__I__legacy"
    )
    stored = datastore._lookup_stored_graph("RNA", "I", "legacy")
    assert isinstance(stored, StoredAssayGraph)
    assert stored.paths.cell_graph_group_path.endswith("graph__1.0__1.5")
    assert stored.dims == 5
    assert stored.k == 3


def test_state_reads_do_not_mutate_artifacts_or_state() -> None:
    datastore, _state = _state_store()
    before = _store_digest(datastore.z)

    datastore.get_normalized_group_path("RNA", "I", "hvgs")
    datastore.get_latest_graph_loc("RNA", "I", "hvgs")
    datastore.load_graph(graph_loc=datastore.get_latest_graph_loc("RNA", "I", "hvgs"))
    stored_assay_graph_from_state(datastore.zw, "RNA", "I", "hvgs")

    assert _store_digest(datastore.z) == before


def test_incomplete_selected_artifact_is_not_silently_used() -> None:
    datastore, state = _state_store()
    datastore.z[artifact_path(state.normalized)].attrs["complete"] = False

    with pytest.raises(RuntimeError, match="incomplete"):
        datastore.get_normalized_group_path("RNA", "I", "hvgs")

    datastore.z[artifact_path(state.normalized)].attrs["complete"] = True
    datastore.z[artifact_path(state.connectivity_map)].attrs["complete"] = False
    with pytest.raises(RuntimeError, match="Graph artifact is incomplete"):
        datastore.load_graph(graph_loc=artifact_path(state.connectivity_map))


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

    with pytest.raises(ArtifactSelectionError) as caught:
        datastore.load_graph(from_assay="RNA", cell_key="I", feat_key="hvgs")

    assert caught.value.code == expected_code
    assert caught.value.context["artifact_id"] == selection.artifact_id
    assert _store_digest(datastore.z) == before


def test_matching_state_does_not_fall_back_to_stale_legacy_paths() -> None:
    datastore, state = _state_store()
    datastore.z.create_group("RNA/normed__I__hvgs")
    state_without_normalized = replace(state, normalized=None)
    datastore.z["RNA/state"].attrs["state"] = state_without_normalized.to_dict()

    with pytest.raises(KeyError, match="no selected normalized"):
        datastore.get_normalized_group_path("RNA", "I", "hvgs")


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

    with pytest.raises(ValueError, match="does not match AssayState"):
        datastore.get_latest_graph_loc("RNA", "I", "hvgs")


def test_state_rejects_incomplete_or_missing_graph_inputs() -> None:
    datastore, state = _state_store()
    datastore.z[artifact_path(state.feature_scaling)].attrs["complete"] = False
    with pytest.raises(RuntimeError, match="Artifact is incomplete"):
        datastore.get_latest_graph_loc("RNA", "I", "hvgs")

    datastore.z[artifact_path(state.batch_correction)].attrs["complete"] = True
    without_scaling = replace(state, feature_scaling=None)
    datastore.z["RNA/state"].attrs["state"] = without_scaling.to_dict()
    with pytest.raises(KeyError, match="feature_scaling"):
        datastore.get_latest_graph_loc("RNA", "I", "hvgs")

    datastore.z["RNA/state"].attrs["state"] = state.to_dict()
    datastore.z[artifact_path(state.feature_scaling)].attrs["complete"] = True
    datastore.z[artifact_path(state.batch_correction)].attrs["complete"] = False
    with pytest.raises(RuntimeError, match="Artifact is incomplete"):
        datastore.get_latest_graph_loc("RNA", "I", "hvgs")


def test_build_mapping_reference_preserves_existing_named_results(
    analyzed_datastore_ephemeral,
) -> None:
    datastore = analyzed_datastore_ephemeral
    start = datastore.get_assay_state("RNA")
    assert start is not None
    assert start.reduction is not None and start.neighbors is not None

    planted = replace(
        start,
        named_results={**dict(start.named_results), "pca": start.reduction},
    )
    write_assay_state(datastore.zw, planted)

    reference = datastore.build_mapping_reference(start.neighbors)
    after = datastore.get_assay_state("RNA")
    assert after is not None
    assert after.named_results["pca"] == start.reduction
    assert after.named_results["mapping_reference"] == reference.ref


def test_mapping_reference_and_graph_do_not_clear_each_other(
    analyzed_datastore_ephemeral,
) -> None:
    datastore = analyzed_datastore_ephemeral
    start = datastore.get_assay_state("RNA")
    assert start is not None
    neighbors = start.neighbors
    graph = start.connectivity_map
    assert neighbors is not None and graph is not None

    reference = datastore.build_mapping_reference(neighbors)
    after_reference = datastore.get_assay_state("RNA")
    assert after_reference is not None
    assert after_reference.named_results["mapping_reference"] == reference.ref
    # Publishing a mapping reference walks up to neighbors, so the graph is not
    # in that lineage and used to be dropped.
    assert after_reference.connectivity_map == graph
    assert after_reference.embedding_initialization is not None

    rebuilt = datastore.build_connectivity_map(neighbors, bandwidth=1.25)
    assert rebuilt != graph
    after_graph = datastore.get_assay_state("RNA")
    assert after_graph is not None
    assert after_graph.connectivity_map == rebuilt
    # The named handle is not derived from the graph, so rebuilding the graph
    # must leave it selectable.
    assert after_graph.named_results["mapping_reference"] == reference.ref
    assert datastore.get_mapping_reference().ref == reference.ref


def test_publishing_recovers_from_an_incomplete_graph_artifact(
    analyzed_datastore_ephemeral,
) -> None:
    datastore = analyzed_datastore_ephemeral
    start = datastore.get_assay_state("RNA")
    assert start is not None
    assert start.neighbors is not None and start.connectivity_map is not None
    datastore.z[artifact_path(start.connectivity_map)].attrs["complete"] = False

    datastore.build_mapping_reference(start.neighbors)

    state = datastore.get_assay_state("RNA")
    assert state is not None
    assert "mapping_reference" in state.named_results
    assert state.connectivity_map is None


def test_named_result_is_dropped_when_the_chain_moves_underneath_it(
    analyzed_datastore_ephemeral,
) -> None:
    datastore = analyzed_datastore_ephemeral
    start = datastore.get_assay_state("RNA")
    assert start is not None and start.neighbors is not None
    datastore.build_mapping_reference(start.neighbors)

    normalized = datastore.run_normalization(
        cell_key="I",
        feat_key="hvgs",
        update_state=False,
    )
    reduction = datastore.run_pca(normalized, dims=5, update_state=False)
    ann_index = datastore.build_ann_index(reduction, update_state=False)
    moved = datastore.query_neighbors(ann_index, coordinates=reduction, k=3)

    state = datastore.get_assay_state("RNA")
    assert state is not None
    assert state.neighbors == moved
    # Carrying the handle here would describe a chain that no longer exists, so
    # it is dropped rather than preserved into an inconsistent state.
    assert "mapping_reference" not in state.named_results
    assert state.connectivity_map is None


def test_derived_writers_do_not_mutate_graph_artifacts() -> None:
    datastore, state = _state_store()
    graph_path = artifact_path(state.connectivity_map)
    before = _store_digest(datastore.z[graph_path])

    datastore.get_diffusion_operator("RNA", "I", "hvgs", t=2)
    diffusion_refs = list_artifacts(
        datastore.z,
        scope="assay",
        assay="RNA",
        kind="diffusion_operator",
    )
    assert len(diffusion_refs) == 1
    # Artifact-backed Paris goes through _run_paris_from_artifacts. The stub
    # store lacks MetaData, so label persistence may fail after hierarchy work;
    # the connectivity artifact itself must stay unchanged either way.
    try:
        datastore._run_paris_from_artifacts(
            graph_ref=state.connectivity_map,
            graph_loc=artifact_path(state.connectivity_map),
            from_assay="RNA",
            label_assay="RNA",
            cell_key="I",
            fixed_cluster_count=2,
            effective_min_cluster_size=None,
            label="paris_cluster",
            force_recalc=False,
        )
    except AttributeError:
        pass
    assert _store_digest(datastore.z[graph_path]) == before
