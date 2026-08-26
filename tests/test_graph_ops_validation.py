"""Early-exit validation tests for graph operations."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

import scarf.graph.state as graph_state
from scarf.datastore._operations.graph import _sampling_fraction
from scarf.datastore.graph_datastore import GraphDataStore
from scarf.graph.feature_projection import resolve_native_graph_inputs
from scarf.graph.state import GraphSelection
from scarf.storage.artifacts import (
    ArtifactRef,
    artifact_path,
    fingerprint_array,
    fingerprint_strings,
    make_provenance,
    new_artifact_id,
)
from scarf.storage.errors import ArtifactResolutionError


class _BareGraphStore(GraphDataStore):
    @property
    def assay_names(self) -> list[str]:
        return []


def _bare_store() -> _BareGraphStore:
    store = object.__new__(_BareGraphStore)
    store.z = zarr.open_group(store=MemoryStore(), mode="w")
    store.workspace = None
    store.zarr_mode = "r+"
    store._defaultAssay = None
    store.nthreads = 1
    return store


def _complete_artifact(
    store: _BareGraphStore,
    kind: str,
    *,
    assay: str | None = "RNA",
    inputs: dict[str, object] | None = None,
    execution_options: dict[str, object] | None = None,
    operation: str | None = None,
    parameters: dict[str, object] | None = None,
    arrays: dict[str, np.ndarray] | None = None,
) -> ArtifactRef:
    ref = ArtifactRef(
        scope="assay" if assay is not None else "datastore",
        assay=assay,
        kind=kind,
        artifact_id=new_artifact_id(),
    )
    group = store.zw.create_group(artifact_path(ref))
    group.attrs.update(
        {
            "artifact_id": ref.artifact_id,
            "kind": kind,
            "provenance": make_provenance(
                operation=operation or f"test_{kind}",
                parameters=parameters or {},
                inputs=inputs or {},
            ),
            "execution_options": execution_options or {},
            "complete": True,
        }
    )
    for name, values in (arrays or {}).items():
        group.create_array(name, data=values)
    return ref


def _ref(
    kind: str,
    token: str,
    *,
    scope: str = "assay",
    assay: str | None = "RNA",
) -> ArtifactRef:
    return ArtifactRef(
        scope=scope,
        assay=assay,
        kind=kind,
        artifact_id=token * 64,
    )


def test_sampling_fraction_validates_type_and_range():
    assert _sampling_fraction(0.5, "frac") == 0.5
    assert _sampling_fraction(1, "frac") == 1.0
    with pytest.raises(TypeError, match="must be a number"):
        _sampling_fraction(True, "frac")
    with pytest.raises(TypeError, match="must be a number"):
        _sampling_fraction("x", "frac")
    with pytest.raises(ValueError, match="greater than 0 and at most 1"):
        _sampling_fraction(0.0, "frac")
    with pytest.raises(ValueError, match="greater than 0 and at most 1"):
        _sampling_fraction(1.1, "frac")
    with pytest.raises(ValueError, match="greater than 0 and at most 1"):
        _sampling_fraction(float("nan"), "frac")


def test_require_complete_artifact_rejects_kind_and_assay_mismatch(monkeypatch):
    store = _bare_store()
    ref = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="neighbors",
        artifact_id="b" * 64,
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError(
            "storage lookup must not run for local validation failures"
        )

    monkeypatch.setattr(
        "scarf.datastore._operations.graph.require_complete_artifact",
        fail_if_called,
    )
    with pytest.raises(ValueError, match="Expected 'ann_index' artifact"):
        store._require_complete_artifact(ref, "ann_index")

    wrong_assay = ArtifactRef(
        scope="assay",
        assay="ADT",
        kind="neighbors",
        artifact_id="c" * 64,
    )
    with pytest.raises(ValueError, match="must belong to assay 'RNA'"):
        store._require_complete_artifact(wrong_assay, "neighbors", assay="RNA")


def test_artifact_input_ref_requires_named_input():
    store = _bare_store()
    ref = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="neighbors",
        artifact_id="b" * 64,
    )
    store._require_complete_artifact = Mock(return_value=SimpleNamespace(inputs={}))
    with pytest.raises(ValueError, match="has no 'coordinates' input"):
        store._artifact_input_ref(ref, "coordinates", "reduction")


def test_run_lsi_and_custom_reduction_validate_before_work():
    store = _bare_store()
    store._run_reduction_artifact = Mock(
        side_effect=AssertionError("reduction must not run")
    )

    with pytest.raises(ValueError, match="solver must be"):
        store.run_lsi(solver="mystery")
    with pytest.raises(TypeError, match="n_iter must be an integer"):
        store.run_lsi(n_iter=True)
    with pytest.raises(ValueError, match="n_iter must be nonnegative"):
        store.run_lsi(n_iter=-1)
    with pytest.raises(TypeError, match="n_oversamples must be an integer"):
        store.run_lsi(n_oversamples=True)
    with pytest.raises(ValueError, match="n_oversamples must be nonnegative"):
        store.run_lsi(n_oversamples=-2)

    with pytest.raises(ValueError, match="two-dimensional matrix"):
        store.run_custom_reduction(np.arange(4))
    with pytest.raises(ValueError, match="two-dimensional matrix"):
        store.run_custom_reduction(np.zeros((5, 0)))
    store._run_reduction_artifact.assert_not_called()


def test_explicit_normalized_ref_can_rebuild_past_unavailable_current_graph(
    monkeypatch,
):
    store = _bare_store()
    store.zarr_loc = "memory"
    missing_graph = _ref("connectivity_map", "9")
    state_group = store.zw.create_group("RNA/state")
    state_group.attrs["state"] = graph_state.AssayState(
        assay="RNA",
        cell_key="I",
        connectivity_map=missing_graph,
    ).to_dict()
    normalized = _complete_artifact(
        store,
        "normalized",
        execution_options={"cell_key": "I"},
        arrays={"data": np.zeros((4, 3), dtype=np.float32)},
    )
    expected = _ref("reduction", "8")

    def reduction_without_numerics(_store, **kwargs):
        assert kwargs["normalized"] == normalized
        return expected

    monkeypatch.setattr(
        type(store),
        "_run_reduction_artifact_impl",
        reduction_without_numerics,
    )

    assert (
        store.run_pca(
            normalized,
            dims=2,
            local_cache=False,
            update_state=False,
        )
        == expected
    )


def test_artifact_chain_rejects_corrupt_coordinate_links():
    store = _bare_store()
    first_coordinates = _complete_artifact(store, "reduction")
    second_coordinates = _complete_artifact(store, "reduction")
    ann = _complete_artifact(
        store,
        "ann_index",
        inputs={"coordinates": first_coordinates},
    )
    neighbors = _complete_artifact(
        store,
        "neighbors",
        inputs={
            "ann_index": ann,
            "coordinates": second_coordinates,
        },
    )

    with pytest.raises(
        ValueError,
        match="Neighbors and ANN index use different coordinates",
    ):
        store._artifact_chain_state(neighbors)

    ann_without_coordinates = _complete_artifact(store, "ann_index")
    with pytest.raises(ValueError, match="ANN artifact has no coordinates input"):
        store._artifact_chain_state(ann_without_coordinates)

    imported = _complete_artifact(store, "imported_coordinates")
    imported_ann = _complete_artifact(
        store,
        "ann_index",
        inputs={"coordinates": imported},
    )
    with pytest.raises(
        ValueError,
        match="Imported-coordinate artifact has no cell_key",
    ):
        store._artifact_chain_state(imported_ann)


def test_build_ann_index_rejects_invalid_coordinate_contracts_before_loading():
    store = _bare_store()
    store._coordinate_source = Mock(
        side_effect=AssertionError("coordinates must not be loaded")
    )
    detached = ArtifactRef(
        scope="datastore",
        kind="reduction",
        artifact_id="1" * 64,
    )
    reduction = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="reduction",
        artifact_id="3" * 64,
    )

    with pytest.raises(ValueError, match="Coordinate artifact has no assay"):
        store.build_ann_index(detached)
    with pytest.raises(TypeError, match="ann_parallel must be a boolean"):
        store.build_ann_index(reduction, ann_parallel="yes")

    store._coordinate_source.assert_not_called()


def test_query_neighbors_rejects_ann_coordinate_contracts_before_loading():
    store = _bare_store()
    ann = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="ann_index",
        artifact_id="4" * 64,
    )
    stored_coordinates = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="reduction",
        artifact_id="5" * 64,
    )
    other_coordinates = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="reduction",
        artifact_id="6" * 64,
    )
    store._coordinate_source = Mock(
        side_effect=AssertionError("coordinates must not be loaded")
    )

    store._require_complete_artifact = Mock(
        return_value=SimpleNamespace(inputs={}, parameters={}, path="ann")
    )
    with pytest.raises(ValueError, match="ANN artifact has no coordinates input"):
        store.query_neighbors(ann)

    store._require_complete_artifact = Mock(
        return_value=SimpleNamespace(
            inputs={"coordinates": stored_coordinates.to_dict()},
            parameters={"ann_metric": "l2"},
            path="ann",
        )
    )
    with pytest.raises(
        ValueError,
        match="coordinates do not match the ANN artifact input",
    ):
        store.query_neighbors(ann, coordinates=other_coordinates)

    invalid_coordinates = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="normalized",
        artifact_id="7" * 64,
    )
    store._require_complete_artifact = Mock(
        return_value=SimpleNamespace(
            inputs={"coordinates": invalid_coordinates.to_dict()},
            parameters={"ann_metric": "l2"},
            path="ann",
        )
    )
    with pytest.raises(ValueError, match="ANN coordinates must be"):
        store.query_neighbors(ann)

    store._coordinate_source.assert_not_called()


def test_assay_state_rejects_malformed_models_and_storage() -> None:
    with pytest.raises(ValueError, match="valid assay name"):
        graph_state.AssayState(assay="", cell_key="I")
    with pytest.raises(ValueError, match="non-empty cell_key"):
        graph_state.AssayState(assay="RNA", cell_key="")

    detached = _ref("normalized", "1", scope="datastore", assay=None)
    with pytest.raises(ValueError, match="must reference assay"):
        graph_state.AssayState(
            assay="RNA",
            cell_key="I",
            normalized=detached,
        )

    wrong_named_kind = _ref("neighbors", "2")
    with pytest.raises(ValueError, match="requires kind 'mapping_reference'"):
        graph_state.AssayState(
            assay="RNA",
            cell_key="I",
            named_results={"mapping_reference": wrong_named_kind},
        )

    valid_fields = {"assay": "RNA", "cell_key": "I"}
    with pytest.raises(ValueError, match="named_results must be a mapping"):
        graph_state.AssayState.from_dict({**valid_fields, "named_results": []})
    with pytest.raises(ValueError, match="Every named result"):
        graph_state.AssayState.from_dict(
            {**valid_fields, "named_results": {1: wrong_named_kind.to_dict()}}
        )
    with pytest.raises(
        graph_state.IncompatibleAnalysisStateError,
        match="missing required fields",
    ) as missing:
        graph_state.AssayState.from_dict(valid_fields)
    assert missing.value.code == "invalid_analysis_state"
    with pytest.raises(
        graph_state.IncompatibleAnalysisStateError,
        match="removed feature-key contract",
    ) as legacy:
        graph_state.AssayState.from_dict({**valid_fields, "feat_key": "I"})
    assert legacy.value.code == "legacy_feature_contract"
    with pytest.raises(ValueError, match="Invalid assay name"):
        graph_state.assay_state_path("RNA/bad")

    root = zarr.open_group(store=MemoryStore(), mode="w")
    state_group = root.create_group("RNA/state")
    state_group.attrs["state"] = []
    with pytest.raises(ValueError, match="must be a mapping"):
        graph_state.read_assay_state(root, "RNA")

    state_group.attrs["state"] = graph_state.AssayState(
        assay="ADT",
        cell_key="I",
    ).to_dict()
    with pytest.raises(ValueError, match="names assay 'ADT'"):
        graph_state.read_assay_state(root, "RNA")


def test_integrated_graph_resolution_rejects_bad_indexes_and_missing_labels() -> None:
    store = _bare_store()
    store._integratedGraphsLoc = "integratedGraphs"
    index = store.zw.create_group("integratedGraphs")
    integrated = _ref("integrated_graph", "5", scope="datastore", assay=None)

    index.attrs["artifacts"] = []
    with pytest.raises(RuntimeError, match="artifact index is invalid"):
        graph_state.integrated_graph_label(store, integrated)

    index.attrs["artifacts"] = {"joint": "bad"}
    with pytest.raises(RuntimeError, match="index for 'joint' is invalid"):
        graph_state.integrated_graph_label(store, integrated)

    index.attrs["artifacts"] = {}
    with pytest.raises(KeyError, match="not registered under a label"):
        graph_state.integrated_graph_label(store, integrated)


@pytest.mark.parametrize(
    ("kind", "state_fields", "expected_code"),
    [
        (
            "connectivity_map",
            {"assay": "RNA", "cell_key": "I", "feat_key": "I"},
            "legacy_feature_contract",
        ),
        (
            "integrated_graph",
            {"assay": "RNA", "cell_key": "I", "unexpected": True},
            "invalid_analysis_state",
        ),
    ],
)
def test_explicit_graph_rejects_incompatible_state_before_execution(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    state_fields: dict[str, object],
    expected_code: str,
) -> None:
    store = _bare_store()
    store._integratedGraphsLoc = "integratedGraphs"
    store._get_assay = Mock(return_value=SimpleNamespace(name="RNA"))
    store._store_to_sparse = Mock(
        side_effect=AssertionError("graph execution must not start")
    )
    selection = _complete_artifact(
        store,
        "cell_selection",
        assay=None,
        execution_options={"source_column": "I"},
    )
    graph = _complete_artifact(
        store,
        kind,
        assay="RNA" if kind == "connectivity_map" else None,
    )
    if kind == "integrated_graph":
        store.zw.create_group("integratedGraphs").attrs["artifacts"] = {
            "joint": graph.to_dict()
        }
    state_group = store.zw.create_group("RNA/state")
    state_group.attrs["state"] = state_fields
    before = dict(state_group.attrs)
    monkeypatch.setattr(
        "scarf.graph.feature_projection.graph_cell_selection",
        lambda *_args: selection,
    )
    monkeypatch.setattr(
        "scarf.graph.feature_projection.graph_source_assays",
        lambda *_args: ("RNA",),
    )
    monkeypatch.setattr(
        graph_state,
        "validate_cell_selection_artifact",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(graph_state.IncompatibleAnalysisStateError) as error:
        store.load_graph(graph, from_assay="RNA")

    assert error.value.code == expected_code
    store._store_to_sparse.assert_not_called()
    assert dict(state_group.attrs) == before


def test_selection_artifact_rejects_changed_array_geometry() -> None:
    store = _bare_store()
    ids = np.asarray(["c0", "c1"])
    cell_data = store.zw.create_group("cellData")
    cell_data.create_array("ids", data=ids)
    cell_data.create_array("I", data=np.ones(2, dtype=bool))
    selection = _complete_artifact(
        store,
        "cell_selection",
        assay=None,
        inputs={"ordered_row_ids_fingerprint": fingerprint_strings(ids)},
        arrays={"values": np.ones(1, dtype=bool)},
    )

    with pytest.raises(
        ArtifactResolutionError,
        match="row identity does not match its metadata table",
    ) as error:
        graph_state.validate_cell_selection_artifact(store.zw, selection, "I")
    assert error.value.code == "row_identity_mismatch"


def _imported_coordinate_store(
    *,
    inputs: dict[str, object],
    parameters: dict[str, object] | None = None,
    arrays: dict[str, np.ndarray] | None = None,
) -> tuple[_BareGraphStore, ArtifactRef]:
    store = _bare_store()
    ids = np.asarray(["c0", "c1"])
    cell_data = store.zw.create_group("cellData")
    cell_data.create_array("ids", data=ids)
    cell_data.create_array("I", data=np.ones(2, dtype=bool))
    selection = _complete_artifact(
        store,
        "cell_selection",
        assay=None,
        inputs={
            "ordered_row_ids_fingerprint": fingerprint_strings(ids),
            "values_fingerprint": fingerprint_array(np.ones(2, dtype=bool)),
        },
        execution_options={"source_column": "I"},
        arrays={"values": np.ones(2, dtype=bool)},
    )
    coordinates = _complete_artifact(
        store,
        "imported_coordinates",
        operation="import_dimreduc",
        inputs={"cell_selection": selection, **inputs},
        execution_options={"cell_key": "I", "block_rows": 2},
        parameters=parameters,
        arrays=arrays,
    )
    return store, coordinates


def test_imported_coordinate_validation_rejects_missing_core_contracts() -> None:
    store, coordinates = _imported_coordinate_store(inputs={})
    with pytest.raises(ValueError, match="source digest is missing"):
        graph_state.validate_imported_coordinates_artifact(store.zw, coordinates)

    store, coordinates = _imported_coordinate_store(
        inputs={
            "source_digest": {"bytes_hex": "0" * 64},
            "payload_fingerprints": {},
        }
    )
    with pytest.raises(ValueError, match="has no data array"):
        graph_state.validate_imported_coordinates_artifact(store.zw, coordinates)

    store, coordinates = _imported_coordinate_store(
        inputs={
            "source_digest": {"bytes_hex": "0" * 64},
            "payload_fingerprints": {"data": "0" * 64},
        },
        parameters={"dimreduc_key": "pca", "dims": True},
        arrays={"data": np.zeros((2, 2), dtype=np.float32)},
    )
    with pytest.raises(ValueError, match="dimensions are missing"):
        graph_state.validate_imported_coordinates_artifact(store.zw, coordinates)


def test_graph_selection_validators_reject_detached_and_mixed_chains() -> None:
    store = _bare_store()
    detached_connectivity = _ref(
        "connectivity_map",
        "6",
        scope="datastore",
        assay=None,
    )
    detached_neighbors = _ref("neighbors", "7", scope="datastore", assay=None)
    detached_normalized = _ref("normalized", "8", scope="datastore", assay=None)
    with pytest.raises(ValueError, match="has no assay"):
        graph_state.validate_artifact_graph_selection(
            store.zw,
            detached_connectivity,
            "I",
        )
    with pytest.raises(ValueError, match="has no assay"):
        graph_state.validate_neighbors_artifact_selection(
            store.zw,
            detached_neighbors,
            "I",
        )
    with pytest.raises(ValueError, match="has no assay"):
        graph_state.validate_normalized_artifact_selection(
            store.zw,
            detached_normalized,
            "I",
        )

    invalid_coordinates = _complete_artifact(store, "normalized")
    ann = _complete_artifact(
        store,
        "ann_index",
        inputs={"coordinates": invalid_coordinates},
    )
    neighbors = _complete_artifact(
        store,
        "neighbors",
        inputs={
            "ann_index": ann,
            "coordinates": invalid_coordinates,
        },
    )
    with pytest.raises(ValueError, match="Neighbor coordinates must be"):
        graph_state.validate_neighbors_artifact_selection(
            store.zw,
            neighbors,
            "I",
        )

    first_reduction = _complete_artifact(store, "reduction")
    second_reduction = _complete_artifact(store, "reduction")
    mismatched_ann = _complete_artifact(
        store,
        "ann_index",
        inputs={"coordinates": second_reduction},
    )
    mismatched_neighbors = _complete_artifact(
        store,
        "neighbors",
        inputs={
            "ann_index": mismatched_ann,
            "coordinates": first_reduction,
        },
    )
    with pytest.raises(ValueError, match="different coordinates"):
        graph_state.validate_neighbors_artifact_selection(
            store.zw,
            mismatched_neighbors,
            "I",
        )


def _connectivity_chain(
    store: _BareGraphStore,
    coordinates: ArtifactRef,
    *,
    ann_coordinates: ArtifactRef | None = None,
    normalized: ArtifactRef | None = None,
    feature_scaling: ArtifactRef | None = None,
) -> ArtifactRef:
    if normalized is not None and feature_scaling is not None:
        coordinates = _complete_artifact(
            store,
            "reduction",
            inputs={
                "normalized": normalized,
                "feature_scaling": feature_scaling,
            },
        )
    ann = _complete_artifact(
        store,
        "ann_index",
        inputs={"coordinates": ann_coordinates or coordinates},
    )
    neighbors = _complete_artifact(
        store,
        "neighbors",
        inputs={"ann_index": ann, "coordinates": coordinates},
    )
    return _complete_artifact(
        store,
        "connectivity_map",
        inputs={"neighbors": neighbors},
    )


def test_native_graph_projection_rejects_invalid_artifact_chains() -> None:
    store = _bare_store()
    with pytest.raises(ArtifactResolutionError) as wrong_scope:
        resolve_native_graph_inputs(
            store.zw,
            _ref("connectivity_map", "9", scope="datastore", assay=None),
        )
    assert wrong_scope.value.code == "wrong_scope"
    with pytest.raises(ArtifactResolutionError) as wrong_kind:
        resolve_native_graph_inputs(
            store.zw,
            _ref("ann_index", "a"),
        )
    assert wrong_kind.value.code == "unsupported_graph_kind"

    imported = _complete_artifact(store, "imported_coordinates")
    imported_graph = _connectivity_chain(store, imported)
    with pytest.raises(ArtifactResolutionError) as imported_error:
        resolve_native_graph_inputs(store.zw, imported_graph)
    assert imported_error.value.code == "corrupt_payload"

    reduction = _complete_artifact(store, "reduction")
    other_reduction = _complete_artifact(store, "reduction")
    mismatched = _connectivity_chain(
        store,
        reduction,
        ann_coordinates=other_reduction,
    )
    with pytest.raises(
        ArtifactResolutionError,
        match="different coordinate artifacts",
    ) as mismatched_error:
        resolve_native_graph_inputs(store.zw, mismatched)
    assert mismatched_error.value.code == "corrupt_payload"

    normalized = _complete_artifact(store, "normalized", execution_options={})
    scaling = _complete_artifact(store, "feature_scaling")
    missing_selection_keys = _connectivity_chain(
        store,
        reduction,
        normalized=normalized,
        feature_scaling=scaling,
    )
    with pytest.raises(
        ArtifactResolutionError, match="cell_selection"
    ) as missing_error:
        resolve_native_graph_inputs(store.zw, missing_selection_keys)
    assert missing_error.value.code == "corrupt_payload"


def test_normalization_requires_explicit_feature_selection() -> None:
    store = _bare_store()
    with pytest.raises(ValueError, match="No assay was provided"):
        store.run_normalization(features="all_features")

    store._defaultAssay = "RNA"
    with pytest.raises(TypeError, match="required keyword-only argument: 'features'"):
        store.run_normalization()


def _reduction_validation_store(
    shape: tuple[int, int],
    *,
    execution_options: dict[str, object] | None = None,
) -> tuple[_BareGraphStore, ArtifactRef]:
    store = _bare_store()
    group = store.zw.create_group("normalized")
    group.create_array("data", data=np.zeros(shape, dtype=np.float32))
    normalized = _ref("normalized", "b")
    store._require_complete_artifact = Mock(
        return_value=SimpleNamespace(
            path="normalized",
            execution_options=(
                {"cell_key": "I"} if execution_options is None else execution_options
            ),
        )
    )
    return store, normalized


def _run_reduction_validation(
    store: _BareGraphStore,
    normalized: ArtifactRef,
    *,
    method: str,
    dims: int,
    batch_size: int,
    custom_loadings: np.ndarray | None = None,
    lsi_skip_first: bool = False,
) -> ArtifactRef:
    return store._run_reduction_artifact_impl(
        method=method,
        normalized=normalized,
        from_assay=None,
        dims=dims,
        pca_cell_key=None,
        feat_scaling=True,
        lsi_skip_first=lsi_skip_first,
        custom_loadings=custom_loadings,
        rand_state=1,
        batch_size=batch_size,
        show_elbow_plot=False,
        update_state=False,
        invalidate_cache=False,
    )


def test_reduction_stages_validate_contracts_before_compute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "scarf.datastore._operations.graph.validate_normalized_artifact_selection",
        lambda *_args, **_kwargs: None,
    )

    store, _normalized = _reduction_validation_store((4, 4))
    with pytest.raises(ValueError, match="Normalized artifact has no assay"):
        _run_reduction_validation(
            store,
            _ref("normalized", "c", scope="datastore", assay=None),
            method="custom",
            dims=2,
            batch_size=4,
            custom_loadings=np.ones((4, 2)),
        )

    store, normalized = _reduction_validation_store((4, 4), execution_options={})
    with pytest.raises(ValueError, match="has no cell_key"):
        _run_reduction_validation(
            store,
            normalized,
            method="custom",
            dims=2,
            batch_size=4,
            custom_loadings=np.ones((4, 2)),
        )

    store, normalized = _reduction_validation_store((4, 3))
    with pytest.raises(ValueError, match="two-dimensional"):
        _run_reduction_validation(
            store,
            normalized,
            method="custom",
            dims=2,
            batch_size=4,
            custom_loadings=np.ones(3),
        )
    with pytest.raises(ValueError, match="rows must match"):
        _run_reduction_validation(
            store,
            normalized,
            method="custom",
            dims=2,
            batch_size=4,
            custom_loadings=np.ones((2, 2)),
        )
    with pytest.raises(ValueError, match="at least one dimension"):
        _run_reduction_validation(
            store,
            normalized,
            method="custom",
            dims=2,
            batch_size=4,
            custom_loadings=np.ones((3, 0)),
        )

    store, normalized = _reduction_validation_store((4, 4))
    store.cells = SimpleNamespace(fetch=Mock(return_value=np.ones(4, dtype=int)))
    with pytest.raises(TypeError, match="one boolean value"):
        _run_reduction_validation(
            store,
            normalized,
            method="pca",
            dims=2,
            batch_size=4,
        )

    store, normalized = _reduction_validation_store((4, 2))
    store.cells = SimpleNamespace(fetch=Mock(return_value=np.ones(4, dtype=bool)))
    with pytest.raises(ValueError, match="selected features"):
        _run_reduction_validation(
            store,
            normalized,
            method="pca",
            dims=2,
            batch_size=4,
        )

    store, normalized = _reduction_validation_store((4, 4))
    store.cells = SimpleNamespace(fetch=Mock(return_value=np.ones(4, dtype=bool)))
    with pytest.raises(ValueError, match="batch_size"):
        _run_reduction_validation(
            store,
            normalized,
            method="pca",
            dims=2,
            batch_size=2,
        )

    store, normalized = _reduction_validation_store((4, 4))
    store.cells = SimpleNamespace(
        fetch=Mock(return_value=np.ones(4, dtype=bool)),
        fetch_all=Mock(return_value=np.ones(4, dtype=int)),
    )
    with pytest.raises(TypeError, match="boolean column"):
        _run_reduction_validation(
            store,
            normalized,
            method="pca",
            dims=2,
            batch_size=4,
        )

    store, normalized = _reduction_validation_store((2, 2))
    with pytest.raises(ValueError, match="exceed the normalized matrix rank"):
        _run_reduction_validation(
            store,
            normalized,
            method="lsi",
            dims=2,
            batch_size=2,
            lsi_skip_first=True,
        )


def test_harmony_validates_artifact_and_batch_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "scarf.datastore._operations.graph.validate_normalized_artifact_selection",
        lambda *_args, **_kwargs: None,
    )
    store = _bare_store()
    store._require_complete_artifact = Mock(
        return_value=SimpleNamespace(execution_options={"cell_key": "I"})
    )
    with pytest.raises(ValueError, match="Reduction artifact has no assay"):
        store.run_harmony(
            [],
            reduction=_ref("reduction", "d", scope="datastore", assay=None),
        )

    reduction = _ref("reduction", "e")
    normalized = _ref("normalized", "f")
    store._artifact_input_ref = Mock(return_value=normalized)
    store._require_complete_artifact = Mock(
        side_effect=lambda ref, *_args, **_kwargs: SimpleNamespace(
            execution_options=({"cell_key": "I"} if ref == normalized else {})
        )
    )
    with pytest.raises(ValueError, match="non-empty list"):
        store.run_harmony([], reduction=reduction)
    with pytest.raises(ValueError, match="must be unique"):
        store.run_harmony(["batch", "batch"], reduction=reduction)

    store._require_complete_artifact = Mock(
        side_effect=lambda *_args, **_kwargs: SimpleNamespace(execution_options={})
    )
    with pytest.raises(ValueError, match="has no cell_key"):
        store.run_harmony(["batch"], reduction=reduction)


def test_stage_entrypoints_reject_missing_selected_artifacts() -> None:
    store = _bare_store()
    with pytest.raises(ValueError, match="No assay was provided"):
        store._selected_artifact(None, "normalized", "normalized")
    with pytest.raises(ValueError, match="No assay was provided"):
        store.build_embedding_initialization()
    with pytest.raises(ValueError, match="No assay was provided"):
        store.build_ann_index()

    store._defaultAssay = "RNA"
    with pytest.raises(KeyError, match="no selected artifact state"):
        store._selected_artifact(None, "normalized", "normalized")
    with pytest.raises(KeyError, match="no selected reduction"):
        store.build_embedding_initialization()
    with pytest.raises(KeyError, match="no selected reduction"):
        store.build_ann_index()


def test_embedding_initialization_rejects_detached_and_tiny_sources() -> None:
    store = _bare_store()
    with pytest.raises(ValueError, match="Reduction artifact has no assay"):
        store._build_embedding_initialization(
            _ref("reduction", "1", scope="datastore", assay=None),
            n_centroids=2,
            rand_state=1,
            batch_size=None,
            invalidate_cache=False,
        )

    store._coordinate_source = Mock(return_value=(SimpleNamespace(data=None), 1, 2))
    with pytest.raises(ValueError, match="at least two cells and centroids"):
        store._build_embedding_initialization(
            _ref("reduction", "2"),
            n_centroids=2,
            rand_state=1,
            batch_size=None,
            invalidate_cache=False,
        )


def _neighbor_query_store(
    *,
    n_cells: int,
    metric: str,
) -> tuple[_BareGraphStore, ArtifactRef, ArtifactRef]:
    store = _bare_store()
    ann = _ref("ann_index", "3")
    coordinates = _ref("reduction", "4")
    ann_status = SimpleNamespace(
        inputs={"coordinates": coordinates.to_dict()},
        parameters={"ann_metric": metric},
        path="ann",
    )
    store._require_complete_artifact = Mock(
        side_effect=lambda ref, *_args, **_kwargs: (
            ann_status
            if ref == ann
            else SimpleNamespace(inputs={}, parameters={}, path="coordinates")
        )
    )
    store._coordinate_source = Mock(
        return_value=(SimpleNamespace(data=None), n_cells, 2)
    )
    return store, ann, coordinates


def test_neighbor_query_and_connectivity_fail_before_expensive_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _bare_store()
    store._require_complete_artifact = Mock(
        return_value=SimpleNamespace(inputs={}, parameters={}, path="ann")
    )
    with pytest.raises(ValueError, match="ANN artifact has no assay"):
        store.query_neighbors(
            _ref("ann_index", "5", scope="datastore", assay=None),
            update_state=False,
        )

    store, ann, _coordinates = _neighbor_query_store(n_cells=1, metric="l2")
    with pytest.raises(ValueError, match="at least two cells"):
        store.query_neighbors(ann, update_state=False)

    store, ann, _coordinates = _neighbor_query_store(
        n_cells=3,
        metric="euclidean",
    )
    with pytest.raises(ValueError, match="supported distance metric"):
        store.query_neighbors(ann, update_state=False)

    store, ann, _coordinates = _neighbor_query_store(n_cells=3, metric="l2")
    result = _ref("neighbors", "6")
    store._plan_assay_artifact = Mock(
        return_value=SimpleNamespace(ref=result, reused=False)
    )
    store._resolve_ann_index = Mock(
        side_effect=ArtifactResolutionError(
            "ANN artifact has no persisted Zarr index bytes",
            code="corrupt_payload",
            context={"artifact_id": ann.artifact_id},
        )
    )
    start_write = Mock(side_effect=AssertionError("artifact write must not start"))
    monkeypatch.setattr(
        "scarf.datastore._operations.graph.start_artifact",
        start_write,
    )
    with pytest.raises(ArtifactResolutionError) as caught:
        store.query_neighbors(ann, update_state=False)
    assert caught.value.code == "corrupt_payload"
    start_write.assert_not_called()

    detached_neighbors = _ref("neighbors", "7", scope="datastore", assay=None)
    store._require_complete_artifact = Mock(
        return_value=SimpleNamespace(path="neighbors")
    )
    with pytest.raises(ValueError, match="Neighbors artifact has no assay"):
        store.build_connectivity_map(detached_neighbors, update_state=False)


def test_run_umap_rejects_cell_key_that_does_not_match_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _bare_store()
    graph = _complete_artifact(store, "connectivity_map")
    cells = _complete_artifact(store, "cell_selection", assay=None)
    other_cells = _complete_artifact(store, "cell_selection", assay=None)
    monkeypatch.setattr(
        "scarf.datastore._operations.embeddings.resolve_graph_selection",
        lambda *_args, **_kwargs: GraphSelection(
            graph_loc=artifact_path(graph),
            graph_ref=graph,
            from_assay="RNA",
            cell_key="I",
            integrated_label=None,
        ),
    )
    monkeypatch.setattr(store, "load_graph", Mock(return_value=np.eye(2)))
    monkeypatch.setattr(
        store,
        "_get_ini_embed",
        Mock(return_value=(np.zeros((2, 2)), graph)),
    )
    monkeypatch.setattr(store, "_ensure_cell_selection", Mock(return_value=cells))
    monkeypatch.setattr(store, "_graph_cell_selection", Mock(return_value=other_cells))
    with pytest.raises(ValueError, match="cell_key does not match the graph"):
        store.run_umap(graph, n_epochs=1)
