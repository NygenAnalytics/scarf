"""Early-exit validation tests for graph operations."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

import scarf.graph.state as graph_state
from scarf.assay import ATACassay
from scarf.datastore._operations.graph import _sampling_fraction
from scarf.datastore.graph_datastore import GraphDataStore
from scarf.storage.artifacts import (
    ArtifactRef,
    artifact_path,
    fingerprint_strings,
    make_provenance,
    new_artifact_id,
)


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


def test_get_latest_keys_requires_default_assay_and_skips_resolution_when_explicit():
    store = _bare_store()
    with pytest.raises(ValueError, match="No default assay"):
        store._get_latest_keys(None, None, None)

    store._defaultAssay = "RNA"
    store._get_latest_cell_key = Mock(
        side_effect=AssertionError("explicit keys must not resolve cell key")
    )
    store._get_latest_feat_key = Mock(
        side_effect=AssertionError("explicit keys must not resolve feat key")
    )
    assert store._get_latest_keys("ADT", "custom", "feats") == (
        "ADT",
        "custom",
        "feats",
    )
    store._get_latest_cell_key.assert_not_called()
    store._get_latest_feat_key.assert_not_called()


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
        match="Imported coordinates are not part of the normalized AssayState",
    ):
        store._artifact_chain_state(imported_ann)


def test_artifact_chain_requires_paired_selection_overrides():
    store = _bare_store()
    normalized = _complete_artifact(
        store,
        "normalized",
        execution_options={"cell_key": "cells", "feat_key": "features"},
    )

    with pytest.raises(
        ValueError,
        match="cell_key and feat_key overrides must be provided together",
    ):
        store._artifact_chain_state(
            normalized,
            cell_key_override="other_cells",
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"log_transform": "yes"}, "log_transform must be a boolean"),
        (
            {"renormalize_subset": 1},
            "renormalize_subset must be a boolean",
        ),
    ],
)
def test_run_normalization_validates_atac_boolean_parameters_before_work(
    kwargs,
    message,
):
    store = _bare_store()
    store._defaultAssay = "RNA"
    assay = object.__new__(ATACassay)
    assay.z = store.zw.create_group("RNA")
    assay.z.create_group("featureData")
    assay.feats = SimpleNamespace(
        columns=["I"],
        fetch_all=Mock(return_value=np.ones(3, dtype=bool)),
    )
    assay.normMethod = "tfidf"
    assay.sf = None
    store.cells = SimpleNamespace(
        fetch_all=Mock(return_value=np.ones(3, dtype=bool)),
    )
    store._get_assay = Mock(return_value=assay)
    store._resolve_selection_input = Mock(
        side_effect=AssertionError("selection artifacts must not be resolved")
    )

    with pytest.raises(TypeError, match=message):
        store.run_normalization(
            from_assay="RNA",
            feat_key="I",
            **kwargs,
        )

    store._resolve_selection_input.assert_not_called()


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
    imported = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="imported_coordinates",
        artifact_id="2" * 64,
    )
    reduction = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="reduction",
        artifact_id="3" * 64,
    )

    with pytest.raises(ValueError, match="Coordinate artifact has no assay"):
        store.build_ann_index(detached)
    with pytest.raises(
        ValueError,
        match="Imported coordinates cannot activate AssayState",
    ):
        store.build_ann_index(imported)
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
        graph_state.AssayState(assay="", cell_key="I", feat_key="I")
    with pytest.raises(ValueError, match="requires cell_key and feat_key"):
        graph_state.AssayState(assay="RNA", cell_key="", feat_key="I")

    detached = _ref("normalized", "1", scope="datastore", assay=None)
    with pytest.raises(ValueError, match="must reference assay"):
        graph_state.AssayState(
            assay="RNA",
            cell_key="I",
            feat_key="I",
            normalized=detached,
        )

    wrong_named_kind = _ref("neighbors", "2")
    with pytest.raises(ValueError, match="requires kind 'mapping_reference'"):
        graph_state.AssayState(
            assay="RNA",
            cell_key="I",
            feat_key="I",
            named_results={"mapping_reference": wrong_named_kind},
        )

    valid_fields = {"assay": "RNA", "cell_key": "I", "feat_key": "I"}
    with pytest.raises(TypeError, match="named_results must be a mapping"):
        graph_state.AssayState.from_dict({**valid_fields, "named_results": []})
    with pytest.raises(TypeError, match="Every named result"):
        graph_state.AssayState.from_dict(
            {**valid_fields, "named_results": {1: wrong_named_kind.to_dict()}}
        )
    with pytest.raises(ValueError, match="Invalid assay name"):
        graph_state.assay_state_path("RNA/bad")

    root = zarr.open_group(store=MemoryStore(), mode="w")
    state_group = root.create_group("RNA/state")
    state_group.attrs["state"] = []
    with pytest.raises(TypeError, match="must be a mapping"):
        graph_state.read_assay_state(root, "RNA")

    state_group.attrs["state"] = graph_state.AssayState(
        assay="ADT",
        cell_key="I",
        feat_key="I",
    ).to_dict()
    with pytest.raises(ValueError, match="names assay 'ADT'"):
        graph_state.read_assay_state(root, "RNA")


def test_legacy_graph_selection_reports_unresolvable_and_stale_inputs() -> None:
    store = _bare_store()
    with pytest.raises(ValueError, match="provenance cannot be resolved"):
        graph_state.validate_legacy_graph_selection(
            store,
            "not-an-encoded-graph",
            "RNA",
            "I",
            "I",
        )

    graph_loc = "RNA/normed__I__I/reduction__pca__2__I/ann__l2__50__50__48__1/knn__2"
    with pytest.raises(ValueError, match="normalized data is missing"):
        graph_state.validate_legacy_graph_selection(
            store,
            graph_loc,
            "RNA",
            "I",
            "I",
        )

    normalized = store.zw.create_group("RNA/normed__I__I")
    normalized.attrs["subset_hash"] = 0
    store.cells = SimpleNamespace(active_index=Mock(return_value=np.array([0])))
    store._get_assay = Mock(
        return_value=SimpleNamespace(
            feats=SimpleNamespace(active_index=Mock(side_effect=KeyError("gone")))
        )
    )
    with pytest.raises(ValueError, match="selection columns cannot be validated"):
        graph_state.validate_legacy_graph_selection(
            store,
            graph_loc,
            "RNA",
            "I",
            "I",
        )


def test_named_result_and_selected_paths_reject_stale_state() -> None:
    store = _bare_store()
    mapping = _complete_artifact(store, "mapping_reference", inputs={})
    batch_correction = _ref("batch_correction", "3")
    corrected_state = graph_state.AssayState(
        assay="RNA",
        cell_key="I",
        feat_key="I",
        batch_correction=batch_correction,
    )
    assert (
        graph_state.named_result_mismatch(
            store.zw,
            "mapping_reference",
            mapping,
            corrected_state,
        )
        == "Plain PCA mapping reference cannot select batch correction"
    )

    empty_state = graph_state.AssayState(assay="RNA", cell_key="I", feat_key="I")
    reason = graph_state.named_result_mismatch(
        store.zw,
        "mapping_reference",
        mapping,
        empty_state,
    )
    assert reason is not None
    assert reason.startswith("Mapping reference state is missing")

    with pytest.raises(ValueError, match="Mapping reference state is missing"):
        graph_state.write_assay_state(
            store.zw,
            graph_state.AssayState(
                assay="RNA",
                cell_key="I",
                feat_key="I",
                named_results={"mapping_reference": mapping},
            ),
        )

    missing_normalized = _ref("normalized", "4")
    store.zw.create_group("RNA/state").attrs["state"] = graph_state.AssayState(
        assay="RNA",
        cell_key="I",
        feat_key="I",
        normalized=missing_normalized,
    ).to_dict()
    with pytest.raises(KeyError, match="normalized artifact does not exist"):
        graph_state.normalized_path_from_state(store.zw, "RNA", "I", "I")

    store.zw["RNA/state"].attrs["state"] = empty_state.to_dict()
    with pytest.raises(KeyError, match="no selected embedding initialization"):
        graph_state.embedding_initialization_path_from_state(
            store.zw,
            "RNA",
            "I",
            "I",
        )

    assert graph_state._parameters(store.zw, missing_normalized) == {}
    with pytest.raises(ValueError, match="has no 'coordinates' artifact input"):
        graph_state._input_ref(store.zw, mapping, "coordinates")


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

    store._get_latest_keys = Mock(return_value=("RNA", "I", "I"))
    store._resolve_integrated_graph_path = Mock(return_value="missing/integrated")
    with pytest.raises(KeyError, match="does not exist"):
        graph_state.resolve_graph_selection(
            store,
            None,
            from_assay=None,
            cell_key=None,
            feat_key=None,
            integrated_graph="missing",
        )


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
        graph_state.ArtifactSelectionError,
        match="no longer matches its artifact",
    ) as error:
        graph_state.validate_cell_selection_artifact(store.zw, selection, "I")
    assert error.value.code == "selection_values_changed"


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
        inputs={"ordered_row_ids_fingerprint": fingerprint_strings(ids)},
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
            "I",
        )
    with pytest.raises(ValueError, match="has no assay"):
        graph_state.validate_neighbors_artifact_selection(
            store.zw,
            detached_neighbors,
            "I",
            "I",
        )
    with pytest.raises(ValueError, match="has no assay"):
        graph_state.validate_normalized_artifact_selection(
            store.zw,
            detached_normalized,
            "I",
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


def test_stored_graph_resolution_rejects_invalid_artifact_chains() -> None:
    store = _bare_store()
    with pytest.raises(ValueError, match="must be assay-scoped"):
        graph_state.stored_assay_graph_from_ref(
            store.zw,
            _ref("connectivity_map", "9", scope="datastore", assay=None),
        )
    with pytest.raises(ValueError, match="Expected a connectivity_map"):
        graph_state.stored_assay_graph_from_ref(
            store.zw,
            _ref("neighbors", "a"),
        )

    imported = _complete_artifact(store, "imported_coordinates")
    invalid = _connectivity_chain(store, imported)
    with pytest.raises(ValueError, match="reduction or batch_correction"):
        graph_state.stored_assay_graph_from_ref(store.zw, invalid)

    reduction = _complete_artifact(store, "reduction")
    other_reduction = _complete_artifact(store, "reduction")
    mismatched = _connectivity_chain(
        store,
        reduction,
        ann_coordinates=other_reduction,
    )
    with pytest.raises(ValueError, match="different coordinates"):
        graph_state.stored_assay_graph_from_ref(store.zw, mismatched)

    normalized = _complete_artifact(store, "normalized", execution_options={})
    scaling = _complete_artifact(store, "feature_scaling")
    missing_selection_keys = _connectivity_chain(
        store,
        reduction,
        normalized=normalized,
        feature_scaling=scaling,
    )
    with pytest.raises(ValueError, match="selection keys are missing"):
        graph_state.stored_assay_graph_from_ref(
            store.zw,
            missing_selection_keys,
        )


def test_provenance_scalar_helpers_reject_wrong_types() -> None:
    assert graph_state._optional_str(None) is None
    assert graph_state._optional_bool(None) is None
    with pytest.raises(TypeError, match="must be a string"):
        graph_state._optional_str(1)
    with pytest.raises(TypeError, match="must be an integer"):
        graph_state._optional_int(True)
    with pytest.raises(TypeError, match="must be boolean"):
        graph_state._optional_bool("yes")
    with pytest.raises(TypeError, match="must be a string"):
        graph_state._required_str({"value": 1}, "value", "test")
    with pytest.raises(TypeError, match="must be an integer"):
        graph_state._required_int({"value": True}, "value", "test")
    with pytest.raises(TypeError, match="must be numeric"):
        graph_state._required_float({"value": "one"}, "value", "test")
    with pytest.raises(TypeError, match="must be boolean"):
        graph_state._required_bool({"value": 1}, "value", "test")


def test_normalization_rejects_missing_and_invalid_selections() -> None:
    store = _bare_store()
    with pytest.raises(ValueError, match="No assay was provided"):
        store.run_normalization(feat_key="I")

    store._defaultAssay = "RNA"
    with pytest.raises(ValueError, match="feat_key is required"):
        store.run_normalization()

    feats = SimpleNamespace(columns=[], fetch_all=Mock())
    assay = SimpleNamespace(feats=feats)
    store._get_assay = Mock(return_value=assay)
    with pytest.raises(KeyError, match="Feature selection column"):
        store.run_normalization(feat_key="I")

    feats.columns = ["I"]
    store.cells = SimpleNamespace(fetch_all=Mock(return_value=np.ones(2, dtype=int)))
    feats.fetch_all.return_value = np.ones(2, dtype=bool)
    with pytest.raises(TypeError, match="selections must be boolean"):
        store.run_normalization(feat_key="I")

    store.cells.fetch_all.return_value = np.zeros(2, dtype=bool)
    with pytest.raises(ValueError, match="requires selected cells and features"):
        store.run_normalization(feat_key="I")


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
                {"cell_key": "I", "feat_key": "I"}
                if execution_options is None
                else execution_options
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
    with pytest.raises(ValueError, match="has no cell_key or feat_key"):
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
        return_value=SimpleNamespace(
            execution_options={"cell_key": "I", "feat_key": "I"}
        )
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
            execution_options=(
                {"cell_key": "I", "feat_key": "I"} if ref == normalized else {}
            )
        )
    )
    with pytest.raises(ValueError, match="non-empty list"):
        store.run_harmony([], reduction=reduction)
    with pytest.raises(ValueError, match="must be unique"):
        store.run_harmony(["batch", "batch"], reduction=reduction)

    store._require_complete_artifact = Mock(
        side_effect=lambda *_args, **_kwargs: SimpleNamespace(execution_options={})
    )
    with pytest.raises(ValueError, match="has no cell_key or feat_key"):
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


def test_neighbor_query_and_connectivity_fail_before_expensive_work() -> None:
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
    store._resolve_ann_index = Mock(return_value=None)
    with pytest.raises(RuntimeError, match="no readable index"):
        store.query_neighbors(ann, update_state=False)

    detached_neighbors = _ref("neighbors", "7", scope="datastore", assay=None)
    store._require_complete_artifact = Mock(
        return_value=SimpleNamespace(path="neighbors")
    )
    with pytest.raises(ValueError, match="Neighbors artifact has no assay"):
        store.build_connectivity_map(detached_neighbors, update_state=False)
