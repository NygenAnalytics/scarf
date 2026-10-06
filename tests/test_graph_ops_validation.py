"""Early-exit validation tests for explicit-reference graph operations."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.datastore._operations.graph import _sampling_fraction
from scarf.datastore.graph_datastore import GraphDataStore
from scarf.graph.feature_projection import (
    resolve_native_graph_inputs,
)
from scarf.embeddings.imported_storage import validate_imported_coordinates_artifact
from scarf.storage.artifacts import (
    ArtifactRef,
    ArtifactStatus,
    artifact_path,
    fingerprint_array,
    fingerprint_strings,
    make_provenance,
    new_artifact_id,
)
from scarf.storage.errors import ArtifactResolutionError
from scarf.storage.selections import validate_stored_selection_integrity


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


def _cell_selection(
    store: _BareGraphStore,
    values: np.ndarray | None = None,
) -> ArtifactRef:
    ids = np.asarray(["c0", "c1"])
    selected = np.ones(2, dtype=bool) if values is None else values
    if "cellData" not in store.zw:
        cells = store.zw.create_group("cellData")
        cells.create_array("ids", data=ids)
        cells.create_array("I", data=np.ones(2, dtype=bool))
    return _complete_artifact(
        store,
        "cell_selection",
        assay=None,
        inputs={
            "ordered_row_ids_fingerprint": fingerprint_strings(ids),
            "values_fingerprint": fingerprint_array(selected),
        },
        execution_options={"source_column": "I"},
        arrays={"values": selected},
    )


def test_sampling_fraction_validates_type_and_range() -> None:
    assert _sampling_fraction(0.5, "frac") == 0.5
    assert _sampling_fraction(1, "frac") == 1.0
    for value in (True, "x"):
        with pytest.raises(TypeError, match="must be a number"):
            _sampling_fraction(value, "frac")
    for value in (0.0, 1.1, float("nan")):
        with pytest.raises(ValueError, match="greater than 0 and at most 1"):
            _sampling_fraction(value, "frac")


def test_require_complete_artifact_rejects_kind_and_assay_before_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _bare_store()
    storage_lookup = Mock(side_effect=AssertionError("storage lookup must not run"))
    monkeypatch.setattr(
        "scarf.datastore._operations.graph.require_complete_artifact",
        storage_lookup,
    )

    with pytest.raises(ValueError, match="Expected 'ann_index' artifact"):
        store._require_complete_artifact(_ref("neighbors", "b"), "ann_index")
    with pytest.raises(ValueError, match="must belong to assay 'RNA'"):
        store._require_complete_artifact(
            _ref("neighbors", "c", assay="ADT"),
            "neighbors",
            assay="RNA",
        )
    storage_lookup.assert_not_called()


def _status(
    ref: ArtifactRef,
    *,
    path: str,
    inputs: dict[str, object] | None = None,
    parameters: dict[str, object] | None = None,
) -> ArtifactStatus:
    return ArtifactStatus(
        ref=ref,
        path=path,
        exists=True,
        complete=True,
        provenance={
            "operation": "recorded",
            "parameters": parameters or {},
            "inputs": inputs or {},
        },
    )


def test_artifact_input_ref_requires_named_input() -> None:
    store = _bare_store()
    neighbors = _ref("neighbors", "b")
    store._require_complete_artifact = Mock(
        return_value=_status(neighbors, path="neighbors")
    )
    with pytest.raises(ValueError, match="has no 'coordinates' input"):
        store._artifact_input_ref(neighbors, "coordinates", "reduction")


def test_reduction_entrypoints_validate_and_forward_explicit_normalized_ref() -> None:
    store = _bare_store()
    normalized = _ref("normalized", "1")
    expected = _ref("reduction", "2")
    store._run_reduction_artifact = Mock(return_value=expected)

    with pytest.raises(ValueError, match="solver must be"):
        store.run_lsi(normalized, solver="mystery")
    with pytest.raises(TypeError, match="n_iter must be an integer"):
        store.run_lsi(normalized, n_iter=True)
    with pytest.raises(ValueError, match="n_oversamples must be at least 0"):
        store.run_lsi(normalized, n_oversamples=-1)
    with pytest.raises(ValueError, match="two-dimensional matrix"):
        store.run_custom_reduction(np.arange(4), normalized)

    assert store.run_pca(normalized, dims=2, local_cache=False) == expected
    assert store._run_reduction_artifact.call_args.kwargs["normalized"] == normalized


def test_build_ann_index_rejects_bad_coordinate_contract_before_loading() -> None:
    store = _bare_store()
    store._coordinate_source = Mock(
        side_effect=AssertionError("coordinates must not be loaded")
    )

    with pytest.raises(ValueError, match="Coordinate artifact has no assay"):
        store.build_ann_index(_ref("reduction", "1", scope="datastore", assay=None))
    with pytest.raises(TypeError, match="ann_parallel must be a boolean"):
        store.build_ann_index(_ref("reduction", "2"), ann_parallel="yes")
    store._coordinate_source.assert_not_called()


def test_query_neighbors_rejects_malformed_ann_lineage_before_loading() -> None:
    store = _bare_store()
    ann = _ref("ann_index", "3")
    coordinates = _ref("reduction", "4")
    other = _ref("reduction", "5")
    store._coordinate_source = Mock(
        side_effect=AssertionError("coordinates must not be loaded")
    )

    store._require_complete_artifact = Mock(return_value=_status(ann, path="ann"))
    with pytest.raises(ValueError, match="has no 'coordinates' input"):
        store.query_neighbors(ann)

    store._require_complete_artifact = Mock(
        return_value=_status(
            ann,
            path="ann",
            inputs={"coordinates": coordinates.to_dict()},
            parameters={"ann_metric": "l2"},
        )
    )
    with pytest.raises(ValueError, match="do not match the ANN artifact input"):
        store.query_neighbors(ann, coordinates=other)
    store._coordinate_source.assert_not_called()


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"ann_metric": "l2", "ann_parallel": False}, "ann_ef search depth"),
        (
            {"ann_metric": "l2", "ann_ef": True, "ann_parallel": False},
            "ann_ef search depth",
        ),
        # The query's thread count follows the recorded ann_parallel flag.
        ({"ann_metric": "l2", "ann_ef": 50}, "ann_parallel record"),
        (
            {"ann_metric": "l2", "ann_ef": 50, "ann_parallel": 1},
            "ann_parallel record",
        ),
    ],
)
def test_query_neighbors_requires_recorded_ann_search_settings(
    parameters: dict[str, object],
    message: str,
) -> None:
    store = _bare_store()
    ann = _ref("ann_index", "6")
    coordinates = _ref("reduction", "7")
    store._require_complete_artifact = Mock(
        side_effect=lambda ref, *_args, **_kwargs: (
            _status(
                ann,
                path="ann",
                inputs={"coordinates": coordinates.to_dict()},
                parameters=parameters,
            )
            if ref == ann
            else _status(ref, path="coordinates")
        )
    )
    store._coordinate_source = Mock(return_value=(SimpleNamespace(data=None), 3, 2))
    store._resolve_ann_index = Mock(
        side_effect=AssertionError("the ANN index must not be loaded")
    )

    with pytest.raises(ValueError, match=message):
        store.query_neighbors(ann)
    store._resolve_ann_index.assert_not_called()


def test_cell_selection_validation_rejects_changed_array_geometry() -> None:
    store = _bare_store()
    selection = _cell_selection(store, np.ones(1, dtype=bool))

    with pytest.raises(ArtifactResolutionError) as caught:
        validate_stored_selection_integrity(
            store.zw,
            selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
    assert caught.value.code == "row_identity_mismatch"


def _imported_coordinate_store(
    *,
    inputs: dict[str, object],
    parameters: dict[str, object] | None = None,
    arrays: dict[str, np.ndarray] | None = None,
) -> tuple[_BareGraphStore, ArtifactRef]:
    store = _bare_store()
    selection = _cell_selection(store)
    coordinates = _complete_artifact(
        store,
        "imported_coordinates",
        operation="import_dimreduc",
        inputs={"cell_selection": selection, **inputs},
        execution_options={"block_rows": 2},
        parameters=parameters,
        arrays=arrays,
    )
    return store, coordinates


def test_imported_coordinate_validation_rejects_missing_core_contracts() -> None:
    store, coordinates = _imported_coordinate_store(inputs={})
    with pytest.raises(ArtifactResolutionError, match="source digest is missing"):
        validate_imported_coordinates_artifact(store.zw, coordinates)

    store, coordinates = _imported_coordinate_store(
        inputs={
            "source_digest": {"bytes_hex": "0" * 64},
            "payload_fingerprints": {},
        }
    )
    with pytest.raises(ArtifactResolutionError, match="has no data array"):
        validate_imported_coordinates_artifact(store.zw, coordinates)

    store, coordinates = _imported_coordinate_store(
        inputs={
            "source_digest": {"bytes_hex": "0" * 64},
            "payload_fingerprints": {"data": "0" * 64},
        },
        parameters={"dimreduc_key": "pca", "dims": True},
        arrays={"data": np.zeros((2, 2), dtype=np.float32)},
    )
    with pytest.raises(ArtifactResolutionError, match="dimensions are missing"):
        validate_imported_coordinates_artifact(store.zw, coordinates)


def _connectivity_chain(
    store: _BareGraphStore,
    coordinates: ArtifactRef,
    *,
    ann_coordinates: ArtifactRef | None = None,
) -> ArtifactRef:
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


def test_native_graph_resolution_rejects_scope_kind_and_mixed_lineage() -> None:
    store = _bare_store()
    with pytest.raises(ArtifactResolutionError) as wrong_scope:
        resolve_native_graph_inputs(
            store.zw,
            _ref("connectivity_map", "6", scope="datastore", assay=None),
        )
    assert wrong_scope.value.code == "wrong_scope"

    with pytest.raises(ArtifactResolutionError) as wrong_kind:
        resolve_native_graph_inputs(store.zw, _ref("ann_index", "7"))
    assert wrong_kind.value.code == "unsupported_graph_kind"

    first = _complete_artifact(store, "reduction")
    second = _complete_artifact(store, "reduction")
    mixed = _connectivity_chain(store, first, ann_coordinates=second)
    with pytest.raises(
        ArtifactResolutionError,
        match="different coordinate artifacts",
    ) as caught:
        resolve_native_graph_inputs(store.zw, mixed)
    assert caught.value.code == "corrupt_payload"


def test_normalization_requires_explicit_selection_refs() -> None:
    store = _bare_store()
    with pytest.raises(TypeError, match="cell_selection must be an ArtifactRef"):
        store.run_normalization("I", _ref("feature_selection", "8"))
    with pytest.raises(TypeError, match="features must be an ArtifactRef"):
        store.run_normalization(
            _ref("cell_selection", "9", scope="datastore", assay=None),
            "all_features",
        )


def test_harmony_validates_public_batch_contract_before_snapshot() -> None:
    store = _bare_store()
    reduction = _ref("reduction", "a")
    store._resolve_harmony_reduction = Mock()

    with pytest.raises(ValueError, match="non-empty list"):
        store.run_harmony(reduction, [])
    with pytest.raises(ValueError, match="must be unique"):
        store.run_harmony(reduction, ["batch", "batch"])
    with pytest.raises(ValueError, match="non-empty strings"):
        store.run_harmony(reduction, [""])


def test_embedding_initialization_rejects_detached_and_tiny_sources() -> None:
    store = _bare_store()
    with pytest.raises(ValueError, match="Coordinate artifact has no assay"):
        store.build_embedding_initialization(
            _ref("reduction", "b", scope="datastore", assay=None),
            n_centroids=2,
        )

    store._coordinate_source = Mock(return_value=(SimpleNamespace(data=None), 1, 2))
    with pytest.raises(ValueError, match="at least two cells and centroids"):
        store.build_embedding_initialization(_ref("reduction", "c"), n_centroids=2)


def test_neighbor_and_connectivity_fail_before_expensive_compute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _bare_store()
    ann = _ref("ann_index", "d")
    coordinates = _ref("reduction", "e")
    store._require_complete_artifact = Mock(
        side_effect=lambda ref, *_args, **_kwargs: (
            _status(
                ann,
                path="ann",
                inputs={"coordinates": coordinates.to_dict()},
                parameters={"ann_metric": "l2"},
            )
            if ref == ann
            else _status(ref, path="coordinates")
        )
    )
    store._coordinate_source = Mock(return_value=(SimpleNamespace(data=None), 1, 2))
    with pytest.raises(ValueError, match="at least two cells"):
        store.query_neighbors(ann)

    detached = _ref("neighbors", "f", scope="datastore", assay=None)
    store._require_complete_artifact = Mock(
        return_value=SimpleNamespace(path="neighbors")
    )
    compute = Mock(side_effect=AssertionError("connectivity compute must not run"))
    monkeypatch.setattr(
        "scarf.neighbors.graph.build_connectivity_arrays",
        compute,
    )
    with pytest.raises(ArtifactResolutionError) as caught:
        store.build_connectivity_map(detached)
    assert caught.value.code == "wrong_scope"
    compute.assert_not_called()


def test_public_graph_entrypoints_require_explicit_refs() -> None:
    store = _bare_store()
    with pytest.raises(TypeError, match="required positional argument"):
        store.run_pca()
    with pytest.raises(TypeError, match="required positional argument"):
        store.build_embedding_initialization()
    with pytest.raises(TypeError, match="required positional argument"):
        store.build_ann_index()
    with pytest.raises(TypeError, match="required positional argument"):
        store.query_neighbors()
    with pytest.raises(TypeError, match="required positional argument"):
        store.build_connectivity_map()


def test_run_umap_rejects_non_artifact_graph_before_resolution() -> None:
    store = _bare_store()
    with pytest.raises(TypeError, match="graph must be an ArtifactRef"):
        store.run_umap(
            "RNA",
            np.zeros((2, 2)),
        )


def _neighbors_payload(
    store: _BareGraphStore,
    *,
    indices: np.ndarray | None = None,
    distances: np.ndarray | None = None,
    attrs: dict[str, object] | None = None,
) -> ArtifactRef:
    """Store a neighbors artifact over four cells with two neighbors each."""
    ref = _complete_artifact(
        store,
        "neighbors",
        arrays={
            "indices": (
                np.array([[1, 2], [0, 2], [0, 1], [0, 1]], dtype=np.uint32)
                if indices is None
                else indices
            ),
            "distances": (
                np.full((4, 2), 0.5, dtype=np.float32)
                if distances is None
                else distances
            ),
        },
    )
    group = store.zw[artifact_path(ref)]
    group.attrs.update({"n_cells": 4, "n_neighbors": 2, "self_hit_rate": 100.0})
    group.attrs.update(attrs or {})
    for name, value in (attrs or {}).items():
        if value is None:
            del group.attrs[name]
    return ref


def test_neighbor_payload_validation_requires_a_neighbors_artifact() -> None:
    from scarf.graph.distances import (
        validate_distance_provenance,
        validate_neighbors_payload,
    )

    store = _bare_store()
    connectivity = _ref("connectivity_map", "a")

    with pytest.raises(ValueError, match="Distance provenance requires a neighbors"):
        validate_distance_provenance(store.zw, connectivity)
    with pytest.raises(ValueError, match="validation requires a neighbors artifact"):
        validate_neighbors_payload(store.zw, connectivity)


def test_neighbor_payload_validation_returns_the_validated_dimensions() -> None:
    from scarf.graph.distances import validate_neighbors_payload

    store = _bare_store()
    for rate in (0, 37.5, 100):
        payload = validate_neighbors_payload(
            store.zw, _neighbors_payload(store, attrs={"self_hit_rate": rate})
        )
        assert (payload.n_cells, payload.n_neighbors) == (4, 2)
        np.testing.assert_array_equal(payload.indices[:, 0], [1, 0, 0, 0])


@pytest.mark.parametrize("rate", [None, True, "100", float("nan"), -1.0, 100.5])
def test_neighbor_payload_validation_rejects_an_invalid_self_hit_rate(rate) -> None:
    from scarf.graph.distances import validate_neighbors_payload

    store = _bare_store()
    neighbors = _neighbors_payload(store, attrs={"self_hit_rate": rate})

    with pytest.raises(
        ArtifactResolutionError, match="invalid self_hit_rate"
    ) as caught:
        validate_neighbors_payload(store.zw, neighbors)
    assert caught.value.code == "corrupt_payload"
    assert caught.value.context["artifact_id"] == neighbors.artifact_id


@pytest.mark.parametrize(
    ("indices", "distances"),
    [
        (np.array([[1, 2], [0, 2], [0, 1], [0, 4]], dtype=np.uint32), None),
        (np.array([[1, 2], [0, 2], [2, 1], [0, 1]], dtype=np.uint32), None),
        (None, np.array([[0.5, 0.5]] * 3 + [[0.5, np.inf]], dtype=np.float32)),
        (None, np.array([[0.5, 0.5]] * 3 + [[0.5, np.nan]], dtype=np.float32)),
        (None, np.array([[0.5, 0.5]] * 3 + [[0.5, -0.5]], dtype=np.float32)),
    ],
    ids=[
        "index_past_last_cell",
        "self_neighbor",
        "infinite_distance",
        "nan_distance",
        "negative_distance",
    ],
)
def test_neighbor_payload_validation_rejects_invalid_indices_or_distances(
    indices, distances
) -> None:
    from scarf.graph.distances import validate_neighbors_payload

    store = _bare_store()
    neighbors = _neighbors_payload(store, indices=indices, distances=distances)

    with pytest.raises(
        ArtifactResolutionError, match="contain invalid indices or distances"
    ) as caught:
        validate_neighbors_payload(store.zw, neighbors)
    assert caught.value.code == "corrupt_payload"
