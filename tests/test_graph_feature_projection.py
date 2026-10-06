import hashlib

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

import scarf.graph.feature_projection as feature_projection_module
from scarf.datastore.graph_datastore import GraphDataStore
from scarf.embeddings.imported import write_imported_coordinates
from scarf.graph.feature_projection import (
    graph_cell_selection,
    graph_source_assays,
    resolve_coordinate_inputs,
    resolve_graph_assay_inputs,
    resolve_graph_source_assay,
    resolve_native_graph_inputs,
)
from scarf.storage.artifacts import (
    ArtifactRef,
    artifact_path,
    fingerprint_array,
    fingerprint_stored_arrays,
    fingerprint_stored_strings,
    make_provenance,
    new_artifact_id,
)
from scarf.storage.errors import ArtifactResolutionError


def _artifact(
    root: zarr.Group,
    kind: str,
    *,
    assay: str | None,
    inputs: dict[str, object] | None = None,
    parameters: dict[str, object] | None = None,
    operation: str | None = None,
) -> ArtifactRef:
    ref = ArtifactRef(
        scope="assay" if assay is not None else "datastore",
        assay=assay,
        kind=kind,
        artifact_id=new_artifact_id(),
    )
    group = root.create_group(artifact_path(ref))
    group.attrs.update(
        {
            "artifact_id": ref.artifact_id,
            "kind": kind,
            "provenance": make_provenance(
                operation=operation or f"test_{kind}",
                parameters=parameters or {},
                inputs=inputs or {},
            ),
            "execution_options": {},
            "complete": True,
        }
    )
    return ref


def _feature_selection(root: zarr.Group, assay: str) -> ArtifactRef:
    feature_data_path = f"{assay}/featureData"
    if feature_data_path not in root:
        feature_data = root.create_group(feature_data_path)
        feature_data.create_array(
            "ids",
            data=np.asarray(["f0", "f1", "f2", "f3"]),
        )
    else:
        feature_data = root[feature_data_path]
    row_fingerprint = fingerprint_stored_strings(feature_data["ids"])
    values = np.ones(4, dtype=bool)
    all_features = _artifact(
        root,
        "feature_selection",
        assay=assay,
        parameters={
            "dataset_fingerprint": "test-dataset",
            "ordered_feature_ids_fingerprint": row_fingerprint,
        },
        operation="create_all_features",
    )
    all_group = root[artifact_path(all_features)]
    all_group.create_array("values", data=values)
    all_group.attrs["ordered_feature_ids_fingerprint"] = row_fingerprint
    all_group.attrs["payload_fingerprint"] = fingerprint_stored_arrays(
        all_group,
        ("values",),
    )
    selection = _artifact(
        root,
        "feature_selection",
        assay=assay,
        inputs={"all_features": all_features},
        parameters={"values_fingerprint": fingerprint_array(values)},
        operation="set_feature_selection",
    )
    selection_group = root[artifact_path(selection)]
    selection_group.create_array("values", data=values)
    selection_group.attrs["ordered_feature_ids_fingerprint"] = row_fingerprint
    selection_group.attrs["payload_fingerprint"] = fingerprint_stored_arrays(
        selection_group,
        ("values",),
    )
    return selection


def _cell_selection(root: zarr.Group) -> ArtifactRef:
    cell_ids = np.asarray(["c0", "c1", "c2"])
    values = np.ones(3, dtype=bool)
    cell_data = root.create_group("cellData")
    cell_data.create_array("ids", data=cell_ids)
    cell_data.create_array("I", data=values)
    selection = _artifact(
        root,
        "cell_selection",
        assay=None,
        inputs={
            "ordered_row_ids_fingerprint": fingerprint_stored_strings(cell_data["ids"]),
            "values_fingerprint": fingerprint_array(values),
        },
    )
    group = root[artifact_path(selection)]
    group.create_array("values", data=values)
    group.attrs["execution_options"] = {"source_column": "I"}
    return selection


def _native_chain(
    root: zarr.Group,
    assay: str,
    *,
    cell_selection: ArtifactRef,
    feature_selection: ArtifactRef | None = None,
    batch_corrected: bool = False,
    imported: bool = False,
) -> tuple[ArtifactRef, ArtifactRef, ArtifactRef]:
    root.require_group(assay).attrs.update(
        {"prepared": True, "dataset_fingerprint": "test-dataset"}
    )
    n_cells = int(root[artifact_path(cell_selection)]["values"][:].sum())
    if imported:
        coordinate_values = np.arange(n_cells * 2, dtype=np.float32).reshape(n_cells, 2)
        coordinates = write_imported_coordinates(
            root,
            assay=assay,
            dimreduc_key="pca",
            role="pca",
            coordinates=coordinate_values,
            source_digest=hashlib.sha256(b"projection-import").digest(),
            payload_fingerprints={"data": fingerprint_array(coordinate_values)},
            source_cell_ids=np.asarray(root["cellData/ids"][:]),
            cell_selection=cell_selection,
            block_rows=2,
        )
    else:
        if feature_selection is None:
            feature_selection = _feature_selection(root, assay)
        normalized = _artifact(
            root,
            "normalized",
            assay=assay,
            inputs={
                "cell_selection": cell_selection,
                "feature_selection": feature_selection,
                "dataset_fingerprint": "test-dataset",
            },
        )
        root[artifact_path(normalized)].create_array(
            "data", data=np.zeros((n_cells, 4), dtype=np.float32)
        )
        reduction = _artifact(
            root,
            "reduction",
            assay=assay,
            inputs={"normalized": normalized},
        )
        root[artifact_path(reduction)].create_array(
            "data", data=np.zeros((n_cells, 2), dtype=np.float64)
        )
        root[artifact_path(reduction)].create_array(
            "loadings", data=np.zeros((4, 2), dtype=np.float64)
        )
        coordinates = (
            _artifact(
                root,
                "batch_correction",
                assay=assay,
                inputs={"reduction": reduction},
            )
            if batch_corrected
            else reduction
        )
    if batch_corrected and not imported:
        root[artifact_path(coordinates)].create_array(
            "data", data=np.zeros((n_cells, 2), dtype=np.float64)
        )
    ann_index = _artifact(
        root,
        "ann_index",
        assay=assay,
        inputs={"coordinates": coordinates},
    )
    neighbors = _artifact(
        root,
        "neighbors",
        assay=assay,
        inputs={"ann_index": ann_index, "coordinates": coordinates},
    )
    connectivity = _artifact(
        root,
        "connectivity_map",
        assay=assay,
        inputs={"neighbors": neighbors},
    )
    return connectivity, neighbors, coordinates


def _bare_embedding_store(root: zarr.Group) -> GraphDataStore:
    store = object.__new__(GraphDataStore)
    store.z = root
    store.workspace = None
    return store


@pytest.fixture
def root() -> zarr.Group:
    return zarr.open_group(store=MemoryStore(), mode="w")


@pytest.mark.parametrize("batch_corrected", [False, True])
def test_native_projection_follows_named_inputs(
    root: zarr.Group,
    batch_corrected: bool,
) -> None:
    cells = _cell_selection(root)
    features = _feature_selection(root, "RNA")
    connectivity, neighbors, coordinates = _native_chain(
        root,
        "RNA",
        cell_selection=cells,
        feature_selection=features,
        batch_corrected=batch_corrected,
    )

    ancestry = resolve_native_graph_inputs(root, connectivity)

    assert ancestry.neighbors == neighbors
    assert ancestry.coordinates == coordinates
    assert graph_cell_selection(root, connectivity) == cells
    assert ancestry.feature_selection == features


def test_imported_projection_has_no_feature_selection(root: zarr.Group) -> None:
    cells = _cell_selection(root)
    connectivity, _neighbors, _coordinates = _native_chain(
        root,
        "RNA",
        cell_selection=cells,
        imported=True,
    )

    assert graph_cell_selection(root, connectivity) == cells
    assert resolve_native_graph_inputs(root, connectivity).feature_selection is None


def test_native_projection_ignores_live_cell_alias_drift(root: zarr.Group) -> None:
    cells = _cell_selection(root)
    connectivity, _neighbors, _coordinates = _native_chain(
        root,
        "RNA",
        cell_selection=cells,
    )
    root["cellData/I"][0] = False

    assert resolve_native_graph_inputs(root, connectivity).cell_selection == cells


def test_native_graph_classifies_missing_incomplete_and_malformed_records(
    root: zarr.Group,
) -> None:
    cells = _cell_selection(root)
    connectivity, _neighbors, _coordinates = _native_chain(
        root,
        "RNA",
        cell_selection=cells,
    )
    missing = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="connectivity_map",
        artifact_id="0" * 64,
    )

    with pytest.raises(ArtifactResolutionError) as missing_error:
        resolve_native_graph_inputs(root, missing)
    assert missing_error.value.code == "missing_artifact"

    root[artifact_path(connectivity)].attrs["complete"] = False
    with pytest.raises(ArtifactResolutionError) as incomplete:
        resolve_native_graph_inputs(root, connectivity)
    assert incomplete.value.code == "incomplete_artifact"

    root[artifact_path(connectivity)].attrs["complete"] = "yes"
    with pytest.raises(ArtifactResolutionError) as malformed:
        resolve_native_graph_inputs(root, connectivity)
    assert malformed.value.code == "corrupt_payload"


def test_neighbors_reject_non_reference_coordinates(
    root: zarr.Group,
) -> None:
    cells = _cell_selection(root)
    connectivity, neighbors, _coordinates = _native_chain(
        root,
        "RNA",
        cell_selection=cells,
    )
    group = root[artifact_path(neighbors)]
    provenance = dict(group.attrs["provenance"])
    inputs = dict(provenance["inputs"])
    inputs["coordinates"] = "RNA/normed__I__hvgs/reduction__pca"
    provenance["inputs"] = inputs
    group.attrs["provenance"] = provenance

    with pytest.raises(ArtifactResolutionError) as caught:
        resolve_native_graph_inputs(root, connectivity)
    assert caught.value.code == "corrupt_payload"


@pytest.mark.parametrize(
    "mutation",
    ["missing", "extra_field", "modern_feature_input"],
)
def test_native_projection_classifies_modern_named_edge_damage_as_corruption(
    root: zarr.Group,
    mutation: str,
) -> None:
    cells = _cell_selection(root)
    connectivity, neighbors, _coordinates = _native_chain(
        root,
        "RNA",
        cell_selection=cells,
    )
    group = root[artifact_path(neighbors)]
    provenance = dict(group.attrs["provenance"])
    inputs = dict(provenance["inputs"])
    if mutation == "missing":
        del inputs["coordinates"]
    else:
        raw_coordinates = dict(inputs["coordinates"])
        if mutation == "extra_field":
            raw_coordinates["unexpected"] = True
        else:
            raw_coordinates["feature_selection"] = _feature_selection(
                root,
                "RNA",
            ).to_dict()
        inputs["coordinates"] = raw_coordinates
    provenance["inputs"] = inputs
    group.attrs["provenance"] = provenance

    with pytest.raises(ArtifactResolutionError) as caught:
        resolve_native_graph_inputs(root, connectivity)

    assert caught.value.code == "corrupt_payload"
    assert caught.value.context["input_name"] == "coordinates"


def test_native_projection_classifies_coordinate_disagreement_as_corruption(
    root: zarr.Group,
) -> None:
    cells = _cell_selection(root)
    connectivity, neighbors, _coordinates = _native_chain(
        root,
        "RNA",
        cell_selection=cells,
    )
    neighbor_inputs = root[artifact_path(neighbors)].attrs["provenance"]["inputs"]
    ann_index = ArtifactRef.from_dict(neighbor_inputs["ann_index"])
    different_coordinates = _artifact(root, "reduction", assay="RNA")
    ann_group = root[artifact_path(ann_index)]
    provenance = dict(ann_group.attrs["provenance"])
    inputs = dict(provenance["inputs"])
    inputs["coordinates"] = different_coordinates.to_dict()
    provenance["inputs"] = inputs
    ann_group.attrs["provenance"] = provenance

    with pytest.raises(ArtifactResolutionError) as caught:
        resolve_native_graph_inputs(root, connectivity)

    assert caught.value.code == "corrupt_payload"
    assert caught.value.context["input_name"] == "coordinates"


def test_native_assay_resolution_rejects_another_assay(root: zarr.Group) -> None:
    cells = _cell_selection(root)
    connectivity, neighbors, _coordinates = _native_chain(
        root,
        "RNA",
        cell_selection=cells,
    )

    assert resolve_graph_assay_inputs(root, connectivity, "RNA").neighbors == neighbors
    with pytest.raises(ArtifactResolutionError) as caught:
        resolve_graph_assay_inputs(root, connectivity, "ADT")
    assert caught.value.code == "wrong_assay"
    assert caught.value.context["expected_assay"] == "ADT"


@pytest.mark.parametrize(
    ("assays", "source_count"),
    [
        ([], 0),
        (["RNA"], 1),
        (["RNA", "RNA"], 2),
    ],
)
def test_integrated_graph_rejects_invalid_assay_cardinality(
    root: zarr.Group,
    assays: list[str],
    source_count: int,
) -> None:
    cells = _cell_selection(root)
    connectivity, _neighbors, _coordinates = _native_chain(
        root,
        "RNA",
        cell_selection=cells,
    )
    integrated = _artifact(
        root,
        "integrated_graph",
        assay=None,
        inputs={
            **{f"source_{index}": connectivity for index in range(source_count)},
            "cell_selection": cells,
        },
        parameters={"method": "snn", "assays": assays},
        operation="integrate_assays",
    )

    with pytest.raises(ArtifactResolutionError) as caught:
        graph_source_assays(root, integrated)
    assert caught.value.code == "corrupt_payload"


@pytest.mark.parametrize("mutation", ["missing_source", "extra_ref_field"])
def test_integrated_graph_rejects_malformed_source_shape(
    root: zarr.Group,
    mutation: str,
) -> None:
    cells = _cell_selection(root)
    rna_connectivity, _rna_neighbors, _rna_coordinates = _native_chain(
        root,
        "RNA",
        cell_selection=cells,
    )
    adt_connectivity, _adt_neighbors, _adt_coordinates = _native_chain(
        root,
        "ADT",
        cell_selection=cells,
    )
    source_0: object = rna_connectivity
    inputs: dict[str, object] = {
        "source_0": source_0,
        "source_1": adt_connectivity,
        "cell_selection": cells,
    }
    if mutation == "missing_source":
        inputs.pop("source_1")
    else:
        source_0 = {**rna_connectivity.to_dict(), "unexpected": "value"}
        inputs["source_0"] = source_0
    integrated = _artifact(
        root,
        "integrated_graph",
        assay=None,
        inputs=inputs,
        parameters={"method": "snn", "assays": ["RNA", "ADT"]},
        operation="integrate_assays",
    )

    with pytest.raises(ArtifactResolutionError) as caught:
        graph_source_assays(root, integrated)
    assert caught.value.code == "corrupt_payload"


def test_integrated_snn_resolves_persisted_assay_branch(root: zarr.Group) -> None:
    cells = _cell_selection(root)
    rna_features = _feature_selection(root, "RNA")
    adt_features = _feature_selection(root, "ADT")
    rna_connectivity, rna_neighbors, _rna_coordinates = _native_chain(
        root,
        "RNA",
        cell_selection=cells,
        feature_selection=rna_features,
    )
    adt_connectivity, adt_neighbors, _adt_coordinates = _native_chain(
        root,
        "ADT",
        cell_selection=cells,
        feature_selection=adt_features,
    )
    integrated = _artifact(
        root,
        "integrated_graph",
        assay=None,
        inputs={
            "source_0": rna_connectivity,
            "source_1": adt_connectivity,
            "cell_selection": cells,
        },
        parameters={"method": "snn", "assays": ["RNA", "ADT"]},
        operation="integrate_assays",
    )

    rna_branch = resolve_graph_assay_inputs(root, integrated, "RNA")
    adt_branch = resolve_graph_assay_inputs(root, integrated, "ADT")

    assert rna_branch.neighbors == rna_neighbors
    assert rna_branch.feature_selection == rna_features
    assert adt_branch.neighbors == adt_neighbors
    assert adt_branch.feature_selection == adt_features
    with pytest.raises(ArtifactResolutionError) as caught:
        resolve_graph_assay_inputs(root, integrated, "ATAC")
    assert caught.value.code == "wrong_assay"
    assert caught.value.context["expected_assay"] == "ATAC"


def test_integrated_snn_rejects_extra_source_ref_fields(
    root: zarr.Group,
) -> None:
    cells = _cell_selection(root)
    connectivity, _neighbors, _coordinates = _native_chain(
        root,
        "RNA",
        cell_selection=cells,
    )
    adt_connectivity, _adt_neighbors, _adt_coordinates = _native_chain(
        root,
        "ADT",
        cell_selection=cells,
    )
    integrated = _artifact(
        root,
        "integrated_graph",
        assay=None,
        inputs={
            "source_0": connectivity,
            "source_1": adt_connectivity,
            "cell_selection": cells,
        },
        parameters={"method": "snn", "assays": ["RNA", "ADT"]},
        operation="integrate_assays",
    )
    group = root[artifact_path(integrated)]
    provenance = dict(group.attrs["provenance"])
    inputs = dict(provenance["inputs"])
    source = dict(inputs["source_0"])
    source["unexpected"] = True
    inputs["source_0"] = source
    provenance["inputs"] = inputs
    group.attrs["provenance"] = provenance

    with pytest.raises(ArtifactResolutionError) as caught:
        resolve_graph_assay_inputs(root, integrated, "RNA")
    assert caught.value.code == "corrupt_payload"


def test_integrated_wnn_projection_validates_coordinate_bundle(
    root: zarr.Group,
) -> None:
    cells = _cell_selection(root)
    features = _feature_selection(root, "RNA")
    _connectivity, neighbors, coordinates = _native_chain(
        root,
        "RNA",
        cell_selection=cells,
        feature_selection=features,
        batch_corrected=True,
    )
    adt_features = _feature_selection(root, "ADT")
    _adt_connectivity, adt_neighbors, adt_coordinates = _native_chain(
        root,
        "ADT",
        cell_selection=cells,
        feature_selection=adt_features,
        batch_corrected=True,
    )
    integrated = _artifact(
        root,
        "integrated_graph",
        assay=None,
        inputs={
            "source_0": {
                "neighbors": neighbors,
                "coordinates": coordinates,
            },
            "source_1": {
                "neighbors": adt_neighbors,
                "coordinates": adt_coordinates,
            },
            "cell_selection": cells,
        },
        parameters={
            "method": "wnn",
            "assays": ["RNA", "ADT"],
            "l2_normalize": True,
        },
        operation="integrate_assays",
    )

    assert graph_cell_selection(root, integrated) == cells
    assert graph_source_assays(root, integrated) == ("RNA", "ADT")
    branch = resolve_graph_assay_inputs(root, integrated, "RNA")
    assert branch.neighbors == neighbors
    assert branch.coordinates == coordinates

    _newer_connectivity, newer_neighbors, _newer_coordinates = _native_chain(
        root,
        "RNA",
        cell_selection=cells,
        feature_selection=features,
        batch_corrected=True,
    )
    assert newer_neighbors != neighbors
    assert resolve_graph_assay_inputs(root, integrated, "RNA").neighbors == neighbors

    integrated_group = root[artifact_path(integrated)]
    original_provenance = dict(integrated_group.attrs["provenance"])
    provenance = dict(original_provenance)
    inputs = dict(provenance["inputs"])
    source = dict(inputs["source_0"])
    raw_coordinates = dict(source["coordinates"])
    raw_coordinates["unexpected"] = True
    source["coordinates"] = raw_coordinates
    inputs["source_0"] = source
    provenance["inputs"] = inputs
    integrated_group.attrs["provenance"] = provenance
    with pytest.raises(ArtifactResolutionError) as malformed_coordinates:
        resolve_graph_assay_inputs(root, integrated, "RNA")
    assert malformed_coordinates.value.code == "corrupt_payload"

    integrated_group.attrs["provenance"] = original_provenance
    provenance = dict(original_provenance)
    inputs = dict(provenance["inputs"])
    source = dict(inputs["source_0"])
    raw_neighbors = dict(source["neighbors"])
    raw_neighbors["unexpected"] = True
    source["neighbors"] = raw_neighbors
    inputs["source_0"] = source
    provenance["inputs"] = inputs
    integrated_group.attrs["provenance"] = provenance
    with pytest.raises(ArtifactResolutionError) as malformed:
        resolve_graph_assay_inputs(root, integrated, "RNA")
    assert malformed.value.code == "corrupt_payload"

    wrong_coordinates = _artifact(root, "reduction", assay="RNA")
    broken = _artifact(
        root,
        "integrated_graph",
        assay=None,
        inputs={
            "source_0": {
                "neighbors": neighbors,
                "coordinates": wrong_coordinates,
            },
            "source_1": {
                "neighbors": adt_neighbors,
                "coordinates": adt_coordinates,
            },
            "cell_selection": cells,
        },
        parameters={
            "method": "wnn",
            "assays": ["RNA", "ADT"],
            "l2_normalize": True,
        },
        operation="integrate_assays",
    )
    with pytest.raises(ArtifactResolutionError) as caught:
        graph_source_assays(root, broken)
    assert caught.value.code == "corrupt_payload"


def test_integrated_wnn_rejects_imported_coordinates(root: zarr.Group) -> None:
    cells = _cell_selection(root)
    _connectivity, neighbors, coordinates = _native_chain(
        root,
        "RNA",
        cell_selection=cells,
        imported=True,
    )
    _adt_connectivity, adt_neighbors, adt_coordinates = _native_chain(
        root,
        "ADT",
        cell_selection=cells,
    )
    integrated = _artifact(
        root,
        "integrated_graph",
        assay=None,
        inputs={
            "source_0": {
                "neighbors": neighbors,
                "coordinates": coordinates,
            },
            "source_1": {
                "neighbors": adt_neighbors,
                "coordinates": adt_coordinates,
            },
            "cell_selection": cells,
        },
        parameters={
            "method": "wnn",
            "assays": ["RNA", "ADT"],
            "l2_normalize": True,
        },
        operation="integrate_assays",
    )

    with pytest.raises(ArtifactResolutionError) as caught:
        graph_source_assays(root, integrated)
    assert caught.value.code == "wrong_kind"


def test_ini_embed_requires_initialization_from_the_graph_reduction(
    root: zarr.Group,
) -> None:
    cells = _cell_selection(root)
    features = _feature_selection(root, "RNA")
    _graph, _neighbors, coordinates = _native_chain(
        root,
        "RNA",
        cell_selection=cells,
        feature_selection=features,
    )
    other_graph, _other_neighbors, other_coordinates = _native_chain(
        root,
        "RNA",
        cell_selection=cells,
        feature_selection=features,
    )
    store = _bare_embedding_store(root)
    initialization = _artifact(
        root,
        "embedding_initialization",
        assay="RNA",
        inputs={"coordinates": coordinates},
        operation="build_embedding_initialization",
    )
    with pytest.raises(
        ValueError,
        match="does not belong to the graph coordinates",
    ):
        store._get_ini_embed(initialization, other_graph, 2)
    assert other_coordinates != coordinates


@pytest.mark.parametrize(
    ("damage", "code"),
    [
        ("incomplete", "incomplete_artifact"),
        ("row_ids", "row_identity_mismatch"),
        ("selection", "selection_values_changed"),
    ],
)
def test_native_resolution_reuses_records_only_within_one_call(
    root, monkeypatch, damage, code
):
    from collections import Counter

    cells = _cell_selection(root)
    graph, neighbors, coordinates = _native_chain(root, "RNA", cell_selection=cells)
    inspections = Counter()
    original = feature_projection_module.inspect_artifact

    def inspect(group, ref):
        inspections[ref] += 1
        return original(group, ref)

    monkeypatch.setattr(feature_projection_module, "inspect_artifact", inspect)
    result = resolve_native_graph_inputs(root, graph)
    assert result.cell_selection == cells
    assert inspections[neighbors] == inspections[coordinates] == 1
    if damage == "incomplete":
        root[artifact_path(neighbors)].attrs["complete"] = False
    elif damage == "row_ids":
        root["cellData/ids"][:] = np.array(["c2", "c1", "c0"])
    else:
        root[artifact_path(cells)]["values"][0] = False
    # The second call reads the damaged records again instead of reusing them.
    with pytest.raises(ArtifactResolutionError) as caught:
        resolve_native_graph_inputs(root, graph)
    assert caught.value.code == code
    damaged = neighbors if damage == "incomplete" else cells
    assert caught.value.context["artifact_id"] == damaged.artifact_id


def _replace_input(root: zarr.Group, ref: ArtifactRef, name: str, value) -> None:
    """Rewrite one named provenance input of a stored artifact."""
    group = root[artifact_path(ref)]
    provenance = dict(group.attrs["provenance"])
    inputs = dict(provenance["inputs"])
    inputs[name] = value.to_dict() if isinstance(value, ArtifactRef) else value
    provenance["inputs"] = inputs
    group.attrs["provenance"] = provenance


def _input(root: zarr.Group, ref: ArtifactRef, name: str) -> ArtifactRef:
    raw = root[artifact_path(ref)].attrs["provenance"]["inputs"][name]
    return ArtifactRef.from_dict(raw)


def _two_assay_snn(root: zarr.Group, **graph_options) -> dict[str, ArtifactRef]:
    cells = _cell_selection(root)
    rna, rna_neighbors, rna_coordinates = _native_chain(
        root, "RNA", cell_selection=cells
    )
    adt, adt_neighbors, _adt_coordinates = _native_chain(
        root, "ADT", cell_selection=cells
    )
    options = {
        "inputs": {"source_0": rna, "source_1": adt, "cell_selection": cells},
        "parameters": {"method": "snn", "assays": ["RNA", "ADT"]},
        "operation": "integrate_assays",
    } | graph_options
    return {
        "cells": cells,
        "rna": rna,
        "rna_neighbors": rna_neighbors,
        "rna_coordinates": rna_coordinates,
        "adt": adt,
        "adt_neighbors": adt_neighbors,
        "integrated": _artifact(root, "integrated_graph", assay=None, **options),
    }


@pytest.mark.parametrize(
    ("replacement", "code", "context"),
    [
        ("rna_coordinates", "wrong_kind", {"expected_kind": "neighbors"}),
        ("adt_neighbors", "wrong_assay", {"expected_assay": "RNA"}),
    ],
)
def test_native_projection_rejects_an_input_of_another_kind_or_assay(
    root: zarr.Group, replacement: str, code: str, context: dict[str, str]
) -> None:
    refs = _two_assay_snn(root)
    _replace_input(root, refs["rna"], "neighbors", refs[replacement])

    with pytest.raises(ArtifactResolutionError) as caught:
        resolve_native_graph_inputs(root, refs["rna"])

    assert caught.value.code == code
    assert caught.value.context["artifact_id"] == refs[replacement].artifact_id
    assert context.items() <= caught.value.context.items()


def test_native_projection_rejects_unsupported_neighbor_coordinates(
    root: zarr.Group,
) -> None:
    cells = _cell_selection(root)
    connectivity, neighbors, coordinates = _native_chain(
        root, "RNA", cell_selection=cells
    )
    features = _input(
        root, _input(root, coordinates, "normalized"), "feature_selection"
    )
    _replace_input(root, neighbors, "coordinates", features)

    with pytest.raises(
        ArtifactResolutionError, match="unsupported artifact kind"
    ) as caught:
        resolve_native_graph_inputs(root, connectivity)

    assert caught.value.code == "unsupported_graph_kind"
    assert caught.value.context["actual_kind"] == "feature_selection"


def test_native_projection_follows_neighbors_on_normalized_values(
    root: zarr.Group,
) -> None:
    cells = _cell_selection(root)
    features = _feature_selection(root, "RNA")
    connectivity, neighbors, coordinates = _native_chain(
        root, "RNA", cell_selection=cells, feature_selection=features
    )
    normalized = _input(root, coordinates, "normalized")
    ann_index = _input(root, neighbors, "ann_index")
    _replace_input(root, ann_index, "coordinates", normalized)
    _replace_input(root, neighbors, "coordinates", normalized)

    ancestry = resolve_native_graph_inputs(root, connectivity)

    # A graph on normalized values has no reduction.
    assert ancestry.coordinates == ancestry.normalized == normalized
    assert ancestry.reduction is None
    assert ancestry.cell_selection == cells
    assert ancestry.feature_selection == features


def test_coordinate_resolution_requires_assay_scoped_supported_coordinates(
    root: zarr.Group,
) -> None:
    cells = _cell_selection(root)
    _connectivity, _neighbors, coordinates = _native_chain(
        root, "RNA", cell_selection=cells
    )
    unscoped = ArtifactRef(
        scope="datastore", kind="reduction", artifact_id=new_artifact_id()
    )

    with pytest.raises(ArtifactResolutionError, match="has no assay") as caught:
        resolve_coordinate_inputs(root, unscoped)
    assert caught.value.code == "wrong_scope"

    # Normalized values are coordinates without a reduction.
    normalized = _input(root, coordinates, "normalized")
    lineage = resolve_coordinate_inputs(root, normalized)
    assert (lineage.coordinates, lineage.reduction, lineage.normalized) == (
        normalized,
        None,
        normalized,
    )
    assert lineage.cell_selection == cells

    features = _input(root, normalized, "feature_selection")
    with pytest.raises(ArtifactResolutionError, match="Coordinates must be") as caught:
        resolve_coordinate_inputs(root, features)
    assert caught.value.code == "unsupported_graph_kind"

    root[artifact_path(normalized)].attrs["complete"] = False
    with pytest.raises(ArtifactResolutionError) as caught:
        resolve_coordinate_inputs(root, normalized)
    assert caught.value.code == "incomplete_artifact"


@pytest.mark.parametrize(
    ("array", "values", "code"),
    [
        ("loadings", None, "payload_missing"),
        ("loadings", np.zeros((3, 2)), "column_mismatch"),
        ("loadings", np.zeros(4), "column_mismatch"),
        ("data", np.zeros((2, 2)), "row_mismatch"),
        ("data", np.zeros((3, 3)), "row_mismatch"),
        ("data", np.zeros((3, 2), dtype=np.int32), "row_mismatch"),
    ],
    ids=[
        "missing_loadings",
        "loadings_rows",
        "one_dimensional_loadings",
        "data_rows",
        "data_columns",
        "integer_data",
    ],
)
def test_coordinate_resolution_rejects_reduction_payload_damage(
    root: zarr.Group, array: str, values: np.ndarray | None, code: str
) -> None:
    cells = _cell_selection(root)
    _connectivity, _neighbors, coordinates = _native_chain(
        root, "RNA", cell_selection=cells
    )
    group = root[artifact_path(coordinates)]
    assert resolve_coordinate_inputs(root, coordinates).cell_selection == cells
    if values is None:
        del group[array]
    else:
        group.create_array(array, data=values, overwrite=True)

    with pytest.raises(ArtifactResolutionError) as caught:
        resolve_coordinate_inputs(root, coordinates)

    assert caught.value.code == code
    assert caught.value.context["artifact_id"] == coordinates.artifact_id


@pytest.mark.parametrize(
    "graph_options",
    [
        {"operation": "merge_graphs"},
        {"parameters": {"method": "knn", "assays": ["RNA", "ADT"]}},
        {"parameters": {"method": "snn", "assays": "RNA,ADT"}},
        {"parameters": {"method": "snn", "assays": ["RNA", "ADT"], "k": 3}},
        {"parameters": {"method": "wnn", "assays": ["RNA", "ADT"]}},
        {
            "parameters": {
                "method": "wnn",
                "assays": ["RNA", "ADT"],
                "l2_normalize": "yes",
            }
        },
    ],
    ids=[
        "other_operation",
        "unknown_method",
        "assays_not_a_list",
        "extra_parameter",
        "wnn_without_l2_normalize",
        "wnn_non_boolean_l2_normalize",
    ],
)
def test_integrated_graph_rejects_invalid_source_parameters(
    root: zarr.Group, graph_options: dict
) -> None:
    integrated = _two_assay_snn(root, **graph_options)["integrated"]

    with pytest.raises(
        ArtifactResolutionError, match="invalid source parameters"
    ) as caught:
        graph_source_assays(root, integrated)

    assert caught.value.code == "corrupt_payload"


def test_integrated_wnn_requires_a_neighbor_coordinate_bundle(
    root: zarr.Group,
) -> None:
    refs = _two_assay_snn(
        root,
        parameters={"method": "wnn", "assays": ["RNA", "ADT"], "l2_normalize": True},
    )

    # An SNN-style connectivity reference is not a WNN source bundle.
    with pytest.raises(ArtifactResolutionError, match="no source bundle") as caught:
        graph_source_assays(root, refs["integrated"])

    assert caught.value.code == "corrupt_payload"
    assert caught.value.context["input_name"] == "source_0"


def test_integrated_graph_sources_must_share_its_cell_selection(
    root: zarr.Group,
) -> None:
    refs = _two_assay_snn(root)
    # The same cells under another selection record are a different input.
    cells = refs["cells"]
    other_cells = _artifact(
        root,
        "cell_selection",
        assay=None,
        inputs=root[artifact_path(cells)].attrs["provenance"]["inputs"],
    )
    other_group = root[artifact_path(other_cells)]
    other_group.create_array("values", data=np.ones(3, dtype=bool))
    other_group.attrs["execution_options"] = {"source_column": "I"}
    _replace_input(root, refs["integrated"], "cell_selection", other_cells)

    with pytest.raises(
        ArtifactResolutionError, match="shared cell selection"
    ) as caught:
        graph_cell_selection(root, refs["integrated"])

    assert caught.value.code == "corrupt_payload"
    assert caught.value.context["input_name"] == "cell_selection"


def test_graph_resolvers_reject_non_graph_artifacts(root: zarr.Group) -> None:
    cells = _cell_selection(root)
    _connectivity, _neighbors, coordinates = _native_chain(
        root, "RNA", cell_selection=cells
    )

    for resolve in (
        lambda: graph_cell_selection(root, coordinates),
        lambda: graph_source_assays(root, coordinates),
        lambda: resolve_graph_assay_inputs(root, coordinates, "RNA"),
    ):
        with pytest.raises(
            ArtifactResolutionError, match="connectivity_map, neighbors, or integrated"
        ) as caught:
            resolve()
        assert caught.value.code == "unsupported_graph_kind"


def test_graph_source_assay_resolution_checks_the_requested_assay(
    root: zarr.Group,
) -> None:
    refs = _two_assay_snn(root)

    assert resolve_graph_source_assay(root, refs["rna"], None, parameter_name="x") == (
        "RNA"
    )
    assert resolve_graph_source_assay(root, refs["rna"], "RNA", parameter_name="x") == (
        "RNA"
    )
    with pytest.raises(ArtifactResolutionError, match="x does not match") as native:
        resolve_graph_source_assay(root, refs["rna"], "ADT", parameter_name="x")
    assert native.value.code == "wrong_assay"
    assert native.value.context["expected_assay"] == "RNA"

    integrated = refs["integrated"]
    assert (
        resolve_graph_source_assay(root, integrated, "ADT", parameter_name="x") == "ADT"
    )
    with pytest.raises(ValueError, match="x is required for an integrated graph"):
        resolve_graph_source_assay(root, integrated, None, parameter_name="x")
    with pytest.raises(
        ArtifactResolutionError, match="no source assay 'ATAC'"
    ) as other:
        resolve_graph_source_assay(root, integrated, "ATAC", parameter_name="x")
    assert other.value.code == "wrong_assay"
    assert other.value.context["expected_assay"] == "RNA,ADT"


def _second_cell_selection(root: zarr.Group, values: np.ndarray) -> ArtifactRef:
    """Store another cell selection over the rows ``_cell_selection`` created."""
    selection = _artifact(
        root,
        "cell_selection",
        assay=None,
        inputs={
            "ordered_row_ids_fingerprint": fingerprint_stored_strings(
                root["cellData/ids"]
            ),
            "values_fingerprint": fingerprint_array(values),
        },
    )
    group = root[artifact_path(selection)]
    group.create_array("values", data=values)
    group.attrs["execution_options"] = {"source_column": "I"}
    return selection


@pytest.mark.parametrize(
    "damage",
    [
        "wrong_kind",
        "datastore_scope",
        "incomplete",
        "operation",
        "no_coordinates",
        "other_assay",
        "other_cells",
    ],
)
def test_ini_embed_rejects_an_initialization_from_other_inputs(
    root: zarr.Group, damage: str
) -> None:
    cells = _cell_selection(root)
    graph, _neighbors, coordinates = _native_chain(root, "RNA", cell_selection=cells)
    inputs: dict[str, object] = {"coordinates": coordinates}
    assay: str | None = "RNA"
    operation = "build_embedding_initialization"
    if damage == "datastore_scope":
        assay = None
    elif damage == "operation":
        operation = "run_kmeans"
    elif damage == "no_coordinates":
        inputs = {}
    elif damage == "other_assay":
        _adt_graph, _adt_neighbors, adt_coordinates = _native_chain(
            root, "ADT", cell_selection=cells
        )
        inputs = {"coordinates": adt_coordinates}
    elif damage == "other_cells":
        subset = _second_cell_selection(root, np.array([True, True, False]))
        _subset_graph, _subset_neighbors, subset_coordinates = _native_chain(
            root, "RNA", cell_selection=subset
        )
        inputs = {"coordinates": subset_coordinates}
    initialization = _artifact(
        root,
        "embedding_initialization",
        assay=assay,
        inputs=inputs,
        operation=operation,
    )
    if damage == "incomplete":
        root[artifact_path(initialization)].attrs["complete"] = False
    if damage == "wrong_kind":
        initialization = coordinates
    message = {
        "wrong_kind": "must be an embedding_initialization ref",
        "datastore_scope": "must be an assay-scoped artifact",
        "incomplete": "unavailable or incomplete",
        "operation": "has an invalid operation",
        "no_coordinates": "has no coordinate source",
        "other_assay": "source belongs to another assay",
        "other_cells": "does not match the graph cell selection",
    }[damage]

    with pytest.raises(ValueError, match=message):
        _bare_embedding_store(root)._get_ini_embed(initialization, graph, 2)
