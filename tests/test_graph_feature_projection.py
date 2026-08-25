import hashlib
from dataclasses import replace

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

import scarf.graph.feature_projection as feature_projection_module
from scarf.datastore.graph_datastore import GraphDataStore
from scarf.embeddings.imported import write_imported_coordinates
from scarf.graph.errors import IncompatibleAnalysisStateError
from scarf.graph.feature_projection import (
    graph_cell_selection,
    graph_source_assays,
    project_normalized_feature_selections,
    resolve_graph_assay_inputs,
    resolve_native_graph_inputs,
)
from scarf.graph.state import AssayState, write_assay_state
from scarf.storage.artifacts import (
    ArtifactRef,
    artifact_path,
    fingerprint_array,
    fingerprint_stored_arrays,
    fingerprint_stored_strings,
    inspect_artifact,
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
    if imported:
        coordinate_values = np.arange(6, dtype=np.float32).reshape(3, 2)
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
            cell_key="I",
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
            },
        )
        reduction = _artifact(
            root,
            "reduction",
            assay=assay,
            inputs={"normalized": normalized},
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
    assert project_normalized_feature_selections(root, connectivity) == (features,)


def test_imported_projection_has_no_feature_selection(root: zarr.Group) -> None:
    cells = _cell_selection(root)
    connectivity, _neighbors, _coordinates = _native_chain(
        root,
        "RNA",
        cell_selection=cells,
        imported=True,
    )

    assert graph_cell_selection(root, connectivity) == cells
    assert project_normalized_feature_selections(root, connectivity) == ()


def test_native_projection_validates_live_cell_selection(root: zarr.Group) -> None:
    cells = _cell_selection(root)
    connectivity, _neighbors, _coordinates = _native_chain(
        root,
        "RNA",
        cell_selection=cells,
    )
    root["cellData/I"][0] = False

    with pytest.raises(ArtifactResolutionError) as caught:
        resolve_native_graph_inputs(root, connectivity)

    assert caught.value.code == "selection_values_changed"


def test_native_projection_rejects_scalar_embedded_in_coordinate_ref(
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
    coordinates = dict(inputs["coordinates"])
    coordinates["feat_key"] = "I__hvgs"
    inputs["coordinates"] = coordinates
    provenance["inputs"] = inputs
    group.attrs["provenance"] = provenance

    with pytest.raises(IncompatibleAnalysisStateError) as caught:
        resolve_native_graph_inputs(root, connectivity)
    assert caught.value.code == "legacy_feature_contract"


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


def test_neighbors_string_coordinates_use_the_legacy_contract(
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

    with pytest.raises(IncompatibleAnalysisStateError) as caught:
        resolve_native_graph_inputs(root, connectivity)
    assert caught.value.code == "legacy_feature_contract"

    inputs["feat_key"] = "I__hvgs"
    del inputs["coordinates"]
    provenance["inputs"] = inputs
    group.attrs["provenance"] = provenance
    with pytest.raises(IncompatibleAnalysisStateError) as missing_with_feat_key:
        resolve_native_graph_inputs(root, connectivity)
    assert missing_with_feat_key.value.code == "legacy_feature_contract"


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


def test_integrated_snn_with_imported_branches_projects_no_features(
    root: zarr.Group,
) -> None:
    cells = _cell_selection(root)
    rna_connectivity, _rna_neighbors, _rna_coordinates = _native_chain(
        root,
        "RNA",
        cell_selection=cells,
        imported=True,
    )
    adt_connectivity, _adt_neighbors, _adt_coordinates = _native_chain(
        root,
        "ADT",
        cell_selection=cells,
        imported=True,
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

    assert project_normalized_feature_selections(root, integrated) == ()


def test_integrated_snn_projects_only_native_branches_in_persisted_order(
    root: zarr.Group,
) -> None:
    cells = _cell_selection(root)
    rna_features = _feature_selection(root, "RNA")
    rna_connectivity, _rna_neighbors, _rna_coordinates = _native_chain(
        root,
        "RNA",
        cell_selection=cells,
        feature_selection=rna_features,
    )
    adt_features = _feature_selection(root, "ADT")
    adt_connectivity, _adt_neighbors, _adt_coordinates = _native_chain(
        root,
        "ADT",
        cell_selection=cells,
        feature_selection=adt_features,
    )
    atac_connectivity, _atac_neighbors, _atac_coordinates = _native_chain(
        root,
        "ATAC",
        cell_selection=cells,
        imported=True,
    )
    integrated = _artifact(
        root,
        "integrated_graph",
        assay=None,
        inputs={
            "source_0": atac_connectivity,
            "source_1": adt_connectivity,
            "source_2": rna_connectivity,
            "cell_selection": cells,
        },
        parameters={"method": "snn", "assays": ["ATAC", "ADT", "RNA"]},
        operation="integrate_assays",
    )

    assert project_normalized_feature_selections(root, integrated) == (
        adt_features,
        rna_features,
    )


def test_integrated_projection_deduplicates_exact_refs_in_first_seen_order(
    root: zarr.Group,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cells = _cell_selection(root)
    rna_features = _feature_selection(root, "RNA")
    adt_features = _feature_selection(root, "ADT")
    sources: list[ArtifactRef] = []
    ancestry_by_source = {}
    for assay in ("RNA", "ADT", "ATAC"):
        features = (
            rna_features
            if assay == "RNA"
            else adt_features
            if assay == "ADT"
            else _feature_selection(root, assay)
        )
        connectivity, _neighbors, _coordinates = _native_chain(
            root,
            assay,
            cell_selection=cells,
            feature_selection=features,
        )
        sources.append(connectivity)
        ancestry_by_source[connectivity] = resolve_native_graph_inputs(
            root,
            connectivity,
        )
    integrated = _artifact(
        root,
        "integrated_graph",
        assay=None,
        inputs={
            **{f"source_{index}": source for index, source in enumerate(sources)},
            "cell_selection": cells,
        },
        parameters={"method": "snn", "assays": ["RNA", "ADT", "ATAC"]},
        operation="integrate_assays",
    )
    projected_by_source = {
        sources[0]: rna_features,
        sources[1]: adt_features,
        sources[2]: rna_features,
    }

    def resolve_with_duplicate(source_root, source):
        assert source_root is root
        return replace(
            ancestry_by_source[source],
            feature_selection=projected_by_source[source],
        )

    monkeypatch.setattr(
        feature_projection_module,
        "resolve_native_graph_inputs",
        resolve_with_duplicate,
    )

    assert project_normalized_feature_selections(root, integrated) == (
        rna_features,
        adt_features,
    )


def test_integrated_snn_rejects_scalar_embedded_in_source_ref(
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
    source["feat_key"] = "I__hvgs"
    inputs["source_0"] = source
    provenance["inputs"] = inputs
    group.attrs["provenance"] = provenance

    with pytest.raises(IncompatibleAnalysisStateError) as caught:
        resolve_graph_assay_inputs(root, integrated, "RNA")
    assert caught.value.code == "legacy_feature_contract"


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
    assert project_normalized_feature_selections(root, integrated) == (
        features,
        adt_features,
    )
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
    raw_coordinates["feat_key"] = "I__hvgs"
    source["coordinates"] = raw_coordinates
    inputs["source_0"] = source
    provenance["inputs"] = inputs
    integrated_group.attrs["provenance"] = provenance
    with pytest.raises(IncompatibleAnalysisStateError) as malformed_coordinates:
        resolve_graph_assay_inputs(root, integrated, "RNA")
    assert malformed_coordinates.value.code == "legacy_feature_contract"

    integrated_group.attrs["provenance"] = original_provenance
    provenance = dict(original_provenance)
    inputs = dict(provenance["inputs"])
    source = dict(inputs["source_0"])
    raw_neighbors = dict(source["neighbors"])
    raw_neighbors["feat_key"] = "I__hvgs"
    source["neighbors"] = raw_neighbors
    inputs["source_0"] = source
    provenance["inputs"] = inputs
    integrated_group.attrs["provenance"] = provenance
    with pytest.raises(IncompatibleAnalysisStateError) as malformed:
        resolve_graph_assay_inputs(root, integrated, "RNA")
    assert malformed.value.code == "legacy_feature_contract"

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
        project_normalized_feature_selections(root, broken)
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
        project_normalized_feature_selections(root, integrated)
    assert caught.value.code == "wrong_kind"


def test_ini_embed_requires_state_initialization_from_the_graph_reduction(
    root: zarr.Group,
) -> None:
    cells = _cell_selection(root)
    features = _feature_selection(root, "RNA")
    graph, _neighbors, coordinates = _native_chain(
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
    with pytest.raises(KeyError, match="no embedding initialization"):
        store._get_ini_embed("RNA", "I", graph, 2)

    raw_normalized = (inspect_artifact(root, coordinates).inputs or {}).get(
        "normalized"
    )
    assert isinstance(raw_normalized, dict)
    initialization = _artifact(
        root,
        "embedding_initialization",
        assay="RNA",
        inputs={"reduction": coordinates},
    )
    write_assay_state(
        root,
        AssayState(
            assay="RNA",
            cell_key="I",
            normalized=ArtifactRef.from_dict(raw_normalized),
            reduction=coordinates,
            embedding_initialization=initialization,
        ),
    )
    with pytest.raises(
        ValueError,
        match="does not belong to the graph reduction",
    ):
        store._get_ini_embed("RNA", "I", other_graph, 2)
    assert other_coordinates != coordinates
