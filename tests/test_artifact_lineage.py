import networkx as nx
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.datastore.base_datastore import BaseDataStore
from scarf.storage.lineage import ArtifactLineage
from scarf.mapping.reference import MappingReference
from scarf.storage.artifacts import (
    ArtifactRef,
    ArtifactScope,
    ExternalArtifactRef,
    artifact_path,
    make_provenance,
)


def _ref(
    kind: str,
    token: str,
    *,
    scope: ArtifactScope = "assay",
) -> ArtifactRef:
    return ArtifactRef(
        scope=scope,
        assay="RNA" if scope == "assay" else None,
        kind=kind,
        artifact_id=token * 64,
    )


def _write_artifact(
    root: zarr.Group,
    ref: ArtifactRef,
    *,
    operation: str,
    inputs: dict,
    parameters: dict | None = None,
    execution_options: dict | None = None,
    complete: bool = True,
) -> None:
    group = root.create_group(artifact_path(ref))
    group.attrs.update(
        {
            "artifact_id": ref.artifact_id,
            "kind": ref.kind,
            "provenance": make_provenance(
                operation=operation,
                parameters=parameters or {},
                inputs=inputs,
            ),
            "execution_options": execution_options or {},
            "complete": complete,
        }
    )


class _ReferenceDatastore:
    def __init__(self, root: zarr.Group) -> None:
        self.zw = root

    def _get_assay(self, assay_name: str) -> zarr.Group:
        return self.zw[assay_name]


def _mapping_reference(
    root: zarr.Group,
    ref: ArtifactRef,
    fingerprint: str,
) -> MappingReference:
    assert ref.assay is not None
    if ref.assay not in root:
        root.create_group(ref.assay)
    root[ref.assay].attrs["dataset_fingerprint"] = fingerprint
    reference = object.__new__(MappingReference)
    object.__setattr__(reference, "datastore", _ReferenceDatastore(root))
    object.__setattr__(reference, "ref", ref)
    object.__setattr__(reference, "assay_name", ref.assay)
    object.__setattr__(reference, "dataset_fingerprint", fingerprint)
    return reference


def test_lineage_merges_nested_inputs_and_named_outputs() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    selection = _ref("cell_selection", "1", scope="datastore")
    normalized = _ref("normalized", "2")
    reduction = _ref("reduction", "3")
    embedding = _ref("embedding", "a")
    _write_artifact(
        root,
        selection,
        operation="manual_selection",
        inputs={},
    )
    _write_artifact(
        root,
        normalized,
        operation="run_normalization",
        inputs={"selection": {"cells": selection}},
    )
    _write_artifact(
        root,
        reduction,
        operation="run_pca",
        inputs={
            "normalized": normalized,
            "copies": [normalized],
            "source_fingerprint": {"value_fingerprint": "f" * 64},
        },
        parameters={"dims": 25},
        execution_options={"local_cache": False},
    )
    _write_artifact(
        root,
        embedding,
        operation="run_umap",
        inputs={"coordinates": reduction},
    )

    lineage = ArtifactLineage.from_store(
        root,
        {
            "pca": reduction,
            "selected_reduction": reduction,
            "umap": embedding,
        },
    )
    reversed_outputs = ArtifactLineage.from_store(
        root,
        {
            "umap": embedding,
            "selected_reduction": reduction,
            "pca": reduction,
        },
    )

    assert nx.is_frozen(lineage.graph)
    assert set(lineage.graph) == {selection, normalized, reduction, embedding}
    assert lineage.graph.edges[selection, normalized]["inputs"] == ("selection.cells",)
    assert lineage.graph.edges[normalized, reduction]["inputs"] == (
        "copies[0]",
        "normalized",
    )
    assert lineage.graph.nodes[reduction]["outputs"] == (
        "pca",
        "selected_reduction",
    )
    assert lineage.graph.nodes[embedding]["outputs"] == ("umap",)
    assert lineage.to_mermaid() == reversed_outputs.to_mermaid()
    assert lineage.to_mermaid() == "\n".join(
        [
            "flowchart LR",
            (
                '    artifact0["datastore / cell_selection | manual_selection | '
                '111111111111"]'
            ),
            ('    artifact1["RNA / normalized | run_normalization | 222222222222"]'),
            (
                '    artifact2["RNA / reduction | run_pca | 333333333333 | '
                'outputs: pca, selected_reduction"]'
            ),
            (
                '    artifact3["RNA / embedding | run_umap | aaaaaaaaaaaa | '
                'outputs: umap"]'
            ),
            '    artifact0 -->|"selection.cells"| artifact1',
            '    artifact1 -->|"copies[0], normalized"| artifact2',
            '    artifact2 -->|"coordinates"| artifact3',
        ]
    )

    markdown = lineage.to_markdown()
    assert markdown.startswith("```mermaid\nflowchart LR\n")
    assert "- Parameters: `dims=25`" in markdown
    assert "- Execution options: `local_cache=false`" in markdown
    assert "source_fingerprint=" in markdown
    assert lineage._repr_markdown_() == markdown


def test_datastore_lineage_builds_a_bounded_markdown_report() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    ref = _ref("quality_metric", "4", scope="datastore")
    _write_artifact(
        root,
        ref,
        operation="calculate_qc",
        inputs={"fingerprint": "a" * 500},
        parameters={"payload": "b" * 500},
    )
    datastore = BaseDataStore.__new__(BaseDataStore)
    datastore.z = root
    datastore.workspace = None

    lineage = datastore.lineage(ref)
    markdown = lineage.to_markdown()

    assert isinstance(lineage, ArtifactLineage)
    assert lineage.outputs == {"output": ref}
    assert "a" * 200 not in markdown
    assert "b" * 200 not in markdown
    assert "..." in markdown


def test_lineage_renders_missing_and_incomplete_artifacts() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    missing = _ref("normalized", "5")
    consumer = _ref("reduction", "6")
    unfinished = _ref("ann_index", "7")
    _write_artifact(
        root,
        consumer,
        operation="run_pca",
        inputs={"normalized": missing},
    )
    _write_artifact(
        root,
        unfinished,
        operation="build_ann_index",
        inputs={"coordinates": consumer},
        complete=False,
    )

    lineage = ArtifactLineage.from_store(root, unfinished)
    mermaid = lineage.to_mermaid()

    assert len(lineage.graph) == 3
    assert "status: missing" in mermaid
    assert "status: incomplete" in mermaid
    assert lineage.graph.nodes[missing]["status"].exists is False
    assert lineage.graph.nodes[unfinished]["status"].complete is False


def test_lineage_rejects_cyclic_provenance() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    first = _ref("normalized", "8")
    second = _ref("reduction", "9")
    _write_artifact(
        root,
        first,
        operation="run_normalization",
        inputs={"reduction": second},
    )
    _write_artifact(
        root,
        second,
        operation="run_pca",
        inputs={"normalized": first},
    )

    with pytest.raises(ValueError, match="dependency cycle"):
        ArtifactLineage.from_store(root, second)


def test_lineage_renders_one_unresolved_external_node_without_opening_paths(
    monkeypatch,
) -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    mapping_ref = _ref("mapping_reference", "a")
    external = ExternalArtifactRef("reference-dataset", mapping_ref)
    projection = _ref("projection", "b")
    _write_artifact(
        root,
        projection,
        operation="map_query",
        inputs={
            "mapping_reference": external,
            "query_fingerprint": "q" * 64,
        },
    )

    def reject_open(*args, **kwargs):
        raise AssertionError("Lineage must not discover or open an external path")

    monkeypatch.setattr(zarr, "open_group", reject_open)
    lineage = ArtifactLineage.from_store(root, projection)
    markdown = lineage.to_markdown()

    assert set(lineage.graph) == {projection, external}
    assert lineage.graph.nodes[external]["status"] is None
    assert lineage.graph.edges[external, projection]["inputs"] == ("mapping_reference",)
    assert "unresolved external" in markdown
    assert "reference-dataset" in markdown
    assert "status: missing" not in markdown
    assert "mapping_reference=" not in markdown
    assert "query_fingerprint=" in markdown


def test_lineage_rejects_malformed_external_artifact_inputs() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    projection = _ref("projection", "0")
    _write_artifact(
        root,
        projection,
        operation="map_query",
        inputs={
            "mapping_reference": {
                "type": "external_artifact",
                "dataset_fingerprint": "reference-dataset",
                "path": "/tmp/reference.zarr",
            }
        },
    )

    with pytest.raises(
        ValueError,
        match="Invalid external artifact reference.*mapping_reference",
    ):
        ArtifactLineage.from_store(root, projection)


def test_lineage_rejects_external_root_with_wrong_fingerprint() -> None:
    query_root = zarr.open_group(store=MemoryStore(), mode="w")
    external_root = zarr.open_group(store=MemoryStore(), mode="w")
    external_root.create_group("RNA").attrs["dataset_fingerprint"] = "wrong-dataset"
    external = ExternalArtifactRef(
        "expected-dataset",
        _ref("mapping_reference", "5"),
    )
    projection = _ref("projection", "6")
    _write_artifact(
        query_root,
        projection,
        operation="map_query",
        inputs={"mapping_reference": external},
    )

    with pytest.raises(
        ValueError,
        match="Expected 'expected-dataset', received 'wrong-dataset'",
    ):
        ArtifactLineage.from_store(
            query_root,
            projection,
            external_roots={"expected-dataset": external_root},
        )


def test_lineage_rejects_external_root_without_stored_fingerprint() -> None:
    query_root = zarr.open_group(store=MemoryStore(), mode="w")
    external_root = zarr.open_group(store=MemoryStore(), mode="w")
    external_root.create_group("RNA")
    external = ExternalArtifactRef(
        "expected-dataset",
        _ref("mapping_reference", "7"),
    )
    projection = _ref("projection", "8")
    _write_artifact(
        query_root,
        projection,
        operation="map_query",
        inputs={"mapping_reference": external},
    )

    with pytest.raises(
        ValueError,
        match="assay 'RNA' has no stored dataset_fingerprint.*expected-dataset",
    ):
        ArtifactLineage.from_store(
            query_root,
            projection,
            external_roots={"expected-dataset": external_root},
        )


def test_lineage_traverses_multiple_namespaced_external_roots() -> None:
    query_root = zarr.open_group(store=MemoryStore(), mode="w")
    first_root = zarr.open_group(store=MemoryStore(), mode="w")
    second_root = zarr.open_group(store=MemoryStore(), mode="w")
    shared_ref = _ref("mapping_reference", "c")
    shared_input = _ref("reduction", "d")
    for root, operation, fingerprint in (
        (first_root, "first_reference", "first-dataset"),
        (second_root, "second_reference", "second-dataset"),
    ):
        _write_artifact(
            root,
            shared_input,
            operation="run_pca",
            inputs={},
        )
        _write_artifact(
            root,
            shared_ref,
            operation=operation,
            inputs={"reduction": shared_input},
        )
        root["RNA"].attrs["dataset_fingerprint"] = fingerprint

    first = ExternalArtifactRef("first-dataset", shared_ref)
    second = ExternalArtifactRef("second-dataset", shared_ref)
    projection = _ref("projection", "e")
    _write_artifact(
        query_root,
        projection,
        operation="map_query",
        inputs={"references": [first, second]},
    )

    lineage = ArtifactLineage.from_store(
        query_root,
        projection,
        external_roots={
            "first-dataset": first_root,
            "second-dataset": second_root,
        },
    )
    first_input = ExternalArtifactRef("first-dataset", shared_input)
    second_input = ExternalArtifactRef("second-dataset", shared_input)

    assert set(lineage.graph) == {
        projection,
        first,
        second,
        first_input,
        second_input,
    }
    assert first != second
    assert first_input != second_input
    assert lineage.graph.edges[first_input, first]["inputs"] == ("reduction",)
    assert lineage.graph.edges[second_input, second]["inputs"] == ("reduction",)
    assert lineage.graph.nodes[first]["status"].operation == "first_reference"
    assert lineage.graph.nodes[second]["status"].operation == "second_reference"
    assert "unresolved external" not in lineage.to_markdown()


def test_lineage_traverses_external_datastore_scoped_dependencies() -> None:
    query_root = zarr.open_group(store=MemoryStore(), mode="w")
    reference_root = zarr.open_group(store=MemoryStore(), mode="w")
    reference_root.create_group("RNA").attrs["dataset_fingerprint"] = (
        "reference-dataset"
    )
    cell_selection = _ref("cell_selection", "3", scope="datastore")
    mapping_ref = _ref("mapping_reference", "4")
    _write_artifact(
        reference_root,
        cell_selection,
        operation="manual_selection",
        inputs={},
    )
    _write_artifact(
        reference_root,
        mapping_ref,
        operation="build_mapping_reference",
        inputs={"cell_selection": cell_selection},
    )
    external = ExternalArtifactRef("reference-dataset", mapping_ref)
    projection = _ref("projection", "5")
    _write_artifact(
        query_root,
        projection,
        operation="map_query",
        inputs={"mapping_reference": external},
    )

    lineage = ArtifactLineage.from_store(
        query_root,
        projection,
        external_roots={"reference-dataset": reference_root},
    )

    operations = {
        lineage.graph.nodes[node]["status"].operation for node in lineage.graph
    }
    assert operations == {
        "map_query",
        "build_mapping_reference",
        "manual_selection",
    }
    assert "external reference-da / datastore" in lineage.to_markdown()


def test_datastore_lineage_resolves_explicit_mapping_references() -> None:
    query_root = zarr.open_group(store=MemoryStore(), mode="w")
    first_root = zarr.open_group(store=MemoryStore(), mode="w")
    second_root = zarr.open_group(store=MemoryStore(), mode="w")
    first_ref = _ref("mapping_reference", "f")
    second_ref = _ref("mapping_reference", "1")
    _write_artifact(
        first_root,
        first_ref,
        operation="build_mapping_reference",
        inputs={},
    )
    _write_artifact(
        second_root,
        second_ref,
        operation="build_mapping_reference",
        inputs={},
    )
    first_reference = _mapping_reference(first_root, first_ref, "first-dataset")
    second_reference = _mapping_reference(second_root, second_ref, "second-dataset")
    projection = _ref("projection", "2")
    first_external = first_reference.external_ref
    second_external = second_reference.external_ref
    _write_artifact(
        query_root,
        projection,
        operation="map_query",
        inputs={"references": [first_external, second_external]},
    )
    datastore = BaseDataStore.__new__(BaseDataStore)
    datastore.z = query_root
    datastore.workspace = None

    partly_resolved = datastore.lineage(
        projection,
        references=first_reference,
    )
    resolved = datastore.lineage(
        projection,
        references=(first_reference, second_reference),
    )

    assert partly_resolved.graph.nodes[first_external]["status"].complete
    assert partly_resolved.graph.nodes[second_external]["status"] is None
    assert resolved.graph.nodes[first_external]["status"].complete
    assert resolved.graph.nodes[second_external]["status"].complete


def test_datastore_lineage_validates_reference_fingerprints_and_root_conflicts() -> (
    None
):
    query_root = zarr.open_group(store=MemoryStore(), mode="w")
    target = _ref("projection", "3")
    _write_artifact(
        query_root,
        target,
        operation="map_query",
        inputs={},
    )
    datastore = BaseDataStore.__new__(BaseDataStore)
    datastore.z = query_root
    datastore.workspace = None

    first_root = zarr.open_group(store=MemoryStore(), mode="w")
    second_root = zarr.open_group(store=MemoryStore(), mode="w")
    reference_ref = _ref("mapping_reference", "4")
    first = _mapping_reference(first_root, reference_ref, "shared-dataset")
    second = _mapping_reference(second_root, reference_ref, "shared-dataset")
    same_root = _mapping_reference(first_root, reference_ref, "shared-dataset")
    datastore.lineage(target, references=(first, same_root))
    with pytest.raises(ValueError, match="conflicting roots"):
        datastore.lineage(target, references=(first, second))

    first_root["RNA"].attrs["dataset_fingerprint"] = "changed-dataset"
    with pytest.raises(
        ValueError,
        match="Expected 'shared-dataset', received 'changed-dataset'",
    ):
        datastore.lineage(target, references=first)

    del first_root["RNA"].attrs["dataset_fingerprint"]
    with pytest.raises(
        ValueError,
        match="no stored dataset fingerprint.*build_mapping_reference",
    ):
        datastore.lineage(target, references=first)
