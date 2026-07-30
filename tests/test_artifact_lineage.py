import networkx as nx
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.datastore.base_datastore import BaseDataStore
from scarf.lineage import ArtifactLineage
from scarf.storage.artifacts import (
    ArtifactRef,
    ArtifactScope,
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
