"""Regression tests for DataStore presentation helpers."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import zarr
from networkx import DiGraph
from scipy.sparse import csr_matrix
from zarr.storage import MemoryStore

from scarf.datastore._operations.presentation import _PresentationOperationsMixin
from scarf.storage.artifacts import (
    ArtifactRef,
    artifact_path,
    make_provenance,
    new_artifact_id,
)


class _PresentationStore(_PresentationOperationsMixin):
    def __init__(self, root: zarr.Group) -> None:
        self.zw = root


def _presentation_store() -> tuple[_PresentationStore, MemoryStore]:
    backing = MemoryStore()
    root = zarr.open_group(store=backing, mode="w")
    root.attrs["title"] = "small store"
    root.create_array(
        "root_values",
        data=np.asarray([True, False], dtype=np.bool_),
        chunks=(1,),
    )
    branch = root.create_group("branch")
    branch.attrs.update({"label": "selected", "rank": 1})
    values = branch.create_array(
        "values",
        data=np.arange(6, dtype=np.float32).reshape(2, 3),
        chunks=(1, 3),
    )
    values.attrs["units"] = "counts"
    nested = branch.create_group("nested")
    nested.create_array(
        "deep",
        data=np.arange(4, dtype=np.int16),
        chunks=(2,),
    )
    root.create_group("sibling").create_array(
        "hidden",
        data=np.asarray([1], dtype=np.uint8),
    )
    return _PresentationStore(root), backing


def _write_complete_artifact(
    root: zarr.Group,
    kind: str,
    *,
    assay: str | None = "RNA",
    inputs: dict[str, object] | None = None,
    arrays: dict[str, np.ndarray] | None = None,
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
                operation=f"test_{kind}",
                parameters={},
                inputs=inputs or {},
            ),
            "execution_options": {},
            "complete": True,
        }
    )
    for name, values in (arrays or {}).items():
        group.create_array(name, data=values)
    return ref


@pytest.mark.parametrize("start", ["branch", "/branch", "branch/", "/branch/"])
def test_show_zarr_tree_normalizes_path_and_filters_to_subtree(
    start: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store, _backing = _presentation_store()

    store.show_zarr_tree(start=start, depth=0)
    captured = capsys.readouterr().out

    assert "/branch" in captured
    assert "nested" in captured
    assert "values" in captured
    assert "deep" not in captured
    assert "root_values" not in captured
    assert "sibling" not in captured


def test_show_zarr_tree_depth_controls_nested_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    store, _backing = _presentation_store()

    store.show_zarr_tree(start="branch", depth=0)
    shallow = capsys.readouterr().out
    store.show_zarr_tree(start="branch", depth=1)
    deep = capsys.readouterr().out

    assert "deep" not in shallow
    assert "deep (4,) int16" in deep


def test_show_zarr_tree_formats_arrays_without_mutating_attributes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    store, _backing = _presentation_store()
    branch_attrs = dict(store.zw["branch"].attrs)
    array_attrs = dict(store.zw["branch/values"].attrs)

    store.show_zarr_tree(start="branch", depth=1)
    captured = capsys.readouterr().out

    assert "values (2, 3) float32" in captured
    assert "values: shape=(2, 3), dtype=float32, chunks=(1, 3)" in captured
    assert dict(store.zw["branch"].attrs) == branch_attrs
    assert dict(store.zw["branch/values"].attrs) == array_attrs


@pytest.mark.parametrize(
    ("start", "error_type"),
    [
        ("does_not_exist", KeyError),
        ("branch/values", TypeError),
    ],
)
def test_show_zarr_tree_rejects_invalid_start_path(
    start: str,
    error_type: type[Exception],
) -> None:
    store, _backing = _presentation_store()

    with pytest.raises(error_type):
        store.show_zarr_tree(start=start, depth=1)


@pytest.mark.parametrize(
    ("depth", "error_type"),
    [
        (-1, ValueError),
        ("one", TypeError),
    ],
)
def test_show_zarr_tree_rejects_invalid_depth(
    depth: object,
    error_type: type[Exception],
) -> None:
    store, _backing = _presentation_store()

    with pytest.raises(error_type):
        store.show_zarr_tree(depth=depth)


def test_show_zarr_tree_operates_on_read_only_memory_store(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _writable, backing = _presentation_store()
    root = zarr.open_group(store=backing.with_read_only(True), mode="r")
    store = _PresentationStore(root)
    before_root_attrs = dict(root.attrs)
    before_branch_attrs = dict(root["branch"].attrs)

    store.show_zarr_tree(start="/", depth=1)
    captured = capsys.readouterr().out

    assert {"branch", "root_values", "sibling"} <= set(captured.split())
    assert dict(root.attrs) == before_root_attrs
    assert dict(root["branch"].attrs) == before_branch_attrs


def test_to_anndata_exports_an_empty_feature_selection() -> None:
    store, _backing = _presentation_store()
    features = SimpleNamespace(
        N=2,
        columns=["ids", "names"],
        to_pandas_dataframe=Mock(
            return_value=pd.DataFrame({"ids": ["f0", "f1"], "names": ["g0", "g1"]})
        ),
        fetch_all=Mock(return_value=np.asarray(["f0", "f1"])),
    )
    assay = SimpleNamespace(
        feats=features,
        to_raw_sparse=Mock(return_value=csr_matrix([[1, 2], [3, 4]])),
    )
    store._get_assay = Mock(return_value=assay)
    store.cells = SimpleNamespace(
        columns=["ids"],
        active_index=Mock(return_value=np.asarray([0, 1])),
        to_pandas_dataframe=Mock(return_value=pd.DataFrame({"ids": ["c0", "c1"]})),
    )

    exported = store.to_anndata(feature_indexes=[])

    assert exported.shape == (2, 0)
    assert list(exported.obs_names) == ["c0", "c1"]
    assay.to_raw_sparse.assert_called_once_with("I")


def test_membership_strength_rejects_a_different_graph_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _backing = _presentation_store()
    graph_ref = _write_complete_artifact(store.zw, "connectivity_map")
    requested = ArtifactRef(
        scope="datastore",
        kind="cell_selection",
        artifact_id="1" * 64,
    )
    graph_selection = ArtifactRef(
        scope="datastore",
        kind="cell_selection",
        artifact_id="2" * 64,
    )
    store.get_latest_graph_loc = Mock(return_value=artifact_path(graph_ref))
    store._get_graph_ncells_k = Mock(return_value=(2, 1))
    store._ensure_cell_selection = Mock(return_value=requested)
    store._graph_cell_selection = Mock(return_value=graph_selection)
    store._selection_artifacts_match = Mock(return_value=False)
    monkeypatch.setattr(
        "scarf.datastore._operations.presentation.resolve_stored_graph_input",
        Mock(return_value=graph_ref),
    )

    with pytest.raises(ValueError, match="cell_key does not match"):
        store.calc_membership_strength("RNA", "I", "I", "clusters")


@pytest.mark.parametrize(
    ("edges", "message"),
    [
        (np.asarray([[0, 1]], dtype=np.uint32), "stored cell and k dimensions"),
        (
            np.asarray([[0, 1], [0, 0]], dtype=np.uint32),
            "cell-major order",
        ),
    ],
)
def test_membership_strength_validates_legacy_edge_layout(
    monkeypatch: pytest.MonkeyPatch,
    edges: np.ndarray,
    message: str,
) -> None:
    store, _backing = _presentation_store()
    graph = store.zw.create_group("legacy_graph")
    graph.create_array("edges", data=edges)
    selection = ArtifactRef(
        scope="datastore",
        kind="cell_selection",
        artifact_id="3" * 64,
    )
    result = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="membership_strength",
        artifact_id="4" * 64,
    )
    store.get_latest_graph_loc = Mock(return_value="legacy_graph")
    store._get_graph_ncells_k = Mock(return_value=(2, 1))
    store._ensure_cell_selection = Mock(return_value=selection)
    store._resolve_cell_data_provenance_input = Mock(return_value=selection)
    store.cells = SimpleNamespace(fetch=Mock(return_value=np.asarray([0, 1])))
    monkeypatch.setattr(
        "scarf.datastore._operations.presentation.resolve_stored_graph_input",
        Mock(return_value={"legacy_graph_fingerprint": "a" * 64}),
    )
    monkeypatch.setattr(
        "scarf.datastore._operations.presentation.validate_legacy_graph_selection",
        Mock(),
    )
    monkeypatch.setattr(
        "scarf.datastore._operations.presentation.plan_cell_data_artifact",
        Mock(return_value=SimpleNamespace(ref=result, reused=False)),
    )
    monkeypatch.setattr(
        "scarf.datastore._operations.presentation.column_display",
        Mock(return_value=None),
    )

    with pytest.raises(ValueError, match=message):
        store.calc_membership_strength("RNA", "I", "I", "clusters")


def test_membership_strength_reuses_values_and_display_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _backing = _presentation_store()
    graph_ref = _write_complete_artifact(store.zw, "connectivity_map")
    selection = ArtifactRef(
        scope="datastore",
        kind="cell_selection",
        artifact_id="5" * 64,
    )
    result = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="membership_strength",
        artifact_id="6" * 64,
    )
    result_group = store.zw.create_group(artifact_path(result))
    result_group.create_array(
        "values",
        data=np.asarray([0.5, 1.0], dtype=np.float32),
    )
    store.get_latest_graph_loc = Mock(return_value=artifact_path(graph_ref))
    store._get_graph_ncells_k = Mock(return_value=(2, 1))
    store._ensure_cell_selection = Mock(return_value=selection)
    store._graph_cell_selection = Mock(return_value=selection)
    store._selection_artifacts_match = Mock(return_value=True)
    store._resolve_cell_data_provenance_input = Mock(return_value=selection)
    insert = Mock()
    store.cells = SimpleNamespace(insert=insert)
    link = Mock()
    monkeypatch.setattr(
        "scarf.datastore._operations.presentation.resolve_stored_graph_input",
        Mock(return_value=graph_ref),
    )
    monkeypatch.setattr(
        "scarf.datastore._operations.presentation.plan_cell_data_artifact",
        Mock(return_value=SimpleNamespace(ref=result, reused=True)),
    )
    monkeypatch.setattr(
        "scarf.datastore._operations.presentation.column_display",
        Mock(return_value={"palette": "preserved"}),
    )
    monkeypatch.setattr(
        "scarf.datastore._operations.presentation.link_cell_data_column",
        link,
    )

    assert store.calc_membership_strength("RNA", "I", "I", "clusters") is None

    inserted = insert.call_args.args[1]
    np.testing.assert_allclose(inserted, [0.5, 1.0])
    assert link.call_args.kwargs["preserved_display"] == {"palette": "preserved"}
    assert link.call_args.kwargs["default_display"]["minimum"] == 0.0
    assert link.call_args.kwargs["default_display"]["maximum"] == 1.0


def test_smart_label_handles_empty_and_unmatched_base_labels() -> None:
    store, _backing = _presentation_store()
    store.cells = SimpleNamespace(fetch=Mock(return_value=np.asarray([])))
    assert store.smart_label("clusters", "base") == []
    with pytest.raises(ValueError, match="selects no cells"):
        store.smart_label("clusters", "base", new_col_name="labels")

    values = {
        "clusters": np.asarray(["a", "a", "a"]),
        "base": np.asarray(["X", "X", "Y"]),
    }
    store.cells = SimpleNamespace(fetch=lambda name, **_kwargs: values[name])
    assert store.smart_label("clusters", "base") == ["X-Ya", "X-Ya", "X-Ya"]


def test_prepare_cluster_tree_rejects_unresolved_inputs() -> None:
    store, _backing = _presentation_store()
    store._get_latest_keys = Mock(return_value=("RNA", "I", "I"))
    with pytest.raises(ValueError, match="provide a value for `cluster_key`"):
        store._prepare_cluster_tree()

    store._resolve_integrated_graph_path = Mock(return_value=None)
    with pytest.raises(KeyError, match="does not exist"):
        store._prepare_cluster_tree(
            integrated_graph="missing",
            cluster_key="clusters",
        )

    cell_data = store.zw.create_group("cellData")
    cell_data.create_array("clusters", data=np.asarray([0, 1]))
    graph_ref = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="connectivity_map",
        artifact_id="7" * 64,
    )
    store.get_latest_graph_loc = Mock(return_value=artifact_path(graph_ref))
    with pytest.raises(ValueError, match="no source artifact for this graph"):
        store._prepare_cluster_tree(cluster_key="clusters")


def test_artifact_cluster_tree_requires_cut_and_hierarchy_provenance() -> None:
    store, _backing = _presentation_store()
    cell_data = store.zw.create_group("cellData")
    cluster_column = cell_data.create_array(
        "clusters",
        data=np.asarray([0, 1]),
    )
    graph_ref = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="connectivity_map",
        artifact_id="8" * 64,
    )

    with pytest.raises(ValueError, match="no source artifact"):
        store._prepare_artifact_cluster_tree(
            graph_ref=graph_ref,
            graph_loc=artifact_path(graph_ref),
            from_assay="RNA",
            cell_key="I",
            feat_key="I",
            integrated_graph=None,
            cluster_key="clusters",
            fill_by_value=None,
            invalidate_cache=False,
        )

    cut_ref = _write_complete_artifact(
        store.zw,
        "cluster_cut",
        inputs={"connectivity_map": graph_ref},
        arrays={"labels": np.asarray([0, 1])},
    )
    cluster_column.attrs["source_artifact"] = cut_ref.to_dict()
    with pytest.raises(ValueError, match="no hierarchy input"):
        store._prepare_artifact_cluster_tree(
            graph_ref=graph_ref,
            graph_loc=artifact_path(graph_ref),
            from_assay="RNA",
            cell_key="I",
            feat_key="I",
            integrated_graph=None,
            cluster_key="clusters",
            fill_by_value=None,
            invalidate_cache=False,
        )


def test_legacy_cluster_tree_materializes_and_reuses_table_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _backing = _presentation_store()
    store.zw.create_group("legacy_graph")
    store.zw.create_array(
        "legacy_dendrogram",
        data=np.asarray([[0, 1, 1.0, 2]], dtype=np.float64),
    )
    cell_data = store.zw.create_group("cellData")
    cell_data.create_array("clusters", data=np.asarray([0, 1]))
    object_clusters = np.asarray(["A", "B"], dtype=object)
    colors = np.asarray([0.25, 0.75])
    store._get_latest_keys = Mock(return_value=("RNA", "I", "I"))
    store.get_latest_graph_loc = Mock(return_value="legacy_graph")
    store.resources = None
    store.cells = SimpleNamespace(fetch=Mock(return_value=object_clusters))
    store.get_cell_vals = Mock(return_value=colors)

    coalesced = DiGraph()
    coalesced.add_edges_from([(2, 0), (2, 1)])
    coalesced.nodes[2]["nleaves"] = 2
    coalesced.nodes[0].update({"nleaves": 1, "partition_id": "A"})
    coalesced.nodes[1].update({"nleaves": 1, "partition_id": "B"})
    make_digraph = Mock(return_value=object())
    coalesce_tree = Mock(return_value=coalesced)
    monkeypatch.setattr(
        "scarf.datastore._operations.paris_persistence."
        "resolve_compatibility_dendrogram",
        Mock(return_value=("legacy_dendrogram", "generation-1")),
    )
    monkeypatch.setattr(
        "scarf.clustering.cluster_tree.make_digraph",
        make_digraph,
    )
    monkeypatch.setattr(
        "scarf.clustering.cluster_tree.CoalesceTree",
        coalesce_tree,
    )

    materialized = store._prepare_cluster_tree(
        cluster_key="clusters",
        fill_by_value="signal",
    )

    cache_path = materialized["coalesced_location"]
    assert store.zw[cache_path].attrs["complete"] is True
    assert materialized["graph"] is coalesced
    np.testing.assert_array_equal(materialized["color_values"], colors)
    make_digraph.assert_called_once()
    coalesce_tree.assert_called_once()

    make_digraph.reset_mock()
    coalesce_tree.reset_mock()
    make_digraph.side_effect = AssertionError("cached tree must not be rebuilt")
    coalesce_tree.side_effect = AssertionError("cached tree must not be rebuilt")
    cached = store._prepare_cluster_tree(
        cluster_key="clusters",
        fill_by_value="signal",
    )

    assert set(cached["graph"].edges) == {(2, 0), (2, 1)}
    assert cached["graph"].nodes[0]["partition_id"] == "A"
    assert cached["graph"].nodes[1]["partition_id"] == "B"
    make_digraph.assert_not_called()
    coalesce_tree.assert_not_called()

    make_digraph.side_effect = None
    coalesce_tree.side_effect = None
    make_digraph.return_value = object()
    coalesce_tree.return_value = coalesced
    store.cells.fetch = Mock(return_value=np.asarray([0, 1]))
    numeric = store._prepare_cluster_tree(cluster_key="clusters")
    assert numeric["coalesced_location"] != cache_path
