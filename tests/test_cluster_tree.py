from types import SimpleNamespace

import matplotlib
import networkx as nx
import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import scarf.clustering.cluster_tree as cluster_tree_module
import scarf.clustering.paris as paris_module
import scarf.datastore._operations.paris_persistence as paris_persistence
import scarf.plotting as splt
from scarf.clustering.cluster_tree import CoalesceTree, make_digraph
from scarf.datastore._operations.presentation import _PresentationOperationsMixin
from scarf.plotting.cluster_tree import _hierarchy_positions, _tree_color_series
from scarf.storage.artifacts import (
    ArtifactRef,
    artifact_path,
    inspect_artifact,
    make_provenance,
    parse_artifact_path,
)


class _ClusterTreeStore(_PresentationOperationsMixin):
    def __init__(self, root: zarr.Group, graph_ref: ArtifactRef) -> None:
        self.zw = root
        self.graph_ref = graph_ref

    @staticmethod
    def get_cell_vals(
        *,
        from_assay: str,
        cell_key: str,
        k: str,
    ) -> np.ndarray:
        raise AssertionError(f"Unexpected color lookup for {from_assay}/{cell_key}/{k}")


def _write_complete_artifact(
    root: zarr.Group,
    ref: ArtifactRef,
    *,
    operation: str,
    inputs: dict[str, object],
    arrays: dict[str, np.ndarray] | None = None,
) -> zarr.Group:
    group = root.create_group(artifact_path(ref))
    group.attrs.update(
        {
            "artifact_id": ref.artifact_id,
            "kind": ref.kind,
            "provenance": make_provenance(
                operation=operation,
                parameters={},
                inputs=inputs,
            ),
            "execution_options": {},
            "created_at_ns": 1,
            "complete": True,
        }
    )
    for name, values in (arrays or {}).items():
        group.create_array(name, data=values)
    return group


def _artifact_cluster_tree_store(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_ClusterTreeStore, dict[str, ArtifactRef], MemoryStore]:
    backing = MemoryStore()
    root = zarr.open_group(store=backing, mode="w")
    refs = {
        "selection": ArtifactRef(
            scope="datastore",
            kind="cell_selection",
            artifact_id="1" * 64,
        ),
        "graph": ArtifactRef(
            scope="assay",
            assay="RNA",
            kind="connectivity_map",
            artifact_id="2" * 64,
        ),
        "hierarchy": ArtifactRef(
            scope="assay",
            assay="RNA",
            kind="cluster_hierarchy",
            artifact_id="3" * 64,
        ),
        "cut": ArtifactRef(
            scope="assay",
            assay="RNA",
            kind="cluster_cut",
            artifact_id="4" * 64,
        ),
    }
    _write_complete_artifact(
        root,
        refs["selection"],
        operation="manual_selection",
        inputs={},
    )
    _write_complete_artifact(
        root,
        refs["graph"],
        operation="build_connectivity_map",
        inputs={"cell_selection": refs["selection"]},
    )
    _write_complete_artifact(
        root,
        refs["hierarchy"],
        operation="fit_paris_hierarchy",
        inputs={"connectivity_map": refs["graph"]},
    )
    clusters = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    _write_complete_artifact(
        root,
        refs["cut"],
        operation="cut_paris_hierarchy",
        inputs={
            "cluster_hierarchy": refs["hierarchy"],
            "connectivity_map": refs["graph"],
            "cell_selection": refs["selection"],
        },
        arrays={"labels": clusters},
    )
    cell_data = root.create_group("cellData")
    cluster_column = cell_data.create_array("clusters", data=clusters)
    cluster_column.attrs["source_artifact"] = refs["cut"].to_dict()

    monkeypatch.setattr(
        paris_persistence,
        "load_hierarchy_group",
        lambda _group, _label: (object(), object()),
    )

    def materialize_dendrogram(
        _hierarchy: object,
        *,
        compatibility: bool,
    ) -> np.ndarray:
        assert compatibility is True
        return _balanced_linkage()

    monkeypatch.setattr(
        paris_module,
        "hierarchy_to_dendrogram",
        materialize_dendrogram,
    )
    return _ClusterTreeStore(root, refs["graph"]), refs, backing


def _prepare_artifact_tree(store: _ClusterTreeStore, **kwargs: object):
    invalidate_cache = bool(kwargs.pop("invalidate_cache", False))
    fill_by_value = kwargs.pop("fill_by_value", None)
    return store._prepare_artifact_cluster_tree(
        graph_ref=store.graph_ref,
        from_assay="RNA",
        cell_key="I",
        cluster_key="clusters",
        fill_by_value=fill_by_value,
        invalidate_cache=invalidate_cache,
        **kwargs,
    )


def _partition_ids(graph) -> dict[int, object]:
    return {
        int(node): attributes["partition_id"]
        for node, attributes in graph.nodes(data=True)
        if "partition_id" in attributes
    }


def _balanced_linkage() -> np.ndarray:
    return np.asarray(
        [
            [0, 1, 1, 2],
            [2, 3, 1, 2],
            [4, 5, 1, 2],
            [6, 7, 1, 2],
            [8, 9, 2, 4],
            [10, 11, 2, 4],
            [12, 13, 10, 8],
        ],
        dtype=np.float64,
    )


def _prepared_plot_tree(color_values: np.ndarray | None) -> dict[str, object]:
    graph = nx.DiGraph([(2, 0), (2, 1)])
    graph.nodes[0].update(nleaves=3, partition_id=0)
    graph.nodes[1].update(nleaves=3, partition_id=1)
    graph.nodes[2].update(nleaves=6)
    return {
        "graph": graph,
        "clusters": np.asarray([0, 0, 0, 1, 1, 1]),
        "color_values": color_values,
        "from_assay": "RNA",
        "cell_key": "I",
        "graph_ref": ArtifactRef(
            scope="assay",
            assay="RNA",
            kind="connectivity_map",
            artifact_id="9" * 64,
        ),
        "cluster_key": "clusters",
        "coalesced_location": "RNA/artifacts/cluster_tree/example",
    }


def test_hierarchy_positions_support_undirected_rooted_trees() -> None:
    graph = nx.Graph([(1, 0), (1, 2), (2, 3)])

    positions = _hierarchy_positions(
        graph,
        root=1,
        width=3.0,
        vert_gap=0.5,
    )

    assert set(positions) == set(graph)
    assert positions[1][1] == 0.0
    assert positions[3][1] == -1.0
    assert all(0.0 <= x <= 3.0 for x, _ in positions.values())


def test_cluster_tree_renders_categorical_pies_into_external_axis() -> None:
    calls: list[dict[str, object]] = []
    prepared = _prepared_plot_tree(
        np.asarray(["A", "B", "A", "B", "A", "B"], dtype=object)
    )

    def prepare(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return prepared

    store = SimpleNamespace(_prepare_cluster_tree=prepare)
    figure, ax = plt.subplots(figsize=(4, 4))
    result = splt.cluster_tree(
        store,
        cluster_key="clusters",
        fill_by_value="cell_type",
        color_key={"A": "#ff0000", "B": "#0000ff"},
        show_labels=False,
        ax=ax,
        show=False,
    )

    assert calls == [
        {
            "graph": None,
            "from_assay": None,
            "cell_key": None,
            "cluster_key": "clusters",
            "fill_by_value": "cell_type",
        }
    ]
    assert result.owns_figure is False
    assert result.figure is figure
    assert len(ax.collections) >= 5
    assert result.tables["cluster_summary"]["n_cells"].tolist() == [3, 3]
    assert isinstance(result.scales[0], splt.CategoricalScale)
    assert result.scales[0].order == ("A", "B")
    result.close()
    assert plt.fignum_exists(figure.number)
    plt.close(figure)


def test_cluster_tree_renders_continuous_values_and_closes_owned_figure() -> None:
    uniform, uniform_is_categorical = _tree_color_series(
        np.ones(4),
        force_ints_as_cats=False,
    )
    numeric, numeric_is_categorical = _tree_color_series(
        np.asarray([1, 2, 3]),
        force_ints_as_cats=False,
    )
    np.testing.assert_array_equal(uniform, np.ones(4))
    assert uniform_is_categorical is False
    np.testing.assert_array_equal(numeric, [1.0, 2.0, 3.0])
    assert numeric_is_categorical is False

    prepared = _prepared_plot_tree(np.asarray([0.0, 1.0, 2.0, 4.0, 5.0, 6.0]))
    store = SimpleNamespace(_prepare_cluster_tree=lambda **_kwargs: prepared)
    result = splt.cluster_tree(
        store,
        fill_by_value="score",
        force_ints_as_cats=False,
        cmap="viridis",
        figsize=(4, 3),
        show=False,
    )

    assert result.owns_figure is True
    assert set(result.axes) == {"tree", "colorbar"}
    assert isinstance(result.scales[0], splt.ColorScale)
    assert result.scales[0].vmin == pytest.approx(1.0)
    assert result.scales[0].vmax == pytest.approx(5.0)
    assert [text.get_text() for text in result.axes["tree"].texts] == ["0", "1"]
    assert np.isfinite(result.tables["positions"][["x", "y"]]).all().all()
    figure_number = result.figure.number
    result.close()
    assert not plt.fignum_exists(figure_number)


def test_make_digraph_preserves_linkage_topology_and_leaf_clusters() -> None:
    dendrogram = _balanced_linkage()
    clusters = np.asarray([0, 0, 1, 1, 2, 2, 3, 3])

    graph = make_digraph(dendrogram, clust_info=clusters)

    assert set(graph.nodes) == set(range(15))
    assert set(graph.edges) == {
        (8, 0),
        (8, 1),
        (9, 2),
        (9, 3),
        (10, 4),
        (10, 5),
        (11, 6),
        (11, 7),
        (12, 8),
        (12, 9),
        (13, 10),
        (13, 11),
        (14, 12),
        (14, 13),
    }
    assert [graph.nodes[leaf]["cluster"] for leaf in range(8)] == clusters.tolist()
    assert graph.nodes[14]["nleaves"] == 8
    assert graph.nodes[14]["dist"] == 10


def test_make_digraph_rejects_mismatched_cluster_info() -> None:
    dendrogram = _balanced_linkage()
    with pytest.raises(ValueError, match="cluster information"):
        make_digraph(dendrogram, clust_info=np.zeros(3))


def test_coalesce_tree_retains_cluster_holding_nodes_and_ancestors() -> None:
    clusters = np.asarray([0, 0, 1, 1, 2, 2, 3, 3])
    graph = make_digraph(_balanced_linkage(), clust_info=clusters)

    coalesced = CoalesceTree(graph, clusters)

    assert set(coalesced.nodes) == set(range(8, 15))
    assert set(coalesced.edges) == {
        (12, 8),
        (12, 9),
        (13, 10),
        (13, 11),
        (14, 12),
        (14, 13),
    }
    assert {
        int(node): int(attributes["partition_id"])
        for node, attributes in coalesced.nodes(data=True)
        if "partition_id" in attributes
    } == {8: 0, 9: 1, 10: 2, 11: 3}


def test_coalesce_tree_rejects_non_monophyletic_clusters() -> None:
    clusters = np.asarray([0, 1, 0, 1, 2, 2, 3, 3])
    graph = make_digraph(_balanced_linkage(), clust_info=clusters)

    with pytest.raises(ValueError, match="not monophyletic"):
        CoalesceTree(graph, clusters)


def test_artifact_cluster_tree_cache_hit_is_compute_free_and_read_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _refs, backing = _artifact_cluster_tree_store(monkeypatch)
    prepared = _prepare_artifact_tree(store)

    def fail_recompute(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("cache hit recomputed the cluster tree")

    monkeypatch.setattr(
        paris_module,
        "hierarchy_to_dendrogram",
        fail_recompute,
    )
    monkeypatch.setattr(
        cluster_tree_module,
        "CoalesceTree",
        fail_recompute,
    )
    store.zw = zarr.open_group(store=backing.with_read_only(True), mode="r")

    cached = _prepare_artifact_tree(store)

    assert cached["coalesced_location"] == prepared["coalesced_location"]
    assert set(cached["graph"].edges) == set(prepared["graph"].edges)
    assert _partition_ids(cached["graph"]) == _partition_ids(prepared["graph"])


def test_artifact_cluster_tree_invalidation_forces_cache_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _refs, _backing = _artifact_cluster_tree_store(monkeypatch)
    prepared = _prepare_artifact_tree(store)
    original_coalesce = cluster_tree_module.CoalesceTree
    original_materialize = paris_module.hierarchy_to_dendrogram
    calls = {"coalesce": 0, "dendrogram": 0}

    def track_coalesce(*args: object, **kwargs: object):
        calls["coalesce"] += 1
        return original_coalesce(*args, **kwargs)

    def track_materialize(*args: object, **kwargs: object):
        calls["dendrogram"] += 1
        return original_materialize(*args, **kwargs)

    monkeypatch.setattr(cluster_tree_module, "CoalesceTree", track_coalesce)
    monkeypatch.setattr(
        paris_module,
        "hierarchy_to_dendrogram",
        track_materialize,
    )

    invalidated = _prepare_artifact_tree(store, invalidate_cache=True)

    assert calls == {"coalesce": 1, "dendrogram": 1}
    assert invalidated["coalesced_location"] != prepared["coalesced_location"]
    assert inspect_artifact(
        store.zw,
        parse_artifact_path(invalidated["coalesced_location"]),
    ).complete


@pytest.mark.parametrize(
    ("damage", "expected_dendrogram_calls"),
    [
        ("incomplete_coalesced", 0),
        ("missing_coalesced_array", 0),
        ("missing_dendrogram_array", 1),
    ],
)
def test_artifact_cluster_tree_does_not_reuse_incomplete_or_malformed_cache(
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
    expected_dendrogram_calls: int,
) -> None:
    store, _refs, _backing = _artifact_cluster_tree_store(monkeypatch)
    prepared = _prepare_artifact_tree(store)
    coalesced_ref = parse_artifact_path(prepared["coalesced_location"])
    coalesced_group = store.zw[artifact_path(coalesced_ref)]
    coalesced_status = inspect_artifact(store.zw, coalesced_ref)
    assert coalesced_status.inputs is not None
    dendrogram_ref = ArtifactRef.from_dict(coalesced_status.inputs["dendrogram"])

    if damage == "incomplete_coalesced":
        coalesced_group.attrs["complete"] = False
    elif damage == "missing_coalesced_array":
        del coalesced_group["nodelist"]
    else:
        del store.zw[artifact_path(dendrogram_ref)]["data"]

    original_coalesce = cluster_tree_module.CoalesceTree
    original_materialize = paris_module.hierarchy_to_dendrogram
    calls = {"coalesce": 0, "dendrogram": 0}

    def track_coalesce(*args: object, **kwargs: object):
        calls["coalesce"] += 1
        return original_coalesce(*args, **kwargs)

    def track_materialize(*args: object, **kwargs: object):
        calls["dendrogram"] += 1
        return original_materialize(*args, **kwargs)

    monkeypatch.setattr(cluster_tree_module, "CoalesceTree", track_coalesce)
    monkeypatch.setattr(
        paris_module,
        "hierarchy_to_dendrogram",
        track_materialize,
    )

    repaired = _prepare_artifact_tree(store)

    assert calls == {
        "coalesce": 1,
        "dendrogram": expected_dendrogram_calls,
    }
    assert repaired["coalesced_location"] != prepared["coalesced_location"]
    assert inspect_artifact(
        store.zw,
        parse_artifact_path(repaired["coalesced_location"]),
    ).complete


def test_artifact_cluster_tree_rejects_graph_from_different_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, refs, _backing = _artifact_cluster_tree_store(monkeypatch)
    other_selection = ArtifactRef(
        scope="datastore",
        kind="cell_selection",
        artifact_id="5" * 64,
    )
    other_graph = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="connectivity_map",
        artifact_id="6" * 64,
    )
    _write_complete_artifact(
        store.zw,
        other_selection,
        operation="manual_selection",
        inputs={},
    )
    _write_complete_artifact(
        store.zw,
        other_graph,
        operation="build_connectivity_map",
        inputs={"cell_selection": other_selection},
    )

    with pytest.raises(
        ValueError,
        match="Cluster cut does not belong to the requested graph",
    ):
        store._prepare_artifact_cluster_tree(
            graph_ref=other_graph,
            from_assay="RNA",
            cell_key="I",
            cluster_key="clusters",
            fill_by_value=None,
            invalidate_cache=False,
        )

    assert refs["graph"] != other_graph


def test_artifact_cluster_tree_rejects_graph_scope_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, refs, _backing = _artifact_cluster_tree_store(monkeypatch)
    datastore_graph = ArtifactRef(
        scope="datastore",
        kind="connectivity_map",
        artifact_id="7" * 64,
    )
    _write_complete_artifact(
        store.zw,
        datastore_graph,
        operation="build_connectivity_map",
        inputs={"cell_selection": refs["selection"]},
    )

    with pytest.raises(
        ValueError,
        match="Cluster cut does not belong to the requested graph",
    ):
        store._prepare_artifact_cluster_tree(
            graph_ref=datastore_graph,
            from_assay="RNA",
            cell_key="I",
            cluster_key="clusters",
            fill_by_value=None,
            invalidate_cache=False,
        )
