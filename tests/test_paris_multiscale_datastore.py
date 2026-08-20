import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import zarr
from scipy.sparse import csr_matrix
from zarr.storage import MemoryStore

from scarf.clustering._paris_core import ParisHierarchy
from scarf.clustering.paris_multiscale import PlateauForest
from scarf.datastore._operations.clustering import _ClusteringOperationsMixin
from scarf.datastore._operations.presentation import _PresentationOperationsMixin
from scarf.datastore._operations.paris_persistence import (
    LATEST_PARIS_GENERATION,
    estimate_paris_adaptive_cut_peak_bytes,
    estimate_paris_peak_bytes,
    estimate_hierarchy_group_peak_bytes,
    generation_location,
    load_hierarchy_group,
)
from scarf.storage.artifacts import (
    ArtifactRef,
    artifact_path,
    inspect_artifact,
    list_artifacts,
)
from scarf.storage.selections import resolve_selection_artifact
from scarf.storage.budget import ResourceBudget


class _Cells:
    def __init__(self, active: np.ndarray, root: zarr.Group) -> None:
        self.active = active
        self.data: dict[str, np.ndarray] = {
            "I": active.copy(),
            "ids": np.asarray([f"cell_{i}" for i in range(len(active))]),
        }
        self.writes: list[str] = []
        self.root = root
        self.N = len(active)
        cell_data = root.create_group("cellData")
        cell_data.create_array("I", data=active.copy())
        cell_data.create_array("ids", data=self.data["ids"])

    @property
    def columns(self) -> list[str]:
        return list(self.data)

    def fetch_all(self, name: str) -> np.ndarray:
        return self.data[name]

    def fetch(self, name: str, key: str = "I") -> np.ndarray:
        return self.data[name][self.data[key].astype(bool)]

    def insert(
        self,
        name: str,
        values: np.ndarray,
        *,
        fill_value: object = np.nan,
        key: str = "I",
        overwrite: bool = False,
    ) -> None:
        del overwrite
        active = self.data[key].astype(bool)
        incoming = np.asarray(values)
        if incoming.shape != (int(active.sum()),):
            raise ValueError("metadata values do not match the active key")
        filled = np.full(active.shape, fill_value, dtype=incoming.dtype)
        filled[active] = incoming
        self.data[name] = filled
        self.writes.append(name)
        cell_data = self.root["cellData"]
        if name in cell_data:
            del cell_data[name]
        cell_data.create_array(name, data=filled)


class _Store(_ClusteringOperationsMixin, _PresentationOperationsMixin):
    _integratedGraphsLoc = "integratedGraphs"

    def __init__(
        self,
        graph: csr_matrix,
        *,
        extra_cells: int = 0,
    ) -> None:
        self.zw = zarr.open_group(store=MemoryStore(), mode="w")
        active = np.zeros(graph.shape[0] + extra_cells, dtype=bool)
        active[: graph.shape[0]] = True
        self.cells = _Cells(active, self.zw)
        self.nthreads = 2
        self.resources = ResourceBudget(8 * 1024**3, self.nthreads)
        self.zarr_mode = "r+"
        self.graphs: dict[str, csr_matrix] = {}
        self.latest_graph_calls = 0
        self.raise_on_latest_graph = False
        self.load_graph_calls = 0
        self._write_graph("RNA/graph", graph, k=3)

    def _write_graph(self, location: str, graph: csr_matrix, *, k: int) -> None:
        group = self.zw.create_group(location, overwrite=True)
        coo = graph.tocoo()
        group.create_array(
            "edges",
            data=np.column_stack((coo.row, coo.col)).astype(np.uint64),
        )
        group.create_array("weights", data=coo.data.astype(np.float64))
        group.attrs.update(
            {
                "n_cells": graph.shape[0],
                "n_neighbors": k,
            }
        )
        self.graphs[location] = graph

    def add_integrated_graph(
        self, label: str, graph: csr_matrix, *, k: int = 3
    ) -> None:
        self._write_graph(f"{self._integratedGraphsLoc}/{label}", graph, k=k)

    def _resolve_integrated_graph_path(self, label: str) -> str:
        return f"{self._integratedGraphsLoc}/{label}"

    def _get_latest_keys(
        self,
        from_assay: str | None,
        cell_key: str | None,
        feat_key: str | None,
    ) -> tuple[str, str, str]:
        return from_assay or "RNA", cell_key or "I", feat_key or "hvgs"

    def get_latest_graph_loc(
        self,
        from_assay: str,
        cell_key: str,
        feat_key: str,
    ) -> str:
        del from_assay, cell_key, feat_key
        self.latest_graph_calls += 1
        if self.raise_on_latest_graph:
            raise AssertionError("standard graph lookup was not expected")
        return "RNA/graph"

    def _lookup_stored_graph(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        graph_loc: str | None = None,
    ):
        from scarf.graph.paths import AssayGraphPaths, StoredAssayGraph

        if graph_loc is not None:
            path = graph_loc
        else:
            path = self.get_latest_graph_loc(
                from_assay or "RNA",
                cell_key or "I",
                feat_key or "hvgs",
            )
        return StoredAssayGraph(
            paths=AssayGraphPaths(
                normalized_group_path="RNA/normed__I__hvgs",
                reduction_group_path="RNA/normed__I__hvgs/reduction__pca__10__I",
                neighbor_index_group_path=(
                    "RNA/normed__I__hvgs/reduction__pca__10__I/ann__l2__50__50__16__1"
                ),
                nearest_neighbors_group_path=(
                    "RNA/normed__I__hvgs/reduction__pca__10__I/"
                    "ann__l2__50__50__16__1/knn__11"
                ),
                cell_graph_group_path=path,
            ),
            from_assay=from_assay or "RNA",
            cell_key=cell_key or "I",
            feat_key=feat_key or "hvgs",
        )

    def _get_graph_ncells_k(self, graph_loc: str) -> tuple[int, int]:
        attrs = self.zw[graph_loc].attrs
        return int(attrs["n_cells"]), int(attrs["n_neighbors"])

    def load_graph(self, *, graph_loc: str, **_kwargs: object) -> csr_matrix:
        self.load_graph_calls += 1
        return self.graphs[graph_loc]

    @staticmethod
    def _col_renamer(from_assay: str, cell_key: str, label: str) -> str:
        if cell_key == "I":
            return f"{from_assay}_{label}"
        return f"{from_assay}_{cell_key}_{label}"

    def get_cell_vals(
        self,
        *,
        from_assay: str,
        cell_key: str,
        k: str,
    ) -> np.ndarray:
        del from_assay
        return self.cells.fetch(k, key=cell_key)

    def _ensure_cell_selection(self, column: str) -> ArtifactRef:
        return resolve_selection_artifact(
            self.zw,
            scope="datastore",
            kind="cell_selection",
            values=self.cells.fetch_all(column),
            row_ids=self.cells.fetch_all("ids"),
            operation="manual_selection",
            parameters={},
            inputs={},
            source_column=column,
        )


def _block_graph() -> csr_matrix:
    graph = np.zeros((14, 14), dtype=np.float64)
    for start, stop in ((0, 6), (6, 14)):
        rows, columns = np.triu_indices(stop - start, k=1)
        graph[start + rows, start + columns] = 1
    graph[5, 6] = 0.01
    return csr_matrix(graph)


def _disconnected_graph() -> csr_matrix:
    edges = (
        (0, 1, 9.0),
        (1, 2, 4.0),
        (2, 3, 1.0),
        (4, 5, 8.0),
        (5, 6, 3.0),
        (6, 7, 0.5),
    )
    return csr_matrix(
        (
            [weight for _left, _right, weight in edges],
            (
                [left for left, _right, _weight in edges],
                [right for _left, right, _weight in edges],
            ),
        ),
        shape=(8, 8),
    )


def _load_artifact_hierarchy(
    store: _Store,
    artifact_id: str,
) -> tuple[ParisHierarchy, PlateauForest]:
    ref = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="cluster_hierarchy",
        artifact_id=artifact_id,
    )
    group = store.zw[artifact_path(ref)]
    return load_hierarchy_group(group, artifact_id)


def _column_artifact(store: _Store, column: str) -> ArtifactRef:
    raw_ref = store.zw["cellData"][column].attrs["source_artifact"]
    return ArtifactRef.from_dict(raw_ref)


def test_auto_cut_persists_typed_hierarchy_and_reuses_diagnostics() -> None:
    store = _Store(_block_graph(), extra_cells=2)
    first = store.run_paris_clustering(min_cluster_size=2)
    graph_group = store.zw["RNA/graph"]
    generation_id = first.hierarchy_generation_id
    assert generation_id is not None
    hierarchy, forest = _load_artifact_hierarchy(store, generation_id)

    assert hierarchy.n_leaves == 14
    assert forest.n_leaves == 14
    assert LATEST_PARIS_GENERATION not in graph_group.attrs
    assert "latest_dendrogram" not in graph_group.attrs
    assert first.n_clusters == 2
    assert first.labels[:6].tolist() == [1] * 6
    assert first.labels[6:].tolist() == [2] * 8
    assert store.cells.data["RNA_paris_cluster"][-2:].tolist() == [-1, -1]
    assert set(store.cells.data["RNA_paris_cluster"][:-2]) == {1, 2}
    assert store.cells.writes == ["RNA_paris_cluster"]
    assert first.ref is not None
    assert first.ref == ArtifactRef.from_dict(
        store.zw["cellData/RNA_paris_cluster"].attrs["source_artifact"]
    )

    second = store.run_paris_clustering(min_cluster_size=2)
    assert second.ref == first.ref
    assert second.hierarchy_generation_id == generation_id
    assert np.array_equal(second.labels, first.labels)
    assert second.diagnostics == first.diagnostics
    assert store.load_graph_calls == 1
    assert store.cells.writes == ["RNA_paris_cluster", "RNA_paris_cluster"]


def test_fit_uses_additive_graph_and_fixed_cut_materializes_linkage_lazily() -> None:
    graph = _block_graph()
    store = _Store(graph)
    result = store.run_paris_clustering(n_clusters=3)
    generation_id = result.hierarchy_generation_id
    assert generation_id is not None
    hierarchy, _forest = _load_artifact_hierarchy(store, generation_id)
    expected_children = np.asarray(
        [
            [0, 1],
            [7, 8],
            [2, 3],
            [9, 10],
            [4, 14],
            [11, 12],
            [13, 15],
            [16, 18],
            [5, 21],
            [17, 19],
            [20, 23],
            [6, 24],
            [22, 25],
        ],
        dtype=np.int64,
    )
    expected_heights = np.asarray(
        [
            0.2906300860265055,
            0.5696349686119507,
            0.2906300860265055,
            0.5696349686119507,
            0.2906300860265055,
            0.5696349686119507,
            0.5696349686119507,
            0.2906300860265055,
            0.2912113461985585,
            0.5696349686119507,
            0.5696349686119507,
            0.5704487328528249,
            1954.034061846082,
        ]
    )

    np.testing.assert_array_equal(hierarchy.children, expected_children)
    np.testing.assert_allclose(hierarchy.heights, expected_heights, rtol=1e-12)
    dendrograms = list_artifacts(
        store.zw,
        scope="assay",
        assay="RNA",
        kind="dendrogram",
    )
    assert len(dendrograms) == 1
    assert artifact_path(dendrograms[0]) in store.zw
    assert "latest_dendrogram" not in store.zw["RNA/graph"].attrs
    assert result.mode == "fixed"
    assert result.n_clusters == 3
    assert set(result.labels) == {1, 2, 3}
    assert result.diagnostics == ()


def test_fixed_cut_uses_raw_hierarchy_for_disconnected_graph() -> None:
    store = _Store(_disconnected_graph())

    result = store.run_paris_clustering(n_clusters=3)

    assert result.n_clusters == 3
    assert np.unique(result.labels).size == 3
    assert result.labels[0] == result.labels[1]
    assert result.labels[2] == result.labels[3]
    assert np.unique(result.labels[4:]).size == 1
    assert len({result.labels[0], result.labels[2], result.labels[4]}) == 3
    generation_id = result.hierarchy_generation_id
    assert generation_id is not None
    hierarchy, _forest = _load_artifact_hierarchy(store, generation_id)
    assert hierarchy.synthetic_joins.sum() == 1
    dendrogram_ref = list_artifacts(
        store.zw,
        scope="assay",
        assay="RNA",
        kind="dendrogram",
    )[0]
    compatibility = np.asarray(store.zw[artifact_path(dendrogram_ref)]["data"][:])
    assert compatibility[hierarchy.synthetic_joins, 2].tolist() == [0.0]


def test_auto_cut_applies_the_modularity_split_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scarf.clustering._paris_modularity as modularity

    store = _Store(_block_graph())
    seen_graph_shapes: list[tuple[int, int]] = []

    def veto_all_splits(
        _hierarchy: ParisHierarchy,
        forest: PlateauForest,
        graph: csr_matrix,
    ) -> np.ndarray:
        seen_graph_shapes.append(graph.shape)
        return np.zeros(forest.representatives.size, dtype=np.float64)

    monkeypatch.setattr(modularity, "modularity_split_gains", veto_all_splits)
    result = store.run_paris_clustering(min_cluster_size=2)

    assert seen_graph_shapes == [(14, 14)]
    assert result.n_clusters == 1
    assert result.labels.tolist() == [1] * 14


def test_auto_cut_defaults_minimum_cluster_size_to_graph_k_plus_one() -> None:
    store = _Store(_block_graph())

    result = store.run_paris_clustering()

    assert result.min_cluster_size == 4


def test_incomplete_adaptive_cache_is_recomputed() -> None:
    store = _Store(_block_graph())
    first = store.run_paris_clustering(min_cluster_size=2)
    first_cut = _column_artifact(store, "RNA_paris_cluster")
    del store.zw[artifact_path(first_cut)]["labels"]

    second = store.run_paris_clustering(min_cluster_size=2)
    second_cut = _column_artifact(store, "RNA_paris_cluster")

    assert np.array_equal(second.labels, first.labels)
    assert second_cut != first_cut
    assert "labels" in store.zw[artifact_path(second_cut)]
    assert second.hierarchy_generation_id == first.hierarchy_generation_id
    assert store.load_graph_calls == 2


def test_force_recalculation_retains_only_referenced_generations() -> None:
    store = _Store(_block_graph())
    first = store.run_paris_clustering(label="first", min_cluster_size=2)
    second = store.run_paris_clustering(
        label="second",
        min_cluster_size=2,
        force_recalc=True,
    )
    assert first.hierarchy_generation_id != second.hierarchy_generation_id
    hierarchy_ids = {
        ref.artifact_id
        for ref in list_artifacts(
            store.zw,
            scope="assay",
            assay="RNA",
            kind="cluster_hierarchy",
        )
    }
    assert hierarchy_ids == {
        first.hierarchy_generation_id,
        second.hierarchy_generation_id,
    }

    store.run_paris_clustering(n_clusters=2, label="first")
    assert {
        ref.artifact_id
        for ref in list_artifacts(
            store.zw,
            scope="assay",
            assay="RNA",
            kind="cluster_hierarchy",
        )
    } == hierarchy_ids
    assert "paris_hierarchy" not in store.zw["RNA/graph"]


def test_stale_generation_pointer_recomputes_without_crashing() -> None:
    store = _Store(_block_graph())
    first = store.run_paris_clustering(n_clusters=2)
    # A released-layout pointer must not influence new artifact reuse.
    store.zw["RNA/graph"].attrs[LATEST_PARIS_GENERATION] = "missing-generation"
    assert generation_location("RNA/graph", "missing-generation") not in store.zw

    second = store.run_paris_clustering(n_clusters=2)

    assert second.hierarchy_generation_id == first.hierarchy_generation_id
    assert store.zw["RNA/graph"].attrs[LATEST_PARIS_GENERATION] == "missing-generation"
    assert np.array_equal(second.labels, first.labels)


def test_adaptive_cache_collection_prunes_only_unusable_configurations() -> None:
    store = _Store(_block_graph())
    first = store.run_paris_clustering(
        min_cluster_size=2,
        label="small",
    )
    second = store.run_paris_clustering(
        min_cluster_size=3,
        label="large",
    )

    cuts = list_artifacts(
        store.zw,
        scope="assay",
        assay="RNA",
        kind="cluster_cut",
    )
    assert len(cuts) == 2
    assert first.hierarchy_generation_id == second.hierarchy_generation_id
    assert "adaptive_clustering" not in store.zw["RNA/graph"]


def test_stale_legacy_dendrogram_warns_once_and_is_not_reused() -> None:
    store = _Store(_block_graph())
    graph_group = store.zw["RNA/graph"]
    graph_group.create_array("dendrogram", data=np.zeros((13, 4)))
    graph_group.attrs["latest_dendrogram"] = "RNA/graph/dendrogram"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = store.run_paris_clustering(n_clusters=2)
    assert caught == []
    second = store.run_paris_clustering(n_clusters=2)

    assert first.hierarchy_generation_id == second.hierarchy_generation_id
    assert (
        str(store.zw["RNA/graph"].attrs["latest_dendrogram"]) == "RNA/graph/dendrogram"
    )
    assert (
        len(
            list_artifacts(
                store.zw,
                scope="assay",
                assay="RNA",
                kind="dendrogram",
            )
        )
        == 1
    )


def test_integrated_graph_is_resolved_without_standard_graph_lookup() -> None:
    graph = _block_graph()
    store = _Store(graph)
    store.add_integrated_graph("joint", graph)
    store.raise_on_latest_graph = True

    result = store.run_paris_clustering(
        integrated_graph="joint",
        min_cluster_size=2,
    )

    assert result.label_key == "joint_paris_cluster"
    assert LATEST_PARIS_GENERATION not in store.zw["integratedGraphs/joint"].attrs
    assert (
        len(
            list_artifacts(
                store.zw,
                scope="datastore",
                kind="cluster_hierarchy",
            )
        )
        == 1
    )
    assert store.latest_graph_calls == 0


def test_memory_preflight_fails_before_loading_edges() -> None:
    store = _Store(_block_graph())
    store.resources = ResourceBudget(memoryBytes=1, workers=1)
    with pytest.raises(MemoryError, match="resource budget"):
        store.run_paris_clustering()

    assert store.load_graph_calls == 0
    assert LATEST_PARIS_GENERATION not in store.zw["RNA/graph"].attrs


def test_cached_hierarchy_adaptive_preflight_fails_before_graph_load() -> None:
    store = _Store(_block_graph())
    fixed = store.run_paris_clustering(n_clusters=2)
    assert store.load_graph_calls == 1
    generation_id = fixed.hierarchy_generation_id
    assert generation_id is not None
    hierarchy_ref = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="cluster_hierarchy",
        artifact_id=generation_id,
    )
    cached_estimate = estimate_hierarchy_group_peak_bytes(
        store.zw[artifact_path(hierarchy_ref)],
        "adaptive",
    )
    graph_group = store.zw["RNA/graph"]
    edges = graph_group["edges"]
    weights = graph_group["weights"]
    graph_estimate = estimate_paris_adaptive_cut_peak_bytes(
        14,
        int(edges.shape[0]),
        np.dtype(edges.dtype).itemsize,
        np.dtype(weights.dtype).itemsize,
    )
    assert graph_estimate > cached_estimate

    store.resources = ResourceBudget(
        memoryBytes=(cached_estimate + graph_estimate) // 2,
        workers=1,
    )
    with pytest.raises(MemoryError, match="^Paris adaptive cut"):
        store.run_paris_clustering(
            n_clusters="auto",
            min_cluster_size=2,
            label="guarded",
        )

    assert store.load_graph_calls == 1
    assert "RNA/graph/adaptive_clustering/guarded" not in store.zw


def test_cached_fixed_and_adaptive_cuts_preflight_hierarchy_loading() -> None:
    store = _Store(_block_graph())
    store.run_paris_clustering(n_clusters=2, label="fixed")
    store.run_paris_clustering(
        n_clusters="auto",
        min_cluster_size=2,
        label="adaptive",
    )
    writes_before = store.cells.writes.copy()
    loads_before = store.load_graph_calls

    store.resources = ResourceBudget(memoryBytes=1, workers=1)
    with pytest.raises(MemoryError, match="Cached Paris fixed cut"):
        store.run_paris_clustering(n_clusters=3, label="other_fixed")
    with pytest.raises(MemoryError, match="Cached Paris adaptive cut"):
        store.run_paris_clustering(
            n_clusters="auto",
            min_cluster_size=2,
            label="adaptive",
        )

    assert store.cells.writes == writes_before
    assert store.load_graph_calls == loads_before


def test_memory_estimate_accounts_for_parallel_contraction_tables() -> None:
    n_cells = 500_000
    serial = estimate_paris_peak_bytes(
        n_cells,
        7_500_000,
        4,
        4,
        nthreads=1,
    )
    parallel = estimate_paris_peak_bytes(
        n_cells,
        7_500_000,
        4,
        4,
        nthreads=8,
    )

    assert parallel > serial
    assert parallel - serial >= n_cells * (8 - 1) * 8
    adaptive_cut_peak = estimate_paris_adaptive_cut_peak_bytes(
        n_cells,
        7_500_000,
        4,
        4,
    )
    assert 0 < adaptive_cut_peak < serial


def test_memory_estimate_switches_to_int64_before_doubled_counts_overflow() -> None:
    int32_max = int(np.iinfo(np.int32).max)
    edge_threshold = int32_max // 2
    low_edge_estimate = estimate_paris_adaptive_cut_peak_bytes(
        1_000,
        edge_threshold,
        4,
        4,
    )
    high_edge_estimate = estimate_paris_adaptive_cut_peak_bytes(
        1_000,
        edge_threshold + 1,
        4,
        4,
    )
    leaf_threshold = (int32_max + 1) // 2
    low_leaf_estimate = estimate_paris_adaptive_cut_peak_bytes(
        leaf_threshold,
        1,
        4,
        4,
    )
    high_leaf_estimate = estimate_paris_adaptive_cut_peak_bytes(
        leaf_threshold + 1,
        1,
        4,
        4,
    )

    assert high_edge_estimate > low_edge_estimate * 1.2
    assert high_leaf_estimate > low_leaf_estimate * 1.2


def test_interrupted_replacement_keeps_previous_generation_authoritative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scarf.datastore._operations.paris_persistence as cache

    store = _Store(_block_graph())
    first = store.run_paris_clustering(min_cluster_size=2)
    previous_generation = first.hierarchy_generation_id
    previous_cut = _column_artifact(store, "RNA_paris_cluster")

    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(cache, "write_hierarchy_group", fail_write)
    with pytest.raises(OSError, match="interrupted"):
        store.run_paris_clustering(force_recalc=True)

    assert _column_artifact(store, "RNA_paris_cluster") == previous_cut
    hierarchy_ref = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="cluster_hierarchy",
        artifact_id=previous_generation,
    )
    assert inspect_artifact(store.zw, hierarchy_ref).complete
    assert store.cells.writes == ["RNA_paris_cluster"]


def test_graph_rebuild_clears_generation_pointer() -> None:
    graph = _block_graph()
    store = _Store(graph)
    first = store.run_paris_clustering(min_cluster_size=2)
    store._write_graph("RNA/graph", graph * 2, k=3)
    assert LATEST_PARIS_GENERATION not in store.zw["RNA/graph"].attrs

    second = store.run_paris_clustering(min_cluster_size=2)
    assert second.hierarchy_generation_id != first.hierarchy_generation_id


def test_cluster_tree_cache_tracks_generation_and_cluster_identity() -> None:
    store = _Store(_block_graph())
    store.run_paris_clustering(
        min_cluster_size=2,
        label="first",
    )
    prepared = store._prepare_cluster_tree(cluster_key="RNA_first")
    cached = store._prepare_cluster_tree(cluster_key="RNA_first")
    assert prepared["coalesced_location"] == cached["coalesced_location"]

    values = store.cells.data["RNA_first"].copy()
    values[values == 1] = -1
    values[values == 2] = 1
    values[values == -1] = 2
    store.cells.data["RNA_first"] = values
    changed = store._prepare_cluster_tree(cluster_key="RNA_first")
    assert changed["coalesced_location"] == prepared["coalesced_location"]

    store.run_paris_clustering(
        n_clusters=3,
        label="second",
    )
    second_tree = store._prepare_cluster_tree(cluster_key="RNA_second")
    assert second_tree["coalesced_location"] != prepared["coalesced_location"]


def test_topacedo_uses_the_generation_recorded_for_adaptive_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _Store(_block_graph())
    result = store.run_paris_clustering(min_cluster_size=2)
    captured: dict[str, np.ndarray] = {}

    class Sampler:
        def __init__(
            self,
            _graph: csr_matrix,
            _clusters: np.ndarray,
            dendrogram: np.ndarray,
            *_args: object,
        ) -> None:
            captured["dendrogram"] = dendrogram
            n_cells = len(_clusters)
            self.densities = np.ones(n_cells)
            self.meanSnn = np.ones(n_cells)
            self.seeds = np.asarray([0], dtype=np.int64)
            self.rand_state = int(_args[-1])

        def run(self) -> tuple[np.ndarray, list[tuple[int, int]]]:
            if self.rand_state == 99:
                return np.asarray([0], dtype=np.int64), []
            if self.rand_state == 100:
                return np.asarray([0], dtype=np.int64), [(0, len(self.densities))]
            return np.asarray([0, 1], dtype=np.int64), [(0, 1)]

    monkeypatch.setitem(
        sys.modules,
        "topacedo",
        SimpleNamespace(TopacedoSampler=Sampler),
    )
    monkeypatch.setattr(
        "scarf.datastore._operations.clustering.validate_legacy_graph_selection",
        lambda *_args, **_kwargs: None,
    )
    sampling_ref = store.run_topacedo_sampler(cluster_key=result.label_key)

    output_columns = [
        "RNA_sketched",
        "RNA_cell_density",
        "RNA_snn_value",
        "RNA_sketch_seeds",
    ]
    output_refs = {
        ArtifactRef.from_dict(store.zw[f"cellData/{column}"].attrs["source_artifact"])
        for column in output_columns
    }
    assert output_refs == {sampling_ref}
    assert sampling_ref.kind == "sampling"
    sampling_group = store.zw[artifact_path(sampling_ref)]
    assert set(sampling_group.array_keys()) == {
        "sampled",
        "density",
        "mean_snn",
        "seeds",
        "edges",
    }
    np.testing.assert_array_equal(sampling_group["edges"][:], [[0, 1]])
    monkeypatch.setitem(sys.modules, "topacedo", None)
    assert store.run_topacedo_sampler(cluster_key=result.label_key) == sampling_ref
    monkeypatch.setitem(
        sys.modules,
        "topacedo",
        SimpleNamespace(TopacedoSampler=Sampler),
    )
    empty_ref = store.run_topacedo_sampler(
        cluster_key=result.label_key,
        rand_state=99,
    )
    empty_group = store.zw[artifact_path(empty_ref)]
    assert empty_group["edges"].shape == (0, 2)
    assert empty_group["edges"].chunks == (1, 2)
    with pytest.raises(ValueError, match="edge endpoints"):
        store.run_topacedo_sampler(
            cluster_key=result.label_key,
            rand_state=100,
        )
    assert captured["dendrogram"].shape == (13, 4)
    dendrogram_ref = list_artifacts(
        store.zw,
        scope="assay",
        assay="RNA",
        kind="dendrogram",
    )[0]
    dendrogram_inputs = inspect_artifact(store.zw, dendrogram_ref).inputs
    assert dendrogram_inputs is not None
    assert (
        ArtifactRef.from_dict(dendrogram_inputs["cluster_hierarchy"]).artifact_id
        == result.hierarchy_generation_id
    )


@pytest.mark.parametrize(
    ("arguments", "error_type", "message"),
    [
        ({"n_clusters": True}, TypeError, "integer or 'auto'"),
        ({"n_clusters": np.bool_(True)}, TypeError, "integer or 'auto'"),
        ({"n_clusters": "unknown"}, ValueError, "integer or 'auto'"),
        ({"n_clusters": 0}, ValueError, "positive"),
        ({"n_clusters": 15}, ValueError, "graph size"),
        (
            {"n_clusters": 2, "min_cluster_size": 3},
            ValueError,
            "only valid",
        ),
        (
            {"n_clusters": "auto", "min_cluster_size": 1},
            ValueError,
            "at least 2",
        ),
    ],
)
def test_run_paris_clustering_validates_cut_arguments(
    arguments: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    store = _Store(_block_graph())
    with pytest.raises(error_type, match=message):
        store.run_paris_clustering(**arguments)


def test_run_paris_clustering_accepts_numpy_integer_cut_parameters() -> None:
    store = _Store(_block_graph())

    fixed = store.run_paris_clustering(n_clusters=np.int64(2), label="fixed")
    adaptive = store.run_paris_clustering(
        n_clusters="auto",
        min_cluster_size=np.int32(2),
        label="adaptive",
    )

    assert fixed.n_clusters == 2
    assert adaptive.labels.shape == (14,)
