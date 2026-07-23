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
from scarf.clustering.paris import fit_paris_hierarchy
from scarf.datastore._operations.clustering import _ClusteringOperationsMixin
from scarf.datastore._operations.presentation import _PresentationOperationsMixin
from scarf.datastore._operations.paris_persistence import (
    ADAPTIVE_SCHEMA_VERSION,
    LATEST_PARIS_GENERATION,
    adaptive_config_digest,
    estimate_cached_paris_peak_bytes,
    estimate_paris_adaptive_cut_peak_bytes,
    estimate_paris_peak_bytes,
    generation_location,
    load_adaptive_result,
    load_hierarchy_generation,
)
from scarf.storage.budget import (
    ResourceBudget,
    _get_resource_budget_override,
    set_resource_budget,
)


class _Cells:
    def __init__(self, active: np.ndarray) -> None:
        self.active = active
        self.data: dict[str, np.ndarray] = {"I": active.copy()}
        self.writes: list[str] = []

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
        self.cells = _Cells(active)
        self.nthreads = 2
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

    def _get_latest_keys(
        self,
        from_assay: str | None,
        cell_key: str | None,
        feat_key: str | None,
    ) -> tuple[str, str, str]:
        return from_assay or "RNA", cell_key or "I", feat_key or "hvgs"

    def _get_latest_graph_loc(
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


def test_auto_cut_persists_typed_hierarchy_and_reuses_diagnostics() -> None:
    store = _Store(_block_graph(), extra_cells=2)
    first = store.run_paris_clustering(min_cluster_size=2)
    graph_group = store.zw["RNA/graph"]
    generation_id = str(graph_group.attrs[LATEST_PARIS_GENERATION])
    hierarchy, forest = load_hierarchy_generation(
        store.zw,
        "RNA/graph",
        generation_id,
    )

    assert hierarchy.n_leaves == 14
    assert forest.n_leaves == 14
    assert "latest_dendrogram" not in graph_group.attrs
    assert first.n_clusters == 2
    assert first.labels[:6].tolist() == [1] * 6
    assert first.labels[6:].tolist() == [2] * 8
    assert store.cells.data["RNA_paris_cluster"][-2:].tolist() == [-1, -1]
    assert set(store.cells.data["RNA_paris_cluster"][:-2]) == {1, 2}
    assert store.cells.writes == ["RNA_paris_cluster"]

    second = store.run_paris_clustering(min_cluster_size=2)
    assert str(graph_group.attrs[LATEST_PARIS_GENERATION]) == generation_id
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
    hierarchy, _forest = load_hierarchy_generation(
        store.zw,
        "RNA/graph",
        generation_id,
    )
    expected = fit_paris_hierarchy(graph, n_threads=1)

    assert np.array_equal(hierarchy.children, expected.children)
    assert np.array_equal(hierarchy.heights, expected.heights)
    dendrogram_loc = str(store.zw["RNA/graph"].attrs["latest_dendrogram"])
    assert dendrogram_loc == (
        f"{generation_location('RNA/graph', generation_id)}/dendrogram"
    )
    assert dendrogram_loc in store.zw
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
    hierarchy, _forest = load_hierarchy_generation(
        store.zw,
        "RNA/graph",
        generation_id,
    )
    assert hierarchy.synthetic_joins.sum() == 1
    dendrogram_loc = str(store.zw["RNA/graph"].attrs["latest_dendrogram"])
    compatibility = np.asarray(store.zw[dendrogram_loc][:])
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


def test_stale_adaptive_cache_schema_is_recomputed() -> None:
    store = _Store(_block_graph())
    first = store.run_paris_clustering(min_cluster_size=2)
    generation_id = first.hierarchy_generation_id
    assert generation_id is not None
    hierarchy, _forest = load_hierarchy_generation(
        store.zw,
        "RNA/graph",
        generation_id,
    )
    digest = adaptive_config_digest(generation_id, 2)
    location = f"RNA/graph/adaptive_clustering/paris_cluster/{digest}"
    config = store.zw[location]

    assert (
        load_adaptive_result(
            store.zw,
            "RNA/graph",
            "paris_cluster",
            digest,
            hierarchy,
        )
        is not None
    )
    config.attrs["schema_version"] = ADAPTIVE_SCHEMA_VERSION - 1
    assert (
        load_adaptive_result(
            store.zw,
            "RNA/graph",
            "paris_cluster",
            digest,
            hierarchy,
        )
        is None
    )

    second = store.run_paris_clustering(min_cluster_size=2)

    assert np.array_equal(second.labels, first.labels)
    assert store.zw[location].attrs["schema_version"] == ADAPTIVE_SCHEMA_VERSION
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
    hierarchy_group = store.zw["RNA/graph/paris_hierarchy/v2"]
    assert set(hierarchy_group.group_keys()) == {
        first.hierarchy_generation_id,
        second.hierarchy_generation_id,
    }

    store.run_paris_clustering(n_clusters=2, label="first")
    assert set(hierarchy_group.group_keys()) == {second.hierarchy_generation_id}
    assert "RNA/graph/adaptive_clustering/first" not in store.zw
    assert "RNA/graph/adaptive_clustering/second" in store.zw


def test_adaptive_cache_collection_prunes_only_unusable_configurations() -> None:
    store = _Store(_block_graph())
    result = store.run_paris_clustering(min_cluster_size=2)
    generation_id = result.hierarchy_generation_id
    assert generation_id is not None
    label_location = "RNA/graph/adaptive_clustering/paris_cluster"
    reusable_digest = adaptive_config_digest(generation_id, 2)

    stale_digest = adaptive_config_digest(generation_id, 3)
    incomplete_digest = adaptive_config_digest(generation_id, 4)
    orphan_digest = adaptive_config_digest("missing-generation", 2)
    for digest, config_generation, min_size, schema, complete in (
        (
            stale_digest,
            generation_id,
            3,
            ADAPTIVE_SCHEMA_VERSION - 1,
            True,
        ),
        (
            incomplete_digest,
            generation_id,
            4,
            ADAPTIVE_SCHEMA_VERSION,
            False,
        ),
        (
            orphan_digest,
            "missing-generation",
            2,
            ADAPTIVE_SCHEMA_VERSION,
            True,
        ),
    ):
        config = store.zw.create_group(f"{label_location}/{digest}")
        config.attrs.update(
            {
                "complete": complete,
                "schema_version": schema,
                "hierarchy_generation_id": config_generation,
                "min_cluster_size": min_size,
                "final_label_key": "RNA_paris_cluster",
            }
        )

    store.run_paris_clustering(n_clusters=2)

    assert f"{label_location}/{reusable_digest}" in store.zw
    assert f"{label_location}/{stale_digest}" not in store.zw
    assert f"{label_location}/{incomplete_digest}" not in store.zw
    assert f"{label_location}/{orphan_digest}" not in store.zw
    assert "active_digest" not in store.zw[label_location].attrs


def test_stale_legacy_dendrogram_warns_once_and_is_not_reused() -> None:
    store = _Store(_block_graph())
    graph_group = store.zw["RNA/graph"]
    graph_group.create_array("dendrogram", data=np.zeros((13, 4)))
    graph_group.attrs["latest_dendrogram"] = "RNA/graph/dendrogram"

    with pytest.warns(UserWarning, match="predates canonical additive graphs"):
        first = store.run_paris_clustering(n_clusters=2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        second = store.run_paris_clustering(n_clusters=2)
    assert caught == []

    assert first.hierarchy_generation_id == second.hierarchy_generation_id
    assert (
        str(store.zw["RNA/graph"].attrs["latest_dendrogram"]) != "RNA/graph/dendrogram"
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
    assert LATEST_PARIS_GENERATION in store.zw["integratedGraphs/joint"].attrs
    assert store.latest_graph_calls == 0


def test_memory_preflight_fails_before_loading_edges() -> None:
    store = _Store(_block_graph())
    previous = _get_resource_budget_override()
    try:
        set_resource_budget(ResourceBudget(memoryBytes=1, workers=1, workingCopies=1))
        with pytest.raises(MemoryError, match="resource budget"):
            store.run_paris_clustering()
    finally:
        set_resource_budget(previous)

    assert store.load_graph_calls == 0
    assert LATEST_PARIS_GENERATION not in store.zw["RNA/graph"].attrs


def test_cached_hierarchy_adaptive_preflight_fails_before_graph_load() -> None:
    store = _Store(_block_graph())
    fixed = store.run_paris_clustering(n_clusters=2)
    assert store.load_graph_calls == 1
    generation_id = fixed.hierarchy_generation_id
    assert generation_id is not None
    cached_estimate = estimate_cached_paris_peak_bytes(
        store.zw,
        "RNA/graph",
        generation_id,
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

    previous = _get_resource_budget_override()
    try:
        set_resource_budget(
            ResourceBudget(
                memoryBytes=(cached_estimate + graph_estimate) // 2,
                workers=1,
                workingCopies=1,
            )
        )
        with pytest.raises(MemoryError, match="^Paris adaptive cut"):
            store.run_paris_clustering(
                n_clusters="auto",
                min_cluster_size=2,
                label="guarded",
            )
    finally:
        set_resource_budget(previous)

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

    previous = _get_resource_budget_override()
    try:
        set_resource_budget(ResourceBudget(memoryBytes=1, workers=1, workingCopies=1))
        with pytest.raises(MemoryError, match="Cached Paris fixed cut"):
            store.run_paris_clustering(n_clusters=3, label="other_fixed")
        with pytest.raises(MemoryError, match="Cached Paris adaptive cut"):
            store.run_paris_clustering(
                n_clusters="auto",
                min_cluster_size=2,
                label="adaptive",
            )
    finally:
        set_resource_budget(previous)

    assert store.cells.writes == writes_before
    assert store.load_graph_calls == loads_before


def test_memory_estimate_accounts_for_parallel_contraction_tables() -> None:
    n_cells = 500_000
    serial = estimate_paris_peak_bytes(
        n_cells,
        7_500_000,
        4,
        4,
        n_threads=1,
    )
    parallel = estimate_paris_peak_bytes(
        n_cells,
        7_500_000,
        4,
        4,
        n_threads=8,
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

    def fail_write(
        root: zarr.Group,
        graph_loc: str,
        *_args: object,
        **_kwargs: object,
    ) -> tuple[str, str]:
        location = f"{graph_loc}/paris_hierarchy/v2/incomplete"
        group = root.create_group(location, overwrite=True)
        group.attrs["complete"] = False
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(cache, "write_hierarchy_generation", fail_write)
    with pytest.raises(OSError, match="interrupted"):
        store.run_paris_clustering(force_recalc=True)

    assert (
        str(store.zw["RNA/graph"].attrs[LATEST_PARIS_GENERATION]) == previous_generation
    )
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
    first = store.run_paris_clustering(
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
    assert changed["coalesced_location"] != prepared["coalesced_location"]

    store.cells.data["RNA_first"] = first.labels.copy()
    second = store.run_paris_clustering(
        min_cluster_size=2,
        label="second",
        force_recalc=True,
    )
    old_generation = store._prepare_cluster_tree(cluster_key="RNA_first")
    assert first.hierarchy_generation_id in old_generation["coalesced_location"]
    assert second.hierarchy_generation_id not in old_generation["coalesced_location"]


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

        def run(self) -> tuple[np.ndarray, list[tuple[int, int]]]:
            return np.asarray([0, 1], dtype=np.int64), [(0, 1)]

    monkeypatch.setitem(
        sys.modules,
        "topacedo",
        SimpleNamespace(TopacedoSampler=Sampler),
    )
    edges = store.run_topacedo_sampler(
        cluster_key=result.label_key,
        return_edges=True,
    )

    assert edges == [(0, 1)]
    assert captured["dendrogram"].shape == (13, 4)
    assert result.hierarchy_generation_id in str(
        store.zw["RNA/graph"].attrs["latest_dendrogram"]
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


def test_run_clustering_is_a_deprecated_none_returning_shim() -> None:
    store = _Store(_block_graph())
    with pytest.warns(FutureWarning, match="run_clustering is deprecated"):
        returned = store.run_clustering(n_clusters=np.int64(2))

    assert returned is None
    assert store.cells.writes == ["RNA_cluster"]

    with (
        pytest.warns(FutureWarning),
        pytest.raises(
            ValueError,
            match="balanced-cut mode has been removed",
        ),
    ):
        store.run_clustering(
            n_clusters=2,
            balanced_cut=True,
            max_size=100,
            min_size=10,
        )
    with (
        pytest.warns(FutureWarning),
        pytest.raises(
            ValueError,
            match="n_clusters=None",
        ),
    ):
        store.run_clustering()


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


def test_deprecated_symmetry_flags_warn_and_are_ignored() -> None:
    store = _Store(_block_graph())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        store.run_clustering(
            n_clusters=2,
            symmetric_graph=True,
            graph_upper_only=True,
        )

    assert [item.category for item in caught] == [FutureWarning, FutureWarning]
    assert store.load_graph_calls == 1
