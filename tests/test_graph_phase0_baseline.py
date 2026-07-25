"""Graph-first Phase 0 baselines for master-format and v1_prep layouts.

Pins path conventions, latest-pointer lookup, load_graph no-mutation, and
workspace-relative resolution before later maintenance phases change metadata.
"""

import hashlib
from typing import Any

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.datastore.graph_datastore import GraphDataStore


class _MemoryGraphStore(GraphDataStore):
    @property
    def assay_names(self) -> list[str]:
        return self._assay_names


@pytest.fixture(autouse=True)
def _isolate_path_parser_tests_from_selection_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "scarf.datastore._operations.graph.validate_legacy_graph_selection",
        lambda *_args, **_kwargs: None,
    )


def _memory_graph_store(
    *,
    workspace: str | None = None,
    assay_names: list[str] | None = None,
) -> _MemoryGraphStore:
    store = _MemoryGraphStore.__new__(_MemoryGraphStore)
    store.z = zarr.open_group(store=MemoryStore(), mode="w")
    store.workspace = workspace
    store.zarr_mode = "r+"
    store._defaultAssay = "RNA"
    store._assay_names = assay_names or ["RNA"]
    store._integratedGraphsLoc = "integratedGraphs"
    store._cachedMagicOperator = None
    store._cachedMagicOperatorLoc = None
    store.nthreads = 1
    return store


def _add_edges_weights(group: zarr.Group, n_cells: int = 3) -> None:
    group.attrs["n_cells"] = n_cells
    group.attrs["n_neighbors"] = 2
    group.create_array(
        "edges",
        data=np.array(
            [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]],
            dtype=np.uint64,
        ),
    )
    group.create_array(
        "weights",
        data=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
    )


def _add_knn_indices(knn_group: zarr.Group, n_cells: int = 3, k: int = 2) -> None:
    knn_group.create_array(
        "indices",
        data=np.array([[1, 2], [0, 2], [0, 1]], dtype=np.uint64)[:n_cells, :k],
    )


def _install_master_format_assay_graph(
    store: _MemoryGraphStore,
    *,
    assay: str = "RNA",
    cell_key: str = "I",
    feat_key: str = "hvgs",
) -> dict[str, str]:
    """Install a master-era parameter-encoded graph chain without v1_prep attrs."""
    normed_loc = f"{assay}/normed__{cell_key}__{feat_key}"
    reduction_loc = f"{normed_loc}/reduction__pca__10__{cell_key}"
    ann_loc = f"{reduction_loc}/ann__l2__50__50__16__1"
    knn_loc = f"{ann_loc}/knn__11"
    graph_loc = f"{knn_loc}/graph__1.0__1.5"
    kmeans_loc = f"{reduction_loc}/kmeans__100__1"

    normed = store.zw.create_group(normed_loc)
    reduction = store.zw.create_group(reduction_loc)
    ann = store.zw.create_group(ann_loc)
    knn = store.zw.create_group(knn_loc)
    graph = store.zw.create_group(graph_loc)
    store.zw.create_group(kmeans_loc)
    _add_knn_indices(knn)
    _add_edges_weights(graph)

    # Master Harmony: unsuffixed ANN path + attrs/child, no contract-hash suffix.
    ann.attrs["isHarmonized"] = True
    reduction.create_array(
        "harmonizedData",
        data=np.zeros((3, 10), dtype=np.float32),
    )

    normed.attrs["latest_reduction"] = reduction_loc
    reduction.attrs["latest_ann"] = ann_loc
    reduction.attrs["latest_kmeans"] = kmeans_loc
    ann.attrs["latest_knn"] = knn_loc
    knn.attrs["latest_graph"] = graph_loc
    return {
        "normed": normed_loc,
        "reduction": reduction_loc,
        "ann": ann_loc,
        "knn": knn_loc,
        "graph": graph_loc,
        "kmeans": kmeans_loc,
    }


def _group_attr_snapshot(group: zarr.Group) -> dict[str, Any]:
    return {key: group.attrs[key] for key in group.attrs}


def _collect_paths(root: zarr.Group, prefix: str = "") -> list[str]:
    paths: list[str] = []
    for key in root:
        path = f"{prefix}/{key}" if prefix else key
        paths.append(path)
        child = root[key]
        if hasattr(child, "keys"):
            paths.extend(_collect_paths(child, path))
    return paths


def _store_keys_digest(root: zarr.Group) -> str:
    digest = hashlib.sha256()
    for path in sorted(_collect_paths(root)):
        digest.update(path.encode())
        node = root[path]
        if hasattr(node, "attrs"):
            for key in sorted(node.attrs):
                digest.update(key.encode())
                digest.update(repr(node.attrs[key]).encode())
        if hasattr(node, "shape"):
            digest.update(repr(tuple(node.shape)).encode())
            digest.update(str(node.dtype).encode())
            digest.update(np.asarray(node[:]).tobytes())
    return digest.hexdigest()


def test_normalized_group_path_convention_is_stable() -> None:
    assay, cell_key, feat_key = "RNA", "I", "hvgs"
    expected = f"{assay}/normed__{cell_key}__{feat_key}"
    assert expected == "RNA/normed__I__hvgs"
    store = _memory_graph_store()
    assert store.get_normalized_group_path(assay, cell_key, feat_key) == expected
    assert store.get_normalized_group_path(assay, cell_key, feat_key) == (
        "RNA/normed__I__hvgs"
    )


def test_get_normalized_group_path_before_and_after_graph_exists() -> None:
    store = _memory_graph_store()
    before = store.get_normalized_group_path("RNA", "I", "hvgs")
    assert before == "RNA/normed__I__hvgs"
    assert before not in store.zw

    locs = _install_master_format_assay_graph(store)
    after = store.get_normalized_group_path("RNA", "I", "hvgs")
    assert after == before
    assert after == locs["normed"]
    assert after in store.zw


def test_get_latest_graph_loc_matches_master_chain() -> None:
    store = _memory_graph_store()
    locs = _install_master_format_assay_graph(store)

    assert store.get_latest_graph_loc("RNA", "I", "hvgs") == locs["graph"]


def test_lookup_and_load_graph_do_not_mutate_master_format_store() -> None:
    store = _memory_graph_store()
    locs = _install_master_format_assay_graph(store)
    before = _store_keys_digest(store.z)
    before_knn_attrs = _group_attr_snapshot(store.zw[locs["knn"]])

    graph_loc = store.get_latest_graph_loc("RNA", "I", "hvgs")
    graph = store.load_graph(
        from_assay="RNA",
        cell_key="I",
        feat_key="hvgs",
        graph_loc=graph_loc,
    )
    assert graph.shape == (3, 3)
    assert graph.nnz == 6

    after = _store_keys_digest(store.z)
    after_knn_attrs = _group_attr_snapshot(store.zw[locs["knn"]])
    assert after == before
    assert after_knn_attrs == before_knn_attrs


def test_master_format_chain_has_no_v1_prep_suffixes_or_schema_attrs() -> None:
    store = _memory_graph_store()
    locs = _install_master_format_assay_graph(store)

    assert "__unscaled" not in locs["ann"]
    assert "__harmony_" not in locs["ann"]
    assert "ann_idx_bytes" not in store.zw[locs["ann"]]
    assert "schemaVersion" not in store.zw[locs["ann"]].attrs
    assert "schema_version" not in store.zw[locs["normed"]].attrs
    assert store.zw[locs["ann"]].attrs["isHarmonized"] is True
    assert "harmonizedData" in store.zw[locs["reduction"]]


def test_v1_prep_suffix_conventions_are_distinct_from_master() -> None:
    from scarf.graph.encoded_paths import parse_assay_graph_paths

    store = _memory_graph_store()
    assay = "RNA"
    cell_key = "I"
    feat_key = "hvgs"
    normed_loc = f"{assay}/normed__{cell_key}__{feat_key}"
    reduction_loc = f"{normed_loc}/reduction__pca__10__{cell_key}"
    base_ann = f"{reduction_loc}/ann__l2__50__50__16__1"
    unscaled_ann = f"{base_ann}__unscaled"
    harmony_ann = f"{base_ann}__harmony_0123456789abcdef"
    knn_loc = f"{harmony_ann}/knn__11"
    graph_loc = f"{knn_loc}/graph__1.0__1.5"

    store.zw.create_group(normed_loc).attrs["latest_reduction"] = reduction_loc
    reduction = store.zw.create_group(reduction_loc)
    reduction.attrs["latest_ann"] = harmony_ann
    store.zw.create_group(unscaled_ann)
    ann = store.zw.create_group(harmony_ann)
    ann.attrs["latest_knn"] = knn_loc
    ann.attrs["featureScaling"] = False
    ann.attrs["isHarmonized"] = True
    knn = store.zw.create_group(knn_loc)
    knn.attrs["latest_graph"] = graph_loc
    graph = store.zw.create_group(graph_loc)
    _add_knn_indices(knn)
    _add_edges_weights(graph)

    assert store.get_latest_graph_loc(assay, cell_key, feat_key) == graph_loc
    assert "__unscaled" in unscaled_ann
    assert "__harmony_" in harmony_ann

    parsed = parse_assay_graph_paths(graph_loc)
    assert parsed.paths.cell_graph_group_path == graph_loc
    assert parsed.harmony_contract_hash == "0123456789abcdef"
    assert parsed.feat_scaling is True

    unscaled_harmony = f"{unscaled_ann}__harmony_abcdef0123456789"
    unscaled_parsed = parse_assay_graph_paths(
        f"{unscaled_harmony}/knn__11/graph__1.0__1.5"
    )
    assert unscaled_parsed.feat_scaling is False
    assert unscaled_parsed.harmony_contract_hash == "abcdef0123456789"


def test_parse_master_and_v1_prep_assay_graph_paths() -> None:
    from scarf.graph.encoded_paths import (
        make_cell_graph_group_path,
        make_nearest_neighbors_group_path,
        make_neighbor_index_group_path,
        make_normalized_group_path,
        make_reduction_group_path,
        parse_assay_graph_paths,
    )
    from scarf.graph.paths import AssayGraphPaths

    master = (
        "RNA/normed__I__hvgs/reduction__pca__10__I/"
        "ann__l2__50__50__16__1/knn__11/graph__1.0__1.5"
    )
    stored = parse_assay_graph_paths(master)
    assert stored.from_assay == "RNA"
    assert stored.cell_key == "I"
    assert stored.feat_key == "hvgs"
    assert stored.reduction_method == "pca"
    assert stored.dims == 10
    assert stored.k == 11
    assert stored.feat_scaling is True
    assert stored.harmony_contract_hash is None
    assert stored.local_connectivity == 1.0
    assert stored.bandwidth == 1.5

    normalized = make_normalized_group_path("RNA", "I", "hvgs")
    reduction = make_reduction_group_path(normalized, "pca", 10, "I")
    neighbor_index = make_neighbor_index_group_path(
        reduction,
        "l2",
        50,
        50,
        16,
        1,
        feat_scaling=False,
        harmony_contract_hash="deadbeefcafebabe",
    )
    nearest_neighbors = make_nearest_neighbors_group_path(neighbor_index, 11)
    cell_graph = make_cell_graph_group_path(nearest_neighbors, 1.0, 1.5)
    built = AssayGraphPaths(
        normalized_group_path=normalized,
        reduction_group_path=reduction,
        neighbor_index_group_path=neighbor_index,
        nearest_neighbors_group_path=nearest_neighbors,
        cell_graph_group_path=cell_graph,
        kmeans_initialization_group_path=None,
    )
    assert built.neighbor_index_group_path.endswith(
        "ann__l2__50__50__16__1__unscaled__harmony_deadbeefcafebabe"
    )
    reparsed = parse_assay_graph_paths(built.cell_graph_group_path)
    assert reparsed.feat_scaling is False
    assert reparsed.harmony_contract_hash == "deadbeefcafebabe"


def test_workspace_relative_latest_graph_lookup() -> None:
    store = _memory_graph_store(workspace="analysis")
    store.z.create_group("analysis")
    locs = _install_master_format_assay_graph(store)

    assert store.get_latest_graph_loc("RNA", "I", "hvgs") == locs["graph"]
    graph = store.load_graph(
        from_assay="RNA",
        cell_key="I",
        feat_key="hvgs",
        graph_loc=locs["graph"],
    )
    assert graph.shape == (3, 3)


def test_get_ini_embed_needs_only_reduction_and_kmeans() -> None:
    # Initial embedding must not require an ANN index, KNN graph, or cell graph:
    # only the reduction and its k-means initialization.
    store = _memory_graph_store()
    assay, cell_key, feat_key = "RNA", "I", "hvgs"
    normed_loc = f"{assay}/normed__{cell_key}__{feat_key}"
    reduction_loc = f"{normed_loc}/reduction__pca__10__{cell_key}"
    kmeans_loc = f"{reduction_loc}/kmeans__4__1"
    store.zw.create_group(normed_loc).attrs["latest_reduction"] = reduction_loc
    store.zw.create_group(reduction_loc).attrs["latest_kmeans"] = kmeans_loc
    kmeans = store.zw.create_group(kmeans_loc)
    rng = np.random.default_rng(0)
    kmeans.create_array("cluster_centers", data=rng.normal(size=(4, 10)))
    kmeans.create_array(
        "cluster_labels", data=np.array([0, 1, 2, 3, 0, 1], dtype=np.uint32)
    )

    # No latest_ann, latest_knn, or latest_graph pointers exist.
    assert "latest_ann" not in store.zw[reduction_loc].attrs
    assert f"{reduction_loc}/ann__l2__50__50__16__1" not in store.zw

    ini = store._get_ini_embed(assay, cell_key, feat_key, 2)
    assert ini.shape == (6, 2)


def test_partial_knn_lookup_never_fabricates_a_graph_path() -> None:
    from scarf.graph.encoded_paths import (
        lookup_latest_assay_graph,
        lookup_latest_nearest_neighbor_paths,
        nearest_neighbor_paths_from_loc,
    )
    from scarf.graph.paths import AssayNearestNeighborPaths

    store = _memory_graph_store()
    normed_loc = "RNA/normed__I__hvgs"
    reduction_loc = f"{normed_loc}/reduction__pca__10__I"
    ann_loc = f"{reduction_loc}/ann__l2__50__50__16__1"
    knn_loc = f"{ann_loc}/knn__11"
    store.zw.create_group(normed_loc).attrs["latest_reduction"] = reduction_loc
    store.zw.create_group(reduction_loc).attrs["latest_ann"] = ann_loc
    store.zw.create_group(ann_loc).attrs["latest_knn"] = knn_loc
    store.zw.create_group(knn_loc)

    latest = lookup_latest_nearest_neighbor_paths(store.zw, "RNA", "I", "hvgs")
    explicit = nearest_neighbor_paths_from_loc(knn_loc)
    assert isinstance(latest, AssayNearestNeighborPaths)
    assert latest == explicit
    assert latest.nearest_neighbors_group_path == knn_loc
    assert not hasattr(latest, "cell_graph_group_path")

    with pytest.raises(KeyError, match="latest_graph"):
        lookup_latest_assay_graph(store.zw, "RNA", "I", "hvgs")


def test_integrated_graph_load_by_explicit_path_is_stable() -> None:
    store = _memory_graph_store()
    graph_loc = "integratedGraphs/joint"
    group = store.zw.create_group(graph_loc)
    _add_edges_weights(group)
    before = _store_keys_digest(store.z)

    graph = store.load_graph(
        from_assay="RNA",
        cell_key="I",
        feat_key="I",
        graph_loc=graph_loc,
    )
    assert graph.shape == (3, 3)
    assert _store_keys_digest(store.z) == before
