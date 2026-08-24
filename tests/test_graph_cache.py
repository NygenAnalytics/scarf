import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.datastore.graph_datastore import GraphDataStore


def _memory_group():
    return zarr.open_group(store=MemoryStore(), mode="w")


def test_get_latest_graph_loc_is_public(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "scarf.datastore._operations.graph.validate_legacy_graph_selection",
        lambda *_args, **_kwargs: None,
    )
    root = _memory_group()
    normed = root.create_group("RNA/normed__I__hvgs")
    reduction = root.create_group(f"{normed.path}/reduction__pca__10__I")
    ann = root.create_group(f"{reduction.path}/ann__l2__50__50__16__1")
    knn = root.create_group(f"{ann.path}/knn__11")
    graph = root.create_group(f"{knn.path}/graph__1.0__1.5")
    normed.attrs["latest_reduction"] = reduction.path
    reduction.attrs["latest_ann"] = ann.path
    ann.attrs["latest_knn"] = knn.path
    knn.attrs["latest_graph"] = graph.path

    store = GraphDataStore.__new__(GraphDataStore)
    store.z = root
    store.workspace = None

    assert store.get_normalized_group_path("RNA", "I", "hvgs") == "RNA/normed__I__hvgs"
    assert store.get_latest_graph_loc("RNA", "I", "hvgs") == graph.path

    del knn.attrs["latest_graph"]
    with pytest.raises(KeyError):
        store.get_latest_graph_loc("RNA", "I", "hvgs")


def test_resolve_local_cache_plan(tmp_path):
    local_root = _memory_group()
    enabled, base, remove = GraphDataStore._resolve_local_cache_plan(
        "/tmp/local.zarr", local_root, "auto"
    )
    assert enabled is False

    enabled, base, remove = GraphDataStore._resolve_local_cache_plan(
        "s3://bucket/path", local_root, False
    )
    assert enabled is False

    enabled, base, remove = GraphDataStore._resolve_local_cache_plan(
        "s3://bucket/path", local_root, str(tmp_path / "cache")
    )
    assert enabled is True
    assert base == str(tmp_path / "cache")
    assert remove is False
