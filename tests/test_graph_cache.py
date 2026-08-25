import zarr
from zarr.storage import MemoryStore

from scarf.datastore.graph_datastore import GraphDataStore


def _memory_group():
    return zarr.open_group(store=MemoryStore(), mode="w")


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
