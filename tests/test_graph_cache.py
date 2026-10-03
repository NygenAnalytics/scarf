import os

import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.datastore.graph_datastore import GraphDataStore


def _memory_group():
    return zarr.open_group(store=MemoryStore(), mode="w")


def test_resolve_local_cache_plan(tmp_path):
    local_root = _memory_group()
    resolve = GraphDataStore._resolve_local_cache_plan

    # A local store is read in place whatever the policy.
    for policy in ("auto", True, str(tmp_path / "unused")):
        assert resolve("/tmp/local.zarr", local_root, policy) == (False, None, False)
    assert not (tmp_path / "unused").exists()
    # A remote store is read in place only when staging is disabled.
    assert resolve("s3://bucket/path", local_root, False) == (False, None, False)

    explicit = str(tmp_path / "cache")
    assert resolve("s3://bucket/path", local_root, explicit) == (True, explicit, False)
    assert os.path.isdir(explicit)

    for policy in ("auto", True):
        enabled, base, remove = resolve("s3://bucket/path", local_root, policy)
        try:
            assert (enabled, remove) == (True, True)
            assert base is not None and os.path.isdir(base)
            assert os.path.basename(base).startswith("scarf_local_cache_")
        finally:
            if base is not None:
                os.rmdir(base)


@pytest.mark.parametrize("policy", [1, None, b"auto"])
def test_resolve_local_cache_plan_rejects_other_policies(policy):
    with pytest.raises(TypeError, match="local_cache must be 'auto', True, False"):
        GraphDataStore._resolve_local_cache_plan(
            "s3://bucket/path", _memory_group(), policy
        )
