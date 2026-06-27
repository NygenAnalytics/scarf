import types

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.storage.zarr_store import (
    get_storage_profile,
    is_remote_zarr_location,
    make_store,
    open_store,
    set_storage_profile,
)
from scarf.utils import load_zarr


@pytest.fixture(autouse=True)
def reset_profile():
    set_storage_profile(None)
    yield
    set_storage_profile(None)


def test_is_remote_zarr_location():
    assert is_remote_zarr_location("s3://bucket/path") is True
    assert is_remote_zarr_location("gs://bucket/path") is True
    assert is_remote_zarr_location("/tmp/foo.zarr") is False
    assert is_remote_zarr_location("file:///tmp/foo.zarr") is False


def test_make_store_local_path_returns_str(tmp_path):
    path = str(tmp_path / "ds.zarr")
    store = make_store(path)
    assert store == path


def test_make_store_memory_store():
    mem = MemoryStore()
    store = make_store(mem)
    assert store is mem


def test_open_store_memory(tmp_path):
    path = str(tmp_path / "ds.zarr")
    root = open_store(path, mode="w")
    root.create_group("g")
    loaded = open_store(path, mode="r")
    assert "g" in loaded


def test_load_zarr_memory_store():
    mem = MemoryStore()
    root = zarr.open_group(store=mem, mode="w")
    root.create_group("assay")
    loaded = load_zarr(mem, mode="r")
    assert "assay" in loaded


def test_make_store_remote_requires_obstore(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name in ("obstore", "obstore.store"):
            raise ImportError("no obstore")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    with pytest.raises(ImportError, match="obstore"):
        make_store("s3://bucket/path")


def test_make_store_remote_auto_cloud_profile(monkeypatch):
    class FakeObstore:
        pass

    class FakeObjectStore:
        def __init__(self, store, read_only=False):
            self.store = store
            self.read_only = read_only

    fake_mod = types.ModuleType("obstore.store")
    fake_mod.from_url = lambda url, **kwargs: FakeObstore()
    monkeypatch.setitem(__import__("sys").modules, "obstore.store", fake_mod)
    monkeypatch.setattr(
        "zarr.storage.ObjectStore",
        FakeObjectStore,
    )
    store = make_store("s3://bucket/path")
    assert isinstance(store, FakeObjectStore)
    assert get_storage_profile() == "cloud"


def test_explicit_profile_not_overridden_by_remote(monkeypatch):
    class FakeObstore:
        pass

    class FakeObjectStore:
        def __init__(self, store, read_only=False):
            self.store = store

    fake_mod = types.ModuleType("obstore.store")
    fake_mod.from_url = lambda url, **kwargs: FakeObstore()
    monkeypatch.setitem(__import__("sys").modules, "obstore.store", fake_mod)
    monkeypatch.setattr("zarr.storage.ObjectStore", FakeObjectStore)

    set_storage_profile("fast_local")
    make_store("s3://bucket/path")
    assert get_storage_profile() == "fast_local"


def test_normed_array_spec_cloud_sharding():
    from scarf.storage.zarr_store import (
        normed_array_spec,
        set_storage_profile,
    )

    set_storage_profile("cloud")
    spec = normed_array_spec(1_000_000, 2000)
    assert spec.dtype == "float32"
    assert spec.shards is not None
    assert spec.shards[1] == 2000


@pytest.mark.parametrize("n_feats", [500, 2000, 3000, 5000, 8192, 30_000])
def test_normed_array_spec_cloud_creates_array(tmp_path, n_feats):
    from scarf.storage.zarr_store import (
        create_numeric_array,
        normed_array_spec,
        set_storage_profile,
    )

    set_storage_profile("cloud")
    spec = normed_array_spec(1_000_000, n_feats)
    root = zarr.open_group(str(tmp_path / f"normed_{n_feats}.zarr"), mode="w")
    create_numeric_array(root, "data", spec)
    assert root["data"].shape == (1_000_000, n_feats)
    if spec.shards is not None:
        assert spec.shards[0] % spec.chunks[0] == 0
        assert spec.shards[1] % spec.chunks[1] == 0


def test_compute_zarr_layout_scales_with_cells():
    from scarf.storage.zarr_store import compute_zarr_layout

    small = compute_zarr_layout(1_000, 2_000, remote=True)
    large = compute_zarr_layout(1_000_000, 50_000, remote=True)
    assert large.countChunks[0] >= small.countChunks[0]
    assert large.countShards[0] >= small.countShards[0]
    assert large.asyncConcurrency >= small.asyncConcurrency
    assert large.streamTargetBytes >= small.streamTargetBytes


def test_marker_batch_size_aligns_and_respects_budget():
    from scarf.storage.zarr_store import compute_zarr_layout, marker_batch_size

    layout = compute_zarr_layout(10_000, 50_000, remote=True)
    col_chunk = layout.countChunks[1]

    bs = marker_batch_size(10_000, 50_000, layout)
    # Chunk-aligned when the budget allows at least one column chunk.
    assert bs % col_chunk == 0
    assert bs <= 50_000
    # Stays within the streaming memory budget (float32).
    assert bs * 10_000 * 4 <= layout.streamTargetBytes

    # Many cells force a smaller, memory-bounded batch.
    huge = compute_zarr_layout(5_000_000, 50_000, remote=True)
    bs_huge = marker_batch_size(5_000_000, 50_000, huge)
    assert bs_huge >= 1
    assert bs_huge < marker_batch_size(10_000, 50_000, huge)

    # Never exceeds the available feature count.
    assert marker_batch_size(1_000, 32, layout) <= 32


def test_marker_batch_size_snaps_to_chunk_divisor_when_below_chunk():
    from scarf.storage.zarr_store import compute_zarr_layout, marker_batch_size

    # Many cells force a sub-chunk batch; it must divide the column chunk so
    # batches never straddle a chunk boundary.
    layout = compute_zarr_layout(5_000_000, 50_000, remote=True)
    col_chunk = layout.countChunks[1]
    bs = marker_batch_size(5_000_000, 50_000, layout)
    assert bs >= 1
    assert bs <= col_chunk
    assert col_chunk % bs == 0


def test_streaming_block_size(tmp_path):
    from scarf.storage.zarr_store import streaming_block_size, set_storage_profile

    set_storage_profile("fast_local")
    root = zarr.open_group(str(tmp_path / "stream.zarr"), mode="w")
    arr = root.create_array(
        "x", shape=(10_000, 500), chunks=(256, 500), dtype="float32"
    )
    block = streaming_block_size(arr)
    assert block >= 256
    assert block <= 10_000


def test_streaming_block_size_aligns_and_shrinks_under_budget(tmp_path):
    from scarf.storage.budget import ResourceBudget, set_resource_budget
    from scarf.storage.zarr_store import set_storage_profile, streaming_block_size

    set_storage_profile("fast_local")
    root = zarr.open_group(str(tmp_path / "stream.zarr"), mode="w")
    arr = root.create_array(
        "x", shape=(100_000, 5_000), chunks=(256, 5_000), dtype="float32"
    )
    try:
        set_resource_budget(ResourceBudget(memoryBytes=8 * 1024**3, workers=1))
        big = streaming_block_size(arr)
        assert big % 256 == 0

        set_resource_budget(ResourceBudget(memoryBytes=8 * 1024**3, workers=16))
        small = streaming_block_size(arr)
        assert small % 256 == 0
        assert small <= big
    finally:
        set_resource_budget(None)


def test_compute_zarr_layout_caps_under_small_budget():
    from scarf.storage.budget import ResourceBudget
    from scarf.storage.zarr_store import compute_zarr_layout

    generous = ResourceBudget(memoryBytes=64 * 1024**3, workers=2)
    tight = ResourceBudget(memoryBytes=512 * 1024**2, workers=8)

    big = compute_zarr_layout(1_000_000, 50_000, remote=True, budget=generous)
    small = compute_zarr_layout(1_000_000, 50_000, remote=True, budget=tight)

    assert small.streamTargetBytes <= tight.perWorkerBytes
    assert small.streamTargetBytes <= big.streamTargetBytes
    assert small.asyncConcurrency <= big.asyncConcurrency
    assert small.prefetchDepth >= 1


def test_compute_zarr_layout_shard_chunk_alignment():
    from scarf.storage.zarr_store import compute_zarr_layout

    layout = compute_zarr_layout(100_000, 50_000, remote=True)
    row_chunk, col_chunk = layout.countChunks
    shard_rows, shard_cols = layout.countShards
    assert shard_cols % col_chunk == 0
    assert shard_rows % row_chunk == 0
    assert col_chunk <= 512


def test_ann_index_round_trip(tmp_path):
    import hnswlib

    from scarf.storage.zarr_store import (
        has_ann_index,
        load_ann_index,
        save_ann_index,
    )

    root = zarr.open_group(str(tmp_path / "ds.zarr"), mode="w")
    g = root.create_group("ann")
    dim = 8
    n = 200
    idx = hnswlib.Index(space="l2", dim=dim)
    idx.init_index(max_elements=n, ef_construction=50, M=16)
    data = np.random.rand(n, dim).astype(np.float32)
    idx.add_items(data)
    save_ann_index(g, idx)
    assert has_ann_index(g)
    loaded = load_ann_index(g, "l2", dim)
    q = data[:5]
    i1, d1 = idx.knn_query(q, k=3)
    i2, d2 = loaded.knn_query(q, k=3)
    np.testing.assert_array_equal(i1, i2)
    np.testing.assert_allclose(d1, d2, rtol=1e-5)
