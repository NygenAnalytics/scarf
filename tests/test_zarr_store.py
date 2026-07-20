import types

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.storage.arrays import create_numeric_array
from scarf.storage.copy import (
    copy_zarr_array,
    copy_zarr_group_tree,
    open_or_create_staged_normed_array,
)
from scarf.storage.layout import normed_array_spec
from scarf.storage.profiles import (
    get_storage_profile,
    is_local_zarr_path,
    is_remote_zarr_location,
    set_storage_profile,
)
from scarf.storage.sharding import (
    accumulate_sparse_to_shards,
    finalize_sharded_counts,
    write_dense_from_row_batches,
)
from scarf.storage.stores import (
    is_remote_datastore,
    make_store,
    open_store,
)
from scarf.storage.types import array_metadata_shards
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


def test_is_local_zarr_path():
    assert is_local_zarr_path("/tmp/foo.zarr") is True
    assert is_local_zarr_path("s3://bucket/path") is False
    assert is_local_zarr_path("gs://bucket/path") is False
    assert is_local_zarr_path(MemoryStore()) is False


def test_load_zarr_forwards_storage_options_to_make_store(monkeypatch):
    captured = {}

    def fake_make_store(location, storage_options=None, read_only=False):
        captured["location"] = location
        captured["storage_options"] = storage_options
        captured["read_only"] = read_only
        return MemoryStore()

    monkeypatch.setattr("scarf.storage.stores.make_store", fake_make_store)
    monkeypatch.setattr(
        "scarf.storage.stores.configure_zarr_io_for_profile", lambda: None
    )
    monkeypatch.setattr("zarr.open_group", lambda **kwargs: object())
    load_zarr(
        "s3://bucket/path",
        mode="r",
        storage_options={"secret_access_key": "secret"},
    )
    assert captured["storage_options"] == {"secret_access_key": "secret"}
    assert captured["read_only"] is True


def _memory_group():
    return zarr.open_group(store=MemoryStore(), mode="w")


def test_is_remote_datastore():
    local_root = _memory_group()
    assert is_remote_datastore("/tmp/foo.zarr", local_root) is False
    assert is_remote_datastore("s3://bucket/path", local_root) is True
    # Missing/empty location must inspect the group store, not treat "" as local.
    assert is_remote_datastore("", local_root) is False
    assert is_remote_datastore(None, local_root) is False


def test_copy_zarr_array_round_trip(tmp_path):
    src_root = zarr.open_group(str(tmp_path / "src.zarr"), mode="w")
    spec = normed_array_spec(64, 8, profile="fast_local")
    src = create_numeric_array(src_root, "data", spec)
    expected = np.random.rand(64, 8).astype(np.float32)
    src[:] = expected

    dst_root = zarr.open_group(str(tmp_path / "dst.zarr"), mode="w")
    dst = create_numeric_array(dst_root, "data", spec)
    copy_zarr_array(src, dst, block_rows=16)
    np.testing.assert_allclose(dst[:], expected, rtol=1e-6)


def test_write_dense_from_row_batches_flushes_at_shard_boundaries():
    root = _memory_group()
    dst = root.create_array(
        "counts",
        shape=(7, 3),
        chunks=(2, 3),
        shards=(4, 3),
        dtype=np.uint16,
        fill_value=0,
    )
    expected = np.arange(21, dtype=np.int64).reshape(7, 3)
    writes = []
    write_dtypes = []

    class RecordingArray:
        def __init__(self, array):
            self._array = array
            self.metadata = array.metadata
            self.shape = array.shape

        def __setitem__(self, selection, value):
            row_slice = selection[0]
            writes.append((row_slice.start, row_slice.stop))
            write_dtypes.append(value.dtype)
            self._array[selection] = value

    rows_written = write_dense_from_row_batches(
        RecordingArray(dst),
        iter(
            [
                expected[:1],
                expected[1:5],
                np.empty((0, 3), dtype=np.int64),
                expected[5:],
            ]
        ),
        dtype=np.uint16,
    )

    assert rows_written == 7
    assert writes == [(0, 4), (4, 7)]
    assert write_dtypes == [np.dtype(np.uint16), np.dtype(np.uint16)]
    np.testing.assert_array_equal(dst[:], expected.astype(np.uint16))


@pytest.mark.parametrize(
    ("workspace", "counts_group_path"),
    [(None, "RNA"), ("workspace", "matrices/RNA")],
)
def test_finalize_sharded_counts_repacks_and_cleans_up(
    monkeypatch, workspace, counts_group_path
):
    from scarf.storage.budget import ResourceBudget

    root = _memory_group()
    counts_group = root.create_group(counts_group_path)
    source = counts_group.create_array(
        "counts",
        shape=(5, 3),
        chunks=(2, 3),
        dtype=np.uint32,
        fill_value=0,
    )
    expected = np.arange(15, dtype=np.uint32).reshape(5, 3)
    source[:] = expected
    assert array_metadata_shards(source) is None

    budget = ResourceBudget(memoryBytes=24, workers=1, workingCopies=1)
    monkeypatch.setattr("scarf.storage.layout.get_resource_budget", lambda: budget)

    result = finalize_sharded_counts(
        root,
        "RNA",
        workspace=workspace,
        profile="fast_local",
    )

    np.testing.assert_array_equal(result[:], expected)
    assert result.chunks == (2, 1)
    assert array_metadata_shards(result) == (2, 3)
    refreshed_group = root[counts_group_path]
    assert "counts__sharded_tmp" not in refreshed_group
    assert refreshed_group.attrs["scarf:zarr_spec"] == {
        "profile": "fast_local",
        "chunks": [2, 1],
        "shards": [2, 3],
        "zarr_format": 3,
    }


def test_accumulate_sparse_to_shards_preserves_offsets_across_zero_runs():
    from scipy.sparse import coo_matrix

    root = _memory_group()
    dst = root.create_array(
        "counts",
        shape=(36, 3),
        chunks=(2, 3),
        dtype=np.uint32,
        fill_value=0,
    )

    zero_batch = coo_matrix((1, 3), dtype=np.uint32)
    batches = [
        coo_matrix(
            (np.array([11], dtype=np.uint32), ([0], [0])),
            shape=(1, 3),
        ),
        *[zero_batch for _ in range(32)],
        coo_matrix(
            (np.array([22, 33], dtype=np.uint32), ([0, 1], [1, 2])),
            shape=(2, 3),
        ),
        zero_batch,
    ]

    rows_written = accumulate_sparse_to_shards(dst, iter(batches), shard_rows=2)

    expected = np.zeros((36, 3), dtype=np.uint32)
    expected[0, 0] = 11
    expected[33, 1] = 22
    expected[34, 2] = 33
    assert rows_written == 36
    np.testing.assert_array_equal(dst[:], expected)


def test_copy_zarr_group_tree(tmp_path):
    from scarf.writers import create_zarr_obj_array

    src_root = zarr.open_group(str(tmp_path / "src.zarr"), mode="w")
    slot = src_root.create_group("I__cluster")
    cluster = slot.create_group("0")
    create_zarr_obj_array(cluster, "score", [1.0, 2.0, 3.0], dtype="float64")

    dst_root = zarr.open_group(str(tmp_path / "dst.zarr"), mode="w")
    dst_slot = dst_root.create_group("I__cluster")
    copy_zarr_group_tree(slot, dst_slot)
    np.testing.assert_array_equal(dst_slot["0"]["score"][:], [1.0, 2.0, 3.0])


def test_open_or_create_staged_normed_array_reuses_shape(tmp_path):
    src_root = zarr.open_group(str(tmp_path / "src.zarr"), mode="w")
    spec = normed_array_spec(32, 4, profile="cloud")
    src = create_numeric_array(src_root, "data", spec)
    src[:] = np.ones((32, 4), dtype=np.float32)

    cache_path = str(tmp_path / "cache" / "abc123" / "normed.zarr")
    staged = open_or_create_staged_normed_array(cache_path, src)
    copy_zarr_array(src, staged, block_rows=8)
    staged.attrs["staged_subset_hash"] = "abc123"
    staged.attrs["staged_complete"] = True

    reopened = open_or_create_staged_normed_array(cache_path, src)
    assert reopened.shape == src.shape
    np.testing.assert_allclose(reopened[:], np.ones((32, 4), dtype=np.float32))


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


def test_normed_array_spec_plain_chunks():
    from scarf.storage.budget import ResourceBudget
    from scarf.storage.layout import normed_array_spec
    from scarf.storage.profiles import set_storage_profile

    set_storage_profile("cloud")
    budget = ResourceBudget(memoryBytes=8 * 1024**3, workers=4, workingCopies=4)
    spec = normed_array_spec(1_000_000, 2000, budget=budget)
    assert spec.dtype == "float32"
    assert spec.shards is None
    assert spec.chunks[1] == 2000
    assert spec.chunks[0] >= 1


@pytest.mark.parametrize("n_cells", [1_000_000, 2_500_000, 5_000_000, 10_000_000])
def test_normed_array_spec_respects_codec_limit(n_cells):
    from scarf.storage.budget import ResourceBudget
    from scarf.storage.layout import _CODEC_MAX_BYTES, normed_array_spec

    budget = ResourceBudget(
        memoryBytes=112 * 1024**3,
        workers=16,
        workingCopies=4,
    )
    spec = normed_array_spec(
        n_cells,
        2000,
        profile="cloud",
        budget=budget,
    )
    assert spec.shards is None
    assert spec.chunks[0] * spec.chunks[1] * 4 <= _CODEC_MAX_BYTES


@pytest.mark.parametrize("n_feats", [500, 2000, 3000, 5000, 8192, 30_000])
def test_normed_array_spec_creates_array(tmp_path, n_feats):
    from scarf.storage.arrays import create_numeric_array
    from scarf.storage.layout import normed_array_spec
    from scarf.storage.profiles import set_storage_profile

    set_storage_profile("cloud")
    spec = normed_array_spec(1_000_000, n_feats)
    root = zarr.open_group(str(tmp_path / f"normed_{n_feats}.zarr"), mode="w")
    create_numeric_array(root, "data", spec)
    assert root["data"].shape == (1_000_000, n_feats)
    assert spec.shards is None


def test_memory_first_layout_worked_example():
    from scarf.storage.budget import ResourceBudget
    from scarf.storage.layout import _CODEC_MAX_BYTES, matrix_layout

    budget = ResourceBudget(memoryBytes=8 * 1024**3, workers=8, workingCopies=4)
    chunks, shards = matrix_layout(1_000_000, 50_000, budget=budget, itemsize=4)
    assert shards is not None
    row_shard, shard_cols = shards
    feature_chunk = chunks[1]
    work = (8 * 1024**3) // 4
    assert feature_chunk == work // (1_000_000 * 4)
    assert shard_cols % feature_chunk == 0
    assert shard_cols >= 50_000
    assert row_shard * shard_cols * 4 <= _CODEC_MAX_BYTES


def test_ceil_pad_awkward_feature_count():
    from scarf.storage.budget import ResourceBudget
    from scarf.storage.layout import matrix_layout

    budget = ResourceBudget(memoryBytes=8 * 1024**3, workers=1, workingCopies=4)
    chunks, shards = matrix_layout(1_000_000, 36_601, budget=budget, itemsize=4)
    assert shards is not None
    feature_chunk = chunks[1]
    shard_cols = shards[1]
    assert feature_chunk >= 1
    assert shard_cols % feature_chunk == 0
    assert shard_cols >= 36_601


def test_float64_halves_row_shard():
    from scarf.storage.budget import ResourceBudget
    from scarf.storage.layout import matrix_layout

    budget = ResourceBudget(memoryBytes=8 * 1024**3, workers=1, workingCopies=4)
    u32, _ = matrix_layout(100_000, 20_000, budget=budget, itemsize=4)
    f64, _ = matrix_layout(100_000, 20_000, budget=budget, itemsize=8)
    assert f64[0] <= u32[0]


def test_matrix_layout_scales_with_cells():
    from scarf.storage.budget import ResourceBudget
    from scarf.storage.layout import matrix_layout

    budget = ResourceBudget(memoryBytes=64 * 1024**3, workers=2, workingCopies=4)
    small_chunks, small_shards = matrix_layout(1_000, 2_000, budget=budget, itemsize=4)
    large_chunks, large_shards = matrix_layout(
        1_000_000, 50_000, budget=budget, itemsize=4
    )
    assert small_shards is not None and large_shards is not None
    assert large_chunks[0] >= small_chunks[0]
    assert large_shards[0] >= small_shards[0]


def test_matrix_layout_shard_chunk_alignment():
    from scarf.storage.budget import ResourceBudget
    from scarf.storage.layout import _CODEC_MAX_BYTES, matrix_layout

    budget = ResourceBudget(memoryBytes=8 * 1024**3, workers=4, workingCopies=4)
    chunks, shards = matrix_layout(100_000, 50_000, budget=budget, itemsize=4)
    assert shards is not None
    row_chunk, col_chunk = chunks
    shard_rows, shard_cols = shards
    assert shard_cols % col_chunk == 0
    assert shard_rows % row_chunk == 0
    assert shard_cols >= 50_000
    assert shard_rows * shard_cols * 4 <= _CODEC_MAX_BYTES


def test_matrix_layout_respects_codec_limit():
    from scarf.storage.budget import get_resource_budget
    from scarf.storage.layout import _CODEC_MAX_BYTES, matrix_layout

    budget = get_resource_budget()
    for n_cells, n_feats in [
        (10_000, 89_796),
        (1_000_000, 50_000),
        (1_000_000, 36_601),
    ]:
        chunks, shards = matrix_layout(n_cells, n_feats, budget=budget, itemsize=4)
        assert shards is not None
        row_shard, shard_cols = shards
        feature_chunk = chunks[1]
        assert row_shard * shard_cols * 4 <= _CODEC_MAX_BYTES
        assert shard_cols % feature_chunk == 0
        assert row_shard % chunks[0] == 0


def test_matrix_layout_target_chunk_bytes_clamps_features():
    from scarf.storage.budget import ResourceBudget
    from scarf.storage.layout import matrix_layout

    budget = ResourceBudget(memoryBytes=48 * 1024**3, workers=4, workingCopies=4)
    chunks, shards = matrix_layout(
        100_000,
        45_525,
        budget=budget,
        itemsize=4,
        targetChunkBytes=256 * 1024 * 1024,
        minFeatureChunk=500,
        maxFeatureChunk=10_000,
    )
    assert shards is not None
    assert 500 <= chunks[1] <= 10_000
    assert chunks[1] <= 45_525
    assert shards[1] % chunks[1] == 0
    assert chunks[0] * chunks[1] * 4 <= 256 * 1024 * 1024 + chunks[1] * 4


def test_count_array_spec_passes_target_chunk_bytes():
    from scarf.storage.budget import ResourceBudget, set_resource_budget
    from scarf.storage.layout import count_array_spec

    budget = ResourceBudget(memoryBytes=24 * 1024**3, workers=4, workingCopies=4)
    try:
        set_resource_budget(budget)
        capped = count_array_spec(
            100_000,
            45_525,
            dtype="uint32",
            remote=True,
            targetChunkBytes=64 * 1024 * 1024,
            minFeatureChunk=500,
            maxFeatureChunk=10_000,
        )
        baseline = count_array_spec(100_000, 45_525, dtype="uint32", remote=True)
    finally:
        set_resource_budget(None)
    assert capped.chunks[1] <= 10_000
    assert capped.chunks[1] < baseline.chunks[1] or baseline.chunks[1] <= 10_000


def test_count_array_spec_applies_cloud_default_target_chunk_bytes():
    from scarf.storage.budget import ResourceBudget, set_resource_budget
    from scarf.storage.layout import (
        DEFAULT_CLOUD_TARGET_CHUNK_BYTES,
        count_array_spec,
        matrix_layout,
    )

    budget = ResourceBudget(memoryBytes=24 * 1024**3, workers=4, workingCopies=4)
    try:
        set_resource_budget(budget)
        cloud = count_array_spec(100_000, 45_525, dtype="uint32", remote=True)
        local = count_array_spec(100_000, 45_525, dtype="uint32", remote=False)
        expected, _ = matrix_layout(
            100_000,
            45_525,
            budget=budget,
            itemsize=4,
            targetChunkBytes=DEFAULT_CLOUD_TARGET_CHUNK_BYTES,
            minFeatureChunk=500,
            maxFeatureChunk=10_000,
        )
        memory_first, _ = matrix_layout(
            100_000,
            45_525,
            budget=budget,
            itemsize=4,
        )
    finally:
        set_resource_budget(None)
    assert cloud.chunks == expected
    assert local.chunks == memory_first
    assert DEFAULT_CLOUD_TARGET_CHUNK_BYTES == 128 * 1024 * 1024


def test_large_atac_count_array_accepts_sparse_writes(tmp_path):
    import numpy as np

    from scarf.storage.arrays import create_numeric_array
    from scarf.storage.layout import count_array_spec

    n_cells, n_feats = 10_000, 89_796
    spec = count_array_spec(n_cells, n_feats, dtype="uint32")
    root = zarr.open_group(str(tmp_path / "atac.zarr"), mode="w")
    arr = create_numeric_array(root, "counts", spec)
    rows = np.arange(1000, dtype=np.int64)
    cols = np.arange(1000, dtype=np.int64)
    arr.set_coordinate_selection((rows, cols), np.ones(1000, dtype=np.uint32))


def test_v2_group_skips_shards(tmp_path):
    from scarf.storage.arrays import create_numeric_array
    from scarf.storage.layout import count_array_spec

    root = zarr.open_group(str(tmp_path / "v2.zarr"), mode="w", zarr_format=2)
    spec = count_array_spec(100, 50, dtype="uint32")
    arr = create_numeric_array(root, "counts", spec)
    assert array_metadata_shards(arr) is None


def test_ann_index_round_trip(tmp_path):
    import hnswlib

    from scarf.storage.ann_index import (
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
