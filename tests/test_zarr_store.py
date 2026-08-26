import types

import numpy as np
import pytest
import zarr
from scipy.sparse import csr_matrix
from zarr.storage import MemoryStore

from scarf.matrix.chunked import ChunkedArray
from scarf.storage.arrays import create_numeric_array
from scarf.storage.budget import ResourceBudget
from scarf.storage.copy import (
    copy_zarr_array,
    copy_zarr_group_tree,
    create_or_open_staged_normed_array,
)
from scarf.storage.layout import (
    ZarrArraySpec,
    _CODEC_MAX_BYTES,
    DEFAULT_TARGET_CHUNK_BYTES,
    DEFAULT_TARGET_SHARD_BYTES,
    bounded_row_sharded_array_spec,
    count_array_spec,
    get_compressors,
    normed_array_spec,
    row_sharded_array_spec,
)
from scarf.storage.profiles import (
    is_local_zarr_path,
    is_remote_zarr_location,
    resolve_storage_profile,
)
from scarf.storage.count_matrix import (
    persist_count_matrix_plan,
    plan_count_matrix_pair,
)
from scarf.storage.schema import create_zarr_count_assay
from scarf.storage.sharding import (
    accumulate_sparse_to_shards,
    sparse_producer_peak_bytes,
    write_dense_from_row_batches,
    write_dense_in_shard_rows,
    write_counts_t,
)
from scarf.storage.stores import (
    is_remote_datastore,
    make_store,
    open_store,
)
from scarf.storage.types import array_metadata_shards
from scarf.utils import load_zarr
from tests.store_probes import RecordingStore


def _planned_counts(group: zarr.Group, values: np.ndarray, name: str = "counts"):
    plan = plan_count_matrix_pair(values.shape[0], values.shape[1], values.dtype)
    counts = group.create_array(
        name,
        shape=plan.counts.shape,
        chunks=plan.counts.chunks,
        shards=plan.counts.shards,
        dtype=values.dtype,
        fill_value=0,
        overwrite=True,
    )
    if values.size:
        counts[:] = values
    persist_count_matrix_plan(group, plan)
    persist_count_matrix_plan(counts, plan)
    return counts


def test_location_classification_is_pure():
    memory = MemoryStore()
    assert is_remote_zarr_location("s3://bucket/path")
    assert is_remote_zarr_location("gs://bucket/path")
    assert not is_remote_zarr_location("/tmp/data.zarr")
    assert is_local_zarr_path("/tmp/data.zarr")
    assert not is_local_zarr_path("s3://bucket/path")
    assert not is_local_zarr_path(memory)
    assert resolve_storage_profile("s3://bucket/path") == "cloud"
    assert resolve_storage_profile("/tmp/data.zarr") == "fast_local"
    assert resolve_storage_profile("s3://bucket/path", "fast_local") == "fast_local"


def test_load_zarr_forwards_storage_options(monkeypatch):
    captured = {}

    def fake_make_store(location, storage_options=None, read_only=False):
        captured.update(
            location=location,
            storageOptions=storage_options,
            readOnly=read_only,
        )
        store = MemoryStore()
        zarr.open_group(store=store, mode="w")
        return store

    monkeypatch.setattr("scarf.storage.stores.make_store", fake_make_store)
    load_zarr(
        "s3://bucket/path",
        mode="r",
        storage_options={"secret_access_key": "secret"},
    )
    assert captured == {
        "location": "s3://bucket/path",
        "storageOptions": {"secret_access_key": "secret"},
        "readOnly": True,
    }


def test_store_opening_and_remote_detection(tmp_path):
    path = str(tmp_path / "data.zarr")
    assert make_store(path) == path
    memory = MemoryStore()
    assert make_store(memory) is memory

    root = open_store(path, mode="w")
    root.create_group("group")
    assert "group" in open_store(path, mode="r")

    memory_root = zarr.open_group(store=memory, mode="w")
    values = memory_root.create_array("values", shape=(4,), dtype="i4")
    assert not is_remote_datastore(None, memory_root)
    assert not is_remote_datastore("", values)
    assert is_remote_datastore("s3://bucket/path", memory_root)


def test_remote_store_uses_obstore_without_mutating_profile(monkeypatch):
    class FakeObstore:
        pass

    class FakeObjectStore:
        def __init__(self, store, read_only=False):
            self.store = store
            self.read_only = read_only

    fake_module = types.ModuleType("obstore.store")
    fake_module.from_url = lambda url, **kwargs: FakeObstore()
    monkeypatch.setitem(__import__("sys").modules, "obstore.store", fake_module)
    monkeypatch.setattr("zarr.storage.ObjectStore", FakeObjectStore)

    store = make_store("s3://bucket/path", read_only=True)
    assert isinstance(store, FakeObjectStore)
    assert store.read_only is True
    assert resolve_storage_profile("/tmp/data.zarr") == "fast_local"


def test_hugging_face_store_uses_fsspec(monkeypatch):
    sentinel = object()
    captured = {}

    def from_url(url, *, storage_options=None, read_only=False):
        captured.update(
            url=url,
            storageOptions=storage_options,
            readOnly=read_only,
        )
        return sentinel

    monkeypatch.setattr("zarr.storage.FsspecStore.from_url", from_url)

    store = make_store(
        "hf://buckets/Nygen/cytebase/demo/data.zarr",
        storage_options={"token": False},
        read_only=True,
    )

    assert store is sentinel
    assert captured == {
        "url": "hf://buckets/Nygen/cytebase/demo/data.zarr",
        "storageOptions": {"token": False},
        "readOnly": True,
    }


def test_count_plan_uses_paired_rotate_once_geometry():
    from scarf.storage.count_matrix import plan_count_matrix_pair

    spec = count_array_spec(250_000, 45_525, "uint16", profile="cloud")
    expected = plan_count_matrix_pair(250_000, 45_525, "uint16", profile="cloud").counts
    assert spec.shards is not None
    assert spec.shards[1] >= 45_525
    assert spec.chunks == expected.chunks
    assert spec.shards == expected.shards
    assert spec.shards[0] % spec.chunks[0] == 0
    assert spec.shards[1] % spec.chunks[1] == 0


@pytest.mark.parametrize(
    ("n_features", "dtype"),
    [
        (101, "uint8"),
        (997, "float64"),
        (45_524, "uint16"),
    ],
)
def test_count_plan_alignment_and_byte_limits_are_shape_independent(
    n_features,
    dtype,
):
    spec = count_array_spec(10_000, n_features, dtype, profile="cloud")
    assert spec.shards is not None
    assert spec.shards[1] >= n_features
    assert all(
        shard % chunk == 0
        for shard, chunk in zip(spec.shards, spec.chunks, strict=True)
    )


def test_count_plan_does_not_depend_on_process_resource_environment(monkeypatch):
    kwargs = {"profile": "cloud"}
    monkeypatch.setenv("SCARF_MEM_BUDGET", "1G")
    monkeypatch.setenv("SCARF_WORKERS", "1")
    small_machine = count_array_spec(10_000, 997, "uint16", **kwargs)
    monkeypatch.setenv("SCARF_MEM_BUDGET", "64G")
    monkeypatch.setenv("SCARF_WORKERS", "32")
    large_machine = count_array_spec(10_000, 997, "uint16", **kwargs)
    assert small_machine == large_machine


def test_count_plan_respects_codec_limit_and_small_dimensions():
    spec = count_array_spec(
        3,
        7,
        "float64",
        profile="fast_local",
    )
    assert all(
        chunk <= size for chunk, size in zip(spec.chunks, spec.shape, strict=True)
    )
    assert np.prod(spec.chunks) * 8 <= _CODEC_MAX_BYTES


def test_numeric_array_adapts_codecs_to_zarr_format():
    from scarf.storage.count_matrix import CountMatrixPolicy

    spec = count_array_spec(
        8,
        4,
        "uint16",
        profile="cloud",
        policy=CountMatrixPolicy(unitBytes=32, chunkBytes=16),
    )
    for zarr_format in (2, 3):
        root = zarr.open_group(
            store=MemoryStore(),
            mode="w",
            zarr_format=zarr_format,
        )
        array = create_numeric_array(root, "counts", spec)
        values = np.arange(32, dtype=np.uint16).reshape(8, 4)
        array[:] = values
        np.testing.assert_array_equal(array[:], values)
        if zarr_format == 2:
            assert array_metadata_shards(array) is None
        else:
            assert array_metadata_shards(array) == spec.shards


@pytest.mark.parametrize(
    ("shape", "chunks", "shards"),
    [
        # A shard extent that is not a whole number of chunks.
        ((10, 9), (4, 4), (10, 9)),
        # A chunk clamped to a narrower shape leaves the shard misaligned.
        ((10, 3), (4, 8), (8, 8)),
        # A shard smaller than its own chunk.
        ((10, 8), (4, 8), (2, 8)),
    ],
)
def test_numeric_array_rejects_a_shard_that_is_not_whole_chunks(shape, chunks, shards):
    root = zarr.open_group(store=MemoryStore(), mode="w")
    spec = ZarrArraySpec(
        shape=shape,
        chunks=chunks,
        shards=shards,
        dtype="uint16",
        compressors=get_compressors("fast_local"),
        fillValue=0,
    )
    with pytest.raises(ValueError, match="whole chunks"):
        create_numeric_array(root, "counts", spec)


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_assay_records_the_metadata_the_stored_array_actually_has(zarr_format):
    root = zarr.open_group(store=MemoryStore(), mode="w", zarr_format=zarr_format)
    counts = create_zarr_count_assay(
        root,
        "RNA",
        None,
        1_009,
        [f"f{i}" for i in range(997)],
        [f"g{i}" for i in range(997)],
        profile="fast_local",
    )
    recorded = root["RNA"].attrs["scarf:zarr_spec"]
    stored_shards = array_metadata_shards(counts)

    assert recorded["chunks"] == list(counts.chunks)
    assert recorded["shards"] == (
        None if stored_shards is None else list(stored_shards)
    )
    assert recorded["zarr_format"] == zarr_format
    if zarr_format == 2:
        assert stored_shards is None


def test_new_assay_in_zarr_v2_stays_chunk_only():
    root = zarr.open_group(
        store=MemoryStore(),
        mode="w",
        zarr_format=2,
    )
    counts = create_zarr_count_assay(
        root,
        "RNA",
        None,
        8,
        ["f0", "f1", "f2", "f3"],
        ["g0", "g1", "g2", "g3"],
        profile="fast_local",
    )
    values = np.arange(32, dtype=np.uint32).reshape(8, 4)
    counts[:] = values

    assert array_metadata_shards(counts) is None
    assert root["RNA"].attrs["scarf:zarr_spec"]["zarr_format"] == 2
    with pytest.raises(ValueError, match="Zarr format 3"):
        write_counts_t(counts, root["RNA"])
    np.testing.assert_array_equal(counts[:], values)


def test_empty_assay_schema_on_zarr_v2_stays_chunk_only() -> None:
    from scarf.storage.schema import create_empty_zarr_count_assay

    v2_assay = zarr.open_group(store=MemoryStore(), mode="w", zarr_format=2)
    create_empty_zarr_count_assay(
        v2_assay,
        "RNA",
        None,
        3,
        4,
        "U10",
        "U10",
        "uint16",
    )
    assert "counts" in v2_assay["RNA"]


def test_normed_plan_respects_codec_limit():
    spec = normed_array_spec(
        10_000_000,
        2_000,
        profile="cloud",
    )
    assert spec.shards is None
    assert spec.chunks[0] * spec.chunks[1] * 4 <= _CODEC_MAX_BYTES


def test_row_sharded_plan_uses_full_width_divisible_chunks():
    spec = row_sharded_array_spec(
        (10_000_000, 100),
        np.float32,
        profile="cloud",
        band_rows=1_000_000,
    )

    assert spec.shards == (1_000_000, 100)
    assert spec.chunks[1] == 100
    assert spec.shards[0] % spec.chunks[0] == 0
    assert np.prod(spec.chunks) * np.dtype(spec.dtype).itemsize <= 128 * 1024**2


def test_bounded_row_sharded_plan_caps_wide_mapping_bands():
    spec = bounded_row_sharded_array_spec(
        (1_000_000, 4_000),
        np.float64,
        profile="cloud",
    )

    assert spec.shards is not None
    assert spec.shards[0] < spec.shape[0]
    assert spec.shards[0] % spec.chunks[0] == 0
    itemsize = np.dtype(spec.dtype).itemsize
    assert np.prod(spec.chunks) * itemsize <= DEFAULT_TARGET_CHUNK_BYTES
    assert np.prod(spec.shards) * itemsize <= DEFAULT_TARGET_SHARD_BYTES


def test_row_sharded_plan_uses_band_chunks_for_zarr_v2():
    spec = row_sharded_array_spec(
        (12, 3),
        np.uint32,
        profile="fast_local",
        band_rows=5,
        zarr_format=2,
    )

    assert spec.shards is None
    assert spec.chunks == (5, 3)


def test_row_sharded_plan_avoids_unit_chunks_for_irregular_rows():
    spec = row_sharded_array_spec(
        (500_009, 100),
        np.float32,
        profile="cloud",
        band_rows=1_000_000,
    )

    assert spec.shards == (500_009, 100)
    assert spec.chunks[0] > 1
    assert spec.shards[0] % spec.chunks[0] == 0


def test_row_sharded_plan_caps_zarr_v2_chunk_bytes():
    spec = row_sharded_array_spec(
        (300_000_000, 2),
        np.uint32,
        profile="cloud",
        band_rows=300_000_000,
        zarr_format=2,
    )

    assert spec.shards is None
    assert np.prod(spec.chunks) * np.dtype(spec.dtype).itemsize <= _CODEC_MAX_BYTES


def test_flat_connectivity_shards_align_to_one_million_cells():
    k = 50
    spec = row_sharded_array_spec(
        (10_000_000 * k, 2),
        np.uint32,
        profile="cloud",
        band_rows=1_000_000 * k,
    )

    assert spec.shards == (1_000_000 * k, 2)


def test_dense_row_batches_flush_at_shard_boundaries():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    destination = root.create_array(
        "counts",
        shape=(7, 3),
        chunks=(2, 3),
        shards=(4, 3),
        dtype=np.uint16,
        fill_value=0,
    )
    expected = np.arange(21, dtype=np.int64).reshape(7, 3)
    rows = write_dense_from_row_batches(
        destination,
        iter([expected[:1], expected[1:5], expected[5:]]),
        dtype=np.uint16,
        resources=ResourceBudget(1024**2, 4),
    )
    assert rows == 7
    np.testing.assert_array_equal(destination[:], expected.astype(np.uint16))


def test_dense_shard_summaries_are_merged_incrementally():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    destination = root.create_array(
        "values",
        shape=(100, 3),
        chunks=(10, 3),
        dtype=np.float32,
    )
    values = np.arange(300, dtype=np.float32).reshape(100, 3)
    merge_count = 0

    def summarize(block):
        return (
            block.sum(axis=0, dtype=np.float64),
            np.square(block, dtype=np.float64).sum(axis=0),
        )

    def merge(accumulated, current):
        nonlocal merge_count
        merge_count += 1
        accumulated[0][:] += current[0]
        accumulated[1][:] += current[1]
        return accumulated

    summary = write_dense_in_shard_rows(
        destination,
        lambda start, end: values[start:end],
        summarize=summarize,
        merge_summary=merge,
    )

    assert merge_count == 9
    np.testing.assert_allclose(summary[0], values.sum(axis=0, dtype=np.float64))
    np.testing.assert_allclose(
        summary[1],
        np.square(values, dtype=np.float64).sum(axis=0),
    )


def test_sparse_batches_write_complete_shards():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    destination = root.create_array(
        "counts",
        shape=(12, 3),
        chunks=(2, 3),
        shards=(4, 3),
        dtype=np.uint16,
        fill_value=0,
    )
    expected = np.arange(36, dtype=np.uint16).reshape(12, 3)
    expected[4:8] = 0
    batches = (
        csr_matrix(expected[start : start + 5]) for start in range(0, len(expected), 5)
    )
    rows = accumulate_sparse_to_shards(
        destination,
        batches,
        resources=ResourceBudget(1024**2, 4),
        producerReserveBytes=sparse_producer_peak_bytes(27, 15, 2),
    )
    assert rows == len(expected)
    np.testing.assert_array_equal(destination[:], expected)


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        (np.uint8, np.array([200, 100], dtype=np.uint8)),
        (bool, np.array([True, True], dtype=bool)),
    ],
)
def test_sparse_duplicate_sum_rejects_destination_overflow(dtype, values):
    from scipy.sparse import coo_matrix

    root = zarr.open_group(store=MemoryStore(), mode="w")
    destination = root.create_array(
        "counts",
        shape=(1, 1),
        chunks=(1, 1),
        shards=(1, 1),
        dtype=dtype,
        fill_value=0,
    )
    batch = coo_matrix(
        (values, (np.array([0, 0]), np.array([0, 0]))),
        shape=(1, 1),
    )

    with pytest.raises(OverflowError, match="destination dtype"):
        accumulate_sparse_to_shards(
            destination,
            iter([batch]),
            resources=ResourceBudget(1024**2, 1),
            producerReserveBytes=sparse_producer_peak_bytes(
                2,
                2,
                values.itemsize,
            ),
        )


def test_empty_sparse_bands_clear_existing_values():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    destination = root.create_array(
        "counts",
        shape=(8, 3),
        chunks=(2, 3),
        shards=(4, 3),
        dtype=np.uint16,
        fill_value=0,
    )
    destination[:] = np.arange(1, 25, dtype=np.uint16).reshape(8, 3)

    rows = accumulate_sparse_to_shards(
        destination,
        iter([csr_matrix((3, 3)), csr_matrix((5, 3))]),
        resources=ResourceBudget(1024**2, 4),
        producerReserveBytes=0,
    )

    assert rows == 8
    np.testing.assert_array_equal(
        destination[:],
        np.zeros((8, 3), dtype=np.uint16),
    )


def test_complete_sparse_bands_put_each_shard_once_without_get():
    store = RecordingStore()
    root = zarr.open_group(store=store, mode="w")
    destination = root.create_array(
        "counts",
        shape=(12, 3),
        chunks=(2, 3),
        shards=(4, 3),
        dtype=np.uint16,
        fill_value=0,
    )
    expected = np.arange(1, 37, dtype=np.uint16).reshape(12, 3)
    store.reset()

    accumulate_sparse_to_shards(
        destination,
        (csr_matrix(expected[start : start + 5]) for start in range(0, 12, 5)),
        resources=ResourceBudget(1024**2, 4),
        producerReserveBytes=sparse_producer_peak_bytes(27, 15, 2),
    )

    operations = store.chunk_ops("counts/c/")
    assert not [key for kind, key in operations if kind == "get"]
    written = [key for kind, key in operations if kind == "set"]
    assert len(written) == len(set(written)) == 3


def test_sparse_band_writes_respect_memory_admission():
    store = RecordingStore(delay=0.01)
    root = zarr.open_group(store=store, mode="w")
    destination = root.create_array(
        "counts",
        shape=(12, 3),
        chunks=(2, 3),
        shards=(4, 3),
        dtype=np.uint16,
        fill_value=0,
    )
    expected = np.arange(1, 37, dtype=np.uint16).reshape(12, 3)
    store.reset()

    accumulate_sparse_to_shards(
        destination,
        (csr_matrix(expected[start : start + 5]) for start in range(0, 12, 5)),
        resources=ResourceBudget(9_000, 4),
        producerReserveBytes=sparse_producer_peak_bytes(27, 15, 2),
    )

    assert store.max_in_flight_for("set") == 1
    np.testing.assert_array_equal(destination[:], expected)


def test_sparse_writer_releases_completed_band_before_reading_more():
    import gc
    import weakref

    from scarf.storage.sharding import (
        SparseRowBand,
        SparseWriteBand,
        write_sparse_bands,
    )

    root = zarr.open_group(store=MemoryStore(), mode="w")
    destination = root.create_array(
        "counts",
        shape=(8, 3),
        chunks=(2, 3),
        shards=(4, 3),
        dtype=np.uint16,
        fill_value=0,
    )
    expected = np.arange(1, 25, dtype=np.uint16).reshape(8, 3)

    def writes():
        for start in (0, 4):
            row = np.repeat(np.arange(4, dtype=np.int64), 3)
            row_ref = weakref.ref(row)
            yield SparseWriteBand(
                destination=destination,
                band=SparseRowBand(
                    start=start,
                    end=start + 4,
                    nColumns=3,
                    row=row,
                    column=np.tile(np.arange(3, dtype=np.int64), 4),
                    data=expected[start : start + 4].ravel(),
                    dtype=np.uint16,
                ),
            )
            del row
            gc.collect()
            assert row_ref() is None

    write_sparse_bands(
        writes(),
        resources=ResourceBudget(7_000, 4),
    )
    np.testing.assert_array_equal(destination[:], expected)


def test_sparse_writer_admits_row_chunks_and_retained_producer_bytes():
    from scarf.storage.sharding import (
        SparseRowBand,
        SparseWriteBand,
        write_sparse_bands,
    )

    root = zarr.open_group(store=MemoryStore(), mode="w")
    row_chunked = root.create_array(
        "row_chunked",
        shape=(4, 3),
        chunks=(2, 3),
        shards=(4, 3),
        dtype=np.uint16,
        fill_value=0,
    )

    def empty_write(destination, producer_bytes=0):
        return SparseWriteBand(
            destination=destination,
            band=SparseRowBand(
                start=0,
                end=4,
                nColumns=3,
                row=np.array([], dtype=np.int64),
                column=np.array([], dtype=np.int64),
                data=np.array([], dtype=np.uint16),
                dtype=np.uint16,
            ),
            producerBytes=producer_bytes,
        )

    with pytest.raises(MemoryError):
        write_sparse_bands(
            iter([empty_write(row_chunked)]),
            resources=ResourceBudget(5_000, 4),
        )

    destination = root.create_array(
        "producer",
        shape=(4, 3),
        chunks=(4, 3),
        shards=(4, 3),
        dtype=np.uint16,
        fill_value=0,
    )
    write = empty_write(destination, producer_bytes=5_000)

    with pytest.raises(MemoryError):
        write_sparse_bands(
            iter([write]),
            resources=ResourceBudget(7_000, 4),
        )

    pulled = False

    def writes():
        nonlocal pulled
        pulled = True
        yield write

    with pytest.raises(MemoryError, match="Sparse producer"):
        write_sparse_bands(
            writes(),
            resources=ResourceBudget(7_000, 4),
            producerReserveBytes=7_001,
        )
    assert pulled is False


def test_padded_shard_geometry_stays_readable_and_transposable():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    expected = np.arange(1, 71, dtype=np.uint16).reshape(10, 7)
    counts = _planned_counts(root, expected)
    resources = ResourceBudget(1024**2, 4)

    rows = write_dense_from_row_batches(
        counts,
        iter([expected[:4], expected[4:]]),
        resources=resources,
    )
    assert rows == 10
    np.testing.assert_array_equal(counts[:], expected)
    np.testing.assert_array_equal(
        ChunkedArray(counts, resources=resources).compute(),
        expected,
    )

    counts_t = write_counts_t(counts, root, resources=resources)
    assert counts_t is not None
    assert counts_t.attrs["complete"] is True
    np.testing.assert_array_equal(counts_t[:], expected.T)


def test_write_counts_t_works_inside_running_event_loop():
    import asyncio

    root = zarr.open_group(store=MemoryStore(), mode="w")
    expected = np.arange(12, dtype=np.uint16).reshape(4, 3)
    counts = _planned_counts(root, expected)

    async def invoke():
        return write_counts_t(
            counts,
            root,
            resources=ResourceBudget(1024**2, 2),
        )

    counts_t = asyncio.run(invoke())
    assert counts_t is not None
    assert counts_t.attrs["complete"] is True
    np.testing.assert_array_equal(counts_t[:], expected.T)


def test_empty_dense_and_transpose_writes_need_no_task_memory():
    from scipy.sparse import coo_matrix

    root = zarr.open_group(store=MemoryStore(), mode="w")
    empty = np.zeros((0, 3), dtype=np.uint8)
    counts = _planned_counts(root, empty)
    resources = ResourceBudget(1, 4)

    assert write_dense_from_row_batches(counts, iter(()), resources=resources) == 0
    counts_t = write_counts_t(counts, root, resources=resources)
    assert counts_t is not None
    assert counts_t.shape == (3, 0)
    assert counts_t.attrs["complete"] is True

    sparse_counts = root.create_array(
        "sparse_counts",
        shape=(0, 3),
        chunks=(1, 3),
        shards=(1, 3),
        dtype=np.uint8,
        fill_value=0,
    )
    rows = accumulate_sparse_to_shards(
        sparse_counts,
        iter([coo_matrix((0, 3), dtype=np.uint8)]),
        resources=resources,
        producerReserveBytes=1024,
    )
    assert rows == 0


def test_copy_array_and_metadata_tree(tmp_path):
    source_root = zarr.open_group(str(tmp_path / "source.zarr"), mode="w")
    spec = normed_array_spec(64, 8, profile="fast_local")
    source = create_numeric_array(source_root, "data", spec)
    expected = np.random.default_rng(0).random((64, 8), dtype=np.float32)
    source[:] = expected

    target_root = zarr.open_group(str(tmp_path / "target.zarr"), mode="w")
    target = create_numeric_array(target_root, "data", spec)
    copy_zarr_array(
        source,
        target,
        resources=ResourceBudget(1024**2, 2),
    )
    np.testing.assert_allclose(target[:], expected)

    metadata = source_root.create_group("metadata")
    score = metadata.create_array("score", data=np.array([1.0, 2.0, 3.0]))
    score.attrs["display"] = {"label": "Score"}
    copied_metadata = target_root.create_group("metadata")
    copy_zarr_group_tree(metadata, copied_metadata)
    np.testing.assert_array_equal(copied_metadata["score"][:], score[:])
    assert copied_metadata["score"].attrs["display"] == {"label": "Score"}


def test_copy_zarr_array_rejects_shape_and_rank_mismatch(tmp_path):
    source_root = zarr.open_group(str(tmp_path / "source.zarr"), mode="w")
    target_root = zarr.open_group(str(tmp_path / "target.zarr"), mode="w")
    source = create_numeric_array(
        source_root,
        "data",
        normed_array_spec(8, 4, profile="fast_local"),
    )
    mismatched = create_numeric_array(
        target_root,
        "data",
        normed_array_spec(8, 3, profile="fast_local"),
    )
    with pytest.raises(ValueError, match="Shape mismatch"):
        copy_zarr_array(source, mismatched)

    vector = source_root.create_array("vector", data=np.arange(8, dtype=np.float32))
    vector_target = target_root.create_array(
        "vector",
        data=np.zeros(8, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="only supports 2D"):
        copy_zarr_array(vector, vector_target)


def test_copy_group_tree_resolves_byte_string_metadata(tmp_path):
    source_root = zarr.open_group(str(tmp_path / "source.zarr"), mode="w")
    target_root = zarr.open_group(str(tmp_path / "target.zarr"), mode="w")
    metadata = source_root.create_group("metadata")
    labels = np.array([b"alpha", b"beta-gamma", b"x"])
    column = metadata.create_array("labels", data=labels)
    column.attrs["display"] = {"label": "Labels"}
    assert np.dtype(column.dtype).kind == "S"

    copied = target_root.create_group("metadata")
    copy_zarr_group_tree(metadata, copied)

    np.testing.assert_array_equal(
        np.asarray(copied["labels"][:]).astype(str),
        ["alpha", "beta-gamma", "x"],
    )
    assert copied["labels"].attrs["display"] == {"label": "Labels"}
    assert np.dtype(copied["labels"].dtype).kind == "U"
    assert np.dtype(copied["labels"].dtype).itemsize // 4 >= len("beta-gamma")


def test_staged_normed_array_reuses_matching_shape(tmp_path):
    path = str(tmp_path / "cache" / "normalized.zarr")
    first = create_or_open_staged_normed_array(path, (32, 4))
    first[:] = 1
    reopened = create_or_open_staged_normed_array(path, (32, 4))
    np.testing.assert_array_equal(reopened[:], np.ones((32, 4), dtype=np.float32))


def test_remote_store_requires_obstore(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def reject_obstore(name, *args, **kwargs):
        if name in {"obstore", "obstore.store"}:
            raise ImportError("missing obstore")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_obstore)
    with pytest.raises(ImportError, match="obstore"):
        make_store("s3://bucket/path")


def test_store_probe_count_only_skips_per_key_logs() -> None:
    from tests.store_probes import StoreProbe

    probe = StoreProbe(countOnly=True)
    probe.enter("get", "a/key", requestedBytes=12)
    probe.record_transfer("get", "a/key", 12)
    probe.enter("set", "b/key", requestedBytes=8)
    probe.record_transfer("set", "b/key", 8)
    payload = probe.to_json()
    assert probe.ops == []
    assert probe.transferred_bytes == []
    assert payload["gets"] == 1
    assert payload["sets"] == 1
    assert payload["readTransferredBytes"] == 12
    assert payload["writeTransferredBytes"] == 8
    assert payload["requestedBytes"] == 20
    assert payload["keysTouched"] == 0
