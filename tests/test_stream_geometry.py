import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.assay.persistence import _read_block
from scarf.matrix.chunked import ChunkedArray
from scarf.storage.budget import ResourceBudget, admit_stream
from scarf.storage.feature_stream import plan_feature_stream
from scarf.storage.geometry import ArrayGeometry, array_geometry
from scarf.storage.partition import (
    affordable_width,
    checked_indices,
    contiguous_ranges,
    partition_indices,
    row_band,
)
from scarf.utils.prefetch import iter_column_blocks

from .store_probes import RecordingStore


def _array(
    *,
    shape: tuple[int, int],
    chunks: tuple[int, int],
    shards: tuple[int, int] | None = None,
    dtype: type[np.generic] = np.uint32,
    compressors: None | str = "auto",
) -> zarr.Array:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    return root.create_array(
        "counts",
        shape=shape,
        chunks=chunks,
        shards=shards,
        dtype=dtype,
        fill_value=0,
        compressors=compressors,
    )


def _geometry(
    *,
    shape: tuple[int, int],
    chunks: tuple[int, int],
    shards: tuple[int, int] | None = None,
    itemsize: int = 4,
) -> ArrayGeometry:
    return ArrayGeometry(
        shape=shape,
        chunks=chunks,
        shards=shards,
        itemsize=itemsize,
    )


def test_nominal_chunk_bytes_counts_the_whole_chunk_at_an_array_edge() -> None:
    array = _array(shape=(12, 7), chunks=(5, 3))
    geometry = array_geometry(array)

    assert geometry is not None
    # The trailing chunk covers two rows and one column of the array, but Zarr
    # stores and decodes the full 5 x 3 chunk.
    assert geometry.nominalChunkBytes() == 5 * 3 * 4


def test_an_edge_chunk_decodes_to_the_nominal_size() -> None:
    array = _array(shape=(12, 7), chunks=(5, 3), compressors=None)
    array[10:12, 6:7] = 1
    geometry = array_geometry(array)

    uncompressed = [
        len(array.store._store_dict[key].to_bytes())
        for key in array.store._store_dict
        if "/c/" in key
    ]

    assert geometry is not None
    assert uncompressed == [geometry.nominalChunkBytes()]


def test_array_geometry_is_absent_for_in_memory_arrays() -> None:
    assert array_geometry(np.zeros((4, 4), dtype=np.uint32)) is None


def test_array_geometry_records_shards_and_falls_back_to_chunks() -> None:
    sharded = array_geometry(_array(shape=(20, 20), chunks=(5, 5), shards=(10, 10)))
    plain = array_geometry(_array(shape=(20, 20), chunks=(5, 5)))

    assert sharded is not None and plain is not None
    assert sharded.shards == (10, 10)
    assert sharded.axisShard(0) == 10
    assert sharded.axisChunk(0) == 5
    assert plain.shards is None
    assert plain.axisShard(0) == plain.axisChunk(0) == 5


def test_bin_of_maps_indices_to_their_chunk() -> None:
    geometry = _geometry(shape=(12, 20), chunks=(5, 5))

    np.testing.assert_array_equal(
        geometry.binOf(1, np.array([0, 4, 5, 17])),
        np.array([0, 0, 1, 3]),
    )


def test_row_band_follows_the_requested_stored_unit() -> None:
    geometry = _geometry(shape=(20, 20), chunks=(5, 5), shards=(10, 10))

    assert row_band(geometry, fallback=999) == 10
    assert row_band(geometry, unit="chunk", fallback=999) == 5


def test_row_band_uses_the_fallback_without_geometry() -> None:
    assert row_band(None, fallback=10_000) == 10_000
    assert row_band(None, fallback=0) == 1


def test_contiguous_ranges_cover_the_axis_without_overlap() -> None:
    assert contiguous_ranges(12, 5) == [(0, 5), (5, 10), (10, 12)]
    assert contiguous_ranges(0, 5) == []


def test_affordable_width_finds_the_largest_accepted_width() -> None:
    assert affordable_width(lambda width: width <= 3, 10) == 3
    assert affordable_width(lambda width: width <= 20, 10) == 10
    assert affordable_width(lambda _width: False, 10) == 0


def test_partition_indices_emits_one_block_per_chunk_by_default() -> None:
    geometry = _geometry(shape=(12, 20), chunks=(5, 5))

    blocks = partition_indices(geometry, 1, np.array([0, 1, 5, 6, 17]))

    assert [block.bins for block in blocks] == [(0,), (1,), (3,)]
    np.testing.assert_array_equal(blocks[0].indices, np.array([0, 1]))
    np.testing.assert_array_equal(blocks[2].destinations, np.array([4]))


def test_partition_indices_packs_only_adjacent_chunks() -> None:
    geometry = _geometry(shape=(12, 20), chunks=(5, 5))

    blocks = partition_indices(
        geometry,
        1,
        np.array([0, 1, 5, 6, 17]),
        fits=lambda width: width <= 4,
    )

    assert [block.bins for block in blocks] == [(0, 1), (3,)]
    np.testing.assert_array_equal(blocks[0].destinations, np.array([0, 1, 2, 3]))


def test_partition_indices_preserves_destinations_for_unsorted_indices() -> None:
    geometry = _geometry(shape=(12, 15), chunks=(5, 5))
    requested = np.array([12, 1, 6])

    blocks = partition_indices(
        geometry,
        1,
        requested,
        fits=lambda width: width <= 3,
    )

    destinations = np.concatenate([block.destinations for block in blocks])
    indices = np.concatenate([block.indices for block in blocks])
    restored = np.empty(requested.size, dtype=np.int64)
    restored[destinations] = indices

    np.testing.assert_array_equal(restored, requested)


def test_partition_indices_splits_a_chunk_that_is_too_wide_alone() -> None:
    geometry = _geometry(shape=(12, 5), chunks=(5, 5))

    blocks = partition_indices(
        geometry,
        1,
        np.arange(5),
        fits=lambda width: width <= 2,
    )

    assert [block.indices.size for block in blocks] == [2, 2, 1]
    assert [block.bins for block in blocks] == [(0,), (0,), (0,)]


def test_partition_indices_cuts_fixed_width_blocks_in_request_order() -> None:
    geometry = _geometry(shape=(12, 10), chunks=(5, 5))

    blocks = partition_indices(geometry, 1, np.arange(10), maxWidth=4)

    assert [block.indices.size for block in blocks] == [4, 4, 2]
    np.testing.assert_array_equal(blocks[1].indices, np.arange(4, 8))
    np.testing.assert_array_equal(blocks[1].destinations, np.arange(4, 8))
    assert blocks[1].bins == (0, 1)


def test_partition_indices_rejects_an_index_that_never_fits() -> None:
    geometry = _geometry(shape=(12, 10), chunks=(5, 5))

    with pytest.raises(MemoryError, match="does not fit the operation budget"):
        partition_indices(geometry, 1, np.arange(10), fits=lambda _width: False)


def test_partition_indices_rejects_both_width_and_predicate() -> None:
    geometry = _geometry(shape=(12, 10), chunks=(5, 5))

    with pytest.raises(ValueError, match="either maxWidth or fits"):
        partition_indices(
            geometry,
            1,
            np.arange(4),
            maxWidth=2,
            fits=lambda _width: True,
        )


def test_partition_indices_returns_nothing_for_an_empty_selection() -> None:
    geometry = _geometry(shape=(12, 10), chunks=(5, 5))

    assert partition_indices(geometry, 1, np.array([], dtype=np.int64)) == []


def test_checked_indices_rejects_malformed_selections() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        checked_indices(np.zeros((2, 2), dtype=np.int64), limit=4, name="rows")
    with pytest.raises(TypeError, match="must contain integers"):
        checked_indices(np.array([0.5]), limit=4, name="rows")
    with pytest.raises(IndexError, match="out-of-range"):
        checked_indices(np.array([4]), limit=4, name="rows")
    with pytest.raises(ValueError, match="duplicate"):
        checked_indices(np.array([1, 1]), limit=4, name="rows")


def test_admit_stream_charges_every_concurrent_chunk_decode() -> None:
    resources = ResourceBudget(1_000, 8)

    admission = admit_stream(
        resources,
        nBlocks=8,
        blockBytes=400,
        decodeBytes=100,
    )

    assert (admission.outerWorkers, admission.ioConcurrency) == (2, 1)
    peak = admission.outerWorkers * (400 + admission.ioConcurrency * 100)
    assert peak <= resources.memoryBytes


def test_admit_stream_without_decode_bytes_matches_a_flat_task_cost() -> None:
    resources = ResourceBudget(1_000, 8)

    admission = admit_stream(resources, nBlocks=8, blockBytes=250)

    assert admission.outerWorkers == 4
    assert admission.outerWorkers * 250 <= resources.memoryBytes


def test_admit_stream_honours_resident_bytes_and_a_requested_depth() -> None:
    resources = ResourceBudget(1_000, 8)

    admission = admit_stream(
        resources,
        nBlocks=8,
        blockBytes=100,
        residentBytes=600,
        requested=2,
    )

    assert admission.outerWorkers == 2
    assert 600 + admission.outerWorkers * 100 <= resources.memoryBytes


def test_admit_stream_reports_when_one_block_cannot_fit() -> None:
    with pytest.raises(MemoryError, match="One task needs"):
        admit_stream(ResourceBudget(100, 4), nBlocks=4, blockBytes=1_000)


def test_row_band_reproduces_every_expression_it_replaced() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    sharded = root.create_array(
        "sharded",
        shape=(200, 8),
        chunks=(10, 8),
        shards=(50, 8),
        dtype=np.uint32,
    )
    chunked = root.create_array("chunked", shape=(200, 8), chunks=(10, 8), dtype="u4")
    scalar = root.create_array("scalar", shape=(), dtype="u4")
    in_memory = np.zeros((200, 8), dtype=np.uint32)

    for array in (sharded, chunked, scalar, in_memory):
        geometry = array_geometry(array)
        chunks = getattr(array, "chunks", None)
        rows = int(array.shape[0]) if array.ndim else 0

        # scarf.storage.layout.array_shard_rows
        shards = getattr(getattr(array, "metadata", None), "shards", None)
        if shards is not None and len(shards) > 0:
            expected_shard = int(shards[0])
        elif chunks is not None and len(chunks) > 0:
            expected_shard = int(chunks[0])
        else:
            expected_shard = max(rows, 1)
        assert row_band(geometry, fallback=max(rows, 1)) == expected_shard

        # scarf.metadata.rows.default_block_rows
        if chunks and len(chunks) > 0 and int(chunks[0]) > 0:
            expected_metadata = int(chunks[0])
        else:
            expected_metadata = max(1, min(rows, 100_000))
        assert (
            row_band(geometry, unit="chunk", fallback=min(rows, 100_000))
            == expected_metadata
        )

        # mapping._projection_block_size, confidence.py and hashing.py
        if chunks is not None and len(chunks) > 0:
            expected_projection = int(chunks[0])
        else:
            expected_projection = min(max(rows, 1), 10_000)
        assert (
            row_band(geometry, unit="chunk", fallback=min(max(rows, 1), 10_000))
            == expected_projection
        )

        # scarf.storage.artifacts._stored_array_chunk_rows
        if array.ndim > 0 and chunks is not None:
            expected_artifact = max(int(chunks[0]), 1)
        else:
            expected_artifact = 1
        assert row_band(geometry, unit="chunk", fallback=1) == expected_artifact


def _plan_with(
    *,
    resources: ResourceBudget,
    features: np.ndarray,
    batch: int | None = 4,
):
    array = _array(shape=(25, 32), chunks=(25, 4))
    cells = np.arange(25)
    return plan_feature_stream(
        array,
        featureAxis=1,
        cellAxis=0,
        featureIndices=features,
        cellIndices=cells,
        resources=resources,
        blockBytes=lambda width: width * cells.size * 4,
        requestedBatchSize=batch,
    )


def test_a_stream_that_cannot_be_admitted_reads_one_chunk_at_a_time() -> None:
    # Room for one block beside its decode, but not for a second read on top,
    # so admission fails and the plan must not leave read-ahead concurrency on.
    plan = _plan_with(
        resources=ResourceBudget(1_000, 8),
        features=np.arange(32),
    )

    assert len(plan.blocks) == 8
    assert (plan.readWorkers, plan.ioConcurrency) == (1, 1)


def test_a_stream_without_read_ahead_uses_current_decode_concurrency() -> None:
    # A second block cannot fit, but the current block can afford two decoded
    # chunks. Preserve that inner concurrency without admitting read-ahead.
    plan = _plan_with(
        resources=ResourceBudget(1_200, 8),
        features=np.arange(32),
    )

    assert len(plan.blocks) == 8
    assert (plan.readWorkers, plan.ioConcurrency) == (1, 2)


def test_a_single_block_stream_budgets_its_own_decode_concurrency() -> None:
    plan = _plan_with(
        resources=ResourceBudget(1_600, 8),
        features=np.arange(4),
    )

    assert len(plan.blocks) == 1
    assert plan.readWorkers == 1
    # One 400 byte block plus three 400 byte decodes exhausts the limit, so the
    # plan admits three concurrent decodes rather than one per worker.
    assert plan.ioConcurrency == 3
    assert 400 + plan.ioConcurrency * 400 <= 1_600


def test_an_empty_selection_plans_no_read_concurrency() -> None:
    plan = _plan_with(
        resources=ResourceBudget(1_000_000, 8),
        features=np.array([], dtype=np.int64),
    )

    assert plan.blocks == ()
    assert (plan.readWorkers, plan.ioConcurrency) == (1, 1)


def test_streamed_row_blocks_decode_only_as_many_chunks_as_budgeted() -> None:
    store = RecordingStore(delay=0.005)
    root = zarr.open_group(store=store, mode="w")
    array = root.create_array(
        "counts",
        shape=(64, 32),
        chunks=(16, 8),
        dtype=np.uint32,
        compressors=None,
        fill_value=0,
    )
    expected = np.arange(64 * 32, dtype=np.uint32).reshape(64, 32)
    array[:] = expected

    # A row block spans the four column chunks of one chunk row, so an
    # unbounded read can hold all four decodes at once. The budget affords one
    # 2048 byte block beside a single 512 byte chunk decode, and no more.
    budgeted = ChunkedArray(array, resources=ResourceBudget(3_000, 4))
    store.reset()
    blocks = list(budgeted.stream_blocks())

    np.testing.assert_array_equal(np.vstack(blocks), expected)
    assert store.max_in_flight_for("get") == 1

    unbudgeted = ChunkedArray(array)
    store.reset()
    list(unbudgeted.stream_blocks())

    assert store.max_in_flight_for("get") > 1, (
        "without a budget Zarr uses its own concurrency, so the probe should "
        "observe the overlap this test bounds"
    )


def test_planned_reads_never_decode_more_chunks_than_budgeted() -> None:
    # Budget chosen so admission has to trade read depth against per-read
    # decodes: charging one decode per read admits four concurrent chunk
    # decodes, which overruns the limit once they are actually in flight.
    store = RecordingStore(delay=0.005)
    root = zarr.open_group(store=store, mode="w")
    array = root.create_array(
        "counts",
        shape=(64, 32),
        chunks=(16, 4),
        dtype=np.uint32,
        compressors=None,
        fill_value=0,
    )
    array[:] = np.arange(64 * 32, dtype=np.uint32).reshape(64, 32)

    cells = np.arange(64)
    resources = ResourceBudget(4_280, 8)
    block_bytes = 4 * cells.size * 4
    plan = plan_feature_stream(
        array,
        featureAxis=1,
        cellAxis=0,
        featureIndices=np.arange(32),
        cellIndices=cells,
        resources=resources,
        blockBytes=lambda width: width * cells.size * 4,
        requestedBatchSize=4,
    )

    assert len(plan.blocks) == 8
    assert (plan.readWorkers, plan.ioConcurrency) == (2, 1)

    store.reset()
    blocks = list(
        iter_column_blocks(
            len(plan.blocks),
            lambda index: _read_block(array, cells, plan.blocks[index].indices),
            workers=plan.readWorkers,
            io_concurrency=plan.ioConcurrency,
        )
    )

    peak = store.max_in_flight_for("get")
    decode = plan.geometry.nominalChunkBytes()
    resident = block_bytes + decode
    live = resident + plan.readWorkers * block_bytes + peak * decode

    assert len(blocks) == 8
    assert peak >= 2, "the probe never observed overlapping reads"
    assert peak <= plan.readWorkers * plan.ioConcurrency
    assert live <= resources.memoryBytes
