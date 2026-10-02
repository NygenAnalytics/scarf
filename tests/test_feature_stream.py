import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from tests.storage_helpers import reset_zarr_runtime
from scarf.storage.budget import ResourceBudget
from scarf.storage.count_matrix import (
    CountMatrixPolicy,
    persist_count_matrix_plan,
    plan_count_matrix_pair,
)
from scarf.storage.feature_stream import plan_feature_stream
from scarf.storage.io_policy import StorageIoPolicy


def setup_function() -> None:
    reset_zarr_runtime()


def teardown_function() -> None:
    reset_zarr_runtime()


def _array(
    *,
    shape: tuple[int, int] = (12, 15),
    chunks: tuple[int, int] = (5, 5),
) -> zarr.Array:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    return root.create_array(
        "counts",
        shape=shape,
        chunks=chunks,
        dtype=np.uint32,
        fill_value=0,
    )


def test_feature_stream_packs_only_adjacent_whole_bins() -> None:
    array = _array(shape=(12, 20))
    plan = plan_feature_stream(
        array,
        featureAxis=1,
        cellAxis=0,
        featureIndices=np.array([0, 1, 5, 6, 17]),
        cellIndices=np.arange(12),
        resources=ResourceBudget(10_000, 4),
        blockBytes=lambda width: width * 100,
    )

    assert [block.bins for block in plan.blocks] == [(0, 1), (3,)]
    np.testing.assert_array_equal(
        plan.blocks[0].destinations,
        np.array([0, 1, 2, 3]),
    )


def test_feature_stream_preserves_destination_order_for_unsorted_features() -> None:
    array = _array()
    plan = plan_feature_stream(
        array,
        featureAxis=1,
        cellAxis=0,
        featureIndices=np.array([12, 1, 6]),
        cellIndices=np.arange(12),
        resources=ResourceBudget(10_000, 2),
        blockBytes=lambda width: width * 100,
    )

    destinations = np.concatenate([block.destinations for block in plan.blocks])
    features = np.concatenate([block.indices for block in plan.blocks])
    restored = np.empty(3, dtype=np.int64)
    restored[destinations] = features
    np.testing.assert_array_equal(restored, np.array([12, 1, 6]))


def test_feature_stream_splits_one_bin_and_counts_repeated_tiles() -> None:
    array = _array(shape=(12, 5), chunks=(5, 5))
    plan = plan_feature_stream(
        array,
        featureAxis=1,
        cellAxis=0,
        featureIndices=np.arange(5),
        cellIndices=np.array([0, 6, 11]),
        resources=ResourceBudget(350, 4),
        blockBytes=lambda width: width * 100,
    )

    assert [block.indices.size for block in plan.blocks] == [2, 2, 1]
    assert plan.repeatedDecodeCount == 6


def test_feature_stream_budgets_an_edge_chunk_at_its_nominal_size() -> None:
    # A 5 x 3 chunk of uint32 decodes to 60 bytes even where it overhangs the
    # 12 x 7 array and only 2 x 1 of its elements are addressable.
    array = _array(shape=(12, 7), chunks=(5, 3))
    kwargs = {
        "featureAxis": 1,
        "cellAxis": 0,
        "featureIndices": np.array([6]),
        "cellIndices": np.array([10, 11]),
        "blockBytes": lambda width: width,
    }

    plan = plan_feature_stream(array, resources=ResourceBudget(61, 1), **kwargs)
    assert plan.geometry.nominalChunkBytes() == 60
    assert len(plan.blocks) == 1

    with pytest.raises(MemoryError):
        plan_feature_stream(array, resources=ResourceBudget(60, 1), **kwargs)


def test_feature_stream_rejects_unaffordable_override() -> None:
    array = _array(shape=(12, 10), chunks=(5, 5))
    with pytest.raises(
        MemoryError,
        match="Requested feature batch width 5.*affordable width is 2",
    ):
        plan_feature_stream(
            array,
            featureAxis=1,
            cellAxis=0,
            featureIndices=np.arange(10),
            cellIndices=np.arange(12),
            resources=ResourceBudget(350, 2),
            blockBytes=lambda width: width * 100,
            requestedBatchSize=5,
        )


def test_feature_stream_accounts_for_resident_bytes() -> None:
    array = _array(shape=(12, 10), chunks=(5, 5))
    plan = plan_feature_stream(
        array,
        featureAxis=1,
        cellAxis=0,
        featureIndices=np.arange(10),
        cellIndices=np.arange(12),
        resources=ResourceBudget(1_000, 4),
        residentBytes=600,
        blockBytes=lambda width: width * 100,
    )

    widest = max(block.indices.size for block in plan.blocks)
    decode = plan.geometry.nominalChunkBytes()
    assert widest == 3
    assert 600 + widest * 100 + decode <= 1_000


def test_feature_stream_keeps_reads_and_their_decodes_inside_the_budget() -> None:
    array = _array(shape=(20, 40), chunks=(5, 5))
    plan = plan_feature_stream(
        array,
        featureAxis=1,
        cellAxis=0,
        featureIndices=np.array([0, 10, 20, 30]),
        cellIndices=np.arange(20),
        resources=ResourceBudget(650, 8),
        blockBytes=lambda _width: 100,
    )

    decode = plan.geometry.nominalChunkBytes()
    in_flight = plan.readWorkers * (100 + plan.ioConcurrency * decode)

    assert len(plan.blocks) == 4
    assert plan.readWorkers == 3
    assert in_flight <= 650
    assert plan.readWorkers * plan.ioConcurrency <= 8


def test_feature_stream_records_optional_shard_geometry() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    array = root.create_array(
        "counts",
        shape=(20, 20),
        chunks=(5, 5),
        shards=(10, 10),
        dtype=np.uint32,
    )
    plan = plan_feature_stream(
        array,
        featureAxis=1,
        cellAxis=0,
        featureIndices=np.array([0]),
        cellIndices=np.array([0]),
        resources=ResourceBudget(1_000, 1),
        blockBytes=lambda _width: 1,
    )

    assert plan.geometry.shards == (10, 10)


def test_feature_stream_packs_many_single_feature_bins() -> None:
    array = _array(shape=(10, 2_000), chunks=(5, 1))
    plan = plan_feature_stream(
        array,
        featureAxis=1,
        cellAxis=0,
        featureIndices=np.arange(2_000),
        cellIndices=np.arange(10),
        resources=ResourceBudget(10_000, 2),
        blockBytes=lambda width: width,
    )

    assert len(plan.blocks) == 1
    assert plan.blocks[0].bins[0] == 0
    assert plan.blocks[0].bins[-1] == 1_999


def _counts_t_with_plan(
    values: np.ndarray,
    *,
    policy: CountMatrixPolicy | None = None,
    store: MemoryStore | None = None,
) -> zarr.Array:
    resolved = policy or CountMatrixPolicy(unitBytes=2_000, chunkBytes=200)
    plan = plan_count_matrix_pair(
        values.shape[0],
        values.shape[1],
        values.dtype,
        policy=resolved,
    )
    root = zarr.open_group(store=MemoryStore() if store is None else store, mode="w")
    group = root.create_group("RNA")
    counts_t = group.create_array(
        "countsT",
        shape=plan.countsT.shape,
        chunks=plan.countsT.chunks,
        shards=plan.countsT.shards,
        dtype=values.dtype,
        overwrite=True,
    )
    counts_t[:] = values.T
    persist_count_matrix_plan(group, plan)
    persist_count_matrix_plan(counts_t, plan)
    return counts_t


def test_map_feature_read_groups_preserves_unsorted_cell_order() -> None:
    from scarf.storage.feature_stream import map_feature_read_groups

    values = np.arange(24, dtype=np.uint16).reshape(6, 4)
    counts_t = _counts_t_with_plan(values)
    cell_idx = np.array([5, 0, 3])
    loaded = list(
        map_feature_read_groups(
            counts_t,
            lambda group: group,
            cell_idx=cell_idx,
            feat_idx=np.arange(4),
            resources=ResourceBudget(8 * 1024 * 1024, 2),
        )
    )
    assert loaded
    stacked = np.empty((values.shape[1], cell_idx.shape[0]), dtype=values.dtype)
    for group in loaded:
        stacked[group.featStart : group.featEnd] = group.values
    np.testing.assert_array_equal(stacked, values.T[:, cell_idx])


def test_map_feature_cell_bands_reduces_in_band_order() -> None:
    from scarf.storage.feature_stream import map_feature_cell_bands

    values = np.arange(24, dtype=np.uint16).reshape(6, 4)
    counts_t = _counts_t_with_plan(values)
    cell_idx = np.array([5, 0, 3])
    dest = np.zeros((4, 3), dtype=np.uint16)
    seen: list[tuple[int, int]] = []

    def accumulate(band):  # type: ignore[no-untyped-def]
        seen.append((band.featStart, band.cellStart))
        dest[band.featStart : band.featEnd, band.selectedDestinations] = band.values[
            :, band.selectedLocal
        ]
        return band.featStart

    list(
        map_feature_cell_bands(
            counts_t,
            accumulate,
            cell_idx=cell_idx,
            resources=ResourceBudget(8 * 1024 * 1024, 2),
        )
    )
    np.testing.assert_array_equal(dest, values.T[:, cell_idx])
    assert seen == sorted(seen)


def test_sparse_feature_selection_reads_only_selected_rows() -> None:
    from scarf.storage.feature_stream import map_feature_cell_bands

    values = np.arange(12 * 40, dtype=np.uint16).reshape(12, 40)
    counts_t = _counts_t_with_plan(values)
    cell_idx = np.array([11, 2, 7])
    feat_idx = np.array([1, 17, 30])
    dest = np.zeros((40, 3), dtype=np.uint16)
    row_sets = []

    def capture(band):  # type: ignore[no-untyped-def]
        row_sets.append(band.rows)
        rows = band.featStart + band.featureRows()
        dest[np.ix_(rows, band.selectedDestinations)] = band.values[
            :, band.selectedLocal
        ]

    list(
        map_feature_cell_bands(
            counts_t,
            capture,
            cell_idx=cell_idx,
            feat_idx=feat_idx,
            resources=ResourceBudget(8 * 1024 * 1024, 2),
        )
    )
    assert row_sets and all(rows is not None for rows in row_sets)
    np.testing.assert_array_equal(dest[feat_idx], values.T[feat_idx][:, cell_idx])
    assert not np.delete(dest, feat_idx, axis=0).any()


@pytest.mark.parametrize(("spare", "readers"), [(0, 2), (-1, 1)])
def test_cell_band_admission_charges_the_zarr_read_of_each_band(
    spare: int, readers: int
) -> None:
    from scarf.storage.feature_stream import map_feature_cell_bands

    # Two 10-feature groups of one 10-cell band, each band one 200-byte chunk.
    values = np.arange(120, dtype=np.uint16).reshape(10, 12)
    counts_t = _counts_t_with_plan(values)
    band_bytes = 200
    # The band, Zarr's shard-level copy, and the chunk decoded next to its
    # compressed bytes.
    read_bytes = 2 * band_bytes + 2 * 200
    # A local and a destination index per cell, and the selected cells.
    index_bytes = 10 * 8 * 2 + 10 * 8
    metrics: dict[str, object] = {}

    list(
        map_feature_cell_bands(
            counts_t,
            lambda _band: None,
            resources=ResourceBudget(index_bytes + 2 * read_bytes + spare, 2),
            io=StorageIoPolicy(readWorkers=4),
            metrics=metrics,
        )
    )

    assert metrics["cellBandBytes"] == band_bytes
    assert metrics["unitBytes"] == read_bytes
    assert metrics["residentBytes"] == index_bytes
    assert metrics["readGroupBytes"] == 200
    assert metrics["actualReadWorkers"] == readers


def test_selected_values_and_persisted_groups_preserve_order() -> None:
    from scarf.storage.feature_stream import (
        map_feature_read_groups,
        persisted_read_group,
        selected_feature_chunk_starts,
        selected_feature_values,
    )

    n_cells, n_feats = 12, 40
    values = np.arange(n_cells * n_feats, dtype=np.uint16).reshape(n_cells, n_feats)
    counts_t = _counts_t_with_plan(values)
    starts = selected_feature_chunk_starts(counts_t)
    assert starts
    groups = list(
        map_feature_read_groups(
            counts_t,
            lambda group: group,
            resources=ResourceBudget(64 * 1024 * 1024, 2),
            progress="test-progress",
        )
    )
    first = groups[0]
    keep = np.ones(first.values.shape[0], dtype=bool)
    assert selected_feature_values(first.values, keep) is first.values
    keep[0] = False
    filtered = selected_feature_values(first.values, keep)
    assert filtered.shape[0] == first.values.shape[0] - 1
    feature_width, _bytes = persisted_read_group(counts_t)
    assert first.featEnd - first.featStart <= feature_width
    starts = [group.featStart for group in groups]
    assert starts
    assert len(starts) == len(set(starts))
    assert min(starts) == 0
    assert max(group.featEnd for group in groups) == n_feats


def test_consume_uses_persisted_two_gib_read_group() -> None:
    from scarf.storage.feature_stream import (
        map_feature_read_groups,
        persisted_read_group,
    )

    values = np.arange(20 * 8, dtype=np.uint16).reshape(20, 8)
    policy = CountMatrixPolicy(unitBytes=2_000_000_000, chunkBytes=100_000_000)
    counts_t = _counts_t_with_plan(values, policy=policy)
    feature_width, _bytes = persisted_read_group(counts_t)
    widths = list(
        map_feature_read_groups(
            counts_t,
            lambda group: group.featEnd - group.featStart,
            resources=ResourceBudget(8 * 1024**3, 2),
        )
    )
    assert widths
    assert all(width <= feature_width for width in widths)
    assert widths[0] == min(8, feature_width)


def test_consume_uses_persisted_read_group_not_default_unit() -> None:
    from scarf.storage.feature_stream import (
        map_feature_read_groups,
        persisted_read_group,
    )

    values = np.arange(80 * 50_001, dtype=np.uint16).reshape(80, 50_001)
    policy = CountMatrixPolicy(unitBytes=20_000, chunkBytes=2_000)
    counts_t = _counts_t_with_plan(values, policy=policy)
    feature_width, _bytes = persisted_read_group(counts_t)
    metrics: dict[str, object] = {}
    widths = list(
        map_feature_read_groups(
            counts_t,
            lambda group: group.featEnd - group.featStart,
            resources=ResourceBudget(8 * 1024**3, 2),
            metrics=metrics,
        )
    )
    assert widths
    assert all(width <= feature_width for width in widths)
    assert sum(widths) == 50_001, metrics
    assert min(widths) < feature_width or 50_001 % feature_width == 0
    assert int(metrics["featureWidth"]) == feature_width


def test_read_group_rows_counts_the_selected_features_of_each_read_group() -> None:
    from scarf.storage.feature_stream import map_feature_read_groups, read_group_rows

    values = np.arange(40 * 300, dtype=np.uint16).reshape(40, 300)
    counts_t = _counts_t_with_plan(values)
    selected = np.arange(3, 300, 7)
    for feat_idx in (None, selected):
        bounds = list(
            map_feature_read_groups(
                counts_t,
                lambda group: (group.featStart, group.featEnd),
                feat_idx=feat_idx,
                resources=ResourceBudget(8 * 1024**3, 2),
            )
        )
        wanted = np.arange(300) if feat_idx is None else feat_idx
        expected = [
            np.count_nonzero((wanted >= start) & (wanted < end))
            for start, end in bounds
        ]
        assert len(bounds) > 1
        np.testing.assert_array_equal(read_group_rows(counts_t, feat_idx), expected)


def test_persisted_read_group_requires_read_group_bytes() -> None:
    from scarf.storage.feature_stream import persisted_read_group

    values = np.arange(24, dtype=np.uint16).reshape(6, 4)
    counts_t = _counts_t_with_plan(values)
    payload = dict(counts_t.attrs["scarf:countMatrixLayout"])
    payload["readGroup"] = dict(payload["readGroup"])
    payload["readGroup"].pop("readGroupBytes")
    counts_t.attrs["scarf:countMatrixLayout"] = payload
    with pytest.raises(ValueError, match="missing a persisted read group"):
        persisted_read_group(counts_t)


def test_persisted_read_group_rejects_byte_mismatch() -> None:
    from scarf.storage.feature_stream import persisted_read_group

    values = np.arange(24, dtype=np.uint16).reshape(6, 4)
    counts_t = _counts_t_with_plan(values)
    payload = dict(counts_t.attrs["scarf:countMatrixLayout"])
    payload["readGroup"] = dict(payload["readGroup"])
    payload["readGroup"]["readGroupBytes"] = int(payload["readGroup"]["featureWidth"])
    counts_t.attrs["scarf:countMatrixLayout"] = payload
    with pytest.raises(ValueError, match="does not match live geometry"):
        persisted_read_group(counts_t)


def test_map_feature_read_groups_early_close_does_not_block() -> None:
    from scarf.storage.feature_stream import map_feature_read_groups

    values = np.arange(40 * 80, dtype=np.uint16).reshape(40, 80)
    counts_t = _counts_t_with_plan(values)
    iterator = map_feature_read_groups(
        counts_t,
        lambda group: group.featStart,
        resources=ResourceBudget(8 * 1024 * 1024, 2),
        io=StorageIoPolicy(readWorkers=2),
    )
    assert next(iterator) == 0
    iterator.close()


def test_map_feature_cell_bands_early_close_does_not_block() -> None:
    from scarf.storage.feature_stream import map_feature_cell_bands

    values = np.arange(40 * 80, dtype=np.uint16).reshape(40, 80)
    counts_t = _counts_t_with_plan(values)
    bands = map_feature_cell_bands(
        counts_t,
        lambda band: band.featStart,
        resources=ResourceBudget(8 * 1024 * 1024, 2),
        io=StorageIoPolicy(readWorkers=4),
        orderedCompute=True,
    )
    next(bands)
    bands.close()


def test_feature_stream_empty_selection_and_keep_guard() -> None:
    from scarf.storage.feature_stream import (
        map_feature_cell_bands,
        map_feature_read_groups,
        selected_feature_chunk_starts,
        selected_feature_values,
    )

    values = np.arange(20 * 8, dtype=np.uint16).reshape(20, 8)
    counts_t = _counts_t_with_plan(values)
    assert selected_feature_chunk_starts(counts_t, np.array([], dtype=np.int64)) == []
    with pytest.raises(ValueError, match="1-D mask"):
        selected_feature_values(np.zeros((3, 4)), np.ones(2, dtype=bool))
    assert (
        list(
            map_feature_read_groups(
                counts_t,
                lambda group: group.featStart,
                feat_idx=np.array([], dtype=np.int64),
                resources=ResourceBudget(1024 * 1024, 1),
            )
        )
        == []
    )
    for empty in ({"cell_idx": np.array([], dtype=np.int64)}, {"feat_idx": []}):
        assert (
            list(
                map_feature_cell_bands(
                    counts_t,
                    lambda band: band.featStart,
                    resources=ResourceBudget(1024 * 1024, 1),
                    **empty,
                )
            )
            == []
        )


def test_feature_stream_plans_reject_malformed_requests() -> None:
    array = _array()
    kwargs = {
        "featureIndices": np.arange(3),
        "cellIndices": np.arange(12),
        "resources": ResourceBudget(10_000, 1),
        "blockBytes": lambda width: width,
    }
    with pytest.raises(ValueError, match="featureAxis must be 0 or 1"):
        plan_feature_stream(array, featureAxis=2, cellAxis=0, **kwargs)
    with pytest.raises(ValueError, match="must differ"):
        plan_feature_stream(array, featureAxis=1, cellAxis=1, **kwargs)
    with pytest.raises(ValueError, match="chunked two-dimensional"):
        plan_feature_stream(np.zeros((12, 15)), featureAxis=1, cellAxis=0, **kwargs)
    for width in (True, 2.5):
        with pytest.raises(TypeError, match="positive integer"):
            plan_feature_stream(
                array, featureAxis=1, cellAxis=0, requestedBatchSize=width, **kwargs
            )
    with pytest.raises(ValueError, match="greater than zero"):
        plan_feature_stream(
            array, featureAxis=1, cellAxis=0, requestedBatchSize=0, **kwargs
        )
    with pytest.raises(MemoryError, match="Resident data needs"):
        plan_feature_stream(
            array, featureAxis=1, cellAxis=0, residentBytes=10_000, **kwargs
        )
    with pytest.raises(ValueError, match="positive byte count"):
        plan_feature_stream(
            array, featureAxis=1, cellAxis=0, **{**kwargs, "blockBytes": lambda _w: 0}
        )


def test_handoff_waits_for_room_and_stops_waiting_once_closed() -> None:
    import asyncio

    from scarf.storage.feature_stream import _iter_bounded_handoff

    def run(deliver, _stop):  # type: ignore[no-untyped-def]
        async def produce() -> None:
            # One slot: the later items wait until the first is taken, and
            # whichever still waits when the consumer closes stops waiting.
            await asyncio.gather(deliver(1), deliver(2), deliver(3))

        asyncio.run(produce())

    stream = _iter_bounded_handoff(in_flight=1, run=run)
    assert next(stream) == 1
    stream.close()


def test_cell_major_bands_visit_every_feature_group_of_a_band_first() -> None:
    from scarf.storage.feature_stream import map_feature_cell_bands

    values = np.arange(40 * 80, dtype=np.uint16).reshape(40, 80)
    counts_t = _counts_t_with_plan(values)
    order = list(
        map_feature_cell_bands(
            counts_t,
            lambda band: (band.cellStart, band.featStart),
            resources=ResourceBudget(8 * 1024 * 1024, 2),
            cellMajorOrder=True,
        )
    )
    assert order == sorted(order)
    assert len({feat for _cell, feat in order}) > 1


def test_feature_group_ranges_merge_only_adjacent_chunks() -> None:
    from scarf.storage.feature_stream import _feature_group_ranges

    counts_t = _counts_t_with_plan(np.arange(40 * 80, dtype=np.uint16).reshape(40, 80))
    chunk = int(counts_t.chunks[0])
    n_feats = int(counts_t.shape[0])
    assert _feature_group_ranges(counts_t, feat_idx=None, featureWidth=10_000) == [
        (0, n_feats)
    ]
    separate = _feature_group_ranges(counts_t, feat_idx=None, featureWidth=chunk)
    assert separate == [
        (start, min(start + chunk, n_feats)) for start in range(0, n_feats, chunk)
    ]
    gapped = _feature_group_ranges(
        counts_t, feat_idx=np.array([0, 2 * chunk]), featureWidth=10_000
    )
    assert gapped == [(0, chunk), (2 * chunk, min(3 * chunk, n_feats))]


def test_read_groups_are_processed_in_order_one_at_a_time() -> None:
    import threading
    import time

    from scarf.storage.feature_stream import map_feature_read_groups

    values = np.arange(40 * 80, dtype=np.uint16).reshape(40, 80)
    counts_t = _counts_t_with_plan(values)
    lock = threading.Lock()
    live = 0
    peak = 0

    def watch(group):  # type: ignore[no-untyped-def]
        nonlocal live, peak
        with lock:
            live += 1
            peak = max(peak, live)
        time.sleep(0.01)
        with lock:
            live -= 1
        return group.featStart

    metrics: dict[str, object] = {}
    starts = list(
        map_feature_read_groups(
            counts_t,
            watch,
            resources=ResourceBudget(8 * 1024 * 1024, 4),
            io=StorageIoPolicy(readWorkers=8),
            metrics=metrics,
        )
    )
    # One group is processed while the next one is read.
    assert int(metrics["requestedGroupsInFlight"]) == 2
    assert int(metrics["effectiveGroupsInFlight"]) == 2
    assert int(metrics["effectiveComputeWorkers"]) == 1
    assert starts == sorted(starts)
    assert peak == 1


def test_map_feature_read_groups_parallel_matches_sequential() -> None:
    from scarf.storage.feature_stream import map_feature_read_groups

    values = np.arange(120, dtype=np.uint16).reshape(10, 12)
    counts_t = _counts_t_with_plan(values)
    cell_idx = np.array([9, 0, 4, 2])
    kwargs = {
        "cell_idx": cell_idx,
        "feat_idx": np.arange(12),
        "resources": ResourceBudget(8 * 1024 * 1024, 2),
    }
    sequential = {
        group.featStart: np.asarray(group.values).copy()
        for group in map_feature_read_groups(
            counts_t,
            lambda group: group,
            io=StorageIoPolicy(readWorkers=1),
            **kwargs,
        )
    }
    parallel = {
        group.featStart: np.asarray(group.values).copy()
        for group in map_feature_read_groups(
            counts_t,
            lambda group: group,
            io=StorageIoPolicy(readWorkers=3),
            **kwargs,
        )
    }
    assert sequential.keys() == parallel.keys()
    for start, expected in sequential.items():
        np.testing.assert_array_equal(parallel[start], expected)


@pytest.mark.parametrize(
    ("read_workers", "inner_reads", "reason"),
    [(8, 4, None), (64, 10, "2 units in flight hold at most 10 reads each")],
)
def test_map_feature_read_groups_share_the_requested_read_width(
    read_workers: int, inner_reads: int, reason: str | None
) -> None:
    from scarf.storage.feature_stream import map_feature_read_groups

    values = np.arange(100 * 40, dtype=np.uint16).reshape(100, 40)
    counts_t = _counts_t_with_plan(
        values,
        policy=CountMatrixPolicy(unitBytes=2_000, chunkBytes=200),
    )
    resources = ResourceBudget(1024 * 1024, 2)
    metrics: dict[str, object] = {}
    groups = list(
        map_feature_read_groups(
            counts_t,
            lambda group: group,
            resources=resources,
            io=StorageIoPolicy(readWorkers=read_workers),
            metrics=metrics,
        )
    )

    dest = np.empty_like(values.T)
    covered = np.zeros(values.shape[1], dtype=bool)
    for group in groups:
        dest[group.featStart : group.featEnd] = group.values
        covered[group.featStart : group.featEnd] = True
    np.testing.assert_array_equal(dest, values.T)
    assert np.all(covered)
    # Two groups are in flight, and each reads up to half of the requested
    # width, but never more bands than it has.
    assert metrics["requestedGroupsInFlight"] == 2
    assert metrics["effectiveGroupsInFlight"] == 2
    assert metrics["requestedChunkReadsInFlight"] == read_workers
    assert metrics["innerReads"] == inner_reads
    assert metrics["effectiveChunkReadsInFlight"] == 2 * inner_reads
    assert metrics["cellBandCount"] == 10
    assert metrics["reductionReason"] == (
        None if reason is None else f"{2 * inner_reads} reads used because {reason}"
    )
    assert int(metrics["peakHeldBytes"]) <= resources.memoryBytes


def test_map_feature_read_groups_keep_the_requested_reads_in_flight() -> None:
    from scarf.storage.feature_stream import map_feature_read_groups

    from .store_probes import RecordingStore

    # Two read groups of ten bands each, read from a store whose every get
    # waits, so the reads of a group overlap as far as the plan lets them.
    values = np.arange(100 * 40, dtype=np.uint16).reshape(100, 40)
    store = RecordingStore(delay=0.02)
    counts_t = _counts_t_with_plan(
        values,
        policy=CountMatrixPolicy(unitBytes=4_000, chunkBytes=400),
        store=store,
    )
    store.reset()
    starts = list(
        map_feature_read_groups(
            counts_t,
            lambda group: group.featStart,
            resources=ResourceBudget(64 * 1024**2, 2),
            io=StorageIoPolicy(readWorkers=16),
        )
    )

    assert starts == [0, 20]
    # A group used to read at most two bands at once, whatever the request.
    assert 8 < store.max_in_flight_for("get") <= 16


def test_read_group_units_are_sized_by_the_selected_cells() -> None:
    from scarf.storage.feature_stream import (
        map_feature_read_groups,
        persisted_read_group,
        read_group_stream_floor,
    )
    from scarf.storage.geometry import array_geometry

    values = np.arange(100 * 40, dtype=np.uint16).reshape(100, 40)
    counts_t = _counts_t_with_plan(values)
    feature_width, read_group_bytes = persisted_read_group(counts_t)
    geometry = array_geometry(counts_t)
    assert geometry is not None
    cell_idx = np.arange(0, 100, 4)
    metrics: dict[str, object] = {}
    list(
        map_feature_read_groups(
            counts_t,
            lambda group: group,
            cell_idx=cell_idx,
            resources=ResourceBudget(8 * 1024 * 1024, 2),
            metrics=metrics,
        )
    )

    # The destination holds the 25 selected of the 100 stored cells.
    assert read_group_bytes == 100 * feature_width * 2
    assert metrics["unitBytes"] == 25 * feature_width * 2
    band_bytes = int(metrics["cellBandBytes"])
    chunks = -(-feature_width // geometry.axisChunk(0))
    assert metrics["cellBandReadBytes"] == geometry.readBytes(band_bytes, chunks)
    assert metrics["residentBytes"] == 25 * 8 * 2
    assert read_group_stream_floor(counts_t, cell_idx=cell_idx) == (
        int(metrics["unitBytes"])
        + int(metrics["cellBandReadBytes"])
        + int(metrics["residentBytes"])
    )
    assert read_group_stream_floor(counts_t, feat_idx=np.array([], dtype=int)) == 0


def _traced_stream_peak(mapper, counts_t, **kwargs) -> tuple[int, dict[str, object]]:  # type: ignore[no-untyped-def]
    import tracemalloc

    # Load the codecs and pools first, so that only the stream's own buffers
    # are traced.
    list(mapper(counts_t, lambda unit: None, **kwargs))
    metrics: dict[str, object] = {}
    tracemalloc.start()
    try:
        base, _ = tracemalloc.get_traced_memory()
        for _ in mapper(counts_t, lambda unit: None, metrics=metrics, **kwargs):
            pass
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak - base, metrics


@pytest.mark.parametrize("mapper_name", ["read_groups", "cell_bands"])
@pytest.mark.parametrize("budget", [8_000_000, 40_000_000])
def test_feature_streams_reserve_every_traced_buffer(
    mapper_name: str, budget: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scarf.storage import feature_stream

    from .test_async_execution import linger_pool_workers

    mapper = getattr(feature_stream, f"map_feature_{mapper_name}")
    rng = np.random.default_rng(0)
    values = rng.poisson(0.3, size=(100_000, 64)).astype(np.uint16)
    # Four read groups of 16 features over four sharded bands of 25,000 cells.
    counts_t = _counts_t_with_plan(
        values, policy=CountMatrixPolicy(unitBytes=3_200_000, chunkBytes=800_000)
    )
    cell_idx = np.sort(rng.choice(100_000, 70_000, replace=False))
    # Every pool worker lingers after each call, as a descheduled worker does
    # under CPU contention, so a buffer that a worker still references after
    # the stream released it shows up in the trace. One worker runs one codec
    # thread, which drops a chunk it decoded before it decodes the next read's,
    # so Zarr's own buffers stay inside the reservations of the reads in
    # flight.
    linger_pool_workers(monkeypatch, 0.01)

    traced, metrics = _traced_stream_peak(
        mapper, counts_t, cell_idx=cell_idx, resources=ResourceBudget(budget, 1)
    )

    # NumPy and Zarr buffers are traced and the callback allocates nothing, so
    # what the stream allocates must fit the bytes its plan reserves.
    assert metrics["codecWorkers"] == 1
    assert traced <= int(metrics["reservedBytes"]) <= budget


def test_map_feature_cell_bands_parallel_matches_sequential() -> None:
    from scarf.storage.feature_stream import map_feature_cell_bands

    values = np.arange(120, dtype=np.uint16).reshape(10, 12)
    counts_t = _counts_t_with_plan(values)
    cell_idx = np.array([9, 0, 4, 2])

    def collect(groups_in_flight: int) -> list[tuple[int, int, np.ndarray]]:
        collected: list[tuple[int, int, np.ndarray]] = []

        def capture(band):  # type: ignore[no-untyped-def]
            collected.append(
                (
                    band.featStart,
                    band.cellStart,
                    np.asarray(band.values[:, band.selectedLocal]).copy(),
                )
            )
            return None

        list(
            map_feature_cell_bands(
                counts_t,
                capture,
                cell_idx=cell_idx,
                resources=ResourceBudget(8 * 1024 * 1024, 2),
                io=StorageIoPolicy(readWorkers=groups_in_flight),
            )
        )
        return collected

    sequential = collect(1)
    parallel = collect(3)
    assert [(feat, cell) for feat, cell, _values in sequential] == [
        (feat, cell) for feat, cell, _values in parallel
    ]
    for (_feat, _cell, expected), (_pfeat, _pcell, observed) in zip(
        sequential, parallel, strict=True
    ):
        np.testing.assert_array_equal(observed, expected)


def test_hvg_stats_mask_matches_gathered_cells() -> None:
    from scarf.assay.rna import _hvg_stats_gene_major

    values = np.arange(20, dtype=np.uint16).reshape(4, 5)
    selected = np.array([0, 2, 4], dtype=np.int64)
    inv = np.array([1.0, 0.5, 2.0], dtype=np.float64)
    dest = np.array([0, -1, 1, 2], dtype=np.int64)
    gathered_nz = np.zeros(3, dtype=np.float64)
    gathered_s1 = np.zeros(3, dtype=np.float64)
    gathered_s2 = np.zeros(3, dtype=np.float64)
    masked_nz = np.zeros(3, dtype=np.float64)
    masked_s1 = np.zeros(3, dtype=np.float64)
    masked_s2 = np.zeros(3, dtype=np.float64)
    _hvg_stats_gene_major(
        values[:, selected],
        inv,
        1000.0,
        dest,
        gathered_nz,
        gathered_s1,
        gathered_s2,
    )
    _hvg_stats_gene_major(
        values,
        inv,
        1000.0,
        dest,
        masked_nz,
        masked_s1,
        masked_s2,
        selected=selected,
    )
    np.testing.assert_allclose(masked_nz, gathered_nz)
    np.testing.assert_allclose(masked_s1, gathered_s1)
    np.testing.assert_allclose(masked_s2, gathered_s2)


def test_early_close_joins_the_producer_and_surfaces_its_failure() -> None:
    import asyncio

    from scarf.storage.feature_stream import _iter_bounded_handoff

    def run(deliver, _stop):
        async def produce():
            await deliver(1)
            await deliver(2)
            raise ValueError("producer failed")

        asyncio.run(produce())

    stream = _iter_bounded_handoff(in_flight=1, run=run)
    assert next(stream) == 1
    with pytest.raises(ValueError, match="producer failed"):
        stream.close()


@pytest.mark.parametrize("ordered", [False, True])
def test_stream_closes_when_a_unit_starts_after_stop(monkeypatch, ordered) -> None:
    import asyncio
    import threading
    import time
    import types

    import scarf.storage.feature_stream as feature_stream

    counts_t = _counts_t_with_plan(np.arange(40 * 80, dtype=np.uint16).reshape(40, 80))
    stops: list[threading.Event] = []

    class SlowStop(threading.Event):
        def __init__(self) -> None:
            super().__init__()
            stops.append(self)

        def set(self) -> None:
            # Widen the gap between acknowledging an item and stopping.
            time.sleep(0.2)
            super().set()

    class RacingTaskGroup(asyncio.TaskGroup):
        created = 0

        def create_task(self, coro, **kwargs):  # type: ignore[no-untyped-def]
            task = super().create_task(coro, **kwargs)
            RacingTaskGroup.created += 1
            if RacingTaskGroup.created == 2:
                # The scheduler saw no stop request; the unit starts after one.
                stops[0].wait(10)
            return task

    # Only the stream's own scheduler sees the patched classes; Zarr reads
    # start their own task groups.
    monkeypatch.setattr(
        feature_stream,
        "threading",
        types.SimpleNamespace(Event=SlowStop, Thread=threading.Thread),
    )
    patched_asyncio = types.SimpleNamespace(
        **{name: getattr(asyncio, name) for name in dir(asyncio) if name[0] != "_"}
    )
    patched_asyncio.TaskGroup = RacingTaskGroup
    monkeypatch.setattr(feature_stream, "asyncio", patched_asyncio)
    stream = feature_stream.map_feature_cell_bands(
        counts_t,
        lambda band: band.featStart,
        resources=ResourceBudget(64 * 1024**2, 2),
        io=StorageIoPolicy(readWorkers=1),
        orderedCompute=ordered,
    )
    finished = threading.Event()

    def consume() -> None:
        assert next(stream) == 0
        stream.close()
        finished.set()

    consumer = threading.Thread(target=consume, daemon=True)
    consumer.start()
    consumer.join(10)
    assert finished.is_set()


def test_feature_streams_honor_cooperative_shutdown() -> None:
    from scarf.storage.feature_stream import map_feature_read_groups
    from scarf.utils.shutdown import ShutdownRequested, ShutdownToken, shutdown_scope

    counts_t = _counts_t_with_plan(np.arange(40 * 80, dtype=np.uint16).reshape(40, 80))
    token = ShutdownToken()
    processed: list[int] = []

    def process(group):  # type: ignore[no-untyped-def]
        processed.append(group.featStart)
        token.request(reason="operator stop")
        return group.featStart

    with shutdown_scope(token), pytest.raises(ShutdownRequested):
        list(
            map_feature_read_groups(
                counts_t,
                process,
                resources=ResourceBudget(64 * 1024**2, 2),
                io=StorageIoPolicy(readWorkers=1),
            )
        )
    assert len(processed) == 1


def test_feature_stream_reports_reach_the_callers_report_scope() -> None:
    from scarf.storage.execution import execution_report_scope
    from scarf.storage.feature_stream import map_feature_cell_bands

    counts_t = _counts_t_with_plan(np.arange(40 * 80, dtype=np.uint16).reshape(40, 80))
    with execution_report_scope() as reports:
        list(
            map_feature_cell_bands(
                counts_t,
                lambda band: band.featStart,
                resources=ResourceBudget(64 * 1024**2, 2),
            )
        )
    assert [report.unitKind for report in reports] == ["countsTCellBand"]
    assert reports[0].extra["featureGroupCount"] >= 1


@pytest.mark.parametrize("mapper", ["read_groups", "cell_bands"])
@pytest.mark.parametrize("cell", [-1, 6])
def test_feature_streams_refuse_out_of_range_cells(mapper, cell) -> None:
    from scarf.storage import feature_stream

    counts_t = _counts_t_with_plan(np.arange(24, dtype=np.uint16).reshape(6, 4))
    stream = getattr(feature_stream, f"map_feature_{mapper}")
    with pytest.raises(IndexError, match="cell_idx contains an out-of-range index"):
        list(
            stream(
                counts_t,
                lambda unit: unit,
                cell_idx=np.array([0, cell]),
                feat_idx=np.arange(4),
                resources=ResourceBudget(8 * 1024 * 1024, 2),
            )
        )
