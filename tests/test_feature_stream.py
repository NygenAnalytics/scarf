import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.storage.async_execution import reset_zarr_runtime_for_tests
from scarf.storage.budget import ResourceBudget
from scarf.storage.count_matrix import (
    CountMatrixPolicy,
    persist_count_matrix_plan,
    plan_count_matrix_pair,
)
from scarf.storage.feature_stream import plan_feature_stream
from scarf.storage.io_policy import StorageIoPolicy


def setup_function() -> None:
    reset_zarr_runtime_for_tests()


def teardown_function() -> None:
    reset_zarr_runtime_for_tests()


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
    assert plan.readWorkers == 2
    assert in_flight + 100 + decode <= 650
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
) -> zarr.Array:
    resolved = policy or CountMatrixPolicy(unitBytes=2_000, chunkBytes=200)
    plan = plan_count_matrix_pair(
        values.shape[0],
        values.shape[1],
        values.dtype,
        policy=resolved,
    )
    root = zarr.open_group(store=MemoryStore(), mode="w")
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


def test_cell_band_admission_charges_the_live_band_buffer() -> None:
    from scarf.storage.feature_stream import map_feature_cell_bands

    values = np.arange(120, dtype=np.uint16).reshape(10, 12)
    counts_t = _counts_t_with_plan(values)
    metrics: dict[str, object] = {}

    list(
        map_feature_cell_bands(
            counts_t,
            lambda _band: None,
            resources=ResourceBudget(400, 2),
            io=StorageIoPolicy(readWorkers=4),
            metrics=metrics,
        )
    )

    assert metrics["cellBandBytes"] == 200
    assert metrics["unitBytes"] == 200
    assert metrics["readGroupBytes"] == 200
    assert metrics["actualReadWorkers"] == 2


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
        orderedCompute=True,
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
    with pytest.raises(ValueError, match="extraItemsize"):
        list(
            map_feature_read_groups(
                counts_t,
                lambda group: group.featStart,
                resources=ResourceBudget(1024 * 1024, 1),
                extraItemsize=-1,
            )
        )
    assert (
        list(
            map_feature_cell_bands(
                counts_t,
                lambda band: band.featStart,
                cell_idx=np.array([], dtype=np.int64),
                resources=ResourceBudget(1024 * 1024, 1),
            )
        )
        == []
    )


def test_feature_group_ranges_skips_overlap_and_merges_adjacent() -> None:
    from scarf.storage.feature_stream import _feature_group_ranges

    overlapping = _counts_t_with_plan(
        np.arange(40 * 80, dtype=np.uint16).reshape(40, 80)
    )
    merged = _feature_group_ranges(
        overlapping,
        feat_idx=None,
        feat_starts=[0, 0, overlapping.chunks[0], 2 * overlapping.chunks[0]],
        featureWidth=10_000,
    )
    expected_end = min(3 * int(overlapping.chunks[0]), int(overlapping.shape[0]))
    assert merged == [(0, expected_end)]


def test_groups_in_flight_is_clamped_and_handoff_is_bounded() -> None:
    from scarf.storage.feature_stream import map_feature_read_groups

    values = np.arange(40 * 80, dtype=np.uint16).reshape(40, 80)
    counts_t = _counts_t_with_plan(values)
    live = 0
    peak = 0

    def watch(group):  # type: ignore[no-untyped-def]
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        live -= 1
        return group.featStart

    metrics: dict[str, object] = {}
    list(
        map_feature_read_groups(
            counts_t,
            watch,
            resources=ResourceBudget(8 * 1024 * 1024, 4),
            io=StorageIoPolicy(readWorkers=8),
            metrics=metrics,
        )
    )
    assert int(metrics["requestedGroupsInFlight"]) == 8
    assert int(metrics["effectiveGroupsInFlight"]) <= 8
    assert peak <= int(metrics["effectiveGroupsInFlight"])


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


def test_map_feature_read_groups_uses_bounded_inner_reads() -> None:
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
            io=StorageIoPolicy(readWorkers=8),
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
    assert metrics["requestedGroupsInFlight"] == 4
    assert metrics["effectiveGroupsInFlight"] == 4
    assert metrics["requestedChunkReadsInFlight"] == 8
    assert metrics["effectiveChunkReadsInFlight"] == 8
    assert metrics["innerReads"] == 2
    assert metrics["cellBandCount"] == 10
    assert int(metrics["peakHeldBytes"]) <= resources.memoryBytes


def test_map_feature_read_groups_charges_extra_itemsize() -> None:
    from scarf.storage.feature_stream import map_feature_read_groups

    values = np.arange(24, dtype=np.uint16).reshape(6, 4)
    counts_t = _counts_t_with_plan(values)
    resources = ResourceBudget(8 * 1024 * 1024, 2)
    baseline: dict[str, object] = {}
    extra: dict[str, object] = {}
    list(
        map_feature_read_groups(
            counts_t,
            lambda group: group,
            resources=resources,
            metrics=baseline,
        )
    )
    list(
        map_feature_read_groups(
            counts_t,
            lambda group: group,
            resources=resources,
            metrics=extra,
            extraItemsize=12,
        )
    )
    feature_width = int(extra["featureWidth"])
    assert int(extra["unitBytes"]) == int(baseline["unitBytes"]) + (
        12 * feature_width * values.shape[0]
    )
    assert int(extra["peakHeldBytes"]) <= resources.memoryBytes
    assert int(extra["peakHeldBytes"]) >= int(baseline["peakHeldBytes"])


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
