import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.storage.async_execution import reset_zarr_runtime_for_tests
from scarf.storage.budget import ResourceBudget
from scarf.storage.feature_stream import plan_feature_stream


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


def test_map_feature_read_groups_preserves_unsorted_cell_order() -> None:
    from scarf.storage.feature_stream import map_feature_read_groups

    store = MemoryStore()
    root = zarr.open_group(store=store, mode="w")
    values = np.arange(24, dtype=np.uint16).reshape(6, 4)
    counts_t = root.create_array(
        "countsT",
        data=values.T,
        chunks=(2, 3),
        shards=(2, 6),
    )
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
    stacked = np.concatenate([group.values for group in loaded], axis=0)
    np.testing.assert_array_equal(stacked, values.T[:, cell_idx])


def test_map_feature_cell_bands_reduces_in_band_order() -> None:
    from scarf.storage.feature_stream import map_feature_cell_bands

    store = MemoryStore()
    root = zarr.open_group(store=store, mode="w")
    values = np.arange(24, dtype=np.uint16).reshape(6, 4)
    counts_t = root.create_array(
        "countsT",
        data=values.T,
        chunks=(2, 3),
        shards=(2, 6),
    )
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


def test_load_feature_strip_and_selected_values_preserve_order() -> None:
    from scarf.storage.feature_stream import (
        load_feature_strip,
        map_feature_read_groups,
        selected_feature_chunk_starts,
        selected_feature_values,
    )
    from scarf.storage.sharding import write_counts_t

    store = MemoryStore()
    root = zarr.open_group(store=store, mode="w")
    n_cells, n_feats = 12, 40
    values = np.arange(n_cells * n_feats, dtype=np.uint16).reshape(n_cells, n_feats)
    counts = root.create_array(
        "counts",
        shape=values.shape,
        chunks=(4, 8),
        dtype=values.dtype,
        fill_value=0,
    )
    counts[:] = values
    write_counts_t(
        counts,
        root,
        resources=ResourceBudget(8 * 1024**3, 2),
        maxShardBytes=192,
    )
    counts_t = root["countsT"]
    starts = selected_feature_chunk_starts(counts_t)
    assert len(starts) > 1
    first = load_feature_strip(counts_t, starts[0])
    keep = np.ones(first.values.shape[0], dtype=bool)
    assert selected_feature_values(first.values, keep) is first.values
    keep[0] = False
    filtered = selected_feature_values(first.values, keep)
    assert filtered.shape[0] == first.values.shape[0] - 1
    seen = list(
        map_feature_read_groups(
            counts_t,
            lambda group: group.featStart,
            resources=ResourceBudget(64 * 1024 * 1024, 2),
            progress="test-progress",
            readGroupChunks=1,
        )
    )
    assert seen == starts


def test_planned_read_group_chunks_uses_the_target_unit() -> None:
    from scarf.storage.feature_stream import planned_read_group_chunks

    array = _array(shape=(40, 50), chunks=(10, 25))
    assert planned_read_group_chunks(array, targetReadUnitBytes=4_000) == 2
    assert planned_read_group_chunks(array, targetReadUnitBytes=1) == 1


def test_map_feature_read_groups_parallel_matches_sequential() -> None:
    from scarf.storage.feature_stream import map_feature_read_groups

    store = MemoryStore()
    root = zarr.open_group(store=store, mode="w")
    values = np.arange(120, dtype=np.uint16).reshape(10, 12)
    counts_t = root.create_array(
        "countsT",
        data=values.T,
        chunks=(3, 5),
        shards=(3, 10),
    )
    cell_idx = np.array([9, 0, 4, 2])
    kwargs = {
        "cell_idx": cell_idx,
        "feat_idx": np.arange(12),
        "resources": ResourceBudget(8 * 1024 * 1024, 2),
        "readGroupChunks": 1,
    }
    sequential = {
        group.featStart: np.asarray(group.values).copy()
        for group in map_feature_read_groups(
            counts_t,
            lambda group: group,
            readGroupsInFlight=1,
            **kwargs,
        )
    }
    parallel = {
        group.featStart: np.asarray(group.values).copy()
        for group in map_feature_read_groups(
            counts_t,
            lambda group: group,
            readGroupsInFlight=3,
            **kwargs,
        )
    }
    assert sequential.keys() == parallel.keys()
    for start, expected in sequential.items():
        np.testing.assert_array_equal(parallel[start], expected)


def test_map_feature_cell_bands_parallel_matches_sequential() -> None:
    from scarf.storage.feature_stream import map_feature_cell_bands

    store = MemoryStore()
    root = zarr.open_group(store=store, mode="w")
    values = np.arange(120, dtype=np.uint16).reshape(10, 12)
    counts_t = root.create_array(
        "countsT",
        data=values.T,
        chunks=(3, 5),
        shards=(3, 10),
    )
    cell_idx = np.array([9, 0, 4, 2])

    def collect(read_groups_in_flight: int) -> list[tuple[int, int, np.ndarray]]:
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
                readGroupChunks=1,
                readGroupsInFlight=read_groups_in_flight,
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
