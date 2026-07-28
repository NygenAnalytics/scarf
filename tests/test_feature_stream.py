import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.storage.budget import ResourceBudget
from scarf.storage.feature_stream import plan_feature_stream


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
