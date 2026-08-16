from inspect import getsource

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.storage.count_matrix import (
    DEFAULT_LAYOUT_STRATEGY,
    CountMatrixLayoutPolicy,
    persist_count_matrix_plan,
    plan_count_matrix_pair,
    plan_layout_candidates,
    validate_count_matrix_pair,
)
from scarf.storage.layout import _CODEC_MAX_BYTES


def _shape(plan, which: str) -> tuple[tuple[int, int], tuple[int, int]]:
    array = plan.counts if which == "counts" else plan.countsT
    return array.chunks, array.shards or ()


def test_uint16_50k_gene_examples_match_the_agreed_geometry() -> None:
    cases = {
        10_000: {
            "counts": ((10_000, 5_000), (10_000, 50_000)),
            "countsT": ((50_000, 1_000), (50_000, 10_000)),
            "readGroup": (50_000, 1, 10),
            "amp": 1.0,
        },
        100_000: {
            "counts": ((10_000, 5_000), (10_000, 50_000)),
            "countsT": ((5_000, 10_000), (5_000, 100_000)),
            "readGroup": (5_000, 1, 10),
            "amp": 1.0,
        },
        500_000: {
            "counts": ((10_000, 5_000), (10_000, 50_000)),
            "countsT": ((1_000, 50_000), (5_000, 100_000)),
            "readGroup": (1_000, 5, 10),
            "amp": 1.0,
        },
        1_000_000: {
            "counts": ((10_000, 5_000), (10_000, 50_000)),
            "countsT": ((500, 100_000), (5_000, 100_000)),
            "readGroup": (500, 10, 10),
            "amp": 1.0,
        },
    }
    for n_cells, expected in cases.items():
        plans = plan_layout_candidates(n_cells, 50_000, "uint16")
        for plan in plans.values():
            assert _shape(plan, "counts") == expected["counts"]
            assert _shape(plan, "countsT") == expected["countsT"]
            assert plan.readGroup.featureWidth == expected["readGroup"][0]
            assert plan.readGroup.shardsTouched == expected["readGroup"][1]
            assert plan.readGroup.chunksTouched == expected["readGroup"][2]
            assert plan.sourceDecodeAmplification == expected["amp"]
            assert plan.chunksPerShard == 10


def test_five_and_ten_million_strategies_diverge() -> None:
    five = plan_layout_candidates(5_000_000, 50_000, "uint16")
    ten = plan_layout_candidates(10_000_000, 50_000, "uint16")

    assert _shape(five["keepAspect"], "countsT") == ((100, 100_000), (5_000, 100_000))
    assert five["keepAspect"].readGroup.shardsTouched == 50
    assert five["keepAspect"].sourceDecodeAmplification == 1.0

    assert _shape(five["rotateOnce"], "countsT") == ((100, 500_000), (500, 1_000_000))
    assert _shape(five["rotateEach"], "countsT") == ((100, 500_000), (500, 1_000_000))
    assert five["rotateOnce"].readGroup.shardsTouched == 5
    assert five["rotateOnce"].sourceDecodeAmplification == 10.0

    assert _shape(ten["keepAspect"], "countsT") == ((50, 100_000), (5_000, 100_000))
    assert ten["keepAspect"].readGroup.shardsTouched == 100
    assert _shape(ten["rotateOnce"], "countsT") == ((50, 1_000_000), (500, 1_000_000))
    assert ten["rotateOnce"].readGroup.shardsTouched == 10
    assert ten["rotateOnce"].sourceDecodeAmplification == 10.0


def test_one_hundred_million_separates_rotate_once_from_rotate_each() -> None:
    plans = plan_layout_candidates(100_000_000, 50_000, "uint16")
    assert _shape(plans["keepAspect"], "countsT") == ((5, 100_000), (5_000, 100_000))
    assert plans["keepAspect"].readGroup.shardsTouched == 1_000
    assert _shape(plans["rotateOnce"], "countsT") == ((5, 1_000_000), (500, 1_000_000))
    assert plans["rotateOnce"].sourceDecodeAmplification == 10.0
    assert _shape(plans["rotateEach"], "countsT") == ((5, 10_000_000), (50, 10_000_000))
    assert plans["rotateEach"].sourceDecodeAmplification == 100.0
    assert plans["rotateEach"].readGroup.shardsTouched == 10


def test_two_hundred_thousand_read_group_spans_two_shards() -> None:
    plan = plan_count_matrix_pair(200_000, 50_000, "uint16")
    assert _shape(plan, "countsT") == ((2_500, 20_000), (5_000, 100_000))
    assert plan.readGroup.shardsTouched == 2
    assert plan.readGroup.chunksTouched == 10
    assert plan.sourceDecodeAmplification == 1.0


def test_gene_width_and_dtype_change_the_cell_band() -> None:
    uint16 = plan_count_matrix_pair(1_000_000, 25_000, "uint16")
    uint32 = plan_count_matrix_pair(1_000_000, 25_000, "uint32")
    wide = plan_count_matrix_pair(1_000_000, 100_000, "uint16")
    assert uint16.counts.chunks[0] == 20_000
    assert uint32.counts.chunks[0] == 10_000
    assert wide.counts.chunks[0] == 5_000
    assert uint16.countsT.shards == (2_500, 200_000)
    assert wide.counts.chunks == (5_000, 10_000)
    assert wide.countsT.shards == (1_000, 500_000)
    assert wide.sourceDecodeAmplification == 10.0


def test_tiny_policy_is_deterministic() -> None:
    policy = CountMatrixLayoutPolicy(
        targetReadUnitBytes=1_000,
        targetChunkBytes=100,
    )
    first = plan_count_matrix_pair(17, 40, "uint16", policy=policy)
    second = plan_count_matrix_pair(17, 40, "uint16", policy=policy)
    validate_count_matrix_pair(first, expected=second)
    assert first.counts.shards[1] == 40
    assert first.countsT.shards[0] % first.countsT.chunks[0] == 0
    assert first.countsT.shards[1] % first.countsT.chunks[1] == 0


def test_supported_dtypes_are_deterministic() -> None:
    fingerprints = set()
    for dtype in ("uint8", "uint16", "uint32", "int32", "float32"):
        plan = plan_count_matrix_pair(1_000, 1_000, dtype)
        fingerprints.add(plan.fingerprint)
        assert plan.itemsize == np.dtype(dtype).itemsize
    assert len(fingerprints) == 5


def test_small_matrices_use_actual_extents() -> None:
    plan = plan_count_matrix_pair(8, 7, "uint16")
    assert plan.counts.shape == (8, 7)
    assert plan.counts.shards[1] == 7
    assert plan.readGroup.featureWidth == 7


def test_codec_limit_failures_are_explicit() -> None:
    n_cells = (_CODEC_MAX_BYTES // 2) + 16
    policy = CountMatrixLayoutPolicy(
        targetReadUnitBytes=n_cells * 4,
        targetChunkBytes=n_cells * 2,
    )
    with pytest.raises(ValueError, match="codec input"):
        plan_count_matrix_pair(n_cells, 1, "uint16", policy=policy)


def test_destination_buffer_is_one_physical_shard() -> None:
    plan = plan_count_matrix_pair(1_000_000, 50_000, "uint16")
    assert plan.destinationBufferBytes == 100_000 * 5_000 * 2
    assert plan.sourceBufferBytes == 10_000 * 5_000 * 2
    assert plan.sourceDecodeAmplification == 1.0


def test_policy_metadata_round_trip() -> None:
    plan = plan_count_matrix_pair(10_000, 50_000, "uint16")
    root = zarr.open_group(store=MemoryStore(), mode="w")
    persist_count_matrix_plan(root, plan)
    stored = root.attrs["scarf:countMatrixLayout"]
    assert stored["fingerprint"] == plan.fingerprint
    assert stored["policy"]["targetReadUnitBytes"] == 1_000_000_000
    assert stored["strategy"] == "rotateOnce"
    replay = plan_count_matrix_pair(10_000, 50_000, "uint16")
    validate_count_matrix_pair(replay, expected=plan)


def test_planner_does_not_use_a_fixed_feature_envelope() -> None:
    source = getsource(plan_count_matrix_pair)
    assert "featureEnvelope" not in source
    assert "maxCountsTCellBand" not in source
    plan = plan_count_matrix_pair(100, 50_001, "uint16")
    assert plan.counts.shape == (100, 50_001)


def test_default_strategy_is_rotate_once() -> None:
    assert DEFAULT_LAYOUT_STRATEGY == "rotateOnce"
    plan = plan_count_matrix_pair(5_000_000, 50_000, "uint16")
    assert plan.strategy == "rotateOnce"
    assert plan.sourceDecodeAmplification == 10.0
    assert plan.readGroup.shardsTouched == 5


def test_product_layout_stays_on_the_current_branch() -> None:
    from scarf.storage.count_matrix import (
        accepted_layout_branch,
        apply_recorded_layout_branch,
        override_accepted_layout_branch,
        uses_experimental_product_layout,
    )

    assert accepted_layout_branch() == "current"
    assert uses_experimental_product_layout() is False
    assert apply_recorded_layout_branch("B") == "current"
    assert apply_recorded_layout_branch("E") == "current"
    with override_accepted_layout_branch("A"):
        assert accepted_layout_branch() == "A"
        assert uses_experimental_product_layout() is True
    assert accepted_layout_branch() == "current"
    with pytest.raises(ValueError, match="unsupported layout branch"):
        apply_recorded_layout_branch("Z")
