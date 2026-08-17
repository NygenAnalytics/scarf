from inspect import getsource

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.storage.count_matrix import (
    CountMatrixPolicy,
    create_count_matrix_array,
    load_count_matrix_plan,
    persist_count_matrix_plan,
    plan_count_matrix_pair,
    require_count_matrix_layout,
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
        plan = plan_count_matrix_pair(n_cells, 50_000, "uint16")
        assert _shape(plan, "counts") == expected["counts"]
        assert _shape(plan, "countsT") == expected["countsT"]
        assert plan.readGroup.featureWidth == expected["readGroup"][0]
        assert plan.readGroup.shardsTouched == expected["readGroup"][1]
        assert plan.readGroup.chunksTouched == expected["readGroup"][2]
        assert plan.sourceDecodeAmplification == expected["amp"]
        assert plan.chunksPerShard == 10


def test_five_and_ten_million_rotate_once_locks() -> None:
    five = plan_count_matrix_pair(5_000_000, 50_000, "uint16")
    ten = plan_count_matrix_pair(10_000_000, 50_000, "uint16")
    assert _shape(five, "countsT") == ((100, 500_000), (500, 1_000_000))
    assert five.readGroup.shardsTouched == 5
    assert five.sourceDecodeAmplification == 10.0
    assert _shape(ten, "countsT") == ((50, 1_000_000), (500, 1_000_000))
    assert ten.readGroup.shardsTouched == 10
    assert ten.sourceDecodeAmplification == 10.0


def test_one_hundred_million_stays_at_ten_x_decode() -> None:
    plan = plan_count_matrix_pair(100_000_000, 50_000, "uint16")
    assert _shape(plan, "countsT") == ((5, 1_000_000), (500, 1_000_000))
    assert plan.sourceDecodeAmplification == 10.0


def test_two_hundred_thousand_read_group_spans_two_shards() -> None:
    plan = plan_count_matrix_pair(200_000, 50_000, "uint16")
    assert _shape(plan, "countsT") == ((2_500, 20_000), (5_000, 100_000))
    assert plan.readGroup.shardsTouched == 2
    assert plan.readGroup.chunksTouched == 10
    assert plan.sourceDecodeAmplification == 1.0


def test_cellxgene_width_at_500k_and_1m() -> None:
    for n_cells in (500_000, 1_000_000):
        plan = plan_count_matrix_pair(n_cells, 45_525, "uint16")
        assert plan.counts.shape == (n_cells, 45_525)
        assert plan.countsT.shards[0] % plan.countsT.chunks[0] == 0
        assert plan.countsT.shards[1] % plan.countsT.chunks[1] == 0
        assert plan.readGroup.featureWidth >= 1
        assert plan.readGroup.readGroupBytes > 0


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
    policy = CountMatrixPolicy(unitBytes=1_000, chunkBytes=100)
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


def test_awkward_and_prime_widths_keep_short_last_units() -> None:
    for n_feats in (1, 50_000, 50_001, 50_027):
        plan = plan_count_matrix_pair(100, n_feats, "uint16")
        assert plan.counts.shape == (100, n_feats)
        assert plan.counts.chunks[1] <= n_feats
        assert plan.countsT.shards[0] % plan.countsT.chunks[0] == 0
        assert plan.countsT.shards[1] % plan.countsT.chunks[1] == 0
        raw = int(plan.counts.chunks[0]) * int(plan.counts.chunks[1]) * plan.itemsize
        assert raw <= _CODEC_MAX_BYTES


def test_empty_matrices_persist_a_plan() -> None:
    for n_cells, n_feats in ((0, 12), (8, 0), (0, 0)):
        plan = plan_count_matrix_pair(n_cells, n_feats, "uint16")
        assert plan.counts.shape == (n_cells, n_feats)
        assert plan.countsT.shape == (n_feats, n_cells)
        root = zarr.open_group(store=MemoryStore(), mode="w")
        persist_count_matrix_plan(root, plan)
        stored = load_count_matrix_plan(root)
        assert stored["policy"]["unitBytes"] == 1_000_000_000
        assert stored["fingerprint"] == plan.fingerprint


def test_codec_limit_failures_are_explicit() -> None:
    n_cells = (_CODEC_MAX_BYTES // 2) + 16
    policy = CountMatrixPolicy(
        unitBytes=n_cells * 4,
        chunkBytes=n_cells * 2,
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
    assert stored["policy"]["unitBytes"] == 1_000_000_000
    assert stored["policy"]["chunkBytes"] == 100_000_000
    replay = plan_count_matrix_pair(10_000, 50_000, "uint16")
    validate_count_matrix_pair(replay, expected=plan)


def test_require_count_matrix_layout_rejects_read_group_mismatch() -> None:
    plan = plan_count_matrix_pair(10, 20, "uint16")
    root = zarr.open_group(store=MemoryStore(), mode="w")
    group = root.create_group("RNA")
    counts = create_count_matrix_array(group, "counts", plan.counts)
    counts_t = create_count_matrix_array(group, "countsT", plan.countsT)
    persist_count_matrix_plan(group, plan)
    persist_count_matrix_plan(counts, plan)
    persist_count_matrix_plan(counts_t, plan)
    payload = dict(group.attrs["scarf:countMatrixLayout"])
    payload["readGroup"] = dict(payload["readGroup"])
    payload["readGroup"]["readGroupBytes"] = 1
    for node in (group, counts, counts_t):
        node.attrs["scarf:countMatrixLayout"] = payload
    with pytest.raises(ValueError, match="read group"):
        require_count_matrix_layout(group, counts, counts_t)


def test_old_policy_keys_fail_closed() -> None:
    plan = plan_count_matrix_pair(10_000, 50_000, "uint16")
    root = zarr.open_group(store=MemoryStore(), mode="w")
    persist_count_matrix_plan(root, plan)
    stored = dict(root.attrs["scarf:countMatrixLayout"])
    stored["policy"] = {
        "targetReadUnitBytes": 1_000_000_000,
        "targetChunkBytes": 100_000_000,
    }
    root.attrs["scarf:countMatrixLayout"] = stored
    with pytest.raises(ValueError, match="retired keys"):
        load_count_matrix_plan(root)


def test_planner_does_not_use_a_fixed_feature_envelope() -> None:
    source = getsource(plan_count_matrix_pair)
    assert "featureEnvelope" not in source
    assert "maxCountsTCellBand" not in source
    plan = plan_count_matrix_pair(100, 50_001, "uint16")
    assert plan.counts.shape == (100, 50_001)
