"""Zarr v3 outer-block and inner-chunk semantics used by count-matrix consumers."""

import asyncio

import numpy as np
from zarr.storage import MemoryStore

from profiling.phase_checks import (
    PHASE0_COUNTS_CHUNKS,
    PHASE0_COUNTS_SHARDS,
    PHASE0_COUNTST_CHUNKS,
    PHASE0_COUNTST_SHARDS,
    fill_synthetic_pair,
    synthetic_count_values,
)


def test_blocks_address_outer_shards_and_clip_the_logical_edge() -> None:
    counts, _counts_t = fill_synthetic_pair(MemoryStore())
    values = synthetic_count_values()

    assert counts.shards == PHASE0_COUNTS_SHARDS
    assert counts.chunks == PHASE0_COUNTS_CHUNKS
    assert np.asarray(counts.blocks[0, 0]).shape == (50, 117)
    assert np.asarray(counts.blocks[0]).shape == (50, 117)
    last = np.asarray(counts.blocks[4, 0])
    assert last.shape == (21, 117)
    np.testing.assert_array_equal(last, values[200:221, :])


def test_ordinary_slices_select_inner_chunks() -> None:
    counts, _counts_t = fill_synthetic_pair(MemoryStore())
    values = synthetic_count_values()
    inner = np.asarray(counts[0:50, 0:20])
    assert inner.shape == (50, 20)
    np.testing.assert_array_equal(inner, values[0:50, 0:20])
    edge = np.asarray(counts[200:221, 100:117])
    assert edge.shape == (21, 17)
    np.testing.assert_array_equal(edge, values[200:221, 100:117])


def test_countst_omitted_block_axis_spans_remaining_shards() -> None:
    _counts, counts_t = fill_synthetic_pair(MemoryStore())
    assert counts_t.shards == PHASE0_COUNTST_SHARDS
    assert counts_t.chunks == PHASE0_COUNTST_CHUNKS
    assert np.asarray(counts_t.blocks[0, 0]).shape == (20, 100)
    assert np.asarray(counts_t.blocks[0]).shape == (20, 221)
    assert np.asarray(counts_t.blocks[5, 2]).shape == (17, 21)


def test_async_read_and_write_roundtrip_preserves_edge_values() -> None:
    counts, counts_t = fill_synthetic_pair(MemoryStore())
    values = synthetic_count_values()

    async def _roundtrip() -> None:
        got = np.asarray(
            await counts.async_array.getitem((slice(200, 221), slice(100, 117)))
        )
        np.testing.assert_array_equal(got, values[200:221, 100:117])
        marker = np.array([[42]], dtype=values.dtype)
        await counts.async_array.setitem((slice(0, 1), slice(0, 1)), marker)
        written = np.asarray(
            await counts.async_array.getitem((slice(0, 1), slice(0, 1)))
        )
        np.testing.assert_array_equal(written, marker)
        np.testing.assert_array_equal(np.asarray(counts_t[:]), values.T)

    asyncio.run(_roundtrip())
