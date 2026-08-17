"""Zarr v3 outer-block and inner-chunk semantics used by count-matrix consumers."""

import asyncio

import numpy as np
import zarr
from zarr.storage import MemoryStore

from scarf.storage.count_matrix import (
    CountMatrixPolicy,
    persist_count_matrix_plan,
    plan_count_matrix_pair,
)


def _synthetic_values(n_cells: int = 221, n_feats: int = 117) -> np.ndarray:
    return (
        np.arange(n_cells * n_feats, dtype=np.uint16).reshape(n_cells, n_feats)
        % np.iinfo(np.uint16).max
    )


def _fill_synthetic_pair(store: MemoryStore) -> tuple[zarr.Array, zarr.Array]:
    values = _synthetic_values()
    policy = CountMatrixPolicy(unitBytes=20_000, chunkBytes=2_000)
    plan = plan_count_matrix_pair(
        values.shape[0],
        values.shape[1],
        values.dtype,
        policy=policy,
    )
    root = zarr.open_group(store=store, mode="w")
    group = root.create_group("RNA")
    counts = group.create_array(
        "counts",
        shape=plan.counts.shape,
        chunks=plan.counts.chunks,
        shards=plan.counts.shards,
        dtype=values.dtype,
        overwrite=True,
    )
    counts_t = group.create_array(
        "countsT",
        shape=plan.countsT.shape,
        chunks=plan.countsT.chunks,
        shards=plan.countsT.shards,
        dtype=values.dtype,
        overwrite=True,
    )
    counts[:] = values
    counts_t[:] = values.T
    persist_count_matrix_plan(group, plan)
    persist_count_matrix_plan(counts, plan)
    persist_count_matrix_plan(counts_t, plan)
    return counts, counts_t


def test_blocks_address_outer_shards_and_clip_the_logical_edge() -> None:
    counts, _counts_t = _fill_synthetic_pair(MemoryStore())
    values = _synthetic_values()
    shard_rows = int(counts.shards[0])
    n_cells = int(counts.shape[0])
    n_feats = int(counts.shape[1])
    last_start = (n_cells // shard_rows) * shard_rows
    last_rows = n_cells - last_start
    assert last_rows < shard_rows
    assert np.asarray(counts.blocks[0, 0]).shape[1] == n_feats
    last = np.asarray(counts.blocks[last_start // shard_rows, 0])
    assert last.shape == (last_rows, n_feats)
    np.testing.assert_array_equal(last, values[last_start:, :])


def test_ordinary_slices_select_inner_chunks() -> None:
    counts, _counts_t = _fill_synthetic_pair(MemoryStore())
    values = _synthetic_values()
    chunk_rows, chunk_cols = (int(v) for v in counts.chunks)
    inner = np.asarray(counts[0:chunk_rows, 0:chunk_cols])
    assert inner.shape == (chunk_rows, chunk_cols)
    np.testing.assert_array_equal(inner, values[0:chunk_rows, 0:chunk_cols])
    n_cells, n_feats = (int(v) for v in counts.shape)
    edge = np.asarray(
        counts[
            n_cells - last_len(n_cells, chunk_rows) :,
            n_feats - last_len(n_feats, chunk_cols) :,
        ]
    )
    np.testing.assert_array_equal(
        edge,
        values[n_cells - edge.shape[0] :, n_feats - edge.shape[1] :],
    )


def last_len(axis: int, step: int) -> int:
    remainder = axis % step
    return step if remainder == 0 else remainder


def test_countst_omitted_block_axis_spans_remaining_shards() -> None:
    _counts, counts_t = _fill_synthetic_pair(MemoryStore())
    shard_feats, shard_cells = (int(v) for v in counts_t.shards)
    n_feats, n_cells = (int(v) for v in counts_t.shape)
    first = np.asarray(counts_t.blocks[0])
    assert first.shape[0] == min(shard_feats, n_feats)
    assert first.shape[1] == n_cells
    last_feat_idx = (n_feats - 1) // shard_feats
    last_cell_idx = (n_cells - 1) // shard_cells
    last = np.asarray(counts_t.blocks[last_feat_idx, last_cell_idx])
    assert last.shape[0] == last_len(n_feats, shard_feats)
    assert last.shape[1] == last_len(n_cells, shard_cells)


def test_async_read_and_write_roundtrip_preserves_edge_values() -> None:
    counts, counts_t = _fill_synthetic_pair(MemoryStore())
    values = _synthetic_values()
    n_cells, n_feats = (int(v) for v in counts.shape)
    chunk_rows, chunk_cols = (int(v) for v in counts.chunks)
    feat_start = n_feats - last_len(n_feats, chunk_cols)
    cell_start = n_cells - last_len(n_cells, chunk_rows)

    async def _roundtrip() -> None:
        got = np.asarray(
            await counts.async_array.getitem(
                (slice(cell_start, n_cells), slice(feat_start, n_feats))
            )
        )
        np.testing.assert_array_equal(got, values[cell_start:, feat_start:])
        marker = np.array([[42]], dtype=values.dtype)
        await counts.async_array.setitem((slice(0, 1), slice(0, 1)), marker)
        written = np.asarray(
            await counts.async_array.getitem((slice(0, 1), slice(0, 1)))
        )
        np.testing.assert_array_equal(written, marker)
        expected_t = values.T.copy()
        expected_t[0, 0] = 42
        np.testing.assert_array_equal(np.asarray(counts_t[:]), values.T)

    asyncio.run(_roundtrip())
