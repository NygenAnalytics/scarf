"""Budget-admitted feature-shard consume resolver and helpers."""

import numpy as np
import pytest
import zarr

from scarf.storage.budget import ResourceBudget
from scarf.storage.feature_shards import (
    admitted_pending_limit,
    estimate_feature_shard_bytes,
    load_feature_shard,
    map_feature_shards,
    plan_feature_shard_consume_for_array,
    resolve_feature_shard_consume,
    selected_strip_starts,
    shard_values_for_selection,
    strip_feature_starts,
)
from scarf.storage.sharding import write_counts_t


def _strip_counts_t(
    n_feats: int = 40,
    n_cells: int = 12,
    *,
    maxShardBytes: int | None = None,
) -> zarr.Array:
    store = zarr.storage.MemoryStore()
    root = zarr.open_group(store=store, mode="w", zarr_format=3)
    values = np.arange(n_cells * n_feats, dtype=np.uint16).reshape(n_cells, n_feats)
    counts = root.create_array(
        "counts",
        shape=values.shape,
        chunks=(min(4, n_cells), min(8, n_feats)),
        dtype=values.dtype,
        fill_value=0,
    )
    counts[:] = values
    write_kwargs: dict[str, int] = {}
    if maxShardBytes is not None:
        write_kwargs["maxShardBytes"] = maxShardBytes
    write_counts_t(
        counts,
        root,
        resources=ResourceBudget(8 * 1024**3, 2),
        **write_kwargs,
    )
    return root["countsT"]


def test_resolve_feature_shard_consume_argument_env_default_precedence(monkeypatch):
    monkeypatch.setenv("SCARF_FEATURE_SHARD_PREFETCH_DEPTH", "3")
    monkeypatch.setenv("SCARF_FEATURE_SHARD_READ_CONCURRENCY", "4")
    monkeypatch.setenv("SCARF_FEATURE_SHARD_NUMBA_THREADS", "5")
    env_plan = resolve_feature_shard_consume(nthreads=8)
    assert env_plan.prefetchDepth == 3
    assert env_plan.readConcurrency == 4  # clamped to inFlight=4
    assert env_plan.numbaThreads == 5
    assert env_plan.source == "env"

    arg_plan = resolve_feature_shard_consume(
        nthreads=8,
        prefetchDepth=1,
        readConcurrency=2,
        numbaThreads=3,
    )
    assert arg_plan.prefetchDepth == 1
    assert arg_plan.readConcurrency == 2
    assert arg_plan.numbaThreads == 3
    assert arg_plan.source == "argument"

    monkeypatch.delenv("SCARF_FEATURE_SHARD_PREFETCH_DEPTH")
    monkeypatch.delenv("SCARF_FEATURE_SHARD_READ_CONCURRENCY")
    monkeypatch.delenv("SCARF_FEATURE_SHARD_NUMBA_THREADS")
    default_plan = resolve_feature_shard_consume(nthreads=8)
    assert default_plan.prefetchDepth == 1
    assert default_plan.readConcurrency == 2
    assert default_plan.numbaThreads == 4
    assert default_plan.source == "default"


def test_resolve_feature_shard_consume_rejects_invalid_values():
    with pytest.raises(ValueError, match="prefetchDepth"):
        resolve_feature_shard_consume(nthreads=4, prefetchDepth=-1)
    with pytest.raises(ValueError, match="readConcurrency"):
        resolve_feature_shard_consume(nthreads=4, readConcurrency=0)
    with pytest.raises(ValueError, match="numbaThreads"):
        resolve_feature_shard_consume(nthreads=4, numbaThreads=0)


def test_resolve_feature_shard_consume_memory_and_worker_caps():
    shard_bytes = 10 * 1024 * 1024
    tight = resolve_feature_shard_consume(
        nthreads=8,
        prefetchDepth=4,
        readConcurrency=8,
        resources=ResourceBudget(14 * 1024 * 1024, 8),
        shardBytes=shard_bytes,
    )
    assert tight.inFlight == 1
    assert tight.prefetchDepth == 0
    assert tight.readConcurrency == 1
    assert tight.estimatedResidentBytes >= shard_bytes

    with pytest.raises(MemoryError, match="One feature shard"):
        resolve_feature_shard_consume(
            nthreads=4,
            resources=ResourceBudget(1024, 4),
            shardBytes=10_000,
            residentBytes=0,
        )


def test_selected_strip_starts_and_no_copy_full_selection():
    counts_t = _strip_counts_t()
    starts = strip_feature_starts(counts_t)
    assert starts
    gene_strip = int(counts_t.chunks[0])
    feat_idx = np.arange(min(gene_strip, counts_t.shape[0]), dtype=np.int64)
    pruned = selected_strip_starts(counts_t, feat_idx)
    assert pruned == [starts[0]]

    shard = load_feature_shard(counts_t, starts[0])
    keep = np.ones(shard.values.shape[0], dtype=bool)
    viewed = shard_values_for_selection(shard.values, keep)
    assert viewed is shard.values
    keep[0] = False
    filtered = shard_values_for_selection(shard.values, keep)
    assert filtered is not shard.values
    assert filtered.shape[0] == shard.values.shape[0] - 1


def test_map_feature_shards_bounds_pending_and_runs_progress():
    counts_t = _strip_counts_t(n_feats=40, n_cells=12, maxShardBytes=192)
    expected = strip_feature_starts(counts_t)
    assert len(expected) > 1
    plan = plan_feature_shard_consume_for_array(
        counts_t,
        resources=ResourceBudget(64 * 1024 * 1024, 4),
        prefetchDepth=1,
        readConcurrency=4,
    )
    assert plan.readConcurrency <= plan.inFlight
    seen = []

    def process(shard):
        seen.append((shard.featStart, shard.featEnd))
        return shard.featStart

    starts = list(
        map_feature_shards(
            counts_t,
            process,
            plan=plan,
            progress="test-progress",
        )
    )
    assert starts == [s for s, _ in seen]
    assert starts == expected


def test_map_feature_shards_visits_every_strip_when_in_flight_is_one():
    counts_t = _strip_counts_t(n_feats=40, n_cells=12, maxShardBytes=192)
    expected = strip_feature_starts(counts_t)
    assert len(expected) > 1
    plan = plan_feature_shard_consume_for_array(
        counts_t,
        resources=ResourceBudget(64 * 1024 * 1024, 2),
        prefetchDepth=0,
        readConcurrency=1,
    )
    assert plan.inFlight == 1
    seen = list(map_feature_shards(counts_t, lambda shard: shard.featStart, plan=plan))
    assert seen == expected


def test_estimate_feature_shard_bytes_charges_full_decode_plus_gather():
    full = estimate_feature_shard_bytes(geneStrip=10, nCells=100, itemsize=2)
    subset = estimate_feature_shard_bytes(
        geneStrip=10, nCells=100, itemsize=2, selectedCells=20
    )
    assert full == 10 * 100 * 2
    assert subset == (10 * 100 * 2) + (10 * 20 * 2)


def test_admitted_pending_limit_leaves_room_for_current_shard():
    assert admitted_pending_limit(2, holdingCurrent=False) == 2
    assert admitted_pending_limit(2, holdingCurrent=True) == 1
    assert admitted_pending_limit(1, holdingCurrent=True) == 0


def test_plan_does_not_undercharge_when_cells_are_subset():
    counts_t = _strip_counts_t(n_feats=40, n_cells=20)
    gene_strip = int(counts_t.chunks[0])
    itemsize = int(np.dtype(counts_t.dtype).itemsize)
    decode = gene_strip * 20 * itemsize
    subset = plan_feature_shard_consume_for_array(
        counts_t,
        resources=ResourceBudget(64 * 1024 * 1024, 2),
        cell_idx=np.arange(5),
    )
    assert subset.estimatedResidentBytes >= decode
