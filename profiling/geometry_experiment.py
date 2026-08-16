"""Cheap 1M/5M-emulating comparison of the three countsT aspect strategies."""

import time
from typing import Any

import numpy as np
import zarr
from zarr.storage import MemoryStore

from scarf.storage.budget import ResourceBudget
from scarf.storage.count_matrix import (
    LAYOUT_STRATEGIES,
    CountMatrixLayoutPolicy,
    LayoutStrategy,
    create_count_matrix_array,
    persist_count_matrix_plan,
    plan_count_matrix_pair,
    plan_layout_candidates,
)
from scarf.storage.sharding import write_counts_t_experimental

SCALE_POLICY = CountMatrixLayoutPolicy(
    targetReadUnitBytes=100_000,
    targetChunkBytes=10_000,
)
SCALE_GENES = 500
SCALE_DTYPE = "uint16"
# 100x smaller cell extents than the 50k-gene / 1GB production lattice.
SCALE_CASES = (
    ("1M-eq", 10_000),
    ("5M-eq", 50_000),
)


def _values(n_cells: int, n_feats: int) -> np.ndarray:
    cells = np.arange(n_cells, dtype=np.uint32)[:, None]
    feats = np.arange(n_feats, dtype=np.uint32)[None, :]
    return ((cells * np.uint32(n_feats) + feats) % np.uint32(65_521)).astype(np.uint16)


def planned_production_table() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n_cells in (1_000_000, 5_000_000, 10_000_000, 100_000_000):
        for strategy, plan in plan_layout_candidates(n_cells, 50_000, "uint16").items():
            rows.append(
                {
                    "cells": n_cells,
                    "strategy": strategy,
                    "countsChunks": plan.counts.chunks,
                    "countsShards": plan.counts.shards,
                    "countsTChunks": plan.countsT.chunks,
                    "countsTShards": plan.countsT.shards,
                    "readGroupGenes": plan.readGroup.featureWidth,
                    "shardsTouched": plan.readGroup.shardsTouched,
                    "chunksTouched": plan.readGroup.chunksTouched,
                    "writeAmp": plan.sourceDecodeAmplification,
                }
            )
    return rows


def run_scaled_strategy(
    n_cells: int,
    strategy: LayoutStrategy,
) -> dict[str, Any]:
    plan = plan_count_matrix_pair(
        n_cells,
        SCALE_GENES,
        SCALE_DTYPE,
        policy=SCALE_POLICY,
        strategy=strategy,
    )
    values = _values(n_cells, SCALE_GENES)
    root = zarr.open_group(store=MemoryStore(), mode="w")
    group = root.create_group("RNA")
    counts = create_count_matrix_array(group, "counts", plan.counts)
    counts[:] = values
    persist_count_matrix_plan(group, plan)
    persist_count_matrix_plan(counts, plan)
    metrics: dict[str, Any] = {}
    started = time.perf_counter()
    counts_t = write_counts_t_experimental(
        counts,
        group,
        policy=SCALE_POLICY,
        strategy=strategy,
        resources=ResourceBudget(64 * 1024 * 1024, 2),
        metrics=metrics,
    )
    write_sec = time.perf_counter() - started
    np.testing.assert_array_equal(np.asarray(counts_t[:]), values.T)

    read_feats = int(plan.readGroup.featureWidth)
    started = time.perf_counter()
    assembled = np.asarray(counts_t[:read_feats, :])
    read_sec = time.perf_counter() - started
    shard_cells = int(plan.counts.shards[0]) if plan.counts.shards else n_cells
    started = time.perf_counter()
    counts_shard = np.asarray(counts[: min(shard_cells, n_cells), :])
    counts_read_sec = time.perf_counter() - started
    logical = n_cells * SCALE_GENES * 2
    decode = int(metrics.get("sourceDecodeBytes", 0))
    return {
        "strategy": strategy,
        "nCells": n_cells,
        "countsTChunks": plan.countsT.chunks,
        "countsTShards": plan.countsT.shards,
        "readGroupGenes": read_feats,
        "shardsTouched": plan.readGroup.shardsTouched,
        "chunksTouched": plan.readGroup.chunksTouched,
        "plannedAmp": plan.sourceDecodeAmplification,
        "observedAmp": decode / logical if logical else 0.0,
        "writeSec": write_sec,
        "readGroupSec": read_sec,
        "countsShardSec": counts_read_sec,
        "readGroupBytes": int(assembled.nbytes),
        "countsShardBytes": int(counts_shard.nbytes),
        "sourceDecodeBytes": decode,
        "destinationCommits": metrics.get("destinationCommits"),
        "peakLedgerBytes": metrics.get("peakLedgerBytes"),
    }


def run_scaled_suite() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, n_cells in SCALE_CASES:
        for strategy in LAYOUT_STRATEGIES:
            row = run_scaled_strategy(n_cells, strategy)
            row["scale"] = label
            rows.append(row)
    return rows


def _print_table(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> None:
    widths = {key: max(len(key), *(len(str(row[key])) for row in rows)) for key in keys}
    header = "  ".join(key.ljust(widths[key]) for key in keys)
    print(header)
    print("  ".join("-" * widths[key] for key in keys))
    for row in rows:
        print("  ".join(str(row[key]).ljust(widths[key]) for key in keys))


def main() -> None:
    print("Production plans at 50k genes, uint16, U=1GB, Q=100MB")
    planned = planned_production_table()
    _print_table(
        planned,
        (
            "cells",
            "strategy",
            "countsTShards",
            "countsTChunks",
            "shardsTouched",
            "chunksTouched",
            "writeAmp",
        ),
    )
    print()
    print("Scaled I/O: U=100KB, Q=10KB, 500 genes, 1M-eq=10k cells, 5M-eq=50k cells")
    measured = run_scaled_suite()
    _print_table(
        measured,
        (
            "scale",
            "strategy",
            "countsTShards",
            "plannedAmp",
            "observedAmp",
            "shardsTouched",
            "writeSec",
            "readGroupSec",
            "peakLedgerBytes",
        ),
    )


if __name__ == "__main__":
    main()
