"""Benchmark 1: ChunkedArray axis=0 reductions (sum/var/mean_and_std) via map_shards.

Sweeps worker counts and reports wall-time scaling for a synthetic local Zarr
array, then asserts every reduction is bit-identical across worker counts
(map_shards preserves the collect-then-combine order regardless of thread
count).

Run:
    uv run python -m tests.performance.bench_reductions
    uv run python -m tests.performance.bench_reductions --n-cells 100000 --workers 1,2,4,8,16
"""

import argparse
import math
from collections.abc import Callable

import numpy as np

from scarf.chunked import ChunkedArray

from ._common import (
    add_size_args,
    add_worker_sweep_arg,
    best_of,
    local_zarr_array,
    parse_worker_list,
    print_row,
    resource_budget,
    section,
    speedup_note,
)

type Reduction = tuple[np.ndarray, ...]


def _as_tuple(value: object) -> Reduction:
    return value if isinstance(value, tuple) else (np.asarray(value),)


def _assert_identical(op_name: str, baseline: Reduction, result: Reduction) -> None:
    for a, b in zip(baseline, result):
        if not np.array_equal(a, b):
            raise AssertionError(f"{op_name} result changed across worker counts")


def _reduction_ops(
    ca_by_workers: Callable[[int], ChunkedArray],
) -> dict[str, Callable[[int], object]]:
    return {
        "sum(axis=0)": lambda w: ca_by_workers(w).sum(axis=0).compute(w),
        "var(axis=0)": lambda w: ca_by_workers(w).var(axis=0).compute(w),
        "mean_and_std": lambda w: ca_by_workers(w).mean_and_std(nthreads=w),
    }


def run(args: argparse.Namespace) -> None:
    workers_list = parse_worker_list(args.workers)
    n_shards = math.ceil(args.n_cells / args.shard_rows)
    print(
        f"dataset: {args.n_cells} x {args.n_cols} float32, "
        f"shard_rows={args.shard_rows} ({n_shards} shards)",
        flush=True,
    )

    with local_zarr_array(args.n_cells, args.n_cols, args.shard_rows) as arr:

        def make_ca(workers: int) -> ChunkedArray:
            return ChunkedArray(arr, nthreads=workers)

        for op_name, op in _reduction_ops(make_ca).items():
            section(f"axis=0 {op_name}")
            baseline_seconds = 0.0
            baseline_result: Reduction | None = None
            for workers in workers_list:
                with resource_budget(workers):
                    result, seconds = best_of(lambda w=workers: op(w), n=args.repeats)
                result = _as_tuple(result)
                if baseline_result is None:
                    baseline_seconds, baseline_result = seconds, result
                print_row(
                    f"workers={workers}",
                    seconds,
                    speedup_note(baseline_seconds, seconds),
                )
                _assert_identical(op_name, baseline_result, result)

    print("\nAll reductions bit-identical across worker counts.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_size_args(parser)
    add_worker_sweep_arg(parser)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
