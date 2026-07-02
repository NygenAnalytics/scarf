"""Benchmark 2: projection/embeddings reducer (z-score + dot with loadings) via
map_blocks, sweeping worker counts.

Mirrors the per-block reducer AnnStream applies during embedding/ANN-fitting
passes (see scarf/ann.py's ``_reduced_blocks``): z-score each block against a
fixed ``mu``/``sigma`` then project with a fixed loadings matrix. Since each
block is computed independently (no cross-block arithmetic), results are
exactly reproducible across worker counts, unlike a sum where combine order
would matter if it changed.

Run:
    uv run python -m tests.performance.bench_projection
    uv run python -m tests.performance.bench_projection --dims 30 --workers 1,2,4,8
"""

import argparse
import math

import numpy as np

from scarf.chunked import ChunkedArray

from ._common import (
    DEFAULT_DIMS,
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


def make_reducer(
    n_cols: int, dims: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal(n_cols)
    sigma = np.abs(rng.standard_normal(n_cols)) + 0.1
    loadings = rng.standard_normal((n_cols, dims))
    return mu, sigma, loadings


def run(args: argparse.Namespace) -> None:
    workers_list = parse_worker_list(args.workers)
    n_shards = math.ceil(args.n_cells / args.shard_rows)
    mu, sigma, loadings = make_reducer(args.n_cols, args.dims)

    print(
        f"dataset: {args.n_cells} x {args.n_cols} float32 -> {args.dims} dims, "
        f"shard_rows={args.shard_rows} ({n_shards} shards)",
        flush=True,
    )

    def reducer(block: np.ndarray) -> np.ndarray:
        return np.asarray(((block - mu) / sigma).dot(loadings))

    with local_zarr_array(args.n_cells, args.n_cols, args.shard_rows) as arr:
        section("projection (z-score + dot)")
        baseline_seconds = 0.0
        baseline_result: np.ndarray | None = None
        for workers in workers_list:
            with resource_budget(workers):
                ca = ChunkedArray(arr, nthreads=workers)

                def op() -> np.ndarray:
                    parts = ca.map_blocks(
                        lambda _i, s, e: reducer(ca._materialize_range(s, e)),
                        nthreads=workers,
                    )
                    return np.vstack(parts)

                result, seconds = best_of(op, n=args.repeats)
            if baseline_result is None:
                baseline_seconds, baseline_result = seconds, result
            print_row(
                f"workers={workers}", seconds, speedup_note(baseline_seconds, seconds)
            )
            if not np.array_equal(baseline_result, result):
                raise AssertionError("projection result changed across worker counts")

    print("\nProjection output identical across worker counts.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_size_args(parser)
    parser.add_argument("--dims", type=int, default=DEFAULT_DIMS)
    add_worker_sweep_arg(parser)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
