"""Benchmark 3: write_dense_in_shard_rows (parallel produce + single-threaded
writer) vs a naive fully-serial baseline.

Run:
    uv run python -m tests.performance.bench_write
    uv run python -m tests.performance.bench_write --workers 8
"""

import argparse
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import zarr

from scarf.storage.zarr_store import iter_shard_row_slices, write_dense_in_shard_rows

from ._common import (
    add_size_args,
    best_of,
    print_row,
    resource_budget,
    section,
    synthetic_matrix,
)


def naive_serial_write(
    dst: zarr.Array, produce: Callable[[int, int], np.ndarray], shard_rows: int
) -> None:
    for start, end in iter_shard_row_slices(dst.shape[0], shard_rows):
        dst[start:end, :] = produce(start, end)


def run(args: argparse.Namespace) -> None:
    print(
        f"dataset: {args.n_cells} x {args.n_cols} float32, "
        f"shard_rows={args.shard_rows}, produce workers={args.workers}",
        flush=True,
    )
    source = synthetic_matrix(args.n_cells, args.n_cols, seed=0)

    def produce(start: int, end: int) -> np.ndarray:
        block = source[start:end, :]
        return np.sqrt(np.abs(block)) * 2.0 - 1.0

    section("write_dense_in_shard_rows vs naive serial")
    seconds_by_label: dict[str, float] = {}
    with tempfile.TemporaryDirectory(prefix="scarf-perf-write-") as tmp:
        for label, run_write in (
            (
                "naive serial",
                lambda dst: naive_serial_write(dst, produce, args.shard_rows),
            ),
            (
                "parallel produce + 1 writer",
                lambda dst: write_dense_in_shard_rows(
                    dst, produce, shard_rows=args.shard_rows
                ),
            ),
        ):
            store_path = str(Path(tmp) / label.replace(" ", "_"))
            root = zarr.open_group(store_path, mode="w")
            dst = root.create_array(
                "data",
                shape=(args.n_cells, args.n_cols),
                chunks=(args.shard_rows, args.n_cols),
                dtype="float32",
            )
            with resource_budget(args.workers):
                _, seconds = best_of(lambda dst=dst: run_write(dst), n=args.repeats)
            seconds_by_label[label] = seconds
            print_row(label, seconds)

    base = seconds_by_label["naive serial"]
    parallel = seconds_by_label["parallel produce + 1 writer"]
    print(f"\nspeedup: x{base / parallel:.2f}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_size_args(parser)
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Worker budget for the parallel-produce path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
