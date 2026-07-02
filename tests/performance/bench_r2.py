"""Benchmark 4 (optional, R2): read a sharded array from Cloudflare R2 and
sweep across-shard depth vs io_concurrency, showing that the budget split in
scarf.storage.budget.shard_parallelism bounds request fan-out instead of
multiplying to workers-squared in-flight requests.

Credentials follow tests/r2_profile.py's convention (reads tests/.env):
    SCARF_R2_ENDPOINT, SCARF_R2_ACCESS_KEY_ID, SCARF_R2_SECRET_ACCESS_KEY
Bucket/prefix reuse the existing tests/zarr_cloud_exp convention (same file):
    R2_BUCKET, R2_PREFIX
Skips cleanly (exit 0) if any of these are missing. Always deletes the probe
data it uploads, even on failure.

Run:
    uv run python -m tests.performance.bench_r2
    uv run python -m tests.performance.bench_r2 --across 1,2,4,8 --io-concurrency 1,4,8
"""

import argparse
import math
import os
import sys
import uuid
from dataclasses import dataclass

import numpy as np
import zarr

from scarf.parallel import stream_shards
from scarf.storage.zarr_store import iter_shard_row_slices, make_store

from ._common import best_of, parse_worker_list, print_row, section, synthetic_matrix
from tests.r2_profile import load_env, storage_options

DEFAULT_R2_N_CELLS = 20_000
DEFAULT_R2_N_COLS = 200
DEFAULT_R2_SHARD_ROWS = 2_500


def _skip(reason: str) -> None:
    print(f"Skipping R2 benchmark: {reason}", flush=True)


@dataclass(frozen=True)
class R2Location:
    bucket: str
    key_prefix: str
    uri: str


def _resolve_location(run_id: str) -> R2Location | None:
    load_env()
    bucket = os.environ.get("R2_BUCKET")
    if not bucket:
        return None
    prefix = os.environ.get("R2_PREFIX", "").strip("/")
    rel = f"scarf-perf/{run_id}"
    key_prefix = f"{prefix}/{rel}" if prefix else rel
    uri = f"s3://{bucket}/{key_prefix}/data.zarr"
    return R2Location(bucket=bucket, key_prefix=key_prefix, uri=uri)


def _delete_prefix(admin_store: object, prefix: str) -> None:
    paths = [
        meta["path"]
        for batch in admin_store.list(prefix)  # type: ignore[attr-defined]
        for meta in batch
    ]
    if paths:
        admin_store.delete(paths)  # type: ignore[attr-defined]


def run(args: argparse.Namespace) -> int:
    run_id = uuid.uuid4().hex[:12]
    location = _resolve_location(run_id)
    if location is None:
        _skip("R2_BUCKET not set in tests/.env")
        return 0

    try:
        opts = storage_options(location.uri)
    except RuntimeError as exc:
        _skip(str(exc))
        return 0
    assert opts is not None

    from obstore.store import from_url

    admin_store = from_url(f"s3://{location.bucket}", **opts)
    uri = location.uri

    n_cells, n_cols, shard_rows = args.n_cells, args.n_cols, args.shard_rows
    n_shards = math.ceil(n_cells / shard_rows)
    print(
        f"remote dataset: {n_cells} x {n_cols} float32, "
        f"shard_rows={shard_rows} ({n_shards} shards), uri={uri}",
        flush=True,
    )

    try:
        data = synthetic_matrix(n_cells, n_cols, seed=0)
        store = make_store(uri, storage_options=opts)
        root = zarr.open_group(store=store, mode="w")
        dst = root.create_array(
            "data",
            shape=(n_cells, n_cols),
            chunks=(shard_rows, n_cols),
            dtype="float32",
        )
        print("uploading probe array...", flush=True)
        for start, end in iter_shard_row_slices(n_cells, shard_rows):
            dst[start:end, :] = data[start:end, :]

        src = zarr.open_group(store=make_store(uri, storage_options=opts), mode="r")[
            "data"
        ]
        slices = list(iter_shard_row_slices(n_cells, shard_rows))

        def read_band(bounds: tuple[int, int]) -> np.ndarray:
            start, end = bounds
            return np.asarray(src[start:end, :])

        def timed_read(across: int, io_concurrency: int) -> float:
            # Set zarr's async.concurrency directly here (this benchmark drives
            # stream_shards' workers directly and does not pass its
            # io_concurrency kwarg) so the D2 sweep still takes effect when
            # across=1, where stream_shards runs a plain serial generator.
            def op() -> list[np.ndarray]:
                with zarr.config.set({"async.concurrency": max(1, io_concurrency)}):
                    return list(stream_shards(slices, read_band, workers=across))

            result, seconds = best_of(op, n=args.repeats)
            if not np.array_equal(np.vstack(result), data):
                raise AssertionError("read-back data does not match uploaded data")
            return seconds

        across_list = parse_worker_list(args.across)
        io_list = parse_worker_list(args.io_concurrency)
        fixed_io = io_list[-1]
        section(f"D1: across-shard depth sweep (io_concurrency={fixed_io} fixed)")
        baseline = timed_read(across_list[0], fixed_io)
        print_row(f"across={across_list[0]}", baseline)
        for across in across_list[1:]:
            seconds = timed_read(across, fixed_io)
            print_row(f"across={across}", seconds, f"(x{baseline / seconds:.2f})")

        section("D2: io_concurrency sweep (across=1 fixed, sequential band reads)")
        baseline_io = timed_read(1, io_list[0])
        print_row(f"io_concurrency={io_list[0]}", baseline_io)
        for io_concurrency in io_list[1:]:
            seconds = timed_read(1, io_concurrency)
            print_row(
                f"io_concurrency={io_concurrency}",
                seconds,
                f"(x{baseline_io / seconds:.2f})",
            )
    finally:
        print(
            f"\ncleaning up remote probe data under {location.key_prefix}...",
            flush=True,
        )
        _delete_prefix(admin_store, location.key_prefix)

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-cells", type=int, default=DEFAULT_R2_N_CELLS)
    parser.add_argument("--n-cols", type=int, default=DEFAULT_R2_N_COLS)
    parser.add_argument("--shard-rows", type=int, default=DEFAULT_R2_SHARD_ROWS)
    parser.add_argument("--across", type=str, default="1,2,4,8")
    parser.add_argument("--io-concurrency", type=str, default="1,4,8")
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(run(parse_args()))
