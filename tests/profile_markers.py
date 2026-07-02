#!/usr/bin/env python3
"""Profile marker search bottlenecks on an open Zarr store (local or R2)."""

import argparse
import cProfile
import io
import os
import pstats
import time
from pathlib import Path

import numpy as np
from numba import set_num_threads

from scarf.assay import _read_block
from scarf.datastore.datastore import DataStore
from scarf.markers import _batch_stats, _marker_stats_batch
from scarf.utils import set_verbosity

_ENV_PATH = Path(__file__).resolve().parent / ".env"


def _load_env() -> None:
    if not _ENV_PATH.is_file():
        return
    for line in _ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())


def _storage_options(uri: str) -> dict[str, str] | None:
    if not uri.startswith("s3://"):
        return None
    return {
        "AWS_ACCESS_KEY_ID": os.environ["R2_ACCESS_KEY_ID"],
        "AWS_SECRET_ACCESS_KEY": os.environ["R2_SECRET_ACCESS_KEY"],
        "AWS_ENDPOINT_URL": os.environ["R2_ENDPOINT"],
        "AWS_REGION": "auto",
    }


def _timed(label: str, seconds: float, extra: str = "") -> None:
    suffix = f"  {extra}" if extra else ""
    print(f"  {label:28} {seconds:7.2f}s{suffix}")


def profile_one_batch(
    ds: DataStore,
    *,
    group_key: str,
    feat_key: str,
    batch_size: int,
    batch_index: int,
    n_threads: int,
) -> dict[str, float]:
    assay = ds.RNA
    cell_idx = assay.cells.active_index("I")
    feat_idx = assay.feats.active_index(feat_key)
    batches = [
        feat_idx[s : s + batch_size] for s in range(0, len(feat_idx), batch_size)
    ]
    cols = batches[batch_index]
    zarr_arr = assay.rawData._backing

    groups = assay.cells.fetch(group_key, "I")
    group_set = np.array(sorted(set(groups)))
    idx_map = dict(zip(group_set, range(len(group_set)), strict=True))
    int_indices = np.array([idx_map[x] for x in groups])
    group_counts = (
        __import__("pandas").Series(groups).value_counts().reindex(group_set).values
    )
    n_total = len(groups)

    scalar = assay.cells.fetch_all(assay.name + "_nCounts")[cell_idx]
    sf = float(assay.sf)
    scalar_col = np.asarray(scalar, dtype=np.float32).reshape(-1, 1)
    scalar_col[scalar_col == 0] = 1

    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    raw = _read_block(zarr_arr, cell_idx, cols)
    timings["zarr_read"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    normed = (sf * raw.astype(np.float32)) / scalar_col
    timings["normalize_f32"] = time.perf_counter() - t0

    set_num_threads(n_threads)
    data64 = np.ascontiguousarray(normed, dtype=np.float64)

    t0 = time.perf_counter()
    _ = _marker_stats_batch(
        data64,
        int_indices,
        group_counts.astype(np.float64),
        float(n_total),
    )
    timings["numba_kernel"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = _batch_stats(normed, int_indices, group_counts, n_total)
    timings["batch_stats_total"] = time.perf_counter() - t0

    return timings


def profile_postprocess(
    ds: DataStore,
    *,
    group_key: str,
    feat_key: str,
    batch_size: int,
    n_threads: int,
) -> float:
    from scarf.markers import find_markers_by_rank

    assay = ds.RNA
    t0 = time.perf_counter()
    find_markers_by_rank(
        assay=assay,
        group_key=group_key,
        cell_key="I",
        feat_key=feat_key,
        batch_size=batch_size,
        use_prenormed=False,
        prenormed_store=None,
        n_threads=n_threads,
    )
    return time.perf_counter() - t0


def main() -> int:
    _load_env()
    set_verbosity("INFO")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", required=True)
    parser.add_argument("--group-key", default="RNA_leiden_cluster")
    parser.add_argument("--feat-key", default="I")
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--nthreads", type=int, default=4)
    parser.add_argument("--full", action="store_true", help="Run full marker search")
    parser.add_argument(
        "--cprofile",
        action="store_true",
        help="Run cProfile on one _batch_stats call",
    )
    args = parser.parse_args()
    opts = _storage_options(args.uri)

    print("Opening datastore...")
    t_open = time.perf_counter()
    ds = DataStore(
        args.uri,
        default_assay="RNA",
        assay_types={"RNA": "RNA"},
        nthreads=args.nthreads,
        storage_options=opts,
        zarrProfile="cloud" if args.uri.startswith("s3://") else None,
    )
    _timed("open", time.perf_counter() - t_open)

    assay = ds.RNA
    n_cells = len(assay.cells.active_index("I"))
    n_feats = len(assay.feats.active_index(args.feat_key))
    backing = assay.rawData._backing
    chunks = getattr(backing, "chunks", None)
    col_chunk = int(chunks[1]) if chunks and len(chunks) > 1 else n_feats
    batch_size = max(1, min(col_chunk, n_feats))
    n_batches = max(1, (n_feats + batch_size - 1) // batch_size)
    n_groups = len(set(assay.cells.fetch(args.group_key, "I")))

    print(
        f"shape={n_cells} cells x {n_feats} features  "
        f"batch_size={batch_size} ({n_batches} batches)  groups={n_groups}"
    )
    print(f"chunks={chunks}  dtype={backing.dtype}  threads={args.nthreads}")
    print()

    print(f"=== single batch {args.batch_index + 1}/{n_batches} breakdown ===")
    for threads in (1, args.nthreads, min(8, os.cpu_count() or 4)):
        print(f"-- numba threads={threads} --")
        gc = __import__("gc")
        gc.collect()
        timings = profile_one_batch(
            ds,
            group_key=args.group_key,
            feat_key=args.feat_key,
            batch_size=batch_size,
            batch_index=args.batch_index,
            n_threads=threads,
        )
        cols = min(batch_size, n_feats - args.batch_index * batch_size)
        for key in (
            "zarr_read",
            "normalize_f32",
            "numba_kernel",
            "batch_stats_total",
        ):
            _timed(key, timings[key])
        stats_s = timings["batch_stats_total"]
        other_s = max(
            0.0,
            stats_s - timings["numba_kernel"],
        )
        _timed("pvalue_conversion", other_s)
        est_total = (
            timings["zarr_read"]
            + timings["normalize_f32"]
            + timings["batch_stats_total"]
        )
        _timed("batch_est_total", est_total)
        print(
            f"    stats fraction: {100 * stats_s / est_total:.0f}%  "
            f"({cols} genes x {n_cells} cells x {n_groups} groups)"
        )
        print()

    if args.cprofile:
        print("=== cProfile on _batch_stats (warm cache) ===")
        assay = ds.RNA
        cell_idx = assay.cells.active_index("I")
        feat_idx = assay.feats.active_index(args.feat_key)
        cols = feat_idx[:batch_size]
        raw = _read_block(assay.rawData._backing, cell_idx, cols)
        scalar = assay.cells.fetch_all(assay.name + "_nCounts")[cell_idx]
        sf = float(assay.sf)
        scalar_col = np.asarray(scalar, dtype=np.float32).reshape(-1, 1)
        scalar_col[scalar_col == 0] = 1
        normed = (sf * raw.astype(np.float32)) / scalar_col
        groups = assay.cells.fetch(args.group_key, "I")
        group_set = np.array(sorted(set(groups)))
        idx_map = dict(zip(group_set, range(len(group_set)), strict=True))
        int_indices = np.array([idx_map[x] for x in groups])
        group_counts = (
            __import__("pandas").Series(groups).value_counts().reindex(group_set).values
        )

        set_num_threads(args.nthreads)
        _batch_stats(normed, int_indices, group_counts, len(groups))

        pr = cProfile.Profile()
        pr.enable()
        _batch_stats(normed, int_indices, group_counts, len(groups))
        pr.disable()
        stream = io.StringIO()
        ps = pstats.Stats(pr, stream=stream).sort_stats("cumtime")
        ps.print_stats(25)
        print(stream.getvalue())

    if args.full:
        print("=== full marker search ===")
        gc = __import__("gc")
        gc.collect()
        total = profile_postprocess(
            ds,
            group_key=args.group_key,
            feat_key=args.feat_key,
            batch_size=batch_size,
            n_threads=args.nthreads,
        )
        _timed("find_markers_by_rank", total)
        _timed("per_batch_avg", total / n_batches)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
