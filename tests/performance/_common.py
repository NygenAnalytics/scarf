"""Shared helpers for the opt-in performance suite in tests/performance/.

Not a test module (no ``test_*``/``bench_*`` prefix collected by pytest); just
plain helpers imported by the individual ``bench_*.py`` scripts. Keep this
generic so future perf work beyond shard-parallel processing can reuse it.
"""

import argparse
import contextlib
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TypeVar

import numpy as np
import zarr

from scarf.storage.budget import ResourceBudget, set_resource_budget
from tests.zarr_cloud_exp.profiler import profile

DEFAULT_N_CELLS = 60_000
DEFAULT_N_COLS = 1_000
DEFAULT_SHARD_ROWS = 10_000
DEFAULT_DIMS = 50
DEFAULT_WORKERS = (1, 2, 4, 8)

T = TypeVar("T")


def synthetic_matrix(
    n_cells: int, n_cols: int, seed: int = 0, dtype: str = "float32"
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_cells, n_cols)).astype(dtype)


@contextlib.contextmanager
def local_zarr_array(
    n_cells: int,
    n_cols: int,
    shard_rows: int,
    seed: int = 0,
    dtype: str = "float32",
) -> Iterator[zarr.Array]:
    """Write a synthetic dense matrix to a real local-disk Zarr store.

    Row-banded in ``shard_rows`` chunks, matching the on-disk geometry the
    shard-parallel primitives assume. Yields the array reopened read-only
    after one warm-up pass over every band, so timed reads measure
    decompress+compute rather than first-touch page-cache misses (mirrors the
    "warm page cache" setup used for the original design benchmarks).
    """
    with tempfile.TemporaryDirectory(prefix="scarf-perf-") as tmp:
        root = zarr.open_group(tmp, mode="w")
        arr = root.create_array(
            "data",
            shape=(n_cells, n_cols),
            chunks=(shard_rows, n_cols),
            dtype=dtype,
        )
        rng = np.random.default_rng(seed)
        for start in range(0, n_cells, shard_rows):
            end = min(start + shard_rows, n_cells)
            arr[start:end, :] = rng.standard_normal((end - start, n_cols)).astype(dtype)

        warm = zarr.open_group(tmp, mode="r")["data"]
        for start in range(0, n_cells, shard_rows):
            end = min(start + shard_rows, n_cells)
            np.asarray(warm[start:end, :])
        yield warm


@contextlib.contextmanager
def resource_budget(
    workers: int, working_copies: int = 8, memory_gb: float = 32.0
) -> Iterator[ResourceBudget]:
    """Install a generous budget so ``workingCopies``/memory never clamp the sweep.

    Only ``workers`` varies across calls; ``workingCopies`` is kept at least as
    large as ``workers`` so the read-ahead depth in ``shard_parallelism`` is not
    clamped by an incidentally small ``workingCopies``.
    """
    budget = ResourceBudget(
        memoryBytes=int(memory_gb * 1024**3),
        workers=workers,
        workingCopies=max(working_copies, workers),
    )
    set_resource_budget(budget)
    try:
        yield budget
    finally:
        set_resource_budget(None)


_UNSET = object()


def best_of(fn: Callable[[], T], n: int = 3) -> tuple[T, float]:
    """Run ``fn`` ``n`` times, returning the last result and the fastest wall time."""
    best_seconds = float("inf")
    result: object = _UNSET
    for _ in range(n):
        with profile() as r:
            result = fn()
        best_seconds = min(best_seconds, r.seconds)
    assert result is not _UNSET
    return result, best_seconds  # type: ignore[return-value]


def print_row(label: str, seconds: float, note: str = "") -> None:
    suffix = f"  {note}" if note else ""
    print(f"  {label:<28} {seconds:8.3f}s{suffix}", flush=True)


def speedup_note(baseline_seconds: float, seconds: float) -> str:
    if seconds <= 0:
        return ""
    return f"(x{baseline_seconds / seconds:.2f})"


def parse_worker_list(text: str) -> list[int]:
    return [int(w) for w in text.split(",") if w.strip()]


def add_size_args(
    parser: argparse.ArgumentParser,
    *,
    n_cells: int = DEFAULT_N_CELLS,
    n_cols: int = DEFAULT_N_COLS,
    shard_rows: int = DEFAULT_SHARD_ROWS,
) -> None:
    parser.add_argument("--n-cells", type=int, default=n_cells)
    parser.add_argument("--n-cols", type=int, default=n_cols)
    parser.add_argument("--shard-rows", type=int, default=shard_rows)
    parser.add_argument("--repeats", type=int, default=3, help="Best-of-N runs")


def add_worker_sweep_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workers",
        type=str,
        default=",".join(str(w) for w in DEFAULT_WORKERS),
        help="Comma-separated worker counts to sweep, e.g. 1,2,4,8",
    )


def section(title: str) -> None:
    print(f"\n{title}", flush=True)
    print("-" * len(title), flush=True)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]
