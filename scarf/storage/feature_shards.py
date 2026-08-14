"""Whole-shard iteration over strip-sharded feature-major arrays.

Operating point (measured on R2 for marker/HVG consume):
compute runs on the calling thread (avoids Numba + Python thread-pool
deadlocks) and prefetches the next strip(s) under a budget-admitted
``FeatureShardConsumePlan``.

Measurement overrides (optional, not production defaults):
``SCARF_FEATURE_SHARD_PREFETCH_DEPTH``,
``SCARF_FEATURE_SHARD_READ_CONCURRENCY``,
``SCARF_FEATURE_SHARD_NUMBA_THREADS``.
"""

from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TypeVar
import os

import numpy as np
import zarr

from .budget import ResourceBudget
from .types import array_metadata_shards, as_zarr_array
from ..utils.progress import tqdmbar

# Cloud/R2-tuned whole-shard consume defaults (see performance_reconfig.md).
FEATURE_SHARD_COMPUTE_WORKERS = 2
FEATURE_SHARD_PREFETCH_DEPTH = 1
FEATURE_SHARD_READ_CONCURRENCY = 2
# Extra per in-flight shard for index/copy temporaries beyond the primary buffer.
_SHARD_INDEX_OVERHEAD_FRACTION = 0.25

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class FeatureShard:
    """One gene-strip shard loaded from a strip-sharded feature-major array."""

    featStart: int
    featEnd: int
    values: np.ndarray
    readSec: float
    blockBytes: int


@dataclass(frozen=True, slots=True)
class FeatureShardConsumePlan:
    """Resolved whole-shard consume knobs for one feature-wise scan."""

    prefetchDepth: int
    readConcurrency: int
    numbaThreads: int
    inFlight: int
    estimatedResidentBytes: int
    requestedPrefetchDepth: int
    requestedReadConcurrency: int
    requestedNumbaThreads: int
    source: str


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    return int(raw)


def _resolve_requested_knobs(
    *,
    nthreads: int,
    prefetchDepth: int | None,
    readConcurrency: int | None,
    numbaThreads: int | None,
) -> tuple[int, int, int, str]:
    env_prefetch = _env_int("SCARF_FEATURE_SHARD_PREFETCH_DEPTH")
    env_read = _env_int("SCARF_FEATURE_SHARD_READ_CONCURRENCY")
    env_threads = _env_int("SCARF_FEATURE_SHARD_NUMBA_THREADS")

    if prefetchDepth is not None:
        resolved_prefetch = prefetchDepth
        prefetch_source = "argument"
    elif env_prefetch is not None:
        resolved_prefetch = env_prefetch
        prefetch_source = "env"
    else:
        resolved_prefetch = FEATURE_SHARD_PREFETCH_DEPTH
        prefetch_source = "default"

    if readConcurrency is not None:
        resolved_read = readConcurrency
        read_source = "argument"
    elif env_read is not None:
        resolved_read = env_read
        read_source = "env"
    else:
        resolved_read = FEATURE_SHARD_READ_CONCURRENCY
        read_source = "default"

    if numbaThreads is not None:
        resolved_threads = numbaThreads
        thread_source = "argument"
    elif env_threads is not None:
        resolved_threads = env_threads
        thread_source = "env"
    else:
        # Historical operating point: reserve half the worker budget for the
        # (now unused) second compute worker. Measurement runs should override.
        resolved_threads = max(1, int(nthreads) // FEATURE_SHARD_COMPUTE_WORKERS)
        thread_source = "default"

    sources = {prefetch_source, read_source, thread_source}
    if sources == {"default"}:
        source = "default"
    elif "argument" in sources:
        source = "argument"
    elif "env" in sources:
        source = "env"
    else:
        source = "default"
    return (
        int(resolved_prefetch),
        int(resolved_read),
        int(resolved_threads),
        source,
    )


def estimate_feature_shard_bytes(
    *,
    geneStrip: int,
    nCells: int,
    itemsize: int,
    selectedCells: int | None = None,
) -> int:
    """Estimate peak resident bytes for one gene-strip load.

    ``load_feature_shard`` always decodes the full cell axis, then copies a
    cell subset when ``cell_idx`` is not a complete range. Peak is the decode
    buffer plus that gather copy.
    """
    decode = max(1, int(geneStrip) * max(0, int(nCells)) * max(1, int(itemsize)))
    if selectedCells is None or int(selectedCells) >= int(nCells):
        return decode
    gather = int(geneStrip) * max(0, int(selectedCells)) * max(1, int(itemsize))
    return decode + gather


def admitted_pending_limit(in_flight: int, *, holdingCurrent: bool) -> int:
    """Return how many unread shard futures may be queued.

    The currently processed shard stays resident, so the queue must leave one
    slot free while that shard is held.
    """
    bound = max(1, int(in_flight))
    if not holdingCurrent:
        return bound
    return max(0, bound - 1)


def selected_strip_starts(
    counts_t: zarr.Array,
    feat_idx: Sequence[int] | np.ndarray | None,
) -> list[int]:
    """Return strip starts that intersect ``feat_idx`` (all strips when None)."""
    starts = strip_feature_starts(counts_t)
    if feat_idx is None:
        return starts
    indexes = np.asarray(feat_idx, dtype=np.int64)
    if indexes.size == 0:
        return []
    gene_strip = int(counts_t.chunks[0])
    needed = {int(i) // gene_strip * gene_strip for i in indexes.tolist()}
    return [start for start in starts if start in needed]


def shard_values_for_selection(
    values: np.ndarray,
    keep: np.ndarray,
) -> np.ndarray:
    """Return selected feature rows without copying when every row is kept."""
    if keep.ndim != 1 or keep.shape[0] != values.shape[0]:
        raise ValueError("keep must be a 1-D mask over shard feature rows")
    if bool(np.all(keep)):
        return values
    return np.ascontiguousarray(values[keep])


def resolve_feature_shard_consume(
    *,
    nthreads: int,
    prefetchDepth: int | None = None,
    readConcurrency: int | None = None,
    numbaThreads: int | None = None,
    resources: ResourceBudget | None = None,
    shardBytes: int | None = None,
    nStrips: int | None = None,
    residentBytes: int = 0,
    extraBytesPerShard: int = 0,
) -> FeatureShardConsumePlan:
    """Resolve prefetch / read / Numba knobs under an optional memory budget.

    When ``resources`` and ``shardBytes`` are provided, admit at most
    ``prefetchDepth + 1`` in-flight shard buffers (plus index/copy overhead)
    and clamp read concurrency to that queue depth. Profiling argument/env
    overrides remain explicit measurement controls.
    """
    requested_prefetch, requested_read, requested_threads, source = (
        _resolve_requested_knobs(
            nthreads=nthreads,
            prefetchDepth=prefetchDepth,
            readConcurrency=readConcurrency,
            numbaThreads=numbaThreads,
        )
    )
    if requested_prefetch < 0:
        raise ValueError("prefetchDepth must be >= 0")
    if requested_read < 1:
        raise ValueError("readConcurrency must be >= 1")
    if requested_threads < 1:
        raise ValueError("numbaThreads must be >= 1")

    workers = max(1, int(nthreads))
    if resources is not None:
        workers = max(1, min(workers, int(resources.workers)))

    # Split CPU between Zarr decode/read work and caller-thread Numba compute.
    numba_cap = max(1, workers - min(workers // 2, max(0, requested_read - 1)))
    effective_threads = min(requested_threads, numba_cap, workers)

    if resources is None or shardBytes is None:
        in_flight = max(1, requested_prefetch + 1)
        effective_read = min(requested_read, in_flight)
        extra = max(0, int(extraBytesPerShard))
        per_shard = (
            0
            if shardBytes is None
            else int(shardBytes)
            + extra
            + max(1, int(int(shardBytes) * _SHARD_INDEX_OVERHEAD_FRACTION))
        )
        return FeatureShardConsumePlan(
            prefetchDepth=requested_prefetch,
            readConcurrency=effective_read,
            numbaThreads=effective_threads,
            inFlight=in_flight,
            estimatedResidentBytes=0 if shardBytes is None else in_flight * per_shard,
            requestedPrefetchDepth=requested_prefetch,
            requestedReadConcurrency=requested_read,
            requestedNumbaThreads=requested_threads,
            source=source,
        )

    extra = max(0, int(extraBytesPerShard))
    per_shard = (
        int(shardBytes)
        + extra
        + max(1, int(int(shardBytes) * _SHARD_INDEX_OVERHEAD_FRACTION))
    )
    available = int(resources.memoryBytes) - max(0, int(residentBytes))
    if available < per_shard:
        raise MemoryError(
            f"One feature shard needs about {per_shard} bytes in addition to "
            f"{max(0, int(residentBytes))} resident bytes, but the operation "
            f"limit is {resources.memoryBytes} bytes"
        )
    max_inflight = max(1, available // per_shard)
    if nStrips is not None:
        max_inflight = min(max_inflight, max(1, int(nStrips)))
    requested_inflight = max(1, requested_prefetch + 1)
    in_flight = min(requested_inflight, max_inflight)
    effective_prefetch = max(0, in_flight - 1)
    effective_read = min(requested_read, in_flight, workers)
    return FeatureShardConsumePlan(
        prefetchDepth=effective_prefetch,
        readConcurrency=max(1, effective_read),
        numbaThreads=effective_threads,
        inFlight=in_flight,
        estimatedResidentBytes=in_flight * per_shard,
        requestedPrefetchDepth=requested_prefetch,
        requestedReadConcurrency=requested_read,
        requestedNumbaThreads=requested_threads,
        source=source,
    )


def plan_feature_shard_consume_for_array(
    counts_t: zarr.Array,
    *,
    resources: ResourceBudget,
    cell_idx: np.ndarray | None = None,
    feat_idx: Sequence[int] | np.ndarray | None = None,
    prefetchDepth: int | None = None,
    readConcurrency: int | None = None,
    numbaThreads: int | None = None,
    residentBytes: int = 0,
    extraBytesPerShard: int = 0,
) -> FeatureShardConsumePlan:
    """Build a consume plan from strip geometry and the active resource budget."""
    array = as_zarr_array(counts_t)
    starts = selected_strip_starts(array, feat_idx)
    selected_cells = (
        int(array.shape[1]) if cell_idx is None else int(np.asarray(cell_idx).shape[0])
    )
    shard_bytes = estimate_feature_shard_bytes(
        geneStrip=int(array.chunks[0]),
        nCells=int(array.shape[1]),
        itemsize=int(np.dtype(array.dtype).itemsize),
        selectedCells=selected_cells,
    )
    return resolve_feature_shard_consume(
        nthreads=resources.workers,
        prefetchDepth=prefetchDepth,
        readConcurrency=readConcurrency,
        numbaThreads=numbaThreads,
        resources=resources,
        shardBytes=shard_bytes,
        nStrips=len(starts),
        residentBytes=residentBytes,
        extraBytesPerShard=extraBytesPerShard,
    )


def strip_feature_starts(counts_t: zarr.Array) -> list[int]:
    """Return feature-axis starts for each gene-strip shard."""
    n_feats = int(counts_t.shape[0])
    gene_strip = int(counts_t.chunks[0])
    shards = array_metadata_shards(counts_t)
    if shards is None:
        raise ValueError("Feature-shard iteration requires a strip-sharded array")
    if int(shards[0]) != gene_strip:
        raise ValueError(
            f"Expected shard feature extent {gene_strip}, got {tuple(shards)}"
        )
    return list(range(0, n_feats, gene_strip))


def load_feature_shard(
    counts_t: zarr.Array,
    feat_start: int,
    *,
    cell_idx: np.ndarray | None = None,
) -> FeatureShard:
    """Load one gene strip (optionally gathering active cells)."""
    import time

    gene_strip = int(counts_t.chunks[0])
    n_feats = int(counts_t.shape[0])
    feat_end = min(feat_start + gene_strip, n_feats)
    started = time.perf_counter()
    block = np.ascontiguousarray(np.asarray(counts_t[feat_start:feat_end, :]))
    read_sec = time.perf_counter() - started
    block_bytes = int(block.nbytes)
    if cell_idx is not None:
        cell_idx = np.asarray(cell_idx)
        n_cells = int(counts_t.shape[1])
        if len(cell_idx) != n_cells or not np.array_equal(cell_idx, np.arange(n_cells)):
            block = np.ascontiguousarray(block[:, cell_idx])
    return FeatureShard(
        featStart=int(feat_start),
        featEnd=int(feat_end),
        values=block,
        readSec=float(read_sec),
        blockBytes=block_bytes,
    )


def map_feature_shards(
    counts_t: zarr.Array,
    process: Callable[[FeatureShard], T],
    *,
    cell_idx: np.ndarray | None = None,
    feat_idx: Sequence[int] | np.ndarray | None = None,
    prefetchDepth: int | None = None,
    readConcurrency: int | None = None,
    numbaThreads: int | None = None,
    feat_starts: Sequence[int] | None = None,
    resources: ResourceBudget | None = None,
    plan: FeatureShardConsumePlan | None = None,
    progress: str | None = None,
    residentBytes: int = 0,
    extraBytesPerShard: int = 0,
) -> Iterator[T]:
    """Map ``process`` over whole gene-strip shards with budget-admitted prefetch.

    ``process`` always runs on the calling thread so Numba parallel kernels
    remain safe. When ``feat_idx`` is set and ``feat_starts`` is omitted, only
    intersecting strips are loaded.
    """
    array = as_zarr_array(counts_t)
    if feat_starts is not None:
        starts = list(feat_starts)
    else:
        starts = selected_strip_starts(array, feat_idx)
    if not starts:
        return iter(())

    if plan is None:
        if resources is None:
            plan = resolve_feature_shard_consume(
                nthreads=max(1, os.cpu_count() or 1),
                prefetchDepth=prefetchDepth,
                readConcurrency=readConcurrency,
                numbaThreads=numbaThreads,
            )
        else:
            plan = plan_feature_shard_consume_for_array(
                array,
                resources=resources,
                cell_idx=cell_idx,
                feat_idx=feat_idx,
                prefetchDepth=prefetchDepth,
                readConcurrency=readConcurrency,
                numbaThreads=numbaThreads,
                residentBytes=residentBytes,
                extraBytesPerShard=extraBytesPerShard,
            )

    in_flight = max(1, int(plan.inFlight))
    read_concurrency = max(1, min(int(plan.readConcurrency), in_flight))

    def load(start: int) -> FeatureShard:
        return load_feature_shard(array, start, cell_idx=cell_idx)

    def run() -> Iterator[T]:
        bar = None if progress is None else tqdmbar(desc=progress, total=len(starts))
        try:
            with ThreadPoolExecutor(max_workers=read_concurrency) as readers:
                start_iter = iter(starts)
                pending: list[Future[FeatureShard]] = []

                def fill(*, holding_current: bool) -> None:
                    limit = admitted_pending_limit(
                        in_flight, holdingCurrent=holding_current
                    )
                    while len(pending) < limit:
                        try:
                            nxt = next(start_iter)
                        except StopIteration:
                            break
                        pending.append(readers.submit(load, nxt))

                fill(holding_current=False)
                while pending:
                    if len(pending) > in_flight:
                        raise RuntimeError(
                            "feature-shard prefetch exceeded admitted in-flight bound"
                        )
                    shard = pending.pop(0).result()
                    fill(holding_current=True)
                    try:
                        yield process(shard)
                    finally:
                        if bar is not None:
                            bar.update(1)
                    # Drop the generator reference before refilling. Otherwise
                    # inFlight=1 never queues the next strip.
                    del shard
                    fill(holding_current=False)
        finally:
            if bar is not None:
                bar.close()

    return run()
