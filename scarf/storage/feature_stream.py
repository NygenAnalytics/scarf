"""Geometry-aware planning and bounded reads for feature-column streams."""

import asyncio
import operator
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np

from .async_execution import AsyncStorageRunner
from .budget import ResourceBudget, admit_stream, resolve_budget
from .count_matrix import TARGET_READ_UNIT_BYTES
from .geometry import ArrayGeometry, array_geometry
from .partition import (
    IndexBlock,
    affordable_width,
    checked_indices,
    partition_indices,
)
from .types import as_zarr_array

type BlockBytes = Callable[[int], int]
T = TypeVar("T")

__all__ = [
    "FeatureCellBand",
    "FeatureReadGroup",
    "FeatureStreamPlan",
    "feature_column_chunk",
    "load_feature_strip",
    "map_feature_cell_bands",
    "map_feature_read_groups",
    "plan_feature_stream",
    "planned_read_group_chunks",
    "selected_feature_chunk_starts",
    "selected_feature_values",
]


@dataclass(frozen=True, slots=True)
class FeatureStreamPlan:
    """Ordered feature blocks and their admitted read concurrency."""

    geometry: ArrayGeometry
    featureAxis: int
    blocks: tuple[IndexBlock, ...]
    readWorkers: int
    ioConcurrency: int
    repeatedDecodeCount: int


@dataclass(frozen=True, slots=True)
class FeatureReadGroup:
    """One inner-chunk feature group gathered into requested cell order."""

    featStart: int
    featEnd: int
    values: np.ndarray
    readSec: float
    blockBytes: int


@dataclass(frozen=True, slots=True)
class FeatureCellBand:
    """One feature group intersected with one physical cell band.

    ``values`` is the raw decoded band. ``selectedLocal`` indexes active cells
    inside that band. ``selectedDestinations`` maps those cells onto the
    caller-requested selected-cell order.
    """

    featStart: int
    featEnd: int
    cellStart: int
    cellEnd: int
    values: np.ndarray
    selectedLocal: np.ndarray
    selectedDestinations: np.ndarray
    readSec: float
    blockBytes: int


def _axis(value: int, *, name: str) -> int:
    resolved = operator.index(value)
    if resolved not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1")
    return int(resolved)


def _plane(array: Any) -> ArrayGeometry:
    geometry = array_geometry(array)
    if geometry is None or len(geometry.shape) != 2:
        raise ValueError("Feature streams require a chunked two-dimensional array")
    return geometry


def feature_column_chunk(array: Any, *, featureAxis: int) -> int:
    """Return one physical feature-chunk width."""
    return _plane(array).axisChunk(_axis(featureAxis, name="featureAxis"))


def _positive_requested(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("requestedBatchSize must be a positive integer")
    try:
        resolved = operator.index(value)
    except TypeError:
        raise TypeError("requestedBatchSize must be a positive integer") from None
    if resolved < 1:
        raise ValueError("requestedBatchSize must be greater than zero")
    return int(resolved)


def _owned_bytes(blockBytes: BlockBytes, width: int) -> int:
    value = int(blockBytes(max(1, int(width))))
    if value < 1:
        raise ValueError("blockBytes must return a positive byte count")
    return value


def _repeated_decodes(
    blocks: Sequence[IndexBlock],
    *,
    cell_bin_count: int,
) -> int:
    touches: dict[int, int] = {}
    for block in blocks:
        for feature_bin in block.bins:
            touches[feature_bin] = touches.get(feature_bin, 0) + 1
    return sum(max(0, count - 1) * cell_bin_count for count in touches.values())


def plan_feature_stream(
    array: Any,
    *,
    featureAxis: int,
    cellAxis: int,
    featureIndices: Sequence[int] | np.ndarray,
    cellIndices: Sequence[int] | np.ndarray,
    resources: ResourceBudget,
    blockBytes: BlockBytes,
    residentBytes: int = 0,
    requestedBatchSize: int | None = None,
) -> FeatureStreamPlan:
    """Plan variable-width feature blocks from physical chunk geometry."""
    feature_axis = _axis(featureAxis, name="featureAxis")
    cell_axis = _axis(cellAxis, name="cellAxis")
    if feature_axis == cell_axis:
        raise ValueError("featureAxis and cellAxis must differ")

    geometry = _plane(array)
    feature_indices = checked_indices(
        featureIndices,
        limit=geometry.shape[feature_axis],
        name="featureIndices",
    )
    cell_indices = checked_indices(
        cellIndices,
        limit=geometry.shape[cell_axis],
        name="cellIndices",
    )
    requested = _positive_requested(requestedBatchSize)
    resident = max(0, int(residentBytes))
    available = resources.memoryBytes - resident
    if available <= 0:
        raise MemoryError(
            f"Resident data needs {resident} bytes, but the operation limit is "
            f"{resources.memoryBytes} bytes"
        )

    decode_bytes = geometry.nominalChunkBytes()

    def fits(width: int) -> bool:
        return _owned_bytes(blockBytes, width) + decode_bytes <= available

    if feature_indices.size == 0:
        return FeatureStreamPlan(
            geometry=geometry,
            featureAxis=feature_axis,
            blocks=(),
            readWorkers=1,
            ioConcurrency=1,
            repeatedDecodeCount=0,
        )

    if requested is not None:
        blocks = partition_indices(
            geometry,
            feature_axis,
            feature_indices,
            maxWidth=requested,
        )
        if any(not fits(block.indices.size) for block in blocks):
            raise MemoryError(
                f"Requested feature batch width {requested} does not fit; "
                f"the affordable width is {affordable_width(fits, requested)}"
            )
    else:
        blocks = partition_indices(
            geometry,
            feature_axis,
            feature_indices,
            fits=fits,
        )

    block_bytes = max(_owned_bytes(blockBytes, block.indices.size) for block in blocks)
    prefetchable = len(blocks) - 1
    # A read-ahead stream holds the block being consumed while the next ones load.
    # A single-block stream holds nothing before its own read.
    held = resident + (block_bytes + decode_bytes if prefetchable else 0)
    read_workers = 1
    io_concurrency = 1
    try:
        admission = admit_stream(
            resources,
            nBlocks=max(1, prefetchable),
            blockBytes=block_bytes,
            decodeBytes=decode_bytes,
            residentBytes=held,
        )
    except MemoryError:
        # A second materialized block may not fit, while the current block can
        # still use the remaining budget for concurrent chunk decodes.
        current_admission = admit_stream(
            resources,
            nBlocks=1,
            blockBytes=block_bytes,
            decodeBytes=decode_bytes,
            residentBytes=resident,
        )
        io_concurrency = current_admission.ioConcurrency
    else:
        io_concurrency = admission.ioConcurrency
        if prefetchable:
            read_workers = admission.outerWorkers

    cell_bins = int(np.unique(geometry.binOf(cell_axis, cell_indices)).size)
    return FeatureStreamPlan(
        geometry=geometry,
        featureAxis=feature_axis,
        blocks=tuple(blocks),
        readWorkers=read_workers,
        ioConcurrency=io_concurrency,
        repeatedDecodeCount=_repeated_decodes(blocks, cell_bin_count=cell_bins),
    )


def selected_feature_values(values: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Return selected feature rows without copying when every row is kept."""
    if keep.ndim != 1 or keep.shape[0] != values.shape[0]:
        raise ValueError("keep must be a 1-D mask over feature rows")
    if bool(np.all(keep)):
        return values
    return np.ascontiguousarray(values[keep])


def load_feature_strip(
    counts_t: Any,
    feat_start: int,
    *,
    cell_idx: np.ndarray | None = None,
) -> FeatureReadGroup:
    """Load one inner-chunk feature strip across the full cell axis.

    This is the whole-strip baseline used by the Phase 3 comparison. Product
    consumers use ``map_feature_read_groups`` or ``map_feature_cell_bands``.
    """
    array = as_zarr_array(counts_t)
    geometry = _plane(array)
    gene_strip = geometry.axisChunk(0)
    n_feats = int(geometry.shape[0])
    feat_end = min(int(feat_start) + gene_strip, n_feats)
    started = time.perf_counter()
    block = np.ascontiguousarray(np.asarray(array[int(feat_start) : feat_end, :]))
    read_sec = time.perf_counter() - started
    if cell_idx is not None:
        selected = np.asarray(cell_idx)
        n_cells = int(geometry.shape[1])
        if selected.shape[0] != n_cells or not np.array_equal(
            selected, np.arange(n_cells)
        ):
            block = np.ascontiguousarray(block[:, selected])
    return FeatureReadGroup(
        featStart=int(feat_start),
        featEnd=int(feat_end),
        values=block,
        readSec=float(read_sec),
        blockBytes=int(block.nbytes),
    )


def selected_feature_chunk_starts(
    array: Any,
    feat_idx: Sequence[int] | np.ndarray | None = None,
) -> list[int]:
    """Return inner-chunk feature starts that intersect the selection."""
    plane = _plane(array)
    feat_chunk = plane.axisChunk(0)
    n_feats = plane.shape[0]
    if feat_idx is None:
        return list(range(0, n_feats, feat_chunk))
    indices = np.asarray(feat_idx, dtype=np.int64)
    if indices.size == 0:
        return []
    bins = np.unique(indices // feat_chunk)
    return [int(item) * feat_chunk for item in bins]


def _feature_group_ranges(
    array: Any,
    *,
    feat_idx: Sequence[int] | np.ndarray | None,
    feat_starts: Sequence[int] | None,
    readGroupChunks: int,
) -> list[tuple[int, int]]:
    geometry = _plane(array)
    n_feats = int(geometry.shape[0])
    feat_chunk = geometry.axisChunk(0)
    group_width = max(1, int(readGroupChunks)) * feat_chunk
    if feat_starts is None:
        starts = selected_feature_chunk_starts(array, feat_idx)
        width = group_width
    else:
        starts = [int(value) for value in feat_starts]
        width = feat_chunk
    merged: list[tuple[int, int]] = []
    for start in starts:
        feat_end = min(start + width, n_feats)
        if merged and start < merged[-1][1]:
            continue
        if (
            merged
            and start == merged[-1][1]
            and feat_end - merged[-1][0] <= group_width
        ):
            merged[-1] = (merged[-1][0], feat_end)
        else:
            merged.append((start, feat_end))
    return merged


def planned_read_group_chunks(
    array: Any,
    *,
    targetReadUnitBytes: int = TARGET_READ_UNIT_BYTES,
) -> int:
    """Return how many inner feature chunks fit in one target read unit."""
    geometry = _plane(array)
    n_cells = max(1, int(geometry.shape[1]))
    feat_chunk = max(1, geometry.axisChunk(0))
    itemsize = max(1, geometry.itemsize)
    group_feats = max(1, int(targetReadUnitBytes) // (n_cells * itemsize))
    return max(1, group_feats // feat_chunk)


def _selected_cell_bands(
    selected_cells: np.ndarray,
    *,
    n_cells: int,
    cell_chunk: int,
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    bands: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for cell_start in range(0, n_cells, cell_chunk):
        cell_end = min(cell_start + cell_chunk, n_cells)
        in_band = (selected_cells >= cell_start) & (selected_cells < cell_end)
        if not np.any(in_band):
            continue
        local = np.asarray(selected_cells[in_band] - cell_start, dtype=np.int64)
        destinations = np.flatnonzero(in_band).astype(np.int64, copy=False)
        bands.append((cell_start, cell_end, local, destinations))
    return bands


def _groups_in_flight(budget: ResourceBudget, unit_bytes: int) -> int:
    unit = max(1, int(unit_bytes))
    by_memory = max(1, int(budget.memoryBytes) // unit)
    by_workers = max(1, int(budget.workers))
    return max(1, min(by_memory, by_workers))


def map_feature_read_groups(
    counts_t: Any,
    process: Callable[[FeatureReadGroup], T],
    *,
    cell_idx: np.ndarray | None = None,
    feat_idx: Sequence[int] | np.ndarray | None = None,
    feat_starts: Sequence[int] | None = None,
    resources: ResourceBudget | None = None,
    progress: str | None = None,
    readGroupChunks: int | None = None,
    readGroupsInFlight: int | None = None,
) -> Iterator[T]:
    """Map ``process`` over bounded feature/cell read groups.

    Ordinary slices drive the reads. Adjacent inner chunks are grouped only
    when ``readGroupChunks`` is greater than 1. Independent groups stay in
    flight up to the admitted memory/worker limit. ``process`` is serialized
    so Numba kernels stay on one thread pool.
    """
    array = as_zarr_array(counts_t)
    geometry = _plane(array)
    _n_feats, n_cells = (int(value) for value in geometry.shape)
    feat_chunk = geometry.axisChunk(0)
    cell_chunk = geometry.axisChunk(1)
    group_chunks = (
        planned_read_group_chunks(array)
        if readGroupChunks is None
        else max(1, int(readGroupChunks))
    )
    merged = _feature_group_ranges(
        array,
        feat_idx=feat_idx,
        feat_starts=feat_starts,
        readGroupChunks=group_chunks,
    )
    if not merged:
        return iter(())

    if cell_idx is None:
        selected_cells = np.arange(n_cells, dtype=np.int64)
    else:
        selected_cells = np.asarray(cell_idx, dtype=np.int64)
    n_selected = int(selected_cells.shape[0])
    budget = resources or resolve_budget()
    itemsize = geometry.itemsize
    bands = _selected_cell_bands(
        selected_cells,
        n_cells=n_cells,
        cell_chunk=cell_chunk,
    )
    max_local = max((feat_end - feat_start) for feat_start, feat_end in merged)
    max_band = (
        max(cell_end - cell_start for cell_start, cell_end, _local, _dest in bands)
        if bands
        else 1
    )
    unit_bytes = max_local * n_selected * itemsize + max_local * max_band * itemsize
    if unit_bytes > budget.memoryBytes:
        raise MemoryError(
            "One feature read group plus its source cell band exceeds "
            "the operation memory limit"
        )
    in_flight = (
        max(1, int(readGroupsInFlight))
        if readGroupsInFlight is not None
        else _groups_in_flight(budget, unit_bytes)
    )

    from ..utils.progress import tqdmbar

    results: list[T] = []
    progress_bar = tqdmbar(desc=progress, total=len(merged)) if progress else None

    async def _operation(runner: AsyncStorageRunner) -> None:
        source = array.async_array
        turn = asyncio.Condition()
        next_idx = 0

        async def _one_group(idx: int, feat_start: int, feat_end: int) -> None:
            nonlocal next_idx
            n_local = feat_end - feat_start
            destination_bytes = max(1, n_local * n_selected * itemsize)
            async with runner.reserve_bytes(destination_bytes):
                dest = np.empty((n_local, n_selected), dtype=array.dtype)
                started = time.perf_counter()
                for cell_start, cell_end, local, destinations in bands:
                    read_bytes = n_local * (cell_end - cell_start) * itemsize
                    async with runner.reserve_bytes(read_bytes):
                        async with runner.read_lane():
                            block = np.asarray(
                                await source.getitem(
                                    (
                                        slice(feat_start, feat_end),
                                        slice(cell_start, cell_end),
                                    )
                                )
                            )
                        dest[:, destinations] = block[:, local]
                group = FeatureReadGroup(
                    featStart=int(feat_start),
                    featEnd=int(feat_end),
                    values=dest,
                    readSec=time.perf_counter() - started,
                    blockBytes=int(dest.nbytes),
                )
                async with turn:
                    while next_idx != idx:
                        await turn.wait()
                    results.append(process(group))
                    next_idx += 1
                    turn.notify_all()
                    if progress_bar is not None:
                        progress_bar.update(1)

        pending: set[asyncio.Task[None]] = set()
        async with asyncio.TaskGroup() as tasks:
            for idx, (feat_start, feat_end) in enumerate(merged):
                pending.add(tasks.create_task(_one_group(idx, feat_start, feat_end)))
                if len(pending) >= in_flight:
                    done, pending = await asyncio.wait(
                        pending,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for completed in done:
                        completed.result()

    try:
        AsyncStorageRunner(
            budget,
            chunksPerShard=max(1, geometry.axisShard(0) // feat_chunk),
            readGroupsInFlight=in_flight,
        ).run(_operation)
    finally:
        if progress_bar is not None:
            progress_bar.close()
    return iter(results)


def map_feature_cell_bands(
    counts_t: Any,
    process: Callable[[FeatureCellBand], T],
    *,
    cell_idx: np.ndarray | None = None,
    feat_idx: Sequence[int] | np.ndarray | None = None,
    feat_starts: Sequence[int] | None = None,
    resources: ResourceBudget | None = None,
    progress: str | None = None,
    readGroupChunks: int | None = None,
    readGroupsInFlight: int | None = None,
) -> Iterator[T]:
    """Map ``process`` over cell-band slices in deterministic feature order.

    Each callback receives one raw decoded band. Active cells are described by
    ``selectedLocal`` rather than a gathered copy. Independent bands stay in
    flight up to the admitted memory/worker limit. ``process`` runs in band
    order so reductions stay deterministic.
    """
    array = as_zarr_array(counts_t)
    geometry = _plane(array)
    _n_feats, n_cells = (int(value) for value in geometry.shape)
    feat_chunk = geometry.axisChunk(0)
    cell_chunk = geometry.axisChunk(1)
    group_chunks = (
        planned_read_group_chunks(array)
        if readGroupChunks is None
        else max(1, int(readGroupChunks))
    )
    merged = _feature_group_ranges(
        array,
        feat_idx=feat_idx,
        feat_starts=feat_starts,
        readGroupChunks=group_chunks,
    )
    if not merged:
        return iter(())

    if cell_idx is None:
        selected_cells = np.arange(n_cells, dtype=np.int64)
    else:
        selected_cells = np.asarray(cell_idx, dtype=np.int64)
    bands = _selected_cell_bands(
        selected_cells,
        n_cells=n_cells,
        cell_chunk=cell_chunk,
    )
    work = [
        (feat_start, feat_end, cell_start, cell_end, local, destinations)
        for feat_start, feat_end in merged
        for cell_start, cell_end, local, destinations in bands
    ]
    if not work:
        return iter(())

    budget = resources or resolve_budget()
    itemsize = geometry.itemsize
    max_local = max(feat_end - feat_start for feat_start, feat_end, *_rest in work)
    max_band = max(
        cell_end - cell_start for _fs, _fe, cell_start, cell_end, _local, _dest in work
    )
    unit_bytes = max(1, max_local * max_band * itemsize)
    if unit_bytes > budget.memoryBytes:
        raise MemoryError("One feature cell band exceeds the operation memory limit")
    in_flight = (
        max(1, int(readGroupsInFlight))
        if readGroupsInFlight is not None
        else _groups_in_flight(budget, unit_bytes)
    )

    from ..utils.progress import tqdmbar

    results: list[T] = []
    progress_bar = tqdmbar(desc=progress, total=len(work)) if progress else None

    async def _operation(runner: AsyncStorageRunner) -> None:
        source = array.async_array
        turn = asyncio.Condition()
        next_idx = 0

        async def _one_band(
            idx: int,
            feat_start: int,
            feat_end: int,
            cell_start: int,
            cell_end: int,
            local: np.ndarray,
            destinations: np.ndarray,
        ) -> None:
            nonlocal next_idx
            n_local = feat_end - feat_start
            read_bytes = max(1, n_local * (cell_end - cell_start) * itemsize)
            started = time.perf_counter()
            async with runner.reserve_bytes(read_bytes):
                async with runner.read_lane():
                    block = np.asarray(
                        await source.getitem(
                            (
                                slice(feat_start, feat_end),
                                slice(cell_start, cell_end),
                            )
                        )
                    )
                band = FeatureCellBand(
                    featStart=int(feat_start),
                    featEnd=int(feat_end),
                    cellStart=int(cell_start),
                    cellEnd=int(cell_end),
                    values=block,
                    selectedLocal=local,
                    selectedDestinations=destinations,
                    readSec=time.perf_counter() - started,
                    blockBytes=int(block.nbytes),
                )
                async with turn:
                    while next_idx != idx:
                        await turn.wait()
                    results.append(process(band))
                    next_idx += 1
                    turn.notify_all()
                    if progress_bar is not None:
                        progress_bar.update(1)

        pending: set[asyncio.Task[None]] = set()
        async with asyncio.TaskGroup() as tasks:
            for idx, spec in enumerate(work):
                pending.add(tasks.create_task(_one_band(idx, *spec)))
                if len(pending) >= in_flight:
                    done, pending = await asyncio.wait(
                        pending,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for completed in done:
                        completed.result()

    try:
        AsyncStorageRunner(
            budget,
            chunksPerShard=max(1, geometry.axisShard(0) // feat_chunk),
            readGroupsInFlight=in_flight,
        ).run(_operation)
    finally:
        if progress_bar is not None:
            progress_bar.close()
    return iter(results)
