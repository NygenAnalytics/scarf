"""Geometry-aware planning and bounded reads for feature-column streams."""

import asyncio
import math
import operator
import queue
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np

from .async_execution import AsyncStorageRunner
from .budget import ResourceBudget, admit_stream, resolve_budget
from .execution import (
    ExecutionReport,
    OperationPlan,
    WorkShape,
    auto_read_width,
    plan_operation,
    record_execution_report,
)
from .count_matrix import (
    REBUILD_REMEDY,
    load_count_matrix_plan,
    read_group_from_payload,
)
from .geometry import ArrayGeometry, array_geometry
from .io_policy import DEFAULT_STORAGE_IO_POLICY, StorageIoPolicy
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
    "map_feature_cell_bands",
    "map_feature_read_groups",
    "plan_feature_stream",
    "persisted_read_group",
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
    unitIndex: int = 0


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
    unitIndex: int = 0


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


def persisted_read_group(array: Any) -> tuple[int, int]:
    """Return persisted ``(featureWidth, readGroupBytes)`` for consume grouping."""
    payload = load_count_matrix_plan(array)
    feature_width, read_group_bytes = read_group_from_payload(payload)
    geometry = _plane(array)
    expected_bytes = int(geometry.shape[1]) * feature_width * geometry.itemsize
    if read_group_bytes != expected_bytes:
        raise ValueError(
            "persisted read group does not match live geometry. " + REBUILD_REMEDY
        )
    return feature_width, read_group_bytes


def _feature_group_ranges(
    array: Any,
    *,
    feat_idx: Sequence[int] | np.ndarray | None,
    feat_starts: Sequence[int] | None,
    featureWidth: int,
) -> list[tuple[int, int]]:
    geometry = _plane(array)
    n_feats = int(geometry.shape[0])
    group_width = max(1, int(featureWidth))
    if feat_starts is None:
        starts = selected_feature_chunk_starts(array, feat_idx)
    else:
        starts = [int(value) for value in feat_starts]
    merged: list[tuple[int, int]] = []
    for start in starts:
        feat_end = min(start + geometry.axisChunk(0), n_feats)
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


def _plan_feature_consume(
    budget: ResourceBudget,
    *,
    io: StorageIoPolicy,
    nUnits: int,
    unitBytes: int,
    scratchBytes: int = 0,
    innerReadBytes: int = 0,
    maxInnerReads: int | None = None,
    chunksPerShard: int = 1,
    ordered: bool,
) -> OperationPlan:
    return plan_operation(
        budget,
        WorkShape(
            nUnits=max(1, int(nUnits)),
            unitBytes=max(1, int(unitBytes)),
            scratchBytes=max(0, int(scratchBytes)),
            innerReadBytes=max(0, int(innerReadBytes)),
            maxInnerReads=maxInnerReads,
            ordered=ordered,
            writes=False,
            chunksPerShard=max(1, int(chunksPerShard)),
        ),
        policy=io,
    )


def _iter_bounded_handoff(
    *,
    in_flight: int,
    run: Callable[[Callable[[Any], None], threading.Event], None],
) -> Iterator[Any]:
    handoff: queue.Queue[Any] = queue.Queue(maxsize=max(1, int(in_flight)))
    release: queue.Queue[None] = queue.Queue()
    stop = threading.Event()
    sentinel = object()
    error: list[BaseException] = []

    def deliver(item: Any) -> None:
        if stop.is_set():
            return
        while not stop.is_set():
            try:
                handoff.put(item, timeout=0.05)
                break
            except queue.Full:
                continue
        else:
            return
        while not stop.is_set():
            try:
                release.get(timeout=0.05)
                return
            except queue.Empty:
                continue

    def _worker() -> None:
        try:
            run(deliver, stop)
        except BaseException as exc:
            error.append(exc)
        finally:
            while True:
                try:
                    handoff.put(sentinel, timeout=0.05)
                    break
                except queue.Full:
                    if not stop.is_set():
                        continue
                    try:
                        handoff.get_nowait()
                    except queue.Empty:
                        continue

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    try:
        while True:
            item = handoff.get()
            if item is sentinel:
                break
            try:
                yield item
            finally:
                release.put(None)
    finally:
        stop.set()
        while thread.is_alive():
            try:
                item = handoff.get(timeout=0.05)
                if item is not sentinel:
                    release.put(None)
            except queue.Empty:
                release.put(None)
        thread.join()
    if error:
        raise error[0]


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


def map_feature_read_groups(
    counts_t: Any,
    process: Callable[[FeatureReadGroup], T],
    *,
    cell_idx: np.ndarray | None = None,
    feat_idx: Sequence[int] | np.ndarray | None = None,
    feat_starts: Sequence[int] | None = None,
    resources: ResourceBudget | None = None,
    progress: str | None = None,
    io: StorageIoPolicy | None = None,
    metrics: dict[str, Any] | None = None,
    scratchBytes: int = 0,
    extraItemsize: int = 0,
    orderedCompute: bool = False,
) -> Iterator[T]:
    """Map ``process`` over persisted read groups with bounded handoff."""
    array = as_zarr_array(counts_t)
    geometry = _plane(array)
    _n_feats, n_cells = (int(value) for value in geometry.shape)
    feat_chunk = geometry.axisChunk(0)
    cell_chunk = geometry.axisChunk(1)
    feature_width, read_group_bytes = persisted_read_group(array)
    merged = _feature_group_ranges(
        array,
        feat_idx=feat_idx,
        feat_starts=feat_starts,
        featureWidth=feature_width,
    )
    if not merged:
        return iter(())

    if cell_idx is None:
        selected_cells = np.arange(n_cells, dtype=np.int64)
    else:
        selected_cells = np.asarray(cell_idx, dtype=np.int64)
    n_selected = int(selected_cells.shape[0])
    budget = resources or resolve_budget()
    resolved_io = io or DEFAULT_STORAGE_IO_POLICY
    itemsize = geometry.itemsize
    bands = _selected_cell_bands(
        selected_cells,
        n_cells=n_cells,
        cell_chunk=cell_chunk,
    )
    requested_chunk_reads = (
        int(resolved_io.readWorkers)
        if resolved_io.readWorkers is not None
        else auto_read_width(budget.workers)
    )
    compute_width = (
        1
        if orderedCompute
        else min(
            budget.workers,
            int(resolved_io.computeWorkers or budget.workers),
        )
    )
    requested_group_reads = min(
        requested_chunk_reads,
        max(1, 2 * compute_width),
    )
    available_group_reads = max(1, min(len(merged), requested_group_reads))
    requested_inner_reads = min(
        max(1, len(bands)),
        max(1, math.ceil(requested_chunk_reads / available_group_reads)),
    )
    group_io = StorageIoPolicy(
        readWorkers=requested_group_reads,
        computeWorkers=resolved_io.computeWorkers,
        writeWorkers=resolved_io.writeWorkers,
    )
    max_band_bytes = max(
        (
            (feat_end - feat_start) * (cell_end - cell_start) * itemsize
            for feat_start, feat_end in merged
            for cell_start, cell_end, _local, _destinations in bands
        ),
        default=1,
    )
    if extraItemsize < 0:
        raise ValueError("extraItemsize must not be negative")
    extra_itemsize = operator.index(extraItemsize)
    extra_unit_bytes = extra_itemsize * feature_width * n_selected
    plan = _plan_feature_consume(
        budget,
        io=group_io,
        nUnits=len(merged),
        unitBytes=read_group_bytes + extra_unit_bytes,
        scratchBytes=scratchBytes,
        innerReadBytes=max_band_bytes,
        maxInnerReads=requested_inner_reads,
        chunksPerShard=max(1, len(bands)),
        ordered=orderedCompute,
    )
    in_flight = plan.readWorkers
    fetch_seconds = 0.0
    compute_seconds = 0.0
    compute_wait_seconds = 0.0
    units_completed = 0
    if metrics is not None:
        metrics.clear()
        metrics.update(plan.as_metrics())
        metrics.update(
            {
                "requestedGroupsInFlight": requested_group_reads,
                "effectiveGroupsInFlight": in_flight,
                "requestedChunkReadsInFlight": requested_chunk_reads,
                "effectiveChunkReadsInFlight": in_flight * plan.innerReads,
                "readGroupBytes": read_group_bytes,
                "cellBandBytes": max_band_bytes,
                "cellBandCount": len(bands),
                "featureWidth": feature_width,
                "unitKind": "countsTReadGroup",
            }
        )

    from ..utils.progress import tqdmbar

    progress_bar = tqdmbar(desc=progress, total=len(merged)) if progress else None

    def _run(deliver: Callable[[T], None], stop: threading.Event) -> None:
        nonlocal fetch_seconds, compute_seconds, compute_wait_seconds, units_completed

        async def _operation(runner: AsyncStorageRunner) -> None:
            nonlocal fetch_seconds, compute_seconds, compute_wait_seconds
            nonlocal units_completed
            source = array.async_array
            turn = asyncio.Condition()
            next_idx = 0

            async def _one_group(idx: int, feat_start: int, feat_end: int) -> None:
                nonlocal next_idx, fetch_seconds, compute_seconds
                nonlocal compute_wait_seconds, units_completed
                if stop.is_set():
                    async with turn:
                        while next_idx != idx:
                            await turn.wait()
                        next_idx += 1
                        turn.notify_all()
                    return
                n_local = feat_end - feat_start
                extra_live = extra_itemsize * n_local * n_selected
                destination_bytes = max(1, n_local * n_selected * itemsize + extra_live)
                async with runner.reserve_bytes(destination_bytes):
                    dest = np.empty((n_local, n_selected), dtype=array.dtype)

                    async def _read_band(
                        cell_start: int,
                        cell_end: int,
                        local: np.ndarray,
                        destinations: np.ndarray,
                    ) -> float:
                        read_bytes = n_local * (cell_end - cell_start) * itemsize
                        async with runner.read_lane():
                            async with runner.reserve_bytes(read_bytes):
                                started = time.perf_counter()
                                block = np.asarray(
                                    await source.getitem(
                                        (
                                            slice(feat_start, feat_end),
                                            slice(cell_start, cell_end),
                                        )
                                    )
                                )
                                read_seconds = time.perf_counter() - started
                                dest[:, destinations] = block[:, local]
                        return read_seconds

                    read_seconds = sum(
                        await asyncio.gather(*(_read_band(*band) for band in bands))
                    )
                    group = FeatureReadGroup(
                        featStart=int(feat_start),
                        featEnd=int(feat_end),
                        values=dest,
                        readSec=read_seconds,
                        blockBytes=int(dest.nbytes),
                        unitIndex=idx,
                    )
                    fetch_seconds += group.readSec
                    wait_started = time.perf_counter()
                    if orderedCompute:
                        async with turn:
                            while next_idx != idx:
                                await turn.wait()
                            compute_wait_seconds += time.perf_counter() - wait_started
                            compute_started = time.perf_counter()
                            item = await runner.compute(lambda: process(group))
                            compute_seconds += time.perf_counter() - compute_started
                            await asyncio.to_thread(deliver, item)
                            next_idx += 1
                            turn.notify_all()
                    else:
                        compute_started = time.perf_counter()
                        item = await runner.compute(lambda: process(group))
                        compute_seconds += time.perf_counter() - compute_started
                        await asyncio.to_thread(deliver, item)
                    units_completed += 1
                    if progress_bar is not None:
                        progress_bar.update(1)

            pending: set[asyncio.Task[None]] = set()
            async with asyncio.TaskGroup() as tasks:
                for idx, (feat_start, feat_end) in enumerate(merged):
                    if stop.is_set():
                        break
                    pending.add(
                        tasks.create_task(_one_group(idx, feat_start, feat_end))
                    )
                    if len(pending) >= in_flight:
                        done, pending = await asyncio.wait(
                            pending,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        for completed in done:
                            completed.result()

        runner = AsyncStorageRunner(
            budget,
            operation=plan,
            chunksPerShard=max(1, geometry.axisShard(0) // feat_chunk),
            readGroupsInFlight=in_flight,
        )
        try:
            runner.run(_operation)
        finally:
            if progress_bar is not None:
                progress_bar.close()
            report = record_execution_report(
                ExecutionReport(
                    plan=plan,
                    unitKind="countsTReadGroup",
                    actualReadWorkers=in_flight,
                    actualComputeWorkers=runner.plan.computeWorkerLimit,
                    actualWriteWorkers=1,
                    fetchSeconds=fetch_seconds,
                    computeSeconds=compute_seconds,
                    readerWaitSeconds=runner.readerWaitSeconds,
                    computeWaitSeconds=compute_wait_seconds,
                    unitsCompleted=units_completed,
                    peakHeldBytes=runner.ledger.peak_bytes(),
                    extra={
                        "requestedGroupsInFlight": requested_group_reads,
                        "effectiveGroupsInFlight": in_flight,
                        "requestedChunkReadsInFlight": requested_chunk_reads,
                        "effectiveChunkReadsInFlight": in_flight * plan.innerReads,
                        "readGroupBytes": read_group_bytes,
                        "cellBandBytes": max_band_bytes,
                        "cellBandCount": len(bands),
                        "featureWidth": feature_width,
                    },
                )
            )
            if metrics is not None:
                metrics.update(report.as_metrics())

    return _iter_bounded_handoff(in_flight=in_flight, run=_run)


def map_feature_cell_bands(
    counts_t: Any,
    process: Callable[[FeatureCellBand], T],
    *,
    cell_idx: np.ndarray | None = None,
    feat_idx: Sequence[int] | np.ndarray | None = None,
    feat_starts: Sequence[int] | None = None,
    resources: ResourceBudget | None = None,
    progress: str | None = None,
    io: StorageIoPolicy | None = None,
    metrics: dict[str, Any] | None = None,
    scratchBytes: int = 0,
    orderedCompute: bool = True,
    cellMajorOrder: bool = False,
) -> Iterator[T]:
    """Map ``process`` over cell-band slices in deterministic traversal order."""
    array = as_zarr_array(counts_t)
    geometry = _plane(array)
    _n_feats, n_cells = (int(value) for value in geometry.shape)
    feat_chunk = geometry.axisChunk(0)
    cell_chunk = geometry.axisChunk(1)
    feature_width, read_group_bytes = persisted_read_group(array)
    merged = _feature_group_ranges(
        array,
        feat_idx=feat_idx,
        feat_starts=feat_starts,
        featureWidth=feature_width,
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
    if cellMajorOrder:
        work = [
            (feat_start, feat_end, cell_start, cell_end, local, destinations)
            for cell_start, cell_end, local, destinations in bands
            for feat_start, feat_end in merged
        ]
    else:
        work = [
            (feat_start, feat_end, cell_start, cell_end, local, destinations)
            for feat_start, feat_end in merged
            for cell_start, cell_end, local, destinations in bands
        ]
    if not work:
        return iter(())

    budget = resources or resolve_budget()
    resolved_io = io or DEFAULT_STORAGE_IO_POLICY
    itemsize = geometry.itemsize
    max_band_bytes = max(
        (feat_end - feat_start) * (cell_end - cell_start) * itemsize
        for feat_start, feat_end, cell_start, cell_end, _local, _destinations in work
    )
    plan = _plan_feature_consume(
        budget,
        io=resolved_io,
        nUnits=len(work),
        unitBytes=max_band_bytes,
        scratchBytes=scratchBytes,
        ordered=orderedCompute,
    )
    in_flight = plan.readWorkers
    fetch_seconds = 0.0
    compute_seconds = 0.0
    compute_wait_seconds = 0.0
    units_completed = 0
    if metrics is not None:
        metrics.clear()
        metrics.update(plan.as_metrics())
        metrics.update(
            {
                "requestedGroupsInFlight": plan.requestedReadWorkers,
                "effectiveGroupsInFlight": in_flight,
                "readGroupBytes": read_group_bytes,
                "cellBandBytes": max_band_bytes,
                "featureWidth": feature_width,
                "featureGroupCount": len(merged),
                "cellBandCount": len(bands),
                "cellMajorOrder": cellMajorOrder,
                "unitKind": "countsTCellBand",
            }
        )

    from ..utils.progress import tqdmbar

    progress_bar = tqdmbar(desc=progress, total=len(work)) if progress else None

    def _run(deliver: Callable[[T], None], stop: threading.Event) -> None:
        nonlocal fetch_seconds, compute_seconds, compute_wait_seconds, units_completed

        async def _operation(runner: AsyncStorageRunner) -> None:
            nonlocal fetch_seconds, compute_seconds, compute_wait_seconds
            nonlocal units_completed
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
                nonlocal next_idx, fetch_seconds, compute_seconds
                nonlocal compute_wait_seconds, units_completed
                if stop.is_set():
                    async with turn:
                        while next_idx != idx:
                            await turn.wait()
                        next_idx += 1
                        turn.notify_all()
                    return
                n_local = feat_end - feat_start
                read_bytes = max(1, n_local * (cell_end - cell_start) * itemsize)
                async with runner.reserve_bytes(read_bytes):
                    async with runner.read_lane():
                        started = time.perf_counter()
                        block = np.asarray(
                            await source.getitem(
                                (
                                    slice(feat_start, feat_end),
                                    slice(cell_start, cell_end),
                                )
                            )
                        )
                        read_seconds = time.perf_counter() - started
                    band = FeatureCellBand(
                        featStart=int(feat_start),
                        featEnd=int(feat_end),
                        cellStart=int(cell_start),
                        cellEnd=int(cell_end),
                        values=block,
                        selectedLocal=local,
                        selectedDestinations=destinations,
                        readSec=read_seconds,
                        blockBytes=int(block.nbytes),
                        unitIndex=idx,
                    )
                    fetch_seconds += band.readSec
                    wait_started = time.perf_counter()
                    if orderedCompute:
                        async with turn:
                            while next_idx != idx:
                                await turn.wait()
                            compute_wait_seconds += time.perf_counter() - wait_started
                            compute_started = time.perf_counter()
                            item = await runner.compute(lambda: process(band))
                            compute_seconds += time.perf_counter() - compute_started
                            await asyncio.to_thread(deliver, item)
                            next_idx += 1
                            turn.notify_all()
                    else:
                        compute_started = time.perf_counter()
                        item = await runner.compute(lambda: process(band))
                        compute_seconds += time.perf_counter() - compute_started
                        await asyncio.to_thread(deliver, item)
                    units_completed += 1
                    if progress_bar is not None:
                        progress_bar.update(1)

            pending: set[asyncio.Task[None]] = set()
            async with asyncio.TaskGroup() as tasks:
                for idx, spec in enumerate(work):
                    if stop.is_set():
                        break
                    pending.add(tasks.create_task(_one_band(idx, *spec)))
                    if len(pending) >= in_flight:
                        done, pending = await asyncio.wait(
                            pending,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        for completed in done:
                            completed.result()

        runner = AsyncStorageRunner(
            budget,
            operation=plan,
            chunksPerShard=max(1, geometry.axisShard(0) // feat_chunk),
            readGroupsInFlight=in_flight,
        )
        try:
            runner.run(_operation)
        finally:
            if progress_bar is not None:
                progress_bar.close()
            report = record_execution_report(
                ExecutionReport(
                    plan=plan,
                    unitKind="countsTCellBand",
                    actualReadWorkers=in_flight,
                    actualComputeWorkers=runner.plan.computeWorkerLimit,
                    actualWriteWorkers=1,
                    fetchSeconds=fetch_seconds,
                    computeSeconds=compute_seconds,
                    readerWaitSeconds=runner.readerWaitSeconds,
                    computeWaitSeconds=compute_wait_seconds,
                    unitsCompleted=units_completed,
                    peakHeldBytes=runner.ledger.peak_bytes(),
                )
            )
            if metrics is not None:
                metrics.update(report.as_metrics())

    return _iter_bounded_handoff(in_flight=in_flight, run=_run)
