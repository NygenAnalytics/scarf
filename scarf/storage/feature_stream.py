"""Geometry-aware planning and bounded reads for feature-column streams."""

import asyncio
import contextvars
import math
import operator
import queue
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from concurrent.futures import Future
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any, TypeVar

import numpy as np

from .async_execution import AsyncStorageRunner
from .budget import ResourceBudget, resolve_budget
from .execution import admit_stream
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

    ``values`` is the raw decoded band. When ``rows`` is set, ``values`` holds
    only those group-local feature rows; otherwise it holds every row of the
    group. ``selectedLocal`` indexes active cells inside that band.
    ``selectedDestinations`` maps those cells onto the caller-requested
    selected-cell order.
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
    rows: np.ndarray | None = None

    def featureRows(self) -> np.ndarray:
        """Return the group-local feature row of each row in ``values``."""
        if self.rows is not None:
            return self.rows
        return np.arange(self.featEnd - self.featStart, dtype=np.int64)


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
    """Plan variable-width feature blocks from physical chunk geometry.

    A block is one Zarr read of the selected cells and its features.
    ``blockBytes`` counts what the caller holds for a block, its read's result
    included, and the plan adds what the read holds beside the result.
    """
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

    n_cells = int(cell_indices.size)
    cell_chunks = -(-geometry.axisShard(cell_axis) // geometry.axisChunk(cell_axis))
    feature_chunks = -(
        -geometry.axisShard(feature_axis) // geometry.axisChunk(feature_axis)
    )
    per_chunk = np.bincount(geometry.binOf(feature_axis, feature_indices))
    # A block spans at most as many chunks as the fewest selected features
    # per chunk that add up to its width.
    smallest_first = np.cumsum(np.sort(per_chunk[per_chunk > 0]))

    def read_bytes(width: int, bins: int | None = None) -> tuple[int, int]:
        """Return what a block read holds beside its result, and per decode.

        Each concurrent decode can hold the compressed bytes of the chunks the
        block touches in its own shard, over ``bins`` feature chunks or as
        many as ``width`` features can span.
        """
        if bins is None:
            bins = max(1, int(np.searchsorted(smallest_first, width, side="right")))
        result = n_cells * width * geometry.itemsize
        chunks = cell_chunks * min(bins, feature_chunks)
        decode = geometry.readBytes(0, chunks, decodes=1)
        return geometry.readBytes(result, chunks, decodes=1) - result - decode, decode

    def fits(width: int, bins: int | None = None) -> bool:
        return (
            _owned_bytes(blockBytes, width) + sum(read_bytes(width, bins)) <= available
        )

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
        if any(not fits(block.indices.size, len(block.bins)) for block in blocks):
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

    copy_bytes, decode_bytes = read_bytes(
        max(block.indices.size for block in blocks),
        max(len(block.bins) for block in blocks),
    )
    block_bytes = copy_bytes + max(
        _owned_bytes(blockBytes, block.indices.size) for block in blocks
    )
    admission = admit_stream(
        resources,
        nBlocks=len(blocks),
        blockBytes=block_bytes,
        decodeBytes=decode_bytes,
        residentBytes=resident,
    )
    read_workers = admission.readWorkers
    io_concurrency = admission.ioConcurrency

    cell_bins = int(
        np.count_nonzero(np.bincount(geometry.binOf(cell_axis, cell_indices)))
    )
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
    featureWidth: int,
) -> list[tuple[int, int]]:
    geometry = _plane(array)
    n_feats = int(geometry.shape[0])
    group_width = max(1, int(featureWidth))
    merged: list[tuple[int, int]] = []
    # Chunk starts ascend, so each starts at or after the previous group's end.
    for start in selected_feature_chunk_starts(array, feat_idx):
        feat_end = min(start + geometry.axisChunk(0), n_feats)
        if (
            merged
            and start == merged[-1][1]
            and feat_end - merged[-1][0] <= group_width
        ):
            merged[-1] = (merged[-1][0], feat_end)
        else:
            merged.append((start, feat_end))
    return merged


def _sparse_group_rows(
    groups: list[tuple[int, int]],
    feat_idx: Sequence[int] | np.ndarray | None,
    *,
    n_feats: int,
) -> dict[tuple[int, int], np.ndarray]:
    """Return the selected local rows of groups where fewer than half are wanted.

    Zarr copies every decoded row it returns, so reading only the selected rows
    of a sparse selection avoids copying the rest of each group.
    """
    if feat_idx is None:
        return {}
    selected = np.zeros(n_feats, dtype=bool)
    selected[np.asarray(feat_idx, dtype=np.int64)] = True
    sparse: dict[tuple[int, int], np.ndarray] = {}
    for start, stop in groups:
        rows = np.flatnonzero(selected[start:stop])
        if 2 * rows.size < stop - start:
            sparse[(start, stop)] = rows
    return sparse


def _plan_feature_consume(
    budget: ResourceBudget,
    *,
    io: StorageIoPolicy,
    nUnits: int,
    unitBytes: int,
    scratchBytes: int = 0,
    innerReadBytes: int = 0,
    maxInnerReads: int | None = None,
    maxUnitsInFlight: int | None = None,
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
            maxUnitsInFlight=maxUnitsInFlight,
            ordered=ordered,
            writes=False,
            chunksPerShard=max(1, int(chunksPerShard)),
        ),
        policy=io,
    )


def _iter_bounded_handoff(
    *,
    in_flight: int,
    run: Callable[[Callable[[Any], Awaitable[None]], threading.Event], None],
) -> Iterator[Any]:
    handoff: queue.Queue[Any] = queue.Queue(maxsize=max(1, int(in_flight)))
    stop = threading.Event()
    sentinel = object()
    error: list[BaseException] = []

    async def deliver(item: Any) -> None:
        if stop.is_set():
            return
        released: Future[None] = Future()
        while not stop.is_set():
            try:
                handoff.put_nowait((item, released))
                break
            except queue.Full:
                await asyncio.sleep(0.01)
        else:
            return
        acknowledged = asyncio.wrap_future(released)
        while not stop.is_set():
            try:
                await asyncio.wait_for(asyncio.shield(acknowledged), timeout=0.05)
                return
            except TimeoutError:
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
                        _, released = handoff.get_nowait()
                        released.set_result(None)
                    except queue.Empty:
                        continue

    # The producer runs in a copy of the caller's context, so shutdown requests,
    # report scopes, and validation scopes reach it.
    thread = threading.Thread(
        target=contextvars.copy_context().run, args=(_worker,), daemon=True
    )
    thread.start()
    try:
        while True:
            entry = handoff.get()
            if entry is sentinel:
                break
            item, released = entry
            del entry
            try:
                yield item
            finally:
                # The producer may free the item's buffers once it is released.
                del item
                released.set_result(None)
    finally:
        stop.set()
        while thread.is_alive():
            try:
                entry = handoff.get(timeout=0.05)
                if entry is not sentinel:
                    _, released = entry
                    released.set_result(None)
            except queue.Empty:
                continue
        thread.join()
        if error:
            raise error[0]


def _copy_band(
    dest: np.ndarray, block: np.ndarray, local: np.ndarray, destinations: np.ndarray
) -> None:
    """Copy selected band columns into their destination columns."""
    if destinations[-1] - destinations[0] + 1 == len(destinations) and np.array_equal(
        local, np.arange(block.shape[1])
    ):
        np.copyto(dest[:, destinations[0] : destinations[-1] + 1], block)
        return
    from ..utils.strided import copy_columns

    copy_columns(block, local, dest, destinations)


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


def _touched_chunks(geometry: ArrayGeometry, n_features: int) -> int:
    """Return the inner chunks that one cell band of a feature group touches.

    Feature groups start at a chunk boundary and every band is one cell chunk.
    """
    return max(1, -(-int(n_features) // geometry.axisChunk(0)))


@dataclass(frozen=True, slots=True)
class _CellSelection:
    """Selected cells of one stream and the bands they fall in."""

    cells: np.ndarray
    bands: list[tuple[int, int, np.ndarray, np.ndarray]]
    indexBytes: int


def _cell_selection(
    geometry: ArrayGeometry, cell_idx: np.ndarray | None
) -> _CellSelection:
    """Split the selected cells into cell bands and count their index bytes.

    The bands hold a local and a destination index for every selected cell,
    and the stream also owns the selected cells when it creates them.
    """
    n_cells = int(geometry.shape[1])
    if cell_idx is None:
        cells = np.arange(n_cells, dtype=np.int64)
    else:
        cells = np.asarray(cell_idx, dtype=np.int64)
        if cells.size and (int(cells.min()) < 0 or int(cells.max()) >= n_cells):
            raise IndexError("cell_idx contains an out-of-range index")
    bands = _selected_cell_bands(
        cells, n_cells=n_cells, cell_chunk=geometry.axisChunk(1)
    )
    index_bytes = sum(
        int(local.nbytes) + int(destinations.nbytes)
        for _start, _end, local, destinations in bands
    )
    if cells is not cell_idx:
        index_bytes += int(cells.nbytes)
    return _CellSelection(cells=cells, bands=bands, indexBytes=index_bytes)


@dataclass(frozen=True, slots=True)
class _ReadGroupLayout:
    """Read groups of one stream and the bytes its plan reserves for them.

    ``unitBytes`` is the destination of the widest group over the selected
    cells, and ``readBytes`` what one band read of that group holds while
    Zarr decodes it.
    """

    geometry: ArrayGeometry
    groups: list[tuple[int, int]]
    selection: _CellSelection
    featureWidth: int
    readGroupBytes: int
    bandBytes: int
    unitBytes: int
    readBytes: int


def _read_group_layout(
    array: Any,
    *,
    cell_idx: np.ndarray | None,
    feat_idx: Sequence[int] | np.ndarray | None,
) -> _ReadGroupLayout | None:
    geometry = _plane(array)
    feature_width, read_group_bytes = persisted_read_group(array)
    groups = _feature_group_ranges(array, feat_idx=feat_idx, featureWidth=feature_width)
    if not groups:
        return None
    selection = _cell_selection(geometry, cell_idx)
    widest = max(feat_end - feat_start for feat_start, feat_end in groups)
    band_cells = max(
        (cell_end - cell_start for cell_start, cell_end, _l, _d in selection.bands),
        default=0,
    )
    band_bytes = widest * band_cells * geometry.itemsize
    return _ReadGroupLayout(
        geometry=geometry,
        groups=groups,
        selection=selection,
        featureWidth=feature_width,
        readGroupBytes=read_group_bytes,
        bandBytes=band_bytes,
        unitBytes=widest * int(selection.cells.shape[0]) * geometry.itemsize,
        readBytes=geometry.readBytes(band_bytes, _touched_chunks(geometry, widest)),
    )


def read_group_stream_bytes(
    counts_t: Any,
    *,
    cell_idx: np.ndarray | None = None,
    feat_idx: Sequence[int] | np.ndarray | None = None,
) -> tuple[int, int]:
    """Return what ``map_feature_read_groups`` reserves per group and once.

    Each read group in flight holds its destination over the selected cells
    and at least one band read while Zarr decodes it, and the stream holds its
    band indices once.
    """
    layout = _read_group_layout(
        as_zarr_array(counts_t), cell_idx=cell_idx, feat_idx=feat_idx
    )
    if layout is None:
        return 0, 0
    return layout.unitBytes + layout.readBytes, layout.selection.indexBytes


def read_group_stream_floor(
    counts_t: Any,
    *,
    cell_idx: np.ndarray | None = None,
    feat_idx: Sequence[int] | np.ndarray | None = None,
) -> int:
    """Return the fewest bytes ``map_feature_read_groups`` reserves.

    That is one read group gathered over the selected cells, one band read in
    flight, and the stream's band indices. Callers that size their own
    buffers up front leave this much of the budget to the stream.
    """
    return sum(read_group_stream_bytes(counts_t, cell_idx=cell_idx, feat_idx=feat_idx))


def read_group_rows(
    counts_t: Any,
    feat_idx: Sequence[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Return how many selected features each ``map_feature_read_groups`` group holds.

    The counts follow the stream's group order.
    """
    array = as_zarr_array(counts_t)
    feature_width, _ = persisted_read_group(array)
    groups = _feature_group_ranges(array, feat_idx=feat_idx, featureWidth=feature_width)
    bounds = np.asarray(groups, dtype=np.int64).reshape(-1, 2)
    if feat_idx is None:
        return bounds[:, 1] - bounds[:, 0]
    selected = np.sort(np.asarray(feat_idx, dtype=np.int64))
    return np.searchsorted(selected, bounds[:, 1]) - np.searchsorted(
        selected, bounds[:, 0]
    )


def _stream_units(
    units: Sequence[Any],
    load: Callable[[AsyncStorageRunner, int, Any], AbstractAsyncContextManager[Any]],
    process: Callable[[Any], T],
    *,
    plan: OperationPlan,
    unitKind: str,
    orderedCompute: bool,
    progress: str | None,
    metrics: dict[str, Any] | None,
    details: dict[str, Any],
) -> Iterator[T]:
    """Load, process, and hand off units in order with bounded read-ahead.

    ``load`` holds its buffer reservations until the unit's result has been
    handed off, and the stream drops its own references to the unit and the
    result before ``load`` releases them. With ``orderedCompute`` the units
    are processed in order; otherwise results arrive in completion order.
    """
    in_flight = plan.readWorkers
    fetch_seconds = 0.0
    compute_seconds = 0.0
    compute_wait_seconds = 0.0
    units_completed = 0
    if metrics is not None:
        metrics.clear()
        metrics.update(plan.as_metrics())
        metrics.update({**details, "unitKind": unitKind})

    from ..utils.progress import tqdmbar

    progress_bar = tqdmbar(desc=progress, total=len(units)) if progress else None

    def run(deliver: Callable[[T], Awaitable[None]], stop: threading.Event) -> None:
        nonlocal fetch_seconds, compute_seconds, compute_wait_seconds
        nonlocal units_completed

        async def operation(runner: AsyncStorageRunner) -> None:
            nonlocal fetch_seconds, compute_seconds, compute_wait_seconds
            nonlocal units_completed
            turn = asyncio.Condition()
            next_idx = 0

            async def one_unit(idx: int, unit: Any) -> None:
                nonlocal next_idx, fetch_seconds, compute_seconds
                nonlocal compute_wait_seconds, units_completed
                if stop.is_set():
                    # Units start in order, so every later unit also sees the
                    # stop request and none of them waits for this one's turn.
                    return
                async with load(runner, idx, unit) as item:
                    fetch_seconds += item.readSec
                    wait_started = time.perf_counter()
                    if orderedCompute:
                        async with turn:
                            while next_idx != idx:
                                await turn.wait()
                            compute_wait_seconds += time.perf_counter() - wait_started
                            compute_started = time.perf_counter()
                            result = await runner.compute(partial(process, item))
                            compute_seconds += time.perf_counter() - compute_started
                            await deliver(result)
                            next_idx += 1
                            turn.notify_all()
                    else:
                        compute_started = time.perf_counter()
                        result = await runner.compute(partial(process, item))
                        compute_seconds += time.perf_counter() - compute_started
                        await deliver(result)
                    del item, result
                    units_completed += 1
                    if progress_bar is not None:
                        progress_bar.update(1)

            pending: set[asyncio.Task[None]] = set()
            async with asyncio.TaskGroup() as tasks:
                for idx, unit in enumerate(units):
                    if stop.is_set():
                        break
                    pending.add(tasks.create_task(one_unit(idx, unit)))
                    if len(pending) >= in_flight:
                        done, pending = await asyncio.wait(
                            pending,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        for completed in done:
                            completed.result()

        runner = AsyncStorageRunner(operation=plan)
        try:
            runner.run(operation)
        finally:
            if progress_bar is not None:
                progress_bar.close()
            report = record_execution_report(
                ExecutionReport(
                    plan=plan,
                    unitKind=unitKind,
                    actualReadWorkers=in_flight,
                    actualComputeWorkers=runner.plan.computeWorkers,
                    actualWriteWorkers=1,
                    fetchSeconds=fetch_seconds,
                    computeSeconds=compute_seconds,
                    readerWaitSeconds=runner.readerWaitSeconds,
                    computeWaitSeconds=compute_wait_seconds,
                    unitsCompleted=units_completed,
                    peakHeldBytes=runner.ledger.peak_bytes(),
                    extra=dict(details),
                )
            )
            if metrics is not None:
                metrics.update(report.as_metrics())

    return _iter_bounded_handoff(in_flight=in_flight, run=run)


def map_feature_read_groups(
    counts_t: Any,
    process: Callable[[FeatureReadGroup], T],
    *,
    cell_idx: np.ndarray | None = None,
    feat_idx: Sequence[int] | np.ndarray | None = None,
    resources: ResourceBudget | None = None,
    progress: str | None = None,
    io: StorageIoPolicy | None = None,
    metrics: dict[str, Any] | None = None,
    scratchBytes: int = 0,
    orderedCompute: bool = True,
) -> Iterator[T]:
    """Map ``process`` over persisted read groups with bounded handoff.

    A read group holds every selected cell. With ``orderedCompute`` one
    compute worker processes the groups one at a time in order while the next
    group is read. Otherwise up to the policy's compute workers each process
    one group while one more group is read, and results arrive in completion
    order. The plan reserves each group's destination over the selected cells,
    what each band read holds while Zarr decodes it, the stream's band
    indices, and ``scratchBytes`` for the caller, which must cover every
    ``process`` call that can run at once.
    """
    array = as_zarr_array(counts_t)
    layout = _read_group_layout(array, cell_idx=cell_idx, feat_idx=feat_idx)
    if layout is None:
        return iter(())
    geometry = layout.geometry
    merged = layout.groups
    bands = layout.selection.bands
    n_selected = int(layout.selection.cells.shape[0])
    budget = resources or resolve_budget()
    resolved_io = io or DEFAULT_STORAGE_IO_POLICY
    itemsize = geometry.itemsize
    requested_chunk_reads = (
        int(resolved_io.readWorkers)
        if resolved_io.readWorkers is not None
        else auto_read_width(budget.workers)
    )
    # Each computing group has one more group read beside it, and the band
    # reads of the groups in flight share the requested read width.
    compute_width = (
        1
        if orderedCompute
        else min(budget.workers, resolved_io.computeWorkers or budget.workers)
    )
    requested_group_reads = min(requested_chunk_reads, compute_width + 1)
    available_group_reads = max(1, min(len(merged), requested_group_reads))
    requested_inner_reads = min(
        max(1, len(bands)),
        max(1, math.ceil(requested_chunk_reads / available_group_reads)),
    )
    chunk_io = StorageIoPolicy(
        readWorkers=requested_chunk_reads,
        computeWorkers=resolved_io.computeWorkers,
        writeWorkers=resolved_io.writeWorkers,
    )
    plan = _plan_feature_consume(
        budget,
        io=chunk_io,
        nUnits=len(merged),
        unitBytes=layout.unitBytes,
        scratchBytes=scratchBytes + layout.selection.indexBytes,
        innerReadBytes=layout.readBytes,
        maxInnerReads=requested_inner_reads,
        maxUnitsInFlight=requested_group_reads,
        chunksPerShard=max(1, geometry.axisShard(0) // geometry.axisChunk(0)),
        ordered=orderedCompute,
    )
    source = array.async_array

    async def read_group(
        runner: AsyncStorageRunner, idx: int, feat_start: int, feat_end: int
    ) -> FeatureReadGroup:
        n_local = feat_end - feat_start
        chunks = _touched_chunks(geometry, n_local)
        dest = np.empty((n_local, n_selected), dtype=array.dtype)

        async def read_band(
            cell_start: int,
            cell_end: int,
            local: np.ndarray,
            destinations: np.ndarray,
        ) -> float:
            read_bytes = geometry.readBytes(
                n_local * (cell_end - cell_start) * itemsize, chunks
            )
            async with runner.read_lane():
                async with runner.reserve_bytes(read_bytes):
                    started = time.perf_counter()
                    block = np.asarray(
                        await runner.io(
                            source.getitem(
                                (
                                    slice(feat_start, feat_end),
                                    slice(cell_start, cell_end),
                                )
                            )
                        )
                    )
                    read_seconds = time.perf_counter() - started
                    await runner.offload(
                        partial(_copy_band, dest, block, local, destinations)
                    )
                    del block
            return read_seconds

        read_seconds = sum(await asyncio.gather(*(read_band(*band) for band in bands)))
        return FeatureReadGroup(
            featStart=int(feat_start),
            featEnd=int(feat_end),
            values=dest,
            readSec=read_seconds,
            blockBytes=int(dest.nbytes),
            unitIndex=idx,
        )

    @asynccontextmanager
    async def load(
        runner: AsyncStorageRunner, idx: int, unit: tuple[int, int]
    ) -> AsyncIterator[FeatureReadGroup]:
        feat_start, feat_end = unit
        destination_bytes = (feat_end - feat_start) * n_selected * itemsize
        async with runner.reserve_bytes(max(1, destination_bytes)):
            group = await read_group(runner, idx, feat_start, feat_end)
            yield group
            del group

    return _stream_units(
        merged,
        load,
        process,
        plan=plan,
        unitKind="countsTReadGroup",
        orderedCompute=orderedCompute,
        progress=progress,
        metrics=metrics,
        details={
            "requestedGroupsInFlight": requested_group_reads,
            "effectiveGroupsInFlight": plan.readWorkers,
            "requestedChunkReadsInFlight": requested_chunk_reads,
            "effectiveChunkReadsInFlight": plan.readWorkers * plan.innerReads,
            "readGroupBytes": layout.readGroupBytes,
            "cellBandBytes": layout.bandBytes,
            "cellBandReadBytes": layout.readBytes,
            "cellBandCount": len(bands),
            "featureWidth": layout.featureWidth,
        },
    )


def map_feature_cell_bands(
    counts_t: Any,
    process: Callable[[FeatureCellBand], T],
    *,
    cell_idx: np.ndarray | None = None,
    feat_idx: Sequence[int] | np.ndarray | None = None,
    resources: ResourceBudget | None = None,
    progress: str | None = None,
    io: StorageIoPolicy | None = None,
    metrics: dict[str, Any] | None = None,
    scratchBytes: int = 0,
    orderedCompute: bool = True,
    cellMajorOrder: bool = False,
) -> Iterator[T]:
    """Map ``process`` over cell-band slices in deterministic traversal order.

    A band stays reserved from its read until its result has been handed off,
    so the plan charges what the read holds while Zarr decodes it, together
    with the stream's band and row indices and ``scratchBytes`` for the caller.
    """
    array = as_zarr_array(counts_t)
    geometry = _plane(array)
    n_feats = int(geometry.shape[0])
    feature_width, read_group_bytes = persisted_read_group(array)
    merged = _feature_group_ranges(
        array,
        feat_idx=feat_idx,
        featureWidth=feature_width,
    )
    if not merged:
        return iter(())
    sparse_rows = _sparse_group_rows(merged, feat_idx, n_feats=n_feats)
    selection = _cell_selection(geometry, cell_idx)
    bands = selection.bands
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
    widest = max(feat_end - feat_start for feat_start, feat_end in merged)
    max_band_bytes = (
        widest
        * max(cell_end - cell_start for cell_start, cell_end, _l, _d in bands)
        * itemsize
    )
    plan = _plan_feature_consume(
        budget,
        io=resolved_io,
        nUnits=len(work),
        unitBytes=geometry.readBytes(max_band_bytes, _touched_chunks(geometry, widest)),
        scratchBytes=scratchBytes
        + selection.indexBytes
        + sum(int(rows.nbytes) for rows in sparse_rows.values()),
        chunksPerShard=max(1, geometry.axisShard(0) // geometry.axisChunk(0)),
        ordered=orderedCompute,
    )
    source = array.async_array

    @asynccontextmanager
    async def load(
        runner: AsyncStorageRunner, idx: int, unit: tuple[Any, ...]
    ) -> AsyncIterator[FeatureCellBand]:
        feat_start, feat_end, cell_start, cell_end, local, destinations = unit
        n_local = feat_end - feat_start
        read_bytes = geometry.readBytes(
            n_local * (cell_end - cell_start) * itemsize,
            _touched_chunks(geometry, n_local),
        )
        rows = sparse_rows.get((feat_start, feat_end))
        async with runner.reserve_bytes(read_bytes):
            async with runner.read_lane():
                started = time.perf_counter()
                cells = slice(cell_start, cell_end)
                if rows is None:
                    block = np.asarray(
                        await runner.io(
                            source.getitem((slice(feat_start, feat_end), cells))
                        )
                    )
                else:
                    block = np.asarray(
                        await runner.io(
                            source.get_orthogonal_selection((rows + feat_start, cells))
                        )
                    )
                read_seconds = time.perf_counter() - started
            yield FeatureCellBand(
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
                rows=rows,
            )
            del block

    return _stream_units(
        work,
        load,
        process,
        plan=plan,
        unitKind="countsTCellBand",
        orderedCompute=orderedCompute,
        progress=progress,
        metrics=metrics,
        details={
            "requestedGroupsInFlight": plan.requestedReadWorkers,
            "effectiveGroupsInFlight": plan.readWorkers,
            "readGroupBytes": read_group_bytes,
            "cellBandBytes": max_band_bytes,
            "featureWidth": feature_width,
            "featureGroupCount": len(merged),
            "cellBandCount": len(bands),
            "cellMajorOrder": cellMajorOrder,
        },
    )
