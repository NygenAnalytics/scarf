"""Geometry-aware planning and bounded reads for feature-column streams."""

import asyncio
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
        width = group_width
    else:
        starts = [int(value) for value in feat_starts]
        width = group_width
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


def _admit_groups_in_flight(
    budget: ResourceBudget,
    *,
    io: StorageIoPolicy,
    readGroupBytes: int,
) -> tuple[int, int | None]:
    memory_max = max(1, int(budget.memoryBytes) // max(1, int(readGroupBytes)))
    requested = io.groupsInFlight
    if requested is None:
        return min(memory_max, max(1, int(budget.workers))), None
    return min(int(requested), memory_max), int(requested)


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
            release.put(None)
            yield item
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
    in_flight, requested = _admit_groups_in_flight(
        budget,
        io=resolved_io,
        readGroupBytes=read_group_bytes,
    )
    if metrics is not None:
        metrics.clear()
        metrics.update(
            {
                "requestedGroupsInFlight": requested,
                "effectiveGroupsInFlight": in_flight,
                "readGroupBytes": read_group_bytes,
                "featureWidth": feature_width,
            }
        )

    from ..utils.progress import tqdmbar

    progress_bar = tqdmbar(desc=progress, total=len(merged)) if progress else None

    def _run(deliver: Callable[[T], None], stop: threading.Event) -> None:
        async def _operation(runner: AsyncStorageRunner) -> None:
            source = array.async_array
            turn = asyncio.Condition()
            next_idx = 0

            async def _one_group(idx: int, feat_start: int, feat_end: int) -> None:
                nonlocal next_idx
                if stop.is_set():
                    async with turn:
                        while next_idx != idx:
                            await turn.wait()
                        next_idx += 1
                        turn.notify_all()
                    return
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
                        item = process(group)
                        await asyncio.to_thread(deliver, item)
                        next_idx += 1
                        turn.notify_all()
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

        try:
            AsyncStorageRunner(
                budget,
                chunksPerShard=max(1, geometry.axisShard(0) // feat_chunk),
                readGroupsInFlight=in_flight,
            ).run(_operation)
        finally:
            if progress_bar is not None:
                progress_bar.close()

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
) -> Iterator[T]:
    """Map ``process`` over cell-band slices in deterministic feature order."""
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
    in_flight, requested = _admit_groups_in_flight(
        budget,
        io=resolved_io,
        readGroupBytes=read_group_bytes,
    )
    if metrics is not None:
        metrics.clear()
        metrics.update(
            {
                "requestedGroupsInFlight": requested,
                "effectiveGroupsInFlight": in_flight,
                "readGroupBytes": read_group_bytes,
                "featureWidth": feature_width,
            }
        )

    from ..utils.progress import tqdmbar

    progress_bar = tqdmbar(desc=progress, total=len(work)) if progress else None

    def _run(deliver: Callable[[T], None], stop: threading.Event) -> None:
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
                if stop.is_set():
                    async with turn:
                        while next_idx != idx:
                            await turn.wait()
                        next_idx += 1
                        turn.notify_all()
                    return
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
                        item = process(band)
                        await asyncio.to_thread(deliver, item)
                        next_idx += 1
                        turn.notify_all()
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

        try:
            AsyncStorageRunner(
                budget,
                chunksPerShard=max(1, geometry.axisShard(0) // feat_chunk),
                readGroupsInFlight=in_flight,
            ).run(_operation)
        finally:
            if progress_bar is not None:
                progress_bar.close()

    return _iter_bounded_handoff(in_flight=in_flight, run=_run)
