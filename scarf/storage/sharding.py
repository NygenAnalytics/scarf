import asyncio
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import zarr

from .async_execution import AsyncStorageRunner
from .budget import ResourceBudget, resolve_budget
from .execution import admitted_worker_split
from .execution import (
    ExecutionReport,
    OperationPlan,
    WorkShape,
    plan_operation,
    record_execution_report,
)
from .count_matrix import (
    DEFAULT_COUNT_MATRIX_POLICY,
    CountMatrixPolicy,
    create_count_matrix_array,
    load_count_matrix_plan,
    persist_count_matrix_plan,
    plan_count_matrix_pair,
    policy_from_payload,
    validate_count_matrix_source,
)
from .io_policy import DEFAULT_STORAGE_IO_POLICY, StorageIoPolicy
from .geometry import ArrayGeometry, array_geometry
from .layout import (
    ZarrArraySpec,
    _encoded_chunk_bound,
    _group_zarr_format,
    array_shard_rows,
    iter_shard_row_slices,
)
from .parallel import _close_iterator, in_shard_context, stream_shards
from .partition import affordable_width
from .profiles import StorageProfile, resolve_storage_profile
from .types import array_metadata_shards, as_zarr_array, writable
from ..utils.arrays import canonicalize_sparse, checked_sparse_cast

if TYPE_CHECKING:
    from .identity import CountSummary


@dataclass(frozen=True, slots=True)
class _DenseWriteBand:
    start: int
    end: int
    values: np.ndarray


@dataclass(frozen=True, slots=True)
class SparseRowBand:
    start: int
    end: int
    nColumns: int
    row: np.ndarray
    column: np.ndarray
    data: np.ndarray
    dtype: Any

    @property
    def sparseBytes(self) -> int:
        return int(self.row.nbytes + self.column.nbytes + self.data.nbytes)

    def dense(self) -> np.ndarray:
        dense = np.zeros(
            (self.end - self.start, self.nColumns),
            dtype=self.dtype,
        )
        dense[self.row, self.column] = checked_sparse_cast(self.data, self.dtype)
        return dense


@dataclass(frozen=True, slots=True)
class SparseWriteBand:
    destination: zarr.Array
    band: SparseRowBand
    producerBytes: int = 0


@dataclass(frozen=True, slots=True)
class SparseImportPlan:
    batchRows: int
    producerReserveBytes: int
    writeTasks: int


def _destination_geometry(
    destination: zarr.Array | ZarrArraySpec,
) -> tuple[ArrayGeometry, np.dtype[Any]]:
    if isinstance(destination, ZarrArraySpec):
        if len(destination.shape) != 2 or len(destination.chunks) != 2:
            raise ValueError(
                "Sparse import destination specifications must be two-dimensional"
            )
        geometry = ArrayGeometry(
            shape=(int(destination.shape[0]), int(destination.shape[1])),
            chunks=(int(destination.chunks[0]), int(destination.chunks[1])),
            shards=(
                None
                if destination.shards is None
                else tuple(int(value) for value in destination.shards)
            ),
            itemsize=max(1, int(np.dtype(destination.dtype).itemsize)),
        )
        return geometry, np.dtype(destination.dtype)
    resolved = array_geometry(destination)
    if resolved is None or len(resolved.shape) != 2:
        raise ValueError("Sparse import destinations must be two-dimensional arrays")
    return resolved, np.dtype(destination.dtype)


def sparse_matrix_bytes(*matrices: Any) -> int:
    """Return unique array bytes owned by SciPy sparse matrices."""
    arrays = (
        getattr(matrix, name, None)
        for matrix in matrices
        for name in ("data", "row", "col", "indices", "indptr")
    )
    unique = {id(array): array for array in arrays if isinstance(array, np.ndarray)}
    return int(sum(array.nbytes for array in unique.values()))


def sparse_producer_peak_bytes(
    bufferedValues: int,
    sourceValues: int,
    valueItemsize: int,
) -> int:
    """Bound one sparse pull through projection, buffering, and band splitting."""
    buffered_values = max(0, int(bufferedValues))
    source_values = max(0, int(sourceValues))
    value_bytes = max(1, int(valueItemsize))
    index_bytes = np.dtype(np.int64).itemsize
    mask_bytes = np.dtype(np.bool_).itemsize
    triplet_bytes = value_bytes + 2 * index_bytes
    buffered_bytes = buffered_values * (
        2 * triplet_bytes + index_bytes + 2 * mask_bytes
    )
    source_bytes = 2 * source_values * triplet_bytes
    source_indexes = source_values * (2 * index_bytes + 2 * mask_bytes)
    canonical_value_bytes = max(value_bytes, np.dtype(np.int64).itemsize)
    canonicalization_bytes = source_values * (
        4 * canonical_value_bytes + 6 * index_bytes + 4
    )
    return buffered_bytes + source_bytes + source_indexes + canonicalization_bytes


def row_band_task_count(nRows: int, bandRows: int) -> int:
    """Return the number of fixed-height row bands covering a matrix."""
    rows = max(0, int(nRows))
    band_rows = int(bandRows)
    if band_rows <= 0:
        raise ValueError("bandRows must be positive")
    if rows == 0:
        return 0
    return (rows + band_rows - 1) // band_rows


def resolve_sparse_import_batch(
    destinations: Sequence[zarr.Array],
    *,
    nRows: int,
    resources: ResourceBudget,
    maxWindowNnz: Callable[[int], int],
    sourceDtype: Any,
    batchRows: int | None = None,
    residentBytes: int = 0,
    producerStagingBytes: Callable[[int], int] | None = None,
    extraProducerBytes: Callable[[int], int] | None = None,
) -> SparseImportPlan:
    """Resolve sparse source rows for physical destination arrays."""
    return _resolve_sparse_import_geometries(
        tuple(_destination_geometry(destination) for destination in destinations),
        nRows=nRows,
        resources=resources,
        maxWindowNnz=maxWindowNnz,
        sourceDtype=sourceDtype,
        batchRows=batchRows,
        residentBytes=residentBytes,
        producerStagingBytes=producerStagingBytes,
        extraProducerBytes=extraProducerBytes,
    )


def resolve_sparse_import_spec(
    destinations: Sequence[ZarrArraySpec],
    *,
    nRows: int,
    resources: ResourceBudget,
    maxWindowNnz: Callable[[int], int],
    sourceDtype: Any,
    batchRows: int | None = None,
    residentBytes: int = 0,
    producerStagingBytes: Callable[[int], int] | None = None,
    extraProducerBytes: Callable[[int], int] | None = None,
) -> SparseImportPlan:
    """Resolve sparse source rows before destination arrays are created."""
    return _resolve_sparse_import_geometries(
        tuple(_destination_geometry(destination) for destination in destinations),
        nRows=nRows,
        resources=resources,
        maxWindowNnz=maxWindowNnz,
        sourceDtype=sourceDtype,
        batchRows=batchRows,
        residentBytes=residentBytes,
        producerStagingBytes=producerStagingBytes,
        extraProducerBytes=extraProducerBytes,
    )


def _resolve_sparse_import_geometries(
    destinations: Sequence[tuple[ArrayGeometry, np.dtype[Any]]],
    *,
    nRows: int,
    resources: ResourceBudget,
    maxWindowNnz: Callable[[int], int],
    sourceDtype: Any,
    batchRows: int | None = None,
    residentBytes: int = 0,
    producerStagingBytes: Callable[[int], int] | None = None,
    extraProducerBytes: Callable[[int], int] | None = None,
) -> SparseImportPlan:
    """Resolve source rows while admitting one complete destination write band."""
    destination_list = tuple(destinations)
    if not destination_list:
        raise ValueError("At least one sparse import destination is required")
    rows = int(nRows)
    if rows < 0:
        raise ValueError("nRows cannot be negative")
    for geometry, _dtype in destination_list:
        if geometry.shape[0] != rows:
            raise ValueError(
                "Sparse import destinations must match the source row count"
            )
    if batchRows is not None and int(batchRows) <= 0:
        raise ValueError("batch_size must be positive")

    source_dtype = np.dtype(sourceDtype)
    value_itemsize = max(
        source_dtype.itemsize,
        *(dtype.itemsize for _geometry, dtype in destination_list),
    )
    staging_bytes = producerStagingBytes or (lambda _: 0)
    extra_bytes = extraProducerBytes or (lambda _: 0)
    maximum_shard_rows = max(
        geometry.axisShard(0) for geometry, _dtype in destination_list
    )
    write_tasks = sum(
        row_band_task_count(rows, geometry.axisShard(0))
        for geometry, _dtype in destination_list
    )

    band_requirements: list[tuple[ArrayGeometry, np.dtype[Any], int, int]] = []
    if rows:
        for geometry, dtype in destination_list:
            band_rows = min(rows, geometry.axisShard(0))
            band_values = max(0, int(maxWindowNnz(band_rows)))
            band_sparse_bytes = band_values * (
                source_dtype.itemsize + 2 * np.dtype(np.int64).itemsize
            )
            band_requirements.append((geometry, dtype, band_values, band_sparse_bytes))

    def reserve(width: int) -> int:
        source_values = max(0, int(maxWindowNnz(width)))
        buffered_values = max(
            0,
            int(maxWindowNnz(width + maximum_shard_rows)),
        )
        return int(
            sparse_producer_peak_bytes(
                buffered_values,
                source_values,
                value_itemsize,
            )
            + max(0, int(staging_bytes(width)))
            + max(0, int(extra_bytes(width)))
        )

    def admit(width: int) -> int:
        producer_reserve = reserve(width)
        for (
            geometry,
            destination_dtype,
            band_values,
            band_sparse_bytes,
        ) in band_requirements:
            dense_bytes, inner_bytes, n_chunks = _band_geometry(geometry)
            conversion_bytes = (
                band_values * destination_dtype.itemsize
                if source_dtype != destination_dtype
                else 0
            )
            if source_dtype.kind == "f" and destination_dtype.kind in "biu":
                conversion_bytes += band_values * (source_dtype.itemsize + 1)
            admitted_worker_split(
                resources,
                nTasks=1,
                residentBytes=(
                    max(0, int(residentBytes)) + producer_reserve + band_sparse_bytes
                ),
                taskBytes=lambda inner: _row_band_task_peak(
                    sourceBytes=conversion_bytes,
                    denseBytes=dense_bytes,
                    innerChunkBytes=inner_bytes,
                    nChunks=n_chunks,
                    innerConcurrency=inner,
                ),
                requested=1,
            )
        if not band_requirements:
            admitted_worker_split(
                resources,
                nTasks=1,
                residentBytes=max(0, int(residentBytes)) + producer_reserve,
                taskBytes=lambda _: 1,
                requested=1,
            )
        return producer_reserve

    if batchRows is not None:
        resolved_rows = int(batchRows)
        producer_reserve = admit(resolved_rows)
    elif rows == 0:
        resolved_rows = 1
        producer_reserve = admit(resolved_rows)
    else:
        preferred_rows = min(
            rows,
            min(geometry.axisShard(0) for geometry, _dtype in destination_list),
        )

        def fits(width: int) -> bool:
            try:
                admit(width)
            except MemoryError:
                return False
            return True

        resolved_rows = affordable_width(fits, preferred_rows)
        if resolved_rows < 1:
            try:
                admit(1)
            except MemoryError as exc:
                raise MemoryError(
                    "Automatic sparse import cannot fit one source row and one "
                    "complete destination row band. Increase mem_budget or use "
                    "smaller target chunk and shard byte limits."
                ) from exc
            raise RuntimeError("Sparse import admission failed unexpectedly")
        producer_reserve = reserve(resolved_rows)

    return SparseImportPlan(
        batchRows=resolved_rows,
        producerReserveBytes=producer_reserve,
        writeTasks=write_tasks,
    )


def aligned_row_windows(
    nRows: int,
    shardRows: int,
    nWindows: int,
) -> list[tuple[int, int]]:
    """Split rows into shard-aligned windows for disjoint producers."""
    rows = max(0, int(nRows))
    shard = max(1, int(shardRows))
    if rows == 0:
        return []
    n_shards = (rows + shard - 1) // shard
    windows = max(1, min(int(nWindows), n_shards))
    base, extra = divmod(n_shards, windows)
    ranges: list[tuple[int, int]] = []
    shard_index = 0
    for window in range(windows):
        count = base + (1 if window < extra else 0)
        start = shard_index * shard
        shard_index += count
        end = min(rows, shard_index * shard)
        if start < end:
            ranges.append((start, end))
    return ranges


class SparseShardBuffer:
    """Accumulate sparse source batches into immutable destination row bands."""

    def __init__(
        self,
        destination: zarr.Array | ZarrArraySpec,
        *,
        startRow: int = 0,
        endRow: int | None = None,
    ) -> None:
        geometry, dtype = _destination_geometry(destination)
        self.nRows = max(0, int(geometry.shape[0]))
        self.nColumns = max(0, int(geometry.shape[1]))
        self.shardRows = geometry.axisShard(0)
        self.dtype = dtype
        self.startRow = max(0, int(startRow))
        self.endRow = self.nRows if endRow is None else int(endRow)
        if self.startRow > self.endRow or self.endRow > self.nRows:
            raise ValueError("Sparse buffer row window is outside the destination")
        self.rows = self.startRow
        if self.startRow < self.endRow:
            next_boundary = ((self.startRow // self.shardRows) + 1) * self.shardRows
            self._nextFlush = min(self.endRow, next_boundary)
        else:
            self._nextFlush = self.startRow
        self._rows: list[np.ndarray] = []
        self._columns: list[np.ndarray] = []
        self._data: list[np.ndarray] = []

    @property
    def residentBytes(self) -> int:
        arrays = (*self._rows, *self._columns, *self._data)
        return int(sum(array.nbytes for array in arrays))

    def add(self, batch: Any) -> Iterator[SparseRowBand]:
        from scipy.sparse import coo_matrix

        coo = batch if hasattr(batch, "row") else coo_matrix(batch)
        if coo.shape[1] != self.nColumns:
            raise ValueError(
                f"Sparse batch has {coo.shape[1]} columns, expected {self.nColumns}"
            )
        if self.rows + int(coo.shape[0]) > self.nRows:
            raise ValueError("Sparse batches contain more rows than the destination")
        if coo.nnz:
            coo = coo.tocoo(copy=False)
            if not bool(getattr(coo, "has_canonical_format", False)):
                coo = canonicalize_sparse(coo, self.dtype)
            self._rows.append(np.asarray(coo.row, dtype=np.int64) + self.rows)
            self._columns.append(np.asarray(coo.col, dtype=np.int64))
            self._data.append(np.asarray(coo.data))
        self.rows += int(coo.shape[0])
        while self._nextFlush <= self.rows:
            end = self._nextFlush
            start = max(self.startRow, end - self.shardRows)
            self._nextFlush += self.shardRows
            yield self._take(start, end)

    def finish(self) -> Iterator[SparseRowBand]:
        if self.rows != self.endRow:
            raise ValueError(
                f"Sparse stream contains {self.rows} rows, expected {self.endRow}"
            )
        # An empty window never flushed, so its trailing band must not start
        # before the window does.
        trailing_start = max(self.startRow, self._nextFlush - self.shardRows)
        if trailing_start < self.rows:
            yield self._take(trailing_start, self.rows)
        elif self._rows or self._columns or self._data:
            raise RuntimeError("Sparse coordinates remain after the final row band")

    def _take(self, start: int, end: int) -> SparseRowBand:
        row, column, data = self._drain()
        selected = row < end
        keep = ~selected
        if keep.any():
            self._rows = [row[keep]]
            self._columns = [column[keep]]
            self._data = [data[keep]]
        local_rows = row[selected] - start
        return SparseRowBand(
            start=start,
            end=end,
            nColumns=self.nColumns,
            row=local_rows,
            column=column[selected],
            data=data[selected],
            dtype=self.dtype,
        )

    def _drain(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        def joined(parts: list[np.ndarray], dtype: Any) -> np.ndarray:
            if not parts:
                return np.array([], dtype=dtype)
            return np.concatenate(parts)

        values = (
            joined(self._rows, np.int64),
            joined(self._columns, np.int64),
            joined(self._data, self.dtype),
        )
        self._rows = []
        self._columns = []
        self._data = []
        return values


def _band_geometry(
    destination: zarr.Array | ArrayGeometry,
    nRows: int | None = None,
) -> tuple[int, int, int]:
    if isinstance(destination, ArrayGeometry):
        geometry = destination
    else:
        resolved = array_geometry(destination)
        if resolved is None:
            raise ValueError("Destination array has no chunk geometry")
        geometry = resolved
    rows = geometry.axisShard(0) if nRows is None else max(1, int(nRows))
    columns = max(1, geometry.shape[1])
    itemsize = max(1, geometry.itemsize)
    chunk_rows = min(rows, geometry.chunks[0])
    chunk_columns = min(columns, geometry.chunks[1])
    n_chunks = (
        (rows + chunk_rows - 1)
        // chunk_rows
        * ((columns + chunk_columns - 1) // chunk_columns)
    )
    dense_bytes = rows * columns * itemsize
    inner_bytes = chunk_rows * chunk_columns * itemsize
    return dense_bytes, inner_bytes, n_chunks


def _shard_index_bound(nChunks: int) -> int:
    """Conservative encoded-size bound for a sharding index."""
    return max(1, int(nChunks)) * 16 + 1024


def _row_band_task_peak(
    *,
    sourceBytes: int,
    denseBytes: int,
    innerChunkBytes: int,
    nChunks: int,
    innerConcurrency: int,
) -> int:
    encoded_parts = max(1, int(nChunks)) * _encoded_chunk_bound(innerChunkBytes)
    return (
        max(0, int(sourceBytes))
        + max(0, int(denseBytes))
        + min(max(1, int(nChunks)), max(1, int(innerConcurrency)))
        * max(0, int(innerChunkBytes))
        + encoded_parts
        + encoded_parts
        + _shard_index_bound(nChunks)
    )


def plan_dense_write(
    destination: zarr.Array,
    resources: ResourceBudget,
    nTasks: int,
    io: StorageIoPolicy | None = None,
    *,
    residentBytes: int = 0,
    producerBytes: int = 0,
    resultBytes: int = 0,
    mirror: zarr.Array | None = None,
) -> OperationPlan:
    dense_bytes, inner_bytes, n_chunks = _band_geometry(destination)
    # Keep encoded chunks and the assembled shard alongside each dense band.
    unit_bytes = (
        _row_band_task_peak(
            sourceBytes=producerBytes + resultBytes,
            denseBytes=dense_bytes,
            innerChunkBytes=inner_bytes,
            nChunks=n_chunks,
            innerConcurrency=1,
        )
        - inner_bytes
    )
    if mirror is not None:
        mirror_dense, mirror_inner, mirror_chunks = _band_geometry(mirror)
        conversion = mirror_dense if mirror.dtype != destination.dtype else 0
        unit_bytes = max(
            unit_bytes,
            _row_band_task_peak(
                sourceBytes=producerBytes + resultBytes + conversion,
                denseBytes=dense_bytes,
                innerChunkBytes=mirror_inner,
                nChunks=mirror_chunks,
                innerConcurrency=1,
            )
            - mirror_inner,
        )
        inner_bytes = max(inner_bytes, mirror_inner)
        n_chunks = max(n_chunks, mirror_chunks)
    return plan_operation(
        resources,
        WorkShape(
            nUnits=max(1, int(nTasks)),
            unitBytes=unit_bytes,
            residentBytes=residentBytes + resultBytes,
            innerReadBytes=inner_bytes,
            maxInnerReads=n_chunks,
            writes=True,
            chunksPerShard=max(1, n_chunks),
        ),
        policy=io,
    )


def _sparse_task_working_bytes(
    write: SparseWriteBand,
    innerConcurrency: int,
) -> int:
    dense_bytes, inner_bytes, n_chunks = _band_geometry(
        write.destination,
        write.band.end - write.band.start,
    )
    source_dtype = np.dtype(write.band.data.dtype)
    destination_dtype = np.dtype(write.band.dtype)
    values = int(write.band.data.size)
    conversion_bytes = (
        values * destination_dtype.itemsize if source_dtype != destination_dtype else 0
    )
    if source_dtype.kind == "f" and destination_dtype.kind in "biu":
        conversion_bytes += values * (source_dtype.itemsize + 1)
    return _row_band_task_peak(
        sourceBytes=conversion_bytes,
        denseBytes=dense_bytes,
        innerChunkBytes=inner_bytes,
        nChunks=n_chunks,
        innerConcurrency=innerConcurrency,
    )


def _sparse_batch_plan(
    pending: deque[SparseWriteBand],
    resources: ResourceBudget,
    residentBytes: int,
    producerReserveBytes: int,
    nTasks: int,
    io: StorageIoPolicy,
) -> OperationPlan:
    sparse_bytes = sum(write.band.sparseBytes for write in pending)
    producer_bytes = max(write.producerBytes for write in pending)
    geometries = [
        _band_geometry(write.destination, write.band.end - write.band.start)
        for write in pending
    ]
    inner_bytes = max(geometry[1] for geometry in geometries)
    chunks = max(geometry[2] for geometry in geometries)
    unit = max(
        _sparse_task_working_bytes(write, 1) - geometry[1]
        for write, geometry in zip(pending, geometries, strict=True)
    )
    return plan_operation(
        resources,
        WorkShape(
            nUnits=nTasks,
            unitBytes=unit,
            residentBytes=residentBytes
            + max(producerReserveBytes, producer_bytes)
            + sparse_bytes,
            innerReadBytes=inner_bytes,
            maxInnerReads=chunks,
            chunksPerShard=chunks,
            writes=True,
        ),
        policy=io,
    )


def write_sparse_bands(
    writes: Iterator[SparseWriteBand],
    *,
    resources: ResourceBudget,
    residentBytes: int = 0,
    producerReserveBytes: int = 0,
    msg: str | None = None,
    total: int | None = None,
    io: StorageIoPolicy | None = None,
    countSummaries: dict[str, "CountSummary"] | None = None,
) -> None:
    """Densify and write complete sparse row bands within CPU and memory limits.

    ``countSummaries`` maps destination paths to summaries filled from each band.
    """
    pending: deque[SparseWriteBand] = deque()
    source = iter(writes)
    exhausted = False
    progress = None
    resolved_io = io or DEFAULT_STORAGE_IO_POLICY
    if msg is not None:
        from ..utils.progress import tqdmbar

        progress = tqdmbar(desc=msg, total=total)

    def pull() -> SparseWriteBand:
        sparse_bytes = sum(write.band.sparseBytes for write in pending)
        producer_bytes = max(
            producerReserveBytes,
            max((write.producerBytes for write in pending), default=0),
        )
        reserved = residentBytes + producer_bytes + sparse_bytes
        if reserved > resources.memoryBytes:
            raise MemoryError(
                f"Sparse producer needs about {reserved} bytes before a write "
                f"task, but the operation limit is {resources.memoryBytes} bytes"
            )
        return next(source)

    def fill() -> None:
        nonlocal exhausted
        if not pending and not exhausted:
            try:
                pending.append(pull())
            except StopIteration:
                exhausted = True
                return
        while not exhausted:
            operation = _sparse_batch_plan(
                pending,
                resources,
                residentBytes,
                producerReserveBytes,
                resources.workers,
                resolved_io,
            )
            capacity = min(operation.computeWorkers, operation.writeWorkers)
            if len(pending) >= capacity:
                return
            try:
                pending.append(pull())
            except StopIteration:
                exhausted = True

    def write_one(item: SparseWriteBand) -> None:
        dense = item.band.dense()
        if countSummaries is not None:
            countSummaries[item.destination.path].update(item.band.start, dense)
        writable(item.destination)[item.band.start : item.band.end, :] = dense

    try:
        fill()
        while pending:
            operation = _sparse_batch_plan(
                pending,
                resources,
                residentBytes,
                producerReserveBytes,
                len(pending),
                resolved_io,
            )
            admitted = min(operation.computeWorkers, operation.writeWorkers)
            inner = operation.ioConcurrency
            batch = [pending.popleft() for _ in range(admitted)]
            for _ in stream_shards(
                batch,
                write_one,
                workers=admitted,
                within_block_threads=1,
                io_concurrency=inner,
            ):
                if progress is not None:
                    progress.update()
            record_execution_report(
                ExecutionReport(
                    plan=operation,
                    unitKind="countsImportBand",
                    actualReadWorkers=1,
                    actualComputeWorkers=admitted,
                    actualWriteWorkers=admitted,
                    unitsCompleted=len(batch),
                )
            )
            del batch
            fill()
    finally:
        from contextlib import ExitStack

        with ExitStack() as cleanup:
            if progress is not None:
                cleanup.callback(progress.close)
            cleanup.callback(_close_iterator, source)


def write_dense_from_row_batches(
    dst: zarr.Array,
    batches: Iterator[np.ndarray],
    *,
    dtype: Any | None = None,
    msg: str | None = None,
    resources: ResourceBudget | None = None,
    io: StorageIoPolicy | None = None,
    producerReserveBytes: int | None = None,
    residentBytes: int = 0,
    countSummary: "CountSummary | None" = None,
) -> int:
    """Align source batches to destination row bands and write them in parallel."""
    resources = resources or resolve_budget()
    source = iter(batches)
    if int(dst.shape[0]) == 0:
        try:
            for batch in source:
                values = np.asarray(batch)
                if values.ndim != 2 or values.shape[1] != dst.shape[1]:
                    raise ValueError("Dense source batch has an invalid shape")
                if values.shape[0]:
                    raise ValueError(
                        "Dense stream contains rows for an empty destination"
                    )
        finally:
            _close_iterator(source)
        return 0
    shard_rows = array_shard_rows(dst)
    target_dtype = np.dtype(dst.dtype if dtype is None else dtype)
    first: list[np.ndarray] = []
    try:
        for batch in source:
            values = np.asarray(batch)
            if values.ndim != 2 or values.shape[1] != dst.shape[1]:
                raise ValueError("Dense source batch has an invalid shape")
            if values.shape[0]:
                first.append(values)
                break
        if not first:
            raise ValueError(f"Dense stream contains 0 rows, expected {dst.shape[0]}")
        allocation = first[0]
        while isinstance(allocation.base, np.ndarray):
            allocation = allocation.base
        producer_bytes = (
            2 * allocation.nbytes
            if producerReserveBytes is None
            else producerReserveBytes
        )
        n_bands = (int(dst.shape[0]) + shard_rows - 1) // shard_rows
        operation = plan_dense_write(
            dst,
            resources,
            n_bands,
            io=io,
            residentBytes=residentBytes + producer_bytes,
            producerBytes=shard_rows
            * dst.shape[1]
            * max(0, target_dtype.itemsize - dst.dtype.itemsize),
        )
        del batch, values, allocation
    except BaseException:
        _close_iterator(source)
        raise

    def source_batches() -> Iterator[np.ndarray]:
        yield first.pop()
        yield from source

    def aligned() -> Iterator[_DenseWriteBand]:
        n_columns = int(dst.shape[1])
        buffer = np.empty((shard_rows, n_columns), dtype=target_dtype)
        buffered_rows = 0
        position = 0
        try:
            for batch in source_batches():
                values = np.asarray(batch)
                if values.ndim != 2 or values.shape[1] != dst.shape[1]:
                    raise ValueError("Dense source batch has an invalid shape")
                if values.shape[0] == 0:
                    continue
                if values.nbytes > producer_bytes:
                    raise MemoryError(
                        f"Dense source batch needs {values.nbytes} bytes, exceeding "
                        f"its {producer_bytes}-byte producer reservation"
                    )
                if position + buffered_rows + values.shape[0] > dst.shape[0]:
                    raise ValueError(
                        "Dense stream contains more rows than its destination"
                    )
                source_start = 0
                while source_start < int(values.shape[0]):
                    copied = min(
                        shard_rows - buffered_rows,
                        int(values.shape[0]) - source_start,
                    )
                    buffer[buffered_rows : buffered_rows + copied] = values[
                        source_start : source_start + copied
                    ]
                    buffered_rows += copied
                    source_start += copied
                    if buffered_rows == shard_rows:
                        yield _DenseWriteBand(
                            position,
                            position + shard_rows,
                            buffer,
                        )
                        position += shard_rows
                        buffer = np.empty(
                            (shard_rows, n_columns),
                            dtype=target_dtype,
                        )
                        buffered_rows = 0
            if buffered_rows:
                yield _DenseWriteBand(
                    position,
                    position + buffered_rows,
                    buffer[:buffered_rows],
                )
        finally:
            _close_iterator(source)

    target = writable(dst)

    def write_band(band: _DenseWriteBand) -> int:
        if countSummary is not None:
            countSummary.update(band.start, band.values)
        target[band.start : band.end, :] = band.values
        return band.end - band.start

    total_rows = int(
        sum(
            stream_shards(
                aligned(),
                write_band,
                workers=min(operation.computeWorkers, operation.writeWorkers),
                within_block_threads=1,
                io_concurrency=operation.ioConcurrency,
                msg=msg or "Writing Zarr array",
                total=n_bands,
            )
        )
    )
    if total_rows != int(dst.shape[0]):
        raise ValueError(
            f"Dense stream contains {total_rows} rows, expected {dst.shape[0]}"
        )
    return total_rows


def write_dense_in_shard_rows(
    dst: zarr.Array,
    produce: Callable[[int, int], np.ndarray],
    *,
    msg: str | None = None,
    also_write_to: zarr.Array | None = None,
    resources: ResourceBudget | None = None,
    summarize: Callable[[np.ndarray], Any] | None = None,
    merge_summary: Callable[[Any, Any], Any] | None = None,
    io: StorageIoPolicy | None = None,
    residentBytes: int = 0,
    producerBytes: int = 0,
    resultBytes: int = 0,
    countSummary: "CountSummary | None" = None,
) -> Any | None:
    """Produce and write complete destination row bands in the same worker."""
    if (summarize is None) != (merge_summary is None):
        raise ValueError("summarize and merge_summary must be provided together")
    merger = merge_summary
    resources = resources or resolve_budget()
    n_rows = int(dst.shape[0])
    if n_rows == 0:
        return None
    rows = array_shard_rows(dst)
    if also_write_to is not None and (
        tuple(also_write_to.shape) != tuple(dst.shape)
        or array_shard_rows(also_write_to) != rows
    ):
        raise ValueError("Mirror array must have matching shape and row-band layout")
    slices = list(iter_shard_row_slices(n_rows, rows))
    operation = plan_dense_write(
        dst,
        resources,
        len(slices),
        io=io,
        residentBytes=residentBytes,
        producerBytes=producerBytes,
        resultBytes=resultBytes,
        mirror=also_write_to,
    )
    workers = (
        1
        if in_shard_context()
        else min(operation.computeWorkers, operation.writeWorkers)
    )
    inner = operation.ioConcurrency
    target = writable(dst)
    mirror = None if also_write_to is None else writable(also_write_to)

    def produce_and_write(bounds: tuple[int, int]) -> Any:
        start, end = bounds
        block = np.asarray(produce(start, end))
        expected = (end - start, int(dst.shape[1]))
        if block.shape != expected:
            raise ValueError(
                f"Dense producer returned shape {block.shape}, expected {expected}"
            )
        if countSummary is not None:
            countSummary.update(start, block)
        target[start:end, :] = block
        if mirror is not None:
            mirror[start:end, :] = block
        return None if summarize is None else summarize(block)

    summary: Any | None = None
    completed = 0
    for result in stream_shards(
        slices,
        produce_and_write,
        workers=workers,
        within_block_threads=operation.threadsPerComputeWorker,
        io_concurrency=inner,
        msg=msg or "Writing Zarr array",
        total=len(slices),
    ):
        completed += 1
        if summarize is not None:
            assert merger is not None
            summary = result if summary is None else merger(summary, result)
    record_execution_report(
        ExecutionReport(
            plan=operation,
            unitKind="countsRowBand",
            actualReadWorkers=workers,
            actualComputeWorkers=workers,
            actualWriteWorkers=workers,
            unitsCompleted=completed,
        )
    )
    return summary


def accumulate_sparse_to_shards(
    dst: zarr.Array,
    data_stream: Iterator[Any],
    *,
    resources: ResourceBudget | None = None,
    residentBytes: int = 0,
    producerReserveBytes: int,
    msg: str | None = None,
    io: StorageIoPolicy | None = None,
    countSummary: "CountSummary | None" = None,
) -> int:
    """Write one complete dense row band per sparse destination object."""
    resources = resources or resolve_budget()
    source = iter(data_stream)
    if int(dst.shape[0]) == 0:
        try:
            for batch in source:
                if len(batch.shape) != 2 or int(batch.shape[1]) != int(dst.shape[1]):
                    raise ValueError("Sparse source batch has an invalid shape")
                if int(batch.shape[0]):
                    raise ValueError(
                        "Sparse stream contains rows for an empty destination"
                    )
        finally:
            _close_iterator(source)
        return 0
    buffer = SparseShardBuffer(dst)

    def writes() -> Iterator[SparseWriteBand]:
        from scipy.sparse import coo_matrix

        try:
            for batch in source:
                coo = batch if hasattr(batch, "row") else coo_matrix(batch)
                source_bytes = sparse_matrix_bytes(batch, coo)
                for band in buffer.add(coo):
                    yield SparseWriteBand(
                        dst,
                        band,
                        source_bytes + buffer.residentBytes,
                    )
                del batch, coo
            for band in buffer.finish():
                yield SparseWriteBand(dst, band, buffer.residentBytes)
        finally:
            _close_iterator(source)

    try:
        write_sparse_bands(
            writes(),
            resources=resources,
            residentBytes=residentBytes,
            producerReserveBytes=producerReserveBytes,
            msg=msg,
            total=(int(dst.shape[0]) + array_shard_rows(dst) - 1)
            // array_shard_rows(dst),
            io=io,
            countSummaries=None if countSummary is None else {dst.path: countSummary},
        )
    finally:
        _close_iterator(source)
    return buffer.rows


def counts_t_spec(
    counts: ZarrArraySpec,
    *,
    profile: StorageProfile,
) -> ZarrArraySpec:
    """Return the paired rotateOnce feature-major layout derived from counts."""
    if len(counts.shape) != 2 or len(counts.chunks) != 2:
        raise ValueError("counts must be a two-dimensional array specification")
    n_cells = int(counts.shape[0])
    n_feats = int(counts.shape[1])
    return plan_count_matrix_pair(
        n_cells,
        n_feats,
        counts.dtype,
        profile=profile,
    ).countsT


def is_paired_counts_t_layout(
    *,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    shards: tuple[int, ...] | None,
    dtype: Any,
) -> bool:
    """Return True for a structurally valid paired countsT lattice."""
    del dtype
    if len(shape) != 2 or len(chunks) != 2:
        return False
    if shards is None or len(shards) != 2:
        return False
    n_feats, n_cells = (int(value) for value in shape)
    chunk_f, chunk_c = (int(value) for value in chunks)
    shard_f, shard_c = (int(value) for value in shards)
    if n_feats < 0 or n_cells < 0:
        return False
    if min(chunk_f, chunk_c, shard_f, shard_c) < 1:
        return False
    if shard_f % chunk_f or shard_c % chunk_c:
        return False
    return True


def is_readable_counts_t_layout(
    *,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    shards: tuple[int, ...] | None,
    dtype: Any,
) -> bool:
    """Return True for a countsT array the bounded reader can consume."""
    return is_paired_counts_t_layout(
        shape=shape, chunks=chunks, shards=shards, dtype=dtype
    )


def _counts_t_write_peak(spec: ZarrArraySpec, innerConcurrency: int = 1) -> int:
    shards = spec.shards or spec.chunks
    itemsize = int(np.dtype(spec.dtype).itemsize)
    n_chunks = int(np.prod(shards)) // int(np.prod(spec.chunks))
    return _row_band_task_peak(
        sourceBytes=0,
        denseBytes=int(np.prod(shards)) * itemsize,
        innerChunkBytes=int(np.prod(spec.chunks)) * itemsize,
        nChunks=n_chunks,
        innerConcurrency=innerConcurrency,
    )


def _counts_t_read_peak(spec: ZarrArraySpec) -> int:
    chunk_bytes = int(np.prod(spec.chunks)) * int(np.dtype(spec.dtype).itemsize)
    n_chunks = int(np.prod(spec.shards or spec.chunks)) // int(np.prod(spec.chunks))
    return (
        2 * chunk_bytes
        + _encoded_chunk_bound(chunk_bytes)
        + _shard_index_bound(n_chunks)
    )


def preflight_counts_t_spec(
    counts: ZarrArraySpec,
    *,
    profile: StorageProfile,
    resources: ResourceBudget,
    residentBytes: int = 0,
    policy: CountMatrixPolicy | None = None,
) -> ZarrArraySpec:
    """Admit paired ``countsT`` construction from the destination-shard writer."""
    if len(counts.shape) != 2:
        raise ValueError("counts must be a two-dimensional array specification")
    plan = plan_count_matrix_pair(
        int(counts.shape[0]),
        int(counts.shape[1]),
        counts.dtype,
        policy=policy or DEFAULT_COUNT_MATRIX_POLICY,
        profile=profile,
    )
    if 0 in counts.shape:
        return plan.countsT
    needed = (
        _counts_t_write_peak(plan.countsT)
        + _counts_t_read_peak(plan.counts)
        + max(0, int(residentBytes))
    )
    if needed > int(resources.memoryBytes):
        raise MemoryError(
            "countsT write needs at least "
            f"{needed} bytes, but the operation limit is "
            f"{int(resources.memoryBytes)} bytes"
        )
    return plan.countsT


def _counts_t_matches_plan(counts_t: zarr.Array, plan: Any) -> bool:
    if counts_t.attrs.get("complete") is not True:
        return False
    try:
        recorded = load_count_matrix_plan(counts_t)
    except ValueError:
        return False
    if recorded.get("fingerprint") != plan.fingerprint:
        return False
    stored = array_metadata_shards(counts_t)
    stored_shards = None if stored is None else tuple(int(value) for value in stored)
    return bool(
        tuple(int(value) for value in counts_t.shape) == plan.countsT.shape
        and tuple(int(value) for value in counts_t.chunks) == plan.countsT.chunks
        and stored_shards == plan.countsT.shards
    )


def write_counts_t(
    counts: zarr.Array,
    group: zarr.Group,
    *,
    profile: StorageProfile | None = None,
    resources: ResourceBudget | None = None,
    residentBytes: int = 0,
    policy: CountMatrixPolicy | None = None,
    io: StorageIoPolicy | None = None,
    metrics: dict[str, Any] | None = None,
    overwrite: bool = False,
    featureSets: Sequence[np.ndarray] = (),
) -> zarr.Array:
    """Write paired rotateOnce feature-major ``countsT``.

    Each index set in ``featureSets`` also gets its per-cell count total, taken
    from the decoded source blocks and saved next to the count summaries.
    """
    if _group_zarr_format(group) < 3:
        raise ValueError("paired countsT requires Zarr format 3")
    resources = resources or resolve_budget()
    resolved_profile = profile or resolve_storage_profile(group.store)
    if policy is None:
        resolved_policy = policy_from_payload(load_count_matrix_plan(counts))
    else:
        resolved_policy = policy
    resolved_io = io or DEFAULT_STORAGE_IO_POLICY
    plan = plan_count_matrix_pair(
        int(counts.shape[0]),
        int(counts.shape[1]),
        counts.dtype,
        policy=resolved_policy,
        profile=resolved_profile,
    )
    validate_count_matrix_source(counts, expected=plan)
    from ..utils.strided import transpose_into
    from .identity import count_fingerprint, feature_sums_key, write_feature_sums

    source_fingerprint = count_fingerprint(counts)
    if "countsT" in group and not overwrite:
        existing = as_zarr_array(group["countsT"], name="countsT")
        if existing.attrs.get(
            "source_fingerprint"
        ) == source_fingerprint and _counts_t_matches_plan(existing, plan):
            return existing
        raise ValueError(
            "Existing countsT is incomplete or mismatched; use overwrite=True to rewrite it"
        )
    preflight_counts_t_spec(
        plan.counts,
        profile=resolved_profile,
        resources=resources,
        residentBytes=max(0, int(residentBytes)),
        policy=resolved_policy,
    )
    counts_t = as_zarr_array(
        create_count_matrix_array(group, "countsT", plan.countsT),
        name="countsT",
    )
    counts_t.attrs.update({"complete": False, "source_fingerprint": source_fingerprint})
    persist_count_matrix_plan(group, plan)
    persist_count_matrix_plan(counts, plan)
    persist_count_matrix_plan(counts_t, plan)
    n_cells = int(counts.shape[0])
    n_feats = int(counts.shape[1])
    if n_cells == 0 or n_feats == 0:
        counts_t.attrs["complete"] = True
        return counts_t
    feature_sets = [np.unique(np.asarray(f, dtype=np.int64)) for f in featureSets]
    set_totals = [np.zeros(n_cells, dtype=np.float64) for _ in feature_sets]
    dest_feat_shard = int(plan.countsT.shards[0]) if plan.countsT.shards else n_feats
    dest_cell_band = int(plan.countsT.shards[1]) if plan.countsT.shards else n_cells
    source_cell_chunk = int(plan.counts.chunks[0])
    source_feat_chunk = int(plan.counts.chunks[1])
    source_feat_shard = int(plan.counts.shards[1]) if plan.counts.shards else n_feats
    max_group_chunks = max(1, source_feat_shard // source_feat_chunk)
    itemsize = int(np.dtype(counts.dtype).itemsize)
    dest_unit_bytes = max(1, dest_feat_shard * dest_cell_band * itemsize)
    source_read_bytes = _counts_t_read_peak(plan.counts)
    resident_bytes = max(0, int(residentBytes))
    available_bytes = max(0, int(resources.memoryBytes) - resident_bytes)
    cell_chunks_per_dest = max(1, -(-dest_cell_band // max(1, source_cell_chunk)))
    feat_chunks_per_dest = max(1, -(-dest_feat_shard // max(1, source_feat_chunk)))
    touched_source_chunks = cell_chunks_per_dest * feat_chunks_per_dest
    dest_jobs = [
        (
            feat_start,
            min(feat_start + dest_feat_shard, n_feats),
            cell_start,
            min(cell_start + dest_cell_band, n_cells),
        )
        for cell_start in range(0, n_cells, dest_cell_band)
        for feat_start in range(0, n_feats, dest_feat_shard)
    ]
    encode_chunk_bytes = int(np.prod(plan.countsT.chunks)) * itemsize
    minimum_write_peak = _counts_t_write_peak(plan.countsT)
    encode_concurrency = min(
        resources.workers,
        max(
            1,
            (available_bytes - source_read_bytes - minimum_write_peak)
            // encode_chunk_bytes
            + 1,
        ),
    )
    destination_peak_bytes = _counts_t_write_peak(plan.countsT, encode_concurrency)
    encoding_bytes = destination_peak_bytes - dest_unit_bytes
    # A destination finishes its source reads before it encodes, so the reads
    # borrow the encoding workspace instead of competing with commits for bytes.
    source_reads_per_destination = max(
        1, min(touched_source_chunks, encoding_bytes // source_read_bytes)
    )
    working_bytes = max(
        encoding_bytes, source_reads_per_destination * source_read_bytes
    )
    requested_read_workers = (
        resources.workers
        if resolved_io.readWorkers is None
        else int(resolved_io.readWorkers)
    )
    transpose_io = StorageIoPolicy(
        readWorkers=max(1, requested_read_workers),
        computeWorkers=resolved_io.computeWorkers,
        writeWorkers=resolved_io.writeWorkers,
    )
    operation = plan_operation(
        resources,
        WorkShape(
            nUnits=len(dest_jobs),
            unitBytes=dest_unit_bytes + working_bytes,
            residentBytes=resident_bytes,
            writes=True,
            # Zarr encodes up to ioConcurrency inner chunks of one shard at once.
            chunksPerShard=min(
                encode_concurrency, max(max_group_chunks, touched_source_chunks)
            ),
        ),
        policy=transpose_io,
    )
    effective_commits_in_flight = operation.writeWorkers
    dest_in_flight = operation.readWorkers
    observed: dict[str, Any] = {
        "mode": "destination-shard",
        **operation.as_metrics(),
        "requestedSourceReadsInFlight": operation.requestedReadWorkers
        or operation.innerReads,
        "requestedDestCommitsInFlight": operation.requestedWriteWorkers
        or operation.writeWorkers,
        "requestedDestShardsInFlight": operation.requestedReadWorkers or dest_in_flight,
        "effectiveDestShardsInFlight": dest_in_flight,
        "effectiveDestinationShardsInFlight": dest_in_flight,
        "requestedComputeWorkers": operation.requestedComputeWorkers
        or operation.computeWorkers,
        "effectiveSourceReadsInFlight": dest_in_flight * source_reads_per_destination,
        "sourceReadsPerDestination": source_reads_per_destination,
        "sourceReadGroups": 0,
        "sourceLogicalBytes": 0,
        "sourceDecodeBytes": 0,
        "sourceRepeatedDecodeCount": 0,
        "sourceRepeatedDecodeBytes": 0,
        "destinationCommits": 0,
        "destinationLogicalBytes": 0,
        "destinationOwners": 0,
        "plannedDestinationSetBytes": dest_unit_bytes,
        "destinationEncodingBytes": encoding_bytes,
        "destinationWorkingBytes": working_bytes,
        "kind": "observed",
    }
    seen_source_chunks: set[tuple[int, int]] = set()

    def _chunk_starts(start: int, stop: int, chunk: int) -> range:
        return range((start // chunk) * chunk, stop, chunk)

    async def _operation(runner: AsyncStorageRunner) -> None:
        source = counts.async_array
        destination = writable(counts_t).async_array

        async def _process_destination(
            feat_start: int, feat_end: int, cell_start: int, cell_end: int
        ) -> None:
            async with runner.reserve_bytes(dest_unit_bytes + working_bytes):
                buffer = np.empty(
                    (feat_end - feat_start, cell_end - cell_start),
                    dtype=counts.dtype,
                )

                async def _read_and_scatter(
                    source_cell_start: int, source_feat_start: int
                ) -> None:
                    source_cell_end = min(
                        n_cells, source_cell_start + source_cell_chunk
                    )
                    source_feat_end = min(
                        n_feats, source_feat_start + source_feat_chunk
                    )
                    # The destination reservation covers these reads.
                    payload = np.asarray(
                        await runner.io(
                            source.getitem(
                                (
                                    slice(source_cell_start, source_cell_end),
                                    slice(source_feat_start, source_feat_end),
                                )
                            )
                        )
                    )

                    feats = slice(
                        max(source_feat_start, feat_start),
                        min(source_feat_end, feat_end),
                    )
                    cells = slice(
                        max(source_cell_start, cell_start),
                        min(source_cell_end, cell_end),
                    )

                    def _scatter() -> list[np.ndarray]:
                        block = payload[
                            cells.start - source_cell_start : cells.stop
                            - source_cell_start,
                            feats.start - source_feat_start : feats.stop
                            - source_feat_start,
                        ]
                        transpose_into(
                            block,
                            buffer[
                                feats.start - feat_start : feats.stop - feat_start,
                                cells.start - cell_start : cells.stop - cell_start,
                            ],
                        )
                        # Each (cell, feature) is scattered once, so these
                        # partial totals add up to exact per-cell sums.
                        return [
                            block[
                                :,
                                features[
                                    np.searchsorted(features, feats.start) : (
                                        np.searchsorted(features, feats.stop)
                                    )
                                ]
                                - feats.start,
                            ].sum(axis=1, dtype=np.float64)
                            for features in feature_sets
                        ]

                    partials = await runner.compute(_scatter)
                    for totals, partial in zip(set_totals, partials, strict=True):
                        totals[cells] += partial
                    observed["sourceReadGroups"] = int(observed["sourceReadGroups"]) + 1
                    observed["sourceLogicalBytes"] = int(
                        observed["sourceLogicalBytes"]
                    ) + int(payload.nbytes)
                    chunk_key = (
                        source_cell_start // source_cell_chunk,
                        source_feat_start // source_feat_chunk,
                    )
                    decode_bytes = source_cell_chunk * source_feat_chunk * itemsize
                    observed["sourceDecodeBytes"] = (
                        int(observed["sourceDecodeBytes"]) + decode_bytes
                    )
                    if chunk_key in seen_source_chunks:
                        observed["sourceRepeatedDecodeCount"] = (
                            int(observed["sourceRepeatedDecodeCount"]) + 1
                        )
                        observed["sourceRepeatedDecodeBytes"] = (
                            int(observed["sourceRepeatedDecodeBytes"]) + decode_bytes
                        )
                    else:
                        seen_source_chunks.add(chunk_key)

                pending: set[asyncio.Task[None]] = set()
                async with asyncio.TaskGroup() as read_tasks:
                    for source_cell_start in _chunk_starts(
                        cell_start, cell_end, source_cell_chunk
                    ):
                        for source_feat_start in _chunk_starts(
                            feat_start, feat_end, source_feat_chunk
                        ):
                            pending.add(
                                read_tasks.create_task(
                                    _read_and_scatter(
                                        source_cell_start, source_feat_start
                                    )
                                )
                            )
                            if len(pending) >= source_reads_per_destination:
                                done, pending = await asyncio.wait(
                                    pending,
                                    return_when=asyncio.FIRST_COMPLETED,
                                )
                                for completed in done:
                                    completed.result()

                async with runner.commit_lane():
                    await runner.io(
                        destination.setitem(
                            (slice(feat_start, feat_end), slice(cell_start, cell_end)),
                            buffer,
                        )
                    )
                observed["destinationCommits"] = int(observed["destinationCommits"]) + 1
                observed["destinationLogicalBytes"] = int(
                    observed["destinationLogicalBytes"]
                ) + int(buffer.nbytes)

        pending_dest: set[asyncio.Task[None]] = set()
        async with asyncio.TaskGroup() as dest_tasks:
            for job in dest_jobs:
                pending_dest.add(dest_tasks.create_task(_process_destination(*job)))
                if len(pending_dest) >= dest_in_flight:
                    done, pending_dest = await asyncio.wait(
                        pending_dest,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for completed in done:
                        completed.result()

    runner = AsyncStorageRunner(
        operation=operation,
    )
    try:
        runner.run(_operation)
    except BaseException:
        observed["terminalStatus"] = "error"
        observed["peakLedgerBytes"] = runner.ledger.peak_bytes()
        observed["heldLedgerBytes"] = runner.ledger.held_bytes()
        if metrics is not None:
            metrics.clear()
            metrics.update(observed)
        counts_t.attrs["complete"] = False
        raise
    observed["destinationOwners"] = len(dest_jobs)
    observed["effectiveComputeWorkers"] = runner.plan.computeWorkers
    observed["effectiveComputeWorkerLimit"] = runner.plan.computeWorkers
    observed["effectiveDestCommitsInFlight"] = effective_commits_in_flight
    observed["terminalStatus"] = "ok"
    observed["peakLedgerBytes"] = runner.ledger.peak_bytes()
    observed["heldLedgerBytes"] = runner.ledger.held_bytes()
    observed["plannedDestinationBufferBytes"] = plan.destinationBufferBytes
    observed["plannedSourceBufferBytes"] = plan.sourceBufferBytes
    observed["sourceDecodeAmplification"] = plan.sourceDecodeAmplification
    if metrics is not None:
        metrics.clear()
        metrics.update(observed)
    if feature_sets:
        write_feature_sums(
            group,
            counts,
            {
                feature_sums_key(features): (features, totals)
                for features, totals in zip(feature_sets, set_totals, strict=True)
            },
        )
    record_execution_report(
        ExecutionReport(
            plan=operation,
            unitKind="countsRowShard",
            actualReadWorkers=dest_in_flight,
            actualComputeWorkers=int(observed["effectiveComputeWorkers"]),
            actualWriteWorkers=effective_commits_in_flight,
            unitsCompleted=int(observed["destinationOwners"]),
            peakHeldBytes=int(observed["peakLedgerBytes"]),
            extra=observed,
        )
    )
    counts_t.attrs["complete"] = True
    return counts_t
