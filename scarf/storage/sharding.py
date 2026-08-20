import asyncio
import contextvars
import threading
from collections import deque
from collections.abc import Callable, Coroutine, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import zarr

from .async_execution import AsyncStorageRunner
from .budget import (
    ResourceBudget,
    admitted_worker_split,
    resolve_budget,
)
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
from .parallel import _close_iterator, stream_shards
from .partition import affordable_width
from .profiles import StorageProfile, resolve_storage_profile
from .types import array_metadata_shards, as_zarr_array
from ..utils.arrays import canonicalize_sparse, checked_sparse_cast


def _run_async(factory: Callable[[], Coroutine[Any, Any, None]]) -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(factory())
        return

    error: list[BaseException] = []
    context = contextvars.copy_context()

    def run() -> None:
        try:
            context.run(lambda: asyncio.run(factory()))
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    if error:
        raise error[0]


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


def sparse_write_task_count(
    destinations: Sequence[zarr.Array],
    nRows: int,
) -> int:
    """Return the exact number of destination row bands an import will write."""
    rows = max(0, int(nRows))
    if rows == 0:
        return 0
    return int(
        sum(
            row_band_task_count(rows, array_shard_rows(destination))
            for destination in destinations
        )
    )


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
        trailing_start = self._nextFlush - self.shardRows
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


def _writer_count(
    destination: zarr.Array,
    resources: ResourceBudget,
    nTasks: int,
    io: StorageIoPolicy | None = None,
) -> tuple[int, int]:
    dense_bytes, inner_bytes, n_chunks = _band_geometry(destination)
    operation = plan_operation(
        resources,
        WorkShape(
            nUnits=max(1, int(nTasks)),
            unitBytes=max(1, dense_bytes),
            decodeBytes=max(0, inner_bytes),
            writes=True,
            chunksPerShard=max(1, n_chunks),
        ),
        policy=io,
    )
    return operation.writeWorkers, operation.ioConcurrency


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
) -> tuple[int, int]:
    sparse_bytes = sum(write.band.sparseBytes for write in pending)
    producer_bytes = max(write.producerBytes for write in pending)
    return admitted_worker_split(
        resources,
        nTasks=nTasks,
        residentBytes=(
            residentBytes + max(producerReserveBytes, producer_bytes) + sparse_bytes
        ),
        taskBytes=lambda inner: max(
            _sparse_task_working_bytes(write, inner) for write in pending
        ),
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
) -> None:
    """Densify and write complete sparse row bands within CPU and memory limits."""
    pending: deque[SparseWriteBand] = deque()
    source = iter(writes)
    exhausted = False
    progress = None
    resolved_io = io or DEFAULT_STORAGE_IO_POLICY
    if msg is not None:
        from ..utils.progress import tqdmbar

        progress = tqdmbar(desc=msg, total=total)

    operation = None
    actual_write_workers = 0

    def _plan_pending() -> OperationPlan:
        unit = max(_sparse_task_working_bytes(write, 1) for write in pending)
        return plan_operation(
            resources,
            WorkShape(
                nUnits=max(1, int(total) if total else len(pending)),
                unitBytes=unit,
                residentBytes=residentBytes,
                writes=True,
                ordered=False,
            ),
            policy=resolved_io,
        )

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
        nonlocal exhausted, operation
        if not pending and not exhausted:
            try:
                pending.append(pull())
            except StopIteration:
                exhausted = True
                return
        while not exhausted:
            operation = _plan_pending()
            capacity, _ = _sparse_batch_plan(
                pending,
                resources,
                residentBytes,
                producerReserveBytes,
                resources.workers,
            )
            capacity = min(capacity, operation.writeWorkers)
            if len(pending) >= capacity:
                return
            try:
                pending.append(pull())
            except StopIteration:
                exhausted = True

    def write_one(item: SparseWriteBand) -> None:
        item.destination[item.band.start : item.band.end, :] = item.band.dense()

    try:
        fill()
        while pending:
            operation = _plan_pending()
            admitted, inner = _sparse_batch_plan(
                pending,
                resources,
                residentBytes,
                producerReserveBytes,
                len(pending),
            )
            admitted = min(admitted, operation.writeWorkers)
            inner = min(inner, operation.ioConcurrency)
            actual_write_workers = max(actual_write_workers, admitted)
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
            del batch
            fill()
    finally:
        _close_iterator(source)
        if progress is not None:
            progress.close()
        if operation is not None:
            record_execution_report(
                ExecutionReport(
                    plan=operation,
                    unitKind="countsImportBand",
                    actualReadWorkers=1,
                    actualComputeWorkers=operation.computeWorkers,
                    actualWriteWorkers=actual_write_workers or operation.writeWorkers,
                )
            )


def write_dense_from_row_batches(
    dst: zarr.Array,
    batches: Iterator[np.ndarray],
    *,
    dtype: Any | None = None,
    msg: str | None = None,
    resources: ResourceBudget | None = None,
    io: StorageIoPolicy | None = None,
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

    def aligned() -> Iterator[_DenseWriteBand]:
        target_dtype = np.dtype(dst.dtype if dtype is None else dtype)
        n_columns = int(dst.shape[1])
        buffer = np.empty((shard_rows, n_columns), dtype=target_dtype)
        buffered_rows = 0
        position = 0
        try:
            for batch in source:
                values = np.asarray(batch)
                if values.ndim != 2 or values.shape[1] != dst.shape[1]:
                    raise ValueError("Dense source batch has an invalid shape")
                if values.shape[0] == 0:
                    continue
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

    n_bands = (int(dst.shape[0]) + shard_rows - 1) // shard_rows
    try:
        workers, inner = _writer_count(dst, resources, n_bands, io=io)
    except BaseException:
        _close_iterator(source)
        raise

    def write_band(band: _DenseWriteBand) -> int:
        dst[band.start : band.end, :] = band.values
        return band.end - band.start

    total_rows = int(
        sum(
            stream_shards(
                aligned(),
                write_band,
                workers=workers,
                within_block_threads=1,
                io_concurrency=inner,
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
    dense_bytes, inner_bytes, n_chunks = _band_geometry(dst)
    operation = plan_operation(
        resources,
        WorkShape(
            nUnits=max(1, len(slices)),
            unitBytes=max(1, dense_bytes),
            decodeBytes=max(0, inner_bytes),
            writes=True,
            ordered=True,
            chunksPerShard=max(1, n_chunks),
        ),
        policy=io,
    )
    workers = operation.writeWorkers
    inner = operation.ioConcurrency

    def produce_and_write(bounds: tuple[int, int]) -> Any:
        start, end = bounds
        block = np.asarray(produce(start, end))
        expected = (end - start, int(dst.shape[1]))
        if block.shape != expected:
            raise ValueError(
                f"Dense producer returned shape {block.shape}, expected {expected}"
            )
        dst[start:end, :] = block
        if also_write_to is not None:
            also_write_to[start:end, :] = block
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
            actualReadWorkers=operation.readWorkers,
            actualComputeWorkers=operation.computeWorkers,
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
    needed = (
        int(plan.destinationBufferBytes)
        + int(plan.sourceBufferBytes)
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


class _SharedSourceDecode:
    """Share a source decode across concurrent destination consumers."""

    def __init__(self) -> None:
        self._futures: dict[tuple[int, int, int, int], asyncio.Future[np.ndarray]] = {}
        self._held: dict[tuple[int, int, int, int], int] = {}
        self._users: dict[tuple[int, int, int, int], int] = {}
        self._lock = asyncio.Lock()

    async def get(
        self,
        key: tuple[int, int, int, int],
        read_bytes: int,
        runner: AsyncStorageRunner,
        load: Callable[[], Coroutine[Any, Any, np.ndarray]],
    ) -> tuple[np.ndarray, bool]:
        async with self._lock:
            future = self._futures.get(key)
            if future is None:
                future = asyncio.get_running_loop().create_future()
                self._futures[key] = future
                owner = True
            else:
                owner = False
            self._users[key] = self._users.get(key, 0) + 1
        if owner:
            try:
                await runner.ledger.acquire(read_bytes)
                self._held[key] = read_bytes
                async with runner.read_lane():
                    payload = await load()
                future.set_result(payload)
            except BaseException as exc:
                if not future.done():
                    future.set_exception(exc)
                held = self._held.pop(key, 0)
                if held:
                    await runner.ledger.release(held)
                async with self._lock:
                    self._futures.pop(key, None)
                    self._users.pop(key, None)
                raise
        return await future, owner

    async def release(
        self,
        key: tuple[int, int, int, int],
        runner: AsyncStorageRunner,
    ) -> None:
        async with self._lock:
            left = self._users.get(key, 1) - 1
            if left > 0:
                self._users[key] = left
                return
            self._users.pop(key, None)
            self._futures.pop(key, None)
            held = self._held.pop(key, 0)
        if held:
            await runner.ledger.release(held)


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
) -> zarr.Array:
    """Write paired rotateOnce feature-major ``countsT``."""
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
    if "countsT" in group:
        existing = as_zarr_array(group["countsT"], name="countsT")
        if existing.attrs.get("complete") is True and _counts_t_matches_plan(
            existing, plan
        ):
            persist_count_matrix_plan(group, plan)
            persist_count_matrix_plan(counts, plan)
            persist_count_matrix_plan(existing, plan)
            return existing
    counts_t = as_zarr_array(
        create_count_matrix_array(group, "countsT", plan.countsT),
        name="countsT",
    )
    counts_t.attrs["complete"] = False
    persist_count_matrix_plan(group, plan)
    persist_count_matrix_plan(counts, plan)
    persist_count_matrix_plan(counts_t, plan)
    n_cells = int(counts.shape[0])
    n_feats = int(counts.shape[1])
    if n_cells == 0 or n_feats == 0:
        counts_t.attrs["complete"] = True
        return counts_t
    preflight_counts_t_spec(
        plan.counts,
        profile=resolved_profile,
        resources=resources,
        residentBytes=max(0, int(residentBytes)),
        policy=resolved_policy,
    )
    owners: set[tuple[int, int]] = set()
    dest_feat_shard = int(plan.countsT.shards[0]) if plan.countsT.shards else n_feats
    dest_cell_band = int(plan.countsT.shards[1]) if plan.countsT.shards else n_cells
    source_cell_chunk = int(plan.counts.chunks[0])
    source_feat_chunk = int(plan.counts.chunks[1])
    source_feat_shard = int(plan.counts.shards[1]) if plan.counts.shards else n_feats
    max_group_chunks = max(1, source_feat_shard // source_feat_chunk)
    itemsize = int(np.dtype(counts.dtype).itemsize)
    dest_unit_bytes = max(1, dest_feat_shard * dest_cell_band * itemsize)
    source_chunk_bytes = max(1, source_cell_chunk * source_feat_chunk * itemsize)
    resident_bytes = max(0, int(residentBytes))
    available_bytes = max(0, int(resources.memoryBytes) - resident_bytes)
    cell_chunks_per_dest = max(1, -(-dest_cell_band // max(1, source_cell_chunk)))
    feat_chunks_per_dest = max(1, -(-dest_feat_shard // max(1, source_feat_chunk)))
    touched_source_chunks = cell_chunks_per_dest * feat_chunks_per_dest
    destination_cell_jobs = max(1, -(-n_cells // max(1, dest_cell_band)))
    destination_feature_jobs = max(1, -(-n_feats // max(1, dest_feat_shard)))
    max_destinations_per_set = 1
    n_dest_sets = destination_cell_jobs * -(
        -destination_feature_jobs // max_destinations_per_set
    )
    planned_destination_set_bytes = dest_unit_bytes * max_destinations_per_set
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
            nUnits=n_dest_sets,
            unitBytes=planned_destination_set_bytes,
            residentBytes=resident_bytes,
            innerReadBytes=max(int(plan.sourceBufferBytes), source_chunk_bytes),
            writes=True,
            chunksPerShard=max(max_group_chunks, touched_source_chunks),
        ),
        policy=transpose_io,
    )
    resolved_read_group_chunks = 1
    source_reads_per_destination = operation.innerReads
    requested_reads_in_flight = operation.requestedReadWorkers or operation.innerReads
    requested_commits_in_flight = (
        operation.requestedWriteWorkers or operation.writeWorkers
    )
    requested_compute_workers = (
        operation.requestedComputeWorkers or operation.computeWorkers
    )
    effective_reads_in_flight = source_reads_per_destination
    effective_commits_in_flight = operation.writeWorkers
    effective_compute_workers = operation.computeWorkers
    dest_in_flight = operation.readWorkers
    observed: dict[str, Any] = {
        "mode": "destination-shard",
        **operation.as_metrics(),
        "readGroupChunks": resolved_read_group_chunks,
        "requestedSourceReadsInFlight": requested_reads_in_flight,
        "requestedDestCommitsInFlight": requested_commits_in_flight,
        "requestedDestShardsInFlight": operation.requestedReadWorkers or dest_in_flight,
        "effectiveDestShardsInFlight": dest_in_flight,
        "effectiveDestinationShardsInFlight": dest_in_flight,
        "requestedComputeWorkers": requested_compute_workers,
        "effectiveSourceReadsInFlight": dest_in_flight * source_reads_per_destination,
        "sourceReadsPerDestination": source_reads_per_destination,
        "sourceChunksPerDestinationInFlight": (
            source_reads_per_destination * resolved_read_group_chunks
        ),
        "sourceReadGroups": 0,
        "sourceLogicalBytes": 0,
        "sourceDecodeBytes": 0,
        "sourceRepeatedDecodeCount": 0,
        "sourceRepeatedDecodeBytes": 0,
        "destinationCommits": 0,
        "destinationLogicalBytes": 0,
        "destinationOwners": 0,
        "fusedDestinationStrips": 0,
        "destinationSets": n_dest_sets,
        "plannedDestinationSetBytes": planned_destination_set_bytes,
        "kind": "observed",
    }
    seen_source_chunks: set[tuple[int, int]] = set()

    def _shard_key(feat_start: int, cell_start: int) -> tuple[int, int]:
        return (feat_start // dest_feat_shard, cell_start // dest_cell_band)

    def _destination_ranges(
        feat_start: int,
        feat_end: int,
    ) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        start = feat_start
        while start < feat_end:
            boundary = ((start // dest_feat_shard) + 1) * dest_feat_shard
            end = min(feat_end, boundary)
            ranges.append((start, end))
            start = end
        return ranges

    def _source_feature_ranges(
        feat_start: int,
        feat_end: int,
    ) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        shard_start = (feat_start // source_feat_shard) * source_feat_shard
        group_width = resolved_read_group_chunks * source_feat_chunk
        start = shard_start + ((feat_start - shard_start) // group_width) * group_width
        while start < feat_end:
            shard_end = ((start // source_feat_shard) + 1) * source_feat_shard
            end = min(n_feats, shard_end, start + group_width)
            if end <= start:
                raise RuntimeError("source read-group planner did not advance")
            ranges.append((start, end))
            start = end
        return ranges

    def _source_cell_ranges(
        cell_start: int,
        cell_end: int,
    ) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        start = (cell_start // source_cell_chunk) * source_cell_chunk
        while start < cell_end:
            end = min(n_cells, start + source_cell_chunk)
            if end <= start:
                raise RuntimeError("source cell-range planner did not advance")
            ranges.append((start, end))
            start = end
        return ranges

    def _source_read_admission_bytes(
        source_feat_start: int,
        source_feat_end: int,
    ) -> int:
        first_chunk = source_feat_start // source_feat_chunk
        last_chunk = (source_feat_end - 1) // source_feat_chunk
        touched_chunks = last_chunk - first_chunk + 1
        return source_cell_chunk * touched_chunks * source_feat_chunk * itemsize

    async def _operation(runner: AsyncStorageRunner) -> None:
        source = counts.async_array
        destination = counts_t.async_array
        shared_decodes: _SharedSourceDecode | None = None

        async def _process_destination_set(
            destination_ranges: list[tuple[int, int]],
            *,
            cell_start: int,
            cell_end: int,
        ) -> None:
            if not destination_ranges:
                return
            output_bytes = sum(
                (feat_end - feat_start) * (cell_end - cell_start) * itemsize
                for feat_start, feat_end in destination_ranges
            )
            min_feat = destination_ranges[0][0]
            max_feat = destination_ranges[-1][1]
            source_ranges = _source_feature_ranges(min_feat, max_feat)
            max_read_bytes = max(
                _source_read_admission_bytes(start, end) for start, end in source_ranges
            )
            available_for_reads = available_bytes - output_bytes
            if available_for_reads < max_read_bytes:
                raise MemoryError(
                    "countsT cannot admit one source read while "
                    "holding its destination buffer"
                )
            effective_reads = min(
                effective_reads_in_flight,
                max(1, available_for_reads // max_read_bytes),
            )
            observed["effectiveSourceReadsInFlight"] = max(
                int(observed["effectiveSourceReadsInFlight"]),
                int(dest_in_flight * effective_reads),
            )
            observed["sourceReadsPerDestination"] = max(
                int(observed["sourceReadsPerDestination"]),
                int(effective_reads),
            )
            observed["sourceChunksPerDestinationInFlight"] = max(
                int(observed["sourceChunksPerDestinationInFlight"]),
                int(effective_reads * resolved_read_group_chunks),
            )

            async with runner.reserve_bytes(output_bytes):
                buffers = {
                    (feat_start, feat_end): np.empty(
                        (feat_end - feat_start, cell_end - cell_start),
                        dtype=counts.dtype,
                    )
                    for feat_start, feat_end in destination_ranges
                }

                read_work = [
                    (
                        source_cell_start,
                        source_cell_end,
                        source_feat_start,
                        source_feat_end,
                    )
                    for source_cell_start, source_cell_end in _source_cell_ranges(
                        cell_start, cell_end
                    )
                    for source_feat_start, source_feat_end in source_ranges
                ]

                async def _read_and_scatter(
                    source_cell_start: int,
                    source_cell_end: int,
                    source_feat_start: int,
                    source_feat_end: int,
                    read_bytes: int,
                ) -> None:
                    key = (
                        source_cell_start,
                        source_cell_end,
                        source_feat_start,
                        source_feat_end,
                    )

                    async def _load() -> np.ndarray:
                        return np.asarray(
                            await source.getitem(
                                (
                                    slice(source_cell_start, source_cell_end),
                                    slice(source_feat_start, source_feat_end),
                                )
                            )
                        )

                    if shared_decodes is None:
                        raise RuntimeError("source decode cache is not ready")
                    payload, loaded = await shared_decodes.get(
                        key, read_bytes, runner, _load
                    )
                    try:

                        def _scatter() -> None:
                            for (
                                destination_feat_start,
                                destination_feat_end,
                            ), buffer in buffers.items():
                                overlap_start = max(
                                    source_feat_start,
                                    destination_feat_start,
                                )
                                overlap_end = min(
                                    source_feat_end,
                                    destination_feat_end,
                                )
                                if overlap_start >= overlap_end:
                                    continue
                                source_slice = slice(
                                    overlap_start - source_feat_start,
                                    overlap_end - source_feat_start,
                                )
                                cell_overlap_start = max(source_cell_start, cell_start)
                                cell_overlap_end = min(source_cell_end, cell_end)
                                if cell_overlap_start >= cell_overlap_end:
                                    continue
                                source_cell_slice = slice(
                                    cell_overlap_start - source_cell_start,
                                    cell_overlap_end - source_cell_start,
                                )
                                destination_slice = slice(
                                    overlap_start - destination_feat_start,
                                    overlap_end - destination_feat_start,
                                )
                                cell_slice = slice(
                                    cell_overlap_start - cell_start,
                                    cell_overlap_end - cell_start,
                                )
                                buffer[destination_slice, cell_slice] = payload[
                                    source_cell_slice, source_slice
                                ].T

                        await runner.compute(_scatter)
                        observed["sourceReadGroups"] = (
                            int(observed["sourceReadGroups"]) + 1
                        )
                        observed["sourceLogicalBytes"] = int(
                            observed["sourceLogicalBytes"]
                        ) + int(payload.nbytes)
                        if loaded:
                            cell_chunk_index = source_cell_start // source_cell_chunk
                            first_feature_chunk = source_feat_start // source_feat_chunk
                            last_feature_chunk = (
                                source_feat_end - 1
                            ) // source_feat_chunk
                            for feature_chunk_index in range(
                                first_feature_chunk,
                                last_feature_chunk + 1,
                            ):
                                chunk_key = (cell_chunk_index, feature_chunk_index)
                                decode_bytes = (
                                    source_cell_chunk * source_feat_chunk * itemsize
                                )
                                observed["sourceDecodeBytes"] = (
                                    int(observed["sourceDecodeBytes"]) + decode_bytes
                                )
                                if chunk_key in seen_source_chunks:
                                    observed["sourceRepeatedDecodeCount"] = (
                                        int(observed["sourceRepeatedDecodeCount"]) + 1
                                    )
                                    observed["sourceRepeatedDecodeBytes"] = (
                                        int(observed["sourceRepeatedDecodeBytes"])
                                        + decode_bytes
                                    )
                                else:
                                    seen_source_chunks.add(chunk_key)
                    finally:
                        await shared_decodes.release(key, runner)

                pending: set[asyncio.Task[None]] = set()
                async with asyncio.TaskGroup() as read_tasks:
                    for (
                        source_cell_start,
                        source_cell_end,
                        source_feat_start,
                        source_feat_end,
                    ) in read_work:
                        read_bytes = _source_read_admission_bytes(
                            source_feat_start,
                            source_feat_end,
                        )
                        task = read_tasks.create_task(
                            _read_and_scatter(
                                source_cell_start,
                                source_cell_end,
                                source_feat_start,
                                source_feat_end,
                                read_bytes,
                            )
                        )
                        pending.add(task)
                        if len(pending) >= effective_reads:
                            done, pending = await asyncio.wait(
                                pending,
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                            for completed in done:
                                completed.result()

                async def _commit(
                    feat_start: int,
                    feat_end: int,
                    buffer: np.ndarray,
                ) -> None:
                    async with runner.commit_lane():
                        await destination.setitem(
                            (
                                slice(feat_start, feat_end),
                                slice(cell_start, cell_end),
                            ),
                            buffer,
                        )
                    observed["destinationCommits"] = (
                        int(observed["destinationCommits"]) + 1
                    )
                    observed["destinationLogicalBytes"] = int(
                        observed["destinationLogicalBytes"]
                    ) + int(buffer.nbytes)

                pending_commits: set[asyncio.Task[None]] = set()
                async with asyncio.TaskGroup() as commit_tasks:
                    for (feat_start, feat_end), buffer in buffers.items():
                        task = commit_tasks.create_task(
                            _commit(feat_start, feat_end, buffer)
                        )
                        pending_commits.add(task)
                        if len(pending_commits) >= effective_commits_in_flight:
                            done, pending_commits = await asyncio.wait(
                                pending_commits,
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                            for completed in done:
                                completed.result()

        dest_jobs: list[tuple[int, int, int, int]] = []
        for cell_start in range(0, n_cells, dest_cell_band):
            cell_end = min(cell_start + dest_cell_band, n_cells)
            for feat_start in range(0, n_feats, dest_feat_shard):
                feat_end = min(feat_start + dest_feat_shard, n_feats)
                key = _shard_key(feat_start, cell_start)
                if key in owners:
                    raise RuntimeError(f"destination shard {key} already has an owner")
                owners.add(key)
                dest_jobs.append((feat_start, feat_end, cell_start, cell_end))

        dest_sets: list[list[tuple[int, int, int, int]]] = []
        index = 0
        while index < len(dest_jobs):
            current = dest_jobs[index]
            destination_set = [current]
            if index + 1 < len(dest_jobs):
                following = dest_jobs[index + 1]
                same_cell_band = current[2:] == following[2:]
                adjacent = current[1] == following[0]
                shared_source_chunk = (
                    current[1] - 1
                ) // source_feat_chunk == following[0] // source_feat_chunk
                if same_cell_band and adjacent and shared_source_chunk:
                    destination_set.append(following)
                    index += 1
            dest_sets.append(destination_set)
            index += 1
        shared_decodes = _SharedSourceDecode()
        observed["fusedDestinationStrips"] = sum(
            1 for destination_set in dest_sets if len(destination_set) > 1
        )
        observed["destinationSets"] = len(dest_sets)

        async def _one_destination_set(
            jobs: list[tuple[int, int, int, int]],
        ) -> None:
            cell_start = jobs[0][2]
            cell_end = jobs[0][3]
            await _process_destination_set(
                [(feat_start, feat_end) for feat_start, feat_end, _, _ in jobs],
                cell_start=cell_start,
                cell_end=cell_end,
            )

        pending_dest: set[asyncio.Task[None]] = set()
        async with asyncio.TaskGroup() as dest_tasks:
            for jobs in dest_sets:
                pending_dest.add(dest_tasks.create_task(_one_destination_set(jobs)))
                if len(pending_dest) >= dest_in_flight:
                    done, pending_dest = await asyncio.wait(
                        pending_dest,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for completed in done:
                        completed.result()

    dest_chunk_feats = int(plan.countsT.chunks[0])
    runner = AsyncStorageRunner(
        ResourceBudget(max(1, available_bytes), resources.workers),
        operation=operation,
        chunksPerShard=max(1, dest_feat_shard // max(1, dest_chunk_feats)),
        readGroupsInFlight=effective_reads_in_flight,
        destinationCommitsInFlight=effective_commits_in_flight,
        computeWorkerLimit=effective_compute_workers,
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
    observed["destinationOwners"] = len(owners)
    observed["effectiveComputeWorkers"] = runner.plan.computeWorkerLimit
    observed["effectiveComputeWorkerLimit"] = runner.plan.computeWorkerLimit
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


def finalize_rna_counts_t(
    counts: zarr.Array,
    group: zarr.Group,
    *,
    profile: StorageProfile | None = None,
    resources: ResourceBudget | None = None,
    mem_budget: int | str | None = None,
    nthreads: int | None = None,
    residentBytes: int = 0,
    policy: CountMatrixPolicy | None = None,
    io: StorageIoPolicy | None = None,
) -> zarr.Array:
    """Write mandatory paired ``countsT`` after RNA ``counts`` is complete."""
    resolved = resources
    if resolved is None:
        resolved = resolve_budget(mem_budget, nthreads)
    return write_counts_t(
        counts,
        group,
        profile=profile,
        resources=resolved,
        residentBytes=residentBytes,
        policy=policy,
        io=io,
    )
