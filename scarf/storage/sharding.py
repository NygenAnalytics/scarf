import asyncio
import contextvars
import threading
from collections import deque
from collections.abc import Callable, Coroutine, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import zarr

from .arrays import create_numeric_array
from .async_execution import AsyncStorageRunner
from .budget import (
    ResourceBudget,
    admitted_worker_split,
    resolve_budget,
)
from .count_matrix import (
    DEFAULT_LAYOUT_STRATEGY,
    EXPERIMENTAL_POLICY,
    LAYOUT_STRATEGIES,
    CountMatrixLayoutPolicy,
    LayoutStrategy,
    create_count_matrix_array,
    load_count_matrix_plan,
    persist_count_matrix_plan,
    plan_count_matrix_pair,
    validate_count_matrix_source,
)
from .geometry import ArrayGeometry, array_geometry
from .layout import (
    ZarrArraySpec,
    _encoded_chunk_bound,
    _group_zarr_format,
    array_shard_rows,
    get_compressors,
    iter_shard_row_slices,
)
from .parallel import _close_iterator, stream_shards
from .partition import affordable_width
from .profiles import StorageProfile, resolve_storage_profile
from .types import as_zarr_array
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


class SparseShardBuffer:
    """Accumulate sparse source batches into immutable destination row bands."""

    def __init__(self, destination: zarr.Array) -> None:
        self.nRows = max(0, int(destination.shape[0]))
        self.nColumns = max(0, int(destination.shape[1]))
        self.shardRows = array_shard_rows(destination)
        self.dtype = np.dtype(destination.dtype)
        self.rows = 0
        self._nextFlush = self.shardRows
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
            start = end - self.shardRows
            self._nextFlush += self.shardRows
            yield self._take(start, end)

    def finish(self) -> Iterator[SparseRowBand]:
        if self.rows != self.nRows:
            raise ValueError(
                f"Sparse stream contains {self.rows} rows, expected {self.nRows}"
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
) -> tuple[int, int]:
    dense_bytes, inner_bytes, n_chunks = _band_geometry(destination)
    return admitted_worker_split(
        resources,
        nTasks=nTasks,
        taskBytes=lambda inner: _row_band_task_peak(
            sourceBytes=0,
            denseBytes=dense_bytes,
            innerChunkBytes=inner_bytes,
            nChunks=n_chunks,
            innerConcurrency=inner,
        ),
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
) -> None:
    """Densify and write complete sparse row bands within CPU and memory limits."""
    pending: deque[SparseWriteBand] = deque()
    source = iter(writes)
    exhausted = False
    progress = None
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
            capacity, _ = _sparse_batch_plan(
                pending,
                resources,
                residentBytes,
                producerReserveBytes,
                resources.workers,
            )
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
            admitted, inner = _sparse_batch_plan(
                pending,
                resources,
                residentBytes,
                producerReserveBytes,
                len(pending),
            )
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


def write_dense_from_row_batches(
    dst: zarr.Array,
    batches: Iterator[np.ndarray],
    *,
    dtype: Any | None = None,
    msg: str | None = None,
    resources: ResourceBudget | None = None,
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
        workers, inner = _writer_count(dst, resources, n_bands)
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
    workers, inner = _writer_count(dst, resources, len(slices))

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
    for result in stream_shards(
        slices,
        produce_and_write,
        workers=workers,
        within_block_threads=1,
        io_concurrency=inner,
        msg=msg or "Writing Zarr array",
        total=len(slices),
    ):
        if summarize is not None:
            assert merger is not None
            summary = result if summary is None else merger(summary, result)
    return summary


def accumulate_sparse_to_shards(
    dst: zarr.Array,
    data_stream: Iterator[Any],
    *,
    resources: ResourceBudget | None = None,
    residentBytes: int = 0,
    producerReserveBytes: int,
    msg: str | None = None,
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
        )
    finally:
        _close_iterator(source)
    return buffer.rows


# Strip-sharded countsT: gene strips (~1 GiB) with ~128 MiB cell chunks.
# Selected on the 1M R2 layout matrix (1/2/4 GiB). Validation is structural
# and does not require these exact targets.
COUNTS_T_MAX_SHARD_BYTES = 1 * 1024**3
COUNTS_T_TARGET_CHUNK_BYTES = 128 * 1024**2
# Upload concurrency, and the max number of finished strips held in RAM.
# The writer does not keep more in-memory strips than it can upload at once.
COUNTS_T_MAX_UPLOAD_WORKERS = 4


@dataclass(frozen=True, slots=True)
class CountsTWritePlan:
    """Admitted strip ``countsT`` write geometry and concurrency."""

    layout: ZarrArraySpec
    nStrips: int
    stripsPerBatch: int
    sourceCellRows: int
    readWorkers: int
    uploadWorkers: int
    stripBytes: int
    peakBytes: int
    residentBytes: int


def counts_t_write_plan_details(
    plan: CountsTWritePlan,
    *,
    nFeats: int,
    nCells: int,
    itemsize: int,
) -> dict[str, int | float]:
    """Observability fields for ``writeCountsT`` result details.

    The bounded in-memory path reads each source feature column once across
    batches, so estimated read amplification is ~1.0 relative to the logical
    matrix. Scratch transpose is only considered when gates report worse
    amplification or admission failure.
    """
    logical_bytes = max(0, int(nFeats) * int(nCells) * max(1, int(itemsize)))
    n_batches = (
        1
        if plan.nStrips <= 0
        else (plan.nStrips + max(1, plan.stripsPerBatch) - 1)
        // max(1, plan.stripsPerBatch)
    )
    # Feature strips are partitioned across batches; each batch streams all cells.
    source_bytes = logical_bytes
    amplification = (
        float(source_bytes) / float(logical_bytes) if logical_bytes > 0 else 0.0
    )
    return {
        "logicalMatrixBytes": logical_bytes,
        "sourceBytesRequested": source_bytes,
        "estimatedReadAmplification": amplification,
        "nStrips": int(plan.nStrips),
        "stripsPerBatch": int(plan.stripsPerBatch),
        "nBatches": int(n_batches),
        "readWorkers": int(plan.readWorkers),
        "uploadWorkers": int(plan.uploadWorkers),
        "admittedPeakBytes": int(plan.peakBytes),
        "stripBytes": int(plan.stripBytes),
        "sourceCellRows": int(plan.sourceCellRows),
        "residentBytes": int(plan.residentBytes),
    }


def choose_strip_layout(
    n_feats: int,
    n_cells: int,
    itemsize: int,
    *,
    maxShardBytes: int = COUNTS_T_MAX_SHARD_BYTES,
    targetChunkBytes: int = COUNTS_T_TARGET_CHUNK_BYTES,
) -> dict[str, int]:
    """Gene-strip shards with near-target cell chunks and padded shard tails.

    The shard cell extent is a multiple of ``cellChunk`` and may exceed
    ``nCells`` when the logical cell count is not chunk-aligned. Reads still
    observe the logical array shape.
    """
    n_feats = max(0, int(n_feats))
    n_cells = max(0, int(n_cells))
    itemsize = max(1, int(itemsize))
    if n_feats == 0 or n_cells == 0:
        return {
            "geneStrip": max(1, n_feats) if n_feats else 1,
            "cellChunk": max(1, n_cells) if n_cells else 1,
            "shardCells": max(1, n_cells) if n_cells else 1,
            "shardBytes": 0,
            "chunkBytes": 0,
            "nShardsFeat": 0,
        }
    row_bytes = n_cells * itemsize
    gene_strip = max(1, min(n_feats, int(maxShardBytes) // max(1, row_bytes)))
    target_cells = max(1, int(targetChunkBytes) // max(1, gene_strip * itemsize))
    cell_chunk = max(1, min(n_cells, target_cells))
    shard_cells = ((n_cells + cell_chunk - 1) // cell_chunk) * cell_chunk
    return {
        "geneStrip": int(gene_strip),
        "cellChunk": int(cell_chunk),
        "shardCells": int(shard_cells),
        "shardBytes": int(gene_strip * n_cells * itemsize),
        "chunkBytes": int(gene_strip * cell_chunk * itemsize),
        "nShardsFeat": int((n_feats + gene_strip - 1) // gene_strip),
    }


def _counts_t_layout(
    n_cells: int,
    n_feats: int,
    dtype: Any,
    profile: StorageProfile,
    *,
    maxShardBytes: int = COUNTS_T_MAX_SHARD_BYTES,
    targetChunkBytes: int = COUNTS_T_TARGET_CHUNK_BYTES,
) -> ZarrArraySpec:
    """Return the strip-sharded feature-major layout for ``countsT``."""
    itemsize = max(1, int(np.dtype(dtype).itemsize))
    chosen = choose_strip_layout(
        n_feats,
        n_cells,
        itemsize,
        maxShardBytes=maxShardBytes,
        targetChunkBytes=targetChunkBytes,
    )
    gene_strip = int(chosen["geneStrip"])
    cell_chunk = int(chosen["cellChunk"])
    shard_cells = int(chosen["shardCells"])
    return ZarrArraySpec(
        shape=(max(0, int(n_feats)), max(0, int(n_cells))),
        chunks=(gene_strip, cell_chunk),
        dtype=dtype,
        compressors=get_compressors(profile, zarrFormat=3),
        shards=(gene_strip, shard_cells),
        fillValue=0,
        overwrite=True,
    )


def counts_t_spec(
    counts: ZarrArraySpec,
    *,
    profile: StorageProfile,
    maxShardBytes: int = COUNTS_T_MAX_SHARD_BYTES,
    targetChunkBytes: int = COUNTS_T_TARGET_CHUNK_BYTES,
) -> ZarrArraySpec:
    """Return the strip-sharded feature-major layout derived from counts."""
    if len(counts.shape) != 2 or len(counts.chunks) != 2:
        raise ValueError("counts must be a two-dimensional array specification")
    n_cells = int(counts.shape[0])
    n_feats = int(counts.shape[1])
    return _counts_t_layout(
        n_cells,
        n_feats,
        counts.dtype,
        profile,
        maxShardBytes=maxShardBytes,
        targetChunkBytes=targetChunkBytes,
    )


def is_strip_counts_t_layout(
    *,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    shards: tuple[int, ...] | None,
    dtype: Any,
) -> bool:
    """Return True for a structural strip-sharded ``countsT`` layout.

    Accepts padded shard tails and any gene-strip / cell-chunk sizes that obey
    the strip contract. Write-time byte targets are not part of validation.
    """
    del dtype
    if len(shape) != 2 or len(chunks) != 2:
        return False
    if shards is None or len(shards) != 2:
        return False
    n_feats, n_cells = (int(value) for value in shape)
    gene_strip, cell_chunk = (int(value) for value in chunks)
    shard_feats, shard_cells = (int(value) for value in shards)
    if n_feats < 0 or n_cells < 0:
        return False
    if gene_strip < 1 or cell_chunk < 1:
        return False
    if shard_feats != gene_strip:
        return False
    if shard_cells < cell_chunk or shard_cells % cell_chunk != 0:
        return False
    if n_cells > 0 and shard_cells < n_cells:
        return False
    if n_feats > 0 and gene_strip > n_feats:
        return False
    return True


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
    return is_strip_counts_t_layout(
        shape=shape, chunks=chunks, shards=shards, dtype=dtype
    ) or is_paired_counts_t_layout(
        shape=shape, chunks=chunks, shards=shards, dtype=dtype
    )


def _counts_t_write_overhead(*, stripBytes: int, available: int) -> int:
    """Allocator / decode headroom scaled to the admitted strip size."""
    strip_bytes = max(0, int(stripBytes))
    available = max(0, int(available))
    if strip_bytes == 0:
        return min(64 * 1024, available) if available else 64 * 1024
    if strip_bytes <= 128 * 1024**2:
        return max(64 * 1024, strip_bytes // 4)
    return max(strip_bytes // 8, 1 * 1024**2)


def _counts_t_peak_bytes(
    *,
    stripsPerBatch: int,
    geneStrip: int,
    nFeats: int,
    nCells: int,
    itemsize: int,
    sourceCellRows: int,
    readWorkers: int,
    uploadWorkers: int,
    available: int,
) -> int:
    batch = max(1, int(stripsPerBatch))
    gene_strip = max(1, int(geneStrip))
    n_cells = max(1, int(nCells))
    itemsize = max(1, int(itemsize))
    source_rows = max(1, int(sourceCellRows))
    read_workers = max(1, int(readWorkers))
    upload_workers = max(1, int(uploadWorkers))
    strip_bytes = gene_strip * n_cells * itemsize
    feat_width = min(max(1, int(nFeats)), batch * gene_strip)
    read_each = source_rows * feat_width * itemsize * 2
    encode_each = (
        _encoded_chunk_bound(strip_bytes)
        + _encoded_chunk_bound(strip_bytes)
        + _shard_index_bound(1)
    )
    overhead = _counts_t_write_overhead(stripBytes=strip_bytes, available=available)
    return (
        batch * strip_bytes
        + read_workers * read_each
        + min(upload_workers, batch) * encode_each
        + overhead
    )


def plan_counts_t_write(
    *,
    nFeats: int,
    nCells: int,
    dtype: Any,
    sourceCellRows: int,
    resources: ResourceBudget,
    residentBytes: int = 0,
    profile: StorageProfile,
    maxShardBytes: int = COUNTS_T_MAX_SHARD_BYTES,
    targetChunkBytes: int = COUNTS_T_TARGET_CHUNK_BYTES,
) -> CountsTWritePlan:
    """Admit strip ``countsT`` write concurrency from one shared memory model.

    Hold at most ``COUNTS_T_MAX_UPLOAD_WORKERS`` finished strips in RAM, and
    prefer extra upload and read workers over a larger in-memory batch.
    """
    layout = _counts_t_layout(
        nCells,
        nFeats,
        dtype,
        profile,
        maxShardBytes=maxShardBytes,
        targetChunkBytes=targetChunkBytes,
    )
    gene_strip = int(layout.chunks[0])
    n_feats = max(0, int(nFeats))
    n_cells = max(0, int(nCells))
    itemsize = max(1, int(np.dtype(dtype).itemsize))
    source_rows = max(1, int(sourceCellRows))
    resident = max(0, int(residentBytes))
    available = max(0, int(resources.memoryBytes) - resident)
    n_strips = max(1, row_band_task_count(n_feats, gene_strip)) if n_feats else 1
    strip_bytes = gene_strip * max(1, n_cells) * itemsize if n_cells and n_feats else 0
    if n_cells == 0 or n_feats == 0:
        return CountsTWritePlan(
            layout=layout,
            nStrips=0,
            stripsPerBatch=1,
            sourceCellRows=source_rows,
            readWorkers=1,
            uploadWorkers=1,
            stripBytes=0,
            peakBytes=0,
            residentBytes=resident,
        )

    max_read = max(1, int(resources.workers))
    max_upload = max(1, min(max_read, COUNTS_T_MAX_UPLOAD_WORKERS))
    max_batch = min(n_strips, max_upload)

    def peak(batch: int, read_workers: int, upload_workers: int) -> int:
        return _counts_t_peak_bytes(
            stripsPerBatch=batch,
            geneStrip=gene_strip,
            nFeats=n_feats,
            nCells=n_cells,
            itemsize=itemsize,
            sourceCellRows=source_rows,
            readWorkers=read_workers,
            uploadWorkers=upload_workers,
            available=available,
        )

    one_peak = peak(1, 1, 1)
    if available and one_peak > available:
        raise MemoryError(
            "countsT strip write needs at least "
            f"{one_peak} bytes with the current resource budget "
            f"({available} bytes available after residentBytes={resident})"
        )

    best_batch = 1
    best_read = 1
    best_upload = 1
    best_peak = one_peak
    for read_workers in range(1, max_read + 1):
        for upload_workers in range(1, max_upload + 1):
            # Do not hold more finished strips than this candidate can upload.
            limit = min(max_batch, upload_workers)
            batch = 1
            while batch < limit:
                nxt = batch + 1
                if peak(nxt, read_workers, min(upload_workers, nxt)) > available:
                    break
                batch = nxt
            candidate_upload = min(upload_workers, batch)
            candidate_peak = peak(batch, read_workers, candidate_upload)
            if candidate_peak > available:
                continue
            better_upload = candidate_upload > best_upload
            better_read = candidate_upload == best_upload and read_workers > best_read
            better_batch = (
                candidate_upload == best_upload
                and read_workers == best_read
                and batch > best_batch
            )
            better_peak = (
                candidate_upload == best_upload
                and read_workers == best_read
                and batch == best_batch
                and candidate_peak < best_peak
            )
            if better_upload or better_read or better_batch or better_peak:
                best_batch = batch
                best_read = read_workers
                best_upload = candidate_upload
                best_peak = candidate_peak

    return CountsTWritePlan(
        layout=layout,
        nStrips=n_strips,
        stripsPerBatch=best_batch,
        sourceCellRows=source_rows,
        readWorkers=best_read,
        uploadWorkers=best_upload,
        stripBytes=strip_bytes,
        peakBytes=best_peak,
        residentBytes=resident,
    )


def plan_counts_t_write_for_array(
    counts: zarr.Array | ZarrArraySpec,
    *,
    profile: StorageProfile,
    resources: ResourceBudget,
    residentBytes: int = 0,
    sourceCellRows: int | None = None,
    maxShardBytes: int = COUNTS_T_MAX_SHARD_BYTES,
    targetChunkBytes: int = COUNTS_T_TARGET_CHUNK_BYTES,
) -> CountsTWritePlan:
    """Admit a strip write plan for a counts array or array specification."""
    if isinstance(counts, ZarrArraySpec):
        n_cells = int(counts.shape[0])
        n_feats = int(counts.shape[1])
        dtype = counts.dtype
        source_rows = (
            max(1, int(sourceCellRows))
            if sourceCellRows is not None
            else max(1, int(counts.chunks[0]))
        )
    else:
        n_cells = int(counts.shape[0])
        n_feats = int(counts.shape[1])
        dtype = counts.dtype
        source_rows = (
            max(1, int(sourceCellRows))
            if sourceCellRows is not None
            else max(1, int(array_shard_rows(counts)))
        )
    return plan_counts_t_write(
        nFeats=n_feats,
        nCells=n_cells,
        dtype=dtype,
        sourceCellRows=source_rows,
        resources=resources,
        residentBytes=residentBytes,
        profile=profile,
        maxShardBytes=maxShardBytes,
        targetChunkBytes=targetChunkBytes,
    )


def preflight_counts_t_spec(
    counts: ZarrArraySpec,
    *,
    profile: StorageProfile,
    resources: ResourceBudget,
    residentBytes: int = 0,
) -> ZarrArraySpec:
    """Admit strip ``countsT`` construction with the shared write plan."""
    return plan_counts_t_write_for_array(
        counts,
        profile=profile,
        resources=resources,
        residentBytes=residentBytes,
    ).layout


def write_counts_t(
    counts: zarr.Array,
    group: zarr.Group,
    *,
    profile: StorageProfile | None = None,
    resources: ResourceBudget | None = None,
    residentBytes: int = 0,
    maxShardBytes: int = COUNTS_T_MAX_SHARD_BYTES,
    targetChunkBytes: int = COUNTS_T_TARGET_CHUNK_BYTES,
) -> zarr.Array | None:
    """Write strip-sharded feature-major ``countsT``.

    Streams cell-major ``counts`` in memory-bounded gene-strip batches so each
    source cell band is read once per batch (not once per strip). A batch holds
    at most as many strips as admitted upload workers. Within a batch, cell
    bands are fetched concurrently and completed strips are uploaded
    concurrently. Read and upload concurrency are admitted together.
    """
    from concurrent.futures import Future, ThreadPoolExecutor, as_completed

    from ..utils.logging import logger
    from ..utils.progress import tqdmbar

    if _group_zarr_format(group) < 3:
        return None
    resources = resources or resolve_budget()
    resolved_profile = profile or resolve_storage_profile(group.store)
    plan = plan_counts_t_write_for_array(
        counts,
        profile=resolved_profile,
        resources=resources,
        residentBytes=residentBytes,
        maxShardBytes=maxShardBytes,
        targetChunkBytes=targetChunkBytes,
    )
    layout = plan.layout
    n_cells = int(counts.shape[0])
    n_feats = int(counts.shape[1])
    gene_strip, cell_chunk = (int(value) for value in layout.chunks)
    source_rows = plan.sourceCellRows
    strips_per_batch = plan.stripsPerBatch
    read_workers = plan.readWorkers
    upload_workers = plan.uploadWorkers

    counts_t = create_numeric_array(group, "countsT", layout)
    counts_t.attrs["complete"] = False
    if n_cells == 0 or n_feats == 0:
        counts_t.attrs["complete"] = True
        return counts_t

    strips = [
        (feat_start, min(feat_start + gene_strip, n_feats))
        for feat_start in range(0, n_feats, gene_strip)
    ]
    cell_starts = list(range(0, n_cells, source_rows))

    logger.debug(
        f"Writing strip countsT shape=({n_feats}, {n_cells}) "
        f"shards=({gene_strip}, {layout.shards[1] if layout.shards else n_cells}) "
        f"chunks=({gene_strip}, {cell_chunk}) "
        f"strips={len(strips)} stripsPerBatch={strips_per_batch} "
        f"readWorkers={read_workers} uploadWorkers={upload_workers} "
        f"peakBytes={plan.peakBytes}"
    )
    progress = tqdmbar(desc="Writing transposed counts", total=len(strips))
    try:
        for batch_start in range(0, len(strips), strips_per_batch):
            batch = strips[batch_start : batch_start + strips_per_batch]
            feat_lo = int(batch[0][0])
            feat_hi = int(batch[-1][1])
            buffers = {
                g0: np.zeros((g1 - g0, n_cells), dtype=counts.dtype) for g0, g1 in batch
            }

            def _read_band(
                cell_start: int,
                *,
                _feat_lo: int = feat_lo,
                _feat_hi: int = feat_hi,
            ) -> tuple[int, int, np.ndarray]:
                cell_end = min(cell_start + source_rows, n_cells)
                block = np.asarray(counts[cell_start:cell_end, _feat_lo:_feat_hi])
                return cell_start, cell_end, block

            with ThreadPoolExecutor(max_workers=read_workers) as pool:
                pending: list[Future[tuple[int, int, np.ndarray]]] = []
                start_iter = iter(cell_starts)

                def fill() -> None:
                    while len(pending) < read_workers:
                        try:
                            nxt = next(start_iter)
                        except StopIteration:
                            break
                        pending.append(pool.submit(_read_band, nxt))

                fill()
                while pending:
                    future = pending.pop(0)
                    cell_start, cell_end, block = future.result()
                    fill()
                    for g0, g1 in batch:
                        buffers[g0][:, cell_start:cell_end] = np.ascontiguousarray(
                            block[:, g0 - feat_lo : g1 - feat_lo].T
                        )
                    del block

            def _write_strip(g0: int, g1: int) -> None:
                counts_t[g0:g1, :] = buffers[g0]

            with ThreadPoolExecutor(max_workers=upload_workers) as pool:
                upload_futures = [pool.submit(_write_strip, g0, g1) for g0, g1 in batch]
                for upload_future in as_completed(upload_futures):
                    upload_future.result()
            buffers.clear()
            progress.update(len(batch))
    except BaseException:
        counts_t.attrs["complete"] = False
        raise
    finally:
        progress.close()
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
) -> zarr.Array | None:
    """Write mandatory strip ``countsT`` after RNA ``counts`` is complete.

    Resolves a resource budget when the caller has none. Returns ``None`` when
    the destination group is Zarr format < 3 (cannot hold strip shards).
    """
    resolved = resources
    if resolved is None:
        resolved = resolve_budget(mem_budget, nthreads)
    return write_counts_t(
        counts,
        group,
        profile=profile,
        resources=resolved,
        residentBytes=residentBytes,
    )


def write_counts_t_experimental(
    counts: zarr.Array,
    group: zarr.Group,
    *,
    policy: CountMatrixLayoutPolicy | None = None,
    profile: StorageProfile | None = None,
    resources: ResourceBudget | None = None,
    readGroupChunks: int = 1,
    readGroupsInFlight: int = 1,
    destinationCommitsInFlight: int = 1,
    strategy: LayoutStrategy | None = None,
    metrics: dict[str, Any] | None = None,
) -> zarr.Array:
    """Write countsT with the experimental paired layout and async runner.

    This path is private until a later phase selects it as the product writer.
    """
    if _group_zarr_format(group) < 3:
        raise ValueError("experimental countsT requires Zarr format 3")
    resources = resources or resolve_budget()
    resolved_profile = profile or resolve_storage_profile(group.store)
    resolved_policy = policy or EXPERIMENTAL_POLICY
    resolved_strategy: LayoutStrategy = DEFAULT_LAYOUT_STRATEGY
    if strategy is not None:
        resolved_strategy = strategy
    else:
        try:
            recorded = load_count_matrix_plan(counts).get("strategy")
        except ValueError:
            recorded = None
        if recorded in LAYOUT_STRATEGIES:
            resolved_strategy = recorded
    plan = plan_count_matrix_pair(
        int(counts.shape[0]),
        int(counts.shape[1]),
        counts.dtype,
        policy=resolved_policy,
        strategy=resolved_strategy,
        profile=resolved_profile,
    )
    validate_count_matrix_source(counts, expected=plan)
    requested_read_group_chunks = int(readGroupChunks)
    if requested_read_group_chunks < 1:
        raise ValueError("readGroupChunks must be positive")
    requested_reads_in_flight = int(readGroupsInFlight)
    requested_commits_in_flight = int(destinationCommitsInFlight)
    if requested_reads_in_flight < 1:
        raise ValueError("readGroupsInFlight must be positive")
    if requested_commits_in_flight < 1:
        raise ValueError("destinationCommitsInFlight must be positive")
    if "countsT" in group:
        existing = as_zarr_array(group["countsT"], name="countsT")
        complete = bool(existing.attrs.get("complete", False))
        if complete:
            return existing
        del group["countsT"]
    counts_t = as_zarr_array(
        create_count_matrix_array(group, "countsT", plan.countsT),
        name="countsT",
    )
    counts_t.attrs["complete"] = False
    persist_count_matrix_plan(group, plan)
    persist_count_matrix_plan(counts_t, plan)
    n_cells = int(counts.shape[0])
    n_feats = int(counts.shape[1])
    if n_cells == 0 or n_feats == 0:
        counts_t.attrs["complete"] = True
        return counts_t

    owners: set[tuple[int, int]] = set()
    dest_feat_shard = int(plan.countsT.shards[0]) if plan.countsT.shards else n_feats
    dest_cell_band = int(plan.countsT.shards[1]) if plan.countsT.shards else n_cells
    source_cell_chunk = int(plan.counts.chunks[0])
    source_feat_chunk = int(plan.counts.chunks[1])
    source_feat_shard = int(plan.counts.shards[1]) if plan.counts.shards else n_feats
    max_group_chunks = max(1, source_feat_shard // source_feat_chunk)
    resolved_read_group_chunks = min(requested_read_group_chunks, max_group_chunks)
    itemsize = int(np.dtype(counts.dtype).itemsize)
    observed: dict[str, Any] = {
        "mode": "destination-shard",
        "strategy": plan.strategy,
        "readGroupChunks": resolved_read_group_chunks,
        "requestedReadGroupsInFlight": requested_reads_in_flight,
        "requestedDestinationCommitsInFlight": requested_commits_in_flight,
        "effectiveReadGroupsInFlight": 1,
        "sourceReadGroups": 0,
        "sourceLogicalBytes": 0,
        "sourceDecodeBytes": 0,
        "sourceRepeatedDecodeCount": 0,
        "sourceRepeatedDecodeBytes": 0,
        "destinationCommits": 0,
        "destinationLogicalBytes": 0,
        "destinationOwners": 0,
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
        start = feat_start
        while start < feat_end:
            chunk_start = (start // source_feat_chunk) * source_feat_chunk
            shard_end = ((start // source_feat_shard) + 1) * source_feat_shard
            grouped_end = chunk_start + resolved_read_group_chunks * source_feat_chunk
            end = min(feat_end, shard_end, grouped_end)
            if end <= start:
                raise RuntimeError("source read-group planner did not advance")
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
            available_for_reads = resources.memoryBytes - output_bytes
            if available_for_reads < max_read_bytes:
                raise MemoryError(
                    "experimental countsT cannot admit one source read while "
                    "holding its destination buffer"
                )
            effective_reads = min(
                requested_reads_in_flight,
                max(1, available_for_reads // max_read_bytes),
            )
            observed["effectiveReadGroupsInFlight"] = max(
                int(observed["effectiveReadGroupsInFlight"]),
                int(effective_reads),
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
                        min(source_cell_start + source_cell_chunk, cell_end),
                        source_feat_start,
                        source_feat_end,
                    )
                    for source_cell_start in range(
                        cell_start,
                        cell_end,
                        source_cell_chunk,
                    )
                    for source_feat_start, source_feat_end in source_ranges
                ]

                async def _read_and_scatter(
                    source_cell_start: int,
                    source_cell_end: int,
                    source_feat_start: int,
                    source_feat_end: int,
                    read_bytes: int,
                    reservation: dict[str, bool],
                ) -> None:
                    try:
                        async with runner.read_lane():
                            payload = np.asarray(
                                await source.getitem(
                                    (
                                        slice(source_cell_start, source_cell_end),
                                        slice(source_feat_start, source_feat_end),
                                    )
                                )
                            )

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
                                destination_slice = slice(
                                    overlap_start - destination_feat_start,
                                    overlap_end - destination_feat_start,
                                )
                                cell_slice = slice(
                                    source_cell_start - cell_start,
                                    source_cell_end - cell_start,
                                )
                                buffer[destination_slice, cell_slice] = payload[
                                    :, source_slice
                                ].T

                        await runner.compute(_scatter)
                        observed["sourceReadGroups"] = (
                            int(observed["sourceReadGroups"]) + 1
                        )
                        observed["sourceLogicalBytes"] = int(
                            observed["sourceLogicalBytes"]
                        ) + int(payload.nbytes)
                        cell_chunk_index = source_cell_start // source_cell_chunk
                        first_feature_chunk = source_feat_start // source_feat_chunk
                        last_feature_chunk = (source_feat_end - 1) // source_feat_chunk
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
                        if not reservation["released"]:
                            await runner.ledger.release(read_bytes)
                            reservation["released"] = True

                pending: set[asyncio.Task[None]] = set()
                reservations: list[tuple[int, dict[str, bool]]] = []
                try:
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
                            await runner.ledger.acquire(read_bytes)
                            reservation = {"released": False}
                            reservations.append((read_bytes, reservation))
                            task = read_tasks.create_task(
                                _read_and_scatter(
                                    source_cell_start,
                                    source_cell_end,
                                    source_feat_start,
                                    source_feat_end,
                                    read_bytes,
                                    reservation,
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
                finally:
                    for read_bytes, reservation in reservations:
                        if not reservation["released"]:
                            await runner.ledger.release(read_bytes)
                            reservation["released"] = True

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
                        if len(pending_commits) >= requested_commits_in_flight:
                            done, pending_commits = await asyncio.wait(
                                pending_commits,
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                            for completed in done:
                                completed.result()

        for cell_start in range(0, n_cells, dest_cell_band):
            cell_end = min(cell_start + dest_cell_band, n_cells)
            for feat_start in range(0, n_feats, dest_feat_shard):
                feat_end = min(feat_start + dest_feat_shard, n_feats)
                key = _shard_key(feat_start, cell_start)
                if key in owners:
                    raise RuntimeError(f"destination shard {key} already has an owner")
                owners.add(key)
                await _process_destination_set(
                    [(feat_start, feat_end)],
                    cell_start=cell_start,
                    cell_end=cell_end,
                )

    dest_chunk_feats = int(plan.countsT.chunks[0])
    runner = AsyncStorageRunner(
        resources,
        chunksPerShard=max(1, dest_feat_shard // max(1, dest_chunk_feats)),
        readGroupsInFlight=requested_reads_in_flight,
        destinationCommitsInFlight=requested_commits_in_flight,
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
    observed["terminalStatus"] = "ok"
    observed["peakLedgerBytes"] = runner.ledger.peak_bytes()
    observed["heldLedgerBytes"] = runner.ledger.held_bytes()
    observed["plannedDestinationBufferBytes"] = plan.destinationBufferBytes
    observed["plannedSourceBufferBytes"] = plan.sourceBufferBytes
    observed["sourceDecodeAmplification"] = plan.sourceDecodeAmplification
    if metrics is not None:
        metrics.clear()
        metrics.update(observed)
    counts_t.attrs["complete"] = True
    return counts_t
