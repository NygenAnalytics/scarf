import asyncio
import contextvars
import threading
from collections import deque
from collections.abc import Callable, Coroutine, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import zarr

from .arrays import create_numeric_array
from .budget import (
    ResourceBudget,
    admitted_worker_split,
    resolve_budget,
)
from .layout import (
    ZarrArraySpec,
    _group_zarr_format,
    array_shard_rows,
    get_compressors,
    iter_shard_row_slices,
)
from .parallel import _close_iterator, stream_shards
from .profiles import StorageProfile, resolve_storage_profile
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
    destination: zarr.Array,
    nRows: int | None = None,
) -> tuple[int, int, int]:
    rows = array_shard_rows(destination) if nRows is None else max(1, int(nRows))
    columns = max(1, int(destination.shape[1]))
    itemsize = max(1, int(np.dtype(destination.dtype).itemsize))
    chunks = tuple(int(value) for value in destination.chunks)
    chunk_rows = min(rows, chunks[0])
    chunk_columns = min(columns, chunks[1])
    n_chunks = (
        (rows + chunk_rows - 1)
        // chunk_rows
        * ((columns + chunk_columns - 1) // chunk_columns)
    )
    dense_bytes = rows * columns * itemsize
    inner_bytes = chunk_rows * chunk_columns * itemsize
    return dense_bytes, inner_bytes, n_chunks


def _encoded_chunk_bound(rawBytes: int) -> int:
    """Conservative encoded-size bound for the supported compressors."""
    raw_bytes = max(0, int(rawBytes))
    return raw_bytes + raw_bytes // 128 + 1024


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
) -> None:
    """Densify and write complete sparse row bands within CPU and memory limits."""
    pending: deque[SparseWriteBand] = deque()
    source = iter(writes)
    exhausted = False

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
                pass
            del batch
            fill()
    finally:
        _close_iterator(source)


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
        )
    finally:
        _close_iterator(source)
    return buffer.rows


def write_counts_t(
    counts: zarr.Array,
    group: zarr.Group,
    *,
    profile: StorageProfile | None = None,
    resources: ResourceBudget | None = None,
    feature_major_layout: bool = False,
) -> zarr.Array | None:
    """Write the optional feature-major count matrix using public Zarr APIs."""
    from ..utils.logging import logger

    if _group_zarr_format(group) < 3:
        return None
    resources = resources or resolve_budget()
    resolved_profile = profile or resolve_storage_profile(group.store)
    n_cells = int(counts.shape[0])
    n_feats = int(counts.shape[1])
    itemsize = max(1, int(np.dtype(counts.dtype).itemsize))
    if feature_major_layout:
        chunk_elements = max(1, int(np.prod(counts.chunks, dtype=np.int64)))
        cell_chunk = max(1, min(max(1, n_cells), chunk_elements))
        feature_chunk = max(
            1,
            min(
                max(1, n_feats),
                chunk_elements // cell_chunk,
            ),
        )
    else:
        source_rows = array_shard_rows(counts)
        feature_chunk = max(1, min(max(1, n_feats), int(counts.chunks[1])))
        cell_chunk = max(1, min(max(1, n_cells), source_rows))
    inner_bytes = feature_chunk * cell_chunk * itemsize

    counts_t = create_numeric_array(
        group,
        "countsT",
        ZarrArraySpec(
            shape=(n_feats, n_cells),
            chunks=(feature_chunk, cell_chunk),
            dtype=counts.dtype,
            compressors=get_compressors(resolved_profile, zarrFormat=3),
            shards=None,
            fillValue=0,
            overwrite=True,
        ),
    )
    counts_t.attrs["complete"] = False
    if n_cells == 0 or n_feats == 0:
        counts_t.attrs["complete"] = True
        return counts_t

    n_tasks = (
        (n_cells + cell_chunk - 1)
        // cell_chunk
        * ((n_feats + feature_chunk - 1) // feature_chunk)
    )
    workers, inner = admitted_worker_split(
        resources,
        nTasks=n_tasks,
        taskBytes=lambda concurrency: _row_band_task_peak(
            sourceBytes=inner_bytes,
            denseBytes=inner_bytes,
            innerChunkBytes=inner_bytes,
            nChunks=1,
            innerConcurrency=concurrency,
        ),
    )
    logger.info(
        f"Writing countsT shape=({n_feats}, {n_cells}) "
        f"chunks=({feature_chunk}, {cell_chunk}) "
        f"tasks={n_tasks} workers={workers}"
    )

    def inner_chunk_values(
        task: tuple[int, int],
    ) -> tuple[tuple[slice, slice], np.ndarray]:
        feat_start, cell_start = task
        feat_end = min(feat_start + feature_chunk, n_feats)
        cell_end = min(cell_start + cell_chunk, n_cells)
        block = np.asarray(
            counts[cell_start:cell_end, feat_start:feat_end],
        )
        return (
            (slice(feat_start, feat_end), slice(cell_start, cell_end)),
            np.ascontiguousarray(block.T),
        )

    async def copy_all() -> None:
        remaining = (
            (feat_start, cell_start)
            for cell_start in range(0, n_cells, cell_chunk)
            for feat_start in range(0, n_feats, feature_chunk)
        )

        async def copy_worker() -> None:
            for task in remaining:
                selection, values = await asyncio.to_thread(
                    inner_chunk_values,
                    task,
                )
                await counts_t.async_array.setitem(selection, values)
                del selection, values

        async with asyncio.TaskGroup() as group_tasks:
            for _ in range(workers):
                group_tasks.create_task(copy_worker())

    with zarr.config.set({"async.concurrency": inner}):
        try:
            _run_async(copy_all)
        except ExceptionGroup as error:
            if len(error.exceptions) == 1:
                raise error.exceptions[0] from error
            raise
    counts_t.attrs["complete"] = True
    return counts_t
