from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import DTypeLike, NDArray
from scipy.sparse import (
    coo_matrix,
    csr_matrix,
    hstack,
    issparse,
    vstack,
)

from .errors import MatrixSourceError, ResourceLimitError
from .values import decode_text_values, read_window
from .._sparse import SparseRowStore
from ...utils.arrays import cumulative_nnz, has_duplicates


type MatrixBlock = NDArray[Any] | coo_matrix | csr_matrix

_INDEX_BYTES = np.dtype(np.int64).itemsize


@dataclass(frozen=True)
class MemoryEstimate:
    residentBytes: int = 0
    workingBytes: int = 0
    outputBytes: int = 0

    @property
    def blockBytes(self) -> int:
        """Bytes allocated while reading one block, excluding resident state."""
        return self.workingBytes + self.outputBytes


@dataclass(frozen=True)
class SourceLimits:
    maxFeatures: int = np.iinfo(np.int32).max
    maxCells: int = np.iinfo(np.int32).max
    maxNnz: int = np.iinfo(np.int64).max
    maxBlockBytes: int = 512 * 1024 * 1024
    maxMetadataBytes: int = 256 * 1024 * 1024
    tileCells: int = 1024
    compressedChunkNnz: int = 4 * 1024 * 1024

    def __post_init__(self) -> None:
        for field_name in (
            "maxFeatures",
            "maxCells",
            "maxNnz",
            "maxBlockBytes",
            "maxMetadataBytes",
            "tileCells",
            "compressedChunkNnz",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")


DEFAULT_LIMITS = SourceLimits()


@runtime_checkable
class MatrixSource(Protocol):
    @property
    def shape(self) -> tuple[int, int]: ...

    @property
    def dtype(self) -> np.dtype[Any]: ...

    @property
    def row_names(self) -> tuple[str, ...] | None: ...

    @property
    def column_names(self) -> tuple[str, ...] | None: ...

    @property
    def is_sparse(self) -> bool: ...

    @property
    def zero_preserving(self) -> bool: ...

    @property
    def resident_bytes(self) -> int: ...

    def read_cells(self, start: int, stop: int) -> MatrixBlock: ...

    def estimate_read_memory(self, start: int, stop: int) -> MemoryEstimate: ...


def _matrix_sources(source: MatrixSource) -> Iterator[MatrixSource]:
    from .operations import _ResolvedSubassignment

    pending: list[tuple[Any, bool]] = [(source, False)]
    visited: set[int] = set()
    while pending:
        current, expanded = pending.pop()
        if expanded:
            yield current
            continue
        if id(current) in visited:
            continue
        visited.add(id(current))
        if isinstance(current, LayerPlacement | _ResolvedLayerPlacement):
            pending.append((current.source, False))
            continue
        if isinstance(current, _ResolvedSubassignment):
            if isinstance(current.value, MatrixSource):
                pending.append((current.value, False))
            continue
        pending.append((current, True))
        for value in getattr(current, "__dict__", {}).values():
            if isinstance(
                value,
                MatrixSource
                | LayerPlacement
                | _ResolvedLayerPlacement
                | _ResolvedSubassignment,
            ):
                pending.append((value, False))
            elif (
                isinstance(value, list | tuple)
                and value
                and isinstance(
                    value[0],
                    MatrixSource
                    | LayerPlacement
                    | _ResolvedLayerPlacement
                    | _ResolvedSubassignment,
                )
            ):
                pending.extend((item, False) for item in value)


def release_temporary_storage(source: MatrixSource) -> None:
    for current in _matrix_sources(source):
        if isinstance(current, BaseMatrixSource) and current._rowStore is not None:
            current._rowStore.close()
            current._rowStore = None
        if (
            isinstance(current, TransposeMatrixSource)
            and current._transposeDirectory is not None
        ):
            current._transposeDirectory.cleanup()
            current._transposeDirectory = None


def prepare_matrix_sources(source: MatrixSource, max_bytes: int | None = None) -> None:
    for current in _matrix_sources(source):
        if not isinstance(current, BaseMatrixSource):
            continue
        limits = current._limits
        try:
            if max_bytes is not None:
                available = max_bytes - source.resident_bytes
                if available <= 0:
                    raise MemoryError("Seurat source preparation exceeds mem_budget")
                current._limits = replace(
                    limits, maxBlockBytes=min(limits.maxBlockBytes, available)
                )
                estimate = current.estimate_read_memory(0, min(1, current.n_cells))
                if estimate.blockBytes > available:
                    raise MemoryError("Seurat source preparation exceeds mem_budget")
            current._prepare_for_read()
        finally:
            current._limits = limits


def _metadata_bytes(names: Sequence[str] | None) -> int:
    if names is None:
        return 0
    return sum(len(value.encode("utf-8")) + 8 for value in names)


def _normalize_names(
    names: Sequence[str | bytes] | NDArray[Any] | None,
    length: int,
    axis: str,
    limits: SourceLimits,
) -> tuple[str, ...] | None:
    if names is None:
        return None
    values = decode_text_values(
        names,
        object_path=f"{axis} names",
        max_bytes=limits.maxMetadataBytes,
    )
    if len(values) != length:
        raise MatrixSourceError(
            f"{axis} names have length {len(values)}; expected {length}"
        )
    return values


def _validate_shape(shape: Sequence[int], limits: SourceLimits) -> tuple[int, int]:
    if len(shape) != 2:
        raise MatrixSourceError("matrix shape must have exactly two dimensions")
    n_features, n_cells = (int(shape[0]), int(shape[1]))
    if n_features < 0 or n_cells < 0:
        raise MatrixSourceError("matrix dimensions cannot be negative")
    if n_features > limits.maxFeatures:
        raise ResourceLimitError(
            f"feature count {n_features} exceeds maxFeatures={limits.maxFeatures}"
        )
    if n_cells > limits.maxCells:
        raise ResourceLimitError(
            f"cell count {n_cells} exceeds maxCells={limits.maxCells}"
        )
    return n_features, n_cells


def _validate_window(start: int, stop: int, n_cells: int) -> tuple[int, int]:
    if isinstance(start, bool) or isinstance(stop, bool):
        raise TypeError("cell bounds must be integers")
    start = int(start)
    stop = int(stop)
    if start < 0 or stop < start or stop > n_cells:
        raise IndexError(f"cell window [{start}, {stop}) is outside [0, {n_cells})")
    return start, stop


def _array_shape(values: Any) -> tuple[int, ...]:
    shape = getattr(values, "shape", None)
    if shape is None:
        try:
            return (len(values),)
        except TypeError as error:
            raise TypeError("array-like object must expose shape or len") from error
    return tuple(int(value) for value in shape)


def _array_dtype(values: Any) -> np.dtype[Any]:
    dtype = getattr(values, "dtype", None)
    return np.dtype(np.asarray(values).dtype if dtype is None else dtype)


def _array_resident_bytes(values: Any) -> int:
    if isinstance(values, np.ndarray):
        return int(values.nbytes)
    return 0


def _normalize_indexes(
    indexes: Sequence[int] | NDArray[Any],
    upper_bound: int,
    axis: str,
) -> NDArray[np.int64]:
    values = np.asarray(indexes)
    if values.ndim != 1:
        raise MatrixSourceError(f"{axis} indexes must be one-dimensional")
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError(f"{axis} indexes must contain integers")
    values = values.astype(np.int64, copy=False)
    if values.size and (np.any(values < 0) or np.any(values >= upper_bound)):
        raise IndexError(f"{axis} indexes contain an out-of-range value")
    return values


def _index_runs(indexes: NDArray[np.int64]) -> Iterator[tuple[int, int, int]]:
    """Yield (offset, start, stop) for each run of consecutive indexes."""
    if indexes.size == 0:
        return
    breaks = np.flatnonzero(np.diff(indexes) != 1) + 1
    offsets = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [indexes.size]))
    for offset, end in zip(offsets.tolist(), ends.tolist(), strict=True):
        yield offset, int(indexes[offset]), int(indexes[end - 1]) + 1


def _block_to_csr(
    block: Any,
    *,
    dtype: DTypeLike | None = None,
) -> csr_matrix:
    """Return a two-dimensional block from ``read_cells`` as CSR."""
    if issparse(block):
        result = csr_matrix(block)
        if dtype is not None:
            result = result.astype(dtype, copy=False)
        return result
    values = np.asarray(block)
    if dtype is not None:
        values = values.astype(dtype, copy=False)
    return csr_matrix(values)


def _block_to_dense(
    block: Any,
    *,
    dtype: DTypeLike | None = None,
) -> NDArray[Any]:
    """Return a two-dimensional block from ``read_cells`` as a dense array."""
    values = block.toarray() if issparse(block) else np.asarray(block)
    if dtype is not None:
        values = values.astype(dtype, copy=False)
    return values


def _empty_block(
    rows: int,
    columns: int,
    dtype: DTypeLike,
    sparse: bool,
) -> MatrixBlock:
    if sparse:
        return cast(MatrixBlock, csr_matrix((rows, columns), dtype=dtype))
    return np.empty((rows, columns), dtype=dtype)


def _value_bound(source: MatrixSource, estimate: MemoryEstimate, rows: int) -> int:
    """Upper bound on the values stored in a block of ``rows`` source cells."""
    dense = rows * source.shape[0]
    if not source.is_sparse:
        return dense
    per_value = source.dtype.itemsize + _INDEX_BYTES
    payload = max(0, estimate.outputBytes - (rows + 1) * _INDEX_BYTES)
    return min(dense, -(-payload // per_value))


def _block_output_bytes(
    dtype: np.dtype[Any],
    rows: int,
    columns: int,
    *,
    sparse: bool,
    values: int,
) -> int:
    """Output bytes of a block; a sparse block holds at most ``values`` entries."""
    if not sparse:
        return rows * columns * dtype.itemsize
    stored = min(values, rows * columns)
    return stored * (dtype.itemsize + _INDEX_BYTES) + (rows + 1) * _INDEX_BYTES


def _read_selected_cells(
    source: MatrixSource,
    indexes: NDArray[np.int64],
) -> MatrixBlock:
    if indexes.size == 0:
        return _empty_block(0, source.shape[0], source.dtype, source.is_sparse)
    pieces = [
        source.read_cells(run_start, run_stop)
        for _, run_start, run_stop in _index_runs(indexes)
    ]
    if source.is_sparse:
        return cast(
            MatrixBlock,
            vstack(
                [_block_to_csr(piece, dtype=source.dtype) for piece in pieces],
                format="csr",
                dtype=source.dtype,
            ),
        )
    return np.vstack(
        [_block_to_dense(piece, dtype=source.dtype) for piece in pieces],
        dtype=source.dtype,
    )


def _selected_estimate(
    source: MatrixSource,
    indexes: NDArray[np.int64],
) -> tuple[int, int]:
    """Return (block bytes, stored-value bound) for reading selected cells."""
    block_bytes = 0
    values = 0
    for _, run_start, run_stop in _index_runs(indexes):
        estimate = source.estimate_read_memory(run_start, run_stop)
        block_bytes += estimate.blockBytes
        values += _value_bound(source, estimate, run_stop - run_start)
    return block_bytes, values


def validate_compressed_pointers(
    read: Callable[[int, int], NDArray[Any]],
    count: int,
    limits: SourceLimits,
    *,
    label: str,
) -> int:
    """Validate a nondecreasing pointer vector that starts at zero; return its end.

    ``read(start, stop)`` returns the ``stop - start`` pointers of that window.
    """
    chunk = max(1, min(limits.compressedChunkNnz, count))
    previous: int | None = None
    final = 0
    for start in range(0, count, chunk):
        stop = min(count, start + chunk)
        pointers = read(start, stop)
        if not np.issubdtype(pointers.dtype, np.integer):
            raise TypeError(f"{label} must contain integers")
        if np.any(pointers[1:] < pointers[:-1]):
            raise MatrixSourceError(f"{label} must be nondecreasing")
        if previous is not None and int(pointers[0]) < previous:
            raise MatrixSourceError(f"{label} must be nondecreasing")
        if start == 0 and int(pointers[0]) != 0:
            raise MatrixSourceError(f"{label} must start at zero")
        previous = int(pointers[-1])
        final = previous
    if final > limits.maxNnz:
        raise ResourceLimitError(f"{label} nnz {final} exceeds maxNnz={limits.maxNnz}")
    return final


def validate_minor_indexes(
    read: Callable[[int, int], NDArray[Any]],
    nnz: int,
    upper_bound: int,
    limits: SourceLimits,
    *,
    label: str,
) -> None:
    """Check that every stored minor-axis index lies in [0, upper_bound)."""
    for start in range(0, nnz, limits.compressedChunkNnz):
        stop = min(nnz, start + limits.compressedChunkNnz)
        indexes = read(start, stop)
        if not np.issubdtype(indexes.dtype, np.integer):
            raise TypeError(f"{label} must contain integers")
        if indexes.size and (np.any(indexes < 0) or np.any(indexes >= upper_bound)):
            raise MatrixSourceError(f"{label} contains an out-of-range value")


class BaseMatrixSource(ABC):
    def __init__(
        self,
        shape: Sequence[int],
        dtype: DTypeLike,
        *,
        row_names: Sequence[str | bytes] | NDArray[Any] | None = None,
        column_names: Sequence[str | bytes] | NDArray[Any] | None = None,
        is_sparse: bool,
        zero_preserving: bool = True,
        limits: SourceLimits = DEFAULT_LIMITS,
    ) -> None:
        self._limits = limits
        self._shape = _validate_shape(shape, limits)
        self._dtype: np.dtype[Any] = cast(np.dtype[Any], np.dtype(dtype))
        if self._dtype.hasobject or self._dtype.fields is not None:
            raise TypeError(f"matrix dtype {self._dtype} is not numeric")
        if self._dtype.kind not in "biufc":
            raise TypeError(f"matrix dtype {self._dtype} is not numeric")
        self._row_names = _normalize_names(row_names, self._shape[0], "row", limits)
        self._column_names = _normalize_names(
            column_names, self._shape[1], "column", limits
        )
        # The names never change, so their bytes are counted once rather than
        # in every read estimate.
        metadata_size = _metadata_bytes(self._row_names) + _metadata_bytes(
            self._column_names
        )
        self._metadataBytes = metadata_size
        if metadata_size > limits.maxMetadataBytes:
            raise ResourceLimitError(
                f"matrix names exceed maxMetadataBytes={limits.maxMetadataBytes}"
            )
        self._is_sparse = bool(is_sparse)
        self._zero_preserving = bool(zero_preserving)
        self._rowStore: SparseRowStore | None = None
        self._tempDir: Path | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype

    @property
    def row_names(self) -> tuple[str, ...] | None:
        return self._row_names

    @property
    def column_names(self) -> tuple[str, ...] | None:
        return self._column_names

    @property
    def is_sparse(self) -> bool:
        return self._is_sparse

    @property
    def zero_preserving(self) -> bool:
        return self._zero_preserving

    @property
    def n_features(self) -> int:
        return self.shape[0]

    @property
    def n_cells(self) -> int:
        return self.shape[1]

    @property
    def resident_bytes(self) -> int:
        return self._metadataBytes + (
            0 if self._rowStore is None else self._rowStore.indptr.nbytes
        )

    def _window(self, start: int, stop: int) -> tuple[int, int]:
        return _validate_window(start, stop, self.n_cells)

    def _admit(self, estimate: MemoryEstimate) -> None:
        if estimate.blockBytes > self._limits.maxBlockBytes:
            raise ResourceLimitError(
                f"matrix block needs {estimate.blockBytes} bytes; "
                f"maxBlockBytes={self._limits.maxBlockBytes}"
            )

    def _output_bytes(self, rows: int, values: int) -> int:
        return _block_output_bytes(
            self.dtype,
            rows,
            self.n_features,
            sparse=self.is_sparse,
            values=values,
        )

    def _row_store_memory(
        self,
        start: int,
        stop: int,
        *,
        nnz: int,
        source_bytes: int = 0,
    ) -> MemoryEstimate:
        if start == stop:
            return MemoryEstimate(self.resident_bytes, 0, 8)
        preparation = 0
        if self._rowStore is None:
            count = min(nnz, (stop - start) * self.n_features)
            minimum = (self.n_cells + 1) * 32 + source_bytes + 192
            preparation = max(
                minimum,
                min(self._limits.maxBlockBytes, minimum + nnz * 192),
            )
        else:
            count = int(self._rowStore.indptr[stop] - self._rowStore.indptr[start])
        output = count * (self.dtype.itemsize + 8) + (stop - start + 1) * 8
        return MemoryEstimate(
            self.resident_bytes, max(output, preparation - output), output
        )

    def _prepare_for_read(self) -> None:
        pass

    def _prepare_row_store(
        self,
        chunks: Callable[[], Iterator[coo_matrix]],
        *,
        source_bytes: int = 0,
    ) -> SparseRowStore:
        if self._rowStore is None:
            if (
                self.n_cells + 1
            ) * 8 + self._metadataBytes > self._limits.maxMetadataBytes:
                raise ResourceLimitError("Sparse row pointers exceed maxMetadataBytes")
            try:
                # Cell rows keep the source dtype and their duplicate
                # coordinates, as cell-compressed sources do, so the writer
                # sums the duplicates without wrapping a narrow dtype.
                self._rowStore = SparseRowStore(
                    chunks,
                    (self.n_cells, self.n_features),
                    self.dtype,
                    max_bytes=self._limits.maxBlockBytes - source_bytes,
                    max_nnz=self._limits.maxNnz,
                    sum_duplicates=False,
                    temp_dir=self._tempDir,
                )
            except MemoryError as error:
                raise ResourceLimitError(
                    f"{error}; maxBlockBytes={self._limits.maxBlockBytes}"
                ) from error
        return self._rowStore

    @abstractmethod
    def read_cells(self, start: int, stop: int) -> MatrixBlock:
        """Return cells ``[start, stop)`` as a cell-by-feature block."""

    @abstractmethod
    def estimate_read_memory(self, start: int, stop: int) -> MemoryEstimate:
        """Return the memory that reading cells ``[start, stop)`` needs."""


class CompressedMatrixSource(BaseMatrixSource):
    """Sparse source stored as compressed pointers, minor indexes, and values.

    When the compressed axis runs over cells, blocks are sliced directly.
    Otherwise the entries are transposed once into a disk-backed row store.
    """

    _decodeWorkingBytes = 0

    def __init__(
        self,
        shape: Sequence[int],
        dtype: DTypeLike,
        *,
        cells_compressed: bool,
        nnz: int,
        row_names: Sequence[str | bytes] | NDArray[Any] | None = None,
        column_names: Sequence[str | bytes] | NDArray[Any] | None = None,
        limits: SourceLimits = DEFAULT_LIMITS,
    ) -> None:
        super().__init__(
            shape,
            dtype,
            row_names=row_names,
            column_names=column_names,
            is_sparse=True,
            limits=limits,
        )
        self._cellsCompressed = bool(cells_compressed)
        self._nnz = int(nnz)

    @property
    def nnz(self) -> int:
        return self._nnz

    @abstractmethod
    def _read_pointers(self, start: int, stop: int) -> NDArray[np.int64]:
        """Return compressed pointers ``[start, stop)``."""

    @abstractmethod
    def _read_entries(
        self, start: int, stop: int
    ) -> tuple[NDArray[Any], NDArray[np.int64]]:
        """Return the values and minor indexes of entries ``[start, stop)``."""

    def _cell_bounds(
        self,
        start: int,
        stop: int,
    ) -> tuple[NDArray[np.int64], int, int]:
        pointers = self._read_pointers(start, stop + 1)
        data_start = int(pointers[0])
        data_stop = int(pointers[-1])
        return pointers - data_start, data_start, data_stop

    def estimate_read_memory(self, start: int, stop: int) -> MemoryEstimate:
        start, stop = self._window(start, stop)
        if not self._cellsCompressed:
            return self._row_store_memory(
                start,
                stop,
                nnz=self.nnz,
                source_bytes=(self.n_features + 1) * _INDEX_BYTES,
            )
        pointers, data_start, data_stop = self._cell_bounds(start, stop)
        output = (data_stop - data_start) * (
            self.dtype.itemsize + _INDEX_BYTES
        ) + pointers.nbytes
        return MemoryEstimate(
            self.resident_bytes, output + self._decodeWorkingBytes, output
        )

    def read_cells(self, start: int, stop: int) -> csr_matrix:
        start, stop = self._window(start, stop)
        self._admit(self.estimate_read_memory(start, stop))
        if self._cellsCompressed:
            pointers, data_start, data_stop = self._cell_bounds(start, stop)
            data, indexes = self._read_entries(data_start, data_stop)
            return csr_matrix(
                (data, indexes, pointers),
                shape=(stop - start, self.n_features),
                dtype=self.dtype,
            )
        if start == stop:
            return csr_matrix((0, self.n_features), dtype=self.dtype)
        return self._prepare_row_store(
            self._feature_chunks,
            source_bytes=(self.n_features + 1) * _INDEX_BYTES,
        ).read(start, stop)

    def _prepare_for_read(self) -> None:
        if not self._cellsCompressed and self.n_cells:
            self._prepare_row_store(
                self._feature_chunks,
                source_bytes=(self.n_features + 1) * _INDEX_BYTES,
            )

    def _feature_chunks(self) -> Iterator[coo_matrix]:
        chunk_nnz = max(
            1,
            min(
                self._limits.compressedChunkNnz,
                (
                    self._limits.maxBlockBytes
                    - (self.n_cells + 1) * 32
                    - (self.n_features + 1) * _INDEX_BYTES
                )
                // 384,
            ),
        )
        pointers = self._read_pointers(0, self.n_features + 1)
        for start in range(0, self.nnz, chunk_nnz):
            stop = min(self.nnz, start + chunk_nnz)
            features = (
                np.searchsorted(pointers, np.arange(start, stop), side="right") - 1
            )
            data, cells = self._read_entries(start, stop)
            yield coo_matrix(
                (data, (cells, features)),
                shape=(self.n_cells, self.n_features),
            )


class DenseMatrixSource(BaseMatrixSource):
    def __init__(
        self,
        values: Any,
        shape: Sequence[int] | None = None,
        *,
        dtype: DTypeLike | None = None,
        row_names: Sequence[str | bytes] | NDArray[Any] | None = None,
        column_names: Sequence[str | bytes] | NDArray[Any] | None = None,
        limits: SourceLimits = DEFAULT_LIMITS,
    ) -> None:
        normalized_values = (
            np.asarray(values)
            if isinstance(values, Sequence) and not isinstance(values, str | bytes)
            else values
        )
        source_shape = _array_shape(normalized_values)
        if len(source_shape) == 1:
            if shape is None:
                raise MatrixSourceError(
                    "shape is required for a flat R column-major vector"
                )
            logical_shape = _validate_shape(shape, limits)
            if source_shape[0] != logical_shape[0] * logical_shape[1]:
                raise MatrixSourceError(
                    f"flat dense source has {source_shape[0]} values; "
                    f"expected {logical_shape[0] * logical_shape[1]}"
                )
            self._flat = True
        elif len(source_shape) == 2:
            logical_shape = _validate_shape(
                source_shape if shape is None else shape, limits
            )
            if source_shape != logical_shape:
                raise MatrixSourceError(
                    f"dense source shape {source_shape} does not match {logical_shape}"
                )
            self._flat = False
        else:
            raise MatrixSourceError(
                "dense matrix source must be one or two-dimensional"
            )
        source_dtype = (
            _array_dtype(normalized_values) if dtype is None else np.dtype(dtype)
        )
        self._values = normalized_values
        super().__init__(
            logical_shape,
            source_dtype,
            row_names=row_names,
            column_names=column_names,
            is_sparse=False,
            limits=limits,
        )

    @property
    def resident_bytes(self) -> int:
        return super().resident_bytes + _array_resident_bytes(self._values)

    def estimate_read_memory(self, start: int, stop: int) -> MemoryEstimate:
        start, stop = self._window(start, stop)
        output = (stop - start) * self.n_features * self.dtype.itemsize
        return MemoryEstimate(self.resident_bytes, output, output)

    def read_cells(self, start: int, stop: int) -> NDArray[Any]:
        start, stop = self._window(start, stop)
        estimate = self.estimate_read_memory(start, stop)
        self._admit(estimate)
        if self._flat:
            flat = read_window(
                self._values,
                start * self.n_features,
                stop * self.n_features,
                dtype=self.dtype,
            )
            return np.ascontiguousarray(flat.reshape(stop - start, self.n_features))
        try:
            feature_by_cell = np.asarray(self._values[:, start:stop])
        except (IndexError, TypeError, ValueError) as error:
            raise MatrixSourceError(
                "two-dimensional source does not support bounded column slicing"
            ) from error
        expected_shape = (self.n_features, stop - start)
        if feature_by_cell.shape != expected_shape:
            raise MatrixSourceError(
                f"dense block has shape {feature_by_cell.shape}; "
                f"expected {expected_shape}"
            )
        return np.ascontiguousarray(feature_by_cell.T, dtype=self.dtype)


class CscMatrixSource(CompressedMatrixSource):
    _SUPPORTED_CLASSES = frozenset(
        {
            "dgCMatrix",
            "lgCMatrix",
            "ngCMatrix",
            "igCMatrix",
            "CsparseMatrix",
        }
    )

    def __init__(
        self,
        x: Any | None,
        i: Any,
        p: Any,
        shape: Sequence[int],
        *,
        dtype: DTypeLike | None = None,
        class_name: str = "dgCMatrix",
        row_names: Sequence[str | bytes] | NDArray[Any] | None = None,
        column_names: Sequence[str | bytes] | NDArray[Any] | None = None,
        limits: SourceLimits = DEFAULT_LIMITS,
    ) -> None:
        logical_shape = _validate_shape(shape, limits)
        if class_name not in self._SUPPORTED_CLASSES:
            raise MatrixSourceError(f"unsupported Matrix class {class_name!r}")
        pointer_shape = _array_shape(p)
        if pointer_shape != (logical_shape[1] + 1,):
            raise MatrixSourceError(
                f"CSC p slot has shape {pointer_shape}; "
                f"expected ({logical_shape[1] + 1},)"
            )
        index_shape = _array_shape(i)
        if len(index_shape) != 1:
            raise MatrixSourceError("CSC i slot must be one-dimensional")
        self._x = x
        self._i = i
        self._p = p
        nnz = validate_compressed_pointers(
            lambda start, stop: read_window(p, start, stop),
            logical_shape[1] + 1,
            limits,
            label="CSC p slot",
        )
        if index_shape != (nnz,):
            raise MatrixSourceError(
                f"CSC i slot has length {index_shape[0]}; expected {nnz}"
            )
        validate_minor_indexes(
            lambda start, stop: read_window(i, start, stop),
            nnz,
            logical_shape[0],
            limits,
            label="CSC i slot",
        )
        if x is None:
            source_dtype = np.dtype(bool if dtype is None else dtype)
        else:
            value_shape = _array_shape(x)
            if value_shape != (nnz,):
                raise MatrixSourceError(
                    f"CSC x slot has shape {value_shape}; expected ({nnz},)"
                )
            source_dtype = _array_dtype(x) if dtype is None else np.dtype(dtype)
        super().__init__(
            logical_shape,
            source_dtype,
            cells_compressed=True,
            nnz=nnz,
            row_names=row_names,
            column_names=column_names,
            limits=limits,
        )

    @property
    def resident_bytes(self) -> int:
        values = 0 if self._x is None else _array_resident_bytes(self._x)
        return (
            super().resident_bytes
            + values
            + _array_resident_bytes(self._i)
            + _array_resident_bytes(self._p)
        )

    def _read_pointers(self, start: int, stop: int) -> NDArray[np.int64]:
        return read_window(self._p, start, stop, dtype=np.int64)

    def _read_entries(
        self, start: int, stop: int
    ) -> tuple[NDArray[Any], NDArray[np.int64]]:
        indexes = read_window(self._i, start, stop, dtype=np.int64)
        if self._x is None:
            return np.ones(stop - start, dtype=self.dtype), indexes
        return read_window(self._x, start, stop, dtype=self.dtype), indexes


class MappedMatrixSource(BaseMatrixSource):
    def __init__(
        self,
        source: MatrixSource,
        *,
        feature_indices: Sequence[int] | NDArray[Any] | None = None,
        cell_indices: Sequence[int] | NDArray[Any] | None = None,
        limits: SourceLimits = DEFAULT_LIMITS,
    ) -> None:
        self.source = source
        self.featureIndices = (
            None
            if feature_indices is None
            else _normalize_indexes(feature_indices, source.shape[0], "feature")
        )
        self.cellIndices = (
            None
            if cell_indices is None
            else _normalize_indexes(cell_indices, source.shape[1], "cell")
        )
        n_features = (
            source.shape[0]
            if self.featureIndices is None
            else int(self.featureIndices.size)
        )
        n_cells = (
            source.shape[1] if self.cellIndices is None else int(self.cellIndices.size)
        )
        row_names = source.row_names
        if row_names is not None and self.featureIndices is not None:
            row_names = tuple(row_names[index] for index in self.featureIndices)
        column_names = source.column_names
        if column_names is not None and self.cellIndices is not None:
            column_names = tuple(column_names[index] for index in self.cellIndices)
        super().__init__(
            (n_features, n_cells),
            source.dtype,
            row_names=row_names,
            column_names=column_names,
            is_sparse=source.is_sparse,
            zero_preserving=source.zero_preserving,
            limits=limits,
        )

    @property
    def resident_bytes(self) -> int:
        mapping_bytes = 0
        if self.featureIndices is not None:
            mapping_bytes += self.featureIndices.nbytes
        if self.cellIndices is not None:
            mapping_bytes += self.cellIndices.nbytes
        return int(super().resident_bytes + self.source.resident_bytes + mapping_bytes)

    def estimate_read_memory(self, start: int, stop: int) -> MemoryEstimate:
        start, stop = self._window(start, stop)
        rows = stop - start
        if self.cellIndices is None:
            estimate = self.source.estimate_read_memory(start, stop)
            child_bytes = estimate.blockBytes
            values = _value_bound(self.source, estimate, rows)
        else:
            child_bytes, values = _selected_estimate(
                self.source, self.cellIndices[start:stop]
            )
        output = self._output_bytes(rows, values)
        return MemoryEstimate(self.resident_bytes, child_bytes + output, output)

    def read_cells(self, start: int, stop: int) -> MatrixBlock:
        start, stop = self._window(start, stop)
        estimate = self.estimate_read_memory(start, stop)
        self._admit(estimate)
        if self.cellIndices is None:
            block = self.source.read_cells(start, stop)
        else:
            block = _read_selected_cells(self.source, self.cellIndices[start:stop])
        if self.featureIndices is None:
            return block
        if issparse(block):
            return _block_to_csr(block)[:, self.featureIndices].tocsr()
        return np.ascontiguousarray(np.asarray(block)[:, self.featureIndices])


class TransposeMatrixSource(BaseMatrixSource):
    def __init__(
        self,
        source: MatrixSource,
        *,
        tile_cells: int | None = None,
        limits: SourceLimits = DEFAULT_LIMITS,
    ) -> None:
        self.source = source
        self.tile_cells = limits.tileCells if tile_cells is None else int(tile_cells)
        self._transposeDirectory: TemporaryDirectory[str] | None = None
        if self.tile_cells <= 0:
            raise ValueError("tile_cells must be positive")
        super().__init__(
            (source.shape[1], source.shape[0]),
            source.dtype,
            row_names=source.column_names,
            column_names=source.row_names,
            is_sparse=source.is_sparse,
            zero_preserving=source.zero_preserving,
            limits=limits,
        )

    @property
    def resident_bytes(self) -> int:
        return super().resident_bytes + self.source.resident_bytes

    def estimate_read_memory(self, start: int, stop: int) -> MemoryEstimate:
        start, stop = self._window(start, stop)
        if self.is_sparse:
            return self._row_store_memory(
                start, stop, nnz=self.n_cells * self.n_features
            )
        output = (stop - start) * self.n_features * self.dtype.itemsize
        working = output
        if (
            start != stop
            and self.n_features
            and self._transposeDirectory is None
            and not (
                type(self.source) is DenseMatrixSource
                and isinstance(self.source._values, np.ndarray)
            )
        ):
            working = max(
                working,
                max(
                    estimate.workingBytes + 2 * estimate.outputBytes
                    for _, _, estimate in self._source_windows()
                )
                + min(
                    1024 * 1024,
                    max(1, self._limits.maxBlockBytes // 4),
                    self.n_cells * self.n_features * self.dtype.itemsize,
                )
                - output,
            )
        return MemoryEstimate(self.resident_bytes, working, output)

    def _source_windows(self) -> Iterator[tuple[int, int, MemoryEstimate]]:
        start = 0
        available = self._limits.maxBlockBytes
        if self.is_sparse:
            available -= (self.n_cells + 1) * 32
        while start < self.source.shape[1]:
            width = min(self.tile_cells, self.source.shape[1] - start)
            while True:
                estimate = self.source.estimate_read_memory(start, start + width)
                if estimate.blockBytes <= available // 2:
                    break
                if width == 1:
                    raise ResourceLimitError(
                        "Transpose source tile exceeds maxBlockBytes"
                    )
                width = max(1, width // 2)
            yield start, start + width, estimate
            start += width

    def _source_blocks(self) -> Iterator[tuple[int, MatrixBlock]]:
        for start, stop, _ in self._source_windows():
            yield start, self.source.read_cells(start, stop)

    def _transposed_chunks(self) -> Iterator[coo_matrix]:
        for start, block in self._source_blocks():
            values = coo_matrix(block)
            yield coo_matrix(
                (values.data, (values.col, values.row + start)),
                shape=(self.n_cells, self.n_features),
            )

    def _prepare_for_read(self) -> None:
        if not self.n_cells or not self.n_features:
            return
        self._admit(self.estimate_read_memory(0, 1))
        if self.is_sparse:
            self._prepare_row_store(self._transposed_chunks)
        elif not (
            type(self.source) is DenseMatrixSource
            and isinstance(self.source._values, np.ndarray)
        ):
            self._prepare_dense()

    def _prepare_dense(self) -> Path:
        import h5py

        if self._transposeDirectory is None:
            directory = TemporaryDirectory(prefix="scarf-transpose-", dir=self._tempDir)
            path = Path(directory.name) / "matrix.h5"
            try:
                required = self.n_cells * self.n_features * self.dtype.itemsize
                import shutil

                if required > shutil.disk_usage(directory.name).free:
                    raise OSError(f"Dense transpose needs {required} temporary bytes")
                with h5py.File(path, "w") as handle:
                    chunk_bytes = min(
                        1024 * 1024, max(1, self._limits.maxBlockBytes // 4)
                    )
                    width = min(
                        self.tile_cells,
                        self.n_features,
                        max(1, chunk_bytes // self.dtype.itemsize),
                    )
                    height = min(
                        self.n_cells,
                        max(1, chunk_bytes // (width * self.dtype.itemsize)),
                    )
                    target = handle.create_dataset(
                        "matrix",
                        shape=(self.n_cells, self.n_features),
                        chunks=(height, width),
                        dtype=self.dtype,
                    )
                    for source_start, block in self._source_blocks():
                        values = _block_to_dense(block, dtype=self.dtype)
                        target[:, source_start : source_start + values.shape[0]] = (
                            values.T
                        )
                self._transposeDirectory = directory
            except BaseException:
                directory.cleanup()
                raise
        return Path(self._transposeDirectory.name) / "matrix.h5"

    def read_cells(self, start: int, stop: int) -> MatrixBlock:
        start, stop = self._window(start, stop)
        estimate = self.estimate_read_memory(start, stop)
        self._admit(estimate)
        if start == stop:
            return _empty_block(0, self.n_features, self.dtype, self.is_sparse)
        if self.is_sparse:
            return self._prepare_row_store(self._transposed_chunks).read(start, stop)
        if self.n_features == 0:
            return np.empty((stop - start, 0), dtype=self.dtype)
        if type(self.source) is DenseMatrixSource and isinstance(
            self.source._values, np.ndarray
        ):
            values = self.source._values
            if self.source._flat:
                values = values.reshape(self.source.shape, order="F")
            return np.ascontiguousarray(values[start:stop], dtype=self.dtype)
        import h5py

        with h5py.File(self._prepare_dense(), "r") as handle:
            return np.asarray(handle["matrix"][start:stop])


def _matching_names(
    sources: Sequence[MatrixSource],
    axis: Literal["row_names", "column_names"],
) -> tuple[str, ...] | None:
    first = sources[0].row_names if axis == "row_names" else sources[0].column_names
    for source in sources[1:]:
        current = source.row_names if axis == "row_names" else source.column_names
        if first is not None and current is not None and current != first:
            raise MatrixSourceError(f"{axis.replace('_', ' ')} conflict across sources")
        if first is None:
            first = current
    return first


class FeatureBindMatrixSource(BaseMatrixSource):
    def __init__(
        self,
        sources: Sequence[MatrixSource],
        *,
        limits: SourceLimits = DEFAULT_LIMITS,
    ) -> None:
        if not sources:
            raise ValueError("feature bind requires at least one source")
        self.sources = tuple(sources)
        n_cells = self.sources[0].shape[1]
        if any(source.shape[1] != n_cells for source in self.sources):
            raise MatrixSourceError("feature bind sources must have equal cell counts")
        column_names = _matching_names(self.sources, "column_names")
        row_names = (
            tuple(name for source in self.sources for name in source.row_names or ())
            if all(source.row_names is not None for source in self.sources)
            else None
        )
        dtype = np.result_type(*(source.dtype for source in self.sources))
        super().__init__(
            (sum(source.shape[0] for source in self.sources), n_cells),
            dtype,
            row_names=row_names,
            column_names=column_names,
            is_sparse=all(source.is_sparse for source in self.sources),
            zero_preserving=all(source.zero_preserving for source in self.sources),
            limits=limits,
        )

    @property
    def resident_bytes(self) -> int:
        return super().resident_bytes + sum(
            source.resident_bytes for source in self.sources
        )

    def estimate_read_memory(self, start: int, stop: int) -> MemoryEstimate:
        start, stop = self._window(start, stop)
        child_bytes = 0
        values = 0
        for source in self.sources:
            estimate = source.estimate_read_memory(start, stop)
            child_bytes += estimate.blockBytes
            values += _value_bound(source, estimate, stop - start)
        output = self._output_bytes(stop - start, values)
        return MemoryEstimate(self.resident_bytes, child_bytes + output, output)

    def read_cells(self, start: int, stop: int) -> MatrixBlock:
        start, stop = self._window(start, stop)
        estimate = self.estimate_read_memory(start, stop)
        self._admit(estimate)
        blocks = [source.read_cells(start, stop) for source in self.sources]
        if self.is_sparse:
            return cast(
                MatrixBlock,
                hstack(
                    [_block_to_csr(block, dtype=self.dtype) for block in blocks],
                    format="csr",
                    dtype=self.dtype,
                ),
            )
        return np.hstack([_block_to_dense(block, dtype=self.dtype) for block in blocks])


class CellBindMatrixSource(BaseMatrixSource):
    def __init__(
        self,
        sources: Sequence[MatrixSource],
        *,
        limits: SourceLimits = DEFAULT_LIMITS,
    ) -> None:
        if not sources:
            raise ValueError("cell bind requires at least one source")
        self.sources = tuple(sources)
        n_features = self.sources[0].shape[0]
        if any(source.shape[0] != n_features for source in self.sources):
            raise MatrixSourceError("cell bind sources must have equal feature counts")
        row_names = _matching_names(self.sources, "row_names")
        column_names = (
            tuple(name for source in self.sources for name in source.column_names or ())
            if all(source.column_names is not None for source in self.sources)
            else None
        )
        dtype = np.result_type(*(source.dtype for source in self.sources))
        self._offsets = np.cumsum(
            [0, *(source.shape[1] for source in self.sources)],
            dtype=np.int64,
        )
        super().__init__(
            (n_features, int(self._offsets[-1])),
            dtype,
            row_names=row_names,
            column_names=column_names,
            is_sparse=all(source.is_sparse for source in self.sources),
            zero_preserving=all(source.zero_preserving for source in self.sources),
            limits=limits,
        )

    @property
    def resident_bytes(self) -> int:
        return int(
            super().resident_bytes
            + self._offsets.nbytes
            + sum(source.resident_bytes for source in self.sources)
        )

    def _pieces(self, start: int, stop: int) -> list[tuple[MatrixSource, int, int]]:
        pieces: list[tuple[MatrixSource, int, int]] = []
        for index, source in enumerate(self.sources):
            source_global_start = int(self._offsets[index])
            source_global_stop = int(self._offsets[index + 1])
            overlap_start = max(start, source_global_start)
            overlap_stop = min(stop, source_global_stop)
            if overlap_start < overlap_stop:
                pieces.append(
                    (
                        source,
                        overlap_start - source_global_start,
                        overlap_stop - source_global_start,
                    )
                )
        return pieces

    def estimate_read_memory(self, start: int, stop: int) -> MemoryEstimate:
        start, stop = self._window(start, stop)
        child_bytes = 0
        values = 0
        for source, local_start, local_stop in self._pieces(start, stop):
            estimate = source.estimate_read_memory(local_start, local_stop)
            child_bytes += estimate.blockBytes
            values += _value_bound(source, estimate, local_stop - local_start)
        output = self._output_bytes(stop - start, values)
        return MemoryEstimate(self.resident_bytes, child_bytes + output, output)

    def read_cells(self, start: int, stop: int) -> MatrixBlock:
        start, stop = self._window(start, stop)
        estimate = self.estimate_read_memory(start, stop)
        self._admit(estimate)
        blocks = [
            source.read_cells(local_start, local_stop)
            for source, local_start, local_stop in self._pieces(start, stop)
        ]
        if not blocks:
            return _empty_block(0, self.n_features, self.dtype, self.is_sparse)
        if self.is_sparse:
            return cast(
                MatrixBlock,
                vstack(
                    [_block_to_csr(block, dtype=self.dtype) for block in blocks],
                    format="csr",
                    dtype=self.dtype,
                ),
            )
        return np.vstack([_block_to_dense(block, dtype=self.dtype) for block in blocks])


@dataclass(frozen=True)
class LayerPlacement:
    source: MatrixSource
    featureIndices: Sequence[int] | NDArray[Any] | None = None
    cellIndices: Sequence[int] | NDArray[Any] | None = None
    name: str | None = None


@dataclass(frozen=True)
class _ResolvedLayerPlacement:
    source: MatrixSource
    featureIndices: NDArray[np.int64]
    cellIndices: NDArray[np.int64]
    name: str
    cellsSorted: bool

    def local_cells(self, start: int, stop: int) -> NDArray[np.int64]:
        if self.cellsSorted:
            left, right = np.searchsorted(self.cellIndices, (start, stop), side="left")
            return np.arange(int(left), int(right), dtype=np.int64)
        return np.flatnonzero(
            (self.cellIndices >= start) & (self.cellIndices < stop)
        ).astype(np.int64, copy=False)


def _map_names(
    local_names: tuple[str, ...] | None,
    global_names: tuple[str, ...],
    axis: str,
) -> NDArray[np.int64]:
    if local_names is None:
        raise MatrixSourceError(
            f"{axis} indexes are required when source names are absent"
        )
    if len(set(global_names)) != len(global_names):
        raise MatrixSourceError(
            f"global {axis} names must be unique for name-based stitching"
        )
    lookup = {name: index for index, name in enumerate(global_names)}
    missing = [name for name in local_names if name not in lookup]
    if missing:
        raise MatrixSourceError(
            f"source {axis} name {missing[0]!r} is absent from the global axis"
        )
    return np.asarray([lookup[name] for name in local_names], dtype=np.int64)


class LayerStitchMatrixSource(BaseMatrixSource):
    def __init__(
        self,
        layers: Sequence[LayerPlacement | MatrixSource],
        *,
        row_names: Sequence[str | bytes] | NDArray[Any],
        column_names: Sequence[str | bytes] | NDArray[Any],
        dtype: DTypeLike | None = None,
        limits: SourceLimits = DEFAULT_LIMITS,
    ) -> None:
        raw_rows = _normalize_names(row_names, len(row_names), "row", limits)
        raw_columns = _normalize_names(
            column_names, len(column_names), "column", limits
        )
        assert raw_rows is not None
        assert raw_columns is not None
        if not layers:
            raise ValueError("layer stitching requires at least one layer")
        resolved: list[_ResolvedLayerPlacement] = []
        for index, item in enumerate(layers):
            placement = (
                item if isinstance(item, LayerPlacement) else LayerPlacement(item)
            )
            features = (
                _map_names(placement.source.row_names, raw_rows, "feature")
                if placement.featureIndices is None
                else _normalize_indexes(
                    placement.featureIndices, len(raw_rows), "feature"
                )
            )
            cells = (
                _map_names(placement.source.column_names, raw_columns, "cell")
                if placement.cellIndices is None
                else _normalize_indexes(placement.cellIndices, len(raw_columns), "cell")
            )
            if features.size != placement.source.shape[0]:
                raise MatrixSourceError(
                    f"layer {index} maps {features.size} features; "
                    f"source has {placement.source.shape[0]}"
                )
            if cells.size != placement.source.shape[1]:
                raise MatrixSourceError(
                    f"layer {index} maps {cells.size} cells; "
                    f"source has {placement.source.shape[1]}"
                )
            if has_duplicates(features):
                raise MatrixSourceError(f"layer {index} repeats a global feature")
            if has_duplicates(cells):
                raise MatrixSourceError(f"layer {index} repeats a global cell")
            resolved.append(
                _ResolvedLayerPlacement(
                    placement.source,
                    features,
                    cells,
                    placement.name or f"layer[{index}]",
                    bool(cells.size < 2 or np.all(cells[1:] > cells[:-1])),
                )
            )
        self._validate_conflicts(resolved, len(raw_rows), len(raw_columns))
        self.layers = tuple(resolved)
        result_dtype = (
            np.result_type(*(layer.source.dtype for layer in resolved))
            if dtype is None
            else np.dtype(dtype)
        )
        super().__init__(
            (len(raw_rows), len(raw_columns)),
            result_dtype,
            row_names=raw_rows,
            column_names=raw_columns,
            is_sparse=True,
            zero_preserving=all(layer.source.zero_preserving for layer in resolved),
            limits=limits,
        )

    @staticmethod
    def _validate_conflicts(
        layers: Sequence[_ResolvedLayerPlacement],
        n_features: int,
        n_cells: int,
    ) -> None:
        for left in range(len(layers)):
            features = np.zeros(n_features, dtype=bool)
            cells = np.zeros(n_cells, dtype=bool)
            features[layers[left].featureIndices] = True
            cells[layers[left].cellIndices] = True
            for right in range(left + 1, len(layers)):
                if np.any(features[layers[right].featureIndices]) and np.any(
                    cells[layers[right].cellIndices]
                ):
                    raise MatrixSourceError(
                        f"layer coordinate conflict between "
                        f"{layers[left].name!r} and {layers[right].name!r}"
                    )

    @property
    def resident_bytes(self) -> int:
        mappings = sum(
            layer.featureIndices.nbytes + layer.cellIndices.nbytes
            for layer in self.layers
        )
        return int(
            super().resident_bytes
            + mappings
            + sum(layer.source.resident_bytes for layer in self.layers)
        )

    def estimate_read_memory(self, start: int, stop: int) -> MemoryEstimate:
        start, stop = self._window(start, stop)
        child_bytes = 0
        values = 0
        for layer in self.layers:
            layer_bytes, layer_values = _selected_estimate(
                layer.source, layer.local_cells(start, stop)
            )
            child_bytes += layer_bytes
            values += layer_values
        output = self._output_bytes(stop - start, values)
        return MemoryEstimate(self.resident_bytes, child_bytes + 2 * output, output)

    def read_cells(self, start: int, stop: int) -> csr_matrix:
        start, stop = self._window(start, stop)
        estimate = self.estimate_read_memory(start, stop)
        self._admit(estimate)
        data_parts: list[NDArray[Any]] = []
        row_parts: list[NDArray[np.int64]] = []
        column_parts: list[NDArray[np.int64]] = []
        for layer in self.layers:
            local_cells = layer.local_cells(start, stop)
            if local_cells.size == 0:
                continue
            block = _read_selected_cells(layer.source, local_cells)
            sparse_block = _block_to_csr(block, dtype=self.dtype).tocoo(copy=False)
            if sparse_block.nnz == 0:
                continue
            global_rows = (
                layer.cellIndices[local_cells[sparse_block.row]].astype(
                    np.int64, copy=False
                )
                - start
            )
            global_columns = layer.featureIndices[sparse_block.col].astype(
                np.int64, copy=False
            )
            data_parts.append(np.asarray(sparse_block.data, dtype=self.dtype))
            row_parts.append(global_rows)
            column_parts.append(global_columns)
        if not data_parts:
            return csr_matrix((stop - start, self.n_features), dtype=self.dtype)
        # Layers never share a coordinate, so stitching only re-indexes their
        # entries. A layer's duplicate coordinates are kept for the writer,
        # which sums them without wrapping a narrow dtype.
        rows = np.concatenate(row_parts)
        columns = np.concatenate(column_parts)
        order = np.lexsort((columns, rows))
        return csr_matrix(
            (
                np.concatenate(data_parts)[order],
                columns[order],
                cumulative_nnz(np.bincount(rows, minlength=stop - start)),
            ),
            shape=(stop - start, self.n_features),
        )


class RenamedMatrixSource(BaseMatrixSource):
    """Replace a source's axis names; ``None`` leaves that axis unnamed."""

    def __init__(
        self,
        source: MatrixSource,
        *,
        row_names: Sequence[str | bytes] | NDArray[Any] | None,
        column_names: Sequence[str | bytes] | NDArray[Any] | None,
        limits: SourceLimits = DEFAULT_LIMITS,
    ) -> None:
        self.source = source
        super().__init__(
            source.shape,
            source.dtype,
            row_names=row_names,
            column_names=column_names,
            is_sparse=source.is_sparse,
            zero_preserving=source.zero_preserving,
            limits=limits,
        )

    @property
    def resident_bytes(self) -> int:
        return super().resident_bytes + self.source.resident_bytes

    def estimate_read_memory(self, start: int, stop: int) -> MemoryEstimate:
        return self.source.estimate_read_memory(start, stop)

    def read_cells(self, start: int, stop: int) -> MatrixBlock:
        start, stop = self._window(start, stop)
        return self.source.read_cells(start, stop)


class DtypeMatrixSource(BaseMatrixSource):
    def __init__(
        self,
        source: MatrixSource,
        dtype: DTypeLike,
        *,
        limits: SourceLimits = DEFAULT_LIMITS,
    ) -> None:
        self.source = source
        super().__init__(
            source.shape,
            dtype,
            row_names=source.row_names,
            column_names=source.column_names,
            is_sparse=source.is_sparse,
            zero_preserving=source.zero_preserving,
            limits=limits,
        )

    @property
    def resident_bytes(self) -> int:
        return super().resident_bytes + self.source.resident_bytes

    def estimate_read_memory(self, start: int, stop: int) -> MemoryEstimate:
        start, stop = self._window(start, stop)
        child = self.source.estimate_read_memory(start, stop)
        output = self._output_bytes(
            stop - start, _value_bound(self.source, child, stop - start)
        )
        return MemoryEstimate(self.resident_bytes, child.blockBytes, output)

    def read_cells(self, start: int, stop: int) -> MatrixBlock:
        start, stop = self._window(start, stop)
        estimate = self.estimate_read_memory(start, stop)
        self._admit(estimate)
        block = self.source.read_cells(start, stop)
        if issparse(block):
            return _block_to_csr(block, dtype=self.dtype)
        return np.asarray(block).astype(self.dtype, copy=False)
