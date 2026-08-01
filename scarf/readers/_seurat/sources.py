from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import DTypeLike, NDArray
from scipy.sparse import (
    coo_matrix,
    csr_matrix,
    hstack,
    spmatrix,
    vstack,
)

from .errors import MatrixSourceError, ResourceLimitError


type MatrixBlock = NDArray[Any] | coo_matrix | csr_matrix


@dataclass(frozen=True)
class MemoryEstimate:
    residentBytes: int = 0
    workingBytes: int = 0
    outputBytes: int = 0

    @property
    def peakBytes(self) -> int:
        return self.residentBytes + self.workingBytes + self.outputBytes

    @property
    def resident_bytes(self) -> int:
        return self.residentBytes

    @property
    def working_bytes(self) -> int:
        return self.workingBytes

    @property
    def output_bytes(self) -> int:
        return self.outputBytes

    @property
    def peak_bytes(self) -> int:
        return self.peakBytes


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

    @property
    def max_features(self) -> int:
        return self.maxFeatures

    @property
    def max_cells(self) -> int:
        return self.maxCells

    @property
    def max_nnz(self) -> int:
        return self.maxNnz

    @property
    def max_block_bytes(self) -> int:
        return self.maxBlockBytes

    @property
    def max_metadata_bytes(self) -> int:
        return self.maxMetadataBytes

    @property
    def tile_cells(self) -> int:
        return self.tileCells

    @property
    def compressed_chunk_nnz(self) -> int:
        return self.compressedChunkNnz


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


def _metadata_bytes(names: tuple[str, ...] | None) -> int:
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
    if isinstance(names, str | bytes):
        raise TypeError(f"{axis} names must be a sequence")
    values: list[str] = []
    size = 0
    for value in names:
        if isinstance(value, bytes | np.bytes_):
            try:
                decoded = bytes(value).decode("utf-8")
            except UnicodeDecodeError as error:
                raise MatrixSourceError(
                    f"{axis} names contain invalid UTF-8"
                ) from error
        elif isinstance(value, str | np.str_):
            decoded = str(value)
        else:
            raise TypeError(f"{axis} names must contain only strings")
        if "\x00" in decoded:
            raise MatrixSourceError(f"{axis} names contain a NUL character")
        size += len(decoded.encode("utf-8")) + 8
        if size > limits.maxMetadataBytes:
            raise ResourceLimitError(
                f"{axis} names exceed maxMetadataBytes={limits.maxMetadataBytes}"
            )
        values.append(decoded)
    if len(values) != length:
        raise MatrixSourceError(
            f"{axis} names have length {len(values)}; expected {length}"
        )
    return tuple(values)


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
    if dtype is None:
        return np.asarray(values).dtype
    return cast(np.dtype[Any], np.dtype(dtype))


def _array_resident_bytes(values: Any) -> int:
    if isinstance(values, np.ndarray):
        return int(values.nbytes)
    return 0


def _read_1d(
    values: Any,
    start: int,
    stop: int,
    *,
    dtype: DTypeLike | None = None,
) -> NDArray[Any]:
    try:
        result = np.asarray(values[start:stop])
    except (IndexError, TypeError, ValueError) as error:
        raise MatrixSourceError(
            "array-like object does not support bounded slicing"
        ) from error
    if result.ndim != 1:
        result = result.reshape(-1)
    if result.size != stop - start:
        raise MatrixSourceError(
            f"bounded array read returned {result.size} values; expected {stop - start}"
        )
    if dtype is not None:
        result = result.astype(dtype, copy=False)
    return result


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


def _block_to_csr(
    block: MatrixBlock | spmatrix,
    *,
    dtype: DTypeLike | None = None,
) -> csr_matrix:
    if isinstance(block, spmatrix):
        result = block.tocsr(copy=False)
        if dtype is not None:
            result = result.astype(dtype, copy=False)
        return cast(csr_matrix, result)
    values = np.asarray(block)
    if values.ndim != 2:
        raise MatrixSourceError("matrix block must be two-dimensional")
    if dtype is not None:
        values = values.astype(dtype, copy=False)
    return csr_matrix(values)


def _block_to_dense(
    block: MatrixBlock | spmatrix,
    *,
    dtype: DTypeLike | None = None,
) -> NDArray[Any]:
    values = block.toarray() if isinstance(block, spmatrix) else np.asarray(block)
    if values.ndim != 2:
        raise MatrixSourceError("matrix block must be two-dimensional")
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


def _read_selected_cells(
    source: MatrixSource,
    indexes: NDArray[np.int64],
) -> MatrixBlock:
    if indexes.size == 0:
        return _empty_block(0, source.shape[0], source.dtype, source.is_sparse)
    pieces: list[MatrixBlock] = []
    run_start = 0
    for position in range(1, indexes.size + 1):
        run_finished = (
            position == indexes.size or indexes[position] != indexes[position - 1] + 1
        )
        if not run_finished:
            continue
        source_start = int(indexes[run_start])
        source_stop = int(indexes[position - 1]) + 1
        pieces.append(source.read_cells(source_start, source_stop))
        run_start = position
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


class BaseMatrixSource:
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
        metadata_size = _metadata_bytes(self._row_names) + _metadata_bytes(
            self._column_names
        )
        if metadata_size > limits.maxMetadataBytes:
            raise ResourceLimitError(
                f"matrix names exceed maxMetadataBytes={limits.maxMetadataBytes}"
            )
        self._is_sparse = bool(is_sparse)
        self._zero_preserving = bool(zero_preserving)

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
    def rowNames(self) -> tuple[str, ...] | None:
        return self.row_names

    @property
    def columnNames(self) -> tuple[str, ...] | None:
        return self.column_names

    @property
    def is_sparse(self) -> bool:
        return self._is_sparse

    @property
    def sparse(self) -> bool:
        return self.is_sparse

    @property
    def zero_preserving(self) -> bool:
        return self._zero_preserving

    @property
    def zeroPreserving(self) -> bool:
        return self.zero_preserving

    @property
    def n_features(self) -> int:
        return self.shape[0]

    @property
    def n_cells(self) -> int:
        return self.shape[1]

    @property
    def resident_bytes(self) -> int:
        return _metadata_bytes(self.row_names) + _metadata_bytes(self.column_names)

    @property
    def residentBytes(self) -> int:
        return self.resident_bytes

    def _window(self, start: int, stop: int) -> tuple[int, int]:
        return _validate_window(start, stop, self.n_cells)

    def _admit(self, estimate: MemoryEstimate) -> None:
        if estimate.workingBytes + estimate.outputBytes > self._limits.maxBlockBytes:
            raise ResourceLimitError(
                "matrix block needs "
                f"{estimate.workingBytes + estimate.outputBytes} bytes; "
                f"maxBlockBytes={self._limits.maxBlockBytes}"
            )

    def read_cells(self, start: int, stop: int) -> MatrixBlock:
        raise NotImplementedError

    def estimate_read_memory(self, start: int, stop: int) -> MemoryEstimate:
        raise NotImplementedError

    def memory_estimate(self, start: int, stop: int) -> MemoryEstimate:
        return self.estimate_read_memory(start, stop)

    def estimate_read_bytes(self, start: int, stop: int) -> int:
        return self.estimate_read_memory(start, stop).peakBytes

    def estimate_memory(self, start: int, stop: int) -> MemoryEstimate:
        return self.estimate_read_memory(start, stop)

    def estimated_peak_bytes(self, start: int, stop: int) -> int:
        return self.estimate_read_bytes(start, stop)


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
            flat = _read_1d(
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


RColumnMajorMatrixSource = DenseMatrixSource
InMemoryDenseMatrixSource = DenseMatrixSource
LazyDenseMatrixSource = DenseMatrixSource


class CscMatrixSource(BaseMatrixSource):
    _SUPPORTED_CLASSES = frozenset(
        {
            "dgCMatrix",
            "lgCMatrix",
            "ngCMatrix",
            "igCMatrix",
            "CsparseMatrix",
            "CSC",
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
        self.class_name = class_name
        self._nnz = self._validate_structure(logical_shape, limits)
        if x is None:
            source_dtype = np.dtype(bool if dtype is None else dtype)
        else:
            value_shape = _array_shape(x)
            if value_shape != (self._nnz,):
                raise MatrixSourceError(
                    f"CSC x slot has shape {value_shape}; expected ({self._nnz},)"
                )
            source_dtype = _array_dtype(x) if dtype is None else np.dtype(dtype)
        super().__init__(
            logical_shape,
            source_dtype,
            row_names=row_names,
            column_names=column_names,
            is_sparse=True,
            limits=limits,
        )

    def _validate_structure(
        self,
        shape: tuple[int, int],
        limits: SourceLimits,
    ) -> int:
        chunk_size = max(1, min(limits.compressedChunkNnz, shape[1] + 1))
        previous: int | None = None
        last = 0
        for start in range(0, shape[1] + 1, chunk_size):
            stop = min(shape[1] + 1, start + chunk_size)
            pointers = _read_1d(self._p, start, stop)
            if not np.issubdtype(pointers.dtype, np.integer):
                raise TypeError("CSC p slot must contain integers")
            pointers = pointers.astype(np.int64, copy=False)
            if previous is not None and pointers.size and int(pointers[0]) < previous:
                raise MatrixSourceError("CSC p slot must be nondecreasing")
            if pointers.size > 1 and np.any(pointers[1:] < pointers[:-1]):
                raise MatrixSourceError("CSC p slot must be nondecreasing")
            if start == 0 and (not pointers.size or int(pointers[0]) != 0):
                raise MatrixSourceError("CSC p slot must start at zero")
            if pointers.size:
                previous = int(pointers[-1])
                last = previous
        if last < 0:
            raise MatrixSourceError("CSC p slot cannot contain negative offsets")
        if last > limits.maxNnz:
            raise ResourceLimitError(f"CSC nnz {last} exceeds maxNnz={limits.maxNnz}")
        if _array_shape(self._i) != (last,):
            raise MatrixSourceError(
                f"CSC i slot has length {_array_shape(self._i)[0]}; expected {last}"
            )
        for start in range(0, last, limits.compressedChunkNnz):
            stop = min(last, start + limits.compressedChunkNnz)
            indexes = _read_1d(self._i, start, stop)
            if not np.issubdtype(indexes.dtype, np.integer):
                raise TypeError("CSC i slot must contain integers")
            if indexes.size and (np.any(indexes < 0) or np.any(indexes >= shape[0])):
                raise MatrixSourceError("CSC i slot contains an out-of-range row")
        return last

    @property
    def nnz(self) -> int:
        return self._nnz

    @property
    def resident_bytes(self) -> int:
        values = 0 if self._x is None else _array_resident_bytes(self._x)
        return (
            super().resident_bytes
            + values
            + _array_resident_bytes(self._i)
            + _array_resident_bytes(self._p)
        )

    def _range(self, start: int, stop: int) -> tuple[NDArray[Any], int, int]:
        pointers = _read_1d(self._p, start, stop + 1)
        if not np.issubdtype(pointers.dtype, np.integer):
            raise TypeError("CSC p slot must contain integers")
        pointers = pointers.astype(np.int64, copy=False)
        data_start = int(pointers[0])
        data_stop = int(pointers[-1])
        return pointers - data_start, data_start, data_stop

    def estimate_read_memory(self, start: int, stop: int) -> MemoryEstimate:
        start, stop = self._window(start, stop)
        pointers, data_start, data_stop = self._range(start, stop)
        nnz = data_stop - data_start
        index_size = np.dtype(np.int64).itemsize
        output = nnz * (self.dtype.itemsize + index_size) + pointers.nbytes
        working = output
        return MemoryEstimate(self.resident_bytes, working, output)

    def read_cells(self, start: int, stop: int) -> csr_matrix:
        start, stop = self._window(start, stop)
        estimate = self.estimate_read_memory(start, stop)
        self._admit(estimate)
        pointers, data_start, data_stop = self._range(start, stop)
        indexes = _read_1d(self._i, data_start, data_stop, dtype=np.int64)
        if self._x is None:
            data = np.ones(data_stop - data_start, dtype=self.dtype)
        else:
            data = _read_1d(self._x, data_start, data_stop, dtype=self.dtype)
        return csr_matrix(
            (data, indexes, pointers),
            shape=(stop - start, self.n_features),
            dtype=self.dtype,
        )


CSCMatrixSource = CscMatrixSource
MatrixCscSource = CscMatrixSource


class MappedMatrixSource(BaseMatrixSource):
    def __init__(
        self,
        source: MatrixSource,
        *,
        feature_indices: Sequence[int] | NDArray[Any] | None = None,
        cell_indices: Sequence[int] | NDArray[Any] | None = None,
        row_names: Sequence[str | bytes] | NDArray[Any] | None = None,
        column_names: Sequence[str | bytes] | NDArray[Any] | None = None,
        limits: SourceLimits = DEFAULT_LIMITS,
    ) -> None:
        self.source = source
        self.feature_indices = (
            None
            if feature_indices is None
            else _normalize_indexes(feature_indices, source.shape[0], "feature")
        )
        self.cell_indices = (
            None
            if cell_indices is None
            else _normalize_indexes(cell_indices, source.shape[1], "cell")
        )
        n_features = (
            source.shape[0]
            if self.feature_indices is None
            else int(self.feature_indices.size)
        )
        n_cells = (
            source.shape[1]
            if self.cell_indices is None
            else int(self.cell_indices.size)
        )
        mapped_rows = row_names
        if mapped_rows is None and source.row_names is not None:
            mapped_rows = (
                source.row_names
                if self.feature_indices is None
                else tuple(source.row_names[index] for index in self.feature_indices)
            )
        mapped_columns = column_names
        if mapped_columns is None and source.column_names is not None:
            mapped_columns = (
                source.column_names
                if self.cell_indices is None
                else tuple(source.column_names[index] for index in self.cell_indices)
            )
        super().__init__(
            (n_features, n_cells),
            source.dtype,
            row_names=mapped_rows,
            column_names=mapped_columns,
            is_sparse=source.is_sparse,
            zero_preserving=source.zero_preserving,
            limits=limits,
        )

    @property
    def resident_bytes(self) -> int:
        mapping_bytes = 0
        if self.feature_indices is not None:
            mapping_bytes += self.feature_indices.nbytes
        if self.cell_indices is not None:
            mapping_bytes += self.cell_indices.nbytes
        return int(super().resident_bytes + self.source.resident_bytes + mapping_bytes)

    def estimate_read_memory(self, start: int, stop: int) -> MemoryEstimate:
        start, stop = self._window(start, stop)
        rows = stop - start
        if self.cell_indices is None:
            source_estimate = self.source.estimate_read_memory(start, stop)
        elif rows == 0:
            source_estimate = MemoryEstimate()
        else:
            selected = self.cell_indices[start:stop]
            source_estimate = MemoryEstimate(
                workingBytes=sum(
                    self.source.estimate_read_memory(
                        int(index), int(index) + 1
                    ).peakBytes
                    for index in selected
                )
            )
        output = rows * self.n_features * self.dtype.itemsize
        if self.is_sparse:
            output += rows * np.dtype(np.int64).itemsize
        return MemoryEstimate(
            self.resident_bytes,
            source_estimate.workingBytes + output,
            output,
        )

    def read_cells(self, start: int, stop: int) -> MatrixBlock:
        start, stop = self._window(start, stop)
        estimate = self.estimate_read_memory(start, stop)
        self._admit(estimate)
        if self.cell_indices is None:
            block = self.source.read_cells(start, stop)
        else:
            block = _read_selected_cells(self.source, self.cell_indices[start:stop])
        if self.feature_indices is None:
            return block
        if isinstance(block, spmatrix):
            return cast(
                MatrixBlock,
                block.tocsr(copy=False)[:, self.feature_indices].tocsr(),
            )
        return np.ascontiguousarray(np.asarray(block)[:, self.feature_indices])


SubsetMatrixSource = MappedMatrixSource
ReorderedMatrixSource = MappedMatrixSource


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
        output = (stop - start) * self.n_features * self.dtype.itemsize
        tile = min(self.tile_cells, self.source.shape[1])
        working = 0
        for source_start in range(0, self.source.shape[1], max(1, tile)):
            source_stop = min(self.source.shape[1], source_start + max(1, tile))
            working = max(
                working,
                self.source.estimate_read_memory(source_start, source_stop).peakBytes,
            )
        return MemoryEstimate(self.resident_bytes, working, output)

    def read_cells(self, start: int, stop: int) -> MatrixBlock:
        start, stop = self._window(start, stop)
        estimate = self.estimate_read_memory(start, stop)
        self._admit(estimate)
        if start == stop:
            return _empty_block(0, self.n_features, self.dtype, self.is_sparse)
        if self.is_sparse:
            pieces: list[csr_matrix] = []
            for source_start in range(0, self.source.shape[1], self.tile_cells):
                source_stop = min(self.source.shape[1], source_start + self.tile_cells)
                block = _block_to_csr(
                    self.source.read_cells(source_start, source_stop),
                    dtype=self.dtype,
                )
                pieces.append(block[:, start:stop].T.tocsr())
            if not pieces:
                return csr_matrix((stop - start, self.n_features), dtype=self.dtype)
            return cast(
                MatrixBlock,
                hstack(pieces, format="csr", dtype=self.dtype),
            )
        output = np.empty((stop - start, self.source.shape[1]), dtype=self.dtype)
        for source_start in range(0, self.source.shape[1], self.tile_cells):
            source_stop = min(self.source.shape[1], source_start + self.tile_cells)
            block = _block_to_dense(
                self.source.read_cells(source_start, source_stop),
                dtype=self.dtype,
            )
            output[:, source_start:source_stop] = block[:, start:stop].T
        return output


TransposeSource = TransposeMatrixSource


def _matching_names(
    sources: Sequence[MatrixSource],
    axis: str,
) -> tuple[str, ...] | None:
    if axis == "row_names":
        first = sources[0].row_names
    elif axis == "column_names":
        first = sources[0].column_names
    else:
        raise ValueError("axis must be row_names or column_names")
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
        child = sum(
            source.estimate_read_memory(start, stop).peakBytes
            for source in self.sources
        )
        output = (stop - start) * self.n_features * self.dtype.itemsize
        return MemoryEstimate(self.resident_bytes, child, output)

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
        child = sum(
            source.estimate_read_memory(local_start, local_stop).peakBytes
            for source, local_start, local_stop in self._pieces(start, stop)
        )
        output = (stop - start) * self.n_features * self.dtype.itemsize
        return MemoryEstimate(self.resident_bytes, child, output)

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


RowBindMatrixSource = FeatureBindMatrixSource
ColumnBindMatrixSource = CellBindMatrixSource
FeatureBindSource = FeatureBindMatrixSource
CellBindSource = CellBindMatrixSource


@dataclass(frozen=True, init=False)
class LayerPlacement:
    source: MatrixSource
    featureIndices: Sequence[int] | NDArray[Any] | None = None
    cellIndices: Sequence[int] | NDArray[Any] | None = None
    name: str | None = None

    def __init__(
        self,
        source: MatrixSource,
        feature_indices: Sequence[int] | NDArray[Any] | None = None,
        cell_indices: Sequence[int] | NDArray[Any] | None = None,
        name: str | None = None,
        *,
        featureIndices: Sequence[int] | NDArray[Any] | None = None,
        cellIndices: Sequence[int] | NDArray[Any] | None = None,
    ) -> None:
        if feature_indices is not None and featureIndices is not None:
            raise TypeError("provide only one feature index spelling")
        if cell_indices is not None and cellIndices is not None:
            raise TypeError("provide only one cell index spelling")
        object.__setattr__(self, "source", source)
        object.__setattr__(
            self,
            "featureIndices",
            feature_indices if featureIndices is None else featureIndices,
        )
        object.__setattr__(
            self,
            "cellIndices",
            cell_indices if cellIndices is None else cellIndices,
        )
        object.__setattr__(self, "name", name)

    @property
    def feature_indices(self) -> Sequence[int] | NDArray[Any] | None:
        return self.featureIndices

    @property
    def cell_indices(self) -> Sequence[int] | NDArray[Any] | None:
        return self.cellIndices


@dataclass(frozen=True)
class _ResolvedLayerPlacement:
    source: MatrixSource
    featureIndices: NDArray[np.int64]
    cellIndices: NDArray[np.int64]
    name: str


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
            if np.unique(features).size != features.size:
                raise MatrixSourceError(f"layer {index} repeats a global feature")
            if np.unique(cells).size != cells.size:
                raise MatrixSourceError(f"layer {index} repeats a global cell")
            self._validate_layer_names(
                placement.source,
                features,
                cells,
                raw_rows,
                raw_columns,
                index,
            )
            resolved.append(
                _ResolvedLayerPlacement(
                    placement.source,
                    features,
                    cells,
                    placement.name or f"layer[{index}]",
                )
            )
        self._validate_conflicts(resolved)
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
    def _validate_layer_names(
        source: MatrixSource,
        feature_indexes: NDArray[np.int64],
        cell_indexes: NDArray[np.int64],
        row_names: tuple[str, ...],
        column_names: tuple[str, ...],
        index: int,
    ) -> None:
        if source.row_names is not None:
            expected = tuple(row_names[position] for position in feature_indexes)
            if source.row_names != expected:
                raise MatrixSourceError(
                    f"layer {index} feature names conflict with global mapping"
                )
        if source.column_names is not None:
            expected = tuple(column_names[position] for position in cell_indexes)
            if source.column_names != expected:
                raise MatrixSourceError(
                    f"layer {index} cell names conflict with global mapping"
                )

    @staticmethod
    def _validate_conflicts(
        layers: Sequence[_ResolvedLayerPlacement],
    ) -> None:
        feature_sets = [set(layer.featureIndices.tolist()) for layer in layers]
        cell_sets = [set(layer.cellIndices.tolist()) for layer in layers]
        for left in range(len(layers)):
            for right in range(left + 1, len(layers)):
                if feature_sets[left].intersection(feature_sets[right]) and cell_sets[
                    left
                ].intersection(cell_sets[right]):
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
        child = 0
        for layer in self.layers:
            selected = np.flatnonzero(
                (layer.cellIndices >= start) & (layer.cellIndices < stop)
            )
            child += sum(
                layer.source.estimate_read_memory(
                    int(position), int(position) + 1
                ).peakBytes
                for position in selected
            )
        output = (stop - start) * self.n_features * self.dtype.itemsize
        return MemoryEstimate(self.resident_bytes, child, output)

    def read_cells(self, start: int, stop: int) -> csr_matrix:
        start, stop = self._window(start, stop)
        estimate = self.estimate_read_memory(start, stop)
        self._admit(estimate)
        data_parts: list[NDArray[Any]] = []
        row_parts: list[NDArray[np.int64]] = []
        column_parts: list[NDArray[np.int64]] = []
        for layer in self.layers:
            local_cells: NDArray[np.int64] = np.flatnonzero(
                (layer.cellIndices >= start) & (layer.cellIndices < stop)
            ).astype(np.int64, copy=False)
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
        return coo_matrix(
            (
                np.concatenate(data_parts),
                (np.concatenate(row_parts), np.concatenate(column_parts)),
            ),
            shape=(stop - start, self.n_features),
            dtype=self.dtype,
        ).tocsr()


LayerStitchSource = LayerStitchMatrixSource


class RenamedMatrixSource(BaseMatrixSource):
    def __init__(
        self,
        source: MatrixSource,
        *,
        row_names: Sequence[str | bytes] | NDArray[Any] | None = None,
        column_names: Sequence[str | bytes] | NDArray[Any] | None = None,
        limits: SourceLimits = DEFAULT_LIMITS,
    ) -> None:
        self.source = source
        super().__init__(
            source.shape,
            source.dtype,
            row_names=source.row_names if row_names is None else row_names,
            column_names=(
                source.column_names if column_names is None else column_names
            ),
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
        output = (stop - start) * self.n_features * self.dtype.itemsize
        return MemoryEstimate(self.resident_bytes, child.peakBytes, output)

    def read_cells(self, start: int, stop: int) -> MatrixBlock:
        start, stop = self._window(start, stop)
        estimate = self.estimate_read_memory(start, stop)
        self._admit(estimate)
        block = self.source.read_cells(start, stop)
        if isinstance(block, spmatrix):
            return cast(MatrixBlock, block.astype(self.dtype, copy=False))
        return np.asarray(block).astype(self.dtype, copy=False)


RenameMatrixSource = RenamedMatrixSource
CastMatrixSource = DtypeMatrixSource
