import os
import re
import shutil
import sqlite3
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, overload

import numpy as np
from numpy.typing import NDArray

from ._rds import (
    R_INT_NA,
    AltRepValue,
    LazyAtomicVector,
    LazyStringVector,
    PairValue,
    RdsClosedError,
    RdsDocument,
    RdsLimits,
    RNode,
    RType,
    get_attribute,
    get_slot,
    iter_attributes,
    iter_named,
    open_rds,
)
from ._seurat import (
    CellBindMatrixSource,
    DEFAULT_LIMITS,
    FragmentSource,
    LayerPlacement,
    LayerStitchMatrixSource,
    MatrixBlock,
    MatrixSource,
    MatrixSourceError,
    MemoryEstimate,
    SourceLimits,
    UnsupportedMatrixOperation,
    fragment_source_from_slots,
    matrix_source_from_slots,
)


_VECTOR_BLOCK_SIZE = 65_536
_SCALAR_MATRIX_SLOTS = frozenset(
    {
        "asSparse",
        "as_sparse",
        "dataset",
        "dir",
        "directory",
        "dtype",
        "along",
        "buffer_size",
        "compressed",
        "filepath",
        "function",
        "group",
        "layer",
        "name",
        "mode",
        "op",
        "operation",
        "path",
        "right",
        "sparseLayout",
        "sparse_layout",
        "threads",
        "tile_width",
        "transpose",
        "type",
        "version",
        "fillValue",
        "fill_value",
        "invert",
        "keepNonzero",
        "keep_nonzero",
    }
)
_MATRIX_PARAMETER_SLOTS = frozenset(
    {"Rvalue", "active_transforms", "col_params", "row_params"}
)
_SAFE_DELAYED_PRIMITIVES = frozenset(
    {
        "!",
        "!=",
        "%%",
        "%/%",
        "&",
        "*",
        "+",
        "-",
        "/",
        "<",
        "<=",
        "==",
        ">",
        ">=",
        "^",
        "abs",
        "acos",
        "acosh",
        "asin",
        "asinh",
        "atan",
        "atanh",
        "ceiling",
        "cos",
        "cosh",
        "exp",
        "expm1",
        "floor",
        "identity",
        "is.finite",
        "is.infinite",
        "is.na",
        "is.nan",
        "log",
        "log10",
        "log1p",
        "log2",
        "pmax",
        "pmin",
        "round",
        "sign",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
        "trunc",
        "|",
    }
)
_ROOT_IMPORTED_SLOTS = frozenset(
    {"active.assay", "active.ident", "assays", "class", "meta.data", "reductions"}
)
_ROOT_IGNORED_SLOTS = frozenset(
    {
        "commands",
        "graphs",
        "images",
        "misc",
        "neighbors",
        "tools",
    }
)


class SeuratImportError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        object_path: str,
        code: str,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.objectPath = object_path
        self.object_path = object_path
        self.code = code
        self.context = dict(context or {})
        super().__init__(f"{message} at {object_path} [{code}]")


@dataclass(frozen=True, slots=True)
class SeuratDiagnostic:
    code: str
    message: str
    objectPath: str
    context: dict[str, Any]

    @classmethod
    def from_error(cls, error: SeuratImportError) -> "SeuratDiagnostic":
        return cls(
            code=error.code,
            message=error.message,
            objectPath=error.objectPath,
            context=dict(error.context),
        )


@dataclass(frozen=True, slots=True)
class SeuratNotice:
    code: str
    message: str
    objectPath: str
    context: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SeuratItemInspection:
    name: str
    importable: bool
    sourceClass: str | None
    dimensions: tuple[int, ...] | None
    objectPath: str
    dtype: str | None = None
    backend: str | None = None
    memoryEstimate: MemoryEstimate | None = None
    blockingDiagnostic: SeuratDiagnostic | None = None
    notices: tuple[SeuratNotice, ...] = ()


@dataclass(frozen=True, slots=True)
class SeuratInspectResult:
    source: str
    sourceDigest: str
    payloadDigest: str
    compression: str
    activeAssay: str
    nCells: int
    assays: tuple[SeuratItemInspection, ...]
    reductions: tuple[SeuratItemInspection, ...]
    cellMetadata: SeuratItemInspection
    activeIdentity: SeuratItemInspection
    notices: tuple[SeuratNotice, ...] = ()

    def assay(self, name: str) -> SeuratItemInspection:
        for item in self.assays:
            if item.name == name:
                return item
        raise KeyError(name)

    def reduction(self, name: str) -> SeuratItemInspection:
        for item in self.reductions:
            if item.name == name:
                return item
        raise KeyError(name)


@dataclass(frozen=True, slots=True)
class SeuratColumnBlock:
    values: NDArray[Any] | tuple[str | bytes | None, ...]
    missing: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class _CachedLayerSpec:
    assay: str
    layer: str
    path: str
    sourceClass: str
    package: str
    loader: str
    objectPath: str


def _parse_cache_loader_list(value: str, *, object_path: str) -> tuple[str, ...]:
    result: list[str] = []
    index = 0
    while index < len(value):
        while index < len(value) and value[index].isspace():
            index += 1
        if index >= len(value) or value[index] not in {"'", '"'}:
            raise _error(
                "SaveSeuratRds composite loader contains an invalid string list",
                object_path=object_path,
                code="unsupported_sidecar_cache_recipe",
            )
        quote = value[index]
        index += 1
        characters: list[str] = []
        while index < len(value) and value[index] != quote:
            character = value[index]
            if character == "\\":
                index += 1
                if index >= len(value) or value[index] not in {quote, "\\"}:
                    raise _error(
                        "SaveSeuratRds composite loader uses an unsupported escape",
                        object_path=object_path,
                        code="unsupported_sidecar_cache_recipe",
                    )
                character = value[index]
            characters.append(character)
            index += 1
        if index >= len(value):
            raise _error(
                "SaveSeuratRds composite loader has an unterminated string",
                object_path=object_path,
                code="unsupported_sidecar_cache_recipe",
            )
        index += 1
        result.append("".join(characters))
        while index < len(value) and value[index].isspace():
            index += 1
        if index == len(value):
            break
        if value[index] != ",":
            raise _error(
                "SaveSeuratRds composite loader string list is malformed",
                object_path=object_path,
                code="unsupported_sidecar_cache_recipe",
            )
        index += 1
    if not result:
        raise _error(
            "SaveSeuratRds composite loader has no leaf recipes",
            object_path=object_path,
            code="unsupported_sidecar_cache_recipe",
        )
    return tuple(result)


class SeuratStringVector(Sequence[str]):
    def __init__(
        self,
        values: LazyStringVector,
        document: RdsDocument,
        *,
        object_path: str,
    ) -> None:
        self._values = values
        self._document = document
        self.objectPath = object_path

    @property
    def shape(self) -> tuple[int]:
        return (len(self),)

    def __len__(self) -> int:
        return len(self._values)

    def _ensure_open(self) -> None:
        if self._document.closed:
            raise RdsClosedError("RDS document is closed", path=self.objectPath)

    def read_block(self, start: int, stop: int) -> tuple[str, ...]:
        self._ensure_open()
        if start < 0 or stop < start or stop > len(self):
            raise IndexError(
                f"identifier window [{start}, {stop}) is outside [0, {len(self)})"
            )
        return tuple(
            _as_text(value, object_path=f"{self.objectPath}/{start + offset}")
            for offset, value in enumerate(self._values.read_block(start, stop))
        )

    def iter_blocks(
        self, block_size: int = _VECTOR_BLOCK_SIZE
    ) -> Iterator[tuple[str, ...]]:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        for start in range(0, len(self), block_size):
            yield self.read_block(start, min(len(self), start + block_size))

    @overload
    def __getitem__(self, key: int) -> str: ...

    @overload
    def __getitem__(self, key: slice) -> tuple[str, ...]: ...

    def __getitem__(self, key: int | slice) -> str | tuple[str, ...]:
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            if step == 1:
                return self.read_block(start, stop)
            return tuple(self[index] for index in range(start, stop, step))
        index = key + len(self) if key < 0 else key
        if index < 0 or index >= len(self):
            raise IndexError("identifier index out of range")
        return self.read_block(index, index + 1)[0]

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Sequence)
            and not isinstance(other, str | bytes)
            and _identifiers_equal(self, other)
        )


class _IndexedStringVector(Sequence[str]):
    def __init__(
        self,
        values: Sequence[str],
        indexes: NDArray[np.int64],
        *,
        object_path: str,
    ) -> None:
        self._values = values
        self._indexes = indexes
        self.objectPath = object_path

    def __len__(self) -> int:
        return int(self._indexes.size)

    def read_block(self, start: int, stop: int) -> tuple[str, ...]:
        if start < 0 or stop < start or stop > len(self):
            raise IndexError(
                f"identifier window [{start}, {stop}) is outside [0, {len(self)})"
            )
        return tuple(self._values[int(index)] for index in self._indexes[start:stop])

    @overload
    def __getitem__(self, key: int) -> str: ...

    @overload
    def __getitem__(self, key: slice) -> tuple[str, ...]: ...

    def __getitem__(self, key: int | slice) -> str | tuple[str, ...]:
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            if step == 1:
                return self.read_block(start, stop)
            return tuple(self[index] for index in range(start, stop, step))
        index = key + len(self) if key < 0 else key
        if index < 0 or index >= len(self):
            raise IndexError("identifier index out of range")
        return self._values[int(self._indexes[index])]

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Sequence)
            and not isinstance(other, str | bytes)
            and _identifiers_equal(self, other)
        )


class SeuratMembership:
    def __init__(
        self,
        length: int,
        positions: NDArray[np.int64] | None = None,
    ) -> None:
        self._length = int(length)
        if self._length < 0:
            raise ValueError("membership length cannot be negative")
        if positions is None:
            self._positions = None
            return
        normalized = np.asarray(positions, dtype=np.int64)
        if normalized.ndim != 1:
            raise ValueError("membership positions must be one-dimensional")
        if normalized.size and (
            int(normalized.min()) < 0 or int(normalized.max()) >= self._length
        ):
            raise ValueError("membership position is out of range")
        if normalized.size > 1 and np.any(normalized[1:] <= normalized[:-1]):
            normalized = np.sort(normalized)
            if np.any(normalized[1:] == normalized[:-1]):
                raise ValueError("membership positions contain duplicates")
        if normalized.size == self._length:
            self._positions = None
        else:
            self._positions = normalized

    @property
    def allIncluded(self) -> bool:
        return self._positions is None

    def __len__(self) -> int:
        return self._length

    def read_block(self, start: int, stop: int) -> NDArray[np.bool_]:
        if start < 0 or stop < start or stop > len(self):
            raise IndexError(
                f"membership window [{start}, {stop}) is outside [0, {len(self)})"
            )
        if self._positions is None:
            return np.ones(stop - start, dtype=np.bool_)
        result = np.zeros(stop - start, dtype=np.bool_)
        left = int(np.searchsorted(self._positions, start, side="left"))
        right = int(np.searchsorted(self._positions, stop, side="left"))
        result[self._positions[left:right] - start] = True
        return result

    @overload
    def __getitem__(self, key: int) -> bool: ...

    @overload
    def __getitem__(self, key: slice) -> NDArray[np.bool_]: ...

    def __getitem__(self, key: int | slice) -> bool | NDArray[np.bool_]:
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            block = self.read_block(start, stop)
            return block if step == 1 else block[::step]
        index = key + len(self) if key < 0 else key
        if index < 0 or index >= len(self):
            raise IndexError("membership index out of range")
        return bool(self.read_block(index, index + 1)[0])

    def __array__(
        self,
        dtype: Any | None = None,
        copy: bool | None = None,
    ) -> NDArray[Any]:
        values = self.read_block(0, len(self))
        if copy:
            values = values.copy()
        return values.astype(dtype, copy=False) if dtype is not None else values


class SeuratMetadataColumn:
    def __init__(
        self,
        *,
        name: str,
        kind: str,
        values: LazyAtomicVector | LazyStringVector,
        length: int,
        document: RdsDocument,
        source_indices: NDArray[np.int64] | None = None,
        levels: tuple[str, ...] = (),
        ordered: bool = False,
        object_path: str,
    ) -> None:
        self.name = name
        self.kind = kind
        self.length = length
        self.levels = levels
        self.ordered = ordered
        self.objectPath = object_path
        self._values = values
        self._document = document
        self._sourceIndices = source_indices

    @property
    def sourceIndices(self) -> NDArray[np.int64] | None:
        return self._sourceIndices

    def _ensure_open(self) -> None:
        if self._document.closed:
            raise RdsClosedError("RDS document is closed", path=self.objectPath)

    def _window(self, start: int, stop: int) -> tuple[int, int]:
        if isinstance(start, bool) or isinstance(stop, bool):
            raise TypeError("metadata bounds must be integers")
        start = int(start)
        stop = int(stop)
        if start < 0 or stop < start or stop > self.length:
            raise IndexError(
                f"metadata window [{start}, {stop}) is outside [0, {self.length})"
            )
        return start, stop

    def _read_atomic(self, start: int, stop: int) -> NDArray[Any]:
        values = self._values
        if not isinstance(values, LazyAtomicVector):
            raise TypeError(f"{self.objectPath} is not an atomic column")
        if self._sourceIndices is None:
            return values.read_block(start, stop)
        indexes = self._sourceIndices[start:stop]
        output = np.empty(indexes.size, dtype=values.dtype)
        run_start = 0
        for position in range(1, indexes.size + 1):
            finished = (
                position == indexes.size
                or indexes[position] != indexes[position - 1] + 1
            )
            if not finished:
                continue
            source_start = int(indexes[run_start])
            source_stop = int(indexes[position - 1]) + 1
            output[run_start:position] = values.read_block(source_start, source_stop)
            run_start = position
        return output

    def _read_strings(self, start: int, stop: int) -> tuple[str | bytes | None, ...]:
        values = self._values
        if not isinstance(values, LazyStringVector):
            raise TypeError(f"{self.objectPath} is not a character column")
        if self._sourceIndices is None:
            return tuple(values.read_block(start, stop))
        return tuple(values[int(index)] for index in self._sourceIndices[start:stop])

    def read_block(self, start: int, stop: int) -> SeuratColumnBlock:
        self._ensure_open()
        start, stop = self._window(start, stop)
        if self.kind == "character":
            string_values = self._read_strings(start, stop)
            string_missing = np.fromiter(
                (value is None for value in string_values),
                dtype=np.bool_,
                count=len(string_values),
            )
            return SeuratColumnBlock(string_values, string_missing)

        atomic_values = self._read_atomic(start, stop)
        if self.kind in {"integer", "factor"}:
            atomic_missing = atomic_values == R_INT_NA
        elif self.kind == "logical":
            atomic_missing = atomic_values == R_INT_NA
            atomic_values = atomic_values == 1
        elif self.kind == "real":
            atomic_missing = np.isnan(atomic_values)
        else:
            raise AssertionError(f"unknown metadata kind {self.kind!r}")
        return SeuratColumnBlock(
            atomic_values,
            np.asarray(atomic_missing, dtype=np.bool_),
        )

    def read_decoded_block(
        self, start: int, stop: int
    ) -> tuple[str | bytes | int | float | bool | None, ...]:
        block = self.read_block(start, stop)
        if self.kind == "factor":
            numeric = np.asarray(block.values)
            return tuple(
                None if block.missing[index] else self.levels[int(value) - 1]
                for index, value in enumerate(numeric)
            )
        if isinstance(block.values, tuple):
            return block.values
        return tuple(
            None if block.missing[index] else value.item()
            for index, value in enumerate(block.values)
        )


@dataclass(frozen=True, slots=True)
class SeuratMetadata:
    rowIds: Sequence[str]
    columns: tuple[SeuratMetadataColumn, ...]
    objectPath: str

    @property
    def columnNames(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    def column(self, name: str) -> SeuratMetadataColumn:
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(name)


class SeuratNumericVector:
    def __init__(
        self,
        values: LazyAtomicVector,
        document: RdsDocument,
        *,
        object_path: str,
    ) -> None:
        self._values = values
        self._document = document
        self.objectPath = object_path

    @property
    def length(self) -> int:
        return len(self._values)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._values.dtype

    def __len__(self) -> int:
        return self.length

    def read_block(self, start: int, stop: int) -> NDArray[Any]:
        if self._document.closed:
            raise RdsClosedError("RDS document is closed", path=self.objectPath)
        return self._values.read_block(start, stop)


class SeuratRMatrix:
    def __init__(
        self,
        values: LazyAtomicVector,
        shape: tuple[int, int],
        *,
        row_ids: Sequence[str],
        column_ids: Sequence[str],
        document: RdsDocument,
        object_path: str,
    ) -> None:
        if len(values) != shape[0] * shape[1]:
            raise SeuratImportError(
                f"matrix has {len(values)} values but shape {shape} needs "
                f"{shape[0] * shape[1]}",
                object_path=object_path,
                code="matrix_length_mismatch",
            )
        self._values = values
        self.shape = shape
        self.rowIds = row_ids
        self.columnIds = column_ids
        self._document = document
        self.objectPath = object_path

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._values.dtype

    def read_rows(self, start: int, stop: int) -> NDArray[Any]:
        if self._document.closed:
            raise RdsClosedError("RDS document is closed", path=self.objectPath)
        if isinstance(start, bool) or isinstance(stop, bool):
            raise TypeError("matrix bounds must be integers")
        start = int(start)
        stop = int(stop)
        if start < 0 or stop < start or stop > self.shape[0]:
            raise IndexError(
                f"matrix window [{start}, {stop}) is outside [0, {self.shape[0]})"
            )
        output = np.empty((stop - start, self.shape[1]), dtype=self.dtype)
        for column in range(self.shape[1]):
            offset = column * self.shape[0]
            output[:, column] = self._values.read_block(offset + start, offset + stop)
        return output

    def read_cells(self, start: int, stop: int) -> NDArray[Any]:
        return self.read_rows(start, stop)


class _OwnedMatrixSource:
    def __init__(
        self,
        source: MatrixSource,
        document: RdsDocument,
        *,
        object_path: str,
    ) -> None:
        self._source = source
        self._document = document
        self.objectPath = object_path

    def _ensure_open(self) -> None:
        if self._document.closed:
            raise RdsClosedError("RDS document is closed", path=self.objectPath)

    @property
    def shape(self) -> tuple[int, int]:
        self._ensure_open()
        return self._source.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        self._ensure_open()
        return self._source.dtype

    @property
    def row_names(self) -> tuple[str, ...] | None:
        self._ensure_open()
        return self._source.row_names

    @property
    def column_names(self) -> tuple[str, ...] | None:
        self._ensure_open()
        return self._source.column_names

    @property
    def is_sparse(self) -> bool:
        self._ensure_open()
        return self._source.is_sparse

    @property
    def zero_preserving(self) -> bool:
        self._ensure_open()
        return self._source.zero_preserving

    @property
    def resident_bytes(self) -> int:
        self._ensure_open()
        return self._source.resident_bytes

    def read_cells(self, start: int, stop: int) -> MatrixBlock:
        self._ensure_open()
        return self._source.read_cells(start, stop)

    def estimate_read_memory(self, start: int, stop: int) -> MemoryEstimate:
        self._ensure_open()
        return self._source.estimate_read_memory(start, stop)


@dataclass(frozen=True, slots=True)
class SeuratAssay:
    name: str
    sourceClass: str
    counts: MatrixSource
    featureIds: Sequence[str]
    cellIds: Sequence[str]
    assayCellIds: Sequence[str]
    cellMembership: SeuratMembership
    featureMetadata: SeuratMetadata
    notices: tuple[SeuratNotice, ...]
    objectPath: str

    @property
    def matrix(self) -> MatrixSource:
        return self.counts

    @property
    def dimensions(self) -> tuple[int, int]:
        return self.counts.shape


@dataclass(frozen=True, slots=True)
class SeuratReduction:
    name: str
    sourceClass: str
    role: str
    cellEmbeddings: SeuratRMatrix
    featureLoadings: SeuratRMatrix | None
    stdev: SeuratNumericVector | None
    assayUsed: str
    key: str
    globalReduction: bool
    imported: bool
    computedByScarf: bool
    notices: tuple[SeuratNotice, ...]
    objectPath: str

    @property
    def dimensions(self) -> tuple[int, int]:
        return self.cellEmbeddings.shape


@dataclass(frozen=True, slots=True)
class _LogMap:
    rowIds: Sequence[str]
    layerNames: Sequence[str]
    values: LazyAtomicVector
    objectPath: str
    maximumIndexBytes: int

    def membership(self, layer_name: str) -> NDArray[np.int64]:
        try:
            column = self.layerNames.index(layer_name)
        except ValueError as error:
            raise SeuratImportError(
                f"LogMap has no column for layer {layer_name!r}",
                object_path=self.objectPath,
                code="logmap_layer_missing",
                context={"layer": layer_name},
            ) from error
        required_bytes = len(self.rowIds) * (
            np.dtype(np.int32).itemsize + np.dtype(np.int64).itemsize
        )
        if required_bytes > self.maximumIndexBytes:
            raise SeuratImportError(
                "LogMap membership exceeds its memory budget",
                object_path=f"{self.objectPath}/{layer_name}",
                code="metadata_index_limit",
                context={
                    "requiredBytes": required_bytes,
                    "maximumBytes": self.maximumIndexBytes,
                },
            )
        start = column * len(self.rowIds)
        values = self.values.read_block(start, start + len(self.rowIds))
        invalid = (values != 0) & (values != 1)
        if np.any(invalid):
            offset = int(np.flatnonzero(invalid)[0])
            raise SeuratImportError(
                "LogMap membership must contain only TRUE or FALSE",
                object_path=f"{self.objectPath}/{layer_name}",
                code="invalid_logmap_membership",
                context={"row": offset, "value": int(values[offset])},
            )
        return np.flatnonzero(values == 1).astype(np.int64, copy=False)


def _error(
    message: str,
    *,
    object_path: str,
    code: str,
    **context: Any,
) -> SeuratImportError:
    return SeuratImportError(
        message,
        object_path=object_path,
        code=code,
        context=context,
    )


def _as_text(value: str | bytes | None, *, object_path: str) -> str:
    if value is None:
        raise _error(
            "identifier is missing",
            object_path=object_path,
            code="missing_id",
        )
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise _error(
                "identifier is not valid UTF-8",
                object_path=object_path,
                code="invalid_id_encoding",
            ) from error
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise _error(
            "identifier is not valid Unicode",
            object_path=object_path,
            code="invalid_id_encoding",
        ) from error
    if not value:
        raise _error(
            "identifier is empty",
            object_path=object_path,
            code="missing_id",
        )
    if "\x00" in value:
        raise _error(
            "identifier contains a NUL character",
            object_path=object_path,
            code="invalid_id",
        )
    return value


def _string_values(node: RNode, *, object_path: str) -> LazyStringVector:
    if node.type is not RType.STRING or not isinstance(node.value, LazyStringVector):
        raise _error(
            "expected a character vector",
            object_path=object_path,
            code="invalid_character_vector",
            rType=node.type.name,
        )
    return node.value


def _read_text_vector(
    node: RNode,
    *,
    object_path: str,
    maximum_bytes: int = DEFAULT_LIMITS.maxMetadataBytes,
) -> tuple[str, ...]:
    values = _string_values(node, object_path=object_path)
    result: list[str] = []
    used_bytes = 0
    for start in range(0, len(values), _VECTOR_BLOCK_SIZE):
        block = values.read_block(start, min(len(values), start + _VECTOR_BLOCK_SIZE))
        for offset, value in enumerate(block):
            text = _as_text(value, object_path=f"{object_path}/{start + offset}")
            used_bytes += len(text.encode("utf-8")) + 8
            if used_bytes > maximum_bytes:
                raise _error(
                    "text vector exceeds its metadata memory budget",
                    object_path=object_path,
                    code="metadata_index_limit",
                    requiredBytes=used_bytes,
                    maximumBytes=maximum_bytes,
                )
            result.append(text)
    return tuple(result)


def _identifier_block(
    values: Sequence[str],
    start: int,
    stop: int,
    *,
    object_path: str,
) -> tuple[str, ...]:
    read_block = getattr(values, "read_block", None)
    raw = read_block(start, stop) if callable(read_block) else values[start:stop]
    if len(raw) != stop - start:
        raise _error(
            "identifier source returned an invalid block length",
            object_path=object_path,
            code="invalid_id_source",
            expected=stop - start,
            actual=len(raw),
        )
    return tuple(
        _as_text(value, object_path=f"{object_path}/{start + offset}")
        for offset, value in enumerate(raw)
    )


def _identifiers_equal(left: Sequence[str], right: Sequence[str]) -> bool:
    if len(left) != len(right):
        return False
    for start in range(0, len(left), _VECTOR_BLOCK_SIZE):
        stop = min(len(left), start + _VECTOR_BLOCK_SIZE)
        if _identifier_block(
            left,
            start,
            stop,
            object_path="$left_ids",
        ) != _identifier_block(
            right,
            start,
            stop,
            object_path="$right_ids",
        ):
            return False
    return True


def _identifier_database(
    *,
    scratch_dir: str | os.PathLike[str] | None,
    maximum_bytes: int,
    object_path: str,
) -> tuple[sqlite3.Connection, str]:
    if maximum_bytes < 8_192:
        raise _error(
            "identifier index budget is too small",
            object_path=object_path,
            code="metadata_index_limit",
            maximumBytes=maximum_bytes,
        )
    file = tempfile.NamedTemporaryFile(
        prefix="scarf-seurat-ids-",
        suffix=".sqlite3",
        dir=scratch_dir,
        delete=False,
    )
    path = file.name
    file.close()
    try:
        free_bytes = shutil.disk_usage(os.path.dirname(path)).free
        allowed_bytes = min(maximum_bytes, free_bytes)
        if allowed_bytes < 8_192:
            raise _error(
                "identifier index has insufficient scratch space",
                object_path=object_path,
                code="metadata_index_limit",
                maximumBytes=maximum_bytes,
                freeBytes=free_bytes,
            )
        connection = sqlite3.connect(path)
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        page_size = int(connection.execute("PRAGMA page_size").fetchone()[0])
        connection.execute(
            f"PRAGMA max_page_count={max(2, allowed_bytes // page_size)}"
        )
        connection.execute(
            "CREATE TABLE ids "
            "(value TEXT PRIMARY KEY, position INTEGER NOT NULL, matched INTEGER NOT NULL) "
            "WITHOUT ROWID"
        )
        return connection, path
    except Exception:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        raise


def _close_identifier_database(connection: sqlite3.Connection, path: str) -> None:
    connection.close()
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def _populate_identifier_database(
    connection: sqlite3.Connection,
    values: Sequence[str],
    *,
    object_path: str,
) -> None:
    try:
        for start in range(0, len(values), _VECTOR_BLOCK_SIZE):
            stop = min(len(values), start + _VECTOR_BLOCK_SIZE)
            block = _identifier_block(
                values,
                start,
                stop,
                object_path=object_path,
            )
            connection.executemany(
                "INSERT INTO ids(value, position, matched) VALUES (?, ?, 0)",
                ((value, start + offset) for offset, value in enumerate(block)),
            )
            connection.commit()
    except sqlite3.IntegrityError as error:
        raise _error(
            "identifiers are duplicated",
            object_path=object_path,
            code="duplicate_id",
        ) from error
    except sqlite3.OperationalError as error:
        raise _error(
            "identifier index exceeds its disk budget",
            object_path=object_path,
            code="metadata_index_limit",
        ) from error


def _validate_unique_ids(
    values: Sequence[str],
    *,
    object_path: str,
    scratch_dir: str | os.PathLike[str] | None,
    maximum_bytes: int,
) -> None:
    if not values:
        return
    connection, path = _identifier_database(
        scratch_dir=scratch_dir,
        maximum_bytes=maximum_bytes,
        object_path=object_path,
    )
    try:
        _populate_identifier_database(
            connection,
            values,
            object_path=object_path,
        )
    finally:
        _close_identifier_database(connection, path)


def _validate_unique(values: tuple[str, ...], *, object_path: str) -> None:
    seen: set[str] = set()
    for index, value in enumerate(values):
        if value in seen:
            raise _error(
                f"identifier {value!r} is duplicated",
                object_path=f"{object_path}/{index}",
                code="duplicate_id",
                id=value,
            )
        seen.add(value)


def _class_names(node: RNode, *, object_path: str) -> tuple[str, ...]:
    class_node = get_attribute(node, "class")
    if class_node is None:
        return ()
    names = _read_text_vector(class_node, object_path=f"{object_path}/class")
    if not names:
        raise _error(
            "class vector is empty",
            object_path=f"{object_path}/class",
            code="empty_class",
        )
    return names


def _require_slot(node: RNode, name: str, *, object_path: str) -> RNode:
    value = get_slot(node, name)
    if value is None:
        raise _error(
            f"required slot {name!r} is missing",
            object_path=f"{object_path}/{name}",
            code="missing_slot",
            slot=name,
        )
    return value


def _assay_storage_kind(
    node: RNode,
    classes: tuple[str, ...],
    *,
    object_path: str,
) -> str:
    if "Assay5T" in classes:
        raise _error(
            "transposed Assay5 storage is not supported",
            object_path=object_path,
            code="unsupported_assay_class",
            classNames=classes,
        )
    has_assay5_slots = all(
        get_slot(node, name) is not None for name in ("layers", "cells", "features")
    )
    if "Assay5" in classes or has_assay5_slots:
        return "assay5"

    has_layers = get_slot(node, "layers") is not None
    has_legacy_slots = all(
        get_slot(node, name) is not None for name in ("counts", "meta.features")
    )
    if "Assay" in classes or (has_legacy_slots and not has_layers):
        return "legacy"

    raise _error(
        "assay does not expose legacy Assay or Assay5 capabilities",
        object_path=object_path,
        code="unsupported_assay_class",
        classNames=classes,
    )


def _named_nodes(node: RNode, *, object_path: str) -> tuple[tuple[str, RNode], ...]:
    try:
        raw = tuple(iter_named(node))
    except (TypeError, ValueError) as error:
        if (
            node.type is RType.VECTOR
            and isinstance(node.value, tuple)
            and not node.value
            and get_attribute(node, "names") is None
        ):
            return ()
        raise _error(
            "expected a named list",
            object_path=object_path,
            code="invalid_named_list",
            cause=str(error),
        ) from error
    result: list[tuple[str, RNode]] = []
    names: list[str] = []
    for index, (name, value) in enumerate(raw):
        normalized = _as_text(name, object_path=f"{object_path}/names/{index}")
        result.append((normalized, value))
        names.append(normalized)
    _validate_unique(tuple(names), object_path=f"{object_path}/names")
    return tuple(result)


def _read_integer_vector(
    node: RNode,
    *,
    object_path: str,
    expected_length: int | None = None,
) -> NDArray[np.int64]:
    if node.type not in {RType.INTEGER, RType.LOGICAL} or not isinstance(
        node.value, LazyAtomicVector
    ):
        raise _error(
            "expected an integer vector",
            object_path=object_path,
            code="invalid_integer_vector",
            rType=node.type.name,
        )
    if expected_length is not None and len(node.value) != expected_length:
        raise _error(
            f"integer vector has length {len(node.value)}; expected {expected_length}",
            object_path=object_path,
            code="length_mismatch",
            actual=len(node.value),
            expected=expected_length,
        )
    return node.value.read_block(0, len(node.value)).astype(np.int64, copy=False)


def _read_text_scalar(node: RNode, *, object_path: str) -> str:
    values = _string_values(node, object_path=object_path)
    if len(values) != 1:
        raise _error(
            f"expected one string, found {len(values)}",
            object_path=object_path,
            code="invalid_scalar",
            actualLength=len(values),
        )
    return _as_text(values[0], object_path=object_path)


def _read_logical_scalar(node: RNode, *, object_path: str) -> bool:
    if node.type is not RType.LOGICAL or not isinstance(node.value, LazyAtomicVector):
        raise _error(
            "expected one logical value",
            object_path=object_path,
            code="invalid_logical_scalar",
            rType=node.type.name,
        )
    if len(node.value) != 1:
        raise _error(
            f"expected one logical value, found {len(node.value)}",
            object_path=object_path,
            code="invalid_logical_scalar",
            actualLength=len(node.value),
        )
    value = int(node.value[0])
    if value not in {0, 1}:
        raise _error(
            "logical scalar cannot be missing",
            object_path=object_path,
            code="invalid_logical_scalar",
            value=value,
        )
    return bool(value)


def _node_has_content(node: RNode) -> bool:
    if node.is_null:
        return False
    if isinstance(node.value, LazyAtomicVector | LazyStringVector | tuple):
        return len(node.value) > 0
    return True


def _matrix_dimensions(node: RNode, *, object_path: str) -> tuple[int, int]:
    dim = get_attribute(node, "Dim") or get_attribute(node, "dim")
    if dim is None:
        raise _error(
            "matrix dimensions are missing",
            object_path=f"{object_path}/dim",
            code="missing_matrix_dimensions",
        )
    values = _read_integer_vector(
        dim,
        object_path=f"{object_path}/dim",
        expected_length=2,
    )
    shape = (int(values[0]), int(values[1]))
    if shape[0] < 0 or shape[1] < 0:
        raise _error(
            f"matrix dimensions cannot be negative: {shape}",
            object_path=f"{object_path}/dim",
            code="invalid_matrix_dimensions",
            dimensions=shape,
        )
    return shape


def _matrix_dimnames(
    node: RNode,
    *,
    document: RdsDocument,
    object_path: str,
    require_both: bool,
) -> tuple[Sequence[str] | None, Sequence[str] | None]:
    dimnames = get_attribute(node, "Dimnames") or get_attribute(node, "dimnames")
    if dimnames is None:
        if require_both:
            raise _error(
                "matrix Dimnames are missing",
                object_path=f"{object_path}/Dimnames",
                code="missing_dimnames",
            )
        return None, None
    if dimnames.type is not RType.VECTOR or not isinstance(dimnames.value, tuple):
        raise _error(
            "matrix Dimnames must contain two axes",
            object_path=f"{object_path}/Dimnames",
            code="invalid_dimnames",
        )
    if len(dimnames.value) != 2:
        raise _error(
            f"matrix Dimnames has {len(dimnames.value)} axes; expected 2",
            object_path=f"{object_path}/Dimnames",
            code="invalid_dimnames",
            actualLength=len(dimnames.value),
        )
    axes: list[Sequence[str] | None] = []
    for index, axis in enumerate(dimnames.value):
        axis_path = f"{object_path}/Dimnames/{index}"
        if axis.is_null:
            if require_both:
                raise _error(
                    "matrix axis identifiers are missing",
                    object_path=axis_path,
                    code="missing_dimnames",
                )
            axes.append(None)
            continue
        axes.append(
            SeuratStringVector(
                _string_values(axis, object_path=axis_path),
                document,
                object_path=axis_path,
            )
        )
    return axes[0], axes[1]


def _unwrap_atomic(node: RNode, *, object_path: str) -> LazyAtomicVector:
    if node.type in {
        RType.LOGICAL,
        RType.INTEGER,
        RType.REAL,
        RType.COMPLEX,
        RType.RAW,
    } and isinstance(node.value, LazyAtomicVector):
        return node.value
    if node.type is RType.ALTREP and isinstance(node.value, AltRepValue):
        altrep = node.value
        if altrep.known and altrep.class_name in {
            "wrap_complex",
            "wrap_integer",
            "wrap_logical",
            "wrap_raw",
            "wrap_real",
        }:
            state = altrep.state
            if state.type is RType.PAIRLIST and isinstance(state.value, PairValue):
                return _unwrap_atomic(state.value.car, object_path=object_path)
        raise _error(
            f"ALTREP class {altrep.class_name!r} is not usable as an atomic matrix",
            object_path=object_path,
            code="unsupported_altrep",
            className=altrep.class_name,
            packageName=altrep.package_name,
        )
    raise _error(
        "expected an atomic vector",
        object_path=object_path,
        code="invalid_atomic_vector",
        rType=node.type.name,
    )


def _matrix_slot_value(
    node: RNode,
    *,
    name: str,
    object_path: str,
    reader: "SeuratReader",
) -> Any:
    if node.is_null:
        return None
    if node.type in {
        RType.LOGICAL,
        RType.INTEGER,
        RType.REAL,
        RType.COMPLEX,
        RType.RAW,
    }:
        atomic_values = _unwrap_atomic(node, object_path=object_path)
        if name in _MATRIX_PARAMETER_SLOTS:
            dimensions = get_attribute(node, "dim")
            if dimensions is not None:
                raw_shape = _unwrap_atomic(
                    dimensions,
                    object_path=f"{object_path}/dim",
                )
                shape_values = raw_shape.read_block(0, len(raw_shape))
                if (
                    shape_values.ndim != 1
                    or shape_values.size != 2
                    or np.any(shape_values < 0)
                ):
                    raise _error(
                        "matrix parameter has invalid dimensions",
                        object_path=f"{object_path}/dim",
                        code="invalid_matrix_parameter",
                    )
                shape = (int(shape_values[0]), int(shape_values[1]))
                if shape[0] * shape[1] != len(atomic_values):
                    raise _error(
                        "matrix parameter dimensions do not match its values",
                        object_path=object_path,
                        code="invalid_matrix_parameter",
                    )
                if atomic_values.nbytes > reader._maximumIndexBytes:
                    raise _error(
                        "matrix parameter exceeds its metadata memory budget",
                        object_path=object_path,
                        code="metadata_index_limit",
                        requiredBytes=atomic_values.nbytes,
                        maximumBytes=reader._maximumIndexBytes,
                    )
                return atomic_values.read_block(0, len(atomic_values)).reshape(
                    shape,
                    order="F",
                )
        if name in _SCALAR_MATRIX_SLOTS and len(atomic_values) == 1:
            return atomic_values[0]
        return atomic_values
    if node.type is RType.ALTREP:
        return _unwrap_atomic(node, object_path=object_path)
    if node.type is RType.STRING:
        string_values = _string_values(node, object_path=object_path)
        if name in _SCALAR_MATRIX_SLOTS and len(string_values) == 1:
            return _as_text(string_values[0], object_path=object_path)
        return string_values
    if node.type is RType.CHAR:
        if isinstance(node.value, str | bytes):
            return _as_text(node.value, object_path=object_path)
        return None
    if node.type in {RType.BUILTIN, RType.SPECIAL}:
        function_name = node.value
        if isinstance(function_name, str) and function_name in _SAFE_DELAYED_PRIMITIVES:
            return function_name
        raise _error(
            "matrix operation contains an unsupported R primitive",
            object_path=object_path,
            code="unsupported_matrix_function",
            functionName=function_name,
        )
    if node.type in {
        RType.CLOSURE,
        RType.PROMISE,
        RType.BYTECODE,
        RType.BYTECODE_DEFINITION,
        RType.BYTECODE_REFERENCE,
    }:
        raise _error(
            "matrix operation contains executable R semantics",
            object_path=object_path,
            code="unsupported_matrix_function",
            nodeType=node.type.name,
        )
    if node.type is RType.VECTOR and isinstance(node.value, tuple):
        if name in {"Dimnames", "dimnames"}:
            axes: list[LazyStringVector | None] = []
            for index, axis in enumerate(node.value):
                if axis.is_null:
                    axes.append(None)
                else:
                    axes.append(
                        _string_values(
                            axis,
                            object_path=f"{object_path}/{index}",
                        )
                    )
            return tuple(axes)
        if name == "sources":
            return tuple(
                reader._matrix_source_from_node(
                    child,
                    object_path=f"{object_path}/{index}",
                )
                for index, child in enumerate(node.value)
            )
        if name == "fragments_list":
            return tuple(
                reader._fragment_source_from_node(
                    child,
                    object_path=f"{object_path}/{index}",
                )
                for index, child in enumerate(node.value)
            )
        names_node = get_attribute(node, "names")
        if names_node is not None:
            return {
                child_name: _matrix_slot_value(
                    child,
                    name=child_name,
                    object_path=f"{object_path}/{child_name}",
                    reader=reader,
                )
                for child_name, child in _named_nodes(node, object_path=object_path)
            }
        return tuple(
            _matrix_slot_value(
                child,
                name=name,
                object_path=f"{object_path}/{index}",
                reader=reader,
            )
            for index, child in enumerate(node.value)
        )
    if node.type is RType.S4:
        if name == "fragments":
            return reader._fragment_source_from_node(
                node,
                object_path=object_path,
            )
        return reader._matrix_source_from_node(node, object_path=object_path)
    raise _error(
        f"matrix slot uses unsupported R type {node.type.name}",
        object_path=object_path,
        code="unsupported_matrix_slot",
        rType=node.type.name,
    )


def _frame_row_ids(
    node: RNode,
    *,
    document: RdsDocument,
    object_path: str,
    expected_length: int,
    compact_ids: Sequence[str] | None,
) -> Sequence[str]:
    row_names = get_attribute(node, "row.names")
    if row_names is None:
        raise _error(
            "data frame row.names are missing",
            object_path=f"{object_path}/row.names",
            code="missing_row_names",
        )
    if row_names.type is RType.STRING:
        result = SeuratStringVector(
            _string_values(
                row_names,
                object_path=f"{object_path}/row.names",
            ),
            document,
            object_path=f"{object_path}/row.names",
        )
        if len(result) != expected_length:
            raise _error(
                f"data frame has {len(result)} row names; expected {expected_length}",
                object_path=f"{object_path}/row.names",
                code="length_mismatch",
                actual=len(result),
                expected=expected_length,
            )
        return result
    if row_names.type is RType.INTEGER:
        compact = _read_integer_vector(
            row_names,
            object_path=f"{object_path}/row.names",
        )
        if (
            compact_ids is not None
            and compact.shape == (2,)
            and int(compact[0]) == R_INT_NA
            and abs(int(compact[1])) == expected_length
        ):
            return compact_ids
    raise _error(
        "data frame row.names do not carry explicit identifiers",
        object_path=f"{object_path}/row.names",
        code="invalid_row_names",
    )


def _alignment(
    source_ids: Sequence[str],
    target_ids: Sequence[str],
    *,
    object_path: str,
    scratch_dir: str | os.PathLike[str] | None,
    maximum_bytes: int,
) -> NDArray[np.int64] | None:
    if _identifiers_equal(source_ids, target_ids):
        return None
    if len(source_ids) * np.dtype(np.int64).itemsize > maximum_bytes:
        raise _error(
            "identifier alignment exceeds its memory budget",
            object_path=object_path,
            code="metadata_index_limit",
            requiredBytes=len(source_ids) * np.dtype(np.int64).itemsize,
            maximumBytes=maximum_bytes,
        )
    connection, path = _identifier_database(
        scratch_dir=scratch_dir,
        maximum_bytes=maximum_bytes,
        object_path=object_path,
    )
    try:
        _populate_identifier_database(
            connection,
            source_ids,
            object_path=object_path,
        )
        connection.execute(
            "CREATE TABLE target "
            "(position INTEGER PRIMARY KEY, value TEXT NOT NULL UNIQUE)"
        )
        for start in range(0, len(target_ids), _VECTOR_BLOCK_SIZE):
            stop = min(len(target_ids), start + _VECTOR_BLOCK_SIZE)
            block = _identifier_block(
                target_ids,
                start,
                stop,
                object_path=object_path,
            )
            connection.executemany(
                "INSERT INTO target(position, value) VALUES (?, ?)",
                ((start + offset, value) for offset, value in enumerate(block)),
            )
            connection.commit()
        missing = tuple(
            str(row[0])
            for row in connection.execute(
                "SELECT target.value FROM target "
                "LEFT JOIN ids ON ids.value = target.value "
                "WHERE ids.value IS NULL LIMIT 5"
            )
        )
        extra = tuple(
            str(row[0])
            for row in connection.execute(
                "SELECT ids.value FROM ids "
                "LEFT JOIN target ON target.value = ids.value "
                "WHERE target.value IS NULL LIMIT 5"
            )
        )
        if missing or extra or len(source_ids) != len(target_ids):
            raise _error(
                "metadata row identifiers conflict with the target axis",
                object_path=object_path,
                code="metadata_id_conflict",
                missing=tuple(missing),
                extra=extra,
            )
        result = np.empty(len(target_ids), dtype=np.int64)
        cursor = connection.execute(
            "SELECT ids.position FROM target "
            "JOIN ids ON ids.value = target.value "
            "ORDER BY target.position"
        )
        start = 0
        while rows := cursor.fetchmany(_VECTOR_BLOCK_SIZE):
            stop = start + len(rows)
            result[start:stop] = np.fromiter(
                (int(row[0]) for row in rows),
                dtype=np.int64,
                count=len(rows),
            )
            start = stop
        if start != len(target_ids):
            raise RuntimeError("identifier alignment row count changed")
        return result
    except sqlite3.IntegrityError as error:
        raise _error(
            "target identifiers are duplicated",
            object_path=object_path,
            code="duplicate_id",
        ) from error
    except sqlite3.OperationalError as error:
        raise _error(
            "identifier alignment exceeds its disk budget",
            object_path=object_path,
            code="metadata_index_limit",
        ) from error
    finally:
        _close_identifier_database(connection, path)


def _positions_in_target(
    source_ids: Sequence[str],
    target_ids: Sequence[str],
    *,
    object_path: str,
    scratch_dir: str | os.PathLike[str] | None,
    maximum_bytes: int,
    missing_message: str = "Assay5 LogMap contains cells absent from global metadata",
    missing_code: str = "assay_cell_id_conflict",
) -> NDArray[np.int64]:
    required_bytes = len(source_ids) * np.dtype(np.int64).itemsize
    if required_bytes > maximum_bytes:
        raise _error(
            "identifier mapping exceeds its memory budget",
            object_path=object_path,
            code="metadata_index_limit",
            requiredBytes=required_bytes,
            maximumBytes=maximum_bytes,
        )
    if _identifiers_equal(source_ids, target_ids):
        return np.arange(len(source_ids), dtype=np.int64)
    connection, path = _identifier_database(
        scratch_dir=scratch_dir,
        maximum_bytes=maximum_bytes,
        object_path=object_path,
    )
    try:
        _populate_identifier_database(
            connection,
            target_ids,
            object_path=object_path,
        )
        connection.execute(
            "CREATE TABLE requested "
            "(position INTEGER PRIMARY KEY, value TEXT NOT NULL UNIQUE)"
        )
        for start in range(0, len(source_ids), _VECTOR_BLOCK_SIZE):
            stop = min(len(source_ids), start + _VECTOR_BLOCK_SIZE)
            block = _identifier_block(
                source_ids,
                start,
                stop,
                object_path=object_path,
            )
            connection.executemany(
                "INSERT INTO requested(position, value) VALUES (?, ?)",
                ((start + offset, value) for offset, value in enumerate(block)),
            )
            connection.commit()
        missing = tuple(
            str(row[0])
            for row in connection.execute(
                "SELECT requested.value FROM requested "
                "LEFT JOIN ids ON ids.value = requested.value "
                "WHERE ids.value IS NULL LIMIT 5"
            )
        )
        if missing:
            raise _error(
                missing_message,
                object_path=object_path,
                code=missing_code,
                missing=missing,
            )
        result = np.empty(len(source_ids), dtype=np.int64)
        cursor = connection.execute(
            "SELECT ids.position FROM requested "
            "JOIN ids ON ids.value = requested.value "
            "ORDER BY requested.position"
        )
        start = 0
        while rows := cursor.fetchmany(_VECTOR_BLOCK_SIZE):
            stop = start + len(rows)
            result[start:stop] = np.fromiter(
                (int(row[0]) for row in rows),
                dtype=np.int64,
                count=len(rows),
            )
            start = stop
        if start != len(source_ids):
            raise RuntimeError("identifier mapping row count changed")
        return result
    except sqlite3.IntegrityError as error:
        raise _error(
            "source identifiers are duplicated",
            object_path=object_path,
            code="duplicate_id",
        ) from error
    except sqlite3.OperationalError as error:
        raise _error(
            "identifier mapping exceeds its disk budget",
            object_path=object_path,
            code="metadata_index_limit",
        ) from error
    finally:
        _close_identifier_database(connection, path)


def _validate_column_values(
    values: LazyAtomicVector,
    *,
    kind: str,
    levels: tuple[str, ...],
    object_path: str,
) -> None:
    if kind not in {"logical", "factor"}:
        return
    for start in range(0, len(values), _VECTOR_BLOCK_SIZE):
        block = values.read_block(start, min(len(values), start + _VECTOR_BLOCK_SIZE))
        if kind == "logical":
            invalid = (block != 0) & (block != 1) & (block != R_INT_NA)
        else:
            invalid = (block != R_INT_NA) & ((block < 1) | (block > len(levels)))
        if np.any(invalid):
            index = int(np.flatnonzero(invalid)[0])
            raise _error(
                f"{kind} column contains an invalid encoded value",
                object_path=f"{object_path}/{start + index}",
                code=f"invalid_{kind}_value",
                value=int(block[index]),
            )


def _metadata_column(
    node: RNode,
    *,
    name: str,
    length: int,
    alignment: NDArray[np.int64] | None,
    document: RdsDocument,
    object_path: str,
    maximum_metadata_bytes: int,
) -> SeuratMetadataColumn:
    classes = _class_names(node, object_path=object_path)
    levels: tuple[str, ...] = ()
    ordered = "ordered" in classes
    if "factor" in classes:
        if node.type is not RType.INTEGER or not isinstance(
            node.value, LazyAtomicVector
        ):
            raise _error(
                "factor column must use integer codes",
                object_path=object_path,
                code="invalid_factor",
                rType=node.type.name,
            )
        levels_node = get_attribute(node, "levels")
        if levels_node is None:
            raise _error(
                "factor levels are missing",
                object_path=f"{object_path}/levels",
                code="missing_factor_levels",
            )
        levels = _read_text_vector(
            levels_node,
            object_path=f"{object_path}/levels",
            maximum_bytes=maximum_metadata_bytes,
        )
        _validate_unique(levels, object_path=f"{object_path}/levels")
        kind = "factor"
        values: LazyAtomicVector | LazyStringVector = node.value
    elif node.type is RType.LOGICAL and isinstance(node.value, LazyAtomicVector):
        kind = "logical"
        values = node.value
    elif node.type is RType.INTEGER and isinstance(node.value, LazyAtomicVector):
        kind = "integer"
        values = node.value
    elif node.type is RType.REAL and isinstance(node.value, LazyAtomicVector):
        kind = "real"
        values = node.value
    elif node.type is RType.STRING and isinstance(node.value, LazyStringVector):
        kind = "character"
        values = node.value
    else:
        raise _error(
            f"metadata column uses unsupported R type {node.type.name}",
            object_path=object_path,
            code="unsupported_metadata_type",
            rType=node.type.name,
            classNames=classes,
        )
    if len(values) != length:
        raise _error(
            f"metadata column has length {len(values)}; expected {length}",
            object_path=object_path,
            code="length_mismatch",
            actual=len(values),
            expected=length,
        )
    if isinstance(values, LazyAtomicVector):
        _validate_column_values(
            values,
            kind=kind,
            levels=levels,
            object_path=object_path,
        )
    return SeuratMetadataColumn(
        name=name,
        kind=kind,
        values=values,
        length=length,
        document=document,
        source_indices=alignment,
        levels=levels,
        ordered=ordered,
        object_path=object_path,
    )


def _data_frame(
    node: RNode,
    *,
    target_ids: Sequence[str],
    document: RdsDocument,
    object_path: str,
    allow_compact: bool,
    scratch_dir: str | os.PathLike[str] | None,
    maximum_index_bytes: int,
) -> SeuratMetadata:
    classes = _class_names(node, object_path=object_path)
    if "data.frame" not in classes:
        raise _error(
            "metadata slot is not a data.frame",
            object_path=object_path,
            code="invalid_data_frame",
            classNames=classes,
        )
    if node.type is not RType.VECTOR or not isinstance(node.value, tuple):
        raise _error(
            "data.frame columns are not stored as a generic vector",
            object_path=object_path,
            code="invalid_data_frame",
            rType=node.type.name,
        )
    source_ids = _frame_row_ids(
        node,
        document=document,
        object_path=object_path,
        expected_length=len(target_ids),
        compact_ids=target_ids if allow_compact else None,
    )
    index = _alignment(
        source_ids,
        target_ids,
        object_path=f"{object_path}/row.names",
        scratch_dir=scratch_dir,
        maximum_bytes=maximum_index_bytes,
    )
    named_columns = _named_nodes(node, object_path=object_path)
    columns = tuple(
        _metadata_column(
            column,
            name=name,
            length=len(target_ids),
            alignment=index,
            document=document,
            object_path=f"{object_path}/{name}",
            maximum_metadata_bytes=maximum_index_bytes,
        )
        for name, column in named_columns
    )
    return SeuratMetadata(
        rowIds=target_ids,
        columns=columns,
        objectPath=object_path,
    )


class SeuratReader:
    def __init__(
        self,
        source: Any,
        *,
        limits: RdsLimits | None = None,
        temp_dir: str | os.PathLike[str] | None = None,
        assays: Sequence[str] | None = None,
        assay_layers: Mapping[str, Sequence[str]] | None = None,
        reductions: Sequence[str] | None = None,
        sidecar_path_remaps: Mapping[str | os.PathLike[str], str | os.PathLike[str]]
        | None = None,
        matrix_limits: SourceLimits = DEFAULT_LIMITS,
    ) -> None:
        self.source = source
        self._sidecarPathRemaps = sidecar_path_remaps
        self._matrixLimits = matrix_limits
        self._scratchDir = temp_dir
        self._maximumIndexBytes = matrix_limits.maxMetadataBytes
        self._requestedAssayLayers = assay_layers
        self._rdsPath = (
            os.fspath(source) if isinstance(source, str | os.PathLike) else None
        )
        self._document = open_rds(source, limits=limits, temp_dir=temp_dir)
        self._assayModels: dict[str, SeuratAssay] = {}
        self._assayErrors: dict[str, SeuratImportError] = {}
        self._reductionModels: dict[str, SeuratReduction] = {}
        self._reductionErrors: dict[str, SeuratImportError] = {}
        try:
            self._initialize(assays=assays, reductions=reductions)
        except Exception:
            self._document.close()
            raise

    @property
    def closed(self) -> bool:
        return self._document.closed

    @property
    def document(self) -> RdsDocument:
        self._ensure_open()
        return self._document

    @property
    def tempPaths(self) -> tuple[str, ...]:
        return self._document.temp_paths

    @property
    def inspection(self) -> SeuratInspectResult:
        self._ensure_open()
        return self._inspection

    @property
    def cellMetadata(self) -> SeuratMetadata:
        self._ensure_open()
        return self._cellMetadata

    @property
    def activeIdentity(self) -> SeuratMetadataColumn:
        self._ensure_open()
        if self._activeIdentityError is not None:
            raise self._activeIdentityError
        if self._activeIdentity is None:
            raise AssertionError("active identity status is inconsistent")
        return self._activeIdentity

    @property
    def assayNames(self) -> tuple[str, ...]:
        return self._selectedAssays

    @property
    def reductionNames(self) -> tuple[str, ...]:
        return self._selectedReductions

    @property
    def assays(self) -> tuple[SeuratAssay, ...]:
        return tuple(self.get_assay(name) for name in self._selectedAssays)

    @property
    def reductions(self) -> tuple[SeuratReduction, ...]:
        return tuple(self.get_reduction(name) for name in self._selectedReductions)

    def _ensure_open(self) -> None:
        if self._document.closed:
            raise RdsClosedError("RDS document is closed", path="$")

    def close(self) -> None:
        self._document.close()

    def __enter__(self) -> "SeuratReader":
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        self.close()

    def inspect(self) -> SeuratInspectResult:
        return self.inspection

    def get_assay(self, name: str | None = None) -> SeuratAssay:
        self._ensure_open()
        selected = self.activeAssay if name is None else name
        if selected not in self._selectedAssays:
            raise _error(
                f"assay {selected!r} was not selected for inspection",
                object_path=f"assays/{selected}",
                code="assay_not_selected",
                assay=selected,
            )
        error = self._assayErrors.get(selected)
        if error is not None:
            raise error
        try:
            return self._assayModels[selected]
        except KeyError as error:
            raise AssertionError("assay inspection status is inconsistent") from error

    def get_reduction(self, name: str) -> SeuratReduction:
        self._ensure_open()
        if name not in self._selectedReductions:
            raise _error(
                f"reduction {name!r} was not selected for inspection",
                object_path=f"reductions/{name}",
                code="reduction_not_selected",
                reduction=name,
            )
        error = self._reductionErrors.get(name)
        if error is not None:
            raise error
        try:
            return self._reductionModels[name]
        except KeyError as error:
            raise AssertionError(
                "reduction inspection status is inconsistent"
            ) from error

    def _load_save_seurat_cache(self, root: RNode) -> None:
        self._cachedLayers: dict[str, tuple[_CachedLayerSpec, ...]] = {}
        self._saveCachePresent = False
        self._saveCacheError: SeuratImportError | None = None
        tools = get_slot(root, "tools")
        if tools is None or not _node_has_content(tools):
            return
        try:
            tool_nodes = _named_nodes(tools, object_path="tools")
        except SeuratImportError:
            return
        cache_node = next(
            (node for name, node in tool_nodes if name == "SaveSeuratRds"),
            None,
        )
        if cache_node is None:
            return
        self._saveCachePresent = True
        try:
            classes = _class_names(
                cache_node,
                object_path="tools/SaveSeuratRds",
            )
            if "data.frame" not in classes:
                raise _error(
                    "SaveSeuratRds layer cache is not a data.frame",
                    object_path="tools/SaveSeuratRds",
                    code="invalid_sidecar_cache",
                    classNames=classes,
                )
            columns = dict(
                _named_nodes(
                    cache_node,
                    object_path="tools/SaveSeuratRds",
                )
            )
            required = ("layer", "path", "class", "pkg", "fxn", "assay")
            missing = tuple(name for name in required if name not in columns)
            if missing:
                raise _error(
                    "SaveSeuratRds layer cache is missing required columns",
                    object_path="tools/SaveSeuratRds",
                    code="invalid_sidecar_cache",
                    missingColumns=missing,
                )
            decoded: dict[str, tuple[str, ...]] = {}
            remaining_bytes = self._maximumIndexBytes
            for name in required:
                values = _read_text_vector(
                    columns[name],
                    object_path=f"tools/SaveSeuratRds/{name}",
                    maximum_bytes=remaining_bytes,
                )
                decoded[name] = values
                remaining_bytes -= sum(
                    len(value.encode("utf-8")) + 8 for value in values
                )
            lengths = {len(values) for values in decoded.values()}
            if len(lengths) != 1:
                raise _error(
                    "SaveSeuratRds layer cache columns have different lengths",
                    object_path="tools/SaveSeuratRds",
                    code="invalid_sidecar_cache",
                    columnLengths={
                        name: len(values) for name, values in decoded.items()
                    },
                )
            cached: dict[str, list[_CachedLayerSpec]] = {}
            seen: set[tuple[str, str]] = set()
            for index in range(lengths.pop()):
                assay = decoded["assay"][index]
                layer = decoded["layer"][index]
                key = (assay, layer)
                if key in seen:
                    raise _error(
                        "SaveSeuratRds layer cache contains a duplicate layer",
                        object_path=f"tools/SaveSeuratRds/{index}",
                        code="invalid_sidecar_cache",
                        assay=assay,
                        layer=layer,
                    )
                seen.add(key)
                if assay not in self._assayNodes:
                    continue
                cached.setdefault(assay, []).append(
                    _CachedLayerSpec(
                        assay=assay,
                        layer=layer,
                        path=decoded["path"][index],
                        sourceClass=decoded["class"][index],
                        package=decoded["pkg"][index],
                        loader=decoded["fxn"][index],
                        objectPath=f"tools/SaveSeuratRds/{index}",
                    )
                )
            self._cachedLayers = {
                assay: tuple(values) for assay, values in cached.items()
            }
        except SeuratImportError as error:
            self._saveCacheError = error

    def _cached_layer_source(self, cached: _CachedLayerSpec) -> MatrixSource:
        loader = " ".join(cached.loader.split())
        composite = re.fullmatch(
            r"function\(x\) \{ paths <- unlist\(x = strsplit"
            r"\(x = x, split = ','\)\); fxns <- list\(\s*(?P<loaders>.*)\s*\); "
            r"mats <- vector\(mode = 'list', length = length\(x = paths\)\); "
            r"for \(i in seq_along\(paths\)\) \{ fn <- "
            r"eval\(str2lang\(fxns\[\[i\]\]\)\); mats\[\[i\]\] <- "
            r"fn\(paths\[i\]\); \}; return\(Reduce\(cbind, mats\)\); \}",
            loader,
        )
        if composite is not None:
            if cached.package != "BPCells":
                raise _error(
                    "SaveSeuratRds composite loader has an unexpected package",
                    object_path=f"{cached.objectPath}/pkg",
                    code="unsupported_sidecar_cache_recipe",
                    package=cached.package,
                )
            loaders = _parse_cache_loader_list(
                composite.group("loaders"),
                object_path=f"{cached.objectPath}/fxn",
            )
            paths = tuple(path.strip() for path in cached.path.split(","))
            if len(paths) != len(loaders) or any(not path for path in paths):
                raise _error(
                    "SaveSeuratRds composite loader paths do not match its recipes",
                    object_path=f"{cached.objectPath}/path",
                    code="invalid_sidecar_cache",
                    pathCount=len(paths),
                    loaderCount=len(loaders),
                )
            sources = tuple(
                self._cached_layer_source(
                    _CachedLayerSpec(
                        assay=cached.assay,
                        layer=cached.layer,
                        path=path,
                        sourceClass=cached.sourceClass,
                        package=cached.package,
                        loader=nested_loader,
                        objectPath=f"{cached.objectPath}/fxn/{index}",
                    )
                )
                for index, (path, nested_loader) in enumerate(zip(paths, loaders))
            )
            try:
                return CellBindMatrixSource(sources, limits=self._matrixLimits)
            except Exception as cause:
                raise self._wrapped_item_error(
                    cause,
                    object_path=f"assays/{cached.assay}/layers/{cached.layer}",
                    code="invalid_matrix",
                ) from cause
        specification: dict[str, Any]
        if cached.package == "BPCells" and re.fullmatch(
            r"function\(x\) BPCells::open_matrix_dir\(dir = x\)",
            loader,
        ):
            specification = {"class": "MatrixDir", "dir": cached.path}
        elif cached.package == "BPCells" and (
            match := re.fullmatch(
                r"function\(x\) BPCells::open_matrix_hdf5"
                r"\(path = x, group = (?P<quote>['\"])(?P<group>[^'\"]+)"
                r"(?P=quote)\)",
                loader,
            )
        ):
            specification = {
                "class": "MatrixH5",
                "filepath": cached.path,
                "group": match.group("group"),
            }
        elif cached.package == "BPCells" and (
            match := re.fullmatch(
                r"function\(x\) BPCells::open_matrix_anndata_hdf5"
                r"\(path = x, group = (?P<quote>['\"])(?P<group>[^'\"]+)"
                r"(?P=quote)\)",
                loader,
            )
        ):
            specification = {
                "class": "AnnDataMatrixH5",
                "filepath": cached.path,
                "group": match.group("group"),
            }
        elif cached.package == "HDF5Array" and (
            match := re.fullmatch(
                r"function\(x\) HDF5Array::HDF5Array"
                r"\(filepath = x, name = (?P<quote>['\"])(?P<name>[^'\"]+)"
                r"(?P=quote), as\.sparse = (?P<sparse>TRUE|FALSE)\)",
                loader,
            )
        ):
            specification = {
                "class": "HDF5ArraySeed",
                "filepath": cached.path,
                "name": match.group("name"),
                "asSparse": match.group("sparse") == "TRUE",
            }
        elif cached.package == "HDF5Array" and (
            match := re.fullmatch(
                r"function\(x\) HDF5Array::H5ADMatrix"
                r"\(filepath = x(?:, layer = (?P<quote>['\"])"
                r"(?P<layer>[^'\"]+)(?P=quote))?\s*\)",
                loader,
            )
        ):
            specification = {
                "class": "H5ADMatrixSeed",
                "filepath": cached.path,
            }
            if match.group("layer") is not None:
                specification["layer"] = match.group("layer")
        else:
            raise _error(
                "SaveSeuratRds loader recipe is not in the safe built-in profile",
                object_path=f"{cached.objectPath}/fxn",
                code="unsupported_sidecar_cache_recipe",
                assay=cached.assay,
                layer=cached.layer,
                sourceClass=cached.sourceClass,
                package=cached.package,
            )
        try:
            return matrix_source_from_slots(
                specification,
                object_path=f"assays/{cached.assay}/layers/{cached.layer}",
                rds_path=self._rdsPath,
                absolute_prefix_remaps=self._sidecarPathRemaps,
                limits=self._matrixLimits,
            )
        except Exception as cause:
            raise self._wrapped_item_error(
                cause,
                object_path=f"assays/{cached.assay}/layers/{cached.layer}",
                code=(
                    "unsupported_matrix"
                    if isinstance(cause, UnsupportedMatrixOperation)
                    else "invalid_matrix"
                ),
            ) from cause

    def _initialize(
        self,
        *,
        assays: Sequence[str] | None,
        reductions: Sequence[str] | None,
    ) -> None:
        root = self._document.root
        root_classes = _class_names(root, object_path="$")
        if "Seurat" not in root_classes:
            raise _error(
                "RDS root is not a Seurat object",
                object_path="$",
                code="not_seurat",
                classNames=root_classes,
            )
        assay_list = _require_slot(root, "assays", object_path="")
        reduction_list = _require_slot(root, "reductions", object_path="")
        self._assayNodes = dict(_named_nodes(assay_list, object_path="assays"))
        self._reductionNodes = dict(
            _named_nodes(reduction_list, object_path="reductions")
        )
        self._load_save_seurat_cache(root)
        active_node = _require_slot(root, "active.assay", object_path="")
        self.activeAssay = _read_text_scalar(
            active_node,
            object_path="active.assay",
        )
        if self.activeAssay not in self._assayNodes:
            raise _error(
                f"active assay {self.activeAssay!r} does not exist",
                object_path="active.assay",
                code="active_assay_missing",
                activeAssay=self.activeAssay,
                availableAssays=tuple(self._assayNodes),
            )

        metadata_node = _require_slot(root, "meta.data", object_path="")
        if metadata_node.type is not RType.VECTOR or not isinstance(
            metadata_node.value, tuple
        ):
            raise _error(
                "cell meta.data is not a data.frame vector",
                object_path="meta.data",
                code="invalid_data_frame",
            )
        self.cellIds = _frame_row_ids(
            metadata_node,
            document=self._document,
            object_path="meta.data",
            expected_length=len(metadata_node.value[0].value)
            if metadata_node.value
            and isinstance(
                metadata_node.value[0].value,
                LazyAtomicVector | LazyStringVector,
            )
            else self._frame_length(metadata_node, object_path="meta.data"),
            compact_ids=None,
        )
        _validate_unique_ids(
            self.cellIds,
            object_path="meta.data/row.names",
            scratch_dir=self._scratchDir,
            maximum_bytes=self._maximumIndexBytes,
        )
        self._cellMetadata = _data_frame(
            metadata_node,
            target_ids=self.cellIds,
            document=self._document,
            object_path="meta.data",
            allow_compact=False,
            scratch_dir=self._scratchDir,
            maximum_index_bytes=self._maximumIndexBytes,
        )

        self._activeIdentity: SeuratMetadataColumn | None = None
        self._activeIdentityError: SeuratImportError | None = None
        try:
            self._activeIdentity = self._build_active_identity(root)
        except SeuratImportError as error:
            self._activeIdentityError = error

        self._selectedAssays = self._selected_names(
            assays,
            available=tuple(self._assayNodes),
            kind="assay",
        )
        self._assayLayers = self._layer_overrides(
            self._requestedAssayLayers,
            available=tuple(self._assayNodes),
        )
        self._selectedReductions = self._selected_names(
            reductions,
            available=tuple(self._reductionNodes),
            kind="reduction",
        )
        assay_inspections = tuple(
            self._inspect_assay(name) for name in self._selectedAssays
        )
        reduction_inspections = tuple(
            self._inspect_reduction(name) for name in self._selectedReductions
        )
        root_notices = self._root_notices(root)
        metadata_inspection = SeuratItemInspection(
            name="meta.data",
            importable=True,
            sourceClass="data.frame",
            dimensions=(len(self.cellIds), len(self._cellMetadata.columns)),
            objectPath="meta.data",
            dtype="mixed",
            backend=type(self._cellMetadata).__name__,
        )
        active_inspection = (
            SeuratItemInspection(
                name="active.ident",
                importable=False,
                sourceClass=None,
                dimensions=None,
                objectPath="active.ident",
                blockingDiagnostic=SeuratDiagnostic.from_error(
                    self._activeIdentityError
                ),
            )
            if self._activeIdentityError is not None
            else SeuratItemInspection(
                name="active.ident",
                importable=True,
                sourceClass="factor",
                dimensions=(len(self.cellIds),),
                objectPath="active.ident",
                dtype="factor",
                backend=type(self._activeIdentity).__name__,
            )
        )
        self._inspection = SeuratInspectResult(
            source=self._document.source.name,
            sourceDigest=self._document.source.source_sha256,
            payloadDigest=self._document.source.payload_sha256,
            compression=self._document.source.compression.value,
            activeAssay=self.activeAssay,
            nCells=len(self.cellIds),
            assays=assay_inspections,
            reductions=reduction_inspections,
            cellMetadata=metadata_inspection,
            activeIdentity=active_inspection,
            notices=root_notices,
        )

    @staticmethod
    def _frame_length(node: RNode, *, object_path: str) -> int:
        if node.type is not RType.VECTOR or not isinstance(node.value, tuple):
            raise _error(
                "data.frame columns are not stored as a vector",
                object_path=object_path,
                code="invalid_data_frame",
            )
        if not node.value:
            row_names = get_attribute(node, "row.names")
            if row_names is None:
                return 0
            if row_names.type is RType.STRING and isinstance(
                row_names.value, LazyStringVector
            ):
                return len(row_names.value)
            compact = _read_integer_vector(
                row_names,
                object_path=f"{object_path}/row.names",
            )
            if (
                compact.shape == (2,)
                and int(compact[0]) == R_INT_NA
                and int(compact[1]) <= 0
            ):
                return -int(compact[1])
            raise _error(
                "cannot determine data.frame row count",
                object_path=object_path,
                code="invalid_data_frame",
            )
        first = node.value[0].value
        if isinstance(first, LazyAtomicVector | LazyStringVector):
            return len(first)
        raise _error(
            "first data.frame column is not block-readable",
            object_path=f"{object_path}/0",
            code="unsupported_metadata_type",
        )

    @staticmethod
    def _selected_names(
        requested: Sequence[str] | None,
        *,
        available: tuple[str, ...],
        kind: str,
    ) -> tuple[str, ...]:
        if requested is None:
            return available
        if isinstance(requested, str | bytes):
            raise TypeError(f"{kind} selection must be a sequence of names")
        result = tuple(str(value) for value in requested)
        if len(set(result)) != len(result):
            raise ValueError(f"{kind} selection contains duplicate names")
        return result

    @staticmethod
    def _layer_overrides(
        requested: Mapping[str, Sequence[str]] | None,
        *,
        available: tuple[str, ...],
    ) -> dict[str, tuple[str, ...]]:
        if requested is None:
            return {}
        if not isinstance(requested, Mapping):
            raise TypeError("assay_layers must map assay names to layer-name sequences")
        result: dict[str, tuple[str, ...]] = {}
        for raw_assay, raw_layers in requested.items():
            assay = str(raw_assay)
            if assay not in available:
                raise ValueError(
                    f"assay layer override references unknown assay {assay!r}"
                )
            if isinstance(raw_layers, str | bytes):
                raise TypeError(
                    f"layer override for assay {assay!r} must be a sequence"
                )
            layers = tuple(str(value) for value in raw_layers)
            if not layers:
                raise ValueError(
                    f"layer override for assay {assay!r} must not be empty"
                )
            if len(set(layers)) != len(layers):
                raise ValueError(
                    f"layer override for assay {assay!r} contains duplicates"
                )
            result[assay] = layers
        return result

    def _build_active_identity(self, root: RNode) -> SeuratMetadataColumn:
        if "active.ident" in self._cellMetadata.columnNames:
            raise _error(
                "active.ident conflicts with a cell metadata column of the same name",
                object_path="active.ident",
                code="active_identity_conflict",
                column="active.ident",
            )
        node = _require_slot(root, "active.ident", object_path="")
        classes = _class_names(node, object_path="active.ident")
        if "factor" not in classes:
            raise _error(
                "active.ident must be a factor",
                object_path="active.ident",
                code="invalid_active_identity",
                classNames=classes,
            )
        names_node = get_attribute(node, "names")
        if names_node is None:
            raise _error(
                "active.ident names are missing",
                object_path="active.ident/names",
                code="missing_active_identity_names",
            )
        source_ids = SeuratStringVector(
            _string_values(
                names_node,
                object_path="active.ident/names",
            ),
            self._document,
            object_path="active.ident/names",
        )
        index = _alignment(
            source_ids,
            self.cellIds,
            object_path="active.ident/names",
            scratch_dir=self._scratchDir,
            maximum_bytes=self._maximumIndexBytes,
        )
        return _metadata_column(
            node,
            name="active.ident",
            length=len(self.cellIds),
            alignment=index,
            document=self._document,
            object_path="active.ident",
            maximum_metadata_bytes=self._maximumIndexBytes,
        )

    def _root_notices(self, root: RNode) -> tuple[SeuratNotice, ...]:
        notices: list[SeuratNotice] = []
        for name, node in iter_attributes(root):
            if name in _ROOT_IMPORTED_SLOTS or not _node_has_content(node):
                continue
            if name == "tools" and self._saveCachePresent:
                notices.append(
                    SeuratNotice(
                        code="used_save_seurat_rds_cache",
                        message="recognized SaveSeuratRds sidecar cache was inspected",
                        objectPath="tools/SaveSeuratRds",
                        context={
                            "cachedAssays": tuple(self._cachedLayers),
                            "valid": self._saveCacheError is None,
                        },
                    )
                )
                continue
            if name in _ROOT_IGNORED_SLOTS or name not in {
                "project.name",
                "version",
            }:
                notices.append(
                    SeuratNotice(
                        code="ignored_seurat_slot",
                        message=f"Seurat slot {name!r} is not imported",
                        objectPath=name,
                        context={"rType": node.type.name},
                    )
                )
        return tuple(notices)

    def _inspect_assay(self, name: str) -> SeuratItemInspection:
        object_path = f"assays/{name}"
        node = self._assayNodes.get(name)
        if node is None:
            error = _error(
                f"assay {name!r} does not exist",
                object_path=object_path,
                code="assay_not_found",
                availableAssays=tuple(self._assayNodes),
            )
            self._assayErrors[name] = error
            return SeuratItemInspection(
                name=name,
                importable=False,
                sourceClass=None,
                dimensions=None,
                objectPath=object_path,
                blockingDiagnostic=SeuratDiagnostic.from_error(error),
            )
        classes: tuple[str, ...] = ()
        blocked_error: SeuratImportError
        try:
            classes = _class_names(node, object_path=object_path)
            storage_kind = _assay_storage_kind(
                node,
                classes,
                object_path=object_path,
            )
            if storage_kind == "assay5":
                assay = self._build_assay5(name, node)
            else:
                assay = self._build_legacy_assay(name, node)
            self._assayModels[name] = assay
            source = getattr(assay.counts, "_source", assay.counts)
            estimate = assay.counts.estimate_read_memory(
                0,
                min(1, assay.dimensions[1]),
            )
            return SeuratItemInspection(
                name=name,
                importable=True,
                sourceClass=assay.sourceClass,
                dimensions=assay.dimensions,
                objectPath=object_path,
                dtype=assay.counts.dtype.str,
                backend=type(source).__name__,
                memoryEstimate=estimate,
                notices=assay.notices,
            )
        except SeuratImportError as error:
            blocked_error = error
            self._assayErrors[name] = blocked_error
        except Exception as cause:
            blocked_error = self._wrapped_item_error(
                cause,
                object_path=object_path,
                code=(
                    "unsupported_matrix"
                    if isinstance(cause, UnsupportedMatrixOperation)
                    else "invalid_assay"
                ),
            )
            self._assayErrors[name] = blocked_error
        return SeuratItemInspection(
            name=name,
            importable=False,
            sourceClass=classes[0] if classes else None,
            dimensions=None,
            objectPath=object_path,
            blockingDiagnostic=SeuratDiagnostic.from_error(blocked_error),
        )

    def _inspect_reduction(self, name: str) -> SeuratItemInspection:
        object_path = f"reductions/{name}"
        node = self._reductionNodes.get(name)
        if node is None:
            error = _error(
                f"reduction {name!r} does not exist",
                object_path=object_path,
                code="reduction_not_found",
                availableReductions=tuple(self._reductionNodes),
            )
            self._reductionErrors[name] = error
            return SeuratItemInspection(
                name=name,
                importable=False,
                sourceClass=None,
                dimensions=None,
                objectPath=object_path,
                blockingDiagnostic=SeuratDiagnostic.from_error(error),
            )
        classes: tuple[str, ...] = ()
        blocked_error: SeuratImportError
        try:
            classes = _class_names(node, object_path=object_path)
            if "DimReduc" not in classes:
                raise _error(
                    "reduction does not expose DimReduc capabilities",
                    object_path=object_path,
                    code="unsupported_reduction_class",
                    classNames=classes,
                )
            reduction = self._build_reduction(name, node)
            self._reductionModels[name] = reduction
            output_bytes = (
                min(1, reduction.dimensions[0])
                * reduction.dimensions[1]
                * reduction.cellEmbeddings.dtype.itemsize
            )
            return SeuratItemInspection(
                name=name,
                importable=True,
                sourceClass=reduction.sourceClass,
                dimensions=reduction.dimensions,
                objectPath=object_path,
                dtype=reduction.cellEmbeddings.dtype.str,
                backend=type(reduction.cellEmbeddings).__name__,
                memoryEstimate=MemoryEstimate(
                    workingBytes=output_bytes,
                    outputBytes=output_bytes,
                ),
                notices=reduction.notices,
            )
        except SeuratImportError as error:
            blocked_error = error
            self._reductionErrors[name] = blocked_error
        except Exception as cause:
            blocked_error = self._wrapped_item_error(
                cause,
                object_path=object_path,
                code="invalid_reduction",
            )
            self._reductionErrors[name] = blocked_error
        return SeuratItemInspection(
            name=name,
            importable=False,
            sourceClass=classes[0] if classes else None,
            dimensions=None,
            objectPath=object_path,
            blockingDiagnostic=SeuratDiagnostic.from_error(blocked_error),
        )

    @staticmethod
    def _wrapped_item_error(
        cause: Exception,
        *,
        object_path: str,
        code: str,
    ) -> SeuratImportError:
        source_path = getattr(cause, "objectPath", object_path)
        return _error(
            str(cause),
            object_path=str(source_path),
            code=code,
            causeType=type(cause).__name__,
        )

    def _fragment_source_from_node(
        self,
        node: RNode,
        *,
        object_path: str,
    ) -> FragmentSource:
        classes = _class_names(node, object_path=object_path)
        if node.type is not RType.S4:
            raise _error(
                f"fragment source uses unsupported R type {node.type.name}",
                object_path=object_path,
                code="unsupported_fragment_structure",
                rType=node.type.name,
                classNames=classes,
            )
        slots: dict[str, Any] = {}
        for name, value in iter_attributes(node):
            if name == "class":
                continue
            slots[name] = _matrix_slot_value(
                value,
                name=name,
                object_path=f"{object_path}/{name}",
                reader=self,
            )
        try:
            return fragment_source_from_slots(
                {"class": classes, "slots": slots},
                object_path=object_path,
                rds_path=self._rdsPath,
                absolute_prefix_remaps=self._sidecarPathRemaps,
                limits=self._matrixLimits,
            )
        except Exception as cause:
            raise self._wrapped_item_error(
                cause,
                object_path=object_path,
                code=(
                    "unsupported_matrix"
                    if isinstance(cause, UnsupportedMatrixOperation)
                    else "invalid_matrix"
                ),
            ) from cause

    def _matrix_source_from_node(
        self,
        node: RNode,
        *,
        object_path: str,
    ) -> MatrixSource:
        classes = _class_names(node, object_path=object_path)
        if node.type in {
            RType.LOGICAL,
            RType.INTEGER,
            RType.REAL,
            RType.COMPLEX,
            RType.RAW,
            RType.ALTREP,
        }:
            values = _unwrap_atomic(node, object_path=object_path)
            shape = _matrix_dimensions(node, object_path=object_path)
            specification: dict[str, Any] = {
                "class": classes or ("matrix", "array"),
                ".Data": values,
                "dim": shape,
                "Dimnames": (None, None),
            }
        elif node.type is RType.S4:
            slots: dict[str, Any] = {}
            for name, value in iter_attributes(node):
                if name in {"class", "Dimnames", "dimnames"}:
                    continue
                slots[name] = _matrix_slot_value(
                    value,
                    name=name,
                    object_path=f"{object_path}/{name}",
                    reader=self,
                )
            if classes and classes[0] in {
                "dgeMatrix",
                "lgeMatrix",
                "ngeMatrix",
                "igeMatrix",
                "denseMatrix",
            }:
                if "dim" not in slots and "Dim" in slots:
                    slots["dim"] = slots["Dim"]
            specification = {"class": classes, "slots": slots}
        else:
            raise _error(
                f"matrix uses unsupported R type {node.type.name}",
                object_path=object_path,
                code="unsupported_matrix_structure",
                rType=node.type.name,
                classNames=classes,
            )
        try:
            return matrix_source_from_slots(
                specification,
                object_path=object_path,
                rds_path=self._rdsPath,
                absolute_prefix_remaps=self._sidecarPathRemaps,
                limits=self._matrixLimits,
            )
        except SeuratImportError:
            raise
        except Exception as cause:
            raise self._wrapped_item_error(
                cause,
                object_path=object_path,
                code=(
                    "unsupported_matrix"
                    if isinstance(cause, UnsupportedMatrixOperation)
                    else "invalid_matrix"
                ),
            ) from cause

    def _build_legacy_assay(self, name: str, node: RNode) -> SeuratAssay:
        object_path = f"assays/{name}"
        override = self._assayLayers.get(name)
        if override is not None and override != ("counts",):
            raise _error(
                "legacy Assay layer override must select only 'counts'",
                object_path=f"{object_path}/counts",
                code="invalid_layer_override",
                selectedLayers=override,
            )
        counts_node = _require_slot(node, "counts", object_path=object_path)
        source = self._matrix_source_from_node(
            counts_node,
            object_path=f"{object_path}/counts",
        )
        feature_ids, cell_ids = _matrix_dimnames(
            counts_node,
            document=self._document,
            object_path=f"{object_path}/counts",
            require_both=True,
        )
        assert feature_ids is not None
        assert cell_ids is not None
        _validate_unique_ids(
            feature_ids,
            object_path=f"{object_path}/counts/Dimnames/0",
            scratch_dir=self._scratchDir,
            maximum_bytes=self._maximumIndexBytes,
        )
        source_row_conflict = source.row_names is not None and not _identifiers_equal(
            source.row_names,
            feature_ids,
        )
        source_column_conflict = (
            source.column_names is not None
            and not _identifiers_equal(source.column_names, cell_ids)
        )
        if source_row_conflict or source_column_conflict:
            raise _error(
                "matrix source identifiers conflict with serialized Dimnames",
                object_path=f"{object_path}/counts/Dimnames",
                code="matrix_id_conflict",
            )
        if not _identifiers_equal(cell_ids, self.cellIds):
            raise _error(
                "legacy assay cell identifiers conflict with global cell order",
                object_path=f"{object_path}/counts/Dimnames/1",
                code="assay_cell_id_conflict",
            )
        metadata_node = _require_slot(
            node,
            "meta.features",
            object_path=object_path,
        )
        feature_metadata = _data_frame(
            metadata_node,
            target_ids=feature_ids,
            document=self._document,
            object_path=f"{object_path}/meta.features",
            allow_compact=True,
            scratch_dir=self._scratchDir,
            maximum_index_bytes=self._maximumIndexBytes,
        )
        notices: list[SeuratNotice] = []
        for slot_name, slot in iter_attributes(node):
            if slot_name in {"data", "scale.data"} and _node_has_content(slot):
                notices.append(
                    SeuratNotice(
                        code="ignored_normalized_layer",
                        message=f"normalized slot {slot_name!r} is not imported",
                        objectPath=f"{object_path}/{slot_name}",
                        context={},
                    )
                )
            elif slot_name not in {
                "class",
                "counts",
                "meta.features",
            } and _node_has_content(slot):
                notices.append(
                    SeuratNotice(
                        code="ignored_assay_slot",
                        message=f"assay slot {slot_name!r} is not imported",
                        objectPath=f"{object_path}/{slot_name}",
                        context={"rType": slot.type.name},
                    )
                )
        owned = _OwnedMatrixSource(
            source,
            self._document,
            object_path=f"{object_path}/counts",
        )
        return SeuratAssay(
            name=name,
            sourceClass=_class_names(node, object_path=object_path)[0],
            counts=owned,
            featureIds=feature_ids,
            cellIds=self.cellIds,
            assayCellIds=cell_ids,
            cellMembership=SeuratMembership(len(self.cellIds)),
            featureMetadata=feature_metadata,
            notices=tuple(notices),
            objectPath=object_path,
        )

    def _build_logmap(self, node: RNode, *, object_path: str) -> _LogMap:
        classes = _class_names(node, object_path=object_path)
        if "LogMap" not in classes:
            raise _error(
                "membership slot is not a LogMap",
                object_path=object_path,
                code="invalid_logmap",
                classNames=classes,
            )
        if node.type is not RType.LOGICAL:
            raise _error(
                "LogMap values must be logical",
                object_path=object_path,
                code="invalid_logmap",
                rType=node.type.name,
            )
        values = _unwrap_atomic(node, object_path=object_path)
        shape = _matrix_dimensions(node, object_path=object_path)
        row_ids, layer_names = _matrix_dimnames(
            node,
            document=self._document,
            object_path=object_path,
            require_both=True,
        )
        assert row_ids is not None
        assert layer_names is not None
        _validate_unique_ids(
            row_ids,
            object_path=f"{object_path}/Dimnames/0",
            scratch_dir=self._scratchDir,
            maximum_bytes=self._maximumIndexBytes,
        )
        _validate_unique_ids(
            layer_names,
            object_path=f"{object_path}/Dimnames/1",
            scratch_dir=self._scratchDir,
            maximum_bytes=self._maximumIndexBytes,
        )
        if shape != (len(row_ids), len(layer_names)):
            raise _error(
                "LogMap dimensions conflict with Dimnames",
                object_path=f"{object_path}/dim",
                code="logmap_dimension_mismatch",
                dimensions=shape,
                rowIds=len(row_ids),
                layerNames=len(layer_names),
            )
        if len(values) != shape[0] * shape[1]:
            raise _error(
                "LogMap value count conflicts with dimensions",
                object_path=object_path,
                code="logmap_dimension_mismatch",
                values=len(values),
                dimensions=shape,
            )
        result = _LogMap(
            rowIds=row_ids,
            layerNames=layer_names,
            values=values,
            objectPath=object_path,
            maximumIndexBytes=self._maximumIndexBytes,
        )
        return result

    def _cached_axis_positions(
        self,
        source_ids: Sequence[str] | None,
        target_ids: Sequence[str],
        source_size: int,
        *,
        object_path: str,
    ) -> NDArray[np.int64]:
        if source_ids is None:
            if source_size != len(target_ids):
                raise _error(
                    "SaveSeuratRds removed the layer membership map and the "
                    "sidecar has no identifiers to reconstruct it",
                    object_path=object_path,
                    code="irrecoverable_sidecar_cache",
                    sourceSize=source_size,
                    targetSize=len(target_ids),
                )
            return np.arange(source_size, dtype=np.int64)
        if len(source_ids) != source_size:
            raise _error(
                "sidecar identifier length conflicts with its matrix dimension",
                object_path=object_path,
                code="sidecar_dimension_mismatch",
                sourceSize=source_size,
                identifierCount=len(source_ids),
            )
        return _positions_in_target(
            source_ids,
            target_ids,
            object_path=object_path,
            scratch_dir=self._scratchDir,
            maximum_bytes=self._maximumIndexBytes,
        )

    def _build_assay5(self, name: str, node: RNode) -> SeuratAssay:
        object_path = f"assays/{name}"
        layers_node = _require_slot(node, "layers", object_path=object_path)
        layer_nodes = _named_nodes(
            layers_node,
            object_path=f"{object_path}/layers",
        )
        cached_layers = self._cachedLayers.get(name, ())
        in_object_names = {layer_name for layer_name, _ in layer_nodes}
        duplicates = tuple(
            cached.layer for cached in cached_layers if cached.layer in in_object_names
        )
        if duplicates:
            raise _error(
                "SaveSeuratRds cache duplicates an in-object layer",
                object_path="tools/SaveSeuratRds",
                code="invalid_sidecar_cache",
                assay=name,
                duplicateLayers=duplicates,
            )
        layer_entries: tuple[
            tuple[str, RNode | _CachedLayerSpec],
            ...,
        ] = (
            *layer_nodes,
            *((cached.layer, cached) for cached in cached_layers),
        )
        cells = self._build_logmap(
            _require_slot(node, "cells", object_path=object_path),
            object_path=f"{object_path}/cells",
        )
        features = self._build_logmap(
            _require_slot(node, "features", object_path=object_path),
            object_path=f"{object_path}/features",
        )
        global_positions = _positions_in_target(
            cells.rowIds,
            self.cellIds,
            object_path=f"{object_path}/cells/Dimnames/0",
            scratch_dir=self._scratchDir,
            maximum_bytes=self._maximumIndexBytes,
        )
        if global_positions.size > 1 and np.any(
            global_positions[1:] <= global_positions[:-1]
        ):
            raise _error(
                "Assay5 cell identifiers do not follow global cell order",
                object_path=f"{object_path}/cells/Dimnames/0",
                code="assay_cell_order_conflict",
            )
        resident_index_bytes = int(global_positions.nbytes) + len(self.cellIds)
        available_count_layers = tuple(
            (layer_name, layer_value)
            for layer_name, layer_value in layer_entries
            if layer_name == "counts"
            or (layer_name.startswith("counts.") and len(layer_name) > len("counts."))
        )
        if not available_count_layers and self._saveCacheError is not None:
            raise self._saveCacheError
        override = self._assayLayers.get(name)
        if override is None:
            count_layers = available_count_layers
        else:
            raw_count_names = {layer_name for layer_name, _ in available_count_layers}
            invalid = tuple(
                layer_name
                for layer_name in override
                if layer_name not in raw_count_names
            )
            if invalid:
                raise _error(
                    "Assay5 layer override contains a non-count or missing layer",
                    object_path=f"{object_path}/layers",
                    code="invalid_layer_override",
                    invalidLayers=invalid,
                    availableCountLayers=tuple(
                        layer_name for layer_name, _ in available_count_layers
                    ),
                )
            selected = set(override)
            count_layers = tuple(
                (layer_name, layer_node)
                for layer_name, layer_node in available_count_layers
                if layer_name in selected
            )
        if not count_layers:
            raise _error(
                "Assay5 has no raw count-bearing layer",
                object_path=f"{object_path}/layers",
                code="counts_layer_missing",
            )
        for layer_name, layer_value in count_layers:
            if (
                not isinstance(layer_value, _CachedLayerSpec)
                and layer_name not in cells.layerNames
            ):
                raise _error(
                    f"cell LogMap has no column for layer {layer_name!r}",
                    object_path=f"{object_path}/cells",
                    code="logmap_layer_missing",
                    layer=layer_name,
                )
            if (
                not isinstance(layer_value, _CachedLayerSpec)
                and layer_name not in features.layerNames
            ):
                raise _error(
                    f"feature LogMap has no column for layer {layer_name!r}",
                    object_path=f"{object_path}/features",
                    code="logmap_layer_missing",
                    layer=layer_name,
                )
        placements: list[LayerPlacement] = []
        for layer_name, layer_value in count_layers:
            layer_path = f"{object_path}/layers/{layer_name}"
            if isinstance(layer_value, _CachedLayerSpec):
                source = self._cached_layer_source(layer_value)
                feature_indices = self._cached_axis_positions(
                    source.row_names,
                    features.rowIds,
                    source.shape[0],
                    object_path=f"{layer_path}/Dimnames/0",
                )
                assay_cell_indices = self._cached_axis_positions(
                    source.column_names,
                    cells.rowIds,
                    source.shape[1],
                    object_path=f"{layer_path}/Dimnames/1",
                )
            else:
                source = self._matrix_source_from_node(
                    layer_value,
                    object_path=layer_path,
                )
                feature_indices = features.membership(layer_name)
                assay_cell_indices = cells.membership(layer_name)
            resident_index_bytes += int(
                feature_indices.nbytes + assay_cell_indices.nbytes
            )
            if resident_index_bytes * 4 > self._maximumIndexBytes:
                raise _error(
                    "Assay5 stitching indexes exceed their memory budget",
                    object_path=f"{object_path}/layers",
                    code="metadata_index_limit",
                    requiredBytes=resident_index_bytes * 4,
                    maximumBytes=self._maximumIndexBytes,
                )
            if source.shape != (feature_indices.size, assay_cell_indices.size):
                raise _error(
                    f"layer shape {source.shape} does not match LogMap membership "
                    f"({feature_indices.size}, {assay_cell_indices.size})",
                    object_path=layer_path,
                    code="layer_membership_dimension_mismatch",
                    sourceDimensions=source.shape,
                    featureMembership=int(feature_indices.size),
                    cellMembership=int(assay_cell_indices.size),
                )
            expected_features = _IndexedStringVector(
                features.rowIds,
                feature_indices,
                object_path=f"{layer_path}/Dimnames/0",
            )
            expected_cells = _IndexedStringVector(
                cells.rowIds,
                assay_cell_indices,
                object_path=f"{layer_path}/Dimnames/1",
            )
            if source.row_names is not None and not _identifiers_equal(
                source.row_names,
                expected_features,
            ):
                raise _error(
                    "layer feature identifiers conflict with feature LogMap",
                    object_path=f"{layer_path}/Dimnames/0",
                    code="layer_feature_id_conflict",
                )
            if source.column_names is not None and not _identifiers_equal(
                source.column_names, expected_cells
            ):
                raise _error(
                    "layer cell identifiers conflict with cell LogMap",
                    object_path=f"{layer_path}/Dimnames/1",
                    code="layer_cell_id_conflict",
                )
            layer_global_positions = global_positions[assay_cell_indices]
            resident_index_bytes += int(layer_global_positions.nbytes)
            if resident_index_bytes * 4 > self._maximumIndexBytes:
                raise _error(
                    "Assay5 stitching indexes exceed their memory budget",
                    object_path=f"{object_path}/layers",
                    code="metadata_index_limit",
                    requiredBytes=resident_index_bytes * 4,
                    maximumBytes=self._maximumIndexBytes,
                )
            placements.append(
                LayerPlacement(
                    source,
                    feature_indices=feature_indices,
                    cell_indices=layer_global_positions,
                    name=layer_name,
                )
            )
        try:
            stitched = LayerStitchMatrixSource(
                placements,
                row_names=features.rowIds,
                column_names=self.cellIds,
                limits=self._matrixLimits,
            )
        except MatrixSourceError as cause:
            raise _error(
                str(cause),
                object_path=f"{object_path}/layers",
                code="layer_stitch_conflict",
                causeType=type(cause).__name__,
            ) from cause

        metadata_node = _require_slot(
            node,
            "meta.data",
            object_path=object_path,
        )
        feature_metadata = _data_frame(
            metadata_node,
            target_ids=features.rowIds,
            document=self._document,
            object_path=f"{object_path}/meta.data",
            allow_compact=True,
            scratch_dir=self._scratchDir,
            maximum_index_bytes=self._maximumIndexBytes,
        )
        notices: list[SeuratNotice] = []
        selected_count_names = {layer_name for layer_name, _ in count_layers}
        available_count_names = {layer_name for layer_name, _ in available_count_layers}
        for layer_name, layer_value in count_layers:
            if isinstance(layer_value, _CachedLayerSpec):
                notices.append(
                    SeuratNotice(
                        code="restored_sidecar_cache_layer",
                        message=(
                            f"raw count layer {layer_name!r} was reconstructed "
                            "from a safe SaveSeuratRds cache recipe"
                        ),
                        objectPath=layer_value.objectPath,
                        context={
                            "package": layer_value.package,
                            "sourceClass": layer_value.sourceClass,
                        },
                    )
                )
        for layer_name, _ in layer_entries:
            if layer_name in selected_count_names:
                continue
            normalized = (
                layer_name == "data"
                or layer_name.startswith("data.")
                or layer_name == "scale.data"
                or layer_name.startswith("scale.data.")
            )
            unselected_count = layer_name in available_count_names
            notices.append(
                SeuratNotice(
                    code=(
                        "ignored_unselected_count_layer"
                        if unselected_count
                        else (
                            "ignored_normalized_layer"
                            if normalized
                            else "ignored_non_count_layer"
                        )
                    ),
                    message=(
                        f"raw count layer {layer_name!r} was not selected"
                        if unselected_count
                        else f"layer {layer_name!r} is not a raw count-bearing layer"
                    ),
                    objectPath=f"{object_path}/layers/{layer_name}",
                    context={},
                )
            )
        for slot_name, slot in iter_attributes(node):
            if slot_name not in {
                "cells",
                "class",
                "features",
                "layers",
                "meta.data",
            } and _node_has_content(slot):
                notices.append(
                    SeuratNotice(
                        code="ignored_assay_slot",
                        message=f"assay slot {slot_name!r} is not imported",
                        objectPath=f"{object_path}/{slot_name}",
                        context={"rType": slot.type.name},
                    )
                )
        owned = _OwnedMatrixSource(
            stitched,
            self._document,
            object_path=f"{object_path}/layers",
        )
        return SeuratAssay(
            name=name,
            sourceClass=_class_names(node, object_path=object_path)[0],
            counts=owned,
            featureIds=features.rowIds,
            cellIds=self.cellIds,
            assayCellIds=cells.rowIds,
            cellMembership=SeuratMembership(
                len(self.cellIds),
                global_positions,
            ),
            featureMetadata=feature_metadata,
            notices=tuple(notices),
            objectPath=object_path,
        )

    def _r_matrix(
        self,
        node: RNode,
        *,
        object_path: str,
        require_names: bool,
    ) -> SeuratRMatrix:
        values = _unwrap_atomic(node, object_path=object_path)
        shape = _matrix_dimensions(node, object_path=object_path)
        rows, columns = _matrix_dimnames(
            node,
            document=self._document,
            object_path=object_path,
            require_both=require_names,
        )
        if rows is None:
            rows = tuple(str(index) for index in range(shape[0]))
        if columns is None:
            columns = tuple(str(index) for index in range(shape[1]))
        if len(rows) != shape[0] or len(columns) != shape[1]:
            raise _error(
                "matrix dimensions conflict with Dimnames",
                object_path=f"{object_path}/Dimnames",
                code="dimnames_length_mismatch",
                dimensions=shape,
                rowIds=len(rows),
                columnIds=len(columns),
            )
        if require_names:
            _validate_unique_ids(
                rows,
                object_path=f"{object_path}/Dimnames/0",
                scratch_dir=self._scratchDir,
                maximum_bytes=self._maximumIndexBytes,
            )
            _validate_unique_ids(
                columns,
                object_path=f"{object_path}/Dimnames/1",
                scratch_dir=self._scratchDir,
                maximum_bytes=self._maximumIndexBytes,
            )
        return SeuratRMatrix(
            values,
            shape,
            row_ids=rows,
            column_ids=columns,
            document=self._document,
            object_path=object_path,
        )

    def _assay_for_reduction(self, name: str) -> SeuratAssay:
        if name in self._assayModels:
            return self._assayModels[name]
        if name in self._assayErrors:
            raise self._assayErrors[name]
        node = self._assayNodes.get(name)
        if node is None:
            raise _error(
                f"reduction references missing assay {name!r}",
                object_path="reductions",
                code="reduction_assay_missing",
                assay=name,
            )
        classes = _class_names(node, object_path=f"assays/{name}")
        storage_kind = _assay_storage_kind(
            node,
            classes,
            object_path=f"assays/{name}",
        )
        if storage_kind == "assay5":
            assay = self._build_assay5(name, node)
        else:
            assay = self._build_legacy_assay(name, node)
        self._assayModels[name] = assay
        return assay

    def _build_reduction(self, name: str, node: RNode) -> SeuratReduction:
        object_path = f"reductions/{name}"
        embeddings = self._r_matrix(
            _require_slot(node, "cell.embeddings", object_path=object_path),
            object_path=f"{object_path}/cell.embeddings",
            require_names=True,
        )
        if not _identifiers_equal(embeddings.rowIds, self.cellIds):
            raise _error(
                "reduction cell identifiers do not exactly match global cell order",
                object_path=f"{object_path}/cell.embeddings/Dimnames/0",
                code="reduction_cell_id_conflict",
            )
        assay_used = _read_text_scalar(
            _require_slot(node, "assay.used", object_path=object_path),
            object_path=f"{object_path}/assay.used",
        )
        assay = self._assay_for_reduction(assay_used)
        key = _read_text_scalar(
            _require_slot(node, "key", object_path=object_path),
            object_path=f"{object_path}/key",
        )
        global_reduction = _read_logical_scalar(
            _require_slot(node, "global", object_path=object_path),
            object_path=f"{object_path}/global",
        )

        loadings_node = get_slot(node, "feature.loadings")
        feature_loadings: SeuratRMatrix | None = None
        if loadings_node is not None:
            loading_values = _unwrap_atomic(
                loadings_node,
                object_path=f"{object_path}/feature.loadings",
            )
            loading_dim = get_attribute(loadings_node, "Dim") or get_attribute(
                loadings_node, "dim"
            )
            loading_shape = (
                (0, 0)
                if loading_dim is None and len(loading_values) == 0
                else _matrix_dimensions(
                    loadings_node,
                    object_path=f"{object_path}/feature.loadings",
                )
            )
            if loading_shape[0] > 0 and loading_shape[1] > 0:
                feature_loadings = self._r_matrix(
                    loadings_node,
                    object_path=f"{object_path}/feature.loadings",
                    require_names=True,
                )
                if not _identifiers_equal(
                    feature_loadings.columnIds,
                    embeddings.columnIds,
                ):
                    raise _error(
                        "loading component identifiers conflict with embeddings",
                        object_path=(f"{object_path}/feature.loadings/Dimnames/1"),
                        code="loading_component_id_conflict",
                    )
                _positions_in_target(
                    feature_loadings.rowIds,
                    assay.featureIds,
                    object_path=f"{object_path}/feature.loadings/Dimnames/0",
                    scratch_dir=self._scratchDir,
                    maximum_bytes=self._maximumIndexBytes,
                    missing_message=(
                        "loading features are absent from the referenced assay"
                    ),
                    missing_code="loading_feature_id_conflict",
                )

        stdev_node = get_slot(node, "stdev")
        stdev: SeuratNumericVector | None = None
        if stdev_node is not None:
            stdev_values = _unwrap_atomic(
                stdev_node,
                object_path=f"{object_path}/stdev",
            )
            if len(stdev_values) not in {0, embeddings.shape[1]}:
                raise _error(
                    f"stdev has length {len(stdev_values)}; expected "
                    f"{embeddings.shape[1]}",
                    object_path=f"{object_path}/stdev",
                    code="stdev_length_mismatch",
                    actual=len(stdev_values),
                    expected=embeddings.shape[1],
                )
            if len(stdev_values):
                stdev = SeuratNumericVector(
                    stdev_values,
                    self._document,
                    object_path=f"{object_path}/stdev",
                )

        notices: list[SeuratNotice] = []
        for slot_name, slot in iter_attributes(node):
            if slot_name == "feature.loadings.projected" and _node_has_content(slot):
                notices.append(
                    SeuratNotice(
                        code="ignored_projected_loadings",
                        message="projected feature loadings are not imported",
                        objectPath=f"{object_path}/{slot_name}",
                        context={},
                    )
                )
            elif slot_name not in {
                "assay.used",
                "cell.embeddings",
                "class",
                "feature.loadings",
                "global",
                "key",
                "stdev",
            } and _node_has_content(slot):
                notices.append(
                    SeuratNotice(
                        code="ignored_reduction_slot",
                        message=f"reduction slot {slot_name!r} is not imported",
                        objectPath=f"{object_path}/{slot_name}",
                        context={"rType": slot.type.name},
                    )
                )
        normalized_name = name.casefold()
        role = (
            "displayEmbedding"
            if normalized_name in {"umap", "tsne", "t-sne"}
            else "graphCoordinates"
        )
        return SeuratReduction(
            name=name,
            sourceClass=_class_names(node, object_path=object_path)[0],
            role=role,
            cellEmbeddings=embeddings,
            featureLoadings=feature_loadings,
            stdev=stdev,
            assayUsed=assay_used,
            key=key,
            globalReduction=global_reduction,
            imported=True,
            computedByScarf=False,
            notices=tuple(notices),
            objectPath=object_path,
        )


def inspect_seurat(
    source: Any,
    *,
    limits: RdsLimits | None = None,
    temp_dir: str | os.PathLike[str] | None = None,
    assays: Sequence[str] | None = None,
    assay_layers: Mapping[str, Sequence[str]] | None = None,
    reductions: Sequence[str] | None = None,
    sidecar_path_remaps: Mapping[str | os.PathLike[str], str | os.PathLike[str]]
    | None = None,
    matrix_limits: SourceLimits = DEFAULT_LIMITS,
) -> SeuratInspectResult:
    with SeuratReader(
        source,
        limits=limits,
        temp_dir=temp_dir,
        assays=assays,
        assay_layers=assay_layers,
        reductions=reductions,
        sidecar_path_remaps=sidecar_path_remaps,
        matrix_limits=matrix_limits,
    ) as reader:
        return reader.inspection


__all__ = [
    "SeuratAssay",
    "SeuratColumnBlock",
    "SeuratDiagnostic",
    "SeuratImportError",
    "SeuratInspectResult",
    "SeuratItemInspection",
    "SeuratMembership",
    "SeuratMetadata",
    "SeuratMetadataColumn",
    "SeuratNotice",
    "SeuratNumericVector",
    "SeuratRMatrix",
    "SeuratReader",
    "SeuratReduction",
    "inspect_seurat",
]
