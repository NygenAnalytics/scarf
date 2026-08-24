import math
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from .bpcells import (
    _BPArrayStore,
    _DirectoryArrayStore,
    _HDF5ArrayStore,
    _StoredBP128Array,
    _require_numeric_array,
)
from .errors import (
    MatrixSourceError,
    ResourceLimitError,
    UnsafeSidecarError,
    UnsupportedMatrixOperation,
)
from .paths import SidecarPathResolver, require_filesystem_path
from .sources import (
    DEFAULT_LIMITS,
    BaseMatrixSource,
    MemoryEstimate,
    SourceLimits,
    _normalize_names,
)


_FRAGMENT_VERSION_PATTERN = re.compile(r"^(packed|unpacked)-fragments-v([12])$")
_UINT32_MAX = int(np.iinfo(np.uint32).max)
_DICT_ENTRY_BYTES = 160
_FRAGMENT_BLOCK_SAFETY_FACTOR = 4


@dataclass(frozen=True)
class FragmentCapability:
    className: str
    accepted: bool
    reason: str | None = None


_FRAGMENT_CAPABILITIES = (
    FragmentCapability("UnpackedMemFragments", True),
    FragmentCapability("PackedMemFragments", True),
    FragmentCapability("FragmentsDir", True),
    FragmentCapability("FragmentsHDF5", True),
    FragmentCapability(
        "FragmentsTsv", False, "fragment TSV execution is not implemented"
    ),
    FragmentCapability("ShiftFragments", True),
    FragmentCapability("SelectLength", True),
    FragmentCapability("ChrSelectName", True),
    FragmentCapability("ChrSelectIndex", True),
    FragmentCapability("CellSelectName", True),
    FragmentCapability("CellSelectIndex", True),
    FragmentCapability("CellMerge", True),
    FragmentCapability("ChrRename", True),
    FragmentCapability("CellRename", True),
    FragmentCapability("CellPrefix", True),
    FragmentCapability("RegionSelect", True),
    FragmentCapability("MergeFragments", True),
    FragmentCapability(
        "IterableFragments", False, "abstract fragment sources cannot execute"
    ),
)


class FragmentCapabilityRegistry:
    def __init__(self) -> None:
        self.capabilities = {
            capability.className: capability for capability in _FRAGMENT_CAPABILITIES
        }

    @property
    def acceptedClasses(self) -> tuple[str, ...]:
        return tuple(
            capability.className
            for capability in _FRAGMENT_CAPABILITIES
            if capability.accepted
        )

    def recognizes(self, class_name: str | None) -> bool:
        return class_name in self.capabilities

    def resolve(
        self,
        classes: tuple[str, ...],
        *,
        object_path: str,
    ) -> FragmentCapability:
        if not classes:
            raise UnsupportedMatrixOperation(
                object_path,
                "fragment-source",
                None,
                "fragment class is missing",
            )
        primary = classes[0]
        capability = self.capabilities.get(primary)
        if capability is None:
            raise UnsupportedMatrixOperation(
                object_path,
                "fragment-source",
                primary,
                "unknown or custom fragment class",
            )
        unsupported_bases = [
            class_name
            for class_name in classes[1:]
            if class_name != "IterableFragments"
        ]
        if unsupported_bases:
            raise UnsupportedMatrixOperation(
                object_path,
                "fragment-source",
                unsupported_bases[0],
                "unknown or custom fragment base class",
            )
        if not capability.accepted:
            raise UnsupportedMatrixOperation(
                object_path,
                "fragment-source",
                primary,
                capability.reason,
            )
        return capability


FRAGMENT_CAPABILITY_REGISTRY = FragmentCapabilityRegistry()


@dataclass(frozen=True)
class FragmentBlock:
    cellIds: NDArray[np.uint32]
    starts: NDArray[np.uint32]
    ends: NDArray[np.uint32]

    @property
    def size(self) -> int:
        return int(self.cellIds.size)


@runtime_checkable
class FragmentSource(Protocol):
    @property
    def chromosomeNames(self) -> tuple[str, ...]: ...

    @property
    def cellNames(self) -> tuple[str, ...]: ...

    @property
    def recordCount(self) -> int: ...

    @property
    def residentBytes(self) -> int: ...

    @property
    def metadataBytes(self) -> int: ...

    @property
    def blockWorkingBytes(self) -> int: ...

    def iter_chromosome(self, chromosome_id: int) -> Iterator[FragmentBlock]: ...


class _FragmentWrapper:
    def __init__(self, source: FragmentSource) -> None:
        self.source = source

    @property
    def chromosomeNames(self) -> tuple[str, ...]:
        return self.source.chromosomeNames

    @property
    def cellNames(self) -> tuple[str, ...]:
        return self.source.cellNames

    @property
    def recordCount(self) -> int:
        return self.source.recordCount

    @property
    def residentBytes(self) -> int:
        return self.source.residentBytes

    @property
    def metadataBytes(self) -> int:
        return self.source.metadataBytes

    @property
    def blockWorkingBytes(self) -> int:
        return 2 * self.source.blockWorkingBytes


class ShiftedFragmentSource(_FragmentWrapper):
    def __init__(
        self,
        source: FragmentSource,
        shift_start: int,
        shift_end: int,
    ) -> None:
        super().__init__(source)
        self.shiftStart = shift_start
        self.shiftEnd = shift_end

    def iter_chromosome(self, chromosome_id: int) -> Iterator[FragmentBlock]:
        for block in self.source.iter_chromosome(chromosome_id):
            starts = block.starts.astype(np.int64) + self.shiftStart
            ends = block.ends.astype(np.int64) + self.shiftEnd
            if (
                np.any(starts < 0)
                or np.any(ends < starts)
                or np.any(starts > _UINT32_MAX)
                or np.any(ends > _UINT32_MAX)
            ):
                raise MatrixSourceError(
                    "shifted fragment coordinates leave the valid uint32 range"
                )
            yield FragmentBlock(
                block.cellIds,
                starts.astype(np.uint32),
                ends.astype(np.uint32),
            )


class LengthSelectedFragmentSource(_FragmentWrapper):
    def __init__(
        self,
        source: FragmentSource,
        minimum: int,
        maximum: int,
    ) -> None:
        super().__init__(source)
        if minimum < 0 or maximum < minimum:
            raise MatrixSourceError("fragment length bounds are invalid")
        self.minimum = minimum
        self.maximum = maximum

    def iter_chromosome(self, chromosome_id: int) -> Iterator[FragmentBlock]:
        for block in self.source.iter_chromosome(chromosome_id):
            lengths = block.ends.astype(np.uint64) - block.starts.astype(np.uint64)
            keep = (lengths >= self.minimum) & (lengths <= self.maximum)
            if np.any(keep):
                yield FragmentBlock(
                    block.cellIds[keep],
                    block.starts[keep],
                    block.ends[keep],
                )


class ChromosomeSelectedFragmentSource(_FragmentWrapper):
    def __init__(
        self,
        source: FragmentSource,
        selection: Sequence[int],
        names: Sequence[str],
    ) -> None:
        super().__init__(source)
        self.selection = tuple(int(value) for value in selection)
        self._chromosomeNames = tuple(names)

    @property
    def chromosomeNames(self) -> tuple[str, ...]:
        return self._chromosomeNames

    @property
    def metadataBytes(self) -> int:
        return self.source.metadataBytes + _metadata_bytes(self._chromosomeNames)

    def iter_chromosome(self, chromosome_id: int) -> Iterator[FragmentBlock]:
        if chromosome_id < 0 or chromosome_id >= len(self.selection):
            raise IndexError("fragment chromosome ID is out of range")
        yield from self.source.iter_chromosome(self.selection[chromosome_id])


class CellMappedFragmentSource(_FragmentWrapper):
    def __init__(
        self,
        source: FragmentSource,
        mapping: NDArray[np.int64],
        names: Sequence[str],
    ) -> None:
        super().__init__(source)
        if mapping.shape != (len(source.cellNames),):
            raise MatrixSourceError("fragment cell mapping has an invalid length")
        self.mapping = mapping
        self._cellNames = tuple(names)

    @property
    def cellNames(self) -> tuple[str, ...]:
        return self._cellNames

    @property
    def residentBytes(self) -> int:
        return self.source.residentBytes + self.mapping.nbytes

    @property
    def metadataBytes(self) -> int:
        return (
            self.source.metadataBytes
            + self.mapping.nbytes
            + _metadata_bytes(self._cellNames)
        )

    def iter_chromosome(self, chromosome_id: int) -> Iterator[FragmentBlock]:
        for block in self.source.iter_chromosome(chromosome_id):
            mapped = self.mapping[block.cellIds]
            keep = mapped >= 0
            if np.any(keep):
                yield FragmentBlock(
                    mapped[keep].astype(np.uint32),
                    block.starts[keep],
                    block.ends[keep],
                )


class RenamedFragmentSource(_FragmentWrapper):
    def __init__(
        self,
        source: FragmentSource,
        *,
        chromosome_names: Sequence[str] | None = None,
        cell_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__(source)
        self._chromosomeNames = (
            source.chromosomeNames
            if chromosome_names is None
            else tuple(chromosome_names)
        )
        self._cellNames = source.cellNames if cell_names is None else tuple(cell_names)
        if len(self._chromosomeNames) != len(source.chromosomeNames):
            raise MatrixSourceError(
                "replacement chromosome names have an invalid length"
            )
        if len(self._cellNames) != len(source.cellNames):
            raise MatrixSourceError("replacement cell names have an invalid length")

    @property
    def chromosomeNames(self) -> tuple[str, ...]:
        return self._chromosomeNames

    @property
    def cellNames(self) -> tuple[str, ...]:
        return self._cellNames

    @property
    def metadataBytes(self) -> int:
        return (
            self.source.metadataBytes
            + _metadata_bytes(self._chromosomeNames)
            + _metadata_bytes(self._cellNames)
        )

    def iter_chromosome(self, chromosome_id: int) -> Iterator[FragmentBlock]:
        yield from self.source.iter_chromosome(chromosome_id)


class RegionSelectedFragmentSource(_FragmentWrapper):
    def __init__(
        self,
        source: FragmentSource,
        regions: Mapping[int, tuple[NDArray[np.uint32], NDArray[np.uint32]]],
        *,
        invert: bool,
        metadata_bytes: int,
    ) -> None:
        super().__init__(source)
        self.regions = dict(regions)
        self.invert = invert
        self._regionMetadataBytes = metadata_bytes

    @property
    def residentBytes(self) -> int:
        return self.source.residentBytes + self._regionMetadataBytes

    @property
    def metadataBytes(self) -> int:
        return self.source.metadataBytes + self._regionMetadataBytes

    def iter_chromosome(self, chromosome_id: int) -> Iterator[FragmentBlock]:
        region = self.regions.get(chromosome_id)
        for block in self.source.iter_chromosome(chromosome_id):
            overlaps = np.zeros(block.size, dtype=bool)
            if region is not None:
                starts, ends = region
                for region_start, region_end in zip(starts, ends, strict=True):
                    overlaps |= (block.starts <= region_end) & (
                        block.ends >= region_start
                    )
            keep = ~overlaps if self.invert else overlaps
            if np.any(keep):
                yield FragmentBlock(
                    block.cellIds[keep],
                    block.starts[keep],
                    block.ends[keep],
                )


class MergedFragmentSource:
    def __init__(self, sources: Sequence[FragmentSource]) -> None:
        if not sources:
            raise MatrixSourceError("fragment merge requires at least one source")
        self.sources = tuple(sources)
        chromosome_names: list[str] = []
        for source in self.sources:
            for name in source.chromosomeNames:
                if name not in chromosome_names:
                    chromosome_names.append(name)
        self._chromosomeNames = tuple(chromosome_names)
        self._cellNames = tuple(
            name for source in self.sources for name in source.cellNames
        )
        self._cellOffsets: list[int] = []
        offset = 0
        for source in self.sources:
            self._cellOffsets.append(offset)
            offset += len(source.cellNames)

    @property
    def chromosomeNames(self) -> tuple[str, ...]:
        return self._chromosomeNames

    @property
    def cellNames(self) -> tuple[str, ...]:
        return self._cellNames

    @property
    def recordCount(self) -> int:
        return sum(source.recordCount for source in self.sources)

    @property
    def residentBytes(self) -> int:
        return sum(source.residentBytes for source in self.sources)

    @property
    def metadataBytes(self) -> int:
        return sum(source.metadataBytes for source in self.sources) + _metadata_bytes(
            self._chromosomeNames + self._cellNames
        )

    @property
    def blockWorkingBytes(self) -> int:
        return 2 * max(source.blockWorkingBytes for source in self.sources)

    def iter_chromosome(self, chromosome_id: int) -> Iterator[FragmentBlock]:
        if chromosome_id < 0 or chromosome_id >= len(self.chromosomeNames):
            raise IndexError("fragment chromosome ID is out of range")
        name = self.chromosomeNames[chromosome_id]
        for source, cell_offset in zip(
            self.sources,
            self._cellOffsets,
            strict=True,
        ):
            if name not in source.chromosomeNames:
                continue
            source_id = source.chromosomeNames.index(name)
            for block in source.iter_chromosome(source_id):
                shifted_cells = block.cellIds.astype(np.uint64) + cell_offset
                if np.any(shifted_cells > _UINT32_MAX):
                    raise MatrixSourceError("merged fragment cell IDs overflow uint32")
                yield FragmentBlock(
                    shifted_cells.astype(np.uint32),
                    block.starts,
                    block.ends,
                )


def _class_names(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        result = tuple(str(item) for item in value)
        if not result:
            raise MatrixSourceError("fragment class vector cannot be empty")
        return result
    return (str(value),)


def _slot_mapping(
    specification: Mapping[str, Any], object_path: str
) -> Mapping[str, Any]:
    nested = specification.get("slots")
    if nested is None:
        return specification
    if not isinstance(nested, Mapping):
        raise TypeError(f"fragment slots at {object_path} must be a mapping")
    return nested


def _vector_length(value: Any, object_path: str) -> int:
    try:
        return len(value)
    except TypeError:
        shape = getattr(value, "shape", None)
        if shape is None:
            raise TypeError(f"vector at {object_path} has no bounded length") from None
        normalized = tuple(int(item) for item in shape)
        if len(normalized) != 1:
            raise MatrixSourceError(f"vector at {object_path} must be one-dimensional")
        return normalized[0]


def _read_vector_slice(
    value: Any,
    start: int,
    stop: int,
    object_path: str,
) -> NDArray[Any]:
    if hasattr(value, "read_block"):
        result = np.asarray(value.read_block(start, stop))
    else:
        try:
            result = np.asarray(value[start:stop])
        except (IndexError, TypeError, ValueError) as error:
            raise MatrixSourceError(
                f"vector at {object_path} does not support bounded slicing"
            ) from error
    if result.ndim != 1:
        result = result.reshape(-1)
    if result.size != stop - start:
        raise MatrixSourceError(
            f"vector at {object_path} returned {result.size} values; "
            f"expected {stop - start}"
        )
    return result


def _single_value(value: Any, object_path: str) -> Any:
    if isinstance(value, os.PathLike):
        return value
    if isinstance(
        value,
        str
        | bytes
        | bool
        | int
        | float
        | np.str_
        | np.bytes_
        | np.bool_
        | np.integer
        | np.floating,
    ):
        return value
    length = _vector_length(value, object_path)
    if length != 1:
        raise MatrixSourceError(f"value at {object_path} must be scalar")
    return _read_vector_slice(value, 0, 1, object_path)[0]


def _decode_text(value: Any, object_path: str) -> str:
    if isinstance(value, bytes | np.bytes_):
        try:
            result = bytes(value).decode("utf-8")
        except UnicodeDecodeError as error:
            raise MatrixSourceError(
                f"text at {object_path} is not valid UTF-8"
            ) from error
    elif isinstance(value, str | np.str_):
        result = str(value)
    else:
        raise TypeError(f"text at {object_path} must contain strings")
    if "\x00" in result:
        raise MatrixSourceError(f"text at {object_path} contains a NUL character")
    return result


def _text_scalar(value: Any, object_path: str) -> str:
    return _decode_text(_single_value(value, object_path), object_path)


def _bool_scalar(value: Any, object_path: str) -> bool:
    scalar = _single_value(value, object_path)
    if isinstance(scalar, bool | np.bool_):
        return bool(scalar)
    if isinstance(scalar, int | np.integer) and int(scalar) in {0, 1}:
        return bool(scalar)
    raise TypeError(f"logical value at {object_path} must be TRUE or FALSE")


def _positive_int_scalar(value: Any, object_path: str) -> int:
    scalar = _single_value(value, object_path)
    if isinstance(scalar, bool | np.bool_) or not isinstance(
        scalar, int | float | np.integer | np.floating
    ):
        raise TypeError(f"integer value at {object_path} must be numeric")
    numeric = float(scalar)
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric <= 0:
        raise MatrixSourceError(f"integer value at {object_path} must be positive")
    return int(numeric)


def _integer_scalar(
    value: Any,
    object_path: str,
    *,
    allow_missing: bool = False,
) -> int | None:
    scalar = _single_value(value, object_path)
    if isinstance(scalar, bool | np.bool_) or not isinstance(
        scalar, int | float | np.integer | np.floating
    ):
        raise TypeError(f"integer value at {object_path} must be numeric")
    numeric = float(scalar)
    if (
        allow_missing
        and isinstance(scalar, int | np.integer)
        and int(scalar) == np.iinfo(np.int32).min
    ):
        return None
    if (
        not math.isfinite(numeric)
        or not numeric.is_integer()
        or numeric < np.iinfo(np.int32).min
        or numeric > np.iinfo(np.int32).max
    ):
        raise MatrixSourceError(f"integer value at {object_path} is invalid")
    return int(numeric)


def _text_values(
    value: Any,
    object_path: str,
    limits: SourceLimits,
) -> tuple[str, ...]:
    if isinstance(value, str | bytes | np.str_ | np.bytes_):
        raw_values: Sequence[Any] = (value,)
    else:
        raw_values = value
    length = _vector_length(raw_values, object_path)
    if length * 8 > limits.maxMetadataBytes:
        raise ResourceLimitError(
            f"text at {object_path} exceeds maxMetadataBytes={limits.maxMetadataBytes}"
        )
    output: list[str] = []
    total = 0
    for start in range(0, length, 4096):
        stop = min(length, start + 4096)
        if hasattr(raw_values, "read_block"):
            values = raw_values.read_block(start, stop)
        else:
            values = raw_values[start:stop]
        for index, value_item in enumerate(values, start=start):
            decoded = _decode_text(value_item, f"{object_path}[{index}]")
            total += len(decoded.encode("utf-8")) + 8
            if total > limits.maxMetadataBytes:
                raise ResourceLimitError(
                    f"text at {object_path} exceeds "
                    f"maxMetadataBytes={limits.maxMetadataBytes}"
                )
            output.append(decoded)
    return tuple(output)


def _metadata_bytes(values: Sequence[str]) -> int:
    return sum(len(value.encode("utf-8")) + 8 for value in values)


def _resolve_sidecar(
    value: Any,
    *,
    rds_path: str | os.PathLike[str] | None,
    absolute_prefix_remaps: Mapping[str | os.PathLike[str], str | os.PathLike[str]]
    | None,
    expect: str,
) -> Path:
    if rds_path is not None:
        return SidecarPathResolver(
            rds_path,
            absolute_prefix_remaps=absolute_prefix_remaps,
        ).resolve(value, expect=expect)
    if expect == "file":
        return require_filesystem_path(value, "fragment HDF5 source")
    if not isinstance(value, str | os.PathLike):
        raise TypeError("fragment sidecar directory must be a filesystem path")
    path = Path(value).expanduser().resolve(strict=False)
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_dir():
        raise UnsafeSidecarError(f"fragment sidecar path {path} is not a directory")
    return path


def _coerce_unsigned(
    values: NDArray[Any],
    dtype: np.dtype[Any],
    object_path: str,
) -> NDArray[Any]:
    raw = np.asarray(values)
    if raw.dtype.kind == "i" and raw.dtype.itemsize == dtype.itemsize:
        if dtype == np.dtype(np.uint32):
            return raw.astype(np.int32, copy=False).view(np.uint32).copy()
        if np.any(raw < 0):
            raise MatrixSourceError(f"array at {object_path} contains a negative value")
        return raw.astype(dtype, copy=False)
    if raw.dtype.kind in "iu":
        if raw.dtype.kind == "i" and np.any(raw < 0):
            raise MatrixSourceError(f"array at {object_path} contains a negative value")
        widened = raw.astype(object)
        upper = int(np.iinfo(dtype).max)
        if any(int(value) > upper for value in widened):
            raise MatrixSourceError(f"array at {object_path} exceeds {dtype}")
        return raw.astype(dtype)
    if raw.dtype.kind == "f":
        numeric = raw.astype(np.float64, copy=False)
        if (
            np.any(~np.isfinite(numeric))
            or np.any(numeric < 0)
            or np.any(numeric != np.floor(numeric))
        ):
            raise MatrixSourceError(
                f"array at {object_path} must contain finite unsigned integers"
            )
        upper = min(int(np.iinfo(dtype).max), 2**53)
        if np.any(numeric > upper):
            raise MatrixSourceError(
                f"array at {object_path} cannot be represented exactly as {dtype}"
            )
        return numeric.astype(dtype)
    raise TypeError(f"array at {object_path} must contain numeric integers")


class _MemoryArrayStore:
    def __init__(
        self,
        slots: Mapping[str, Any],
        version: str,
        dtypes: Mapping[str, np.dtype[Any]],
        limits: SourceLimits,
        object_path: str,
    ) -> None:
        self.slots = slots
        self._version = version
        self.dtypes = dict(dtypes)
        self._limits = limits
        self.objectPath = object_path

    @property
    def version(self) -> str:
        return self._version

    @property
    def residentBytes(self) -> int:
        total = 0
        for value in self.slots.values():
            if isinstance(value, np.ndarray):
                total += int(value.nbytes)
        return total

    def has(self, name: str) -> bool:
        return name in self.slots

    def numeric_info(self, name: str) -> tuple[np.dtype[Any], int]:
        if name not in self.slots:
            raise MatrixSourceError(f"BPCells numeric array {name!r} is missing")
        if name not in self.dtypes:
            raise MatrixSourceError(f"BPCells array {name!r} is not numeric")
        value = self.slots[name]
        length = _vector_length(value, f"{self.objectPath}@{name}")
        return self.dtypes[name], length

    def read_numeric(
        self,
        name: str,
        start: int = 0,
        stop: int | None = None,
    ) -> NDArray[Any]:
        dtype, length = self.numeric_info(name)
        stop = length if stop is None else int(stop)
        start = int(start)
        if start < 0 or stop < start or stop > length:
            raise IndexError(
                f"BPCells memory array {name!r} window [{start}, {stop}) "
                f"is outside [0, {length})"
            )
        values = _read_vector_slice(
            self.slots[name],
            start,
            stop,
            f"{self.objectPath}@{name}",
        )
        return _coerce_unsigned(values, dtype, f"{self.objectPath}@{name}")

    def read_text(self, name: str) -> tuple[str, ...]:
        if name not in self.slots:
            raise MatrixSourceError(f"BPCells text array {name!r} is missing")
        return _text_values(
            self.slots[name],
            f"{self.objectPath}@{name}",
            self._limits,
        )


def _memory_dtypes(version: str) -> dict[str, np.dtype[Any]]:
    match = _FRAGMENT_VERSION_PATTERN.fullmatch(version)
    if match is None:
        raise MatrixSourceError(f"unsupported BPCells fragment format {version!r}")
    compression, version_text = match.groups()
    pointer_dtype = np.dtype(np.uint32 if version_text == "1" else np.uint64)
    result = {
        "chr_ptr": pointer_dtype,
        "end_max": np.dtype(np.uint32),
    }
    if compression == "unpacked":
        result.update(
            {
                "cell": np.dtype(np.uint32),
                "start": np.dtype(np.uint32),
                "end": np.dtype(np.uint32),
            }
        )
    else:
        for prefix in ("cell", "start", "end"):
            result[f"{prefix}_data"] = np.dtype(np.uint32)
            result[f"{prefix}_idx"] = np.dtype(np.uint32)
            result[f"{prefix}_idx_offsets"] = np.dtype(np.uint64)
        result["start_starts"] = np.dtype(np.uint32)
    return result


class _PackedD1Reader:
    def __init__(
        self,
        store: _BPArrayStore,
        count: int,
        *,
        require_offsets: bool,
        limits: SourceLimits,
    ) -> None:
        self.store = store
        self.count = count
        self.reader = _StoredBP128Array(
            store,
            "start",
            count,
            "plain",
            require_offsets=require_offsets,
            limits=limits,
        )
        self.blockCount = math.ceil(count / 128)
        _require_numeric_array(
            store,
            "start_starts",
            np.dtype(np.uint32),
            self.blockCount,
        )

    @property
    def indexOffsets(self) -> NDArray[np.int64]:
        return self.reader.indexOffsets

    def read(self, start: int, stop: int) -> NDArray[np.uint32]:
        start = int(start)
        stop = int(stop)
        if start < 0 or stop < start or stop > self.count:
            raise IndexError(
                f"BPCells packed start window [{start}, {stop}) "
                f"is outside [0, {self.count})"
            )
        if start == stop:
            return np.empty(0, dtype=np.uint32)
        first_block = start // 128
        final_block = (stop - 1) // 128
        decoded: list[NDArray[np.uint32]] = []
        for block in range(first_block, final_block + 1):
            block_start = block * 128
            block_stop = min(self.count, block_start + 128)
            deltas = self.reader.read(block_start, block_stop).astype(
                np.uint64, copy=False
            )
            initial = int(self.store.read_numeric("start_starts", block, block + 1)[0])
            values = (
                np.cumsum(deltas, dtype=np.uint64) + np.uint64(initial)
            ) & np.uint64(_UINT32_MAX)
            decoded.append(values.astype(np.uint32))
        combined = np.concatenate(decoded)
        local_start = start - first_block * 128
        return combined[local_start : local_start + stop - start]


class StoredFragmentSource:
    def __init__(
        self,
        store: _BPArrayStore,
        *,
        chromosome_names: Sequence[str] | None = None,
        cell_names: Sequence[str] | None = None,
        object_path: str,
        class_name: str,
        limits: SourceLimits = DEFAULT_LIMITS,
    ) -> None:
        match = _FRAGMENT_VERSION_PATTERN.fullmatch(store.version)
        if match is None:
            raise MatrixSourceError(
                f"unsupported BPCells fragment format {store.version!r}"
            )
        compression, version_text = match.groups()
        self.store = store
        self.compression = compression
        self.formatVersion = int(version_text)
        self.objectPath = object_path
        self.className = class_name
        self._limits = limits
        pointer_dtype = np.dtype(np.uint32 if self.formatVersion == 1 else np.uint64)
        pointer_length = _require_numeric_array(store, "chr_ptr", pointer_dtype)
        if pointer_length % 2:
            raise MatrixSourceError("BPCells chr_ptr must contain start/end pairs")
        pointer_bytes = pointer_length * pointer_dtype.itemsize
        if pointer_bytes > limits.maxMetadataBytes:
            raise ResourceLimitError(
                f"BPCells chr_ptr exceeds maxMetadataBytes={limits.maxMetadataBytes}"
            )
        flat_pointers = store.read_numeric("chr_ptr").astype(np.uint64, copy=False)
        self._chrPointers = flat_pointers.reshape(-1, 2)
        self._validate_pointers()
        self._recordCount = (
            int(self._chrPointers[-1, 1]) if self._chrPointers.size else 0
        )
        if self._recordCount > limits.maxNnz:
            raise ResourceLimitError(
                f"fragment count {self._recordCount} exceeds maxNnz={limits.maxNnz}"
            )
        _require_numeric_array(
            store,
            "end_max",
            np.dtype(np.uint32),
            math.ceil(self._recordCount / 128),
        )
        source_chromosomes = (
            tuple(chromosome_names)
            if chromosome_names is not None
            else store.read_text("chr_names")
        )
        source_cells = (
            tuple(cell_names)
            if cell_names is not None
            else store.read_text("cell_names")
        )
        normalized_chromosomes = _normalize_names(
            source_chromosomes,
            self._chrPointers.shape[0],
            "chromosome",
            limits,
        )
        normalized_cells = _normalize_names(
            source_cells,
            len(source_cells),
            "cell",
            limits,
        )
        assert normalized_chromosomes is not None
        assert normalized_cells is not None
        self._chromosomeNames = normalized_chromosomes
        self._cellNames = normalized_cells
        self._cellReader: _StoredBP128Array | None = None
        self._startReader: _PackedD1Reader | None = None
        self._endReader: _StoredBP128Array | None = None
        offset_bytes = 0
        if compression == "unpacked":
            for name in ("cell", "start", "end"):
                _require_numeric_array(
                    store,
                    name,
                    np.dtype(np.uint32),
                    self._recordCount,
                )
        else:
            require_offsets = self.formatVersion == 2
            self._cellReader = _StoredBP128Array(
                store,
                "cell",
                self._recordCount,
                "plain",
                require_offsets=require_offsets,
                limits=limits,
            )
            self._startReader = _PackedD1Reader(
                store,
                self._recordCount,
                require_offsets=require_offsets,
                limits=limits,
            )
            self._endReader = _StoredBP128Array(
                store,
                "end",
                self._recordCount,
                "plain",
                require_offsets=require_offsets,
                limits=limits,
            )
            offset_bytes = (
                self._cellReader.indexOffsets.nbytes
                + self._startReader.indexOffsets.nbytes
                + self._endReader.indexOffsets.nbytes
            )
        metadata_bytes = (
            pointer_bytes
            + offset_bytes
            + _metadata_bytes(self._chromosomeNames)
            + _metadata_bytes(self._cellNames)
        )
        if metadata_bytes > limits.maxMetadataBytes:
            raise ResourceLimitError(
                f"fragment metadata exceeds maxMetadataBytes={limits.maxMetadataBytes}"
            )
        self.metadataBytes = metadata_bytes
        overhead = 4096 if compression == "packed" else 0
        bytes_per_record = 128
        minimum_block_bytes = (
            overhead + bytes_per_record * _FRAGMENT_BLOCK_SAFETY_FACTOR
        )
        if limits.maxBlockBytes < minimum_block_bytes:
            raise ResourceLimitError(
                "fragment decoding requires at least "
                f"{minimum_block_bytes} bytes; maxBlockBytes={limits.maxBlockBytes}"
            )
        self.blockRecords = min(
            limits.compressedChunkNnz,
            max(
                1,
                (limits.maxBlockBytes - overhead)
                // (bytes_per_record * _FRAGMENT_BLOCK_SAFETY_FACTOR),
            ),
            max(1, self._recordCount),
        )
        self._blockWorkingBytes = overhead + self.blockRecords * bytes_per_record
        self._validate_records()

    def _validate_pointers(self) -> None:
        previous = 0
        for chromosome_id, pair in enumerate(self._chrPointers):
            start = int(pair[0])
            stop = int(pair[1])
            if start != previous:
                raise MatrixSourceError(
                    f"BPCells chr_ptr chromosome {chromosome_id} starts at {start}; "
                    f"expected {previous}"
                )
            if stop < start:
                raise MatrixSourceError(
                    f"BPCells chr_ptr chromosome {chromosome_id} has decreasing bounds"
                )
            previous = stop

    def _read_records(self, start: int, stop: int) -> FragmentBlock:
        if self.compression == "unpacked":
            cells = self.store.read_numeric("cell", start, stop).astype(
                np.uint32, copy=False
            )
            starts = self.store.read_numeric("start", start, stop).astype(
                np.uint32, copy=False
            )
            ends = self.store.read_numeric("end", start, stop).astype(
                np.uint32, copy=False
            )
        else:
            assert self._cellReader is not None
            assert self._startReader is not None
            assert self._endReader is not None
            cells = self._cellReader.read(start, stop)
            starts = self._startReader.read(start, stop)
            lengths = self._endReader.read(start, stop)
            widened = starts.astype(np.uint64) + lengths.astype(np.uint64)
            if np.any(widened > _UINT32_MAX):
                raise MatrixSourceError(
                    "packed fragment end coordinate overflows uint32"
                )
            ends = widened.astype(np.uint32)
        return FragmentBlock(cells, starts, ends)

    def _validate_records(self) -> None:
        expected_end_max: list[int] = []
        expected_start = 0
        current_end_max = 0
        previous_end_max = 0
        expected_index = 0

        def flush_expected() -> None:
            nonlocal expected_index
            if not expected_end_max:
                return
            actual = self.store.read_numeric(
                "end_max",
                expected_index,
                expected_index + len(expected_end_max),
            ).astype(np.uint32, copy=False)
            expected = np.asarray(expected_end_max, dtype=np.uint32)
            if not np.array_equal(actual, expected):
                mismatch = int(np.flatnonzero(actual != expected)[0])
                raise MatrixSourceError(
                    "BPCells end_max is inconsistent at block "
                    f"{expected_index + mismatch}"
                )
            expected_index += len(expected_end_max)
            expected_end_max.clear()

        for chromosome_id in range(len(self._chromosomeNames)):
            previous_end_max = max(previous_end_max, current_end_max)
            current_end_max = 0
            previous_start: int | None = None
            pair = self._chrPointers[chromosome_id]
            chromosome_start = int(pair[0])
            chromosome_stop = int(pair[1])
            for block_start in range(
                chromosome_start,
                chromosome_stop,
                self.blockRecords,
            ):
                block_stop = min(chromosome_stop, block_start + self.blockRecords)
                block = self._read_records(block_start, block_stop)
                if block.size == 0:
                    continue
                if np.any(block.cellIds >= len(self._cellNames)):
                    local = int(
                        np.flatnonzero(block.cellIds >= len(self._cellNames))[0]
                    )
                    raise MatrixSourceError(
                        f"fragment cell ID is out of range at record {block_start + local}"
                    )
                if np.any(block.ends < block.starts):
                    local = int(np.flatnonzero(block.ends < block.starts)[0])
                    raise MatrixSourceError(
                        f"fragment end precedes start at record {block_start + local}"
                    )
                if previous_start is not None and int(block.starts[0]) < previous_start:
                    raise MatrixSourceError(
                        f"fragment starts are not sorted on chromosome {chromosome_id}"
                    )
                if block.starts.size > 1 and np.any(
                    block.starts[1:] < block.starts[:-1]
                ):
                    raise MatrixSourceError(
                        f"fragment starts are not sorted on chromosome {chromosome_id}"
                    )
                previous_start = int(block.starts[-1])
                local_start = 0
                while local_start < block.size:
                    global_position = block_start + local_start
                    take = min(block.size - local_start, 128 - global_position % 128)
                    local_stop = local_start + take
                    current_end_max = max(
                        current_end_max,
                        int(np.max(block.ends[local_start:local_stop])),
                    )
                    global_position += take
                    if global_position % 128 == 0:
                        expected_end_max.append(max(current_end_max, previous_end_max))
                        previous_end_max = 0
                    local_start = local_stop
                flush_expected()
                expected_start = block_stop
            if expected_start != chromosome_stop:
                raise MatrixSourceError(
                    f"fragment chromosome {chromosome_id} ended unexpectedly"
                )
        if self._recordCount % 128:
            expected_end_max.append(max(current_end_max, previous_end_max))
        flush_expected()
        if expected_index != math.ceil(self._recordCount / 128):
            raise MatrixSourceError("BPCells end_max has an invalid length")

    @property
    def chromosomeNames(self) -> tuple[str, ...]:
        return self._chromosomeNames

    @property
    def cellNames(self) -> tuple[str, ...]:
        return self._cellNames

    @property
    def recordCount(self) -> int:
        return self._recordCount

    @property
    def residentBytes(self) -> int:
        store_bytes = int(getattr(self.store, "residentBytes", 0))
        return self.metadataBytes + store_bytes

    @property
    def blockWorkingBytes(self) -> int:
        return self._blockWorkingBytes

    def iter_chromosome(self, chromosome_id: int) -> Iterator[FragmentBlock]:
        chromosome_id = int(chromosome_id)
        if chromosome_id < 0 or chromosome_id >= len(self.chromosomeNames):
            raise IndexError(
                f"fragment chromosome ID {chromosome_id} is outside "
                f"[0, {len(self.chromosomeNames)})"
            )
        pair = self._chrPointers[chromosome_id]
        start = int(pair[0])
        stop = int(pair[1])
        for block_start in range(start, stop, self.blockRecords):
            yield self._read_records(
                block_start,
                min(stop, block_start + self.blockRecords),
            )


def _optional_override_names(
    slots: Mapping[str, Any],
    name: str,
    object_path: str,
    limits: SourceLimits,
) -> tuple[str, ...] | None:
    value = slots.get(name)
    if value is None:
        return None
    values = _text_values(value, f"{object_path}@{name}", limits)
    return values or None


def fragment_source_from_slots(
    specification: Mapping[str, Any],
    *,
    object_path: str = "$",
    rds_path: str | os.PathLike[str] | None = None,
    absolute_prefix_remaps: Mapping[str | os.PathLike[str], str | os.PathLike[str]]
    | None = None,
    limits: SourceLimits = DEFAULT_LIMITS,
) -> FragmentSource:
    if not isinstance(specification, Mapping):
        raise TypeError(f"fragment source at {object_path} must be a mapping")
    slots = _slot_mapping(specification, object_path)
    classes = _class_names(
        specification.get(
            "className",
            specification.get(
                "class",
                slots.get("className", slots.get("class")),
            ),
        )
    )
    capability = FRAGMENT_CAPABILITY_REGISTRY.resolve(
        classes,
        object_path=object_path,
    )
    class_name = capability.className

    def resolve_fragment(value: Any, path: str) -> FragmentSource:
        if isinstance(value, FragmentSource):
            return value
        if isinstance(value, Mapping):
            return fragment_source_from_slots(
                value,
                object_path=path,
                rds_path=rds_path,
                absolute_prefix_remaps=absolute_prefix_remaps,
                limits=limits,
            )
        raise TypeError(f"fragment input at {path} must be a source or mapping")

    if class_name == "MergeFragments":
        values = slots.get("fragments_list")
        if not isinstance(values, Sequence) or isinstance(
            values, str | bytes | bytearray
        ):
            raise TypeError(f"fragments_list at {object_path} must be a sequence")
        return MergedFragmentSource(
            [
                resolve_fragment(value, f"{object_path}@fragments_list[{index}]")
                for index, value in enumerate(values)
            ]
        )
    if class_name not in {
        "UnpackedMemFragments",
        "PackedMemFragments",
        "FragmentsDir",
        "FragmentsHDF5",
    }:
        source = resolve_fragment(
            slots.get("fragments"),
            f"{object_path}@fragments",
        )
        if class_name == "ShiftFragments":
            shift_start = _integer_scalar(
                slots.get("shift_start"),
                f"{object_path}@shift_start",
            )
            shift_end = _integer_scalar(
                slots.get("shift_end"),
                f"{object_path}@shift_end",
            )
            assert shift_start is not None and shift_end is not None
            return ShiftedFragmentSource(source, shift_start, shift_end)
        if class_name == "SelectLength":
            minimum = _integer_scalar(
                slots.get("min_len"),
                f"{object_path}@min_len",
                allow_missing=True,
            )
            maximum = _integer_scalar(
                slots.get("max_len"),
                f"{object_path}@max_len",
                allow_missing=True,
            )
            return LengthSelectedFragmentSource(
                source,
                0 if minimum is None else minimum,
                np.iinfo(np.int32).max if maximum is None else maximum,
            )
        if class_name in {"ChrSelectName", "ChrSelectIndex"}:
            if class_name == "ChrSelectName":
                names = _text_values(
                    slots.get("chr_names"),
                    f"{object_path}@chr_names",
                    limits,
                )
                if len(set(names)) != len(names):
                    raise MatrixSourceError(
                        "fragment chromosome selection contains duplicates"
                    )
                missing = [name for name in names if name not in source.chromosomeNames]
                if missing:
                    raise MatrixSourceError(
                        f"fragment chromosome selection contains unknown names {missing!r}"
                    )
                selection = tuple(source.chromosomeNames.index(name) for name in names)
            else:
                raw = _unsigned_vector(
                    slots.get("chr_index_selection"),
                    "chr_index_selection",
                    object_path,
                    limits,
                )
                if np.any(raw == 0) or np.any(raw > len(source.chromosomeNames)):
                    raise MatrixSourceError(
                        "fragment chromosome selection contains an invalid R index"
                    )
                selection = tuple(int(value) - 1 for value in raw)
                if len(set(selection)) != len(selection):
                    raise MatrixSourceError(
                        "fragment chromosome selection contains duplicates"
                    )
                names = tuple(source.chromosomeNames[index] for index in selection)
            return ChromosomeSelectedFragmentSource(source, selection, names)
        if class_name in {"CellSelectName", "CellSelectIndex"}:
            if class_name == "CellSelectName":
                names = _text_values(
                    slots.get("cell_names"),
                    f"{object_path}@cell_names",
                    limits,
                )
                if len(set(names)) != len(names):
                    raise MatrixSourceError(
                        "fragment cell selection contains duplicates"
                    )
                missing = [name for name in names if name not in source.cellNames]
                if missing:
                    raise MatrixSourceError(
                        f"fragment cell selection contains unknown names {missing!r}"
                    )
                selection = tuple(source.cellNames.index(name) for name in names)
            else:
                raw = _unsigned_vector(
                    slots.get("cell_index_selection"),
                    "cell_index_selection",
                    object_path,
                    limits,
                )
                if np.any(raw == 0) or np.any(raw > len(source.cellNames)):
                    raise MatrixSourceError(
                        "fragment cell selection contains an invalid R index"
                    )
                selection = tuple(int(value) - 1 for value in raw)
                if len(set(selection)) != len(selection):
                    raise MatrixSourceError(
                        "fragment cell selection contains duplicates"
                    )
                names = tuple(source.cellNames[index] for index in selection)
            mapping = np.full(len(source.cellNames), -1, dtype=np.int64)
            mapping[np.asarray(selection, dtype=np.int64)] = np.arange(
                len(selection),
                dtype=np.int64,
            )
            return CellMappedFragmentSource(source, mapping, names)
        if class_name == "CellMerge":
            group_names = _text_values(
                slots.get("group_names"),
                f"{object_path}@group_names",
                limits,
            )
            group_ids = _unsigned_vector(
                slots.get("group_ids"),
                "group_ids",
                object_path,
                limits,
            )
            if group_ids.shape != (len(source.cellNames),) or np.any(
                group_ids >= len(group_names)
            ):
                raise MatrixSourceError(
                    "fragment cell merge groups do not match the source cells"
                )
            return CellMappedFragmentSource(
                source,
                group_ids.astype(np.int64),
                group_names,
            )
        if class_name in {"ChrRename", "CellRename", "CellPrefix"}:
            if class_name == "ChrRename":
                return RenamedFragmentSource(
                    source,
                    chromosome_names=_text_values(
                        slots.get("chr_names"),
                        f"{object_path}@chr_names",
                        limits,
                    ),
                )
            if class_name == "CellRename":
                return RenamedFragmentSource(
                    source,
                    cell_names=_text_values(
                        slots.get("cell_names"),
                        f"{object_path}@cell_names",
                        limits,
                    ),
                )
            prefix = _text_scalar(
                slots.get("prefix"),
                f"{object_path}@prefix",
            )
            return RenamedFragmentSource(
                source,
                cell_names=tuple(prefix + name for name in source.cellNames),
            )
        if class_name == "RegionSelect":
            chromosome_levels = _text_values(
                slots.get("chr_levels"),
                f"{object_path}@chr_levels",
                limits,
            )
            chromosome_ids = _unsigned_vector(
                slots.get("chr_id"),
                "chr_id",
                object_path,
                limits,
            )
            starts = _unsigned_vector(
                slots.get("start"),
                "start",
                object_path,
                limits,
            )
            ends = _unsigned_vector(
                slots.get("end"),
                "end",
                object_path,
                limits,
            )
            if not (chromosome_ids.shape == starts.shape == ends.shape) or np.any(
                chromosome_ids >= len(chromosome_levels)
            ):
                raise MatrixSourceError("fragment region metadata is inconsistent")
            if np.any(ends < starts):
                raise MatrixSourceError("fragment region end precedes its start")
            source_regions: dict[
                int,
                tuple[list[np.uint32], list[np.uint32]],
            ] = {}
            for chromosome_id, start, end in zip(
                chromosome_ids,
                starts,
                ends,
                strict=True,
            ):
                name = chromosome_levels[int(chromosome_id)]
                if name not in source.chromosomeNames:
                    continue
                source_id = source.chromosomeNames.index(name)
                region_starts, region_ends = source_regions.setdefault(
                    source_id,
                    ([], []),
                )
                region_starts.append(start)
                region_ends.append(end)
            normalized_regions = {
                chromosome_id: (
                    np.asarray(region_starts, dtype=np.uint32),
                    np.asarray(region_ends, dtype=np.uint32),
                )
                for chromosome_id, (
                    region_starts,
                    region_ends,
                ) in source_regions.items()
            }
            metadata_bytes = (
                chromosome_ids.nbytes
                + starts.nbytes
                + ends.nbytes
                + _metadata_bytes(chromosome_levels)
            )
            if metadata_bytes > limits.maxMetadataBytes:
                raise ResourceLimitError(
                    "fragment region metadata exceeds "
                    f"maxMetadataBytes={limits.maxMetadataBytes}"
                )
            return RegionSelectedFragmentSource(
                source,
                normalized_regions,
                invert=_bool_scalar(
                    slots.get("invert_selection", False),
                    f"{object_path}@invert_selection",
                ),
                metadata_bytes=metadata_bytes,
            )
        raise AssertionError(class_name)

    store: _BPArrayStore
    if class_name in {"UnpackedMemFragments", "PackedMemFragments"}:
        if "version" not in slots:
            raise MatrixSourceError(
                f"fragment source at {object_path} has no version slot"
            )
        version = _text_scalar(slots["version"], f"{object_path}@version")
        store = cast(
            _BPArrayStore,
            _MemoryArrayStore(
                slots,
                version,
                _memory_dtypes(version),
                limits,
                object_path,
            ),
        )
        chromosome_names = None
        cell_names = None
    elif class_name == "FragmentsDir":
        if "dir" not in slots:
            raise MatrixSourceError(f"fragment source at {object_path} has no dir slot")
        path = _resolve_sidecar(
            _single_value(slots["dir"], f"{object_path}@dir"),
            rds_path=rds_path,
            absolute_prefix_remaps=absolute_prefix_remaps,
            expect="directory",
        )
        store = cast(_BPArrayStore, _DirectoryArrayStore(path, limits))
        chromosome_names = _optional_override_names(
            slots, "chr_names", object_path, limits
        )
        cell_names = _optional_override_names(slots, "cell_names", object_path, limits)
    elif class_name == "FragmentsHDF5":
        path_value = slots.get("path", slots.get("filepath"))
        if path_value is None or "group" not in slots:
            raise MatrixSourceError(
                f"fragment source at {object_path} requires path and group slots"
            )
        path = _resolve_sidecar(
            _single_value(path_value, f"{object_path}@path"),
            rds_path=rds_path,
            absolute_prefix_remaps=absolute_prefix_remaps,
            expect="file",
        )
        group = _text_scalar(slots["group"], f"{object_path}@group")
        store = cast(_BPArrayStore, _HDF5ArrayStore(path, group, limits))
        chromosome_names = _optional_override_names(
            slots, "chr_names", object_path, limits
        )
        cell_names = _optional_override_names(slots, "cell_names", object_path, limits)
    else:
        raise AssertionError(class_name)
    match = _FRAGMENT_VERSION_PATTERN.fullmatch(store.version)
    if match is None:
        raise MatrixSourceError(
            f"unsupported BPCells fragment format {store.version!r}"
        )
    compression, _version = match.groups()
    expected_compression = {
        "UnpackedMemFragments": "unpacked",
        "PackedMemFragments": "packed",
    }.get(class_name)
    if expected_compression is not None and compression != expected_compression:
        raise MatrixSourceError(f"{class_name} cannot contain {store.version!r}")
    if class_name in {"FragmentsDir", "FragmentsHDF5"}:
        if "compressed" not in slots:
            raise MatrixSourceError(
                f"fragment source at {object_path} has no compressed slot"
            )
        declared_compressed = _bool_scalar(
            slots["compressed"],
            f"{object_path}@compressed",
        )
        if declared_compressed != (compression == "packed"):
            raise MatrixSourceError(
                f"fragment compressed slot at {object_path} disagrees with "
                f"format {store.version!r}"
            )
    if "buffer_size" in slots:
        _positive_int_scalar(slots["buffer_size"], f"{object_path}@buffer_size")
    return StoredFragmentSource(
        store,
        chromosome_names=chromosome_names,
        cell_names=cell_names,
        object_path=object_path,
        class_name=class_name,
        limits=limits,
    )


def _unsigned_vector(
    value: Any,
    name: str,
    object_path: str,
    limits: SourceLimits,
) -> NDArray[np.uint32]:
    length = _vector_length(value, f"{object_path}@{name}")
    required = length * np.dtype(np.uint32).itemsize
    if required > limits.maxMetadataBytes:
        raise ResourceLimitError(
            f"{name} at {object_path} exceeds "
            f"maxMetadataBytes={limits.maxMetadataBytes}"
        )
    raw = _read_vector_slice(value, 0, length, f"{object_path}@{name}")
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} at {object_path} must contain integers")
    if raw.dtype.kind == "f":
        numeric = raw.astype(np.float64, copy=False)
        invalid = (
            ~np.isfinite(numeric)
            | (numeric < 0)
            | (numeric != np.floor(numeric))
            | (numeric > _UINT32_MAX)
        )
    else:
        numeric = raw
        invalid = (numeric < 0) | (numeric > _UINT32_MAX)
    if np.any(invalid):
        raise MatrixSourceError(
            f"{name} at {object_path} contains an invalid uint32 value"
        )
    return numeric.astype(np.uint32)


def _shape_value(
    value: Any,
    object_path: str,
) -> tuple[int, int]:
    raw = _read_vector_slice(
        value,
        0,
        _vector_length(value, object_path),
        object_path,
    )
    if raw.size != 2 or raw.dtype.kind not in "iuf":
        raise MatrixSourceError(f"dim at {object_path} must contain two integers")
    numeric = raw.astype(np.float64, copy=False)
    if (
        np.any(~np.isfinite(numeric))
        or np.any(numeric < 0)
        or np.any(numeric != np.floor(numeric))
    ):
        raise MatrixSourceError(f"dim at {object_path} must contain two integers")
    return int(numeric[0]), int(numeric[1])


class FragmentDerivedMatrixSource(BaseMatrixSource):
    def __init__(
        self,
        fragments: FragmentSource,
        *,
        matrix_type: str,
        chromosome_ids: Any,
        starts: Any,
        ends: Any,
        chromosome_levels: Any,
        mode: Any,
        tile_widths: Any | None = None,
        transpose: Any = True,
        shape: Any | None = None,
        row_names: Sequence[str | bytes] | NDArray[Any] | None = None,
        column_names: Sequence[str | bytes] | NDArray[Any] | None = None,
        object_path: str = "$",
        limits: SourceLimits = DEFAULT_LIMITS,
    ) -> None:
        if not isinstance(fragments, FragmentSource):
            raise TypeError(
                f"fragment input at {object_path}@fragments is not a FragmentSource"
            )
        if matrix_type not in {"PeakMatrix", "TileMatrix"}:
            raise UnsupportedMatrixOperation(
                object_path,
                "fragment-derived",
                matrix_type,
                "unknown fragment-derived matrix class",
            )
        self.fragments = fragments
        self.matrixType = matrix_type
        self.objectPath = object_path
        self.operation = "fragment-derived"
        self.logicalTranspose = _bool_scalar(transpose, f"{object_path}@transpose")
        self.chromosomeIds = _unsigned_vector(
            chromosome_ids, "chr_id", object_path, limits
        )
        self.starts = _unsigned_vector(starts, "start", object_path, limits)
        self.ends = _unsigned_vector(ends, "end", object_path, limits)
        if not (self.chromosomeIds.size == self.starts.size == self.ends.size):
            raise MatrixSourceError(
                f"{matrix_type} at {object_path} requires equal chr_id, start, and end lengths"
            )
        if np.any(self.ends < self.starts):
            raise MatrixSourceError(
                f"{matrix_type} at {object_path} contains an end before its start"
            )
        self.chromosomeLevels = _text_values(
            chromosome_levels,
            f"{object_path}@chr_levels",
            limits,
        )
        if self.chromosomeLevels != fragments.chromosomeNames:
            raise MatrixSourceError(
                f"{matrix_type} chromosome levels at {object_path} do not match "
                "the fragment source"
            )
        if self.chromosomeIds.size and np.any(
            self.chromosomeIds >= len(self.chromosomeLevels)
        ):
            raise MatrixSourceError(
                f"{matrix_type} at {object_path} has an out-of-range chromosome ID"
            )
        self.mode = _text_scalar(mode, f"{object_path}@mode")
        self.tileWidths: NDArray[np.uint32] | None
        if matrix_type == "PeakMatrix":
            if self.mode not in {"insertions", "fragments", "overlaps"}:
                raise MatrixSourceError(
                    f"PeakMatrix mode {self.mode!r} at {object_path} is invalid"
                )
            if tile_widths is not None:
                raise MatrixSourceError("PeakMatrix cannot contain tile widths")
            self.tileWidths = None
            self._validate_peak_order()
            derived_features = int(self.chromosomeIds.size)
            self._featureOffsets = None
        else:
            if self.mode not in {"insertions", "fragments"}:
                raise MatrixSourceError(
                    f"TileMatrix mode {self.mode!r} at {object_path} is invalid"
                )
            if tile_widths is None:
                raise MatrixSourceError(
                    f"TileMatrix at {object_path} has no tile_width slot"
                )
            self.tileWidths = _unsigned_vector(
                tile_widths, "tile_width", object_path, limits
            )
            if self.tileWidths.size != self.chromosomeIds.size:
                raise MatrixSourceError(
                    f"TileMatrix at {object_path} requires one width per range"
                )
            if np.any(self.tileWidths == 0):
                raise MatrixSourceError(
                    f"TileMatrix at {object_path} contains a zero tile width"
                )
            self._validate_tile_order()
            widths = self.tileWidths.astype(np.uint64)
            spans = self.ends.astype(np.uint64) - self.starts.astype(np.uint64)
            tile_counts = (spans + widths - 1) // widths
            total_tiles = int(np.sum(tile_counts, dtype=np.uint64))
            if total_tiles > _UINT32_MAX:
                raise ResourceLimitError(
                    f"TileMatrix at {object_path} exceeds uint32 feature capacity"
                )
            offsets = np.empty(tile_counts.size + 1, dtype=np.uint64)
            offsets[0] = 0
            np.cumsum(tile_counts, dtype=np.uint64, out=offsets[1:])
            self._featureOffsets = offsets
            derived_features = total_tiles
        native_shape = (len(fragments.cellNames), derived_features)
        logical_shape = native_shape[::-1] if self.logicalTranspose else native_shape
        if shape is not None:
            declared_shape = _shape_value(shape, f"{object_path}@dim")
            if declared_shape != logical_shape:
                raise MatrixSourceError(
                    f"{matrix_type} dim {declared_shape} at {object_path} does not "
                    f"match derived shape {logical_shape}"
                )
        region_metadata = (
            self.chromosomeIds.nbytes
            + self.starts.nbytes
            + self.ends.nbytes
            + (0 if self.tileWidths is None else self.tileWidths.nbytes)
            + (0 if self._featureOffsets is None else self._featureOffsets.nbytes)
            + _metadata_bytes(self.chromosomeLevels)
            + self.chromosomeIds.size * np.dtype(np.int64).itemsize
        )
        if region_metadata + fragments.residentBytes > limits.maxMetadataBytes:
            raise ResourceLimitError(
                "fragment-derived metadata exceeds "
                f"maxMetadataBytes={limits.maxMetadataBytes}"
            )
        self._build_chromosome_indexes()
        self.regionMetadataBytes = int(region_metadata)
        super().__init__(
            logical_shape,
            np.uint32,
            row_names=row_names,
            column_names=column_names,
            is_sparse=True,
            limits=limits,
        )

    def _validate_peak_order(self) -> None:
        for index in range(1, self.chromosomeIds.size):
            previous = (
                int(self.chromosomeIds[index - 1]),
                int(self.ends[index - 1]),
                int(self.starts[index - 1]),
            )
            current = (
                int(self.chromosomeIds[index]),
                int(self.ends[index]),
                int(self.starts[index]),
            )
            if current < previous:
                raise MatrixSourceError(
                    f"PeakMatrix peaks at {self.objectPath} are not sorted by "
                    "(chr, end, start)"
                )

    def _validate_tile_order(self) -> None:
        for index in range(1, self.chromosomeIds.size):
            previous_chromosome = int(self.chromosomeIds[index - 1])
            chromosome = int(self.chromosomeIds[index])
            if chromosome < previous_chromosome:
                raise MatrixSourceError(
                    f"TileMatrix ranges at {self.objectPath} are not sorted by chromosome"
                )
            if chromosome == previous_chromosome and int(self.ends[index - 1]) > int(
                self.starts[index]
            ):
                raise MatrixSourceError(
                    f"TileMatrix ranges at {self.objectPath} overlap"
                )

    def _build_chromosome_indexes(self) -> None:
        indexes: list[NDArray[np.int64]] = []
        for chromosome_id in range(len(self.chromosomeLevels)):
            selected = np.flatnonzero(self.chromosomeIds == chromosome_id).astype(
                np.int64, copy=False
            )
            if self.matrixType == "PeakMatrix" and selected.size:
                order = np.lexsort(
                    (
                        self.ends[selected],
                        self.starts[selected],
                    )
                )
                selected = selected[order]
            indexes.append(selected)
        self._chromosomeIndexes = tuple(indexes)

    @property
    def resident_bytes(self) -> int:
        return int(
            super().resident_bytes
            + self.fragments.residentBytes
            + self.regionMetadataBytes
        )

    def _selected_peak_indexes(
        self,
        chromosome_id: int,
        start: int,
        stop: int,
    ) -> NDArray[np.int64]:
        indexes = self._chromosomeIndexes[chromosome_id]
        if self.logicalTranspose:
            return indexes
        return indexes[(indexes >= start) & (indexes < stop)]

    def _peak_contributions(
        self,
        start: int,
        stop: int,
    ) -> Iterator[tuple[int, int, int]]:
        for chromosome_id in range(len(self.chromosomeLevels)):
            candidates = self._selected_peak_indexes(chromosome_id, start, stop)
            next_candidate = 0
            active: list[int] = []
            for block in self.fragments.iter_chromosome(chromosome_id):
                if block.size == 0:
                    continue
                block_end_max = int(np.max(block.ends))
                while (
                    next_candidate < candidates.size
                    and int(self.starts[candidates[next_candidate]]) < block_end_max
                ):
                    active.append(int(candidates[next_candidate]))
                    next_candidate += 1
                last_start = int(block.starts[-1])
                if self.logicalTranspose:
                    keep_cells = (block.cellIds >= start) & (block.cellIds < stop)
                    if np.any(keep_cells):
                        cells = block.cellIds[keep_cells]
                        fragment_starts = block.starts[keep_cells]
                        fragment_ends = block.ends[keep_cells]
                    else:
                        cells = np.empty(0, dtype=np.uint32)
                        fragment_starts = np.empty(0, dtype=np.uint32)
                        fragment_ends = np.empty(0, dtype=np.uint32)
                else:
                    cells = block.cellIds
                    fragment_starts = block.starts
                    fragment_ends = block.ends
                retained: list[int] = []
                for feature in active:
                    peak_start = int(self.starts[feature])
                    peak_end = int(self.ends[feature])
                    if cells.size:
                        if self.mode == "overlaps":
                            mask = (fragment_starts < peak_end) & (
                                fragment_ends > peak_start
                            )
                            counts = None
                        else:
                            start_overlap = (fragment_starts >= peak_start) & (
                                fragment_starts < peak_end
                            )
                            end_overlap = (fragment_ends > peak_start) & (
                                fragment_ends <= peak_end
                            )
                            mask = start_overlap | end_overlap
                            counts = (
                                start_overlap.astype(np.uint8)
                                + end_overlap.astype(np.uint8)
                                if self.mode == "insertions"
                                else None
                            )
                        for position in np.flatnonzero(mask):
                            cell_id = int(cells[position])
                            value = int(counts[position]) if counts is not None else 1
                            if self.logicalTranspose:
                                yield cell_id - start, feature, value
                            else:
                                yield feature - start, cell_id, value
                    if last_start < peak_end:
                        retained.append(feature)
                active = retained

    def _tile_region_for_start(
        self,
        chromosome_id: int,
        position: int,
    ) -> int | None:
        regions = self._chromosomeIndexes[chromosome_id]
        if regions.size == 0:
            return None
        first_region = int(regions[0])
        region_starts = self.starts[first_region : first_region + regions.size]
        local = int(np.searchsorted(region_starts, position, side="right") - 1)
        if local < 0:
            return None
        region = first_region + local
        return region if position < int(self.ends[region]) else None

    def _tile_region_for_end(
        self,
        chromosome_id: int,
        position: int,
    ) -> int | None:
        regions = self._chromosomeIndexes[chromosome_id]
        if regions.size == 0:
            return None
        first_region = int(regions[0])
        region_starts = self.starts[first_region : first_region + regions.size]
        local = int(np.searchsorted(region_starts, position, side="left") - 1)
        if local < 0:
            return None
        region = first_region + local
        return region if position <= int(self.ends[region]) else None

    def _tile_feature(self, region: int, position: int, *, end: bool) -> int:
        assert self.tileWidths is not None
        assert self._featureOffsets is not None
        adjusted = position - 1 if end else position
        return int(
            self._featureOffsets[region]
            + (adjusted - int(self.starts[region])) // int(self.tileWidths[region])
        )

    def _tile_contributions(
        self,
        start: int,
        stop: int,
    ) -> Iterator[tuple[int, int, int]]:
        for chromosome_id in range(len(self.chromosomeLevels)):
            for block in self.fragments.iter_chromosome(chromosome_id):
                for position in range(block.size):
                    cell_id = int(block.cellIds[position])
                    if self.logicalTranspose and not start <= cell_id < stop:
                        continue
                    fragment_start = int(block.starts[position])
                    fragment_end = int(block.ends[position])
                    start_region = self._tile_region_for_start(
                        chromosome_id,
                        fragment_start,
                    )
                    end_region = self._tile_region_for_end(
                        chromosome_id,
                        fragment_end,
                    )
                    if start_region is not None:
                        feature = self._tile_feature(
                            start_region,
                            fragment_start,
                            end=False,
                        )
                        if self.logicalTranspose or start <= feature < stop:
                            if self.logicalTranspose:
                                yield cell_id - start, feature, 1
                            else:
                                yield feature - start, cell_id, 1
                    if end_region is not None and (
                        self.mode == "insertions" or end_region != start_region
                    ):
                        feature = self._tile_feature(
                            end_region,
                            fragment_end,
                            end=True,
                        )
                        if self.logicalTranspose or start <= feature < stop:
                            if self.logicalTranspose:
                                yield cell_id - start, feature, 1
                            else:
                                yield feature - start, cell_id, 1

    def _contributions(
        self,
        start: int,
        stop: int,
    ) -> Iterator[tuple[int, int, int]]:
        if self.matrixType == "PeakMatrix":
            yield from self._peak_contributions(start, stop)
        else:
            yield from self._tile_contributions(start, stop)

    def estimate_read_memory(self, start: int, stop: int) -> MemoryEstimate:
        start, stop = self._window(start, stop)
        rows = stop - start
        base_working = self.fragments.blockWorkingBytes + rows * 64
        if self.matrixType == "PeakMatrix":
            base_working += self.n_features * 96
        minimum_output = (rows + 1) * np.dtype(np.int64).itemsize
        if rows == 0:
            return MemoryEstimate(self.resident_bytes, base_working, minimum_output)
        if base_working + minimum_output > self._limits.maxBlockBytes:
            return MemoryEstimate(self.resident_bytes, base_working, minimum_output)
        events = sum(1 for _event in self._contributions(start, stop))
        possible_nnz = min(events, rows * self.n_features)
        output = (
            possible_nnz * (np.dtype(np.uint32).itemsize + np.dtype(np.int64).itemsize)
            + (rows + 1) * np.dtype(np.int64).itemsize
        )
        working = base_working + possible_nnz * _DICT_ENTRY_BYTES
        return MemoryEstimate(self.resident_bytes, working, output)

    def read_cells(self, start: int, stop: int) -> csr_matrix:
        start, stop = self._window(start, stop)
        estimate = self.estimate_read_memory(start, stop)
        self._admit(estimate)
        if start == stop:
            return csr_matrix((0, self.n_features), dtype=np.uint32)
        rows: list[dict[int, int]] = [{} for _index in range(stop - start)]
        nnz = 0
        for row, column, value in self._contributions(start, stop):
            row_values = rows[row]
            previous = row_values.get(column)
            if previous is None:
                if nnz >= self._limits.maxNnz:
                    raise ResourceLimitError(
                        f"fragment-derived block exceeds maxNnz={self._limits.maxNnz}"
                    )
                row_values[column] = value
                nnz += 1
                dynamic_bytes = (
                    self.fragments.blockWorkingBytes
                    + nnz
                    * (
                        _DICT_ENTRY_BYTES
                        + np.dtype(np.uint32).itemsize
                        + np.dtype(np.int64).itemsize
                    )
                    + (len(rows) + 1) * np.dtype(np.int64).itemsize
                )
                if dynamic_bytes > self._limits.maxBlockBytes:
                    raise ResourceLimitError(
                        "fragment-derived block exceeds "
                        f"maxBlockBytes={self._limits.maxBlockBytes}"
                    )
            else:
                updated = previous + value
                if updated > _UINT32_MAX:
                    raise MatrixSourceError("fragment-derived count overflows uint32")
                row_values[column] = updated
        data = np.empty(nnz, dtype=np.uint32)
        indices = np.empty(nnz, dtype=np.int64)
        indptr = np.empty(len(rows) + 1, dtype=np.int64)
        indptr[0] = 0
        position = 0
        for row_index, row_values in enumerate(rows):
            for column, value in row_values.items():
                data[position] = value
                indices[position] = column
                position += 1
            indptr[row_index + 1] = position
        return csr_matrix(
            (data, indices, indptr),
            shape=(stop - start, self.n_features),
            dtype=np.uint32,
            copy=False,
        )


def build_fragment_matrix_source(
    specification: Mapping[str, Any],
    *,
    object_path: str = "$",
    limits: SourceLimits = DEFAULT_LIMITS,
) -> FragmentDerivedMatrixSource:
    class_value = specification.get(
        "matrixType",
        specification.get(
            "matrix_type",
            specification.get("className", specification.get("class")),
        ),
    )
    classes = _class_names(class_value)
    matrix_type = classes[0] if classes else ""
    fragments = specification.get("fragments")
    if not isinstance(fragments, FragmentSource):
        raise TypeError(
            f"fragment input at {object_path}@fragments is not a FragmentSource"
        )
    required = {
        "chromosome_ids": specification.get(
            "chrId",
            specification.get("chr_id"),
        ),
        "starts": specification.get("start"),
        "ends": specification.get("end"),
        "chromosome_levels": specification.get(
            "chrLevels",
            specification.get("chr_levels"),
        ),
        "mode": specification.get("mode"),
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise MatrixSourceError(
            f"{matrix_type or 'fragment matrix'} at {object_path} is missing "
            f"{', '.join(missing)}"
        )
    return FragmentDerivedMatrixSource(
        fragments,
        matrix_type=matrix_type,
        chromosome_ids=required["chromosome_ids"],
        starts=required["starts"],
        ends=required["ends"],
        chromosome_levels=required["chromosome_levels"],
        mode=required["mode"],
        tile_widths=specification.get(
            "tileWidths",
            specification.get("tile_width"),
        ),
        transpose=specification.get("transpose", True),
        shape=specification.get(
            "shape",
            specification.get("Dim", specification.get("dim")),
        ),
        row_names=specification.get(
            "rowNames",
            specification.get("row_names"),
        ),
        column_names=specification.get(
            "columnNames",
            specification.get("column_names"),
        ),
        object_path=object_path,
        limits=limits,
    )
