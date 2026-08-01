import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import h5py
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix

from .errors import MatrixSourceError, ResourceLimitError, UnsafeSidecarError
from .paths import require_hdf5_group, validate_hdf5_file
from .sources import (
    DEFAULT_LIMITS,
    BaseMatrixSource,
    MemoryEstimate,
    SourceLimits,
    _validate_shape,
)


_NUMERIC_HEADERS: dict[bytes, np.dtype[Any]] = {
    b"UINT32v1": np.dtype("<u4"),
    b"UINT64v1": np.dtype("<u8"),
    b"FLOATSv1": np.dtype("<f4"),
    b"DOUBLEv1": np.dtype("<f8"),
}
_VERSION_PATTERN = re.compile(r"^(packed|unpacked)-(uint|float|double)-matrix-v([12])$")


def _unpack_bp128_block(
    words: NDArray[Any],
    bits: int,
) -> NDArray[np.uint32]:
    if bits < 0 or bits > 32:
        raise MatrixSourceError(f"BP128 bit width {bits} is outside [0, 32]")
    expected = bits * 4
    packed = np.asarray(words, dtype=np.uint32).reshape(-1)
    if packed.size != expected:
        raise MatrixSourceError(
            f"BP128 block has {packed.size} words; expected {expected}"
        )
    output: NDArray[np.uint32] = np.zeros(128, dtype=np.uint32)
    if bits == 0:
        return output
    vectors = packed.reshape(bits, 4).astype(np.uint64)
    mask = np.uint64(0xFFFFFFFF if bits == 32 else (1 << bits) - 1)
    for vector_index in range(32):
        bit_position = vector_index * bits
        word_index = bit_position // 32
        shift = bit_position & 31
        values = vectors[word_index] >> np.uint64(shift)
        if shift + bits > 32:
            values |= vectors[word_index + 1] << np.uint64(32 - shift)
        output[vector_index * 4 : vector_index * 4 + 4] = (values & mask).astype(
            np.uint32
        )
    return output


def _expanded_bp128_indexes(
    indexes: NDArray[Any],
    index_offsets: NDArray[Any] | None,
) -> NDArray[np.uint64]:
    raw = np.asarray(indexes)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
        raise TypeError("BP128 idx must be a one-dimensional integer array")
    raw_u64 = raw.astype(np.uint64, copy=False)
    if index_offsets is None:
        expanded = raw_u64.copy()
    else:
        offsets = np.asarray(index_offsets)
        if offsets.ndim != 1 or not np.issubdtype(offsets.dtype, np.integer):
            raise TypeError("BP128 idx_offsets must be a one-dimensional integer array")
        offsets = offsets.astype(np.int64, copy=False)
        if (
            offsets.size < 2
            or int(offsets[0]) != 0
            or int(offsets[-1]) != raw.size
            or np.any(offsets[1:] <= offsets[:-1])
        ):
            raise MatrixSourceError(
                "BP128 idx_offsets must partition the complete idx array"
            )
        expanded = raw_u64.copy()
        for segment in range(1, offsets.size - 1):
            start = int(offsets[segment])
            stop = int(offsets[segment + 1])
            expanded[start:stop] += np.uint64(segment) << np.uint64(32)
    if expanded.size and int(expanded[0]) != 0:
        raise MatrixSourceError("BP128 idx must start at zero")
    if expanded.size > 1 and np.any(expanded[1:] < expanded[:-1]):
        raise MatrixSourceError("expanded BP128 idx must be nondecreasing")
    return expanded


def decode_bp128(
    data: NDArray[Any],
    indexes: NDArray[Any],
    count: int,
    *,
    index_offsets: NDArray[Any] | None = None,
    transform: str = "plain",
    starts: NDArray[Any] | None = None,
    start: int = 0,
    stop: int | None = None,
) -> NDArray[np.uint32]:
    count = int(count)
    if count < 0:
        raise ValueError("BP128 count cannot be negative")
    stop = count if stop is None else int(stop)
    start = int(start)
    if start < 0 or stop < start or stop > count:
        raise IndexError(f"BP128 window [{start}, {stop}) is outside [0, {count})")
    block_count = math.ceil(count / 128)
    expanded = _expanded_bp128_indexes(indexes, index_offsets)
    if expanded.shape != (block_count + 1,):
        raise MatrixSourceError(
            f"BP128 idx has length {expanded.size}; expected {block_count + 1}"
        )
    packed = np.asarray(data)
    if packed.ndim != 1 or not np.issubdtype(packed.dtype, np.integer):
        raise TypeError("BP128 data must be a one-dimensional integer array")
    if int(expanded[-1]) != packed.size:
        raise MatrixSourceError(
            f"BP128 data has {packed.size} words; idx ends at {int(expanded[-1])}"
        )
    if transform not in {"plain", "m1", "d1", "d1z"}:
        raise ValueError(f"unknown BP128 transform {transform!r}")
    start_values: NDArray[Any] | None = None
    if transform in {"d1", "d1z"}:
        if starts is None:
            raise MatrixSourceError(f"BP128 {transform} requires starts")
        start_values = np.asarray(starts)
        if (
            start_values.ndim != 1
            or start_values.size != block_count
            or not np.issubdtype(start_values.dtype, np.integer)
        ):
            raise MatrixSourceError(f"BP128 starts must have length {block_count}")
    if start == stop:
        return np.empty(0, dtype=np.uint32)
    first_block = start // 128
    final_block = (stop - 1) // 128
    decoded_blocks: list[NDArray[np.uint32]] = []
    for block in range(first_block, final_block + 1):
        word_start = int(expanded[block])
        word_stop = int(expanded[block + 1])
        word_count = word_stop - word_start
        if word_count % 4:
            raise MatrixSourceError(
                f"BP128 block {block} word count {word_count} is not divisible by four"
            )
        values = _unpack_bp128_block(packed[word_start:word_stop], word_count // 4)
        if transform == "m1":
            widened = values.astype(np.uint64) + 1
            if np.any(widened > np.iinfo(np.uint32).max):
                raise MatrixSourceError("BP128 m1 decode overflows uint32")
            values = widened.astype(np.uint32)
        elif transform in {"d1", "d1z"}:
            assert start_values is not None
            if transform == "d1z":
                encoded = values.astype(np.uint64)
                deltas = (encoded >> np.uint64(1)).astype(np.int64) ^ -(
                    (encoded & np.uint64(1)).astype(np.int64)
                )
            else:
                deltas = values.astype(np.int64)
            decoded = np.cumsum(deltas, dtype=np.int64)
            decoded += int(start_values[block])
            if np.any(decoded < 0) or np.any(decoded > np.iinfo(np.uint32).max):
                raise MatrixSourceError(f"BP128 {transform} decode leaves uint32 range")
            values = decoded.astype(np.uint32)
        decoded_blocks.append(values)
    combined = np.concatenate(decoded_blocks)
    local_start = start - first_block * 128
    return combined[local_start : local_start + stop - start]


def decode_bp128_m1(
    data: NDArray[Any],
    indexes: NDArray[Any],
    count: int,
    *,
    index_offsets: NDArray[Any] | None = None,
    start: int = 0,
    stop: int | None = None,
) -> NDArray[np.uint32]:
    return decode_bp128(
        data,
        indexes,
        count,
        index_offsets=index_offsets,
        transform="m1",
        start=start,
        stop=stop,
    )


def decode_bp128_d1z(
    data: NDArray[Any],
    indexes: NDArray[Any],
    starts: NDArray[Any],
    count: int,
    *,
    index_offsets: NDArray[Any] | None = None,
    start: int = 0,
    stop: int | None = None,
) -> NDArray[np.uint32]:
    return decode_bp128(
        data,
        indexes,
        count,
        index_offsets=index_offsets,
        transform="d1z",
        starts=starts,
        start=start,
        stop=stop,
    )


class _BPArrayStore(Protocol):
    @property
    def version(self) -> str: ...

    def has(self, name: str) -> bool: ...

    def numeric_info(self, name: str) -> tuple[np.dtype[Any], int]: ...

    def read_numeric(
        self,
        name: str,
        start: int = 0,
        stop: int | None = None,
    ) -> NDArray[Any]: ...

    def read_text(self, name: str) -> tuple[str, ...]: ...


def _read_vector_window(
    values: Any,
    start: int,
    stop: int,
) -> NDArray[Any]:
    read_block = getattr(values, "read_block", None)
    if callable(read_block):
        return np.asarray(read_block(start, stop))
    return np.asarray(values[start:stop])


class _MemoryArrayStore:
    def __init__(
        self,
        version: str,
        arrays: Mapping[str, Any],
        dtypes: Mapping[str, np.dtype[Any]],
        *,
        float_bit_arrays: frozenset[str],
        text: Mapping[str, Sequence[str | bytes] | NDArray[Any] | None],
        limits: SourceLimits,
    ) -> None:
        self._version = version
        self._arrays = dict(arrays)
        self._dtypes = dict(dtypes)
        self._floatBitArrays = float_bit_arrays
        self._text = dict(text)
        self._limits = limits

    @property
    def version(self) -> str:
        return self._version

    def has(self, name: str) -> bool:
        return name in self._arrays or (
            name in self._text and self._text[name] is not None
        )

    def numeric_info(self, name: str) -> tuple[np.dtype[Any], int]:
        if name not in self._arrays or name not in self._dtypes:
            raise MatrixSourceError(f"BPCells memory array {name!r} is missing")
        return self._dtypes[name], len(self._arrays[name])

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
        values = _read_vector_window(self._arrays[name], start, stop)
        if name in self._floatBitArrays:
            if values.dtype.kind not in "iu" or values.dtype.itemsize != 4:
                raise TypeError(
                    f"BPCells memory array {name!r} must contain 32-bit "
                    "floating-point bit patterns"
                )
            return np.ascontiguousarray(values, dtype=np.uint32).view(np.float32)
        if dtype.kind in "ui":
            if not np.issubdtype(values.dtype, np.number):
                raise TypeError(f"BPCells memory array {name!r} must be numeric")
            if values.dtype.kind == "f" and (
                np.any(~np.isfinite(values))
                or np.any(values != np.floor(values))
                or np.any(values < 0)
            ):
                raise MatrixSourceError(
                    f"BPCells memory array {name!r} contains invalid integer values"
                )
        return values.astype(dtype, copy=False)

    def read_text(self, name: str) -> tuple[str, ...]:
        values = self._text.get(name)
        if values is None:
            raise MatrixSourceError(f"BPCells memory text array {name!r} is missing")
        output: list[str] = []
        used_bytes = 0
        for start in range(0, len(values), 4096):
            stop = min(len(values), start + 4096)
            read_block = getattr(values, "read_block", None)
            block = (
                read_block(start, stop) if callable(read_block) else values[start:stop]
            )
            for value in block:
                if isinstance(value, bytes):
                    try:
                        decoded = value.decode("utf-8")
                    except UnicodeDecodeError as error:
                        raise MatrixSourceError(
                            f"BPCells memory text array {name!r} is not valid UTF-8"
                        ) from error
                elif isinstance(value, str):
                    decoded = value
                else:
                    raise TypeError(
                        f"BPCells memory text array {name!r} must contain strings"
                    )
                used_bytes += len(decoded.encode("utf-8")) + 8
                if used_bytes > self._limits.maxMetadataBytes:
                    raise ResourceLimitError(
                        f"BPCells memory text array {name!r} exceeds "
                        f"maxMetadataBytes={self._limits.maxMetadataBytes}"
                    )
                output.append(decoded)
        return tuple(output)


class _DirectoryArrayStore:
    def __init__(
        self,
        path: str | os.PathLike[str],
        limits: SourceLimits,
    ) -> None:
        self.path = Path(path).expanduser().resolve(strict=False)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        if not self.path.is_dir():
            raise UnsafeSidecarError(
                f"BPCells directory {self.path} is not a directory"
            )
        self._limits = limits
        values = self.read_text("version")
        if len(values) != 1:
            raise MatrixSourceError(
                "BPCells directory version must contain exactly one line"
            )
        self._version = values[0]

    @property
    def version(self) -> str:
        return self._version

    def _child(self, name: str) -> Path:
        if "/" in name or name in {"", ".", ".."}:
            raise UnsafeSidecarError(f"invalid BPCells array name {name!r}")
        child = self.path / name
        resolved = child.resolve(strict=False)
        if not (resolved == self.path or resolved.is_relative_to(self.path)):
            raise UnsafeSidecarError(f"BPCells array {name!r} escapes its directory")
        return resolved

    def has(self, name: str) -> bool:
        child = self._child(name)
        return child.exists() and child.is_file()

    def numeric_info(self, name: str) -> tuple[np.dtype[Any], int]:
        child = self._child(name)
        if not child.exists():
            raise MatrixSourceError(f"BPCells numeric array {name!r} is missing")
        if not child.is_file():
            raise UnsafeSidecarError(
                f"BPCells numeric array {name!r} is not a regular file"
            )
        with child.open("rb") as handle:
            header = handle.read(8)
        if header not in _NUMERIC_HEADERS:
            raise MatrixSourceError(
                f"BPCells numeric array {name!r} has unknown 8-byte header {header!r}"
            )
        dtype = _NUMERIC_HEADERS[header]
        payload_bytes = child.stat().st_size - 8
        if payload_bytes < 0 or payload_bytes % dtype.itemsize:
            raise MatrixSourceError(
                f"BPCells numeric array {name!r} has a truncated payload"
            )
        return dtype, payload_bytes // dtype.itemsize

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
                f"BPCells array {name!r} window [{start}, {stop}) "
                f"is outside [0, {length})"
            )
        child = self._child(name)
        with child.open("rb") as handle:
            handle.seek(8 + start * dtype.itemsize)
            payload = handle.read((stop - start) * dtype.itemsize)
        if len(payload) != (stop - start) * dtype.itemsize:
            raise MatrixSourceError(f"short read from BPCells numeric array {name!r}")
        return np.frombuffer(payload, dtype=dtype).copy()

    def read_text(self, name: str) -> tuple[str, ...]:
        child = self._child(name)
        if not child.exists():
            raise MatrixSourceError(f"BPCells text array {name!r} is missing")
        if not child.is_file():
            raise UnsafeSidecarError(
                f"BPCells text array {name!r} is not a regular file"
            )
        size = child.stat().st_size
        if size > self._limits.maxMetadataBytes:
            raise ResourceLimitError(
                f"BPCells text array {name!r} exceeds "
                f"maxMetadataBytes={self._limits.maxMetadataBytes}"
            )
        try:
            text = child.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise MatrixSourceError(
                f"BPCells text array {name!r} is not valid UTF-8"
            ) from error
        return tuple(text.splitlines())


class _HDF5ArrayStore:
    def __init__(
        self,
        path: str | os.PathLike[str] | Any,
        group: str,
        limits: SourceLimits,
    ) -> None:
        self.path = validate_hdf5_file(path, limits=limits)
        self.group = "/" + group.strip("/")
        self._limits = limits
        with h5py.File(self.path, mode="r") as handle:
            node = require_hdf5_group(handle, self.group)
            if "version" not in node.attrs:
                raise MatrixSourceError(
                    f"BPCells HDF5 group {self.group!r} has no version attribute"
                )
            value = np.asarray(node.attrs["version"]).reshape(-1)
            if value.size != 1:
                raise MatrixSourceError("BPCells HDF5 version attribute must be scalar")
            scalar = value[0]
            if isinstance(scalar, bytes | np.bytes_):
                self._version = bytes(scalar).decode("utf-8")
            elif isinstance(scalar, str | np.str_):
                self._version = str(scalar)
            else:
                raise MatrixSourceError("BPCells HDF5 version attribute must be text")

    @property
    def version(self) -> str:
        return self._version

    def has(self, name: str) -> bool:
        with h5py.File(self.path, mode="r") as handle:
            group = require_hdf5_group(handle, self.group)
            return name in group and isinstance(group[name], h5py.Dataset)

    def numeric_info(self, name: str) -> tuple[np.dtype[Any], int]:
        with h5py.File(self.path, mode="r") as handle:
            group = require_hdf5_group(handle, self.group)
            if name not in group or not isinstance(group[name], h5py.Dataset):
                raise MatrixSourceError(
                    f"BPCells HDF5 numeric array {name!r} is missing"
                )
            node = group[name]
            assert isinstance(node, h5py.Dataset)
            if node.ndim != 1 or node.dtype.kind not in "uif":
                raise TypeError(
                    f"BPCells HDF5 numeric array {name!r} must be numeric and "
                    "one-dimensional"
                )
            return np.dtype(node.dtype), int(node.size)

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
                f"BPCells HDF5 array {name!r} window [{start}, {stop}) "
                f"is outside [0, {length})"
            )
        with h5py.File(self.path, mode="r") as handle:
            group = require_hdf5_group(handle, self.group)
            node = group[name]
            assert isinstance(node, h5py.Dataset)
            return np.asarray(node[start:stop], dtype=dtype)

    def read_text(self, name: str) -> tuple[str, ...]:
        with h5py.File(self.path, mode="r") as handle:
            group = require_hdf5_group(handle, self.group)
            if name not in group or not isinstance(group[name], h5py.Dataset):
                raise MatrixSourceError(f"BPCells HDF5 text array {name!r} is missing")
            node = group[name]
            assert isinstance(node, h5py.Dataset)
            if node.ndim > 1:
                raise MatrixSourceError(
                    f"BPCells HDF5 text array {name!r} must be scalar or 1D"
                )
            string_info = h5py.check_string_dtype(node.dtype)
            if string_info is None:
                raise MatrixSourceError(
                    f"BPCells HDF5 text array {name!r} must contain strings"
                )
            count = int(node.size)
            if count * 8 > self._limits.maxMetadataBytes:
                raise ResourceLimitError(
                    f"BPCells HDF5 text array {name!r} exceeds "
                    f"maxMetadataBytes={self._limits.maxMetadataBytes}"
                )
            if (
                string_info.length is not None
                and count * (string_info.length + 8) > self._limits.maxMetadataBytes
            ):
                raise ResourceLimitError(
                    f"BPCells HDF5 text array {name!r} exceeds "
                    f"maxMetadataBytes={self._limits.maxMetadataBytes}"
                )
            output: list[str] = []
            size = 0
            for start in range(0, count, 4096):
                stop = min(count, start + 4096)
                values = np.asarray(
                    node[()] if node.ndim == 0 else node[start:stop]
                ).reshape(-1)
                for value in values:
                    if isinstance(value, bytes | np.bytes_):
                        try:
                            decoded = bytes(value).decode("utf-8")
                        except UnicodeDecodeError as error:
                            raise MatrixSourceError(
                                f"BPCells HDF5 text array {name!r} is not valid UTF-8"
                            ) from error
                    elif isinstance(value, str | np.str_):
                        decoded = str(value)
                    else:
                        raise MatrixSourceError(
                            f"BPCells HDF5 text array {name!r} must contain strings"
                        )
                    size += len(decoded.encode("utf-8")) + 8
                    if size > self._limits.maxMetadataBytes:
                        raise ResourceLimitError(
                            f"BPCells HDF5 text array {name!r} exceeds "
                            f"maxMetadataBytes={self._limits.maxMetadataBytes}"
                        )
                    output.append(decoded)
        return tuple(output)


def _require_numeric_array(
    store: _BPArrayStore,
    name: str,
    dtype: np.dtype[Any],
    length: int | None = None,
) -> int:
    actual_dtype, actual_length = store.numeric_info(name)
    if actual_dtype.kind != dtype.kind or actual_dtype.itemsize != dtype.itemsize:
        raise MatrixSourceError(
            f"BPCells array {name!r} has dtype {actual_dtype}; expected {dtype}"
        )
    if length is not None and actual_length != length:
        raise MatrixSourceError(
            f"BPCells array {name!r} has length {actual_length}; expected {length}"
        )
    return actual_length


class _StoredBP128Array:
    def __init__(
        self,
        store: _BPArrayStore,
        prefix: str,
        count: int,
        transform: str,
        *,
        require_offsets: bool,
        limits: SourceLimits,
    ) -> None:
        self.store = store
        self.prefix = prefix
        self.count = int(count)
        self.transform = transform
        self.blockCount = math.ceil(self.count / 128)
        self.limits = limits
        _require_numeric_array(
            store, f"{prefix}_idx", np.dtype("uint32"), self.blockCount + 1
        )
        self.dataLength = _require_numeric_array(
            store, f"{prefix}_data", np.dtype("uint32")
        )
        offset_name = f"{prefix}_idx_offsets"
        if require_offsets:
            if not store.has(offset_name):
                raise MatrixSourceError(
                    f"BPCells packed v2 array {prefix!r} has no idx_offsets"
                )
            offset_length = _require_numeric_array(
                store, offset_name, np.dtype("uint64")
            )
            if offset_length * np.dtype(np.uint64).itemsize > limits.maxMetadataBytes:
                raise ResourceLimitError(
                    f"BPCells {offset_name} exceeds metadata limit"
                )
            self.indexOffsets = store.read_numeric(offset_name).astype(
                np.int64, copy=False
            )
        else:
            self.indexOffsets = np.array([0, self.blockCount + 1], dtype=np.int64)
        if (
            self.indexOffsets.size < 2
            or int(self.indexOffsets[0]) != 0
            or int(self.indexOffsets[-1]) != self.blockCount + 1
            or np.any(self.indexOffsets[1:] <= self.indexOffsets[:-1])
        ):
            raise MatrixSourceError(f"BPCells {offset_name} does not partition idx")
        if transform in {"d1", "d1z"}:
            _require_numeric_array(
                store,
                f"{prefix}_starts",
                np.dtype("uint32"),
                self.blockCount,
            )
        self._validate_index_end()

    def _idx_high(self, position: int) -> int:
        boundaries = self.indexOffsets[1:-1]
        return int(np.searchsorted(boundaries, position, side="right")) << 32

    def _expanded_idx_pair(self, block: int) -> tuple[int, int]:
        raw = self.store.read_numeric(f"{self.prefix}_idx", block, block + 2).astype(
            np.uint64, copy=False
        )
        first = int(raw[0]) + self._idx_high(block)
        second = int(raw[1]) + self._idx_high(block + 1)
        if second < first:
            raise MatrixSourceError(
                f"BPCells {self.prefix}_idx decreases at block {block}"
            )
        return first, second

    def _validate_index_end(self) -> None:
        first = self.store.read_numeric(f"{self.prefix}_idx", 0, 1)
        if first.size != 1 or int(first[0]) != 0:
            raise MatrixSourceError(f"BPCells {self.prefix}_idx must start at zero")
        raw_end = self.store.read_numeric(
            f"{self.prefix}_idx", self.blockCount, self.blockCount + 1
        )
        expanded_end = int(raw_end[0]) + self._idx_high(self.blockCount)
        if expanded_end != self.dataLength:
            raise MatrixSourceError(
                f"BPCells {self.prefix}_idx ends at {expanded_end}; "
                f"data has length {self.dataLength}"
            )

    def read(self, start: int, stop: int) -> NDArray[np.uint32]:
        start = int(start)
        stop = int(stop)
        if start < 0 or stop < start or stop > self.count:
            raise IndexError(
                f"BPCells packed window [{start}, {stop}) is outside [0, {self.count})"
            )
        if start == stop:
            return np.empty(0, dtype=np.uint32)
        first_block = start // 128
        final_block = (stop - 1) // 128
        output: list[NDArray[np.uint32]] = []
        for block in range(first_block, final_block + 1):
            word_start, word_stop = self._expanded_idx_pair(block)
            word_count = word_stop - word_start
            if word_count % 4:
                raise MatrixSourceError(
                    f"BPCells {self.prefix} block {block} has invalid word count"
                )
            values = _unpack_bp128_block(
                self.store.read_numeric(f"{self.prefix}_data", word_start, word_stop),
                word_count // 4,
            )
            if self.transform == "m1":
                widened = values.astype(np.uint64) + 1
                if np.any(widened > np.iinfo(np.uint32).max):
                    raise MatrixSourceError("BP128 m1 decode overflows uint32")
                values = widened.astype(np.uint32)
            elif self.transform in {"d1", "d1z"}:
                encoded = values.astype(np.uint64)
                if self.transform == "d1z":
                    deltas = (encoded >> np.uint64(1)).astype(np.int64) ^ -(
                        (encoded & np.uint64(1)).astype(np.int64)
                    )
                else:
                    deltas = encoded.astype(np.int64)
                start_value = int(
                    self.store.read_numeric(f"{self.prefix}_starts", block, block + 1)[
                        0
                    ]
                )
                decoded = np.cumsum(deltas, dtype=np.int64) + start_value
                if np.any(decoded < 0) or np.any(decoded > np.iinfo(np.uint32).max):
                    raise MatrixSourceError(
                        f"BP128 {self.transform} decode leaves uint32 range"
                    )
                values = decoded.astype(np.uint32)
            output.append(values)
        combined = np.concatenate(output)
        local_start = start - first_block * 128
        return combined[local_start : local_start + stop - start]


class BPCellsMatrixSource(BaseMatrixSource):
    def __init__(
        self,
        store: _BPArrayStore,
        *,
        limits: SourceLimits = DEFAULT_LIMITS,
    ) -> None:
        match = _VERSION_PATTERN.fullmatch(store.version)
        if match is None:
            raise MatrixSourceError(
                f"unsupported BPCells matrix format {store.version!r}"
            )
        compression, datatype, version_text = match.groups()
        self.store = store
        self.compression = compression
        self.datatype = datatype
        self.formatVersion = int(version_text)
        _require_numeric_array(store, "shape", np.dtype("uint32"), 2)
        raw_shape = store.read_numeric("shape")
        shape = _validate_shape(
            (int(raw_shape[0]), int(raw_shape[1])),
            limits,
        )
        order_values = store.read_text("storage_order")
        if len(order_values) != 1 or order_values[0] not in {"row", "col"}:
            raise MatrixSourceError(
                "BPCells storage_order must contain exactly 'row' or 'col'"
            )
        self.storageOrder = order_values[0]
        compressed_axis = shape[1] if self.storageOrder == "col" else shape[0]
        pointer_dtype = np.dtype("uint32" if self.formatVersion == 1 else "uint64")
        _require_numeric_array(store, "idxptr", pointer_dtype, compressed_axis + 1)
        self._nnz = self._validate_pointers(compressed_axis, limits)
        if self._nnz > limits.maxNnz:
            raise ResourceLimitError(
                f"BPCells nnz {self._nnz} exceeds maxNnz={limits.maxNnz}"
            )
        matrix_dtype: np.dtype[Any]
        self._indexReader: _StoredBP128Array | None
        self._valueReader: _StoredBP128Array | None
        if datatype == "uint":
            matrix_dtype = np.dtype(np.uint32)
        elif datatype == "float":
            matrix_dtype = np.dtype(np.float32)
        else:
            matrix_dtype = np.dtype(np.float64)
        if compression == "packed":
            self._indexReader = _StoredBP128Array(
                store,
                "index",
                self._nnz,
                "d1z",
                require_offsets=self.formatVersion == 2,
                limits=limits,
            )
            self._valueReader = (
                _StoredBP128Array(
                    store,
                    "val",
                    self._nnz,
                    "m1",
                    require_offsets=self.formatVersion == 2,
                    limits=limits,
                )
                if datatype == "uint"
                else None
            )
            if datatype != "uint":
                _require_numeric_array(store, "val", matrix_dtype, self._nnz)
        else:
            self._indexReader = None
            self._valueReader = None
            _require_numeric_array(store, "index", np.dtype("uint32"), self._nnz)
            _require_numeric_array(store, "val", matrix_dtype, self._nnz)
        row_names = self._optional_names("row_names", shape[0])
        column_names = self._optional_names("col_names", shape[1])
        super().__init__(
            shape,
            matrix_dtype,
            row_names=row_names,
            column_names=column_names,
            is_sparse=True,
            limits=limits,
        )
        self._validate_indexes(limits)

    def _optional_names(
        self,
        name: str,
        length: int,
    ) -> tuple[str, ...] | None:
        if not self.store.has(name):
            return None
        values = self.store.read_text(name)
        if len(values) == 0:
            return None
        if len(values) != length:
            raise MatrixSourceError(
                f"BPCells {name} has length {len(values)}; expected {length}"
            )
        return values

    def _validate_pointers(
        self,
        compressed_axis: int,
        limits: SourceLimits,
    ) -> int:
        previous: int | None = None
        final = 0
        chunk = max(1, min(compressed_axis + 1, limits.compressedChunkNnz))
        for start in range(0, compressed_axis + 1, chunk):
            stop = min(compressed_axis + 1, start + chunk)
            values = self.store.read_numeric("idxptr", start, stop)
            if values.size > 1 and np.any(values[1:] < values[:-1]):
                raise MatrixSourceError("BPCells idxptr must be nondecreasing")
            if previous is not None and values.size and int(values[0]) < previous:
                raise MatrixSourceError("BPCells idxptr must be nondecreasing")
            if start == 0 and (not values.size or int(values[0]) != 0):
                raise MatrixSourceError("BPCells idxptr must start at zero")
            if values.size:
                previous = int(values[-1])
                final = previous
        return final

    def _read_indexes(self, start: int, stop: int) -> NDArray[np.uint32]:
        if self._indexReader is not None:
            return self._indexReader.read(start, stop)
        return self.store.read_numeric("index", start, stop).astype(
            np.uint32, copy=False
        )

    def _read_values(self, start: int, stop: int) -> NDArray[Any]:
        if self._valueReader is not None:
            return self._valueReader.read(start, stop)
        return self.store.read_numeric("val", start, stop).astype(
            self.dtype, copy=False
        )

    def _validate_indexes(self, limits: SourceLimits) -> None:
        minor_axis = self.shape[0] if self.storageOrder == "col" else self.shape[1]
        for start in range(0, self._nnz, limits.compressedChunkNnz):
            stop = min(self._nnz, start + limits.compressedChunkNnz)
            indexes = self._read_indexes(start, stop)
            if indexes.size and np.any(indexes >= minor_axis):
                raise MatrixSourceError("BPCells index contains an out-of-range value")

    @property
    def nnz(self) -> int:
        return self._nnz

    @property
    def resident_bytes(self) -> int:
        offsets = 0
        if self._indexReader is not None:
            offsets += self._indexReader.indexOffsets.nbytes
        if self._valueReader is not None:
            offsets += self._valueReader.indexOffsets.nbytes
        return int(super().resident_bytes + offsets)

    def _bounds(
        self,
        start: int,
        stop: int,
    ) -> tuple[NDArray[np.int64], int, int]:
        pointers = self.store.read_numeric("idxptr", start, stop + 1).astype(
            np.int64, copy=False
        )
        data_start = int(pointers[0])
        data_stop = int(pointers[-1])
        return pointers - data_start, data_start, data_stop

    def estimate_read_memory(self, start: int, stop: int) -> MemoryEstimate:
        start, stop = self._window(start, stop)
        index_size = np.dtype(np.int64).itemsize
        if self.storageOrder == "col":
            pointers, data_start, data_stop = self._bounds(start, stop)
            nnz = data_stop - data_start
            output = nnz * (self.dtype.itemsize + index_size) + pointers.nbytes
            working = output + 128 * np.dtype(np.uint32).itemsize
            return MemoryEstimate(self.resident_bytes, working, output)
        max_output_nnz = min(self.nnz, (stop - start) * self.n_features)
        output = max_output_nnz * (self.dtype.itemsize + 2 * index_size)
        output += (stop - start + 1) * index_size
        working = min(self.nnz, self._limits.compressedChunkNnz) * (
            self.dtype.itemsize + 2 * index_size
        )
        return MemoryEstimate(self.resident_bytes, working, output)

    def read_cells(self, start: int, stop: int) -> csr_matrix:
        start, stop = self._window(start, stop)
        estimate = self.estimate_read_memory(start, stop)
        self._admit(estimate)
        if self.storageOrder == "col":
            pointers, data_start, data_stop = self._bounds(start, stop)
            return csr_matrix(
                (
                    self._read_values(data_start, data_stop),
                    self._read_indexes(data_start, data_stop).astype(
                        np.int64, copy=False
                    ),
                    pointers,
                ),
                shape=(stop - start, self.n_features),
                dtype=self.dtype,
            )
        return self._read_row_stored(start, stop)

    def _read_row_stored(self, start: int, stop: int) -> csr_matrix:
        if start == stop:
            return csr_matrix((0, self.n_features), dtype=self.dtype)
        data_parts: list[NDArray[Any]] = []
        row_parts: list[NDArray[np.int64]] = []
        column_parts: list[NDArray[np.int64]] = []
        retained = 0
        for feature in range(self.n_features):
            bounds = self.store.read_numeric("idxptr", feature, feature + 2)
            vector_start = int(bounds[0])
            vector_stop = int(bounds[1])
            for chunk_start in range(
                vector_start,
                vector_stop,
                self._limits.compressedChunkNnz,
            ):
                chunk_stop = min(
                    vector_stop,
                    chunk_start + self._limits.compressedChunkNnz,
                )
                cells = self._read_indexes(chunk_start, chunk_stop).astype(
                    np.int64, copy=False
                )
                keep = (cells >= start) & (cells < stop)
                count = int(np.count_nonzero(keep))
                if count == 0:
                    continue
                retained += count
                required = retained * (
                    self.dtype.itemsize + 2 * np.dtype(np.int64).itemsize
                )
                if required > self._limits.maxBlockBytes:
                    raise ResourceLimitError(
                        "BPCells sparse block exceeds "
                        f"maxBlockBytes={self._limits.maxBlockBytes}"
                    )
                data_parts.append(self._read_values(chunk_start, chunk_stop)[keep])
                row_parts.append(cells[keep] - start)
                column_parts.append(np.full(count, feature, dtype=np.int64))
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


class BPCellsMemoryMatrixSource(BPCellsMatrixSource):
    def __init__(
        self,
        version: str,
        arrays: Mapping[str, Any],
        *,
        shape: tuple[int, int],
        storage_order: str,
        row_names: Sequence[str | bytes] | NDArray[Any] | None = None,
        column_names: Sequence[str | bytes] | NDArray[Any] | None = None,
        float_bit_arrays: frozenset[str] = frozenset(),
        limits: SourceLimits = DEFAULT_LIMITS,
    ) -> None:
        match = _VERSION_PATTERN.fullmatch(version)
        if match is None:
            raise MatrixSourceError(f"unsupported BPCells matrix format {version!r}")
        compression, datatype, version_text = match.groups()
        if storage_order not in {"row", "col"}:
            raise MatrixSourceError("BPCells memory storage order must be row or col")
        pointer_dtype = np.dtype("uint32" if int(version_text) == 1 else "uint64")
        value_dtypes: dict[str, np.dtype[Any]] = {
            "uint": np.dtype(np.uint32),
            "float": np.dtype(np.float32),
            "double": np.dtype(np.float64),
        }
        value_dtype = value_dtypes[datatype]
        normalized_arrays = {
            **arrays,
            "shape": np.asarray(shape, dtype=np.uint32),
        }
        dtypes: dict[str, np.dtype[Any]] = {
            "shape": np.dtype(np.uint32),
            "idxptr": pointer_dtype,
            "index": np.dtype(np.uint32),
            "index_data": np.dtype(np.uint32),
            "index_starts": np.dtype(np.uint32),
            "index_idx": np.dtype(np.uint32),
            "index_idx_offsets": np.dtype(np.uint64),
            "val": value_dtype,
            "val_data": np.dtype(np.uint32),
            "val_idx": np.dtype(np.uint32),
            "val_idx_offsets": np.dtype(np.uint64),
        }
        if compression == "unpacked":
            required = {"idxptr", "index", "val"}
        elif datatype == "uint":
            required = {
                "idxptr",
                "index_data",
                "index_starts",
                "index_idx",
                "val_data",
                "val_idx",
            }
        else:
            required = {
                "idxptr",
                "index_data",
                "index_starts",
                "index_idx",
                "val",
            }
        if int(version_text) == 2 and compression == "packed":
            required.add("index_idx_offsets")
            if datatype == "uint":
                required.add("val_idx_offsets")
        missing = sorted(required.difference(normalized_arrays))
        if missing:
            raise MatrixSourceError(
                f"BPCells memory matrix is missing arrays {missing!r}"
            )
        store = _MemoryArrayStore(
            version,
            normalized_arrays,
            dtypes,
            float_bit_arrays=float_bit_arrays,
            text={
                "storage_order": (storage_order,),
                "row_names": row_names,
                "col_names": column_names,
            },
            limits=limits,
        )
        super().__init__(store, limits=limits)


class BPCellsDirectoryMatrixSource(BPCellsMatrixSource):
    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        limits: SourceLimits = DEFAULT_LIMITS,
    ) -> None:
        super().__init__(_DirectoryArrayStore(path, limits), limits=limits)
        self.path = Path(path).expanduser().resolve(strict=False)


class BPCellsHDF5MatrixSource(BPCellsMatrixSource):
    def __init__(
        self,
        path: str | os.PathLike[str] | Any,
        *,
        group: str,
        limits: SourceLimits = DEFAULT_LIMITS,
    ) -> None:
        store = _HDF5ArrayStore(path, group, limits)
        super().__init__(store, limits=limits)
        self.path = store.path
        self.group = store.group


BPCellsDirectorySource = BPCellsDirectoryMatrixSource
BPCellsDirMatrixSource = BPCellsDirectoryMatrixSource
BPCellsH5MatrixSource = BPCellsHDF5MatrixSource
BPCellsHDF5Source = BPCellsHDF5MatrixSource
