import codecs
import struct
from collections.abc import Iterator, Sequence
from typing import Any, overload

import numpy as np
import numpy.typing as npt

from ._storage import RandomAccessStorage
from ._types import RType


_STRING_DESCRIPTOR = struct.Struct("<B3xQqI")
STRING_DESCRIPTOR_BYTES = _STRING_DESCRIPTOR.size
STRING_SOURCE_PAYLOAD = 0
STRING_SOURCE_DECODED = 1


def decode_r_string(raw: bytes, gp: int, default_encoding: str | None) -> str | bytes:
    """Decode one serialized R string using its CHARSXP encoding flags.

    Byte strings stay bytes. Latin-1, UTF-8, and ASCII flags pick their codec;
    unflagged strings use the writer's native encoding, or UTF-8 when the
    header names none or an unknown one.
    """
    if gp & (1 << 1):
        return raw
    if gp & (1 << 2):
        return raw.decode("latin-1")
    if gp & (1 << 3):
        return raw.decode("utf-8", errors="surrogateescape")
    if gp & (1 << 6):
        return raw.decode("ascii", errors="surrogateescape")
    encoding = default_encoding or "utf-8"
    try:
        codecs.lookup(encoding)
    except LookupError:
        encoding = "utf-8"
    return raw.decode(encoding, errors="surrogateescape")


class LazyVector:
    """Block-readable vector data owned by an RDS document."""

    @property
    def length(self) -> int:
        raise NotImplementedError

    def __len__(self) -> int:
        return self.length


class LazyAtomicVector(LazyVector):
    """A numeric or raw vector backed by a byte range."""

    _DTYPES: dict[RType, str] = {
        RType.LOGICAL: "i4",
        RType.INTEGER: "i4",
        RType.REAL: "f8",
        RType.COMPLEX: "c16",
        RType.RAW: "u1",
    }

    def __init__(
        self,
        storage: RandomAccessStorage,
        *,
        offset: int,
        length: int,
        r_type: RType,
        byte_order: str,
        path: str,
    ) -> None:
        if r_type not in self._DTYPES:
            raise ValueError(f"{r_type.name} is not an atomic vector type")
        self._storage = storage
        self.offset = offset
        self._length = length
        self.r_type = r_type
        self.byte_order = byte_order
        self.path = path
        self._dtype = np.dtype(self._DTYPES[r_type])

    @property
    def length(self) -> int:
        return self._length

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype

    @property
    def item_size(self) -> int:
        return self._dtype.itemsize

    @property
    def nbytes(self) -> int:
        return self.length * self.item_size

    def read_block(self, start: int, stop: int) -> npt.NDArray[Any]:
        """Read a half-open element range into a native-order array."""
        if start < 0 or stop < start or stop > self.length:
            raise IndexError(
                f"invalid block [{start}:{stop}] for vector of length {self.length}"
            )
        count = stop - start
        data = self._storage.read_at(
            self.offset + start * self.item_size,
            count * self.item_size,
            path=self.path,
        )
        wire_dtype = self._dtype.newbyteorder("<" if self.byte_order == "<" else ">")
        return np.frombuffer(data, dtype=wire_dtype, count=count).astype(
            self._dtype,
            copy=True,
        )

    @overload
    def __getitem__(self, key: int) -> Any: ...

    @overload
    def __getitem__(self, key: slice) -> npt.NDArray[Any]: ...

    def __getitem__(self, key: int | slice) -> Any | npt.NDArray[Any]:
        if isinstance(key, slice):
            start, stop, step = key.indices(self.length)
            if step != 1:
                raise ValueError("lazy vector slices require a step of 1")
            return self.read_block(start, stop)
        index = key
        if index < 0:
            index += self.length
        if index < 0 or index >= self.length:
            raise IndexError("vector index out of range")
        return self.read_block(index, index + 1)[0].item()


class LazyStringVector(LazyVector, Sequence[str | bytes | None]):
    """A character vector backed by a fixed-width descriptor index."""

    def __init__(
        self,
        index_storage: RandomAccessStorage,
        *,
        descriptor_offset: int,
        length: int,
        payload_storage: RandomAccessStorage,
        decoded_storage: RandomAccessStorage | None,
        default_encoding: str | None,
        path: str,
    ) -> None:
        self._index_storage = index_storage
        self.descriptor_offset = descriptor_offset
        self._length = length
        self._payload_storage = payload_storage
        self._decoded_storage = decoded_storage
        self.default_encoding = default_encoding
        self.path = path

    @property
    def length(self) -> int:
        return self._length

    def _descriptor(self, index: int) -> tuple[int, int, int, int]:
        data = self._index_storage.read_at(
            self.descriptor_offset + index * STRING_DESCRIPTOR_BYTES,
            STRING_DESCRIPTOR_BYTES,
            path=f"{self.path}[{index}]",
        )
        return _STRING_DESCRIPTOR.unpack(data)

    def _decode(self, raw: bytes | None, gp: int) -> str | bytes | None:
        if raw is None:
            return None
        return decode_r_string(raw, gp, self.default_encoding)

    def read_block(self, start: int, stop: int) -> list[str | bytes | None]:
        """Read and decode a half-open string range."""
        if start < 0 or stop < start or stop > self.length:
            raise IndexError(
                f"invalid block [{start}:{stop}] for vector of length {self.length}"
            )
        result: list[str | bytes | None] = []
        for block_start in range(start, stop, 1024):
            block_stop = min(stop, block_start + 1024)
            descriptors = list(
                _STRING_DESCRIPTOR.iter_unpack(
                    self._index_storage.read_at(
                        self.descriptor_offset + block_start * STRING_DESCRIPTOR_BYTES,
                        (block_stop - block_start) * STRING_DESCRIPTOR_BYTES,
                        path=self.path,
                    )
                )
            )
            index = 0
            while index < len(descriptors):
                source, offset, length, _gp = descriptors[index]
                if length == -1:
                    result.append(None)
                    index += 1
                    continue
                end = offset + length
                last = index + 1
                while last < len(descriptors):
                    next_source, next_offset, next_length, _ = descriptors[last]
                    if (
                        next_source != source
                        or next_length < 0
                        or next_offset < offset
                        or next_offset + next_length > offset + max(length, 1024 * 1024)
                    ):
                        break
                    end = max(end, next_offset + next_length)
                    last += 1
                storage = (
                    self._payload_storage
                    if source == STRING_SOURCE_PAYLOAD
                    else self._decoded_storage
                )
                if storage is None:
                    raise RuntimeError("decoded string storage is unavailable")
                raw = storage.read_at(
                    offset, end - offset, path=f"{self.path}[{block_start + index}]"
                )
                for _, position, size, gp in descriptors[index:last]:
                    result.append(
                        self._decode(
                            raw[position - offset : position - offset + size], gp
                        )
                    )
                index = last
        return result

    @overload
    def __getitem__(self, key: int) -> str | bytes | None: ...

    @overload
    def __getitem__(self, key: slice) -> list[str | bytes | None]: ...

    def __getitem__(
        self,
        key: int | slice,
    ) -> str | bytes | None | list[str | bytes | None]:
        if isinstance(key, slice):
            start, stop, step = key.indices(self.length)
            if step != 1:
                raise ValueError("lazy string slices require a step of 1")
            return self.read_block(start, stop)
        index = key
        if index < 0:
            index += self.length
        if index < 0 or index >= self.length:
            raise IndexError("string vector index out of range")
        source, offset, length, gp = self._descriptor(index)
        if length == -1:
            return None
        storage = (
            self._payload_storage
            if source == STRING_SOURCE_PAYLOAD
            else self._decoded_storage
        )
        if storage is None:
            raise RuntimeError("decoded string storage is unavailable")
        raw = storage.read_at(offset, length, path=f"{self.path}[{index}]")
        return self._decode(raw, gp)

    def __iter__(self) -> Iterator[str | bytes | None]:
        for index in range(self.length):
            yield self[index]


def pack_string_descriptor(
    source: int,
    offset: int,
    length: int,
    gp: int,
) -> bytes:
    """Pack one internal string descriptor."""
    return _STRING_DESCRIPTOR.pack(source, offset, length, gp)
