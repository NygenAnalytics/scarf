import bz2
import gzip
import hashlib
import io
import lzma
import os
import pathlib
import shutil
import tempfile
import threading
from collections.abc import Buffer
from dataclasses import dataclass
from typing import Any, Protocol, cast

from ._types import (
    RdsClosedError,
    RdsCompression,
    RdsFormatError,
    RdsLimitError,
    RdsLimits,
    RdsSourceMetadata,
)


class BinaryStream(Protocol):
    def read(self, size: int = -1) -> bytes: ...

    def seek(self, offset: int, whence: int = 0) -> int: ...

    def tell(self) -> int: ...

    def close(self) -> None: ...


RdsInput = str | os.PathLike[str] | BinaryStream | Buffer


class RandomAccessStorage:
    """Synchronized random access to one binary backing stream."""

    def __init__(
        self,
        file: BinaryStream,
        *,
        name: str,
        close_file: bool,
        restore_position: int | None = None,
    ) -> None:
        self._file = file
        self.name = name
        self._close_file = close_file
        self._restore_position = restore_position
        self._closed = False
        self._lock = threading.RLock()

    @property
    def closed(self) -> bool:
        return self._closed

    def tell(self) -> int:
        with self._lock:
            self._ensure_open()
            return self._file.tell()

    def seek(self, offset: int, whence: int = 0) -> int:
        with self._lock:
            self._ensure_open()
            return self._file.seek(offset, whence)

    def read(self, size: int = -1) -> bytes:
        with self._lock:
            self._ensure_open()
            return self._file.read(size)

    def read_at(self, offset: int, size: int, *, path: str = "$") -> bytes:
        if offset < 0 or size < 0:
            raise ValueError("offset and size must be nonnegative")
        with self._lock:
            self._ensure_open(path)
            current = self._file.tell()
            try:
                self._file.seek(offset)
                data = self._file.read(size)
            finally:
                self._file.seek(current)
        if len(data) != size:
            raise RdsFormatError(
                f"unexpected end of stream while reading {size} bytes",
                path=path,
                offset=offset,
            )
        return data

    def size(self) -> int:
        with self._lock:
            self._ensure_open()
            current = self._file.tell()
            try:
                return self._file.seek(0, os.SEEK_END)
            finally:
                self._file.seek(current)

    def flush(self) -> None:
        with self._lock:
            self._ensure_open()
            flush = getattr(self._file, "flush", None)
            if flush is not None:
                flush()

    def _append_unchecked(self, data: bytes) -> int:
        with self._lock:
            self._ensure_open()
            offset = self._file.seek(0, os.SEEK_END)
            written = self._file.write(data)  # type: ignore[attr-defined]
            if written is not None and written != len(data):
                raise OSError("short temporary-file write")
            return offset

    def _ensure_open(self, path: str = "$") -> None:
        if self._closed:
            raise RdsClosedError("RDS document is closed", path=path)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._restore_position is not None:
                try:
                    self._file.seek(self._restore_position)
                except (OSError, ValueError):
                    pass
            if self._close_file:
                self._file.close()


class TempManager:
    """Own capped temporary files for one document."""

    def __init__(
        self,
        limits: RdsLimits,
        *,
        temp_dir: str | os.PathLike[str] | None,
    ) -> None:
        self._maximum = limits.max_temp_bytes
        self._used = 0
        self._temp_dir = None if temp_dir is None else os.fspath(temp_dir)
        self._storages: list[RandomAccessStorage] = []
        self._paths: list[pathlib.Path] = []
        self._shared: dict[str, RandomAccessStorage] = {}
        self._closed = False
        self._lock = threading.RLock()

    @property
    def used_bytes(self) -> int:
        return self._used

    @property
    def paths(self) -> tuple[pathlib.Path, ...]:
        return tuple(self._paths)

    @property
    def storages(self) -> tuple[RandomAccessStorage, ...]:
        return tuple(self._storages)

    def create(self, label: str) -> RandomAccessStorage:
        with self._lock:
            if self._closed:
                raise RdsClosedError("RDS document is closed", path="$temp")
            file = tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix=f"scarf-rds-{label}-",
                suffix=".tmp",
                dir=self._temp_dir,
                delete=False,
            )
            path = pathlib.Path(file.name)
            storage = RandomAccessStorage(
                file,
                name=str(path),
                close_file=True,
            )
            self._paths.append(path)
            self._storages.append(storage)
            return storage

    def shared(self, label: str) -> RandomAccessStorage:
        """Return one append-only temporary storage per label."""
        with self._lock:
            storage = self._shared.get(label)
            if storage is not None:
                return storage
            storage = self.create(label)
            self._shared[label] = storage
            return storage

    def append(self, storage: RandomAccessStorage, data: bytes, *, path: str) -> int:
        with self._lock:
            actual = self._used + len(data)
            if actual > self._maximum:
                raise RdsLimitError(
                    "max_temp_bytes",
                    actual,
                    self._maximum,
                    path=path,
                )
            try:
                free_bytes = shutil.disk_usage(pathlib.Path(storage.name).parent).free
            except OSError as error:
                raise OSError(
                    f"Cannot inspect free scratch space for {storage.name}"
                ) from error
            if len(data) > free_bytes:
                raise RdsLimitError(
                    "scratch_free_bytes",
                    len(data),
                    free_bytes,
                    path=path,
                )
            offset = storage._append_unchecked(data)
            self._used = actual
            return offset

    def flush(self) -> None:
        for storage in self._storages:
            if not storage.closed:
                storage.flush()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for storage in reversed(self._storages):
                storage.close()
            for path in reversed(self._paths):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass


class BufferedTempWriter:
    """Chunk small records into bounded temporary-file writes."""

    def __init__(
        self,
        manager: TempManager,
        storage: RandomAccessStorage,
        *,
        chunk_bytes: int = 256 * 1024,
    ) -> None:
        self._manager = manager
        self.storage = storage
        self._chunk_bytes = chunk_bytes
        self._buffer = bytearray()
        self._base = storage.size()
        self._committed = 0

    @property
    def position(self) -> int:
        return self._base + self._committed + len(self._buffer)

    def write(self, data: bytes, *, path: str) -> int:
        offset = self.position
        self._buffer.extend(data)
        if len(self._buffer) >= self._chunk_bytes:
            self.flush(path=path)
        return offset

    def flush(self, *, path: str) -> None:
        if not self._buffer:
            return
        data = bytes(self._buffer)
        actual = self._manager.append(self.storage, data, path=path)
        expected = self._base + self._committed
        if actual != expected:
            raise RuntimeError("temporary writer was interleaved")
        self._committed += len(data)
        self._buffer.clear()


@dataclass(slots=True)
class PreparedPayload:
    storage: RandomAccessStorage
    metadata: RdsSourceMetadata
    temp_manager: TempManager
    direct_storage: RandomAccessStorage | None


@dataclass(slots=True)
class _OpenedInput:
    file: BinaryStream
    name: str
    size: int | None
    initial_position: int
    owned: bool


_COMPRESSION_MAGIC: tuple[tuple[bytes, RdsCompression], ...] = (
    (b"\x1f\x8b", RdsCompression.GZIP),
    (b"BZh", RdsCompression.BZIP2),
    (b"\xfd7zXZ\x00", RdsCompression.XZ),
    (b"\x28\xb5\x2f\xfd", RdsCompression.ZSTD),
)


def _open_input(source: RdsInput) -> _OpenedInput:
    if isinstance(source, (str, os.PathLike)):
        path = pathlib.Path(source)
        path_file = path.open("rb")
        return _OpenedInput(
            file=path_file,
            name=str(path),
            size=path.stat().st_size,
            initial_position=0,
            owned=True,
        )
    if isinstance(source, Buffer):
        data = bytes(source)
        bytes_file = io.BytesIO(data)
        return _OpenedInput(
            file=bytes_file,
            name="<bytes>",
            size=len(data),
            initial_position=0,
            owned=True,
        )
    try:
        initial = source.tell()
        end = source.seek(0, os.SEEK_END)
        source.seek(initial)
    except (AttributeError, OSError, ValueError) as error:
        raise TypeError("RDS input stream must be seekable") from error
    return _OpenedInput(
        file=source,
        name=str(getattr(source, "name", "<stream>")),
        size=end,
        initial_position=initial,
        owned=False,
    )


def _detect_compression(prefix: bytes) -> RdsCompression:
    for magic, compression in _COMPRESSION_MAGIC:
        if prefix.startswith(magic):
            return compression
    return RdsCompression.NONE


def _sha256_range(file: BinaryStream, start: int, stop: int) -> str:
    digest = hashlib.sha256()
    file.seek(start)
    remaining = stop - start
    while remaining:
        chunk = file.read(min(1024 * 1024, remaining))
        if not chunk:
            raise OSError("Input stream ended while computing its digest")
        digest.update(chunk)
        remaining -= len(chunk)
    file.seek(start)
    return digest.hexdigest()


def _decompressor(
    compression: RdsCompression,
    source: BinaryStream,
) -> BinaryStream:
    if compression is RdsCompression.GZIP:
        return gzip.GzipFile(fileobj=source, mode="rb")
    if compression is RdsCompression.BZIP2:
        return cast(BinaryStream, bz2.BZ2File(cast(Any, source), mode="rb"))
    if compression is RdsCompression.XZ:
        return cast(BinaryStream, lzma.LZMAFile(cast(Any, source), mode="rb"))
    if compression is RdsCompression.ZSTD:
        import zstandard

        return cast(
            BinaryStream,
            zstandard.ZstdDecompressor().stream_reader(
                cast(Any, source),
                read_across_frames=True,
                closefd=False,
            ),
        )
    raise AssertionError(f"no decompressor for {compression}")


def prepare_payload(
    source: RdsInput,
    limits: RdsLimits,
    *,
    temp_dir: str | os.PathLike[str] | None,
) -> PreparedPayload:
    """Open an input and spool compressed payloads within the temp cap."""
    manager = TempManager(limits, temp_dir=temp_dir)
    opened = _open_input(source)
    direct: RandomAccessStorage | None = None
    try:
        if opened.size is None:
            raise AssertionError("seekable input size was not recorded")
        source_bytes = opened.size - opened.initial_position
        source_sha256 = _sha256_range(
            opened.file,
            opened.initial_position,
            opened.size,
        )
        opened.file.seek(opened.initial_position)
        prefix = opened.file.read(8)
        opened.file.seek(opened.initial_position)
        compression = _detect_compression(prefix)
        if compression is RdsCompression.NONE:
            direct = RandomAccessStorage(
                opened.file,
                name=opened.name,
                close_file=opened.owned,
                restore_position=None if opened.owned else opened.initial_position,
            )
            payload_bytes = (
                direct.size() - opened.initial_position
                if opened.size is None
                else opened.size - opened.initial_position
            )
            return PreparedPayload(
                storage=direct,
                metadata=RdsSourceMetadata(
                    name=opened.name,
                    compression=compression,
                    source_bytes=source_bytes,
                    payload_bytes=payload_bytes,
                    source_sha256=source_sha256,
                    payload_sha256=source_sha256,
                    spooled=False,
                ),
                temp_manager=manager,
                direct_storage=direct,
            )

        spool = manager.create("payload")
        reader = _decompressor(compression, opened.file)
        payload_bytes = 0
        payload_digest = hashlib.sha256()
        try:
            while True:
                chunk = reader.read(1024 * 1024)
                if not chunk:
                    break
                manager.append(spool, chunk, path="$source")
                payload_bytes += len(chunk)
                payload_digest.update(chunk)
        except RdsLimitError:
            raise
        except Exception as error:
            raise RdsFormatError(
                f"invalid {compression.value} compressed stream",
                path="$source",
            ) from error
        finally:
            reader.close()
        spool.flush()
        spool.seek(0)
        if opened.owned:
            opened.file.close()
        else:
            opened.file.seek(opened.initial_position)
        return PreparedPayload(
            storage=spool,
            metadata=RdsSourceMetadata(
                name=opened.name,
                compression=compression,
                source_bytes=source_bytes,
                payload_bytes=payload_bytes,
                source_sha256=source_sha256,
                payload_sha256=payload_digest.hexdigest(),
                spooled=True,
            ),
            temp_manager=manager,
            direct_storage=None,
        )
    except Exception:
        if direct is not None:
            direct.close()
        elif opened.owned:
            opened.file.close()
        else:
            try:
                opened.file.seek(opened.initial_position)
            except (OSError, ValueError):
                pass
        manager.close()
        raise
