"""Test recording of Zarr Store operations.

This wrapper is instrumentation only. It is not a product request limiter.
"""

import asyncio
import threading
from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from zarr.abc.store import ByteRequest, Store
from zarr.core.buffer import Buffer, BufferPrototype
from zarr.storage import MemoryStore


def _nbytes(value: object) -> int:
    if value is None:
        return 0
    try:
        return int(len(value))  # type: ignore[arg-type]
    except TypeError:
        return 0


def _requested_bytes(byte_range: object | None) -> int | None:
    if byte_range is None:
        return None
    start = getattr(byte_range, "start", None)
    end = getattr(byte_range, "end", None)
    if start is not None and end is not None:
        return max(0, int(end) - int(start))
    suffix = getattr(byte_range, "suffix", None)
    if suffix is not None:
        return max(0, int(suffix))
    return None


@dataclass(frozen=True, slots=True)
class StoreOperationSummary:
    gets: int
    sets: int
    deletes: int
    rangeGets: int
    partialGets: int
    requestedBytes: int
    transferredBytes: int
    maxInFlight: int
    keysTouched: int


class StoreProbe:
    """Shared operation log for a store and any clone Zarr makes of it.

    Args:
        delay: Seconds every tracked operation awaits, making overlap observable.
        fail_on: Key whose write raises instead of storing bytes.
        countOnly: Increment counters without retaining per-key logs.
    """

    def __init__(
        self,
        *,
        delay: float = 0.0,
        fail_on: str | None = None,
        countOnly: bool = False,
    ):
        self.delay = delay
        self.fail_on = fail_on
        self.countOnly = countOnly
        self.ops: list[tuple[str, str]] = []
        self.byte_ranges: list[tuple[str, str, object | None]] = []
        self.requested_bytes: list[tuple[str, str, int | None]] = []
        self.transferred_bytes: list[tuple[str, str, int]] = []
        self.max_in_flight = 0
        self._in_flight = 0
        self._in_flight_by_kind: dict[str, int] = {}
        self._max_in_flight_by_kind: dict[str, int] = {}
        self._count_by_kind: dict[str, int] = {}
        self._requested_total = 0
        self._transferred_total = 0
        self._read_requested_total = 0
        self._read_transferred_total = 0
        self._write_transferred_total = 0
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self.ops.clear()
            self.byte_ranges.clear()
            self.requested_bytes.clear()
            self.transferred_bytes.clear()
            self.max_in_flight = 0
            self._in_flight = 0
            self._in_flight_by_kind.clear()
            self._max_in_flight_by_kind.clear()
            self._count_by_kind.clear()
            self._requested_total = 0
            self._transferred_total = 0
            self._read_requested_total = 0
            self._read_transferred_total = 0
            self._write_transferred_total = 0

    def chunk_ops(self, prefix: str) -> list[tuple[str, str]]:
        return [(kind, key) for kind, key in self.ops if key.startswith(prefix)]

    def enter(
        self,
        kind: str,
        key: str,
        byte_range: object | None = None,
        requestedBytes: int | None = None,
    ) -> None:
        requested = int(requestedBytes or 0)
        with self._lock:
            self._count_by_kind[kind] = self._count_by_kind.get(kind, 0) + 1
            self._requested_total += requested
            if kind in {"get", "get_ranges", "get_partial_values"}:
                self._read_requested_total += requested
            if not self.countOnly:
                self.ops.append((kind, key))
                self.byte_ranges.append((kind, key, byte_range))
                self.requested_bytes.append((kind, key, requestedBytes))
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
            self._in_flight_by_kind[kind] = self._in_flight_by_kind.get(kind, 0) + 1
            self._max_in_flight_by_kind[kind] = max(
                self._max_in_flight_by_kind.get(kind, 0),
                self._in_flight_by_kind[kind],
            )

    def record_transfer(self, kind: str, key: str, nbytes: int) -> None:
        transferred = int(nbytes)
        with self._lock:
            self._transferred_total += transferred
            if kind in {"get", "get_ranges", "get_partial_values"}:
                self._read_transferred_total += transferred
            elif kind == "set":
                self._write_transferred_total += transferred
            if not self.countOnly:
                self.transferred_bytes.append((kind, key, transferred))

    def leave(self, kind: str) -> None:
        with self._lock:
            self._in_flight -= 1
            self._in_flight_by_kind[kind] -= 1

    def max_in_flight_for(self, kind: str) -> int:
        with self._lock:
            return self._max_in_flight_by_kind.get(kind, 0)

    def summary(self) -> StoreOperationSummary:
        with self._lock:
            keys = {key for _kind, key in self.ops}
            return StoreOperationSummary(
                gets=int(self._count_by_kind.get("get", 0)),
                sets=int(self._count_by_kind.get("set", 0)),
                deletes=int(self._count_by_kind.get("delete", 0)),
                rangeGets=int(self._count_by_kind.get("get_ranges", 0)),
                partialGets=int(self._count_by_kind.get("get_partial_values", 0)),
                requestedBytes=int(self._requested_total),
                transferredBytes=int(self._transferred_total),
                maxInFlight=int(self.max_in_flight),
                keysTouched=len(keys),
            )

    def to_json(self) -> dict[str, int]:
        summary = self.summary()
        return {
            "gets": summary.gets,
            "sets": summary.sets,
            "deletes": summary.deletes,
            "rangeGets": summary.rangeGets,
            "partialGets": summary.partialGets,
            "requestedBytes": summary.requestedBytes,
            "transferredBytes": summary.transferredBytes,
            "readRequestedBytes": max(
                int(self._read_requested_total),
                int(self._read_transferred_total),
            ),
            "readTransferredBytes": int(self._read_transferred_total),
            "writeTransferredBytes": int(self._write_transferred_total),
            "maxInFlight": summary.maxInFlight,
            "keysTouched": summary.keysTouched,
        }


class RecordingMemoryStore(MemoryStore):
    """MemoryStore that records object operations, ranges, overlap, and bytes."""

    def __init__(
        self,
        store_dict=None,
        *,
        read_only: bool = False,
        delay: float = 0.0,
        fail_on: str | None = None,
        probe: StoreProbe | None = None,
    ):
        super().__init__(store_dict, read_only=read_only)
        self.probe = probe or StoreProbe(delay=delay, fail_on=fail_on)

    def with_read_only(self, read_only: bool = False) -> "RecordingMemoryStore":
        return type(self)(
            store_dict=self._store_dict,
            read_only=read_only,
            probe=self.probe,
        )

    @property
    def ops(self) -> list[tuple[str, str]]:
        return self.probe.ops

    @property
    def max_in_flight(self) -> int:
        return self.probe.max_in_flight

    def max_in_flight_for(self, kind: str) -> int:
        return self.probe.max_in_flight_for(kind)

    @property
    def byte_ranges(self) -> list[tuple[str, str, object | None]]:
        return self.probe.byte_ranges

    def reset(self) -> None:
        self.probe.reset()

    def chunk_ops(self, prefix: str) -> list[tuple[str, str]]:
        return self.probe.chunk_ops(prefix)

    async def _tracked(self, kind, key, start, byte_range=None):
        self.probe.enter(
            kind,
            key,
            byte_range,
            requestedBytes=_requested_bytes(byte_range),
        )
        try:
            if self.probe.delay:
                await asyncio.sleep(self.probe.delay)
            result = await start()
            transferred = _nbytes(result) if kind != "set" else _nbytes(result) or 0
            if kind == "set":
                transferred = 0
            self.probe.record_transfer(kind, key, transferred)
            return result
        finally:
            self.probe.leave(kind)

    async def get(self, key, prototype, byte_range=None):
        return await self._tracked(
            "get",
            key,
            lambda: super(RecordingMemoryStore, self).get(key, prototype, byte_range),
            byte_range,
        )

    async def set(self, key, value, byte_range=None):
        if key == self.probe.fail_on:
            self.probe.enter("set", key, byte_range, requestedBytes=_nbytes(value))
            self.probe.record_transfer("set", key, 0)
            self.probe.leave("set")
            raise RuntimeError("injected write failure")
        self.probe.enter(
            "set",
            key,
            byte_range,
            requestedBytes=_nbytes(value),
        )
        try:
            if self.probe.delay:
                await asyncio.sleep(self.probe.delay)
            result = await super().set(key, value, byte_range)
            self.probe.record_transfer("set", key, _nbytes(value))
            return result
        finally:
            self.probe.leave("set")

    async def delete(self, key):
        return await self._tracked(
            "delete",
            key,
            lambda: super(RecordingMemoryStore, self).delete(key),
        )

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        pairs = list(key_ranges)
        label = pairs[0][0] if pairs else ""
        requested = sum(_requested_bytes(item[1]) or 0 for item in pairs)
        self.probe.enter(
            "get_partial_values",
            label,
            pairs,
            requestedBytes=requested,
        )
        try:
            if self.probe.delay:
                await asyncio.sleep(self.probe.delay)
            result = await super().get_partial_values(prototype, pairs)
            transferred = sum(_nbytes(item) for item in result)
            self.probe.record_transfer("get_partial_values", label, transferred)
            return result
        finally:
            self.probe.leave("get_partial_values")

    async def get_ranges(
        self,
        key: str,
        byte_ranges: Sequence[ByteRequest | None],
        *,
        prototype: BufferPrototype,
        max_concurrency: int = 10,
        max_gap_bytes: int = 1 << 20,
        max_coalesced_bytes: int = 16 << 20,
    ) -> AsyncIterator[Sequence[tuple[int, Buffer | None]]]:
        requested = sum(_requested_bytes(item) or 0 for item in byte_ranges)
        self.probe.enter(
            "get_ranges",
            key,
            tuple(byte_ranges),
            requestedBytes=requested,
        )
        transferred = 0
        try:
            if self.probe.delay:
                await asyncio.sleep(self.probe.delay)
            async for group in super().get_ranges(
                key,
                byte_ranges,
                prototype=prototype,
                max_concurrency=max_concurrency,
                max_gap_bytes=max_gap_bytes,
                max_coalesced_bytes=max_coalesced_bytes,
            ):
                transferred += sum(_nbytes(item[1]) for item in group)
                yield group
        finally:
            self.probe.record_transfer("get_ranges", key, transferred)
            self.probe.leave("get_ranges")


class RecordingStoreWrapper(Store):
    """Wrap any Zarr Store and record get, range, write, overlap, and bytes."""

    def __init__(self, inner: Store, *, probe: StoreProbe | None = None):
        super().__init__(read_only=inner.read_only)
        self._inner = inner
        self.probe = probe or StoreProbe()
        self._is_open = bool(getattr(inner, "_is_open", False))

    def with_read_only(self, read_only: bool = False) -> "RecordingStoreWrapper":
        return RecordingStoreWrapper(
            self._inner.with_read_only(read_only),
            probe=self.probe,
        )

    @property
    def ops(self) -> list[tuple[str, str]]:
        return self.probe.ops

    @property
    def max_in_flight(self) -> int:
        return self.probe.max_in_flight

    def max_in_flight_for(self, kind: str) -> int:
        return self.probe.max_in_flight_for(kind)

    def reset(self) -> None:
        self.probe.reset()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RecordingStoreWrapper) and self._inner == other._inner

    @property
    def supports_writes(self) -> bool:
        return self._inner.supports_writes

    @property
    def supports_deletes(self) -> bool:
        return self._inner.supports_deletes

    @property
    def supports_listing(self) -> bool:
        return self._inner.supports_listing

    async def _open(self) -> None:
        open_inner = getattr(self._inner, "_open", None)
        if callable(open_inner):
            await open_inner()
        await super()._open()

    def close(self) -> None:
        self._inner.close()
        super().close()

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        self.probe.enter(
            "get",
            key,
            byte_range,
            requestedBytes=_requested_bytes(byte_range),
        )
        try:
            if self.probe.delay:
                await asyncio.sleep(self.probe.delay)
            result = await self._inner.get(key, prototype, byte_range)
            self.probe.record_transfer("get", key, _nbytes(result))
            return result
        finally:
            self.probe.leave("get")

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        pairs = list(key_ranges)
        label = pairs[0][0] if pairs else ""
        requested = sum(_requested_bytes(item[1]) or 0 for item in pairs)
        self.probe.enter(
            "get_partial_values",
            label,
            pairs,
            requestedBytes=requested,
        )
        try:
            if self.probe.delay:
                await asyncio.sleep(self.probe.delay)
            result = await self._inner.get_partial_values(prototype, pairs)
            self.probe.record_transfer(
                "get_partial_values",
                label,
                sum(_nbytes(item) for item in result),
            )
            return result
        finally:
            self.probe.leave("get_partial_values")

    async def get_ranges(
        self,
        key: str,
        byte_ranges: Sequence[ByteRequest | None],
        *,
        prototype: BufferPrototype,
        max_concurrency: int = 10,
        max_gap_bytes: int = 1 << 20,
        max_coalesced_bytes: int = 16 << 20,
    ) -> AsyncIterator[Sequence[tuple[int, Buffer | None]]]:
        requested = sum(_requested_bytes(item) or 0 for item in byte_ranges)
        self.probe.enter(
            "get_ranges",
            key,
            tuple(byte_ranges),
            requestedBytes=requested,
        )
        transferred = 0
        try:
            if self.probe.delay:
                await asyncio.sleep(self.probe.delay)
            async for group in self._inner.get_ranges(
                key,
                byte_ranges,
                prototype=prototype,
                max_concurrency=max_concurrency,
                max_gap_bytes=max_gap_bytes,
                max_coalesced_bytes=max_coalesced_bytes,
            ):
                transferred += sum(_nbytes(item[1]) for item in group)
                yield group
        finally:
            self.probe.record_transfer("get_ranges", key, transferred)
            self.probe.leave("get_ranges")

    async def exists(self, key: str) -> bool:
        return await self._inner.exists(key)

    async def set(self, key: str, value: Buffer) -> None:
        if key == self.probe.fail_on:
            self.probe.enter("set", key, None, requestedBytes=_nbytes(value))
            self.probe.record_transfer("set", key, 0)
            self.probe.leave("set")
            raise RuntimeError("injected write failure")
        self.probe.enter("set", key, None, requestedBytes=_nbytes(value))
        try:
            if self.probe.delay:
                await asyncio.sleep(self.probe.delay)
            await self._inner.set(key, value)
            self.probe.record_transfer("set", key, _nbytes(value))
        finally:
            self.probe.leave("set")

    async def delete(self, key: str) -> None:
        self.probe.enter("delete", key)
        try:
            if self.probe.delay:
                await asyncio.sleep(self.probe.delay)
            await self._inner.delete(key)
            self.probe.record_transfer("delete", key, 0)
        finally:
            self.probe.leave("delete")

    def list(self) -> AsyncIterator[str]:
        return self._inner.list()

    def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        return self._inner.list_prefix(prefix)

    def list_dir(self, prefix: str) -> AsyncIterator[str]:
        return self._inner.list_dir(prefix)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def wrap_recording_store(
    store: Store,
    *,
    probe: StoreProbe | None = None,
) -> RecordingStoreWrapper:
    if isinstance(store, RecordingStoreWrapper):
        return store
    return RecordingStoreWrapper(store, probe=probe)
