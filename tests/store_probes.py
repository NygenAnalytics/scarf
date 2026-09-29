"""Zarr store wrappers that expose object operations to tests."""

import asyncio
from collections.abc import AsyncIterator, Iterable, Sequence
from typing import Any

from zarr.abc.store import ByteRequest
from zarr.core.buffer import Buffer, BufferPrototype
from zarr.storage import MemoryStore

from profiling import recording_store
from profiling.recording_store import _nbytes, _requested_bytes, wrap_recording_store


class StoreProbe(recording_store.StoreProbe):
    """Profiling totals plus a per-key log, per-kind overlap, delays, and failures.

    Args:
        delay: Seconds every operation of a ``RecordingStore`` awaits, making overlap
            observable.
        fail_on: Key whose ``RecordingStore`` write raises instead of storing bytes.
    """

    def __init__(self, *, delay: float = 0.0, fail_on: str | None = None) -> None:
        super().__init__()
        self.delay = delay
        self.fail_on = fail_on
        self.ops: list[tuple[str, str]] = []
        self._kind_in_flight: dict[str, int] = {}
        self._kind_max_in_flight: dict[str, int] = {}

    def reset(self) -> None:
        super().reset()
        with self._lock:
            self.ops.clear()
            self._kind_max_in_flight = dict(self._kind_in_flight)

    def enter(
        self,
        kind: str,
        key: str,
        byte_range: object | None = None,
        requestedBytes: int | None = None,
    ) -> None:
        super().enter(kind, key, byte_range, requestedBytes)
        with self._lock:
            self.ops.append((kind, key))
            current = self._kind_in_flight.get(kind, 0) + 1
            self._kind_in_flight[kind] = current
            self._kind_max_in_flight[kind] = max(
                self._kind_max_in_flight.get(kind, 0), current
            )

    def leave(self, kind: str) -> None:
        super().leave(kind)
        with self._lock:
            self._kind_in_flight[kind] -= 1

    @property
    def max_in_flight(self) -> int:
        return self.to_json()["maxInFlight"]

    def max_in_flight_for(self, kind: str) -> int:
        with self._lock:
            return self._kind_max_in_flight.get(kind, 0)

    def chunk_ops(self, prefix: str) -> list[tuple[str, str]]:
        return [(kind, key) for kind, key in self.ops if key.startswith(prefix)]


class RecordingStore(MemoryStore):
    """MemoryStore that records object operations, overlap, and bytes."""

    def __init__(
        self,
        store_dict: Any = None,
        *,
        read_only: bool = False,
        delay: float = 0.0,
        fail_on: str | None = None,
        probe: StoreProbe | None = None,
    ):
        super().__init__(store_dict, read_only=read_only)
        self.probe = probe or StoreProbe(delay=delay, fail_on=fail_on)

    def with_read_only(self, read_only: bool = False) -> "RecordingStore":
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

    def reset(self) -> None:
        self.probe.reset()

    def chunk_ops(self, prefix: str) -> list[tuple[str, str]]:
        return self.probe.chunk_ops(prefix)

    async def _tracked(self, kind: str, key: str, start: Any, byte_range: Any = None):
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
            self.probe.record_transfer(kind, key, _nbytes(result))
            return result
        finally:
            self.probe.leave(kind)

    async def get(self, key, prototype, byte_range=None):
        return await self._tracked(
            "get",
            key,
            lambda: super(RecordingStore, self).get(key, prototype, byte_range),
            byte_range,
        )

    async def set(self, key, value, byte_range=None):
        self.probe.enter("set", key, byte_range, requestedBytes=_nbytes(value))
        try:
            if key == self.probe.fail_on:
                raise RuntimeError("injected write failure")
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
            lambda: super(RecordingStore, self).delete(key),
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


__all__ = [
    "RecordingStore",
    "StoreProbe",
    "wrap_recording_store",
]
