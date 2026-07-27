"""Zarr store wrappers that expose object operations to tests."""

import asyncio
import threading

from zarr.storage import MemoryStore


class StoreProbe:
    """Shared operation log for a store and any clone Zarr makes of it.

    Args:
        delay: Seconds every operation awaits, making real overlap observable.
        fail_on: Key whose write raises instead of storing bytes.
    """

    def __init__(self, *, delay: float = 0.0, fail_on: str | None = None):
        self.delay = delay
        self.fail_on = fail_on
        self.ops: list[tuple[str, str]] = []
        self.byte_ranges: list[tuple[str, str, object | None]] = []
        self.max_in_flight = 0
        self._in_flight = 0
        self._in_flight_by_kind: dict[str, int] = {}
        self._max_in_flight_by_kind: dict[str, int] = {}
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self.ops.clear()
            self.byte_ranges.clear()
            self.max_in_flight = 0
            self._in_flight = 0
            self._in_flight_by_kind.clear()
            self._max_in_flight_by_kind.clear()

    def chunk_ops(self, prefix: str) -> list[tuple[str, str]]:
        return [(kind, key) for kind, key in self.ops if key.startswith(prefix)]

    def enter(self, kind: str, key: str, byte_range: object | None = None) -> None:
        with self._lock:
            self.ops.append((kind, key))
            self.byte_ranges.append((kind, key, byte_range))
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
            self._in_flight_by_kind[kind] = self._in_flight_by_kind.get(kind, 0) + 1
            self._max_in_flight_by_kind[kind] = max(
                self._max_in_flight_by_kind.get(kind, 0),
                self._in_flight_by_kind[kind],
            )

    def leave(self, kind: str) -> None:
        with self._lock:
            self._in_flight -= 1
            self._in_flight_by_kind[kind] -= 1

    def max_in_flight_for(self, kind: str) -> int:
        with self._lock:
            return self._max_in_flight_by_kind.get(kind, 0)


class RecordingStore(MemoryStore):
    """MemoryStore that records object operations and observed overlap."""

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

    # Zarr clones a store to change read-only state; the clone keeps reporting
    # to the same probe so the test still observes every operation.
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

    @property
    def byte_ranges(self) -> list[tuple[str, str, object | None]]:
        return self.probe.byte_ranges

    def reset(self) -> None:
        self.probe.reset()

    def chunk_ops(self, prefix: str) -> list[tuple[str, str]]:
        return self.probe.chunk_ops(prefix)

    # The awaitable is built inside the tracker so a cancelled task never leaves
    # an unawaited coroutine behind.
    async def _tracked(self, kind, key, start, byte_range=None):
        self.probe.enter(kind, key, byte_range)
        try:
            if self.probe.delay:
                await asyncio.sleep(self.probe.delay)
            return await start()
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
        if key == self.probe.fail_on:
            self.probe.enter("set", key, byte_range)
            self.probe.leave("set")
            raise RuntimeError("injected write failure")
        return await self._tracked(
            "set",
            key,
            lambda: super(RecordingStore, self).set(key, value, byte_range),
            byte_range,
        )

    async def delete(self, key):
        return await self._tracked(
            "delete",
            key,
            lambda: super(RecordingStore, self).delete(key),
        )
