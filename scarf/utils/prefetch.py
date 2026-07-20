import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from ..storage.parallel import stream_shards
from .logging import logger
from .process import process_rss_mb
from .progress import tqdmbar

_T = TypeVar("_T")


def prefetch_blocks(
    block_iter: Iterable[Any],
    fn: Callable[[Any], Any],
    max_ahead: int = 1,
) -> Iterator[Any]:
    """Apply a function with bounded ordered read-ahead."""
    yield from stream_shards(
        block_iter,
        fn,
        workers=max(1, max_ahead),
    )


def _wait_with_heartbeat(
    future: Future[_T],
    *,
    label: str,
    interval_s: float = 30.0,
) -> _T:
    from concurrent.futures import TimeoutError as FuturesTimeoutError

    waited = 0.0
    while True:
        try:
            return future.result(timeout=interval_s)
        except FuturesTimeoutError:
            waited += interval_s
            logger.info(f"{label}: still waiting ({waited:.0f}s elapsed)")


class ColumnBlockPipeline:
    """Read ordered column blocks with bounded RAM and disk prefetch."""

    def __init__(
        self,
        n_blocks: int,
        read_block: Callable[[int], np.ndarray],
        *,
        ram_ahead: int = 1,
        disk_ahead: int = 5,
        scratch_dir: str | None = None,
    ) -> None:
        self.n_blocks = n_blocks
        self.read_block = read_block
        self.disk_ahead = max(0, int(disk_ahead))
        self._own_scratch = scratch_dir is None
        self._scratch = scratch_dir or tempfile.mkdtemp(prefix="scarf_col_blocks_")
        self._disk_done: set[int] = set()
        self._disk_pending: dict[int, Future[int]] = {}
        self._disk_future: Future[int] | None = None
        self._ram_futures: dict[int, Future[np.ndarray]] = {}
        self.ram_ahead = max(1, int(ram_ahead))
        pool_workers = max(2, self.ram_ahead + (1 if self.disk_ahead > 0 else 0))
        self._pool = ThreadPoolExecutor(max_workers=pool_workers)
        self._lock = threading.Lock()

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
        if self._own_scratch:
            shutil.rmtree(self._scratch, ignore_errors=True)

    def __enter__(self) -> "ColumnBlockPipeline":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _disk_path(self, block_idx: int) -> Path:
        return Path(self._scratch) / f"block_{block_idx:04d}.npy"

    def _write_disk(self, block_idx: int) -> int:
        np.save(
            self._disk_path(block_idx),
            self.read_block(block_idx),
            allow_pickle=False,
        )
        with self._lock:
            self._disk_done.add(block_idx)
            self._disk_pending.pop(block_idx, None)
        return block_idx

    def _load_disk(self, block_idx: int) -> np.ndarray:
        with self._lock:
            pending = self._disk_pending.pop(block_idx, None)
            self._disk_done.add(block_idx)
        if pending is not None:
            pending.result()
        path = self._disk_path(block_idx)
        if not path.is_file():
            return self.read_block(block_idx)
        array = np.load(path)
        path.unlink(missing_ok=True)
        return np.asarray(array)

    def _advance_disk_queue(self, anchor_idx: int) -> None:
        if self.disk_ahead <= 0:
            return
        with self._lock:
            if self._disk_future is not None and not self._disk_future.done():
                return
            self._disk_future = None
            window_end = min(self.n_blocks, anchor_idx + 2 + self.disk_ahead)
            for index in range(anchor_idx + 2, window_end):
                if index in self._disk_done or self._disk_path(index).is_file():
                    continue
                if index in self._disk_pending:
                    continue
                future = self._pool.submit(self._write_disk, index)
                self._disk_pending[index] = future
                self._disk_future = future
                return

    def _schedule_ram(self, block_idx: int) -> None:
        if block_idx >= self.n_blocks:
            return
        with self._lock:
            if block_idx in self._ram_futures:
                return
            if block_idx in self._disk_done or self._disk_path(block_idx).is_file():
                future = self._pool.submit(self._load_disk, block_idx)
            elif block_idx in self._disk_pending:

                def wait_and_load(idx: int = block_idx) -> np.ndarray:
                    pending = self._disk_pending.get(idx)
                    if pending is not None:
                        pending.result()
                    return self._load_disk(idx)

                future = self._pool.submit(wait_and_load)
            else:
                future = self._pool.submit(self.read_block, block_idx)
            self._ram_futures[block_idx] = future

    def schedule_ahead(self, from_block_idx: int) -> None:
        for offset in range(1, self.ram_ahead + 1):
            self._schedule_ram(from_block_idx + offset)
        self._advance_disk_queue(from_block_idx)

    def take(
        self,
        block_idx: int,
        *,
        wait_label: str | None = None,
    ) -> tuple[np.ndarray, float, str]:
        start_time = time.perf_counter()
        source = "r2"
        label = wait_label or f"block {block_idx + 1}/{self.n_blocks} read"

        def wait_on(future: Future[Any], source_name: str) -> np.ndarray:
            nonlocal source
            source = source_name
            return _wait_with_heartbeat(future, label=label)

        if block_idx == 0:
            array = self.read_block(0)
        else:
            with self._lock:
                ram_future = self._ram_futures.pop(block_idx, None)
            if ram_future is not None:
                array = wait_on(ram_future, "ram")
            elif self._disk_path(block_idx).is_file() or block_idx in self._disk_done:
                array = self._load_disk(block_idx)
                source = "disk"
            else:
                array = wait_on(
                    self._pool.submit(self.read_block, block_idx),
                    "r2",
                )
        wait_seconds = time.perf_counter() - start_time
        self.schedule_ahead(block_idx)
        return array, wait_seconds, source


REMOTE_COLUMN_DISK_AHEAD = 5


def remote_column_disk_ahead(*, remote: bool, n_blocks: int = 1) -> int:
    """Return the disk staging depth for a column-block pipeline."""
    if not remote or n_blocks < 5:
        return 0
    return REMOTE_COLUMN_DISK_AHEAD


def remote_column_ram_ahead(*, remote: bool, n_blocks: int) -> int:
    """Return the RAM read-ahead depth for a column-block pipeline."""
    if not remote:
        return 1
    from ..storage.budget import READ_AHEAD

    return max(1, min(READ_AHEAD, n_blocks - 1))


def iter_column_blocks(
    n_blocks: int,
    read_block: Callable[[int], np.ndarray],
    *,
    remote: bool = False,
    disk_ahead: int | None = None,
    scratch_dir: str | None = None,
    msg: str | None = None,
) -> Iterator[tuple[int, np.ndarray, float, str]]:
    """Yield column blocks in order with bounded prefetch."""
    if n_blocks <= 0:
        return
    ahead = (
        remote_column_disk_ahead(remote=remote, n_blocks=n_blocks)
        if disk_ahead is None
        else max(0, int(disk_ahead))
    )
    ram_ahead = remote_column_ram_ahead(remote=remote, n_blocks=n_blocks)
    block_range = range(n_blocks)
    with ColumnBlockPipeline(
        n_blocks,
        read_block,
        ram_ahead=ram_ahead,
        disk_ahead=ahead,
        scratch_dir=scratch_dir,
    ) as pipeline:
        for block_idx in tqdmbar(block_range, desc=msg or "", total=n_blocks):
            if msg:
                logger.debug(
                    f"{msg}: reading block {block_idx + 1}/{n_blocks} "
                    f"(rss {process_rss_mb():.0f} MiB)"
                )
            wait_label = (
                f"{msg}: block {block_idx + 1}/{n_blocks} read" if msg else None
            )
            array, wait_seconds, source = pipeline.take(
                block_idx,
                wait_label=wait_label,
            )
            yield block_idx, array, wait_seconds, source
