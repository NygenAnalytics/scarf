import time
from collections.abc import Callable, Iterator

import numpy as np

from ..storage.budget import DEFAULT_READ_AHEAD_BLOCKS
from ..storage.parallel import stream_shards
from .logging import logger
from .process import process_rss_mb


def iter_column_blocks(
    n_blocks: int,
    read_block: Callable[[int], np.ndarray],
    *,
    workers: int = DEFAULT_READ_AHEAD_BLOCKS,
    io_concurrency: int | None = None,
    msg: str | None = None,
) -> Iterator[tuple[int, np.ndarray, float, str]]:
    """Yield column blocks in order with caller-planned read-ahead."""
    if n_blocks <= 0:
        return
    worker_budget = max(1, int(workers))
    outer_workers = min(worker_budget, n_blocks)
    resolved_io = (
        max(1, worker_budget // outer_workers)
        if io_concurrency is None
        else max(1, int(io_concurrency))
    )

    def timed_read(block_idx: int) -> tuple[int, np.ndarray, float, str]:
        if msg:
            logger.debug(
                f"{msg}: reading block {block_idx + 1}/{n_blocks} "
                f"(rss {process_rss_mb():.0f} MiB)"
            )
        started = time.perf_counter()
        array = np.asarray(read_block(block_idx))
        return block_idx, array, time.perf_counter() - started, "direct"

    yield from stream_shards(
        range(n_blocks),
        timed_read,
        workers=outer_workers,
        io_concurrency=resolved_io,
        msg=msg,
        total=n_blocks,
    )
