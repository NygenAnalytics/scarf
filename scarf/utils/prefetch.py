import time
from collections.abc import Callable, Iterator

import numpy as np

from ..storage.budget import READ_AHEAD
from ..storage.parallel import stream_shards
from .logging import logger
from .process import process_rss_mb


def iter_column_blocks(
    n_blocks: int,
    read_block: Callable[[int], np.ndarray],
    *,
    workers: int = READ_AHEAD,
    msg: str | None = None,
) -> Iterator[tuple[int, np.ndarray, float, str]]:
    """Yield column blocks in order with shallow read-ahead."""
    if n_blocks <= 0:
        return
    worker_budget = max(1, int(workers))
    outer_workers = min(worker_budget, READ_AHEAD, n_blocks)
    io_concurrency = max(1, worker_budget // outer_workers)

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
        io_concurrency=io_concurrency,
        msg=msg,
        total=n_blocks,
    )
