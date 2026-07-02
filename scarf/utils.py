"""Utility methods.

- Methods:
    - clean_array: returns input array with nan and infinite values removed
    - controlled_compute: materializes a ChunkedArray into NumPy
    - rescale_array: performs edge trimming on values of the input vector
    - show_progress: materializes a ChunkedArray and shows a progress bar
    - system_call: executes a command in the underlying operative system
    - rolling_window: applies rolling window mean over a vector
"""

import resource
import shutil
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import zarr
from zarr.abc.store import Store
from loguru import logger
from numba import jit
from numpy.typing import NDArray

from ._types import ZarrMode

__all__ = [
    "logger",
    "tqdmbar",
    "tqdm_params",
    "set_verbosity",
    "get_log_level",
    "system_call",
    "rescale_array",
    "clean_array",
    "load_zarr",
    "permute_into_chunks",
    "show_dask_progress",
    "controlled_compute",
    "prefetch_blocks",
    "ColumnBlockPipeline",
    "iter_column_blocks",
    "remote_column_disk_ahead",
    "remote_column_ram_ahead",
    "process_rss_mb",
    "rss_peak_tracker",
    "rolling_window",
]

logger.remove()


def stdout_is_interactive() -> bool:
    """True when stdout is a TTY (interactive terminal)."""
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _flushing_stdout_sink(message: Any) -> None:
    sys.stdout.write(str(message))
    sys.stdout.flush()


logger.add(
    _flushing_stdout_sink,
    colorize=stdout_is_interactive(),
    format="{message}\n"
    if not stdout_is_interactive()
    else "<level>{level}</level>: {message}",
    level="INFO",
)

tqdm_params = {
    "bar_format": "{desc}: {percentage:3.0f}%| {bar} {n_fmt}/{total_fmt} [{elapsed}]",
    "ncols": 500,
    "colour": "#34abeb",
}

type ZARRLOC = str | Store


def get_log_level() -> int:
    """Return the current minimum log level configured for Scarf's logger."""
    # noinspection PyUnresolvedReferences
    return int(logger._core.min_level)  # type: ignore[attr-defined]


def is_notebook() -> bool:
    """Return True when running inside a Jupyter notebook kernel."""
    try:
        shell = get_ipython().__class__.__name__  # type: ignore[name-defined]
        if shell == "ZMQInteractiveShell":
            return True
        elif shell == "TerminalInteractiveShell":
            return False
        else:
            return False
    except NameError:
        return False


def tqdmbar(*args: Any, **kwargs: Any) -> Any:
    """Return a tqdm progress bar with Scarf defaults and log-level aware disable."""
    params = dict(tqdm_params)
    for i in kwargs:
        if i in params:
            del params[i]
    if "disable" not in kwargs and "disable" not in params:
        if get_log_level() <= 20 and stdout_is_interactive():
            params["disable"] = False
        else:
            params["disable"] = True
    if is_notebook():
        from tqdm import tqdm_notebook

        return tqdm_notebook(*args, **kwargs, **params)
    else:
        from tqdm.auto import tqdm

        return tqdm(*args, **kwargs, **params)


def set_verbosity(level: str | None = None, filepath: str | None = None) -> None:
    """Set verbosity level of Scarf's output. Setting value of level='CRITICAL'
    should silence all logs. Progress bars are automatically disabled for
    levels above 'INFO'.

    Args:
        level: A valid level name. Run without any parameter to see available options
        filepath: The output file path. All logs will be saved to this file. If no file path is
                  is provided then all the logs are printed on standard output.

    Returns:
        None
    """
    # noinspection PyUnresolvedReferences
    available_levels = logger._core.levels.keys()  # type: ignore[attr-defined]

    if level is None or level not in available_levels:
        raise ValueError(
            f"Please provide a value for level: {', '.join(available_levels)}"
        )
    logger.remove()
    interactive = stdout_is_interactive() and filepath is None
    if filepath is None:
        logger.add(
            _flushing_stdout_sink,
            colorize=interactive,
            format="{message}\n"
            if not interactive
            else "<level>{level}</level>: {message}",
            level=level,
        )
        return None
    logger.add(
        filepath,  # type: ignore[arg-type]
        colorize=True,
        format="<level>{level}</level>: {message}",
        level=level,
    )


def process_rss_mb() -> float:
    """Return this process's resident set size in MiB."""
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


@contextmanager
def rss_peak_tracker(
    interval_s: float = 0.25,
) -> Iterator[Callable[[], float]]:
    """Sample process RSS in the background and yield a peak reader.

    The reader returns the maximum RSS observed since the tracker started,
    including a final sample on exit.
    """
    peak = process_rss_mb()
    stop = threading.Event()

    def sample_loop() -> None:
        nonlocal peak
        while not stop.wait(interval_s):
            peak = max(peak, process_rss_mb())

    thread = threading.Thread(target=sample_loop, daemon=True)
    thread.start()
    try:
        yield lambda: peak
    finally:
        stop.set()
        thread.join(timeout=2.0)
        peak = max(peak, process_rss_mb())


_T = TypeVar("_T")


def _wait_with_heartbeat(
    future: Future[_T],
    *,
    label: str,
    interval_s: float = 30.0,
) -> _T:
    """Wait on ``future``, logging every ``interval_s`` while blocked."""
    from concurrent.futures import TimeoutError as FuturesTimeoutError

    waited = 0.0
    while True:
        try:
            return future.result(timeout=interval_s)
        except FuturesTimeoutError:
            waited += interval_s
            logger.info(f"{label}: still waiting ({waited:.0f}s elapsed)")


class ColumnBlockPipeline:
    """Ordered block reads with one RAM read-ahead and optional disk staging.

    The next block is prefetched into RAM while the current block is processed.
    Further blocks (up to ``disk_ahead``) are staged on local disk one at a time
    so at most one remote read runs for disk spillover while compute continues.
    """

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
        arr = np.load(path)
        path.unlink(missing_ok=True)
        return np.asarray(arr)

    def _advance_disk_queue(self, anchor_idx: int) -> None:
        """Start at most one serial disk fetch within the staging window."""
        if self.disk_ahead <= 0:
            return
        with self._lock:
            if self._disk_future is not None and not self._disk_future.done():
                return
            self._disk_future = None
            window_end = min(self.n_blocks, anchor_idx + 2 + self.disk_ahead)
            for idx in range(anchor_idx + 2, window_end):
                if idx in self._disk_done or self._disk_path(idx).is_file():
                    continue
                if idx in self._disk_pending:
                    continue
                future = self._pool.submit(self._write_disk, idx)
                self._disk_pending[idx] = future
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
        """Start prefetching blocks after ``from_block_idx``."""
        for offset in range(1, self.ram_ahead + 1):
            self._schedule_ram(from_block_idx + offset)
        self._advance_disk_queue(from_block_idx)

    def take(
        self,
        block_idx: int,
        *,
        wait_label: str | None = None,
    ) -> tuple[np.ndarray, float, str]:
        """Return block ``block_idx``, wait time in seconds, and read source."""
        t0 = time.perf_counter()
        source = "r2"
        label = wait_label or f"block {block_idx + 1}/{self.n_blocks} read"

        def wait_on(future: Future[Any], src: str) -> np.ndarray:
            nonlocal source
            source = src
            return _wait_with_heartbeat(future, label=label)

        if block_idx == 0:
            arr = self.read_block(0)
        else:
            with self._lock:
                ram_future = self._ram_futures.pop(block_idx, None)
            if ram_future is not None:
                arr = wait_on(ram_future, "ram")
            elif self._disk_path(block_idx).is_file() or block_idx in self._disk_done:
                arr = self._load_disk(block_idx)
                source = "disk"
            else:
                arr = wait_on(self._pool.submit(self.read_block, block_idx), "r2")
        wait_sec = time.perf_counter() - t0
        self.schedule_ahead(block_idx)
        return arr, wait_sec, source


REMOTE_COLUMN_DISK_AHEAD = 5


def remote_column_disk_ahead(*, remote: bool, n_blocks: int = 1) -> int:
    """Disk staging depth for remote column-block reads.

    Disk spill only helps long pipelines with many trailing blocks. For a
    handful of batches (e.g. marker search) it steals thread-pool workers
    from RAM read-ahead and re-fetches the same R2 column chunks via disk.
    """
    if not remote or n_blocks < 5:
        return 0
    return REMOTE_COLUMN_DISK_AHEAD


def remote_column_ram_ahead(*, remote: bool, n_blocks: int) -> int:
    """In-flight RAM prefetches for remote column-block reads."""
    if not remote:
        return 1
    from scarf.storage.budget import READ_AHEAD

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
    """Iterate column blocks with RAM read-ahead and optional disk staging.

    Yields ``(block_idx, array, read_wait_sec, source)`` where ``source`` is
    ``"r2"``, ``"ram"``, or ``"disk"``.
    """
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
            arr, wait_sec, source = pipeline.take(block_idx, wait_label=wait_label)
            yield block_idx, arr, wait_sec, source


def rescale_array(a: np.ndarray, frac: float = 0.9) -> np.ndarray:
    """Performs edge trimming on values of the input vector.

    Performs edge trimming on values of the input vector and constrains them
    between frac and 1-frac density of a normal distribution created with the
    sample mean and std. dev. of a.

    Args:
        a: numeric vector
        frac: Value between 0 and 1.

    Returns:
        The input array, edge trimmed and constrained.
    """
    from scipy.stats import norm

    loc = (np.median(a) + np.median(a[::-1])) / 2
    dist = norm(loc, np.std(a))
    minv, maxv = dist.ppf(1 - frac), dist.ppf(frac)
    a[a < minv] = minv
    a[a > maxv] = maxv
    return a


def clean_array(x: NDArray[Any] | list[Any], fill_val: int | float = 0) -> NDArray[Any]:
    """Returns input array with nan and infinite values removed.

    Args:
        x: input array
        fill_val: value to fill zero values with (default: 0)
    """
    arr = np.asarray(x, dtype=np.float64)
    arr = np.nan_to_num(arr, copy=True)
    arr[(arr == np.inf) | (arr == -np.inf)] = 0
    arr[arr == 0] = fill_val
    return arr


def load_zarr(
    zarr_loc: ZARRLOC,
    mode: ZarrMode,
    synchronizer: Any = None,
    storage_options: dict[str, Any] | None = None,
) -> zarr.Group:
    """Open a Zarr group at the given path, URI, or store object.

    Args:
        zarr_loc: Path, remote URI (s3://, gs://, ...), or a zarr Store instance.
        mode: Zarr open mode, e.g. ``'r'``, ``'r+'``, or ``'w'``.
        synchronizer: Optional synchronizer (ignored under Zarr v3).
        storage_options: Credentials and backend options for remote URIs (obstore S3Config keys).

    Returns:
        Root Zarr group.
    """
    from .storage.zarr_store import configure_zarr_io_for_profile, make_store

    if synchronizer is not None:
        logger.debug("ThreadSynchronizer is ignored under Zarr v3")
    store = make_store(
        zarr_loc,
        storage_options=storage_options,
        read_only=(mode == "r"),
    )
    configure_zarr_io_for_profile()
    if isinstance(store, str):
        return zarr.open_group(store, mode=mode)
    return zarr.open_group(store=store, mode=mode)


def prefetch_blocks(
    block_iter: Iterable[Any],
    fn: Callable[[Any], Any],
    max_ahead: int = 1,
) -> Iterator[Any]:
    """Apply ``fn`` to blocks with bounded read-ahead while preserving order.

    Args:
        block_iter: Iterable of block objects to process.
        fn: Callable invoked on each block; its return value is yielded.
        max_ahead: Maximum number of blocks to compute ahead of the consumer.

    Yields:
        Results of ``fn(block)`` in the same order as ``block_iter``.
    """
    from .parallel import stream_shards

    yield from stream_shards(block_iter, fn, workers=max(1, max_ahead))


def controlled_compute(arr: Any, nthreads: int) -> np.ndarray:
    """Materializes a ChunkedArray, Block or deferred reduction into NumPy.

    Args:
        arr: A ChunkedArray, Block, deferred reduction, or an already-evaluated
             NumPy array.
        nthreads: number of threads to use for computation

    Returns:
        Result of computation as a NumPy array.
    """
    if hasattr(arr, "compute"):
        return np.asarray(arr.compute(nthreads))
    return np.asarray(arr)


def show_dask_progress(
    arr: Any, msg: str | None = None, nthreads: int = 1
) -> np.ndarray:
    """Materializes a ChunkedArray/reduction while showing a progress bar.

    Args:
        arr: A ChunkedArray, Block or deferred reduction.
        msg: message to log, default None
        nthreads: number of threads to use for computation, default 1

    Returns:
        Result of computation as a NumPy array.
    """
    if hasattr(arr, "compute"):
        return np.asarray(arr.compute(nthreads, msg))
    return np.asarray(arr)


def system_call(command: str) -> None:
    """Executes a command in the underlying operative system.

    Args:
        command: Shell command string to run.

    Returns:
        None
    """
    import shlex
    import subprocess

    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)
    while True:
        output = process.stdout.readline()  # type: ignore[union-attr]
        if process.poll() is not None:
            break
        if output:
            logger.info(output.strip())
    process.poll()
    return None


@jit(nopython=True)
def rolling_window(a: np.ndarray, w: int) -> np.ndarray:
    """Apply a centered rolling mean with window size w along axis 0.

    Args:
        a: 2D numeric array.
        w: Window size (number of rows).

    Returns:
        Array of the same shape as a with smoothed values.
    """
    n, m = a.shape
    b = np.zeros(shape=(n, m))
    for i in range(n):
        if i < w:
            x = i
            y = w - i
        elif (n - i) < w:
            x = w - (n - i)
            y = n - i
        else:
            x = w // 2
            y = w // 2
        x = i - x
        y = i + y
        for j in range(m):
            b[i, j] = a[x:y, j].mean()
    return b


def permute_into_chunks(size: int, chunk_size: int, seed: int = 42) -> list[np.ndarray]:
    """
    Permute the chunks of an array of the given size.

    Args:
        size: The size of the array to be permuted.
        chunk_size: The size of the chunks to permute.
        seed: Random seed for chunk permutations.

    Returns:
        List of permuted chunk index arrays.
    Examples:
    >>> permute_into_chunks(10, 3)
    [array([2, 1, 0]), array([3, 5, 4]), array([7, 8, 6]), array([9])]
    """
    rng = np.random.default_rng(seed=seed)
    arr = np.arange(size)
    start = 0
    end = len(arr) - len(arr) % chunk_size
    chunks = [arr[i : i + chunk_size] for i in range(start, end, chunk_size)]
    p_values = [rng.permutation(chunk) for chunk in chunks]
    # add the remaining elements
    if end < len(arr):
        p_values.append(rng.permutation(arr[end:]))
    return p_values
