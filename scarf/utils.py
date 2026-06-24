"""Utility methods.

- Methods:
    - clean_array: returns input array with nan and infinite values removed
    - controlled_compute: materializes a ChunkedArray into NumPy
    - rescale_array: performs edge trimming on values of the input vector
    - show_progress: materializes a ChunkedArray and shows a progress bar
    - system_call: executes a command in the underlying operative system
    - rolling_window: applies rolling window mean over a vector
"""

import sys

import numpy as np
import zarr
from loguru import logger
from numba import jit

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
    "rolling_window",
]

logger.remove()
logger.add(
    sys.stdout, colorize=True, format="<level>{level}</level>: {message}", level="INFO"
)

tqdm_params = {
    "bar_format": "{desc}: {percentage:3.0f}%| {bar} {n_fmt}/{total_fmt} [{elapsed}]",
    "ncols": 500,
    "colour": "#34abeb",
}

type ZARRLOC = str


def get_log_level():
    # noinspection PyUnresolvedReferences
    return logger._core.min_level  # type: ignore


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__  # type: ignore
        if shell == "ZMQInteractiveShell":
            return True
        elif shell == "TerminalInteractiveShell":
            return False
        else:
            return False
    except NameError:
        return False


def tqdmbar(*args, **kwargs):
    params = dict(tqdm_params)
    for i in kwargs:
        if i in params:
            del params[i]
    if "disable" not in kwargs and "disable" not in params:
        if get_log_level() <= 20:
            params["disable"] = False
        else:
            params["disable"] = True
    if is_notebook():
        from tqdm import tqdm_notebook

        return tqdm_notebook(*args, **kwargs, **params)
    else:
        from tqdm.auto import tqdm

        return tqdm(*args, **kwargs, **params)


def set_verbosity(level: str | None = None, filepath: str | None = None):
    """Set verbosity level of Scarf's output. Setting value of level='CRITICAL'
    should silence all logs. Progress bars are automatically disabled for
    levels above 'INFO'.

    Args:
        level: A valid level name. Run without any parameter to see available options
        filepath: The output file path. All logs will be saved to this file. If no file path is
                  is provided then all the logs are printed on standard output.

    Returns:
    """
    # noinspection PyUnresolvedReferences
    available_levels = logger._core.levels.keys()  # type: ignore

    if level is None or level not in available_levels:
        raise ValueError(
            f"Please provide a value for level: {', '.join(available_levels)}"
        )
    logger.remove()
    if filepath is None:
        filepath = sys.stdout  # type: ignore
    logger.add(
        filepath,  # type: ignore
        colorize=True,
        format="<level>{level}</level>: {message}",
        level=level,
    )


def rescale_array(a: np.ndarray, frac: float = 0.9) -> np.ndarray:
    """Performs edge trimming on values of the input vector.

    Performs edge trimming on values of the input vector and constraints them between within frac and 1-frac density of
    normal distribution created with the sample mean and std. dev. of a.

    Args:
        a: numeric vector
        frac: Value between 0 and 1.

    Return:
        The input array, edge trimmed and constrained.
    """
    from scipy.stats import norm

    loc = (np.median(a) + np.median(a[::-1])) / 2
    dist = norm(loc, np.std(a))
    minv, maxv = dist.ppf(1 - frac), dist.ppf(frac)
    a[a < minv] = minv
    a[a > maxv] = maxv
    return a


def clean_array(x, fill_val: int = 0):
    """Returns input array with nan and infinite values removed.

    Args:
        x (np.ndarray): input array
        fill_val: value to fill zero values with (default: 0)
    """
    x = np.nan_to_num(x, copy=True)
    x[(x == np.inf) | (x == -np.inf)] = 0
    x[x == 0] = fill_val
    return x


def load_zarr(
    zarr_loc: ZARRLOC, mode: str, synchronizer=None
) -> zarr.Group:
    if synchronizer is not None:
        logger.debug("ThreadSynchronizer is ignored under Zarr v3")
    return zarr.open_group(zarr_loc, mode=mode)


def controlled_compute(arr, nthreads):
    """Materializes a ChunkedArray, Block or deferred reduction into NumPy.

    Args:
        arr: A ChunkedArray, Block, deferred reduction, or an already-evaluated
             NumPy array.
        nthreads: number of threads to use for computation

    Returns:
        Result of computation as a NumPy array.
    """
    if hasattr(arr, "compute"):
        return arr.compute(nthreads)
    return np.asarray(arr)


def show_dask_progress(arr, msg: str | None = None, nthreads: int = 1):
    """Materializes a ChunkedArray/reduction while showing a progress bar.

    Args:
        arr: A ChunkedArray, Block or deferred reduction.
        msg: message to log, default None
        nthreads: number of threads to use for computation, default 1

    Returns:
        Result of computation as a NumPy array.
    """
    if hasattr(arr, "compute"):
        return arr.compute(nthreads, msg)
    return np.asarray(arr)


def system_call(command):
    """Executes a command in the underlying operative system."""
    import shlex
    import subprocess

    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)
    while True:
        output = process.stdout.readline()  # type: ignore
        if process.poll() is not None:
            break
        if output:
            logger.info(output.strip())
    process.poll()
    return None


@jit(nopython=True)
def rolling_window(a, w):
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
        size: The size of the array to be permuted
        chunk_size: The size of the chunks to permute
    Returns:
        A permuted array of the given size
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
