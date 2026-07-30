from typing import Any

import numpy as np

from .logging import progress_enabled


def controlled_compute(arr: Any, nthreads: int) -> np.ndarray:
    """Materialize a deferred array with a bounded thread count."""
    if hasattr(arr, "compute"):
        return np.asarray(arr.compute(nthreads))
    return np.asarray(arr)


def show_dask_progress(
    arr: Any,
    msg: str | None = None,
    nthreads: int = 1,
) -> np.ndarray:
    """Materialize a deferred array while reporting progress."""
    if hasattr(arr, "compute"):
        progress_msg = msg if progress_enabled() else None
        return np.asarray(arr.compute(nthreads, progress_msg))
    return np.asarray(arr)
