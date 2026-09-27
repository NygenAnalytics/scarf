from collections.abc import Callable
from functools import cache, wraps


def restore_numba_threads[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    """Restore Numba's process thread count after a call."""

    @wraps(fn)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        import numba

        previous = numba.get_num_threads()
        try:
            return fn(*args, **kwargs)
        finally:
            numba.set_num_threads(previous)

    return wrapped


@cache
def threadsafe_threading_layer() -> bool:
    """Whether two threads may launch parallel kernels at the same time.

    Numba's workqueue threading layer aborts the process on concurrent
    launches; the TBB and OpenMP layers are thread-safe.
    """
    import numba

    numba.get_num_threads()  # selects the threading layer
    return str(numba.threading_layer()) != "workqueue"
