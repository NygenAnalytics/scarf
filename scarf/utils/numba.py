from collections.abc import Callable
from functools import wraps


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
