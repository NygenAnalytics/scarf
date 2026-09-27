"""Run one callable on a worker thread while the caller continues."""

import contextvars
import functools
import threading
from collections.abc import Callable
from concurrent.futures import Future

from .numba import threadsafe_threading_layer


class BackgroundTask[T]:
    """Run ``fn`` on a worker thread in a copy of the caller's context.

    ``fn`` must not call ``threadpoolctl`` or import C extensions that are not
    loaded yet. A thread-pool scan holds the dynamic loader lock while it waits
    for the GIL, and an extension import holds the GIL while it waits for that
    lock, so the two deadlock when they run on different threads. With
    ``inline=True``, or without a thread-safe Numba threading layer, ``fn`` runs
    before the constructor returns.
    """

    def __init__(self, fn: Callable[[], T], *, name: str, inline: bool = False):
        self._future: Future[T] = Future()
        if inline or not threadsafe_threading_layer():
            self._run(fn)
            return
        work = functools.partial(contextvars.copy_context().run, fn)
        threading.Thread(target=self._run, args=(work,), name=name, daemon=True).start()

    def _run(self, fn: Callable[[], T]) -> None:
        try:
            self._future.set_result(fn())
        except BaseException as error:
            self._future.set_exception(error)

    def done(self) -> bool:
        return self._future.done()

    def result(self) -> T:
        """Wait for ``fn`` and return its value or raise its error."""
        return self._future.result()
