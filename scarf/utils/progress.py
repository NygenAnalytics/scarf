from collections.abc import Iterable, Iterator
from typing import Any

from .logging import progress_enabled

tqdm_params = {
    "bar_format": "{desc}: {percentage:3.0f}%| {bar} {n_fmt}/{total_fmt} [{elapsed}]",
    "dynamic_ncols": True,
    "colour": "#34abeb",
}


def is_notebook() -> bool:
    """Return whether the current shell is a Jupyter kernel."""
    try:
        shell = get_ipython().__class__.__name__  # type: ignore[name-defined]
        if shell == "ZMQInteractiveShell":
            return True
        if shell == "TerminalInteractiveShell":
            return False
        return False
    except NameError:
        return False


def tqdmbar(*args: Any, **kwargs: Any) -> Any:
    """Create a progress bar using Scarf's display defaults."""
    params = dict(tqdm_params)
    for name in kwargs:
        if name in params:
            del params[name]
    if "disable" in kwargs:
        kwargs["disable"] = not progress_enabled() or bool(kwargs["disable"])
    else:
        params["disable"] = not progress_enabled()
    from tqdm.auto import tqdm

    return tqdm(*args, **kwargs, **params)


def iter_progress[T](
    iterable: Iterable[T],
    *,
    desc: str | None = None,
    total: int | None = None,
    disable: bool | None = None,
) -> Iterator[T]:
    """Yield values while updating and reliably closing a progress bar."""
    iterator = iter(iterable)
    kwargs: dict[str, Any] = {"desc": desc, "total": total}
    if disable is not None:
        kwargs["disable"] = disable
    progress = tqdmbar(**kwargs)
    try:
        while True:
            try:
                item = next(iterator)
            except StopIteration:
                return
            try:
                yield item
            finally:
                del item
            progress.update()
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()
        progress.close()
