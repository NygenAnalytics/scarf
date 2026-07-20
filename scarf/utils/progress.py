from typing import Any

from .logging import get_log_level, stdout_is_interactive

tqdm_params = {
    "bar_format": "{desc}: {percentage:3.0f}%| {bar} {n_fmt}/{total_fmt} [{elapsed}]",
    "ncols": 500,
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
    if "disable" not in kwargs and "disable" not in params:
        params["disable"] = not (
            get_log_level() <= 20 and (stdout_is_interactive() or is_notebook())
        )
    if is_notebook():
        from tqdm import tqdm_notebook

        return tqdm_notebook(*args, **kwargs, **params)
    from tqdm.auto import tqdm

    return tqdm(*args, **kwargs, **params)
