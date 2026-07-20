import sys
from typing import Any

from loguru import logger

logger.remove()


def stdout_is_interactive() -> bool:
    """Return whether stdout is an interactive terminal."""
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _flushing_stdout_sink(message: Any) -> None:
    sys.stdout.write(str(message))
    sys.stdout.flush()


logger.add(
    _flushing_stdout_sink,
    colorize=stdout_is_interactive(),
    format=(
        "{message}\n"
        if not stdout_is_interactive()
        else "<level>{level}</level>: {message}"
    ),
    level="INFO",
)


def get_log_level() -> int:
    """Return the current minimum Scarf log level."""
    return int(logger._core.min_level)  # type: ignore[attr-defined]


def set_verbosity(level: str | None = None, filepath: str | None = None) -> None:
    """Set Scarf's logging level and optional output file."""
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
            format=(
                "{message}\n"
                if not interactive
                else "<level>{level}</level>: {message}"
            ),
            level=level,
        )
        return
    logger.add(
        filepath,  # type: ignore[arg-type]
        colorize=True,
        format="<level>{level}</level>: {message}",
        level=level,
    )
