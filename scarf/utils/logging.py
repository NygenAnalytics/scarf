import sys
from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass(slots=True)
class _OutputConfig:
    level: str = "INFO"
    progress: bool = True
    timestamps: bool = False
    filepath: str | None = None


_config = _OutputConfig()
_handler_id: int | None = None


def stdout_is_interactive() -> bool:
    """Return whether stdout is an interactive terminal."""
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _flushing_stdout_sink(message: Any) -> None:
    sys.stdout.write(str(message))
    sys.stdout.flush()


def _log_format() -> str:
    if _config.timestamps:
        return "{time:YYYY-MM-DDTHH:mm:ss.SSSZ} | <level>{level}</level> | {message}"
    return "<level>{level}</level>: {message}"


def _remove_scarf_handler() -> None:
    global _handler_id
    if _handler_id is None:
        return
    try:
        logger.remove(_handler_id)
    except ValueError:
        pass
    _handler_id = None


def _install_scarf_handler() -> None:
    global _handler_id
    _remove_scarf_handler()
    if _config.filepath is None:
        _handler_id = logger.add(
            _flushing_stdout_sink,
            colorize=None,
            format=_log_format(),
            level=_config.level,
        )
        return
    _handler_id = logger.add(
        _config.filepath,
        colorize=False,
        format=_log_format(),
        level=_config.level,
    )


def _resolve_level(level: str | None) -> str:
    if level is None:
        raise ValueError("Please provide a value for level recognized by Loguru")
    try:
        return logger.level(level).name
    except ValueError as error:
        raise ValueError(
            f"Please provide a value for level recognized by Loguru: {level!r}"
        ) from error


try:
    logger.remove(0)
except ValueError:
    pass
_install_scarf_handler()


def get_log_level() -> int:
    """Return the current minimum Scarf log level."""
    return logger.level(_config.level).no


def progress_enabled() -> bool:
    """Return whether Scarf progress reporting is enabled."""
    return _config.progress


def configure_output(
    *,
    level: str | None = None,
    progress: bool | None = None,
    timestamps: bool | None = None,
) -> None:
    """Configure Scarf's logging and progress output.

    Unspecified settings retain their current values.
    """
    update_handler = False
    if level is not None:
        _config.level = _resolve_level(level)
        update_handler = True
    if progress is not None:
        if not isinstance(progress, bool):
            raise TypeError("progress must be a bool")
        _config.progress = progress
    if timestamps is not None:
        if not isinstance(timestamps, bool):
            raise TypeError("timestamps must be a bool")
        _config.timestamps = timestamps
        update_handler = True
    if update_handler:
        _install_scarf_handler()


def set_verbosity(level: str | None = None, filepath: str | None = None) -> None:
    """Set Scarf's logging level and optional output file."""
    _config.level = _resolve_level(level)
    _config.filepath = filepath
    _install_scarf_handler()
