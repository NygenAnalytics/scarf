from importlib import import_module as _import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scarf.utils.arrays import (
        array_digest as array_digest,
        clean_array as clean_array,
        permute_into_chunks as permute_into_chunks,
        rescale_array as rescale_array,
        rolling_window as rolling_window,
    )
    from scarf.utils.compute import (
        controlled_compute as controlled_compute,
        compute_with_progress as compute_with_progress,
    )
    from scarf.utils.logging import (
        configure_output as configure_output,
        get_log_level as get_log_level,
        logger as logger,
        set_verbosity as set_verbosity,
    )
    from scarf.utils.prefetch import (
        iter_column_blocks as iter_column_blocks,
    )
    from scarf.utils.process import (
        process_rss_mb as process_rss_mb,
        rss_peak_tracker as rss_peak_tracker,
        system_call as system_call,
    )
    from scarf.utils.progress import (
        tqdm_params as tqdm_params,
        tqdmbar as tqdmbar,
    )
    from scarf.storage.stores import load_zarr as load_zarr

__all__ = [
    "logger",
    "tqdmbar",
    "tqdm_params",
    "configure_output",
    "set_verbosity",
    "get_log_level",
    "system_call",
    "rescale_array",
    "clean_array",
    "load_zarr",
    "permute_into_chunks",
    "compute_with_progress",
    "controlled_compute",
    "iter_column_blocks",
    "process_rss_mb",
    "rss_peak_tracker",
    "array_digest",
    "rolling_window",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "logger": (".logging", "logger"),
    "tqdmbar": (".progress", "tqdmbar"),
    "tqdm_params": (".progress", "tqdm_params"),
    "configure_output": (".logging", "configure_output"),
    "set_verbosity": (".logging", "set_verbosity"),
    "get_log_level": (".logging", "get_log_level"),
    "system_call": (".process", "system_call"),
    "rescale_array": (".arrays", "rescale_array"),
    "clean_array": (".arrays", "clean_array"),
    "load_zarr": ("..storage.stores", "load_zarr"),
    "permute_into_chunks": (".arrays", "permute_into_chunks"),
    "compute_with_progress": (".compute", "compute_with_progress"),
    "controlled_compute": (".compute", "controlled_compute"),
    "iter_column_blocks": (".prefetch", "iter_column_blocks"),
    "process_rss_mb": (".process", "process_rss_mb"),
    "rss_peak_tracker": (".process", "rss_peak_tracker"),
    "array_digest": (".arrays", "array_digest"),
    "rolling_window": (".arrays", "rolling_window"),
    "stdout_is_interactive": (".logging", "stdout_is_interactive"),
    "is_notebook": (".progress", "is_notebook"),
}

for _export_name in _LAZY_EXPORTS:
    globals().pop(_export_name, None)
del _export_name


def __getattr__(name: str) -> Any:
    export = _LAZY_EXPORTS.get(name)
    if export is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = export
    value = getattr(_import_module(module_name, __name__), attribute_name)
    if name in __all__ and hasattr(value, "__module__"):
        value.__module__ = __name__
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(_LAZY_EXPORTS))
