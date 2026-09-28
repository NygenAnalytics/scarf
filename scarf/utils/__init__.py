from typing import TYPE_CHECKING

from .._facade import lazy_facade as _lazy_facade

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
        logger as logger,
        set_verbosity as set_verbosity,
    )
    from scarf.utils.process import (
        process_rss_mb as process_rss_mb,
    )
    from scarf.utils.progress import (
        tqdm_params as tqdm_params,
        tqdmbar as tqdmbar,
    )

__all__ = [
    "logger",
    "tqdmbar",
    "tqdm_params",
    "configure_output",
    "set_verbosity",
    "rescale_array",
    "clean_array",
    "permute_into_chunks",
    "compute_with_progress",
    "controlled_compute",
    "process_rss_mb",
    "array_digest",
    "rolling_window",
]

__getattr__, __dir__ = _lazy_facade(
    __name__,
    {
        "logger": ".logging",
        "tqdmbar": ".progress",
        "tqdm_params": ".progress",
        "configure_output": ".logging",
        "set_verbosity": ".logging",
        "rescale_array": ".arrays",
        "clean_array": ".arrays",
        "permute_into_chunks": ".arrays",
        "compute_with_progress": ".compute",
        "controlled_compute": ".compute",
        "process_rss_mb": ".process",
        "array_digest": ".arrays",
        "rolling_window": ".arrays",
    },
    set_module=True,
)
