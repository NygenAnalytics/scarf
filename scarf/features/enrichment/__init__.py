from typing import TYPE_CHECKING

from ..._facade import lazy_facade as _lazy_facade

if TYPE_CHECKING:
    from .net import read_gmt as read_gmt
    from .results import EnrichmentResult as EnrichmentResult

__all__ = ["EnrichmentResult", "read_gmt"]

__getattr__, __dir__ = _lazy_facade(
    __name__,
    {
        "EnrichmentResult": ".results",
        "read_gmt": ".net",
    },
)
