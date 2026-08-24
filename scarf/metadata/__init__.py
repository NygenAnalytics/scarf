"""Zarr-backed metadata tables."""

from .rows import MetaDataRowBlock
from .table import MetaData, zarrGroup as zarrGroup

__all__ = ["MetaData", "MetaDataRowBlock"]

MetaData.__module__ = __name__
MetaDataRowBlock.__module__ = __name__
