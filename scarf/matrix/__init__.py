"""Blockwise matrix abstractions."""

from . import _reductions as _reduction_module
from .chunked import ChunkedArray

__all__ = ["ChunkedArray"]

setattr(_reduction_module, "ChunkedArray", ChunkedArray)

ChunkedArray.__module__ = __name__
for _member in ChunkedArray.__dict__.values():
    if isinstance(_member, (classmethod, staticmethod)):
        _member = _member.__func__
    if isinstance(_member, property):
        _member = _member.fget
    if callable(_member) and hasattr(_member, "__module__"):
        _member.__module__ = __name__

del _member
