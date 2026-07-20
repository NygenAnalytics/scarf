"""Blockwise matrix abstractions."""

from . import blocks as _blocks
from . import _reductions as _reduction_module
from .blocks import Block
from .chunked import ChunkedArray

__all__ = ["Block", "ChunkedArray"]

setattr(_blocks, "ChunkedArray", ChunkedArray)
setattr(_reduction_module, "ChunkedArray", ChunkedArray)

for _public_class in (Block, ChunkedArray):
    _public_class.__module__ = __name__
    for _member in _public_class.__dict__.values():
        if isinstance(_member, (classmethod, staticmethod)):
            _member = _member.__func__
        if isinstance(_member, property):
            _member = _member.fget
        if callable(_member) and hasattr(_member, "__module__"):
            _member.__module__ = __name__

del _member
del _public_class
