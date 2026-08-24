"""Stored chunk geometry of a streamed array."""

import operator
from dataclasses import dataclass
from typing import Any

import numpy as np

from .types import array_metadata_shards

__all__ = ["ArrayGeometry", "array_geometry"]


@dataclass(frozen=True, slots=True)
class ArrayGeometry:
    """Chunk and shard extents an array is physically stored in."""

    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    shards: tuple[int, ...] | None
    itemsize: int

    def axisChunk(self, axis: int) -> int:
        """Return the chunk extent along one axis."""
        return max(1, self.chunks[axis])

    def axisShard(self, axis: int) -> int:
        """Return the shard extent along one axis, or its chunk extent."""
        if self.shards is None:
            return self.axisChunk(axis)
        return max(1, self.shards[axis])

    def nominalChunkBytes(self) -> int:
        """Return the bytes one whole chunk decodes to.

        Zarr stores and decodes a full chunk even where the chunk overhangs the
        array edge, so this deliberately ignores the clipped extent.
        """
        elements = 1
        for extent in self.chunks:
            elements *= max(1, extent)
        return elements * self.itemsize

    def binOf(self, axis: int, indices: np.ndarray) -> np.ndarray:
        """Map indices along one axis to the chunk they live in."""
        return np.asarray(indices, dtype=np.int64) // self.axisChunk(axis)


def array_geometry(array: Any) -> ArrayGeometry | None:
    """Read stored geometry, or ``None`` when the array has no chunk layout.

    In-memory arrays have no chunks, so callers get one explicit place to decide
    what an unchunked array should mean for them.
    """
    chunks = getattr(array, "chunks", None)
    if chunks is None:
        return None
    resolved_chunks = tuple(operator.index(value) for value in chunks)
    resolved_shape = tuple(operator.index(value) for value in array.shape)
    if not resolved_chunks:
        return None
    if len(resolved_chunks) != len(resolved_shape):
        raise ValueError(
            f"Array chunks {resolved_chunks} do not match shape {resolved_shape}"
        )
    if any(value < 1 for value in resolved_chunks):
        raise ValueError(f"Array chunks must be positive, got {resolved_chunks}")
    if any(value < 0 for value in resolved_shape):
        raise ValueError(f"Array dimensions cannot be negative, got {resolved_shape}")

    raw_shards = array_metadata_shards(array)
    shards = None
    if raw_shards is not None:
        shards = tuple(operator.index(value) for value in raw_shards)
        if len(shards) != len(resolved_shape):
            raise ValueError(
                f"Array shards {shards} do not match shape {resolved_shape}"
            )

    return ArrayGeometry(
        shape=resolved_shape,
        chunks=resolved_chunks,
        shards=shards,
        itemsize=max(1, int(np.dtype(array.dtype).itemsize)),
    )
