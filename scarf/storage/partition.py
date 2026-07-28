"""Row bands and chunk-aligned index blocks for streamed arrays."""

import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .geometry import ArrayGeometry

type Fits = Callable[[int], bool]

__all__ = [
    "IndexBlock",
    "affordable_width",
    "checked_indices",
    "contiguous_ranges",
    "partition_indices",
    "row_band",
]


@dataclass(frozen=True, slots=True)
class IndexBlock:
    """Selected indices that share physical chunks, and where they belong."""

    indices: np.ndarray
    destinations: np.ndarray
    bins: tuple[int, ...]


def row_band(
    geometry: ArrayGeometry | None,
    *,
    unit: Literal["shard", "chunk"] = "shard",
    fallback: int,
) -> int:
    """Return rows per block from stored geometry, or ``fallback`` without it.

    ``unit`` selects the stored extent to follow: ``shard`` for the object a
    write lands in, ``chunk`` for the unit a read decodes.
    """
    if geometry is None:
        return max(1, int(fallback))
    extent = geometry.axisShard(0) if unit == "shard" else geometry.axisChunk(0)
    return max(1, extent)


def contiguous_ranges(nRows: int, band: int) -> list[tuple[int, int]]:
    """Split a row axis into consecutive half-open ranges of ``band`` rows."""
    rows = max(0, int(nRows))
    step = max(1, int(band))
    return [(start, min(start + step, rows)) for start in range(0, rows, step)]


def affordable_width(fits: Fits, maxWidth: int) -> int:
    """Return the largest width up to ``maxWidth`` that ``fits`` accepts."""
    limit = max(0, int(maxWidth))
    if limit == 0 or not fits(1):
        return 0
    low = 1
    high = limit
    while low < high:
        candidate = (low + high + 1) // 2
        if fits(candidate):
            low = candidate
        else:
            high = candidate - 1
    return low


def checked_indices(
    values: Sequence[int] | np.ndarray,
    *,
    limit: int,
    name: str,
) -> np.ndarray:
    """Validate a one-dimensional selection of distinct in-range indices."""
    indexes = np.asarray(values)
    if indexes.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if indexes.size == 0:
        return np.empty(0, dtype=np.int64)
    if not np.issubdtype(indexes.dtype, np.integer):
        raise TypeError(f"{name} must contain integers")
    indexes = indexes.astype(np.int64, copy=False)
    if np.any(indexes < 0) or np.any(indexes >= limit):
        raise IndexError(f"{name} contains an out-of-range index")
    if np.unique(indexes).size != indexes.size:
        raise ValueError(f"{name} cannot contain duplicate indexes")
    return indexes


def _bins(geometry: ArrayGeometry, axis: int, indices: np.ndarray) -> tuple[int, ...]:
    return tuple(int(value) for value in np.unique(geometry.binOf(axis, indices)))


def _bin_groups(
    geometry: ArrayGeometry,
    axis: int,
    indices: np.ndarray,
) -> list[tuple[int, np.ndarray]]:
    """Group index positions by chunk, ascending, keeping request order inside."""
    bins = geometry.binOf(axis, indices)
    order = np.argsort(bins, kind="stable")
    ordered = bins[order]
    edges = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            (np.flatnonzero(np.diff(ordered)) + 1).astype(np.int64),
            np.array([order.size], dtype=np.int64),
        )
    )
    return [
        (int(ordered[edges[position]]), order[edges[position] : edges[position + 1]])
        for position in range(edges.size - 1)
    ]


def partition_indices(
    geometry: ArrayGeometry,
    axis: int,
    indices: Sequence[int] | np.ndarray,
    *,
    maxWidth: int | None = None,
    fits: Fits | None = None,
) -> list[IndexBlock]:
    """Group selected indices into blocks aligned to physical chunks.

    With ``maxWidth`` the selection is cut into fixed-width blocks in request
    order. With ``fits`` adjacent chunks are packed while the predicate accepts
    the combined width, and a chunk too wide on its own is split. With neither,
    every chunk becomes one block.
    """
    if maxWidth is not None and fits is not None:
        raise ValueError("Pass either maxWidth or fits, not both")
    resolved = checked_indices(indices, limit=geometry.shape[axis], name="indices")
    if resolved.size == 0:
        return []
    positions = np.arange(resolved.size, dtype=np.int64)

    if maxWidth is not None:
        width = max(1, operator.index(maxWidth))
        return [
            IndexBlock(
                indices=resolved[start : start + width],
                destinations=positions[start : start + width],
                bins=_bins(geometry, axis, resolved[start : start + width]),
            )
            for start in range(0, resolved.size, width)
        ]

    groups = _bin_groups(geometry, axis, resolved)
    if fits is None:
        return [
            IndexBlock(
                indices=resolved[member],
                destinations=positions[member],
                bins=(value,),
            )
            for value, member in groups
        ]

    blocks: list[IndexBlock] = []
    pending: list[np.ndarray] = []
    pending_bins: list[int] = []
    pending_width = 0
    previous_bin: int | None = None

    def flush() -> None:
        nonlocal pending_width, previous_bin
        if not pending:
            return
        member = np.concatenate(pending)
        blocks.append(
            IndexBlock(
                indices=resolved[member],
                destinations=positions[member],
                bins=tuple(pending_bins),
            )
        )
        pending.clear()
        pending_bins.clear()
        pending_width = 0
        previous_bin = None

    for value, member in groups:
        if not fits(member.size):
            flush()
            width = affordable_width(fits, member.size)
            if width < 1:
                raise MemoryError(
                    "One index of the selection does not fit the operation budget"
                )
            for start in range(0, member.size, width):
                piece = member[start : start + width]
                blocks.append(
                    IndexBlock(
                        indices=resolved[piece],
                        destinations=positions[piece],
                        bins=(value,),
                    )
                )
            continue
        adjacent = previous_bin is None or value == previous_bin + 1
        if pending and (not adjacent or not fits(pending_width + member.size)):
            flush()
        pending.append(member)
        pending_bins.append(value)
        pending_width += int(member.size)
        previous_bin = value
    flush()
    return blocks
