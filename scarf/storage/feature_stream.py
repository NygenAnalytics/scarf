"""Geometry-aware planning for feature-column streams."""

import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .budget import ResourceBudget, admit_stream
from .geometry import ArrayGeometry, array_geometry
from .partition import (
    IndexBlock,
    affordable_width,
    checked_indices,
    partition_indices,
)

type BlockBytes = Callable[[int], int]

__all__ = [
    "FeatureStreamPlan",
    "feature_column_chunk",
    "plan_feature_stream",
]


@dataclass(frozen=True, slots=True)
class FeatureStreamPlan:
    """Ordered feature blocks and their admitted read concurrency."""

    geometry: ArrayGeometry
    featureAxis: int
    blocks: tuple[IndexBlock, ...]
    readWorkers: int
    ioConcurrency: int
    repeatedDecodeCount: int


def _axis(value: int, *, name: str) -> int:
    resolved = operator.index(value)
    if resolved not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1")
    return int(resolved)


def _plane(array: Any) -> ArrayGeometry:
    geometry = array_geometry(array)
    if geometry is None or len(geometry.shape) != 2:
        raise ValueError("Feature streams require a chunked two-dimensional array")
    return geometry


def feature_column_chunk(array: Any, *, featureAxis: int) -> int:
    """Return one physical feature-chunk width."""
    return _plane(array).axisChunk(_axis(featureAxis, name="featureAxis"))


def _positive_requested(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("requestedBatchSize must be a positive integer")
    try:
        resolved = operator.index(value)
    except TypeError:
        raise TypeError("requestedBatchSize must be a positive integer") from None
    if resolved < 1:
        raise ValueError("requestedBatchSize must be greater than zero")
    return int(resolved)


def _owned_bytes(blockBytes: BlockBytes, width: int) -> int:
    value = int(blockBytes(max(1, int(width))))
    if value < 1:
        raise ValueError("blockBytes must return a positive byte count")
    return value


def _repeated_decodes(
    blocks: Sequence[IndexBlock],
    *,
    cell_bin_count: int,
) -> int:
    touches: dict[int, int] = {}
    for block in blocks:
        for feature_bin in block.bins:
            touches[feature_bin] = touches.get(feature_bin, 0) + 1
    return sum(max(0, count - 1) * cell_bin_count for count in touches.values())


def plan_feature_stream(
    array: Any,
    *,
    featureAxis: int,
    cellAxis: int,
    featureIndices: Sequence[int] | np.ndarray,
    cellIndices: Sequence[int] | np.ndarray,
    resources: ResourceBudget,
    blockBytes: BlockBytes,
    residentBytes: int = 0,
    requestedBatchSize: int | None = None,
) -> FeatureStreamPlan:
    """Plan variable-width feature blocks from physical chunk geometry."""
    feature_axis = _axis(featureAxis, name="featureAxis")
    cell_axis = _axis(cellAxis, name="cellAxis")
    if feature_axis == cell_axis:
        raise ValueError("featureAxis and cellAxis must differ")

    geometry = _plane(array)
    feature_indices = checked_indices(
        featureIndices,
        limit=geometry.shape[feature_axis],
        name="featureIndices",
    )
    cell_indices = checked_indices(
        cellIndices,
        limit=geometry.shape[cell_axis],
        name="cellIndices",
    )
    requested = _positive_requested(requestedBatchSize)
    resident = max(0, int(residentBytes))
    available = resources.memoryBytes - resident
    if available <= 0:
        raise MemoryError(
            f"Resident data needs {resident} bytes, but the operation limit is "
            f"{resources.memoryBytes} bytes"
        )

    decode_bytes = geometry.nominalChunkBytes()

    def fits(width: int) -> bool:
        return _owned_bytes(blockBytes, width) + decode_bytes <= available

    if feature_indices.size == 0:
        return FeatureStreamPlan(
            geometry=geometry,
            featureAxis=feature_axis,
            blocks=(),
            readWorkers=1,
            ioConcurrency=1,
            repeatedDecodeCount=0,
        )

    if requested is not None:
        blocks = partition_indices(
            geometry,
            feature_axis,
            feature_indices,
            maxWidth=requested,
        )
        if any(not fits(block.indices.size) for block in blocks):
            raise MemoryError(
                f"Requested feature batch width {requested} does not fit; "
                f"the affordable width is {affordable_width(fits, requested)}"
            )
    else:
        blocks = partition_indices(
            geometry,
            feature_axis,
            feature_indices,
            fits=fits,
        )

    block_bytes = max(_owned_bytes(blockBytes, block.indices.size) for block in blocks)
    prefetchable = len(blocks) - 1
    # A read-ahead stream holds the block being consumed while the next ones load.
    # A single-block stream holds nothing before its own read.
    held = resident + (block_bytes + decode_bytes if prefetchable else 0)
    read_workers = 1
    io_concurrency = 1
    try:
        admission = admit_stream(
            resources,
            nBlocks=max(1, prefetchable),
            blockBytes=block_bytes,
            decodeBytes=decode_bytes,
            residentBytes=held,
        )
    except MemoryError:
        # A second materialized block may not fit, while the current block can
        # still use the remaining budget for concurrent chunk decodes.
        current_admission = admit_stream(
            resources,
            nBlocks=1,
            blockBytes=block_bytes,
            decodeBytes=decode_bytes,
            residentBytes=resident,
        )
        io_concurrency = current_admission.ioConcurrency
    else:
        io_concurrency = admission.ioConcurrency
        if prefetchable:
            read_workers = admission.outerWorkers

    cell_bins = int(np.unique(geometry.binOf(cell_axis, cell_indices)).size)
    return FeatureStreamPlan(
        geometry=geometry,
        featureAxis=feature_axis,
        blocks=tuple(blocks),
        readWorkers=read_workers,
        ioConcurrency=io_concurrency,
        repeatedDecodeCount=_repeated_decodes(blocks, cell_bin_count=cell_bins),
    )
