"""Compatibility surface for the pre-planner marker batching helpers."""

import warnings
from typing import Any

from ...storage.budget import DEFAULT_READ_AHEAD_BLOCKS, ResourceBudget
from ...storage.feature_stream import feature_column_chunk as _feature_column_chunk

_MARKER_BYTES_PER_CELL_FEATURE = 32

__all__ = ["feature_column_chunk", "resolve_marker_gene_batch_size"]


def feature_column_chunk(assay: Any, n_features: int) -> int:
    """Return the on-disk feature width of one stored chunk."""
    # RNA feature-column streams (markers, HVG, pseudotime) prefer countsT
    # when present; other assays keep cell-major batch sizing.
    from ...assay import RNAassay

    if isinstance(assay, RNAassay):
        counts_t = getattr(assay, "rawDataT", None)
        if counts_t is not None:
            return _feature_column_chunk(counts_t, featureAxis=0)
    backing = getattr(getattr(assay, "rawData", None), "_backing", None)
    if backing is not None:
        return _feature_column_chunk(backing, featureAxis=1)
    return max(1, int(n_features))


def resolve_marker_gene_batch_size(
    *,
    n_features: int,
    n_cells: int,
    column_chunk: int,
    resources: ResourceBudget,
) -> int:
    """Choose a marker batch that fits the supplied memory limit.

    Superseded by ``scarf.storage.feature_stream.plan_feature_stream``, which
    reads the stored chunk geometry instead of assuming a fixed per-element
    footprint. Kept so callers written against the earlier releases still work.
    """
    warnings.warn(
        "resolve_marker_gene_batch_size is deprecated; call "
        "scarf.storage.feature_stream.plan_feature_stream instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    n_features = max(1, int(n_features))
    n_cells = max(1, int(n_cells))
    column_chunk = max(1, int(column_chunk))
    bytes_per_feature = n_cells * _MARKER_BYTES_PER_CELL_FEATURE
    concurrency = min(resources.workers, DEFAULT_READ_AHEAD_BLOCKS)
    budget_cap = resources.memoryBytes // (concurrency * bytes_per_feature)
    if budget_cap < 1:
        raise MemoryError(
            "One marker feature batch does not fit the operation memory limit"
        )
    return min(column_chunk, n_features, budget_cap)
