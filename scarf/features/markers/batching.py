from typing import Any

from ...storage.budget import READ_AHEAD, ResourceBudget

_MARKER_BYTES_PER_CELL_FEATURE = 32

__all__ = [
    "feature_column_chunk",
    "resolve_feature_batch_size",
    "resolve_marker_gene_batch_size",
]


def feature_column_chunk(assay: Any, n_features: int) -> int:
    """Return the on-disk feature width of one stored chunk."""
    # RNA feature-column streams (markers, HVG, pseudotime) prefer countsT
    # when present; other assays keep cell-major batch sizing.
    from ...assay import RNAassay

    if isinstance(assay, RNAassay):
        counts_t = getattr(assay, "rawDataT", None)
        if counts_t is not None:
            chunks = getattr(counts_t, "chunks", None)
            if chunks and len(chunks) > 0:
                return max(1, int(chunks[0]))
    backing = getattr(getattr(assay, "rawData", None), "_backing", None)
    chunks = getattr(backing, "chunks", None)
    if chunks and len(chunks) > 1:
        return max(1, int(chunks[1]))
    return max(1, int(n_features))


def resolve_marker_gene_batch_size(
    *,
    n_features: int,
    n_cells: int,
    column_chunk: int,
    resources: ResourceBudget,
) -> int:
    """Choose a marker batch that fits the supplied memory limit."""
    n_features = max(1, int(n_features))
    n_cells = max(1, int(n_cells))
    column_chunk = max(1, int(column_chunk))
    bytes_per_feature = n_cells * _MARKER_BYTES_PER_CELL_FEATURE
    concurrency = min(resources.workers, READ_AHEAD)
    budget_cap = resources.memoryBytes // (concurrency * bytes_per_feature)
    if budget_cap < 1:
        raise MemoryError(
            "One marker feature batch does not fit the operation memory limit"
        )
    return min(column_chunk, n_features, budget_cap)


def resolve_feature_batch_size(
    assay: Any,
    *,
    n_features: int,
    n_cells: int,
    resources: ResourceBudget,
    requested: int | None = None,
) -> int:
    """Resolve a feature-column batch, defaulting to the stored chunk width."""
    if requested is not None:
        return max(1, int(requested))
    return resolve_marker_gene_batch_size(
        n_features=n_features,
        n_cells=n_cells,
        column_chunk=feature_column_chunk(assay, n_features),
        resources=resources,
    )
