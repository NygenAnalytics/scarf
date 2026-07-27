from ...storage.budget import READ_AHEAD, ResourceBudget

_MARKER_BYTES_PER_CELL_FEATURE = 32

__all__ = ["resolve_marker_gene_batch_size"]


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
