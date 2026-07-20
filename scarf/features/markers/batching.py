_MARKER_BYTES_PER_CELL_FEATURE = 32

__all__ = ["resolve_marker_gene_batch_size"]


def resolve_marker_gene_batch_size(
    *,
    n_features: int,
    n_cells: int,
    column_chunk: int,
    memory_bytes: int | None = None,
    working_copies: int | None = None,
) -> int:
    """Choose an automatic marker batch that fits the active memory budget."""
    from ...storage.budget import get_resource_budget

    n_features = max(1, int(n_features))
    n_cells = max(1, int(n_cells))
    column_chunk = max(1, int(column_chunk))
    if memory_bytes is None or working_copies is None:
        budget = get_resource_budget()
        memory_bytes = budget.memoryBytes if memory_bytes is None else memory_bytes
        working_copies = (
            budget.workingCopies if working_copies is None else working_copies
        )
    work = max(1, int(memory_bytes)) // max(1, int(working_copies))
    budget_cap = max(1, work // (n_cells * _MARKER_BYTES_PER_CELL_FEATURE))
    return max(1, min(column_chunk, n_features, budget_cap))
