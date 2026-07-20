from typing import Any

import numpy as np
import zarr


def _read_block(
    zarr_arr: zarr.Array,
    row_idx: np.ndarray,
    col_idx: np.ndarray,
) -> np.ndarray:
    """Read ``zarr_arr[row_idx, col_idx]`` returning rows/cols in index order.

    A basic slice is used only for an index run that is provably contiguous
    (consecutive ascending integers), so it selects exactly the requested
    positions and never includes neighbouring rows or columns. Any other
    selection falls back to orthogonal (fancy) indexing, which preserves the
    order of the index arrays. This centralizes the read path so callers never
    hand-roll ``slice(idx[0], idx[-1] + 1)``.
    """
    from ..matrix._indexing import is_contiguous

    def axis_sel(idx: np.ndarray) -> slice | np.ndarray:
        idx = np.asarray(idx)
        if idx.size > 0 and is_contiguous(idx):
            return slice(int(idx[0]), int(idx[-1]) + 1)
        return idx

    row_sel = axis_sel(row_idx)
    col_sel = axis_sel(col_idx)
    if isinstance(row_sel, slice) and isinstance(col_sel, slice):
        return np.asarray(zarr_arr[row_sel, col_sel])
    return np.asarray(zarr_arr.get_orthogonal_selection((row_sel, col_sel)))


def _feature_stats_tile_shape(
    n_cells: int,
    n_features: int,
    *,
    row_chunk: int,
    col_chunk: int,
    budget: Any | None = None,
    target_bytes: int | None = None,
) -> tuple[int, int]:
    """Choose a dense stats tile that fits the active memory budget.

    Full-width feature chunks (common on small matrices) would otherwise
    materialize ``n_cells x n_features`` uint32+float64 temporaries at once.
    """
    from ..storage.budget import ResourceBudget, get_resource_budget

    if budget is None:
        resolved = get_resource_budget()
    elif isinstance(budget, ResourceBudget):
        resolved = budget
    else:
        raise TypeError("budget must be a ResourceBudget or None")
    work = resolved.memoryBytes // max(1, resolved.workingCopies)
    if target_bytes is None:
        target_bytes = min(work // 4, 256 * 1024 * 1024)
    target_bytes = max(8 * 1024 * 1024, int(target_bytes))
    rows = max(1, min(int(n_cells), max(1, int(row_chunk))))
    cols = max(1, min(int(n_features), max(1, int(col_chunk))))
    # One uint32 source band plus one float64 scaled band.
    bytes_per_element = 12
    while rows > 1 and rows * cols * bytes_per_element > target_bytes:
        rows = max(1, rows // 2)
    while cols > 1 and rows * cols * bytes_per_element > target_bytes:
        cols = max(1, cols // 2)
    return rows, cols
