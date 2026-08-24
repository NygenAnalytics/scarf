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
