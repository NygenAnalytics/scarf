"""Bulk aggregation of cell groups over stored or normalized matrices."""

from dataclasses import replace

import numpy as np
import zarr
from numba import njit, prange

from ..matrix import ChunkedArray
from ..storage.budget import ResourceBudget
from ..storage.feature_stream import FeatureCellBand, map_feature_cell_bands
from ..storage.geometry import array_geometry
from ..storage.io_policy import DEFAULT_STORAGE_IO_POLICY, StorageIoPolicy


@njit(cache=True, nogil=True, parallel=True, error_model="numpy")
def _accumulate_group_counts(
    raw: np.ndarray,
    selected: np.ndarray,
    group_codes: np.ndarray,
    scalars: np.ndarray | None,
    size_factor: float,
    values: np.ndarray,
    fractions: np.ndarray | None,
) -> None:
    # Features own disjoint output entries; cell order stays fixed within each.
    for feature in prange(raw.shape[0]):  # type: ignore[no-untyped-call, attr-defined]
        for i in range(len(selected)):
            group = group_codes[i]
            count = raw[feature, selected[i]]
            if scalars is None:
                values[feature, group] += count
            else:
                values[feature, group] += np.float64(count) * size_factor / scalars[i]
            if fractions is not None and count > 0:
                fractions[feature, group] += 1


def aggregate_rna_groups(
    counts_t: zarr.Array,
    cell_indices: np.ndarray,
    group_codes: np.ndarray,
    n_groups: int,
    *,
    scalars: np.ndarray | None,
    size_factor: float,
    return_fraction: bool,
    resources: ResourceBudget,
    io: StorageIoPolicy | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if group_codes.shape != cell_indices.shape or np.any(
        (group_codes < 0) | (group_codes >= n_groups)
    ):
        raise ValueError("Bulk group codes must align with selected cells")
    if scalars is not None and scalars.shape != cell_indices.shape:
        raise ValueError("Bulk normalization scalars must align with selected cells")
    geometry = array_geometry(counts_t)
    assert geometry is not None
    n_features = int(counts_t.shape[0])
    # Raw sums are exact: integer counts add in the integer dtype NumPy gives
    # their sum, floating counts in float64. Means are float64.
    dtype = np.empty(0, dtype=counts_t.dtype).sum().dtype
    if scalars is not None or dtype.kind == "f":
        dtype = np.dtype(np.float64)
    band_cells = min(int(counts_t.shape[1]), geometry.axisChunk(1))
    output_bytes = n_features * n_groups * (dtype.itemsize + 8 * return_fraction)
    # Include DataFrame construction/filter copies and selected-cell bookkeeping.
    resident_bytes = 4 * output_bytes + 64 * len(cell_indices) + 16 * n_groups
    # Reserve band-local group codes and normalization scalars.
    scratch_bytes = resident_bytes + min(len(cell_indices), band_cells) * (
        group_codes.dtype.itemsize + (0 if scalars is None else scalars.dtype.itemsize)
    )

    def accumulate(band: FeatureCellBand) -> None:
        columns = slice(band.featStart, band.featEnd)
        _accumulate_group_counts(
            band.values,
            band.selectedLocal,
            group_codes[band.selectedDestinations],
            None if scalars is None else scalars[band.selectedDestinations],
            size_factor,
            values[:, columns].T,
            None if fractions is None else fractions[:, columns].T,
        )

    # The stream plans when it is created, so a budget that cannot also hold
    # the output fails before the output is allocated.
    bands = map_feature_cell_bands(
        counts_t,
        accumulate,
        cell_idx=cell_indices,
        resources=resources,
        io=replace(io or DEFAULT_STORAGE_IO_POLICY, computeWorkers=1),
        progress="Aggregating RNA groups",
        scratchBytes=scratch_bytes,
        orderedCompute=True,
    )
    values = np.zeros((n_groups, n_features), dtype=dtype)
    fractions = (
        np.zeros((n_groups, n_features), dtype=np.float64) if return_fraction else None
    )
    group_sizes = np.bincount(group_codes, minlength=n_groups)
    for _ in bands:
        pass
    denominators = np.maximum(group_sizes, 1)[:, None]
    if scalars is not None:
        values /= denominators
    if fractions is not None:
        fractions /= denominators
    return values.T, None if fractions is None else fractions.T


def aggregate_normalized_groups(
    normalized: ChunkedArray,
    group_codes: np.ndarray,
    n_groups: int,
    *,
    nthreads: int,
) -> np.ndarray:
    """Average normalized rows within each cell group in one streaming pass.

    ``normalized`` holds one row per entry of ``group_codes``. Its
    normalization was fitted once over all of its rows, so a group's profile
    does not depend on which other groups are requested. Rows coded ``-1``
    contribute to the fit but to no group.

    Args:
        normalized: Lazy normalized matrix with cells as rows.
        group_codes: Group code of each row, or ``-1``.
        n_groups: Number of groups.
        nthreads: Worker count for streaming row blocks.

    Returns:
        A features-by-groups array of means. Empty groups are zero.
    """
    codes = np.asarray(group_codes, dtype=np.int64)
    if codes.shape != (normalized.shape[0],) or np.any(
        (codes < -1) | (codes >= n_groups)
    ):
        raise ValueError("Bulk group codes must align with normalized rows")
    sums = np.zeros((n_groups, normalized.shape[1]), dtype=np.float64)
    rows, n_features = normalized.chunksize
    # The sums stay for the whole pass. A block is copied to float64 unless it
    # already is, and once more for the rows of one group at a time.
    block_copies = 1 if normalized.dtype == np.float64 else 2
    start = 0
    for block in normalized._stream_blocks(
        nthreads=nthreads,
        msg="Aggregating normalized groups",
        prefetch=None,
        row_mask=None,
        resident_bytes=sums.nbytes + (block_copies * rows + 1) * n_features * 8,
    ):
        values = np.asarray(block, dtype=np.float64)
        del block
        block_codes = codes[start : start + len(values)]
        for code in np.unique(block_codes[block_codes >= 0]):
            sums[code] += values[block_codes == code].sum(axis=0)
        start += len(values)
        # The stream only reserves the blocks it reads, not one kept here.
        del values
    sizes = np.bincount(codes[codes >= 0], minlength=n_groups)
    sums /= np.maximum(sizes, 1)[:, None]
    return sums.T
