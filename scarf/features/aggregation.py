"""RNA bulk aggregation over the stored feature-major count matrix."""

from dataclasses import replace

import numpy as np
import zarr
from numba import njit, prange

from ..storage.budget import ResourceBudget
from ..storage.execution import WorkShape, plan_operation
from ..storage.feature_stream import (
    FeatureCellBand,
    map_feature_cell_bands,
    persisted_read_group,
)
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
    result_dtype = (
        np.dtype(np.float64)
        if scalars is not None
        else np.empty(0, dtype=counts_t.dtype).sum().dtype
    )
    # Numba requires at least float32 for floating-point arrays.
    cast_counts = np.dtype(counts_t.dtype) == np.dtype(np.float16)
    dtype = (
        np.dtype(np.float32) if result_dtype == np.dtype(np.float16) else result_dtype
    )
    feature_width, _ = persisted_read_group(counts_t)
    band_features = min(n_features, max(feature_width, geometry.axisChunk(0)))
    band_cells = min(int(counts_t.shape[1]), geometry.axisChunk(1))
    band_elements = band_features * band_cells
    output_bytes = n_features * n_groups * (dtype.itemsize + 8 * return_fraction)
    # Include DataFrame construction/filter copies and selected-cell bookkeeping.
    resident_bytes = 4 * output_bytes + 64 * len(cell_indices) + 16 * n_groups
    # Reserve band-local group codes and normalization scalars.
    scratch_bytes = resident_bytes + min(len(cell_indices), band_cells) * (
        group_codes.dtype.itemsize + (0 if scalars is None else scalars.dtype.itemsize)
    )
    if cast_counts:
        scratch_bytes += band_elements * np.dtype(np.float32).itemsize
    stream_io = replace(io or DEFAULT_STORAGE_IO_POLICY, computeWorkers=1)
    decode_bytes = (
        max(1, -(-band_features // geometry.axisChunk(0)))
        * geometry.nominalChunkBytes()
    )
    plan = plan_operation(
        resources,
        WorkShape(
            nUnits=max(1, -(-n_features // geometry.axisChunk(0)))
            * max(1, -(-int(counts_t.shape[1]) // geometry.axisChunk(1))),
            unitBytes=max(1, band_elements * geometry.itemsize),
            decodeBytes=decode_bytes,
            scratchBytes=scratch_bytes,
            ordered=True,
        ),
        stream_io,
    )
    stream_io = replace(stream_io, readWorkers=plan.readWorkers)
    scratch_bytes += plan.readWorkers * decode_bytes
    values = np.zeros((n_groups, n_features), dtype=dtype)
    fractions = (
        np.zeros((n_groups, n_features), dtype=np.float64) if return_fraction else None
    )
    group_sizes = np.bincount(group_codes, minlength=n_groups)

    def accumulate(band: FeatureCellBand) -> None:
        columns = slice(band.featStart, band.featEnd)
        _accumulate_group_counts(
            band.values.astype(np.float32) if cast_counts else band.values,
            band.selectedLocal,
            group_codes[band.selectedDestinations],
            None if scalars is None else scalars[band.selectedDestinations],
            size_factor,
            values[:, columns].T,
            None if fractions is None else fractions[:, columns].T,
        )

    for _ in map_feature_cell_bands(
        counts_t,
        accumulate,
        cell_idx=cell_indices,
        resources=resources,
        io=stream_io,
        progress="Aggregating RNA groups",
        scratchBytes=scratch_bytes,
        orderedCompute=True,
    ):
        pass
    denominators = np.maximum(group_sizes, 1)[:, None]
    if scalars is not None:
        values /= denominators
    if fractions is not None:
        fractions /= denominators
    return values.T.astype(result_dtype, copy=False), (
        None if fractions is None else fractions.T
    )
