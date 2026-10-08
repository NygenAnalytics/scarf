"""Bulk aggregation of cell groups over stored or normalized matrices.

It also builds the bulk profiles that ``DataStore.make_bulk`` returns: the
cells of each bulk column, each column's mean or summed feature profile, and
the frames labelled by feature.
"""

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pandas as pd
import zarr
from numba import njit, prange
from numpy.typing import NDArray

from ..assay import Assay, RNAassay, lib_size_feature_stream_eligible
from ..assay.normalization import library_size_divisors
from ..matrix import ChunkedArray
from ..metadata import MetaData
from ..metadata.rows import read_metadata_rows_chunkwise
from ..storage.budget import ResourceBudget
from ..storage.feature_stream import FeatureCellBand, map_feature_cell_bands
from ..storage.geometry import array_geometry
from ..storage.io_policy import DEFAULT_STORAGE_IO_POLICY, StorageIoPolicy
from ..utils.compute import controlled_compute
from ..utils.progress import iter_progress


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


def _require_finite_groups(
    values: np.ndarray,
    group_names: Sequence[str] | None,
    *,
    statistic: str,
) -> None:
    """Raise unless every value of a groups-by-features result is finite."""
    if values.dtype.kind != "f":
        return
    finite = np.isfinite(values)
    if finite.all():
        return
    group, feature = (int(index) for index in np.argwhere(~finite)[0])
    label = repr(group_names[group]) if group_names is not None else str(group)
    raise ValueError(
        f"Bulk {statistic} of feature {feature} in group {label} is not finite"
    )


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
    group_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return features-by-groups count sums, or means normalized by ``scalars``."""
    if group_codes.shape != cell_indices.shape or np.any(
        (group_codes < 0) | (group_codes >= n_groups)
    ):
        raise ValueError("Bulk group codes must align with selected cells")
    if scalars is not None:
        if scalars.shape != cell_indices.shape:
            raise ValueError(
                "Bulk normalization scalars must align with selected cells"
            )
        if not (np.isfinite(scalars) & (scalars > 0)).all():
            raise ValueError(
                "Bulk normalization scalars must be finite and positive; "
                "library_size_divisors maps a zero total to 1"
            )
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
    _require_finite_groups(
        values, group_names, statistic="sum" if scalars is None else "mean"
    )
    return values.T, None if fractions is None else fractions.T


def aggregate_normalized_groups(
    normalized: ChunkedArray,
    group_codes: np.ndarray,
    n_groups: int,
    *,
    nthreads: int,
    group_names: Sequence[str] | None = None,
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
        group_names: Optional group names used in errors.

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
    _require_finite_groups(sums, group_names, statistic="mean")
    return sums.T


def _bulk_replicates(cells: NDArray[Any], n_reps: int, seed: int) -> list[NDArray[Any]]:
    """Split one group's cells into ``n_reps`` pseudo-replicates.

    Each group draws its permutation from a new ``RandomState(seed)``, and
    each replicate lists its cells in ascending order.
    """
    v_list = list(cells)
    random_state = np.random.RandomState(seed)
    shuffled_idx = random_state.choice(v_list, len(v_list), replace=False)
    rep_idx = np.array_split(shuffled_idx, n_reps)
    return [np.array(sorted(x)) for x in rep_idx]


def bulk_column_rows(
    group_values: NDArray[Any],
    group_order: Sequence[Any],
    labelled: NDArray[np.bool_],
    cell_idx: NDArray[np.int64],
    *,
    secondary: tuple[NDArray[Any], Sequence[Any]] | None,
    null_vals: Sequence[Any] | None,
    secondary_null_vals: Sequence[Any] | None,
    pseudo_reps: int,
    random_seed: int,
) -> dict[str, NDArray[Any]]:
    """Return the global cell indices of each bulk column, keyed by column name."""
    if null_vals is None:
        null_vals = []
    if secondary_null_vals is None:
        secondary_null_vals = []
    if secondary is None:
        sec_group_values: NDArray[Any] = np.array([None], dtype=object)
        sec_groups_set: Sequence[Any] = [None]
    else:
        sec_group_values, sec_groups_set = secondary
    column_rows: dict[str, NDArray[Any]] = {}
    for g in group_order:
        if g in null_vals:
            continue
        for sg in sec_groups_set:
            if sg in secondary_null_vals:
                continue
            if sg is None and len(sec_group_values) == 1:
                selected_rows = np.flatnonzero((group_values == g) & labelled)
            else:
                selected_rows = np.flatnonzero(
                    (group_values == g) & (sec_group_values == sg) & labelled
                )
            g_idx = cell_idx[selected_rows]
            rep_indices = _bulk_replicates(g_idx, pseudo_reps, random_seed)
            for n, idx in enumerate(rep_indices):
                if sg is None and len(sec_group_values) == 1:
                    col_name = f"{g}"
                else:
                    col_name = f"{g}_{sg}"
                if pseudo_reps > 1:
                    col_name += f"_Rep{n + 1}"
                if col_name in column_rows:
                    raise ValueError(
                        f"Bulk column name {col_name!r} is produced by more "
                        "than one group; rename the group values"
                    )
                column_rows[col_name] = idx
    return column_rows


def bulk_codes(
    cell_idx: NDArray[np.int64],
    column_rows: Mapping[str, NDArray[Any]],
) -> NDArray[np.int64]:
    """Return the bulk column of each selected cell, or -1 for cells in none.

    Codes number the columns in the order of ``column_rows``.
    """
    codes = np.full(len(cell_idx), -1, dtype=np.int64)
    for code, idx in enumerate(column_rows.values()):
        codes[np.searchsorted(cell_idx, idx)] = code
    return codes


def bulk_read_cells(
    assay: Assay,
    aggr_type: str,
    cell_idx: NDArray[np.int64],
    column_rows: Mapping[str, NDArray[Any]],
) -> NDArray[np.int64]:
    """Return the global indices of the cells that ``aggregate_bulk_profiles`` reads."""
    if not column_rows or len(cell_idx) == 0:
        return np.zeros(0, dtype=np.int64)
    stream_rna = isinstance(assay, RNAassay) and (
        aggr_type == "sum"
        or (aggr_type == "mean" and lib_size_feature_stream_eligible(assay))
    )
    if stream_rna or aggr_type == "sum":
        return cell_idx[bulk_codes(cell_idx, column_rows) >= 0]
    if aggr_type == "mean":
        return cell_idx
    return np.zeros(0, dtype=np.int64)


def aggregate_bulk_profiles(
    assay: Assay,
    column_rows: Mapping[str, NDArray[Any]],
    cell_idx: NDArray[np.int64],
    *,
    aggr_type: str,
    return_fraction: bool,
    cells: MetaData,
    resources: ResourceBudget,
    nthreads: int,
) -> tuple[dict[str, NDArray[Any]], dict[str, NDArray[Any]]]:
    """Return each bulk column's feature profile and expressing fractions."""
    vals: dict[str, NDArray[Any]] = {}
    fracs: dict[str, NDArray[Any]] = {}
    codes = bulk_codes(cell_idx, column_rows)
    stream_rna = isinstance(assay, RNAassay) and (
        aggr_type == "sum"
        or (aggr_type == "mean" and lib_size_feature_stream_eligible(assay))
    )
    if stream_rna and column_rows:
        assert isinstance(assay, RNAassay)
        included = codes >= 0
        selected = cell_idx[included]
        # The divisors of ``normed``: a cell without counts averages in as
        # zeros rather than dividing zero by zero.
        scalars = (
            library_size_divisors(
                read_metadata_rows_chunkwise(cells, f"{assay.name}_nCounts", selected),
                source=f"{assay.name}_nCounts",
            )
            if aggr_type == "mean"
            else None
        )
        # An RNA assay opens only with a complete countsT and a size factor.
        values, fractions = aggregate_rna_groups(
            cast(zarr.Array, assay.rawDataT),
            selected,
            codes[included],
            len(column_rows),
            scalars=scalars,
            size_factor=cast(int, assay.sf),
            return_fraction=return_fraction,
            resources=resources,
            io=assay.storageIo,
            group_names=list(column_rows),
        )
        vals = {name: values[:, i] for i, name in enumerate(column_rows)}
        if fractions is not None:
            fracs = {name: fractions[:, i] for i, name in enumerate(column_rows)}
    else:
        if aggr_type not in ("sum", "mean"):
            raise ValueError("ERROR: `aggr_type` can only be either 'sum' or 'mean'")
        if aggr_type == "mean" and column_rows and len(cell_idx):
            # Fit the normalization once over every selected cell.
            means = aggregate_normalized_groups(
                assay.normed(
                    cell_idx=cell_idx,
                    feat_idx=np.arange(assay.feats.N, dtype=np.int64),
                ),
                codes,
                len(column_rows),
                nthreads=nthreads,
                group_names=list(column_rows),
            )
            vals = {name: means[:, i] for i, name in enumerate(column_rows)}
        # Raw sums are exact: integer counts keep NumPy's integer accumulator.
        sum_dtype = np.float64 if assay.rawData.dtype.kind == "f" else None
        for col_name, idx in iter_progress(
            column_rows.items(),
            desc="Aggregating pseudo-replicates",
            total=len(column_rows),
        ):
            if len(idx) == 0:
                vals[col_name] = np.zeros(assay.feats.N)
                if return_fraction:
                    fracs[col_name] = np.zeros(assay.feats.N)
                continue
            if aggr_type == "sum":
                vals[col_name] = controlled_compute(
                    assay.rawData[idx].sum(axis=0, dtype=sum_dtype), nthreads
                )
            if return_fraction:
                fracs[col_name] = (assay.rawData[idx] > 0).mean(axis=0).compute()
    return vals, fracs


def aligned_feature_labels(values: np.ndarray, index: pd.Index) -> np.ndarray:
    """Return feature labels as a hashable NumPy array aligned to ``index``."""
    labels = np.asarray(values, dtype=object).reshape(-1)
    return labels[np.asarray(index, dtype=np.intp)]


def bulk_frames(
    vals: Mapping[str, NDArray[Any]],
    fracs: Mapping[str, NDArray[Any]],
    features: MetaData,
    *,
    remove_empty_features: bool,
    feature_label: str,
    return_fraction: bool,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Return the bulk profiles, and fractions on request, as frames."""
    vals_df = pd.DataFrame(vals)

    empty_idx = None
    if remove_empty_features:
        empty_idx = vals_df.sum(axis=1) != 0
        vals_df = vals_df.loc[empty_idx]

    if feature_label == "id":
        vals_df.set_index(
            aligned_feature_labels(features.fetch_all("ids"), vals_df.index),
            inplace=True,
            drop=True,
        )
    elif feature_label == "name":
        vals_df.set_index(
            aligned_feature_labels(features.fetch_all("names"), vals_df.index),
            inplace=True,
            drop=True,
        )

    if return_fraction:
        fracs_df = pd.DataFrame(fracs)
        if empty_idx is not None:
            fracs_df = fracs_df[empty_idx]
        fracs_df.set_index(vals_df.index, inplace=True, drop=True)
        return vals_df, fracs_df
    return vals_df
