from typing import Any

import numba
import numpy as np
import pandas as pd
from numba import set_num_threads

from ...assay import Assay, ATACassay, RNAassay, lib_size_feature_stream_eligible
from ...assay.normalization import (
    clr_values,
    inverse_document_frequency,
    norm_clr,
    norm_dummy,
    norm_tf_idf,
    reject_unknown_normalization_params,
    tfidf_values,
)
from ...utils.arguments import integer_argument
from ...utils.arrays import has_duplicates
from ...utils.compute import compute_with_progress
from ...utils.logging import logger
from ...utils.numba import restore_numba_threads
from ..statistical import adjust_pvalues
from .rank import (
    _batch_stats,
    _marker_stats_gene_major,
    batch_rank_scratch_bytes,
    gene_major_rank_scratch_bytes,
)
from .regression import _REG_OK, _REG_SENTINEL, _regression_batch_results
from .table import (
    RankMarkerResult,
    _validate_rank_marker_groups,
    stored_table_bytes,
)

__all__ = ["find_markers_by_rank", "find_markers_by_regression"]


# Normalizations whose values the rank search computes from raw-count batches.
_NORMALIZATION_ADAPTERS = {norm_tf_idf: "tfidf", norm_clr: "clr", norm_dummy: "dummy"}
# Bytes per value that a dense adapter holds while it normalizes and ranks a
# read group: its float64 intermediates and the dense kernel's C-order copy.
_DENSE_VALUE_BYTES = {"tfidf": 16, "clr": 16, "dummy": 8}


def rank_marker_adapter(assay: Assay) -> str | None:
    """Return the raw-count adapter that rank marker search applies, if any.

    Adapters compute normalized values from ``countsT`` batches themselves.
    Library-size normalization of RNA counts of any storage dtype, with or
    without subset renormalization, has one adapter. Without one, marker
    search reads ``iter_normed_feature_wise``.
    """
    if lib_size_feature_stream_eligible(assay):
        if not isinstance(assay, RNAassay):
            raise TypeError("Library-size marker search requires an RNAassay instance")
        return "lib_size"
    if getattr(assay, "rawDataT", None) is None:
        return None
    return _NORMALIZATION_ADAPTERS.get(assay.normMethod)


@restore_numba_threads
def find_markers_by_rank(
    assay: Assay,
    groups: np.ndarray,
    cell_idx: np.ndarray,
    feat_idx: np.ndarray,
    nthreads: int = 1,
    *,
    writers: int = 1,
    **norm_params: Any,
) -> RankMarkerResult:
    """Rank one-versus-rest markers of every group over explicit cells and features.

    Library-size normalized RNA counts of any storage dtype, with or without
    ``renormalize_subset``, are ranked from ``countsT`` by one zero-aware
    kernel, which requires finite non-negative counts and cell totals.
    TF-IDF, CLR, and unnormalized counts of an assay with ``countsT`` are
    ranked from it by the dense kernel, and every other assay or
    normalization from ``iter_normed_feature_wise``. The search reserves
    within the assay's memory budget everything it holds: its streamed blocks
    and their Zarr reads, kernel scratch, the result, and the stored tables
    of ``writers`` groups that are finished from the result at once.

    Args:
        assay: Assay whose features are ranked.
        groups: Group label of each cell in ``cell_idx``.
        cell_idx: Unique assay cell indices.
        feat_idx: Unique assay feature indices.
        nthreads: Thread limit for normalized feature batches, and the worker
            count when the assay has no resource budget.
        writers: Groups whose stored tables are finished and written from the
            result at once.
        **norm_params: ``log_transform`` and ``renormalize_subset``.

    Returns:
        The rank statistics of every group over the sorted features.

    Raises:
        IndexError: If a cell or feature index is past the end of the assay.
        ValueError: If the indices are empty, negative, or repeated, or do not
            align with ``groups``, there are fewer than two groups, a group has
            fewer than two cells, or a library-size normalized value or the
            total of a selected cell (its ``<assay>_nCounts`` value, or with
            ``renormalize_subset`` its sum over the tested features) is
            negative or not finite.
    """
    reject_unknown_normalization_params(
        norm_params,
        caller="find_markers_by_rank",
    )
    groups = np.asarray(groups)
    cell_idx = np.asarray(cell_idx, dtype=np.int64)
    feat_idx = np.asarray(feat_idx, dtype=np.int64)
    if groups.ndim != 1 or cell_idx.ndim != 1 or feat_idx.ndim != 1:
        raise ValueError("groups, cell_idx, and feat_idx must be one-dimensional")
    if len(groups) != len(cell_idx):
        raise ValueError("groups must align with cell_idx")
    if len(cell_idx) == 0 or len(feat_idx) == 0:
        raise ValueError("Marker search requires non-empty cell and feature indices")
    if (
        np.any(cell_idx < 0)
        or has_duplicates(cell_idx)
        or np.any(feat_idx < 0)
        or has_duplicates(feat_idx)
    ):
        raise ValueError(
            "cell_idx and feat_idx must contain unique non-negative indices"
        )
    writers = integer_argument(writers, "writers", minimum=1)
    group_ids, codes = np.unique(groups, return_inverse=True)
    codes = codes.astype(np.int64, copy=False)
    group_sizes = np.bincount(codes, minlength=len(group_ids))
    _validate_rank_marker_groups(group_sizes)
    feature_index = np.sort(feat_idx)
    statistics = np.zeros((len(feature_index), len(group_ids), 8), dtype=np.float64)
    # The search holds these until the last group's stored table is written.
    held_bytes = (
        codes.nbytes
        + feature_index.nbytes
        + statistics.nbytes
        + writers * stored_table_bytes(len(feature_index))
    )
    adapter = rank_marker_adapter(assay)
    if adapter is None:
        _rank_normed_batches(
            assay,
            codes,
            group_sizes,
            cell_idx,
            feature_index,
            statistics,
            nthreads=nthreads,
            held_bytes=held_bytes,
            norm_params=norm_params,
        )
    else:
        _rank_counts_t(
            assay,
            adapter,
            codes,
            group_sizes,
            cell_idx,
            feature_index,
            statistics,
            nthreads=nthreads,
            held_bytes=held_bytes,
            norm_params=norm_params,
        )
    return RankMarkerResult(
        group_ids=group_ids,
        group_sizes=group_sizes,
        feature_index=feature_index,
        statistics=statistics,
    )


def _rank_counts_t(
    assay: Assay,
    adapter: str,
    codes: np.ndarray,
    group_sizes: np.ndarray,
    cell_idx: np.ndarray,
    feature_index: np.ndarray,
    statistics: np.ndarray,
    *,
    nthreads: int,
    held_bytes: int,
    norm_params: dict[str, Any],
) -> None:
    """Rank ``countsT`` read groups in order with the threads of one worker."""
    from ...storage.budget import resolve_budget
    from ...storage.feature_stream import (
        FeatureReadGroup,
        map_feature_read_groups,
        persisted_read_group,
        selected_feature_values,
    )

    # Every adapter reads countsT, so its presence selects one.
    counts_t = assay.rawDataT
    assert counts_t is not None
    # The stream leaves the columns of cells past the end unwritten.
    if int(cell_idx.max()) >= int(counts_t.shape[1]):
        raise IndexError("cell_idx contains an out-of-range index")
    if int(feature_index[-1]) >= int(counts_t.shape[0]):
        raise IndexError("feat_idx contains an out-of-range index")
    n_cells = len(codes)
    n_groups = len(group_sizes)
    resources = getattr(assay, "resources", None) or resolve_budget(workers=nthreads)
    # One compute worker ranks each read group with all of its threads.
    threads = min(
        max(1, int(resources.workers)),
        max(1, int(numba.config.NUMBA_NUM_THREADS)),
    )
    group_features = min(len(feature_index), persisted_read_group(counts_t)[0])
    dest_of = np.full(int(counts_t.shape[0]), -1, dtype=np.int64)
    dest_of[feature_index] = np.arange(len(feature_index), dtype=np.int64)
    # Float32 sizes would round a product of group sizes above 2**24.
    group_counts = group_sizes.astype(np.float64)
    log_transform = bool(norm_params.get("log_transform", False))
    totals: np.ndarray | None = None
    cell_scale: np.ndarray | None = None
    feature_scale: np.ndarray | None = None
    size_factor = 1.0
    if adapter == "lib_size":
        # Eligibility for the adapter requires a size factor.
        assert assay.sf is not None
        size_factor = float(assay.sf)
        if norm_params.get("renormalize_subset", False):
            # The subset totals that ``normed`` divides by.
            source = f"The tested features of {assay.name} hold"
            totals = compute_with_progress(
                assay.rawData[:, feature_index][cell_idx, :].sum(
                    axis=1, dtype=np.float64
                ),
                "Normalizing with feature subset",
                nthreads,
            )
        else:
            column = assay.name + "_nCounts"
            source = f"{column} holds"
            totals = np.asarray(
                assay.cells.fetch_all(column)[cell_idx],
                dtype=np.float64,
            )
        if not (np.isfinite(totals) & (totals >= 0)).all():
            raise ValueError(
                f"{source} negative or non-finite totals of selected cells; "
                "library-size normalization requires finite non-negative counts"
            )
        totals[totals == 0] = 1
        state_bytes = totals.nbytes
        compute_bytes = gene_major_rank_scratch_bytes(
            n_cells=n_cells,
            n_groups=n_groups,
            n_features=group_features,
            nthreads=threads,
        )
    else:
        state_bytes = 0
        if adapter == "tfidf":
            if not isinstance(assay, ATACassay):
                raise TypeError("TF-IDF marker search requires an ATACassay instance")
            _, (term_totals, n_docs, document_frequency) = assay._fit_tf_idf(
                cell_idx, feature_index, **norm_params
            )
            cell_scale = np.asarray(term_totals, dtype=np.float64)
            feature_scale = inverse_document_frequency(n_docs, document_frequency)
            state_bytes = cell_scale.nbytes + feature_scale.nbytes
        # A read group's values, the copy of its selected raw rows, and the
        # dense kernel's scratch exist one read group at a time.
        value_bytes = _DENSE_VALUE_BYTES[adapter] + np.dtype(counts_t.dtype).itemsize
        compute_bytes = group_features * n_cells * value_bytes + (
            batch_rank_scratch_bytes(
                n_cells=n_cells,
                n_groups=n_groups,
                n_features=group_features,
                nthreads=threads,
            )
        )

    def process_group(group: FeatureReadGroup) -> None:
        local_dest = dest_of[group.featStart : group.featEnd]
        if adapter == "lib_size":
            assert totals is not None
            invalid = _marker_stats_gene_major(
                group.values,
                totals,
                size_factor,
                log_transform,
                codes,
                group_counts,
                float(n_cells),
                local_dest,
                threads,
                statistics,
            )
            if invalid >= 0:
                raise ValueError(
                    f"Feature {group.featStart + invalid} of {assay.name} has a "
                    "negative or non-finite library-size normalized value; "
                    "library-size normalization requires finite non-negative "
                    "counts"
                )
            return
        selected = local_dest >= 0
        raw = selected_feature_values(group.values, selected)
        rows = local_dest[selected]
        if adapter == "tfidf":
            assert cell_scale is not None
            assert feature_scale is not None
            values = tfidf_values(raw.T, cell_scale, feature_scale[rows])
        elif adapter == "clr":
            # Every selected cell is in the batch, so CLR is fitted on all
            # of them.
            values = clr_values(raw.T)
        else:
            values = np.asarray(raw.T)
        statistics[rows] = _batch_stats(
            values,
            codes,
            group_sizes,
            n_cells,
            feature_labels=feature_index[rows],
        )

    logger.debug(
        f"Marker search read groups: features={len(feature_index)} "
        f"groups={n_groups} adapter={adapter} workers={resources.workers} "
        f"threads={threads} memoryBytes={resources.memoryBytes}"
    )
    for _ in map_feature_read_groups(
        counts_t,
        process_group,
        cell_idx=cell_idx,
        feat_idx=feature_index,
        resources=resources,
        progress="Finding markers",
        io=getattr(assay, "storageIo", None),
        scratchBytes=held_bytes + dest_of.nbytes + state_bytes + compute_bytes,
    ):
        pass


def _rank_normed_batches(
    assay: Assay,
    codes: np.ndarray,
    group_sizes: np.ndarray,
    cell_idx: np.ndarray,
    feature_index: np.ndarray,
    statistics: np.ndarray,
    *,
    nthreads: int,
    held_bytes: int,
    norm_params: dict[str, Any],
) -> None:
    """Rank normalized feature batches with the dense kernel on this thread."""
    worker_limit = getattr(
        getattr(assay, "resources", None),
        "workers",
        nthreads,
    )
    threads = min(
        max(1, nthreads),
        max(1, int(worker_limit)),
        numba.config.NUMBA_NUM_THREADS,
    )
    set_num_threads(threads)
    n_cells = len(codes)
    n_groups = len(group_sizes)
    start = 0
    for batch in assay.iter_normed_feature_wise(
        cell_idx=cell_idx,
        feat_idx=feature_index,
        batch_size=None,
        msg="Finding markers",
        # The dense kernel's C-order float64 copy of each value, and its
        # statistics of each feature spread over the feature's cells.
        scratch_itemsize=8 + -(-64 * n_groups // n_cells),
        resident_bytes=held_bytes
        + batch_rank_scratch_bytes(
            n_cells=n_cells,
            n_groups=n_groups,
            n_features=0,
            nthreads=threads,
        ),
        **norm_params,
    ):
        if isinstance(batch, pd.DataFrame):
            values = batch.to_numpy()
            labels = np.asarray(batch.columns)
        else:
            feature_major, labels = batch
            values = np.asarray(feature_major).T
        stop = start + values.shape[1]
        if not np.array_equal(labels, feature_index[start:stop]):
            raise RuntimeError(
                "Normalized feature batches must follow the requested features"
            )
        statistics[start:stop] = _batch_stats(
            values,
            codes,
            group_sizes,
            n_cells,
            feature_labels=np.asarray(labels),
        )
        start = stop
    if start != len(feature_index):
        raise RuntimeError("Normalized feature batches must cover every feature")


@restore_numba_threads
def find_markers_by_regression(
    assay: Assay,
    cell_idx: np.ndarray,
    feat_idx: np.ndarray,
    regressor: np.ndarray,
    min_cells: int,
    batch_size: int | None = None,
    **norm_params: Any,
) -> pd.DataFrame:
    """Find features correlated with a continuous variable."""
    reject_unknown_normalization_params(
        norm_params,
        caller="find_markers_by_regression",
    )
    cell_idx = np.asarray(cell_idx, dtype=np.int64)
    feat_idx = np.asarray(feat_idx, dtype=np.int64)
    if cell_idx.ndim != 1 or feat_idx.ndim != 1:
        raise ValueError("cell_idx and feat_idx must be one-dimensional")
    if len(cell_idx) == 0 or len(feat_idx) == 0:
        raise ValueError("Marker regression requires non-empty indices")
    if (
        np.any(cell_idx < 0)
        or has_duplicates(cell_idx)
        or np.any(feat_idx < 0)
        or has_duplicates(feat_idx)
    ):
        raise ValueError(
            "cell_idx and feat_idx must contain unique non-negative indices"
        )
    regressor = np.asarray(regressor, dtype=np.float64)
    if regressor.ndim != 1:
        raise ValueError("regressor must be one-dimensional")
    if len(regressor) != len(cell_idx):
        raise ValueError("regressor must align with cell_idx")
    if not np.isfinite(regressor).all():
        raise ValueError("regressor must contain only finite values")
    if regressor.size < 2 or np.unique(regressor).size < 2:
        raise ValueError("regressor must contain at least two distinct values")
    if min_cells < 1:
        raise ValueError("min_cells must be at least 1")

    nthreads = getattr(assay, "nthreads", 1)
    set_num_threads(min(max(1, int(nthreads)), numba.config.NUMBA_NUM_THREADS))
    # Pearson r does not depend on the regressor's scale, and a power-of-two
    # scale is exact, so scaling the regressor below one in magnitude keeps
    # its centered sums finite and leaves r unchanged.
    regressor = np.ldexp(regressor, -np.frexp(np.abs(regressor).max())[1])
    x_centered = regressor - regressor.mean()
    ssxm = float(np.dot(x_centered, x_centered) / regressor.shape[0])

    labels: list[Any] = []
    r_parts: list[np.ndarray] = []
    p_parts: list[np.ndarray] = []
    status_parts: list[np.ndarray] = []
    for feature_major, raw_labels in assay.iter_normed_feature_wise(
        cell_idx=cell_idx,
        feat_idx=feat_idx,
        batch_size=batch_size,
        msg="Finding correlated features",
        as_dataframe=False,
        **norm_params,
    ):
        data = np.asarray(feature_major, dtype=np.float64).T
        feat_labels = np.asarray(raw_labels)
        if data.ndim != 2 or data.shape[0] != regressor.shape[0]:
            raise ValueError(
                "Regressor length does not match the number of selected cells"
            )
        r_vals, p_vals, status = _regression_batch_results(
            data,
            x_centered,
            ssxm,
            regressor,
            min_cells,
            feat_labels,
        )
        labels.extend(feat_labels.tolist())
        r_parts.append(r_vals)
        p_parts.append(p_vals)
        status_parts.append(status)

    if not labels:
        return pd.DataFrame(columns=["r_value", "p_value", "p_value_adjusted"])
    r_values = np.concatenate(r_parts)
    p_values = np.concatenate(p_parts)
    status = np.concatenate(status_parts)
    adjusted = np.full(p_values.shape, np.nan, dtype=np.float64)
    tested = status == _REG_OK
    if np.any(tested):
        adjusted[tested] = adjust_pvalues(p_values[tested], "fdr_bh")
    untested = status == _REG_SENTINEL
    p_values = p_values.astype(np.float64, copy=True)
    p_values[untested] = np.nan
    return pd.DataFrame(
        {
            "r_value": r_values,
            "p_value": p_values,
            "p_value_adjusted": adjusted,
        },
        index=labels,
    )
