from typing import Any

import numba
import numpy as np
import pandas as pd
from numba import set_num_threads
from scipy.special import ndtr

from ...assay import Assay, RNAassay, lib_size_feature_stream_eligible
from ...assay.normalization import (
    norm_clr,
    norm_dummy,
    norm_tf_idf,
    reject_unknown_normalization_params,
)
from ...utils.logging import logger
from ...utils.numba import restore_numba_threads
from .correction import _bh_adjusted_pvalues
from .rank import (
    _batch_stats,
    _marker_stats_gene_major,
    gene_major_rank_scratch_bytes,
    sort_marker_results,
)
from .regression import (
    _REG_NONFINITE,
    _REG_OK,
    _REG_SENTINEL,
    _regression_batch_results,
)
from .table import MARKER_STAT_COLUMNS

__all__ = ["find_markers_by_rank", "find_markers_by_regression"]

_KERNEL_STAT_COLUMNS = (
    "score",
    "mean",
    "mean_rest",
    "frac_exp",
    "frac_exp_rest",
    "fold_change",
    "p_value",
    "auc",
)


def _tfidf_feature_values(
    raw: np.ndarray,
    cell_scale: np.ndarray,
    feature_scale: np.ndarray,
) -> np.ndarray:
    cells = np.asarray(raw.T, dtype=np.float64)
    denom = np.asarray(cell_scale, dtype=np.float64).reshape(-1, 1)
    scale = np.asarray(feature_scale, dtype=np.float64).reshape(1, -1)
    return np.asarray((cells / denom) * scale, dtype=np.float64)


def _clr_feature_values(raw: np.ndarray) -> np.ndarray:
    cells = np.asarray(raw.T, dtype=np.float64)
    scale = np.exp(np.log1p(cells).sum(axis=0) / max(1, cells.shape[0]))
    return np.asarray(np.log1p(cells / scale), dtype=np.float64)


def _lib_size_feature_values(
    raw: np.ndarray,
    scalar: np.ndarray,
    size_factor: float,
    log_transform: bool,
) -> np.ndarray:
    cells = (
        np.float32(size_factor) * raw.T.astype(np.float32, copy=False)
    ) / np.asarray(scalar, dtype=np.float32).reshape(-1, 1)
    if log_transform:
        cells = np.log1p(cells)
    return np.asarray(cells, dtype=np.float32)


def _validate_rank_marker_groups(group_counts: np.ndarray, n_total: int) -> None:
    populated = np.asarray(group_counts, dtype=np.float64)
    if populated.size < 2:
        raise ValueError("Rank marker search requires at least two populated groups")
    if np.any(populated < 2):
        raise ValueError(
            "Rank marker search requires at least two cells in every group"
        )
    reference = n_total - populated
    if np.any(reference < 2):
        raise ValueError(
            "Rank marker search requires at least two cells in every "
            "one-versus-rest reference complement"
        )


@restore_numba_threads
def find_markers_by_rank(
    assay: Assay,
    group_key: str,
    cell_key: str,
    feat_key: str,
    nthreads: int = 1,
    **norm_params: Any,
) -> dict[Any, pd.DataFrame]:
    """Identify marker features for groups with rank-based statistics."""
    reject_unknown_normalization_params(
        norm_params,
        caller="find_markers_by_rank",
    )
    groups = assay.cells.fetch(group_key, cell_key)
    group_set = np.array(sorted(set(groups)))
    n_groups = len(group_set)
    idx_map = dict(zip(group_set, range(n_groups)))
    int_indices = np.asarray([idx_map[x] for x in groups], dtype=np.int64)
    out_cols = ["feature_index", *MARKER_STAT_COLUMNS]
    results: dict[Any, pd.DataFrame] = {}
    worker_limit = getattr(
        getattr(assay, "resources", None),
        "workers",
        nthreads,
    )
    set_num_threads(
        min(
            max(1, nthreads),
            max(1, int(worker_limit)),
            numba.config.NUMBA_NUM_THREADS,
        )
    )
    group_counts = pd.Series(groups).value_counts().reindex(group_set).values
    n_total = len(groups)
    _validate_rank_marker_groups(group_counts, n_total)

    renormalize_subset = bool(norm_params.get("renormalize_subset", False))
    log_transform = bool(norm_params.get("log_transform", False))
    use_fast = lib_size_feature_stream_eligible(
        assay,
        renormalize_subset=renormalize_subset,
    )
    if use_fast and isinstance(assay, RNAassay):
        raw_source, _, _ = assay._raw_feature_stream_source()
        use_fast = np.issubdtype(raw_source.dtype, np.unsignedinteger)
    elif use_fast:
        raise TypeError("Fast raw-count marker search requires an RNAassay instance")

    counts_t = getattr(assay, "rawDataT", None)
    has_selection = hasattr(assay.cells, "active_index") and hasattr(
        assay.feats, "active_index"
    )
    cell_idx = assay.cells.active_index(cell_key) if has_selection else None
    feat_idx = assay.feats.active_index(feat_key) if has_selection else None
    adapter: str | None = None
    cell_scale: np.ndarray | None = None
    feature_scale: np.ndarray | None = None
    scalar_values: np.ndarray | None = None
    size_factor = 1.0
    if (
        use_fast
        and isinstance(assay, RNAassay)
        and counts_t is not None
        and cell_idx is not None
        and feat_idx is not None
    ):
        adapter = "rna_lib_size_unsigned"
        scalar = assay.cells.fetch_all(assay.name + "_nCounts")[cell_idx]
        size_factor = float(assay.sf) if assay.sf is not None else 1.0
        if assay.sf is None:
            raise ValueError("RNA library-size normalization requires a size factor")
        scalar_values = np.asarray(scalar, dtype=np.float32)
        scalar_values[scalar_values == 0] = 1
    elif (
        counts_t is not None
        and cell_idx is not None
        and feat_idx is not None
        and assay.normMethod is norm_tf_idf
    ):
        adapter = "tfidf"
        assay.normed(cell_idx, feat_idx, **norm_params)
        cell_scale = np.asarray(assay.n_term_per_doc, dtype=np.float64)
        docs = float(getattr(assay, "n_docs", len(cell_idx)))
        feature_scale = np.log2(
            1.0 + (docs / (np.asarray(assay.n_docs_per_term, dtype=np.float64) + 1.0))
        )
    elif (
        counts_t is not None
        and cell_idx is not None
        and feat_idx is not None
        and assay.normMethod is norm_clr
    ):
        adapter = "clr"
    elif (
        counts_t is not None
        and cell_idx is not None
        and feat_idx is not None
        and assay.normMethod is norm_dummy
    ):
        adapter = "dummy"
    elif (
        counts_t is not None
        and cell_idx is not None
        and feat_idx is not None
        and lib_size_feature_stream_eligible(
            assay, renormalize_subset=renormalize_subset
        )
        and not use_fast
    ):
        adapter = "lib_size"
        scalar = assay.cells.fetch_all(assay.name + "_nCounts")[cell_idx]
        size_factor = float(assay.sf) if assay.sf is not None else 1.0
        scalar_values = np.asarray(scalar, dtype=np.float32)
        scalar_values[scalar_values == 0] = 1

    if adapter is not None:
        if counts_t is None or cell_idx is None or feat_idx is None:
            raise ValueError(
                f"Assay {assay.name!r} requires sharded countsT for marker search"
            )
        from ...storage.budget import resolve_budget
        from ...storage.feature_stream import (
            map_feature_read_groups,
            selected_feature_values,
        )

        stats_matrix = np.zeros((len(feat_idx), n_groups, 8), dtype=np.float64)
        n_feats = int(counts_t.shape[0])
        dest_of = np.full(n_feats, -1, dtype=np.int64)
        dest_of[feat_idx] = np.arange(len(feat_idx), dtype=np.int64)
        resources = getattr(assay, "resources", None) or resolve_budget(
            workers=nthreads
        )
        threads = min(
            max(1, int(resources.workers)),
            max(1, int(numba.config.NUMBA_NUM_THREADS)),
        )
        group_counts32 = np.asarray(group_counts, dtype=np.float32)
        previous_threads = numba.get_num_threads()
        set_num_threads(threads)
        logger.debug(
            f"Marker search bounded groups: features={len(feat_idx)} "
            f"groups={n_groups} adapter={adapter} workers={resources.workers} "
            f"numbaThreads={threads} memoryBytes={resources.memoryBytes}"
        )

        def process_group(group: Any) -> None:
            local_dest = dest_of[group.featStart : group.featEnd]
            selected = local_dest >= 0
            if not np.any(selected):
                return None
            if adapter == "rna_lib_size_unsigned":
                assert scalar_values is not None
                _marker_stats_gene_major(
                    group.values,
                    scalar_values,
                    np.float32(size_factor),
                    log_transform,
                    int_indices,
                    group_counts32,
                    np.float32(n_total),
                    local_dest,
                    stats_matrix,
                )
                return None
            raw = selected_feature_values(group.values, selected)
            destinations = local_dest[selected].astype(np.int64, copy=False)
            if adapter == "tfidf":
                assert cell_scale is not None
                assert feature_scale is not None
                values = _tfidf_feature_values(
                    raw, cell_scale, feature_scale[destinations]
                )
            elif adapter == "clr":
                values = _clr_feature_values(raw)
            elif adapter == "dummy":
                values = np.asarray(raw.T)
            else:
                assert scalar_values is not None
                values = _lib_size_feature_values(
                    raw, scalar_values, size_factor, log_transform
                )
            stats_matrix[destinations] = _batch_stats(
                values,
                int_indices,
                group_counts,
                n_total,
                feature_labels=destinations,
            )
            return None

        consume_metrics: dict[str, object] = {}
        scratch = int(stats_matrix.nbytes) + gene_major_rank_scratch_bytes(
            n_cells=n_total,
            n_groups=n_groups,
            nthreads=threads,
        )
        extra_itemsize = (
            0
            if adapter == "rna_lib_size_unsigned"
            else int(np.dtype(np.float64).itemsize)
        )
        try:
            for _ in map_feature_read_groups(
                counts_t,
                process_group,
                cell_idx=cell_idx,
                feat_idx=feat_idx,
                resources=resources,
                progress="Finding markers",
                io=getattr(assay, "storageIo", None),
                metrics=consume_metrics,
                scratchBytes=scratch,
                extraItemsize=extra_itemsize,
                orderedCompute=False,
            ):
                pass
        finally:
            set_num_threads(previous_threads)
        if adapter == "rna_lib_size_unsigned":
            z_values = np.asarray(stats_matrix[:, :, 6], dtype=np.float64)
            stats_matrix[:, :, 6] = 2.0 * ndtr(-np.abs(z_values))
    else:
        batch_stats = []
        iterator = iter(
            assay.iter_normed_feature_wise(
                cell_key=cell_key,
                feat_key=feat_key,
                batch_size=None,
                msg="Finding markers",
                **norm_params,
            )
        )
        while True:
            try:
                mat = next(iterator)
            except StopIteration:
                break
            if isinstance(mat, pd.DataFrame):
                values = mat.to_numpy()
                feature_labels = np.asarray(mat.columns)
            else:
                feature_major, feature_labels = mat
                values = np.asarray(feature_major).T
            stats = _batch_stats(
                values,
                int_indices,
                group_counts,
                n_total,
                feature_labels=np.asarray(feature_labels),
            )
            batch_stats.append(stats)
        stats_matrix = np.vstack(batch_stats)
    feat_index = assay.feats.active_index(feat_key)
    pval_col = "p_value"
    for n, i in enumerate(group_set):
        kernel = pd.DataFrame(
            stats_matrix[:, n, :],
            columns=list(_KERNEL_STAT_COLUMNS),
            index=feat_index,
        )
        adjusted = _bh_adjusted_pvalues(
            kernel[pval_col].to_numpy(dtype=np.float64, copy=False)
        )
        df = kernel.copy()
        df["p_value_adjusted"] = adjusted
        df = df.loc[:, list(MARKER_STAT_COLUMNS)]
        cols_to_round = [
            col for col in df.columns if col not in {pval_col, "p_value_adjusted"}
        ]
        df.loc[:, cols_to_round] = df.loc[:, cols_to_round].round(5)
        df["feature_index"] = df.index
        results[i] = sort_marker_results(df)[out_cols]
    return results


@restore_numba_threads
def find_markers_by_regression(
    assay: Assay,
    cell_key: str,
    feat_key: str,
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
    regressor = np.asarray(regressor, dtype=np.float64)
    if regressor.ndim != 1:
        raise ValueError("regressor must be one-dimensional")
    if not np.isfinite(regressor).all():
        raise ValueError("regressor must contain only finite values")
    if regressor.size < 2 or np.unique(regressor).size < 2:
        raise ValueError("regressor must contain at least two distinct values")
    if min_cells < 1:
        raise ValueError("min_cells must be at least 1")

    nthreads = getattr(assay, "nthreads", 1)
    set_num_threads(min(max(1, int(nthreads)), numba.config.NUMBA_NUM_THREADS))
    x_centered = regressor - regressor.mean()
    ssxm = float(np.dot(x_centered, x_centered) / regressor.shape[0])

    labels: list[Any] = []
    r_parts: list[np.ndarray] = []
    p_parts: list[np.ndarray] = []
    status_parts: list[np.ndarray] = []
    for feature_batch in assay.iter_normed_feature_wise(
        cell_key=cell_key,
        feat_key=feat_key,
        batch_size=batch_size,
        msg="Finding correlated features",
        **norm_params,
    ):
        if not isinstance(feature_batch, pd.DataFrame):
            raise TypeError("Expected normalized feature batches as DataFrames.")
        if feature_batch.shape[0] != regressor.shape[0]:
            raise ValueError(
                "Regressor length does not match the number of selected cells"
            )
        data = np.ascontiguousarray(feature_batch.to_numpy(dtype=np.float64))
        feat_labels = np.asarray(feature_batch.columns)
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
    if np.any(status == _REG_NONFINITE):
        raise ValueError("Regression results contain non-finite feature status")
    adjusted = np.full(p_values.shape, np.nan, dtype=np.float64)
    tested = status == _REG_OK
    if np.any(tested):
        adjusted[tested] = _bh_adjusted_pvalues(p_values[tested])
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
