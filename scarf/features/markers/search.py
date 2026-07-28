from typing import Any

import numba
import numpy as np
import pandas as pd
from numba import set_num_threads

from ...assay import Assay, RNAassay, lib_size_feature_stream_eligible
from ...utils.logging import logger
from ...utils.numba import restore_numba_threads
from .rank import _batch_stats, sort_marker_results
from .regression import _regression_batch_results

__all__ = ["find_markers_by_rank", "find_markers_by_regression"]


@restore_numba_threads
def find_markers_by_rank(
    assay: Assay,
    group_key: str,
    cell_key: str,
    feat_key: str,
    batch_size: int,
    n_threads: int,
    **norm_params: Any,
) -> dict[Any, pd.DataFrame]:
    """Identify marker features for groups with rank-based statistics."""
    groups = assay.cells.fetch(group_key, cell_key)
    group_set = np.array(sorted(set(groups)))
    n_groups = len(group_set)
    idx_map = dict(zip(group_set, range(n_groups)))
    int_indices = np.array([idx_map[x] for x in groups])
    out_cols = [
        "feature_index",
        "score",
        "mean",
        "mean_rest",
        "frac_exp",
        "frac_exp_rest",
        "fold_change",
        "p_value",
    ]
    results: dict[Any, pd.DataFrame] = {}
    set_num_threads(min(max(1, n_threads), numba.config.NUMBA_NUM_THREADS))
    group_counts = pd.Series(groups).value_counts().reindex(group_set).values
    n_total = len(groups)

    renormalize_subset = bool(norm_params.get("renormalize_subset", False))
    log_transform = bool(norm_params.get("log_transform", False))
    use_fast = lib_size_feature_stream_eligible(
        assay,
        renormalize_subset=renormalize_subset,
    )

    if use_fast:
        if not isinstance(assay, RNAassay):
            raise TypeError(
                "Fast raw-count marker search requires an RNAassay instance"
            )
        cell_idx = assay.cells.active_index(cell_key)
        feat_idx = assay.feats.active_index(feat_key)
        n_batches = max(1, (len(feat_idx) + batch_size - 1) // batch_size)
        logger.debug(
            f"Marker search (fast): {len(feat_idx)} features, {n_groups} groups, "
            f"{n_batches} batches of {batch_size}"
        )
        scalar = assay.cells.fetch_all(assay.name + "_nCounts")[cell_idx]
        sf = assay.sf
        if sf is None:
            raise ValueError("RNA library-size normalization requires a size factor")
        scalar_col = np.asarray(scalar, dtype=np.float32).reshape(-1, 1)
        scalar_col[scalar_col == 0] = 1
        batch_stats: list[np.ndarray] = []
        for (
            _block_idx,
            raw,
            _cols,
            _read_sec,
            _source,
        ) in assay.iter_raw_column_blocks(
            cell_idx=cell_idx,
            feat_idx=feat_idx,
            batch_size=batch_size,
            msg="Finding markers",
        ):
            normed = (float(sf) * raw.astype(np.float32)) / scalar_col
            if log_transform:
                normed = np.log1p(normed)
            stats = _batch_stats(normed, int_indices, group_counts, n_total)
            batch_stats.append(stats)
        stats_matrix = np.vstack(batch_stats)
    else:
        batch_stats = []
        iterator = iter(
            assay.iter_normed_feature_wise(
                cell_key=cell_key,
                feat_key=feat_key,
                batch_size=batch_size,
                msg="Finding markers",
                **norm_params,
            )
        )
        while True:
            try:
                mat = next(iterator)
            except StopIteration:
                break
            values = mat.to_numpy() if isinstance(mat, pd.DataFrame) else mat[0]
            stats = _batch_stats(
                values,
                int_indices,
                group_counts,
                n_total,
            )
            batch_stats.append(stats)
        stats_matrix = np.vstack(batch_stats)
    feat_index = assay.feats.active_index(feat_key)
    pval_col = "p_value"
    for n, i in enumerate(group_set):
        df = pd.DataFrame(
            stats_matrix[:, n, :],
            columns=out_cols[1:],
            index=feat_index,
        )
        cols_to_round = [col for col in df.columns if col != pval_col]
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
    regressor = np.asarray(regressor, dtype=np.float64)
    if regressor.ndim != 1:
        raise ValueError("regressor must be one-dimensional")
    if not np.isfinite(regressor).all():
        raise ValueError("regressor must contain only finite values")
    if regressor.size < 2 or np.unique(regressor).size < 2:
        raise ValueError("regressor must contain at least two distinct values")
    if min_cells < 1:
        raise ValueError("min_cells must be at least 1")

    n_threads = getattr(assay, "nthreads", 1)
    set_num_threads(min(max(1, int(n_threads)), numba.config.NUMBA_NUM_THREADS))
    x_centered = regressor - regressor.mean()
    ssxm = float(np.dot(x_centered, x_centered) / regressor.shape[0])

    labels: list[Any] = []
    r_parts: list[np.ndarray] = []
    p_parts: list[np.ndarray] = []
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
        r_vals, p_vals = _regression_batch_results(
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

    if not labels:
        return pd.DataFrame(columns=["r_value", "p_value"])
    return pd.DataFrame(
        {
            "r_value": np.concatenate(r_parts),
            "p_value": np.concatenate(p_parts),
        },
        index=labels,
    )
