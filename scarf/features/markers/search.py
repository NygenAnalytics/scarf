import time
from typing import Any

import numba
import numpy as np
import pandas as pd
from numba import set_num_threads
from scipy.special import ndtr

from ...assay import Assay, RNAassay, lib_size_feature_stream_eligible
from ...storage.feature_stream import plan_feature_stream
from ...utils.logging import logger
from ...utils.numba import restore_numba_threads
from ...utils.process import process_rss_mb
from .rank import (
    _batch_stats,
    _marker_stats_gene_major,
    gene_major_rank_scratch_bytes,
    sort_marker_results,
)
from .regression import _regression_batch_results

__all__ = ["find_markers_by_rank", "find_markers_by_regression"]


@restore_numba_threads
def find_markers_by_rank(
    assay: Assay,
    group_key: str,
    cell_key: str,
    feat_key: str,
    batch_size: int | None = None,
    n_threads: int = 1,
    **norm_params: Any,
) -> dict[Any, pd.DataFrame]:
    """Identify marker features for groups with rank-based statistics."""
    groups = assay.cells.fetch(group_key, cell_key)
    group_set = np.array(sorted(set(groups)))
    n_groups = len(group_set)
    idx_map = dict(zip(group_set, range(n_groups)))
    int_indices = np.asarray([idx_map[x] for x in groups], dtype=np.int64)
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
    worker_limit = getattr(
        getattr(assay, "resources", None),
        "workers",
        n_threads,
    )
    set_num_threads(
        min(
            max(1, n_threads),
            max(1, int(worker_limit)),
            numba.config.NUMBA_NUM_THREADS,
        )
    )
    group_counts = pd.Series(groups).value_counts().reindex(group_set).values
    n_total = len(groups)

    renormalize_subset = bool(norm_params.get("renormalize_subset", False))
    log_transform = bool(norm_params.get("log_transform", False))
    use_fast = lib_size_feature_stream_eligible(
        assay,
        renormalize_subset=renormalize_subset,
    )
    if use_fast and isinstance(assay, RNAassay):
        raw_source, _, _ = assay._raw_feature_stream_source()
        use_fast = np.issubdtype(raw_source.dtype, np.unsignedinteger)

    if use_fast:
        if not isinstance(assay, RNAassay):
            raise TypeError(
                "Fast raw-count marker search requires an RNAassay instance"
            )
        cell_idx = assay.cells.active_index(cell_key)
        feat_idx = assay.feats.active_index(feat_key)
        scalar = assay.cells.fetch_all(assay.name + "_nCounts")[cell_idx]
        sf = assay.sf
        if sf is None:
            raise ValueError("RNA library-size normalization requires a size factor")
        scalar_values = np.asarray(scalar, dtype=np.float32)
        scalar_values[scalar_values == 0] = 1
        group_counts32 = np.asarray(group_counts, dtype=np.float32)
        stats_matrix = np.zeros(
            (len(feat_idx), n_groups, 7),
            dtype=np.float64,
        )
        raw_source, feature_axis, cell_axis = assay._raw_feature_stream_source()
        active_threads = numba.get_num_threads()
        resident_bytes = (
            scalar_values.nbytes
            + int_indices.nbytes
            + group_counts32.nbytes
            + stats_matrix.nbytes
            + gene_major_rank_scratch_bytes(
                n_cells=len(cell_idx),
                n_groups=n_groups,
                n_threads=active_threads,
            )
        )
        orientation_buffers = 1 if feature_axis == 0 else 2
        raw_itemsize = max(1, int(np.dtype(raw_source.dtype).itemsize))
        plan = plan_feature_stream(
            raw_source,
            featureAxis=feature_axis,
            cellAxis=cell_axis,
            featureIndices=feat_idx,
            cellIndices=cell_idx,
            resources=assay.resources,
            residentBytes=resident_bytes,
            blockBytes=lambda width: max(
                1,
                len(cell_idx) * width * raw_itemsize * orientation_buffers,
            ),
            requestedBatchSize=batch_size,
        )
        logger.debug(
            f"Marker search (fast): {len(feat_idx)} features, "
            f"{n_groups} groups, {len(plan.blocks)} blocks, "
            f"repeated chunk decodes={plan.repeatedDecodeCount}"
        )
        logger.info(
            f"Marker search plan: features={len(feat_idx)} groups={n_groups} "
            f"blocks={len(plan.blocks)} readWorkers={plan.readWorkers} "
            f"ioConcurrency={plan.ioConcurrency} numbaThreads={active_threads} "
            f"repeatedDecodes={plan.repeatedDecodeCount}"
        )
        block_idx = 0
        for block, raw, read_sec, source in assay.iter_raw_feature_major_blocks(
            cell_idx=cell_idx,
            plan=plan,
            msg="Finding markers",
        ):
            block_idx += 1
            compute_started = time.perf_counter()
            cpu_started = time.process_time()
            _marker_stats_gene_major(
                raw,
                scalar_values,
                np.float32(sf),
                log_transform,
                int_indices,
                group_counts32,
                np.float32(n_total),
                block.destinations,
                stats_matrix,
            )
            compute_seconds = time.perf_counter() - compute_started
            cpu_seconds = time.process_time() - cpu_started
            effective_cores = (
                cpu_seconds / compute_seconds if compute_seconds > 0 else 0.0
            )
            logger.info(
                f"Marker block {block_idx}/{len(plan.blocks)}: "
                f"width={block.indices.size} read={read_sec:.1f}s ({source}) "
                f"compute={compute_seconds:.1f}s cpu={cpu_seconds:.1f}s "
                f"effectiveCores={effective_cores:.2f} "
                f"rss={process_rss_mb():.0f} MiB"
            )
            del raw
        p_values = stats_matrix[:, :, 6]
        np.abs(p_values, out=p_values)
        np.negative(p_values, out=p_values)
        ndtr(p_values, out=p_values)
        p_values *= 2.0
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
