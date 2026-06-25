"""Utility functions for the mapping."""

from typing import Any, cast

import numpy as np
import pandas as pd

from .assay import Assay
from .chunked import ChunkedArray
from .utils import controlled_compute, show_dask_progress, logger, tqdmbar

__all__ = ["align_features", "coral"]


def _streaming_covariance(data: ChunkedArray, nthreads: int, msg: str) -> np.ndarray:
    """Computes the (features x features) covariance by streaming row-blocks.

    Uses the identity cov = (XtX - n * mean (x) mean) / (n - 1), accumulating
    the cross-product XtX and the column sums over row-blocks so peak memory
    stays bounded by a single block plus the small (features x features) matrix.
    """
    n_cols = data.shape[1]
    xtx = np.zeros((n_cols, n_cols), dtype=np.float64)
    col_sum = np.zeros(n_cols, dtype=np.float64)
    n_rows = 0
    for block in tqdmbar(data.blocks, total=data.numblocks[0], desc=msg):
        a = controlled_compute(block, nthreads).astype(np.float64, copy=False)
        xtx += a.T @ a
        col_sum += a.sum(axis=0)
        n_rows += a.shape[0]
    mean = col_sum / n_rows
    cov = (xtx - n_rows * np.outer(mean, mean)) / (n_rows - 1)
    return cov


def _cov_diaged(data: ChunkedArray, nthreads: int, msg: str) -> np.ndarray:
    a = _streaming_covariance(data, nthreads, msg)
    a[a == np.inf] = 0
    a[a == np.nan] = 0
    return a + np.eye(a.shape[0])


def _correlation_alignment(
    s: ChunkedArray, t: ChunkedArray, nthreads: int
) -> ChunkedArray:
    from scipy.linalg import fractional_matrix_power as fmp
    from threadpoolctl import threadpool_limits

    s_cov = _cov_diaged(s, nthreads, "CORAL: Computing source covariance")
    t_cov = _cov_diaged(t, nthreads, "CORAL: Computing target covariance")
    logger.info(
        "Calculating fractional power of covariance matrices. This might take a while... "
    )
    with threadpool_limits(limits=nthreads):
        a_coral = np.dot(fmp(s_cov, -0.5), fmp(t_cov, 0.5))
    logger.info("Fractional power calculation complete")
    return s.dot(a_coral)


def coral(
    source_data: ChunkedArray,
    target_data: ChunkedArray,
    assay: Assay,
    feat_key: str,
    cell_key: str,
    nthreads: int,
) -> None:
    """Apply CORAL batch correction and write corrected data to Zarr.

    Args:
        source_data: Source ChunkedArray (reference modality).
        target_data: Target ChunkedArray to align.
        assay: Target Assay whose Zarr group receives corrected data.
        feat_key: Feature selection key used in output path.
        cell_key: Cell selection key used in output path.
        nthreads: Threads for streaming statistics and writes.
    """
    from .writers import dask_to_zarr
    from .utils import clean_array

    sm = clean_array(
        show_dask_progress(
            source_data.mean(axis=0),
            "CORAL: Computing source feature means",
            nthreads,
        )
    )
    sd = clean_array(
        show_dask_progress(
            source_data.std(axis=0),
            "CORAL: Computing source feature stdev",
            nthreads,
        ),
        1,
    )
    tm = clean_array(
        show_dask_progress(
            target_data.mean(axis=0),
            "CORAL: Computing target feature means",
            nthreads,
        )
    )
    td = clean_array(
        show_dask_progress(
            target_data.std(axis=0),
            "CORAL: Computing target feature stdev",
            nthreads,
        ),
        1,
    )
    data = _correlation_alignment(
        (source_data - sm) / sd, (target_data - tm) / td, nthreads
    )
    dask_to_zarr(
        data,
        assay.z,
        f"normed__{cell_key}__{feat_key}/data_coral",
        data.chunksize,
        nthreads,
        msg="Writing out coral corrected data",
    )


def _order_features(
    s_assay: Assay,
    t_assay: Assay,
    s_feat_ids: np.ndarray,
    filter_null: bool,
    exclude_missing: bool,
    nthreads: int,
    target_cell_key: str = "I",
) -> tuple[np.ndarray, np.ndarray]:
    s_ids = pd.Series(s_assay.feats.fetch_all("ids"))
    t_ids = pd.Series(t_assay.feats.fetch_all("ids"))
    t_idx = t_ids.isin(s_feat_ids)
    if t_idx.sum() == 0:
        raise ValueError(
            "ERROR: None of the features from reference were found in the target data"
        )
    if filter_null:
        if exclude_missing is False:
            logger.warning(
                "`filter_null` has not effect because `exclude_missing` is False"
            )
        else:
            t_idx[t_idx] = (
                controlled_compute(
                    t_assay.rawData[:, list(t_idx[t_idx].index)][
                        t_assay.cells.active_index(target_cell_key), :
                    ].sum(axis=0),
                    nthreads,
                )
                != 0
            )
    t_idx = t_idx[t_idx].index
    if exclude_missing:
        s_idx = s_ids.isin(t_ids.values[t_idx])
    else:
        s_idx = s_ids.isin(s_feat_ids)
    s_idx = s_idx[s_idx].index
    t_idx_map = {v: k for k, v in t_ids.to_dict().items()}
    t_re_idx = np.array(
        [t_idx_map[x] if x in t_idx_map else -1 for x in s_ids.values[s_idx]]
    )
    if len(s_idx) != len(t_re_idx):
        raise AssertionError(
            "ERROR: Feature ordering failed. Please report this issue. "
            f"This is an unexpected scenario. Source has {len(s_idx)} features while target has "
            f"{len(t_re_idx)} features"
        )
    return s_idx.values, t_re_idx


def align_features(
    source_assay: Assay,
    target_assay: Assay,
    source_cell_key: str,
    source_feat_key: str,
    target_feat_key: str,
    target_cell_key: str,
    filter_null: bool,
    exclude_missing: bool,
    nthreads: int,
) -> np.ndarray:
    """Aligns target features to source features.

    Args:
        source_assay: Reference assay with features to align to.
        target_assay: Target assay whose features are reordered and saved.
        source_cell_key: Cell key for source normalization params.
        source_feat_key: Feature key on the source assay.
        target_feat_key: Feature key label for saved target data.
        target_cell_key: Cell key on the target assay.
        filter_null: Drop target features with zero counts in selected cells.
        exclude_missing: Exclude source features absent from target.
        nthreads: Threads for streaming alignment.

    Returns:
        Target feature index array aligned to source order.
    """
    from .writers import create_zarr_dataset

    source_feat_ids = source_assay.feats.fetch(
        "ids", key=source_cell_key + "__" + source_feat_key
    )
    s_idx, t_idx = _order_features(
        source_assay,
        target_assay,
        source_feat_ids,
        filter_null,
        exclude_missing,
        nthreads,
        target_cell_key,
    )
    logger.info(f"{(t_idx == -1).sum()} features missing in target data")
    normed_loc = f"normed__{source_cell_key}__{source_feat_key}"
    norm_params = cast(
        dict[str, Any], source_assay.z[normed_loc].attrs["subset_params"]
    )
    sorted_t_idx = np.array(sorted(t_idx[t_idx != -1]))

    normed_data = target_assay.normed(
        target_assay.cells.active_index(target_cell_key), sorted_t_idx, **norm_params
    )
    normed_loc = f"normed__{target_cell_key}__{target_feat_key}"
    og = create_zarr_dataset(
        target_assay.z,
        f"{normed_loc}/data",
        (1000, len(t_idx)),
        "float64",
        (normed_data.shape[0], len(t_idx)),
    )
    pos_start, pos_end = 0, 0
    unsorter_idx = np.argsort(np.argsort(t_idx[t_idx != -1]))
    for i in tqdmbar(
        normed_data.blocks,
        total=normed_data.numblocks[0],
        desc=f"({target_assay.name}) Writing aligned data to {normed_loc}",
    ):
        pos_end += i.shape[0]
        a = np.ones((i.shape[0], len(t_idx)))
        a[:, np.where(t_idx != -1)[0]] = controlled_compute(i, nthreads)[
            :, unsorter_idx
        ]
        og[pos_start:pos_end, :] = a
        pos_start = pos_end
    return s_idx
