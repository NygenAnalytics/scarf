from typing import TYPE_CHECKING, cast

import numpy as np

from ..graph.encoded_paths import make_normalized_leaf_name
from ..matrix import ChunkedArray
from ..utils.compute import show_dask_progress
from ..utils.logging import logger

if TYPE_CHECKING:
    from ..assay import Assay


def _streaming_covariance(
    data: ChunkedArray,
    nthreads: int,
    msg: str,
) -> np.ndarray:
    """Computes covariance while streaming row blocks."""
    n_cols = data.shape[1]
    xtx = np.zeros((n_cols, n_cols), dtype=np.float64)
    col_sum = np.zeros(n_cols, dtype=np.float64)
    n_rows = 0
    for block in data.stream_blocks(nthreads=nthreads, msg=msg):
        a = block.astype(np.float64, copy=False)
        xtx += a.T @ a
        col_sum += a.sum(axis=0)
        n_rows += a.shape[0]
    if n_rows < 2:
        raise ValueError("CORAL requires at least two cells in each dataset")
    mean = col_sum / n_rows
    cov = (xtx - n_rows * np.outer(mean, mean)) / (n_rows - 1)
    if not np.all(np.isfinite(cov)):
        raise ValueError("CORAL covariance contains non-finite values")
    return cov


def _cov_diaged(data: ChunkedArray, nthreads: int, msg: str) -> np.ndarray:
    a = _streaming_covariance(data, nthreads, msg)
    a = (a + a.T) / 2
    if not np.all(np.isfinite(a)):
        raise ValueError("CORAL covariance contains non-finite values")
    return cast(np.ndarray, a + np.eye(a.shape[0], dtype=a.dtype))


def _symmetric_matrix_power(matrix: np.ndarray, power: float) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError("CORAL covariance decomposition failed")
    if np.any(eigenvalues <= 0):
        raise ValueError("CORAL covariance must be positive definite")
    result = (eigenvectors * np.power(eigenvalues, power)) @ eigenvectors.T
    if not np.all(np.isfinite(result)):
        raise ValueError("CORAL covariance transform contains non-finite values")
    return cast(np.ndarray, result)


def _correlation_alignment(
    s: ChunkedArray,
    t: ChunkedArray,
    nthreads: int,
) -> ChunkedArray:
    from threadpoolctl import threadpool_limits

    if s.shape[1] != t.shape[1]:
        raise ValueError("CORAL source and target must have the same feature count")
    s_cov = _cov_diaged(s, nthreads, "CORAL: Computing source covariance")
    t_cov = _cov_diaged(t, nthreads, "CORAL: Computing target covariance")
    logger.info(
        "Calculating fractional power of covariance matrices. This might take a while... "
    )
    with threadpool_limits(limits=nthreads):
        a_coral = _symmetric_matrix_power(s_cov, -0.5) @ _symmetric_matrix_power(
            t_cov, 0.5
        )
    if not np.all(np.isfinite(a_coral)):
        raise ValueError("CORAL transform contains non-finite values")
    logger.info("Fractional power calculation complete")
    return s.dot(a_coral)


def coral(
    source_data: ChunkedArray,
    target_data: ChunkedArray,
    assay: "Assay",
    feat_key: str,
    cell_key: str,
    nthreads: int,
) -> None:
    """Apply CORAL batch correction and write corrected data to Zarr."""
    from ..storage.materialize import dask_to_zarr
    from ..utils.arrays import clean_array

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
    standardized = _correlation_alignment(
        (source_data - sm) / sd,
        (target_data - tm) / td,
        nthreads,
    )
    data = standardized * td + tm
    normed_loc = make_normalized_leaf_name(cell_key, feat_key)
    dask_to_zarr(
        data,
        assay.z,
        f"{normed_loc}/data_coral",
        nthreads,
        msg="Writing out coral corrected data",
        resources=assay.resources,
    )
