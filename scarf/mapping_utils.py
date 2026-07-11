"""Numerical and feature-alignment utilities for reference mapping."""

import hashlib
from typing import Any, cast

import numpy as np
import pandas as pd

from .assay import Assay
from .chunked import ChunkedArray
from .utils import controlled_compute, show_dask_progress, logger

__all__ = [
    "align_features",
    "array_hash",
    "array_store_hash",
    "conformal_prediction_sets",
    "coral",
    "distance_weights",
]


def array_hash(values: np.ndarray | list[Any]) -> str:
    """Return a stable content hash for numeric or identifier arrays."""
    arr = np.asarray(values)
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode())
    digest.update(arr.dtype.str.encode())
    if arr.dtype.kind in {"O", "S", "U"}:
        digest.update(
            "\x1f".join(str(value) for value in arr.reshape(-1)).encode("utf-8")
        )
    else:
        digest.update(np.ascontiguousarray(arr).tobytes())
    return digest.hexdigest()


def array_store_hash(values: Any) -> str:
    """Hash a row-addressable array without materializing it in memory."""
    shape = tuple(int(value) for value in values.shape)
    dtype = np.dtype(values.dtype)
    digest = hashlib.sha256()
    digest.update(str(shape).encode())
    digest.update(dtype.str.encode())
    if not shape:
        digest.update(np.asarray(values[...]).tobytes())
        return digest.hexdigest()
    chunks = getattr(values, "chunks", None)
    row_chunk = (
        int(chunks[0])
        if chunks is not None and len(chunks) > 0
        else min(max(shape[0], 1), 10_000)
    )
    for start in range(0, shape[0], row_chunk):
        stop = min(start + row_chunk, shape[0])
        block = np.asarray(values[start:stop])
        if dtype.kind in {"O", "S", "U"}:
            digest.update(
                "\x1f".join(str(value) for value in block.reshape(-1)).encode("utf-8")
            )
        else:
            digest.update(np.ascontiguousarray(block).tobytes())
    return digest.hexdigest()


def distance_weights(distances: np.ndarray) -> np.ndarray:
    """Convert HNSW squared-L2 distances into normalized inverse-L2 weights."""
    values = np.asarray(distances, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Expected a two-dimensional distance array")
    if not np.all(np.isfinite(values)):
        raise ValueError("Neighbor distances must be finite")
    if np.any(values < 0):
        raise ValueError("Neighbor distances must be non-negative")

    l2_distances = np.sqrt(values)
    weights = np.zeros_like(l2_distances)
    zero_mask = l2_distances == 0
    zero_count = zero_mask.sum(axis=1)
    rows_with_zero = zero_count > 0
    if rows_with_zero.any():
        weights[rows_with_zero] = (
            zero_mask[rows_with_zero] / zero_count[rows_with_zero, np.newaxis]
        )
    rows_without_zero = ~rows_with_zero
    if rows_without_zero.any():
        inverse = 1.0 / l2_distances[rows_without_zero]
        weights[rows_without_zero] = inverse / inverse.sum(axis=1, keepdims=True)
    return weights


def conformal_prediction_sets(
    label_scores: np.ndarray,
    calibration_nonconformity: np.ndarray,
    alpha: float = 0.1,
) -> np.ndarray:
    """Return class-membership masks from split-conformal p-values."""
    scores = np.asarray(label_scores, dtype=np.float64)
    calibration = np.asarray(calibration_nonconformity, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("label_scores must be a two-dimensional array")
    if calibration.ndim != 1 or calibration.size == 0:
        raise ValueError("calibration_nonconformity must be a non-empty vector")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(calibration)):
        raise ValueError("Conformal inputs must be finite")
    nonconformity = 1.0 - scores
    p_values = (
        (calibration[np.newaxis, np.newaxis, :] >= nonconformity[:, :, np.newaxis]).sum(
            axis=2
        )
        + 1
    ) / (len(calibration) + 1)
    return cast(np.ndarray, p_values > alpha)


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
    s: ChunkedArray, t: ChunkedArray, nthreads: int
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
    assay: Assay,
    feat_key: str,
    cell_key: str,
    nthreads: int,
) -> None:
    """Apply CORAL batch correction and write corrected data to Zarr.

    Args:
        source_data: Query ChunkedArray to align to the reference.
        target_data: Reference ChunkedArray defining the target distribution.
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
    standardized = _correlation_alignment(
        (source_data - sm) / sd, (target_data - tm) / td, nthreads
    )
    # The alignment is computed in standardized coordinates. Restore the
    # reference feature scale so the result can use the reference ANN
    # transform exactly as an uncorrected query would.
    data = standardized * td + tm
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
    missing_feature_policy: str,
    nthreads: int,
    target_cell_key: str = "I",
) -> tuple[np.ndarray, np.ndarray]:
    s_ids = pd.Series(s_assay.feats.fetch_all("ids"))
    t_ids = pd.Series(t_assay.feats.fetch_all("ids"))
    if s_ids.duplicated().any():
        duplicates = s_ids[s_ids.duplicated()].iloc[:5].tolist()
        raise ValueError(f"Reference feature identifiers must be unique: {duplicates}")
    if t_ids.duplicated().any():
        duplicates = t_ids[t_ids.duplicated()].iloc[:5].tolist()
        raise ValueError(f"Target feature identifiers must be unique: {duplicates}")
    selected_ids = pd.Series(s_feat_ids)
    if selected_ids.duplicated().any():
        duplicates = selected_ids[selected_ids.duplicated()].iloc[:5].tolist()
        raise ValueError(
            f"Selected reference feature identifiers must be unique: {duplicates}"
        )
    t_idx = t_ids.isin(s_feat_ids)
    if t_idx.sum() == 0:
        raise ValueError(
            "ERROR: None of the features from reference were found in the target data"
        )
    if filter_null:
        if missing_feature_policy != "intersection":
            logger.warning(
                "`filter_null` has no effect unless missing_feature_policy is 'intersection'"
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
    if len(t_idx) == 0:
        raise ValueError("No target features remain after applying the feature policy")
    if missing_feature_policy == "intersection":
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
    missing_feature_policy: str | None = None,
    missing_feature_values: np.ndarray | None = None,
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
        exclude_missing: Deprecated alias for ``missing_feature_policy='intersection'``.
        nthreads: Threads for streaming alignment.
        missing_feature_policy: One of ``'zero'``, ``'intersection'``, ``'error'``,
            or ``'reference_mean'``.
            ``'zero'`` fills absent target features with zero;
            ``'intersection'`` retains only shared features; ``'error'``
            rejects incomplete feature overlap; and ``'reference_mean'`` fills
            missing values from the stored reference mean.
        missing_feature_values: Per-reference-feature values used for absent
            target features. Required when ``missing_feature_policy`` is
            ``'reference_mean'`` and target features are absent.

    Returns:
        Target feature index array aligned to source order.
    """
    from .writers import create_zarr_dataset

    if missing_feature_policy is None:
        missing_feature_policy = "intersection" if exclude_missing else "zero"
    if missing_feature_policy not in {
        "zero",
        "intersection",
        "error",
        "reference_mean",
    }:
        raise ValueError(
            "missing_feature_policy must be one of 'zero', 'intersection', 'error', "
            "or 'reference_mean'"
        )
    if exclude_missing and missing_feature_policy != "intersection":
        raise ValueError(
            "exclude_missing=True is only compatible with missing_feature_policy='intersection'"
        )

    source_feature_key = (
        source_feat_key
        if source_feat_key == "I"
        else f"{source_cell_key}__{source_feat_key}"
    )
    source_feat_ids = source_assay.feats.fetch("ids", key=source_feature_key)
    s_idx, t_idx = _order_features(
        source_assay,
        target_assay,
        source_feat_ids,
        filter_null,
        missing_feature_policy,
        nthreads,
        target_cell_key,
    )
    n_missing = int((t_idx == -1).sum())
    if missing_feature_policy == "error" and n_missing:
        raise ValueError(
            f"Target data is missing {n_missing} required reference features"
        )
    logger.info(f"{n_missing} features missing in target data")
    if missing_feature_values is not None:
        missing_feature_values = np.asarray(missing_feature_values)
        if missing_feature_values.shape != (len(t_idx),):
            raise ValueError(
                "missing_feature_values must have one value per aligned reference feature"
            )
    if missing_feature_policy == "reference_mean" and n_missing:
        if missing_feature_values is None:
            raise ValueError(
                "reference_mean feature handling requires missing_feature_values"
            )
        if not np.all(np.isfinite(missing_feature_values)):
            raise ValueError("missing_feature_values must be finite")
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
    for i in normed_data.stream_blocks(
        nthreads=nthreads,
        msg=f"({target_assay.name}) Writing aligned data to {normed_loc}",
    ):
        pos_end += i.shape[0]
        if missing_feature_values is None:
            a = np.zeros((i.shape[0], len(t_idx)), dtype=i.dtype)
        else:
            a = np.broadcast_to(missing_feature_values, (i.shape[0], len(t_idx))).copy()
        a[:, np.where(t_idx != -1)[0]] = i[:, unsorter_idx]
        og[pos_start:pos_end, :] = a
        pos_start = pos_end
    return s_idx
