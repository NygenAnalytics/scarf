from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..matrix import ChunkedArray
from ..utils.logging import logger

_GRAM_PCA_MAX_FEATURES = 4096


@dataclass(frozen=True, slots=True)
class _GramPcaModel:
    components_: np.ndarray
    explained_variance_: np.ndarray
    explained_variance_ratio_: np.ndarray
    singular_values_: np.ndarray
    mean_: np.ndarray
    n_components_: int
    n_features_in_: int
    n_samples_seen_: int


def _mutable_fit_block(block: np.ndarray) -> np.ndarray:
    values = np.asarray(block)
    if values.flags.owndata and values.flags.writeable and values.flags.c_contiguous:
        return values
    return np.array(values, copy=True, order="C")


def _gram_pca_dispatch(
    n_features: int,
    block_rows: int,
    n_blocks: int,
) -> tuple[bool, str | None]:
    if n_blocks <= 1:
        return False, "the input has only one row block"
    if n_features > block_rows:
        return (
            False,
            f"{n_features} features exceed {block_rows} rows per block",
        )
    if n_features > _GRAM_PCA_MAX_FEATURES:
        return (
            False,
            f"{n_features} features exceed the {_GRAM_PCA_MAX_FEATURES}-feature limit",
        )
    return True, None


def _fit_sklearn_incremental_pca(
    data: ChunkedArray,
    *,
    dims: int,
    batch_size: int,
    row_mask: np.ndarray | None,
    scale: Callable[[np.ndarray], np.ndarray] | None,
    nthreads: int,
) -> tuple[np.ndarray, Any]:
    from numpy.linalg import LinAlgError
    from sklearn.decomposition import IncrementalPCA

    model = IncrementalPCA(
        n_components=dims + 1,
        batch_size=batch_size,
    )
    end_reservoir: np.ndarray | None = None
    carry_over: np.ndarray | None = None
    for block in data._stream_blocks(
        nthreads=nthreads,
        msg="Fitting PCA",
        prefetch=None,
        row_mask=row_mask,
    ):
        if scale is not None:
            block = scale(block)
        if carry_over is not None:
            block = np.vstack((carry_over, block))
            carry_over = None
        if len(block) < (dims + 1):
            carry_over = block
            continue
        if end_reservoir is None:
            end_reservoir = block
            continue
        try:
            model.partial_fit(_mutable_fit_block(block), check_input=False)
        except LinAlgError:
            carry_over = block

    if carry_over is not None:
        fit_batch = (
            np.vstack((end_reservoir, carry_over))
            if end_reservoir is not None
            else carry_over
        )
    else:
        assert end_reservoir is not None
        fit_batch = end_reservoir
    model.partial_fit(_mutable_fit_block(fit_batch), check_input=False)
    return model.components_[:-1, :].T, model


def _fit_gram_pca(
    data: ChunkedArray,
    *,
    dims: int,
    row_mask: np.ndarray | None,
    selected_samples: int,
    scale: Callable[[np.ndarray], np.ndarray] | None,
    nthreads: int,
) -> tuple[np.ndarray, _GramPcaModel]:
    from scipy.linalg import blas, eigh
    from sklearn.utils.extmath import svd_flip
    from threadpoolctl import threadpool_limits

    n_features = data.shape[1]
    n_components = dims + 1
    gram = np.zeros(
        (n_features, n_features),
        dtype=np.float64,
        order="F",
    )
    column_sum = np.zeros(n_features, dtype=np.float64)
    n_samples_seen = 0

    with threadpool_limits(limits=nthreads):
        for block in data._stream_blocks(
            nthreads=nthreads,
            msg="Fitting PCA",
            prefetch=None,
            row_mask=row_mask,
        ):
            if scale is not None:
                block = scale(block)
            values = np.asfortranarray(block, dtype=np.float64)
            gram = blas.dsyrk(
                1.0,
                values,
                beta=1.0,
                c=gram,
                trans=1,
                lower=0,
                overwrite_c=1,
            )
            column_sum += values.sum(axis=0, dtype=np.float64)
            n_samples_seen += len(values)

        if n_samples_seen != selected_samples:
            raise RuntimeError(
                f"PCA streamed {n_samples_seen} rows, expected {selected_samples}"
            )

        mean = column_sum / n_samples_seen
        gram = blas.dsyr(
            -float(n_samples_seen),
            mean,
            lower=0,
            a=gram,
            overwrite_a=1,
        )
        gram /= n_samples_seen - 1
        total_variance = float(np.trace(gram))
        if not np.isfinite(total_variance) or total_variance <= 0:
            raise ValueError("PCA input must have positive finite variance")

        eigenvalues, eigenvectors = eigh(
            gram,
            lower=False,
            subset_by_index=(
                n_features - n_components,
                n_features - 1,
            ),
            overwrite_a=True,
            check_finite=False,
            driver="evr",
        )

    eigenvalues = np.clip(eigenvalues[::-1], 0.0, None)
    components = np.array(
        eigenvectors[:, ::-1].T,
        dtype=np.float64,
        copy=True,
        order="C",
    )
    _, components = svd_flip(
        None,
        components,
        u_based_decision=False,
    )
    model = _GramPcaModel(
        components_=components,
        explained_variance_=eigenvalues,
        explained_variance_ratio_=eigenvalues / total_variance,
        singular_values_=np.sqrt(eigenvalues * (n_samples_seen - 1)),
        mean_=mean,
        n_components_=n_components,
        n_features_in_=n_features,
        n_samples_seen_=n_samples_seen,
    )
    return model.components_[:-1, :].T, model


def fit_incremental_pca(
    data: ChunkedArray,
    *,
    dims: int,
    batch_size: int,
    use_for_pca: np.ndarray,
    scale: Callable[[np.ndarray], np.ndarray] | None,
    nthreads: int,
) -> tuple[np.ndarray, Any]:
    """Fit streaming PCA and return loadings with the fitted model."""
    use_for_pca = np.asarray(use_for_pca)
    if use_for_pca.dtype != bool or use_for_pca.shape != (data.shape[0],):
        raise ValueError("use_for_pca must be a boolean vector matching data rows")

    n_components = dims + 1
    selected_samples = int(np.count_nonzero(use_for_pca))
    if selected_samples < n_components:
        raise ValueError(f"PCA requires at least {n_components} selected rows")
    if data.shape[1] < n_components:
        raise ValueError(f"PCA requires at least {n_components} features")

    subset_samples = selected_samples != data.shape[0]
    row_mask = use_for_pca if subset_samples else None
    n_features = data.shape[1]
    block_rows = data.chunksize[0]
    use_gram, fallback_reason = _gram_pca_dispatch(
        n_features,
        block_rows,
        data.numblocks[0],
    )
    if use_gram:
        accumulator_mib = n_features * n_features * 8 / (1024**2)
        logger.info(
            "Fitting PCA with the Gram covariance solver "
            f"({n_features} features, {block_rows} rows per block, "
            f"{accumulator_mib:.0f} MiB accumulator)"
        )
        return _fit_gram_pca(
            data,
            dims=dims,
            row_mask=row_mask,
            selected_samples=selected_samples,
            scale=scale,
            nthreads=nthreads,
        )

    logger.info(
        "Fitting PCA with the IncrementalPCA solver "
        f"({n_features} features, {block_rows} rows per block; "
        f"falling back because {fallback_reason})"
    )
    return _fit_sklearn_incremental_pca(
        data,
        dims=dims,
        batch_size=batch_size,
        row_mask=row_mask,
        scale=scale,
        nthreads=nthreads,
    )


def fit_lsi(
    data: ChunkedArray,
    *,
    dims: int,
    skip_first: bool,
    params: dict[str, Any],
    random_state: int,
    nthreads: int,
) -> np.ndarray:
    """Fit LSI loadings over materialized matrix blocks."""
    from sklearn.decomposition import TruncatedSVD

    reserved = {"n_components", "random_state"}
    for key in list(params):
        if key in reserved:
            del params[key]
            logger.warning(f"Provided parameter, {key}, for LSI model will not be used")

    matrix = np.vstack(
        list(data.stream_blocks(nthreads=nthreads, msg="Fitting LSI model"))
    )
    model = TruncatedSVD(
        n_components=dims + int(skip_first),
        random_state=random_state,
        **params,
    )
    model.fit(matrix)
    components = model.components_.T
    if skip_first:
        return np.asarray(components[:, 1:])
    return np.asarray(components)
