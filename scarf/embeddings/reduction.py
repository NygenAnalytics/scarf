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
        logger.debug(
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

    logger.debug(
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
    """Fit uncentered LSI loadings with a streamed or materialized solver."""
    reserved = {"n_components", "random_state"}
    for key in list(params):
        if key in reserved:
            del params[key]
            logger.warning(f"Provided parameter, {key}, for LSI model will not be used")

    n_components = dims + int(skip_first)
    if n_components > min(data.shape):
        raise ValueError("LSI components cannot exceed the input matrix rank")
    solver_params = dict(params)
    solver = solver_params.pop("solver", "streaming")
    if solver == "streaming":
        allowed = {"n_iter", "n_oversamples"}
        unsupported = sorted(set(solver_params) - allowed)
        if unsupported:
            joined = ", ".join(unsupported)
            raise ValueError(f"Streaming LSI does not support parameters: {joined}")
        return _fit_streaming_lsi(
            data,
            n_components=n_components,
            skip_first=skip_first,
            n_iter=solver_params.get("n_iter", 5),
            n_oversamples=solver_params.get("n_oversamples", 10),
            random_state=random_state,
            nthreads=nthreads,
        )
    if solver != "materialized":
        raise ValueError("LSI solver must be 'streaming' or 'materialized'")
    return _fit_materialized_lsi(
        data,
        n_components=n_components,
        skip_first=skip_first,
        params=solver_params,
        random_state=random_state,
        nthreads=nthreads,
    )


def _fit_materialized_lsi(
    data: ChunkedArray,
    *,
    n_components: int,
    skip_first: bool,
    params: dict[str, Any],
    random_state: int,
    nthreads: int,
) -> np.ndarray:
    from sklearn.decomposition import TruncatedSVD

    matrix = np.vstack(
        list(
            data.stream_blocks(nthreads=nthreads, msg="Fitting materialized LSI model")
        )
    )
    model = TruncatedSVD(
        n_components=n_components,
        random_state=random_state,
        **params,
    )
    model.fit(matrix)
    components = model.components_.T
    if skip_first:
        return np.asarray(components[:, 1:])
    return np.asarray(components)


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    resolved = int(value)
    if resolved < 0:
        raise ValueError(f"{name} must be nonnegative")
    return resolved


def _streaming_lsi_accumulator_bytes(n_features: int, width: int) -> int:
    itemsize = np.dtype(np.float64).itemsize
    return 3 * n_features * width * itemsize + 2 * width * width * itemsize


def _streaming_lsi_resident_bytes(data: ChunkedArray, width: int) -> int:
    block_rows = min(int(data.chunksize[0]), int(data.shape[0]))
    block_temporaries = (
        block_rows * (int(data.shape[1]) + width) * np.dtype(np.float64).itemsize
    )
    return (
        _streaming_lsi_accumulator_bytes(int(data.shape[1]), width) + block_temporaries
    )


def _stream_lsi_gram_action(
    data: ChunkedArray,
    basis: np.ndarray,
    *,
    nthreads: int,
    message: str,
) -> np.ndarray:
    result = np.zeros_like(basis, dtype=np.float64)
    rows_seen = 0
    resident_bytes = _streaming_lsi_resident_bytes(data, basis.shape[1])
    for block in data._stream_blocks(
        nthreads=nthreads,
        msg=message,
        prefetch=1,
        row_mask=None,
        resident_bytes=resident_bytes,
    ):
        values = np.asarray(block)
        projected = values @ basis
        result += values.T @ projected
        rows_seen += len(values)
    if rows_seen != data.shape[0]:
        raise RuntimeError(f"LSI streamed {rows_seen} rows, expected {data.shape[0]}")
    if not np.isfinite(result).all():
        raise ValueError("LSI input must contain only finite values")
    return result


def _fit_streaming_lsi(
    data: ChunkedArray,
    *,
    n_components: int,
    skip_first: bool,
    n_iter: Any,
    n_oversamples: Any,
    random_state: int,
    nthreads: int,
) -> np.ndarray:
    iterations = _nonnegative_integer(n_iter, "n_iter")
    oversamples = _nonnegative_integer(n_oversamples, "n_oversamples")
    width = min(min(data.shape), n_components + oversamples)
    rng = np.random.default_rng(random_state)
    basis = rng.standard_normal((data.shape[1], width), dtype=np.float64)
    basis, _ = np.linalg.qr(basis, mode="reduced")

    for iteration in range(iterations + 1):
        basis = _stream_lsi_gram_action(
            data,
            basis,
            nthreads=nthreads,
            message=f"Fitting streaming LSI model ({iteration + 1}/{iterations + 2})",
        )
        basis, _ = np.linalg.qr(basis, mode="reduced")

    projected_gram = np.zeros((width, width), dtype=np.float64)
    rows_seen = 0
    resident_bytes = _streaming_lsi_resident_bytes(data, width)
    for block in data._stream_blocks(
        nthreads=nthreads,
        msg=f"Fitting streaming LSI model ({iterations + 2}/{iterations + 2})",
        prefetch=1,
        row_mask=None,
        resident_bytes=resident_bytes,
    ):
        projected = np.asarray(block) @ basis
        projected_gram += projected.T @ projected
        rows_seen += len(projected)
    if rows_seen != data.shape[0]:
        raise RuntimeError(f"LSI streamed {rows_seen} rows, expected {data.shape[0]}")
    if not np.isfinite(projected_gram).all():
        raise ValueError("LSI input must contain only finite values")

    eigenvalues, rotations = np.linalg.eigh(projected_gram)
    order = np.argsort(eigenvalues)[::-1][:n_components]
    loadings = np.asarray(basis @ rotations[:, order], dtype=np.float64)
    largest = np.argmax(np.abs(loadings), axis=0)
    signs = np.sign(loadings[largest, np.arange(loadings.shape[1])])
    signs[signs == 0] = 1
    loadings *= signs
    if skip_first:
        return loadings[:, 1:]
    return loadings
