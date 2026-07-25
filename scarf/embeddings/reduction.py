from collections.abc import Callable
from typing import Any

import numpy as np

from ..matrix import ChunkedArray
from ..utils.logging import logger


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
    from numpy.linalg import LinAlgError
    from sklearn.decomposition import IncrementalPCA

    model = IncrementalPCA(
        n_components=dims + 1,
        batch_size=batch_size,
    )
    subset_samples = use_for_pca.sum() != data.shape[0]
    start = 0
    end_reservoir: np.ndarray | None = None
    carry_over: np.ndarray | None = None
    for block in data.stream_blocks(nthreads=nthreads, msg="Fitting PCA"):
        if subset_samples:
            end = start + block.shape[0]
            block = block[use_for_pca[start:end]]
            start = end
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
            model.partial_fit(np.array(block, copy=True), check_input=False)
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
    try:
        model.partial_fit(np.array(fit_batch, copy=True), check_input=False)
    except LinAlgError:
        logger.warning(
            f"{fit_batch.shape[0]} samples were not used in PCA fitting "
            "due to LinAlgError",
            flush=True,
        )
    return model.components_[:-1, :].T, model


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
