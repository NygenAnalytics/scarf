import inspect
import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from ...utils.arguments import integer_argument
from .models import ClusterFn, HarmonyResult

_DATA_ARGUMENTS = frozenset({"data_mat", "meta_data"})


def _require_integer(
    values: dict[str, Any],
    name: str,
    minimum: int,
    *,
    optional: bool = False,
) -> None:
    if name not in values or (optional and values[name] is None):
        return
    integer_argument(values[name], f"Harmony {name}", minimum=minimum)


def _require_real(
    values: dict[str, Any],
    name: str,
    *,
    positive: bool,
) -> None:
    if name not in values:
        return
    value = values[name]
    if isinstance(value, bool) or not isinstance(value, int | float | np.number):
        raise TypeError(f"Harmony {name} must be a real number")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0 or (positive and resolved == 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"Harmony {name} must be finite and {qualifier}")


def _require_real_values(
    values: dict[str, Any],
    name: str,
    *,
    positive: bool,
    optional: bool,
) -> None:
    if name not in values:
        return
    value = values[name]
    message = f"Harmony {name} must contain real numbers"
    if value is None:
        if optional:
            return
        raise TypeError(message)
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        raise TypeError(message) from None
    if (
        not np.all(np.isfinite(array))
        or np.any(array < 0)
        or (positive and np.any(array == 0))
    ):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"Harmony {name} values must be finite and {qualifier}")


def validate_harmony_parameters(
    parameters: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Check Harmony keyword parameters that do not depend on the data.

    Checks that need the cell count or the batch levels remain in
    ``fit_harmony``.

    Args:
        parameters: Keyword arguments for ``fit_harmony`` other than the data
            matrix and batch metadata.

    Returns:
        A new dictionary with the same parameters.
    """
    if parameters is None:
        return {}
    if not isinstance(parameters, Mapping):
        raise TypeError("Harmony parameters must be a mapping of keyword arguments")
    values = dict(parameters)
    allowed = set(inspect.signature(fit_harmony).parameters) - _DATA_ARGUMENTS
    unsupported = sorted(str(name) for name in values if name not in allowed)
    if unsupported:
        raise ValueError(f"Unsupported Harmony parameters: {', '.join(unsupported)}")
    _require_integer(values, "nclust", 1, optional=True)
    _require_integer(values, "max_iter_harmony", 0)
    _require_integer(values, "max_iter_kmeans", 1)
    _require_integer(values, "random_state", 0)
    _require_real(values, "tau", positive=False)
    _require_real(values, "block_size", positive=True)
    _require_real(values, "epsilon_cluster", positive=False)
    _require_real(values, "epsilon_harmony", positive=False)
    _require_real_values(values, "sigma", positive=True, optional=False)
    _require_real_values(values, "theta", positive=False, optional=True)
    _require_real_values(values, "lamb", positive=False, optional=True)
    cluster_fn = values.get("cluster_fn", "kmeans")
    if not callable(cluster_fn) and cluster_fn != "kmeans":
        raise ValueError("Harmony cluster_fn must be 'kmeans' or a callable")
    return values


def harmony_cluster_count(n_cells: int, nclust: int | None = None) -> int:
    """Return the number of clusters that ``fit_harmony`` fits for ``n_cells``."""
    if nclust is None:
        return max(1, int(np.min([np.round(n_cells / 30.0), 100])))
    if nclust < 1 or nclust > n_cells:
        raise ValueError("Harmony nclust must be between one and the cell count")
    return nclust


# Rows of distances to every centroid that each OpenMP thread of
# scikit-learn's chunked k-means loops holds while Harmony seeds its clusters.
_KMEANS_THREAD_CHUNK_ROWS = 256
# Python objects a fit holds whatever the data size, such as the optimizer,
# the batch encoding, and the objective history, measured below 96 KiB.
_HARMONY_OBJECT_BYTES = 256 * 1024


def harmony_peak_bytes(
    n_cells: int,
    dims: int,
    n_clusters: int,
    n_levels: int,
    *,
    block_size: float = 0.05,
    nthreads: int = 1,
) -> int:
    """Estimate the most bytes that fitting Harmony holds.

    It includes the input matrix but not the batch labels or a custom ``cluster_fn``.
    """
    cells = integer_argument(n_cells, "n_cells", minimum=1)
    d = integer_argument(dims, "dims", minimum=1)
    k = integer_argument(n_clusters, "n_clusters", minimum=1)
    b = integer_argument(n_levels, "n_levels", minimum=1)
    threads = integer_argument(nthreads, "nthreads", minimum=1)
    if isinstance(block_size, bool) or not isinstance(block_size, Real):
        raise TypeError("block_size must be a real number")
    if not math.isfinite(block_size) or block_size <= 0:
        raise ValueError("block_size must be finite and positive")
    # Every term below is a count of float64 values per cell. The update
    # reassigns cells in equal blocks of this fraction of the cells.
    block = 1.0 / math.ceil(1.0 / float(block_size))
    # The input, original, corrected, and unit-length coordinates; the int64
    # design with its intercept row; two bool copies of the one-hot design;
    # and per-cell labels and codes.
    held = 4 * d + (b + 1) + b / 4 + 1
    # Seeding: a C-ordered copy of the unit-length coordinates, the squared
    # norms, weights, and labels of every cell, and the distances of the
    # ``2 + log(k)`` k-means++ candidates with a product temporary.
    seeding = d + 2 * (2 + math.log(k)) + 8
    # From the first assignment on, the assignments and their distances.
    assigned = held + 2 * k
    # The objective holds weighted assignments and a cross-entropy term
    # with their product, or the term with a float64 copy of the design.
    objective = max(3 * k, 2 * k + b)
    # The update holds exponentiated distances and, for each block, copies
    # of its assignments and a float64 copy of its part of the design.
    update = max(2 * k, k + block * (3 * k + 1.25 * b))
    # A correction round holds the new corrected coordinates beside the
    # products of one cluster's ridge regression, or the temporaries that
    # make the corrected coordinates unit length.
    correction = d + (b + 1) + max(b + 1, 3 * d) + 1
    per_cell = max(
        held + seeding,
        assigned + max(objective, update, correction),
    )
    # scikit-learn's chunked k-means loops hold, per thread, the distances
    # of up to 256 cells to every centroid and the centroid sums.
    threads_bytes = threads * k * (_KMEANS_THREAD_CHUNK_ROWS + d + 1) * 8
    # The ridge penalty matrix lasts the whole fit, and each cluster's ridge
    # system and the solver's copy of it are temporaries of the same size.
    square_bytes = 3 * (b + 1) ** 2 * 8
    return (
        int(math.ceil(8 * cells * per_cell))
        + square_bytes
        + threads_bytes
        + _HARMONY_OBJECT_BYTES
    )


def fit_harmony(
    data_mat: np.ndarray,
    meta_data: pd.DataFrame,
    theta: float | int | np.ndarray | list[float] | None = None,
    lamb: float | int | np.ndarray | list[float] | None = None,
    sigma: float | np.ndarray = 0.1,
    nclust: int | None = None,
    tau: float = 0,
    block_size: float = 0.05,
    max_iter_harmony: int = 50,
    max_iter_kmeans: int = 20,
    epsilon_cluster: float = 1e-5,
    epsilon_harmony: float = 1e-4,
    random_state: int = 0,
    cluster_fn: ClusterFn = "kmeans",
) -> HarmonyResult:
    """Fit Harmony and return corrected coordinates with portable state."""
    from .. import Harmony

    if data_mat.ndim != 2:
        raise ValueError("Harmony data_mat must be two-dimensional")
    if data_mat.shape[1] != len(meta_data):
        raise ValueError(
            "Harmony metadata rows must match the number of embedding columns"
        )
    if data_mat.shape[1] < 2:
        raise ValueError("Harmony requires at least two cells")
    if max_iter_kmeans < 1:
        raise ValueError("Harmony max_iter_kmeans must be at least 1")
    if meta_data.empty:
        raise ValueError("Harmony requires at least one batch metadata column")
    if meta_data.columns.duplicated().any():
        raise ValueError("Harmony batch metadata column names must be unique")
    if meta_data.isna().any().any():
        raise ValueError("Harmony batch metadata cannot contain missing values")
    if not np.all(np.isfinite(data_mat)):
        raise ValueError("Harmony input contains non-finite values")

    n_cells = data_mat.shape[1]
    nclust = harmony_cluster_count(n_cells, nclust)

    sigma_arr = np.asarray(sigma, dtype=np.float64)
    if sigma_arr.ndim == 0:
        sigma_arr = np.full(nclust, float(sigma_arr.item()), dtype=np.float64)
    elif sigma_arr.shape != (nclust,):
        raise ValueError("Harmony sigma must be scalar or have one value per cluster")
    if not np.all(np.isfinite(sigma_arr)) or np.any(sigma_arr <= 0):
        raise ValueError("Harmony sigma values must be finite and positive")

    batch_columns = tuple(str(column) for column in meta_data.columns)
    categorical_metadata = meta_data.astype(
        {column: "category" for column in meta_data.columns}
    )
    # A declared category that no cell has is not a batch level.
    for column in categorical_metadata.columns:
        categorical_metadata[column] = categorical_metadata[
            column
        ].cat.remove_unused_categories()
    phi_frame = pd.get_dummies(categorical_metadata)
    phi = phi_frame.to_numpy().T
    batch_levels = tuple(
        tuple(str(level) for level in categorical_metadata[column].cat.categories)
        for column in categorical_metadata.columns
    )
    phi_n = np.asarray([len(levels) for levels in batch_levels], dtype=int)
    # get_dummies writes one indicator per category, column by column.
    assert phi.shape[0] == int(np.sum(phi_n))
    level_names = ", ".join(
        f"{column}={level}"
        for column, levels in zip(batch_columns, batch_levels, strict=True)
        for level in levels
    )

    def _expand_parameter(
        values: float | int | np.ndarray | list[float] | None,
        name: str,
    ) -> np.ndarray:
        if values is None:
            return np.ones(int(np.sum(phi_n)), dtype=np.float64)
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 0:
            return np.full(int(np.sum(phi_n)), float(array.item()))
        if array.shape == (len(phi_n),):
            return np.repeat(array, phi_n)
        if array.shape != (int(np.sum(phi_n)),):
            raise ValueError(
                f"Each Harmony batch level must have a {name}: pass one value, "
                f"one per batch column ({', '.join(batch_columns)}), or one per "
                f"batch level ({level_names}); got shape {array.shape}"
            )
        return array

    theta_arr = _expand_parameter(theta, "theta")
    lamb_arr = _expand_parameter(lamb, "lamb")
    if not np.all(np.isfinite(theta_arr)) or np.any(theta_arr < 0):
        raise ValueError("Harmony theta values must be finite and non-negative")
    if not np.all(np.isfinite(lamb_arr)) or np.any(lamb_arr < 0):
        raise ValueError("Harmony lamb values must be finite and non-negative")

    batch_counts = phi.sum(axis=1)
    batch_proportions = batch_counts / n_cells
    if tau > 0:
        theta_arr = theta_arr * (1 - np.exp(-((batch_counts / (nclust * tau)) ** 2)))

    lamb_mat = np.diag(np.insert(lamb_arr, 0, 0))
    phi_moe = np.vstack((np.repeat(1, n_cells), phi))
    if isinstance(cluster_fn, str) and cluster_fn != "kmeans":
        raise ValueError("Harmony cluster_fn must be 'kmeans' or a callable")

    optimizer = Harmony(
        data_mat,
        phi,
        phi_moe,
        batch_proportions,
        sigma_arr,
        theta_arr,
        max_iter_harmony,
        max_iter_kmeans,
        epsilon_cluster,
        epsilon_harmony,
        nclust,
        block_size,
        lamb_mat,
        random_state,
        cluster_fn,
    )
    corrected = optimizer.result()
    if not (
        np.all(np.isfinite(corrected))
        and np.all(np.isfinite(optimizer.R))
        and np.all(np.isfinite(optimizer.Y))
    ):
        raise ValueError(
            "Harmony produced non-finite corrected coordinates, assignments, "
            "or centroids"
        )

    cluster_backend = (
        "sklearn.cluster.KMeans"
        if isinstance(cluster_fn, str)
        else (
            f"{getattr(cluster_fn, '__module__', type(cluster_fn).__module__)}."
            f"{getattr(cluster_fn, '__qualname__', type(cluster_fn).__qualname__)}"
        )
    )
    parameters: dict[str, object] = {
        "nclust": int(nclust),
        "sigma": sigma_arr.tolist(),
        "theta": theta_arr.tolist(),
        "lambda": lamb_arr.tolist(),
        "tau": float(tau),
        "blockSize": float(block_size),
        "maxIterHarmony": int(max_iter_harmony),
        "maxIterKmeans": int(max_iter_kmeans),
        "epsilonCluster": float(epsilon_cluster),
        "epsilonHarmony": float(epsilon_harmony),
        "randomState": int(random_state),
        "clusterBackend": cluster_backend,
        "phiColumns": [str(column) for column in phi_frame.columns],
    }
    return HarmonyResult(
        original=optimizer.Z_orig,
        corrected=corrected,
        assignments=optimizer.R,
        centroids=optimizer.Y,
        sigma=sigma_arr.copy(),
        ridge=lamb_mat,
        batch_columns=batch_columns,
        batch_levels=batch_levels,
        parameters=parameters,
    )
