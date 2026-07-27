import numpy as np
import pandas as pd

from .models import ClusterFn, HarmonyResult


def run_harmony(
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
) -> np.ndarray:
    """Run Harmony batch correction on a PCA embedding."""
    from .. import fit_harmony

    return fit_harmony(
        data_mat,
        meta_data,
        theta=theta,
        lamb=lamb,
        sigma=sigma,
        nclust=nclust,
        tau=tau,
        block_size=block_size,
        max_iter_harmony=max_iter_harmony,
        max_iter_kmeans=max_iter_kmeans,
        epsilon_cluster=epsilon_cluster,
        epsilon_harmony=epsilon_harmony,
        random_state=random_state,
        cluster_fn=cluster_fn,
    ).corrected


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
    if meta_data.empty:
        raise ValueError("Harmony requires at least one batch metadata column")
    if meta_data.columns.duplicated().any():
        raise ValueError("Harmony batch metadata column names must be unique")
    if meta_data.isna().any().any():
        raise ValueError("Harmony batch metadata cannot contain missing values")
    if not np.all(np.isfinite(data_mat)):
        raise ValueError("Harmony input contains non-finite values")

    n_cells = data_mat.shape[1]
    if nclust is None:
        nclust = max(1, int(np.min([np.round(n_cells / 30.0), 100])))
    elif nclust < 1 or nclust > n_cells:
        raise ValueError("Harmony nclust must be between one and the cell count")

    sigma_arr = np.asarray(sigma, dtype=np.float64)
    if sigma_arr.ndim == 0:
        sigma_arr = np.full(nclust, float(sigma_arr.item()), dtype=np.float64)
    elif sigma_arr.shape != (nclust,):
        raise ValueError("Harmony sigma must be scalar or have one value per cluster")
    if not np.all(np.isfinite(sigma_arr)) or np.any(sigma_arr <= 0):
        raise ValueError("Harmony sigma values must be finite and positive")

    categorical_metadata = meta_data.astype(
        {column: "category" for column in meta_data.columns}
    )
    phi_frame = pd.get_dummies(categorical_metadata)
    phi = phi_frame.to_numpy().T
    phi_n = np.asarray(
        [meta_data[column].nunique() for column in meta_data.columns],
        dtype=int,
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
            raise ValueError(f"Each Harmony batch level must have a {name}")
        return array

    theta_arr = _expand_parameter(theta, "theta")
    lamb_arr = _expand_parameter(lamb, "lambda")
    if not np.all(np.isfinite(theta_arr)) or np.any(theta_arr < 0):
        raise ValueError("Harmony theta values must be finite and non-negative")
    if not np.all(np.isfinite(lamb_arr)) or np.any(lamb_arr < 0):
        raise ValueError("Harmony lambda values must be finite and non-negative")

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

    batch_columns = tuple(str(column) for column in meta_data.columns)
    batch_levels = tuple(
        tuple(str(value) for value in pd.unique(meta_data[column]))
        for column in meta_data.columns
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
        corrected=optimizer.result(),
        assignments=optimizer.R,
        centroids=optimizer.Y,
        sigma=sigma_arr.copy(),
        ridge=lamb_mat,
        batch_columns=batch_columns,
        batch_levels=batch_levels,
        parameters=parameters,
    )
