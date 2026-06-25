from collections.abc import Callable
from functools import partial

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .utils import tqdmbar, logger

type ClusterFn = str | Callable[[np.ndarray, int], np.ndarray]


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
    """Run Harmony batch correction on a PCA embedding.

    Args:
        data_mat: Embedding matrix, shape (n_dims, n_cells).
        meta_data: Batch metadata DataFrame (one column per batch variable).
        theta: Cluster diversity penalty per batch level (default: 1 per level).
        lamb: Ridge penalty per batch level (default: 1 per level).
        sigma: Kernel width(s) for soft k-means clustering.
        nclust: Number of Harmony clusters (default: min(round(N/30), 100)).
        tau: Protects small batches when > 0.
        block_size: Fraction of cells per update block.
        max_iter_harmony: Maximum Harmony iterations.
        max_iter_kmeans: Maximum k-means iterations per round.
        epsilon_cluster: Convergence threshold for clustering.
        epsilon_harmony: Convergence threshold for Harmony.
        random_state: Random seed.
        cluster_fn: Clustering backend (``'kmeans'``).

    Returns:
        Batch-corrected embedding matrix, shape (n_dims, n_cells).
    """

    N = data_mat.shape[1]
    if nclust is None:
        nclust = np.min([np.round(N / 30.0), 100]).astype(int)

    if type(sigma) is float and nclust > 1:
        sigma = np.repeat(sigma, nclust)

    phi = pd.get_dummies(meta_data).to_numpy().T
    phi_n = meta_data.describe().loc["unique"].to_numpy().astype(int)

    if theta is None:
        theta = np.repeat([1] * len(phi_n), phi_n)
    elif isinstance(theta, float) or isinstance(theta, int):
        theta = np.repeat([theta] * len(phi_n), phi_n)
    elif len(theta) == len(phi_n):
        theta = np.repeat([theta], phi_n)

    assert len(theta) == np.sum(phi_n), "each batch variable must have a theta"

    if lamb is None:
        lamb = np.repeat([1] * len(phi_n), phi_n)
    elif isinstance(lamb, float) or isinstance(lamb, int):
        lamb = np.repeat([lamb] * len(phi_n), phi_n)
    elif len(lamb) == len(phi_n):
        lamb = np.repeat([lamb], phi_n)

    assert len(lamb) == np.sum(phi_n), "each batch variable must have a lambda"

    # Number of items in each category.
    N_b = phi.sum(axis=1)
    # Proportion of items in each category.
    Pr_b = N_b / N

    if tau > 0:
        theta = theta * (1 - np.exp(-((N_b / (nclust * tau)) ** 2)))

    lamb_mat = np.diag(np.insert(lamb, 0, 0))

    phi_moe = np.vstack((np.repeat(1, N), phi))

    np.random.seed(random_state)

    sigma_arr = np.asarray(sigma)
    theta_arr = np.asarray(theta)

    ho = Harmony(
        data_mat,
        phi,
        phi_moe,
        Pr_b,
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

    return ho.result()


class Harmony:
    def __init__(
        self,
        Z: np.ndarray,
        Phi: np.ndarray,
        Phi_moe: np.ndarray,
        Pr_b: np.ndarray,
        sigma: np.ndarray,
        theta: np.ndarray,
        max_iter_harmony: int,
        max_iter_kmeans: int,
        epsilon_kmeans: float,
        epsilon_harmony: float,
        K: int,
        block_size: float,
        lamb: np.ndarray,
        random_state: int | None = None,
        cluster_fn: ClusterFn = "kmeans",
    ) -> None:
        self.Z_corr = np.array(Z)
        self.Z_orig = np.array(Z)

        self.Z_cos = self.Z_orig / self.Z_orig.max(axis=0)
        self.Z_cos = self.Z_cos / np.linalg.norm(self.Z_cos, ord=2, axis=0)

        self.Phi = Phi
        self.Phi_moe = Phi_moe
        self.N = self.Z_corr.shape[1]
        self.Pr_b = Pr_b
        self.B = self.Phi.shape[0]  # number of batch variables
        self.d = self.Z_corr.shape[0]
        self.window_size = 3
        self.epsilon_kmeans = epsilon_kmeans
        self.epsilon_harmony = epsilon_harmony

        self.lamb = lamb
        self.sigma = sigma
        self.sigma_prior = sigma
        self.block_size = block_size
        self.K = K  # number of clusters
        self.max_iter_harmony = max_iter_harmony
        self.max_iter_kmeans = max_iter_kmeans
        self.theta = theta

        self.objective_harmony: list[float] = []
        self.objective_kmeans: list[float] = []
        self.objective_kmeans_dist: list[float] = []
        self.objective_kmeans_entropy: list[float] = []
        self.objective_kmeans_cross: list[float] = []
        self.kmeans_rounds: list[int] = []

        self.allocate_buffers()
        resolved_cluster_fn: Callable[[np.ndarray, int], np.ndarray]
        if isinstance(cluster_fn, str):
            resolved_cluster_fn = partial(
                Harmony._cluster_kmeans, random_state=random_state
            )
        else:
            resolved_cluster_fn = cluster_fn
        self.init_cluster(resolved_cluster_fn)
        self.harmonize(self.max_iter_harmony)

    def result(self) -> np.ndarray:
        return self.Z_corr

    def allocate_buffers(self) -> None:
        self._scale_dist = np.zeros((self.K, self.N))
        self.dist_mat = np.zeros((self.K, self.N))
        self.O = np.zeros((self.K, self.B))
        self.E = np.zeros((self.K, self.B))
        self.W = np.zeros((self.B + 1, self.d))
        self.Phi_Rk = np.zeros((self.B + 1, self.N))

    @staticmethod
    def _cluster_kmeans(
        data: np.ndarray, K: int, random_state: int | None
    ) -> np.ndarray:
        # Start with cluster centroids
        centers = (
            KMeans(
                n_clusters=K,
                init="k-means++",
                n_init=10,
                max_iter=25,
                random_state=random_state,
            )
            .fit(data)
            .cluster_centers_
        )
        return np.asarray(centers)

    def init_cluster(self, cluster_fn: Callable[[np.ndarray, int], np.ndarray]) -> None:
        self.Y = cluster_fn(self.Z_cos.T, self.K).T
        # (1) Normalize
        self.Y = self.Y / np.linalg.norm(self.Y, ord=2, axis=0)
        # (2) Assign cluster probabilities
        self.dist_mat = 2 * (1 - np.dot(self.Y.T, self.Z_cos))
        self.R = -self.dist_mat
        self.R = self.R / self.sigma[:, None]
        self.R -= np.max(self.R, axis=0)
        self.R = np.exp(self.R)
        self.R = self.R / np.sum(self.R, axis=0)
        # (3) Batch diversity statistics
        self.E = np.outer(np.sum(self.R, axis=1), self.Pr_b)
        self.O = np.inner(self.R, self.Phi)
        self.compute_objective()
        # Save results
        self.objective_harmony.append(self.objective_kmeans[-1])

    def compute_objective(self) -> None:
        kmeans_error = np.sum(np.multiply(self.R, self.dist_mat))
        # Entropy
        _entropy = np.sum(safe_entropy(self.R) * self.sigma[:, np.newaxis])
        # Cross Entropy
        x = self.R * self.sigma[:, np.newaxis]
        y = np.tile(self.theta[:, np.newaxis], self.K).T
        z = np.log((self.O + 1) / (self.E + 1))
        w = np.dot(y * z, self.Phi)
        _cross_entropy = np.sum(x * w)
        # Save results
        self.objective_kmeans.append(kmeans_error + _entropy + _cross_entropy)
        self.objective_kmeans_dist.append(kmeans_error)
        self.objective_kmeans_entropy.append(_entropy)
        self.objective_kmeans_cross.append(_cross_entropy)

    def harmonize(self, iter_harmony: int = 10) -> int:
        converged = False
        for i in tqdmbar(range(1, iter_harmony + 1), desc="Harmonizing batches"):
            # STEP 1: Clustering
            self.cluster()
            # STEP 2: Regress out covariates
            self.Z_cos, self.Z_corr, self.W, self.Phi_Rk = moe_correct_ridge(
                self.Z_orig,
                self.R,
                self.W,
                self.K,
                self.Phi_Rk,
                self.Phi_moe,
                self.lamb,
            )
            # STEP 3: Check for convergence
            converged = self.check_convergence(1)
            if converged:
                break
        if not converged:
            logger.warning("Stopped before convergence")
        return 0

    def cluster(self) -> int:
        # Z_cos has changed
        # R is assumed to not have changed
        # Update Y to match new integrated data
        self.dist_mat = 2 * (1 - np.dot(self.Y.T, self.Z_cos))
        for i in range(self.max_iter_kmeans):
            # print("kmeans {}".format(i))
            # STEP 1: Update Y
            self.Y = np.dot(self.Z_cos, self.R.T)
            self.Y = self.Y / np.linalg.norm(self.Y, ord=2, axis=0)
            # STEP 2: Update dist_mat
            self.dist_mat = 2 * (1 - np.dot(self.Y.T, self.Z_cos))
            # STEP 3: Update R
            self.update_R()
            # STEP 4: Check for convergence
            self.compute_objective()
            if i > self.window_size:
                converged = self.check_convergence(0)
                if converged:
                    break
        self.kmeans_rounds.append(i)
        self.objective_harmony.append(self.objective_kmeans[-1])
        return 0

    def update_R(self) -> int:
        self._scale_dist = -self.dist_mat
        self._scale_dist = self._scale_dist / self.sigma[:, None]
        self._scale_dist -= np.max(self._scale_dist, axis=0)
        self._scale_dist = np.exp(self._scale_dist)
        # Update cells in blocks
        update_order = np.arange(self.N)
        np.random.shuffle(update_order)
        n_blocks = np.ceil(1 / self.block_size).astype(int)
        blocks = np.array_split(update_order, n_blocks)
        for b in blocks:
            # STEP 1: Remove cells
            self.E -= np.outer(np.sum(self.R[:, b], axis=1), self.Pr_b)
            self.O -= np.dot(self.R[:, b], self.Phi[:, b].T)
            # STEP 2: Recompute R for removed cells
            self.R[:, b] = self._scale_dist[:, b]
            self.R[:, b] = np.multiply(
                self.R[:, b],
                np.dot(
                    np.power((self.E + 1) / (self.O + 1), self.theta), self.Phi[:, b]
                ),
            )
            self.R[:, b] = self.R[:, b] / np.linalg.norm(self.R[:, b], ord=1, axis=0)
            # STEP 3: Put cells back
            self.E += np.outer(np.sum(self.R[:, b], axis=1), self.Pr_b)
            self.O += np.dot(self.R[:, b], self.Phi[:, b].T)
        return 0

    def check_convergence(self, i_type: int) -> bool:
        obj_old = 0.0
        obj_new = 0.0
        # Clustering, compute new window mean
        if i_type == 0:
            okl = len(self.objective_kmeans)
            for i in range(self.window_size):
                obj_old += self.objective_kmeans[okl - 2 - i]
                obj_new += self.objective_kmeans[okl - 1 - i]
            if abs(obj_old - obj_new) / abs(obj_old) < self.epsilon_kmeans:
                return True
            return False
        # Harmony
        if i_type == 1:
            obj_old = self.objective_harmony[-2]
            obj_new = self.objective_harmony[-1]
            e = (obj_old - obj_new) / abs(obj_old)
            logger.info(f"Error after {len(self.objective_harmony)} iterations: {e}")
            if e < self.epsilon_harmony:
                return True
            return False
        return True


def safe_entropy(x: np.ndarray) -> np.ndarray:
    y = np.multiply(x, np.log(x))
    y[~np.isfinite(y)] = 0.0
    return np.asarray(y)


def moe_correct_ridge(
    Z_orig: np.ndarray,
    R: np.ndarray,
    W: np.ndarray,
    K: int,
    Phi_Rk: np.ndarray,
    Phi_moe: np.ndarray,
    lamb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Z_corr = Z_orig.copy()
    for i in range(K):
        Phi_Rk = np.multiply(Phi_moe, R[i, :])
        x = np.dot(Phi_Rk, Phi_moe.T) + lamb
        W = np.dot(np.dot(np.linalg.inv(x), Phi_Rk), Z_orig.T)
        W[0, :] = 0  # do not remove the intercept
        Z_corr -= np.dot(W.T, Phi_Rk)
    Z_cos = Z_corr / np.linalg.norm(Z_corr, ord=2, axis=0)
    return Z_cos, Z_corr, W, Phi_Rk
