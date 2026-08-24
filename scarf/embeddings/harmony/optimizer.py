from collections.abc import Callable
from functools import partial

import numpy as np
from sklearn.cluster import KMeans

from ...utils.logging import logger
from ...utils.progress import tqdmbar
from .models import ClusterFn


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
        self.Z_orig = np.array(Z, dtype=np.float64, copy=True)
        self.Z_corr = np.array(self.Z_orig, copy=True)
        self.Z_cos = _normalize_columns(self.Z_orig)
        self._rng = np.random.RandomState(random_state)

        self.Phi = Phi
        self.Phi_moe = Phi_moe
        self.N = self.Z_corr.shape[1]
        self.Pr_b = Pr_b
        self.B = self.Phi.shape[0]
        self.d = self.Z_corr.shape[0]
        self.window_size = 3
        self.epsilon_kmeans = epsilon_kmeans
        self.epsilon_harmony = epsilon_harmony

        self.lamb = lamb
        self.sigma = sigma
        self.sigma_prior = sigma
        self.block_size = block_size
        self.K = K
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
                Harmony._cluster_kmeans,
                random_state=random_state,
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
        data: np.ndarray,
        K: int,
        random_state: int | None,
    ) -> np.ndarray:
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
        self.Y = _normalize_columns(self.Y)
        self.dist_mat = 2 * (1 - np.dot(self.Y.T, self.Z_cos))
        self.R = -self.dist_mat
        self.R = self.R / self.sigma[:, None]
        self.R -= np.max(self.R, axis=0)
        self.R = np.exp(self.R)
        self.R = self.R / np.sum(self.R, axis=0)
        self.E = np.outer(np.sum(self.R, axis=1), self.Pr_b)
        self.O = np.inner(self.R, self.Phi)
        self.compute_objective()
        self.objective_harmony.append(self.objective_kmeans[-1])

    def compute_objective(self) -> None:
        kmeans_error = np.sum(np.multiply(self.R, self.dist_mat))
        entropy = np.sum(safe_entropy(self.R) * self.sigma[:, np.newaxis])
        x = self.R * self.sigma[:, np.newaxis]
        y = np.tile(self.theta[:, np.newaxis], self.K).T
        z = np.log((self.O + 1) / (self.E + 1))
        w = np.dot(y * z, self.Phi)
        cross_entropy = np.sum(x * w)
        self.objective_kmeans.append(kmeans_error + entropy + cross_entropy)
        self.objective_kmeans_dist.append(kmeans_error)
        self.objective_kmeans_entropy.append(entropy)
        self.objective_kmeans_cross.append(cross_entropy)

    def harmonize(self, iter_harmony: int = 10) -> int:
        converged = False
        progress = tqdmbar(
            desc="Harmonizing batches",
            total=iter_harmony,
        )
        try:
            for _ in range(1, iter_harmony + 1):
                self.cluster()
                self.Z_cos, self.Z_corr, self.W, self.Phi_Rk = moe_correct_ridge(
                    self.Z_orig,
                    self.R,
                    self.W,
                    self.K,
                    self.Phi_Rk,
                    self.Phi_moe,
                    self.lamb,
                )
                progress.update()
                converged = self.check_convergence(1)
                if converged:
                    progress.total = progress.n
                    progress.refresh()
                    break
        finally:
            progress.close()
        self.Y = _normalize_columns(np.dot(self.Z_cos, self.R.T))
        self.dist_mat = 2 * (1 - np.dot(self.Y.T, self.Z_cos))
        if not converged:
            logger.warning("Harmony stopped before convergence")
        return 0

    def cluster(self) -> int:
        self.dist_mat = 2 * (1 - np.dot(self.Y.T, self.Z_cos))
        for iteration in range(self.max_iter_kmeans):
            self.Y = np.dot(self.Z_cos, self.R.T)
            self.Y = self.Y / np.linalg.norm(self.Y, ord=2, axis=0)
            self.dist_mat = 2 * (1 - np.dot(self.Y.T, self.Z_cos))
            self.update_R()
            self.compute_objective()
            if iteration > self.window_size and self.check_convergence(0):
                break
        self.kmeans_rounds.append(iteration)
        self.objective_harmony.append(self.objective_kmeans[-1])
        return 0

    def update_R(self) -> int:
        self._scale_dist = -self.dist_mat
        self._scale_dist = self._scale_dist / self.sigma[:, None]
        self._scale_dist -= np.max(self._scale_dist, axis=0)
        self._scale_dist = np.exp(self._scale_dist)
        update_order = self._rng.permutation(self.N)
        n_blocks = np.ceil(1 / self.block_size).astype(int)
        blocks = np.array_split(update_order, n_blocks)
        for block in blocks:
            self.E -= np.outer(np.sum(self.R[:, block], axis=1), self.Pr_b)
            self.O -= np.dot(self.R[:, block], self.Phi[:, block].T)
            self.R[:, block] = self._scale_dist[:, block]
            self.R[:, block] = np.multiply(
                self.R[:, block],
                np.dot(
                    np.power((self.E + 1) / (self.O + 1), self.theta),
                    self.Phi[:, block],
                ),
            )
            self.R[:, block] = self.R[:, block] / np.linalg.norm(
                self.R[:, block],
                ord=1,
                axis=0,
            )
            self.E += np.outer(np.sum(self.R[:, block], axis=1), self.Pr_b)
            self.O += np.dot(self.R[:, block], self.Phi[:, block].T)
        return 0

    def check_convergence(self, i_type: int) -> bool:
        obj_old = 0.0
        obj_new = 0.0
        if i_type == 0:
            objective_count = len(self.objective_kmeans)
            for offset in range(self.window_size):
                obj_old += self.objective_kmeans[objective_count - 2 - offset]
                obj_new += self.objective_kmeans[objective_count - 1 - offset]
            if abs(obj_old - obj_new) / abs(obj_old) < self.epsilon_kmeans:
                return True
            return False
        if i_type == 1:
            obj_old = self.objective_harmony[-2]
            obj_new = self.objective_harmony[-1]
            error = (obj_old - obj_new) / abs(obj_old)
            logger.debug(
                f"Harmony error after {len(self.objective_harmony)} iterations: {error}"
            )
            if error < self.epsilon_harmony:
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
    for cluster_index in range(K):
        Phi_Rk = np.multiply(Phi_moe, R[cluster_index, :])
        x = np.dot(Phi_Rk, Phi_moe.T) + lamb
        try:
            W = np.linalg.solve(x, np.dot(Phi_Rk, Z_orig.T))
        except np.linalg.LinAlgError as exc:
            raise ValueError("Harmony ridge system is singular") from exc
        W[0, :] = 0
        Z_corr -= np.dot(W.T, Phi_Rk)
    Z_cos = _normalize_columns(Z_corr)
    return Z_cos, Z_corr, W, Phi_Rk


def _normalize_columns(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, ord=2, axis=0)
    normalized = np.zeros_like(values, dtype=np.float64)
    nonzero = norms > 0
    normalized[:, nonzero] = values[:, nonzero] / norms[nonzero]
    return normalized
