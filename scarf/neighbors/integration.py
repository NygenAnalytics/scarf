import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from ..utils.logging import logger
from ..utils.progress import tqdmbar


def wnn_integration(
    name1: str,
    g1: csr_matrix,
    ld1: np.ndarray,
    name2: str,
    g2: csr_matrix,
    ld2: np.ndarray,
    n_threads: int,
) -> coo_matrix:
    """Build a weighted nearest-neighbor graph from two modality graphs."""
    g1 = g1.tocsr(copy=False)
    g2 = g2.tocsr(copy=False)
    ld1 = np.asarray(ld1)
    ld2 = np.asarray(ld2)

    if g1.shape != g2.shape:
        raise ValueError("WNN graphs must have the same shape")
    if g1.shape[0] != g1.shape[1] or g1.shape[0] == 0:
        raise ValueError("WNN graphs must be non-empty square matrices")

    n_cells = g1.shape[0]
    for name, graph in ((name1, g1), (name2, g2)):
        row_degrees = np.diff(graph.indptr)
        if np.unique(row_degrees).size != 1:
            raise ValueError(f"WNN graph for {name} must have a regular row degree")
        if row_degrees[0] < 2:
            raise ValueError(
                f"WNN graph for {name} must have at least two neighbors per cell"
            )

    k1 = int(np.diff(g1.indptr)[0])
    k2 = int(np.diff(g2.indptr)[0])
    nk = min(k1, k2)
    if k1 != k2:
        logger.warning(
            f"WNN graphs have different neighbor counts ({name1}: {k1}, "
            f"{name2}: {k2}). The integrated graph will retain {nk} "
            "neighbors per cell."
        )

    for name, latent_data in ((name1, ld1), (name2, ld2)):
        if latent_data.ndim != 2 or latent_data.shape[1] == 0:
            raise ValueError(f"WNN embedding for {name} must be a non-empty matrix")
        if latent_data.shape[0] != n_cells:
            raise ValueError(
                f"WNN embedding for {name} must have one row per graph cell"
            )
        if not np.all(np.isfinite(latent_data)):
            raise ValueError(f"WNN embedding for {name} contains non-finite values")

    def calc_theta(
        point: np.ndarray,
        estimate: np.ndarray,
        nearest_distance: float,
        anchor_distance: float,
    ) -> float:
        estimate_distance = float(np.sqrt(((point - estimate) ** 2).sum(axis=0)))
        adjusted_distance = max(estimate_distance - nearest_distance, 0)
        bandwidth = max(
            anchor_distance - nearest_distance,
            np.finfo(np.float64).eps,
        )
        return float(np.exp(-adjusted_distance / bandwidth))

    def calc_affinity_ratio(
        point: np.ndarray,
        self_estimate: np.ndarray,
        cross_estimate: np.ndarray,
        nearest_distance: float,
        anchor_distance: float,
        epsilon: float = 1e-4,
    ) -> float:
        theta_self = calc_theta(
            point,
            self_estimate,
            nearest_distance,
            anchor_distance,
        )
        theta_cross = calc_theta(
            point,
            cross_estimate,
            nearest_distance,
            anchor_distance,
        )
        return float(np.clip(theta_self / (theta_cross + epsilon), 0, 200))

    neighbor_indices1 = g1.indices.reshape(n_cells, k1)
    neighbor_indices2 = g2.indices.reshape(n_cells, k2)

    from threadpoolctl import threadpool_limits

    column_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    with threadpool_limits(limits=n_threads):
        for cell_idx in tqdmbar(range(n_cells), desc="Building WNN graph"):
            neighbors1 = neighbor_indices1[cell_idx]
            neighbors2 = neighbor_indices2[cell_idx]
            mixed_k = np.union1d(neighbors1, neighbors2)

            distances1 = np.sqrt(((ld1[cell_idx] - ld1[mixed_k]) ** 2).sum(axis=1))
            distances2 = np.sqrt(((ld2[cell_idx] - ld2[mixed_k]) ** 2).sum(axis=1))

            self_distances1 = distances1[np.searchsorted(mixed_k, neighbors1)]
            self_distances2 = distances2[np.searchsorted(mixed_k, neighbors2)]
            ranked_distances1 = np.sort(self_distances1)
            ranked_distances2 = np.sort(self_distances2)

            score1 = calc_affinity_ratio(
                ld1[cell_idx],
                ld1[neighbors1].mean(axis=0),
                ld1[neighbors2].mean(axis=0),
                float(ranked_distances1[0]),
                float(ranked_distances1[-2]),
            )
            score2 = calc_affinity_ratio(
                ld2[cell_idx],
                ld2[neighbors2].mean(axis=0),
                ld2[neighbors1].mean(axis=0),
                float(ranked_distances2[0]),
                float(ranked_distances2[-2]),
            )

            max_score = max(score1, score2)
            exp1 = np.exp(score1 - max_score)
            exp2 = np.exp(score2 - max_score)
            normalizer = exp1 + exp2
            weighted_distances = distances1 * (exp1 / normalizer) + distances2 * (
                exp2 / normalizer
            )

            indices = np.argsort(weighted_distances)[:nk]
            column_parts.append(mixed_k[indices])
            selected_distances = weighted_distances[indices]
            offsets = selected_distances - selected_distances[0]
            scale = offsets.mean()
            if scale <= np.finfo(np.float64).eps:
                edge_weights = np.ones(selected_distances.shape, dtype=np.float64)
            else:
                edge_weights = np.maximum(
                    np.exp(-(offsets / scale)),
                    np.finfo(np.float64).tiny,
                )
            data_parts.append(edge_weights)

    merged_data = np.hstack(data_parts)
    rows = np.repeat(np.arange(n_cells), nk)
    merged_columns = np.hstack(column_parts)
    if not np.all(np.isfinite(merged_data)):
        raise FloatingPointError("WNN integration produced non-finite graph weights")
    return coo_matrix((merged_data, (rows, merged_columns)), shape=g1.shape)
