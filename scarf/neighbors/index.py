from typing import Any

import numpy as np


def instantiate_knn_index(
    space: str,
    dim: int,
    max_elements: int,
    ef_construction: int,
    M: int,
    random_seed: int,
    ef: int,
    num_threads: int,
) -> Any:
    """Create and configure an hnswlib KNN index."""
    import hnswlib

    ann_idx = hnswlib.Index(space=space, dim=dim)
    ann_idx.init_index(
        max_elements=max_elements,
        ef_construction=ef_construction,
        M=M,
        random_seed=random_seed,
    )
    ann_idx.set_ef(ef)
    ann_idx.set_num_threads(num_threads)
    return ann_idx


def fix_knn_query(
    indices: np.ndarray,
    distances: np.ndarray,
    ref_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Remove self-neighbor entries from KNN query results."""
    fixed_ind, fixed_dist = indices.copy()[:, 1:], distances.copy()[:, 1:]
    mis_idx = indices[:, 0].reshape(1, -1)[0] != ref_idx
    n_mis = int(mis_idx.sum())
    if n_mis > 0:
        for n, i, neighbors, neighbor_distances in zip(
            np.where(mis_idx)[0],
            ref_idx[mis_idx],
            indices[mis_idx],
            distances[mis_idx],
        ):
            self_positions = np.where(neighbors == i)[0]
            if len(self_positions) > 0:
                self_position = self_positions[0]
                neighbors = np.array(
                    list(neighbors[:self_position])
                    + list(neighbors[self_position + 1 :])
                )
                neighbor_distances = np.array(
                    list(neighbor_distances[:self_position])
                    + list(neighbor_distances[self_position + 1 :])
                )
            else:
                neighbors = neighbors[:-1]
                neighbor_distances = neighbor_distances[:-1]
            fixed_ind[n] = neighbors
            fixed_dist[n] = neighbor_distances
    return fixed_ind, fixed_dist, n_mis
