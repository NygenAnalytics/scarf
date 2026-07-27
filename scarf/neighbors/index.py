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
    neighbor_indices = np.asarray(indices)
    neighbor_distances = np.asarray(distances)
    references = np.asarray(ref_idx)
    if (
        neighbor_indices.ndim != 2
        or neighbor_distances.shape != neighbor_indices.shape
        or references.shape != (neighbor_indices.shape[0],)
        or neighbor_indices.shape[1] < 2
    ):
        raise ValueError("KNN query arrays have incompatible shapes")
    matches = neighbor_indices == references[:, np.newaxis]
    has_self = matches.any(axis=1)
    self_positions = matches.argmax(axis=1)
    drop_positions = np.where(
        has_self,
        self_positions,
        neighbor_indices.shape[1] - 1,
    )
    columns = np.arange(neighbor_indices.shape[1])[np.newaxis, :]
    keep = columns != drop_positions[:, np.newaxis]
    output_shape = (neighbor_indices.shape[0], neighbor_indices.shape[1] - 1)
    fixed_indices = neighbor_indices[keep].reshape(output_shape)
    fixed_distances = neighbor_distances[keep].reshape(output_shape)
    missed_self_hits = int(np.count_nonzero(~has_self))
    return fixed_indices, fixed_distances, missed_self_hits
