import numpy as np
from numba import jit
from scipy.sparse import coo_matrix, csr_matrix

from ..utils.progress import tqdmbar


def _is_umap_version_new() -> bool:
    import umap
    from packaging import version

    return version.parse(umap.__version__) >= version.parse("0.5.0")


def smooth_knn_chunk(
    indices: np.ndarray,
    distances: np.ndarray,
    *,
    local_connectivity: float,
    bandwidth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert one KNN distance block into fuzzy graph edges."""
    from umap.umap_ import compute_membership_strengths, smooth_knn_dist

    indices_array = np.asarray(indices)
    distance_array = np.asarray(distances, dtype=np.float32)
    if not distance_array.flags.c_contiguous:
        distance_array = np.ascontiguousarray(distance_array)
    sigmas, rhos = smooth_knn_dist(
        distance_array,
        k=indices_array.shape[1],
        local_connectivity=local_connectivity,
        bandwidth=bandwidth,
    )
    if _is_umap_version_new():
        rows, columns, values, _ = compute_membership_strengths(
            indices_array,
            distance_array,
            sigmas,
            rhos,
        )
    else:
        rows, columns, values = compute_membership_strengths(
            indices_array,
            distance_array,
            sigmas,
            rhos,
        )
    return np.asarray(rows), np.asarray(columns), np.asarray(values)


@jit(nopython=True)
def calc_snn(indices: np.ndarray) -> np.ndarray:
    """Calculate shared-neighbor fractions for a KNN index matrix."""
    ncells, nk = indices.shape
    snn = np.zeros((ncells, nk))
    for i in range(ncells):
        for j in range(nk):
            k = indices[i][j]
            snn[i][j] = len(set(indices[i]).intersection(set(indices[k])))
    return np.asarray(snn / (nk - 1))


def weight_sort_indices(
    i: np.ndarray,
    w: np.ndarray,
    wn: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sort neighbors by adjusted weight and remove duplicates."""
    idx = np.argsort(wn)[::-1]
    i = i[idx]
    w = w[idx]
    _, unique_idx = np.unique(i, return_index=True)
    unique_idx_arr = np.asarray(sorted(unique_idx))
    return i[unique_idx_arr][:n], w[unique_idx_arr][:n]


def merge_graphs(csr_mats: list[csr_matrix]) -> coo_matrix:
    """Merge regular graphs using edge weights and shared neighbors."""
    try:
        assert len({matrix.shape for matrix in csr_mats}) == 1
    except AssertionError:
        raise ValueError("ERROR: All graphs do not have the same shape.")
    try:
        assert len({matrix.size for matrix in csr_mats}) == 1
    except AssertionError:
        raise ValueError("ERROR: All graphs do not have the same number of edges")

    nk = csr_mats[0][0].indices.shape[0]
    snns = []
    for matrix in tqdmbar(csr_mats, desc="Identifying SNNs in graphs"):
        snns.append(
            calc_snn(
                matrix.indices.reshape((matrix.shape[0], matrix[0].indices.shape[0]))
            )
        )
    columns: list[int] = []
    data: list[float] = []
    for row_idx in tqdmbar(
        range(csr_mats[0].shape[0]),
        desc="Merging graph edges",
    ):
        merged_indices = np.hstack([matrix[row_idx].indices for matrix in csr_mats])
        adjusted_weights = np.hstack(
            [
                matrix[row_idx].data + snns[index][row_idx]
                for index, matrix in enumerate(csr_mats)
            ]
        )
        merged_weights = np.hstack([matrix[row_idx].data for matrix in csr_mats])
        merged_indices, merged_weights = weight_sort_indices(
            merged_indices,
            merged_weights,
            adjusted_weights,
            nk,
        )
        columns.extend(merged_indices)
        data.extend(merged_weights)
    shape = csr_mats[0].shape
    rows = np.repeat(range(shape[0]), nk)
    return coo_matrix((data, (rows, columns)), shape=shape)
