from numbers import Real

import numpy as np
from numba import jit, njit
from scipy.sparse import coo_matrix, csr_matrix

from ..utils.progress import iter_progress


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
    rows, columns, values, _ = compute_membership_strengths(
        indices_array,
        distance_array,
        sigmas,
        rhos,
    )
    return np.asarray(rows), np.asarray(columns), np.asarray(values)


def validate_connectivity_parameters(
    local_connectivity: float,
    bandwidth: float,
) -> tuple[float, float]:
    if isinstance(local_connectivity, bool) or not isinstance(
        local_connectivity,
        Real,
    ):
        raise TypeError("local_connectivity must be a real number")
    if isinstance(bandwidth, bool) or not isinstance(bandwidth, Real):
        raise TypeError("bandwidth must be a real number")
    resolved_local_connectivity = float(local_connectivity)
    resolved_bandwidth = float(bandwidth)
    if not np.isfinite(resolved_local_connectivity) or resolved_local_connectivity < 0:
        raise ValueError("local_connectivity must be finite and non-negative")
    if not np.isfinite(resolved_bandwidth) or resolved_bandwidth <= 0:
        raise ValueError("bandwidth must be finite and greater than zero")
    return resolved_local_connectivity, resolved_bandwidth


def build_connectivity_arrays(
    indices: np.ndarray,
    distances: np.ndarray,
    *,
    local_connectivity: float,
    bandwidth: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build edge and weight arrays retaining every KNN entry."""
    local_connectivity, bandwidth = validate_connectivity_parameters(
        local_connectivity,
        bandwidth,
    )
    neighbor_indices = np.asarray(indices)
    neighbor_distances = np.asarray(distances)
    if neighbor_indices.ndim != 2 or neighbor_distances.shape != neighbor_indices.shape:
        raise ValueError("Neighbor indices and distances must have matching matrices")
    if not np.issubdtype(neighbor_indices.dtype, np.integer):
        raise TypeError("Neighbor indices must be integers")
    if np.any(neighbor_indices < 0) or np.any(
        neighbor_indices >= neighbor_indices.shape[0]
    ):
        raise ValueError("Neighbor indices are outside the cell range")
    if not np.all(np.isfinite(neighbor_distances)) or np.any(neighbor_distances < 0):
        raise ValueError("Neighbor distances must be finite and non-negative")
    rows, columns, values = smooth_knn_chunk(
        neighbor_indices,
        neighbor_distances,
        local_connectivity=local_connectivity,
        bandwidth=bandwidth,
    )
    expected = int(neighbor_indices.size)
    if (
        rows.shape != (expected,)
        or columns.shape != (expected,)
        or values.shape != (expected,)
    ):
        raise ValueError("UMAP membership output does not match the KNN matrix")
    if (
        np.any(rows < 0)
        or np.any(columns < 0)
        or np.any(rows > np.iinfo(np.uint32).max)
        or np.any(columns > np.iinfo(np.uint32).max)
    ):
        raise ValueError("Connectivity endpoints exceed uint32 bounds")
    edges = np.empty((expected, 2), dtype=np.uint32)
    edges[:, 0] = rows
    edges[:, 1] = columns
    weights = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(weights)):
        raise ValueError("Connectivity weights must be finite")
    return edges, weights


def take_nearest_per_row(
    weights: np.ndarray,
    edges: np.ndarray,
    n_cells: int,
    use_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the ``use_k`` nearest stored edges of every cell.

    Edges are stored row major and ordered by increasing distance inside a cell,
    so a row's leading entries are its nearest neighbors. Rows with fewer than
    ``use_k`` stored edges keep all their entries.
    """
    sources = np.asarray(edges[:, 0], dtype=np.intp)
    if np.any(np.diff(sources) < 0):
        raise ValueError("Graph edges are not grouped by source cell")
    counts = np.bincount(sources, minlength=n_cells)
    row_starts = np.repeat(np.cumsum(counts) - counts, counts)
    rank_in_row = np.arange(sources.size, dtype=np.intp) - row_starts
    keep = rank_in_row < use_k
    return weights[keep], edges[keep]


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
    if not csr_mats:
        raise ValueError("At least one graph is required")
    if len({matrix.shape for matrix in csr_mats}) != 1:
        raise ValueError("ERROR: All graphs do not have the same shape.")
    row_counts = [np.diff(matrix.indptr) for matrix in csr_mats]
    if any(
        len(counts) == 0 or not np.all(counts == counts[0]) for counts in row_counts
    ):
        raise ValueError("ERROR: All graphs must have a regular neighbor count")
    neighbor_counts = {int(counts[0]) for counts in row_counts}
    if len(neighbor_counts) != 1:
        raise ValueError("ERROR: All graphs do not have the same number of edges")
    nk = neighbor_counts.pop()
    if nk < 2:
        raise ValueError("SNN integration requires at least two neighbors per cell")
    n_cells = csr_mats[0].shape[0]
    # Regular rows let each graph be read as one cell-by-neighbor matrix.
    neighbor_rows = [matrix.indices.reshape((n_cells, nk)) for matrix in csr_mats]
    weight_rows = [matrix.data.reshape((n_cells, nk)) for matrix in csr_mats]
    snns = [
        calc_snn(neighbors)
        for neighbors in iter_progress(
            neighbor_rows,
            desc="Identifying SNNs in graphs",
        )
    ]
    columns = np.empty(
        n_cells * nk,
        dtype=np.result_type(*(neighbors.dtype for neighbors in neighbor_rows)),
    )
    data = np.empty(
        n_cells * nk,
        dtype=np.result_type(*(weights.dtype for weights in weight_rows)),
    )
    for row_idx in iter_progress(range(n_cells), desc="Merging graph edges"):
        merged_indices, merged_weights = weight_sort_indices(
            np.hstack([neighbors[row_idx] for neighbors in neighbor_rows]),
            np.hstack([weights[row_idx] for weights in weight_rows]),
            np.hstack(
                [
                    weights[row_idx] + snn[row_idx]
                    for weights, snn in zip(weight_rows, snns, strict=True)
                ]
            ),
            nk,
        )
        start = row_idx * nk
        columns[start : start + nk] = merged_indices
        data[start : start + nk] = merged_weights
    rows = np.repeat(np.arange(n_cells), nk)
    return coo_matrix((data, (rows, columns)), shape=csr_mats[0].shape)


@njit(cache=True, nogil=True)
def csr_symmetry_error(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray | None,
    allow_self_loops: bool,
) -> int:
    """Return the first symmetry defect of a CSR matrix with sorted rows.

    Returns 0 for a symmetric matrix, 1 for a self-loop when
    ``allow_self_loops`` is false, 2 for an entry without its transpose, and 3
    for an entry whose transpose holds another value. Passing ``None`` as
    ``data`` checks only the sparsity structure.
    """
    for source in range(indptr.shape[0] - 1):
        for offset in range(indptr[source], indptr[source + 1]):
            target = indices[offset]
            if target == source:
                if allow_self_loops:
                    continue
                return 1
            lower = indptr[target]
            upper = indptr[target + 1]
            while lower < upper:
                middle = (lower + upper) // 2
                if indices[middle] < source:
                    lower = middle + 1
                else:
                    upper = middle
            if lower >= indptr[target + 1] or indices[lower] != source:
                return 2
            if data is not None:
                if data[lower] != data[offset]:
                    return 3
    return 0
