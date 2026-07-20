import numpy as np
from scipy.sparse import csr_matrix

from ._types import NeighborMetric, ZarrArray

_EDGE_BATCH_ROWS = 100_000


def knn_to_csr_matrix(
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    *,
    use_affinities: bool = False,
) -> csr_matrix:
    """Convert k-nearest neighbors data to a Compressed Sparse Row (CSR) matrix.

    Creates a sparse adjacency matrix representation of a KNN graph. Distances
    can optionally be converted to affinities.

    Args:
        neighbor_indices: Indices matrix from k-nearest neighbors, shape (n_samples, k)
        neighbor_distances: Distances matrix from k-nearest neighbors, shape (n_samples, k)
        use_affinities: Convert distances using ``1 / (log1p(distance) + 1)``.

    Returns:
        scipy.sparse.csr_matrix: Sparse adjacency matrix of shape (n_samples, n_samples)
        where non-zero entries represent neighbor weights
    """
    neighbor_indices = np.asarray(neighbor_indices)
    neighbor_distances = np.asarray(neighbor_distances, dtype=np.float64)
    if neighbor_indices.ndim != 2 or neighbor_distances.ndim != 2:
        raise ValueError("Neighbor indices and distances must be two-dimensional")
    if neighbor_indices.shape != neighbor_distances.shape:
        raise ValueError("Neighbor indices and distances must have matching shapes")
    if not np.issubdtype(neighbor_indices.dtype, np.integer):
        raise TypeError("Neighbor indices must contain integers")
    if not np.all(np.isfinite(neighbor_distances)) or np.any(neighbor_distances < 0):
        raise ValueError("Neighbor distances must be finite and non-negative")

    num_samples, num_neighbors = neighbor_indices.shape
    if num_samples == 0 or num_neighbors == 0:
        raise ValueError("KNN data must contain cells and neighbors")
    if np.any(neighbor_indices < 0) or np.any(neighbor_indices >= num_samples):
        raise IndexError("Neighbor index is outside the graph")

    weights = neighbor_distances
    if use_affinities:
        weights = 1 / (np.log1p(neighbor_distances) + 1)

    indptr = np.arange(
        0,
        num_samples * num_neighbors + 1,
        num_neighbors,
        dtype=np.int64,
    )
    return csr_matrix(
        (weights.reshape(-1), neighbor_indices.reshape(-1), indptr),
        shape=(num_samples, num_samples),
    )


def _validated_cluster_labels(
    cluster_labels: np.ndarray, n_nodes: int
) -> tuple[np.ndarray, int]:
    cluster_labels = np.asarray(cluster_labels)
    if len(cluster_labels) != n_nodes:
        raise ValueError("Cluster labels must have one value per graph node")
    if not np.issubdtype(cluster_labels.dtype, np.integer):
        raise TypeError("Cluster labels must contain integers")

    unique_cluster_ids = np.unique(cluster_labels)
    expected_cluster_ids = np.arange(0, len(unique_cluster_ids))
    if not np.array_equal(unique_cluster_ids, expected_cluster_ids):
        raise ValueError("Cluster labels must be contiguous integers starting at 0")
    return cluster_labels, len(unique_cluster_ids)


def _finalize_cluster_similarity(inter_cluster_weights: np.ndarray) -> np.ndarray:
    inter_cluster_weights = (inter_cluster_weights + inter_cluster_weights.T) / 2
    total_cluster_weights = inter_cluster_weights.sum(axis=1)
    weight_union = (
        total_cluster_weights[:, None]
        + total_cluster_weights[None, :]
        - inter_cluster_weights
    )
    similarity_matrix = np.divide(
        inter_cluster_weights,
        weight_union,
        out=np.zeros_like(inter_cluster_weights),
        where=weight_union > 0,
    )
    np.fill_diagonal(similarity_matrix, 1.0)
    return np.asarray(similarity_matrix)


def calculate_weighted_cluster_similarity(
    knn_graph: csr_matrix, cluster_labels: np.ndarray
) -> np.ndarray:
    """Calculate similarity between clusters based on shared weighted edges.

    Uses a weighted Jaccard index to compute similarities between clusters in a KNN graph.

    Args:
        knn_graph: CSR matrix representing the KNN graph, shape (n_samples, n_samples)
        cluster_labels: Cluster assignments for each node, must be contiguous integers
            starting from 0

    Returns:
        np.ndarray: Symmetric matrix of shape (n_clusters, n_clusters) containing
        pairwise similarities between clusters

    Raises:
        ValueError: If cluster labels are not contiguous integers starting from 0
    """
    knn_graph = knn_graph.tocsr(copy=False)
    if knn_graph.shape[0] != knn_graph.shape[1]:
        raise ValueError("KNN graph must be square")
    if not np.all(np.isfinite(knn_graph.data)) or np.any(knn_graph.data < 0):
        raise ValueError("KNN graph weights must be finite and non-negative")

    cluster_labels, num_clusters = _validated_cluster_labels(
        cluster_labels, knn_graph.shape[0]
    )
    inter_cluster_weights = np.zeros((num_clusters, num_clusters), dtype=np.float64)
    for row_start in range(0, knn_graph.shape[0], _EDGE_BATCH_ROWS):
        row_end = min(row_start + _EDGE_BATCH_ROWS, knn_graph.shape[0])
        edge_start = knn_graph.indptr[row_start]
        edge_end = knn_graph.indptr[row_end]
        source_clusters = np.repeat(
            cluster_labels[row_start:row_end],
            np.diff(knn_graph.indptr[row_start : row_end + 1]),
        )
        target_clusters = cluster_labels[knn_graph.indices[edge_start:edge_end]]
        pair_ids = source_clusters * num_clusters + target_clusters
        weight_counts = np.asarray(
            np.bincount(
                pair_ids,
                weights=knn_graph.data[edge_start:edge_end],
                minlength=num_clusters * num_clusters,
            ),
            dtype=np.float64,
        )
        inter_cluster_weights += weight_counts.reshape(num_clusters, num_clusters)

    return _finalize_cluster_similarity(inter_cluster_weights)


def calculate_knn_cluster_similarity(
    neighbor_indices: np.ndarray | ZarrArray,
    neighbor_distances: np.ndarray | ZarrArray,
    cluster_labels: np.ndarray,
    *,
    batch_rows: int = _EDGE_BATCH_ROWS,
) -> np.ndarray:
    """Stream KNN rows and calculate cluster similarity from distance affinities."""
    if neighbor_indices.ndim != 2 or neighbor_distances.ndim != 2:
        raise ValueError("Neighbor indices and distances must be two-dimensional")
    if neighbor_indices.shape != neighbor_distances.shape:
        raise ValueError("Neighbor indices and distances must have matching shapes")
    if neighbor_indices.shape[1] == 0:
        raise ValueError("KNN data must contain neighbors")
    if batch_rows < 1:
        raise ValueError("batch_rows must be greater than zero")

    n_nodes, n_neighbors = neighbor_indices.shape
    cluster_labels, num_clusters = _validated_cluster_labels(cluster_labels, n_nodes)
    inter_cluster_weights = np.zeros((num_clusters, num_clusters), dtype=np.float64)
    for row_start in range(0, n_nodes, batch_rows):
        row_end = min(row_start + batch_rows, n_nodes)
        index_block = np.asarray(neighbor_indices[row_start:row_end])
        distance_block = np.asarray(
            neighbor_distances[row_start:row_end], dtype=np.float64
        )
        if not np.issubdtype(index_block.dtype, np.integer):
            raise TypeError("Neighbor indices must contain integers")
        if np.any(index_block < 0) or np.any(index_block >= n_nodes):
            raise IndexError("Neighbor index is outside the graph")
        if not np.all(np.isfinite(distance_block)) or np.any(distance_block < 0):
            raise ValueError("Neighbor distances must be finite and non-negative")

        affinities = 1 / (np.log1p(distance_block) + 1)
        source_clusters = np.repeat(cluster_labels[row_start:row_end], n_neighbors)
        target_clusters = cluster_labels[index_block.reshape(-1)]
        pair_ids = source_clusters * num_clusters + target_clusters
        weight_counts = np.asarray(
            np.bincount(
                pair_ids,
                weights=affinities.reshape(-1),
                minlength=num_clusters * num_clusters,
            ),
            dtype=np.float64,
        )
        inter_cluster_weights += weight_counts.reshape(num_clusters, num_clusters)

    return _finalize_cluster_similarity(inter_cluster_weights)


def calculate_top_k_neighbor_distances(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    k: int,
    metric: NeighborMetric = "l2",
) -> np.ndarray:
    """Calculate distances to k nearest neighbors between two sets of points.

    For each point in matrix_a, finds the k nearest neighbors in matrix_b
    and returns their distances.

    Args:
        matrix_a: First set of points, shape (m, d)
        matrix_b: Second set of points, shape (n, d)
        k: Number of nearest neighbors to find

    Returns:
        np.ndarray: Matrix of shape (m, k) containing the distances to the
        k nearest neighbors in matrix_b for each point in matrix_a

    Raises:
        ValueError: If the inputs are empty, incompatible, or k is invalid
    """
    matrix_a = np.asarray(matrix_a, dtype=np.float64)
    matrix_b = np.asarray(matrix_b, dtype=np.float64)
    if matrix_a.ndim != 2 or matrix_b.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if matrix_a.shape[1] != matrix_b.shape[1]:
        raise ValueError("Matrices must have the same number of features")
    if matrix_a.shape[0] == 0 or matrix_b.shape[0] == 0:
        raise ValueError("Input matrices must contain at least one point")
    if not np.all(np.isfinite(matrix_a)) or not np.all(np.isfinite(matrix_b)):
        raise ValueError("Input matrices must contain finite values")
    if k < 1:
        raise ValueError("k must be greater than zero")

    # Ensure k is not larger than the number of points in matrix_b
    k = min(k, matrix_b.shape[0])

    products = np.dot(matrix_a, matrix_b.T)
    if metric == "l2":
        a_squared = np.sum(np.square(matrix_a), axis=1, keepdims=True)
        b_squared = np.sum(np.square(matrix_b), axis=1)
        distances = np.sqrt(np.maximum(a_squared + b_squared - 2 * products, 0))
    elif metric == "cosine":
        norms = np.outer(
            np.linalg.norm(matrix_a, axis=1),
            np.linalg.norm(matrix_b, axis=1),
        )
        similarities = np.divide(
            products,
            norms,
            out=np.zeros_like(products),
            where=norms > 0,
        )
        distances = np.clip(1 - similarities, 0, 2)
    elif metric == "ip":
        distances = np.maximum(1 - products, 0)
    else:
        raise ValueError(f"Unsupported neighbor metric: {metric}")

    # Find the k smallest distances for each point in matrix_a
    return np.partition(distances, k - 1, axis=1)[:, :k]
