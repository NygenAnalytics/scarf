"""
Methods and classes for evaluation
"""

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import zarr

zarrArrayType = zarr.Array

from .ann import AnnStream
from .datastore.datastore import DataStore
from .utils import (
    logger,
    tqdmbar,
)


# LISI - The Local Inverse Simpson Index
def compute_lisi(
    distances: zarrArrayType,
    indices: zarrArrayType,
    metadata: pd.DataFrame,
    label_colnames: Iterable[str],
    perplexity: float = 30,
) -> np.ndarray:
    """Compute the Local Inverse Simpson Index (LISI) for each column in metadata.

    LISI measures how well mixed different groups of cells are in the neighborhood of each cell.
    Higher values indicate better mixing of different groups.

    Args:
        distances: Pre-computed distances between cells, stored in zarr array format
        indices: Pre-computed nearest neighbor indices, stored in zarr array format
        metadata: DataFrame containing categorical labels for each cell
        label_colnames: Column names in metadata to compute LISI for
        perplexity: Parameter controlling the effective number of neighbors (default: 30)

    Returns:
        np.ndarray: Matrix of LISI scores with shape (n_cells, n_labels)
        Each column corresponds to LISI scores for one label column in metadata

    Example:
        For metadata with a 'batch' column having 3 categories:
        - LISI ≈ 3: Cell has neighbors from all 3 batches (well mixed)
        - LISI ≈ 1: Cell has neighbors from only 1 batch (poorly mixed)

    References:
        Korsunsky et al. 2019 doi: 10.1038/s41592-019-0619-0
    """

    n_cells = metadata.shape[0]
    n_labels = len(label_colnames)
    # Don't count yourself
    indices = indices[:, 1:]
    distances = distances[:, 1:]
    lisi_df = np.zeros((n_cells, n_labels))
    for i, label in enumerate(label_colnames):
        logger.info(f"Computing LISI for {label}")
        labels = pd.Categorical(metadata[label])
        simpson = compute_simpson(
            distances.T, indices.T, labels, perplexity
        )
        lisi_df[:, i] = 1 / simpson
    return lisi_df


def compute_simpson(
    distances: np.ndarray,
    indices: np.ndarray,
    labels: pd.Categorical,
    perplexity: float,
    tol: float = 1e-5,
) -> np.ndarray:
    """Compute Simpson's diversity index with Gaussian kernel weighting.

    This function implements the core calculation for LISI, computing a diversity score
    based on the distribution of categories in each cell's neighborhood.

    Args:
        distances: Distance matrix between points, shape (n_neighbors, n_points)
        indices: Index matrix for nearest neighbors, shape (n_neighbors, n_points)
        labels: Categorical labels for each point
        perplexity: Target perplexity for Gaussian kernel
        tol: Convergence tolerance for perplexity calibration (default: 1e-5)

    Returns:
        np.ndarray: Array of Simpson's diversity indices, one per point
    """
    n = distances.shape[1]
    P = np.zeros(distances.shape[0])
    simpson = np.zeros(n)
    logU = np.log(perplexity)
    # Loop through each cell.
    for i in tqdmbar(range(n), desc="Computing Simpson's Diversity Index"):
        beta = 1
        betamin = -np.inf
        betamax = np.inf
        # Compute Hdiff
        P = np.exp(-distances[:, i] * beta)
        P_sum = np.sum(P)
        if P_sum == 0:
            H = 0
            P = np.zeros(distances.shape[0])
        else:
            H = np.log(P_sum) + beta * np.sum(distances[:, i] * P) / P_sum
            P = P / P_sum
        Hdiff = H - logU
        n_tries = 50
        for t in range(n_tries):
            # Stop when we reach the tolerance
            if abs(Hdiff) < tol:
                break
            # Update beta
            if Hdiff > 0:
                betamin = beta
                if not np.isfinite(betamax):
                    beta *= 2
                else:
                    beta = (beta + betamax) / 2
            else:
                betamax = beta
                if not np.isfinite(betamin):
                    beta /= 2
                else:
                    beta = (beta + betamin) / 2
            # Compute Hdiff
            P = np.exp(-distances[:, i] * beta)
            P_sum = np.sum(P)
            if P_sum == 0:
                H = 0
                P = np.zeros(distances.shape[0])
            else:
                H = np.log(P_sum) + beta * np.sum(distances[:, i] * P) / P_sum
                P = P / P_sum
            Hdiff = H - logU
        # distancesefault value
        if H == 0:
            simpson[i] = -1
        # Simpson's index
        for label_category in labels.categories:
            ix = indices[:, i]
            q = labels[ix] == label_category
            if np.any(q):
                P_sum = np.sum(P[q])
                simpson[i] += P_sum * P_sum
    return simpson


# SILHOUETTE SCORE - The Silhouette Score
def knn_to_csr_matrix(
    neighbor_indices: np.ndarray, neighbor_distances: np.ndarray
) -> csr_matrix:
    """Convert k-nearest neighbors data to a Compressed Sparse Row (CSR) matrix.

    Creates a sparse adjacency matrix representation of the k-nearest neighbors graph
    where edge weights are the distances between points.

    Args:
        neighbor_indices: Indices matrix from k-nearest neighbors, shape (n_samples, k)
        neighbor_distances: Distances matrix from k-nearest neighbors, shape (n_samples, k)

    Returns:
        scipy.sparse.csr_matrix: Sparse adjacency matrix of shape (n_samples, n_samples)
        where non-zero entries represent distances between neighboring points
    """
    num_samples, num_neighbors = neighbor_indices.shape
    row_indices = np.repeat(np.arange(num_samples), num_neighbors)
    return csr_matrix(
        (neighbor_distances[:].flatten(), (row_indices, neighbor_indices[:].flatten())),
        shape=(num_samples, num_samples),
    )


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
        AssertionError: If cluster labels are not contiguous integers starting from 0
    """
    unique_cluster_ids = np.unique(cluster_labels)
    expected_cluster_ids = np.arange(0, len(unique_cluster_ids))
    assert np.array_equal(unique_cluster_ids, expected_cluster_ids), (
        "Cluster labels must be contiguous integers starting at 1"
    )

    num_clusters = len(unique_cluster_ids)
    inter_cluster_weights = np.zeros((num_clusters, num_clusters))

    for cluster_id in unique_cluster_ids:
        nodes_in_cluster = np.where(cluster_labels == cluster_id)[0]
        neighbor_cluster_labels = cluster_labels[knn_graph[nodes_in_cluster].indices]
        neighbor_edge_weights = knn_graph[nodes_in_cluster].data

        for neighbor_cluster, edge_weight in zip(
            neighbor_cluster_labels, neighbor_edge_weights
        ):
            inter_cluster_weights[cluster_id, neighbor_cluster] += edge_weight

    # assert inter_cluster_weights.sum() == knn_graph.data.sum()

    # Ensure symmetry
    inter_cluster_weights = (inter_cluster_weights + inter_cluster_weights.T) / 2

    # Calculate total weights for each cluster
    total_cluster_weights = np.array(
        [inter_cluster_weights[i - 1].sum() for i in unique_cluster_ids]
    )

    # Calculate similarity using weighted Jaccard index
    similarity_matrix = np.zeros((num_clusters, num_clusters))

    for i in range(num_clusters):
        for j in range(i, num_clusters):
            weight_union = (
                total_cluster_weights[i]
                + total_cluster_weights[j]
                - inter_cluster_weights[i, j]
            )
            if weight_union > 0:
                similarity = inter_cluster_weights[i, j] / weight_union
                similarity_matrix[i, j] = similarity_matrix[j, i] = similarity

    # Set diagonal to 1 (self-similarity)
    # np.fill_diagonal(similarity_matrix, 1.0)

    return similarity_matrix


def calculate_top_k_neighbor_distances(
    matrix_a: np.ndarray, matrix_b: np.ndarray, k: int
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
        AssertionError: If matrices don't have the same number of features
    """
    # Check if the matrices have the same number of features (d)
    assert matrix_a.shape[1] == matrix_b.shape[1], (
        "Matrices must have the same number of features"
    )

    # Ensure k is not larger than the number of points in matrix_b
    k = min(k, matrix_b.shape[0])

    # Calculate squared Euclidean distances
    a_squared = np.sum(np.square(matrix_a), axis=1, keepdims=True)
    b_squared = np.sum(np.square(matrix_b), axis=1)

    # Use broadcasting to compute pairwise distances
    distances = a_squared + b_squared - 2 * np.dot(matrix_a, matrix_b.T)

    # Use np.maximum to avoid small negative values due to floating point errors
    distances = np.maximum(distances, 0)

    # Find the k smallest distances for each point in matrix_a
    top_k_distances = np.partition(distances, k, axis=1)[:, :k]

    # Calculate the square root to get Euclidean distances
    return np.sqrt(top_k_distances)


def process_cluster(
    cluster_cells: np.ndarray,
    hvg_data: np.ndarray | zarrArrayType,
    ann_obj: AnnStream,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Process a cluster of cells to prepare data for silhouette scoring.

    Randomly splits cluster cells into two groups and applies dimensionality reduction.

    Args:
        cluster_cells: Indices of cells belonging to the cluster
        hvg_data: Expression data for highly variable genes
        ann_obj: Object containing dimensionality reduction method
        k: Number of cells to sample from cluster

    Returns:
        tuple[np.ndarray, np.ndarray]: Two arrays containing reduced data for
        different subsets of cells from the cluster
    """
    np.random.shuffle(cluster_cells)
    data_cells = np.array(
        [ann_obj.reducer(hvg_data[i]) for i in sorted(cluster_cells[:k])]
    )
    data_cells_2 = np.array(
        [ann_obj.reducer(hvg_data[i]) for i in sorted(cluster_cells[k : 2 * k])]
    )
    return data_cells, data_cells_2


def silhouette_scoring(
    ds: DataStore,
    ann_obj: AnnStream,
    graph: csr_matrix,
    hvg_data: np.ndarray | zarrArrayType,
    assay_type: str,
    res_label: str,
) -> np.ndarray | None:
    """Compute modified silhouette scores for clusters in single-cell data.

    This implementation differs from the standard silhouette score by using
    a graph-based approach and comparing clusters to their nearest neighbors.

    Args:
        ds: DataStore object containing cell metadata
        ann_obj: Object containing dimensionality reduction method
        graph: CSR matrix representing the KNN graph
        hvg_data: Expression data for highly variable genes
        assay_type: Type of assay (e.g., 'RNA', 'ATAC')
        res_label: Label for clustering resolution

    Returns:
        np.ndarray | None: Array of silhouette scores for each cluster,
        or None if cluster labels are not found

    Notes:
        Scores are calculated using a sampling approach for efficiency.
        NaN values indicate clusters that couldn't be scored due to size constraints.
    """
    try:
        clusters = ds.cells.fetch(f"{assay_type}_{res_label}") - 1  # RNA_{res_label}
    except KeyError:
        logger.error(f"Cluster labels not found for {assay_type}_{res_label}")
        return None

    cluster_similarity = calculate_weighted_cluster_similarity(graph, clusters)

    k = 11
    score = []

    for n, i in enumerate(cluster_similarity):
        this_cluster_cells = np.where(clusters == n)[0]
        if len(this_cluster_cells) < 2 * k:
            k = int(len(this_cluster_cells) / 2)
            logger.warning(
                f"Warning: Cluster {n} has fewer than 22 cells. Will adjust k to {k} instead"
            )

    for n, i in tqdmbar(
        enumerate(cluster_similarity),
        total=len(cluster_similarity),
        desc="Calculating Silhouette Scores",
    ):
        this_cluster_cells = np.where(clusters == n)[0]
        np.random.shuffle(this_cluster_cells)
        data_this_cells, data_this_cells_2 = process_cluster(
            # n,
            this_cluster_cells,
            hvg_data,
            ann_obj,
            k,
        )

        if data_this_cells.size == 0 or data_this_cells_2.size == 0:
            logger.warning(f"Warning: Reduced data for cluster {n} is empty. Skipping.")
            score.append(np.nan)
            continue

        k_neighbors = min(k - 1, data_this_cells_2.shape[0] - 1)

        if k_neighbors < 1:
            logger.warning(
                f"Warning: Not enough points in cluster {n} for comparison. Skipping."
            )
            score.append(np.nan)
            continue

        self_dist = calculate_top_k_neighbor_distances(
            data_this_cells, data_this_cells_2, k - 1
        ).mean()

        nearest_cluster = np.argsort(i)[-1]
        nearest_cluster_cells = np.where(clusters == nearest_cluster)[0]
        np.random.shuffle(nearest_cluster_cells)

        if len(nearest_cluster_cells) < k:
            logger.warning(
                f"Warning: Nearest cluster {nearest_cluster} has fewer than {k} cells. Skipping."
            )
            score.append(np.nan)
            continue

        data_nearest_cells, _ = process_cluster(
            # nearest_cluster,
            nearest_cluster_cells,
            hvg_data,
            ann_obj,
            k,
        )

        if data_nearest_cells.size == 0:
            logger.warning(
                f"Warning: Reduced data for nearest cluster {nearest_cluster} is empty. Skipping."
            )
            score.append(np.nan)
            continue

        other_dist = calculate_top_k_neighbor_distances(
            data_this_cells, data_nearest_cells, k - 1
        ).mean()

        score.append((other_dist - self_dist) / max(self_dist, other_dist))

    return np.array(score)


def integration_score(
    batch_labels: Sequence[np.ndarray], metric: str = "ari"
) -> float | None:
    """Calculate integration score between two sets of batch labels.

    Args:
        batch_labels: Sequence containing two arrays of batch labels to compare
        metric: Metric to use for comparison, one of:
            - 'ari': Adjusted Rand Index
            - 'nmi': Normalized Mutual Information

    Returns:
        float | None: Integration score between 0 and 1, or None if metric
        is not recognized

    Notes:
        Higher scores indicate better agreement between batch labels,
        suggesting more effective batch integration.
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    if metric == "ari":
        return adjusted_rand_score(batch_labels[0], batch_labels[1])
    elif metric == "nmi":
        return normalized_mutual_info_score(batch_labels[0], batch_labels[1])
    else:
        logger.error(
            f"Metric {metric} not recognized. Please choose from 'ari' or 'nmi'."
        )
        return None
