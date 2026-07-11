"""
Methods and classes for evaluation
"""

from collections.abc import Iterable, Sequence
from typing import Literal, cast

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import zarr

from .ann import AnnStream
from .chunked import ChunkedArray
from .datastore.datastore import DataStore
from .utils import (
    logger,
    tqdmbar,
)

type ZarrArray = zarr.Array
type MatrixData = np.ndarray | ZarrArray | ChunkedArray
type NeighborMetric = Literal["l2", "cosine", "ip"]

_LISI_BATCH_SIZE = 10_000
_EDGE_BATCH_ROWS = 100_000


# LISI - The Local Inverse Simpson Index
def _effective_perplexity(perplexity: float, n_neighbors: int) -> float:
    if not np.isfinite(perplexity) or perplexity < 1:
        raise ValueError("Perplexity must be a finite value greater than or equal to 1")
    if n_neighbors < 3:
        raise ValueError("LISI requires at least three neighbors per cell")

    max_perplexity = n_neighbors / 3
    if perplexity > max_perplexity:
        logger.warning(
            f"Perplexity {perplexity:g} requires at least "
            f"{int(np.ceil(3 * perplexity))} neighbors, but the graph has "
            f"{n_neighbors}. Using perplexity {max_perplexity:g}."
        )
        return max_perplexity
    return perplexity


def _neighbor_probabilities(
    distances: np.ndarray,
    perplexity: float,
    tol: float,
    max_iter: int = 50,
) -> np.ndarray:
    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[1] == 0:
        raise ValueError("Distances must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(distances)) or np.any(distances < 0):
        raise ValueError("Distances must contain finite, non-negative values")

    centered = distances - distances.min(axis=1, keepdims=True)
    n_points = distances.shape[0]
    beta = np.ones(n_points, dtype=np.float64)
    beta_min = np.full(n_points, -np.inf, dtype=np.float64)
    beta_max = np.full(n_points, np.inf, dtype=np.float64)
    target_entropy = np.log(perplexity)

    for _ in range(max_iter):
        with np.errstate(over="ignore", under="ignore"):
            weights = np.exp(-centered * beta[:, None])
        weight_sums = weights.sum(axis=1)
        entropy = np.log(weight_sums) + (
            beta * np.sum(centered * weights, axis=1) / weight_sums
        )
        entropy_diff = entropy - target_entropy
        active = np.abs(entropy_diff) >= tol
        if not np.any(active):
            break

        too_diffuse = active & (entropy_diff > 0)
        too_concentrated = active & ~too_diffuse

        beta_min[too_diffuse] = beta[too_diffuse]
        bounded_above = too_diffuse & np.isfinite(beta_max)
        beta[bounded_above] = (beta[bounded_above] + beta_max[bounded_above]) / 2
        beta[too_diffuse & ~np.isfinite(beta_max)] *= 2

        beta_max[too_concentrated] = beta[too_concentrated]
        bounded_below = too_concentrated & np.isfinite(beta_min)
        beta[bounded_below] = (beta[bounded_below] + beta_min[bounded_below]) / 2
        beta[too_concentrated & ~np.isfinite(beta_min)] /= 2

    with np.errstate(over="ignore", under="ignore"):
        probabilities = np.exp(-centered * beta[:, None])
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return np.asarray(probabilities)


def _simpson_from_probabilities(
    probabilities: np.ndarray,
    indices: np.ndarray,
    label_codes: np.ndarray,
    n_categories: int,
) -> np.ndarray:
    indices = np.asarray(indices)
    if indices.shape != probabilities.shape:
        raise ValueError("Neighbor indices and probabilities must have matching shapes")
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("Neighbor indices must contain integers")
    if np.any(indices < 0) or np.any(indices >= len(label_codes)):
        raise IndexError("Neighbor index is outside the label array")

    neighbor_codes = label_codes[indices]
    simpson = np.zeros(probabilities.shape[0], dtype=np.float64)
    if n_categories <= probabilities.shape[1]:
        for category in range(n_categories):
            category_mass = np.sum(probabilities * (neighbor_codes == category), axis=1)
            simpson += np.square(category_mass)
    else:
        for neighbor in range(probabilities.shape[1]):
            same_category = neighbor_codes == neighbor_codes[:, neighbor, None]
            category_mass = np.sum(probabilities * same_category, axis=1)
            simpson += probabilities[:, neighbor] * category_mass

    if not np.all(np.isfinite(simpson)) or np.any(simpson <= 0):
        raise FloatingPointError("Could not compute finite Simpson indices")
    return np.clip(simpson, np.finfo(np.float64).tiny, 1.0)


def compute_lisi(
    distances: np.ndarray | ZarrArray,
    indices: np.ndarray | ZarrArray,
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

    if distances.ndim != 2 or indices.ndim != 2:
        raise ValueError("KNN distances and indices must be two-dimensional")
    if distances.shape != indices.shape:
        raise ValueError("KNN distances and indices must have matching shapes")

    n_cells, n_neighbors = distances.shape
    if metadata.shape[0] != n_cells:
        raise ValueError(
            "Metadata rows must match the number of cells in the KNN graph"
        )

    label_cols = list(label_colnames)
    n_labels = len(label_cols)
    if n_labels == 0:
        return np.empty((n_cells, 0), dtype=np.float64)

    effective_perplexity = _effective_perplexity(perplexity, n_neighbors)
    categoricals: list[pd.Categorical] = []
    for label in label_cols:
        categorical = pd.Categorical(metadata[label])
        if np.any(categorical.codes < 0):
            raise ValueError(f"Label column {label!r} contains missing values")
        categoricals.append(categorical)
        logger.info(f"Computing LISI for {label}")

    chunk_rows = _LISI_BATCH_SIZE
    if isinstance(distances, zarr.Array):
        chunk_rows = min(chunk_rows, int(distances.chunks[0]))

    lisi_df = np.empty((n_cells, n_labels), dtype=np.float64)
    starts = range(0, n_cells, chunk_rows)
    total = int(np.ceil(n_cells / chunk_rows))
    for start in tqdmbar(starts, total=total, desc="Computing LISI"):
        end = min(start + chunk_rows, n_cells)
        distance_block = np.asarray(distances[start:end])
        index_block = np.asarray(indices[start:end])
        probabilities = _neighbor_probabilities(
            distance_block, effective_perplexity, tol=1e-5
        )
        for label_index, labels in enumerate(categoricals):
            simpson = _simpson_from_probabilities(
                probabilities,
                index_block,
                np.asarray(labels.codes),
                len(labels.categories),
            )
            lisi_df[start:end, label_index] = 1 / simpson
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
    distances = np.asarray(distances)
    indices = np.asarray(indices)
    if distances.ndim != 2 or indices.ndim != 2:
        raise ValueError("Distances and indices must be two-dimensional")
    if distances.shape != indices.shape:
        raise ValueError("Distances and indices must have matching shapes")
    if np.any(labels.codes < 0):
        raise ValueError("Labels contain missing values")

    n_neighbors = distances.shape[0]
    effective_perplexity = _effective_perplexity(perplexity, n_neighbors)
    probabilities = _neighbor_probabilities(distances.T, effective_perplexity, tol=tol)
    return _simpson_from_probabilities(
        probabilities,
        indices.T,
        np.asarray(labels.codes),
        len(labels.categories),
    )


# SILHOUETTE SCORE - The Silhouette Score
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


def _read_rows(data: MatrixData, row_indices: np.ndarray) -> np.ndarray:
    row_indices = np.asarray(row_indices, dtype=np.int64)
    if isinstance(data, ChunkedArray):
        return data[row_indices].compute()
    if isinstance(data, zarr.Array):
        return np.asarray(data.get_orthogonal_selection((row_indices, slice(None))))
    return np.asarray(data[row_indices])


def _embed_rows(
    row_indices: np.ndarray,
    data: MatrixData,
    ann_obj: AnnStream,
    *,
    data_is_reduced: bool,
) -> np.ndarray:
    rows = _read_rows(data, np.sort(row_indices))
    if data_is_reduced:
        return np.asarray(rows)
    return np.asarray(ann_obj.reducer(rows))


def _sample_cluster_embeddings(
    cluster_cells: np.ndarray,
    data: MatrixData,
    ann_obj: AnnStream,
    count: int,
    rng: np.random.Generator,
    *,
    data_is_reduced: bool,
) -> np.ndarray:
    if count < 1 or count > len(cluster_cells):
        raise ValueError("Sample count must fit within the cluster")
    sampled = rng.choice(cluster_cells, size=count, replace=False)
    return _embed_rows(sampled, data, ann_obj, data_is_reduced=data_is_reduced)


def process_cluster(
    cluster_cells: np.ndarray,
    hvg_data: MatrixData,
    ann_obj: AnnStream,
    k: int,
    *,
    rng: np.random.Generator | None = None,
    data_is_reduced: bool = False,
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
    if k < 1 or len(cluster_cells) < 2 * k:
        raise ValueError("A cluster must contain at least 2 * k cells")
    if rng is None:
        rng = np.random.default_rng(4444)

    selected = rng.choice(cluster_cells, size=2 * k, replace=False)
    data_cells = _embed_rows(
        selected[:k],
        hvg_data,
        ann_obj,
        data_is_reduced=data_is_reduced,
    )
    data_cells_2 = _embed_rows(
        selected[k:],
        hvg_data,
        ann_obj,
        data_is_reduced=data_is_reduced,
    )
    return data_cells, data_cells_2


def silhouette_scoring(
    ds: DataStore,
    ann_obj: AnnStream,
    graph: csr_matrix | None,
    hvg_data: MatrixData,
    assay_type: str,
    res_label: str,
    *,
    cell_key: str = "I",
    random_seed: int = 4444,
    sample_size: int = 11,
    data_is_reduced: bool = False,
    distance_metric: NeighborMetric | None = None,
    neighbor_indices: np.ndarray | ZarrArray | None = None,
    neighbor_distances: np.ndarray | ZarrArray | None = None,
) -> np.ndarray | None:
    """Compute modified silhouette scores for clusters in single-cell data.

    This implementation differs from the standard silhouette score by using
    a graph-based approach and comparing clusters to their nearest neighbors.

    Args:
        ds: DataStore object containing cell metadata
        ann_obj: Object containing dimensionality reduction method
        graph: Optional CSR matrix representing the weighted KNN graph
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
    if sample_size < 1:
        raise ValueError("sample_size must be greater than zero")

    if res_label in ds.cells.columns:
        cluster_column = res_label
    else:
        prefix = assay_type if cell_key == "I" else f"{assay_type}_{cell_key}"
        cluster_column = f"{prefix}_{res_label}"

    try:
        raw_clusters = ds.cells.fetch(cluster_column, key=cell_key)
    except KeyError:
        logger.error(f"Cluster labels not found for {cluster_column}")
        return None

    categorical = pd.Categorical(raw_clusters)
    if np.any(categorical.codes < 0):
        raise ValueError(f"Cluster column {cluster_column!r} contains missing values")
    clusters = np.asarray(categorical.codes, dtype=np.int64)
    if hvg_data.shape[0] != len(clusters):
        raise ValueError(
            "Embedding data and cluster labels must contain the same cells"
        )

    if graph is not None:
        if graph.shape != (len(clusters), len(clusters)):
            raise ValueError("KNN graph and cluster labels must contain the same cells")
        cluster_similarity = calculate_weighted_cluster_similarity(graph, clusters)
    elif neighbor_indices is not None and neighbor_distances is not None:
        cluster_similarity = calculate_knn_cluster_similarity(
            neighbor_indices,
            neighbor_distances,
            clusters,
        )
    else:
        raise ValueError("Provide a KNN graph or neighbor indices and distances")
    num_clusters = len(categorical.categories)
    if num_clusters < 2:
        logger.warning("Silhouette scoring requires at least two clusters")
        return np.full(num_clusters, np.nan)

    order = np.argsort(clusters, kind="stable")
    counts = np.bincount(clusters, minlength=num_clusters)
    boundaries = np.cumsum(counts)
    starts = np.concatenate(([0], boundaries[:-1]))
    cluster_cells = [order[start:end] for start, end in zip(starts, boundaries)]

    metric = distance_metric or cast(NeighborMetric, ann_obj.annMetric)
    if metric not in {"l2", "cosine", "ip"}:
        raise ValueError(f"Unsupported neighbor metric: {metric}")

    rng = np.random.default_rng(random_seed)
    score: list[float] = []
    for cluster_id, similarities in tqdmbar(
        enumerate(cluster_similarity),
        total=len(cluster_similarity),
        desc="Calculating Silhouette Scores",
    ):
        this_cluster_cells = cluster_cells[cluster_id]
        k = min(sample_size, len(this_cluster_cells) // 2)
        if k < 1:
            logger.warning(
                f"Cluster {categorical.categories[cluster_id]!r} has fewer than two cells"
            )
            score.append(np.nan)
            continue

        data_this_cells, data_this_cells_2 = process_cluster(
            this_cluster_cells,
            hvg_data,
            ann_obj,
            k,
            rng=rng,
            data_is_reduced=data_is_reduced,
        )

        self_dist = calculate_top_k_neighbor_distances(
            data_this_cells,
            data_this_cells_2,
            min(k, len(data_this_cells_2)),
            metric=metric,
        ).mean()

        other_similarities = similarities.copy()
        other_similarities[cluster_id] = -np.inf
        nearest_cluster = int(np.argmax(other_similarities))
        nearest_cluster_cells = cluster_cells[nearest_cluster]
        nearest_sample_size = min(k, len(nearest_cluster_cells))
        data_nearest_cells = _sample_cluster_embeddings(
            nearest_cluster_cells,
            hvg_data,
            ann_obj,
            nearest_sample_size,
            rng,
            data_is_reduced=data_is_reduced,
        )

        other_dist = calculate_top_k_neighbor_distances(
            data_this_cells,
            data_nearest_cells,
            min(k, len(data_nearest_cells)),
            metric=metric,
        ).mean()

        denominator = max(self_dist, other_dist)
        if denominator <= np.finfo(np.float64).eps:
            score.append(0.0)
        else:
            score.append(float((other_dist - self_dist) / denominator))

    return np.asarray(score)


def label_concordance_score(
    label_sets: Sequence[np.ndarray],
    metric: Literal["ari", "nmi"] = "ari",
) -> float:
    """Compare two label partitions using ARI or NMI.

    Args:
        label_sets: Two arrays of labels to compare.
        metric: Either ``"ari"`` or ``"nmi"``.

    Returns:
        Label agreement. ARI ranges from -1 to 1 and NMI from 0 to 1.
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    if len(label_sets) != 2:
        raise ValueError("Exactly two label arrays are required")

    first = np.asarray(label_sets[0])
    second = np.asarray(label_sets[1])
    if first.ndim != 1 or second.ndim != 1:
        raise ValueError("Label arrays must be one-dimensional")
    if len(first) != len(second):
        raise ValueError("Label arrays must have matching lengths")
    if pd.isna(first).any() or pd.isna(second).any():
        raise ValueError("Label arrays must not contain missing values")

    if metric == "ari":
        return float(adjusted_rand_score(first, second))
    if metric == "nmi":
        return float(normalized_mutual_info_score(first, second))
    raise ValueError(f"Metric {metric!r} is not one of 'ari' or 'nmi'")


def integration_score(
    batch_labels: Sequence[np.ndarray],
    metric: Literal["ari", "nmi"] = "ari",
) -> float:
    """Backward-compatible name for :func:`label_concordance_score`."""
    return label_concordance_score(batch_labels, metric)


def lisi_batch_mixing_score(
    lisi_scores: np.ndarray,
    batch_labels: Sequence[object] | np.ndarray,
) -> float:
    """Normalize mean batch LISI against the dataset's batch proportions.

    Raw batch LISI depends on how many batches are present and how unevenly
    cells are split between them, so its values are hard to compare across
    datasets. This score rescales the mean batch LISI onto a fixed range by
    dividing the observed neighborhood mixing by the mixing that perfectly
    integrated data would reach. The reference point is the inverse Simpson
    index of the global batch proportions, which is the LISI a neighborhood
    would show if it mirrored the whole dataset.

    Args:
        lisi_scores: Per-cell batch LISI values, typically the array returned
            by :func:`compute_lisi` for a batch label.
        batch_labels: Batch assignment for each cell, aligned with
            ``lisi_scores``.

    Returns:
        A value in ``[0, 1]``. Scores near 1 mean neighborhoods mix batches as
        well as the global composition allows, and scores near 0 mean batches
        stay separated.

    Raises:
        ValueError: If the inputs are misaligned, contain non-finite scores or
            missing labels, or describe fewer than two batches.
    """
    scores = np.asarray(lisi_scores, dtype=np.float64)
    labels = pd.Categorical(batch_labels)
    if scores.ndim != 1 or len(scores) != len(labels):
        raise ValueError("LISI scores and batch labels must be aligned vectors")
    if not np.all(np.isfinite(scores)):
        raise ValueError("LISI scores must contain finite values")
    if np.any(labels.codes < 0):
        raise ValueError("Batch labels must not contain missing values")
    if len(labels.categories) < 2:
        raise ValueError("Batch mixing requires at least two batches")

    counts = np.bincount(labels.codes, minlength=len(labels.categories))
    proportions = counts / counts.sum()
    ideal_lisi = 1 / np.square(proportions).sum()
    normalized = (scores.mean() - 1) / (ideal_lisi - 1)
    return float(np.clip(normalized, 0, 1))
