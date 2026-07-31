from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from ..neighbors.stream import AnnStream
from ..utils.logging import logger
from ..utils.progress import iter_progress
from ._rows import read_matrix_rows
from ._types import MatrixData, NeighborMetric, ZarrArray
from .graph import (
    calculate_knn_cluster_similarity,
    calculate_top_k_neighbor_distances,
    calculate_weighted_cluster_similarity,
)

if TYPE_CHECKING:
    from ..datastore.datastore import DataStore


def _embed_rows(
    row_indices: np.ndarray,
    data: MatrixData,
    ann_obj: AnnStream,
    *,
    data_is_reduced: bool,
) -> np.ndarray:
    rows = read_matrix_rows(data, np.sort(row_indices))
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
    ds: "DataStore",
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
    for cluster_id, similarities in iter_progress(
        enumerate(cluster_similarity),
        total=len(cluster_similarity),
        desc="Calculating silhouette scores",
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
