from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
import zarr

from ..utils.logging import logger
from ..utils.progress import iter_progress
from ._types import ZarrArray

_LISI_BATCH_SIZE = 10_000


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
        logger.debug(f"Preparing LISI labels from {label}")

    chunk_rows = _LISI_BATCH_SIZE
    if isinstance(distances, zarr.Array):
        chunk_rows = min(chunk_rows, int(distances.chunks[0]))

    lisi_df = np.empty((n_cells, n_labels), dtype=np.float64)
    starts = range(0, n_cells, chunk_rows)
    total = int(np.ceil(n_cells / chunk_rows))
    for start in iter_progress(starts, total=total, desc="Computing LISI"):
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


def _lisi_knn_summary(
    distances: np.ndarray | ZarrArray,
    indices: np.ndarray | ZarrArray,
    labels: Sequence[object] | np.ndarray,
    perplexity: float | None,
    scale: bool,
    *,
    invert: bool,
    label_name: str,
) -> float:
    if distances.ndim != 2 or indices.ndim != 2:
        raise ValueError("KNN distances and indices must be two-dimensional")
    if distances.shape != indices.shape:
        raise ValueError("KNN distances and indices must have matching shapes")

    n_cells, n_neighbors = distances.shape
    if n_neighbors < 3:
        raise ValueError("LISI requires at least three neighbors per cell")

    categorical = pd.Categorical(labels)
    if len(categorical) != n_cells:
        raise ValueError(f"{label_name} labels must match the number of cells")
    if np.any(categorical.codes < 0):
        raise ValueError(f"{label_name} labels must not contain missing values")

    n_categories = int(np.unique(categorical.codes).size)
    if n_categories < 2:
        raise ValueError(f"{label_name} LISI requires at least two categories")

    resolved_perplexity = (
        float(np.floor(n_neighbors / 3)) if perplexity is None else perplexity
    )
    metadata = pd.DataFrame({"labels": categorical})
    per_cell = compute_lisi(
        distances,
        indices,
        metadata,
        ["labels"],
        perplexity=resolved_perplexity,
    )[:, 0]
    summary = float(np.nanmedian(per_cell))
    if not scale:
        return summary
    if invert:
        return float((n_categories - summary) / (n_categories - 1))
    return float((summary - 1) / (n_categories - 1))


def ilisi_knn(
    distances: np.ndarray | ZarrArray,
    indices: np.ndarray | ZarrArray,
    batch_labels: Sequence[object] | np.ndarray,
    perplexity: float | None = None,
    scale: bool = True,
) -> float:
    """Compute the median integration LISI score from a self-free KNN graph.

    When scaled, higher values indicate better batch mixing. This follows the
    iLISI aggregation and scaling used by scIB.

    The KNN arrays must exclude each cell from its own neighbor row. Scarf's
    persisted KNN graphs satisfy this requirement. Self-including KNN arrays
    produce a different statistic and are not adjusted automatically.

    References:
        Luecken et al. 2022 doi: 10.1038/s41592-021-01336-8
    """
    return _lisi_knn_summary(
        distances,
        indices,
        batch_labels,
        perplexity,
        scale,
        invert=False,
        label_name="Batch",
    )


def clisi_knn(
    distances: np.ndarray | ZarrArray,
    indices: np.ndarray | ZarrArray,
    cell_labels: Sequence[object] | np.ndarray,
    perplexity: float | None = None,
    scale: bool = True,
) -> float:
    """Compute the median cell-type LISI score from a self-free KNN graph.

    When scaled, higher values indicate better conservation of cell labels.
    This follows the cLISI aggregation and scaling used by scIB.

    The KNN arrays must exclude each cell from its own neighbor row. Scarf's
    persisted KNN graphs satisfy this requirement. Self-including KNN arrays
    produce a different statistic and are not adjusted automatically.

    References:
        Luecken et al. 2022 doi: 10.1038/s41592-021-01336-8
    """
    return _lisi_knn_summary(
        distances,
        indices,
        cell_labels,
        perplexity,
        scale,
        invert=True,
        label_name="Cell",
    )


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
