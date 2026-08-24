"""Portable numerical primitives for Symphony-style reference mapping."""

from typing import cast

import numpy as np

from .models import (
    QueryCorrection,
    ScaledPCAProjectionModel,
    SymphonyCorrectionModel,
)

SYMPHONY_ALGORITHM = "symphony"


def project_pca(values: np.ndarray, model: ScaledPCAProjectionModel) -> np.ndarray:
    """Project normalized expression onto immutable reference PCA loadings."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != model.n_features:
        raise ValueError(
            f"Expected query matrix with {model.n_features} features, got {values.shape}"
        )
    projected = ((values - model.feature_means) / model.feature_scales) @ model.loadings
    if not np.all(np.isfinite(projected)):
        raise ValueError("PCA projection produced non-finite values")
    return cast(np.ndarray, projected)


def scaled_dispersion_sum(values: np.ndarray, model: ScaledPCAProjectionModel) -> float:
    """Return the summed squared scaled deviation of one aligned query block.

    The reference PCA is fitted on z-scored features, so dividing this sum by the
    cell and feature counts returns exactly 1 for the reference itself. A query
    that returns much less than 1 occupies a narrower region of the same space
    and its cells collect near the middle of the reference cloud, where the
    retrieved neighbors stop reflecting the query's own structure.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != model.n_features:
        raise ValueError(
            f"Expected query matrix with {model.n_features} features, got {values.shape}"
        )
    scaled = (values - model.feature_means) / model.feature_scales
    return float(np.einsum("ij,ij->", scaled, scaled))


def soft_cluster_assignments(
    coordinates: np.ndarray, model: SymphonyCorrectionModel
) -> np.ndarray:
    """Calculate cosine-kernel soft assignments to reference centroids."""
    values = np.asarray(coordinates, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != model.n_dims:
        raise ValueError("Query coordinates have incompatible dimensions")
    query_unit = _normalize_rows(values)
    centroids_unit = _normalize_rows(model.centroids)
    distances = 2.0 * (1.0 - query_unit @ centroids_unit.T)
    logits = -distances / model.sigma[np.newaxis, :]
    logits -= logits.max(axis=1, keepdims=True)
    assignments = np.exp(logits)
    assignments /= assignments.sum(axis=1, keepdims=True)
    if not np.all(np.isfinite(assignments)):
        raise ValueError("Soft cluster assignments contain non-finite values")
    return cast(np.ndarray, assignments)


def zero_norm_rows(coordinates: np.ndarray) -> np.ndarray:
    """Return rows without directional information in reference PC space."""
    values = np.asarray(coordinates, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Query coordinates must be two-dimensional")
    return cast(np.ndarray, np.linalg.norm(values, axis=1) == 0)


def initialize_sufficient_statistics(
    n_batches: int, model: SymphonyCorrectionModel
) -> tuple[np.ndarray, np.ndarray]:
    if n_batches < 1:
        raise ValueError("At least one query batch is required")
    return (
        np.zeros((n_batches, model.n_clusters), dtype=np.float64),
        np.zeros((n_batches, model.n_clusters, model.n_dims), dtype=np.float64),
    )


def accumulate_sufficient_statistics(
    counts: np.ndarray,
    sums: np.ndarray,
    coordinates: np.ndarray,
    assignments: np.ndarray,
    batch_codes: np.ndarray,
) -> None:
    """Accumulate per-query-batch, per-cluster weighted coordinate sums."""
    if coordinates.shape[0] != len(batch_codes) or assignments.shape[0] != len(
        batch_codes
    ):
        raise ValueError("Query batch codes must match coordinate rows")
    if assignments.shape[1] != counts.shape[1]:
        raise ValueError("Assignment count does not match reference clusters")
    if np.any(batch_codes < 0) or np.any(batch_codes >= counts.shape[0]):
        raise ValueError("Query batch codes are out of range")

    for batch_code in np.unique(batch_codes):
        batch_mask = batch_codes == batch_code
        batch_assignments = assignments[batch_mask]
        counts[batch_code] += batch_assignments.sum(axis=0)
        sums[batch_code] += batch_assignments.T @ coordinates[batch_mask]


def solve_query_correction(
    counts: np.ndarray,
    sums: np.ndarray,
    model: SymphonyCorrectionModel,
) -> QueryCorrection:
    """Fit Symphony's joint cluster-aware query batch correction."""
    if counts.ndim != 2 or counts.shape[1] != model.n_clusters:
        raise ValueError("Query count statistics have incompatible dimensions")
    if sums.shape != (counts.shape[0], model.n_clusters, model.n_dims):
        raise ValueError("Query sum statistics have incompatible dimensions")
    n_batches = counts.shape[0]
    offsets = np.zeros_like(sums)
    for cluster in range(model.n_clusters):
        cluster_counts = counts[:, cluster]
        design_crossproduct = np.zeros(
            (n_batches + 1, n_batches + 1),
            dtype=np.float64,
        )
        design_crossproduct[0, 0] = cluster_counts.sum() + model.cluster_mass[cluster]
        design_crossproduct[0, 1:] = cluster_counts
        design_crossproduct[1:, 0] = cluster_counts
        design_crossproduct[1:, 1:] = np.diag(cluster_counts + 1.0)

        coordinate_crossproduct = np.empty(
            (n_batches + 1, model.n_dims),
            dtype=np.float64,
        )
        coordinate_crossproduct[0] = (
            sums[:, cluster].sum(axis=0)
            + model.cluster_mass[cluster] * model.corrected_centroids[cluster]
        )
        coordinate_crossproduct[1:] = sums[:, cluster]
        coefficients = np.linalg.solve(
            design_crossproduct,
            coordinate_crossproduct,
        )
        offsets[:, cluster] = coefficients[1:]
    return QueryCorrection(batch_offsets=offsets, batch_counts=counts.copy())


def apply_query_correction(
    coordinates: np.ndarray,
    assignments: np.ndarray,
    batch_codes: np.ndarray,
    model: SymphonyCorrectionModel,
    correction: QueryCorrection,
) -> np.ndarray:
    """Map query PCA coordinates into the fixed corrected reference space."""
    if coordinates.shape[0] != assignments.shape[0] or coordinates.shape[0] != len(
        batch_codes
    ):
        raise ValueError("Query coordinate, assignment, and batch rows must agree")
    if assignments.shape[1] != model.n_clusters:
        raise ValueError("Query assignments have incompatible cluster count")
    if correction.batch_offsets.shape[1:] != (model.n_clusters, model.n_dims):
        raise ValueError("Query correction has incompatible reference dimensions")
    query_offsets = np.einsum(
        "nk,nkd->nd", assignments, correction.batch_offsets[batch_codes]
    )
    corrected = coordinates - query_offsets
    if not np.all(np.isfinite(corrected)):
        raise ValueError("Query correction produced non-finite coordinates")
    return cast(np.ndarray, corrected)


def weighted_centroids(
    coordinates: np.ndarray, assignments: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return soft cluster masses and centroids for reference compression."""
    values = np.asarray(coordinates, dtype=np.float64)
    weights = np.asarray(assignments, dtype=np.float64)
    if values.ndim != 2 or weights.ndim != 2 or values.shape[0] != weights.shape[1]:
        raise ValueError("Coordinate and assignment dimensions do not agree")
    mass = weights.sum(axis=1)
    if np.any(mass <= 0):
        raise ValueError("Reference assignments include an empty cluster")
    centroids = (weights @ values / mass[:, np.newaxis]).astype(np.float64)
    return mass, centroids


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = np.zeros_like(values, dtype=np.float64)
    nonzero = norms[:, 0] > 0
    normalized[nonzero] = values[nonzero] / norms[nonzero]
    return normalized
