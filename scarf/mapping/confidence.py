from typing import Any, cast

import numpy as np

from ..storage.geometry import array_geometry
from ..storage.partition import row_band


def _distance_quantile_summary(
    distances: Any,
    max_samples: int = 100_000,
    n_quantiles: int = 1_001,
) -> tuple[np.ndarray, np.ndarray]:
    """Summarize first-neighbor distances with deterministic row sampling."""
    shape = tuple(int(value) for value in distances.shape)
    if len(shape) not in {1, 2}:
        raise ValueError("Neighbor distances must be one- or two-dimensional")
    n_rows = shape[0]
    if n_rows < 1:
        raise ValueError("Neighbor distances are empty")
    if len(shape) == 2 and shape[1] < 1:
        raise ValueError("Neighbor distances do not contain any neighbors")
    if max_samples < 1 or n_quantiles < 1:
        raise ValueError("Sampling and quantile counts must be positive")

    stride = max(int(np.ceil(n_rows / max_samples)), 1)
    block_size = row_band(
        array_geometry(distances),
        unit="chunk",
        fallback=min(n_rows, 10_000),
    )
    sampled: list[np.ndarray] = []
    for start in range(0, n_rows, block_size):
        stop = min(start + block_size, n_rows)
        block = np.asarray(distances[start:stop])
        if block.ndim == 2:
            block = block[:, 0]
        mask = np.arange(start, stop, dtype=np.int64) % stride == 0
        sampled.append(np.asarray(block[mask], dtype=np.float64))
    values = np.concatenate(sampled)
    quantiles = np.linspace(0.0, 1.0, min(n_quantiles, len(values)))
    return quantiles, np.quantile(values, quantiles)


def _validated_distances(distances: np.ndarray) -> np.ndarray:
    values = np.asarray(distances, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Expected a two-dimensional distance array")
    if not np.all(np.isfinite(values)):
        raise ValueError("Neighbor distances must be finite")
    if np.any(values < 0):
        raise ValueError("Neighbor distances must be non-negative")
    return values


def mapping_score_weights(distances: np.ndarray) -> np.ndarray:
    """Return absolute neighbor weights for reference-side mapping scores.

    The weight of one reference neighbor is ``1 / (log(distance + 1) + 1)``.
    Weights are not normalized per query cell, so a query cell that sits far
    from the reference contributes less total weight than one that lands on the
    reference manifold. Normalizing per row would erase that contrast and make
    the score a plain neighbor count.
    """
    weights: np.ndarray = 1.0 / (np.log1p(_validated_distances(distances)) + 1.0)
    return weights


def distance_weights(distances: np.ndarray) -> np.ndarray:
    """Convert metric distances into normalized inverse-distance weights."""
    values = _validated_distances(distances)

    weights = np.zeros_like(values)
    zero_mask = values == 0
    zero_count = zero_mask.sum(axis=1)
    rows_with_zero = zero_count > 0
    if rows_with_zero.any():
        weights[rows_with_zero] = (
            zero_mask[rows_with_zero] / zero_count[rows_with_zero, np.newaxis]
        )
    rows_without_zero = ~rows_with_zero
    if rows_without_zero.any():
        positive = values[rows_without_zero]
        minimum = positive.min(axis=1, keepdims=True)
        inverse_ratios = minimum / positive
        weights[rows_without_zero] = inverse_ratios / inverse_ratios.sum(
            axis=1,
            keepdims=True,
        )
    return weights


def conformal_prediction_sets(
    label_scores: np.ndarray,
    calibration_nonconformity: np.ndarray,
    alpha: float = 0.1,
) -> np.ndarray:
    """Return class-membership masks from split-conformal p-values."""
    scores = np.asarray(label_scores, dtype=np.float64)
    calibration = np.asarray(calibration_nonconformity, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("label_scores must be a two-dimensional array")
    if calibration.ndim != 1 or calibration.size == 0:
        raise ValueError("calibration_nonconformity must be a non-empty vector")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(calibration)):
        raise ValueError("Conformal inputs must be finite")
    nonconformity = 1.0 - scores
    p_values = (
        (calibration[np.newaxis, np.newaxis, :] >= nonconformity[:, :, np.newaxis]).sum(
            axis=2
        )
        + 1
    ) / (len(calibration) + 1)
    return cast(np.ndarray, p_values > alpha)
