from dataclasses import dataclass
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


def add_mapping_scores(
    scores: np.ndarray,
    scored_rows: np.ndarray,
    neighbor_indices: np.ndarray,
    weights: np.ndarray,
    *,
    skip: np.ndarray,
    groups: np.ndarray | None = None,
) -> None:
    """Add the neighbor weights of one block of query rows to mapping scores.

    ``scores`` holds one row of reference-cell sums per query group, and
    ``scored_rows`` counts the query rows added to each group. ``skip`` marks
    the block's query rows that add nothing, following the caller's policy.
    ``neighbor_indices`` and ``weights``, such as ``mapping_score_weights``,
    hold one row for each query row that is not skipped, in block order.
    ``groups`` gives the group row of every block row; without it, every row
    belongs to group 0.

    Args:
        scores: Group-by-reference-cell sums, updated in place.
        scored_rows: Number of query rows added to each group, updated in
            place.
        neighbor_indices: Reference neighbors of the rows that are not
            skipped.
        weights: Weight of each of those neighbors.
        skip: Boolean mask of the block's query rows that add nothing.
        groups: Optional group row of each block row.
    """
    keep = ~np.asarray(skip, dtype=bool)
    rows = (
        np.zeros(int(np.count_nonzero(keep)), dtype=np.int64)
        if groups is None
        else np.asarray(groups, dtype=np.int64)[keep]
    )
    if (
        neighbor_indices.ndim != 2
        or neighbor_indices.shape[0] != rows.shape[0]
        or weights.shape != neighbor_indices.shape
    ):
        raise ValueError(
            "Neighbor indices and weights need one row per query row that is "
            "not skipped"
        )
    np.add.at(
        scores,
        (
            np.broadcast_to(rows[:, np.newaxis], neighbor_indices.shape),
            neighbor_indices,
        ),
        weights,
    )
    scored_rows += np.bincount(rows, minlength=len(scored_rows))


def finish_mapping_scores(
    scores: np.ndarray,
    scored_rows: np.ndarray,
    *,
    n_neighbors: int,
    multiplier: float,
    log_transform: bool,
) -> None:
    """Scale summed mapping-score weights in place.

    Each group's sums are multiplied by ``multiplier`` over its scored query
    rows times ``n_neighbors``, so a score does not grow with the number of
    query rows or neighbors. A group without scored rows keeps zeros.
    ``log_transform`` then replaces every score with its ``log1p``.

    Args:
        scores: Group-by-reference-cell sums from ``add_mapping_scores``.
        scored_rows: Number of query rows added to each group.
        n_neighbors: Neighbors of each query row.
        multiplier: Scale of a score.
        log_transform: Whether to take ``log1p`` of the scaled scores.
    """
    for row, count in enumerate(scored_rows):
        if count:
            scores[row] *= multiplier / (int(count) * n_neighbors)
    if log_transform:
        np.log1p(scores, out=scores)


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


@dataclass(slots=True)
class _LabelVotes:
    class_codes: np.ndarray
    fractions: np.ndarray
    prediction_codes: np.ndarray
    vote_fraction: np.ndarray
    vote_entropy: np.ndarray
    top_two_margin: np.ndarray
    is_unknown: np.ndarray


def _label_vote_block(
    neighbor_codes: np.ndarray,
    weights: np.ndarray,
    threshold: float,
) -> _LabelVotes:
    """Aggregate categorical votes in neighbor order using bounded row buffers."""
    n_rows, n_neighbors = neighbor_codes.shape
    rows = np.arange(n_rows)[:, None]
    order = np.argsort(neighbor_codes, axis=1, kind="stable")
    codes = np.take_along_axis(neighbor_codes, order, axis=1)
    sorted_weights = np.take_along_axis(weights, order, axis=1)
    sorted_weights[codes < 0] = 0.0
    starts = np.ones(codes.shape, dtype=bool)
    starts[:, 1:] = codes[:, 1:] != codes[:, :-1]
    groups = np.cumsum(starts, axis=1) - 1
    mass = np.zeros_like(weights, dtype=np.float64)
    np.add.at(mass, (rows, groups), sorted_weights)
    first_positions = np.full(codes.shape, n_neighbors, dtype=np.intp)
    np.minimum.at(first_positions, (rows, groups), order)
    class_codes = np.full(codes.shape, -1, dtype=np.int64)
    class_codes[rows, groups] = codes
    first_order = np.argsort(first_positions, axis=1, kind="stable")
    mass = np.take_along_axis(mass, first_order, axis=1)
    class_codes = np.take_along_axis(class_codes, first_order, axis=1)

    labeled_total = np.zeros(n_rows, dtype=np.float64)
    for column in range(n_neighbors):
        labeled_total += mass[:, column]
    entropy = np.zeros(n_rows, dtype=np.float64)
    for column in range(n_neighbors):
        probabilities = np.divide(
            mass[:, column],
            labeled_total,
            out=np.zeros(n_rows, dtype=np.float64),
            where=labeled_total > 0,
        )
        positive = probabilities > 0
        entropy[positive] -= probabilities[positive] * np.log(probabilities[positive])
    total = weights.sum(axis=1, dtype=np.float64)
    fractions = np.divide(
        mass, total[:, None], out=np.zeros_like(mass), where=total[:, None] > 0
    )
    best = fractions.argmax(axis=1)
    top = fractions[np.arange(n_rows), best]
    second = (
        np.partition(fractions, -2, axis=1)[:, -2]
        if n_neighbors > 1
        else np.zeros(n_rows, dtype=np.float64)
    )
    ties = np.count_nonzero(
        (class_codes >= 0) & np.isclose(fractions, top[:, None]), axis=1
    )
    return _LabelVotes(
        class_codes=class_codes,
        fractions=fractions,
        prediction_codes=class_codes[np.arange(n_rows), best],
        vote_fraction=top,
        vote_entropy=entropy,
        top_two_margin=top - second,
        is_unknown=(labeled_total <= 0) | (top < threshold) | (ties != 1),
    )


def _validated_conformal_calibration(
    calibration_nonconformity: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, float]:
    calibration = np.asarray(calibration_nonconformity, dtype=np.float64)
    if calibration.ndim != 1 or calibration.size == 0:
        raise ValueError("calibration_nonconformity must be a non-empty vector")
    if (
        isinstance(alpha, bool | np.bool_)
        or not isinstance(alpha, int | float | np.integer | np.floating)
        or not np.isfinite(alpha)
        or not 0 < float(alpha) < 1
    ):
        raise ValueError("alpha must be strictly between zero and one")
    if not np.all(np.isfinite(calibration)):
        raise ValueError("Conformal inputs must be finite")
    if np.any(calibration < 0) or np.any(calibration > 1):
        raise ValueError("Conformal nonconformity must be in [0, 1]")
    return np.sort(calibration), float(alpha)


def _conformal_membership(
    label_scores: np.ndarray,
    sorted_calibration: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Compare scores to prepared calibration without a three-axis temporary."""
    nonconformity = 1.0 - label_scores
    insertion = np.searchsorted(
        sorted_calibration,
        nonconformity,
        side="left",
    )
    exceedances = len(sorted_calibration) - insertion
    return cast(
        np.ndarray,
        (exceedances + 1) / (len(sorted_calibration) + 1) > alpha,
    )
