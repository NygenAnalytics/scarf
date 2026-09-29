"""Core-backed feature selection and bounded technical-group rank aggregation."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from ...storage.refs import ArtifactRef


def core_hvg_evidence(
    store: Any,
    *,
    assay: str,
    cells: ArtifactRef,
) -> dict[str, ArtifactRef]:
    """Obtain baseline, eligible universes and variability from core Scarf.

    These are feature-axis computations. Requesting all eligible genes does
    not normalize, reduce or construct a graph for another candidate.
    """
    feature_count = int(store.get_assay(assay).feats.N)
    options = {"from_assay": assay, "show_plot": False, "invalidate_cache": False}
    return {
        "scarfDefault": store.select_hvgs(cells, top_n=1000, **options),
        "eligibleDefault": store.select_hvgs(cells, top_n=feature_count, **options),
        "eligibleAll": store.select_hvgs(
            cells, top_n=feature_count, blacklist="", **options
        ),
    }


def rank_core_hvgs(
    store: Any,
    *,
    eligible: ArtifactRef,
    statistics: ArtifactRef,
    top_n: int,
    ranking: np.ndarray | None = None,
) -> ArtifactRef:
    """Select a count on a frozen universe using the core variability payload."""
    mask = np.asarray(store.load_artifact(eligible)["values"][:], dtype=bool)
    variance = np.asarray(
        store.load_artifact(statistics)["corrected_variance"][:], dtype=np.float64
    )
    if mask.shape != variance.shape or not np.isfinite(variance).all():
        raise ValueError("Core HVG statistics do not align with eligible genes")
    indices = np.flatnonzero(mask)
    if ranking is None:
        indices = indices[np.lexsort((indices, -variance[indices]))]
    else:
        ordered = np.asarray(ranking, dtype=np.int64)
        if ordered.ndim != 1 or len(np.unique(ordered)) != len(ordered):
            raise ValueError("HVG ranking must contain unique feature indices")
        if np.any(ordered < 0) or np.any(ordered >= len(mask)):
            raise ValueError("HVG ranking contains invalid feature indices")
        indices = ordered[mask[ordered]]
        if len(indices) != int(mask.sum()):
            raise ValueError("HVG ranking must cover the exact eligible universe")
    if isinstance(top_n, bool) or top_n < 3:
        raise ValueError("RNA representation needs at least three requested genes")
    selected = np.zeros(mask.shape, dtype=bool)
    selected[indices[:top_n]] = True
    if int(selected.sum()) < 3:
        raise ValueError("Fewer than three eligible genes remain")
    return cast(
        ArtifactRef,
        store.set_feature_selection(
            from_assay=eligible.assay, mask=selected, invalidate_cache=False
        ),
    )


@dataclass(frozen=True, slots=True)
class HvgGroupVariability:
    """One technical group's feature variability, streamed into aggregation."""

    group_id: str
    cell_count: int
    corrected_variance: np.ndarray
    detected_features: np.ndarray


@dataclass(frozen=True, slots=True)
class HvgRanking:
    """Batch-aware ranking of the exact eligible feature universe."""

    ranking: np.ndarray


def aggregate_hvg_rankings(
    global_corrected_variance: np.ndarray,
    eligible_features: np.ndarray,
    group_variability: Iterable[HvgGroupVariability],
    *,
    valid_group_count: int,
    candidate_targets: Sequence[int],
) -> HvgRanking:
    """Rank eligible features by their recurrence across technical groups."""
    corrected = np.asarray(global_corrected_variance, dtype=np.float64)
    eligible = np.asarray(eligible_features, dtype=bool)
    if corrected.ndim != 1 or eligible.shape != corrected.shape:
        raise ValueError("Global variability and eligibility must be aligned vectors")
    if not np.isfinite(corrected).all() or (corrected < 0).any():
        raise ValueError("Global corrected variability must be finite and non-negative")
    if isinstance(valid_group_count, bool) or not isinstance(valid_group_count, int):
        raise TypeError("valid_group_count must be an integer")
    if valid_group_count < 2:
        raise ValueError("Batch-aware HVG ranking requires at least two groups")
    if isinstance(candidate_targets, str | bytes):
        raise TypeError("candidate_targets must be a sequence of positive integers")
    targets = tuple(candidate_targets)
    if not targets:
        raise ValueError("At least one HVG candidate target is required")
    for target in targets:
        if isinstance(target, bool) or not isinstance(target, int):
            raise TypeError("HVG candidate targets must be integers")
        if target < 1:
            raise ValueError("HVG candidate targets must be greater than 0")
    eligible_count = int(eligible.sum())
    if eligible_count < 1:
        raise ValueError("HVG ranking requires at least one eligible feature")
    broad_count = min(max(targets), eligible_count)
    indices = np.flatnonzero(eligible)
    recurrence = np.zeros(corrected.shape, dtype=np.int32)
    mean_rank = np.full(corrected.shape, np.inf, dtype=np.float64)
    rank_sum = np.zeros(corrected.shape, dtype=np.float64)
    received = 0
    for group in group_variability:
        received += 1
        if received > valid_group_count:
            raise ValueError("More group summaries were supplied than declared")
        if not isinstance(group.group_id, str) or not group.group_id:
            raise ValueError("Every valid technical group needs a non-empty ID")
        if isinstance(group.cell_count, bool) or not isinstance(group.cell_count, int):
            raise TypeError("Technical-group cell counts must be integers")
        if group.cell_count < 1:
            raise ValueError("Technical-group cell counts must be greater than 0")
        group_corrected = np.asarray(group.corrected_variance, dtype=np.float64)
        detected = np.asarray(group.detected_features, dtype=bool)
        if (
            group_corrected.shape != corrected.shape
            or detected.shape != corrected.shape
        ):
            raise ValueError("Technical-group feature arrays must align globally")
        if not np.isfinite(group_corrected).all() or (group_corrected < 0).any():
            raise ValueError(
                "Technical-group variability must be finite and non-negative"
            )
        group_candidates = np.flatnonzero(eligible & detected)
        if group_candidates.size == 0:
            raise ValueError(
                f"Valid technical group {group.group_id!r} has no rankable features"
            )
        order = group_candidates[
            np.lexsort((group_candidates, -group_corrected[group_candidates]))
        ]
        selected = order[: min(broad_count, len(order))]
        recurrence[selected] += 1
        rank_sum[selected] += np.arange(1, len(selected) + 1, dtype=np.float64) / len(
            order
        )
    if received != valid_group_count:
        raise ValueError(
            f"Expected {valid_group_count} group summaries but received {received}"
        )
    observed = recurrence > 0
    mean_rank[observed] = rank_sum[observed] / recurrence[observed]
    ranking = indices[
        np.lexsort(
            (
                indices,
                -corrected[indices],
                mean_rank[indices],
                -recurrence[indices],
            )
        )
    ].astype(np.int64, copy=False)
    return HvgRanking(ranking=ranking)
