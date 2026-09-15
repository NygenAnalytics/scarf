"""Core-backed feature selection and bounded technical-group rank aggregation."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np

from ...storage.refs import ArtifactRef

HVG_CANDIDATE_TARGETS = (1000, 2000, 4000)


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
    """Bounded feature-axis output of global or batch-aware HVG ranking."""

    ranking_mode: Literal["global", "batchAware"]
    eligible: np.ndarray
    global_corrected_variance: np.ndarray
    recurrence: np.ndarray
    mean_within_group_rank: np.ndarray
    ranking: np.ndarray
    valid_group_count: int
    candidate_counts: tuple[int, ...]

    @property
    def eligible_feature_count(self) -> int:
        return int(self.eligible.sum())

    def candidate_mask(self, top_n: int) -> np.ndarray:
        """Return one registered nested candidate, never an arbitrary count."""
        if top_n not in self.candidate_counts:
            raise ValueError(
                f"top_n must be one of the registered counts {self.candidate_counts}"
            )
        values = np.zeros(self.eligible.shape, dtype=bool)
        values[self.ranking[:top_n]] = True
        return values


def effective_hvg_candidate_counts(
    eligible_feature_count: int,
    targets: Sequence[int] = HVG_CANDIDATE_TARGETS,
) -> tuple[int, ...]:
    """Cap registered HVG counts by eligibility and remove capped duplicates."""
    if isinstance(eligible_feature_count, bool) or not isinstance(
        eligible_feature_count, int
    ):
        raise TypeError("eligible_feature_count must be an integer")
    if eligible_feature_count < 1:
        raise ValueError("eligible_feature_count must be greater than 0")
    if isinstance(targets, str | bytes):
        raise TypeError("targets must be a sequence of positive integers")
    resolved: list[int] = []
    for target in targets:
        if isinstance(target, bool) or not isinstance(target, int):
            raise TypeError("HVG candidate targets must be integers")
        if target < 1:
            raise ValueError("HVG candidate targets must be greater than 0")
        effective = min(target, eligible_feature_count)
        if effective not in resolved:
            resolved.append(effective)
    if not resolved:
        raise ValueError("At least one HVG candidate target is required")
    return tuple(resolved)


def aggregate_hvg_rankings(
    global_corrected_variance: np.ndarray,
    eligible_features: np.ndarray,
    group_variability: Iterable[HvgGroupVariability],
    *,
    valid_group_count: int,
    candidate_targets: Sequence[int] = HVG_CANDIDATE_TARGETS,
) -> HvgRanking:
    """Aggregate global and optional batch-aware feature rankings."""
    corrected = np.asarray(global_corrected_variance, dtype=np.float64)
    eligible = np.asarray(eligible_features, dtype=bool)
    if corrected.ndim != 1 or eligible.shape != corrected.shape:
        raise ValueError("Global variability and eligibility must be aligned vectors")
    if not np.isfinite(corrected).all() or (corrected < 0).any():
        raise ValueError("Global corrected variability must be finite and non-negative")
    if isinstance(valid_group_count, bool) or not isinstance(valid_group_count, int):
        raise TypeError("valid_group_count must be an integer")
    if valid_group_count < 0:
        raise ValueError("valid_group_count must be non-negative")
    eligible_count = int(eligible.sum())
    counts = effective_hvg_candidate_counts(eligible_count, candidate_targets)
    indices = np.flatnonzero(eligible)
    global_order = indices[np.lexsort((indices, -corrected[indices]))].astype(
        np.int64, copy=False
    )
    recurrence = np.zeros(corrected.shape, dtype=np.int32)
    mean_rank = np.full(corrected.shape, np.inf, dtype=np.float64)

    if valid_group_count < 2:
        denominator = max(1, len(global_order))
        mean_rank[global_order] = (
            np.arange(1, len(global_order) + 1, dtype=np.float64) / denominator
        )
        return HvgRanking(
            ranking_mode="global",
            eligible=eligible.copy(),
            global_corrected_variance=corrected.copy(),
            recurrence=recurrence,
            mean_within_group_rank=mean_rank,
            ranking=global_order.copy(),
            valid_group_count=valid_group_count,
            candidate_counts=counts,
        )

    rank_sum = np.zeros(corrected.shape, dtype=np.float64)
    broad_count = max(counts)
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
    return HvgRanking(
        ranking_mode="batchAware",
        eligible=eligible.copy(),
        global_corrected_variance=corrected.copy(),
        recurrence=recurrence,
        mean_within_group_rank=mean_rank,
        ranking=ranking.copy(),
        valid_group_count=valid_group_count,
        candidate_counts=counts,
    )
