"""Registered alternatives and pure scientific response validation."""

import math
from typing import Any

from .models import (
    AnalysisConfig,
    AnnotationDecision,
    Candidate,
    Choice,
    ContextDecision,
    Study,
)


def alternatives(
    candidates: list[Candidate], prepared: dict[str, Any], config: AnalysisConfig
) -> dict[str, Candidate]:
    """Offer single changes from completed native candidates, with no hidden search."""
    result: dict[str, Candidate] = {}
    seen = {
        (row.hvgCount, row.pcaDims, row.neighborsK, row.useHarmony)
        for row in candidates
    }
    cells = prepared["retainedCells"]
    features = prepared["availableFeatures"]
    correction_eligible = (
        prepared.get("correctionEligible", False)
        and config.scoreDoublets
        and config.maxFinalists >= 2
    )
    for parent in candidates:
        if parent.useHarmony:
            continue
        for field, values in (
            ("hvgCount", (2000, 4000)),
            ("pcaDims", (10, 30)),
            ("neighborsK", (21, 41)),
            ("useHarmony", (True,) if correction_eligible else ()),
        ):
            for value in values:
                data = parent.model_dump()
                data.update({field: value, "parentId": parent.candidateId})
                signature = (
                    data["hvgCount"],
                    data["pcaDims"],
                    data["neighborsK"],
                    data["useHarmony"],
                )
                if (
                    signature in seen
                    or data["pcaDims"] >= min(cells, features, data["hvgCount"])
                    or data["neighborsK"] >= cells
                ):
                    continue
                name = f"{parent.candidateId}:{field}:{value}"
                data["candidateId"] = name
                result[name] = Candidate.model_validate(data)
                seen.add(signature)
    return result


def validate_choice(
    value: Choice,
    *,
    options: set[str],
    actions: set[str],
    maximum: int = 1,
    evidence_ids: set[str] | None = None,
) -> None:
    errors = []
    if value.action not in actions:
        errors.append(f"action must be one of {sorted(actions)}")
    if value.action == "defer":
        if value.optionIds:
            errors.append("a deferred decision must not select options")
        if not value.question or not value.question.strip():
            errors.append("defer requires an actionable question")
    else:
        if not 1 <= len(value.optionIds) <= maximum:
            errors.append(f"select between one and {maximum} optionIds")
        unknown = set(value.optionIds) - options
        if unknown:
            errors.append(
                f"unknown optionIds: {sorted(unknown)}; offered: {sorted(options)}"
            )
        if len(set(value.optionIds)) != len(value.optionIds):
            errors.append("optionIds must not contain duplicates")
    if evidence_ids is not None and set(value.evidenceIds) - evidence_ids:
        errors.append(
            f"unknown evidenceIds: {sorted(set(value.evidenceIds) - evidence_ids)}"
        )
    if errors:
        raise ValueError("; ".join(errors))


def validate_context(
    value: ContextDecision,
    *,
    columns: set[str],
    study: Study,
    evidence_ids: set[str] | None = None,
) -> None:
    errors = []
    unknown = set(value.columnRoles) - columns
    if unknown:
        errors.append(f"unknown or held-out columns: {sorted(unknown)}")
    unauthorized = set(value.excludeFeatures) - set(study.featureExclusions)
    if unauthorized:
        errors.append(
            f"feature exclusions must come from the supplied inventory: {sorted(unauthorized)}"
        )
    for column, role in value.columnRoles.items():
        if role == "technical" and column not in study.technicalBatchColumns:
            errors.append(
                f"{column}: a model cannot authorize a new technical batch column"
            )
        if column in study.technicalBatchColumns and role not in {
            "technical",
            "ignore",
        }:
            errors.append(f"{column}: a model cannot change a supplied technical role")
        if column in study.protectedColumns and role != "protected":
            errors.append(f"{column}: a model cannot remove a supplied protected role")
    if evidence_ids is None:
        evidence_ids = {
            "source:summary",
            "design:correction",
            *(f"column:{column}" for column in columns),
            *(f"reference:{index}" for index in range(len(study.referenceFiles))),
        }
    unknown_evidence = set(value.evidenceIds) - evidence_ids
    if unknown_evidence:
        errors.append(f"unknown evidenceIds: {sorted(unknown_evidence)}")
    if errors:
        raise ValueError("; ".join(errors))


def _supports_identity(marker: dict[str, Any]) -> bool:
    try:
        score, fraction = float(marker["score"]), float(marker["fracExp"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        math.isfinite(score)
        and math.isfinite(fraction)
        and score >= 0.25
        and fraction >= 0.2
    )


def validate_annotations(
    value: AnnotationDecision, *, clusters: list[dict[str, Any]]
) -> None:
    offered = {str(row["clusterId"]): row for row in clusters}
    observed = [row.clusterId for row in value.annotations]
    errors = []
    if set(observed) != set(offered) or len(observed) != len(offered):
        errors.append(f"annotate exactly these clusterIds once each: {sorted(offered)}")
    for row in value.annotations:
        if row.clusterId not in offered:
            continue
        markers = {
            str(marker["gene"]): marker
            for marker in [
                *offered[row.clusterId]["markers"],
                *offered[row.clusterId].get("weakMarkers", []),
            ]
        }
        cited = set(row.supportingMarkers) | set(row.contradictingMarkers)
        if cited - markers.keys():
            errors.append(
                f"cluster {row.clusterId}: markers must be observed, unknown {sorted(cited - markers.keys())}"
            )
        if set(row.supportingMarkers) & set(row.contradictingMarkers):
            errors.append(
                f"cluster {row.clusterId}: supporting and contradicting markers must differ"
            )
        if row.identity.strip().lower() != "unassigned":
            positive = [
                gene
                for gene in set(row.supportingMarkers)
                if gene in markers and _supports_identity(markers[gene])
            ]
            if len(positive) < 2:
                errors.append(
                    f"cluster {row.clusterId}: a named identity needs at least two observed positive markers with score >= 0.25 and fracExp >= 0.2; otherwise use unassigned"
                )
        elif row.confidence != "low":
            errors.append(
                f"cluster {row.clusterId}: unassigned identity must have low confidence"
            )
        if not row.identity.strip():
            errors.append(f"cluster {row.clusterId}: identity must not be blank")
    if errors:
        raise ValueError("; ".join(errors))
