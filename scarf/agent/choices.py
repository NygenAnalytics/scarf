"""Registered alternatives and pure scientific response validation."""

import math
from typing import Any

from .models import (
    AnnotationDecision,
    Candidate,
    Choice,
    ContextDecision,
    Study,
)


def native_probe_options(
    baseline: Candidate, prepared: dict[str, Any]
) -> dict[str, dict[str, Candidate]]:
    """Offer feasible independent probes of the exact native baseline."""
    if baseline.useHarmony:
        raise ValueError("Native probes require an uncorrected baseline")
    cells = prepared["retainedCells"]
    features = prepared["availableFeatures"]
    result: dict[str, dict[str, Candidate]] = {}
    for field, values in (
        ("hvgCount", (2000, 4000, 1000)),
        ("pcaDims", (10, 30)),
        ("neighborsK", (21, 41, 11)),
    ):
        offered = {}
        for value in values:
            if value == getattr(baseline, field):
                continue
            data = baseline.model_dump()
            data.update({field: value, "parentId": baseline.candidateId})
            actual_features = (
                features
                if field == "hvgCount"
                else min(features, prepared.get("actualHvgCount", baseline.hvgCount))
            )
            if (
                field == "hvgCount"
                and value > features
                or data["pcaDims"] >= min(cells, actual_features, data["hvgCount"])
                or data["neighborsK"] >= cells
            ):
                continue
            option_id = f"{baseline.candidateId}:{field}:{value}"
            data["candidateId"] = option_id
            offered[option_id] = Candidate.model_validate(data)
        result[field] = offered
    return result


def _deferral_errors(
    value: Choice | ContextDecision,
    *,
    unresolved_fact_ids: set[str] | None,
) -> list[str]:
    errors = []
    if not value.question or not value.question.strip():
        errors.append("defer requires an actionable question")
    if value.deferralReason is None:
        errors.append("defer requires a deferralReason")
    if not value.evidenceIds:
        errors.append("defer requires supplied evidenceIds")
    if value.deferralReason == "missingEssentialInput" and not (
        set(value.evidenceIds) & (unresolved_fact_ids or set())
    ):
        errors.append("missingEssentialInput must cite a deterministic unresolved fact")
    return errors


def resolve_deferral(
    value: Choice | ContextDecision,
    *,
    interaction_mode: str,
    option_order: list[str],
    unresolved_fact_ids: set[str],
) -> dict[str, Any] | None:
    """Resolve a valid optional ambiguity without changing scientific gates."""
    if interaction_mode not in {"strict", "lenient"}:
        raise ValueError("interaction_mode must be strict or lenient")
    if isinstance(value, Choice) and value.action != "defer":
        raise ValueError("Only a deferred choice can receive a policy resolution")
    errors = _deferral_errors(value, unresolved_fact_ids=unresolved_fact_ids)
    if errors:
        raise ValueError("; ".join(errors))
    if interaction_mode == "strict" or value.deferralReason in {
        "missingEssentialInput",
        "unsupportedObjective",
    }:
        return None
    if isinstance(value, ContextDecision):
        if value.deferralReason != "uncertainMetadata":
            raise ValueError("Context can resolve only uncertainMetadata")
        return {"action": "retainDeclaredRoles"}
    if value.deferralReason != "ambiguousSelection":
        raise ValueError("Choices can resolve only ambiguousSelection")
    acceptable = set(value.acceptableOptionIds)
    if (
        len(acceptable) < 2
        or len(acceptable) != len(value.acceptableOptionIds)
        or not acceptable.issubset(option_order)
    ):
        raise ValueError("Ambiguity requires at least two distinct eligible options")
    return {
        "action": "choose",
        "optionIds": [next(option for option in option_order if option in acceptable)],
    }


def validate_choice(
    value: Choice,
    *,
    options: set[str],
    actions: set[str],
    maximum: int = 1,
    evidence_ids: set[str] | None = None,
    unresolved_fact_ids: set[str] | None = None,
) -> None:
    errors = []
    if value.action not in actions:
        errors.append(f"action must be one of {sorted(actions)}")
    if value.action == "defer":
        if value.optionIds:
            errors.append("a deferred decision must not select options")
        errors.extend(_deferral_errors(value, unresolved_fact_ids=unresolved_fact_ids))
        if value.deferralReason == "uncertainMetadata":
            errors.append("uncertainMetadata is a context deferral")
        if value.deferralReason == "ambiguousSelection":
            acceptable = set(value.acceptableOptionIds)
            if (
                len(acceptable) < 2
                or len(acceptable) != len(value.acceptableOptionIds)
                or not acceptable.issubset(options)
            ):
                errors.append(
                    "ambiguousSelection requires at least two distinct eligible acceptableOptionIds"
                )
        elif value.acceptableOptionIds:
            errors.append("acceptableOptionIds are only valid for ambiguousSelection")
    else:
        if value.question is not None or value.deferralReason is not None:
            errors.append("ordinary choices must not contain deferral fields")
        if value.acceptableOptionIds:
            errors.append("ordinary choices must not contain acceptableOptionIds")
        if not value.evidenceIds:
            errors.append("ordinary choices require supplied evidenceIds")
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


def context_evidence_ids(columns: set[str], study: Study) -> set[str]:
    """Identify visible context evidence, including supplied role provenance."""
    identifiers = {
        "source:summary",
        "design:correction",
        *(f"column:{column}" for column in columns),
        *(f"reference:{index}" for index in range(len(study.referenceFiles))),
    }
    for role, declared in (
        ("technical", study.technicalBatchColumns),
        ("protected", study.protectedColumns),
        ("sample", [study.sampleColumn] if study.sampleColumn else []),
        ("capture", [study.captureColumn] if study.captureColumn else []),
    ):
        if columns.intersection(declared):
            identifiers.add(f"study:{role}")
    return identifiers


def validate_context(
    value: ContextDecision,
    *,
    columns: set[str],
    study: Study,
    evidence_ids: set[str] | None = None,
    unresolved_fact_ids: set[str] | None = None,
) -> None:
    errors = []
    if value.question is not None or value.deferralReason is not None:
        errors.extend(_deferral_errors(value, unresolved_fact_ids=unresolved_fact_ids))
        if value.deferralReason == "ambiguousSelection":
            errors.append("ambiguousSelection is a selection deferral")
    elif (value.columnRoles or value.excludeFeatures) and not value.evidenceIds:
        errors.append("context changes require supplied evidenceIds")
    for role in ("sample", "capture"):
        proposed = [
            column for column, selected in value.columnRoles.items() if selected == role
        ]
        supplied = getattr(study, f"{role}Column")
        if len(proposed) > 1:
            errors.append(f"at most one column may have the {role} role")
        if supplied is not None and any(column != supplied for column in proposed):
            errors.append(f"a model cannot replace the supplied {role} column")
    unknown = set(value.columnRoles) - columns
    if unknown:
        errors.append(f"unknown or held-out columns: {sorted(unknown)}")
    unauthorized = set(value.excludeFeatures) - set(study.featureExclusions)
    if unauthorized:
        errors.append(
            f"feature exclusions must come from the supplied inventory: {sorted(unauthorized)}"
        )
    for column, role in value.columnRoles.items():
        declared = {
            name
            for name, present in (
                ("technical", column in study.technicalBatchColumns),
                ("protected", column in study.protectedColumns),
                ("sample", column == study.sampleColumn),
                ("capture", column == study.captureColumn),
            )
            if present
        }
        if role == "technical" and column not in study.technicalBatchColumns:
            errors.append(
                f"{column}: a model cannot authorize a new technical batch column"
            )
        if "technical" in declared and role not in declared:
            errors.append(f"{column}: a model cannot change a supplied technical role")
        if "protected" in declared and role not in declared:
            errors.append(f"{column}: a model cannot remove a supplied protected role")
        for supplied_role in ("sample", "capture"):
            if supplied_role in declared and role not in declared:
                errors.append(
                    f"{column}: a model cannot change a supplied {supplied_role} role"
                )
    if evidence_ids is None:
        evidence_ids = context_evidence_ids(columns, study)
    unknown_evidence = set(value.evidenceIds) - evidence_ids
    if unknown_evidence:
        errors.append(
            f"unknown evidenceIds: {sorted(unknown_evidence)}; offered: {sorted(evidence_ids)}"
        )
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
