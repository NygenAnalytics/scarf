"""Objective evidence requirements derived from bounded, measured study designs."""

import hashlib
import re
from typing import Any, Literal

from ..record_io import canonical_json_bytes
from .contracts import (
    DesignEvidenceCoverage,
    DesignEvidenceRequirement,
    characterization_evidence,
)


def active_batch_safety(result: Any) -> list[Any]:
    """Select the exact final assessed design without unioning alternatives."""
    all_assessments = list(result.batchSafety)
    columns = sorted(result.decision.batchCorrection.batchColumns)
    if not columns:
        tested = {tuple(sorted(item.batchColumns)) for item in all_assessments}
        if len(tested) > 1:
            raise ValueError(
                "The final correction plan must identify one exact assessed batch set; "
                "separate alternatives cannot be combined into an untested design"
            )
        columns = list(next(iter(tested), ()))
    return [item for item in all_assessments if sorted(item.batchColumns) == columns]


def requested_design_questions(
    study_context: str, study_objective: str, columns: list[str]
) -> list[tuple[str, list[str], bool]]:
    """Keep explicit named joint/conditional requests visible before proposals."""
    output: list[tuple[str, list[str], bool]] = []
    for text in (study_context, study_objective):
        for raw in re.split(r"[\n.!?]+", text):
            quote = raw.strip()
            joint = bool(
                re.search(
                    r"\b(joint|jointly|combined|combination|interaction)\b", quote, re.I
                )
            )
            conditional = bool(
                re.search(r"\b(within|conditioned|stratif\w*)\b", quote, re.I)
            )
            if not (joint or conditional) or re.search(
                r"\b(?:do not|not requested|outside scope)\b", quote, re.I
            ):
                continue
            named = sorted(
                name
                for name in columns
                if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", quote, re.I)
            )
            generic = re.search(r"\b(covariates|factors|columns)\b", quote, re.I)
            if (len(named) >= 2 or not named and generic) and (
                quote,
                named,
                conditional,
            ) not in output:
                output.append((quote, named, conditional))
    return output


def _measured_joint_design(comparison: Any) -> bool:
    """Recognize joint descriptive counts without claiming an association."""
    proposal = comparison.proposal
    columns = {
        proposal.response,
        *proposal.explanatoryColumns,
        *([proposal.conditionedOn] if proposal.conditionedOn else []),
    }
    descriptive = comparison.evidence.get("descriptiveDesign", {})
    rows = descriptive.get("jointGroupSupport", [])
    if (
        len(columns) != 3
        or descriptive.get("status") != "computed"
        or not isinstance(rows, list)
        or not rows
        or any(
            comparison.evidence.get("columnKinds", {}).get(name) != "categorical"
            for name in columns
        )
    ):
        return False
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row.get("groups", {})) != columns
            or type(row.get("observationUnits")) is not int
            or type(row.get("independentUnits")) is not int
            or not 0 < row["independentUnits"] <= row["observationUnits"]
        ):
            return False
    return bool(
        sum(row["observationUnits"] for row in rows)
        == descriptive.get("observationUnits")
    )


def requested_design_purpose(
    quote: str,
) -> Literal["effectEstimation", "association", "designCoverage"]:
    """Keep model guidance and validation on the same requested evidence kind."""
    if re.search(r"\bestimat\w*.*\beffect", quote, re.I):
        return "effectEstimation"
    if re.search(r"\bassociat\w*", quote, re.I):
        return "association"
    return "designCoverage"


def objective_evidence(
    *, study_context: str, study_objective: str, experimental_result: Any
) -> tuple[list[DesignEvidenceRequirement], list[DesignEvidenceCoverage]]:
    """Require design coverage and retain the purpose of every proposed question."""
    result = experimental_result
    characterization = result.characterization
    decision = result.decision
    records = {item["name"]: item for item in characterization.columns}
    coefficients = {item["name"]: item for item in characterization.coefficients}
    mentioned = {
        name
        for name, item in records.items()
        if item.get("domain") == "biological"
        and re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", study_objective, re.I)
    }
    conditions = sorted(set(decision.coefficientsOfInterest) | mentioned)
    batch_safety = active_batch_safety(result)
    batch_columns = sorted(
        {
            *decision.batchCorrection.batchColumns,
            *(name for item in batch_safety for name in item.batchColumns),
        }
    )
    requirements = [
        DesignEvidenceRequirement(
            requirementId="studyDesign",
            question=(
                "Establish the observed conditions, independent replication, repeated-unit "
                "coverage, exact batch-design estimability, and capture provenance needed "
                "to interpret the study objective."
            ),
            objectiveQuote=study_objective,
            kind="studyDesign",
            columns=sorted(set(conditions) | set(batch_columns)),
        )
    ]
    evidence_ids = characterization_evidence(characterization)
    evidence_ids.update(item.evidenceId for item in result.batchSafety)
    reasons: list[str] = []
    unavailable: list[str] = []
    non_identifiable = False
    if characterization.status == "failed":
        unavailable.append("Study characterization failed")
    if not records:
        unavailable.append("Observed metadata inventory is unavailable")
    for name in conditions:
        record = coefficients.get(name, {})
        if (
            record.get("scope") != "betweenUnit"
            or not record.get("observationUnit")
            or not record.get("unitLevelCounts")
            or not isinstance(record.get("replication", {}).get("sufficient"), bool)
        ):
            unavailable.append(
                f"{name}: observation-unit and replication evidence is unavailable"
            )
            continue
        if record["replication"]["sufficient"] is False:
            reasons.append(
                f"{name}: measured independent replication is insufficient for effect inference"
            )
        if record.get("independentUnit") and len(record.get("groupOrder", [])) >= 2:
            paired = record.get("pairedCoverage", {})
            if not paired:
                unavailable.append(f"{name}: repeated-unit coverage is unavailable")
            elif paired.get("design") == "mixedOrIncomplete":
                reasons.append(
                    f"{name}: measured pairing is mixed or incomplete; no paired effect is estimated"
                )
    if batch_columns:
        for name in conditions:
            matched = [
                item
                for item in batch_safety
                if item.coefficient == name
                and sorted(item.batchColumns) == batch_columns
            ]
            if not matched or any(item.status == "notComputed" for item in matched):
                unavailable.append(
                    f"{name}: estimability for the complete exact batch set is unavailable"
                )
            elif any(item.status == "unsafe" for item in matched):
                non_identifiable = True
                reasons.append(
                    f"{name}: not identifiable under the exact assessed batch design"
                )
    if not conditions:
        reasons.append("No coefficient-level effect inference is authorized")
    if not batch_columns:
        reasons.append("No technical batch correction is proposed")
    if characterization.captureProvenance is None:
        reasons.append(
            "Physical capture identity is unresolved; capture-dependent decisions are not authorized"
        )
    coverage = [
        DesignEvidenceCoverage(
            requirementId="studyDesign",
            status=(
                "failed"
                if characterization.status == "failed"
                else "unsupported"
                if unavailable
                else "nonIdentifiable"
                if non_identifiable
                else "computed"
            ),
            evidenceIds=sorted(evidence_ids),
            reasons=[*unavailable, *reasons],
        )
    ]
    seen: set[str] = set()
    for comparison in characterization.comparisons:
        proposal = comparison.proposal
        identity = hashlib.sha256(
            canonical_json_bytes(proposal.model_dump(exclude={"rationale"}))
        ).hexdigest()
        requirement_id = f"designQuestion:{identity}"
        if requirement_id in seen:
            raise ValueError("Objective comparisons must have unique current proposals")
        seen.add(requirement_id)
        quote = proposal.objectiveQuote or study_objective
        if quote not in f"{study_context}\n{study_objective}":
            raise ValueError(
                "Objective evidence requirements must quote exact study text"
            )
        if (
            proposal.objectiveQuote
            and quote in study_objective
            and not proposal.essential
        ):
            raise ValueError(
                "A question quoting the explicit study objective must remain essential"
            )
        requirements.append(
            DesignEvidenceRequirement(
                requirementId=requirement_id,
                question=proposal.rationale,
                objectiveQuote=quote,
                kind=proposal.purpose,
                columns=[
                    proposal.response,
                    *proposal.explanatoryColumns,
                    *([proposal.conditionedOn] if proposal.conditionedOn else []),
                ],
                observationUnit=proposal.observationUnit,
                independentUnit=proposal.independentUnit or proposal.observationUnit,
                essential=proposal.essential,
            )
        )
        descriptive = comparison.evidence.get("descriptiveDesign", {})
        computed = comparison.status == "computed"
        answer_reasons = list(comparison.reasons)
        if proposal.purpose == "designCoverage":
            computed = descriptive.get("status") == "computed"
            if computed and comparison.status == "unsupported":
                answer_reasons.append(
                    "Descriptive support answers the design question; the association method remains unsupported"
                )
        elif proposal.purpose == "effectEstimation":
            computed = False
            answer_reasons.append(
                "This workflow does not estimate biological effects or test expression hypotheses"
            )
        if any(
            records.get(column, {}).get(field) != value
            for field, recorded in (
                ("kind", "columnKinds"),
                ("domain", "columnDomains"),
            )
            for column, value in comparison.evidence.get(recorded, {}).items()
        ):
            computed = False
            answer_reasons.append(
                "Comparison evidence has different column roles or kinds from the final design"
            )
        coverage.append(
            DesignEvidenceCoverage(
                requirementId=requirement_id,
                status="computed" if computed else "unsupported",
                evidenceIds=[comparison.evidenceId],
                reasons=answer_reasons,
            )
        )
    for quote, names, conditional in requested_design_questions(
        study_context, study_objective, list(records)
    ):
        purpose = requested_design_purpose(quote)
        matches = [
            index
            for index, item in enumerate(characterization.comparisons, start=1)
            if set(names).issubset(
                {
                    item.proposal.response,
                    *item.proposal.explanatoryColumns,
                    item.proposal.conditionedOn,
                }
            )
            and item.proposal.purpose == purpose
            and (
                (
                    item.proposal.conditionedOn is not None
                    if conditional
                    else len(item.proposal.explanatoryColumns) == 2
                )
                and item.proposal.essential
                or purpose == "designCoverage"
                and coverage[index].status == "computed"
                and _measured_joint_design(item)
            )
        ]
        if matches:
            for index in matches:
                # A model's optional label cannot erase an explicit requirement.
                # Reuse its measured question rather than adding a duplicate that
                # would consume another slot in the bounded study contract.
                if not requirements[index].essential:
                    requirements[index] = requirements[index].model_copy(
                        update={"essential": True}
                    )
            continue
        identifier = "requestedDesign:" + hashlib.sha256(quote.encode()).hexdigest()
        requirements.append(
            DesignEvidenceRequirement(
                requirementId=identifier,
                question=quote,
                objectiveQuote=quote,
                kind=purpose,
                columns=names,
            )
        )
        coverage.append(
            DesignEvidenceCoverage(
                requirementId=identifier,
                status="unsupported",
                reasons=[
                    "The explicitly requested joint or conditional comparison has not been nominated and measured; marginal comparisons do not answer it."
                ],
            )
        )
    if len(requirements) > 13:
        raise ValueError(
            "Objective requirements permit one design summary and eight plus four questions"
        )
    return requirements, coverage


def unmet_objective_requirements(
    requirements: list[DesignEvidenceRequirement],
    coverage: list[DesignEvidenceCoverage],
) -> list[str]:
    """Return essential questions with no measured answer of the required kind."""
    measured = {item.requirementId: item for item in coverage}
    unmet = []
    for requirement in requirements:
        item = measured.get(requirement.requirementId)
        satisfied = item is not None and (
            item.status == "computed"
            or item.status == "nonIdentifiable"
            and requirement.kind in {"studyDesign", "designCoverage"}
        )
        if requirement.essential and not satisfied:
            reasons = (
                "; ".join(item.reasons) if item is not None else "evidence is missing"
            )
            unmet.append(f"{requirement.question} Unresolved: {reasons}")
    return unmet
