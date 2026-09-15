"""Bounded, objective-led comparisons of metadata on independent study units."""

import hashlib
import json
import re
from collections.abc import Sequence
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from ...metrics.association import association_pair, coefficient_estimability
from .. import record_io
from .characterization import _json_scalar, _paired_coverage
from .contracts import (
    CaptureProposal,
    CovariateCharacterization,
    CovariateComparison,
    CovariateProposal,
    ExperimentalContextDependencies,
)

DESIGN_ROUND_LIMITS = (8, 4)
MAX_COMBINATIONS = 32
MAX_STRATA = 16
MAX_DESCRIPTIVE_ROWS = 128


def combination_labels(cells: Any, columns: Sequence[str]) -> np.ndarray:
    """Encode exact, already selection-aligned metadata without writing columns."""
    if len(columns) != 2 or len(set(columns)) != 2:
        raise ValueError("A protected combination requires two distinct columns")
    arrays = [np.asarray(cells.fetch(column)) for column in columns]
    if any(array.ndim != 1 or array.shape != arrays[0].shape for array in arrays):
        raise ValueError("Combination columns must align to the same cell selection")
    return _tuple_labels(arrays)


def _tuple_labels(arrays: Sequence[np.ndarray]) -> np.ndarray:
    output: list[str] = []
    for row in zip(*arrays, strict=True):
        encoded = []
        for value in row:
            value = value.item() if isinstance(value, np.generic) else value
            if pd.isna(value) or isinstance(value, float) and not np.isfinite(value):
                raise ValueError("Combination columns contain missing values")
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            encoded.append([type(value).__name__, value])
        output.append(json.dumps(encoded, ensure_ascii=False, separators=(",", ":")))
    return np.asarray(output, dtype=str)


def _proposal_key(proposal: CovariateProposal) -> str:
    return hashlib.sha256(
        record_io.canonical_json_bytes(proposal.model_dump(exclude={"rationale"}))
    ).hexdigest()


def _association(
    frame: pd.DataFrame, response: str, explanatory: str, kinds: dict[str, Any]
) -> dict[str, Any]:
    if len(frame) < 4:
        return {"status": "notComputed", "reason": "fewerThanFourIndependentUnits"}
    for column in (response, explanatory):
        if kinds[column] == "categorical" and (
            frame[column].nunique() < 2 or frame[column].value_counts().min() < 2
        ):
            return {
                "status": "notComputed",
                "reason": "categoricalGroupsRequireTwoIndependentUnits",
            }
    return association_pair(
        frame[response].to_numpy(),
        frame[explanatory].to_numpy(),
        leftKind=kinds[response],
        rightKind=kinds[explanatory],
    )


def _descriptive_design(
    design: pd.DataFrame,
    *,
    columns: Sequence[str],
    independent: str,
    kinds: dict[str, Any],
) -> dict[str, Any]:
    """Count observations and distinct units without treating repeats as replicates."""
    categorical = list(
        dict.fromkeys(name for name in columns if kinds[name] == "categorical")
    )
    output: dict[str, Any] = {
        "status": "computed",
        "observationUnits": len(design),
        "independentUnits": int(design[independent].nunique()),
        "independentUnit": independent,
        "interpretation": (
            "Descriptive counts only. A unit can occur in multiple groups; group "
            "memberships are not disjoint independent replicates or effect estimates."
        ),
        "groupSupport": {},
        "pairedCoverage": {},
        "sharedIndependentUnits": {},
    }
    if any(design[name].nunique() > MAX_COMBINATIONS for name in categorical):
        output.update(status="unsupported", reason="moreThanThirtyTwoCategoricalLevels")
        return output
    for name in categorical:
        rows: list[dict[str, Any]] = []
        memberships: list[tuple[Any, set[Any]]] = []
        for label, subset in design.groupby(name, sort=False, observed=True):
            memberships.append((_json_scalar(label), set(subset[independent])))
            rows.append(
                {
                    "group": _json_scalar(label),
                    "observationUnits": len(subset),
                    "independentUnits": int(subset[independent].nunique()),
                }
            )
        output["groupSupport"][name] = rows
        overlap = [
            {
                "groups": [left, right],
                "independentUnits": len(left_units & right_units),
            }
            for index, (left, left_units) in enumerate(memberships)
            for right, right_units in memberships[index + 1 :]
        ]
        output["sharedIndependentUnits"][name] = {
            "pairs": overlap[:MAX_DESCRIPTIVE_ROWS],
            "truncated": len(overlap) > MAX_DESCRIPTIVE_ROWS,
        }
        if name != independent and len(rows) >= 2:
            output["pairedCoverage"][name] = _paired_coverage(
                design,
                coefficient=name,
                pair_by=independent,
                group_order=list(design[name].unique()),
            )
    if categorical:
        groups = design.groupby(categorical, sort=False, observed=True)
        if groups.ngroups > MAX_DESCRIPTIVE_ROWS:
            output.update(status="unsupported", reason="moreThan128DescriptiveGroups")
        else:
            rows = []
            for labels, subset in groups:
                if not isinstance(labels, tuple):
                    labels = (labels,)
                rows.append(
                    {
                        "groups": dict(
                            zip(categorical, map(_json_scalar, labels), strict=True)
                        ),
                        "observationUnits": len(subset),
                        "independentUnits": int(subset[independent].nunique()),
                    }
                )
            output["jointGroupSupport"] = rows
    output["continuousSummaries"] = {
        name: {
            "minimum": float(design[name].min()),
            "median": float(design[name].median()),
            "maximum": float(design[name].max()),
            "unit": "observationUnit",
        }
        for name in columns
        if kinds[name] == "continuous"
    }
    return output


def compare_covariates(
    cells: Any,
    characterization: CovariateCharacterization,
    proposal: CovariateProposal,
    *,
    selection_identity: dict[str, Any],
) -> CovariateComparison:
    """Compute descriptive evidence; unsupported designs remain explicit."""
    evidence: dict[str, Any] = {}
    reasons: list[str] = []
    records = {record["name"]: record for record in characterization.columns}
    columns = [proposal.response, *proposal.explanatoryColumns]
    if proposal.conditionedOn is not None:
        columns.append(proposal.conditionedOn)
    unit = proposal.observationUnit
    independent = proposal.independentUnit or unit
    requested = list(dict.fromkeys([*columns, unit, independent]))
    evidence["columnKinds"] = {
        name: records.get(name, {}).get("kind") for name in requested
    }
    evidence["columnDomains"] = {
        name: records.get(name, {}).get("domain") for name in requested
    }
    declared_units = {
        (
            record.get("observationUnit"),
            record.get("independentUnit") or record.get("observationUnit"),
        )
        for record in characterization.coefficients
    }
    declared_pair = (unit, independent) in declared_units
    evidence["unitRoles"] = {
        "observationUnit": unit,
        "independentUnit": independent,
        "declaredInCharacterization": declared_pair,
    }
    if any(name not in cells.columns or name not in records for name in requested):
        reasons.append("unknownObservedColumn")
    elif any(
        records[name].get("kind") != "categorical" for name in {unit, independent}
    ):
        reasons.append("observationAndIndependentUnitsMustBeCategorical")
    elif any(
        records[name].get("domain") not in {"design", "technical"}
        and not (declared_pair and records[name].get("domain") == "biological")
        for name in {unit, independent}
    ):
        reasons.append("observationAndIndependentUnitsMustBeDesignOrTechnical")
    if not reasons:
        kinds = {name: records[name].get("kind", "categorical") for name in columns}
        values = {name: np.asarray(cells.fetch(name)) for name in requested}
        frame = pd.DataFrame(values)
        evidence["cells"] = len(frame)
        evidence["missingCellsByColumn"] = {}
        complete = np.ones(len(frame), dtype=bool)
        for name in requested:
            valid = frame[name].notna().to_numpy()
            if kinds.get(name) == "continuous":
                numeric = pd.to_numeric(frame[name], errors="coerce")
                valid = valid & np.isfinite(numeric.to_numpy(dtype=float))
                frame[name] = numeric
            evidence["missingCellsByColumn"][name] = int((~valid).sum())
            complete &= valid
        evidence["missingCells"] = int((~complete).sum())
        frame = frame.loc[complete]
        if frame.empty:
            reasons.append("noCompleteObservations")
        else:
            grouped = frame.groupby(unit, sort=False, observed=True)
            constant = [*proposal.explanatoryColumns, independent]
            if proposal.conditionedOn is not None:
                constant.append(proposal.conditionedOn)
            if kinds[proposal.response] == "categorical":
                constant.append(proposal.response)
            if any(grouped[name].nunique().gt(1).any() for name in set(constant)):
                reasons.append("explanatoryColumnsMustBeConstantWithinObservationUnit")
            if grouped.ngroups >= len(frame):
                reasons.append("observationUnitIsCellIdentifier")
            design = grouped[requested].first().reset_index(drop=True)
            if kinds[proposal.response] == "continuous":
                design[proposal.response] = (
                    grouped[proposal.response].median().to_numpy()
                )
                evidence["responseAggregation"] = "medianPerObservationUnit"
            evidence["observationUnits"] = len(design)
            if not reasons:
                evidence["descriptiveDesign"] = _descriptive_design(
                    design, columns=columns, independent=independent, kinds=kinds
                )
            if independent != unit:
                grouped_independent = design.groupby(
                    independent, sort=False, observed=True
                )
                constant = [*proposal.explanatoryColumns]
                if proposal.conditionedOn is not None:
                    constant.append(proposal.conditionedOn)
                if kinds[proposal.response] == "categorical":
                    constant.append(proposal.response)
                if any(
                    grouped_independent[name].nunique().gt(1).any() for name in constant
                ):
                    reasons.append("withinIndependentUnitComparisonsAreUnsupported")
                reduced = grouped_independent[columns].first().reset_index(drop=True)
                if kinds[proposal.response] == "continuous":
                    reduced[proposal.response] = (
                        grouped_independent[proposal.response].median().to_numpy()
                    )
                    evidence["independentAggregation"] = "medianOfObservationMedians"
                design = reduced
            evidence["independentUnits"] = len(design)
            evidence["unitOfComparison"] = independent
            if len(design) < 4:
                reasons.append("fewerThanFourIndependentUnits")
            if any(
                kinds[name] == "categorical"
                and design[name].nunique() > MAX_COMBINATIONS
                for name in columns
            ):
                reasons.append("moreThanThirtyTwoCategoricalLevels")
            if not reasons:
                if proposal.conditionedOn is not None:
                    condition = proposal.conditionedOn
                    if kinds[condition] != "categorical":
                        reasons.append("continuousConditioningIsUnsupported")
                    elif design[condition].nunique() > MAX_STRATA:
                        reasons.append("moreThanSixteenConditioningStrata")
                    else:
                        strata: list[dict[str, Any]] = []
                        for label, subset in design.groupby(
                            condition, sort=False, observed=True
                        ):
                            result = _association(
                                subset,
                                proposal.response,
                                proposal.explanatoryColumns[0],
                                kinds,
                            )
                            strata.append(
                                {
                                    "stratum": str(label),
                                    "independentUnits": len(subset),
                                    "association": result,
                                }
                            )
                        evidence["strata"] = strata
                        if any(
                            row["association"].get("status") != "ok" for row in strata
                        ):
                            reasons.append("unsupportedConditioningStrata")
                else:
                    evidence["singleAssociations"] = {
                        name: _association(design, proposal.response, name, kinds)
                        for name in proposal.explanatoryColumns
                    }
                    if any(
                        value.get("status") != "ok"
                        for value in evidence["singleAssociations"].values()
                    ):
                        reasons.append("unsupportedSingleAssociation")
                    if len(proposal.explanatoryColumns) == 2:
                        evidence["jointEstimability"] = coefficient_estimability(
                            design[proposal.response].to_numpy(),
                            coefficientKind=cast(
                                Literal["categorical", "continuous"],
                                kinds[proposal.response],
                            ),
                            technicals={
                                name: design[name].to_numpy()
                                for name in proposal.explanatoryColumns
                            },
                            technicalKinds={
                                name: kinds[name]
                                for name in proposal.explanatoryColumns
                            },
                        )
                        if all(
                            kinds[name] == "categorical"
                            for name in proposal.explanatoryColumns
                        ):
                            labels = _tuple_labels(
                                [
                                    design[name].to_numpy()
                                    for name in proposal.explanatoryColumns
                                ]
                            )
                            if len(np.unique(labels)) > MAX_COMBINATIONS:
                                reasons.append("moreThanThirtyTwoJointGroups")
                            else:
                                design = design.assign(_joint=labels)
                                evidence["jointGroupCounts"] = {
                                    str(key): int(value)
                                    for key, value in design["_joint"]
                                    .value_counts()
                                    .items()
                                }
                                evidence["jointAssociation"] = _association(
                                    design,
                                    proposal.response,
                                    "_joint",
                                    {**kinds, "_joint": "categorical"},
                                )
                                evidence["jointGroupEstimability"] = (
                                    coefficient_estimability(
                                        design[proposal.response].to_numpy(),
                                        coefficientKind=cast(
                                            Literal["categorical", "continuous"],
                                            kinds[proposal.response],
                                        ),
                                        technicals={"joint": labels},
                                        technicalKinds={"joint": "categorical"},
                                    )
                                )
                                if evidence["jointAssociation"].get("status") != "ok":
                                    reasons.append("unsupportedJointAssociation")
    if proposal.protectCombination:
        if any(
            records.get(name, {}).get("domain") != "biological"
            or records.get(name, {}).get("kind") != "categorical"
            for name in proposal.explanatoryColumns
        ):
            reasons.append("protectedCombinationsMustBeCategoricalBiology")
    evidence_id = (
        "designComparison:"
        + hashlib.sha256(
            record_io.canonical_json_bytes(
                {
                    "selection": selection_identity,
                    "proposal": proposal.model_dump(),
                    "evidence": evidence,
                    "reasons": reasons,
                }
            )
        ).hexdigest()
    )
    return CovariateComparison(
        proposal=proposal,
        status="unsupported" if reasons else "computed",
        evidence=evidence,
        reasons=list(dict.fromkeys(reasons)),
        evidenceId=evidence_id,
    )


def evaluate_proposals(
    deps: ExperimentalContextDependencies,
    characterization: CovariateCharacterization,
    proposals: Sequence[CovariateProposal],
) -> None:
    """Consume one bounded round, retaining comparisons from earlier rounds."""
    if deps.designRounds >= len(DESIGN_ROUND_LIMITS):
        raise ValueError("Design comparison permits at most two evidence rounds")
    if len(proposals) > DESIGN_ROUND_LIMITS[deps.designRounds]:
        raise ValueError(
            "Design comparison permits eight initial and four follow-up proposals"
        )
    for proposal in proposals:
        if deps.studyObjective and (
            not proposal.objectiveQuote or "purpose" not in proposal.model_fields_set
        ):
            raise ValueError(
                "Objective comparisons require an exact objectiveQuote and explicit purpose"
            )
        if proposal.objectiveQuote and proposal.objectiveQuote not in (
            f"{deps.studyContext}\n{deps.studyObjective}"
        ):
            raise ValueError("Comparison objectiveQuote must copy exact study text")
        if (
            proposal.objectiveQuote
            and proposal.objectiveQuote in deps.studyObjective
            and not proposal.essential
        ):
            raise ValueError(
                "A question quoting the explicit study objective must remain essential"
            )
    deps.designRounds += 1
    previous = {_proposal_key(item.proposal): item for item in deps.comparisons}
    records = {record["name"]: record for record in characterization.columns}
    for proposal in proposals:
        key = _proposal_key(proposal)
        prior = previous.get(key)
        if prior is not None and any(
            records.get(column, {}).get(field) != value
            for field, recorded in (
                ("kind", "columnKinds"),
                ("domain", "columnDomains"),
            )
            for column, value in prior.evidence.get(recorded, {}).items()
        ):
            del previous[key]
        if prior is not None and key in previous:
            declared_pairs = {
                (
                    record.get("observationUnit"),
                    record.get("independentUnit") or record.get("observationUnit"),
                )
                for record in characterization.coefficients
            }
            declared = (
                proposal.observationUnit,
                proposal.independentUnit or proposal.observationUnit,
            ) in declared_pairs
            if (
                prior.evidence.get("unitRoles", {}).get("declaredInCharacterization")
                != declared
            ):
                del previous[key]
        if key not in previous:
            if prior is not None:
                deps.comparisons.remove(prior)
            comparison = compare_covariates(
                deps.cells,
                characterization,
                proposal,
                selection_identity=deps.cellSelection.to_dict(),
            )
            deps.comparisons.append(comparison)
            previous[key] = comparison
    deps.protectedCombinations = []
    for comparison in deps.comparisons:
        if comparison.proposal.protectCombination:
            columns = sorted(comparison.proposal.explanatoryColumns)
            if any(
                records.get(column, {}).get("domain") != "biological"
                or records.get(column, {}).get("kind") != "categorical"
                for column in columns
            ):
                continue
            if columns not in deps.protectedCombinations:
                deps.protectedCombinations.append(columns)
    characterization.comparisons = list(deps.comparisons)
    characterization.captureProvenance = deps.captureProposal


def accept_capture_proposal(
    deps: ExperimentalContextDependencies,
    characterization: CovariateCharacterization,
    proposal: CaptureProposal,
) -> None:
    """Require an observed capture and exact supporting study statements."""
    study = f"{deps.studyContext}\n{deps.studyObjective}"
    quote = proposal.provenanceQuote
    records = {record["name"]: record for record in characterization.columns}
    record = records.get(proposal.column, {})
    if (
        proposal.column not in deps.cells.columns
        or record.get("kind") != "categorical"
        or record.get("domain") not in {"design", "technical"}
    ):
        raise ValueError(
            "Capture proposal must name an observed categorical design column"
        )
    if (
        quote not in study
        or proposal.column not in quote
        or "capture" not in quote.lower()
        or re.search(r"\b(?:not|unknown|unresolved|uncertain)\b", quote, re.I)
    ):
        raise ValueError(
            "Capture proposal requires an exact study quote identifying that column as a physical capture"
        )
    directed = deps.directions.get("physicalCaptureColumn")
    if directed is not None and directed != proposal.column:
        raise ValueError(
            "Capture proposal conflicts with the caller's physical capture"
        )
    references = proposal.referenceCaptures
    if references:
        labels = np.asarray(deps.cells.fetch(proposal.column)).astype(str)
        reference_quote = proposal.referenceProvenanceQuote
        if (
            len(references) < 2
            or len(references) != len(set(references))
            or not set(references).issubset(set(labels))
        ):
            raise ValueError(
                "Reference pool requires at least two distinct observed captures"
            )
        if (
            not reference_quote
            or reference_quote not in study
            or any(name not in reference_quote for name in references)
            or not any(
                word in reference_quote.lower()
                for word in ("reference", "baseline", "control")
            )
            or re.search(
                r"\b(?:not|unknown|unresolved|uncertain)\b", reference_quote, re.I
            )
        ):
            raise ValueError(
                "Reference captures require an exact study quote supporting their baseline role"
            )
    from .qc_evidence import (
        _directed_capture_source,
        _directed_pooled_reference_captures,
    )

    proposed_deps = deps.model_copy(update={"captureProposal": proposal})
    _directed_capture_source(proposed_deps)
    _directed_pooled_reference_captures(proposed_deps)
    deps.captureProposal = proposal
    characterization.captureProvenance = proposal


def canonical_design_choices(
    deps: ExperimentalContextDependencies, decision: Any
) -> dict[str, Any]:
    """Expose only combinations and capture choices supported by tool evidence."""
    from .qc_evidence import (
        _directed_capture_source,
        _directed_pooled_reference_captures,
    )

    combinations = sorted(deps.protectedCombinations)
    records = {
        record["name"]: record
        for record in (
            deps.characterization.columns if deps.characterization is not None else []
        )
    }
    if any(
        records.get(column, {}).get("domain") != "biological"
        or records.get(column, {}).get("kind") != "categorical"
        for columns in combinations
        for column in columns
    ):
        raise ValueError(
            "Protected combinations must remain categorical biological columns"
        )
    supplied = sorted(sorted(columns) for columns in decision.protectedCombinations)
    if supplied and supplied != combinations:
        raise ValueError(
            "Protected combinations must match the evaluated objective-led proposals"
        )
    capture = _directed_capture_source(deps)
    capture_name = None
    if capture is not None:
        capture_name = capture[0]
        if capture_name is None and capture[1] is not None:
            capture_name = capture[1].name
    references = list(_directed_pooled_reference_captures(deps) or ())
    if (
        decision.physicalCaptureColumn not in {None, capture_name}
        or decision.pooledReferenceCaptures
        and decision.pooledReferenceCaptures != references
    ):
        raise ValueError("Capture choices must match provenance-backed tool evidence")
    return {
        "protectedCombinations": combinations,
        "physicalCaptureColumn": capture_name,
        "pooledReferenceCaptures": references,
        "unsupportedProtection": sorted(
            {
                column
                for column in [
                    *decision.coefficientsOfInterest,
                    *decision.batchCorrection.preserveColumns,
                ]
                if records.get(column, {}).get("kind") == "continuous"
            }
        ),
    }
