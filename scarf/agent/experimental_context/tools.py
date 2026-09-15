"""Read-only Pydantic AI tools for experimental context."""

import hashlib
import json
import math
from collections.abc import Sequence
from copy import deepcopy
from functools import wraps
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np

from ...metadata.queries import reduce_observation_units
from ...metrics.association import coefficient_estimability
from ...storage.artifacts import fingerprint_array
from ...storage.refs import ArtifactRef
from ...utils.logging import logger
from .._deps import AGENT_INSTALL_HINT
from ..record_io import canonical_json_bytes
from ..tools import artifact_reference, core_artifact_reference
from ..types import BatchSafetyEvidence, BatchSafetyStatus
from .characterization import characterize_covariates
from .comparisons import (
    DESIGN_ROUND_LIMITS,
    accept_capture_proposal,
    evaluate_proposals,
)
from .contracts import (
    BatchCorrectionPlan,
    CaptureProposal,
    ColumnDomain,
    ContrastPlan,
    ContrastStatus,
    ContrastTest,
    CovariateCharacterization,
    CovariateEvidence,
    CovariateProposal,
    ExperimentalContextDecision,
    ExperimentalContextDependencies,
    InferenceUnit,
    RepresentationEvaluation,
    characterization_evidence,
)
from .qc_evidence import (
    _artifact_evidence_id,
    _hto_artifact_map,
    _hto_identity_columns,
    _offered_qc_profiles,
)
from .requirements import (
    objective_evidence,
    requested_design_purpose,
    requested_design_questions,
)

try:
    from pydantic_ai import ModelRetry, RunContext
    from pydantic_ai.tools import ToolDefinition
except ImportError as exc:
    raise ImportError(AGENT_INSTALL_HINT) from exc


def compact_context_evidence(evidence: CovariateEvidence) -> dict[str, Any]:
    """Present design findings; retain detailed policy measurements in the journal."""
    payload = evidence.model_dump(mode="json")
    characterization = payload["characterization"]
    coefficients = {
        record["name"]: record for record in characterization["coefficients"]
    }
    for index, report in enumerate(characterization["confounding"]):
        source = coefficients.get(report.get("coefficient"), {})
        for name in list(report):
            if name in source and report[name] == source[name]:
                report.pop(name)
        report["coefficientDetails"] = f"coefficient:{report.get('coefficient')}"
        report["details"] = {"section": "confounding", "record_id": str(index)}
    for plan in payload["contrastPlans"]:
        source = coefficients.get(plan["coefficient"], {})
        for name in ("replication", "estimability", "pairedCoverage"):
            if plan[name] == source.get(name):
                plan.pop(name)
        plan["coefficientDetails"] = f"coefficient:{plan['coefficient']}"
    # These tables duplicate the named coefficient records exactly.
    for name in (
        "unitLevelCounts",
        "groupImbalance",
        "missingness",
        "designStructures",
        "pairedCoverage",
        "coefficientEstimability",
    ):
        characterization.pop(name, None)
    for column in characterization["columns"]:
        counts = column.get("levelCounts", [])
        if len(counts) > 8:
            column["levelCounts"] = counts[:8]
            column["levelCountsOmitted"] = len(counts) - 8
            column["details"] = f"column:{column['name']}"

    def omit_examples(value: Any) -> None:
        if isinstance(value, dict):
            examples = value.pop("incompleteExamples", None)
            if examples is not None:
                value["incompleteExamplesInSavedDetails"] = len(examples)
            for child in value.values():
                omit_examples(child)
        elif isinstance(value, list):
            for child in value:
                omit_examples(child)

    omit_examples(characterization)
    omit_examples(payload["contrastPlans"])
    for record in characterization["coefficients"]:
        record.pop("unitLevelCounts", None)
        record["details"] = f"coefficient:{record['name']}"
    shared_safety: dict[str, Any] = {}
    for profile in payload["qcProfiles"]:
        # The later QC decision receives complete thresholds and retention tables.
        # Context needs capture provenance, adverse evidence and design constraints.
        profile.pop("metricSources", None)
        profile.pop("sourceConcordance", None)
        profile.pop("resolvedBounds", None)
        profile["parameters"] = {
            key: value
            for key, value in profile["parameters"].items()
            if key not in {"resolvedBounds", "captureComparisons", "captureSizes"}
        }
        profile["details"] = f"qcProfile:{profile['profileId']}"
        for failure in profile.get("captureFailureEvidence", []):
            safety = failure.pop("conditionAndUnitSafety", [])
            for check in safety:
                if "requiredGroups" in check and "remainingGroups" in check:
                    check["lostGroups"] = [
                        group
                        for group in check["requiredGroups"]
                        if group not in check["remainingGroups"]
                    ]
                for name in (
                    "requiredGroups",
                    "remainingGroups",
                    "observationUnitsByGroup",
                    "independentUnitsByGroup",
                    "quantilesBeforeExclusion",
                    "quantilesAfterExclusion",
                ):
                    check.pop(name, None)
            identity = hashlib.sha256(canonical_json_bytes(safety)).hexdigest()
            shared_safety[identity] = safety
            failure["designSafetyRef"] = identity
            missingness = failure.pop("metricMissingFractions", {})
            failure["metricMissingness"] = {
                "measuredSources": len(missingness),
                "nonzeroOrUnavailable": {
                    key: value for key, value in missingness.items() if value != 0
                },
            }
    payload["captureDesignSafety"] = shared_safety
    payload["savedDetails"] = (
        "inspect_context_evidence returns one exact saved column, coefficient, "
        "comparison, confounding record, or QC policy/capture. Policy thresholds, "
        "individual donor examples and complete distributions remain saved; "
        "summaries are not a substitute for required evidence."
    )
    return payload


async def inspect_context_evidence(
    ctx: RunContext[ExperimentalContextDependencies],
    section: Literal["column", "coefficient", "comparison", "confounding", "qcProfile"],
    record_id: str,
    capture: str | None = None,
) -> dict[str, Any]:
    """Read one saved evidence record without model calls or recomputation.

    Use a column/coefficient name, comparison evidenceId, zero-based confounding
    index, or profileId. For capture-level QC detail supply one exact capture.
    """
    characterization = ctx.deps.characterization
    if characterization is None:
        raise ModelRetry("Inspect covariates before requesting saved evidence")
    if section == "qcProfile":
        profile = ctx.deps.qcProfiles.get(record_id)
        if profile is None:
            raise ModelRetry("Choose an offered profileId")
        if capture is not None:
            failure = next(
                (
                    row
                    for row in profile.captureFailureEvidence
                    if row.capture == capture
                ),
                None,
            )
            if failure is None:
                raise ModelRetry("Choose one capture recorded in this QC profile")
            bounds = profile.resolvedBounds
            return {
                "profileId": record_id,
                "capture": failure.model_dump(mode="json"),
                "resolvedBounds": [row for row in bounds if row.get("group") == capture]
                if isinstance(bounds, list)
                else deepcopy(bounds),
            }
        result = profile.model_dump(mode="json")
        result.pop("captureFailureEvidence", None)
        result.pop("metricSources", None)
        result.pop("sourceConcordance", None)
        result["parameters"].pop("captureComparisons", None)
        result["parameters"].pop("resolvedBounds", None)
        if profile.sampleColumn is not None or profile.sampleArtifact is not None:
            result.pop("resolvedBounds", None)
            result["captureDetailsRequired"] = (
                "Supply one capture to retrieve its exact thresholds and exclusion safety"
            )
        return result
    if capture is not None:
        raise ModelRetry("A capture can be requested only for a QC profile")
    if section == "comparison":
        for comparison in characterization.comparisons:
            if comparison.evidenceId == record_id:
                return comparison.model_dump(mode="json")
    elif section == "confounding":
        if record_id.isdecimal() and int(record_id) < len(characterization.confounding):
            return deepcopy(characterization.confounding[int(record_id)])
        raise ModelRetry(
            "For section='confounding', record_id must be a zero-based index "
            "string, not an evidence ID. Available record_id values: "
            + json.dumps(
                [str(index) for index in range(len(characterization.confounding))]
            )
        )
    else:
        records = (
            characterization.columns
            if section == "column"
            else characterization.coefficients
        )
        for record in records:
            if record["name"] == record_id:
                return deepcopy(record)
    raise ModelRetry("Choose an exact record from the saved context summary")


def capture_repair_inputs(
    deps: ExperimentalContextDependencies,
) -> dict[str, Any] | None:
    """Offer the exact assessed design for one missing-provenance repair."""
    saved_directions = deps.characterizationInputs.get("directions")
    tested_batch_sets = {
        tuple(sorted(item.batchColumns)) for item in deps.batchSafety.values()
    }
    if (
        deps.designRounds < len(DESIGN_ROUND_LIMITS)
        or deps.captureProposal is not None
        or not isinstance(saved_directions, dict)
        or len(tested_batch_sets) > 1
    ):
        return None
    return {
        "column_domains": deepcopy(saved_directions.get("columnDomains", {})),
        "coefficients_of_interest": list(
            saved_directions.get("coefficientsOfInterest", [])
        ),
        "units_of_inference": deepcopy(saved_directions.get("unitsOfInference", {})),
        "batch_columns": list(next(iter(tested_batch_sets), ())),
        "proposals": [],
    }


def model_evidence_tool(function: Any) -> Any:
    """Keep complete tool results in state and send a deduplicated model view."""

    @wraps(function)
    async def invoke(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = await function(*args, **kwargs)
        payload = compact_context_evidence(result)
        context = args[0] if args else kwargs["ctx"]
        repair_inputs = capture_repair_inputs(context.deps)
        payload["captureRepairAvailable"] = repair_inputs is not None
        if payload["captureRepairAvailable"]:
            payload["captureRepairInstructions"] = (
                "analyze_experimental_design can validate capture_proposal using an "
                "exact supplied study quote. Keep the assessed domains, coefficients, "
                "units and batch columns unchanged and pass proposals=[]. This only "
                "adds capture provenance and dependent QC evidence; it does not "
                "authorize another covariate comparison round."
            )
            payload["captureRepairInputs"] = repair_inputs
        payload["requestedComparisons"] = [
            {
                "question": quote,
                "columns": columns,
                "conditional": conditional,
                "purpose": requested_design_purpose(quote),
                "essential": True,
                "completionEvidence": "evidenceRequirements and evidenceCoverage",
            }
            for quote, columns, conditional in requested_design_questions(
                context.deps.studyContext,
                context.deps.studyObjective,
                [row["name"] for row in result.characterization.columns],
            )
        ]
        return payload

    return invoke


def persist_context_evidence(deps: ExperimentalContextDependencies, key: str) -> None:
    """Commit completed context measurements to the owning stage journal."""
    if deps.checkpointWrite is not None:
        deps.checkpointWrite(
            key,
            {
                "state": deps.model_dump(mode="json"),
                "characterizationInputs": deps.characterizationInputs,
            },
        )


def restore_context_evidence(deps: ExperimentalContextDependencies) -> bool:
    """Restore complete evidence rounds without resetting proposal allowances."""
    if deps.checkpointRead is None:
        return False
    restored = False
    for key in ("inspection", "design1", "design2", "capture"):
        saved = deps.checkpointRead(key)
        if saved is None:
            continue
        state = ExperimentalContextDependencies.model_validate(saved["state"])
        for name, field in ExperimentalContextDependencies.model_fields.items():
            if not field.exclude:
                setattr(deps, name, getattr(state, name))
        deps.characterizationInputs = saved["characterizationInputs"]
        restored = True
    return restored


def characterize_context(
    deps: ExperimentalContextDependencies, directions: dict[str, Any]
) -> CovariateCharacterization:
    """Reuse characterization for the exact frozen stage and declared design."""
    metadata = {}
    for column in deps.cells.columns:
        values = np.asarray(deps.cells.fetch(column))
        metadata[column] = (
            hashlib.sha256(
                repr(
                    [(type(value).__name__, value) for value in values.tolist()]
                ).encode()
            ).hexdigest()
            if values.dtype.hasobject
            else fingerprint_array(values)
        )
    inputs = {
        "directions": deepcopy(directions),
        "cellSelection": deps.cellSelection.to_dict(),
        "studyContext": deps.studyContext,
        "studyObjective": deps.studyObjective,
        "metadata": metadata,
    }
    if deps.characterization is not None and deps.characterizationInputs == inputs:
        return deps.characterization
    result = characterize_covariates(
        deps.store,
        cellSelection=deps.cellSelection,
        studyContext=f"{deps.studyContext}\nStudy objective: {deps.studyObjective}",
        model=None,
        directions=directions,
        groupingArtifacts=_hto_artifact_map(deps),
        inventory=deps.inventoryData,
    )
    if result.status != "failed":
        deps.characterization = result
        deps.characterizationInputs = inputs
    return result


def _prepare_experimental_context_tool(
    ctx: RunContext[ExperimentalContextDependencies],
    tool_definition: ToolDefinition,
) -> ToolDefinition | None:
    """Expose inspection once and at most two ordered design evidence rounds."""
    completed_calls = set(ctx.deps.toolCalls)
    if tool_definition.name == "inspect_cell_covariates":
        return None if tool_definition.name in completed_calls else tool_definition
    if tool_definition.name == "analyze_experimental_design":
        if (
            "inspect_cell_covariates" not in completed_calls
            or ctx.deps.designRounds >= len(DESIGN_ROUND_LIMITS)
            and capture_repair_inputs(ctx.deps) is None
        ):
            return None
        return tool_definition
    if tool_definition.name == "score_current_representation":
        if (
            "analyze_experimental_design" not in completed_calls
            or tool_definition.name in completed_calls
        ):
            return None
        characterization = ctx.deps.characterization
        if characterization is not None and not any(
            record.get("domain") == "technical" and record.get("kind") == "categorical"
            for record in characterization.columns
        ):
            return None
        return tool_definition
    return tool_definition


def contrast_plans_from_characterization(
    characterization: CovariateCharacterization,
) -> list[ContrastPlan]:
    """Build deterministic test licenses from bounded coefficient evidence."""
    reports = {
        report.get("coefficient"): report
        for report in characterization.confounding
        if isinstance(report.get("coefficient"), str)
    }
    plans: list[ContrastPlan] = []
    for record in characterization.coefficients:
        coefficient = record.get("name")
        if not isinstance(coefficient, str):
            continue
        report = reports.get(coefficient, {})
        raw_groups = record.get("groupOrder")
        group_order = (
            list(raw_groups)
            if isinstance(raw_groups, list)
            and all(
                isinstance(value, str | int | float | bool)
                and not (isinstance(value, float) and not math.isfinite(value))
                for value in raw_groups
            )
            else []
        )
        sample_by = record.get("observationUnit")
        sample_by = sample_by if isinstance(sample_by, str) else None
        independent_unit = record.get("independentUnit")
        independent_unit = (
            independent_unit if isinstance(independent_unit, str) else None
        )
        between_unit = record.get("scope") == "betweenUnit"
        replication = dict(record.get("replication") or {})
        replication_passed = replication.get("sufficient") is True
        estimability = dict(
            record.get("estimability") or report.get("estimability") or {}
        )
        estimability_passed = (
            estimability.get("status") == "ok"
            and estimability.get("coefficientEstimable") is True
            and estimability.get("rankDeficient") is not True
        )
        paired_coverage = dict(record.get("pairedCoverage") or {})
        pair_by: str | None = None
        paired_passed: bool | None = None
        mixed_independent_design = False
        if independent_unit is not None:
            if paired_coverage.get("complete") is True:
                pair_by = independent_unit
                paired_passed = True
            elif paired_coverage.get("betweenIndependentUnits") is True:
                sample_by = independent_unit
            else:
                pair_by = independent_unit
                paired_passed = False
                mixed_independent_design = True

        reasons: list[str] = []
        needs_input = False
        if record.get("kind") != "categorical":
            reasons.append("coefficientRequiresExplicitCategoricalGroups")
            needs_input = True
        if not between_unit:
            reasons.append("coefficientIsNotBetweenUnit")
        if sample_by is None:
            reasons.append("sampleByIsUnresolved")
            needs_input = True
        if len(group_order) < 2:
            reasons.append("fewerThanTwoObservedGroups")
            needs_input = True
        if record.get("groupCountsTruncated") is True:
            reasons.append("groupOrderIsTruncated")
            needs_input = True
        if not replication_passed:
            reasons.append("insufficientIndependentReplication")
        if not estimability_passed:
            reasons.append("coefficientIsNotEstimable")

        test: ContrastTest | None = None
        if pair_by is not None:
            if len(group_order) != 2:
                reasons.append("pairedTestsRequireExactlyTwoGroups")
            else:
                test = "wilcoxon"
            if paired_passed is not True:
                reasons.append("pairedCoverageIsIncomplete")
        elif mixed_independent_design:
            reasons.append("independentUnitStructureIsMixed")
        elif len(group_order) == 2:
            test = "mann_whitney"
        elif len(group_order) >= 3:
            test = "kruskal_wallis"

        reasons = list(dict.fromkeys(reasons))
        status: ContrastStatus = (
            "licensed" if not reasons else "needsInput" if needs_input else "blocked"
        )
        evidence_ids = [
            f"column:{coefficient}",
            f"coefficient:{coefficient}",
            f"estimability:{coefficient}",
        ]
        plans.append(
            ContrastPlan(
                coefficient=coefficient,
                groupOrder=group_order,
                sampleBy=sample_by,
                pairBy=pair_by,
                test=test,
                status=status,
                betweenUnitDesign=between_unit,
                replicationPassed=replication_passed,
                estimabilityPassed=estimability_passed,
                pairedCoveragePassed=paired_passed,
                replication=replication,
                estimability=estimability,
                pairedCoverage=paired_coverage,
                blockedReasons=reasons,
                evidenceId=f"contrastPlan:{coefficient}:{status}",
                evidenceIds=evidence_ids,
            )
        )
    return plans


async def inspect_cell_covariates(
    ctx: RunContext[ExperimentalContextDependencies],
) -> CovariateEvidence:
    """Inspect cell metadata without making model-driven choices or writing data."""
    logger.info(
        "Experimental Context covariate inspection started: "
        f"cellSelection={ctx.deps.cellSelection.artifact_id}"
    )
    ctx.deps.htoIdentityColumns = _hto_identity_columns(ctx.deps)
    characterization = characterize_context(ctx.deps, ctx.deps.directions)
    ctx.deps.characterization = characterization
    qc_profiles = _offered_qc_profiles(ctx.deps)
    contrast_plans = contrast_plans_from_characterization(characterization)
    ctx.deps.contrastPlans = {plan.coefficient: plan for plan in contrast_plans}
    evidence_ids = characterization_evidence(characterization)
    evidence_ids.update(profile.evidenceId for profile in qc_profiles)
    evidence_ids.update(source.sourceId for source in ctx.deps.qcMetricSources)
    evidence_ids.update(item.evidenceId for item in ctx.deps.qcSourceConcordance)
    evidence_ids.update(plan.evidenceId for plan in contrast_plans)
    evidence_ids.update(
        failure.evidenceId
        for profile in qc_profiles
        for failure in profile.captureFailureEvidence
    )
    evidence_ids.update(
        f"htoIdentity:{column}" for column in ctx.deps.htoIdentityColumns
    )
    evidence_ids.update(
        _artifact_evidence_id(source) for source in ctx.deps.htoIdentityArtifacts
    )
    ctx.deps.evidenceIds.update(evidence_ids)
    ctx.deps.toolCalls.append("inspect_cell_covariates")
    persist_context_evidence(ctx.deps, "inspection")
    logger.info(
        "Experimental Context covariate inspection completed: "
        f"status={characterization.status}, "
        f"columns={len(characterization.columns)}, "
        f"coefficients={len(characterization.coefficients)}, "
        f"qcProfiles={len(qc_profiles)}, "
        f"htoIdentities={len(ctx.deps.htoIdentityColumns)}, "
        f"evidence={len(evidence_ids)}"
    )
    return CovariateEvidence(
        characterization=characterization,
        qcProfiles=qc_profiles,
        qcMetricSources=ctx.deps.qcMetricSources,
        qcSourceConcordance=ctx.deps.qcSourceConcordance,
        contrastPlans=contrast_plans,
        htoIdentityColumns=ctx.deps.htoIdentityColumns,
        htoIdentityArtifacts=ctx.deps.htoIdentityArtifacts,
        evidenceIds=sorted(evidence_ids),
    )


def _batch_safety_evidence(
    deps: ExperimentalContextDependencies,
    characterization: CovariateCharacterization,
    *,
    coefficients: Sequence[str],
    batch_columns: Sequence[str],
) -> list[BatchSafetyEvidence]:
    column_records = {
        record.get("name"): record
        for record in characterization.columns
        if isinstance(record.get("name"), str)
    }
    coefficient_records = {
        record.get("name"): record
        for record in characterization.coefficients
        if isinstance(record.get("name"), str)
    }
    confounding_reports = {
        report.get("coefficient"): report
        for report in characterization.confounding
        if isinstance(report.get("coefficient"), str)
    }
    canonical_batch_columns = sorted(batch_columns)
    batch_safety: list[BatchSafetyEvidence] = []
    for coefficient in coefficients:
        if not canonical_batch_columns:
            break
        coefficient_record = coefficient_records.get(coefficient)
        report = confounding_reports.get(coefficient)
        coefficient_kind = (
            coefficient_record.get("kind") if coefficient_record is not None else None
        )
        if coefficient_kind not in {"categorical", "continuous"}:
            coefficient_kind = None
        observation_unit = (
            report.get("observationUnit")
            if report is not None
            else (
                coefficient_record.get("observationUnit")
                if coefficient_record is not None
                else None
            )
        )
        unit_constant = {
            pair.get("technical")
            for pair in (report.get("pairs", []) if report is not None else [])
            if isinstance(pair.get("technical"), str)
        }
        effective_batch_columns = [
            name for name in canonical_batch_columns if name in unit_constant
        ]
        estimability: dict[str, Any]
        if (
            coefficient_record is None
            or coefficient_record.get("scope") != "betweenUnit"
            or report is None
            or not isinstance(observation_unit, str)
            or coefficient_kind is None
        ):
            estimability = {
                "status": "notComputed",
                "reason": "unresolvedCoefficientDesign",
            }
        else:
            try:
                design = reduce_observation_units(
                    deps.cells,
                    observation_unit,
                    [coefficient, *effective_batch_columns],
                    cell_key="I",
                )
                estimability = coefficient_estimability(
                    design[coefficient].to_numpy(),
                    coefficientKind=coefficient_kind,
                    technicals={
                        name: design[name].to_numpy()
                        for name in effective_batch_columns
                    },
                    technicalKinds={
                        name: column_records[name]["kind"]
                        for name in effective_batch_columns
                    },
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.debug(
                    "Experimental Context batch estimability was not computed: "
                    f"errorType={type(exc).__name__}"
                )
                estimability = {
                    "status": "notComputed",
                    "reason": type(exc).__name__,
                }
        if estimability.get("status") != "ok":
            safety_status: BatchSafetyStatus = "notComputed"
        elif estimability.get("coefficientEstimable") is True and not bool(
            estimability.get("rankDeficient")
        ):
            safety_status = "safe"
        else:
            safety_status = "unsafe"
        # Sparse descriptive associations cannot erase a computed design constraint.
        joint_checks = [
            {
                "evidenceId": comparison.evidenceId,
                **comparison.evidence["jointGroupEstimability"],
            }
            for comparison in characterization.comparisons
            if comparison.proposal.response == coefficient
            and comparison.proposal.observationUnit == observation_unit
            and comparison.proposal.conditionedOn is None
            and all(
                column_records.get(name, {}).get("kind") == kind
                for name, kind in comparison.evidence.get("columnKinds", {}).items()
            )
            and set(comparison.proposal.explanatoryColumns).issubset(
                canonical_batch_columns
            )
            and "jointGroupEstimability" in comparison.evidence
        ]
        if joint_checks:
            estimability = {**estimability, "jointStratumChecks": joint_checks}
            if any(
                check.get("status") == "ok"
                and check.get("coefficientEstimable") is False
                for check in joint_checks
            ):
                safety_status = "unsafe"
        batch_token = ",".join(canonical_batch_columns)
        safety = BatchSafetyEvidence(
            coefficient=coefficient,
            coefficientKind=coefficient_kind,
            observationUnit=(
                observation_unit if isinstance(observation_unit, str) else None
            ),
            batchColumns=canonical_batch_columns,
            unitConstantBatchColumns=effective_batch_columns,
            status=safety_status,
            estimability=estimability,
            evidenceId=f"batchEstimability:{coefficient}:{batch_token}",
        )
        batch_safety.append(safety)
        deps.batchSafety[safety.evidenceId] = safety
    return batch_safety


async def analyze_experimental_design(
    ctx: RunContext[ExperimentalContextDependencies],
    column_domains: dict[str, ColumnDomain],
    coefficients_of_interest: list[str],
    units_of_inference: dict[str, InferenceUnit],
    batch_columns: list[str],
    proposals: list[CovariateProposal] | None = None,
    capture_proposal: CaptureProposal | None = None,
) -> CovariateEvidence:
    """Validate proposed domains and inference units and compute confounding.

    Args:
        ctx: Pydantic AI run context containing the existing datastore.
        column_domains: Domain assignment for each metadata column under review.
        coefficients_of_interest: Biological columns representing study contrasts.
        units_of_inference: Observation and independent units for each coefficient.
        batch_columns: Exact technical columns proposed for Harmony evaluation.
        proposals: Up to eight initial or four follow-up objective-led comparisons.
            Each uses distinct response/explanatory/conditioning columns: one or
            two explanatory columns without conditioning, or one explanatory
            column with one categorical conditioning column. Observation and
            independent units do not count toward the three-column limit.
            Two explanatory columns together with conditioning are unsupported.
            One explanatory column within one categorical stratum is supported
            subject to observed unit support; descriptive joint counts can remain
            available when the association method is unsupported. Do not discard
            a scientific question or a unit identity just to fit this schema.
        capture_proposal: Exact capture and baseline identities supported by study prose.
            After two comparison rounds, only this provenance repair is permitted:
            keep the assessed design unchanged and pass no proposals. The tool
            validates the supplied quote and refreshes dependent QC evidence.
    """
    logger.info(
        "Experimental Context design analysis started: "
        f"domains={len(column_domains)}, "
        f"coefficients={len(coefficients_of_interest)}, "
        f"inferenceUnits={len(units_of_inference)}, "
        f"batchColumns={len(batch_columns)}"
    )
    capture_repair = ctx.deps.designRounds >= len(DESIGN_ROUND_LIMITS)
    if capture_repair and (capture_proposal is None or proposals):
        raise ModelRetry(
            "Design comparison permits at most two evidence rounds. Only capture "
            "provenance can now be repaired: supply capture_proposal, proposals=[], "
            "and the exact unchanged assessed design."
        )
    if (
        not capture_repair
        and len(proposals or ()) > DESIGN_ROUND_LIMITS[ctx.deps.designRounds]
    ):
        raise ModelRetry(
            "Design comparison permits eight initial and four follow-up proposals"
        )
    directions = dict(ctx.deps.directions)
    directed_domains = dict(column_domains)
    directed_domains.update(dict(directions.get("columnDomains") or {}))
    directions["columnDomains"] = directed_domains
    directed_coefficients = list(
        dict.fromkeys(
            [
                *coefficients_of_interest,
                *(directions.get("coefficientsOfInterest") or []),
            ]
        )
    )
    directions["coefficientsOfInterest"] = directed_coefficients
    directed_units = {
        name: unit.model_dump(exclude_none=True)
        for name, unit in units_of_inference.items()
    }
    directed_units.update(dict(directions.get("unitsOfInference") or {}))
    directions["unitsOfInference"] = directed_units

    proposed_batch_columns = list(batch_columns)
    directed_batch_columns = directions.get("batchColumns")
    if directed_batch_columns is not None:
        if not isinstance(directed_batch_columns, list) or any(
            not isinstance(value, str) or not value.strip()
            for value in directed_batch_columns
        ):
            raise ModelRetry(
                "directions.batchColumns must be a list of exact metadata columns"
            )
        if len(set(directed_batch_columns)) != len(directed_batch_columns):
            raise ModelRetry("directions.batchColumns must be unique")
        if proposed_batch_columns != directed_batch_columns:
            logger.info(
                "Experimental Context replaced model-proposed batch columns with "
                "the exact directed columns"
            )
        proposed_batch_columns = list(directed_batch_columns)
    canonical_batch_columns = sorted(set(proposed_batch_columns))
    if len(canonical_batch_columns) != len(proposed_batch_columns):
        logger.warning(
            "Experimental Context rejected duplicate proposed batch columns: "
            f"{proposed_batch_columns[:20]}"
        )
        raise ModelRetry("Proposed batch columns must be unique")
    inspected_records = {
        record.get("name"): record
        for record in (
            ctx.deps.characterization.columns
            if ctx.deps.characterization is not None
            else []
        )
        if isinstance(record.get("name"), str)
    }
    if ctx.deps.characterization is not None:
        for batch_column in canonical_batch_columns:
            inspected = inspected_records.get(batch_column)
            if inspected is None:
                logger.warning(
                    "Experimental Context rejected unknown proposed batch column "
                    f"before design recomputation: {batch_column!r}"
                )
                raise ModelRetry(f"Unknown batch column {batch_column!r}")
            proposed_domain = directed_domains.get(
                batch_column,
                inspected.get("domain"),
            )
            if proposed_domain != "technical":
                logger.warning(
                    "Experimental Context rejected proposed batch column before "
                    f"design recomputation: {batch_column!r}, "
                    f"domain={proposed_domain!r}, required='technical'"
                )
                raise ModelRetry(
                    f"Batch column {batch_column!r} must be classified as technical"
                )
            if inspected.get("kind") != "categorical":
                logger.warning(
                    "Experimental Context rejected proposed batch column before "
                    f"design recomputation: {batch_column!r}, "
                    f"kind={inspected.get('kind')!r}, required='categorical'"
                )
                raise ModelRetry(
                    f"Batch column {batch_column!r} must be categorical for Harmony"
                )

    if capture_repair:
        characterization = ctx.deps.characterization
        if (
            characterization is None
            or directions != ctx.deps.characterizationInputs.get("directions")
        ):
            raise ModelRetry(
                "Capture provenance repair requires the exact saved domains, "
                "coefficients and inference units; it cannot change the design."
            )
        tested_batch_sets = {
            tuple(sorted(item.batchColumns)) for item in ctx.deps.batchSafety.values()
        }
        if tested_batch_sets != {tuple(canonical_batch_columns)} and (
            tested_batch_sets or canonical_batch_columns
        ):
            raise ModelRetry(
                "Capture provenance repair requires one unambiguous, unchanged "
                "assessed batch-column set. Alternative designs cannot be combined."
            )
        if ctx.deps.captureProposal is not None:
            raise ModelRetry(
                "Capture provenance is already committed; reuse its saved evidence."
            )
        logger.info(
            "Experimental Context capture provenance: validating the supplied study "
            "statement and refreshing dependent QC; reusing measured study design"
        )
    else:
        characterization = characterize_context(ctx.deps, directions)
    if characterization.status == "failed":
        rejection = "; ".join(characterization.notes).strip()
        logger.warning(
            "Experimental Context design characterization rejected the proposed "
            f"directions: {rejection[:1000]}; "
            f"domainColumns={sorted(column_domains)[:50]}, "
            f"coefficients={coefficients_of_interest[:50]}, "
            f"inferenceUnits={sorted(units_of_inference)[:50]}"
        )
        raise ModelRetry("; ".join(characterization.notes))

    # Retain the validated deterministic work even when the proposed Harmony
    # columns below are rejected. A bounded retry or resumed decision can reuse
    # the evidence without rescanning metadata or accepting an unsafe choice.
    ctx.deps.characterization = characterization
    try:
        if not capture_repair:
            evaluate_proposals(ctx.deps, characterization, proposals or ())
        if capture_proposal is not None:
            accept_capture_proposal(ctx.deps, characterization, capture_proposal)
    except ValueError as exc:
        raise ModelRetry(str(exc)) from exc
    if not ctx.deps.htoIdentityColumns:
        ctx.deps.htoIdentityColumns = _hto_identity_columns(ctx.deps)
    try:
        qc_profiles = _offered_qc_profiles(ctx.deps, characterization)
    except Exception:
        if capture_repair:
            ctx.deps.captureProposal = None
            characterization.captureProvenance = None
        raise
    contrast_plans = contrast_plans_from_characterization(characterization)
    ctx.deps.contrastPlans = {plan.coefficient: plan for plan in contrast_plans}
    evidence_ids = characterization_evidence(characterization)
    evidence_ids.update(profile.evidenceId for profile in qc_profiles)
    evidence_ids.update(source.sourceId for source in ctx.deps.qcMetricSources)
    evidence_ids.update(item.evidenceId for item in ctx.deps.qcSourceConcordance)
    evidence_ids.update(plan.evidenceId for plan in contrast_plans)
    evidence_ids.update(
        failure.evidenceId
        for profile in qc_profiles
        for failure in profile.captureFailureEvidence
    )
    evidence_ids.update(
        f"htoIdentity:{column}" for column in ctx.deps.htoIdentityColumns
    )
    evidence_ids.update(
        _artifact_evidence_id(source) for source in ctx.deps.htoIdentityArtifacts
    )
    ctx.deps.evidenceIds.update(evidence_ids)

    column_records = {
        record.get("name"): record
        for record in characterization.columns
        if isinstance(record.get("name"), str)
    }
    for batch_column in canonical_batch_columns:
        record = column_records.get(batch_column)
        if record is None:
            logger.warning(
                "Experimental Context rejected unknown proposed batch column: "
                f"{batch_column!r}"
            )
            raise ModelRetry(f"Unknown batch column {batch_column!r}")
        if record.get("domain") != "technical":
            logger.warning(
                "Experimental Context rejected proposed batch column "
                f"{batch_column!r}: domain={record.get('domain')!r}, "
                "required='technical'"
            )
            raise ModelRetry(
                f"Batch column {batch_column!r} must be classified as technical"
            )
        if record.get("kind") != "categorical":
            logger.warning(
                "Experimental Context rejected proposed batch column "
                f"{batch_column!r}: kind={record.get('kind')!r}, "
                "required='categorical'"
            )
            raise ModelRetry(
                f"Batch column {batch_column!r} must be categorical for Harmony"
            )

    batch_safety = (
        list(ctx.deps.batchSafety.values())
        if capture_repair
        else _batch_safety_evidence(
            ctx.deps,
            characterization,
            coefficients=directed_coefficients,
            batch_columns=canonical_batch_columns,
        )
    )

    evidence_ids.update(item.evidenceId for item in batch_safety)
    ctx.deps.evidenceIds.update(evidence_ids)
    ctx.deps.toolCalls.append("analyze_experimental_design")
    persist_context_evidence(
        ctx.deps, "capture" if capture_repair else f"design{ctx.deps.designRounds}"
    )
    safety_counts = {
        status: sum(item.status == status for item in batch_safety)
        for status in ("safe", "unsafe", "notComputed")
    }
    logger.info(
        "Experimental Context design analysis completed: "
        f"status={characterization.status}, "
        f"batchSafetySafe={safety_counts['safe']}, "
        f"batchSafetyUnsafe={safety_counts['unsafe']}, "
        f"batchSafetyNotComputed={safety_counts['notComputed']}, "
        f"qcProfiles={len(qc_profiles)}, evidence={len(evidence_ids)}"
    )
    requirements, coverage = (
        objective_evidence(
            study_context=ctx.deps.studyContext,
            study_objective=ctx.deps.studyObjective,
            experimental_result=SimpleNamespace(
                characterization=characterization,
                decision=ExperimentalContextDecision(
                    coefficientsOfInterest=directed_coefficients,
                    batchCorrection=BatchCorrectionPlan(
                        action="needsInput",
                        batchColumns=canonical_batch_columns,
                    ),
                ),
                batchSafety=batch_safety,
            ),
        )
        if ctx.deps.studyObjective
        else ([], [])
    )
    return CovariateEvidence(
        characterization=characterization,
        batchSafety=batch_safety,
        evidenceRequirements=requirements,
        evidenceCoverage=coverage,
        qcProfiles=qc_profiles,
        qcMetricSources=ctx.deps.qcMetricSources,
        qcSourceConcordance=ctx.deps.qcSourceConcordance,
        contrastPlans=contrast_plans,
        htoIdentityColumns=ctx.deps.htoIdentityColumns,
        htoIdentityArtifacts=ctx.deps.htoIdentityArtifacts,
        evidenceIds=sorted(evidence_ids),
    )


async def score_current_representation(
    ctx: RunContext[ExperimentalContextDependencies],
    batch_column: str,
    biological_column: str | None = None,
) -> RepresentationEvaluation:
    """Score one explicitly supplied graph without changing datastore state.

    Args:
        ctx: Pydantic AI run context containing the existing datastore.
        batch_column: Categorical technical column used to assess batch mixing.
        biological_column: Optional biological label used to assess preservation.
    """
    logger.info(
        "Experimental Context representation scoring started: "
        f"graphSupplied={ctx.deps.neighbors is not None}, "
        f"biologicalLabelSpecified={biological_column is not None}"
    )
    store = ctx.deps.store
    available_columns = set(store.cells.columns)
    if batch_column not in available_columns:
        raise ModelRetry(f"Unknown batch column {batch_column!r}")
    if biological_column is not None and biological_column not in available_columns:
        raise ModelRetry(f"Unknown biological column {biological_column!r}")
    characterization = ctx.deps.characterization
    if characterization is not None:
        batch_record = next(
            (
                record
                for record in characterization.columns
                if record.get("name") == batch_column
            ),
            None,
        )
        if (
            batch_record is None
            or batch_record.get("domain") != "technical"
            or batch_record.get("kind") != "categorical"
        ):
            raise ModelRetry(
                "Representation scoring requires a characterized categorical "
                "technical batch column"
            )

    neighbors = core_artifact_reference(ctx.deps.neighbors)
    connectivity = core_artifact_reference(ctx.deps.connectivityMap)
    if neighbors is None:
        evaluation = RepresentationEvaluation(
            cellSelection=(
                artifact_reference(ctx.deps.cellSelection)
                if ctx.deps.cellSelection is not None
                else None
            ),
            notes=["No exact neighbors artifact was supplied"],
        )
        ctx.deps.currentRepresentation = evaluation
        ctx.deps.toolCalls.append("score_current_representation")
        logger.info(
            "Experimental Context representation scoring skipped: "
            "no current neighbors artifact"
        )
        return evaluation
    if not isinstance(neighbors, ArtifactRef) or neighbors.kind != "neighbors":
        raise ModelRetry("neighbors must identify an exact neighbors artifact")
    if connectivity is not None and (
        not isinstance(connectivity, ArtifactRef)
        or connectivity.kind not in {"connectivity_map", "integrated_graph"}
    ):
        raise ModelRetry(
            "connectivity_map must identify an exact connectivity graph artifact"
        )

    metrics: dict[str, float] = {}
    notes: list[str] = []
    evidence_ids: list[str] = []
    neighbor_route = f"assay:{neighbors.assay}:neighbors:{neighbors.artifact_id}"
    try:
        value = float(store.metric_ilisi(batch_column, neighbors))
        if math.isfinite(value):
            metrics[f"iLISI:{batch_column}"] = value
            evidence_ids.append(f"metric:iLISI:{batch_column}:{neighbor_route}")
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        logger.debug(
            "Experimental Context iLISI scoring was unavailable: "
            f"errorType={type(exc).__name__}"
        )
        notes.append(f"iLISI could not be scored: {exc}")
    try:
        value = float(store.metric_proportional_batch_mixing(batch_column, neighbors))
        if math.isfinite(value):
            metrics[f"proportionalBatchMixing:{batch_column}"] = value
            evidence_ids.append(
                f"metric:proportionalBatchMixing:{batch_column}:{neighbor_route}"
            )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        logger.debug(
            "Experimental Context batch-mixing scoring was unavailable: "
            f"errorType={type(exc).__name__}"
        )
        notes.append(f"Proportional batch mixing could not be scored: {exc}")
    if biological_column is not None:
        try:
            value = float(store.metric_clisi(biological_column, neighbors))
            if math.isfinite(value):
                metrics[f"cLISI:{biological_column}"] = value
                evidence_ids.append(
                    f"metric:cLISI:{biological_column}:{neighbor_route}"
                )
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            logger.debug(
                "Experimental Context cLISI scoring was unavailable: "
                f"errorType={type(exc).__name__}"
            )
            notes.append(f"cLISI could not be scored: {exc}")
        if connectivity is not None:
            try:
                value = float(
                    store.metric_graph_connectivity(biological_column, connectivity)
                )
                if math.isfinite(value):
                    metrics[f"graphConnectivity:{biological_column}"] = value
                    evidence_ids.append(
                        "metric:graphConnectivity:"
                        f"{biological_column}:assay:{connectivity.assay}:connectivity:"
                        f"{connectivity.artifact_id}"
                    )
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                logger.debug(
                    "Experimental Context connectivity scoring was unavailable: "
                    f"errorType={type(exc).__name__}"
                )
                notes.append(f"Graph connectivity could not be scored: {exc}")

    evaluation = RepresentationEvaluation(
        available=bool(metrics),
        assay=neighbors.assay,
        cellSelection=(
            artifact_reference(ctx.deps.cellSelection)
            if ctx.deps.cellSelection is not None
            else None
        ),
        neighbors=artifact_reference(neighbors),
        connectivityMap=(
            artifact_reference(connectivity) if connectivity is not None else None
        ),
        metrics=metrics,
        notes=notes,
        evidenceIds=evidence_ids,
    )
    ctx.deps.currentRepresentation = evaluation
    ctx.deps.evidenceIds.update(evidence_ids)
    ctx.deps.toolCalls.append("score_current_representation")
    logger.info(
        "Experimental Context representation scoring completed: "
        f"available={evaluation.available}, metrics={len(metrics)}, "
        f"notes={len(notes)}, evidence={len(evidence_ids)}"
    )
    return evaluation
