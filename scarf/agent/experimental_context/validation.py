"""Experimental-context canonicalization and explicit model failures."""

from types import SimpleNamespace
from typing import Any

from ...utils.logging import logger
from .._deps import AGENT_INSTALL_HINT
from ..tools import artifact_reference
from ..types import AgentRunInfo, BatchSafetyEvidence
from .comparisons import canonical_design_choices
from .contracts import (
    CellQcPlan,
    CovariateCharacterization,
    ExperimentalContextDecision,
    ExperimentalContextDependencies,
    ExperimentalContextResult,
    InferenceUnit,
    characterization_evidence,
)
from .qc_evidence import (
    _offered_qc_profiles,
)
from .requirements import (
    active_batch_safety,
    objective_evidence,
    unmet_objective_requirements,
)
from .tools import characterize_context, contrast_plans_from_characterization

try:
    from pydantic import ValidationError
    from pydantic_ai import ModelRetry
except ImportError as exc:
    raise ImportError(AGENT_INSTALL_HINT) from exc


def _validate_batch_correction_plan(
    decision: ExperimentalContextDecision,
    deps: ExperimentalContextDependencies,
    characterization: CovariateCharacterization,
    requested_coefficients: set[str],
    units_of_inference: dict[str, dict[str, Any]],
    records: dict[str, dict[str, Any]],
    coefficient_records: dict[str, dict[str, Any]],
) -> None:
    """Validate one batch plan against exact design, safety, and metric evidence."""
    confounding_reports = {
        report.get("coefficient"): report
        for report in characterization.confounding
        if isinstance(report.get("coefficient"), str)
    }
    plan = decision.batchCorrection
    try:
        active_batch_safety(
            SimpleNamespace(
                decision=decision, batchSafety=list(deps.batchSafety.values())
            )
        )
    except ValueError as exc:
        raise ModelRetry(str(exc)) from exc
    directed_batch_columns = deps.directions.get("batchColumns")
    if directed_batch_columns is not None:
        if not isinstance(directed_batch_columns, list) or any(
            not isinstance(value, str) or not value.strip()
            for value in directed_batch_columns
        ):
            raise ModelRetry(
                "directions.batchColumns must be a list of exact metadata columns"
            )
        canonical_directed_batch = sorted(directed_batch_columns)
        directed_plan_mismatch = (
            (
                plan.action not in {"evaluateHarmony", "unsafe"}
                or sorted(plan.batchColumns) != canonical_directed_batch
            )
            if canonical_directed_batch
            else plan.action != "skip" or bool(plan.batchColumns)
        )
        if directed_plan_mismatch:
            raise ModelRetry(
                "The batch-correction plan must assess the exact directed batch "
                f"columns: {canonical_directed_batch}"
            )
    unknown_columns = sorted(set(decision.columnDomains) - set(records))
    if unknown_columns:
        raise ModelRetry(f"Unknown column domain assignments: {unknown_columns}")
    unit_columns = {
        unit_name
        for unit in units_of_inference.values()
        for unit_name in (
            unit.get("observationUnit"),
            unit.get("independentUnit"),
        )
        if isinstance(unit_name, str)
    }
    if plan.action == "evaluateHarmony" and not plan.batchColumns:
        raise ModelRetry("evaluateHarmony requires at least one batch column")
    if plan.action == "unsafe" and not plan.batchColumns:
        raise ModelRetry("unsafe requires the exact batch columns that were assessed")
    if plan.action == "skip" and plan.batchColumns:
        raise ModelRetry("skip must not include batch columns")
    if plan.action == "needsInput" and not decision.needsInput:
        raise ModelRetry("needsInput action requires at least one concrete question")
    if len(set(plan.batchColumns)) != len(plan.batchColumns):
        raise ModelRetry("Batch columns must be unique")

    for batch_column in plan.batchColumns:
        record = records.get(batch_column)
        if record is None:
            raise ModelRetry(f"Unknown batch column {batch_column!r}")
        if record.get("domain") != "technical":
            raise ModelRetry(
                f"Batch column {batch_column!r} must be classified as technical"
            )
        if record.get("kind") != "categorical":
            raise ModelRetry(
                f"Batch column {batch_column!r} must be categorical for Harmony"
            )
        if batch_column in requested_coefficients or batch_column in unit_columns:
            raise ModelRetry(
                f"Batch column {batch_column!r} cannot be a coefficient or unit of inference"
            )

    if plan.action == "evaluateHarmony":
        mixing_metrics = {"iLISI", "proportionalBatchMixing"}
        preservation_metrics = {"cLISI", "graphConnectivity"}
        if not mixing_metrics.intersection(plan.metricsRequired):
            raise ModelRetry(
                "evaluateHarmony requires iLISI or proportionalBatchMixing"
            )
        if any(
            records.get(column, {}).get("kind") == "categorical"
            for column in plan.preserveColumns
        ) and not preservation_metrics.intersection(plan.metricsRequired):
            raise ModelRetry(
                "evaluateHarmony requires cLISI or graphConnectivity for preservation"
            )
        missing_preserve = sorted(requested_coefficients - set(plan.preserveColumns))
        if missing_preserve:
            raise ModelRetry(
                "preserveColumns must include every coefficient of interest: "
                f"{missing_preserve}"
            )
        unresolved_coefficients = sorted(
            coefficient
            for coefficient in requested_coefficients
            if coefficient_records[coefficient].get("scope") != "betweenUnit"
            or coefficient not in confounding_reports
        )
        if unresolved_coefficients:
            raise ModelRetry(
                "evaluateHarmony requires a between-unit coefficient with a "
                "matching estimability report; use needsInput or unsafe for: "
                f"{unresolved_coefficients}"
            )
        for preserve_column in plan.preserveColumns:
            record = records.get(preserve_column)
            if record is None:
                raise ModelRetry(f"Unknown preservation column {preserve_column!r}")
            if record.get("domain") != "biological":
                raise ModelRetry(
                    f"Preservation column {preserve_column!r} must be biological"
                )

    matched_safety: list[BatchSafetyEvidence] = []
    if plan.action in {"evaluateHarmony", "unsafe"}:
        canonical_batch_columns = sorted(plan.batchColumns)
        for coefficient in sorted(requested_coefficients):
            coefficient_record = coefficient_records[coefficient]
            report = confounding_reports.get(coefficient)
            observation_unit = (
                report.get("observationUnit")
                if report is not None
                else coefficient_record.get("observationUnit")
            )
            unit_constant = {
                pair.get("technical")
                for pair in (report.get("pairs", []) if report is not None else [])
                if isinstance(pair.get("technical"), str)
            }
            expected_effective = [
                name for name in canonical_batch_columns if name in unit_constant
            ]
            candidates = [
                item
                for item in deps.batchSafety.values()
                if item.coefficient == coefficient
                and item.coefficientKind == coefficient_record.get("kind")
                and item.observationUnit == observation_unit
                and item.batchColumns == canonical_batch_columns
                and item.unitConstantBatchColumns == expected_effective
            ]
            if len(candidates) != 1:
                raise ModelRetry(
                    "Call analyze_experimental_design with the exact proposed batch "
                    f"columns before returning a recommendation for {coefficient!r}"
                )
            matched_safety.append(candidates[0])
        missing_safety_evidence = sorted(
            item.evidenceId
            for item in matched_safety
            if item.evidenceId not in plan.evidenceIds
        )
        if missing_safety_evidence:
            raise ModelRetry(
                "Batch-correction recommendations must cite exact batch "
                f"estimability evidence: {missing_safety_evidence}"
            )
        not_computed = [
            item.coefficient for item in matched_safety if item.status == "notComputed"
        ]
        if not_computed:
            raise ModelRetry(
                "Batch estimability could not be computed; use action='needsInput' "
                f"for: {sorted(not_computed)}"
            )
        unsafe_coefficients = [
            item.coefficient for item in matched_safety if item.status == "unsafe"
        ]
        if plan.action == "evaluateHarmony" and unsafe_coefficients:
            raise ModelRetry(
                "Batch correction is unsafe because the biological coefficient is "
                "not estimable after the exact proposed batch columns; use "
                f"action='unsafe' for: {sorted(unsafe_coefficients)}"
            )
        if plan.action == "unsafe" and not unsafe_coefficients:
            raise ModelRetry(
                "The exact proposed batch columns were estimable for every "
                "coefficient; use action='evaluateHarmony' or 'skip'"
            )

    cited_ids = [
        *decision.evidenceIds,
        *plan.evidenceIds,
    ]
    unknown_evidence = sorted(set(cited_ids) - deps.evidenceIds)
    if unknown_evidence:
        raise ModelRetry(f"Unknown evidence IDs: {unknown_evidence}")
    if plan.action in {"evaluateHarmony", "skip", "unsafe"} and not plan.evidenceIds:
        raise ModelRetry("Batch-correction recommendations require evidence IDs")
    current_metric_evidence = set(deps.currentRepresentation.evidenceIds)
    stale_metric_evidence = sorted(
        evidence_id
        for evidence_id in cited_ids
        if evidence_id.startswith("metric:")
        and evidence_id not in current_metric_evidence
    )
    if stale_metric_evidence:
        raise ModelRetry(
            "Metric evidence must come from the returned exact representation: "
            f"{stale_metric_evidence}"
        )


def validate_experimental_context(
    decision: ExperimentalContextDecision,
    deps: ExperimentalContextDependencies,
) -> ExperimentalContextDecision:
    """Recompute and validate every model-authored design choice."""
    if decision.cellQc != CellQcPlan.get_blank():
        raise ModelRetry(
            "Experimental Context must leave cellQc blank; the audited filtering "
            "checkpoint selects from qcProfiles"
        )
    narrative_fields = {
        "rationale": decision.rationale,
        "batchCorrection.rationale": decision.batchCorrection.rationale,
        **{
            f"needsInput[{index}]": question
            for index, question in enumerate(decision.needsInput)
        },
    }
    serialized_field_markers = (
        '"evidenceIds":',
        '"needsInput":',
        '"runInfo":',
        '"batchCorrection":',
        '"cellQc":',
    )
    invalid_narratives = [
        name
        for name, value in narrative_fields.items()
        if any(
            marker in value.replace('\\"', '"') for marker in serialized_field_markers
        )
    ]
    if invalid_narratives:
        raise ModelRetry(
            "Narrative fields must contain plain prose without serialized sibling "
            f"fields: {invalid_narratives}"
        )
    for required in ("inspect_cell_covariates", "analyze_experimental_design"):
        if required not in deps.toolCalls:
            raise ModelRetry(f"Call {required} before returning a decision")
    try:
        canonical_design_choices(deps, decision)
    except ValueError as exc:
        raise ModelRetry(str(exc)) from exc
    directions = dict(deps.directions)
    column_domains = dict(decision.columnDomains)
    column_domains.update(dict(directions.get("columnDomains") or {}))
    directions["columnDomains"] = column_domains
    directions["coefficientsOfInterest"] = list(
        dict.fromkeys(
            [
                *decision.coefficientsOfInterest,
                *(directions.get("coefficientsOfInterest") or []),
            ]
        )
    )
    units_of_inference = {
        name: unit.model_dump(exclude_none=True)
        for name, unit in decision.unitsOfInference.items()
    }
    units_of_inference.update(dict(directions.get("unitsOfInference") or {}))
    directions["unitsOfInference"] = units_of_inference

    previous_inputs = deps.characterizationInputs
    characterization = characterize_context(deps, directions)
    if characterization.status == "failed":
        raise ModelRetry("; ".join(characterization.notes))
    characterization.comparisons = list(deps.comparisons)
    characterization.captureProvenance = deps.captureProposal
    deps.characterization = characterization
    deps.evidenceIds.update(characterization_evidence(characterization))
    contrast_plans = contrast_plans_from_characterization(characterization)
    deps.contrastPlans = {plan.coefficient: plan for plan in contrast_plans}
    deps.evidenceIds.update(plan.evidenceId for plan in contrast_plans)

    if not deps.qcProfiles or previous_inputs != deps.characterizationInputs:
        _offered_qc_profiles(deps, characterization)
    deps.evidenceIds.update(profile.evidenceId for profile in deps.qcProfiles.values())
    deps.evidenceIds.update(source.sourceId for source in deps.qcMetricSources)
    deps.evidenceIds.update(item.evidenceId for item in deps.qcSourceConcordance)

    requested_coefficients = set(directions["coefficientsOfInterest"])
    characterized_coefficients = {
        record.get("name") for record in characterization.coefficients
    }
    missing_coefficients = sorted(
        name
        for name in requested_coefficients
        if name not in characterized_coefficients
    )
    if missing_coefficients:
        raise ModelRetry(
            "Coefficients of interest must be classified as biological: "
            f"{missing_coefficients}"
        )

    coefficient_records: dict[str, dict[str, Any]] = {}
    for record in characterization.coefficients:
        name = record.get("name")
        if isinstance(name, str):
            coefficient_records[name] = record
    records: dict[str, dict[str, Any]] = {}
    for record in characterization.columns:
        name = record.get("name")
        if isinstance(name, str):
            records[name] = record
    _validate_batch_correction_plan(
        decision,
        deps,
        characterization,
        requested_coefficients,
        units_of_inference,
        records,
        coefficient_records,
    )
    canonical_domains = {
        name: records[name]["domain"]
        for name in column_domains
        if name in records
        and records[name].get("domain")
        in {
            "biological",
            "technical",
            "design",
            "ignore",
            "unknown",
        }
    }
    canonical_units = {
        coefficient: InferenceUnit(
            observationUnit=coefficient_records[coefficient].get("observationUnit"),
            independentUnit=coefficient_records[coefficient].get("independentUnit"),
        )
        for coefficient in directions["coefficientsOfInterest"]
        if coefficient in coefficient_records
    }
    try:
        design_choices = canonical_design_choices(deps, decision)
    except ValueError as exc:
        raise ModelRetry(str(exc)) from exc
    validated = decision.model_copy(
        update={
            "columnDomains": canonical_domains,
            "coefficientsOfInterest": list(directions["coefficientsOfInterest"]),
            "unitsOfInference": canonical_units,
            "cellQc": CellQcPlan.get_blank(),
            **design_choices,
        }
    )
    if deps.studyObjective:
        requirements, coverage = objective_evidence(
            study_context=deps.studyContext,
            study_objective=deps.studyObjective,
            experimental_result=SimpleNamespace(
                decision=validated,
                characterization=characterization,
                batchSafety=list(deps.batchSafety.values()),
            ),
        )
        unanswered = unmet_objective_requirements(requirements, coverage)
        if unanswered:
            pending_comparisons = [
                item.question
                for item in requirements
                if item.requirementId.startswith("requestedDesign:")
                and any(
                    row.requirementId == item.requirementId
                    and row.status == "unsupported"
                    for row in coverage
                )
            ]
            if pending_comparisons and deps.designRounds < 2:
                raise ModelRetry(
                    "Explicit study comparisons have not been measured: "
                    + "; ".join(pending_comparisons)
                    + ". Use the remaining analyze_experimental_design round for "
                    "these questions before finalizing. Match their requested purpose: "
                    "descriptive crossing/replication is designCoverage, not an "
                    "association or effect estimate. Preserve unsupported methods "
                    "as limitations; do not substitute marginal comparisons."
                )
            validated = validated.model_copy(
                update={
                    "needsInput": list(
                        dict.fromkeys([*validated.needsInput, *unanswered])
                    ),
                }
            )
    logger.debug(
        "Experimental Context decision validated: "
        f"domains={len(validated.columnDomains)}, "
        f"coefficients={len(validated.coefficientsOfInterest)}, "
        f"qcProfiles={len(deps.qcProfiles)}, "
        f"batchCorrection={validated.batchCorrection.action}, "
        f"needsInput={len(validated.needsInput)}"
    )
    return validated


def failed_experimental_context_result(
    deps: ExperimentalContextDependencies,
    *,
    error: Exception,
    model_name: str,
) -> ExperimentalContextResult:
    """Keep a failed model run from selecting an unsupported scientific default."""
    characterization = deps.characterization or CovariateCharacterization(
        status="failed",
        notes=["Deterministic covariate characterization is unavailable."],
    )
    model_detail = str(error).replace("\n", " ").strip()[:500]
    failure_notes = []
    cause = error.__cause__
    if isinstance(cause, ValidationError):
        details = [
            f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            for item in cause.errors(
                include_url=False, include_input=False, include_context=False
            )[:8]
        ]
        failure_notes.append("Invalid tool arguments: " + "; ".join(details))
    elif isinstance(cause, ModelRetry):
        failure_notes.append("Rejected tool arguments: " + str(cause)[:1000])
    return ExperimentalContextResult(
        status="failed",
        decision=ExperimentalContextDecision(
            rationale="No validated experimental-context decision was available.",
            evidenceIds=sorted(deps.evidenceIds),
        ),
        characterization=characterization,
        cellSelection=artifact_reference(deps.cellSelection),
        cellQc=CellQcPlan.get_blank(),
        qcProfiles=list(deps.qcProfiles.values()),
        qcMetricSources=deps.qcMetricSources,
        qcSourceConcordance=deps.qcSourceConcordance,
        contrastPlans=list(deps.contrastPlans.values()),
        qualityMetricArtifacts=deps.qualityMetricArtifacts,
        htoIdentityColumns=deps.htoIdentityColumns,
        htoIdentityArtifacts=deps.htoIdentityArtifacts,
        batchSafety=list(deps.batchSafety.values()),
        currentRepresentation=deps.currentRepresentation,
        notes=[
            "The model did not produce a validated experimental-context decision.",
            f"Model failure: {model_detail}",
            *failure_notes,
        ],
        runInfo=getattr(error, "agent_run_info", None)
        or AgentRunInfo(
            agentName="experimental_context_failed",
            modelName=model_name,
        ),
    )
