"""Supported assay selection for the automated RNA workflow."""

from collections.abc import Mapping, Sequence
from typing import Any


def validate_rna_directions(directions: Mapping[str, Any]) -> None:
    """Reject automated operations outside the RNA analysis contract."""
    if "hypothesisTesting" in directions:
        raise ValueError(
            "Automated hypothesis testing is not supported; run statistical "
            "analysis separately after the RNA workflow."
        )


def validate_rna_request_fields(request: Any) -> None:
    """Validate routing that does not require inspecting the input store."""
    if len(request.analysisAssays) > 1:
        raise ValueError("analysisAssays must select at most one RNA assay")
    if request.pairedAssays:
        raise ValueError("pairedAssays is unsupported by the single-RNA workflow")
    selected = request.analysisAssays[0] if request.analysisAssays else None
    if selected is not None and request.primaryAssay not in {None, selected}:
        raise ValueError("primaryAssay must match the selected analysisAssays entry")
    selected = selected or request.primaryAssay
    if selected is not None and request.markerAssay not in {None, selected}:
        raise ValueError("markerAssay must match the selected RNA assay")
    validate_rna_directions(request.experimentalDirections)


def selected_rna_assay(request: Any, assay_types: Mapping[str, str]) -> str:
    """Resolve one explicit or uniquely available persisted RNA assay."""
    validate_rna_request_fields(request)
    selected = (
        request.analysisAssays[0] if request.analysisAssays else request.primaryAssay
    )
    if selected is None:
        rna_assays = [
            name for name, assay_type in assay_types.items() if assay_type == "RNA"
        ]
        if len(rna_assays) != 1:
            raise ValueError(
                "The automated workflow requires one RNA assay; "
                f"found {len(rna_assays)}. Select one with primaryAssay or "
                "analysisAssays when multiple RNA assays are present."
            )
        selected = rna_assays[0]
    if selected not in assay_types:
        raise ValueError(f"Unknown requested RNA assay {selected!r}")
    if assay_types[selected] != "RNA":
        raise ValueError(
            f"Selected assay {selected!r} has type {assay_types[selected]!r}; "
            "the automated workflow supports RNA only."
        )
    if request.markerAssay not in {None, selected}:
        raise ValueError("markerAssay must match the selected RNA assay")
    return str(selected)


def selected_store_rna_assay(store: Any, request: Any) -> str:
    """Resolve the workflow assay from the store's persisted summary."""
    return selected_rna_assay(
        request,
        {value.name: value.assay_type for value in store.summary().assays},
    )


def validate_rna_plan(plan: Any, selected: str) -> None:
    """Reject a stale or unsupported cached preprocessing route."""
    if (
        len(plan.assays) != 1
        or plan.assays[0].assay != selected
        or plan.assays[0].assayType != "RNA"
        or not plan.assays[0].graphEligible
        or plan.primaryAssay != selected
        or plan.markerAssay != selected
        or plan.pairedAssays
    ):
        raise ValueError(
            "Preprocessing must contain only the selected RNA assay. "
            "Start a new workflow for incompatible saved analysis state."
        )


def validate_rna_context(report: Any, selected: str) -> None:
    """Keep reused QC evidence on the selected RNA assay."""
    if report.htoIdentityArtifacts:
        raise ValueError(
            "Saved automatic HTO processing is unsupported; start a new RNA workflow."
        )
    for profile in report.qcProfiles:
        if profile.driverAssay != selected or profile.driverAssayType != "RNA":
            raise ValueError(
                "QC evidence must use the selected RNA assay; start a new workflow "
                "for incompatible saved QC state."
            )
    if report.cellQc.driverAssay not in {None, selected}:
        raise ValueError("Cell QC must use the selected RNA assay")


def validate_rna_handoffs(handoffs: Sequence[Any], selected: str) -> None:
    """Keep saved preprocessing on the selected persisted RNA modality."""
    if (
        len(handoffs) != 1
        or handoffs[0].assay != selected
        or handoffs[0].assayType != "RNA"
    ):
        raise ValueError(
            "Saved preprocessing must contain only the selected RNA assay; "
            "start a new workflow for incompatible saved analysis state."
        )


def validate_saved_rna_history(
    store: Any, prefix: str, workflow_run_id: str, selected: str
) -> None:
    """Validate single-RNA ownership before opening resumed work for writes."""
    from ..data_enrichment.contracts import DataEnrichmentReport
    from ..experimental_context.contracts import ExperimentalContextResult
    from ..experimental_context.study import StudyContract, validate_objective_evidence
    from ..parameter_tuning.contracts import ParameterTuningReport
    from . import journal
    from .models import (
        _STAGE_ORDER,
        AutomatedPreprocessingPlan,
        PreprocessedAssayHandoff,
    )

    for stage in _STAGE_ORDER:
        for outcome in journal._stage_outcomes(
            store.zw, prefix, workflow_run_id, stage
        ):
            if outcome.outputs.get("htoIdentityArtifacts"):
                raise ValueError(
                    "Saved automatic HTO processing is unsupported; start a new RNA workflow"
                )
            if stage == "rna_quality_metrics" and outcome.status == "done":
                if "percentageDefinitions" not in outcome.outputs:
                    raise ValueError(
                        "Saved RNA quality metrics lack exact percentage definitions; "
                        "start a new workflow. Existing analysis artifacts remain accessible."
                    )
            for name in ("preprocessingPlan", "resolvedPreprocessingPlan"):
                if outcome.outputs.get(name):
                    validate_rna_plan(
                        AutomatedPreprocessingPlan.model_validate(
                            outcome.outputs[name]
                        ),
                        selected,
                    )
            if stage == "preprocessing" and "assays" in outcome.outputs:
                validate_rna_handoffs(
                    [
                        PreprocessedAssayHandoff.model_validate(v)
                        for v in outcome.outputs["assays"]
                    ],
                    selected,
                )
            if not outcome.reportReferences:
                continue
            if stage == "data_enrichment":
                report = DataEnrichmentReport.model_validate(
                    journal.read_stage_evidence(store, outcome.reportReferences[0])
                )
                if report.status == "done" and (
                    len(report.policies) != 1
                    or report.policies[0].assay != selected
                    or report.policies[0].assayModality != "RNA"
                ):
                    raise ValueError("Saved enrichment includes unsupported assays")
            elif stage == "experimental_context":
                context = ExperimentalContextResult.model_validate(
                    journal.read_stage_evidence(store, outcome.reportReferences[0])
                )
                if context.status == "done":
                    validate_rna_context(context, selected)
                if outcome.status == "done":
                    raw_contract = outcome.outputs.get("studyContract", {})
                    _require_objective_contract(raw_contract)
                    validate_objective_evidence(
                        StudyContract.model_validate(raw_contract), context
                    )
            elif stage == "parameter_tuning":
                tuning = ParameterTuningReport.model_validate(
                    journal.read_stage_evidence(store, outcome.reportReferences[0])
                )
                if (
                    tuning.recommendedIntegrationId is not None
                    or tuning.assayReports
                    or tuning.fromAssay != selected
                ):
                    raise ValueError(
                        "Saved tuning includes unsupported assays or integration"
                    )
    validate_analysis_evidence(journal.analysis_snapshot(store, workflow_run_id))


def _require_objective_contract(value: Any) -> None:
    """Reject historical conclusions without rewriting their evidence."""
    if not isinstance(value, Mapping) or not {
        "evidenceRequirements",
        "evidenceCoverage",
    }.issubset(value):
        raise ValueError(
            "Saved analysis lacks mandatory objective evidence requirements; "
            "start a new workflow to resume or regenerate its report. "
            "Existing analysis artifacts and historical HTML remain accessible."
        )


def validate_analysis_evidence(snapshot: Mapping[str, Any]) -> None:
    """Check scientific completion from the authenticated journal view."""
    from ..experimental_context.contracts import ExperimentalContextResult
    from ..experimental_context.study import StudyContract, validate_objective_evidence
    from .rna_tuning import validate_completed_comparison_evidence

    context_validated = False
    for stage in snapshot.get("stages", []):
        if (
            stage.get("stage") != "experimental_context"
            or stage.get("status") != "done"
        ):
            continue
        raw_contract = stage.get("outputs", {}).get("studyContract", {})
        _require_objective_contract(raw_contract)
        validate_objective_evidence(
            StudyContract.model_validate(raw_contract),
            ExperimentalContextResult.model_validate(stage["report"]),
        )
        context_validated = True
    accepted = [
        review
        for review in snapshot.get("analysisReviews", [])
        if review.get("action") == "accept"
    ]
    for review in accepted:
        validate_completed_comparison_evidence(review)
    if snapshot.get("status") == "completed":
        if not context_validated or not any(
            item.get("scope") == "full" for item in accepted
        ):
            raise ValueError(
                "Completed analysis lacks its mandatory objective and comparison evidence; "
                "start a new workflow. Existing analysis artifacts remain accessible."
            )
