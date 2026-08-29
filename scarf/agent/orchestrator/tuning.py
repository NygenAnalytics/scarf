"""Parameter tuning and multimodal integration workflow stages."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from ...datastore.datastore import DataStore
from ...utils.logging import logger
from ..experimental_context import ExperimentalContextResult
from ..parameter_tuning import (
    ArtifactRecord,
    FinalGraphComparison,
    FinalGraphSelection,
    IntegrationCandidateEvaluation,
    IntegrationMetrics,
    ParameterCandidate,
    ParameterTuningAgent,
    ParameterTuningAssayInput,
    ParameterTuningReport,
    final_graph_options,
    finalize_parameter_tuning_selection,
    validate_final_graph_selection,
)
from ..persistence import (
    AgentInvocation,
    AgentReportLink,
    AgentReportReference,
    AgentWorkflowRun,
    list_agent_reports,
    load_agent_record,
    load_agent_report,
    save_agent_report,
)
from ..types import ArtifactReferenceModel, ExperimentalTuningHandoff
from . import journal
from .models import (
    AutomatedPreprocessingPlan,
    AutomatedWorkflowConfig,
    OrchestrationRequestRecord,
    OrchestrationResumeRecord,
    PreprocessedAssayHandoff,
    WorkflowNeedsInput,
    WorkflowQuestion,
    WorkflowStageAttempt,
    WorkflowStageLink,
    artifact_model_to_ref,
)


class TuningStagesMixin:
    """Execute parameter searches, integration comparisons, and graph selection."""

    model: Any

    def parameter_tuning_stage(
        self,
        store: DataStore,
        workflow: AgentWorkflowRun,
        request_record: OrchestrationRequestRecord,
        parents: Sequence[WorkflowStageLink],
        plan: AutomatedPreprocessingPlan,
        preprocessed: Sequence[PreprocessedAssayHandoff],
        experimental: ExperimentalContextResult,
        enrichment_reference: AgentReportReference,
        experimental_reference: AgentReportReference,
        answers: Mapping[str, Any],
        *,
        resume_record: OrchestrationResumeRecord | None = None,
    ) -> tuple[WorkflowStageAttempt, ParameterTuningReport]:
        prefix = journal._ensure_orchestration_store(store)
        existing = journal._validated_done_outcome(
            store,
            prefix,
            workflow.workflowRunId,
            "parameter_tuning",
            request_record,
            parents,
        )
        if existing is not None:
            logger.info(
                f"Workflow {workflow.workflowRunId}: reusing Parameter Tuning report"
            )
            report = journal.load_stage_report(store, existing, ParameterTuningReport)
            return existing, cast(ParameterTuningReport, report)
        paused = journal._validated_done_outcome(
            store,
            prefix,
            workflow.workflowRunId,
            "parameter_tuning",
            request_record,
            parents,
            required_status="needsInput",
        )
        resumable_report: ParameterTuningReport | None = None
        if paused is not None and paused.reportReferences:
            loaded = journal.load_stage_report(store, paused, ParameterTuningReport)
            candidate_report = cast(ParameterTuningReport, loaded)
            if (
                candidate_report.finalSelection is not None
                and candidate_report.finalSelection.status == "needsInput"
                and candidate_report.assayReports
            ):
                resumable_report = candidate_report
        experimental_handoff = experimental.to_parameter_tuning_handoff()
        tuning_answer = answers.get("parameter_tuning")
        if isinstance(tuning_answer, Mapping):
            tuning_directions = json.dumps(
                dict(tuning_answer),
                sort_keys=True,
            )
        elif isinstance(tuning_answer, str):
            tuning_directions = tuning_answer.strip()
        else:
            tuning_directions = ""
        started = journal._start_attempt(
            store.zw,
            prefix,
            workflow.workflowRunId,
            "parameter_tuning",
            request_record,
            parents,
            inputs={
                "preprocessedAssays": [
                    value.model_dump(mode="json") for value in preprocessed
                ],
                "experimentalTuningHandoff": experimental_handoff.model_dump(
                    mode="json"
                ),
                "primaryAssay": plan.primaryAssay,
                "markerAssay": plan.markerAssay,
                "pairedAssays": plan.pairedAssays,
                "finalGraphOptionId": answers.get("finalGraphOptionId"),
                "parameterTuning": tuning_answer,
                "resumeFromAttempt": (
                    paused.attemptId
                    if paused is not None and resumable_report is not None
                    else None
                ),
            },
            resume_record=resume_record,
        )
        report = ParameterTuningReport.get_blank()
        actions: list[str] = []
        integration_evaluations: list[IntegrationCandidateEvaluation] = []
        candidate_payload: dict[str, list[dict[str, Any]]] = {}
        paired = list(plan.pairedAssays)
        logger.info(
            f"Workflow {workflow.workflowRunId}: Parameter Tuning started for "
            f"{len(preprocessed)} assay(s), paired={len(paired)}"
        )
        try:
            agent = ParameterTuningAgent(
                self.model,
                config=request_record.config.agentRunConfig,
            )
            recovered = journal._recover_persisted_stage_report(
                store,
                started,
                agent_name="parameter_tuning",
                expected_type=ParameterTuningReport,
            )
            if recovered is not None:
                recovered_report, recovered_reference = recovered
                report = cast(ParameterTuningReport, recovered_report)
                integration_evaluations = list(report.integrationEvaluations)
                candidate_payload = {
                    assay: [
                        evaluation.parameters.model_dump(mode="json")
                        for evaluation in assay_report.evaluations
                    ]
                    for assay, assay_report in report.assayReports.items()
                }
                actions.append("recover_persisted_parameter_tuning_report")
                logger.info(
                    f"Workflow {workflow.workflowRunId}: recovering completed "
                    "Parameter Tuning provider result"
                )
                return self.save_parameter_tuning_outcome(
                    store,
                    prefix,
                    workflow,
                    request_record,
                    started,
                    report,
                    plan,
                    preprocessed,
                    integration_evaluations,
                    candidate_payload,
                    paired,
                    enrichment_reference,
                    experimental_reference,
                    experimental_handoff,
                    agent,
                    actions,
                    persisted_reference=recovered_reference,
                )
            if resumable_report is not None:
                assert paused is not None
                resumed_integration_evaluations = list(
                    resumable_report.integrationEvaluations
                )
                report = resumable_report.model_copy(
                    update={
                        "status": "done",
                        "needsInput": None,
                        "finalSelection": None,
                        "recommendedIntegrationId": None,
                        "finalClusterColumn": None,
                        "finalClusterArtifact": None,
                        "graphAssay": None,
                    }
                )
                report = self.select_final_graph(
                    agent,
                    report,
                    resumed_integration_evaluations,
                    marker_assay=plan.markerAssay,
                    answers=answers,
                )
                logger.info(
                    f"Workflow {workflow.workflowRunId}: resumed final graph "
                    "selection without rerunning candidate evaluation"
                )
                resumed_candidate_payload = {
                    assay: [
                        evaluation.parameters.model_dump(mode="json")
                        for evaluation in assay_report.evaluations
                    ]
                    for assay, assay_report in report.assayReports.items()
                }
                return self.save_parameter_tuning_outcome(
                    store,
                    prefix,
                    workflow,
                    request_record,
                    started,
                    report,
                    plan,
                    preprocessed,
                    resumed_integration_evaluations,
                    resumed_candidate_payload,
                    list(plan.pairedAssays),
                    enrichment_reference,
                    experimental_reference,
                    experimental_handoff,
                    agent,
                    ["reuse_parameter_screen_and_integrations"],
                    prior_tuning_reference=paused.reportReferences[0],
                )
            handoff_by_assay = {value.assay: value for value in preprocessed}
            common_k = None
            if paired:
                common_k = min(
                    21,
                    min(handoff_by_assay[assay].nCells - 1 for assay in paired),
                )
                if common_k < 2:
                    raise ValueError("Paired integration requires at least three cells")
            integration_budget = (
                2 * request_record.config.integrationResolutionCandidates
                if len(paired) >= 2
                else 0
            )
            assay_inputs: list[ParameterTuningAssayInput] = []
            for handoff in preprocessed:
                if handoff.normalized is None:
                    raise ValueError(f"Assay {handoff.assay!r} lacks normalization")
                initial_count = (
                    request_record.config.primaryInitialCandidates
                    if handoff.assay == plan.primaryAssay
                    else request_record.config.secondaryInitialCandidates
                )
                neighbors_k = common_k or min(11, handoff.nCells - 1)
                candidates = self.initial_parameter_candidates(
                    workflow.workflowRunId,
                    handoff,
                    count=initial_count,
                    neighbors_k=neighbors_k,
                )
                if (
                    experimental_handoff.batchAction == "evaluateHarmony"
                    and request_record.config.maxHarmonyCandidatesPerAssay == 1
                ):
                    baseline = candidates[0]
                    candidates.append(
                        baseline.model_copy(
                            update={
                                "candidateId": f"{baseline.candidateId}_harmony",
                                "useHarmony": True,
                            }
                        )
                    )
                candidate_payload[handoff.assay] = [
                    value.model_dump(mode="json") for value in candidates
                ]
                logger.info(
                    f"Workflow {workflow.workflowRunId}: planned "
                    f"{len(candidates)} native candidate(s) for assay "
                    f"{handoff.assay!r} (harmony="
                    f"{sum(value.useHarmony for value in candidates)})"
                )
                assay_inputs.append(
                    ParameterTuningAssayInput(
                        normalized=artifact_model_to_ref(handoff.normalized),
                        fromAssay=handoff.assay,
                        cellKey=handoff.cellKey,
                        candidates=candidates,
                        batchColumns=(
                            list(experimental_handoff.batchColumns)
                            if experimental_handoff.batchAction == "evaluateHarmony"
                            else []
                        ),
                        preservationColumns=list(
                            experimental_handoff.preservationColumns
                        ),
                        experimentalHandoff=(
                            None
                            if experimental_handoff.batchAction == "evaluateHarmony"
                            else experimental_handoff
                        ),
                        maxCandidates=(
                            len(candidates)
                            + request_record.config.maxRefinedCandidatesPerAssay
                        ),
                        maxRefinedCandidates=(
                            request_record.config.maxRefinedCandidatesPerAssay
                        ),
                        allowHarmonyRefinement=False,
                        minClusterCells=request_record.config.minClusterCells,
                        identityFeatureLimit=request_record.config.maxIdentityFeatures,
                    )
                )
            planned_native = sum(value.maxCandidates for value in assay_inputs)
            if (
                planned_native + integration_budget
                > request_record.config.maxCandidateBranches
            ):
                raise ValueError(
                    "The native and integrated candidate plan exceeds the global "
                    f"branch limit {request_record.config.maxCandidateBranches}"
                )
            logger.info(
                f"Workflow {workflow.workflowRunId}: executing "
                f"{planned_native} native candidate branch(es) with "
                f"{integration_budget} reserved integration branch(es)"
            )
            report = agent.run_batch(
                store,
                assays=assay_inputs,
                primary_assay=plan.primaryAssay,
                max_total_candidates=(
                    request_record.config.maxCandidateBranches - integration_budget
                ),
                selection_directions=tuning_directions,
            )
            logger.info(
                f"Workflow {workflow.workflowRunId}: Parameter Tuning returned "
                f"status={report.status!r}, evaluated={report.totalCandidates}"
            )
            if report.status == "done":
                for assay, assay_report in report.assayReports.items():
                    normalized = handoff_by_assay[assay].normalized
                    assert normalized is not None
                    agent.promote(
                        store,
                        report=assay_report,
                        normalized=artifact_model_to_ref(normalized),
                        identity_feature_limit=request_record.config.maxIdentityFeatures,
                    )
                    actions.append(f"promote_native:{assay}")
                    logger.info(
                        f"Workflow {workflow.workflowRunId}: promoted native "
                        f"candidate {assay_report.recommendedCandidateId!r} for "
                        f"assay {assay!r}"
                    )
                integration_evaluations = self.evaluate_integrations(
                    store,
                    workflow.workflowRunId,
                    plan,
                    report,
                    experimental_handoff,
                    request_record.config,
                    started=started,
                    parent_reports=[
                        journal._report_link(enrichment_reference),
                        journal._report_link(experimental_reference),
                    ],
                    actions=actions,
                )
                logger.info(
                    f"Workflow {workflow.workflowRunId}: evaluated "
                    f"{len(integration_evaluations)} integration candidate(s)"
                )
                report = self.select_final_graph(
                    agent,
                    report,
                    integration_evaluations,
                    marker_assay=plan.markerAssay,
                    answers=answers,
                )
            return self.save_parameter_tuning_outcome(
                store,
                prefix,
                workflow,
                request_record,
                started,
                report,
                plan,
                preprocessed,
                integration_evaluations,
                candidate_payload,
                paired,
                enrichment_reference,
                experimental_reference,
                experimental_handoff,
                agent,
                actions,
            )
        except Exception as exc:
            failure_artifacts: dict[str, ArtifactReferenceModel] = {}
            for assay, assay_report in report.assayReports.items():
                for evaluation in assay_report.evaluations:
                    for name, artifact in evaluation.artifacts.items():
                        failure_artifacts[
                            f"{assay}_{evaluation.parameters.candidateId}_{name}"
                        ] = ArtifactReferenceModel.model_validate(artifact.model_dump())
            for integration_evaluation in integration_evaluations:
                if integration_evaluation.graphArtifact is not None:
                    failure_artifacts[
                        f"{integration_evaluation.integrationId}_graph"
                    ] = ArtifactReferenceModel.model_validate(
                        integration_evaluation.graphArtifact.model_dump()
                    )
                if integration_evaluation.clusterArtifact is not None:
                    failure_artifacts[
                        f"{integration_evaluation.integrationId}_clusters"
                    ] = ArtifactReferenceModel.model_validate(
                        integration_evaluation.clusterArtifact.model_dump()
                    )
            outcome = journal.finish_exception(
                store,
                prefix,
                workflow,
                started,
                exc,
                artifacts=failure_artifacts,
                actions=actions,
                outputs={
                    "candidatePlan": candidate_payload,
                    "integrationEvaluations": [
                        value.model_dump(mode="json")
                        for value in integration_evaluations
                    ],
                },
            )
            return outcome, ParameterTuningReport.get_blank()

    def select_final_graph(
        self,
        agent: ParameterTuningAgent,
        report: ParameterTuningReport,
        integration_evaluations: Sequence[IntegrationCandidateEvaluation],
        *,
        marker_assay: str,
        answers: Mapping[str, Any],
    ) -> ParameterTuningReport:
        directed_option = answers.get("finalGraphOptionId")
        if not isinstance(directed_option, str) or not directed_option:
            logger.info(
                f"Selecting final graph from native and "
                f"{len(integration_evaluations)} integration evaluation(s)"
            )
            return agent.select_final(
                report=report,
                integration_evaluations=integration_evaluations,
                marker_assay=marker_assay,
            )
        options = final_graph_options(report, integration_evaluations)
        if directed_option not in options:
            raise ValueError("finalGraphOptionId is not an eligible option")
        logger.info(f"Applying caller-selected final graph {directed_option!r}")
        selected_evidence = list(options[directed_option]["evidenceIds"])
        selection = FinalGraphSelection(
            status="done",
            selectedOptionId=directed_option,
            markerAssay=marker_assay,
            confidence="high",
            rationale="The caller selected this persisted eligible option.",
            evidenceIds=selected_evidence,
            comparisons=[
                FinalGraphComparison(
                    optionId=option_id,
                    summary="The caller preferred the selected eligible option.",
                    evidenceIds=[
                        *selected_evidence,
                        *cast(list[str], option["evidenceIds"]),
                    ],
                )
                for option_id, option in options.items()
                if option_id != directed_option
            ],
        )
        selection = validate_final_graph_selection(
            selection,
            report,
            integration_evaluations=integration_evaluations,
            marker_assay=marker_assay,
        )
        return finalize_parameter_tuning_selection(
            report,
            marker_assay=marker_assay,
            integration_evaluations=integration_evaluations,
            recommended_integration_id=selection.integrationId,
            native_assay=selection.nativeAssay,
            final_selection=selection,
        )

    def save_parameter_tuning_outcome(
        self,
        store: DataStore,
        prefix: str,
        workflow: AgentWorkflowRun,
        request_record: OrchestrationRequestRecord,
        started: WorkflowStageAttempt,
        report: ParameterTuningReport,
        plan: AutomatedPreprocessingPlan,
        preprocessed: Sequence[PreprocessedAssayHandoff],
        integration_evaluations: Sequence[IntegrationCandidateEvaluation],
        candidate_payload: Mapping[str, list[dict[str, Any]]],
        paired: Sequence[str],
        enrichment_reference: AgentReportReference,
        experimental_reference: AgentReportReference,
        experimental_handoff: ExperimentalTuningHandoff,
        agent: ParameterTuningAgent,
        actions: Sequence[str],
        *,
        prior_tuning_reference: AgentReportReference | None = None,
        persisted_reference: AgentReportReference | None = None,
    ) -> tuple[WorkflowStageAttempt, ParameterTuningReport]:
        invocation_artifacts = {
            f"{value.assay}_normalized": value.normalized
            for value in preprocessed
            if value.normalized is not None
        }
        for integration_evaluation in integration_evaluations:
            if integration_evaluation.graphArtifact is not None:
                invocation_artifacts[
                    f"{integration_evaluation.integrationId}_graph"
                ] = ArtifactReferenceModel.model_validate(
                    integration_evaluation.graphArtifact.model_dump()
                )
            if integration_evaluation.clusterArtifact is not None:
                invocation_artifacts[
                    f"{integration_evaluation.integrationId}_clusters"
                ] = ArtifactReferenceModel.model_validate(
                    integration_evaluation.clusterArtifact.model_dump()
                )
        stage_artifacts = dict(invocation_artifacts)
        for assay, assay_report in report.assayReports.items():
            for name, artifact in assay_report.selectedArtifacts.items():
                stage_artifacts[f"{assay}_{name}"] = (
                    ArtifactReferenceModel.model_validate(artifact.model_dump())
                )
        if report.finalClusterArtifact is not None:
            stage_artifacts["final_clusters"] = ArtifactReferenceModel.model_validate(
                report.finalClusterArtifact.model_dump()
            )
        checkpoint_ids = {
            f"{journal._stage_execution_id(started)}_integration_{method}"
            for method in {evaluation.method for evaluation in integration_evaluations}
        }
        checkpoint_references = sorted(
            (
                reference
                for reference in list_agent_reports(
                    store,
                    started.workflowRunId,
                    agent_name="parameter_tuning",
                )
                if reference.agentRunId in checkpoint_ids
            ),
            key=lambda value: value.agentRunId,
        )
        if persisted_reference is None:
            saved_report, reference = journal._save_stage_report(
                store,
                started,
                report,
                invocation=AgentInvocation(
                    agentName="parameter_tuning",
                    parentReports=[
                        journal._report_link(enrichment_reference),
                        journal._report_link(experimental_reference),
                        *(
                            [journal._report_link(prior_tuning_reference)]
                            if prior_tuning_reference is not None
                            else []
                        ),
                        *[
                            journal._report_link(value)
                            for value in checkpoint_references
                        ],
                    ],
                    inputs={
                        "assays": dict(candidate_payload),
                        "primaryAssay": plan.primaryAssay,
                        "markerAssay": plan.markerAssay,
                        "pairedAssays": list(paired),
                        "maxCandidateBranches": request_record.config.maxCandidateBranches,
                    },
                    artifacts=stage_artifacts,
                    runConfig=agent.config,
                    experimentalTuningHandoff=experimental_handoff,
                ),
                expected_type=ParameterTuningReport,
            )
            report = cast(ParameterTuningReport, saved_report)
        else:
            reference = persisted_reference
        stage_report_references = [reference, *checkpoint_references]
        operations: list[dict[str, Any]] = []
        for assay, assay_report in report.assayReports.items():
            for candidate_evaluation in assay_report.evaluations:
                operations.append(
                    {
                        "operation": "execute_parameter_candidate",
                        "assay": assay,
                        "candidate": candidate_evaluation.parameters.model_dump(
                            mode="json"
                        ),
                        "phase": candidate_evaluation.phase,
                        "harmonyBatchColumns": list(
                            candidate_evaluation.harmonyBatchColumns
                        ),
                        "updateState": False,
                        "identityFeatureLimit": (
                            request_record.config.maxIdentityFeatures
                        ),
                        "status": candidate_evaluation.status,
                        "artifacts": {
                            name: value.model_dump(mode="json")
                            for name, value in candidate_evaluation.artifacts.items()
                        },
                    }
                )
        token = workflow.workflowRunId[:12]
        seen_integrated_graphs: set[tuple[str, str]] = set()
        for integration_evaluation in integration_evaluations:
            graph_artifact = integration_evaluation.graphArtifact
            graph_id = graph_artifact.artifactId if graph_artifact is not None else ""
            graph_key = (integration_evaluation.method, graph_id)
            if graph_id and graph_key not in seen_integrated_graphs:
                assert graph_artifact is not None
                seen_integrated_graphs.add(graph_key)
                operations.append(
                    {
                        "operation": "integrate_assays",
                        "method": integration_evaluation.method,
                        "assays": list(integration_evaluation.assays),
                        "label": f"agent_{token}_{integration_evaluation.method}",
                        "invalidateCache": True,
                        "l2Normalize": True,
                        "artifact": graph_artifact.model_dump(mode="json"),
                    }
                )
            if graph_artifact is None:
                continue
            operations.append(
                {
                    "operation": "run_leiden_clustering",
                    "integrationId": integration_evaluation.integrationId,
                    "status": integration_evaluation.status,
                    "resolution": integration_evaluation.resolution,
                    "fromAssay": plan.primaryAssay,
                    "cellKey": "I",
                    "backend": "igraph",
                    "symmetricGraph": False,
                    "graphUpperOnly": False,
                    "label": (
                        integration_evaluation.clusterColumn.removeprefix(
                            f"agent_{token}_{integration_evaluation.method}_"
                        )
                        if integration_evaluation.clusterColumn is not None
                        else None
                    ),
                    "randomSeed": 4444,
                    "invalidateCache": False,
                    "artifact": (
                        integration_evaluation.clusterArtifact.model_dump(mode="json")
                        if integration_evaluation.clusterArtifact is not None
                        else None
                    ),
                }
            )
        if report.status == "needsInput":
            needs_input = report.needsInput
            assert needs_input is not None
            outcome = journal._complete_attempt(
                started,
                status="needsInput",
                report_references=stage_report_references,
                artifacts=stage_artifacts,
                outputs={
                    "candidateCount": report.totalCandidates,
                    "operations": operations,
                },
                actions=actions,
                needs_input=WorkflowNeedsInput(
                    questions=[
                        WorkflowQuestion(
                            questionId=(
                                "finalGraphOptionId"
                                if report.finalSelection is not None
                                and report.finalSelection.status == "needsInput"
                                else "parameter_tuning"
                            ),
                            question=needs_input.question,
                            options=list(needs_input.options),
                            evidenceIds=list(needs_input.evidenceIds),
                        )
                    ]
                ),
                notes=report.limitations,
            )
        elif report.status == "failed":
            outcome = journal._complete_attempt(
                started,
                status="failed",
                report_references=stage_report_references,
                artifacts=stage_artifacts,
                outputs={
                    "candidateCount": report.totalCandidates,
                    "operations": operations,
                },
                actions=actions,
                error="; ".join(report.limitations) or "Parameter Tuning failed",
            )
        else:
            outcome = journal._complete_attempt(
                started,
                status="done",
                report_references=stage_report_references,
                artifacts=stage_artifacts,
                outputs={
                    "candidateCount": report.totalCandidates,
                    "recommendedByAssay": report.recommendedByAssay,
                    "recommendedIntegrationId": report.recommendedIntegrationId,
                    "finalClusterColumn": report.finalClusterColumn,
                    "metadataColumns": [
                        evaluation.clusterColumn
                        for evaluation in integration_evaluations
                        if evaluation.clusterColumn is not None
                    ],
                    "operations": operations,
                },
                actions=actions,
                notes=[*report.tradeoffs, *report.limitations],
            )
        journal._save_outcome(store.zw, prefix, outcome)
        logger.info(
            f"Workflow {workflow.workflowRunId}: Parameter Tuning outcome "
            f"status={outcome.status!r}, candidates={report.totalCandidates}, "
            f"integrations={len(integration_evaluations)}"
        )
        if outcome.status == "failed":
            journal.finalize_failed(store, workflow, outcome.error or "tuning failed")
        return outcome, report

    def initial_parameter_candidates(
        self,
        workflow_run_id: str,
        handoff: PreprocessedAssayHandoff,
        *,
        count: int,
        neighbors_k: int,
    ) -> list[ParameterCandidate]:
        max_dimensions = min(handoff.nCells, handoff.nFeatures) - 1
        if neighbors_k < 2 or neighbors_k >= handoff.nCells:
            raise ValueError(
                f"Assay {handoff.assay!r} has no rank-valid graph candidate"
            )
        if handoff.reductionMethod == "identity":
            if handoff.nFeatures < 2:
                raise ValueError(
                    f"Assay {handoff.assay!r} has no rank-valid graph candidate"
                )
            dimensions = handoff.nFeatures
            dimension_values = [dimensions]
        elif max_dimensions < 2:
            raise ValueError(
                f"Assay {handoff.assay!r} has no rank-valid graph candidate"
            )
        elif handoff.reductionMethod == "lsi":
            dimensions = min(50, max_dimensions)
            dimension_values = [
                dimensions,
                min(30, max_dimensions),
                min(70, max_dimensions),
            ]
        else:
            dimensions = min(21, max_dimensions)
            dimension_values = [
                dimensions,
                min(15, max_dimensions),
                min(30, max_dimensions),
            ]
        unique_dimensions = list(
            dict.fromkeys(value for value in dimension_values if value >= 2)
        )
        specifications: list[tuple[int, float]] = [(unique_dimensions[0], 1.0)]
        specifications.extend((value, 1.0) for value in unique_dimensions[1:])
        for resolution in (0.5, 1.5, 0.75, 1.25, 0.35, 1.75):
            if len(specifications) >= count:
                break
            specifications.append((unique_dimensions[0], resolution))
        token = workflow_run_id[:10]
        assay_token = journal._safe_label(handoff.assay).lower()
        if len(assay_token) > 32:
            digest = hashlib.blake2b(
                handoff.assay.encode("utf-8"), digest_size=4
            ).hexdigest()
            assay_token = f"{assay_token[:23]}_{digest}"
        if handoff.reductionMethod == "identity":
            candidates = [
                ParameterCandidate(
                    candidateId=f"w_{token}_{assay_token}_0",
                    reductionMethod="identity",
                    dimensions=handoff.nFeatures,
                    leidenResolution=1.0,
                    neighborsK=neighbors_k,
                )
            ]
            if count > 1 and max_dimensions >= 2:
                candidates.append(
                    ParameterCandidate(
                        candidateId=f"w_{token}_{assay_token}_1",
                        reductionMethod="pca",
                        dimensions=min(21, max_dimensions),
                        leidenResolution=1.0,
                        neighborsK=neighbors_k,
                    )
                )
            for resolution in (0.5, 1.5, 0.75, 1.25, 0.35, 1.75):
                if len(candidates) >= count:
                    break
                index = len(candidates)
                candidates.append(
                    ParameterCandidate(
                        candidateId=f"w_{token}_{assay_token}_{index}",
                        reductionMethod="identity",
                        dimensions=handoff.nFeatures,
                        leidenResolution=resolution,
                        neighborsK=neighbors_k,
                    )
                )
            return candidates
        return [
            ParameterCandidate(
                candidateId=f"w_{token}_{assay_token}_{index}",
                reductionMethod=cast(Any, handoff.reductionMethod),
                dimensions=dimension,
                leidenResolution=resolution,
                neighborsK=neighbors_k,
            )
            for index, (dimension, resolution) in enumerate(specifications[:count])
        ]

    def load_integration_checkpoint(
        self,
        store: DataStore,
        started: WorkflowStageAttempt,
        method: Literal["snn", "wnn"],
    ) -> tuple[list[IntegrationCandidateEvaluation], AgentReportReference] | None:
        checkpoint_id = f"{journal._stage_execution_id(started)}_integration_{method}"
        matches = [
            reference
            for reference in list_agent_reports(
                store,
                started.workflowRunId,
                agent_name="parameter_tuning",
            )
            if reference.agentRunId == checkpoint_id
        ]
        if not matches:
            return None
        if len(matches) != 1:
            raise ValueError("An integration checkpoint has multiple reports")
        reference = matches[0]
        record = load_agent_record(store, reference)
        if (
            record.invocation.inputs.get("orchestrationExecutionId") != checkpoint_id
            or record.invocation.inputs.get("stageExecutionId")
            != journal._stage_execution_id(started)
            or record.invocation.inputs.get("method") != method
        ):
            raise ValueError("Integration checkpoint identity is stale")
        for artifact in record.invocation.artifacts.values():
            store.load_artifact(artifact_model_to_ref(artifact))
        report = load_agent_report(store, reference)
        if not isinstance(report, ParameterTuningReport):
            raise TypeError("Integration checkpoint is not a Parameter Tuning report")
        evaluations = list(report.integrationEvaluations)
        if not evaluations or any(value.method != method for value in evaluations):
            raise ValueError("Integration checkpoint contains the wrong method")
        logger.info(
            f"Workflow {started.workflowRunId}: recovered {method.upper()} "
            f"checkpoint with {len(evaluations)} evaluation(s)"
        )
        return evaluations, reference

    def save_integration_checkpoint(
        self,
        store: DataStore,
        started: WorkflowStageAttempt,
        report: ParameterTuningReport,
        method: Literal["snn", "wnn"],
        evaluations: Sequence[IntegrationCandidateEvaluation],
        parent_reports: Sequence[AgentReportLink],
    ) -> AgentReportReference:
        checkpoint_id = f"{journal._stage_execution_id(started)}_integration_{method}"
        checkpoint_report = report.model_copy(
            update={
                "integrationEvaluations": list(evaluations),
                "recommendedIntegrationId": None,
                "finalClusterColumn": None,
                "finalClusterArtifact": None,
                "finalSelection": None,
            }
        )
        artifacts: dict[str, ArtifactReferenceModel] = {}
        for evaluation in evaluations:
            if evaluation.graphArtifact is not None:
                artifacts[f"{evaluation.integrationId}_graph"] = (
                    ArtifactReferenceModel.model_validate(
                        evaluation.graphArtifact.model_dump()
                    )
                )
            if evaluation.clusterArtifact is not None:
                artifacts[f"{evaluation.integrationId}_clusters"] = (
                    ArtifactReferenceModel.model_validate(
                        evaluation.clusterArtifact.model_dump()
                    )
                )
        invocation = AgentInvocation(
            agentName="parameter_tuning",
            parentReports=list(parent_reports),
            inputs={
                "orchestrationExecutionId": checkpoint_id,
                "stageExecutionId": journal._stage_execution_id(started),
                "method": method,
            },
            artifacts=artifacts,
        )
        try:
            reference = save_agent_report(
                store,
                started.workflowRunId,
                checkpoint_report,
                invocation=invocation,
                agent_run_id=checkpoint_id,
            )
            logger.info(
                f"Workflow {started.workflowRunId}: persisted {method.upper()} "
                f"checkpoint with {len(evaluations)} evaluation(s)"
            )
            return reference
        except FileExistsError:
            recovered = self.load_integration_checkpoint(store, started, method)
            if recovered is None:
                raise
            return recovered[1]

    def evaluate_integrations(
        self,
        store: DataStore,
        workflow_run_id: str,
        plan: AutomatedPreprocessingPlan,
        report: ParameterTuningReport,
        experimental_handoff: ExperimentalTuningHandoff,
        config: AutomatedWorkflowConfig,
        *,
        started: WorkflowStageAttempt | None = None,
        parent_reports: Sequence[AgentReportLink] = (),
        actions: list[str] | None = None,
    ) -> list[IntegrationCandidateEvaluation]:
        assays = list(plan.pairedAssays)
        if len(assays) < 2:
            logger.info("Skipping SNN/WNN evaluation: fewer than two paired assays")
            return []
        selected_k = {
            next(
                evaluation.parameters.neighborsK
                for evaluation in assay_report.evaluations
                if evaluation.candidateId == assay_report.recommendedCandidateId
            )
            for assay, assay_report in report.assayReports.items()
            if assay in assays
        }
        if len(selected_k) != 1:
            raise ValueError("SNN and WNN require one common selected neighborsK")
        primary_report = report.assayReports[plan.primaryAssay]
        primary_evaluation = next(
            value
            for value in primary_report.evaluations
            if value.candidateId == primary_report.recommendedCandidateId
        )
        center = primary_evaluation.parameters.leidenResolution
        count = config.integrationResolutionCandidates
        multipliers = [1.0] if count == 1 else np.linspace(0.5, 1.5, count).tolist()
        resolutions = list(
            dict.fromkeys(max(0.05, round(center * value, 6)) for value in multipliers)
        )
        logger.info(
            f"Evaluating SNN and WNN across {len(resolutions)} resolution(s) "
            f"for {len(assays)} paired assay(s)"
        )
        native_labels: dict[str, np.ndarray[Any, Any]] = {}
        for assay, assay_report in report.assayReports.items():
            if assay not in assays:
                continue
            selected = next(
                value
                for value in assay_report.evaluations
                if value.candidateId == assay_report.recommendedCandidateId
            )
            cluster_model = ArtifactReferenceModel.model_validate(
                selected.artifacts["clusters"].model_dump()
            )
            cluster_group = store.load_artifact(artifact_model_to_ref(cluster_model))
            cluster_values = cast(Any, cluster_group["values"])
            native_labels[assay] = np.asarray(cluster_values[:])
        token = workflow_run_id[:12]
        evaluations: list[IntegrationCandidateEvaluation] = []
        integration_methods: tuple[Literal["snn", "wnn"], ...] = ("snn", "wnn")
        for method in integration_methods:
            evaluations.extend(
                self.evaluate_integration_method(
                    store,
                    method,
                    token,
                    assays,
                    resolutions,
                    native_labels,
                    plan,
                    report,
                    experimental_handoff,
                    config,
                    started=started,
                    parent_reports=parent_reports,
                    actions=actions,
                )
            )
        return evaluations

    def evaluate_integration_method(
        self,
        store: DataStore,
        method: Literal["snn", "wnn"],
        token: str,
        assays: list[str],
        resolutions: Sequence[float],
        native_labels: Mapping[str, np.ndarray[Any, Any]],
        plan: AutomatedPreprocessingPlan,
        report: ParameterTuningReport,
        experimental_handoff: ExperimentalTuningHandoff,
        config: AutomatedWorkflowConfig,
        *,
        started: WorkflowStageAttempt | None,
        parent_reports: Sequence[AgentReportLink],
        actions: list[str] | None,
    ) -> list[IntegrationCandidateEvaluation]:
        if started is not None:
            recovered = self.load_integration_checkpoint(store, started, method)
            if recovered is not None:
                if actions is not None:
                    actions.append(f"recover_integration_checkpoint:{method}")
                return recovered[0]
        logger.info(
            f"Evaluating {method.upper()} integration across "
            f"{len(resolutions)} resolution(s)"
        )
        evaluations: list[IntegrationCandidateEvaluation] = []
        graph_label = f"agent_{token}_{method}"
        try:
            graph_ref = store.integrate_assays(
                assays,
                graph_label,
                method=method,
                invalidate_cache=True,
                l2_normalize=True,
            )
            weights_valid: bool | None = None
            if method == "wnn":
                graph_group = store.load_artifact(graph_ref)
                stored_weights = cast(Any, graph_group["modality_weights"])
                weights = np.asarray(stored_weights[:], dtype=float)
                weights_valid = bool(
                    weights.shape
                    == (len(next(iter(native_labels.values()))), len(assays))
                    and np.all(np.isfinite(weights))
                    and np.all(weights >= 0)
                    and np.allclose(weights.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6)
                )
        except Exception as exc:
            logger.warning(
                f"{method.upper()} graph construction failed "
                f"({type(exc).__name__}); persisting failed evaluations"
            )
            for index, resolution in enumerate(resolutions):
                evaluations.append(
                    IntegrationCandidateEvaluation(
                        integrationId=f"{method}_{token}_{index}",
                        method=cast(Any, method),
                        assays=assays,
                        status="failed",
                        resolution=resolution,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
            if started is not None:
                self.save_integration_checkpoint(
                    store,
                    started,
                    report,
                    method,
                    evaluations,
                    parent_reports,
                )
                if actions is not None:
                    actions.append(f"checkpoint_integration:{method}")
            return evaluations
        for index, resolution in enumerate(resolutions):
            integration_id = f"{method}_{token}_{index}"
            cluster_label = f"agent_{token}_{method}_r_{index}"
            warnings: list[str] = []
            evidence_ids = [f"integration:{integration_id}:clusters"]
            try:
                cluster_ref = store.run_leiden_clustering(
                    graph=graph_ref,
                    from_assay=plan.primaryAssay,
                    cell_key="I",
                    resolution=resolution,
                    backend="igraph",
                    symmetric_graph=False,
                    graph_upper_only=False,
                    label=cluster_label,
                    random_seed=4444,
                    invalidate_cache=False,
                )
                cluster_column = f"{graph_label}_{cluster_label}"
                cluster_group = store.load_artifact(cluster_ref)
                cluster_values = cast(Any, cluster_group["values"])
                values = np.asarray(cluster_values[:])
                _labels, counts = np.unique(values, return_counts=True)
                metrics = IntegrationMetrics(
                    nClusters=int(len(counts)),
                    minClusterCells=int(counts.min()),
                    minClusterFraction=float(counts.min() / len(values)),
                    modalityWeightsValid=weights_valid,
                )
                for assay, native in native_labels.items():
                    metrics.adjustedRandByAssay[assay] = float(
                        adjusted_rand_score(native, values)
                    )
                    metrics.normalizedMutualInformationByAssay[assay] = float(
                        normalized_mutual_info_score(native, values)
                    )
                    evidence_ids.extend(
                        [
                            f"integration:{integration_id}:ari:{assay}",
                            f"integration:{integration_id}:nmi:{assay}",
                        ]
                    )
                for column in experimental_handoff.preservationColumns:
                    try:
                        value = float(
                            store.metric_graph_connectivity(
                                column,
                                graph=graph_ref,
                                from_assay=plan.primaryAssay,
                                cell_key="I",
                            )
                        )
                        if np.isfinite(value):
                            metrics.biologicalConnectivity[column] = value
                            evidence_ids.append(
                                f"integration:{integration_id}:graphConnectivity:{column}"
                            )
                    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
                        warnings.append(
                            f"Graph connectivity for {column!r} unavailable: {exc}"
                        )
                if method == "wnn":
                    evidence_ids.append(f"integration:{integration_id}:modalityWeights")
                reasons: list[str] = []
                missing_connectivity = sorted(
                    set(experimental_handoff.preservationColumns)
                    - set(metrics.biologicalConnectivity)
                )
                if missing_connectivity:
                    reasons.append(
                        "trusted-label connectivity is unavailable for "
                        + ", ".join(missing_connectivity)
                    )
                if metrics.nClusters is None or metrics.nClusters < 2:
                    reasons.append("fewer than two clusters")
                if (
                    metrics.minClusterCells is None
                    or metrics.minClusterCells < config.minClusterCells
                ):
                    reasons.append("smallest cluster is below the configured minimum")
                if method == "wnn" and weights_valid is not True:
                    reasons.append("WNN modality weights are invalid")
                evaluations.append(
                    IntegrationCandidateEvaluation(
                        integrationId=integration_id,
                        method=cast(Any, method),
                        assays=assays,
                        status="done",
                        eligible=not reasons,
                        resolution=resolution,
                        graphArtifact=ArtifactRecord.from_ref(graph_ref),
                        clusterArtifact=ArtifactRecord.from_ref(cluster_ref),
                        clusterColumn=cluster_column,
                        metrics=metrics,
                        evidenceIds=evidence_ids,
                        eligibilityReasons=reasons,
                        warnings=warnings,
                    )
                )
            except Exception as exc:
                evaluations.append(
                    IntegrationCandidateEvaluation(
                        integrationId=integration_id,
                        method=cast(Any, method),
                        assays=assays,
                        status="failed",
                        resolution=resolution,
                        graphArtifact=ArtifactRecord.from_ref(graph_ref),
                        evidenceIds=evidence_ids,
                        warnings=warnings,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        if started is not None:
            self.save_integration_checkpoint(
                store,
                started,
                report,
                method,
                evaluations,
                parent_reports,
            )
            if actions is not None:
                actions.append(f"checkpoint_integration:{method}")
        eligible_count = sum(
            value.status == "done" and value.eligible for value in evaluations
        )
        failed_count = sum(value.status == "failed" for value in evaluations)
        logger.info(
            f"Completed {method.upper()} integration evaluation: "
            f"eligible={eligible_count}, failed={failed_count}, "
            f"total={len(evaluations)}"
        )
        return evaluations
