"""Ingest, enrichment, HTO, and experimental-context workflow stages."""

from collections.abc import Mapping, Sequence
from typing import Any, cast

from ...datastore.datastore import DataStore
from ...utils.logging import logger
from ..data_enrichment import (
    DataEnrichmentAgent,
    DataEnrichmentContext,
    DataEnrichmentReport,
)
from ..experimental_context import (
    ExperimentalContextAgent,
    ExperimentalContextResult,
)
from ..ingest import IngestResult
from ..persistence import (
    AgentInvocation,
    AgentReportReference,
    AgentWorkflowRun,
)
from . import journal
from .models import (
    OrchestrationRequestRecord,
    OrchestrationResumeRecord,
    WorkflowNeedsInput,
    WorkflowQuestion,
    WorkflowStageAttempt,
    WorkflowStageLink,
)


class ContextStagesMixin:
    """Stages that establish study and experimental context."""

    model: Any

    def record_ingest_stage(
        self,
        store: DataStore,
        prefix: str,
        workflow: AgentWorkflowRun,
        request_record: OrchestrationRequestRecord,
        ingest_result: IngestResult,
    ) -> WorkflowStageAttempt:
        existing = journal._validated_done_outcome(
            store,
            prefix,
            workflow.workflowRunId,
            "ingest",
            request_record,
            [],
        )
        if existing is not None:
            logger.info(
                f"Workflow {workflow.workflowRunId}: reusing persisted ingest stage"
            )
            return existing
        started = journal._start_attempt(
            store.zw,
            prefix,
            workflow.workflowRunId,
            "ingest",
            request_record,
            [],
            inputs={
                "sourcePath": request_record.request.sourcePath,
                "zarrPath": ingest_result.zarrPath,
                "format": ingest_result.format,
                "acceptedActions": ingest_result.acceptedActions,
            },
        )
        outcome = journal._complete_attempt(
            started,
            status="done",
            outputs={
                "format": ingest_result.format,
                "assayNames": ingest_result.assayNames,
                "pairingProvenance": (
                    "singleSourceSharedCellAxis"
                    if ingest_result.format in {"h5ad", "10x_h5", "10x_dir"}
                    and len(ingest_result.assayNames) > 1
                    else None
                ),
                "summary": ingest_result.summary,
            },
            actions=ingest_result.actions,
            notes=ingest_result.notes,
        )
        journal._save_outcome(store.zw, prefix, outcome)
        logger.info(
            f"Workflow {workflow.workflowRunId}: ingest recorded "
            f"{len(ingest_result.assayNames)} assay(s)"
        )
        return outcome

    def data_enrichment_stage(
        self,
        store: DataStore,
        workflow: AgentWorkflowRun,
        request_record: OrchestrationRequestRecord,
        parents: Sequence[WorkflowStageLink],
        answers: Mapping[str, Any],
        *,
        resume_record: OrchestrationResumeRecord | None = None,
    ) -> tuple[WorkflowStageAttempt, DataEnrichmentReport]:
        prefix = journal._ensure_orchestration_store(store)
        existing = journal._validated_done_outcome(
            store,
            prefix,
            workflow.workflowRunId,
            "data_enrichment",
            request_record,
            parents,
        )
        if existing is not None:
            logger.info(
                f"Workflow {workflow.workflowRunId}: reusing Data Enrichment report"
            )
            report = journal.load_stage_report(store, existing, DataEnrichmentReport)
            return existing, cast(DataEnrichmentReport, report)
        request = request_record.request
        selected_assays = request.analysisAssays or list(store.assay_names)
        logger.info(
            f"Workflow {workflow.workflowRunId}: Data Enrichment will inspect "
            f"{len(selected_assays)} assay(s)"
        )
        unknown = sorted(set(selected_assays) - set(store.assay_names))
        if unknown:
            return journal.failed_stage(
                store,
                workflow,
                request_record,
                "data_enrichment",
                parents,
                f"Unknown requested assays: {unknown}",
                resume_record=resume_record,
            ), DataEnrichmentReport.get_blank()
        started = journal._start_attempt(
            store.zw,
            prefix,
            workflow.workflowRunId,
            "data_enrichment",
            request_record,
            parents,
            inputs={
                "studyContext": request.studyContext,
                "assays": selected_assays,
                "allowDownload": request_record.config.allowDownloads,
                "dataEnrichmentContext": answers.get("dataEnrichmentContext"),
            },
            resume_record=resume_record,
        )
        actions: list[str] = []
        operations: list[dict[str, Any]] = []
        try:
            reset_selection = self.should_reset_selection(
                request_record, prefix, store, workflow
            )
            context_payload: dict[str, Any] = {"studyContext": request.studyContext}
            supplied_context = answers.get("dataEnrichmentContext")
            if isinstance(supplied_context, Mapping):
                context_payload.update(dict(supplied_context))
            elif isinstance(supplied_context, str) and supplied_context.strip():
                context_payload["experimentalDetails"] = [supplied_context.strip()]
            enrichment_context = DataEnrichmentContext.model_validate(context_payload)
            recovered = journal._recover_persisted_stage_report(
                store,
                started,
                agent_name="data_enrichment",
                expected_type=DataEnrichmentReport,
            )
            if reset_selection:
                if recovered is None:
                    store.filter_cells([], [], [], reset_previous=True)
                logger.info(
                    f"Workflow {workflow.workflowRunId}: reset the shared cell "
                    "selection before agent-managed QC"
                )
                actions.append("reset_cell_selection")
                operations.append(
                    {
                        "operation": "filter_cells",
                        "attrs": [],
                        "lows": [],
                        "highs": [],
                        "resetPrevious": True,
                        "keepBounds": False,
                        "invalidateCache": False,
                    }
                )
            if recovered is not None:
                recovered_report, reference = recovered
                report = cast(DataEnrichmentReport, recovered_report)
                actions.append("recover_persisted_data_enrichment_report")
            else:
                logger.info(
                    f"Workflow {workflow.workflowRunId}: invoking Data Enrichment"
                )
                agent = DataEnrichmentAgent(
                    self.model,
                    config=request_record.config.agentRunConfig,
                )
                report = agent.run(
                    store,
                    context=enrichment_context,
                    assays=selected_assays,
                    cache_dir=request_record.config.cacheDir,
                    allow_download=request_record.config.allowDownloads,
                )
                saved_report, reference = journal._save_stage_report(
                    store,
                    started,
                    report,
                    invocation=AgentInvocation(
                        agentName="data_enrichment",
                        inputs={
                            "context": enrichment_context.model_dump(mode="json"),
                            "assays": selected_assays,
                            "cacheDir": request_record.config.cacheDir,
                            "allowDownload": request_record.config.allowDownloads,
                        },
                        runConfig=agent.config,
                    ),
                    expected_type=DataEnrichmentReport,
                )
                report = cast(DataEnrichmentReport, saved_report)
            logger.info(
                f"Workflow {workflow.workflowRunId}: Data Enrichment returned "
                f"status={report.status!r}, policies={len(report.policies)}, "
                f"inspections={len(report.inspections)}"
            )
            if report.status == "needsInput":
                questions = [
                    WorkflowQuestion(
                        questionId="dataEnrichmentContext",
                        question=(
                            "\n".join(report.unresolvedQuestions)
                            or "Provide the missing study-context details."
                        ),
                        evidenceIds=list(report.evidenceIds),
                    )
                ]
                outcome = journal._complete_attempt(
                    started,
                    status="needsInput",
                    report_references=[reference],
                    actions=actions,
                    outputs={"operations": operations},
                    needs_input=WorkflowNeedsInput(questions=questions),
                    notes=report.limitations,
                )
            elif report.status == "failed":
                outcome = journal._complete_attempt(
                    started,
                    status="failed",
                    report_references=[reference],
                    actions=actions,
                    outputs={"operations": operations},
                    error="; ".join(report.limitations),
                )
            else:
                outcome = journal._complete_attempt(
                    started,
                    status="done",
                    report_references=[reference],
                    actions=actions,
                    outputs={
                        "studyContextSummary": report.studyContextSummary.model_dump(
                            mode="json"
                        ),
                        "operations": operations,
                    },
                    notes=report.limitations,
                )
            journal._save_outcome(store.zw, prefix, outcome)
            if outcome.status == "failed":
                journal.finalize_failed(
                    store, workflow, outcome.error or "enrichment failed"
                )
            return outcome, report
        except Exception as exc:
            outcome = journal.finish_exception(
                store,
                prefix,
                workflow,
                started,
                exc,
                actions=actions,
                outputs={"operations": operations},
            )
            return outcome, DataEnrichmentReport.get_blank()

    def _hto_stage(
        self,
        store: DataStore,
        workflow: AgentWorkflowRun,
        request_record: OrchestrationRequestRecord,
        parents: Sequence[WorkflowStageLink],
        enrichment: DataEnrichmentReport,
        *,
        resume_record: OrchestrationResumeRecord | None = None,
    ) -> WorkflowStageAttempt:
        prefix = journal._ensure_orchestration_store(store)
        existing = journal._validated_done_outcome(
            store,
            prefix,
            workflow.workflowRunId,
            "hto_demultiplexing",
            request_record,
            parents,
        )
        if existing is not None:
            logger.info(
                f"Workflow {workflow.workflowRunId}: reusing HTO demultiplexing stage"
            )
            return existing
        eligible_hto = sum(
            policy.assayModality == "HTO" and policy.demultiplexEligible
            for policy in enrichment.policies
        )
        logger.info(
            f"Workflow {workflow.workflowRunId}: HTO stage found "
            f"{eligible_hto} eligible assay(s)"
        )
        started = journal._start_attempt(
            store.zw,
            prefix,
            workflow.workflowRunId,
            "hto_demultiplexing",
            request_record,
            parents,
            inputs={
                "policies": [
                    value.model_dump(mode="json") for value in enrichment.policies
                ]
            },
            resume_record=resume_record,
        )
        actions: list[str] = []
        outputs: dict[str, Any] = {
            "htoIdentityColumns": [],
            "operations": [],
        }
        try:
            metadata_columns: list[str] = []
            token = workflow.workflowRunId[:12]
            inspections = {value.assay: value for value in enrichment.inspections}
            for policy in enrichment.policies:
                if policy.assayModality == "RNA":
                    inspection = inspections.get(policy.assay)
                    observed_families = (
                        {
                            value.family
                            for value in inspection.families
                            if value.count > 0
                        }
                        if inspection is not None
                        else set()
                    )
                    assay = store.get_assay(policy.assay)
                    if "mitochondrial" in observed_families:
                        column = f"{policy.assay}_percentMito"
                        if column not in store.cells.columns:
                            assay.add_percent_feature(r"^(MT-|mt-)", column)
                            actions.append(f"compute_percent_mito:{policy.assay}")
                        if column in store.cells.columns:
                            metadata_columns.append(column)
                        cast(list[dict[str, Any]], outputs["operations"]).append(
                            {
                                "operation": "add_percent_feature",
                                "assay": policy.assay,
                                "pattern": r"^(MT-|mt-)",
                                "column": column,
                            }
                        )
                    if "ribosomal" in observed_families:
                        column = f"{policy.assay}_percentRibo"
                        pattern = r"^(RPS|RPL|MRPS|MRPL|Rps|Rpl|Mrps|Mrpl)"
                        if column not in store.cells.columns:
                            assay.add_percent_feature(pattern, column)
                            actions.append(f"compute_percent_ribo:{policy.assay}")
                        if column in store.cells.columns:
                            metadata_columns.append(column)
                        cast(list[dict[str, Any]], outputs["operations"]).append(
                            {
                                "operation": "add_percent_feature",
                                "assay": policy.assay,
                                "pattern": pattern,
                                "column": column,
                            }
                        )
                if policy.assayModality != "HTO" or not policy.demultiplexEligible:
                    continue
                label = f"agent_{token}_{journal._safe_label(policy.assay)}_identity"
                column = store.mark_hto_identities(
                    from_assay=policy.assay,
                    cell_key="I",
                    label=label,
                    random_seed=0,
                    invalidate_cache=False,
                )
                cast(list[str], outputs["htoIdentityColumns"]).append(column)
                cast(list[dict[str, Any]], outputs["operations"]).append(
                    {
                        "operation": "mark_hto_identities",
                        "assay": policy.assay,
                        "cellKey": "I",
                        "label": label,
                        "randomSeed": 0,
                        "invalidateCache": False,
                    }
                )
                actions.append(f"demultiplex_hto:{policy.assay}")
            outputs["metadataColumns"] = list(
                dict.fromkeys(
                    [*metadata_columns, *cast(list[str], outputs["htoIdentityColumns"])]
                )
            )
            outcome = journal._complete_attempt(
                started,
                status="done",
                outputs=outputs,
                actions=actions,
            )
            journal._save_outcome(store.zw, prefix, outcome)
            logger.info(
                f"Workflow {workflow.workflowRunId}: HTO stage produced "
                f"{len(cast(list[str], outputs['htoIdentityColumns']))} "
                "identity column(s)"
            )
            return outcome
        except Exception as exc:
            return journal.finish_exception(
                store,
                prefix,
                workflow,
                started,
                exc,
                actions=actions,
                outputs=outputs,
            )

    def experimental_context_stage(
        self,
        store: DataStore,
        workflow: AgentWorkflowRun,
        request_record: OrchestrationRequestRecord,
        parents: Sequence[WorkflowStageLink],
        enrichment_reference: AgentReportReference,
        hto_identity_columns: Sequence[str],
        answers: Mapping[str, Any],
        *,
        resume_record: OrchestrationResumeRecord | None = None,
    ) -> tuple[WorkflowStageAttempt, ExperimentalContextResult]:
        prefix = journal._ensure_orchestration_store(store)
        existing = journal._validated_done_outcome(
            store,
            prefix,
            workflow.workflowRunId,
            "experimental_context",
            request_record,
            parents,
        )
        if existing is not None:
            logger.info(
                f"Workflow {workflow.workflowRunId}: reusing Experimental Context "
                "report"
            )
            report = journal.load_stage_report(
                store, existing, ExperimentalContextResult
            )
            return existing, cast(ExperimentalContextResult, report)
        paused = journal._validated_done_outcome(
            store,
            prefix,
            workflow.workflowRunId,
            "experimental_context",
            request_record,
            parents,
            required_status="needsInput",
        )
        directions = dict(request_record.request.experimentalDirections)
        directions["htoIdentityColumns"] = list(hto_identity_columns)
        supplied_directions = answers.get("experimentalDirections")
        if isinstance(supplied_directions, Mapping):
            directions.update(dict(supplied_directions))
        elif isinstance(supplied_directions, str) and supplied_directions.strip():
            directions["callerAnswer"] = supplied_directions.strip()
        started = journal._start_attempt(
            store.zw,
            prefix,
            workflow.workflowRunId,
            "experimental_context",
            request_record,
            parents,
            inputs={
                "studyContext": request_record.request.studyContext,
                "cellKey": "I",
                "directions": directions,
            },
            resume_record=resume_record,
        )
        logger.info(
            f"Workflow {workflow.workflowRunId}: Experimental Context will evaluate "
            f"{len(hto_identity_columns)} HTO identity column(s)"
        )
        try:
            unsafe_resolution = (
                journal._unsafe_context_resolution(supplied_directions)
                if paused is not None
                and paused.outputs.get("unsafeBatchCorrection") is True
                else None
            )
            actions: list[str] = []
            recovered = journal._recover_persisted_stage_report(
                store,
                started,
                agent_name="experimental_context",
                expected_type=ExperimentalContextResult,
            )
            if recovered is not None:
                recovered_report, reference = recovered
                report = cast(ExperimentalContextResult, recovered_report)
                actions.append("recover_persisted_experimental_context_report")
            else:
                parent_reports = [journal._report_link(enrichment_reference)]
                if unsafe_resolution == "skip":
                    assert paused is not None
                    if not paused.reportReferences:
                        raise ValueError(
                            "Unsafe Experimental Context pause has no persisted report"
                        )
                    prior_report = cast(
                        ExperimentalContextResult,
                        journal.load_stage_report(
                            store,
                            paused,
                            ExperimentalContextResult,
                        ),
                    )
                    prior_plan = prior_report.decision.batchCorrection
                    skip_plan = prior_plan.model_copy(
                        update={
                            "action": "skip",
                            "batchColumns": [],
                            "metricsRequired": [],
                            "rationale": (
                                "The caller explicitly skipped Harmony after "
                                "reviewing the persisted unsafe batch-correction "
                                "evidence."
                            ),
                        }
                    )
                    decision = prior_report.decision.model_copy(
                        update={
                            "batchCorrection": skip_plan,
                            "needsInput": [],
                        }
                    )
                    report = prior_report.model_copy(
                        update={
                            "status": "done",
                            "decision": decision,
                            "notes": [
                                *prior_report.notes,
                                "Caller explicitly skipped Harmony after an unsafe "
                                "result.",
                            ],
                        }
                    )
                    parent_reports.append(
                        journal._report_link(paused.reportReferences[0])
                    )
                    run_config = request_record.config.agentRunConfig
                    actions.append("resolve_unsafe_batch_correction:skip")
                else:
                    logger.info(
                        f"Workflow {workflow.workflowRunId}: invoking Experimental "
                        "Context"
                    )
                    agent = ExperimentalContextAgent(
                        self.model,
                        config=request_record.config.agentRunConfig,
                    )
                    report = agent.run(
                        store,
                        study_context=request_record.request.studyContext,
                        cell_key="I",
                        directions=directions,
                    )
                    run_config = agent.config
                saved_report, reference = journal._save_stage_report(
                    store,
                    started,
                    report,
                    invocation=AgentInvocation(
                        agentName="experimental_context",
                        parentReports=parent_reports,
                        inputs={
                            "studyContext": request_record.request.studyContext,
                            "cellKey": "I",
                            "directions": directions,
                            "unsafeResolution": unsafe_resolution,
                        },
                        runConfig=run_config,
                    ),
                    expected_type=ExperimentalContextResult,
                )
                report = cast(ExperimentalContextResult, saved_report)
            logger.info(
                f"Workflow {workflow.workflowRunId}: Experimental Context returned "
                f"status={report.status!r}, batchAction="
                f"{report.decision.batchCorrection.action!r}"
            )
            if report.status == "needsInput":
                questions = [
                    WorkflowQuestion(
                        questionId="experimentalDirections",
                        question=(
                            "\n".join(report.decision.needsInput)
                            or "Provide the missing experimental-context details."
                        ),
                        evidenceIds=list(report.decision.evidenceIds),
                    )
                ]
                outcome = journal._complete_attempt(
                    started,
                    status="needsInput",
                    report_references=[reference],
                    needs_input=WorkflowNeedsInput(questions=questions),
                    notes=report.notes,
                )
            elif report.status == "failed":
                outcome = journal._complete_attempt(
                    started,
                    status="failed",
                    report_references=[reference],
                    error="; ".join(report.notes) or "Experimental Context failed",
                )
            elif report.decision.batchCorrection.action == "unsafe":
                batch_plan = report.decision.batchCorrection
                outcome = journal._complete_attempt(
                    started,
                    status="needsInput",
                    report_references=[reference],
                    outputs={
                        "unsafeBatchCorrection": True,
                        "batchCorrection": batch_plan.model_dump(mode="json"),
                    },
                    needs_input=WorkflowNeedsInput(
                        questions=[
                            WorkflowQuestion(
                                questionId="experimentalDirections",
                                question=(
                                    "Batch correction is unsafe for the persisted "
                                    "experimental design. Explicitly skip Harmony or "
                                    "provide study-design clarification."
                                ),
                                options=["skipHarmony", "provideClarification"],
                                evidenceIds=list(batch_plan.evidenceIds),
                            )
                        ]
                    ),
                    notes=report.notes,
                )
            else:
                outcome = journal._complete_attempt(
                    started,
                    status="done",
                    report_references=[reference],
                    outputs={
                        "cellQc": report.cellQc.model_dump(mode="json"),
                        "qcProfiles": [
                            value.model_dump(mode="json") for value in report.qcProfiles
                        ],
                        "htoIdentityColumns": report.htoIdentityColumns,
                        "metadataColumns": report.htoIdentityColumns,
                    },
                    actions=actions,
                    notes=report.notes,
                )
            journal._save_outcome(store.zw, prefix, outcome)
            if outcome.status == "failed":
                journal.finalize_failed(
                    store, workflow, outcome.error or "context failed"
                )
            return outcome, report
        except Exception as exc:
            outcome = journal.finish_exception(store, prefix, workflow, started, exc)
            return outcome, ExperimentalContextResult.get_blank()

    def should_reset_selection(
        self,
        request_record: OrchestrationRequestRecord,
        prefix: str,
        store: DataStore,
        workflow: AgentWorkflowRun | None,
    ) -> bool:
        request = request_record.request
        if workflow is None:
            return False
        ingest_outcome = journal._validated_done_outcome(
            store,
            prefix,
            workflow.workflowRunId,
            "ingest",
            request_record,
            [],
        )
        if ingest_outcome is None:
            raise RuntimeError("Cannot resolve reset policy without ingest outcome")
        if ingest_outcome.outputs.get("format") != "zarr":
            return True
        return request.resetCellSelection is True
