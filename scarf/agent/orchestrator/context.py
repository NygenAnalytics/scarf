"""Ingest, enrichment, HTO, and experimental-context workflow stages."""

import re
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np

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
    NamedArtifactSource,
)
from ..ingest import IngestResult
from ..persistence import (
    AgentInvocation,
    AgentReportReference,
    AgentWorkflowRun,
)
from ..types import ArtifactReferenceModel
from . import journal
from .models import (
    OrchestrationRequestRecord,
    OrchestrationResumeRecord,
    WorkflowNeedsInput,
    WorkflowQuestion,
    WorkflowStageAttempt,
    WorkflowStageLink,
    artifact_model_to_ref,
)


class ContextStagesMixin:
    """Stages that establish study and experimental context."""

    model: Any

    @staticmethod
    def _experimental_context_artifacts(
        cell_selection: ArtifactReferenceModel,
        quality_metric_artifacts: Sequence[NamedArtifactSource],
        hto_identity_artifacts: Sequence[NamedArtifactSource],
    ) -> dict[str, ArtifactReferenceModel]:
        artifacts = {"cellSelection": cell_selection}
        for sources, expected_kind in (
            (quality_metric_artifacts, "quality_metric"),
            (hto_identity_artifacts, "hto_identity"),
        ):
            for source in sources:
                if not isinstance(source, NamedArtifactSource):
                    raise TypeError(
                        "Experimental Context artifacts must be named sources"
                    )
                if source.artifact.kind != expected_kind:
                    raise ValueError(
                        f"Experimental Context source {source.name!r} must "
                        f"reference a {expected_kind!r} artifact"
                    )
                if source.name in artifacts:
                    raise ValueError(
                        f"Duplicate Experimental Context artifact name {source.name!r}"
                    )
                artifacts[source.name] = source.artifact
        return artifacts

    @staticmethod
    def _named_stage_artifacts(
        outcome: WorkflowStageAttempt,
        output_name: str,
        expected_kind: str,
    ) -> list[NamedArtifactSource]:
        raw_sources = outcome.outputs.get(output_name)
        if not isinstance(raw_sources, list):
            raise ValueError(f"{output_name} must be a list of named artifacts")
        sources = [
            NamedArtifactSource.model_validate(raw_source) for raw_source in raw_sources
        ]
        named_artifacts: dict[str, ArtifactReferenceModel] = {}
        for source in sources:
            if source.name in named_artifacts:
                raise ValueError(f"{output_name} artifact names must be unique")
            if source.artifact.kind != expected_kind:
                raise ValueError(
                    f"{output_name} must reference {expected_kind!r} artifacts"
                )
            if outcome.artifacts.get(source.name) != source.artifact:
                raise ValueError(
                    f"{output_name} artifact {source.name!r} is not persisted "
                    "as a stage artifact"
                )
            named_artifacts[source.name] = source.artifact
        persisted_artifacts = {
            name: artifact
            for name, artifact in outcome.artifacts.items()
            if artifact.kind == expected_kind
        }
        if named_artifacts != persisted_artifacts:
            raise ValueError(
                f"{output_name} must name every persisted {expected_kind!r} artifact"
            )
        return sources

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
        cell_selection = store.snapshot_cell_selection("I")
        cell_selection_model = ArtifactReferenceModel.from_artifact_ref(cell_selection)
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
                "sourceCellColumn": "I",
            },
        )
        outcome = journal._complete_attempt(
            started,
            status="done",
            artifacts={"cellSelection": cell_selection_model},
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
                "operations": [
                    {
                        "operation": "snapshot_cell_selection",
                        "sourceColumn": "I",
                        "artifact": cell_selection_model.model_dump(mode="json"),
                    }
                ],
            },
            actions=[*ingest_result.actions, "snapshot_cell_selection"],
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
        cell_selection: ArtifactReferenceModel,
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
                artifacts={"cellSelection": cell_selection},
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
                "cellSelection": cell_selection.model_dump(mode="json"),
                "allowDownload": request_record.config.allowDownloads,
                "dataEnrichmentContext": answers.get("dataEnrichmentContext"),
            },
            resume_record=resume_record,
        )
        actions: list[str] = []
        operations: list[dict[str, Any]] = []
        try:
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
                            "cellSelection": cell_selection.model_dump(mode="json"),
                            "cacheDir": request_record.config.cacheDir,
                            "allowDownload": request_record.config.allowDownloads,
                        },
                        artifacts={"cellSelection": cell_selection},
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
                    artifacts={"cellSelection": cell_selection},
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
                    artifacts={"cellSelection": cell_selection},
                    actions=actions,
                    outputs={"operations": operations},
                    error="; ".join(report.limitations),
                )
            else:
                outcome = journal._complete_attempt(
                    started,
                    status="done",
                    report_references=[reference],
                    artifacts={"cellSelection": cell_selection},
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
                artifacts={"cellSelection": cell_selection},
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
        cell_selection: ArtifactReferenceModel,
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
            self._named_stage_artifacts(
                existing,
                "qualityMetricArtifacts",
                "quality_metric",
            )
            self._named_stage_artifacts(
                existing,
                "htoIdentityArtifacts",
                "hto_identity",
            )
            logger.info(
                f"Workflow {workflow.workflowRunId}: reusing HTO demultiplexing stage"
            )
            return existing
        cell_selection_ref = artifact_model_to_ref(cell_selection)
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
                "cellSelection": cell_selection.model_dump(mode="json"),
                "policies": [
                    value.model_dump(mode="json") for value in enrichment.policies
                ],
            },
            resume_record=resume_record,
        )
        actions: list[str] = []
        outputs: dict[str, Any] = {
            "htoIdentityArtifacts": [],
            "qualityMetricArtifacts": [],
            "operations": [],
        }
        artifacts: dict[str, ArtifactReferenceModel] = {"cellSelection": cell_selection}
        try:
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
                    feature_ids = np.asarray(assay.feats.fetch_all("ids")).astype(str)
                    feature_names = np.asarray(assay.feats.fetch_all("names")).astype(
                        str
                    )
                    family_patterns = (
                        (
                            "mitochondrial",
                            r"^(MT-|mt-)",
                            "percentMito",
                            "percent_mito",
                        ),
                        (
                            "ribosomal",
                            r"^(RPS|RPL|MRPS|MRPL|Rps|Rpl|Mrps|Mrpl)",
                            "percentRibo",
                            "percent_ribo",
                        ),
                    )
                    for (
                        family,
                        pattern,
                        artifact_suffix,
                        action_suffix,
                    ) in family_patterns:
                        if family not in observed_families:
                            continue
                        compiled = re.compile(pattern)
                        mask = np.fromiter(
                            (
                                compiled.search(feature_id) is not None
                                or compiled.search(feature_name) is not None
                                for feature_id, feature_name in zip(
                                    feature_ids,
                                    feature_names,
                                    strict=True,
                                )
                            ),
                            dtype=bool,
                            count=assay.feats.N,
                        )
                        if not mask.any():
                            continue
                        features_ref = store.set_feature_selection(
                            from_assay=policy.assay,
                            mask=mask,
                            invalidate_cache=False,
                        )
                        metric_ref = store.run_feature_percentage(
                            cell_selection_ref,
                            features_ref,
                            invalidate_cache=False,
                        )
                        features_model = ArtifactReferenceModel.from_artifact_ref(
                            features_ref
                        )
                        metric_model = ArtifactReferenceModel.from_artifact_ref(
                            metric_ref
                        )
                        artifact_name = f"{policy.assay}_{artifact_suffix}"
                        if artifact_name in artifacts:
                            raise ValueError(
                                f"Duplicate generated artifact name {artifact_name!r}"
                            )
                        source = NamedArtifactSource(
                            name=artifact_name,
                            artifact=metric_model,
                        )
                        artifacts[f"{artifact_name}_features"] = features_model
                        artifacts[artifact_name] = metric_model
                        cast(
                            list[dict[str, Any]],
                            outputs["qualityMetricArtifacts"],
                        ).append(source.model_dump(mode="json"))
                        cast(list[dict[str, Any]], outputs["operations"]).append(
                            {
                                "operation": "run_feature_percentage",
                                "assay": policy.assay,
                                "family": family,
                                "pattern": pattern,
                                "cellSelection": cell_selection.model_dump(mode="json"),
                                "features": features_model.model_dump(mode="json"),
                                "artifact": metric_model.model_dump(mode="json"),
                            }
                        )
                        actions.append(f"compute_{action_suffix}:{policy.assay}")
                if policy.assayModality != "HTO" or not policy.demultiplexEligible:
                    continue
                identity_ref = store.run_hto_demultiplexing(
                    cell_selection_ref,
                    from_assay=policy.assay,
                    random_seed=0,
                    invalidate_cache=False,
                )
                identity_model = ArtifactReferenceModel.from_artifact_ref(identity_ref)
                artifact_name = f"{policy.assay}_htoIdentity"
                if artifact_name in artifacts:
                    raise ValueError(
                        f"Duplicate generated artifact name {artifact_name!r}"
                    )
                source = NamedArtifactSource(
                    name=artifact_name,
                    artifact=identity_model,
                )
                artifacts[artifact_name] = identity_model
                cast(list[dict[str, Any]], outputs["htoIdentityArtifacts"]).append(
                    source.model_dump(mode="json")
                )
                cast(list[dict[str, Any]], outputs["operations"]).append(
                    {
                        "operation": "run_hto_demultiplexing",
                        "assay": policy.assay,
                        "cellSelection": cell_selection.model_dump(mode="json"),
                        "randomSeed": 0,
                        "invalidateCache": False,
                        "artifact": identity_model.model_dump(mode="json"),
                    }
                )
                actions.append(f"demultiplex_hto:{policy.assay}")
            outcome = journal._complete_attempt(
                started,
                status="done",
                artifacts=artifacts,
                outputs=outputs,
                actions=actions,
            )
            journal._save_outcome(store.zw, prefix, outcome)
            logger.info(
                f"Workflow {workflow.workflowRunId}: HTO stage produced "
                f"{len(cast(list[dict[str, Any]], outputs['htoIdentityArtifacts']))} "
                "identity artifact(s)"
            )
            return outcome
        except Exception as exc:
            return journal.finish_exception(
                store,
                prefix,
                workflow,
                started,
                exc,
                artifacts=artifacts,
                actions=actions,
                outputs=outputs,
            )

    def experimental_context_stage(
        self,
        store: DataStore,
        workflow: AgentWorkflowRun,
        request_record: OrchestrationRequestRecord,
        parents: Sequence[WorkflowStageLink],
        cell_selection: ArtifactReferenceModel,
        enrichment_reference: AgentReportReference,
        quality_metric_artifacts: Sequence[NamedArtifactSource],
        hto_identity_artifacts: Sequence[NamedArtifactSource],
        answers: Mapping[str, Any],
        *,
        resume_record: OrchestrationResumeRecord | None = None,
    ) -> tuple[WorkflowStageAttempt, ExperimentalContextResult]:
        prefix = journal._ensure_orchestration_store(store)
        context_artifacts = self._experimental_context_artifacts(
            cell_selection,
            quality_metric_artifacts,
            hto_identity_artifacts,
        )
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
            resolved_report = cast(ExperimentalContextResult, report)
            if existing.artifacts != context_artifacts:
                raise ValueError(
                    "Persisted Experimental Context stage artifacts are stale"
                )
            if resolved_report.cellSelection != cell_selection:
                raise ValueError(
                    "Persisted Experimental Context cell selection is stale"
                )
            if resolved_report.qualityMetricArtifacts != list(quality_metric_artifacts):
                raise ValueError(
                    "Persisted Experimental Context quality artifacts are stale"
                )
            if resolved_report.htoIdentityArtifacts != list(hto_identity_artifacts):
                raise ValueError(
                    "Persisted Experimental Context HTO artifacts are stale"
                )
            record = journal.load_agent_record(
                store,
                existing.reportReferences[0],
            )
            if record.invocation.artifacts != context_artifacts:
                raise ValueError(
                    "Persisted Experimental Context invocation artifacts are stale"
                )
            return existing, resolved_report
        cell_selection_ref = artifact_model_to_ref(cell_selection)
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
                "cellSelection": cell_selection.model_dump(mode="json"),
                "directions": directions,
                "qualityMetricArtifacts": [
                    source.model_dump(mode="json")
                    for source in quality_metric_artifacts
                ],
                "htoIdentityArtifacts": [
                    source.model_dump(mode="json") for source in hto_identity_artifacts
                ],
            },
            resume_record=resume_record,
        )
        logger.info(
            f"Workflow {workflow.workflowRunId}: Experimental Context will evaluate "
            f"{len(quality_metric_artifacts)} quality metric artifact(s) and "
            f"{len(hto_identity_artifacts)} HTO identity artifact(s)"
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
                recovered_record = journal.load_agent_record(store, reference)
                if recovered_record.invocation.artifacts != context_artifacts:
                    raise ValueError(
                        "Recovered Experimental Context invocation artifacts are stale"
                    )
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
                    if paused.artifacts != context_artifacts:
                        raise ValueError(
                            "Paused Experimental Context stage artifacts are stale"
                        )
                    if (
                        prior_report.cellSelection != cell_selection
                        or prior_report.qualityMetricArtifacts
                        != list(quality_metric_artifacts)
                        or prior_report.htoIdentityArtifacts
                        != list(hto_identity_artifacts)
                    ):
                        raise ValueError(
                            "Paused Experimental Context exact inputs are stale"
                        )
                    paused_record = journal.load_agent_record(
                        store,
                        paused.reportReferences[0],
                    )
                    if paused_record.invocation.artifacts != context_artifacts:
                        raise ValueError(
                            "Paused Experimental Context invocation artifacts are stale"
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
                        cell_selection=cell_selection_ref,
                        directions=directions,
                        quality_metric_artifacts=quality_metric_artifacts,
                        hto_identity_artifacts=hto_identity_artifacts,
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
                            "cellSelection": cell_selection.model_dump(mode="json"),
                            "directions": directions,
                            "qualityMetricArtifacts": [
                                source.model_dump(mode="json")
                                for source in quality_metric_artifacts
                            ],
                            "htoIdentityArtifacts": [
                                source.model_dump(mode="json")
                                for source in hto_identity_artifacts
                            ],
                            "unsafeResolution": unsafe_resolution,
                        },
                        artifacts=context_artifacts,
                        runConfig=run_config,
                    ),
                    expected_type=ExperimentalContextResult,
                )
                report = cast(ExperimentalContextResult, saved_report)
            if report.cellSelection != cell_selection:
                raise ValueError(
                    "Experimental Context returned a different cell selection"
                )
            if report.qualityMetricArtifacts != list(quality_metric_artifacts):
                raise ValueError(
                    "Experimental Context returned different quality metric artifacts"
                )
            if report.htoIdentityArtifacts != list(hto_identity_artifacts):
                raise ValueError(
                    "Experimental Context returned different HTO identity artifacts"
                )
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
                    artifacts=context_artifacts,
                    needs_input=WorkflowNeedsInput(questions=questions),
                    notes=report.notes,
                )
            elif report.status == "failed":
                outcome = journal._complete_attempt(
                    started,
                    status="failed",
                    report_references=[reference],
                    artifacts=context_artifacts,
                    error="; ".join(report.notes) or "Experimental Context failed",
                )
            elif report.decision.batchCorrection.action == "unsafe":
                batch_plan = report.decision.batchCorrection
                outcome = journal._complete_attempt(
                    started,
                    status="needsInput",
                    report_references=[reference],
                    artifacts=context_artifacts,
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
                    artifacts=context_artifacts,
                    outputs={
                        "cellQc": report.cellQc.model_dump(mode="json"),
                        "qcProfiles": [
                            value.model_dump(mode="json") for value in report.qcProfiles
                        ],
                        "htoIdentityColumns": report.htoIdentityColumns,
                        "htoIdentityArtifacts": [
                            source.model_dump(mode="json")
                            for source in report.htoIdentityArtifacts
                        ],
                        "qualityMetricArtifacts": [
                            source.model_dump(mode="json")
                            for source in quality_metric_artifacts
                        ],
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
            outcome = journal.finish_exception(
                store,
                prefix,
                workflow,
                started,
                exc,
                artifacts=context_artifacts,
            )
            return outcome, ExperimentalContextResult.get_blank()
