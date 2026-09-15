"""Ingest, RNA enrichment, quality metrics, and experimental-context stages."""

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, cast

from ...datastore.datastore import DataStore
from ...utils.logging import logger
from ..data_enrichment.agent import DataEnrichmentAgent
from ..data_enrichment.contracts import (
    DataEnrichmentContext,
    DataEnrichmentReport,
)
from ..experimental_context.agent import ExperimentalContextAgent
from ..experimental_context.contracts import (
    ExperimentalContextResult,
    NamedArtifactSource,
)
from ..experimental_context.requirements import objective_evidence
from ..experimental_context.study import (
    StudyContract,
    build_study_contract,
    validate_objective_evidence,
)
from ..ingest import IngestResult
from ..ingest.manifest import DatasetManifest, is_author_label_column
from ..record_io import canonical_json_bytes
from ..types import AgentRunInfo, ArtifactReferenceModel
from . import journal
from .models import (
    OrchestrationRequestRecord,
    OrchestrationResumeRecord,
    StageEvidenceReference,
    WorkflowIdentity,
    WorkflowNeedsInput,
    WorkflowQuestion,
    WorkflowStageAttempt,
    WorkflowStageLink,
    artifact_model_to_ref,
)
from .rna import (
    selected_store_rna_assay,
    validate_rna_context,
    validate_rna_directions,
)


def _context_metadata_identity(
    store: DataStore, request_record: OrchestrationRequestRecord
) -> dict[str, str]:
    """Bind added metadata without repeating the original resume fingerprint scan."""
    from ..parameter_tuning.execution import _metadata_column_fingerprint

    original = request_record.inputIdentity.get("data", {}).get("metadata", {})
    # Public resume already validates every original column against this identity.
    # Enrichment may add columns afterward; those also bind committed context work.
    return {
        column: original[column]
        if column in original
        else _metadata_column_fingerprint(store.cells, column)
        for column in sorted(store.cells.columns)
    }


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
        workflow: WorkflowIdentity,
        request_record: OrchestrationRequestRecord,
        ingest_result: IngestResult,
        dataset_manifest: DatasetManifest | None = None,
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
            logger.debug(
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
                "datasetManifest": (
                    dataset_manifest.model_dump(mode="json")
                    if dataset_manifest is not None
                    else None
                ),
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
        logger.debug(
            f"Workflow {workflow.workflowRunId}: ingest recorded "
            f"{len(ingest_result.assayNames)} assay(s)"
        )
        return outcome

    def data_enrichment_stage(
        self,
        store: DataStore,
        workflow: WorkflowIdentity,
        request_record: OrchestrationRequestRecord,
        parents: Sequence[WorkflowStageLink],
        cell_selection: ArtifactReferenceModel,
        answers: Mapping[str, Any],
        *,
        resume_record: OrchestrationResumeRecord | None = None,
    ) -> tuple[WorkflowStageAttempt, DataEnrichmentReport]:
        selected = selected_store_rna_assay(store, request_record.request)
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
            logger.debug(
                f"Workflow {workflow.workflowRunId}: reusing Data Enrichment report"
            )
            report = journal.load_stage_report(store, existing, DataEnrichmentReport)
            report = cast(DataEnrichmentReport, report)
            if (
                len(report.policies) != 1
                or report.policies[0].assay != selected
                or report.policies[0].assayModality != "RNA"
            ):
                raise ValueError(
                    "Saved enrichment includes unsupported assays; start a new RNA workflow."
                )
            return existing, report
        request = request_record.request
        selected_assays = [selected]
        logger.debug(
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
                "studyObjective": request.studyObjective,
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
            context_payload: dict[str, Any] = {
                "studyContext": request.studyContext,
                "studyObjective": request.studyObjective,
            }
            supplied_context = answers.get("dataEnrichmentContext")
            if isinstance(supplied_context, Mapping):
                context_payload.update(dict(supplied_context))
            elif isinstance(supplied_context, str) and supplied_context.strip():
                context_payload["experimentalDetails"] = [supplied_context.strip()]
            enrichment_context = DataEnrichmentContext.model_validate(context_payload)
            recovered = journal._recover_persisted_stage_report(
                store,
                started,
                expected_type=DataEnrichmentReport,
            )
            if recovered is not None:
                recovered_report, reference = recovered
                report = cast(DataEnrichmentReport, recovered_report)
                actions.append("recover_persisted_data_enrichment_report")
            else:
                logger.debug(
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
                    on_attempt=journal.model_attempt_callback(
                        store,
                        prefix,
                        workflow.workflowRunId,
                        "data_enrichment",
                        {
                            "requestSha256": request_record.requestSha256,
                            "configSha256": request_record.configSha256,
                            "parents": [
                                parent.model_dump(mode="json") for parent in parents
                            ],
                            "inputs": started.inputs,
                        },
                    ),
                )
                saved_report, reference = journal._save_stage_report(
                    store,
                    started,
                    report,
                    expected_type=DataEnrichmentReport,
                )
                report = cast(DataEnrichmentReport, saved_report)
            logger.debug(
                f"Workflow {workflow.workflowRunId}: Data Enrichment returned "
                f"status={report.status!r}, policies={len(report.policies)}, "
                f"inspections={len(report.inspections)}"
            )
            if report.status == "needsInput":
                if request_record.config.inputPolicy == "unattended":
                    outcome = journal._complete_attempt(
                        started,
                        status="failed",
                        report_references=[reference],
                        artifacts={"cellSelection": cell_selection},
                        actions=actions,
                        outputs={"operations": operations},
                        error=(
                            "The unattended Data Enrichment stage returned an "
                            "unresolved decision"
                        ),
                        notes=report.limitations,
                    )
                    journal._save_outcome(store.zw, prefix, outcome)
                    return outcome, report
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

    def _rna_quality_metrics_stage(
        self,
        store: DataStore,
        workflow: WorkflowIdentity,
        request_record: OrchestrationRequestRecord,
        parents: Sequence[WorkflowStageLink],
        enrichment: DataEnrichmentReport,
        cell_selection: ArtifactReferenceModel,
        *,
        resume_record: OrchestrationResumeRecord | None = None,
    ) -> WorkflowStageAttempt:
        selected = selected_store_rna_assay(store, request_record.request)
        if (
            len(enrichment.policies) != 1
            or enrichment.policies[0].assay != selected
            or enrichment.policies[0].assayModality != "RNA"
        ):
            raise ValueError(
                "Quality metrics require enrichment of only the selected RNA assay"
            )
        prefix = journal._ensure_orchestration_store(store)
        existing = journal._validated_done_outcome(
            store,
            prefix,
            workflow.workflowRunId,
            "rna_quality_metrics",
            request_record,
            parents,
        )
        if existing is not None:
            if "percentageDefinitions" not in existing.outputs:
                raise ValueError(
                    "Saved RNA quality metrics lack exact percentage definitions; "
                    "start a new workflow. Existing analysis artifacts remain accessible."
                )
            self._named_stage_artifacts(
                existing,
                "qualityMetricArtifacts",
                "quality_metric",
            )
            hto_sources = self._named_stage_artifacts(
                existing,
                "htoIdentityArtifacts",
                "hto_identity",
            )
            if hto_sources:
                raise ValueError(
                    "Saved automatic HTO processing is unsupported; start a new RNA workflow."
                )
            logger.info("Reusing RNA quality metrics")
            return existing
        cell_selection_ref = artifact_model_to_ref(cell_selection)
        logger.info("Computing RNA quality metrics")
        started = journal._start_attempt(
            store.zw,
            prefix,
            workflow.workflowRunId,
            "rna_quality_metrics",
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
            from ..experimental_context.qc_evidence import _rna_percentage_feature_masks

            definitions = []
            for family, suffix, pattern, mask in _rna_percentage_feature_masks(
                store, selected
            ):
                definition = {
                    "family": family,
                    "pattern": pattern,
                    "matchedGenes": int(mask.sum()),
                }
                definitions.append(definition)
                if not mask.any():
                    definition["limitation"] = (
                        "No genes match this symbol-based percentage definition"
                    )
                    continue
                features_ref = store.set_feature_selection(
                    from_assay=selected, mask=mask, invalidate_cache=False
                )
                metric_ref = store.run_feature_percentage(
                    cell_selection_ref, features_ref, invalidate_cache=False
                )
                features_model = ArtifactReferenceModel.from_artifact_ref(features_ref)
                metric_model = ArtifactReferenceModel.from_artifact_ref(metric_ref)
                artifact_name = f"{selected}_{suffix}"
                source = NamedArtifactSource(name=artifact_name, artifact=metric_model)
                artifacts[f"{artifact_name}_features"] = features_model
                artifacts[artifact_name] = metric_model
                cast(list[dict[str, Any]], outputs["qualityMetricArtifacts"]).append(
                    source.model_dump(mode="json")
                )
                cast(list[dict[str, Any]], outputs["operations"]).append(
                    {
                        "operation": "run_feature_percentage",
                        "assay": selected,
                        **definition,
                        "cellSelection": cell_selection.model_dump(mode="json"),
                        "features": features_model.model_dump(mode="json"),
                        "artifact": metric_model.model_dump(mode="json"),
                    }
                )
                actions.append(
                    f"compute_{'percent_mito' if family == 'mitochondrial' else 'percent_ribo'}:{selected}"
                )
            outputs["percentageDefinitions"] = definitions
            outcome = journal._complete_attempt(
                started,
                status="done",
                artifacts=artifacts,
                outputs=outputs,
                actions=actions,
            )
            journal._save_outcome(store.zw, prefix, outcome)
            logger.info("RNA quality metrics completed")
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
        workflow: WorkflowIdentity,
        request_record: OrchestrationRequestRecord,
        parents: Sequence[WorkflowStageLink],
        cell_selection: ArtifactReferenceModel,
        enrichment_reference: StageEvidenceReference,
        quality_metric_artifacts: Sequence[NamedArtifactSource],
        hto_identity_artifacts: Sequence[NamedArtifactSource],
        answers: Mapping[str, Any],
        *,
        resume_record: OrchestrationResumeRecord | None = None,
    ) -> tuple[WorkflowStageAttempt, ExperimentalContextResult]:
        context_revision: dict[str, Any] = {}
        prior_context: ExperimentalContextResult | None = None
        selected = selected_store_rna_assay(store, request_record.request)
        if hto_identity_artifacts:
            raise ValueError(
                "Automatic HTO identities are unsupported by the RNA workflow"
            )
        prefix = journal._ensure_orchestration_store(store)
        context_artifacts = self._experimental_context_artifacts(
            cell_selection,
            quality_metric_artifacts,
            hto_identity_artifacts,
        )
        metadata_identity = _context_metadata_identity(store, request_record)
        existing = journal._validated_done_outcome(
            store,
            prefix,
            workflow.workflowRunId,
            "experimental_context",
            request_record,
            parents,
        )
        if existing is not None:
            saved_metadata = existing.inputs.get("metadataFingerprints")
            if saved_metadata is not None and saved_metadata != metadata_identity:
                raise ValueError(
                    "Experimental Context metadata changed; start a new analysis"
                )
            if saved_metadata is None:
                # Older compatible stages may already have committed downstream
                # metadata identities. Verify them; do not rewrite the old report.
                for tuning in journal._stage_starts(
                    store.zw, prefix, workflow.workflowRunId, "parameter_tuning"
                ):
                    if any(
                        metadata_identity.get(column) != digest
                        for column, digest in tuning.inputs.get(
                            "metadataFingerprints", {}
                        ).items()
                    ):
                        raise ValueError(
                            "Experimental Context metadata differs from saved tuning evidence; "
                            "restore the original inputs or start a new analysis"
                        )
            logger.debug(
                f"Workflow {workflow.workflowRunId}: reusing Experimental Context "
                "report"
            )
            report = journal.load_stage_report(
                store, existing, ExperimentalContextResult
            )
            resolved_report = cast(ExperimentalContextResult, report)
            validate_rna_context(resolved_report, selected)
            saved_contract = StudyContract.model_validate(
                existing.outputs.get("studyContract")
            )
            validate_objective_evidence(saved_contract)
            current_requirements, current_coverage = objective_evidence(
                study_context=request_record.request.studyContext,
                study_objective=request_record.request.studyObjective,
                experimental_result=resolved_report,
            )
            missing_questions = [
                item.model_dump(mode="json")
                for item in current_requirements
                if item.requirementId.startswith("requestedDesign:")
                and any(
                    row.requirementId == item.requirementId
                    and row.status == "unsupported"
                    for row in current_coverage
                )
            ]
            if missing_questions:
                context_revision = {
                    "reassessContextReport": existing.reportReferences[0].model_dump(
                        mode="json"
                    ),
                    "requiredDesignQuestions": missing_questions,
                }
                prior_context = resolved_report
                logger.info(
                    "Experimental context: reassessing explicit study questions missing from prior evidence"
                )
            else:
                validate_objective_evidence(saved_contract, resolved_report)
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
            if not context_revision:
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
        validate_rna_directions(directions)
        if request_record.request.authorLabelPolicy == "holdout":
            held_out_columns = sorted(
                column
                for column in store.cells.columns
                if is_author_label_column(column)
            )
            existing_exclusions = directions.get("excludeColumns")
            if existing_exclusions is None:
                existing_exclusion_list: list[str] = []
            elif isinstance(existing_exclusions, list) and all(
                isinstance(value, str) for value in existing_exclusions
            ):
                existing_exclusion_list = existing_exclusions
            else:
                raise ValueError("experimentalDirections.excludeColumns must be a list")

            referenced_held_out: set[str] = set()

            def find_held_out_references(value: Any) -> None:
                if isinstance(value, str):
                    if value in held_out_columns:
                        referenced_held_out.add(value)
                    return
                if isinstance(value, Mapping):
                    for nested in value.values():
                        find_held_out_references(nested)
                    return
                if isinstance(value, list | tuple | set):
                    for nested in value:
                        find_held_out_references(nested)

            for key, value in directions.items():
                if key != "excludeColumns":
                    find_held_out_references(value)
            if referenced_held_out:
                raise ValueError(
                    "authorLabelPolicy='holdout' forbids runtime use of author "
                    "annotation columns: " + ", ".join(sorted(referenced_held_out))
                )
            directions["excludeColumns"] = sorted(
                {
                    *held_out_columns,
                    *existing_exclusion_list,
                }
            )
        retry_inputs: dict[str, Any] = {}
        failed = journal._validated_done_outcome(
            store,
            prefix,
            workflow.workflowRunId,
            "experimental_context",
            request_record,
            parents,
            required_status="failed",
        )
        if failed is not None and "retryAfterFailedReport" in failed.inputs:
            # A later persistence or validation error must keep the same retry
            # identity so its already committed decision remains recoverable.
            retry_inputs["retryAfterFailedReport"] = failed.inputs[
                "retryAfterFailedReport"
            ]
        if failed is not None and failed.reportReferences:
            failed_report = journal.load_stage_report(
                store, failed, ExperimentalContextResult
            )
            if cast(ExperimentalContextResult, failed_report).status in {
                "failed",
                "needsInput",
            }:
                # An unresolved unattended answer is also an unsuccessful attempt.
                # Keep it immutable while reusing its measured design evidence.
                # A committed successful retry still has a stable recovery key.
                retry_inputs["retryAfterFailedReport"] = failed.reportReferences[
                    0
                ].model_dump(mode="json")
                if (
                    cast(ExperimentalContextResult, failed_report).status
                    == "needsInput"
                ):
                    prior_context = cast(ExperimentalContextResult, failed_report)
                logger.info("Retrying experimental context after the previous failure")
        started = journal._start_attempt(
            store.zw,
            prefix,
            workflow.workflowRunId,
            "experimental_context",
            request_record,
            parents,
            inputs={
                **retry_inputs,
                **context_revision,
                "studyContext": request_record.request.studyContext,
                "studyObjective": request_record.request.studyObjective,
                "cellSelection": cell_selection.model_dump(mode="json"),
                "directions": directions,
                "metadataFingerprints": metadata_identity,
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
        logger.debug(
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
            no_inference_resolution = (
                paused is not None
                and paused.outputs.get("unsafeBatchCorrection") is not True
                and isinstance(supplied_directions, Mapping)
                and supplied_directions.get("coefficientsOfInterest") == []
                and supplied_directions.get("unitsOfInference") == {}
                and isinstance(supplied_directions.get("batchCorrection"), Mapping)
                and supplied_directions["batchCorrection"].get("action") == "skip"
            )
            actions: list[str] = []
            recovered = journal._recover_persisted_stage_report(
                store,
                started,
                expected_type=ExperimentalContextResult,
            )
            if recovered is not None:
                recovered_report, reference = recovered
                report = cast(ExperimentalContextResult, recovered_report)
                actions.append("recover_persisted_experimental_context_report")
            else:
                if unsafe_resolution == "skip" or no_inference_resolution:
                    assert paused is not None
                    if not paused.reportReferences:
                        raise ValueError(
                            "Experimental Context pause has no persisted report"
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
                    prior_plan = prior_report.decision.batchCorrection
                    if no_inference_resolution:
                        plan_updates: dict[str, Any] = {
                            "preserveColumns": [],
                            "rationale": (
                                "The caller explicitly continued without "
                                "coefficient-level inference and skipped Harmony."
                            ),
                        }
                        decision_updates: dict[str, Any] = {
                            "coefficientsOfInterest": [],
                            "unitsOfInference": {},
                        }
                        resolution_note = (
                            "Caller explicitly continued without coefficient-level "
                            "inference and skipped Harmony."
                        )
                        resolution_action = (
                            "resolve_experimental_context:no_inference_skip_harmony"
                        )
                    else:
                        plan_updates = {
                            "rationale": (
                                "The caller explicitly skipped Harmony after reviewing "
                                "the persisted unsafe batch-correction evidence."
                            )
                        }
                        decision_updates = {}
                        resolution_note = (
                            "Caller explicitly skipped Harmony after an unsafe result."
                        )
                        resolution_action = "resolve_unsafe_batch_correction:skip"
                    skip_plan = prior_plan.model_copy(
                        update={
                            "action": "skip",
                            "batchColumns": [],
                            "metricsRequired": [],
                            **plan_updates,
                        }
                    )
                    decision = prior_report.decision.model_copy(
                        update={
                            **decision_updates,
                            "batchCorrection": skip_plan,
                            "needsInput": [],
                        }
                    )
                    report = prior_report.model_copy(
                        update={
                            "status": "done",
                            "decision": decision,
                            "notes": [*prior_report.notes, resolution_note],
                            "runInfo": AgentRunInfo(
                                agentName="experimental_context_resolution"
                            ),
                        }
                    )
                    actions.append(resolution_action)
                else:
                    logger.debug(
                        f"Workflow {workflow.workflowRunId}: invoking Experimental "
                        "Context"
                    )
                    agent = ExperimentalContextAgent(
                        self.model,
                        config=request_record.config.agentRunConfig,
                    )
                    evidence_inputs = {
                        "request": request_record.model_dump(mode="json"),
                        "parents": [
                            parent.model_dump(mode="json") for parent in parents
                        ],
                        "context": {
                            key: value
                            for key, value in started.inputs.items()
                            if key != "retryAfterFailedReport"
                        },
                    }
                    evidence_key = (
                        "experimental_context/evidence/"
                        + hashlib.sha256(
                            canonical_json_bytes(evidence_inputs)
                        ).hexdigest()
                    )
                    result_key = "result"
                    if "retryAfterFailedReport" in retry_inputs:
                        result_key += (
                            "/"
                            + hashlib.sha256(
                                canonical_json_bytes(
                                    retry_inputs["retryAfterFailedReport"]
                                )
                            ).hexdigest()
                        )

                    def read_evidence(key: str) -> dict[str, Any] | None:
                        return journal.load_checkpoint(
                            store,
                            prefix,
                            workflow.workflowRunId,
                            evidence_key
                            + "/"
                            + (result_key if key == "result" else key),
                            evidence_inputs,
                        )

                    def write_evidence(key: str, output: dict[str, Any]) -> None:
                        journal.save_checkpoint(
                            store,
                            prefix,
                            workflow.workflowRunId,
                            evidence_key
                            + "/"
                            + (result_key if key == "result" else key),
                            evidence_inputs,
                            output,
                        )

                    report = agent.run(
                        store,
                        qc_assay=selected,
                        study_context=request_record.request.studyContext,
                        study_objective=request_record.request.studyObjective,
                        cell_selection=cell_selection_ref,
                        directions=directions,
                        quality_metric_artifacts=quality_metric_artifacts,
                        hto_identity_artifacts=hto_identity_artifacts,
                        checkpoint_read=read_evidence,
                        checkpoint_write=write_evidence,
                        on_attempt=journal.model_attempt_callback(
                            store,
                            prefix,
                            workflow.workflowRunId,
                            evidence_key,
                            evidence_inputs,
                        ),
                        previous_context=prior_context,
                    )
                saved_report, reference = journal._save_stage_report(
                    store,
                    started,
                    report,
                    expected_type=ExperimentalContextResult,
                )
                report = cast(ExperimentalContextResult, saved_report)
            if report.cellSelection != cell_selection:
                raise ValueError(
                    "Experimental Context returned a different cell selection"
                )
            if report.status == "done":
                validate_rna_context(report, selected)
            if report.qualityMetricArtifacts != list(quality_metric_artifacts):
                raise ValueError(
                    "Experimental Context returned different quality metric artifacts"
                )
            if report.htoIdentityArtifacts != list(hto_identity_artifacts):
                raise ValueError(
                    "Experimental Context returned different HTO identity artifacts"
                )
            logger.debug(
                f"Workflow {workflow.workflowRunId}: Experimental Context returned "
                f"status={report.status!r}, batchAction="
                f"{report.decision.batchCorrection.action!r}"
            )
            if report.status == "needsInput":
                if request_record.config.inputPolicy == "unattended":
                    outcome = journal._complete_attempt(
                        started,
                        status="failed",
                        report_references=[reference],
                        artifacts=context_artifacts,
                        error=(
                            "The unattended Experimental Context stage returned an "
                            "unresolved decision: "
                            + "; ".join(report.decision.needsInput or report.notes)
                        ),
                        notes=report.notes,
                    )
                    journal._save_outcome(store.zw, prefix, outcome)
                    return outcome, report
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
            elif (
                report.decision.batchCorrection.action == "unsafe"
                and request_record.config.inputPolicy != "unattended"
            ):
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
                if report.decision.batchCorrection.action == "unsafe":
                    actions.append("evaluate_unsafe_harmony_for_diagnosis")
                physical_capture = directions.get("physicalCaptureColumn")
                if not isinstance(physical_capture, str) or not physical_capture:
                    physical_capture = None
                elif physical_capture not in {
                    *store.cells.columns,
                    *report.htoIdentityColumns,
                }:
                    raise ValueError(
                        "physicalCaptureColumn must identify observed metadata or "
                        "an exact HTO identity"
                    )
                study_contract = build_study_contract(
                    study_context=request_record.request.studyContext,
                    study_objective=request_record.request.studyObjective,
                    experimental_result=report,
                    author_label_policy=(request_record.request.authorLabelPolicy),
                    physical_capture_column=report.decision.physicalCaptureColumn
                    or physical_capture,
                )
                try:
                    validate_objective_evidence(study_contract, report)
                except ValueError as exc:
                    unattended = request_record.config.inputPolicy == "unattended"
                    outcome = journal._complete_attempt(
                        started,
                        status="failed" if unattended else "needsInput",
                        report_references=[reference],
                        artifacts=context_artifacts,
                        outputs={
                            "studyContract": study_contract.model_dump(mode="json")
                        },
                        actions=actions,
                        error=str(exc) if unattended else None,
                        needs_input=None
                        if unattended
                        else WorkflowNeedsInput(
                            questions=[
                                WorkflowQuestion(
                                    questionId="experimentalDirections",
                                    question=str(exc),
                                    evidenceIds=list(study_contract.evidenceIds),
                                )
                            ]
                        ),
                        notes=[*report.notes, str(exc)],
                    )
                    journal._save_outcome(store.zw, prefix, outcome)
                    return outcome, report
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
                        "studyContract": study_contract.model_dump(mode="json"),
                    },
                    actions=actions,
                    notes=report.notes,
                )
            journal._save_outcome(store.zw, prefix, outcome)
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
