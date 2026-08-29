"""Final analysis and biological interpretation workflow stages."""

import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

from ...datastore.datastore import DataStore
from ...storage.refs import ArtifactRef
from ...utils.logging import logger
from ..biological_interpretation import (
    BiologicalContext,
    BiologicalInterpretationAgent,
    BiologicalInterpretationReport,
)
from ..data_enrichment import DataEnrichmentReport
from ..experimental_context import ExperimentalContextResult
from ..parameter_tuning import ParameterTuningAgent, ParameterTuningReport
from ..persistence import (
    AgentInvocation,
    AgentReportReference,
    AgentWorkflowRun,
)
from ..types import ArtifactReferenceModel, ExperimentalBiologyHandoff
from . import journal
from .models import (
    AutomatedPreprocessingPlan,
    FinalAnalysisHandoff,
    NativeAnalysisHandoff,
    OrchestrationRequestRecord,
    OrchestrationResumeRecord,
    PreprocessedAssayHandoff,
    ReductionMethod,
    WorkflowNeedsInput,
    WorkflowQuestion,
    WorkflowStageAttempt,
    WorkflowStageLink,
    artifact_model_to_ref,
)


class FinalizationStagesMixin:
    """Finalize layouts, clusters, markers, and biological interpretation."""

    model: Any

    def analysis_finalization_stage(
        self,
        store: DataStore,
        workflow: AgentWorkflowRun,
        request_record: OrchestrationRequestRecord,
        parents: Sequence[WorkflowStageLink],
        plan: AutomatedPreprocessingPlan,
        preprocessed: Sequence[PreprocessedAssayHandoff],
        tuning_report: ParameterTuningReport,
        tuning_reference: AgentReportReference,
        *,
        resume_record: OrchestrationResumeRecord | None = None,
    ) -> tuple[WorkflowStageAttempt, FinalAnalysisHandoff]:
        prefix = journal._ensure_orchestration_store(store)
        existing = journal._validated_done_outcome(
            store,
            prefix,
            workflow.workflowRunId,
            "analysis_finalization",
            request_record,
            parents,
        )
        if existing is not None:
            logger.info(
                f"Workflow {workflow.workflowRunId}: reusing finalized analysis"
            )
            return existing, FinalAnalysisHandoff.model_validate(
                existing.outputs["finalAnalysis"]
            )
        started = journal._start_attempt(
            store.zw,
            prefix,
            workflow.workflowRunId,
            "analysis_finalization",
            request_record,
            parents,
            inputs={
                "parameterReport": tuning_reference.model_dump(mode="json"),
                "preprocessedAssays": [
                    value.model_dump(mode="json") for value in preprocessed
                ],
                "finalClusterColumn": tuning_report.finalClusterColumn,
            },
            resume_record=resume_record,
        )
        artifacts: dict[str, ArtifactReferenceModel] = {}
        actions: list[str] = []
        operations: list[dict[str, Any]] = []
        metadata_columns: list[str] = []
        logger.info(
            f"Workflow {workflow.workflowRunId}: finalizing "
            f"{len(preprocessed)} native analysis route(s) and markers from "
            f"assay {plan.markerAssay!r}"
        )
        try:
            if (
                tuning_report.status != "done"
                or tuning_report.finalClusterArtifact is None
                or not tuning_report.finalClusterColumn
            ):
                raise ValueError("Parameter Tuning has no finalized cluster branch")
            preprocessed_by_assay = {value.assay: value for value in preprocessed}
            plan_by_assay = {value.assay: value for value in plan.assays}
            agent = ParameterTuningAgent(
                self.model,
                config=request_record.config.agentRunConfig,
            )
            token = workflow.workflowRunId[:12]
            native_handoffs, native_umaps = self.finalize_native_analyses(
                store,
                agent,
                request_record,
                tuning_report,
                preprocessed_by_assay,
                token,
                artifacts,
                actions,
                operations,
                metadata_columns,
            )
            graph_method, final_graph, final_umap, final_umap_columns = (
                self.finalize_selected_graph(
                    store,
                    plan,
                    tuning_report,
                    native_handoffs,
                    native_umaps,
                    token,
                    actions,
                    operations,
                    metadata_columns,
                )
            )
            final_clusters = ArtifactReferenceModel.model_validate(
                tuning_report.finalClusterArtifact.model_dump()
            )
            marker_handoff = preprocessed_by_assay[plan.markerAssay]
            if marker_handoff.markerFeatures is None:
                raise ValueError("Marker assay lacks an exact feature panel")
            marker_plan = plan_by_assay[plan.markerAssay]
            marker_ref = store.run_marker_search(
                from_assay=plan.markerAssay,
                group_key=tuning_report.finalClusterColumn,
                cell_key="I",
                features=artifact_model_to_ref(marker_handoff.markerFeatures),
                skip_save=False,
                invalidate_cache=False,
                log_transform=bool(
                    marker_plan.normalizationParameters.get("logTransform", False)
                ),
                renormalize_subset=bool(
                    marker_plan.normalizationParameters.get("renormalizeSubset", False)
                ),
            )
            if not isinstance(marker_ref, ArtifactRef):
                raise TypeError("Saved marker search did not return an artifact")
            marker_model = ArtifactReferenceModel.from_artifact_ref(marker_ref)
            artifacts.update(
                {
                    "final_graph": final_graph,
                    "final_clusters": final_clusters,
                    "final_umap": final_umap,
                    "marker_features": marker_handoff.markerFeatures,
                    "markers": marker_model,
                }
            )
            metadata_columns.append(tuning_report.finalClusterColumn)
            limitations = list(
                dict.fromkeys([*plan.limitations, *tuning_report.limitations])
            )
            if marker_plan.assayType == "ATAC":
                limitations.append(
                    "ATAC peak markers are descriptive and cannot establish "
                    "confident cell identities alone"
                )
            final_analysis = FinalAnalysisHandoff(
                workflowRunId=workflow.workflowRunId,
                primaryAssay=plan.primaryAssay,
                markerAssay=plan.markerAssay,
                cellKey="I",
                nativeAnalyses=native_handoffs,
                graph=final_graph,
                graphMethod=graph_method,
                clusters=final_clusters,
                clusterColumn=tuning_report.finalClusterColumn,
                umap=final_umap,
                umapColumns=final_umap_columns,
                markerFeatures=marker_handoff.markerFeatures,
                markers=marker_model,
                parameterReport=tuning_reference,
                limitations=list(dict.fromkeys(limitations)),
            )
            actions.append(f"run_markers:{plan.markerAssay}")
            operations.append(
                {
                    "operation": "run_marker_search",
                    "assay": plan.markerAssay,
                    "groupKey": tuning_report.finalClusterColumn,
                    "cellKey": "I",
                    "features": marker_handoff.markerFeatures.model_dump(mode="json"),
                    "skipSave": False,
                    "invalidateCache": False,
                    "logTransform": bool(
                        marker_plan.normalizationParameters.get("logTransform", False)
                    ),
                    "renormalizeSubset": bool(
                        marker_plan.normalizationParameters.get(
                            "renormalizeSubset", False
                        )
                    ),
                    "artifact": marker_model.model_dump(mode="json"),
                }
            )
            outcome = journal._complete_attempt(
                started,
                status="done",
                artifacts=artifacts,
                outputs={
                    "finalAnalysis": final_analysis.model_dump(mode="json"),
                    "metadataColumns": list(dict.fromkeys(metadata_columns)),
                    "operations": operations,
                },
                actions=actions,
                notes=final_analysis.limitations,
            )
            journal._save_outcome(store.zw, prefix, outcome)
            logger.info(
                f"Workflow {workflow.workflowRunId}: finalized "
                f"graphMethod={graph_method!r}, nativeLayouts="
                f"{len(native_handoffs)}, markerAssay={plan.markerAssay!r}"
            )
            return outcome, final_analysis
        except Exception as exc:
            outcome = journal.finish_exception(
                store,
                prefix,
                workflow,
                started,
                exc,
                artifacts=artifacts,
                actions=actions,
                outputs={
                    "metadataColumns": list(dict.fromkeys(metadata_columns)),
                    "operations": operations,
                },
            )
            return outcome, FinalAnalysisHandoff.get_blank()

    def finalize_native_analyses(
        self,
        store: DataStore,
        agent: ParameterTuningAgent,
        request_record: OrchestrationRequestRecord,
        tuning_report: ParameterTuningReport,
        preprocessed_by_assay: Mapping[str, PreprocessedAssayHandoff],
        token: str,
        artifacts: dict[str, ArtifactReferenceModel],
        actions: list[str],
        operations: list[dict[str, Any]],
        metadata_columns: list[str],
    ) -> tuple[
        list[NativeAnalysisHandoff],
        dict[str, tuple[ArtifactReferenceModel, list[str]]],
    ]:
        native_handoffs: list[NativeAnalysisHandoff] = []
        native_umaps: dict[str, tuple[ArtifactReferenceModel, list[str]]] = {}
        for assay, assay_report in tuning_report.assayReports.items():
            logger.info(
                f"Finalizing native analysis for assay {assay!r} with candidate "
                f"{assay_report.recommendedCandidateId!r}"
            )
            preprocessed_assay = preprocessed_by_assay[assay]
            normalized = preprocessed_assay.normalized
            if normalized is None:
                raise ValueError(f"Assay {assay!r} lacks normalization")
            promoted = agent.promote(
                store,
                report=assay_report,
                normalized=artifact_model_to_ref(normalized),
                identity_feature_limit=request_record.config.maxIdentityFeatures,
            )
            reduction_key = promoted.parameters.reductionMethod
            reduction_record = promoted.artifacts[reduction_key]
            reduction_ref = artifact_model_to_ref(
                ArtifactReferenceModel.model_validate(reduction_record.model_dump())
            )
            initialization_ref = store.build_embedding_initialization(
                reduction_ref,
                from_assay=assay,
                n_centroids=min(1000, preprocessed_assay.nCells),
                rand_state=4466,
                update_state=True,
                invalidate_cache=False,
            )
            graph_record = promoted.artifacts["connectivityMap"]
            graph_ref = artifact_model_to_ref(
                ArtifactReferenceModel.model_validate(graph_record.model_dump())
            )
            umap_label = (
                f"agent_{token}_{journal._safe_label(assay).lower()}_native_umap"
            )
            umap_ref = store.run_umap(
                graph_ref,
                from_assay=assay,
                cell_key="I",
                label=umap_label,
                parallel=False,
                random_seed=4444,
                invalidate_cache=False,
            )
            umap_columns = [
                f"{assay}_{umap_label}1",
                f"{assay}_{umap_label}2",
            ]
            metadata_columns.extend([*umap_columns, cast(str, promoted.clusterColumn)])
            promoted_artifacts = {
                name: ArtifactReferenceModel.model_validate(value.model_dump())
                for name, value in promoted.artifacts.items()
            }
            umap_model = ArtifactReferenceModel.from_artifact_ref(umap_ref)
            initialization_model = ArtifactReferenceModel.from_artifact_ref(
                initialization_ref
            )
            native_umaps[assay] = (umap_model, umap_columns)
            operations.extend(
                [
                    {
                        "operation": "promote_parameter_candidate",
                        "assay": assay,
                        "candidate": promoted.parameters.model_dump(mode="json"),
                        "normalized": normalized.model_dump(mode="json"),
                        "identityFeatureLimit": (
                            request_record.config.maxIdentityFeatures
                        ),
                        "updateState": True,
                        "artifacts": {
                            name: value.model_dump(mode="json")
                            for name, value in promoted_artifacts.items()
                        },
                    },
                    {
                        "operation": "build_embedding_initialization",
                        "assay": assay,
                        "reduction": ArtifactReferenceModel.model_validate(
                            reduction_record.model_dump()
                        ).model_dump(mode="json"),
                        "nCentroids": min(1000, preprocessed_assay.nCells),
                        "randomSeed": 4466,
                        "updateState": True,
                        "invalidateCache": False,
                        "artifact": initialization_model.model_dump(mode="json"),
                    },
                    {
                        "operation": "run_umap",
                        "assay": assay,
                        "graph": promoted_artifacts["connectivityMap"].model_dump(
                            mode="json"
                        ),
                        "cellKey": "I",
                        "label": umap_label,
                        "parallel": False,
                        "randomSeed": 4444,
                        "invalidateCache": False,
                        "artifact": umap_model.model_dump(mode="json"),
                    },
                ]
            )
            native_handoffs.append(
                NativeAnalysisHandoff(
                    assay=assay,
                    reductionMethod=cast(
                        ReductionMethod, promoted.parameters.reductionMethod
                    ),
                    featureSelection=preprocessed_assay.graphFeatures,
                    markerFeatures=preprocessed_assay.markerFeatures,
                    normalized=normalized,
                    reduction=promoted_artifacts[reduction_key],
                    batchCorrection=promoted_artifacts.get("harmony"),
                    annIndex=promoted_artifacts["annIndex"],
                    embeddingInitialization=initialization_model,
                    neighbors=promoted_artifacts["neighbors"],
                    graph=promoted_artifacts["connectivityMap"],
                    clusters=promoted_artifacts["clusters"],
                    clusterColumn=cast(str, promoted.clusterColumn),
                    umap=umap_model,
                    umapColumns=umap_columns,
                )
            )
            artifacts.update(
                {
                    f"{assay}_{name}": value
                    for name, value in {
                        **promoted_artifacts,
                        "embeddingInitialization": initialization_model,
                        "umap": umap_model,
                    }.items()
                }
            )
            actions.extend([f"promote_native:{assay}", f"run_native_umap:{assay}"])
            logger.info(f"Finalized native UMAP and clusters for assay {assay!r}")
        return native_handoffs, native_umaps

    def finalize_selected_graph(
        self,
        store: DataStore,
        plan: AutomatedPreprocessingPlan,
        tuning_report: ParameterTuningReport,
        native_handoffs: Sequence[NativeAnalysisHandoff],
        native_umaps: Mapping[str, tuple[ArtifactReferenceModel, list[str]]],
        token: str,
        actions: list[str],
        operations: list[dict[str, Any]],
        metadata_columns: list[str],
    ) -> tuple[
        Literal["native", "snn", "wnn"],
        ArtifactReferenceModel,
        ArtifactReferenceModel,
        list[str],
    ]:
        if tuning_report.recommendedIntegrationId is not None:
            selected_integration = next(
                value
                for value in tuning_report.integrationEvaluations
                if value.integrationId == tuning_report.recommendedIntegrationId
            )
            if selected_integration.graphArtifact is None:
                raise ValueError("Selected integration lacks its graph artifact")
            final_graph = ArtifactReferenceModel.model_validate(
                selected_integration.graphArtifact.model_dump()
            )
            graph_method = selected_integration.method
            logger.info(
                f"Finalizing selected {graph_method.upper()} graph "
                f"{selected_integration.integrationId!r}"
            )
            graph_ref = artifact_model_to_ref(final_graph)
            graph_label = f"agent_{token}_{graph_method}"
            umap_label = f"agent_{token}_{graph_method}_final_umap"
            umap_ref = store.run_umap(
                graph_ref,
                from_assay=plan.primaryAssay,
                cell_key="I",
                label=umap_label,
                parallel=False,
                random_seed=4444,
                invalidate_cache=False,
            )
            final_umap = ArtifactReferenceModel.from_artifact_ref(umap_ref)
            final_umap_columns = [
                f"{graph_label}_{umap_label}1",
                f"{graph_label}_{umap_label}2",
            ]
            metadata_columns.extend(final_umap_columns)
            actions.append(f"run_final_umap:{graph_method}")
            operations.append(
                {
                    "operation": "run_umap",
                    "graphMethod": graph_method,
                    "graph": final_graph.model_dump(mode="json"),
                    "fromAssay": plan.primaryAssay,
                    "cellKey": "I",
                    "label": umap_label,
                    "parallel": False,
                    "randomSeed": 4444,
                    "invalidateCache": False,
                    "artifact": final_umap.model_dump(mode="json"),
                }
            )
            return graph_method, final_graph, final_umap, final_umap_columns
        graph_assay = tuning_report.graphAssay
        if graph_assay is None:
            raise ValueError("Native final selection lacks graphAssay")
        native = next(value for value in native_handoffs if value.assay == graph_assay)
        if native.graph is None:
            raise ValueError("Native final selection lacks graph artifact")
        final_umap, final_umap_columns = native_umaps[graph_assay]
        logger.info(
            f"Reusing native graph and UMAP from assay {graph_assay!r} as final"
        )
        return "native", native.graph, final_umap, final_umap_columns

    def biological_interpretation_stage(
        self,
        store: DataStore,
        workflow: AgentWorkflowRun,
        request_record: OrchestrationRequestRecord,
        parents: Sequence[WorkflowStageLink],
        enrichment: DataEnrichmentReport,
        experimental: ExperimentalContextResult,
        tuning_report: ParameterTuningReport,
        final_analysis: FinalAnalysisHandoff,
        enrichment_reference: AgentReportReference,
        experimental_reference: AgentReportReference,
        tuning_reference: AgentReportReference,
        answers: Mapping[str, Any],
        *,
        resume_record: OrchestrationResumeRecord | None = None,
    ) -> WorkflowStageAttempt:
        prefix = journal._ensure_orchestration_store(store)
        existing = journal._validated_done_outcome(
            store,
            prefix,
            workflow.workflowRunId,
            "biological_interpretation",
            request_record,
            parents,
        )
        if existing is not None:
            logger.info(
                f"Workflow {workflow.workflowRunId}: reusing Biological "
                "Interpretation report"
            )
            journal.load_stage_report(store, existing, BiologicalInterpretationReport)
            return existing
        requested_coefficient = answers.get("primaryCoefficient")
        if not isinstance(requested_coefficient, str) or not requested_coefficient:
            directed = request_record.request.experimentalDirections.get(
                "primaryCoefficient"
            )
            requested_coefficient = directed if isinstance(directed, str) else None
        coefficients = list(experimental.decision.coefficientsOfInterest)
        logger.info(
            f"Workflow {workflow.workflowRunId}: Biological Interpretation has "
            f"{len(coefficients)} validated coefficient option(s)"
        )
        started = journal._start_attempt(
            store.zw,
            prefix,
            workflow.workflowRunId,
            "biological_interpretation",
            request_record,
            parents,
            inputs={
                "studyContext": request_record.request.studyContext,
                "studyContextSummary": enrichment.studyContextSummary.model_dump(
                    mode="json"
                ),
                "experimentalContextReport": experimental_reference.model_dump(
                    mode="json"
                ),
                "finalAnalysis": final_analysis.model_dump(mode="json"),
                "primaryCoefficient": requested_coefficient,
                "biologicalInterpretation": answers.get("biologicalInterpretation"),
            },
            resume_record=resume_record,
        )
        if requested_coefficient is None and len(coefficients) > 1:
            logger.info(
                f"Workflow {workflow.workflowRunId}: Biological Interpretation "
                "requires a primary coefficient selection"
            )
            outcome = journal._complete_attempt(
                started,
                status="needsInput",
                needs_input=WorkflowNeedsInput(
                    questions=[
                        WorkflowQuestion(
                            questionId="primaryCoefficient",
                            question=(
                                "Which validated coefficient should constrain "
                                "treatment observations?"
                            ),
                            options=coefficients,
                            evidenceIds=list(experimental.decision.evidenceIds),
                        )
                    ]
                ),
                notes=[
                    "Cluster identities can be interpreted after one treatment "
                    "coefficient is selected"
                ],
            )
            journal._save_outcome(store.zw, prefix, outcome)
            return outcome
        try:
            experimental_handoff: ExperimentalBiologyHandoff | None = None
            if coefficients:
                experimental_handoff = experimental.to_biological_handoff(
                    requested_coefficient
                )
            tuning_handoff = tuning_report.to_biological_handoff()
            marker_policy = next(
                (
                    value
                    for value in enrichment.policies
                    if value.assay == final_analysis.markerAssay
                ),
                None,
            )
            summary = enrichment.studyContextSummary
            biological_context = BiologicalContext(
                organism=(
                    marker_policy.organismName
                    if marker_policy is not None
                    and marker_policy.organismName != "unknown"
                    else ""
                ),
                studyContext=request_record.request.studyContext,
                tissue=", ".join(summary.tissueReferences),
                cellTypeReferences=list(summary.cellTypeReferences),
                experimentalDetails=[
                    *summary.experimentalReferences,
                    *final_analysis.limitations,
                ],
                treatmentQuestion=str(
                    request_record.request.experimentalDirections.get(
                        "treatmentQuestion", ""
                    )
                ),
            )
            biological_answer = answers.get("biologicalInterpretation")
            if isinstance(biological_answer, str) and biological_answer.strip():
                biological_context = biological_context.model_copy(
                    update={
                        "experimentalDetails": [
                            *biological_context.experimentalDetails,
                            biological_answer.strip(),
                        ]
                    }
                )
            elif isinstance(biological_answer, Mapping):
                biological_context = biological_context.model_copy(
                    update={
                        "experimentalDetails": [
                            *biological_context.experimentalDetails,
                            json.dumps(dict(biological_answer), sort_keys=True),
                        ]
                    }
                )
            if (
                final_analysis.clusters is None
                or final_analysis.markers is None
                or final_analysis.markerFeatures is None
            ):
                raise ValueError("Final analysis lacks clusters or marker artifacts")
            recovered = journal._recover_persisted_stage_report(
                store,
                started,
                agent_name="biological_interpretation",
                expected_type=BiologicalInterpretationReport,
            )
            if recovered is not None:
                recovered_report, reference = recovered
                report = cast(BiologicalInterpretationReport, recovered_report)
                recovery_actions = [
                    "recover_persisted_biological_interpretation_report"
                ]
            else:
                recovery_actions = []
                logger.info(
                    f"Workflow {workflow.workflowRunId}: invoking Biological "
                    "Interpretation"
                )
                agent = BiologicalInterpretationAgent(
                    self.model,
                    config=request_record.config.agentRunConfig,
                )
                report = agent.run(
                    store,
                    cluster_column=final_analysis.clusterColumn,
                    cluster=artifact_model_to_ref(final_analysis.clusters),
                    biological_context=biological_context,
                    cell_key=final_analysis.cellKey,
                    from_assay=final_analysis.markerAssay,
                    graph_assay=tuning_handoff.graphAssay,
                    marker_assay_type=(
                        marker_policy.assayModality
                        if marker_policy is not None
                        else None
                    ),
                    tuning_handoff=tuning_handoff,
                    experimental_handoff=experimental_handoff,
                    marker=artifact_model_to_ref(final_analysis.markers),
                    marker_features=artifact_model_to_ref(
                        final_analysis.markerFeatures
                    ),
                    allow_marker_search=False,
                )
                if marker_policy is not None and marker_policy.assayModality == "ATAC":
                    atac_limitation = (
                        "ATAC peak markers are descriptive, so all cell identities "
                        "remain low-confidence hypotheses."
                    )
                    report = report.model_copy(
                        update={
                            "clusterInterpretations": [
                                value.model_copy(
                                    update={
                                        "identityIsHypothesis": True,
                                        "confidence": "low",
                                    }
                                )
                                for value in report.clusterInterpretations
                            ],
                            "limitations": list(
                                dict.fromkeys([*report.limitations, atac_limitation])
                            ),
                        }
                    )
                parent_reports = [
                    journal._report_link(enrichment_reference),
                    journal._report_link(experimental_reference),
                    journal._report_link(tuning_reference),
                ]
                saved_report, reference = journal._save_stage_report(
                    store,
                    started,
                    report,
                    invocation=AgentInvocation(
                        agentName="biological_interpretation",
                        parentReports=parent_reports,
                        inputs={
                            "biologicalContext": biological_context.model_dump(
                                mode="json"
                            ),
                            "clusterColumn": final_analysis.clusterColumn,
                            "cellKey": final_analysis.cellKey,
                            "fromAssay": final_analysis.markerAssay,
                            "graphAssay": tuning_handoff.graphAssay,
                            "markerAssayType": (
                                marker_policy.assayModality
                                if marker_policy is not None
                                else None
                            ),
                            "allowMarkerSearch": False,
                            "studyContextSummary": summary.model_dump(mode="json"),
                            "experimentalContextReport": (
                                experimental_reference.model_dump(mode="json")
                            ),
                        },
                        artifacts={
                            "clusters": final_analysis.clusters,
                            "markers": final_analysis.markers,
                            "markerFeatures": final_analysis.markerFeatures,
                        },
                        runConfig=agent.config,
                        experimentalBiologyHandoff=experimental_handoff,
                        tuningBiologyHandoff=tuning_handoff,
                    ),
                    expected_type=BiologicalInterpretationReport,
                )
                report = cast(BiologicalInterpretationReport, saved_report)
            logger.info(
                f"Workflow {workflow.workflowRunId}: Biological Interpretation "
                f"returned status={report.status!r}, clusters="
                f"{len(report.clusterInterpretations)}, treatments="
                f"{len(report.treatmentObservations)}"
            )
            if report.status == "needsInput":
                needs_input = report.needsInput
                assert needs_input is not None
                outcome = journal._complete_attempt(
                    started,
                    status="needsInput",
                    report_references=[reference],
                    artifacts={
                        "clusters": final_analysis.clusters,
                        "markers": final_analysis.markers,
                    },
                    actions=recovery_actions,
                    needs_input=WorkflowNeedsInput(
                        questions=[
                            WorkflowQuestion(
                                questionId="biologicalInterpretation",
                                question=needs_input.question,
                                options=list(needs_input.requiredInputs),
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
                    report_references=[reference],
                    actions=recovery_actions,
                    error="; ".join(report.limitations)
                    or "Biological Interpretation failed",
                )
            else:
                outcome = journal._complete_attempt(
                    started,
                    status="done",
                    report_references=[reference],
                    artifacts={
                        "clusters": final_analysis.clusters,
                        "markers": final_analysis.markers,
                    },
                    actions=recovery_actions,
                    outputs={
                        "clusterCount": len(report.clusterInterpretations),
                        "treatmentObservationCount": len(report.treatmentObservations),
                    },
                    notes=report.limitations,
                )
            journal._save_outcome(store.zw, prefix, outcome)
            logger.info(
                f"Workflow {workflow.workflowRunId}: Biological Interpretation "
                f"outcome status={outcome.status!r}"
            )
            if outcome.status == "failed":
                journal.finalize_failed(
                    store, workflow, outcome.error or "biology failed"
                )
            return outcome
        except Exception as exc:
            return journal.finish_exception(store, prefix, workflow, started, exc)
