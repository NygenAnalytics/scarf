"""Preprocessing planning and execution stages."""

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np

from ...datastore.datastore import DataStore
from ...datastore.summary import AssaySummary
from ...storage.refs import ArtifactRef
from ...utils.logging import logger
from .. import record_io
from ..data_enrichment import (
    AssayFeatureInspection,
    DataEnrichmentReport,
    FeatureSelectionPolicy,
)
from ..experimental_context import ExperimentalContextResult
from ..persistence import AgentWorkflowRun
from ..types import ArtifactReferenceModel
from . import journal
from .models import (
    AssayPreprocessingPlan,
    AutomatedPreprocessingPlan,
    OrchestrationRequestRecord,
    OrchestrationResumeRecord,
    PreprocessedAssayHandoff,
    ReductionMethod,
    WorkflowNeedsInput,
    WorkflowQuestion,
    WorkflowStageAttempt,
    WorkflowStageLink,
)


class PreprocessingStagesMixin:
    """Stages that plan and execute modality-specific preprocessing."""

    def preprocessing_plan_stage(
        self,
        store: DataStore,
        workflow: AgentWorkflowRun,
        request_record: OrchestrationRequestRecord,
        parents: Sequence[WorkflowStageLink],
        enrichment: DataEnrichmentReport,
        experimental: ExperimentalContextResult,
        ingest_outcome: WorkflowStageAttempt,
        answers: Mapping[str, Any],
        *,
        resume_record: OrchestrationResumeRecord | None = None,
    ) -> tuple[WorkflowStageAttempt, AutomatedPreprocessingPlan]:
        prefix = journal._ensure_orchestration_store(store)
        existing = journal._validated_done_outcome(
            store,
            prefix,
            workflow.workflowRunId,
            "preprocessing_plan",
            request_record,
            parents,
        )
        if existing is not None:
            logger.info(
                f"Workflow {workflow.workflowRunId}: reusing preprocessing plan"
            )
            return existing, AutomatedPreprocessingPlan.model_validate(
                existing.outputs["preprocessingPlan"]
            )
        started = journal._start_attempt(
            store.zw,
            prefix,
            workflow.workflowRunId,
            "preprocessing_plan",
            request_record,
            parents,
            inputs={
                "approvalAnswer": answers.get("approvePlanChecksum"),
                "allowAssumptions": request_record.request.allowAssumptions,
            },
            resume_record=resume_record,
        )
        try:
            plan = self.build_preprocessing_plan(
                store,
                request_record,
                enrichment,
                experimental,
                ingest_outcome,
            )
            route_summary = ", ".join(
                f"{value.assay}:{value.featureMethod}/{value.reductionMethod}"
                for value in plan.assays
            )
            logger.info(
                f"Workflow {workflow.workflowRunId}: preprocessing plan built "
                f"(primary={plan.primaryAssay!r}, marker={plan.markerAssay!r}, "
                f"routes=[{route_summary}])"
            )
        except Exception as exc:
            outcome = journal.finish_exception(store, prefix, workflow, started, exc)
            return outcome, AutomatedPreprocessingPlan.get_blank()
        supplied_approval = answers.get("approvePlanChecksum")
        preapproved = bool(
            request_record.request.experimentalDirections.get(
                "approveAutomatedAnalysis", False
            )
        )
        approved = (
            request_record.request.allowAssumptions
            or preapproved
            or supplied_approval == plan.planChecksum
        )
        if not approved:
            logger.info(
                f"Workflow {workflow.workflowRunId}: preprocessing plan requires "
                "caller approval"
            )
            outcome = journal._complete_attempt(
                started,
                status="needsInput",
                outputs={"preprocessingPlan": plan.model_dump(mode="json")},
                needs_input=WorkflowNeedsInput(
                    questions=[
                        WorkflowQuestion(
                            questionId="approvePlanChecksum",
                            question=(
                                "Approve the persisted preprocessing and bounded "
                                "parameter-tuning plan?"
                            ),
                            planChecksum=plan.planChecksum,
                        )
                    ]
                ),
            )
        else:
            logger.info(
                f"Workflow {workflow.workflowRunId}: preprocessing plan approved"
            )
            outcome = journal._complete_attempt(
                started,
                status="done",
                outputs={"preprocessingPlan": plan.model_dump(mode="json")},
                actions=["approve_preprocessing_plan"],
            )
        journal._save_outcome(store.zw, prefix, outcome)
        return outcome, plan

    def build_preprocessing_plan(
        self,
        store: DataStore,
        request_record: OrchestrationRequestRecord,
        enrichment: DataEnrichmentReport,
        experimental: ExperimentalContextResult,
        ingest_outcome: WorkflowStageAttempt,
    ) -> AutomatedPreprocessingPlan:
        request = request_record.request
        store_summary = store.summary()
        summaries = {value.name: value for value in store_summary.assays}
        policies = {value.assay: value for value in enrichment.policies}
        inspections = {value.assay: value for value in enrichment.inspections}
        selected_names = request.analysisAssays or list(store.assay_names)
        assay_plans: list[AssayPreprocessingPlan] = []
        graph_assays: list[str] = []
        limitations: list[str] = list(enrichment.limitations)
        selected_qc_profile = next(
            (
                value
                for value in experimental.qcProfiles
                if value.profileId == experimental.cellQc.profileId
            ),
            None,
        )
        projected_cells = (
            selected_qc_profile.retainedCells
            if selected_qc_profile is not None and selected_qc_profile.retainedCells > 0
            else store_summary.active_cells
        )
        effective_min_cells = min(20, max(1, projected_cells // 10))
        for assay_name in selected_names:
            summary = summaries[assay_name]
            policy = policies.get(assay_name)
            modality = (
                policy.assayModality
                if policy is not None
                else (
                    summary.assay_type
                    if summary.assay_type in {"RNA", "ATAC", "ADT", "HTO"}
                    else "unsupported"
                )
            )
            plan = self.build_assay_preprocessing_plan(
                store,
                request_record,
                assay_name,
                summary,
                policy,
                inspections.get(assay_name),
                modality,
                effective_min_cells,
            )
            if plan.graphEligible:
                graph_assays.append(assay_name)
            if modality == "unsupported":
                limitations.extend(plan.limitations)
            assay_plans.append(plan)
        if not graph_assays:
            raise ValueError("No supported graph-bearing assay remains")
        if len(graph_assays) > request_record.config.maxGraphAssays:
            raise ValueError(
                "Too many graph-bearing assays; provide analysisAssays to select at "
                f"most {request_record.config.maxGraphAssays}"
            )
        modality_counts: dict[str, int] = {}
        for name in graph_assays:
            modality_counts[summaries[name].assay_type] = (
                modality_counts.get(summaries[name].assay_type, 0) + 1
            )
        duplicate_modalities = sorted(
            name for name, count in modality_counts.items() if count > 1
        )
        if duplicate_modalities and not request.analysisAssays:
            raise ValueError(
                "Multiple same-kind biological assays require explicit "
                f"analysisAssays selection: {duplicate_modalities}"
            )
        primary = request.primaryAssay
        if primary is not None and primary not in graph_assays:
            raise ValueError("primaryAssay must name a graph-bearing selected assay")
        if primary is None:
            primary = next(
                (
                    name
                    for modality in ("RNA", "ADT", "ATAC")
                    for name in graph_assays
                    if summaries[name].assay_type == modality
                ),
                graph_assays[0],
            )
        if request.markerAssay is not None:
            if request.markerAssay not in graph_assays:
                raise ValueError("markerAssay must name a graph-bearing selected assay")
            marker_assay = request.markerAssay
        else:
            marker_assay = next(
                (
                    name
                    for modality in ("RNA", "ADT", "ATAC")
                    for name in graph_assays
                    if summaries[name].assay_type == modality
                ),
                primary,
            )
        ingest_format = ingest_outcome.outputs.get("format")
        if request.pairedAssays:
            paired = list(request.pairedAssays)
            unknown_paired = sorted(set(paired) - set(graph_assays))
            if unknown_paired:
                raise ValueError(
                    f"pairedAssays contains non-graph assays: {unknown_paired}"
                )
            if primary not in paired:
                raise ValueError("pairedAssays must include the primary assay")
        elif (
            len(graph_assays) > 1
            and ingest_outcome.outputs.get("pairingProvenance")
            == "singleSourceSharedCellAxis"
        ):
            paired = list(graph_assays)
        else:
            paired = []
            if len(graph_assays) > 1:
                limitations.append(
                    "Multimodal integration skipped because pairing provenance was not supplied"
                )
        reset_selection = ingest_format != "zarr" or bool(request.resetCellSelection)
        final_plan = AutomatedPreprocessingPlan(
            primaryAssay=primary,
            markerAssay=marker_assay,
            cellQc=experimental.cellQc,
            assays=assay_plans,
            pairedAssays=paired,
            resetCellSelection=reset_selection,
            limitations=list(dict.fromkeys(limitations)),
        )
        checksum = hashlib.sha256(
            record_io.canonical_json_bytes(
                final_plan.model_dump(mode="json", exclude={"planChecksum"})
            )
        ).hexdigest()
        return final_plan.model_copy(update={"planChecksum": checksum})

    def build_assay_preprocessing_plan(
        self,
        store: DataStore,
        request_record: OrchestrationRequestRecord,
        assay_name: str,
        summary: AssaySummary,
        policy: FeatureSelectionPolicy | None,
        inspection: AssayFeatureInspection | None,
        modality: str,
        effective_min_cells: int,
    ) -> AssayPreprocessingPlan:
        excluded: list[str] = []
        evidence_ids: list[str] = []
        if policy is not None:
            excluded = list(
                dict.fromkeys(
                    [
                        *policy.excludeFeatures,
                        *policy.artificialFeatures,
                        *(
                            reference.featureId
                            for reference in policy.exactControlFeatures
                        ),
                        *(
                            reference.featureName
                            for reference in policy.exactControlFeatures
                        ),
                    ]
                )
            )
            evidence_ids = list(policy.evidenceIds)
        if modality == "RNA":
            graph_eligible = summary.total_features >= 3
            return AssayPreprocessingPlan(
                assay=assay_name,
                assayType=summary.assay_type,
                role="graph" if graph_eligible else "unsupported",
                graphEligible=graph_eligible,
                markerEligible=graph_eligible,
                featureMethod="hvg" if graph_eligible else "none",
                reductionMethod="pca" if graph_eligible else "none",
                featureParameters={
                    "topN": min(1000, summary.total_features),
                    "minCells": effective_min_cells,
                    "excludeFamilies": (
                        list(policy.excludeFamilies) if policy is not None else []
                    ),
                },
                normalizationParameters={
                    "logTransform": True,
                    "renormalizeSubset": True,
                },
                reductionParameters={"dimensions": 21},
                exactExcludedFeatures=excluded,
                evidenceIds=evidence_ids,
                limitations=(
                    []
                    if graph_eligible
                    else ["RNA requires at least three features for PCA"]
                ),
            )
        if modality == "ATAC":
            graph_eligible = summary.total_features >= 3
            return AssayPreprocessingPlan(
                assay=assay_name,
                assayType=summary.assay_type,
                role="graph" if graph_eligible else "unsupported",
                graphEligible=graph_eligible,
                markerEligible=graph_eligible,
                featureMethod="prevalentPeaks" if graph_eligible else "none",
                reductionMethod="lsi" if graph_eligible else "none",
                featureParameters={"topN": min(25000, summary.total_features)},
                normalizationParameters={
                    "logTransform": False,
                    "renormalizeSubset": False,
                },
                reductionParameters={"dimensions": 50, "skipFirst": True},
                evidenceIds=evidence_ids,
                limitations=list(
                    dict.fromkeys(
                        [
                            *(
                                []
                                if graph_eligible
                                else [
                                    "ATAC requires at least three peak features for LSI"
                                ]
                            ),
                            *(
                                [
                                    "ATAC feature coordinates are not uniformly "
                                    "valid chrom:start-end intervals; the genome "
                                    "build remains unknown"
                                ]
                                if policy is not None
                                and policy.peakCoordinateStatus
                                in {"partial", "invalid"}
                                else []
                            ),
                        ]
                    )
                ),
            )
        if modality == "ADT":
            assay = store.get_assay(assay_name)
            feature_ids = np.asarray(assay.feats.fetch_all("ids")).astype(str)
            feature_names = np.asarray(assay.feats.fetch_all("names")).astype(str)
            excluded = (
                list(
                    dict.fromkeys(
                        value
                        for reference in policy.exactControlFeatures
                        for value in (reference.featureId, reference.featureName)
                        if value
                    )
                )
                if policy is not None
                else []
            )
            excluded_values = {value for value in excluded if value}
            panel_mask = ~np.isin(feature_ids, list(excluded_values))
            panel_mask &= ~np.isin(feature_names, list(excluded_values))
            selected_count = int(panel_mask.sum())
            graph_eligible = selected_count >= 2
            reduction = (
                "identity"
                if graph_eligible
                and selected_count <= request_record.config.maxIdentityFeatures
                else ("pca" if graph_eligible else "none")
            )
            return AssayPreprocessingPlan(
                assay=assay_name,
                assayType=summary.assay_type,
                role="graph" if graph_eligible else "unsupported",
                graphEligible=graph_eligible,
                markerEligible=graph_eligible,
                featureMethod="panel" if graph_eligible else "none",
                reductionMethod=cast(ReductionMethod, reduction),
                normalizationParameters={
                    "logTransform": False,
                    "renormalizeSubset": False,
                },
                reductionParameters={
                    "dimensions": (
                        selected_count
                        if reduction == "identity"
                        else min(15, max(2, selected_count - 1))
                    )
                },
                exactExcludedFeatures=excluded,
                evidenceIds=evidence_ids,
                limitations=(
                    [
                        "ADT control inventory was truncated; only exact observed "
                        "control features were excluded"
                    ]
                    if inspection is not None and inspection.modalityEvidence.truncated
                    else []
                ),
            )
        if modality == "HTO":
            return AssayPreprocessingPlan(
                assay=assay_name,
                assayType=summary.assay_type,
                role="hto",
                graphEligible=False,
                markerEligible=False,
                featureMethod="none",
                reductionMethod="none",
                evidenceIds=evidence_ids,
            )
        message = f"Unsupported assay {assay_name!r} ({summary.assay_type})"
        return AssayPreprocessingPlan(
            assay=assay_name,
            assayType=summary.assay_type,
            role="unsupported",
            limitations=[message],
        )

    def preprocessing_stage(
        self,
        store: DataStore,
        workflow: AgentWorkflowRun,
        request_record: OrchestrationRequestRecord,
        parents: Sequence[WorkflowStageLink],
        plan: AutomatedPreprocessingPlan,
        experimental: ExperimentalContextResult,
        *,
        resume_record: OrchestrationResumeRecord | None = None,
    ) -> tuple[WorkflowStageAttempt, list[PreprocessedAssayHandoff]]:
        prefix = journal._ensure_orchestration_store(store)
        existing = journal._validated_done_outcome(
            store,
            prefix,
            workflow.workflowRunId,
            "preprocessing",
            request_record,
            parents,
        )
        if existing is not None:
            logger.info(
                f"Workflow {workflow.workflowRunId}: reusing preprocessing artifacts"
            )
            return existing, [
                PreprocessedAssayHandoff.model_validate(value)
                for value in existing.outputs["assays"]
            ]
        started = journal._start_attempt(
            store.zw,
            prefix,
            workflow.workflowRunId,
            "preprocessing",
            request_record,
            parents,
            inputs={"preprocessingPlan": plan.model_dump(mode="json")},
            resume_record=resume_record,
        )
        actions: list[str] = []
        operations: list[dict[str, Any]] = []
        artifacts: dict[str, ArtifactReferenceModel] = {}
        try:
            self.apply_cell_qc(store, experimental, actions, operations)
            active_cells = int(np.asarray(store.cells.fetch_all("I"), dtype=bool).sum())
            logger.info(
                f"Workflow {workflow.workflowRunId}: preprocessing retained "
                f"{active_cells} active cell(s)"
            )
            if active_cells < 3:
                raise ValueError("Preprocessing requires at least three active cells")
            handoffs: list[PreprocessedAssayHandoff] = []
            token = workflow.workflowRunId[:12]
            for assay_plan in plan.assays:
                if not assay_plan.graphEligible:
                    continue
                logger.info(
                    f"Workflow {workflow.workflowRunId}: preprocessing assay "
                    f"{assay_plan.assay!r} via {assay_plan.featureMethod}/"
                    f"{assay_plan.reductionMethod}"
                )
                handoffs.append(
                    self.preprocess_assay(
                        store,
                        assay_plan,
                        active_cells=active_cells,
                        token=token,
                        actions=actions,
                        operations=operations,
                        artifacts=artifacts,
                    )
                )
            outcome = journal._complete_attempt(
                started,
                status="done",
                artifacts={
                    name: value
                    for name, value in artifacts.items()
                    if value is not None
                },
                outputs={
                    "assays": [value.model_dump(mode="json") for value in handoffs],
                    "operations": operations,
                },
                actions=actions,
            )
            journal._save_outcome(store.zw, prefix, outcome)
            logger.info(
                f"Workflow {workflow.workflowRunId}: preprocessing produced "
                f"{len(handoffs)} graph-ready assay handoff(s)"
            )
            return outcome, handoffs
        except Exception as exc:
            outcome = journal.finish_exception(
                store,
                prefix,
                workflow,
                started,
                exc,
                artifacts=artifacts,
                actions=actions,
                outputs={"operations": operations},
            )
            return outcome, []

    def preprocess_assay(
        self,
        store: DataStore,
        assay_plan: AssayPreprocessingPlan,
        *,
        active_cells: int,
        token: str,
        actions: list[str],
        operations: list[dict[str, Any]],
        artifacts: dict[str, ArtifactReferenceModel],
    ) -> PreprocessedAssayHandoff:
        assay = store.get_assay(assay_plan.assay)
        label_base = f"agent_{token}_{journal._safe_label(assay_plan.assay).lower()}"
        min_cells = int(assay_plan.featureParameters.get("minCells", 1))
        marker_features: ArtifactRef
        if assay_plan.featureMethod == "hvg":
            blacklist = self.rna_blacklist(assay_plan)
            graph_label = f"{label_base}_graph_features"
            actual_top_n = min(
                int(assay_plan.featureParameters["topN"]),
                max(1, assay.feats.N - 1),
            )
            hvg_features = store.mark_hvgs(
                from_assay=assay_plan.assay,
                cell_key="I",
                min_cells=min_cells,
                top_n=actual_top_n,
                blacklist=blacklist,
                show_plot=False,
                label=graph_label,
            )
            filtered_graph_label = f"{label_base}_filtered_graph_features"
            graph_features = self.exclude_exact_features(
                store,
                assay_plan,
                hvg_features,
                label=filtered_graph_label,
            )
            detected_label = f"{label_base}_detected_features"
            detected = store.select_detected_features(
                from_assay=assay_plan.assay,
                cell_key="I",
                min_cells=min_cells,
                label=detected_label,
            )
            marker_label = f"{label_base}_marker_features"
            marker_features = self.exclude_exact_features(
                store,
                assay_plan,
                detected,
                label=marker_label,
            )
            actions.extend(
                [
                    f"mark_hvgs:{assay_plan.assay}",
                    f"select_marker_features:{assay_plan.assay}",
                ]
            )
            operations.append(
                {
                    "operation": "mark_hvgs",
                    "assay": assay_plan.assay,
                    "cellKey": "I",
                    "minCells": min_cells,
                    "topN": actual_top_n,
                    "blacklist": blacklist,
                    "showPlot": False,
                    "label": graph_label,
                    "invalidateCache": False,
                    "artifact": ArtifactReferenceModel.from_artifact_ref(
                        hvg_features
                    ).model_dump(mode="json"),
                }
            )
            operations.extend(
                [
                    {
                        "operation": "set_feature_selection",
                        "assay": assay_plan.assay,
                        "source": ArtifactReferenceModel.from_artifact_ref(
                            hvg_features
                        ).model_dump(mode="json"),
                        "exactExcludedFeatures": list(assay_plan.exactExcludedFeatures),
                        "excludeFamilies": list(
                            assay_plan.featureParameters.get("excludeFamilies", [])
                        ),
                        "label": filtered_graph_label,
                        "artifact": ArtifactReferenceModel.from_artifact_ref(
                            graph_features
                        ).model_dump(mode="json"),
                    },
                    {
                        "operation": "select_detected_features",
                        "assay": assay_plan.assay,
                        "cellKey": "I",
                        "minCells": min_cells,
                        "label": detected_label,
                        "artifact": ArtifactReferenceModel.from_artifact_ref(
                            detected
                        ).model_dump(mode="json"),
                    },
                    {
                        "operation": "set_feature_selection",
                        "assay": assay_plan.assay,
                        "source": ArtifactReferenceModel.from_artifact_ref(
                            detected
                        ).model_dump(mode="json"),
                        "exactExcludedFeatures": list(assay_plan.exactExcludedFeatures),
                        "excludeFamilies": list(
                            assay_plan.featureParameters.get("excludeFamilies", [])
                        ),
                        "label": marker_label,
                        "artifact": ArtifactReferenceModel.from_artifact_ref(
                            marker_features
                        ).model_dump(mode="json"),
                    },
                ]
            )
            artifacts.update(
                {
                    f"{assay_plan.assay}_hvg_candidates": (
                        ArtifactReferenceModel.from_artifact_ref(hvg_features)
                    ),
                    f"{assay_plan.assay}_detected_features": (
                        ArtifactReferenceModel.from_artifact_ref(detected)
                    ),
                }
            )
        elif assay_plan.featureMethod == "prevalentPeaks":
            peak_label = f"{label_base}_prevalent_peaks"
            actual_top_n = min(
                int(assay_plan.featureParameters["topN"]),
                assay.feats.N - 1,
            )
            graph_features = store.mark_prevalent_peaks(
                from_assay=assay_plan.assay,
                cell_key="I",
                top_n=actual_top_n,
                label=peak_label,
            )
            marker_features = graph_features
            actions.append(f"mark_prevalent_peaks:{assay_plan.assay}")
            operations.append(
                {
                    "operation": "mark_prevalent_peaks",
                    "assay": assay_plan.assay,
                    "cellKey": "I",
                    "topN": actual_top_n,
                    "label": peak_label,
                    "invalidateCache": False,
                    "artifact": ArtifactReferenceModel.from_artifact_ref(
                        graph_features
                    ).model_dump(mode="json"),
                }
            )
        elif assay_plan.featureMethod == "panel":
            mask = np.ones(assay.feats.N, dtype=bool)
            ids = np.asarray(assay.feats.fetch_all("ids")).astype(str)
            names = np.asarray(assay.feats.fetch_all("names")).astype(str)
            excluded = set(assay_plan.exactExcludedFeatures)
            if excluded:
                mask &= ~np.isin(ids, list(excluded))
                mask &= ~np.isin(names, list(excluded))
            if int(mask.sum()) < 2:
                raise ValueError(
                    f"ADT assay {assay_plan.assay!r} has fewer than two non-control features"
                )
            panel_label = f"{label_base}_panel"
            graph_features = store.set_feature_selection(
                from_assay=assay_plan.assay,
                mask=mask,
                label=panel_label,
            )
            marker_features = graph_features
            actions.append(f"select_adt_panel:{assay_plan.assay}")
            operations.append(
                {
                    "operation": "set_feature_selection",
                    "assay": assay_plan.assay,
                    "selectedFeatures": int(mask.sum()),
                    "exactExcludedFeatures": sorted(excluded),
                    "label": panel_label,
                    "artifact": ArtifactReferenceModel.from_artifact_ref(
                        graph_features
                    ).model_dump(mode="json"),
                }
            )
        else:
            raise ValueError(f"Unsupported feature route {assay_plan.featureMethod!r}")
        normalized = store.run_normalization(
            from_assay=assay_plan.assay,
            cell_key="I",
            features=graph_features,
            log_transform=cast(
                bool,
                assay_plan.normalizationParameters.get("logTransform"),
            ),
            renormalize_subset=cast(
                bool,
                assay_plan.normalizationParameters.get("renormalizeSubset"),
            ),
            update_state=True,
        )
        graph_feature_group = store.load_artifact(graph_features)
        graph_feature_values = cast(Any, graph_feature_group["values"])
        selected_values = np.asarray(graph_feature_values[:], dtype=bool)
        graph_features_model = ArtifactReferenceModel.from_artifact_ref(graph_features)
        marker_features_model = ArtifactReferenceModel.from_artifact_ref(
            marker_features
        )
        normalized_model = ArtifactReferenceModel.from_artifact_ref(normalized)
        handoff = PreprocessedAssayHandoff(
            assay=assay_plan.assay,
            assayType=assay_plan.assayType,
            cellKey="I",
            reductionMethod=assay_plan.reductionMethod,
            graphFeatures=graph_features_model,
            markerFeatures=marker_features_model,
            normalized=normalized_model,
            nCells=active_cells,
            nFeatures=int(selected_values.sum()),
        )
        artifacts.update(
            {
                f"{assay_plan.assay}_graph_features": graph_features_model,
                f"{assay_plan.assay}_marker_features": marker_features_model,
                f"{assay_plan.assay}_normalized": normalized_model,
            }
        )
        actions.append(f"normalize:{assay_plan.assay}")
        operations.append(
            {
                "operation": "run_normalization",
                "assay": assay_plan.assay,
                "cellKey": "I",
                "features": graph_features_model.model_dump(mode="json"),
                "logTransform": assay_plan.normalizationParameters.get("logTransform"),
                "renormalizeSubset": assay_plan.normalizationParameters.get(
                    "renormalizeSubset"
                ),
                "updateState": True,
                "invalidateCache": False,
                "artifact": normalized_model.model_dump(mode="json"),
            }
        )
        logger.info(
            f"Preprocessed assay {assay_plan.assay!r}: "
            f"cells={handoff.nCells}, features={handoff.nFeatures}, "
            f"reduction={handoff.reductionMethod!r}"
        )
        return handoff

    def apply_cell_qc(
        self,
        store: DataStore,
        experimental: ExperimentalContextResult,
        actions: list[str],
        operations: list[dict[str, Any]],
    ) -> None:
        plan = experimental.cellQc
        logger.info(
            f"Applying cell QC action={plan.action!r}, profile={plan.profileId!r}"
        )
        if plan.action == "skip":
            actions.append("skip_cell_qc")
            operations.append(
                {
                    "operation": "skip_cell_qc",
                    "profileId": plan.profileId,
                    "cellKey": plan.cellKey,
                }
            )
            return
        profile = next(
            (
                value
                for value in experimental.qcProfiles
                if value.profileId == plan.profileId
            ),
            None,
        )
        if profile is None:
            raise ValueError("Experimental Context selected an unknown QC profile")
        if plan.action == "globalGaussian":
            store.auto_filter_cells(
                attrs=profile.attributes,
                min_p=float(profile.parameters.get("minP", 0.01)),
                max_p=float(profile.parameters.get("maxP", 0.99)),
                show_qc_plots=False,
            )
            actions.append(f"cell_qc_global:{profile.profileId}")
            operations.append(
                {
                    "operation": "auto_filter_cells",
                    "profileId": profile.profileId,
                    "attrs": list(profile.attributes),
                    "minP": float(profile.parameters.get("minP", 0.01)),
                    "maxP": float(profile.parameters.get("maxP", 0.99)),
                    "showQcPlots": False,
                    "sampleColumn": None,
                    "invalidateCache": False,
                }
            )
            return
        if plan.action == "sampleMad":
            if not plan.sampleColumn:
                raise ValueError("sampleMad QC requires a sample column")
            store.auto_filter_cells(
                attrs=profile.attributes,
                show_qc_plots=False,
                sample_column=plan.sampleColumn,
                n_mads=float(profile.parameters.get("nMads", 3.0)),
                min_cells_per_sample=int(
                    profile.parameters.get("minCellsPerSample", 20)
                ),
            )
            actions.append(f"cell_qc_sample_mad:{profile.profileId}")
            operations.append(
                {
                    "operation": "auto_filter_cells",
                    "profileId": profile.profileId,
                    "attrs": list(profile.attributes),
                    "minP": 0.01,
                    "maxP": 0.99,
                    "showQcPlots": False,
                    "sampleColumn": plan.sampleColumn,
                    "nMads": float(profile.parameters.get("nMads", 3.0)),
                    "minCellsPerSample": int(
                        profile.parameters.get("minCellsPerSample", 20)
                    ),
                    "invalidateCache": False,
                }
            )
            return
        raise ValueError(f"Unsupported cell QC action {plan.action!r}")

    def rna_blacklist(self, plan: AssayPreprocessingPlan) -> str:
        patterns: list[str] = []
        families = set(
            cast(list[str], plan.featureParameters.get("excludeFamilies", []))
        )
        if "mitochondrial" in families:
            patterns.append(r"^(MT-|mt-)")
        if "ribosomal" in families:
            patterns.append(r"^(RPS|RPL|MRPS|MRPL|Rps|Rpl|Mrps|Mrpl)")
        if "histone" in families:
            patterns.append(r"^(HIST|Hist)")
        patterns.extend(
            rf"^{re.escape(value)}$" for value in plan.exactExcludedFeatures if value
        )
        return "|".join(patterns) if patterns else r"(?!)"

    def exclude_exact_features(
        self,
        store: DataStore,
        plan: AssayPreprocessingPlan,
        source: ArtifactRef,
        *,
        label: str,
    ) -> ArtifactRef:
        families = set(
            cast(list[str], plan.featureParameters.get("excludeFamilies", []))
        )
        if not plan.exactExcludedFeatures and not families:
            return source
        assay = store.get_assay(plan.assay)
        source_group = store.load_artifact(source)
        source_values = cast(Any, source_group["values"])
        mask = np.asarray(source_values[:], dtype=bool)
        ids = np.asarray(assay.feats.fetch_all("ids")).astype(str)
        names = np.asarray(assay.feats.fetch_all("names")).astype(str)
        excluded = set(plan.exactExcludedFeatures)
        mask &= ~np.isin(ids, list(excluded))
        mask &= ~np.isin(names, list(excluded))
        family_patterns: list[str] = []
        if "mitochondrial" in families:
            family_patterns.append(r"^(MT-|mt-)")
        if "ribosomal" in families:
            family_patterns.append(r"^(RPS|RPL|MRPS|MRPL|Rps|Rpl|Mrps|Mrpl)")
        if "histone" in families:
            family_patterns.append(r"^(HIST|Hist)")
        if family_patterns:
            technical = np.zeros(len(mask), dtype=bool)
            combined = re.compile("|".join(family_patterns))
            technical |= np.fromiter(
                (combined.search(value) is not None for value in ids),
                dtype=bool,
                count=len(ids),
            )
            technical |= np.fromiter(
                (combined.search(value) is not None for value in names),
                dtype=bool,
                count=len(names),
            )
            mask &= ~technical
        if not mask.any():
            raise ValueError("Exact marker exclusions removed every feature")
        return store.set_feature_selection(
            from_assay=plan.assay,
            mask=mask,
            label=label,
        )
