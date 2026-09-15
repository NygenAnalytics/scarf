"""Validate the selected full RNA analysis and compute its final layout."""

from collections.abc import Sequence
from typing import Any

from ...datastore.datastore import DataStore
from ...graph.feature_projection import graph_cell_selection
from ...utils.logging import logger
from ..parameter_tuning.contracts import ParameterTuningReport
from ..parameter_tuning.selection import promote_parameter_candidate
from ..types import ArtifactReferenceModel
from . import journal
from .models import (
    AutomatedPreprocessingPlan,
    FinalAnalysisHandoff,
    OrchestrationRequestRecord,
    OrchestrationResumeRecord,
    PreprocessedAssayHandoff,
    StageEvidenceReference,
    WorkflowIdentity,
    WorkflowStageAttempt,
    WorkflowStageLink,
    artifact_model_to_ref,
)
from .rna import selected_store_rna_assay, validate_rna_handoffs, validate_rna_plan


class FinalizationStagesMixin:
    """Finish one full-cohort RNA representation using its exact saved artifacts."""

    def analysis_finalization_stage(
        self,
        store: DataStore,
        workflow: WorkflowIdentity,
        request_record: OrchestrationRequestRecord,
        parents: Sequence[WorkflowStageLink],
        plan: AutomatedPreprocessingPlan,
        preprocessed: Sequence[PreprocessedAssayHandoff],
        tuning_report: ParameterTuningReport,
        tuning_reference: StageEvidenceReference,
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
            logger.info("Reusing the validated final RNA analysis")
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
            },
            resume_record=resume_record,
        )
        artifacts: dict[str, ArtifactReferenceModel] = {}
        operations: list[dict[str, Any]] = []
        try:
            assay_name = selected_store_rna_assay(store, request_record.request)
            validate_rna_plan(plan, assay_name)
            validate_rna_handoffs(preprocessed, assay_name)
            handoff = preprocessed[0]
            cells = handoff.cellSelection
            if cells is None or cells != tuning_report.cellSelection:
                raise ValueError(
                    "Finalization requires the original full-cohort selection, not sampled cells"
                )
            if handoff.normalized is None or handoff.markerFeatures is None:
                raise ValueError(
                    "Finalization requires exact normalized and marker features"
                )
            if (
                tuning_report.status != "done"
                or tuning_report.recommendedIntegrationId is not None
                or tuning_report.assayReports
            ):
                raise ValueError(
                    "Finalization requires a completed single-RNA recommendation"
                )
            if journal.read_stage_evidence(
                store, tuning_reference
            ) != tuning_report.model_dump(mode="json"):
                raise ValueError("Final tuning differs from its committed evidence")
            selected = promote_parameter_candidate(
                store,
                report=tuning_report,
                normalized=artifact_model_to_ref(handoff.normalized),
            )
            if selected.parameters.reductionMethod != "pca":
                raise ValueError(
                    "RNA finalization requires the selected PCA representation"
                )
            selected_artifacts = {
                name: ArtifactReferenceModel.model_validate(
                    value.model_dump(mode="json")
                )
                for name, value in selected.artifacts.items()
            }
            required = {"pca", "connectivityMap", "clusters", "markerTable"}
            if not required.issubset(selected_artifacts):
                raise ValueError(
                    f"Final candidate lacks required artifacts: {sorted(required - selected_artifacts.keys())}"
                )
            graph = selected_artifacts["connectivityMap"]
            clusters = selected_artifacts["clusters"]
            markers = selected_artifacts["markerTable"]
            if tuning_report.finalClusterArtifact is None or artifact_model_to_ref(
                clusters
            ) != artifact_model_to_ref(tuning_report.finalClusterArtifact):
                raise ValueError("Finalization changed the selected cluster artifact")
            cells_ref = artifact_model_to_ref(cells)
            graph_ref = artifact_model_to_ref(graph)
            if graph_cell_selection(store.zw, graph_ref) != cells_ref:
                raise ValueError(
                    "Selected graph does not contain the full-cohort selection"
                )
            for label, ref in (("clusters", clusters), ("markers", markers)):
                status = store.inspect_artifact(artifact_model_to_ref(ref))
                if not status.complete or ref.assay != assay_name:
                    raise ValueError(
                        f"Final {label} are incomplete or belong to another assay"
                    )
                inputs = status.inputs or {}
                if inputs.get("cell_selection") != cells_ref.to_dict():
                    raise ValueError(f"Final {label} use a different cell selection")
                parent_key, parent = (
                    ("graph", graph) if label == "clusters" else ("clusters", clusters)
                )
                if inputs.get(parent_key) != artifact_model_to_ref(parent).to_dict():
                    raise ValueError(
                        f"Final {label} do not match the selected {parent_key}"
                    )
            for ref in selected_artifacts.values():
                store.load_artifact(artifact_model_to_ref(ref))
            initialization_ref = store.build_embedding_initialization(
                artifact_model_to_ref(selected_artifacts["pca"]),
                n_centroids=min(1000, handoff.nCells),
                rand_state=4466,
                invalidate_cache=False,
            )
            umap_ref = store.run_umap(
                graph_ref,
                initialization_ref,
                parallel=False,
                random_seed=4444,
                invalidate_cache=False,
            )
            initialization = ArtifactReferenceModel.from_artifact_ref(
                initialization_ref
            )
            umap = ArtifactReferenceModel.from_artifact_ref(umap_ref)
            operations.extend(
                [
                    {
                        "operation": "build_embedding_initialization",
                        "artifact": initialization.model_dump(mode="json"),
                    },
                    {"operation": "run_umap", "artifact": umap.model_dump(mode="json")},
                ]
            )
            doublet_scores = [
                value
                for name, value in sorted(selected_artifacts.items())
                if name.startswith("doubletScore:")
            ]
            doublet_selections = [
                value
                for name, value in sorted(selected_artifacts.items())
                if name.startswith("doubletCellSelection:")
            ]
            if len(doublet_scores) != len(doublet_selections):
                raise ValueError("Advisory doublet scores lack exact selection lineage")
            limitations = list(
                dict.fromkeys([*plan.limitations, *tuning_report.limitations])
            )
            doublet_limitations = [
                warning
                for warning in selected.warnings
                if "doublet" in warning.lower()
                or "physical capture identity" in warning.lower()
            ]
            if not doublet_scores and not any(
                warning.startswith("Advisory doublet scoring was not run for assay ")
                for warning in doublet_limitations
            ):
                raise ValueError(
                    "Selected cluster evidence lacks advisory doublet scores"
                )
            limitations.extend(doublet_limitations)
            artifacts.update(selected_artifacts)
            artifacts.update(
                cellSelection=cells,
                normalized=handoff.normalized,
                graph=graph,
                clusters=clusters,
                markers=markers,
                markerFeatures=handoff.markerFeatures,
                embeddingInitialization=initialization,
                umap=umap,
            )
            final = FinalAnalysisHandoff(
                workflowRunId=workflow.workflowRunId,
                primaryAssay=assay_name,
                markerAssay=assay_name,
                cellSelection=cells,
                graph=graph,
                clusters=clusters,
                embeddingInitialization=initialization,
                umap=umap,
                markerFeatures=handoff.markerFeatures,
                markers=markers,
                doubletScores=doublet_scores,
                doubletScoreSelections=doublet_selections,
                limitations=list(dict.fromkeys(limitations)),
            )
            outcome = journal._complete_attempt(
                started,
                status="done",
                artifacts=artifacts,
                outputs={
                    "finalAnalysis": final.model_dump(mode="json"),
                    "operations": operations,
                },
                actions=["reuse_validated_full_cohort", "run_final_umap"],
                notes=final.limitations,
            )
            journal._save_outcome(store.zw, prefix, outcome)
            logger.info(
                f"Final RNA analysis: {selected.metrics.nClusters} populations; descriptive markers saved"
            )
            return outcome, final
        except Exception as exc:
            outcome = journal.finish_exception(
                store,
                prefix,
                workflow,
                started,
                exc,
                artifacts=artifacts,
                outputs={"operations": operations},
            )
            return outcome, FinalAnalysisHandoff.get_blank()
