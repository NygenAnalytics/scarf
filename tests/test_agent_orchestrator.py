"""Public facade, model, and end-to-end orchestrator contracts."""

import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

import scarf.agent as agent_module
import scarf.agent.orchestrator as orchestrator_module
from scarf.agent.biological_interpretation import (
    BiologicalInterpretationReport,
    ClusterCompositionEvidence,
    ClusterInterpretation,
    ClusterMarkerBatchEvidence,
)
from scarf.agent.data_enrichment import (
    DataEnrichmentReport,
    FeatureSelectionPolicy,
    StudyContextSummary,
)
from scarf.agent.experimental_context import (
    BatchCorrectionPlan,
    CellQcPlan,
    CovariateEvidence,
    ExperimentalContextDecision,
)
from scarf.agent.orchestrator import (
    AgentOrchestrator,
    AssayPreprocessingPlan,
    AutomatedPreprocessingPlan,
    AutomatedWorkflowConfig,
    AutomatedWorkflowRequest,
    AutomatedWorkflowResult,
    AutomatedWorkflowResumeRequest,
    FinalAnalysisHandoff,
    NativeAnalysisHandoff,
    PreprocessedAssayHandoff,
    WorkflowNeedsInput,
    WorkflowQuestion,
    WorkflowStageAttempt,
    WorkflowStageLink,
    artifact_model_to_ref,
)
from scarf.agent.persistence import (
    load_agent_record,
    load_agent_report,
)
from scarf.agent.parameter_tuning import (
    FinalGraphSelection,
    ParameterTuningReport,
)
from scarf.datastore.datastore import DataStore
from tests.test_agent_ingest import _write_h5ad


_PLAN_CHECKSUM = "a" * 64


def _rna_workflow_model() -> tuple[FunctionModel, dict[str, int]]:
    state = {
        "enrichment": 0,
        "context": 0,
        "parameter": 0,
        "biology": 0,
        "requests": 0,
    }

    def prompt_text(messages: list[ModelMessage]) -> str:
        return "\n".join(
            part.content
            for message in messages
            for part in message.parts
            if isinstance(getattr(part, "content", None), str)
        )

    def tool_result(
        messages: list[ModelMessage],
        tool_name: str,
        model_type: Any,
    ) -> Any:
        for message in reversed(messages):
            for part in reversed(message.parts):
                if isinstance(part, ToolReturnPart) and part.tool_name == tool_name:
                    content = part.content
                    if isinstance(content, model_type):
                        return content
                    if isinstance(content, str):
                        return model_type.model_validate_json(content)
                    return model_type.model_validate(content)
        raise AssertionError(f"Missing tool return {tool_name!r}")

    async def reply(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        state["requests"] += 1
        tools = {tool.name for tool in info.function_tools}
        if "inspect_assay_features_batch" in tools or (
            state["enrichment"] == 1 and "find_present_features_batch" in tools
        ):
            if state["enrichment"] == 0:
                state["enrichment"] = 1
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="inspect_assay_features_batch",
                            args={},
                        )
                    ]
                )
            state["enrichment"] = 2
            report = DataEnrichmentReport(
                status="done",
                studyContextSummary=StudyContextSummary(
                    organismReferences=["human"],
                    tissueReferences=["peripheral blood"],
                ),
                policies=[
                    FeatureSelectionPolicy(
                        assay="RNA",
                        species="unknown",
                        rationale="Use the observed RNA feature inventory.",
                        evidenceIds=["assay:RNA:species"],
                    )
                ],
            )
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=info.output_tools[0].name,
                        args=report.model_dump(),
                    )
                ]
            )

        if tools.intersection(
            {
                "inspect_cell_covariates",
                "analyze_experimental_design",
                "score_current_representation",
            }
        ) or state["context"] in {1, 2}:
            if state["context"] == 0:
                state["context"] = 1
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="inspect_cell_covariates",
                            args={},
                        )
                    ]
                )
            if state["context"] == 1:
                state["context"] = 2
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="analyze_experimental_design",
                            args={
                                "column_domains": {},
                                "coefficients_of_interest": [],
                                "units_of_inference": {},
                                "batch_columns": [],
                            },
                        )
                    ]
                )
            state["context"] = 3
            context_evidence = tool_result(
                messages,
                "analyze_experimental_design",
                CovariateEvidence,
            )
            profile = next(
                value
                for value in context_evidence.qcProfiles
                if value.action == "globalGaussian"
            )
            evidence_id = profile.evidenceId
            decision = ExperimentalContextDecision(
                batchCorrection=BatchCorrectionPlan(
                    action="skip",
                    rationale="No trusted technical batch column was supplied.",
                    evidenceIds=[evidence_id],
                ),
                cellQc=CellQcPlan(
                    action=profile.action,
                    profileId=profile.profileId,
                    driverAssay=profile.driverAssay,
                    driverAssayType=profile.driverAssayType,
                    attributes=profile.attributes,
                    artifactMetrics=profile.artifactMetrics,
                    rationale="Apply the bounded global RNA QC profile.",
                    evidenceIds=[evidence_id],
                ),
                rationale="No experimental covariates were supplied.",
                evidenceIds=[evidence_id],
            )
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=info.output_tools[0].name,
                        args=decision.model_dump(),
                    )
                ]
            )

        if (
            tools.intersection(
                {"inspect_cluster_composition", "inspect_cluster_markers_batch"}
            )
            or state["biology"]
        ):
            if state["biology"] == 0:
                state["biology"] = 1
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="inspect_cluster_composition",
                            args={},
                        )
                    ]
                )
            if state["biology"] == 1:
                composition = tool_result(
                    messages,
                    "inspect_cluster_composition",
                    ClusterCompositionEvidence,
                )
                state["biology"] = 2
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="inspect_cluster_markers_batch",
                            args={"cluster_ids": list(composition.clusterCounts)},
                        )
                    ]
                )
            batch = tool_result(
                messages,
                "inspect_cluster_markers_batch",
                ClusterMarkerBatchEvidence,
            )
            interpretations = []
            for cluster in batch.clusters:
                if cluster.evidenceId and cluster.markers:
                    marker = cluster.markers[0]
                    marker_name = marker.featureName or marker.featureId
                    interpretations.append(
                        ClusterInterpretation(
                            clusterId=cluster.clusterId,
                            proposedIdentity=f"{marker_name}-high RNA state",
                            identityIsHypothesis=True,
                            confidence="low",
                            rationale=(
                                f"The observed marker panel is led by {marker_name}."
                            ),
                            evidenceIds=[cluster.evidenceId],
                        )
                    )
            state["biology"] = 3
            report = BiologicalInterpretationReport(
                status="done",
                clusterInterpretations=interpretations,
                evidenceIds=[item.evidenceIds[0] for item in interpretations],
                limitations=["Synthetic data supports marker-linked hypotheses only."],
                stopReason=(
                    "Every cluster with returned marker evidence was reviewed."
                ),
            )
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=info.output_tools[0].name,
                        args=report.model_dump(),
                    )
                ]
            )

        prompt = prompt_text(messages)
        if state["parameter"] == 0:
            match = re.search(
                r'"candidateId"\s*:\s*"([A-Za-z0-9_]+)"',
                prompt,
            )
            assert match is not None
            candidate_id = match.group(1)
            evidence_id = f"candidate:{candidate_id}:clusters"
            assay_report = ParameterTuningReport(
                status="done",
                recommendedCandidateId=candidate_id,
                confidence="high",
                rationale="The only authorized native branch is eligible.",
                evidenceIds=[evidence_id],
                stopReason="The bounded one-candidate screen completed.",
            )
            report = ParameterTuningReport(
                status="done",
                assayReports={"RNA": assay_report},
                rationale="The RNA native screen completed.",
                evidenceIds=[evidence_id],
                stopReason="Native selection completed.",
            )
            state["parameter"] = 1
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=info.output_tools[0].name,
                        args=report.model_dump(),
                    )
                ]
            )

        match = re.search(
            r'"optionId"\s*:\s*"(native:RNA:([A-Za-z0-9_]+))"',
            prompt,
        )
        assert match is not None
        option_id, candidate_id = match.groups()
        evidence_id = f"native:RNA:candidate:{candidate_id}:clusters"
        selection = FinalGraphSelection(
            status="done",
            selectedOptionId=option_id,
            graphMethod="native",
            nativeAssay="RNA",
            nativeCandidateId=candidate_id,
            markerAssay="RNA",
            confidence="high",
            rationale="The sole eligible native graph is selected.",
            evidenceIds=[evidence_id],
        )
        state["parameter"] = 2
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args=selection.model_dump(),
                )
            ]
        )

    return FunctionModel(reply), state


def test_public_orchestrator_models_have_factories_and_camelcase_fields() -> None:
    models = (
        AssayPreprocessingPlan,
        AutomatedPreprocessingPlan,
        AutomatedWorkflowConfig,
        AutomatedWorkflowRequest,
        AutomatedWorkflowResult,
        AutomatedWorkflowResumeRequest,
        FinalAnalysisHandoff,
        NativeAnalysisHandoff,
        PreprocessedAssayHandoff,
        StudyContextSummary,
        CellQcPlan,
        WorkflowNeedsInput,
        WorkflowQuestion,
        WorkflowStageAttempt,
        WorkflowStageLink,
    )

    for model in models:
        assert isinstance(model.get_blank(), model)
        assert isinstance(model.get_example(), model)
        assert all("_" not in field_name for field_name in model.model_fields)
    for model in (
        AutomatedPreprocessingPlan,
        PreprocessedAssayHandoff,
        FinalAnalysisHandoff,
    ):
        assert "cellSelection" in model.model_fields
        assert "cellKey" not in model.model_fields
    for model in (NativeAnalysisHandoff, FinalAnalysisHandoff):
        assert "clusterColumn" not in model.model_fields
        assert "umapColumns" not in model.model_fields


def test_orchestrator_package_preserves_the_public_facade() -> None:
    assert agent_module.AgentOrchestrator is orchestrator_module.AgentOrchestrator
    assert orchestrator_module.__all__ == [
        "AgentOrchestrator",
        "AssayPreprocessingPlan",
        "AutomatedPreprocessingPlan",
        "AutomatedWorkflowConfig",
        "AutomatedWorkflowRequest",
        "AutomatedWorkflowResult",
        "AutomatedWorkflowResumeRequest",
        "FinalAnalysisHandoff",
        "NativeAnalysisHandoff",
        "PreprocessedAssayHandoff",
        "WorkflowNeedsInput",
        "WorkflowQuestion",
        "WorkflowStageAttempt",
        "WorkflowStageLink",
        "artifact_model_to_ref",
    ]


@pytest.mark.slow
def test_rna_h5ad_completes_public_automated_workflow(tmp_path: Path) -> None:
    rng = np.random.default_rng(4444)
    values = rng.poisson(1.0, size=(80, 50)).astype(np.uint16)
    values[:40, :12] += rng.poisson(9.0, size=(40, 12)).astype(np.uint16)
    values[40:, 12:24] += rng.poisson(9.0, size=(40, 12)).astype(np.uint16)
    feature_names = [
        b"CD3D",
        b"CD3E",
        b"IL7R",
        b"LTB",
        b"MALAT1",
        b"CCR7",
        b"LDHB",
        b"NOSIP",
        b"TCF7",
        b"LEF1",
        b"MAL",
        b"LTST1",
        b"MS4A1",
        b"CD79A",
        b"CD37",
        b"CD74",
        b"HLA-DRA",
        b"CD22",
        b"CD83",
        b"CD19",
        b"BANK1",
        b"CD79B",
        b"CD52",
        b"CD48",
        *[f"GENE{index}".encode() for index in range(26)],
    ]
    source = tmp_path / "small_rna.h5ad"
    target = tmp_path / "small_rna.zarr"
    _write_h5ad(
        source,
        values,
        feature_types=[b"Gene Expression"] * values.shape[1],
        feature_names=feature_names,
    )
    model, state = _rna_workflow_model()
    orchestrator = AgentOrchestrator(
        model,
        config=AutomatedWorkflowConfig(
            primaryInitialCandidates=1,
            secondaryInitialCandidates=1,
            maxRefinedCandidatesPerAssay=0,
            maxHarmonyCandidatesPerAssay=0,
            integrationResolutionCandidates=1,
            maxCandidateBranches=1,
            minClusterCells=2,
        ),
    )

    result = orchestrator.run(
        AutomatedWorkflowRequest(
            sourcePath=str(source),
            zarrPath=str(target),
            studyContext=(
                "A human peripheral blood RNA study for a deterministic "
                "acceptance test."
            ),
            allowAssumptions=True,
            primaryAssay="RNA",
            markerAssay="RNA",
            analysisAssays=["RNA"],
        )
    )

    assert result.status == "completed", result.notes
    assert result.currentStage == "biological_interpretation"
    assert result.workflowRun is not None
    assert result.workflowRun.status == "completed"
    report_path = (
        target
        / "agents"
        / "runs"
        / result.workflowRun.workflowRunId
        / "report"
        / "index.html"
    )
    assert report_path.is_file()
    assert "Nygen Analytics" in report_path.read_text(encoding="utf-8")
    assert state["requests"] == 9
    assert [reference.agentName for reference in result.reportReferences] == [
        "data_enrichment",
        "experimental_context",
        "parameter_tuning",
        "biological_interpretation",
    ]
    assert result.finalAnalysis is not None
    assert result.preprocessingPlan is not None
    final = result.finalAnalysis
    assert final.graphMethod == "native"
    assert final.primaryAssay == final.markerAssay == "RNA"
    assert final.graph is not None
    assert final.clusters is not None
    assert final.cellSelection is not None
    assert final.embeddingInitialization is not None
    assert final.umap is not None
    assert final.markers is not None
    assert final.cellSelection.kind == "cell_selection"
    assert final.clusters.kind == "cluster_labels"
    assert final.embeddingInitialization.kind == "embedding_initialization"
    assert final.umap.kind == "embedding"
    assert final.cellSelection != result.preprocessingPlan.cellSelection
    persisted = DataStore(
        str(target),
        default_assay="RNA",
        min_features_per_cell=-1,
        mito_pattern="",
        ribo_pattern="",
        zarr_mode="r",
    )
    umap_inputs = persisted.inspect_artifact(artifact_model_to_ref(final.umap)).inputs
    assert umap_inputs is not None
    assert umap_inputs["graph"] == artifact_model_to_ref(final.graph).to_dict()
    assert (
        umap_inputs["initialization"]
        == artifact_model_to_ref(final.embeddingInitialization).to_dict()
    )
    marker_inputs = persisted.inspect_artifact(
        artifact_model_to_ref(final.markers)
    ).inputs
    assert marker_inputs is not None
    assert marker_inputs["clusters"] == artifact_model_to_ref(final.clusters).to_dict()
    assert (
        marker_inputs["cell_selection"]
        == artifact_model_to_ref(final.cellSelection).to_dict()
    )

    biology_reference = next(
        reference
        for reference in result.reportReferences
        if reference.agentName == "biological_interpretation"
    )
    biology = load_agent_report(target, biology_reference)
    assert isinstance(biology, BiologicalInterpretationReport)
    assert biology.status == "done"
    assert biology.clusterInterpretations
    assert biology.markerArtifact == final.markers
    record = load_agent_record(target, biology_reference)
    assert len(record.invocation.parentReports) == 3
    assert record.invocation.tuningBiologyHandoff is not None
    assert set(record.invocation.artifacts) == {
        "cellSelection",
        "clusters",
        "markers",
        "markerFeatures",
    }
    assert record.invocation.artifacts["cellSelection"] == final.cellSelection
