"""Public facade, model, and end-to-end orchestrator contracts."""

from tests.agent_examples import example
from tests.agent_comparison_examples import observed_action

import json
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
from scarf.agent.data_enrichment import (
    DataEnrichmentReport,
    FeatureSelectionPolicy,
    StudyContextSummary,
)
from scarf.agent.decisions.kernel import DecisionSelection
from scarf.agent.experimental_context import (
    BatchCorrectionPlan,
    CellQcPlan,
    ExperimentalContextDecision,
)
from scarf.agent.parameter_tuning import ParameterTuningReport
from scarf.agent.parameter_tuning.contracts import ParameterCandidateEvaluation
from scarf.agent.orchestrator.journal import (
    analysis_snapshot,
    load_checkpoint,
    _ensure_orchestration_store,
)
from scarf.agent.orchestrator.rna_tuning import TuningAction
from scarf.agent.orchestrator import (
    AgentOrchestrator,
    AutomatedWorkflowConfig,
    AutomatedWorkflowRequest,
    AutomatedWorkflowResumeRequest,
)
from scarf.agent.orchestrator.models import (
    AssayPreprocessingPlan,
    AutomatedPreprocessingPlan,
    AutomatedWorkflowResult,
    FinalAnalysisHandoff,
    PreprocessedAssayHandoff,
    WorkflowNeedsInput,
    WorkflowQuestion,
    WorkflowStageAttempt,
    WorkflowStageLink,
    artifact_model_to_ref,
)
from scarf.datastore.datastore import DataStore
from scarf.storage.refs import ArtifactRef
from scarf.storage.selections import read_stored_selection_indices
from tests.test_agent_ingest import _write_h5ad


_PLAN_CHECKSUM = "a" * 64


def _rna_workflow_model() -> tuple[FunctionModel, dict[str, Any]]:
    state: dict[str, Any] = {
        "enrichment": 0,
        "context": 0,
        "parameter": 0,
        "pca_pauses": 0,
        "pca_prompts": 0,
        "biology": 0,
        "requests": 0,
    }

    def prompt_text(messages: list[ModelMessage]) -> str:
        values: list[str] = []
        for message in messages:
            for part in message.parts:
                content = getattr(part, "content", None)
                if isinstance(content, str):
                    values.append(content)
                elif isinstance(content, tuple):
                    values.extend(item for item in content if isinstance(item, str))
        return "\n".join(values)

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
                    if model_type is dict:
                        return (
                            json.loads(content) if isinstance(content, str) else content
                        )
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
                dict,
            )
            profile = next(
                value
                for value in context_evidence["qcProfiles"]
                if value["registeredProfile"] is not None
            )
            evidence_id = profile["evidenceId"]
            decision = ExperimentalContextDecision(
                batchCorrection=BatchCorrectionPlan(
                    action="skip",
                    rationale="No trusted technical batch column was supplied.",
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

        prompt = prompt_text(messages)
        payload, _ = json.JSONDecoder().raw_decode(prompt[prompt.index("{") :])
        if any(
            {"selectedCandidateId", "correctionNeed", "comparisonConclusions"}.issubset(
                tool.parameters_json_schema.get("properties", {})
            )
            for tool in info.output_tools
        ):
            state["pca_prompts"] += 1
            selected = next(
                (
                    row["candidateId"]
                    for row in payload["candidates"]
                    if row["parameters"]["leidenResolution"] == 0.5
                ),
                payload["currentCandidateId"],
            )
            if payload["comparisonCoverage"]["phase"] == "sensitivity":
                selected = payload["currentCandidateId"]
            action = TuningAction.model_validate(
                observed_action(payload, selected=selected)
            )
            if action.action == "accept" and state["pca_pauses"] == 0:
                state["answer"] = action.model_dump(mode="json")
                action = action.model_copy(
                    update={
                        "action": "defer",
                        "rationale": "Review the completed comparisons before validating the selected combination on all retained cells.",
                    }
                )
                state["pca_pauses"] += 1
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=info.output_tools[0].name, args=action.model_dump()
                    )
                ]
            )
        payload, _ = json.JSONDecoder().raw_decode(prompt[prompt.index("{") :])
        decision = payload["spec"]
        decision_id = decision["decisionId"]
        if decision_id == "pcaPrefix":
            state["pca_prompts"] += 1
        evidence_by_class: dict[str, str] = {}
        evidence_class_by_id: dict[str, str] = {}
        for item in payload["evidence"]["evidence"]:
            evidence_by_class.setdefault(
                item["evidenceClass"],
                item["evidenceId"],
            )
            evidence_class_by_id[item["evidenceId"]] = item["evidenceClass"]
        preferred = decision.get("metricPreferredOptionId")
        selected = (
            next(
                option for option in decision["options"] if option["status"] == "defer"
            )
            if decision_id == "pcaPrefix" and state["pca_pauses"] == 0
            else next(
                option
                for option in decision["options"]
                if option["optionId"] == preferred
            )
            if preferred is not None
            else next(
                option
                for option in decision["options"]
                if option["status"] in {"apply", "skip"}
            )
        )
        if decision_id == "pcaPrefix" and selected["status"] == "defer":
            state["pca_pauses"] += 1
        evidence_ids = list(selected.get("requiredEvidenceIds", []))
        cited_classes = {
            evidence_class_by_id[evidence_id] for evidence_id in evidence_ids
        }
        for evidence_class in selected["requiredEvidenceClasses"]:
            if evidence_class not in cited_classes:
                evidence_ids.append(evidence_by_class[evidence_class])
        selection = DecisionSelection(
            selectedOptionId=selected["optionId"],
            evidenceIds=evidence_ids,
            rationale="Select the first eligible registered option for this test.",
            confidence="high",
        )
        state["parameter"] += 1
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
        assert isinstance(example(model), model)
        assert all("_" not in field_name for field_name in model.model_fields)
    for model in (
        AutomatedPreprocessingPlan,
        PreprocessedAssayHandoff,
        FinalAnalysisHandoff,
    ):
        assert "cellSelection" in model.model_fields
        assert "cellKey" not in model.model_fields
    for model in (FinalAnalysisHandoff,):
        assert "clusterColumn" not in model.model_fields
        assert "umapColumns" not in model.model_fields


def test_orchestrator_package_preserves_the_public_facade() -> None:
    assert not hasattr(agent_module, "AgentOrchestrator")
    assert orchestrator_module.__all__ == [
        "AgentOrchestrator",
        "AutomatedWorkflowConfig",
        "AutomatedWorkflowRequest",
        "AutomatedWorkflowResumeRequest",
    ]


@pytest.mark.slow
def test_rna_h5ad_completes_public_automated_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scarf.agent.orchestrator import rna_tuning as tuning_module

    rng = np.random.default_rng(4444)
    values = rng.poisson(0.2, size=(80, 50)).astype(np.uint16)
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
    pca_diagnostic_calls: list[ArtifactRef] = []
    augment_pca = tuning_module.augment_pca_evaluations

    def track_pca_diagnostics(*args: Any, **kwargs: Any) -> Any:
        pca_diagnostic_calls.append(kwargs["feature_selection"])
        return augment_pca(*args, **kwargs)

    monkeypatch.setattr(tuning_module, "augment_pca_evaluations", track_pca_diagnostics)
    orchestrator = AgentOrchestrator(
        model,
        config=AutomatedWorkflowConfig(screeningCells=60, maxScreeningCells=70),
    )

    request = AutomatedWorkflowRequest(
        sourcePath=str(source),
        zarrPath=str(target),
        studyContext=(
            "A human peripheral blood RNA study for a deterministic acceptance test."
        ),
        studyObjective="Discover stable RNA populations.",
        primaryAssay="RNA",
        markerAssay="RNA",
        analysisAssays=["RNA"],
    )
    paused = orchestrator.run(request)

    assert paused.status == "needsInput"
    assert paused.currentStage == "parameter_tuning"
    assert paused.workflowRunId is not None
    assert paused.needsInput is not None
    assert len(paused.needsInput.questions) == 1
    question = paused.needsInput.questions[0]
    assert question.questionId == "parameter_tuning"
    pca_diagnostics_before_resume = list(pca_diagnostic_calls)
    resume_request = AutomatedWorkflowResumeRequest(
        zarrPath=str(target),
        workflowRunId=paused.workflowRunId,
        answers={"parameter_tuning": state["answer"]},
    )
    editable = DataStore(
        str(target),
        default_assay="RNA",
        min_features_per_cell=-1,
        mito_pattern="",
        ribo_pattern="",
    )
    original_counts = editable.cells.fetch_all("RNA_nCounts")
    changed_counts = original_counts.copy()
    changed_counts[0] += 1
    editable.cells.insert("RNA_nCounts", changed_counts, overwrite=True)
    rejected = orchestrator.resume(resume_request)
    assert rejected.status == "failed"
    assert any("metadata" in note for note in rejected.notes)
    assert pca_diagnostic_calls == pca_diagnostics_before_resume
    editable.cells.insert("RNA_nCounts", original_counts, overwrite=True)
    execute_candidate = tuning_module.RnaTuningRun.execute

    def interrupt_before_full(self: Any, scope: str, *args: Any, **kwargs: Any) -> Any:
        if scope == "full":
            monkeypatch.setattr(
                tuning_module.RnaTuningRun, "execute", execute_candidate
            )
            raise KeyboardInterrupt(
                "interrupt after the screening answer was committed"
            )
        return execute_candidate(self, scope, *args, **kwargs)

    monkeypatch.setattr(tuning_module.RnaTuningRun, "execute", interrupt_before_full)
    with pytest.raises(KeyboardInterrupt, match="screening answer"):
        orchestrator.resume(resume_request)
    assert pca_diagnostic_calls == pca_diagnostics_before_resume
    result = orchestrator.resume(resume_request)

    assert result.status == "completed", result.notes
    assert len(pca_diagnostic_calls) == len(pca_diagnostics_before_resume) + 1
    assert state["pca_prompts"] >= 3
    assert result.currentStage == "analysis_finalization"
    assert result.workflowRunId is not None
    report_path = result.report()
    assert report_path.is_file()
    assert "Scarf analysis summary" in report_path.read_text(encoding="utf-8")
    assert state["requests"] >= 8
    assert state["biology"] == 0
    persisted = DataStore(
        str(target),
        default_assay="RNA",
        min_features_per_cell=-1,
        mito_pattern="",
        ribo_pattern="",
        zarr_mode="r",
    )
    snapshot = analysis_snapshot(persisted, result.workflowRunId)
    stages = {stage["stage"]: stage for stage in snapshot["stages"]}
    preprocessing_plan = AutomatedPreprocessingPlan.model_validate(
        stages["preprocessing_plan"]["outputs"]["preprocessingPlan"]
    )
    assert preprocessing_plan.cellQualityPayload is not None
    final = FinalAnalysisHandoff.model_validate(snapshot["finalAnalysis"])
    assert final.primaryAssay == final.markerAssay == "RNA"
    assert final.graph is not None and final.clusters is not None
    assert final.cellSelection is not None and final.umap is not None
    assert final.embeddingInitialization is not None and final.markers is not None
    assert len(final.doubletScores) == 1
    assert final.cellSelection != preprocessing_plan.cellSelection
    parameter_report = ParameterTuningReport.model_validate(
        stages["parameter_tuning"]["report"]
    )
    tuning_evaluations = parameter_report.evaluations
    pca_evidence = next(
        evaluation
        for evaluation in tuning_evaluations
        if evaluation.metrics.componentVariance
    )
    assert "representationDiagnostic" in pca_evidence.artifacts
    assert any(
        evidence_id.endswith(":pcaLoadings") for evidence_id in pca_evidence.evidenceIds
    )
    cluster_evidence = next(
        evaluation
        for evaluation in tuning_evaluations
        if evaluation.metrics.doubletHighScoreConcentration is not None
    )
    assert cluster_evidence.metrics.markerCoherence is not None
    assert "doubletScore:0" in cluster_evidence.artifacts
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
    doublet_inputs = persisted.inspect_artifact(
        artifact_model_to_ref(final.doubletScores[0])
    ).inputs
    assert doublet_inputs is not None
    assert (
        doublet_inputs["connectivity_map"]
        == artifact_model_to_ref(final.graph).to_dict()
    )
    scored_partition = ArtifactRef.from_dict(doublet_inputs["clusters"])
    assert scored_partition.kind == "cluster_labels"
    assert persisted.inspect_artifact(scored_partition).complete
    selected_evaluation = tuning_evaluations[0]
    assert (
        persisted.inspect_artifact(
            artifact_model_to_ref(selected_evaluation.artifacts["normalized"])
        ).inputs["cell_selection"]
        == artifact_model_to_ref(final.cellSelection).to_dict()
    )
    assert len(tuning_evaluations) == 1
    snapshot = analysis_snapshot(persisted, result.workflowRunId)
    evidence = next(
        stage for stage in snapshot["stages"] if stage["stage"] == "parameter_tuning"
    )["outputs"]["tuningEvidence"]
    assert 4 < evidence["budget"]["scopes"]["sample0"]["reserved"]["partitions"] <= 24
    compared = {
        row["comparisonId"]
        for review in snapshot["analysisReviews"]
        for row in review["comparisonCoverage"]["comparisons"]
    }
    assert {
        "hvgCount:2000",
        "hvgCount:4000",
        "dimensions:10",
        "dimensions:30",
        "neighborsK:21",
        "neighborsK:41",
        "hvgRanking",
        "featurePolicy",
    } <= compared
    assert evidence["budget"]["scopes"]["full"]["reserved"]["graphs"] == 1
    assert evidence["budget"]["scopes"]["full"]["reserved"]["partitions"] == 1
    sample_record = load_checkpoint(
        persisted,
        _ensure_orchestration_store(persisted),
        result.workflowRunId,
        "parameter_tuning/sample0/evaluation0/complete",
        inputs=None,
    )
    assert sample_record is not None
    sample_evaluation = ParameterCandidateEvaluation.model_validate(
        sample_record["evaluation"]
    )
    assert (
        sample_evaluation.parameters.leidenResolution
        == selected_evaluation.parameters.leidenResolution
        == 0.5
    )
    assert sample_evaluation.cellSelection != final.cellSelection

    def selected_rows(reference: Any) -> np.ndarray:
        return read_stored_selection_indices(
            persisted.zw,
            artifact_model_to_ref(reference),
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )

    sample_rows = selected_rows(sample_evaluation.cellSelection)
    full_rows = selected_rows(final.cellSelection)
    sample_labels = np.asarray(
        persisted.load_artifact(
            artifact_model_to_ref(sample_evaluation.artifacts["clusters"])
        )["values"][:]
    )
    full_labels = np.asarray(
        persisted.load_artifact(artifact_model_to_ref(final.clusters))["values"][:]
    )
    from sklearn.metrics import adjusted_rand_score

    assert adjusted_rand_score(
        sample_labels, full_labels[np.searchsorted(full_rows, sample_rows)]
    ) == pytest.approx(1.0)
    markers = result.get_markers(min_score=0.0, min_frac_exp=0.0)
    assert {"CD3D", "MS4A1"}.issubset(set(markers.feature_name))
    plotted = result.plot_embedding(show=False)
    assert plotted.figure is not None
    plotted.close()
    requests_before = state["requests"]
    diagnostics_before = list(pca_diagnostic_calls)
    completed = orchestrator.run(request)
    assert completed.status == "completed"
    assert completed.workflowRunId == result.workflowRunId
    assert analysis_snapshot(persisted, result.workflowRunId)[
        "finalAnalysis"
    ] == final.model_dump(mode="json")
    assert state["requests"] == requests_before
    assert pca_diagnostic_calls == diagnostics_before
    assert "pipeline" not in persisted.zw
