"""Focused contracts for the persisted automated agent orchestrator."""

import hashlib
import json
import re
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import zarr
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

import scarf.agent as agent_module
import scarf.agent.orchestrator as orchestrator_module
import scarf.agent.orchestrator.context as context_module
import scarf.agent.orchestrator.journal as journal_module
import scarf.agent.orchestrator.main as main_module
import scarf.agent.orchestrator.tuning as tuning_module
import scarf.agent.parameter_tuning as parameter_tuning_module
from scarf.agent.biological_interpretation import (
    BiologicalInterpretationReport,
    ClusterCompositionEvidence,
    ClusterInterpretation,
    ClusterMarkerBatchEvidence,
)
from scarf.agent.config import AgentRunConfig
from scarf.agent.data_enrichment import (
    AssayFeatureInspection,
    DataEnrichmentReport,
    FeatureFamilyEvidence,
    FeatureReference,
    FeatureSelectionPolicy,
    StudyContextSummary,
)
from scarf.agent.experimental_context import (
    BatchCorrectionPlan,
    CellQcPlan,
    ExperimentalContextDecision,
    ExperimentalContextResult,
)
from scarf.agent.ingest import IngestResult
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
)
from scarf.agent.orchestrator.models import (
    OrchestrationRequestRecord,
    OrchestrationResumeRecord,
)
from scarf.agent.persistence import (
    AgentInvocation,
    AgentReportReference,
    create_agent_workflow,
    finalize_agent_workflow,
    list_agent_reports,
    load_agent_record,
    load_agent_report,
    load_agent_workflow,
    save_agent_report,
)
from scarf.agent.parameter_tuning import (
    ArtifactRecord,
    FinalGraphNeedsInput,
    FinalGraphSelection,
    IntegrationCandidateEvaluation,
    IntegrationMetrics,
    ParameterCandidateEvaluation,
    ParameterTuningReport,
    final_graph_options,
    finalize_parameter_tuning_selection,
    select_final_parameter_graph,
)
from scarf.agent.types import (
    AgentRunInfo,
    ArtifactReferenceModel,
    BatchSafetyEvidence,
    ExperimentalTuningHandoff,
)
from scarf.datastore.datastore import DataStore
from scarf.storage.budget import ResourceBudget
from scarf.storage.refs import ArtifactRef
from scarf.storage.schema import create_cell_data, create_zarr_count_assay
from scarf.storage.sharding import write_counts_t
from tests.test_agent_ingest import _write_h5ad


_PLAN_CHECKSUM = "a" * 64


def _create_store(path: Path, *, workspace: str | None = None) -> Path:
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    values = np.asarray(
        [
            [4, 0, 1, 0],
            [0, 3, 0, 2],
            [2, 1, 0, 0],
            [0, 0, 5, 1],
        ],
        dtype=np.uint32,
    )
    cell_ids = np.asarray([f"cell-{index}" for index in range(values.shape[0])])
    feature_ids = np.asarray([f"feature-{index}" for index in range(values.shape[1])])
    feature_names = np.asarray(["MT-CO1", "RPS3", "GENE1", "GENE2"])
    create_cell_data(
        root,
        workspace,
        ids=cell_ids,
        names=cell_ids,
        profile="fast_local",
    )
    counts = create_zarr_count_assay(
        root,
        "RNA",
        workspace,
        values.shape[0],
        feat_ids=feature_ids,
        feat_names=feature_names,
        dtype="uint32",
        profile="fast_local",
    )
    counts[:] = values
    count_group = root["RNA"] if workspace is None else root["matrices/RNA"]
    write_counts_t(
        counts,
        count_group,
        resources=ResourceBudget(1024**3, 2),
    )
    active = root if workspace is None else root[workspace]
    active.attrs["assayTypes"] = {"RNA": "RNA"}
    active["RNA"].attrs["dataset_fingerprint"] = "dataset-rna"
    return path


def _record_checksum(payload: dict[str, Any]) -> str:
    content = dict(payload)
    content.pop("contentSha256", None)
    encoded = json.dumps(
        content,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_record_checksum(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    assert payload["contentSha256"] == _record_checksum(payload)
    return payload


def _mock_ingest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_module, "detect_format", lambda _path: "zarr")

    def fake_ingest(
        path: str,
        *,
        zarrPath: str | None,
        model: Any,
        directions: Mapping[str, Any],
    ) -> IngestResult:
        del model, directions
        resolved = str(Path(zarrPath or path).resolve())
        return IngestResult(
            status="done",
            format="zarr",
            zarrPath=resolved,
            assayNames=["RNA"],
            summary={"assays": ["RNA"]},
            actions=["summarize_zarr"],
        )

    monkeypatch.setattr(main_module, "ingest", fake_ingest)


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
            profile_id = "cellQc:RNA:RNA:I:skip"
            evidence_id = f"qcProfile:{profile_id}"
            decision = ExperimentalContextDecision(
                batchCorrection=BatchCorrectionPlan(
                    action="skip",
                    rationale="No trusted technical batch column was supplied.",
                    evidenceIds=[evidence_id],
                ),
                cellQc=CellQcPlan(
                    action="skip",
                    profileId=profile_id,
                    driverAssay="RNA",
                    driverAssayType="RNA",
                    cellKey="I",
                    rationale="Keep every cell in this deterministic fixture.",
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


class _FeatureTable:
    def __init__(self, ids: list[str], names: list[str]) -> None:
        self._values = {
            "ids": np.asarray(ids),
            "names": np.asarray(names),
        }

    def fetch_all(self, column: str) -> np.ndarray:
        return self._values[column]


class _PlanningStore:
    """Narrow datastore surface consumed by preprocessing-plan construction."""

    def __init__(
        self,
        assays: Mapping[str, tuple[str, list[str], list[str]]],
        *,
        active_cells: int = 100,
    ) -> None:
        self.assay_names = list(assays)
        self._assays = {
            name: SimpleNamespace(feats=_FeatureTable(ids, names))
            for name, (_assay_type, ids, names) in assays.items()
        }
        self._summary = SimpleNamespace(
            active_cells=active_cells,
            assays=[
                SimpleNamespace(
                    name=name,
                    assay_type=assay_type,
                    total_features=len(ids),
                )
                for name, (assay_type, ids, _names) in assays.items()
            ],
        )

    def summary(self) -> Any:
        return self._summary

    def get_assay(self, name: str) -> Any:
        return self._assays[name]


def _modality_policy(
    assay: str,
    assay_type: str,
    *,
    controls: list[FeatureReference] | None = None,
    exclude_features: list[str] | None = None,
    artificial_features: list[str] | None = None,
    peak_status: str = "notApplicable",
) -> FeatureSelectionPolicy:
    supported = assay_type in {"RNA", "ATAC", "ADT"}
    modality = (
        assay_type if assay_type in {"RNA", "ATAC", "ADT", "HTO"} else "unsupported"
    )
    return FeatureSelectionPolicy(
        assay=assay,
        assayType=assay_type,
        assayModality=modality,
        graphEligible=supported,
        markerEligible=supported,
        demultiplexEligible=assay_type == "HTO",
        exactControlFeatures=controls or [],
        excludeFeatures=exclude_features or [],
        artificialFeatures=artificial_features or [],
        peakCoordinateStatus=peak_status,
        evidenceIds=[f"assay:{assay}:modality"],
    )


def _planning_inputs(
    assays: Mapping[str, tuple[str, list[str], list[str]]],
    *,
    controls: Mapping[str, list[FeatureReference]] | None = None,
    exclude_features: Mapping[str, list[str]] | None = None,
    artificial_features: Mapping[str, list[str]] | None = None,
    peak_statuses: Mapping[str, str] | None = None,
    primary_assay: str | None = None,
    marker_assay: str | None = None,
    analysis_assays: list[str] | None = None,
    config: AutomatedWorkflowConfig | None = None,
) -> tuple[
    _PlanningStore,
    OrchestrationRequestRecord,
    DataEnrichmentReport,
    ExperimentalContextResult,
    WorkflowStageAttempt,
]:
    store = _PlanningStore(assays)
    policies = [
        _modality_policy(
            name,
            assay_type,
            controls=(controls or {}).get(name),
            exclude_features=(exclude_features or {}).get(name),
            artificial_features=(artificial_features or {}).get(name),
            peak_status=(peak_statuses or {}).get(name, "notApplicable"),
        )
        for name, (assay_type, _ids, _names) in assays.items()
    ]
    enrichment = DataEnrichmentReport(status="done", policies=policies)
    request = AutomatedWorkflowRequest(
        sourcePath="dataset.zarr",
        zarrPath="dataset.zarr",
        studyContext="A bounded plan-construction test.",
        allowAssumptions=True,
        primaryAssay=primary_assay,
        markerAssay=marker_assay,
        analysisAssays=analysis_assays or list(assays),
    )
    request_record = OrchestrationRequestRecord(
        workflowRunId="planning-test",
        request=request,
        config=config or AutomatedWorkflowConfig(),
    )
    experimental = ExperimentalContextResult.get_example()
    ingest_outcome = WorkflowStageAttempt(
        workflowRunId="planning-test",
        stage="ingest",
        attemptId="ingest-test",
        status="done",
        startedAtNs=1,
        completedAtNs=2,
        outputs={"format": "zarr"},
    )
    return store, request_record, enrichment, experimental, ingest_outcome


def _build_plan(
    assays: Mapping[str, tuple[str, list[str], list[str]]],
    **kwargs: Any,
) -> AutomatedPreprocessingPlan:
    inputs = _planning_inputs(assays, **kwargs)
    return AgentOrchestrator(object())._build_preprocessing_plan(*inputs)


def _native_assay_report(assay: str, token: int) -> ParameterTuningReport:
    evaluation = ParameterCandidateEvaluation.get_example().model_copy(
        update={
            "artifacts": {
                "connectivityMap": ArtifactRecord(
                    assay=assay,
                    kind="connectivity_map",
                    artifactId=f"{token:064x}",
                ),
                "clusters": ArtifactRecord(
                    assay=assay,
                    kind="cluster_labels",
                    artifactId=f"{token + 1:064x}",
                ),
            },
            "clusterColumn": f"{assay}_agent_clusters",
            "evidenceIds": ["candidate:baseline:clusters"],
        }
    )
    return ParameterTuningReport(
        status="done",
        fromAssay=assay,
        evaluations=[evaluation],
        recommendedCandidateId=evaluation.candidateId,
        selectedArtifacts=dict(evaluation.artifacts),
        evidenceIds=list(evaluation.evidenceIds),
        stopReason="The bounded screen completed.",
    )


def _native_batch_report(*assays: str) -> ParameterTuningReport:
    reports = {
        assay: _native_assay_report(assay, index * 10 + 1)
        for index, assay in enumerate(assays)
    }
    primary = reports[assays[0]]
    return ParameterTuningReport(
        status="done",
        fromAssay=assays[0],
        cellKey="I",
        evaluations=list(primary.evaluations),
        recommendedCandidateId=primary.recommendedCandidateId,
        selectedArtifacts=dict(primary.selectedArtifacts),
        assayReports=reports,
        recommendedByAssay={
            assay: report.recommendedCandidateId or ""
            for assay, report in reports.items()
        },
        totalCandidates=sum(len(report.evaluations) for report in reports.values()),
        graphAssay=assays[0],
        markerAssay=assays[0],
        runInfo=AgentRunInfo(
            agentName="parameter_tuning",
            runId=uuid.uuid4().hex,
        ),
    )


def _eligible_integration() -> IntegrationCandidateEvaluation:
    return IntegrationCandidateEvaluation(
        integrationId="wnn_resolution_1",
        method="wnn",
        assays=["RNA", "ADT"],
        status="done",
        eligible=True,
        resolution=1.0,
        graphArtifact=ArtifactRecord(
            scope="datastore",
            kind="integrated_graph",
            artifactId="8" * 64,
        ),
        clusterArtifact=ArtifactRecord(
            scope="datastore",
            kind="cluster_labels",
            artifactId="9" * 64,
        ),
        clusterColumn="agent_wnn_cluster",
        metrics=IntegrationMetrics(
            nClusters=2,
            minClusterCells=20,
            modalityWeightsValid=True,
        ),
        evidenceIds=["integration:wnn_resolution_1:clusters"],
    )


class _CheckpointOrchestrator(AgentOrchestrator):
    """Minimal deterministic stage machine exercising persistence and resume."""

    def __init__(self) -> None:
        super().__init__(object())
        self.enrichmentExecutions = 0

    def _continue(
        self,
        store: Any,
        workflow: Any,
        request_record: Any,
        *,
        answers: Mapping[str, Any],
        resume_record: OrchestrationResumeRecord | None = None,
    ) -> AutomatedWorkflowResult:
        prefix = journal_module._ensure_orchestration_store(store)
        ingest = journal_module._validated_done_outcome(
            store,
            prefix,
            workflow.workflowRunId,
            "ingest",
            request_record,
            [],
        )
        assert ingest is not None
        parents = [journal_module._parent_link(ingest)]

        enrichment = journal_module._validated_done_outcome(
            store,
            prefix,
            workflow.workflowRunId,
            "data_enrichment",
            request_record,
            parents,
        )
        if enrichment is None:
            self.enrichmentExecutions += 1
            report = DataEnrichmentReport.get_example()
            report = report.model_copy(
                update={
                    "runInfo": report.runInfo.model_copy(
                        update={"runId": uuid.uuid4().hex}
                    )
                }
            )
            reference = save_agent_report(
                store,
                workflow.workflowRunId,
                report,
                invocation=AgentInvocation(
                    agentName="data_enrichment",
                    inputs={"studyContext": request_record.request.studyContext},
                ),
            )
            started = journal_module._start_attempt(
                store.zw,
                prefix,
                workflow.workflowRunId,
                "data_enrichment",
                request_record,
                parents,
                inputs={"studyContext": request_record.request.studyContext},
                resume_record=resume_record,
            )
            enrichment = journal_module._complete_attempt(
                started,
                status="done",
                report_references=[reference],
            )
            journal_module._save_outcome(store.zw, prefix, enrichment)
        parents = [journal_module._parent_link(enrichment)]

        approved = answers.get("approvePlanChecksum") == _PLAN_CHECKSUM
        completed_plan = journal_module._validated_done_outcome(
            store,
            prefix,
            workflow.workflowRunId,
            "preprocessing_plan",
            request_record,
            parents,
        )
        if completed_plan is None:
            started = journal_module._start_attempt(
                store.zw,
                prefix,
                workflow.workflowRunId,
                "preprocessing_plan",
                request_record,
                parents,
                inputs={"planChecksum": _PLAN_CHECKSUM},
                resume_record=resume_record,
            )
            if not approved:
                paused = journal_module._complete_attempt(
                    started,
                    status="needsInput",
                    needs_input=WorkflowNeedsInput(
                        questions=[
                            WorkflowQuestion(
                                questionId="approvePlanChecksum",
                                question="Approve the preprocessing plan?",
                                planChecksum=_PLAN_CHECKSUM,
                            )
                        ]
                    ),
                )
                journal_module._save_outcome(store.zw, prefix, paused)
                return journal_module.paused_or_failed_result(
                    store,
                    workflow,
                    request_record,
                    paused,
                )
            completed_plan = journal_module._complete_attempt(
                started,
                status="done",
                outputs={"planChecksum": _PLAN_CHECKSUM},
                actions=["approve_preprocessing_plan"],
            )
            journal_module._save_outcome(store.zw, prefix, completed_plan)

        terminal = finalize_agent_workflow(
            store,
            workflow.workflowRunId,
            status="completed",
            message="Checkpoint workflow completed",
        )
        result = AutomatedWorkflowResult(
            status="completed",
            currentStage="preprocessing_plan",
            zarrPath=str(Path(store.zarr_loc).resolve()),
            workflowRun=terminal,
            reportReferences=journal_module.all_report_references(
                store,
                prefix,
                workflow.workflowRunId,
            ),
        )
        result = result.model_copy(
            update={"contentSha256": journal_module._record_checksum(result)}
        )
        journal_module._write_model_once(
            store.zw,
            journal_module._result_key(prefix, workflow.workflowRunId),
            result,
        )
        return result


def _start_paused_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    workspace: str | None = None,
) -> tuple[_CheckpointOrchestrator, AutomatedWorkflowResult, Path]:
    path = _create_store(tmp_path / "data.zarr", workspace=workspace)
    _mock_ingest(monkeypatch)
    orchestrator = _CheckpointOrchestrator()
    result = orchestrator.run(
        AutomatedWorkflowRequest(
            sourcePath=str(path),
            zarrPath=str(path),
            studyContext="A deterministic test study.",
            workspace=workspace,
        )
    )
    assert result.status == "needsInput"
    assert result.workflowRun is not None
    return orchestrator, result, path


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
    assert state["requests"] == 10
    assert [reference.agentName for reference in result.reportReferences] == [
        "data_enrichment",
        "experimental_context",
        "parameter_tuning",
        "biological_interpretation",
    ]
    assert result.finalAnalysis is not None
    final = result.finalAnalysis
    assert final.graphMethod == "native"
    assert final.primaryAssay == final.markerAssay == "RNA"
    assert final.graph is not None
    assert final.clusters is not None
    assert final.markers is not None
    assert final.clusterColumn
    assert len(final.umapColumns) == 2

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
        "clusters",
        "markers",
        "markerFeatures",
    }


def test_orchestration_records_are_plain_json_with_valid_checksums(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _orchestrator, result, path = _start_paused_workflow(tmp_path, monkeypatch)
    workflow_id = result.workflowRun.workflowRunId
    agents = path / "agents"
    orchestration = agents / "orchestrations"
    workflow_path = orchestration / workflow_id

    assert (agents / "zarr.json").is_file()
    assert (agents / "store.json").is_file()
    assert (orchestration / "zarr.json").is_file()
    assert not (workflow_path / "zarr.json").exists()
    assert not list(orchestration.rglob("c"))

    request = _assert_record_checksum(workflow_path / "request.json")
    assert request["workflowRunId"] == workflow_id
    records = sorted((workflow_path / "stages").rglob("*.json"))
    assert records
    for record_path in records:
        _assert_record_checksum(record_path)
        assert record_path.name in {"started.json", "outcome.json"}


def test_approval_resume_reuses_completed_stages_and_persists_answer_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator, paused, path = _start_paused_workflow(tmp_path, monkeypatch)
    workflow_id = paused.workflowRun.workflowRunId
    workflow_path = path / "agents" / "orchestrations" / workflow_id
    paused_outcome_path = next(
        (workflow_path / "stages" / "preprocessing_plan").glob("*/outcome.json")
    )
    paused_outcome = json.loads(paused_outcome_path.read_text())

    completed = orchestrator.resume(
        AutomatedWorkflowResumeRequest(
            zarrPath=str(path),
            workflowRunId=workflow_id,
            answers={"approvePlanChecksum": _PLAN_CHECKSUM},
        )
    )

    assert completed.status == "completed"
    assert completed.workflowRun is not None
    assert completed.workflowRun.status == "completed"
    assert orchestrator.enrichmentExecutions == 1
    assert len(list((workflow_path / "stages" / "ingest").glob("*/outcome.json"))) == 1
    assert (
        len(list((workflow_path / "stages" / "data_enrichment").glob("*/outcome.json")))
        == 1
    )
    assert (
        len(
            list(
                (workflow_path / "stages" / "preprocessing_plan").glob("*/outcome.json")
            )
        )
        == 2
    )

    resume_path = next((workflow_path / "resumes").glob("*.json"))
    resume = _assert_record_checksum(resume_path)
    assert resume["answeredAttempt"]["attemptId"] == paused_outcome["attemptId"]
    assert resume["questionIds"] == ["approvePlanChecksum"]
    assert resume["answers"] == {"approvePlanChecksum": _PLAN_CHECKSUM}
    resumed_start = next(
        payload
        for payload in (
            json.loads(path.read_text())
            for path in (workflow_path / "stages" / "preprocessing_plan").glob(
                "*/started.json"
            )
        )
        if "resumeLineage" in payload["inputs"]
    )
    assert resumed_start["inputs"]["resumeLineage"] == {
        "resumeId": resume["resumeId"],
        "answeredAttempt": resume["answeredAttempt"],
        "questionIds": ["approvePlanChecksum"],
    }

    persisted = _assert_record_checksum(workflow_path / "result.json")
    assert persisted["status"] == "completed"
    assert persisted["workflowRun"]["status"] == "completed"
    assert load_agent_workflow(path, workflow_id).status == "completed"


def test_resume_does_not_mutate_constructor_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator, paused, path = _start_paused_workflow(tmp_path, monkeypatch)
    assert paused.workflowRun is not None
    constructor_config = AutomatedWorkflowConfig(primaryInitialCandidates=2)
    orchestrator.config = constructor_config

    completed = orchestrator.resume(
        AutomatedWorkflowResumeRequest(
            zarrPath=str(path),
            workflowRunId=paused.workflowRun.workflowRunId,
            answers={"approvePlanChecksum": _PLAN_CHECKSUM},
        )
    )

    assert completed.status == "completed"
    assert orchestrator.config == constructor_config


@pytest.mark.parametrize(
    ("answers", "expected_note"),
    [
        ({}, "Missing resume answer"),
        ({"approvePlanChecksum": "approve"}, "persisted plan checksum"),
        (
            {
                "approvePlanChecksum": _PLAN_CHECKSUM,
                "unexpectedAnswer": "ignored",
            },
            "Unknown resume answer key",
        ),
    ],
)
def test_invalid_resume_answers_retain_the_persisted_pause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    answers: dict[str, Any],
    expected_note: str,
) -> None:
    orchestrator, paused, path = _start_paused_workflow(tmp_path, monkeypatch)
    assert paused.workflowRun is not None
    workflow_id = paused.workflowRun.workflowRunId

    result = orchestrator.resume(
        AutomatedWorkflowResumeRequest(
            zarrPath=str(path),
            workflowRunId=workflow_id,
            answers=answers,
        )
    )

    assert result.status == "needsInput"
    assert result.currentStage == "preprocessing_plan"
    assert result.needsInput == paused.needsInput
    assert result.workflowRun is not None
    assert result.workflowRun.status == "running"
    assert any(expected_note in note for note in result.notes)
    workflow_path = path / "agents" / "orchestrations" / workflow_id
    assert (
        len(
            list(
                (workflow_path / "stages" / "preprocessing_plan").glob("*/outcome.json")
            )
        )
        == 1
    )
    assert len(list((workflow_path / "resumes").glob("*.json"))) == 1


def test_interrupted_attempt_supersedes_a_historical_pause_for_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator, paused, path = _start_paused_workflow(tmp_path, monkeypatch)
    assert paused.workflowRun is not None
    workflow_id = paused.workflowRun.workflowRunId
    record, store = orchestrator._load_request_for_resume(
        AutomatedWorkflowResumeRequest(
            zarrPath=str(path),
            workflowRunId=workflow_id,
        )
    )
    prefix = journal_module._ensure_orchestration_store(store)
    journal_module._start_attempt(
        store.zw,
        prefix,
        workflow_id,
        "preprocessing",
        record,
        [],
        inputs={"simulatedCrash": True},
    )

    with pytest.raises(ValueError, match="no active persisted questions"):
        orchestrator.resume(
            AutomatedWorkflowResumeRequest(
                zarrPath=str(path),
                workflowRunId=workflow_id,
                answers={"approvePlanChecksum": _PLAN_CHECKSUM},
            )
        )

    resumed = orchestrator.resume(
        AutomatedWorkflowResumeRequest(
            zarrPath=str(path),
            workflowRunId=workflow_id,
        )
    )

    assert resumed.status == "needsInput"
    assert not any("Missing resume answer" in note for note in resumed.notes)
    workflow_path = path / "agents" / "orchestrations" / workflow_id
    assert (
        len(
            list(
                (workflow_path / "stages" / "preprocessing_plan").glob("*/outcome.json")
            )
        )
        == 2
    )


def test_interrupted_answered_attempt_inherits_exact_persisted_answers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator, paused, path = _start_paused_workflow(tmp_path, monkeypatch)
    assert paused.workflowRun is not None
    workflow_id = paused.workflowRun.workflowRunId
    record, store = orchestrator._load_request_for_resume(
        AutomatedWorkflowResumeRequest(
            zarrPath=str(path),
            workflowRunId=workflow_id,
        )
    )
    prefix = journal_module._ensure_orchestration_store(store)
    paused_outcome = journal_module._stage_outcomes(
        store.zw,
        prefix,
        workflow_id,
        "preprocessing_plan",
    )[0]
    first_resume = OrchestrationResumeRecord(
        workflowRunId=workflow_id,
        resumeId="resume-before-crash",
        createdAtNs=paused_outcome.completedAtNs + 1,
        answeredAttempt=journal_module._parent_link(paused_outcome),
        questionIds=["approvePlanChecksum"],
        answers={"approvePlanChecksum": _PLAN_CHECKSUM},
    )
    first_resume = first_resume.model_copy(
        update={"contentSha256": journal_module._record_checksum(first_resume)}
    )
    journal_module._write_model_once(
        store.zw,
        journal_module._resume_key(
            prefix,
            workflow_id,
            first_resume.resumeId,
        ),
        first_resume,
    )
    journal_module._start_attempt(
        store.zw,
        prefix,
        workflow_id,
        "preprocessing_plan",
        record,
        paused_outcome.parentAttempts,
        inputs={"planChecksum": _PLAN_CHECKSUM},
        resume_record=first_resume,
    )

    with pytest.raises(ValueError, match="Cannot change answers"):
        orchestrator.resume(
            AutomatedWorkflowResumeRequest(
                zarrPath=str(path),
                workflowRunId=workflow_id,
                answers={"approvePlanChecksum": "b" * 64},
            )
        )

    completed = orchestrator.resume(
        AutomatedWorkflowResumeRequest(
            zarrPath=str(path),
            workflowRunId=workflow_id,
        )
    )

    assert completed.status == "completed"
    resume_records = sorted(
        (path / "agents" / "orchestrations" / workflow_id / "resumes").glob("*.json")
    )
    assert len(resume_records) == 2
    inherited = next(
        json.loads(value.read_text())
        for value in resume_records
        if value.stem != first_resume.resumeId
    )
    assert inherited["answers"] == first_resume.answers
    assert inherited["answeredAttempt"] == first_resume.answeredAttempt.model_dump(
        mode="json"
    )
    assert inherited["questionIds"] == first_resume.questionIds


def test_unsafe_experimental_context_pauses_and_explicit_skip_reuses_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _create_store(tmp_path / "unsafe-context.zarr")
    store = DataStore(str(path), default_assay="RNA", min_features_per_cell=0)
    workflow = create_agent_workflow(store, workflow_run_id="unsafe-context")
    enrichment = DataEnrichmentReport.get_example().model_copy(
        update={
            "runInfo": AgentRunInfo(
                agentName="data_enrichment",
                runId=uuid.uuid4().hex,
            )
        }
    )
    enrichment_reference = save_agent_report(
        store,
        workflow.workflowRunId,
        enrichment,
        invocation=AgentInvocation(
            agentName="data_enrichment",
            inputs={"studyContext": "A deliberately confounded study."},
        ),
    )
    request_record = OrchestrationRequestRecord(
        workflowRunId=workflow.workflowRunId,
        request=AutomatedWorkflowRequest(
            sourcePath=str(path),
            zarrPath=str(path),
            studyContext="Treatment is confounded with batch.",
            allowAssumptions=True,
        ),
    )
    example = ExperimentalContextResult.get_example()
    evidence_id = "batchEstimability:treatment:batch"
    unsafe_plan = example.decision.batchCorrection.model_copy(
        update={
            "action": "unsafe",
            "evidenceIds": [evidence_id],
        }
    )
    unsafe_report = example.model_copy(
        update={
            "decision": example.decision.model_copy(
                update={"batchCorrection": unsafe_plan}
            ),
            "batchSafety": [
                BatchSafetyEvidence.get_example().model_copy(
                    update={
                        "status": "unsafe",
                        "estimability": {
                            "status": "ok",
                            "coefficientEstimable": False,
                            "rankDeficient": True,
                        },
                    }
                )
            ],
            "runInfo": AgentRunInfo(
                agentName="experimental_context",
                runId=uuid.uuid4().hex,
            ),
        }
    )

    class UnsafeAgent:
        calls = 0

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.config = AgentRunConfig()

        def run(self, *_args: Any, **_kwargs: Any) -> ExperimentalContextResult:
            type(self).calls += 1
            return unsafe_report

    monkeypatch.setattr(context_module, "ExperimentalContextAgent", UnsafeAgent)
    orchestrator = AgentOrchestrator(object())
    paused_outcome, paused_report = orchestrator._experimental_context_stage(
        store,
        workflow,
        request_record,
        [],
        enrichment_reference,
        [],
        {},
    )

    assert paused_report.status == "done"
    assert paused_outcome.status == "needsInput"
    assert paused_outcome.outputs["unsafeBatchCorrection"] is True
    assert paused_outcome.needsInput is not None
    assert paused_outcome.needsInput.questions[0].options == [
        "skipHarmony",
        "provideClarification",
    ]
    assert journal_module._resume_answer_errors(
        paused_outcome,
        {"experimentalDirections": "unsafe"},
    )
    assert not journal_module._resume_answer_errors(
        paused_outcome,
        {"experimentalDirections": "skipHarmony"},
    )
    assert not journal_module._resume_answer_errors(
        paused_outcome,
        {
            "experimentalDirections": {
                "selection": "provideClarification",
                "clarification": "Batch denotes sequencing lane within each donor.",
            }
        },
    )

    done_outcome, resolved_report = orchestrator._experimental_context_stage(
        store,
        workflow,
        request_record,
        [],
        enrichment_reference,
        [],
        {"experimentalDirections": "skipHarmony"},
    )

    assert done_outcome.status == "done"
    assert done_outcome.actions == ["resolve_unsafe_batch_correction:skip"]
    assert resolved_report.decision.batchCorrection.action == "skip"
    assert resolved_report.decision.batchCorrection.batchColumns == []
    assert resolved_report.decision.batchCorrection.preserveColumns == (
        unsafe_plan.preserveColumns
    )
    assert UnsafeAgent.calls == 1


def test_resume_rejects_changed_dataset_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator, paused, path = _start_paused_workflow(tmp_path, monkeypatch)
    workflow_id = paused.workflowRun.workflowRunId
    root = zarr.open_group(str(path), mode="r+")
    root["RNA"].attrs["dataset_fingerprint"] = "changed-dataset"

    with pytest.raises(ValueError, match="dataset fingerprints"):
        orchestrator.resume(
            AutomatedWorkflowResumeRequest(
                zarrPath=str(path),
                workflowRunId=workflow_id,
                answers={"approvePlanChecksum": _PLAN_CHECKSUM},
            )
        )


def test_resume_rejects_request_envelope_and_destination_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator, paused, path = _start_paused_workflow(tmp_path, monkeypatch)
    workflow_id = paused.workflowRun.workflowRunId
    request_path = path / "agents" / "orchestrations" / workflow_id / "request.json"
    payload = json.loads(request_path.read_text())
    payload["request"]["studyContext"] = "Tampered study context"
    request_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="request checksum"):
        orchestrator.resume(
            AutomatedWorkflowResumeRequest(
                zarrPath=str(path),
                workflowRunId=workflow_id,
            )
        )

    copied = tmp_path / "copied.zarr"
    shutil.copytree(path, copied)
    with pytest.raises(ValueError, match="zarrPath"):
        orchestrator.resume(
            AutomatedWorkflowResumeRequest(
                zarrPath=str(copied),
                workflowRunId=workflow_id,
            )
        )


def test_workspace_records_are_isolated_and_wrong_workspace_cannot_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator, paused, path = _start_paused_workflow(
        tmp_path,
        monkeypatch,
        workspace="workspace_a",
    )
    workflow_id = paused.workflowRun.workflowRunId
    root = zarr.open_group(str(path), mode="r+")
    root.create_group("workspace_b")

    assert (
        path
        / "workspace_a"
        / "agents"
        / "orchestrations"
        / workflow_id
        / "request.json"
    ).is_file()
    assert not (path / "agents").exists()
    assert not (path / "workspace_b" / "agents").exists()

    with pytest.raises(FileNotFoundError):
        orchestrator.resume(
            AutomatedWorkflowResumeRequest(
                zarrPath=str(path),
                workflowRunId=workflow_id,
                workspace="workspace_b",
            )
        )


def test_cancel_finalizes_abandoned_and_prevents_future_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator, paused, path = _start_paused_workflow(tmp_path, monkeypatch)
    workflow_id = paused.workflowRun.workflowRunId
    request = AutomatedWorkflowResumeRequest(
        zarrPath=str(path),
        workflowRunId=workflow_id,
    )

    cancelled = orchestrator.cancel(request, message="Stop this test workflow")

    assert cancelled.status == "abandoned"
    assert cancelled.workflowRun is not None
    assert cancelled.workflowRun.status == "abandoned"
    assert load_agent_workflow(path, workflow_id).status == "abandoned"
    result_path = path / "agents" / "orchestrations" / workflow_id / "result.json"
    persisted = _assert_record_checksum(result_path)
    assert persisted["status"] == "abandoned"
    assert persisted["notes"] == ["Stop this test workflow"]
    with pytest.raises(RuntimeError, match="Cannot resume"):
        orchestrator.resume(request)

    result_path.unlink()
    repaired = orchestrator.resume(request)
    assert repaired.status == "abandoned"
    assert repaired.workflowRun is not None
    assert repaired.workflowRun.status == "abandoned"
    _assert_record_checksum(result_path)
    with pytest.raises(RuntimeError, match="Cannot resume"):
        orchestrator.resume(request)


def test_preprocessing_plan_routes_supported_modalities_and_skips_others() -> None:
    assays = {
        "peaks": (
            "ATAC",
            ["chr1:1-10", "chr1:20-30", "chr2:1-20"],
            ["peak-1", "peak-2", "peak-3"],
        ),
        "tags": ("HTO", ["tag-1", "tag-2"], ["sample-1", "sample-2"]),
        "proteins": (
            "ADT",
            ["adt-1", "adt-2", "adt-3"],
            ["CD3", "CD19", "CD45"],
        ),
        "custom": ("CRISPR", ["guide-1"], ["guide-1"]),
        "transcriptome": (
            "RNA",
            ["gene-1", "gene-2", "gene-3", "gene-4"],
            ["A", "B", "C", "D"],
        ),
    }

    plan = _build_plan(assays)
    routes = {value.assay: value for value in plan.assays}

    assert (plan.primaryAssay, plan.markerAssay, plan.cellKey) == (
        "transcriptome",
        "transcriptome",
        "I",
    )
    assert (
        routes["transcriptome"].role,
        routes["transcriptome"].featureMethod,
        routes["transcriptome"].reductionMethod,
    ) == ("graph", "hvg", "pca")
    assert (
        routes["peaks"].role,
        routes["peaks"].featureMethod,
        routes["peaks"].reductionMethod,
        routes["peaks"].reductionParameters["skipFirst"],
    ) == ("graph", "prevalentPeaks", "lsi", True)
    assert (
        routes["proteins"].role,
        routes["proteins"].featureMethod,
        routes["proteins"].reductionMethod,
    ) == ("graph", "panel", "identity")
    assert routes["tags"].role == "hto"
    assert not routes["tags"].graphEligible
    assert not routes["tags"].markerEligible
    assert routes["tags"].reductionMethod == "none"
    assert routes["tags"].normalizationParameters == {}
    assert routes["custom"].role == "unsupported"
    assert not routes["custom"].graphEligible
    assert any("Unsupported assay 'custom'" in value for value in plan.limitations)


def test_converted_input_always_resets_selection_and_persists_typed_qc() -> None:
    inputs = list(
        _planning_inputs(
            {
                "RNA": (
                    "RNA",
                    ["gene-1", "gene-2", "gene-3"],
                    ["A", "B", "C"],
                )
            }
        )
    )
    request_record = inputs[1]
    request = request_record.request.model_copy(
        update={
            "sourcePath": "dataset.h5ad",
            "resetCellSelection": False,
        }
    )
    inputs[1] = request_record.model_copy(update={"request": request})
    inputs[4] = inputs[4].model_copy(update={"outputs": {"format": "h5ad"}})

    plan = AgentOrchestrator(object())._build_preprocessing_plan(*inputs)

    assert plan.resetCellSelection is True
    assert isinstance(plan.cellQc, CellQcPlan)


@pytest.mark.parametrize(
    ("pairing_provenance", "explicit_pairing", "expected"),
    [
        (None, [], []),
        ("singleSourceSharedCellAxis", [], ["RNA", "ADT"]),
        (None, ["RNA", "ADT"], ["RNA", "ADT"]),
    ],
)
def test_multimodal_pairing_requires_persisted_or_explicit_provenance(
    pairing_provenance: str | None,
    explicit_pairing: list[str],
    expected: list[str],
) -> None:
    inputs = list(
        _planning_inputs(
            {
                "RNA": (
                    "RNA",
                    ["gene-1", "gene-2", "gene-3"],
                    ["A", "B", "C"],
                ),
                "ADT": (
                    "ADT",
                    ["adt-1", "adt-2", "adt-3"],
                    ["CD3", "CD19", "CD45"],
                ),
            },
            primary_assay="RNA",
        )
    )
    request_record = inputs[1]
    inputs[1] = request_record.model_copy(
        update={
            "request": request_record.request.model_copy(
                update={"pairedAssays": explicit_pairing}
            )
        }
    )
    inputs[4] = inputs[4].model_copy(
        update={
            "outputs": {
                "format": "h5ad" if pairing_provenance else "zarr",
                "pairingProvenance": pairing_provenance,
            }
        }
    )

    plan = AgentOrchestrator(object())._build_preprocessing_plan(*inputs)

    assert plan.pairedAssays == expected


def test_percent_features_follow_deterministic_inspection_not_policy_lists(
    tmp_path: Path,
) -> None:
    path = _create_store(tmp_path / "inspected-families.zarr")
    store = DataStore(
        str(path),
        default_assay="RNA",
        min_features_per_cell=-1,
        mito_pattern="",
        ribo_pattern="",
        zarr_mode="r+",
    )
    assert "RNA_percentMito" not in store.cells.columns
    workflow = create_agent_workflow(store, workflow_run_id="inspected-families")
    request_record = OrchestrationRequestRecord(
        workflowRunId=workflow.workflowRunId,
        request=AutomatedWorkflowRequest(
            sourcePath=str(path),
            zarrPath=str(path),
            studyContext="A deterministic feature-family test.",
            allowAssumptions=True,
        ),
        config=AutomatedWorkflowConfig(),
    )
    policy = _modality_policy("RNA", "RNA")
    assert policy.excludeFamilies == []
    assert policy.protectFamilies == []
    enrichment = DataEnrichmentReport(
        status="done",
        policies=[policy],
        inspections=[
            AssayFeatureInspection(
                assay="RNA",
                families=[
                    FeatureFamilyEvidence(
                        family="mitochondrial",
                        count=1,
                        examples=["MT-CO1"],
                        evidenceId="assay:RNA:family:mitochondrial",
                    )
                ],
            )
        ],
    )

    outcome = AgentOrchestrator(object())._hto_stage(
        store,
        workflow,
        request_record,
        [],
        enrichment,
    )

    assert outcome.status == "done"
    assert "RNA_percentMito" in store.cells.columns
    assert "compute_percent_mito:RNA" in outcome.actions
    assert outcome.outputs["operations"] == [
        {
            "operation": "add_percent_feature",
            "assay": "RNA",
            "pattern": r"^(MT-|mt-)",
            "column": "RNA_percentMito",
        }
    ]


def test_hto_demultiplexing_is_checkpointed_once_and_never_graph_bearing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _create_store(tmp_path / "hto-once.zarr")
    store = DataStore(
        str(path),
        default_assay="RNA",
        min_features_per_cell=-1,
        mito_pattern="",
        ribo_pattern="",
        zarr_mode="r+",
    )
    workflow = create_agent_workflow(store, workflow_run_id="hto-once")
    request_record = OrchestrationRequestRecord(
        workflowRunId=workflow.workflowRunId,
        request=AutomatedWorkflowRequest(
            sourcePath=str(path),
            zarrPath=str(path),
            studyContext="A deterministic HTO checkpoint test.",
            allowAssumptions=True,
        ),
        config=AutomatedWorkflowConfig(),
    )
    calls = 0

    def mark_identities(*_args: Any, **kwargs: Any) -> str:
        nonlocal calls
        calls += 1
        column = f"HTO_{kwargs['label']}"
        store.cells.insert(
            column,
            np.asarray(["negative", "singlet", "doublet", "singlet"]),
            overwrite=True,
        )
        return column

    monkeypatch.setattr(store, "mark_hto_identities", mark_identities)
    enrichment = DataEnrichmentReport(
        status="done",
        policies=[
            FeatureSelectionPolicy(
                assay="HTO",
                assayType="HTO",
                assayModality="HTO",
                demultiplexEligible=True,
                exactTagFeatures=[
                    FeatureReference(featureId="tag-1", featureName="sample-1"),
                    FeatureReference(featureId="tag-2", featureName="sample-2"),
                ],
                evidenceIds=["assay:HTO:modality"],
            )
        ],
    )
    orchestrator = AgentOrchestrator(object())

    first = orchestrator._hto_stage(
        store,
        workflow,
        request_record,
        [],
        enrichment,
    )
    second = orchestrator._hto_stage(
        store,
        workflow,
        request_record,
        [],
        enrichment,
    )

    assert first == second
    assert calls == 1
    assert first.outputs["operations"][0]["operation"] == "mark_hto_identities"
    assert all("graph" not in action for action in first.actions)


def test_failed_stage_preserves_completed_operation_journal(tmp_path: Path) -> None:
    path = _create_store(tmp_path / "failure-journal.zarr")
    store = DataStore(
        str(path),
        default_assay="RNA",
        min_features_per_cell=-1,
        mito_pattern="",
        ribo_pattern="",
        zarr_mode="r+",
    )
    workflow = create_agent_workflow(store, workflow_run_id="failure-journal")
    request_record = OrchestrationRequestRecord(
        workflowRunId=workflow.workflowRunId,
        request=AutomatedWorkflowRequest(
            sourcePath=str(path),
            zarrPath=str(path),
            studyContext="A failed-stage journal test.",
            allowAssumptions=True,
        ),
        config=AutomatedWorkflowConfig(),
    )
    prefix = journal_module._ensure_orchestration_store(store)
    started = journal_module._start_attempt(
        store.zw,
        prefix,
        workflow.workflowRunId,
        "preprocessing",
        request_record,
        [],
    )
    operation = {
        "operation": "filter_cells",
        "attrs": ["RNA_nCounts"],
        "resetPrevious": True,
    }

    outcome = journal_module.finish_exception(
        store,
        prefix,
        workflow,
        started,
        RuntimeError("failure after filtering"),
        actions=["cell_qc_global:validated"],
        outputs={"operations": [operation]},
        notes=["The completed selection mutation is retained for audit."],
    )

    assert outcome.status == "failed"
    assert outcome.actions == ["cell_qc_global:validated"]
    assert outcome.outputs["operations"] == [operation]
    assert outcome.notes == ["The completed selection mutation is retained for audit."]
    persisted = journal_module._stage_outcomes(
        store.zw,
        prefix,
        workflow.workflowRunId,
        "preprocessing",
    )
    assert persisted == [outcome]
    assert load_agent_workflow(store, workflow.workflowRunId).status == "failed"


def test_failed_stage_links_report_committed_before_exception(tmp_path: Path) -> None:
    path = _create_store(tmp_path / "failure-report-link.zarr")
    store = DataStore(
        str(path),
        default_assay="RNA",
        min_features_per_cell=-1,
        mito_pattern="",
        ribo_pattern="",
        zarr_mode="r+",
    )
    workflow = create_agent_workflow(store, workflow_run_id="failure-report-link")
    request_record = OrchestrationRequestRecord(
        workflowRunId=workflow.workflowRunId,
        request=AutomatedWorkflowRequest(
            sourcePath=str(path),
            zarrPath=str(path),
            studyContext="A report-link crash test.",
            allowAssumptions=True,
        ),
        config=AutomatedWorkflowConfig(),
    )
    prefix = journal_module._ensure_orchestration_store(store)
    started = journal_module._start_attempt(
        store.zw,
        prefix,
        workflow.workflowRunId,
        "data_enrichment",
        request_record,
        [],
        inputs={"studyContext": request_record.request.studyContext},
    )
    report = DataEnrichmentReport.get_example().model_copy(
        update={
            "runInfo": AgentRunInfo(
                agentName="data_enrichment",
                runId=uuid.uuid4().hex,
            )
        }
    )
    _saved, reference = journal_module._save_stage_report(
        store,
        started,
        report,
        invocation=AgentInvocation(
            agentName="data_enrichment",
            inputs={"studyContext": request_record.request.studyContext},
        ),
        expected_type=DataEnrichmentReport,
    )

    outcome = journal_module.finish_exception(
        store,
        prefix,
        workflow,
        started,
        RuntimeError("failure after report commit"),
    )

    assert outcome.reportReferences == [reference]
    assert load_agent_report(store, reference) == report


@pytest.mark.parametrize(
    ("assay_types", "explicit", "expected"),
    [
        (["ATAC", "ADT", "RNA"], None, "RNA"),
        (["ATAC", "ADT"], None, "ADT"),
        (["ATAC"], None, "ATAC"),
        (["RNA", "ADT", "ATAC"], "ATAC", "ATAC"),
    ],
)
def test_marker_assay_precedence(
    assay_types: list[str],
    explicit: str | None,
    expected: str,
) -> None:
    assays = {
        assay_type: (
            assay_type,
            [f"{assay_type}-1", f"{assay_type}-2", f"{assay_type}-3"],
            [f"{assay_type}-1", f"{assay_type}-2", f"{assay_type}-3"],
        )
        for assay_type in assay_types
    }

    plan = _build_plan(
        assays,
        primary_assay=assay_types[0],
        marker_assay=explicit,
    )

    assert plan.primaryAssay == assay_types[0]
    assert plan.markerAssay == expected


def test_adt_identity_limit_and_exact_observed_control_exclusion() -> None:
    assays = {
        "ADT": (
            "ADT",
            ["adt-1", "control-id", "adt-2", "adt-3"],
            [
                "CD3",
                "Mouse IgG1 isotype control",
                "control response protein",
                "CD19",
            ],
        )
    }
    controls = {
        "ADT": [
            FeatureReference(
                featureId="control-id",
                featureName="Mouse IgG1 isotype control",
            )
        ]
    }

    identity = _build_plan(
        assays,
        controls=controls,
        exclude_features={"ADT": ["adt-1"]},
        artificial_features={"ADT": ["adt-2"]},
        config=AutomatedWorkflowConfig(maxIdentityFeatures=3),
    ).assays[0]
    pca = _build_plan(
        assays,
        controls=controls,
        exclude_features={"ADT": ["adt-1"]},
        artificial_features={"ADT": ["adt-2"]},
        config=AutomatedWorkflowConfig(maxIdentityFeatures=2),
    ).assays[0]

    assert identity.exactExcludedFeatures == [
        "control-id",
        "Mouse IgG1 isotype control",
    ]
    assert identity.reductionMethod == "identity"
    assert identity.reductionParameters["dimensions"] == 3
    assert pca.reductionMethod == "pca"
    assert pca.reductionParameters["dimensions"] == 2


def test_atac_invalid_coordinates_are_limited_without_changing_lsi_route() -> None:
    assays = {
        "ATAC": (
            "ATAC",
            ["chr1:1-20", "not-a-coordinate", "chr2:10-30"],
            ["peak-1", "peak-2", "peak-3"],
        )
    }

    plan = _build_plan(assays, peak_statuses={"ATAC": "invalid"})
    route = plan.assays[0]

    assert route.graphEligible
    assert route.featureMethod == "prevalentPeaks"
    assert route.reductionMethod == "lsi"
    assert route.reductionParameters == {"dimensions": 50, "skipFirst": True}
    assert route.limitations == [
        "ATAC feature coordinates are not uniformly valid chrom:start-end "
        "intervals; the genome build remains unknown"
    ]


@pytest.mark.parametrize(
    ("method", "n_cells", "n_features", "expected_dimensions"),
    [
        ("pca", 5, 3, {2}),
        ("lsi", 6, 4, {3}),
        ("identity", 4, 2, {2}),
    ],
)
def test_initial_candidates_are_rank_valid(
    method: str,
    n_cells: int,
    n_features: int,
    expected_dimensions: set[int],
) -> None:
    orchestrator = AgentOrchestrator(object())
    handoff = PreprocessedAssayHandoff(
        assay="assay",
        assayType="ADT" if method == "identity" else method.upper(),
        reductionMethod=method,
        normalized=ArtifactReferenceModel(
            assay="assay",
            kind="normalized",
            artifactId="1" * 64,
        ),
        nCells=n_cells,
        nFeatures=n_features,
    )

    candidates = orchestrator._initial_parameter_candidates(
        "rank-test",
        handoff,
        count=5,
        neighbors_k=n_cells - 1,
    )

    assert len(candidates) == 5
    assert {value.dimensions for value in candidates} == expected_dimensions
    assert all(2 <= value.dimensions for value in candidates)
    if method != "identity":
        assert all(value.dimensions < min(n_cells, n_features) for value in candidates)
    assert all(2 <= value.neighborsK < n_cells for value in candidates)


def test_initial_candidates_reject_fully_invalid_rank_or_neighbor_count() -> None:
    orchestrator = AgentOrchestrator(object())
    rank_invalid = PreprocessedAssayHandoff(
        assay="RNA",
        assayType="RNA",
        reductionMethod="pca",
        normalized=ArtifactReferenceModel.get_example(),
        nCells=4,
        nFeatures=2,
    )
    identity_invalid = rank_invalid.model_copy(
        update={"assayType": "ADT", "reductionMethod": "identity", "nFeatures": 1}
    )

    with pytest.raises(ValueError, match="no rank-valid graph candidate"):
        orchestrator._initial_parameter_candidates(
            "rank-test",
            rank_invalid,
            count=3,
            neighbors_k=3,
        )
    with pytest.raises(ValueError, match="no rank-valid graph candidate"):
        orchestrator._initial_parameter_candidates(
            "rank-test",
            identity_invalid,
            count=3,
            neighbors_k=3,
        )
    with pytest.raises(ValueError, match="no rank-valid graph candidate"):
        orchestrator._initial_parameter_candidates(
            "rank-test",
            rank_invalid.model_copy(update={"nFeatures": 3}),
            count=3,
            neighbors_k=4,
        )


def test_parameter_tuning_rejects_plan_above_global_branch_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _create_store(tmp_path / "branch-cap.zarr")
    store = DataStore(str(path), default_assay="RNA", min_features_per_cell=0)
    workflow = create_agent_workflow(store, workflow_run_id="branch-cap")
    config = AutomatedWorkflowConfig(
        primaryInitialCandidates=5,
        maxRefinedCandidatesPerAssay=1,
        maxHarmonyCandidatesPerAssay=0,
        maxCandidateBranches=5,
    )
    request_record = OrchestrationRequestRecord(
        workflowRunId=workflow.workflowRunId,
        request=AutomatedWorkflowRequest(
            sourcePath=str(path),
            zarrPath=str(path),
            studyContext="A branch-cap test.",
            allowAssumptions=True,
        ),
        config=config,
    )
    plan = AutomatedPreprocessingPlan(
        primaryAssay="RNA",
        markerAssay="RNA",
        assays=[AssayPreprocessingPlan.get_example()],
    )
    handoff = PreprocessedAssayHandoff.get_example()
    experimental = ExperimentalContextResult.get_example()
    decision = experimental.decision.model_copy(
        update={"batchCorrection": BatchCorrectionPlan(action="skip")}
    )
    experimental = experimental.model_copy(
        update={"decision": decision, "batchSafety": []}
    )

    class UnusedAgent:
        def run_batch(self, *_args: Any, **_kwargs: Any) -> None:
            pytest.fail("The tuning agent must not run above the global branch cap")

    monkeypatch.setattr(
        tuning_module,
        "ParameterTuningAgent",
        lambda *_args, **_kwargs: UnusedAgent(),
    )
    outcome, report = AgentOrchestrator(object())._parameter_tuning_stage(
        store,
        workflow,
        request_record,
        [],
        plan,
        [handoff],
        experimental,
        AgentReportReference.get_example(),
        AgentReportReference.get_example(),
        {},
    )

    assert outcome.status == "failed"
    assert outcome.error is not None
    assert "exceeds the global branch limit 5" in outcome.error
    assert report.status == "failed"
    assert load_agent_workflow(store, workflow.workflowRunId).status == "failed"


def test_final_selection_pause_exposes_exact_options_and_resumes_without_screen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _create_store(tmp_path / "selection-resume.zarr")
    store = DataStore(str(path), default_assay="RNA", min_features_per_cell=0)
    workflow = create_agent_workflow(store, workflow_run_id="selection-resume")
    enrichment = DataEnrichmentReport.get_example().model_copy(
        update={
            "runInfo": AgentRunInfo(
                agentName="data_enrichment",
                runId=uuid.uuid4().hex,
            )
        }
    )
    experimental = ExperimentalContextResult.get_example().model_copy(
        update={
            "runInfo": AgentRunInfo(
                agentName="experimental_context",
                runId=uuid.uuid4().hex,
            )
        }
    )
    enrichment_reference = save_agent_report(
        store,
        workflow.workflowRunId,
        enrichment,
        invocation=AgentInvocation(
            agentName="data_enrichment",
            inputs={"studyContext": "A final-selection resume test."},
        ),
    )
    experimental_reference = save_agent_report(
        store,
        workflow.workflowRunId,
        experimental,
        invocation=AgentInvocation(
            agentName="experimental_context",
            inputs={"cellKey": "I"},
        ),
    )
    request_record = OrchestrationRequestRecord(
        workflowRunId=workflow.workflowRunId,
        request=AutomatedWorkflowRequest(
            sourcePath=str(path),
            zarrPath=str(path),
            studyContext="A final-selection resume test.",
            allowAssumptions=True,
        ),
    )
    plan = AutomatedPreprocessingPlan(
        primaryAssay="RNA",
        markerAssay="RNA",
        assays=[
            AssayPreprocessingPlan.get_example(),
            AssayPreprocessingPlan(
                assay="ADT",
                assayType="ADT",
                role="graph",
                graphEligible=True,
                markerEligible=True,
                featureMethod="panel",
                reductionMethod="identity",
            ),
        ],
        pairedAssays=["RNA", "ADT"],
    )
    preprocessed = [
        PreprocessedAssayHandoff.get_example(),
        PreprocessedAssayHandoff(
            assay="ADT",
            assayType="ADT",
            reductionMethod="identity",
            normalized=ArtifactReferenceModel(
                assay="ADT",
                kind="normalized",
                artifactId="2" * 64,
            ),
            nCells=100,
            nFeatures=5,
        ),
    ]
    integration = _eligible_integration()
    calls = {"screen": 0, "promote": 0, "integrate": 0, "selection": 0}

    def selection_execution(**_kwargs: Any) -> Any:
        return SimpleNamespace(
            output=FinalGraphSelection(
                status="needsInput",
                rationale="The supplied evidence does not resolve the final graph.",
                needsInput=FinalGraphNeedsInput(
                    question="An unconstrained provider question.",
                    options=["invented-option"],
                ),
            ),
            runInfo=AgentRunInfo(
                agentName="parameter_tuning_final_graph",
                runId=uuid.uuid4().hex,
            ),
        )

    monkeypatch.setattr(parameter_tuning_module, "run_agent_sync", selection_execution)

    class CountingAgent:
        config = AgentRunConfig()

        def run_batch(self, *_args: Any, **_kwargs: Any) -> ParameterTuningReport:
            calls["screen"] += 1
            return _native_batch_report("RNA", "ADT")

        def promote(self, *_args: Any, **_kwargs: Any) -> None:
            calls["promote"] += 1

        def select_final(
            self,
            *,
            report: ParameterTuningReport,
            integration_evaluations: list[IntegrationCandidateEvaluation],
            marker_assay: str,
        ) -> ParameterTuningReport:
            calls["selection"] += 1
            return select_final_parameter_graph(
                model=object(),
                report=report,
                integration_evaluations=integration_evaluations,
                marker_assay=marker_assay,
                config=self.config,
            )

    monkeypatch.setattr(
        tuning_module,
        "ParameterTuningAgent",
        lambda *_args, **_kwargs: CountingAgent(),
    )
    orchestrator = AgentOrchestrator(object())

    def evaluate_integrations(*_args: Any, **_kwargs: Any) -> list[Any]:
        calls["integrate"] += 1
        return [integration]

    monkeypatch.setattr(orchestrator, "_evaluate_integrations", evaluate_integrations)
    real_validated_outcome = journal_module._validated_done_outcome
    paused_attempts: list[WorkflowStageAttempt] = []

    def validated_outcome(
        target_store: Any,
        prefix: str,
        workflow_run_id: str,
        stage: str,
        record: OrchestrationRequestRecord,
        parents: list[WorkflowStageLink],
        *,
        required_status: str = "done",
    ) -> WorkflowStageAttempt | None:
        if stage == "parameter_tuning":
            if required_status == "needsInput" and paused_attempts:
                return paused_attempts[-1]
            return None
        return real_validated_outcome(
            target_store,
            prefix,
            workflow_run_id,
            stage,
            record,
            parents,
            required_status=required_status,
        )

    monkeypatch.setattr(
        tuning_module,
        "_validated_done_outcome",
        validated_outcome,
    )
    paused, paused_report = orchestrator._parameter_tuning_stage(
        store,
        workflow,
        request_record,
        [],
        plan,
        preprocessed,
        experimental,
        enrichment_reference,
        experimental_reference,
        {},
    )
    paused_attempts.append(paused)
    expected_options = sorted(final_graph_options(paused_report, [integration]))

    assert paused.status == "needsInput"
    assert paused.needsInput is not None
    assert paused.needsInput.questions[0].questionId == "finalGraphOptionId"
    assert paused.needsInput.questions[0].options == expected_options
    assert paused_report.needsInput is not None
    assert paused_report.needsInput.options == expected_options
    assert paused_report.finalSelection is not None
    assert paused_report.finalSelection.needsInput is not None
    assert paused_report.finalSelection.needsInput.options == expected_options
    assert paused_report.totalCandidates == 3

    completed, completed_report = orchestrator._parameter_tuning_stage(
        store,
        workflow,
        request_record,
        [],
        plan,
        preprocessed,
        experimental,
        enrichment_reference,
        experimental_reference,
        {"finalGraphOptionId": "native:RNA:baseline"},
    )

    assert completed.status == "done"
    assert completed_report.status == "done"
    assert completed_report.finalSelection is not None
    assert completed_report.finalSelection.selectedOptionId == "native:RNA:baseline"
    assert completed_report.totalCandidates == 3
    assert calls == {"screen": 1, "promote": 2, "integrate": 1, "selection": 1}


def test_integration_evaluations_contribute_to_final_candidate_count() -> None:
    report = _native_batch_report("RNA", "ADT")
    integration = _eligible_integration()

    finalized = finalize_parameter_tuning_selection(
        report,
        marker_assay="RNA",
        integration_evaluations=[integration],
        native_assay="RNA",
    )

    assert report.totalCandidates == 2
    assert finalized.totalCandidates == 3


def test_orphan_report_recovery_uses_semantic_stage_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _orchestrator, paused, path = _start_paused_workflow(tmp_path, monkeypatch)
    assert paused.workflowRun is not None
    workflow_id = paused.workflowRun.workflowRunId
    store = DataStore(
        str(path),
        min_features_per_cell=-1,
        mito_pattern="",
        ribo_pattern="",
        zarr_mode="r+",
    )
    started = WorkflowStageAttempt(
        workflowRunId=workflow_id,
        stage="data_enrichment",
        attemptId="crash-attempt-one",
        status="started",
        startedAtNs=1,
        requestSha256="1" * 64,
        configSha256="2" * 64,
        inputs={
            "effectiveContext": "same",
            "resumeLineage": {
                "resumeId": "first-resume",
                "answeredAttempt": WorkflowStageLink.get_example().model_dump(
                    mode="json"
                ),
                "questionIds": ["approvePlanChecksum"],
            },
        },
    )
    report = DataEnrichmentReport.get_example().model_copy(
        update={
            "runInfo": DataEnrichmentReport.get_example().runInfo.model_copy(
                update={"runId": uuid.uuid4().hex}
            )
        }
    )
    persisted_report, reference = journal_module._save_stage_report(
        store,
        started,
        report,
        invocation=AgentInvocation(
            agentName="data_enrichment",
            inputs={"effectiveContext": "same"},
        ),
        expected_type=DataEnrichmentReport,
    )
    assert persisted_report == report

    retried = started.model_copy(
        update={
            "attemptId": "crash-attempt-two",
            "startedAtNs": 2,
            "inputs": {
                "effectiveContext": "same",
                "resumeLineage": {
                    "resumeId": "second-resume",
                    "answeredAttempt": None,
                    "questionIds": [],
                },
            },
        }
    )
    assert journal_module._stage_execution_id(started) == (
        journal_module._stage_execution_id(retried)
    )
    changed_context = retried.model_copy(
        update={"inputs": {"effectiveContext": "different"}}
    )
    assert journal_module._stage_execution_id(started) != (
        journal_module._stage_execution_id(changed_context)
    )
    recovered = journal_module._recover_persisted_stage_report(
        store,
        retried,
        agent_name="data_enrichment",
        expected_type=DataEnrichmentReport,
    )

    assert recovered == (report, reference)
    matching = [
        value
        for value in list_agent_reports(
            store,
            workflow_id,
            agent_name="data_enrichment",
        )
        if value.agentRunId == reference.agentRunId
    ]
    assert matching == [reference]


def test_data_enrichment_recovers_report_after_outcome_write_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator, paused, path = _start_paused_workflow(tmp_path, monkeypatch)
    assert paused.workflowRun is not None
    workflow_id = paused.workflowRun.workflowRunId
    store = DataStore(
        str(path),
        min_features_per_cell=-1,
        mito_pattern="",
        ribo_pattern="",
        zarr_mode="r+",
    )
    prefix = journal_module._ensure_orchestration_store(store)
    request_record = journal_module._read_model(
        store.zw,
        journal_module._request_key(prefix, workflow_id),
        OrchestrationRequestRecord,
    )
    assert isinstance(request_record, OrchestrationRequestRecord)
    workflow = load_agent_workflow(store, workflow_id)
    calls = 0

    class CountingAgent:
        config = AgentRunConfig()

        def run(self, *_args: Any, **_kwargs: Any) -> DataEnrichmentReport:
            nonlocal calls
            calls += 1
            return DataEnrichmentReport.get_example().model_copy(
                update={
                    "runInfo": DataEnrichmentReport.get_example().runInfo.model_copy(
                        update={"runId": uuid.uuid4().hex}
                    )
                }
            )

    monkeypatch.setattr(
        context_module,
        "DataEnrichmentAgent",
        lambda *_args, **_kwargs: CountingAgent(),
    )
    save_outcome = journal_module._save_outcome
    crashed = False

    def crash_after_report(*args: Any, **kwargs: Any) -> None:
        nonlocal crashed
        outcome = args[2]
        if outcome.stage == "data_enrichment" and not crashed:
            crashed = True
            raise KeyboardInterrupt("simulated process crash")
        save_outcome(*args, **kwargs)

    monkeypatch.setattr(context_module, "_save_outcome", crash_after_report)
    with pytest.raises(KeyboardInterrupt, match="simulated process crash"):
        orchestrator._data_enrichment_stage(
            store,
            workflow,
            request_record,
            [],
            {},
        )
    monkeypatch.setattr(context_module, "_save_outcome", save_outcome)

    outcome, recovered = orchestrator._data_enrichment_stage(
        store,
        workflow,
        request_record,
        [],
        {},
    )

    assert outcome.status == "done"
    assert recovered.status == "done"
    assert calls == 1
    assert "recover_persisted_data_enrichment_report" in outcome.actions


def test_long_assay_names_produce_bounded_unique_candidate_ids() -> None:
    orchestrator = AgentOrchestrator(object())
    prefix = "Very long assay name with punctuation / and spaces " + "x" * 120
    first = PreprocessedAssayHandoff(
        assay=f"{prefix} one",
        assayType="RNA",
        reductionMethod="pca",
        normalized=ArtifactReferenceModel.get_example(),
        nCells=100,
        nFeatures=50,
    )
    second = first.model_copy(update={"assay": f"{prefix} two"})

    first_ids = {
        candidate.candidateId
        for candidate in orchestrator._initial_parameter_candidates(
            "long-name-test",
            first,
            count=5,
            neighbors_k=11,
        )
    }
    second_ids = {
        candidate.candidateId
        for candidate in orchestrator._initial_parameter_candidates(
            "long-name-test",
            second,
            count=5,
            neighbors_k=11,
        )
    }

    assert first_ids.isdisjoint(second_ids)
    assert all(len(candidate_id) <= 64 for candidate_id in first_ids | second_ids)
    assert all(
        all(
            character.isdigit() or character.islower() or character in {"_", "-"}
            for character in candidate_id
        )
        for candidate_id in first_ids | second_ids
    )
    assert all(
        len(f"{candidate_id}_harmony") <= 64 for candidate_id in first_ids | second_ids
    )


def test_single_integration_resolution_is_centered_and_workflow_unique() -> None:
    report = _native_batch_report("RNA", "ADT")
    primary_report = report.assayReports["RNA"]
    primary_evaluation = primary_report.evaluations[0]
    centered_evaluation = primary_evaluation.model_copy(
        update={
            "parameters": primary_evaluation.parameters.model_copy(
                update={"leidenResolution": 1.25}
            )
        }
    )
    primary_report = primary_report.model_copy(
        update={"evaluations": [centered_evaluation]}
    )
    report = report.model_copy(
        update={
            "evaluations": [centered_evaluation],
            "assayReports": {
                **report.assayReports,
                "RNA": primary_report,
            },
        }
    )

    class IntegrationStore:
        def __init__(self) -> None:
            self.integration_calls: list[dict[str, Any]] = []
            self.cluster_calls: list[dict[str, Any]] = []

        def load_artifact(self, reference: ArtifactRef) -> dict[str, np.ndarray]:
            if reference.kind == "integrated_graph":
                return {"modality_weights": np.full((4, 2), 0.5)}
            return {"values": np.asarray([0, 0, 1, 1])}

        def integrate_assays(
            self,
            assays: list[str],
            label: str,
            **kwargs: Any,
        ) -> ArtifactRef:
            self.integration_calls.append({"assays": assays, "label": label, **kwargs})
            token = len(self.integration_calls)
            return ArtifactRef(
                scope="datastore",
                kind="integrated_graph",
                artifact_id=f"{100 + token:064x}",
            )

        def run_leiden_clustering(self, **kwargs: Any) -> ArtifactRef:
            self.cluster_calls.append(kwargs)
            token = len(self.cluster_calls)
            return ArtifactRef(
                scope="datastore",
                kind="cluster_labels",
                artifact_id=f"{200 + token:064x}",
            )

    store = IntegrationStore()
    plan = AutomatedPreprocessingPlan(
        primaryAssay="RNA",
        markerAssay="RNA",
        pairedAssays=["RNA", "ADT"],
    )
    evaluations = AgentOrchestrator(object())._evaluate_integrations(
        store,
        "integration-center",
        plan,
        report,
        ExperimentalTuningHandoff(batchAction="skip"),
        AutomatedWorkflowConfig(
            integrationResolutionCandidates=1,
            minClusterCells=1,
        ),
    )

    assert len(evaluations) == 2
    assert {value.method for value in evaluations} == {"snn", "wnn"}
    assert {value.resolution for value in evaluations} == {1.25}
    assert len(store.integration_calls) == 2
    assert all(call["invalidate_cache"] is True for call in store.integration_calls)
    assert {call["method"] for call in store.integration_calls} == {"snn", "wnn"}
    assert len(store.cluster_calls) == 2
    assert {call["resolution"] for call in store.cluster_calls} == {1.25}


def test_integration_requires_trusted_label_connectivity() -> None:
    class IntegrationStore:
        def load_artifact(self, reference: ArtifactRef) -> dict[str, np.ndarray]:
            if reference.kind == "integrated_graph":
                return {"modality_weights": np.full((4, 2), 0.5)}
            return {"values": np.asarray([0, 0, 1, 1])}

        def integrate_assays(
            self,
            _assays: list[str],
            _label: str,
            **_kwargs: Any,
        ) -> ArtifactRef:
            return ArtifactRef(
                scope="datastore",
                kind="integrated_graph",
                artifact_id="7" * 64,
            )

        def run_leiden_clustering(self, **_kwargs: Any) -> ArtifactRef:
            return ArtifactRef(
                scope="datastore",
                kind="cluster_labels",
                artifact_id="8" * 64,
            )

        def metric_graph_connectivity(self, *_args: Any, **_kwargs: Any) -> float:
            raise ValueError("trusted label is unavailable")

    evaluations = AgentOrchestrator(object())._evaluate_integrations(
        IntegrationStore(),
        "integration-connectivity",
        AutomatedPreprocessingPlan(
            primaryAssay="RNA",
            markerAssay="RNA",
            pairedAssays=["RNA", "ADT"],
        ),
        _native_batch_report("RNA", "ADT"),
        ExperimentalTuningHandoff(
            batchAction="skip",
            preservationColumns=["trusted_cell_type"],
        ),
        AutomatedWorkflowConfig(
            integrationResolutionCandidates=1,
            minClusterCells=1,
        ),
    )

    assert evaluations
    assert all(not value.eligible for value in evaluations)
    assert all(
        any(
            "trusted-label connectivity" in reason
            for reason in value.eligibilityReasons
        )
        for value in evaluations
    )


def test_integration_checkpoints_prevent_retry_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _create_store(tmp_path / "integration-checkpoint.zarr")
    store = DataStore(
        str(path),
        default_assay="RNA",
        min_features_per_cell=-1,
        mito_pattern="",
        ribo_pattern="",
        zarr_mode="r+",
    )
    workflow = create_agent_workflow(store, workflow_run_id="integration-checkpoint")
    request_record = OrchestrationRequestRecord(
        workflowRunId=workflow.workflowRunId,
        request=AutomatedWorkflowRequest(
            sourcePath=str(path),
            zarrPath=str(path),
            studyContext="A deterministic integration retry test.",
            allowAssumptions=True,
        ),
        config=AutomatedWorkflowConfig(),
    )
    prefix = journal_module._ensure_orchestration_store(store)
    started = journal_module._start_attempt(
        store.zw,
        prefix,
        workflow.workflowRunId,
        "parameter_tuning",
        request_record,
        [],
        inputs={"semanticInput": "stable"},
    )
    integration_calls: list[dict[str, Any]] = []
    cluster_calls: list[dict[str, Any]] = []

    def load_artifact(reference: ArtifactRef) -> dict[str, np.ndarray]:
        if reference.kind == "integrated_graph":
            return {"modality_weights": np.full((4, 2), 0.5)}
        return {"values": np.asarray([0, 0, 1, 1])}

    def integrate_assays(
        assays: list[str],
        label: str,
        **kwargs: Any,
    ) -> ArtifactRef:
        integration_calls.append({"assays": assays, "label": label, **kwargs})
        return ArtifactRef(
            scope="datastore",
            kind="integrated_graph",
            artifact_id=f"{100 + len(integration_calls):064x}",
        )

    def cluster(**kwargs: Any) -> ArtifactRef:
        cluster_calls.append(kwargs)
        return ArtifactRef(
            scope="datastore",
            kind="cluster_labels",
            artifact_id=f"{200 + len(cluster_calls):064x}",
        )

    monkeypatch.setattr(store, "load_artifact", load_artifact)
    monkeypatch.setattr(store, "integrate_assays", integrate_assays)
    monkeypatch.setattr(store, "run_leiden_clustering", cluster)
    plan = AutomatedPreprocessingPlan(
        primaryAssay="RNA",
        markerAssay="RNA",
        pairedAssays=["RNA", "ADT"],
    )
    report = _native_batch_report("RNA", "ADT")
    config = AutomatedWorkflowConfig(
        integrationResolutionCandidates=1,
        minClusterCells=1,
    )
    first_actions: list[str] = []
    orchestrator = AgentOrchestrator(object())

    first = orchestrator._evaluate_integrations(
        store,
        workflow.workflowRunId,
        plan,
        report,
        ExperimentalTuningHandoff(batchAction="skip"),
        config,
        started=started,
        actions=first_actions,
    )
    retried = started.model_copy(
        update={"attemptId": "retry-attempt", "startedAtNs": started.startedAtNs + 1}
    )
    retry_actions: list[str] = []
    second = orchestrator._evaluate_integrations(
        store,
        workflow.workflowRunId,
        plan,
        report,
        ExperimentalTuningHandoff(batchAction="skip"),
        config,
        started=retried,
        actions=retry_actions,
    )

    assert second == first
    assert len(integration_calls) == 2
    assert len(cluster_calls) == 2
    assert first_actions == [
        "checkpoint_integration:snn",
        "checkpoint_integration:wnn",
    ]
    assert retry_actions == [
        "recover_integration_checkpoint:snn",
        "recover_integration_checkpoint:wnn",
    ]
    checkpoint_reports = list_agent_reports(
        store,
        workflow.workflowRunId,
        agent_name="parameter_tuning",
    )
    assert len(checkpoint_reports) == 2
    assert all("_integration_" in value.agentRunId for value in checkpoint_reports)
