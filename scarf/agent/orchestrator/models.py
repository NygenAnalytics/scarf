"""Public data models for resumable automated agent workflows."""

import re
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from ...storage.refs import ArtifactRef
from ..config import AgentRunConfig
from ..experimental_context import CellQcPlan
from ..persistence import AgentReportReference, AgentWorkflowRun
from ..types import AgentDataModel, ArtifactReferenceModel

type AutomatedWorkflowStatus = Literal["completed", "needsInput", "failed", "abandoned"]
type WorkflowStageStatus = Literal["started", "done", "needsInput", "failed"]
type WorkflowStageName = Literal[
    "ingest",
    "data_enrichment",
    "hto_demultiplexing",
    "experimental_context",
    "preprocessing_plan",
    "preprocessing",
    "parameter_tuning",
    "analysis_finalization",
    "biological_interpretation",
]
type AssayRole = Literal["graph", "hto", "unsupported"]
type ReductionMethod = Literal["pca", "lsi", "identity", "none"]

_RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,127}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ORCHESTRATION_FORMAT = "scarf_agent_orchestrations"
_ORCHESTRATION_VERSION = 1
_STAGE_ORDER: tuple[WorkflowStageName, ...] = (
    "ingest",
    "data_enrichment",
    "hto_demultiplexing",
    "experimental_context",
    "preprocessing_plan",
    "preprocessing",
    "parameter_tuning",
    "analysis_finalization",
    "biological_interpretation",
)


class WorkflowQuestion(AgentDataModel):
    """One stable question that can be answered by a resume request."""

    questionId: str = ""
    question: str = ""
    options: list[str] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)
    planChecksum: str | None = None

    @classmethod
    def get_blank(cls) -> "WorkflowQuestion":
        return cls()

    @classmethod
    def get_example(cls) -> "WorkflowQuestion":
        return cls(
            questionId="approvePlanChecksum",
            question="Approve this preprocessing plan?",
            planChecksum="0" * 64,
        )


class WorkflowNeedsInput(AgentDataModel):
    """All questions blocking the next workflow stage."""

    questions: list[WorkflowQuestion] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "WorkflowNeedsInput":
        return cls()

    @classmethod
    def get_example(cls) -> "WorkflowNeedsInput":
        return cls(questions=[WorkflowQuestion.get_example()])


class WorkflowStageLink(AgentDataModel):
    """Immutable identity of one completed parent stage attempt."""

    stage: WorkflowStageName = "ingest"
    attemptId: str = ""
    contentSha256: str = ""

    @field_validator("attemptId")
    @classmethod
    def validate_attempt_id(cls, value: str) -> str:
        if value and _RUN_ID_PATTERN.fullmatch(value) is None:
            raise ValueError("attemptId must be a lowercase run identifier")
        return value

    @field_validator("contentSha256")
    @classmethod
    def validate_checksum(cls, value: str) -> str:
        if value and _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("contentSha256 must be a lowercase SHA-256 digest")
        return value

    @classmethod
    def get_blank(cls) -> "WorkflowStageLink":
        return cls()

    @classmethod
    def get_example(cls) -> "WorkflowStageLink":
        return cls(stage="ingest", attemptId="attempt-1", contentSha256="0" * 64)


class WorkflowStageAttempt(AgentDataModel):
    """Append-only record for one orchestration stage attempt."""

    workflowRunId: str = ""
    stage: WorkflowStageName = "ingest"
    attemptId: str = ""
    status: WorkflowStageStatus = "started"
    startedAtNs: int = Field(default=0, ge=0)
    completedAtNs: int = Field(default=0, ge=0)
    requestSha256: str = ""
    configSha256: str = ""
    parentAttempts: list[WorkflowStageLink] = Field(default_factory=list)
    reportReferences: list[AgentReportReference] = Field(default_factory=list)
    artifacts: dict[str, ArtifactReferenceModel] = Field(default_factory=dict)
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    actions: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    needsInput: WorkflowNeedsInput | None = None
    error: str | None = None
    contentSha256: str = ""

    @model_validator(mode="after")
    def validate_lifecycle(self) -> "WorkflowStageAttempt":
        if self.status == "started" and self.completedAtNs:
            raise ValueError("A started stage cannot have completedAtNs")
        if self.status != "started" and self.completedAtNs < self.startedAtNs:
            raise ValueError("A completed stage must not precede its start")
        if self.status == "needsInput" and self.needsInput is None:
            raise ValueError("needsInput stage records require questions")
        if self.status == "failed" and not self.error:
            raise ValueError("failed stage records require an error")
        return self

    @classmethod
    def get_blank(cls) -> "WorkflowStageAttempt":
        return cls()

    @classmethod
    def get_example(cls) -> "WorkflowStageAttempt":
        return cls(
            workflowRunId="workflow-1",
            stage="ingest",
            attemptId="attempt-1",
            status="done",
            startedAtNs=1,
            completedAtNs=2,
            requestSha256="0" * 64,
            configSha256="1" * 64,
            contentSha256="2" * 64,
        )


class AssayPreprocessingPlan(AgentDataModel):
    """Exact allowlisted preprocessing route for one assay."""

    assay: str = ""
    assayType: str = "Assay"
    role: AssayRole = "unsupported"
    graphEligible: bool = False
    markerEligible: bool = False
    featureMethod: Literal["hvg", "prevalentPeaks", "panel", "none"] = "none"
    reductionMethod: ReductionMethod = "none"
    featureParameters: dict[str, Any] = Field(default_factory=dict)
    normalizationParameters: dict[str, Any] = Field(default_factory=dict)
    reductionParameters: dict[str, Any] = Field(default_factory=dict)
    exactExcludedFeatures: list[str] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "AssayPreprocessingPlan":
        return cls()

    @classmethod
    def get_example(cls) -> "AssayPreprocessingPlan":
        return cls(
            assay="RNA",
            assayType="RNA",
            role="graph",
            graphEligible=True,
            markerEligible=True,
            featureMethod="hvg",
            reductionMethod="pca",
            featureParameters={"topN": 1000, "minCells": 20},
        )


class AutomatedPreprocessingPlan(AgentDataModel):
    """Dataset-wide preprocessing plan produced before selection changes."""

    primaryAssay: str = ""
    markerAssay: str = ""
    cellKey: str = "I"
    cellQc: CellQcPlan = Field(default_factory=CellQcPlan.get_blank)
    assays: list[AssayPreprocessingPlan] = Field(default_factory=list)
    pairedAssays: list[str] = Field(default_factory=list)
    resetCellSelection: bool = False
    planChecksum: str = ""
    limitations: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "AutomatedPreprocessingPlan":
        return cls()

    @classmethod
    def get_example(cls) -> "AutomatedPreprocessingPlan":
        return cls(
            primaryAssay="RNA",
            markerAssay="RNA",
            assays=[AssayPreprocessingPlan.get_example()],
            planChecksum="0" * 64,
        )


class PreprocessedAssayHandoff(AgentDataModel):
    """Exact normalized input and selections handed to Parameter Tuning."""

    assay: str = ""
    assayType: str = "Assay"
    cellKey: str = "I"
    reductionMethod: ReductionMethod = "none"
    graphFeatures: ArtifactReferenceModel | None = None
    markerFeatures: ArtifactReferenceModel | None = None
    normalized: ArtifactReferenceModel | None = None
    nCells: int = 0
    nFeatures: int = 0

    @classmethod
    def get_blank(cls) -> "PreprocessedAssayHandoff":
        return cls()

    @classmethod
    def get_example(cls) -> "PreprocessedAssayHandoff":
        return cls(
            assay="RNA",
            assayType="RNA",
            reductionMethod="pca",
            graphFeatures=ArtifactReferenceModel.get_example(),
            markerFeatures=ArtifactReferenceModel.get_example(),
            normalized=ArtifactReferenceModel(
                assay="RNA", kind="normalized", artifactId="1" * 64
            ),
            nCells=100,
            nFeatures=1000,
        )


class NativeAnalysisHandoff(AgentDataModel):
    """Selected immutable native analysis chain for one assay."""

    assay: str = ""
    reductionMethod: ReductionMethod = "none"
    featureSelection: ArtifactReferenceModel | None = None
    markerFeatures: ArtifactReferenceModel | None = None
    normalized: ArtifactReferenceModel | None = None
    reduction: ArtifactReferenceModel | None = None
    batchCorrection: ArtifactReferenceModel | None = None
    annIndex: ArtifactReferenceModel | None = None
    embeddingInitialization: ArtifactReferenceModel | None = None
    neighbors: ArtifactReferenceModel | None = None
    graph: ArtifactReferenceModel | None = None
    clusters: ArtifactReferenceModel | None = None
    clusterColumn: str = ""
    umap: ArtifactReferenceModel | None = None
    umapColumns: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "NativeAnalysisHandoff":
        return cls()

    @classmethod
    def get_example(cls) -> "NativeAnalysisHandoff":
        return cls(assay="RNA", reductionMethod="pca")


class FinalAnalysisHandoff(AgentDataModel):
    """Replayable final analysis used by Biological Interpretation."""

    workflowRunId: str = ""
    primaryAssay: str = ""
    markerAssay: str = ""
    cellKey: str = "I"
    nativeAnalyses: list[NativeAnalysisHandoff] = Field(default_factory=list)
    graph: ArtifactReferenceModel | None = None
    graphMethod: Literal["native", "snn", "wnn"] = "native"
    clusters: ArtifactReferenceModel | None = None
    clusterColumn: str = ""
    umap: ArtifactReferenceModel | None = None
    umapColumns: list[str] = Field(default_factory=list)
    markerFeatures: ArtifactReferenceModel | None = None
    markers: ArtifactReferenceModel | None = None
    parameterReport: AgentReportReference | None = None
    limitations: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "FinalAnalysisHandoff":
        return cls()

    @classmethod
    def get_example(cls) -> "FinalAnalysisHandoff":
        return cls(
            workflowRunId="workflow-1",
            primaryAssay="RNA",
            markerAssay="RNA",
            nativeAnalyses=[NativeAnalysisHandoff.get_example()],
            graph=ArtifactReferenceModel(
                assay="RNA", kind="connectivity_map", artifactId="2" * 64
            ),
            clusters=ArtifactReferenceModel(
                assay="RNA", kind="cluster_labels", artifactId="3" * 64
            ),
            clusterColumn="RNA_agent_clusters",
        )


class AutomatedWorkflowConfig(AgentDataModel):
    """Bounded execution policy for automated workflows."""

    primaryInitialCandidates: int = Field(default=5, ge=1)
    secondaryInitialCandidates: int = Field(default=3, ge=1)
    maxRefinedCandidatesPerAssay: int = Field(default=1, ge=0, le=1)
    maxHarmonyCandidatesPerAssay: int = Field(default=1, ge=0, le=1)
    integrationResolutionCandidates: int = Field(default=3, ge=1)
    maxCandidateBranches: int = Field(default=24, ge=1)
    minClusterCells: int = Field(default=20, ge=1)
    maxIdentityFeatures: int = Field(default=64, ge=2)
    maxGraphAssays: int = Field(default=3, ge=1)
    allowDownloads: bool = False
    cacheDir: str | None = None
    agentRunConfig: AgentRunConfig = Field(default_factory=AgentRunConfig)

    @classmethod
    def get_blank(cls) -> "AutomatedWorkflowConfig":
        return cls()

    @classmethod
    def get_example(cls) -> "AutomatedWorkflowConfig":
        return cls()


class AutomatedWorkflowRequest(AgentDataModel):
    """Immutable request for one automated analysis."""

    sourcePath: str = ""
    zarrPath: str | None = None
    studyContext: str = ""
    workspace: str | None = None
    allowAssumptions: bool = False
    primaryAssay: str | None = None
    markerAssay: str | None = None
    analysisAssays: list[str] = Field(default_factory=list)
    pairedAssays: list[str] = Field(default_factory=list)
    resetCellSelection: bool | None = None
    ingestDirections: dict[str, Any] = Field(default_factory=dict)
    experimentalDirections: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_request(self) -> "AutomatedWorkflowRequest":
        if not self.sourcePath.strip():
            raise ValueError("sourcePath must be non-empty")
        if not self.studyContext.strip():
            raise ValueError("studyContext must be non-empty")
        if len(set(self.analysisAssays)) != len(self.analysisAssays):
            raise ValueError("analysisAssays must be unique")
        if len(set(self.pairedAssays)) != len(self.pairedAssays):
            raise ValueError("pairedAssays must be unique")
        if self.pairedAssays and len(self.pairedAssays) < 2:
            raise ValueError("pairedAssays must contain at least two assays")
        return self

    @classmethod
    def get_blank(cls) -> "AutomatedWorkflowRequest":
        return cls(sourcePath="dataset.h5", studyContext="Study context")

    @classmethod
    def get_example(cls) -> "AutomatedWorkflowRequest":
        return cls(
            sourcePath="dataset.h5ad",
            zarrPath="dataset.zarr",
            studyContext="Single-cell profiling of treated human blood.",
        )


class AutomatedWorkflowResumeRequest(AgentDataModel):
    """Answers used to resume one running workflow."""

    zarrPath: str = ""
    workflowRunId: str = ""
    workspace: str | None = None
    answers: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_request(self) -> "AutomatedWorkflowResumeRequest":
        if not self.zarrPath.strip():
            raise ValueError("zarrPath must be non-empty")
        if _RUN_ID_PATTERN.fullmatch(self.workflowRunId) is None:
            raise ValueError("workflowRunId must be a lowercase run identifier")
        return self

    @classmethod
    def get_blank(cls) -> "AutomatedWorkflowResumeRequest":
        return cls(zarrPath="dataset.zarr", workflowRunId="workflow-1")

    @classmethod
    def get_example(cls) -> "AutomatedWorkflowResumeRequest":
        return cls(
            zarrPath="dataset.zarr",
            workflowRunId="workflow-1",
            answers={"approvePlanChecksum": "0" * 64},
        )


class AutomatedWorkflowResult(AgentDataModel):
    """Bounded result of running or resuming an automated workflow."""

    status: AutomatedWorkflowStatus = "failed"
    currentStage: WorkflowStageName = "ingest"
    zarrPath: str | None = None
    workflowRun: AgentWorkflowRun | None = None
    reportReferences: list[AgentReportReference] = Field(default_factory=list)
    preprocessingPlan: AutomatedPreprocessingPlan | None = None
    finalAnalysis: FinalAnalysisHandoff | None = None
    needsInput: WorkflowNeedsInput | None = None
    notes: list[str] = Field(default_factory=list)
    contentSha256: str = ""

    @classmethod
    def get_blank(cls) -> "AutomatedWorkflowResult":
        return cls()

    @classmethod
    def get_example(cls) -> "AutomatedWorkflowResult":
        return cls(
            status="completed",
            currentStage="biological_interpretation",
            zarrPath="dataset.zarr",
            workflowRun=AgentWorkflowRun.get_example(),
            finalAnalysis=FinalAnalysisHandoff.get_example(),
        )


class OrchestrationRequestRecord(AgentDataModel):
    """Stored immutable request and effective configuration."""

    recordType: Literal["automatedWorkflowRequest"] = "automatedWorkflowRequest"
    formatVersion: Literal[1] = 1
    workflowRunId: str = ""
    createdAtNs: int = Field(default=0, ge=0)
    request: AutomatedWorkflowRequest = Field(
        default_factory=AutomatedWorkflowRequest.get_blank
    )
    config: AutomatedWorkflowConfig = Field(
        default_factory=AutomatedWorkflowConfig.get_blank
    )
    requestSha256: str = ""
    configSha256: str = ""
    contentSha256: str = ""

    @classmethod
    def get_blank(cls) -> "OrchestrationRequestRecord":
        return cls()

    @classmethod
    def get_example(cls) -> "OrchestrationRequestRecord":
        return cls(
            workflowRunId="workflow-1",
            createdAtNs=1,
            request=AutomatedWorkflowRequest.get_example(),
            config=AutomatedWorkflowConfig.get_example(),
            requestSha256="0" * 64,
            configSha256="1" * 64,
        )


class OrchestrationResumeRecord(AgentDataModel):
    """One append-only set of answers supplied during resume."""

    recordType: Literal["automatedWorkflowResume"] = "automatedWorkflowResume"
    workflowRunId: str = ""
    resumeId: str = ""
    createdAtNs: int = Field(default=0, ge=0)
    answeredAttempt: WorkflowStageLink | None = None
    questionIds: list[str] = Field(default_factory=list)
    answers: dict[str, Any] = Field(default_factory=dict)
    contentSha256: str = ""

    @classmethod
    def get_blank(cls) -> "OrchestrationResumeRecord":
        return cls()

    @classmethod
    def get_example(cls) -> "OrchestrationResumeRecord":
        return cls(
            workflowRunId="workflow-1",
            resumeId="resume-1",
            createdAtNs=1,
            answers={"approvePlanChecksum": "0" * 64},
        )


def artifact_model_to_ref(value: ArtifactReferenceModel) -> ArtifactRef:
    """Convert an agent artifact model to a validated core artifact reference."""
    return ArtifactRef(
        scope=value.scope,
        assay=value.assay,
        kind=value.kind,
        artifact_id=value.artifactId,
    )
