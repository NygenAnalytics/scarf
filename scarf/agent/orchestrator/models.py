"""Public data models for resumable automated agent workflows."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, field_validator, model_validator

from ...storage.refs import ArtifactRef
from ..cell_quality.profiles import cell_qc_policy
from ..config import AgentRunConfig
from ..decisions.rna import CellQualityExecutorPayload
from ..experimental_context.contracts import CellQcPlan
from ..experimental_context.study import AuthorLabelPolicy
from ..types import AgentDataModel, ArtifactReferenceModel

if TYPE_CHECKING:
    import pandas as pd

    from ...datastore.datastore import DataStore
    from ...plotting._figure import PlotResult

type AutomatedWorkflowStatus = Literal[
    "completed",
    "needsInput",
    "abstained",
    "failed",
    "abandoned",
]
type WorkflowInputPolicy = Literal["pause", "unattended"]
type WorkflowStageStatus = Literal[
    "started", "done", "needsInput", "abstained", "failed"
]
type WorkflowStageName = Literal[
    "ingest",
    "data_enrichment",
    "rna_quality_metrics",
    "experimental_context",
    "preprocessing_plan",
    "preprocessing",
    "parameter_tuning",
    "analysis_finalization",
    "report",
]
type AssayRole = Literal["graph", "hto", "unsupported"]
type ReductionMethod = Literal["pca", "lsi", "identity", "none"]

_RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,127}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_STAGE_ORDER: tuple[WorkflowStageName, ...] = (
    "ingest",
    "data_enrichment",
    "rna_quality_metrics",
    "experimental_context",
    "preprocessing_plan",
    "preprocessing",
    "parameter_tuning",
    "analysis_finalization",
)


@dataclass(frozen=True)
class WorkflowIdentity:
    """Runtime address of a workflow whose state belongs to its journal."""

    workflowRunId: str
    workspace: str | None = None


class StageEvidenceReference(AgentDataModel):
    """Exact evidence checkpoint owned by an orchestration stage."""

    workflowRunId: str
    stage: WorkflowStageName
    key: str
    contentSha256: str


class AnalysisError(RuntimeError):
    """An unattended analysis could not produce a supported completed result."""

    def __init__(self, result: "AutomatedWorkflowResult") -> None:
        self.result = result
        detail = "; ".join(result.notes) or "Required scientific evidence is unresolved"
        address = (
            f" Resume workflow {result.workflowRunId!r} in {result.zarrPath!r}"
            f" (workspace={result.workspace!r})."
            if result.workflowRunId and result.zarrPath
            else ""
        )
        super().__init__(
            f"RNA analysis {result.status} during {result.currentStage}: {detail}.{address}"
        )


class WorkflowQuestion(AgentDataModel):
    """One stable question that can be answered by a resume request."""

    questionId: str = ""
    decisionId: str | None = None
    question: str = ""
    options: list[str] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)
    planChecksum: str | None = None

    @classmethod
    def get_blank(cls) -> "WorkflowQuestion":
        return cls()


class WorkflowNeedsInput(AgentDataModel):
    """All questions blocking the next workflow stage."""

    questions: list[WorkflowQuestion] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "WorkflowNeedsInput":
        return cls()


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
    reportReferences: list[StageEvidenceReference] = Field(default_factory=list)
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
    evidenceIds: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "AssayPreprocessingPlan":
        return cls()


class AutomatedPreprocessingPlan(AgentDataModel):
    """Dataset-wide preprocessing plan produced before selection changes."""

    primaryAssay: str = ""
    markerAssay: str = ""
    cellSelection: ArtifactReferenceModel | None = None
    cellQc: CellQcPlan = Field(default_factory=CellQcPlan.get_blank)
    cellQualityPayload: CellQualityExecutorPayload | None = None
    assays: list[AssayPreprocessingPlan] = Field(default_factory=list)
    pairedAssays: list[str] = Field(default_factory=list)
    planChecksum: str = ""
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_cell_quality_payload(self) -> "AutomatedPreprocessingPlan":
        if (
            self.cellQualityPayload is not None
            and cell_qc_policy(self.cellQc.action, self.cellQc.registeredProfile)
            != self.cellQualityPayload.profile
        ):
            raise ValueError(
                "cellQualityPayload must match the exact selected QC policy"
            )
        return self

    @classmethod
    def get_blank(cls) -> "AutomatedPreprocessingPlan":
        return cls()


class PreprocessedAssayHandoff(AgentDataModel):
    """Exact normalized input and selections handed to Parameter Tuning."""

    assay: str = ""
    assayType: str = "Assay"
    cellSelection: ArtifactReferenceModel | None = None
    reductionMethod: ReductionMethod = "none"
    graphFeatures: ArtifactReferenceModel | None = None
    markerFeatures: ArtifactReferenceModel | None = None
    normalized: ArtifactReferenceModel | None = None
    graphFeatureCandidates: dict[str, ArtifactReferenceModel] = Field(
        default_factory=dict
    )
    normalizedCandidates: dict[str, ArtifactReferenceModel] = Field(
        default_factory=dict
    )
    featureCandidateEvaluations: list[dict[str, Any]] = Field(default_factory=list)
    nCells: int = 0
    nFeatures: int = 0

    @classmethod
    def get_blank(cls) -> "PreprocessedAssayHandoff":
        return cls()


class FinalAnalysisHandoff(AgentDataModel):
    """Exact final RNA artifacts validated by the concluding journal checkpoint."""

    workflowRunId: str = ""
    primaryAssay: str = ""
    markerAssay: str = ""
    cellSelection: ArtifactReferenceModel | None = None
    graph: ArtifactReferenceModel | None = None
    graphMethod: Literal["native"] = "native"
    clusters: ArtifactReferenceModel | None = None
    embeddingInitialization: ArtifactReferenceModel | None = None
    umap: ArtifactReferenceModel | None = None
    markerFeatures: ArtifactReferenceModel | None = None
    markers: ArtifactReferenceModel | None = None
    doubletScores: list[ArtifactReferenceModel] = Field(default_factory=list)
    doubletScoreSelections: list[ArtifactReferenceModel] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "FinalAnalysisHandoff":
        return cls()


class AutomatedWorkflowConfig(AgentDataModel):
    """Bounded execution policy for automated workflows."""

    inputPolicy: WorkflowInputPolicy = Field(
        default="pause",
        exclude_if=lambda value: value == "pause",
    )
    # Older requests always scored doublets. Keep their serialized defaults exact.
    scoreDoublets: bool = Field(
        default=True,
        strict=True,
        exclude_if=lambda value: value is True,
        description="Score advisory doublets; Harmony-eligible runs always retain required doublet diagnostics. The beginner API defaults to false; absent saved fields retain enabled scoring.",
    )
    screeningCells: int | None = Field(
        default=None,
        ge=20,
        description="Fixed initial screening size; null uses 10% of retained cells bounded to 10,000–100,000.",
    )
    maxScreeningCells: int = Field(default=100_000, ge=20)
    maxScreeningEvaluations: int = Field(default=24, ge=4)
    maxTotalScreeningEvaluations: int = Field(default=48, ge=4)
    maxFullGraphs: int = Field(default=4, ge=1)
    maxFullPartitions: int = Field(default=8, ge=1)
    maxFullRepairs: int = Field(default=1, ge=0, le=1)
    randomSeed: int = Field(default=4444, ge=0)
    allowDownloads: bool = False
    cacheDir: str | None = None
    agentRunConfig: AgentRunConfig = Field(default_factory=AgentRunConfig)

    @model_validator(mode="before")
    @classmethod
    def reject_obsolete_configuration(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            obsolete = sorted(
                set(value)
                & {
                    "maxRefinedCandidatesPerAssay",
                    "maxHarmonyCandidatesPerAssay",
                    "runConfoundedHarmonyDiagnostic",
                    "maxCandidateEvaluations",
                    "maxIdentityFeatures",
                    "minClusterCells",
                    "hvgCandidateCounts",
                    "pcaCandidateDimensions",
                    "graphNeighborCandidates",
                    "leidenResolutionCandidates",
                    "maxRevisions",
                    "primaryInitialCandidates",
                    "secondaryInitialCandidates",
                    "integrationResolutionCandidates",
                    "maxCandidateBranches",
                    "maxGraphAssays",
                    "leidenSeeds",
                    "clusterSubsamples",
                    "clusterSubsampleFraction",
                }
            )
            if obsolete:
                raise ValueError(
                    "Unsupported legacy workflow configuration fields: "
                    + ", ".join(obsolete)
                    + ". Create a new single-RNA workflow configuration with "
                    "screening and full-cohort work limits. "
                    "Saved workflows using these fields cannot be resumed or "
                    "regenerated with this release; their analysis artifacts "
                    "remain available through Scarf's artifact APIs."
                )
        return value

    @model_validator(mode="after")
    def validate_work_limits(self) -> "AutomatedWorkflowConfig":
        if (
            self.screeningCells is not None
            and self.maxScreeningCells < self.screeningCells
        ):
            raise ValueError("maxScreeningCells must be at least screeningCells")
        if self.maxTotalScreeningEvaluations < self.maxScreeningEvaluations:
            raise ValueError(
                "Whole-workflow screening allowance must cover one screening population"
            )
        return self

    @classmethod
    def get_blank(cls) -> "AutomatedWorkflowConfig":
        return cls()


class AutomatedWorkflowRequest(AgentDataModel):
    """Immutable request for one automated analysis."""

    sourcePath: str = ""
    zarrPath: str | None = None
    studyContext: str = ""
    studyObjective: str = ""
    authorLabelPolicy: AuthorLabelPolicy = "holdout"
    workspace: str | None = None
    primaryAssay: str | None = None
    markerAssay: str | None = None
    analysisAssays: list[str] = Field(default_factory=list)
    pairedAssays: list[str] = Field(default_factory=list)
    ingestDirections: dict[str, Any] = Field(default_factory=dict)
    experimentalDirections: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_request(self) -> "AutomatedWorkflowRequest":
        if not self.sourcePath.strip():
            raise ValueError("sourcePath must be non-empty")
        if not self.studyContext.strip():
            raise ValueError("studyContext must be non-empty")
        if not self.studyObjective.strip():
            raise ValueError("studyObjective must be non-empty")
        if len(set(self.analysisAssays)) != len(self.analysisAssays):
            raise ValueError("analysisAssays must be unique")
        from .rna import validate_rna_request_fields

        validate_rna_request_fields(self)
        return self

    @classmethod
    def get_blank(cls) -> "AutomatedWorkflowRequest":
        return cls(
            sourcePath="dataset.h5",
            studyContext="Study context",
            studyObjective="Discover stable population structure.",
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


class AutomatedWorkflowResult(AgentDataModel):
    """Small result address; scientific evidence remains in the stage journal."""

    status: AutomatedWorkflowStatus = "failed"
    currentStage: WorkflowStageName = "ingest"
    zarrPath: str | None = None
    workspace: str | None = None
    workflowRunId: str | None = None
    needsInput: WorkflowNeedsInput | None = None
    limitations: list[str] = Field(default_factory=list)
    unresolvedClaims: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def _analysis_store(self) -> "DataStore":
        from .journal import open_analysis_store

        if self.status != "completed":
            raise AnalysisError(self)
        if self.zarrPath is None or self.workflowRunId is None:
            raise RuntimeError("Completed analysis lacks its exact store and workflow")
        return open_analysis_store(
            self.zarrPath, self.workflowRunId, workspace=self.workspace
        )

    def _completed_analysis(self, store: "DataStore") -> FinalAnalysisHandoff:
        from .journal import analysis_snapshot

        assert self.workflowRunId is not None
        snapshot = analysis_snapshot(store, self.workflowRunId)
        if snapshot["status"] != "completed":
            raise RuntimeError(
                "The referenced workflow has no validated final analysis"
            )
        return FinalAnalysisHandoff.model_validate(snapshot["finalAnalysis"])

    def plot_embedding(self, **kwargs: Any) -> "PlotResult":
        """Display the saved cluster map with bounded reads and full-count provenance."""
        from .._plots import plot_final_umap

        store = self._analysis_store()
        final = self._completed_analysis(store)
        if (
            final.umap is None
            or final.clusters is None
            or final.cellSelection is None
            or final.graph is None
        ):
            raise RuntimeError("Completed analysis lacks its final map artifacts")
        for name in (
            "layout",
            "run",
            "color_by",
            "umap",
            "clusters",
            "cell_selection",
            "graph",
        ):
            if name in kwargs:
                raise ValueError("plot_embedding uses the exact completed cluster map")
        return plot_final_umap(
            store,
            umap=artifact_model_to_ref(final.umap),
            clusters=artifact_model_to_ref(final.clusters),
            cell_selection=artifact_model_to_ref(final.cellSelection),
            graph=artifact_model_to_ref(final.graph),
            **kwargs,
        )

    def get_markers(
        self,
        *,
        group_id: str | int | None = None,
        min_score: float = 0.25,
        min_frac_exp: float = 0.2,
    ) -> "pd.DataFrame":
        """Load the final markers using Scarf's established marker filters."""
        store = self._analysis_store()
        final = self._completed_analysis(store)
        if final.markers is None:
            raise RuntimeError("Completed analysis lacks its marker artifact")
        return store.get_markers(
            marker=artifact_model_to_ref(final.markers),
            group_id=group_id,
            min_score=min_score,
            min_frac_exp=min_frac_exp,
        )

    def report(self) -> Path:
        """Return or regenerate the compact report from saved evidence only."""
        from ..report.generator import generate_agent_report

        store = self._analysis_store()
        self._completed_analysis(store)
        assert self.workflowRunId is not None
        return generate_agent_report(store, self.workflowRunId)

    @classmethod
    def get_blank(cls) -> "AutomatedWorkflowResult":
        return cls()


class OrchestrationRequestRecord(AgentDataModel):
    """Stored immutable request and effective configuration."""

    recordType: Literal["automatedWorkflowRequest"] = "automatedWorkflowRequest"
    inputIdentity: dict[str, Any]
    modelIdentity: str
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


class OrchestrationResumeRecord(AgentDataModel):
    """Runtime answers committed as inputs of their owning stage attempt."""

    workflowRunId: str = ""
    answeredAttempt: WorkflowStageLink | None = None
    questionIds: list[str] = Field(default_factory=list)
    answers: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def get_blank(cls) -> "OrchestrationResumeRecord":
        return cls()


def artifact_model_to_ref(value: ArtifactReferenceModel) -> ArtifactRef:
    """Convert an agent artifact model to a validated core artifact reference."""
    return ArtifactRef(
        scope=value.scope,
        assay=value.assay,
        kind=value.kind,
        artifact_id=value.artifactId,
    )
