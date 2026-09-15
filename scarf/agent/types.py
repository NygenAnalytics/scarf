"""Shared Pydantic data structures for Scarf agents."""

from typing import Any, Literal

from ._deps import AGENT_INSTALL_HINT

try:
    from pydantic import BaseModel, ConfigDict, Field
except ImportError as exc:
    raise ImportError(AGENT_INSTALL_HINT) from exc


type StageStatus = Literal["done", "needsInput", "abstained", "failed"]
type BatchCorrectionAction = Literal[
    "skip",
    "evaluateHarmony",
    "unsafe",
    "needsInput",
]
type BatchSafetyStatus = Literal["safe", "unsafe", "notComputed"]


class AgentDataModel(BaseModel):
    """Base class for strict, serializable agent data structures."""

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def get_blank(cls) -> "AgentDataModel":
        """Return an empty but valid value for fallback paths."""
        return cls()


class ArtifactReferenceModel(AgentDataModel):
    type: Literal["artifact"] = "artifact"
    scope: Literal["assay", "datastore"] = "assay"
    assay: str | None = None
    kind: str = ""
    artifactId: str = ""

    @classmethod
    def from_artifact_ref(cls, ref: Any) -> "ArtifactReferenceModel":
        """Convert a core ArtifactRef without making it an agent dependency."""
        return cls(
            scope=getattr(ref, "scope", "assay"),
            assay=getattr(ref, "assay", None),
            kind=str(getattr(ref, "kind", "")),
            artifactId=str(getattr(ref, "artifact_id", "")),
        )


class BatchSafetyEvidence(AgentDataModel):
    """Estimability for one coefficient and exact proposed batch-column set."""

    coefficient: str = ""
    coefficientKind: Literal["categorical", "continuous"] | None = None
    observationUnit: str | None = None
    batchColumns: list[str] = Field(default_factory=list)
    unitConstantBatchColumns: list[str] = Field(default_factory=list)
    status: BatchSafetyStatus = "notComputed"
    estimability: dict[str, Any] = Field(default_factory=dict)
    evidenceId: str = ""

    @classmethod
    def get_blank(cls) -> "BatchSafetyEvidence":
        return cls()


class ExperimentalTuningHandoff(AgentDataModel):
    """Validated Experimental Context inputs for Parameter Tuning."""

    cellSelection: ArtifactReferenceModel | None = None
    batchAction: BatchCorrectionAction = "needsInput"
    batchColumns: list[str] = Field(default_factory=list)
    preservationColumns: list[str] = Field(default_factory=list)
    coefficientsOfInterest: list[str] = Field(default_factory=list)
    batchSafety: list[BatchSafetyEvidence] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)


class ExperimentalBiologyHandoff(AgentDataModel):
    """One explicitly selected experimental coefficient for interpretation."""

    cellSelection: ArtifactReferenceModel | None = None
    conditionColumn: str = ""
    observationUnit: str | None = None
    independentUnit: str | None = None
    coefficientScope: str = ""
    estimability: dict[str, Any] = Field(default_factory=dict)
    evidenceIds: list[str] = Field(default_factory=list)


class TuningBiologyHandoff(AgentDataModel):
    """Exact selected clustering branch for Biological Interpretation."""

    cellSelection: ArtifactReferenceModel | None = None
    fromAssay: str = ""
    graphAssay: str | None = None
    markerAssay: str | None = None
    recommendedCandidateId: str = ""
    clusterArtifact: ArtifactReferenceModel | None = None
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "TuningBiologyHandoff":
        return cls()


class ToolCallInfo(AgentDataModel):
    toolName: str = ""
    callId: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)


class AgentUsageInfo(AgentDataModel):
    inputTokens: int = 0
    outputTokens: int = 0
    totalTokens: int = 0
    requests: int = 0
    toolCalls: int = 0
    availability: Literal["reported", "partial", "unavailable"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class AgentValidationRetry(AgentDataModel):
    """A rejected response or tool call and the feedback supplied for repair."""

    source: Literal["output", "tool", "schema"]
    requestIndex: int = 0
    message: str
    response: dict[str, Any] | str | None = None


class AgentProviderFailure(AgentDataModel):
    """One failed model request, distinct from rejected scientific output."""

    requestIndex: int
    statusCode: int | None = None
    error: str
    retryDelaySeconds: float | None = None


class AgentRunInfo(AgentDataModel):
    agentName: str = ""
    modelName: str = ""
    runId: str = ""
    durationSeconds: float = 0.0
    usage: AgentUsageInfo = Field(default_factory=AgentUsageInfo)
    toolCalls: list[ToolCallInfo] = Field(default_factory=list)
    status: Literal["done", "failed"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    validationRetries: list[AgentValidationRetry] = Field(
        default_factory=list, exclude_if=lambda value: not value
    )
    providerFailures: list[AgentProviderFailure] = Field(
        default_factory=list, exclude_if=lambda value: not value
    )
    errorType: str | None = Field(default=None, exclude_if=lambda value: value is None)
    error: str | None = Field(default=None, exclude_if=lambda value: value is None)


class AgentExecutionResult(AgentDataModel):
    output: Any = None
    runInfo: AgentRunInfo = Field(default_factory=AgentRunInfo)


class EvidenceItem(AgentDataModel):
    id: str
    label: str
    summary: str

    @classmethod
    def get_blank(cls) -> "EvidenceItem":
        return cls(id="", label="", summary="")


class Decision(AgentDataModel):
    selectedId: str = Field(
        description="Exact evidence id string from the provided list, nothing else"
    )
    rationale: str = Field(description="Short reason for the choice")
    evidenceIds: list[str] = Field(
        default_factory=list,
        description="Evidence ids used; must include selectedId and only provided ids",
    )

    @classmethod
    def get_blank(cls) -> "Decision":
        return cls(selectedId="", rationale="")


class NeedsInput(AgentDataModel):
    question: str
    options: list[str] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "NeedsInput":
        return cls(question="")


class StageResult(AgentDataModel):
    status: StageStatus
    decision: Decision | None = None
    needsInput: NeedsInput | None = None
    actions: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "StageResult":
        return cls(status="needsInput")
