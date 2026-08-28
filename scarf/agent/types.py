"""Shared Pydantic data structures for Scarf agents."""

from typing import Any, Literal

from .config._deps import AGENT_INSTALL_HINT

try:
    from pydantic import BaseModel, ConfigDict, Field
except ImportError as exc:
    raise ImportError(AGENT_INSTALL_HINT) from exc


type StageStatus = Literal["done", "needsInput", "failed"]
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

    @classmethod
    def get_example(cls) -> "AgentDataModel":
        """Return a small representative value for tests and fixtures."""
        return cls.get_blank()


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

    @classmethod
    def get_example(cls) -> "ArtifactReferenceModel":
        return cls(
            assay="RNA",
            kind="reduction",
            artifactId="0" * 64,
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

    @classmethod
    def get_example(cls) -> "BatchSafetyEvidence":
        return cls(
            coefficient="treatment",
            coefficientKind="categorical",
            observationUnit="sample",
            batchColumns=["batch"],
            unitConstantBatchColumns=["batch"],
            status="safe",
            estimability={
                "status": "ok",
                "coefficientEstimable": True,
                "rankDeficient": False,
            },
            evidenceId="batchEstimability:treatment:batch",
        )


class ExperimentalTuningHandoff(AgentDataModel):
    """Validated Experimental Context inputs for Parameter Tuning."""

    cellKey: str = "I"
    batchAction: BatchCorrectionAction = "needsInput"
    batchColumns: list[str] = Field(default_factory=list)
    preservationColumns: list[str] = Field(default_factory=list)
    coefficientsOfInterest: list[str] = Field(default_factory=list)
    batchSafety: list[BatchSafetyEvidence] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)


class ExperimentalBiologyHandoff(AgentDataModel):
    """One explicitly selected experimental coefficient for interpretation."""

    cellKey: str = "I"
    conditionColumn: str = ""
    observationUnit: str | None = None
    independentUnit: str | None = None
    coefficientScope: str = ""
    estimability: dict[str, Any] = Field(default_factory=dict)
    evidenceIds: list[str] = Field(default_factory=list)


class TuningBiologyHandoff(AgentDataModel):
    """Exact selected clustering branch for Biological Interpretation."""

    fromAssay: str = ""
    cellKey: str = "I"
    recommendedCandidateId: str = ""
    clusterColumn: str = ""
    clusterArtifact: ArtifactReferenceModel | None = None
    evidenceIds: list[str] = Field(default_factory=list)


class ToolCallInfo(AgentDataModel):
    toolName: str = ""
    callId: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def get_example(cls) -> "ToolCallInfo":
        return cls(toolName="inspect_store", callId="tool-call-1")


class AgentUsageInfo(AgentDataModel):
    inputTokens: int = 0
    outputTokens: int = 0
    totalTokens: int = 0
    requests: int = 0
    toolCalls: int = 0

    @classmethod
    def get_example(cls) -> "AgentUsageInfo":
        return cls(
            inputTokens=100,
            outputTokens=50,
            totalTokens=150,
            requests=2,
            toolCalls=1,
        )


class AgentRunInfo(AgentDataModel):
    agentName: str = ""
    modelName: str = ""
    runId: str = ""
    durationSeconds: float = 0.0
    usage: AgentUsageInfo = Field(default_factory=AgentUsageInfo)
    toolCalls: list[ToolCallInfo] = Field(default_factory=list)

    @classmethod
    def get_example(cls) -> "AgentRunInfo":
        return cls(
            agentName="data_enrichment",
            modelName="example-model",
            runId="example-run",
            durationSeconds=0.1,
            usage=AgentUsageInfo.get_example(),
            toolCalls=[ToolCallInfo.get_example()],
        )


class AgentExecutionResult(AgentDataModel):
    output: Any = None
    runInfo: AgentRunInfo = Field(default_factory=AgentRunInfo)

    @classmethod
    def get_example(cls) -> "AgentExecutionResult":
        return cls(output={}, runInfo=AgentRunInfo.get_example())


class EvidenceItem(AgentDataModel):
    id: str
    label: str
    summary: str

    @classmethod
    def get_blank(cls) -> "EvidenceItem":
        return cls(id="", label="", summary="")

    @classmethod
    def get_example(cls) -> "EvidenceItem":
        return cls(
            id="evidence:example",
            label="example",
            summary="A bounded observed fact.",
        )


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

    @classmethod
    def get_example(cls) -> "Decision":
        return cls(
            selectedId="evidence:example",
            rationale="The evidence directly answers the question.",
            evidenceIds=["evidence:example"],
        )


class NeedsInput(AgentDataModel):
    question: str
    options: list[str] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "NeedsInput":
        return cls(question="")

    @classmethod
    def get_example(cls) -> "NeedsInput":
        return cls(
            question="Which condition column should be used?",
            options=["condition", "treatment"],
        )


class StageResult(AgentDataModel):
    status: StageStatus
    decision: Decision | None = None
    needsInput: NeedsInput | None = None
    actions: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "StageResult":
        return cls(status="needsInput")

    @classmethod
    def get_example(cls) -> "StageResult":
        return cls(status="done", decision=Decision.get_example())
