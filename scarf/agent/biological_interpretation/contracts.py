"""Serializable contracts for biological interpretation."""

from typing import Any, Literal

from pydantic import Field
from pydantic.json_schema import SkipJsonSchema

from ..types import (
    AgentDataModel,
    AgentRunInfo,
    ArtifactReferenceModel,
    ExperimentalBiologyHandoff,
    StageStatus,
)

type InterpretationConfidence = Literal["low", "medium", "high"]
type TreatmentDirection = Literal["higher", "lower", "equal"]

_MAX_CLUSTERS = 20
_MAX_CONDITIONS = 30
_MAX_MARKERS = 25


class BiologicalContext(AgentDataModel):
    """Caller-supplied facts that constrain biological interpretation."""

    organism: str = ""
    studyContext: str = ""
    tissue: str = ""
    cellTypeReferences: list[str] = Field(default_factory=list)
    experimentalDetails: list[str] = Field(default_factory=list)
    treatmentQuestion: str = ""

    @classmethod
    def get_blank(cls) -> "BiologicalContext":
        return cls()


class ConditionClusterSummary(AgentDataModel):
    """Aggregate cluster abundance for one condition without sample identifiers."""

    condition: str = ""
    clusterId: str = ""
    nSamples: int = 0
    meanFraction: float = 0.0
    minFraction: float = 0.0
    maxFraction: float = 0.0
    cellCount: int = 0
    evidenceId: str = ""


class ClusterCompositionEvidence(AgentDataModel):
    """Bounded deterministic evidence about cluster sizes and conditions."""

    clusterArtifact: ArtifactReferenceModel | None = None
    cellSelection: ArtifactReferenceModel | None = None
    totalCells: int = 0
    clusterCounts: dict[str, int] = Field(default_factory=dict)
    sampleColumn: str | None = None
    conditionColumn: str | None = None
    conditionSummaries: list[ConditionClusterSummary] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class MarkerFeature(AgentDataModel):
    """One observed marker feature and its available Scarf statistics."""

    featureId: str = ""
    featureName: str = ""
    featureIndex: int | None = None
    score: float | None = None
    foldChange: float | None = None
    fractionExpressed: float | None = None
    fractionExpressedRest: float | None = None
    mean: float | None = None
    meanRest: float | None = None
    auc: float | None = None
    adjustedPvalue: float | None = None


class ClusterMarkerEvidence(AgentDataModel):
    """Bounded markers for one exact cluster label."""

    clusterId: str = ""
    markers: list[MarkerFeature] = Field(default_factory=list)
    markerArtifact: ArtifactReferenceModel | None = None
    evidenceId: str = ""
    warnings: list[str] = Field(default_factory=list)


class ClusterMarkerBatchEvidence(AgentDataModel):
    """Markers for all model-selected clusters returned by one tool call."""

    clusters: list[ClusterMarkerEvidence] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "ClusterMarkerBatchEvidence":
        return cls()


class ClusterInterpretation(AgentDataModel):
    """One evidence-linked cluster interpretation or hypothesis."""

    clusterId: str = ""
    proposedIdentity: str = "unresolved"
    identityIsHypothesis: bool = True
    confidence: InterpretationConfidence = "low"
    rationale: str = ""
    evidenceIds: list[str] = Field(default_factory=list)


class TreatmentObservation(AgentDataModel):
    """Descriptive treatment observation with no unsupported causal claim."""

    clusterId: str = ""
    referenceCondition: str = ""
    comparisonCondition: str = ""
    direction: TreatmentDirection = "equal"
    observation: str = ""
    isDescriptiveOnly: Literal[True] = True
    evidenceIds: list[str] = Field(default_factory=list)


class FollowUpRecommendation(AgentDataModel):
    """A bounded next analysis tied to an observed uncertainty."""

    question: str = ""
    operation: str = ""
    rationale: str = ""
    requiredInputs: list[str] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)


class BiologicalInterpretationNeedsInput(AgentDataModel):
    question: str = ""
    requiredInputs: list[str] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)


class BiologicalInterpretationReport(AgentDataModel):
    """Structured, evidence-grounded biological review."""

    status: StageStatus = "needsInput"
    clusterInterpretations: list[ClusterInterpretation] = Field(default_factory=list)
    treatmentObservations: list[TreatmentObservation] = Field(default_factory=list)
    followUps: list[FollowUpRecommendation] = Field(default_factory=list)
    clusterArtifact: SkipJsonSchema[ArtifactReferenceModel | None] = None
    markerArtifact: SkipJsonSchema[ArtifactReferenceModel | None] = None
    graphAssay: SkipJsonSchema[str | None] = None
    markerAssay: SkipJsonSchema[str | None] = None
    evidenceIds: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    stopReason: str = ""
    needsInput: BiologicalInterpretationNeedsInput | None = None
    runInfo: SkipJsonSchema[AgentRunInfo] = Field(default_factory=AgentRunInfo)


class BiologicalInterpretationDependencies(AgentDataModel):
    """Runtime state available only to biological interpretation tools."""

    store: Any = Field(default=None, exclude=True)
    cluster: Any = Field(default=None, exclude=True)
    cellSelection: Any = Field(default=None, exclude=True)
    cellIndices: Any = Field(default=None, exclude=True)
    fromAssay: str | None = None
    graphAssay: str | None = None
    markerAssay: str | None = None
    markerAssayType: str | None = None
    sampleColumn: str | None = None
    conditionColumn: str | None = None
    marker: Any = Field(default=None, exclude=True)
    markerFeatures: Any = Field(default=None, exclude=True)
    allowMarkerSearch: bool = False
    maxClusters: int = 12
    maxMarkers: int = 10
    markerMinScore: float = 0.25
    markerMinFraction: float = 0.2
    evidenceIds: set[str] = Field(default_factory=set, exclude=True)
    clusterValues: dict[str, Any] = Field(default_factory=dict, exclude=True)
    markerEvidenceIds: dict[str, str] = Field(default_factory=dict, exclude=True)
    markerEvidence: dict[str, ClusterMarkerEvidence] = Field(
        default_factory=dict,
        exclude=True,
    )
    compositionEvidence: ClusterCompositionEvidence | None = Field(
        default=None,
        exclude=True,
    )
    markerBatch: ClusterMarkerBatchEvidence | None = Field(
        default=None,
        exclude=True,
    )
    markerBatchClusterIds: list[str] = Field(default_factory=list, exclude=True)
    toolCalls: list[str] = Field(default_factory=list, exclude=True)
    conditionEvidence: dict[str, ConditionClusterSummary] = Field(
        default_factory=dict,
        exclude=True,
    )
    designHandoff: ExperimentalBiologyHandoff | None = Field(
        default=None,
        exclude=True,
    )
