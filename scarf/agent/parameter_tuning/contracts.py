import re
from threading import Lock
from typing import Any, Literal

from .._deps import AGENT_INSTALL_HINT
from ..types import (
    AgentDataModel,
    AgentRunInfo,
    ArtifactReferenceModel,
    StageStatus,
)

try:
    from pydantic import Field
    from pydantic.json_schema import SkipJsonSchema
except ImportError as exc:
    raise ImportError(AGENT_INSTALL_HINT) from exc


type CandidateStatus = Literal["done", "failed"]
type TuningConfidence = Literal["low", "medium", "high"]

_CANDIDATE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_]{0,63}$")


class ArtifactRecord(ArtifactReferenceModel):
    """JSON-safe identity for one artifact returned by candidate execution."""

    @classmethod
    def from_ref(cls, ref: Any) -> "ArtifactRecord":
        return cls(
            scope=getattr(ref, "scope", "assay"),
            kind=str(getattr(ref, "kind", "")),
            artifactId=str(getattr(ref, "artifact_id", ref)),
            assay=getattr(ref, "assay", None),
        )


class ParameterCandidate(AgentDataModel):
    """One exact, caller-authorized parameter candidate."""

    candidateId: str = Field(
        default="",
        description="Exact candidate id supplied to candidate execution",
    )
    reductionMethod: Literal["pca"] = "pca"
    dimensions: int = Field(default=21, ge=2)
    leidenResolution: float = Field(default=1.0, gt=0)
    neighborsK: int = Field(default=11, ge=2)
    useHarmony: bool = False


class ParameterMetrics(AgentDataModel):
    """Bounded quality metrics for one candidate branch."""

    nClusters: int | None = None
    minClusterCells: int | None = None
    minClusterFraction: float | None = None
    graphSilhouetteMedian: float | None = None
    pcaSilhouette: float | None = None
    macroF1: float | None = None
    weightedF1: float | None = None
    membershipStrengthMean: float | None = None
    membershipStrengthMedian: float | None = None
    membershipStrengthP10: float | None = None
    membershipStrengthByCluster: dict[str, float] = Field(default_factory=dict)
    membershipStrengthSampleSize: int | None = None
    clusterConnectivity: float | None = None
    seedStability: float | None = None
    subsampleStability: float | None = None
    markerCoherence: float | None = None
    markerSpecificityMedian: float | None = None
    markerSpecificityByCluster: dict[str, float] = Field(default_factory=dict)
    markerAucByCluster: dict[str, float] = Field(default_factory=dict)
    topMarkerGenes: dict[str, list[str]] = Field(default_factory=dict)
    crossUnitSupport: float | None = None
    technicalAssociation: dict[str, float] = Field(default_factory=dict)
    componentVariance: list[float] = Field(default_factory=list)
    pcaExplainedVarianceRatio: list[float] = Field(default_factory=list)
    pcaCumulativeExplainedVarianceRatio: list[float] = Field(default_factory=list)
    topLoadingGenes: dict[str, list[str]] = Field(default_factory=dict)
    loadingFamilyEnrichment: dict[str, float] = Field(default_factory=dict)
    loadingFamilyEnrichmentByComponent: dict[str, dict[str, float]] = Field(
        default_factory=dict
    )
    pcaComponentAssociations: dict[str, dict[str, list[float]]] = Field(
        default_factory=dict
    )
    batchPcaAssociation: dict[str, float] = Field(default_factory=dict)
    technicalPcaAssociation: dict[str, float] = Field(default_factory=dict)
    protectedPcaAssociation: dict[str, float] = Field(default_factory=dict)
    qcPcaAssociation: dict[str, float] = Field(default_factory=dict)
    markerFamilyEnrichment: dict[str, float] = Field(default_factory=dict)
    protectedMarkerFamilies: list[str] = Field(default_factory=list)
    doubletHighScoreConcentration: float | None = None
    doubletScoreQuantiles: dict[str, float] = Field(default_factory=dict)
    doubletScoreByCapture: dict[str, dict[str, float]] = Field(default_factory=dict)
    doubletCaptureCoverage: float | None = None
    batchMixing: dict[str, float] = Field(default_factory=dict)
    biologicalPreservation: dict[str, dict[str, float]] = Field(default_factory=dict)


class ParameterCandidateEvaluation(AgentDataModel):
    """Execution record returned to the model for one candidate."""

    candidateId: str = ""
    harmonyBatchColumns: list[str] = Field(default_factory=list)
    status: CandidateStatus = "failed"
    eligible: bool = False
    parameters: ParameterCandidate = Field(default_factory=ParameterCandidate.get_blank)
    artifacts: dict[str, ArtifactRecord] = Field(default_factory=dict)
    cellSelection: ArtifactReferenceModel | None = None
    clusterColumn: str | None = None
    clusterLabel: str | None = None
    effectiveDimensions: int | None = None
    metrics: ParameterMetrics = Field(default_factory=ParameterMetrics.get_blank)
    evidenceIds: list[str] = Field(default_factory=list)
    eligibilityReasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class ParameterTuningNeedsInput(AgentDataModel):
    """User input required before tuning can produce a recommendation."""

    question: str = ""
    options: list[str] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)


class ParameterTuningReport(AgentDataModel):
    """Grounded recommendation over candidate branches actually executed."""

    status: StageStatus = "failed"
    fromAssay: SkipJsonSchema[str] = ""
    cellSelection: SkipJsonSchema[ArtifactReferenceModel | None] = None
    evaluations: SkipJsonSchema[list[ParameterCandidateEvaluation]] = Field(
        default_factory=list
    )
    recommendedCandidateId: str | None = None
    selectedArtifacts: SkipJsonSchema[dict[str, ArtifactRecord]] = Field(
        default_factory=dict
    )
    confidence: TuningConfidence = "low"
    rationale: str = ""
    evidenceIds: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    stopReason: str = ""
    needsInput: ParameterTuningNeedsInput | None = None
    recommendedByAssay: SkipJsonSchema[dict[str, str]] = Field(default_factory=dict)
    totalCandidates: SkipJsonSchema[int] = 0
    finalClusterColumn: SkipJsonSchema[str | None] = None
    finalClusterArtifact: SkipJsonSchema[ArtifactRecord | None] = None
    graphAssay: SkipJsonSchema[str | None] = None
    markerAssay: SkipJsonSchema[str | None] = None
    runInfo: SkipJsonSchema[AgentRunInfo] = Field(default_factory=AgentRunInfo)


class ParameterTuningDependencies(AgentDataModel):
    """Runtime-only state hidden from the model and shared by tuning tools."""

    store: Any = Field(default=None, exclude=True)
    normalized: Any = Field(default=None, exclude=True)
    cellSelection: Any = Field(default=None, exclude=True)
    normalizedShape: tuple[int, int] | None = None
    fromAssay: str = ""
    candidates: dict[str, ParameterCandidate] = Field(default_factory=dict)
    batchColumns: tuple[str, ...] = ()
    preservationColumns: tuple[str, ...] = ()
    protectedCombinations: tuple[tuple[str, ...], ...] = ()
    columnKinds: dict[str, Literal["categorical", "continuous"]] = Field(
        default_factory=dict
    )
    maxCandidates: int = 5
    minClusterCells: int = 20
    evaluations: dict[str, ParameterCandidateEvaluation] = Field(default_factory=dict)
    executionOrder: list[str] = Field(default_factory=list)
    executionLock: Any = Field(default_factory=Lock, exclude=True, repr=False)
