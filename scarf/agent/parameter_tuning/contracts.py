import re
from threading import Lock
from typing import Any, Literal

from .._deps import AGENT_INSTALL_HINT
from ..types import (
    AgentDataModel,
    AgentRunInfo,
    ArtifactReferenceModel,
    ExperimentalTuningHandoff,
    StageStatus,
    TuningBiologyHandoff,
)

try:
    from pydantic import Field
    from pydantic.json_schema import SkipJsonSchema
except ImportError as exc:
    raise ImportError(AGENT_INSTALL_HINT) from exc


type CandidateStatus = Literal["done", "failed"]
type CandidatePhase = Literal["initial", "refined"]
type ParameterSearchStatus = Literal["complete", "refine"]
type TuningConfidence = Literal["low", "medium", "high"]
type ReductionMethod = Literal["pca", "lsi", "identity"]
type IntegrationMethod = Literal["snn", "wnn"]

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

    @classmethod
    def get_blank(cls) -> "ArtifactRecord":
        return cls()


class ParameterCandidate(AgentDataModel):
    """One exact, caller-authorized parameter candidate."""

    candidateId: str = Field(
        default="",
        description="Exact candidate id supplied to the evaluation tool",
    )
    reductionMethod: ReductionMethod = "pca"
    dimensions: int = Field(default=21, ge=2)
    leidenResolution: float = Field(default=1.0, gt=0)
    neighborsK: int = Field(default=11, ge=2)
    useHarmony: bool = False

    @classmethod
    def get_blank(cls) -> "ParameterCandidate":
        return cls()


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
    neighborPrefixOverlap: float | None = None
    markerFamilyEnrichment: dict[str, float] = Field(default_factory=dict)
    protectedMarkerFamilies: list[str] = Field(default_factory=list)
    doubletHighScoreConcentration: float | None = None
    doubletScoreQuantiles: dict[str, float] = Field(default_factory=dict)
    doubletScoreByCapture: dict[str, dict[str, float]] = Field(default_factory=dict)
    doubletCaptureCoverage: float | None = None
    batchMixing: dict[str, float] = Field(default_factory=dict)
    biologicalPreservation: dict[str, dict[str, float]] = Field(default_factory=dict)
    paretoOptimal: bool | None = None
    dominatedByCandidateIds: list[str] = Field(default_factory=list)
    dominatesCandidateIds: list[str] = Field(default_factory=list)
    dominanceMetrics: dict[str, list[str]] = Field(default_factory=dict)

    @classmethod
    def get_blank(cls) -> "ParameterMetrics":
        return cls()


class ParameterCandidateEvaluation(AgentDataModel):
    """Execution record returned to the model for one candidate."""

    candidateId: str = ""
    phase: CandidatePhase = "initial"
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

    @classmethod
    def get_blank(cls) -> "ParameterCandidateEvaluation":
        return cls()


class IntegrationMetrics(AgentDataModel):
    """Metrics that are valid for an integrated graph comparison."""

    nClusters: int | None = None
    minClusterCells: int | None = None
    minClusterFraction: float | None = None
    adjustedRandByAssay: dict[str, float] = Field(default_factory=dict)
    normalizedMutualInformationByAssay: dict[str, float] = Field(default_factory=dict)
    biologicalConnectivity: dict[str, float] = Field(default_factory=dict)
    modalityWeightsValid: bool | None = None

    @classmethod
    def get_blank(cls) -> "IntegrationMetrics":
        return cls()


class IntegrationCandidateEvaluation(AgentDataModel):
    """One executor-produced SNN or WNN graph and cluster evaluation."""

    integrationId: str = ""
    method: IntegrationMethod = "wnn"
    assays: list[str] = Field(default_factory=list)
    status: CandidateStatus = "failed"
    eligible: bool = False
    cellSelection: ArtifactReferenceModel | None = None
    resolution: float = Field(default=1.0, gt=0)
    graphArtifact: ArtifactRecord | None = None
    clusterArtifact: ArtifactRecord | None = None
    clusterColumn: str | None = None
    metrics: IntegrationMetrics = Field(default_factory=IntegrationMetrics.get_blank)
    evidenceIds: list[str] = Field(default_factory=list)
    eligibilityReasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None

    @classmethod
    def get_blank(cls) -> "IntegrationCandidateEvaluation":
        return cls()


class FinalGraphComparison(AgentDataModel):
    """Evidence-backed comparison against one eligible final graph option."""

    optionId: str = ""
    summary: str = ""
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "FinalGraphComparison":
        return cls()


class FinalGraphNeedsInput(AgentDataModel):
    """Concrete input needed before a final graph can be selected."""

    question: str = ""
    options: list[str] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "FinalGraphNeedsInput":
        return cls()


class FinalGraphSelection(AgentDataModel):
    """Grounded choice among selected native, SNN, and WNN graph options."""

    status: StageStatus = "needsInput"
    selectedOptionId: str | None = None
    graphMethod: SkipJsonSchema[Literal["native", "snn", "wnn"] | None] = None
    nativeAssay: SkipJsonSchema[str | None] = None
    nativeCandidateId: SkipJsonSchema[str | None] = None
    integrationId: SkipJsonSchema[str | None] = None
    markerAssay: SkipJsonSchema[str] = ""
    confidence: TuningConfidence = "low"
    rationale: str = ""
    evidenceIds: list[str] = Field(default_factory=list)
    comparisons: list[FinalGraphComparison] = Field(default_factory=list)
    tradeoffs: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    needsInput: FinalGraphNeedsInput | None = None
    runInfo: SkipJsonSchema[AgentRunInfo] = Field(default_factory=AgentRunInfo)

    @classmethod
    def get_blank(cls) -> "FinalGraphSelection":
        return cls()


class CandidateComparison(AgentDataModel):
    """Evidence-backed comparison against one executed non-selected candidate."""

    candidateId: str = ""
    summary: str = ""
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "CandidateComparison":
        return cls()


class ParameterSearchPlan(AgentDataModel):
    """Validated proposal for one bounded refinement pass."""

    status: ParameterSearchStatus = Field(
        default="complete",
        description=(
            "Summary derived from candidates: refine when candidates is non-empty "
            "and complete when it is empty"
        ),
    )
    candidates: list[ParameterCandidate] = Field(
        default_factory=list,
        description=(
            "Bounded unexecuted refinement candidates, or an empty list when the "
            "initial screen is complete"
        ),
    )
    basedOnCandidateIds: list[str] = Field(default_factory=list)
    harmonyBatchColumns: list[str] = Field(default_factory=list)
    objectives: list[str] = Field(default_factory=list)
    rationale: str = ""
    evidenceIds: list[str] = Field(default_factory=list)
    stoppingCriteria: list[str] = Field(default_factory=list)
    runInfo: SkipJsonSchema[AgentRunInfo] = Field(default_factory=AgentRunInfo)

    @classmethod
    def get_blank(cls) -> "ParameterSearchPlan":
        return cls()


class ParameterTuningBatchSearchPlan(AgentDataModel):
    """One bounded refinement plan for every assay in a batched screen."""

    assayPlans: dict[str, ParameterSearchPlan] = Field(default_factory=dict)
    runInfo: SkipJsonSchema[AgentRunInfo] = Field(default_factory=AgentRunInfo)

    @classmethod
    def get_blank(cls) -> "ParameterTuningBatchSearchPlan":
        return cls()


class ParameterTuningNeedsInput(AgentDataModel):
    """User input required before tuning can produce a recommendation."""

    question: str = ""
    options: list[str] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "ParameterTuningNeedsInput":
        return cls()


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
    comparisons: list[CandidateComparison] = Field(default_factory=list)
    tradeoffs: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    stopReason: str = ""
    needsInput: ParameterTuningNeedsInput | None = None
    searchPlan: SkipJsonSchema[ParameterSearchPlan | None] = None
    assayReports: dict[str, "ParameterTuningReport"] = Field(default_factory=dict)
    recommendedByAssay: SkipJsonSchema[dict[str, str]] = Field(default_factory=dict)
    totalCandidates: SkipJsonSchema[int] = 0
    integrationEvaluations: SkipJsonSchema[list[IntegrationCandidateEvaluation]] = (
        Field(default_factory=list)
    )
    recommendedIntegrationId: SkipJsonSchema[str | None] = None
    finalClusterColumn: SkipJsonSchema[str | None] = None
    finalClusterArtifact: SkipJsonSchema[ArtifactRecord | None] = None
    graphAssay: SkipJsonSchema[str | None] = None
    markerAssay: SkipJsonSchema[str | None] = None
    finalSelection: SkipJsonSchema[FinalGraphSelection | None] = None
    runInfo: SkipJsonSchema[AgentRunInfo] = Field(default_factory=AgentRunInfo)

    @classmethod
    def get_blank(cls) -> "ParameterTuningReport":
        return cls()

    def to_biological_handoff(
        self,
        *,
        marker_assay: str | None = None,
    ) -> TuningBiologyHandoff:
        """Return the exact selected clustering branch for interpretation."""
        if self.status != "done":
            raise ValueError(
                "Parameter Tuning must be done before creating a biology handoff"
            )
        if self.finalClusterArtifact is not None:
            if self.cellSelection is None:
                raise ValueError("Final branch lacks an exact cell selection")
            resolved_marker_assay = marker_assay or self.markerAssay
            if not resolved_marker_assay:
                raise ValueError(
                    "A marker assay is required for an integrated biology handoff"
                )
            if self.finalClusterArtifact.scope == "datastore":
                if self.finalClusterArtifact.assay is not None:
                    raise ValueError(
                        "A datastore-scoped cluster artifact must not name an assay"
                    )
            elif (
                self.graphAssay is not None
                and self.finalClusterArtifact.assay != self.graphAssay
            ):
                raise ValueError("Final cluster artifact does not match graphAssay")
            integration = next(
                (
                    item
                    for item in self.integrationEvaluations
                    if item.integrationId == self.recommendedIntegrationId
                ),
                None,
            )
            if self.finalSelection is not None:
                evidence_ids = self.finalSelection.evidenceIds
            elif integration is not None:
                evidence_ids = integration.evidenceIds
            else:
                prefix = f"candidate:{self.recommendedCandidateId}:"
                evidence_ids = [
                    evidence_id
                    for evidence_id in self.evidenceIds
                    if evidence_id.startswith(prefix)
                ]
            return TuningBiologyHandoff(
                cellSelection=self.cellSelection,
                fromAssay=self.fromAssay,
                graphAssay=self.graphAssay,
                markerAssay=resolved_marker_assay,
                recommendedCandidateId=(
                    self.recommendedIntegrationId
                    or (
                        self.finalSelection.nativeCandidateId
                        if self.finalSelection is not None
                        else None
                    )
                    or self.recommendedCandidateId
                    or "final"
                ),
                clusterArtifact=ArtifactReferenceModel.model_validate(
                    self.finalClusterArtifact.model_dump()
                ),
                evidenceIds=sorted(evidence_ids),
            )
        if self.recommendedCandidateId is None:
            raise ValueError(
                "Parameter Tuning must recommend a candidate before creating a "
                "biology handoff"
            )
        selected = next(
            (
                item
                for item in self.evaluations
                if item.candidateId == self.recommendedCandidateId
            ),
            None,
        )
        if selected is None or selected.status != "done" or not selected.eligible:
            raise ValueError("Recommended candidate is not an eligible execution")
        cluster_artifact = selected.artifacts.get("clusters")
        if cluster_artifact is None or selected.cellSelection is None:
            raise ValueError("Recommended candidate lacks an exact cluster artifact")
        if not self.fromAssay or cluster_artifact.assay != self.fromAssay:
            raise ValueError("Recommended cluster artifact does not match the assay")
        prefix = f"candidate:{selected.candidateId}:"
        return TuningBiologyHandoff(
            cellSelection=selected.cellSelection,
            fromAssay=self.fromAssay,
            graphAssay=self.fromAssay,
            markerAssay=marker_assay or self.markerAssay or self.fromAssay,
            recommendedCandidateId=selected.candidateId,
            clusterArtifact=ArtifactReferenceModel.model_validate(
                cluster_artifact.model_dump()
            ),
            evidenceIds=sorted(
                evidence_id
                for evidence_id in self.evidenceIds
                if evidence_id.startswith(prefix)
            ),
        )


class ParameterTuningDependencies(AgentDataModel):
    """Runtime-only state hidden from the model and shared by tuning tools."""

    store: Any = Field(default=None, exclude=True)
    normalized: Any = Field(default=None, exclude=True)
    cellSelection: Any = Field(default=None, exclude=True)
    normalizedShape: tuple[int, int] | None = None
    fromAssay: str = ""
    candidates: dict[str, ParameterCandidate] = Field(default_factory=dict)
    candidatePhases: dict[str, CandidatePhase] = Field(default_factory=dict)
    batchColumns: tuple[str, ...] = ()
    preservationColumns: tuple[str, ...] = ()
    protectedCombinations: tuple[tuple[str, ...], ...] = ()
    columnKinds: dict[str, Literal["categorical", "continuous"]] = Field(
        default_factory=dict
    )
    harmonyAuthorized: bool = False
    maxCandidates: int = 5
    minClusterCells: int = 20
    identityFeatureLimit: int = 64
    evaluations: dict[str, ParameterCandidateEvaluation] = Field(default_factory=dict)
    executionOrder: list[str] = Field(default_factory=list)
    executionLock: Any = Field(default_factory=Lock, exclude=True, repr=False)

    @classmethod
    def get_blank(cls) -> "ParameterTuningDependencies":
        return cls()


class ParameterTuningAssayInput(AgentDataModel):
    """One assay branch supplied to batched parameter tuning."""

    normalized: Any = Field(default=None, exclude=True)
    candidates: list[ParameterCandidate] = Field(default_factory=list)
    batchColumns: list[str] = Field(default_factory=list)
    preservationColumns: list[str] = Field(default_factory=list)
    experimentalHandoff: ExperimentalTuningHandoff | None = None
    maxCandidates: int = Field(default=5, ge=1)
    maxRefinedCandidates: int = Field(default=0, ge=0)
    allowHarmonyRefinement: bool = True
    minClusterCells: int = Field(default=20, ge=1)
    identityFeatureLimit: int = Field(default=64, ge=2)

    @classmethod
    def get_blank(cls) -> "ParameterTuningAssayInput":
        return cls()


def _default_parameter_candidates() -> list[ParameterCandidate]:
    """Return a small one-factor candidate set around Scarf defaults."""

    return [
        ParameterCandidate(
            candidateId="baseline",
            dimensions=21,
            leidenResolution=1.0,
        ),
        ParameterCandidate(
            candidateId="pca_15",
            dimensions=15,
            leidenResolution=1.0,
        ),
        ParameterCandidate(
            candidateId="pca_30",
            dimensions=30,
            leidenResolution=1.0,
        ),
        ParameterCandidate(
            candidateId="leiden_0_5",
            dimensions=21,
            leidenResolution=0.5,
        ),
        ParameterCandidate(
            candidateId="leiden_1_5",
            dimensions=21,
            leidenResolution=1.5,
        ),
    ]
