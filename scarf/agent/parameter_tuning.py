"""Bounded parameter tuning over explicit Scarf analysis candidates."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from textwrap import dedent
from threading import Lock
from typing import Any, Literal

import numpy as np

from .config import CONFIG, AgentRunConfig
from .config.agent_exec import run_agent_sync
from .types import (
    AgentDataModel,
    AgentRunInfo,
    ArtifactReferenceModel,
    ExperimentalTuningHandoff,
    StageStatus,
    TuningBiologyHandoff,
)
from ..utils.logging import logger

try:
    from pydantic import Field
except ImportError as exc:
    from .config._deps import AGENT_INSTALL_HINT

    raise ImportError(AGENT_INSTALL_HINT) from exc

try:
    from pydantic_ai import RunContext, UnexpectedModelBehavior
except ImportError as exc:
    raise ImportError(AGENT_INSTALL_HINT) from exc


type CandidateStatus = Literal["done", "failed"]
type CandidatePhase = Literal["initial", "refined"]
type ParameterSearchStatus = Literal["complete", "refine"]
type TuningConfidence = Literal["low", "medium", "high"]
type ReductionMethod = Literal["pca", "lsi", "identity"]
type IntegrationMethod = Literal["snn", "wnn"]


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

    @classmethod
    def get_example(cls) -> "ArtifactRecord":
        return cls(
            scope="assay",
            kind="connectivity_map",
            artifactId="a" * 64,
            assay="RNA",
        )


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

    @classmethod
    def get_example(cls) -> "ParameterCandidate":
        return cls(
            candidateId="baseline",
            reductionMethod="pca",
            dimensions=21,
            leidenResolution=1.0,
            neighborsK=11,
            useHarmony=False,
        )


class ParameterMetrics(AgentDataModel):
    """Bounded quality metrics for one candidate branch."""

    nClusters: int | None = None
    minClusterCells: int | None = None
    minClusterFraction: float | None = None
    graphSilhouetteMedian: float | None = None
    pcaSilhouette: float | None = None
    macroF1: float | None = None
    weightedF1: float | None = None
    batchMixing: dict[str, float] = Field(default_factory=dict)
    biologicalPreservation: dict[str, dict[str, float]] = Field(default_factory=dict)

    @classmethod
    def get_blank(cls) -> "ParameterMetrics":
        return cls()

    @classmethod
    def get_example(cls) -> "ParameterMetrics":
        return cls(
            nClusters=8,
            minClusterCells=42,
            minClusterFraction=0.021,
            graphSilhouetteMedian=0.41,
            pcaSilhouette=0.36,
            macroF1=0.82,
            weightedF1=0.86,
            batchMixing={"batch": 0.73},
            biologicalPreservation={
                "cell_type": {"clisi": 0.88, "graphConnectivity": 0.91}
            },
        )


class ParameterCandidateEvaluation(AgentDataModel):
    """Execution record returned to the model for one candidate."""

    candidateId: str = ""
    phase: CandidatePhase = "initial"
    harmonyBatchColumns: list[str] = Field(default_factory=list)
    status: CandidateStatus = "failed"
    eligible: bool = False
    parameters: ParameterCandidate = Field(default_factory=ParameterCandidate.get_blank)
    artifacts: dict[str, ArtifactRecord] = Field(default_factory=dict)
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

    @classmethod
    def get_example(cls) -> "ParameterCandidateEvaluation":
        candidate = ParameterCandidate.get_example()
        return cls(
            candidateId=candidate.candidateId,
            status="done",
            eligible=True,
            parameters=candidate,
            artifacts={"connectivityMap": ArtifactRecord.get_example()},
            clusterColumn="RNA_agent_tuning_baseline",
            clusterLabel="agent_tuning_baseline",
            effectiveDimensions=21,
            metrics=ParameterMetrics.get_example(),
            evidenceIds=["candidate:baseline:clusters"],
        )


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

    @classmethod
    def get_example(cls) -> "IntegrationMetrics":
        return cls(
            nClusters=8,
            minClusterCells=37,
            minClusterFraction=0.0185,
            adjustedRandByAssay={"RNA": 0.71, "ADT": 0.63},
            normalizedMutualInformationByAssay={"RNA": 0.76, "ADT": 0.69},
            modalityWeightsValid=True,
        )


class IntegrationCandidateEvaluation(AgentDataModel):
    """One executor-produced SNN or WNN graph and cluster evaluation."""

    integrationId: str = ""
    method: IntegrationMethod = "snn"
    assays: list[str] = Field(default_factory=list)
    status: CandidateStatus = "failed"
    eligible: bool = False
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

    @classmethod
    def get_example(cls) -> "IntegrationCandidateEvaluation":
        return cls(
            integrationId="wnn_resolution_1",
            method="wnn",
            assays=["RNA", "ADT"],
            status="done",
            eligible=True,
            graphArtifact=ArtifactRecord(
                scope="datastore",
                kind="integrated_graph",
                artifactId="2" * 64,
            ),
            clusterArtifact=ArtifactRecord(
                scope="datastore",
                kind="cluster_labels",
                artifactId="3" * 64,
            ),
            clusterColumn="agent_wnn_cluster",
            metrics=IntegrationMetrics.get_example(),
            evidenceIds=["integration:wnn_resolution_1:clusters"],
        )


class FinalGraphComparison(AgentDataModel):
    """Evidence-backed comparison against one eligible final graph option."""

    optionId: str = ""
    summary: str = ""
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "FinalGraphComparison":
        return cls()

    @classmethod
    def get_example(cls) -> "FinalGraphComparison":
        return cls(
            optionId="native:ADT:baseline",
            summary="The RNA-native option better preserves the requested labels.",
            evidenceIds=[
                "native:RNA:candidate:baseline:clusters",
                "native:ADT:candidate:baseline:clusters",
            ],
        )


class FinalGraphNeedsInput(AgentDataModel):
    """Concrete input needed before a final graph can be selected."""

    question: str = ""
    options: list[str] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "FinalGraphNeedsInput":
        return cls()

    @classmethod
    def get_example(cls) -> "FinalGraphNeedsInput":
        return cls(
            question="Which biological signal must the final graph preserve?",
            options=["cell_type", "condition"],
        )


class FinalGraphSelection(AgentDataModel):
    """Grounded choice among selected native, SNN, and WNN graph options."""

    status: StageStatus = "needsInput"
    selectedOptionId: str | None = None
    graphMethod: Literal["native", "snn", "wnn"] | None = None
    nativeAssay: str | None = None
    nativeCandidateId: str | None = None
    integrationId: str | None = None
    markerAssay: str = ""
    confidence: TuningConfidence = "low"
    rationale: str = ""
    evidenceIds: list[str] = Field(default_factory=list)
    comparisons: list[FinalGraphComparison] = Field(default_factory=list)
    tradeoffs: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    needsInput: FinalGraphNeedsInput | None = None
    runInfo: AgentRunInfo = Field(default_factory=AgentRunInfo)

    @classmethod
    def get_blank(cls) -> "FinalGraphSelection":
        return cls()

    @classmethod
    def get_example(cls) -> "FinalGraphSelection":
        return cls(
            status="done",
            selectedOptionId="native:RNA:baseline",
            graphMethod="native",
            nativeAssay="RNA",
            nativeCandidateId="baseline",
            markerAssay="RNA",
            confidence="medium",
            rationale="The selected native graph has the strongest supported balance.",
            evidenceIds=["native:RNA:candidate:baseline:clusters"],
            runInfo=AgentRunInfo.get_example(),
        )


class CandidateComparison(AgentDataModel):
    """Evidence-backed comparison against one executed non-selected candidate."""

    candidateId: str = ""
    summary: str = ""
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "CandidateComparison":
        return cls()

    @classmethod
    def get_example(cls) -> "CandidateComparison":
        return cls(
            candidateId="pca_15",
            summary="The selected baseline retains larger minimum clusters.",
            evidenceIds=[
                "candidate:baseline:clusters",
                "candidate:pca_15:clusters",
            ],
        )


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
    runInfo: AgentRunInfo = Field(default_factory=AgentRunInfo)

    @classmethod
    def get_blank(cls) -> "ParameterSearchPlan":
        return cls()

    @classmethod
    def get_example(cls) -> "ParameterSearchPlan":
        return cls(
            status="refine",
            candidates=[
                ParameterCandidate(
                    candidateId="refined_pca_18",
                    dimensions=18,
                    leidenResolution=1.0,
                    neighborsK=11,
                    useHarmony=False,
                )
            ],
            basedOnCandidateIds=["baseline", "pca_15"],
            harmonyBatchColumns=[],
            objectives=["Resolve the dimension tradeoff."],
            rationale="The initial screen brackets a narrower dimension range.",
            evidenceIds=[
                "candidate:baseline:clusters",
                "candidate:pca_15:clusters",
            ],
            stoppingCriteria=["Run the proposed candidate once."],
            runInfo=AgentRunInfo.get_example(),
        )


class ParameterTuningBatchSearchPlan(AgentDataModel):
    """One bounded refinement plan for every assay in a batched screen."""

    assayPlans: dict[str, ParameterSearchPlan] = Field(default_factory=dict)
    runInfo: AgentRunInfo = Field(default_factory=AgentRunInfo)

    @classmethod
    def get_blank(cls) -> "ParameterTuningBatchSearchPlan":
        return cls()

    @classmethod
    def get_example(cls) -> "ParameterTuningBatchSearchPlan":
        return cls(assayPlans={"RNA": ParameterSearchPlan.get_example()})


class ParameterTuningNeedsInput(AgentDataModel):
    """User input required before tuning can produce a recommendation."""

    question: str = ""
    options: list[str] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "ParameterTuningNeedsInput":
        return cls()

    @classmethod
    def get_example(cls) -> "ParameterTuningNeedsInput":
        return cls(
            question="Which trusted biological label should be preserved?",
            options=["cell_type", "none"],
            evidenceIds=["candidate:baseline:batchMixing:batch"],
        )


class ParameterTuningReport(AgentDataModel):
    """Grounded recommendation over candidate branches actually executed."""

    status: StageStatus = "failed"
    fromAssay: str = ""
    cellKey: str = "I"
    evaluations: list[ParameterCandidateEvaluation] = Field(default_factory=list)
    recommendedCandidateId: str | None = None
    selectedArtifacts: dict[str, ArtifactRecord] = Field(default_factory=dict)
    confidence: TuningConfidence = "low"
    rationale: str = ""
    evidenceIds: list[str] = Field(default_factory=list)
    comparisons: list[CandidateComparison] = Field(default_factory=list)
    tradeoffs: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    stopReason: str = ""
    needsInput: ParameterTuningNeedsInput | None = None
    searchPlan: ParameterSearchPlan | None = None
    assayReports: dict[str, "ParameterTuningReport"] = Field(default_factory=dict)
    recommendedByAssay: dict[str, str] = Field(default_factory=dict)
    totalCandidates: int = 0
    integrationEvaluations: list[IntegrationCandidateEvaluation] = Field(
        default_factory=list
    )
    recommendedIntegrationId: str | None = None
    finalClusterColumn: str | None = None
    finalClusterArtifact: ArtifactRecord | None = None
    graphAssay: str | None = None
    markerAssay: str | None = None
    finalSelection: FinalGraphSelection | None = None
    runInfo: AgentRunInfo = Field(default_factory=AgentRunInfo)

    @classmethod
    def get_blank(cls) -> "ParameterTuningReport":
        return cls()

    @classmethod
    def get_example(cls) -> "ParameterTuningReport":
        evaluation = ParameterCandidateEvaluation.get_example()
        return cls(
            status="done",
            fromAssay="RNA",
            cellKey="I",
            evaluations=[evaluation],
            recommendedCandidateId=evaluation.candidateId,
            selectedArtifacts=dict(evaluation.artifacts),
            confidence="medium",
            rationale="The baseline balances separation and cluster size.",
            evidenceIds=["candidate:baseline:clusters"],
            tradeoffs=["Higher resolutions produced smaller clusters."],
            limitations=["No trusted biological preservation label was supplied."],
            stopReason="All authorized candidates were evaluated.",
            recommendedByAssay={"RNA": evaluation.candidateId},
            totalCandidates=1,
            graphAssay="RNA",
            markerAssay="RNA",
            finalSelection=FinalGraphSelection.get_example(),
            runInfo=AgentRunInfo.get_example(),
        )

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
            if not self.finalClusterColumn:
                raise ValueError("Final integrated branch lacks a cluster column")
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
                fromAssay=resolved_marker_assay,
                graphAssay=self.graphAssay,
                markerAssay=resolved_marker_assay,
                cellKey=self.cellKey,
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
                clusterColumn=self.finalClusterColumn,
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
        if selected.clusterColumn is None or cluster_artifact is None:
            raise ValueError("Recommended candidate lacks an exact cluster artifact")
        if not self.fromAssay or cluster_artifact.assay != self.fromAssay:
            raise ValueError("Recommended cluster artifact does not match the assay")
        prefix = f"candidate:{selected.candidateId}:"
        return TuningBiologyHandoff(
            fromAssay=self.fromAssay,
            graphAssay=self.fromAssay,
            markerAssay=marker_assay or self.markerAssay or self.fromAssay,
            cellKey=self.cellKey,
            recommendedCandidateId=selected.candidateId,
            clusterColumn=selected.clusterColumn,
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
    normalizedShape: tuple[int, int] | None = None
    fromAssay: str = ""
    cellKey: str = "I"
    candidates: dict[str, ParameterCandidate] = Field(default_factory=dict)
    candidatePhases: dict[str, CandidatePhase] = Field(default_factory=dict)
    batchColumns: tuple[str, ...] = ()
    preservationColumns: tuple[str, ...] = ()
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

    @classmethod
    def get_example(cls) -> "ParameterTuningDependencies":
        candidate = ParameterCandidate.get_example()
        return cls(
            fromAssay="RNA",
            normalizedShape=(1000, 2000),
            candidates={candidate.candidateId: candidate},
            batchColumns=("batch",),
            preservationColumns=("cell_type",),
        )


class ParameterTuningAssayInput(AgentDataModel):
    """One assay branch supplied to batched parameter tuning."""

    normalized: Any = Field(default=None, exclude=True)
    fromAssay: str = ""
    cellKey: str = "I"
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

    @classmethod
    def get_example(cls) -> "ParameterTuningAssayInput":
        return cls(
            normalized=ArtifactRecord(
                assay="RNA",
                kind="normalized",
                artifactId="4" * 64,
            ),
            fromAssay="RNA",
            candidates=get_default_parameter_candidates(),
            experimentalHandoff=ExperimentalTuningHandoff(batchAction="skip"),
        )


def get_default_parameter_candidates() -> list[ParameterCandidate]:
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


def build_initial_parameter_candidates(
    candidates: Sequence[ParameterCandidate],
    *,
    pair_harmony: bool,
) -> list[ParameterCandidate]:
    """Build deterministic initial branches from caller-authorized parameters."""

    initial: list[ParameterCandidate] = []
    for candidate in candidates:
        if pair_harmony and candidate.useHarmony:
            raise ValueError(
                "Initial seed candidates must not set useHarmony when the "
                "experimental handoff controls Harmony pairing"
            )
        initial.append(candidate)
        if pair_harmony:
            payload = candidate.model_dump()
            payload.update(
                {
                    "candidateId": f"{candidate.candidateId}_harmony",
                    "useHarmony": True,
                }
            )
            initial.append(ParameterCandidate.model_validate(payload))
    return initial


def parameter_search_system_prompt() -> str:
    """Build the stable prompt for the bounded refinement-planning call."""

    return dedent(
        """
        You are planning one bounded refinement pass for Scarf parameter tuning.
        The initial candidate screen has already finished. Do not request tools or
        claim that additional candidates ran.

        Return exactly one of these two plan shapes:
        1. status=complete with candidates=[] when the initial screen is sufficient.
        2. status=refine with one or more candidates when an untested candidate
           inside the initial numeric search envelope can resolve a specific
           evidence-backed uncertainty.
        Never return status=complete with candidates. A Harmony candidate always
        uses the exact authorized batch columns supplied in the prompt. You may
        choose between no correction and that approved Harmony configuration, but
        you must not propose or modify batch columns.

        Cite only evidenceIds from the initial evaluations. Identify the successful
        initial candidates that motivate refinement, state focused objectives, and
        provide concrete stopping criteria. Do not invent metrics, artifacts, or
        candidate ids.
        """
    ).strip()


def parameter_search_prompt(
    *,
    from_assay: str,
    cell_key: str,
    evaluations: Sequence[ParameterCandidateEvaluation],
    batch_columns: Sequence[str],
    preservation_columns: Sequence[str],
    harmony_authorized: bool,
    max_refined_candidates: int,
) -> str:
    """Build the planning prompt from deterministic initial evaluations."""

    evaluation_payload = [evaluation.model_dump() for evaluation in evaluations]
    correction_modes = ["none", "harmony"] if harmony_authorized else ["none"]
    return (
        dedent("""
        Inspect the completed initial screen for assay {from_assay} and cell
        selection {cell_key}.

        Initial evaluations:
        {evaluation_payload}

        Authorized correction modes: {correction_modes}
        Exact Harmony batch columns: {batch_columns}
        Trusted biological preservation columns: {preservation_columns}
        Maximum refined candidates: {max_refined_candidates}

        Return one ParameterSearchPlan. Refinement is optional and is limited to
        one deterministic follow-up pass.
        """)
        .strip()
        .format(
            from_assay=from_assay,
            cell_key=cell_key,
            evaluation_payload=json.dumps(
                evaluation_payload,
                indent=2,
                sort_keys=True,
            ),
            correction_modes=json.dumps(correction_modes),
            batch_columns=json.dumps(list(batch_columns)),
            preservation_columns=json.dumps(list(preservation_columns)),
            max_refined_candidates=max_refined_candidates,
        )
    )


def parameter_tuning_system_prompt(min_cluster_cells: int) -> str:
    """Build the stable prompt for final candidate selection."""

    return (
        dedent(
            """
        You are Scarf's parameter tuning selection agent. Every candidate in the
        prompt has already finished deterministic execution. Do not request tools
        or claim that another candidate ran.

        Recommend only a candidate whose evaluation has status=done and
        eligible=true. A candidate is ineligible when it creates fewer than two
        clusters or a cluster with fewer than {min_cluster_cells} cells. Do not
        invent artifact ids, metrics, candidate ids, or evidence ids. Cite only
        evidenceIds recorded in the completed evaluations.

        Balance cluster separation, cluster sizes, batch mixing, and biological
        preservation. High batch mixing alone can indicate overcorrection, so do
        not collapse the metrics into an invented score. UMAP appearance is not
        evidence for parameter quality. When multiple candidates complete,
        return one comparison for every non-selected successful candidate. Each
        comparison must cite evidence from both the selected candidate and that
        comparator. Return a concise structured report.
        """
        )
        .strip()
        .format(min_cluster_cells=min_cluster_cells)
    )


def parameter_tuning_prompt(
    *,
    from_assay: str,
    cell_key: str,
    evaluations: Sequence[ParameterCandidateEvaluation],
    batch_columns: Sequence[str],
    preservation_columns: Sequence[str],
    search_plan: ParameterSearchPlan,
) -> str:
    """Build the final selection prompt from completed evaluations."""

    evaluation_payload = [evaluation.model_dump() for evaluation in evaluations]
    return (
        dedent(
            """
        Select a completed candidate for assay {from_assay} and cell selection
        {cell_key}.

        Completed evaluations:
        {evaluation_payload}

        Validated refinement plan:
        {search_plan}

        Exact Harmony batch columns: {batch_columns}
        Trusted biological preservation columns: {preservation_columns}

        Recommend one eligible candidate or explain why user input is needed.
        Compare the recommendation with every other successful candidate. High
        batch mixing does not by itself justify correction when biological
        preservation declines.
        """
        )
        .strip()
        .format(
            from_assay=from_assay,
            cell_key=cell_key,
            evaluation_payload=json.dumps(
                evaluation_payload,
                indent=2,
                sort_keys=True,
            ),
            search_plan=json.dumps(
                search_plan.model_dump(exclude={"runInfo"}),
                indent=2,
                sort_keys=True,
            ),
            batch_columns=json.dumps(list(batch_columns)),
            preservation_columns=json.dumps(list(preservation_columns)),
        )
    )


def parameter_batch_search_prompt(
    dependencies: Mapping[str, ParameterTuningDependencies],
    max_refined_by_assay: Mapping[str, int],
) -> str:
    """Build one refinement prompt for all modality-specific screens."""

    payload = {
        assay: {
            "evaluations": [
                deps.evaluations[candidate_id].model_dump()
                for candidate_id in deps.executionOrder
            ],
            "authorizedHarmony": deps.harmonyAuthorized,
            "batchColumns": list(deps.batchColumns),
            "preservationColumns": list(deps.preservationColumns),
            "maxRefinedCandidates": max_refined_by_assay[assay],
        }
        for assay, deps in dependencies.items()
    }
    return (
        dedent(
            """
            Plan one optional refinement pass for every assay in this completed
            multimodal initial screen:
            {payload}

            Return exactly one assayPlans entry for every assay. Each entry must
            obey the single-assay ParameterSearchPlan rules. Candidate ids need
            only be unique within their assay. Do not compare metric fields that
            are absent for a modality, and do not request additional tool calls.
            """
        )
        .strip()
        .format(payload=json.dumps(payload, indent=2, sort_keys=True))
    )


def parameter_batch_search_system_prompt() -> str:
    """Build the stable system prompt for batched refinement planning."""

    return (
        dedent(
            """
            {single_assay_rules}

            Return the plans together in one assayPlans mapping.
            """
        )
        .strip()
        .format(single_assay_rules=parameter_search_system_prompt())
    )


def parameter_batch_selection_system_prompt() -> str:
    """Build the stable system prompt for batched native selection."""

    return (
        dedent(
            """
            You are Scarf's batched native parameter selection agent. Every branch
            has already executed. Return one aggregate ParameterTuningReport with
            exactly one grounded single-assay report in assayReports per assay.
            Apply eligibility, evidence, and comparison requirements independently.
            Do not invent joint scores, artifacts, candidates, or evidence. UMAP
            appearance is not evidence. Leave integration fields empty.
            """
        )
        .strip()
        .format()
    )


def parameter_batch_selection_prompt(
    dependencies: Mapping[str, ParameterTuningDependencies],
    search_plans: Mapping[str, ParameterSearchPlan],
    primary_assay: str,
    selection_directions: str = "",
) -> str:
    """Build one native-selection prompt for all executed assay screens."""

    payload = {
        assay: {
            "evaluations": [
                deps.evaluations[candidate_id].model_dump()
                for candidate_id in deps.executionOrder
            ],
            "searchPlan": search_plans[assay].model_dump(exclude={"runInfo"}),
            "minClusterCells": deps.minClusterCells,
            "batchColumns": list(deps.batchColumns),
            "preservationColumns": list(deps.preservationColumns),
        }
        for assay, deps in dependencies.items()
    }
    return (
        dedent(
            """
            Select one eligible native candidate independently for every assay in
            this completed multimodal screen:
            {payload}

            Return a ParameterTuningReport whose assayReports contains exactly one
            single-assay report per assay. Apply the normal evidence and comparison
            rules independently inside each report. The primary assay is
            {primary_assay}. At the aggregate level, summarize cross-assay
            limitations without inventing a joint score. Integration has not run,
            so leave all integration and final-cluster fields empty.

            Caller selection directions, which cannot override eligibility or
            evidence requirements: {selection_directions}
            """
        )
        .strip()
        .format(
            payload=json.dumps(payload, indent=2, sort_keys=True),
            primary_assay=primary_assay,
            selection_directions=selection_directions or "not provided",
        )
    )


def final_graph_selection_system_prompt() -> str:
    """Build stable instructions for the final native/SNN/WNN choice."""

    return (
        dedent(
            """
            You are Scarf's final graph selection agent. Native assay candidates
            and integrated SNN/WNN candidates have already executed. Select only
            an eligible option supplied in the prompt. Do not request tools or
            invent graph options, artifacts, metrics, evidence, or a combined
            score. Compare cluster viability and biological preservation evidence
            that is actually present. ARI and NMI describe agreement, not quality.
            WNN modality weights are usable only when modalityWeightsValid=true.
            UMAP appearance, native-neighbor LISI on an integrated graph, and
            absent metric fields are not evidence. Return one comparison for every
            eligible non-selected option, citing evidence from both options.
            """
        )
        .strip()
        .format()
    )


def final_graph_options(
    report: ParameterTuningReport,
    integration_evaluations: Sequence[IntegrationCandidateEvaluation],
) -> dict[str, dict[str, Any]]:
    """Return the exact eligible graph options and option-scoped evidence."""

    assay_reports = report.assayReports or {report.fromAssay: report}
    options: dict[str, dict[str, Any]] = {}
    for assay, assay_report in assay_reports.items():
        candidate_id = assay_report.recommendedCandidateId
        if candidate_id is None:
            continue
        native_evaluation = next(
            (
                item
                for item in assay_report.evaluations
                if item.candidateId == candidate_id
            ),
            None,
        )
        if (
            native_evaluation is None
            or native_evaluation.status != "done"
            or not native_evaluation.eligible
            or native_evaluation.clusterColumn is None
            or "clusters" not in native_evaluation.artifacts
            or "connectivityMap" not in native_evaluation.artifacts
            or not native_evaluation.evidenceIds
        ):
            continue
        option_id = f"native:{assay}:{candidate_id}"
        option_evidence = [
            f"native:{assay}:{evidence_id}"
            for evidence_id in native_evaluation.evidenceIds
        ]
        evaluation_payload = native_evaluation.model_dump()
        evaluation_payload["evidenceIds"] = option_evidence
        options[option_id] = {
            "optionId": option_id,
            "graphMethod": "native",
            "nativeAssay": assay,
            "nativeCandidateId": candidate_id,
            "evaluation": evaluation_payload,
            "evidenceIds": option_evidence,
        }
    for integration_evaluation in integration_evaluations:
        if (
            integration_evaluation.status != "done"
            or not integration_evaluation.eligible
        ):
            continue
        if (
            integration_evaluation.clusterArtifact is None
            or integration_evaluation.graphArtifact is None
        ):
            continue
        if not integration_evaluation.evidenceIds:
            continue
        if not integration_evaluation.integrationId:
            raise ValueError("Eligible integration evaluations require integrationId")
        if (
            integration_evaluation.graphArtifact.scope != "datastore"
            or integration_evaluation.graphArtifact.assay is not None
            or integration_evaluation.clusterArtifact.scope != "datastore"
            or integration_evaluation.clusterArtifact.assay is not None
            or integration_evaluation.graphArtifact.kind != "integrated_graph"
            or integration_evaluation.clusterArtifact.kind
            not in {"cluster_labels", "cluster_cut"}
        ):
            raise ValueError(
                "Integrated graph and cluster artifacts must be datastore-scoped "
                "without an assay"
            )
        if (
            integration_evaluation.method == "wnn"
            and integration_evaluation.metrics.modalityWeightsValid is not True
        ):
            continue
        option_id = f"integration:{integration_evaluation.integrationId}"
        if option_id in options:
            raise ValueError(
                f"Duplicate integration id {integration_evaluation.integrationId!r}"
            )
        options[option_id] = {
            "optionId": option_id,
            "graphMethod": integration_evaluation.method,
            "integrationId": integration_evaluation.integrationId,
            "evaluation": integration_evaluation.model_dump(),
            "evidenceIds": list(integration_evaluation.evidenceIds),
        }
    return options


def final_graph_selection_prompt(
    *,
    report: ParameterTuningReport,
    integration_evaluations: Sequence[IntegrationCandidateEvaluation],
    marker_assay: str,
) -> str:
    """Build the selection prompt from executor-grounded final graph options."""

    options = final_graph_options(report, integration_evaluations)
    return (
        dedent(
            """
            Select the final graph from these eligible executed options:
            {options}

            The fixed marker assay is {marker_assay}. It determines marker
            extraction and does not imply ownership of an integrated graph.
            Return needsInput only when the supplied evidence cannot resolve a
            scientifically material tradeoff.
            """
        )
        .strip()
        .format(
            options=json.dumps(options, indent=2, sort_keys=True),
            marker_assay=marker_assay,
        )
    )


def normalized_artifact_shape(store: Any, normalized: Any) -> tuple[int, int]:
    """Return the exact cell-by-feature shape of a normalized artifact."""

    group = store.load_artifact(normalized)
    if "data" not in group:
        raise ValueError("Normalized artifact does not contain a data matrix")
    shape = getattr(group["data"], "shape", None)
    if not isinstance(shape, tuple | list) or len(shape) != 2:
        raise ValueError("Normalized artifact data must be two-dimensional")
    n_cells, n_features = map(int, shape)
    if n_cells < 2 or n_features < 2:
        raise ValueError(
            "Parameter tuning requires at least two cells and two selected features"
        )
    return n_cells, n_features


def validate_parameter_candidate_rank(
    candidate: ParameterCandidate,
    normalized_shape: tuple[int, int],
    *,
    identity_feature_limit: int = 64,
) -> int:
    """Validate a candidate before any branch operation and return output rank."""

    n_cells, n_features = normalized_shape
    if candidate.neighborsK >= n_cells:
        raise ValueError(
            f"neighborsK={candidate.neighborsK} requires more than "
            f"{candidate.neighborsK} selected cells; observed {n_cells}"
        )
    if candidate.reductionMethod == "pca":
        if candidate.dimensions + 1 > min(n_cells, n_features):
            raise ValueError(
                f"PCA dimensions={candidate.dimensions} requires at least "
                f"{candidate.dimensions + 1} cells and selected features; "
                f"observed shape {normalized_shape}"
            )
        return candidate.dimensions
    if candidate.reductionMethod == "lsi":
        required_rank = candidate.dimensions + 1
        if required_rank > min(n_cells, n_features):
            raise ValueError(
                "LSI dimensions, including the skipped component, exceed the "
                f"normalized matrix rank for shape {normalized_shape}"
            )
        return candidate.dimensions
    if n_features > identity_feature_limit:
        raise ValueError(
            f"Identity reduction supports at most {identity_feature_limit} selected "
            f"features; observed {n_features}"
        )
    if candidate.dimensions != n_features:
        raise ValueError(
            "Identity reduction dimensions must equal the exact normalized feature "
            f"count {n_features}; received {candidate.dimensions}"
        )
    return n_features


def run_candidate_reduction(
    store: Any,
    *,
    normalized: Any,
    from_assay: str,
    candidate: ParameterCandidate,
    normalized_shape: tuple[int, int],
    update_state: bool,
    identity_feature_limit: int = 64,
) -> tuple[Any, str, int]:
    """Run one validated modality-aware reduction with public Scarf methods."""

    effective_dimensions = validate_parameter_candidate_rank(
        candidate,
        normalized_shape,
        identity_feature_limit=identity_feature_limit,
    )
    common = {
        "from_assay": from_assay,
        "update_state": update_state,
        "invalidate_cache": False,
    }
    if candidate.reductionMethod == "pca":
        ref = store.run_pca(
            normalized,
            dims=candidate.dimensions,
            feat_scaling=True,
            show_elbow_plot=False,
            **common,
        )
        return ref, "pca", effective_dimensions
    if candidate.reductionMethod == "lsi":
        ref = store.run_lsi(
            normalized,
            dims=candidate.dimensions,
            skip_first=True,
            rand_state=CONFIG._PCA_RANDOM_SEED,
            **common,
        )
        return ref, "lsi", effective_dimensions
    loadings = np.eye(normalized_shape[1], dtype=np.float64)
    ref = store.run_custom_reduction(
        loadings,
        normalized,
        **common,
    )
    return ref, "identity", effective_dimensions


def _collect_parameter_candidate_metrics(
    deps: ParameterTuningDependencies,
    *,
    candidate: ParameterCandidate,
    candidate_id: str,
    reduction_ref: Any,
    neighbors_ref: Any,
    graph_ref: Any,
    cluster_ref: Any,
    cluster_column: str,
    evidence_ids: list[str],
    warnings: list[str],
) -> tuple[ParameterMetrics, list[str]]:
    store = deps.store
    cluster_group = store.load_artifact(cluster_ref)
    cluster_data = cluster_group["values"]
    cluster_values = np.asarray(cluster_data[:])
    if cluster_values.ndim != 1 or len(cluster_values) == 0:
        raise ValueError("Cluster artifact must contain one non-empty label vector")
    if np.any(cluster_values < 0):
        raise ValueError("Cluster artifact contains invalid negative labels")
    _, cluster_counts = np.unique(cluster_values, return_counts=True)
    n_clusters = int(len(cluster_counts))
    min_cluster_cells = int(cluster_counts.min())
    min_cluster_fraction = float(min_cluster_cells / len(cluster_values))
    metrics = ParameterMetrics(
        nClusters=n_clusters,
        minClusterCells=min_cluster_cells,
        minClusterFraction=min_cluster_fraction,
    )
    evidence_ids.append(f"candidate:{candidate_id}:clusters")

    try:
        graph_scores = store.metric_graph_silhouette(
            res_label=cluster_column,
            neighbors=neighbors_ref,
            from_assay=deps.fromAssay,
            cell_key=deps.cellKey,
            random_seed=CONFIG._RANDOM_SEED,
            sample_size=11,
        )
        if graph_scores is not None:
            finite_scores = np.asarray(graph_scores, dtype=float)
            finite_scores = finite_scores[np.isfinite(finite_scores)]
            if len(finite_scores):
                metrics.graphSilhouetteMedian = float(np.median(finite_scores))
                evidence_ids.append(f"candidate:{candidate_id}:graphSilhouette")
    except (KeyError, TypeError, ValueError) as exc:
        warnings.append(f"Graph silhouette unavailable: {exc}")

    if candidate.reductionMethod == "pca":
        try:
            separability = store.metric_cluster_separability(
                reduction_ref,
                [cluster_column],
                cell_key=deps.cellKey,
                random_seed=CONFIG._RANDOM_SEED,
            )
            table = separability.clustering_scores
            rows = table.loc[table["clustering"] == cluster_column]
            if len(rows):
                row = rows.iloc[0]
                for field_name, column_name, evidence_name in (
                    ("pcaSilhouette", "silhouette_score", "pcaSilhouette"),
                    ("macroF1", "macro_f1_mean", "macroF1"),
                    ("weightedF1", "weighted_f1_mean", "weightedF1"),
                ):
                    value = row[column_name]
                    if value is not None and np.isfinite(float(value)):
                        setattr(metrics, field_name, float(value))
                        evidence_ids.append(f"candidate:{candidate_id}:{evidence_name}")
        except (KeyError, TypeError, ValueError) as exc:
            warnings.append(f"PCA cluster separability unavailable: {exc}")

    perplexity = max(1.0, float(candidate.neighborsK // 3))
    for column in deps.batchColumns:
        try:
            score = float(
                store.metric_proportional_batch_mixing(
                    column,
                    neighbors=neighbors_ref,
                    from_assay=deps.fromAssay,
                    cell_key=deps.cellKey,
                    perplexity=perplexity,
                )
            )
            if np.isfinite(score):
                metrics.batchMixing[column] = score
                evidence_ids.append(f"candidate:{candidate_id}:batchMixing:{column}")
        except (KeyError, TypeError, ValueError) as exc:
            warnings.append(f"Batch mixing for {column!r} unavailable: {exc}")

    for column in deps.preservationColumns:
        scores: dict[str, float] = {}
        try:
            clisi = float(
                store.metric_clisi(
                    column,
                    neighbors=neighbors_ref,
                    from_assay=deps.fromAssay,
                    cell_key=deps.cellKey,
                    perplexity=None,
                    scale=True,
                )
            )
            if np.isfinite(clisi):
                scores["clisi"] = clisi
                evidence_ids.append(f"candidate:{candidate_id}:clisi:{column}")
        except (KeyError, TypeError, ValueError) as exc:
            warnings.append(f"cLISI for {column!r} unavailable: {exc}")
        try:
            connectivity = float(
                store.metric_graph_connectivity(
                    column,
                    graph=graph_ref,
                    from_assay=deps.fromAssay,
                    cell_key=deps.cellKey,
                )
            )
            if np.isfinite(connectivity):
                scores["graphConnectivity"] = connectivity
                evidence_ids.append(
                    f"candidate:{candidate_id}:graphConnectivity:{column}"
                )
        except (KeyError, TypeError, ValueError) as exc:
            warnings.append(f"Graph connectivity for {column!r} unavailable: {exc}")
        if scores:
            metrics.biologicalPreservation[column] = scores

    eligibility_reasons: list[str] = []
    if n_clusters < 2:
        eligibility_reasons.append("fewer than two clusters")
    if min_cluster_cells < deps.minClusterCells:
        eligibility_reasons.append(
            f"smallest cluster has {min_cluster_cells} cells; "
            f"minimum is {deps.minClusterCells}"
        )
    return metrics, eligibility_reasons


def execute_parameter_candidate(
    deps: ParameterTuningDependencies,
    candidate_id: str,
) -> ParameterCandidateEvaluation:
    """Execute one allowlisted candidate without model involvement."""

    with deps.executionLock:
        if candidate_id in deps.evaluations:
            logger.debug(
                f"Parameter candidate {candidate_id!r} for assay "
                f"{deps.fromAssay!r} reused its completed evaluation"
            )
            return deps.evaluations[candidate_id]
        if candidate_id not in deps.candidates:
            logger.warning(
                f"Parameter candidate {candidate_id!r} is not authorized for "
                f"assay {deps.fromAssay!r}"
            )
            return ParameterCandidateEvaluation(
                candidateId=candidate_id,
                status="failed",
                error=(
                    f"Unknown candidate id {candidate_id!r}; allowed ids are "
                    f"{sorted(deps.candidates)}"
                ),
            )
        if len(deps.executionOrder) >= deps.maxCandidates:
            logger.warning(
                f"Parameter candidate {candidate_id!r} was not executed because "
                f"assay {deps.fromAssay!r} reached its limit of "
                f"{deps.maxCandidates} candidates"
            )
            return ParameterCandidateEvaluation(
                candidateId=candidate_id,
                phase=deps.candidatePhases.get(candidate_id, "initial"),
                harmonyBatchColumns=(
                    list(deps.batchColumns)
                    if deps.candidates[candidate_id].useHarmony
                    else []
                ),
                status="failed",
                parameters=deps.candidates[candidate_id],
                error=f"Candidate execution limit {deps.maxCandidates} reached",
            )

        candidate = deps.candidates[candidate_id]
        deps.executionOrder.append(candidate_id)
        logger.info(
            f"Running parameter candidate {candidate_id!r} for assay "
            f"{deps.fromAssay!r}: method={candidate.reductionMethod}, "
            f"dimensions={candidate.dimensions}, k={candidate.neighborsK}, "
            f"resolution={candidate.leidenResolution}, "
            f"harmony={candidate.useHarmony}"
        )
        if candidate.useHarmony and not deps.batchColumns:
            logger.warning(
                f"Parameter candidate {candidate_id!r} cannot run Harmony because "
                "no batch columns were authorized"
            )
            evaluation = ParameterCandidateEvaluation(
                candidateId=candidate_id,
                phase=deps.candidatePhases.get(candidate_id, "initial"),
                harmonyBatchColumns=[],
                status="failed",
                parameters=candidate,
                error="Harmony candidate requires at least one authorized batch column",
            )
            deps.evaluations[candidate_id] = evaluation
            return evaluation

        store = deps.store
        state_before = None
        if hasattr(store, "get_assay_state"):
            state_before = store.get_assay_state(deps.fromAssay)

        artifacts: dict[str, ArtifactRecord] = {}
        warnings: list[str] = []
        evidence_ids: list[str] = []
        cluster_column: str | None = None

        try:
            normalized_shape = deps.normalizedShape or normalized_artifact_shape(
                store,
                deps.normalized,
            )
            effective_dimensions = validate_parameter_candidate_rank(
                candidate,
                normalized_shape,
                identity_feature_limit=deps.identityFeatureLimit,
            )
            reduction_ref, reduction_key, _ = run_candidate_reduction(
                store,
                normalized=deps.normalized,
                from_assay=deps.fromAssay,
                candidate=candidate,
                normalized_shape=normalized_shape,
                update_state=False,
                identity_feature_limit=deps.identityFeatureLimit,
            )
            artifacts[reduction_key] = ArtifactRecord.from_ref(reduction_ref)
            logger.debug(
                f"Parameter candidate {candidate_id!r}: completed "
                f"{reduction_key} reduction"
            )

            coordinates_ref = reduction_ref
            if candidate.useHarmony:
                coordinates_ref = store.run_harmony(
                    list(deps.batchColumns),
                    reduction=reduction_ref,
                    from_assay=deps.fromAssay,
                    update_state=False,
                    invalidate_cache=False,
                )
                artifacts["harmony"] = ArtifactRecord.from_ref(coordinates_ref)
                logger.debug(
                    f"Parameter candidate {candidate_id!r}: completed Harmony "
                    f"using {len(deps.batchColumns)} batch column(s)"
                )

            ann_ref = store.build_ann_index(
                coordinates=coordinates_ref,
                from_assay=deps.fromAssay,
                ann_metric="l2",
                ann_parallel=False,
                rand_state=CONFIG._PCA_RANDOM_SEED,
                update_state=False,
                invalidate_cache=False,
            )
            artifacts["annIndex"] = ArtifactRecord.from_ref(ann_ref)
            logger.debug(
                f"Parameter candidate {candidate_id!r}: completed ANN indexing"
            )

            neighbors_ref = store.query_neighbors(
                ann_index=ann_ref,
                from_assay=deps.fromAssay,
                coordinates=coordinates_ref,
                k=candidate.neighborsK,
                update_state=False,
                invalidate_cache=False,
            )
            artifacts["neighbors"] = ArtifactRecord.from_ref(neighbors_ref)
            logger.debug(
                f"Parameter candidate {candidate_id!r}: completed neighbor query"
            )

            graph_ref = store.build_connectivity_map(
                neighbors=neighbors_ref,
                from_assay=deps.fromAssay,
                local_connectivity=1.0,
                bandwidth=1.5,
                update_state=False,
                invalidate_cache=False,
            )
            artifacts["connectivityMap"] = ArtifactRecord.from_ref(graph_ref)
            logger.debug(
                f"Parameter candidate {candidate_id!r}: completed connectivity map"
            )

            graph_artifact_id = str(getattr(graph_ref, "artifact_id", ""))
            if not graph_artifact_id:
                raise ValueError("Connectivity artifact has no stable identifier")
            branch_payload = {
                "candidate": candidate.model_dump(),
                "graphArtifactId": graph_artifact_id,
            }
            branch_token = hashlib.blake2b(
                json.dumps(
                    branch_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode(),
                digest_size=6,
            ).hexdigest()
            cluster_label = f"agent_tuning_{candidate_id}_{branch_token}"
            cluster_column = (
                f"{deps.fromAssay}_{cluster_label}"
                if deps.cellKey == "I"
                else f"{deps.fromAssay}_{deps.cellKey}_{cluster_label}"
            )

            cluster_ref = store.run_leiden_clustering(
                graph=graph_ref,
                from_assay=deps.fromAssay,
                cell_key=deps.cellKey,
                resolution=candidate.leidenResolution,
                backend="igraph",
                symmetric_graph=False,
                graph_upper_only=False,
                label=cluster_label,
                random_seed=CONFIG._RANDOM_SEED,
                invalidate_cache=False,
            )
            artifacts["clusters"] = ArtifactRecord.from_ref(cluster_ref)
            logger.debug(
                f"Parameter candidate {candidate_id!r}: completed Leiden clustering"
            )

            metrics, eligibility_reasons = _collect_parameter_candidate_metrics(
                deps,
                candidate=candidate,
                candidate_id=candidate_id,
                reduction_ref=reduction_ref,
                neighbors_ref=neighbors_ref,
                graph_ref=graph_ref,
                cluster_ref=cluster_ref,
                cluster_column=cluster_column,
                evidence_ids=evidence_ids,
                warnings=warnings,
            )

            evaluation = ParameterCandidateEvaluation(
                candidateId=candidate_id,
                phase=deps.candidatePhases.get(candidate_id, "initial"),
                harmonyBatchColumns=(
                    list(deps.batchColumns) if candidate.useHarmony else []
                ),
                status="done",
                eligible=not eligibility_reasons,
                parameters=candidate,
                artifacts=artifacts,
                clusterColumn=cluster_column,
                clusterLabel=cluster_label,
                effectiveDimensions=effective_dimensions,
                metrics=metrics,
                evidenceIds=evidence_ids,
                eligibilityReasons=eligibility_reasons,
                warnings=warnings,
            )
            logger.info(
                f"Completed parameter candidate {candidate_id!r} for assay "
                f"{deps.fromAssay!r}: eligible={evaluation.eligible}, "
                f"clusters={metrics.nClusters}, "
                f"minimum_cluster_cells={metrics.minClusterCells}, "
                f"warnings={len(warnings)}"
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            evaluation = ParameterCandidateEvaluation(
                candidateId=candidate_id,
                phase=deps.candidatePhases.get(candidate_id, "initial"),
                harmonyBatchColumns=(
                    list(deps.batchColumns) if candidate.useHarmony else []
                ),
                status="failed",
                parameters=candidate,
                artifacts=artifacts,
                clusterColumn=cluster_column,
                evidenceIds=evidence_ids,
                warnings=warnings,
                error=str(exc),
            )
            logger.warning(
                f"Parameter candidate {candidate_id!r} for assay "
                f"{deps.fromAssay!r} failed: {exc}"
            )

        state_after = None
        if hasattr(store, "get_assay_state"):
            state_after = store.get_assay_state(deps.fromAssay)
        if state_after != state_before:
            logger.error(
                f"Parameter candidate {candidate_id!r} unexpectedly changed "
                f"assay {deps.fromAssay!r} state"
            )
            raise RuntimeError(
                "Candidate execution unexpectedly changed current assay state"
            )
        deps.evaluations[candidate_id] = evaluation
        return evaluation


async def evaluate_parameter_candidate(
    ctx: RunContext[ParameterTuningDependencies],
    candidate_id: str,
) -> ParameterCandidateEvaluation:
    """Expose deterministic candidate execution as a bounded agent tool."""

    return execute_parameter_candidate(ctx.deps, candidate_id)


def validate_parameter_search_plan(
    plan: ParameterSearchPlan,
    deps: ParameterTuningDependencies,
    *,
    initial_candidate_ids: Sequence[str],
    max_refined_candidates: int,
) -> ParameterSearchPlan:
    """Validate one refinement proposal against the completed initial screen."""

    initial_evaluations = [
        deps.evaluations[candidate_id]
        for candidate_id in initial_candidate_ids
        if candidate_id in deps.evaluations
    ]
    known_evidence = {
        evidence_id
        for evaluation in initial_evaluations
        for evidence_id in evaluation.evidenceIds
    }
    unknown_evidence = sorted(set(plan.evidenceIds) - known_evidence)
    if unknown_evidence:
        raise ValueError(
            f"Parameter search plan cites unknown evidence ids {unknown_evidence}"
        )
    authorized_batch_columns = list(deps.batchColumns) if deps.harmonyAuthorized else []
    if (
        plan.harmonyBatchColumns
        and plan.harmonyBatchColumns != authorized_batch_columns
    ):
        raise ValueError(
            "Parameter search plan cannot modify the exact authorized Harmony "
            "batch columns"
        )
    canonical_status: ParameterSearchStatus = (
        "refine" if plan.candidates else "complete"
    )
    plan = plan.model_copy(
        update={
            "status": canonical_status,
            "harmonyBatchColumns": authorized_batch_columns,
        }
    )
    if plan.status == "complete":
        return plan

    if len(plan.candidates) > max_refined_candidates:
        raise ValueError(
            "Parameter search plan exceeds the refined candidate limit "
            f"{max_refined_candidates}"
        )
    if not plan.rationale.strip():
        raise ValueError("A refinement plan requires a rationale")
    if not plan.objectives:
        raise ValueError("A refinement plan requires focused objectives")
    if not plan.stoppingCriteria:
        raise ValueError("A refinement plan requires stopping criteria")
    if not plan.evidenceIds:
        raise ValueError("A refinement plan requires initial-screen evidence")

    successful_initial_ids = {
        evaluation.candidateId
        for evaluation in initial_evaluations
        if evaluation.status == "done"
    }
    if not plan.basedOnCandidateIds:
        raise ValueError("A refinement plan must identify its initial candidates")
    duplicate_parents = sorted(
        {
            candidate_id
            for candidate_id in plan.basedOnCandidateIds
            if plan.basedOnCandidateIds.count(candidate_id) > 1
        }
    )
    if duplicate_parents:
        raise ValueError(f"Duplicate refinement parent ids {duplicate_parents}")
    invalid_parents = sorted(set(plan.basedOnCandidateIds) - successful_initial_ids)
    if invalid_parents:
        raise ValueError(
            "Refinement parents must be successful initial candidates: "
            f"{invalid_parents}"
        )
    for parent_id in plan.basedOnCandidateIds:
        prefix = f"candidate:{parent_id}:"
        if not any(evidence_id.startswith(prefix) for evidence_id in plan.evidenceIds):
            raise ValueError(
                f"Refinement evidence must cite every parent candidate: {parent_id!r}"
            )
    if deps.harmonyAuthorized:
        parent_candidates = [
            deps.candidates[candidate_id] for candidate_id in plan.basedOnCandidateIds
        ]
        paired_modes: dict[tuple[str, int, float, int], set[bool]] = {}
        for candidate in parent_candidates:
            parameter_key = (
                candidate.reductionMethod,
                candidate.dimensions,
                candidate.leidenResolution,
                candidate.neighborsK,
            )
            paired_modes.setdefault(parameter_key, set()).add(candidate.useHarmony)
        if not any(modes == {False, True} for modes in paired_modes.values()):
            raise ValueError(
                "Harmony refinement requires evidence from one matched corrected "
                "and uncorrected initial pair"
            )

    initial_candidates = [
        deps.candidates[candidate_id] for candidate_id in initial_candidate_ids
    ]
    known_signatures = {
        (
            candidate.reductionMethod,
            candidate.dimensions,
            candidate.leidenResolution,
            candidate.neighborsK,
            candidate.useHarmony,
        )
        for candidate in initial_candidates
    }
    proposed_ids: set[str] = set()
    proposed_signatures: set[tuple[str, int, float, int, bool]] = set()
    for candidate in plan.candidates:
        if not CONFIG._CANDIDATE_ID.fullmatch(candidate.candidateId):
            raise ValueError(
                "Refined candidateId must contain only ASCII letters, numbers, "
                "and underscores"
            )
        if (
            candidate.candidateId in deps.candidates
            or candidate.candidateId in proposed_ids
        ):
            raise ValueError(f"Duplicate refined candidateId {candidate.candidateId!r}")
        proposed_ids.add(candidate.candidateId)
        method_candidates = [
            item
            for item in initial_candidates
            if item.reductionMethod == candidate.reductionMethod
        ]
        if not method_candidates:
            raise ValueError(
                "Refined candidates cannot introduce an untested reduction method: "
                f"{candidate.reductionMethod!r}"
            )
        dimension_bounds = (
            min(item.dimensions for item in method_candidates),
            max(item.dimensions for item in method_candidates),
        )
        resolution_bounds = (
            min(item.leidenResolution for item in method_candidates),
            max(item.leidenResolution for item in method_candidates),
        )
        neighbor_bounds = (
            min(item.neighborsK for item in method_candidates),
            max(item.neighborsK for item in method_candidates),
        )
        if not dimension_bounds[0] <= candidate.dimensions <= dimension_bounds[1]:
            raise ValueError(
                "Refined dimensions must remain inside the initial search envelope "
                f"{dimension_bounds}"
            )
        if not (
            resolution_bounds[0] <= candidate.leidenResolution <= resolution_bounds[1]
        ):
            raise ValueError(
                "Refined Leiden resolution must remain inside the initial search "
                f"envelope {resolution_bounds}"
            )
        if not neighbor_bounds[0] <= candidate.neighborsK <= neighbor_bounds[1]:
            raise ValueError(
                "Refined neighbor count must remain inside the initial search "
                f"envelope {neighbor_bounds}"
            )
        if candidate.useHarmony and (
            not deps.harmonyAuthorized or not deps.batchColumns
        ):
            raise ValueError(
                f"Refined candidate {candidate.candidateId!r} is not authorized "
                "for Harmony"
            )
        signature = (
            candidate.reductionMethod,
            candidate.dimensions,
            candidate.leidenResolution,
            candidate.neighborsK,
            candidate.useHarmony,
        )
        if signature in known_signatures or signature in proposed_signatures:
            raise ValueError(
                f"Refined candidate {candidate.candidateId!r} duplicates an "
                "evaluated or proposed parameter branch"
            )
        proposed_signatures.add(signature)
    return plan


def validate_parameter_batch_search_plan(
    plan: ParameterTuningBatchSearchPlan,
    dependencies: Mapping[str, ParameterTuningDependencies],
    *,
    initial_candidate_ids: Mapping[str, Sequence[str]],
    max_refined_by_assay: Mapping[str, int],
) -> ParameterTuningBatchSearchPlan:
    """Validate every assay entry in one batched refinement response."""

    expected = set(dependencies)
    actual = set(plan.assayPlans)
    if actual != expected:
        raise ValueError(
            "Batched refinement must contain exactly the requested assays: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )
    validated = {
        assay: validate_parameter_search_plan(
            plan.assayPlans[assay],
            dependencies[assay],
            initial_candidate_ids=initial_candidate_ids[assay],
            max_refined_candidates=max_refined_by_assay[assay],
        )
        for assay in dependencies
    }
    return plan.model_copy(update={"assayPlans": validated})


def validate_parameter_tuning_report(
    report: ParameterTuningReport,
    deps: ParameterTuningDependencies,
    *,
    search_plan: ParameterSearchPlan | None = None,
) -> ParameterTuningReport:
    """Ground the model report in candidate executions recorded by the tool."""

    evaluations = [
        deps.evaluations[candidate_id]
        for candidate_id in deps.executionOrder
        if candidate_id in deps.evaluations
    ]
    known_evidence = {
        evidence_id
        for evaluation in evaluations
        for evidence_id in evaluation.evidenceIds
    }
    cited_evidence = set(report.evidenceIds)
    for comparison in report.comparisons:
        cited_evidence.update(comparison.evidenceIds)
    if report.needsInput is not None:
        cited_evidence.update(report.needsInput.evidenceIds)
    unknown_evidence = sorted(cited_evidence - known_evidence)
    if unknown_evidence:
        raise ValueError(
            f"Parameter tuning report cites unknown evidence ids {unknown_evidence}"
        )
    if report.status == "done" and report.recommendedCandidateId is None:
        raise ValueError("A done tuning report must recommend an executed candidate")
    if report.status == "needsInput" and report.needsInput is None:
        raise ValueError("A needsInput tuning report must include a concrete question")
    successful = [
        evaluation for evaluation in evaluations if evaluation.status == "done"
    ]
    comparison_required = len(deps.candidates) > 1 and deps.maxCandidates > 1
    if report.status == "done":
        if not report.evidenceIds:
            raise ValueError("A done tuning report requires recommendation evidence")
        if comparison_required and len(successful) < 2:
            raise ValueError(
                "A completed tuning recommendation requires at least two successful "
                "candidate executions"
            )
        if (
            comparison_required
            and "baseline" in deps.candidates
            and not any(item.candidateId == "baseline" for item in successful)
        ):
            raise ValueError(
                "Evaluate the baseline before completing a multi-candidate comparison"
            )

    selected_artifacts: dict[str, ArtifactRecord] = {}
    if report.recommendedCandidateId is not None:
        selected = deps.evaluations.get(report.recommendedCandidateId)
        if selected is None:
            raise ValueError("Recommended candidate was not executed")
        if selected.status != "done":
            raise ValueError("Recommended candidate execution failed")
        if not selected.eligible:
            raise ValueError("Recommended candidate is not eligible")
        recommendation_prefix = f"candidate:{selected.candidateId}:"
        if not any(
            evidence_id.startswith(recommendation_prefix)
            for evidence_id in report.evidenceIds
        ):
            raise ValueError(
                "Recommendation evidence must include the selected candidate"
            )
        selected_artifacts = dict(selected.artifacts)

    if report.status == "done" and not comparison_required and report.comparisons:
        raise ValueError(
            "Candidate comparisons require a completed multi-candidate evaluation"
        )
    if report.status == "done" and comparison_required:
        assert report.recommendedCandidateId is not None
        successful_ids = {item.candidateId for item in successful}
        expected_comparators = successful_ids - {report.recommendedCandidateId}
        comparison_ids = [item.candidateId for item in report.comparisons]
        duplicate_comparators = sorted(
            {
                candidate_id
                for candidate_id in comparison_ids
                if comparison_ids.count(candidate_id) > 1
            }
        )
        if duplicate_comparators:
            raise ValueError(f"Duplicate candidate comparisons {duplicate_comparators}")
        actual_comparators = set(comparison_ids)
        missing_comparators = sorted(expected_comparators - actual_comparators)
        invalid_comparators = sorted(actual_comparators - expected_comparators)
        if missing_comparators:
            raise ValueError(
                "Completed tuning reports require comparisons for every successful "
                f"non-selected candidate: {missing_comparators}"
            )
        if invalid_comparators:
            raise ValueError(
                "Candidate comparisons must identify successful non-selected "
                f"candidates: {invalid_comparators}"
            )
        selected_prefix = f"candidate:{report.recommendedCandidateId}:"
        for comparison in report.comparisons:
            comparator_prefix = f"candidate:{comparison.candidateId}:"
            if not any(
                evidence_id.startswith(selected_prefix)
                for evidence_id in comparison.evidenceIds
            ):
                raise ValueError(
                    "Each candidate comparison must cite evidence from the "
                    "selected candidate"
                )
            if not any(
                evidence_id.startswith(comparator_prefix)
                for evidence_id in comparison.evidenceIds
            ):
                raise ValueError(
                    "Each candidate comparison must cite evidence from its comparator"
                )
            if not comparison.summary.strip():
                raise ValueError(
                    "Each candidate comparison requires a concise grounded summary"
                )

    return report.model_copy(
        update={
            "fromAssay": deps.fromAssay,
            "cellKey": deps.cellKey,
            "evaluations": evaluations,
            "selectedArtifacts": selected_artifacts,
            "searchPlan": search_plan,
            "assayReports": {},
            "recommendedByAssay": (
                {deps.fromAssay: report.recommendedCandidateId}
                if report.recommendedCandidateId is not None
                else {}
            ),
            "totalCandidates": len(evaluations),
            "integrationEvaluations": [],
            "recommendedIntegrationId": None,
            "finalClusterColumn": None,
            "finalClusterArtifact": None,
            "graphAssay": deps.fromAssay,
            "markerAssay": deps.fromAssay,
            "finalSelection": None,
        }
    )


def validate_parameter_tuning_batch_report(
    report: ParameterTuningReport,
    dependencies: Mapping[str, ParameterTuningDependencies],
    *,
    search_plans: Mapping[str, ParameterSearchPlan],
    primary_assay: str,
) -> ParameterTuningReport:
    """Ground one aggregate response in every assay's executed branches."""

    expected = set(dependencies)
    actual = set(report.assayReports)
    if actual != expected:
        raise ValueError(
            "Batched selection must contain exactly the requested assays: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )
    if primary_assay not in dependencies:
        raise ValueError(f"Unknown primary assay {primary_assay!r}")
    validated_reports = {
        assay: validate_parameter_tuning_report(
            report.assayReports[assay],
            dependencies[assay],
            search_plan=search_plans[assay],
        )
        for assay in dependencies
    }
    known_evidence = {
        evidence_id
        for assay_report in validated_reports.values()
        for evaluation in assay_report.evaluations
        for evidence_id in evaluation.evidenceIds
    }
    unknown_evidence = sorted(set(report.evidenceIds) - known_evidence)
    if unknown_evidence:
        raise ValueError(
            f"Batched tuning report cites unknown evidence ids {unknown_evidence}"
        )
    statuses = {assay_report.status for assay_report in validated_reports.values()}
    if statuses == {"done"}:
        status: StageStatus = "done"
    elif "needsInput" in statuses:
        status = "needsInput"
    else:
        status = "failed"
    primary = validated_reports[primary_assay]
    recommended = {
        assay: assay_report.recommendedCandidateId
        for assay, assay_report in validated_reports.items()
        if assay_report.recommendedCandidateId is not None
    }
    if status == "done" and len(recommended) != len(validated_reports):
        raise ValueError("Every completed assay report must recommend a candidate")
    return report.model_copy(
        update={
            "status": status,
            "fromAssay": primary_assay,
            "cellKey": primary.cellKey,
            "evaluations": primary.evaluations,
            "recommendedCandidateId": primary.recommendedCandidateId,
            "selectedArtifacts": primary.selectedArtifacts,
            "needsInput": primary.needsInput if status != "done" else None,
            "searchPlan": primary.searchPlan,
            "assayReports": validated_reports,
            "recommendedByAssay": recommended,
            "totalCandidates": sum(
                len(item.evaluations) for item in validated_reports.values()
            ),
            "integrationEvaluations": [],
            "recommendedIntegrationId": None,
            "finalClusterColumn": None,
            "finalClusterArtifact": None,
            "graphAssay": primary_assay,
            "markerAssay": primary_assay,
            "finalSelection": None,
        }
    )


def fallback_parameter_tuning_report(
    deps: ParameterTuningDependencies,
    *,
    search_plan: ParameterSearchPlan,
    agent_name: str,
) -> ParameterTuningReport:
    """Retain the first eligible branch when structured selection is unavailable."""

    evaluations = [
        deps.evaluations[candidate_id]
        for candidate_id in deps.executionOrder
        if candidate_id in deps.evaluations
    ]
    successful = [item for item in evaluations if item.status == "done"]
    eligible = [item for item in successful if item.eligible]
    comparison_required = len(deps.candidates) > 1 and deps.maxCandidates > 1
    evidence_by_candidate = {
        item.candidateId: next(
            (
                evidence_id
                for evidence_id in item.evidenceIds
                if evidence_id == f"candidate:{item.candidateId}:clusters"
            ),
            next(iter(item.evidenceIds), None),
        )
        for item in successful
    }
    cannot_recommend = (
        not eligible
        or (comparison_required and len(successful) < 2)
        or any(evidence_by_candidate[item.candidateId] is None for item in successful)
    )
    if cannot_recommend:
        logger.warning(
            f"Parameter tuning fallback for assay {deps.fromAssay!r} requires "
            f"input: completed={len(successful)}, eligible={len(eligible)}"
        )
        known_evidence = sorted(
            {
                evidence_id
                for evaluation in evaluations
                for evidence_id in evaluation.evidenceIds
            }
        )
        report = ParameterTuningReport(
            status="needsInput",
            confidence="low",
            rationale=(
                "Structured model selection was unavailable and the executed "
                "screen does not support a conservative automatic fallback."
            ),
            limitations=[
                "No parameter branch was selected without complete eligible "
                "executor evidence."
            ],
            stopReason="The bounded screen completed without an automatic choice.",
            needsInput=ParameterTuningNeedsInput(
                question="Select one eligible executed parameter candidate.",
                options=[item.candidateId for item in eligible],
                evidenceIds=known_evidence,
            ),
            runInfo=AgentRunInfo(agentName=agent_name),
        )
        return validate_parameter_tuning_report(
            report,
            deps,
            search_plan=search_plan,
        )

    selected = eligible[0]
    logger.warning(
        f"Parameter tuning fallback retained candidate "
        f"{selected.candidateId!r} for assay {deps.fromAssay!r} from "
        f"{len(eligible)} eligible candidate(s)"
    )
    selected_evidence = evidence_by_candidate[selected.candidateId]
    assert selected_evidence is not None
    comparisons: list[CandidateComparison] = []
    if comparison_required:
        for item in successful:
            if item.candidateId == selected.candidateId:
                continue
            comparator_evidence = evidence_by_candidate[item.candidateId]
            assert comparator_evidence is not None
            comparisons.append(
                CandidateComparison(
                    candidateId=item.candidateId,
                    summary=(
                        "This executed branch remains a grounded comparator to the "
                        "conservatively retained first eligible branch."
                    ),
                    evidenceIds=[selected_evidence, comparator_evidence],
                )
            )
    report = ParameterTuningReport(
        status="done",
        recommendedCandidateId=selected.candidateId,
        confidence="low",
        rationale=(
            "Structured model selection was unavailable after bounded retries; "
            "the first eligible authorized branch was retained conservatively."
        ),
        evidenceIds=[selected_evidence],
        comparisons=comparisons,
        tradeoffs=["No model-authored metric trade-off ranking was available."],
        limitations=[
            "The fallback does not claim that the retained branch is metric-optimal."
        ],
        stopReason=(
            "The deterministic screen completed and retained its first eligible "
            "authorized branch."
        ),
        runInfo=AgentRunInfo(agentName=agent_name),
    )
    return validate_parameter_tuning_report(
        report,
        deps,
        search_plan=search_plan,
    )


def fallback_parameter_tuning_batch_report(
    dependencies: Mapping[str, ParameterTuningDependencies],
    *,
    search_plans: Mapping[str, ParameterSearchPlan],
    primary_assay: str,
) -> ParameterTuningReport:
    """Build one grounded aggregate fallback over completed assay screens."""

    logger.warning(
        f"Using parameter tuning batch fallback for {len(dependencies)} assay(s)"
    )
    assay_reports = {
        assay: fallback_parameter_tuning_report(
            deps,
            search_plan=search_plans[assay],
            agent_name="parameter_tuning_batch_fallback",
        )
        for assay, deps in dependencies.items()
    }
    if any(item.status != "done" for item in assay_reports.values()):
        primary = assay_reports[primary_assay]
        if primary.status == "done":
            primary = ParameterTuningReport(
                status="needsInput",
                confidence="low",
                rationale=(
                    "At least one assay lacks a conservative automatic parameter "
                    "selection."
                ),
                limitations=[
                    "The multimodal native screen requires an explicit selection."
                ],
                stopReason="The bounded native screens completed without all choices.",
                needsInput=ParameterTuningNeedsInput(
                    question="Select eligible parameter candidates for every assay.",
                    options=[],
                    evidenceIds=list(primary.evidenceIds),
                ),
                runInfo=AgentRunInfo(agentName="parameter_tuning_batch_fallback"),
            )
            assay_reports[primary_assay] = validate_parameter_tuning_report(
                primary,
                dependencies[primary_assay],
                search_plan=search_plans[primary_assay],
            )
    aggregate = ParameterTuningReport(
        status=(
            "done"
            if all(item.status == "done" for item in assay_reports.values())
            else "needsInput"
        ),
        assayReports=assay_reports,
        rationale=(
            "Structured model selection was unavailable; each completed native "
            "screen used the conservative fallback policy."
        ),
        evidenceIds=list(
            dict.fromkeys(
                evidence_id
                for assay_report in assay_reports.values()
                for evidence_id in assay_report.evidenceIds
            )
        ),
        limitations=[
            "Fallback recommendations retain first eligible authorized branches "
            "without claiming a metric-optimal ranking."
        ],
        stopReason="The bounded native screens completed.",
        runInfo=AgentRunInfo(agentName="parameter_tuning_batch_fallback"),
    )
    logger.warning(
        f"Parameter tuning batch fallback status={aggregate.status}; "
        f"completed_assays={sum(item.status == 'done' for item in assay_reports.values())}"
    )
    return validate_parameter_tuning_batch_report(
        aggregate,
        dependencies,
        search_plans=search_plans,
        primary_assay=primary_assay,
    )


def validate_final_graph_selection(
    selection: FinalGraphSelection,
    report: ParameterTuningReport,
    *,
    integration_evaluations: Sequence[IntegrationCandidateEvaluation],
    marker_assay: str,
) -> FinalGraphSelection:
    """Ground a final graph choice in the exact eligible executor outputs."""

    if report.status != "done":
        raise ValueError("Native parameter tuning must finish before graph selection")
    options = final_graph_options(report, integration_evaluations)
    if not options:
        raise ValueError("No eligible native or integrated graph options are available")
    known_evidence = {
        evidence_id
        for option in options.values()
        for evidence_id in option["evidenceIds"]
    }
    cited = set(selection.evidenceIds)
    for comparison in selection.comparisons:
        cited.update(comparison.evidenceIds)
    if selection.needsInput is not None:
        cited.update(selection.needsInput.evidenceIds)
    unknown = sorted(cited - known_evidence)
    if unknown:
        raise ValueError(f"Final graph selection cites unknown evidence ids {unknown}")
    if selection.status == "needsInput":
        if selection.needsInput is None or not selection.needsInput.question.strip():
            raise ValueError(
                "A needsInput graph selection requires a concrete question"
            )
        return selection.model_copy(
            update={
                "selectedOptionId": None,
                "graphMethod": None,
                "nativeAssay": None,
                "nativeCandidateId": None,
                "integrationId": None,
                "markerAssay": marker_assay,
            }
        )
    if selection.status != "done":
        raise ValueError("Final graph selection must be done or needsInput")
    if selection.selectedOptionId not in options:
        raise ValueError("Selected final graph option is not eligible")
    assert selection.selectedOptionId is not None
    selected = options[selection.selectedOptionId]
    selected_evidence = set(selected["evidenceIds"])
    if not selected_evidence.intersection(selection.evidenceIds):
        raise ValueError(
            "Final graph recommendation must cite selected-option evidence"
        )
    expected_comparators = set(options) - {selection.selectedOptionId}
    comparison_ids = [item.optionId for item in selection.comparisons]
    if len(set(comparison_ids)) != len(comparison_ids):
        raise ValueError("Final graph comparisons must not contain duplicates")
    if set(comparison_ids) != expected_comparators:
        raise ValueError(
            "Final graph selection requires one comparison for every eligible "
            "non-selected option"
        )
    for comparison in selection.comparisons:
        comparator_evidence = set(options[comparison.optionId]["evidenceIds"])
        if not selected_evidence.intersection(comparison.evidenceIds):
            raise ValueError(
                "Every final graph comparison must cite selected-option evidence"
            )
        if not comparator_evidence.intersection(comparison.evidenceIds):
            raise ValueError(
                "Every final graph comparison must cite comparator evidence"
            )
        if not comparison.summary.strip():
            raise ValueError("Every final graph comparison requires a summary")
    return selection.model_copy(
        update={
            "graphMethod": selected["graphMethod"],
            "nativeAssay": selected.get("nativeAssay"),
            "nativeCandidateId": selected.get("nativeCandidateId"),
            "integrationId": selected.get("integrationId"),
            "markerAssay": marker_assay,
            "needsInput": None,
        }
    )


def finalize_parameter_tuning_selection(
    report: ParameterTuningReport,
    *,
    marker_assay: str,
    integration_evaluations: Sequence[IntegrationCandidateEvaluation] = (),
    recommended_integration_id: str | None = None,
    native_assay: str | None = None,
    final_selection: FinalGraphSelection | None = None,
) -> ParameterTuningReport:
    """Attach an executor-selected native or integrated final cluster branch."""

    logger.debug(
        f"Finalizing parameter graph selection: marker_assay={marker_assay!r}, "
        f"integration_candidates={len(integration_evaluations)}"
    )
    if report.status != "done":
        raise ValueError("Parameter tuning must be done before final graph selection")
    if not marker_assay:
        raise ValueError("marker_assay must be non-empty")
    assay_reports = report.assayReports or {report.fromAssay: report}
    if marker_assay not in assay_reports:
        raise ValueError(f"Unknown marker assay {marker_assay!r}")
    evaluations = list(integration_evaluations)
    integration_ids = [item.integrationId for item in evaluations]
    if len(set(integration_ids)) != len(integration_ids):
        raise ValueError("Integration evaluation ids must be unique")
    if recommended_integration_id is not None and native_assay is not None:
        raise ValueError("Choose either an integrated graph or one native assay")
    if recommended_integration_id is not None:
        selected = next(
            (
                item
                for item in evaluations
                if item.integrationId == recommended_integration_id
            ),
            None,
        )
        if selected is None:
            raise ValueError("Recommended integration candidate was not evaluated")
        if selected.status != "done" or not selected.eligible:
            raise ValueError("Recommended integration candidate is not eligible")
        if selected.clusterArtifact is None or not selected.clusterColumn:
            raise ValueError("Recommended integration lacks an exact cluster artifact")
        if (
            selected.clusterArtifact.scope != "datastore"
            or selected.clusterArtifact.assay is not None
        ):
            raise ValueError(
                "Integrated cluster artifacts must be datastore-scoped without assay"
            )
        if (
            selected.graphArtifact is None
            or selected.graphArtifact.scope != "datastore"
        ):
            raise ValueError("Integrated graph artifact must be datastore-scoped")
        cluster_artifact = selected.clusterArtifact
        cluster_column = selected.clusterColumn
        graph_assay = None
    else:
        selected_assay = native_assay or report.fromAssay
        primary = assay_reports.get(selected_assay)
        if primary is None or primary.recommendedCandidateId is None:
            raise ValueError("Selected assay lacks a native tuning recommendation")
        selected_native = next(
            (
                item
                for item in primary.evaluations
                if item.candidateId == primary.recommendedCandidateId
            ),
            None,
        )
        if (
            selected_native is None
            or selected_native.status != "done"
            or not selected_native.eligible
            or selected_native.clusterColumn is None
            or "clusters" not in selected_native.artifacts
        ):
            raise ValueError("Primary native recommendation lacks exact clusters")
        cluster_artifact = selected_native.artifacts["clusters"]
        cluster_column = selected_native.clusterColumn
        graph_assay = selected_assay
    finalized = report.model_copy(
        update={
            "totalCandidates": (
                sum(len(value.evaluations) for value in assay_reports.values())
                + len(evaluations)
            ),
            "integrationEvaluations": evaluations,
            "recommendedIntegrationId": recommended_integration_id,
            "finalClusterColumn": cluster_column,
            "finalClusterArtifact": cluster_artifact,
            "graphAssay": graph_assay,
            "markerAssay": marker_assay,
            "finalSelection": final_selection,
        }
    )
    selected_graph = recommended_integration_id or graph_assay
    logger.info(
        f"Finalized parameter graph selection: graph={selected_graph!r}, "
        f"marker_assay={marker_assay!r}, cluster_column={cluster_column!r}"
    )
    return finalized


def select_final_parameter_graph(
    *,
    model: Any,
    report: ParameterTuningReport,
    integration_evaluations: Sequence[IntegrationCandidateEvaluation],
    marker_assay: str,
    config: AgentRunConfig | None = None,
) -> ParameterTuningReport:
    """Use one bounded provider call to select and attach the final graph."""

    evaluations = list(integration_evaluations)
    if not marker_assay:
        raise ValueError("marker_assay must be non-empty")
    assay_reports = report.assayReports or {report.fromAssay: report}
    if marker_assay not in assay_reports:
        raise ValueError(f"Unknown marker assay {marker_assay!r}")
    options = final_graph_options(report, evaluations)
    if not options:
        raise ValueError("No eligible native or integrated graph options are available")
    logger.info(
        f"Selecting final parameter graph from {len(options)} eligible option(s); "
        f"marker_assay={marker_assay!r}"
    )
    if len(options) == 1:
        option_id, option = next(iter(options.items()))
        logger.info(
            f"Selecting sole eligible final graph option {option_id!r} "
            "without a provider request"
        )
        selection = validate_final_graph_selection(
            FinalGraphSelection(
                status="done",
                selectedOptionId=option_id,
                markerAssay=marker_assay,
                confidence="high",
                rationale="The executor produced exactly one eligible graph option.",
                evidenceIds=list(option["evidenceIds"]),
                limitations=["No alternative eligible final graph required ranking."],
                runInfo=AgentRunInfo(
                    agentName="parameter_tuning_final_graph_deterministic"
                ),
            ),
            report,
            integration_evaluations=evaluations,
            marker_assay=marker_assay,
        )
    else:
        run_config = (config or AgentRunConfig()).with_limits(
            request_limit=6,
            tool_call_limit=5,
            output_token_limit=32768,
            timeout_seconds=600.0,
        )
        try:
            logger.info(
                f"Requesting final graph selection across {len(options)} "
                "eligible options"
            )
            execution = run_agent_sync(
                model=model,
                output_type=FinalGraphSelection,
                system_prompt=final_graph_selection_system_prompt(),
                user_prompt=final_graph_selection_prompt(
                    report=report,
                    integration_evaluations=evaluations,
                    marker_assay=marker_assay,
                ),
                deps_type=ParameterTuningDependencies,
                deps=ParameterTuningDependencies.get_blank(),
                config=run_config,
                name="parameter_tuning_final_graph",
                output_validator=lambda proposed: validate_final_graph_selection(
                    proposed,
                    report,
                    integration_evaluations=evaluations,
                    marker_assay=marker_assay,
                ),
            )
        except UnexpectedModelBehavior:
            option_ids = sorted(options)
            logger.warning(
                "Final graph selection exhausted structured-output retries; "
                f"requesting input for {len(option_ids)} eligible options"
            )
            selection = validate_final_graph_selection(
                FinalGraphSelection(
                    status="needsInput",
                    markerAssay=marker_assay,
                    confidence="low",
                    rationale=(
                        "Structured final-graph selection was unavailable after "
                        "bounded retries."
                    ),
                    limitations=[
                        "No ranking was invented across multiple eligible graphs."
                    ],
                    needsInput=FinalGraphNeedsInput(
                        question="Select one eligible final graph option.",
                        options=option_ids,
                        evidenceIds=sorted(
                            {
                                evidence_id
                                for option in options.values()
                                for evidence_id in option["evidenceIds"]
                            }
                        ),
                    ),
                    runInfo=AgentRunInfo(
                        agentName="parameter_tuning_final_graph_fallback"
                    ),
                ),
                report,
                integration_evaluations=evaluations,
                marker_assay=marker_assay,
            )
        else:
            if not isinstance(execution.output, FinalGraphSelection):
                raise TypeError(
                    "Final graph selector returned an unexpected output type"
                )
            selection = validate_final_graph_selection(
                execution.output,
                report,
                integration_evaluations=evaluations,
                marker_assay=marker_assay,
            ).model_copy(update={"runInfo": execution.runInfo})
            logger.info(
                f"Provider selected final graph option {selection.selectedOptionId!r}"
            )
    if selection.status == "needsInput":
        needs_input = selection.needsInput or FinalGraphNeedsInput.get_blank()
        option_ids = sorted(options)
        canonical_question = "Select one eligible final graph option."
        canonical_needs_input = needs_input.model_copy(
            update={"question": canonical_question, "options": option_ids}
        )
        logger.warning(
            f"Final parameter graph selection needs input; options={len(option_ids)}"
        )
        return report.model_copy(
            update={
                "status": "needsInput",
                "totalCandidates": (
                    sum(len(value.evaluations) for value in assay_reports.values())
                    + len(evaluations)
                ),
                "integrationEvaluations": evaluations,
                "markerAssay": marker_assay,
                "finalSelection": selection.model_copy(
                    update={"needsInput": canonical_needs_input}
                ),
                "needsInput": ParameterTuningNeedsInput(
                    question=canonical_question,
                    options=option_ids,
                    evidenceIds=needs_input.evidenceIds,
                ),
            }
        )
    return finalize_parameter_tuning_selection(
        report,
        marker_assay=marker_assay,
        integration_evaluations=evaluations,
        recommended_integration_id=selection.integrationId,
        native_assay=selection.nativeAssay,
        final_selection=selection,
    )


def promote_parameter_candidate(
    store: Any,
    *,
    report: ParameterTuningReport,
    normalized: Any,
    identity_feature_limit: int = 64,
) -> ParameterCandidateEvaluation:
    """Replay the selected native branch, publish it, and verify artifact reuse."""

    if report.status != "done" or report.recommendedCandidateId is None:
        raise ValueError("A completed native tuning recommendation is required")
    evaluation = next(
        (
            item
            for item in report.evaluations
            if item.candidateId == report.recommendedCandidateId
        ),
        None,
    )
    if evaluation is None or evaluation.status != "done" or not evaluation.eligible:
        raise ValueError("Recommended candidate is not an eligible execution")
    if evaluation.clusterLabel is None or evaluation.clusterColumn is None:
        raise ValueError("Recommended candidate lacks its exact cluster label")
    logger.info(
        f"Promoting parameter candidate {evaluation.candidateId!r} for assay "
        f"{report.fromAssay!r}"
    )
    candidate = evaluation.parameters
    normalized_shape = normalized_artifact_shape(store, normalized)
    reduction_ref, reduction_key, effective_dimensions = run_candidate_reduction(
        store,
        normalized=normalized,
        from_assay=report.fromAssay,
        candidate=candidate,
        normalized_shape=normalized_shape,
        update_state=True,
        identity_feature_limit=identity_feature_limit,
    )
    promoted: dict[str, ArtifactRecord] = {
        reduction_key: ArtifactRecord.from_ref(reduction_ref)
    }
    coordinates_ref = reduction_ref
    if candidate.useHarmony:
        if not evaluation.harmonyBatchColumns:
            raise ValueError("Selected Harmony branch lacks exact batch columns")
        coordinates_ref = store.run_harmony(
            list(evaluation.harmonyBatchColumns),
            reduction=reduction_ref,
            from_assay=report.fromAssay,
            update_state=True,
            invalidate_cache=False,
        )
        promoted["harmony"] = ArtifactRecord.from_ref(coordinates_ref)
    ann_ref = store.build_ann_index(
        coordinates=coordinates_ref,
        from_assay=report.fromAssay,
        ann_metric="l2",
        ann_parallel=False,
        rand_state=CONFIG._PCA_RANDOM_SEED,
        update_state=True,
        invalidate_cache=False,
    )
    promoted["annIndex"] = ArtifactRecord.from_ref(ann_ref)
    neighbors_ref = store.query_neighbors(
        ann_index=ann_ref,
        from_assay=report.fromAssay,
        coordinates=coordinates_ref,
        k=candidate.neighborsK,
        update_state=True,
        invalidate_cache=False,
    )
    promoted["neighbors"] = ArtifactRecord.from_ref(neighbors_ref)
    graph_ref = store.build_connectivity_map(
        neighbors=neighbors_ref,
        from_assay=report.fromAssay,
        local_connectivity=1.0,
        bandwidth=1.5,
        update_state=True,
        invalidate_cache=False,
    )
    promoted["connectivityMap"] = ArtifactRecord.from_ref(graph_ref)
    cluster_ref = store.run_leiden_clustering(
        graph=graph_ref,
        from_assay=report.fromAssay,
        cell_key=report.cellKey,
        resolution=candidate.leidenResolution,
        backend="igraph",
        symmetric_graph=False,
        graph_upper_only=False,
        label=evaluation.clusterLabel,
        random_seed=CONFIG._RANDOM_SEED,
        invalidate_cache=False,
    )
    promoted["clusters"] = ArtifactRecord.from_ref(cluster_ref)
    mismatches = sorted(
        key
        for key, expected in evaluation.artifacts.items()
        if promoted.get(key) != expected
    )
    unexpected = sorted(set(promoted) - set(evaluation.artifacts))
    if mismatches or unexpected:
        logger.error(
            f"Parameter candidate promotion artifact verification failed: "
            f"mismatches={len(mismatches)}, unexpected={len(unexpected)}"
        )
        raise RuntimeError(
            "Promoted branch did not reuse evaluated artifacts: "
            f"mismatches={mismatches}, unexpected={unexpected}"
        )
    promoted_evaluation = evaluation.model_copy(
        update={
            "artifacts": promoted,
            "effectiveDimensions": effective_dimensions,
        }
    )
    logger.info(
        f"Promoted parameter candidate {evaluation.candidateId!r}; "
        f"verified_artifacts={len(promoted)}"
    )
    return promoted_evaluation


def _resolve_experimental_tuning_handoff(
    *,
    cell_key: str,
    batch_columns: Sequence[str],
    preservation_columns: Sequence[str],
    experimental_handoff: ExperimentalTuningHandoff | None,
) -> tuple[str, list[str], list[str]]:
    resolved_cell_key = cell_key
    resolved_batch_columns = list(batch_columns)
    resolved_preservation_columns = list(preservation_columns)
    if experimental_handoff is None:
        return (
            resolved_cell_key,
            resolved_batch_columns,
            resolved_preservation_columns,
        )

    handoff_batch_columns = list(experimental_handoff.batchColumns)
    canonical_batch_columns = sorted(set(handoff_batch_columns))
    if len(canonical_batch_columns) != len(handoff_batch_columns):
        raise ValueError("experimental_handoff batch columns must be unique")
    if cell_key != "I" and cell_key != experimental_handoff.cellKey:
        raise ValueError("cell_key conflicts with experimental_handoff")
    if resolved_batch_columns and sorted(resolved_batch_columns) != (
        canonical_batch_columns
    ):
        raise ValueError("batch_columns conflict with experimental_handoff")
    if resolved_preservation_columns and resolved_preservation_columns != list(
        experimental_handoff.preservationColumns
    ):
        raise ValueError("preservation_columns conflict with experimental_handoff")
    if experimental_handoff.batchAction == "needsInput":
        raise ValueError("Experimental Context requires input before tuning")
    if experimental_handoff.batchAction == "skip" and experimental_handoff.batchColumns:
        raise ValueError("A skip handoff must not contain batch columns")
    if experimental_handoff.batchAction == "evaluateHarmony":
        expected_coefficients = set(experimental_handoff.coefficientsOfInterest)
        safe_coefficients = {
            item.coefficient
            for item in experimental_handoff.batchSafety
            if item.status == "safe" and item.batchColumns == canonical_batch_columns
        }
        if (
            not expected_coefficients
            or not canonical_batch_columns
            or safe_coefficients != expected_coefficients
        ):
            raise ValueError(
                "Harmony handoff lacks safe evidence for every coefficient"
            )
    if experimental_handoff.batchAction == "unsafe":
        expected_coefficients = set(experimental_handoff.coefficientsOfInterest)
        exact_safety = [
            item
            for item in experimental_handoff.batchSafety
            if item.batchColumns == canonical_batch_columns
            and item.coefficient in expected_coefficients
        ]
        if (
            not expected_coefficients
            or {item.coefficient for item in exact_safety} != expected_coefficients
            or any(item.status == "notComputed" for item in exact_safety)
            or not any(item.status == "unsafe" for item in exact_safety)
        ):
            raise ValueError("Unsafe handoff lacks exact unsafe batch evidence")
    if any(
        item.evidenceId not in experimental_handoff.evidenceIds
        for item in experimental_handoff.batchSafety
    ):
        raise ValueError("Experimental handoff does not cite its batch evidence")
    return (
        experimental_handoff.cellKey,
        canonical_batch_columns,
        list(experimental_handoff.preservationColumns),
    )


def prepare_parameter_tuning_dependencies(
    store: Any,
    *,
    normalized: Any,
    from_assay: str,
    cell_key: str = "I",
    candidates: Sequence[ParameterCandidate] | None = None,
    batch_columns: Sequence[str] = (),
    preservation_columns: Sequence[str] = (),
    experimental_handoff: ExperimentalTuningHandoff | None = None,
    max_candidates: int = 5,
    max_refined_candidates: int = 0,
    allow_harmony_refinement: bool = True,
    min_cluster_cells: int = 20,
    identity_feature_limit: int = 64,
) -> tuple[ParameterTuningDependencies, list[str]]:
    """Validate one assay request and construct branch-safe dependencies."""

    if max_candidates < 1:
        raise ValueError("max_candidates must be at least one")
    if max_refined_candidates < 0:
        raise ValueError("max_refined_candidates must be non-negative")
    if min_cluster_cells < 1:
        raise ValueError("min_cluster_cells must be at least one")
    if identity_feature_limit < 2:
        raise ValueError("identity_feature_limit must be at least two")
    normalized_kind = getattr(normalized, "kind", None)
    normalized_assay = getattr(normalized, "assay", None)
    if normalized_kind is not None and normalized_kind != "normalized":
        raise ValueError("normalized must identify a normalized artifact")
    if normalized_assay is not None and normalized_assay != from_assay:
        raise ValueError("normalized artifact belongs to a different assay")
    (
        resolved_cell_key,
        resolved_batch_columns,
        resolved_preservation_columns,
    ) = _resolve_experimental_tuning_handoff(
        cell_key=cell_key,
        batch_columns=batch_columns,
        preservation_columns=preservation_columns,
        experimental_handoff=experimental_handoff,
    )
    if len(set(resolved_batch_columns)) != len(resolved_batch_columns):
        raise ValueError("batch_columns must be unique")
    seed_candidates = (
        get_default_parameter_candidates() if candidates is None else list(candidates)
    )
    if not seed_candidates:
        raise ValueError("candidates must be non-empty")
    if len(seed_candidates) > max_candidates:
        raise ValueError(
            f"Initial candidate count exceeds max_candidates={max_candidates}"
        )
    pair_harmony = (
        experimental_handoff is not None
        and experimental_handoff.batchAction == "evaluateHarmony"
    )
    candidate_values = build_initial_parameter_candidates(
        seed_candidates,
        pair_harmony=pair_harmony,
    )
    if len(candidate_values) + max_refined_candidates > CONFIG._MAX_CANDIDATES_OFFERED:
        raise ValueError(
            "Initial and refined candidates may contain at most "
            f"{CONFIG._MAX_CANDIDATES_OFFERED} values"
        )
    candidate_map: dict[str, ParameterCandidate] = {}
    for candidate in candidate_values:
        if not CONFIG._CANDIDATE_ID.fullmatch(candidate.candidateId):
            raise ValueError(
                "candidateId must contain only ASCII letters, numbers, and underscores"
            )
        if candidate.candidateId in candidate_map:
            raise ValueError(f"Duplicate candidateId {candidate.candidateId!r}")
        if candidate.useHarmony and not resolved_batch_columns:
            raise ValueError(
                f"Candidate {candidate.candidateId!r} requires batch_columns"
            )
        if (
            candidate.useHarmony
            and experimental_handoff is not None
            and experimental_handoff.batchAction != "evaluateHarmony"
        ):
            raise ValueError(
                f"Candidate {candidate.candidateId!r} is not authorized for Harmony"
            )
        candidate_map[candidate.candidateId] = candidate
    harmony_authorized = (
        allow_harmony_refinement
        and bool(resolved_batch_columns)
        and (
            experimental_handoff is None
            or experimental_handoff.batchAction == "evaluateHarmony"
        )
    )
    normalized_shape = normalized_artifact_shape(store, normalized)
    deps = ParameterTuningDependencies(
        store=store,
        normalized=normalized,
        normalizedShape=normalized_shape,
        fromAssay=from_assay,
        cellKey=resolved_cell_key,
        candidates=candidate_map,
        candidatePhases={candidate_id: "initial" for candidate_id in candidate_map},
        batchColumns=tuple(resolved_batch_columns),
        preservationColumns=tuple(resolved_preservation_columns),
        harmonyAuthorized=harmony_authorized,
        maxCandidates=len(candidate_values) + max_refined_candidates,
        minClusterCells=min_cluster_cells,
        identityFeatureLimit=identity_feature_limit,
    )
    return deps, list(candidate_map)


class ParameterTuningAgent:
    """Run bounded tuning over caller-authorized Scarf candidates."""

    def __init__(
        self,
        model: Any,
        *,
        config: AgentRunConfig | None = None,
    ) -> None:
        self.model = model
        self.config = (config or AgentRunConfig()).with_limits(
            request_limit=6,
            tool_call_limit=5,
            output_token_limit=32768,
            timeout_seconds=600.0,
        )

    def run(
        self,
        store: Any,
        *,
        normalized: Any,
        from_assay: str,
        cell_key: str = "I",
        candidates: Sequence[ParameterCandidate] | None = None,
        batch_columns: Sequence[str] = (),
        preservation_columns: Sequence[str] = (),
        experimental_handoff: ExperimentalTuningHandoff | None = None,
        max_candidates: int = 5,
        max_refined_candidates: int = 0,
        min_cluster_cells: int = 20,
        identity_feature_limit: int = 64,
    ) -> ParameterTuningReport:
        """Run deterministic screening, optional refinement, and final selection."""
        return tune_parameters(
            store,
            model=self.model,
            normalized=normalized,
            from_assay=from_assay,
            cell_key=cell_key,
            candidates=candidates,
            batch_columns=batch_columns,
            preservation_columns=preservation_columns,
            experimental_handoff=experimental_handoff,
            max_candidates=max_candidates,
            max_refined_candidates=max_refined_candidates,
            min_cluster_cells=min_cluster_cells,
            identity_feature_limit=identity_feature_limit,
            config=self.config,
        )

    def promote(
        self,
        store: Any,
        *,
        report: ParameterTuningReport,
        normalized: Any,
        identity_feature_limit: int = 64,
    ) -> ParameterCandidateEvaluation:
        """Promote the selected native branch into the active assay state."""

        return promote_parameter_candidate(
            store,
            report=report,
            normalized=normalized,
            identity_feature_limit=identity_feature_limit,
        )

    def run_batch(
        self,
        store: Any,
        *,
        assays: Sequence[ParameterTuningAssayInput],
        primary_assay: str | None = None,
        max_total_candidates: int = 24,
        selection_directions: str = "",
    ) -> ParameterTuningReport:
        """Tune several assays with one planning and one selection request."""

        return tune_parameters_batch(
            store,
            model=self.model,
            assays=assays,
            primary_assay=primary_assay,
            max_total_candidates=max_total_candidates,
            selection_directions=selection_directions,
            config=self.config,
        )

    def select_final(
        self,
        *,
        report: ParameterTuningReport,
        integration_evaluations: Sequence[IntegrationCandidateEvaluation],
        marker_assay: str,
    ) -> ParameterTuningReport:
        """Select native, SNN, or WNN once and attach the final branch."""

        return select_final_parameter_graph(
            model=self.model,
            report=report,
            integration_evaluations=integration_evaluations,
            marker_assay=marker_assay,
            config=self.config,
        )


def _execute_parameter_candidates(
    deps: ParameterTuningDependencies,
    candidate_ids: Sequence[str],
) -> None:
    logger.info(
        f"Executing {len(candidate_ids)} parameter candidate(s) for assay "
        f"{deps.fromAssay!r}"
    )
    for candidate_id in candidate_ids:
        execute_parameter_candidate(deps, candidate_id)


def _register_refined_parameter_candidates(
    deps: ParameterTuningDependencies,
    candidates: Sequence[ParameterCandidate],
) -> None:
    if candidates:
        logger.info(
            f"Executing {len(candidates)} refined parameter candidate(s) for "
            f"assay {deps.fromAssay!r}"
        )
    for candidate in candidates:
        deps.candidates[candidate.candidateId] = candidate
        deps.candidatePhases[candidate.candidateId] = "refined"
        execute_parameter_candidate(deps, candidate.candidateId)


def tune_parameters_batch(
    store: Any,
    *,
    model: Any,
    assays: Sequence[ParameterTuningAssayInput],
    primary_assay: str | None = None,
    max_total_candidates: int = 24,
    selection_directions: str = "",
    config: AgentRunConfig | None = None,
) -> ParameterTuningReport:
    """Execute and select modality-specific native branches in two model calls."""

    assay_inputs = list(assays)
    if not assay_inputs:
        raise ValueError("assays must contain at least one tuning input")
    if max_total_candidates < 1:
        raise ValueError("max_total_candidates must be at least one")
    assay_names = [item.fromAssay for item in assay_inputs]
    if any(not assay for assay in assay_names):
        raise ValueError("Every batched tuning input requires fromAssay")
    if len(set(assay_names)) != len(assay_names):
        raise ValueError("Batched tuning assay names must be unique")
    resolved_primary = primary_assay or assay_names[0]
    if resolved_primary not in assay_names:
        raise ValueError(f"Unknown primary assay {resolved_primary!r}")
    logger.info(
        f"Starting batched parameter tuning for {len(assay_names)} assay(s); "
        f"primary_assay={resolved_primary!r}, "
        f"candidate_limit={max_total_candidates}"
    )

    dependencies: dict[str, ParameterTuningDependencies] = {}
    initial_ids: dict[str, list[str]] = {}
    max_refined_by_assay: dict[str, int] = {}
    for item in assay_inputs:
        deps, candidate_ids = prepare_parameter_tuning_dependencies(
            store,
            normalized=item.normalized,
            from_assay=item.fromAssay,
            cell_key=item.cellKey,
            candidates=item.candidates or None,
            batch_columns=item.batchColumns,
            preservation_columns=item.preservationColumns,
            experimental_handoff=item.experimentalHandoff,
            max_candidates=item.maxCandidates,
            max_refined_candidates=item.maxRefinedCandidates,
            allow_harmony_refinement=item.allowHarmonyRefinement,
            min_cluster_cells=item.minClusterCells,
            identity_feature_limit=item.identityFeatureLimit,
        )
        dependencies[item.fromAssay] = deps
        initial_ids[item.fromAssay] = candidate_ids
        max_refined_by_assay[item.fromAssay] = item.maxRefinedCandidates
    cell_keys = {deps.cellKey for deps in dependencies.values()}
    if len(cell_keys) != 1:
        raise ValueError("Batched tuning inputs must use the same cellKey")
    planned_total = sum(deps.maxCandidates for deps in dependencies.values())
    if planned_total > max_total_candidates:
        raise ValueError(
            f"Batched tuning requests {planned_total} candidate branches; "
            f"the global limit is {max_total_candidates}"
        )
    for assay in assay_names:
        deps = dependencies[assay]
        _execute_parameter_candidates(deps, initial_ids[assay])
    logger.info(
        "Completed batched initial parameter screen: "
        + ", ".join(
            f"{assay}={len(dependencies[assay].evaluations)}" for assay in assay_names
        )
    )

    run_config = (config or AgentRunConfig()).with_limits(
        request_limit=6,
        tool_call_limit=5,
        output_token_limit=32768,
        timeout_seconds=600.0,
    )
    if any(max_refined_by_assay.values()):
        try:
            logger.info(
                "Requesting one batched parameter refinement plan for "
                f"{len(assay_names)} assay(s)"
            )
            planning_execution = run_agent_sync(
                model=model,
                output_type=ParameterTuningBatchSearchPlan,
                system_prompt=parameter_batch_search_system_prompt(),
                user_prompt=parameter_batch_search_prompt(
                    dependencies,
                    max_refined_by_assay,
                ),
                deps_type=ParameterTuningDependencies,
                deps=dependencies[resolved_primary],
                config=run_config,
                name="parameter_batch_search_planning",
                output_validator=(
                    lambda proposed: validate_parameter_batch_search_plan(
                        proposed,
                        dependencies,
                        initial_candidate_ids=initial_ids,
                        max_refined_by_assay=max_refined_by_assay,
                    )
                ),
            )
        except UnexpectedModelBehavior:
            logger.warning(
                "Batched parameter refinement planning exhausted "
                "structured-output retries; skipping optional refinement"
            )
            batch_plan = ParameterTuningBatchSearchPlan(
                assayPlans={
                    assay: ParameterSearchPlan(
                        status="complete",
                        rationale=(
                            "Structured refinement planning was unavailable after "
                            "bounded retries; optional refinement was skipped."
                        ),
                        stoppingCriteria=[
                            "Use the completed deterministic initial screen."
                        ],
                        runInfo=AgentRunInfo(
                            agentName="parameter_batch_search_planning_fallback"
                        ),
                    )
                    for assay in assay_names
                },
                runInfo=AgentRunInfo(
                    agentName="parameter_batch_search_planning_fallback"
                ),
            )
        else:
            if not isinstance(
                planning_execution.output, ParameterTuningBatchSearchPlan
            ):
                raise TypeError("Batched parameter planner returned an unexpected type")
            batch_plan = validate_parameter_batch_search_plan(
                planning_execution.output,
                dependencies,
                initial_candidate_ids=initial_ids,
                max_refined_by_assay=max_refined_by_assay,
            ).model_copy(update={"runInfo": planning_execution.runInfo})
            logger.info(
                "Completed batched parameter refinement plan: "
                + ", ".join(
                    f"{assay}={len(plan.candidates)}"
                    for assay, plan in batch_plan.assayPlans.items()
                )
            )
    else:
        logger.info(
            "Skipping batched parameter refinement because it is not authorized"
        )
        batch_plan = ParameterTuningBatchSearchPlan(
            assayPlans={
                assay: ParameterSearchPlan(
                    status="complete",
                    rationale=(
                        "Refinement was not authorized because "
                        "maxRefinedCandidates is zero."
                    ),
                    stoppingCriteria=[
                        "Use the completed initial screen without refinement."
                    ],
                )
                for assay in assay_names
            }
        )
    for assay, plan in batch_plan.assayPlans.items():
        deps = dependencies[assay]
        _register_refined_parameter_candidates(deps, plan.candidates)

    try:
        logger.info(
            f"Requesting batched parameter selection across "
            f"{sum(len(deps.evaluations) for deps in dependencies.values())} "
            "executed candidates"
        )
        selection_execution = run_agent_sync(
            model=model,
            output_type=ParameterTuningReport,
            system_prompt=parameter_batch_selection_system_prompt(),
            user_prompt=parameter_batch_selection_prompt(
                dependencies,
                batch_plan.assayPlans,
                resolved_primary,
                selection_directions,
            ),
            deps_type=ParameterTuningDependencies,
            deps=dependencies[resolved_primary],
            config=run_config,
            name="parameter_tuning_batch",
            output_validator=lambda proposed: validate_parameter_tuning_batch_report(
                proposed,
                dependencies,
                search_plans=batch_plan.assayPlans,
                primary_assay=resolved_primary,
            ),
        )
    except UnexpectedModelBehavior:
        logger.warning(
            "Batched parameter selection exhausted structured-output retries; "
            "using the conservative executor-evidence fallback"
        )
        return fallback_parameter_tuning_batch_report(
            dependencies,
            search_plans=batch_plan.assayPlans,
            primary_assay=resolved_primary,
        )
    if not isinstance(selection_execution.output, ParameterTuningReport):
        raise TypeError("Batched parameter tuning returned an unexpected type")
    report = validate_parameter_tuning_batch_report(
        selection_execution.output,
        dependencies,
        search_plans=batch_plan.assayPlans,
        primary_assay=resolved_primary,
    )
    completed_report = report.model_copy(
        update={"runInfo": selection_execution.runInfo}
    )
    logger.info(
        f"Completed batched parameter tuning: status={completed_report.status}, "
        f"assays={len(completed_report.assayReports)}, "
        f"candidates={completed_report.totalCandidates}"
    )
    return completed_report


def tune_parameters(
    store: Any,
    *,
    model: Any,
    normalized: Any,
    from_assay: str,
    cell_key: str = "I",
    candidates: Sequence[ParameterCandidate] | None = None,
    batch_columns: Sequence[str] = (),
    preservation_columns: Sequence[str] = (),
    experimental_handoff: ExperimentalTuningHandoff | None = None,
    max_candidates: int = 5,
    max_refined_candidates: int = 0,
    min_cluster_cells: int = 20,
    identity_feature_limit: int = 64,
    config: AgentRunConfig | None = None,
) -> ParameterTuningReport:
    """Run the bounded parameter tuning agent against an existing DataStore."""

    logger.info(
        f"Starting parameter tuning for assay {from_assay!r}; "
        f"candidate_limit={max_candidates}, "
        f"refinement_limit={max_refined_candidates}"
    )
    deps, initial_candidate_ids = prepare_parameter_tuning_dependencies(
        store,
        normalized=normalized,
        from_assay=from_assay,
        cell_key=cell_key,
        candidates=candidates,
        batch_columns=batch_columns,
        preservation_columns=preservation_columns,
        experimental_handoff=experimental_handoff,
        max_candidates=max_candidates,
        max_refined_candidates=max_refined_candidates,
        min_cluster_cells=min_cluster_cells,
        identity_feature_limit=identity_feature_limit,
    )
    run_config = (config or AgentRunConfig()).with_limits(
        request_limit=6,
        tool_call_limit=5,
        output_token_limit=32768,
        timeout_seconds=600.0,
    )
    _execute_parameter_candidates(deps, initial_candidate_ids)
    initial_evaluations = [
        deps.evaluations[candidate_id] for candidate_id in initial_candidate_ids
    ]

    if max_refined_candidates == 0:
        logger.info(
            f"Skipping parameter refinement for assay {from_assay!r} because it "
            "is not authorized"
        )
        plan = ParameterSearchPlan(
            status="complete",
            rationale=(
                "Refinement was not authorized because max_refined_candidates is zero."
            ),
            stoppingCriteria=[
                "Use the completed initial screen without a refinement pass."
            ],
        )
    else:
        try:
            logger.info(
                f"Requesting parameter refinement plan for assay {from_assay!r} "
                f"from {len(initial_evaluations)} initial evaluations"
            )
            planning_execution = run_agent_sync(
                model=model,
                output_type=ParameterSearchPlan,
                system_prompt=parameter_search_system_prompt(),
                user_prompt=parameter_search_prompt(
                    from_assay=from_assay,
                    cell_key=deps.cellKey,
                    evaluations=initial_evaluations,
                    batch_columns=deps.batchColumns,
                    preservation_columns=deps.preservationColumns,
                    harmony_authorized=deps.harmonyAuthorized,
                    max_refined_candidates=max_refined_candidates,
                ),
                deps_type=ParameterTuningDependencies,
                deps=deps,
                config=run_config,
                name="parameter_search_planning",
                output_validator=(
                    lambda proposed_plan: validate_parameter_search_plan(
                        proposed_plan,
                        deps,
                        initial_candidate_ids=initial_candidate_ids,
                        max_refined_candidates=max_refined_candidates,
                    )
                ),
            )
        except UnexpectedModelBehavior:
            logger.warning(
                f"Parameter refinement planning for assay {from_assay!r} "
                "exhausted structured-output retries; skipping optional refinement"
            )
            plan = ParameterSearchPlan(
                status="complete",
                rationale=(
                    "Structured refinement planning was unavailable after bounded "
                    "retries; optional refinement was skipped."
                ),
                stoppingCriteria=["Use the completed deterministic initial screen."],
                runInfo=AgentRunInfo(agentName="parameter_search_planning_fallback"),
            )
        else:
            if not isinstance(planning_execution.output, ParameterSearchPlan):
                raise TypeError(
                    "Parameter search planner returned an unexpected output type"
                )
            plan = validate_parameter_search_plan(
                planning_execution.output,
                deps,
                initial_candidate_ids=initial_candidate_ids,
                max_refined_candidates=max_refined_candidates,
            ).model_copy(update={"runInfo": planning_execution.runInfo})
            logger.info(
                f"Completed parameter refinement plan for assay "
                f"{from_assay!r}: status={plan.status}, "
                f"candidates={len(plan.candidates)}"
            )

    _register_refined_parameter_candidates(deps, plan.candidates)

    evaluations = [
        deps.evaluations[candidate_id]
        for candidate_id in deps.executionOrder
        if candidate_id in deps.evaluations
    ]
    try:
        logger.info(
            f"Requesting parameter selection for assay {from_assay!r} across "
            f"{len(evaluations)} executed candidates"
        )
        selection_execution = run_agent_sync(
            model=model,
            output_type=ParameterTuningReport,
            system_prompt=parameter_tuning_system_prompt(min_cluster_cells),
            user_prompt=parameter_tuning_prompt(
                from_assay=from_assay,
                cell_key=deps.cellKey,
                evaluations=evaluations,
                batch_columns=deps.batchColumns,
                preservation_columns=deps.preservationColumns,
                search_plan=plan,
            ),
            deps_type=ParameterTuningDependencies,
            deps=deps,
            config=run_config,
            name="parameter_tuning",
            output_validator=lambda report: validate_parameter_tuning_report(
                report,
                deps,
                search_plan=plan,
            ),
        )
    except UnexpectedModelBehavior:
        logger.warning(
            f"Parameter selection for assay {from_assay!r} exhausted "
            "structured-output retries; using the conservative executor-evidence "
            "fallback"
        )
        return fallback_parameter_tuning_report(
            deps,
            search_plan=plan,
            agent_name="parameter_tuning_fallback",
        )
    if not isinstance(selection_execution.output, ParameterTuningReport):
        raise TypeError("Parameter tuning agent returned an unexpected output type")
    report = validate_parameter_tuning_report(
        selection_execution.output,
        deps,
        search_plan=plan,
    )
    completed_report = report.model_copy(
        update={"runInfo": selection_execution.runInfo}
    )
    logger.info(
        f"Completed parameter tuning for assay {from_assay!r}: "
        f"status={completed_report.status}, "
        f"selected={completed_report.recommendedCandidateId!r}, "
        f"candidates={len(completed_report.evaluations)}"
    )
    return completed_report


__all__ = [
    "ArtifactRecord",
    "build_initial_parameter_candidates",
    "CandidateComparison",
    "execute_parameter_candidate",
    "final_graph_options",
    "final_graph_selection_prompt",
    "final_graph_selection_system_prompt",
    "FinalGraphComparison",
    "FinalGraphNeedsInput",
    "FinalGraphSelection",
    "finalize_parameter_tuning_selection",
    "IntegrationCandidateEvaluation",
    "IntegrationMetrics",
    "normalized_artifact_shape",
    "ParameterCandidate",
    "ParameterCandidateEvaluation",
    "ParameterMetrics",
    "ParameterSearchPlan",
    "ParameterTuningAssayInput",
    "ParameterTuningAgent",
    "ParameterTuningBatchSearchPlan",
    "ParameterTuningDependencies",
    "ParameterTuningNeedsInput",
    "ParameterTuningReport",
    "evaluate_parameter_candidate",
    "get_default_parameter_candidates",
    "parameter_batch_search_prompt",
    "parameter_batch_search_system_prompt",
    "parameter_batch_selection_prompt",
    "parameter_batch_selection_system_prompt",
    "parameter_search_prompt",
    "parameter_search_system_prompt",
    "parameter_tuning_prompt",
    "parameter_tuning_system_prompt",
    "prepare_parameter_tuning_dependencies",
    "promote_parameter_candidate",
    "run_candidate_reduction",
    "select_final_parameter_graph",
    "tune_parameters",
    "tune_parameters_batch",
    "validate_parameter_batch_search_plan",
    "validate_parameter_candidate_rank",
    "validate_final_graph_selection",
    "validate_parameter_search_plan",
    "validate_parameter_tuning_batch_report",
    "validate_parameter_tuning_report",
]
