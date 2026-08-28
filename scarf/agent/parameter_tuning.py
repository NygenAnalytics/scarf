"""Bounded parameter tuning over explicit Scarf analysis candidates."""

import hashlib
import json
from collections.abc import Sequence
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

try:
    from pydantic import Field
except ImportError as exc:
    from .config._deps import AGENT_INSTALL_HINT

    raise ImportError(AGENT_INSTALL_HINT) from exc

try:
    from pydantic_ai import RunContext
except ImportError as exc:
    raise ImportError(AGENT_INSTALL_HINT) from exc


type CandidateStatus = Literal["done", "failed"]
type CandidatePhase = Literal["initial", "refined"]
type ParameterSearchStatus = Literal["complete", "refine"]
type TuningConfidence = Literal["low", "medium", "high"]


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
            metrics=ParameterMetrics.get_example(),
            evidenceIds=["candidate:baseline:clusters"],
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
            runInfo=AgentRunInfo.get_example(),
        )

    def to_biological_handoff(self) -> TuningBiologyHandoff:
        """Return the exact selected clustering branch for interpretation."""
        if self.status != "done" or self.recommendedCandidateId is None:
            raise ValueError(
                "Parameter Tuning must be done before creating a biology handoff"
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
    fromAssay: str = ""
    cellKey: str = "I"
    candidates: dict[str, ParameterCandidate] = Field(default_factory=dict)
    candidatePhases: dict[str, CandidatePhase] = Field(default_factory=dict)
    batchColumns: tuple[str, ...] = ()
    preservationColumns: tuple[str, ...] = ()
    harmonyAuthorized: bool = False
    maxCandidates: int = 5
    minClusterCells: int = 20
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
            candidates={candidate.candidateId: candidate},
            batchColumns=("batch",),
            preservationColumns=("cell_type",),
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
        dedent(
            """
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


def execute_parameter_candidate(
    deps: ParameterTuningDependencies,
    candidate_id: str,
) -> ParameterCandidateEvaluation:
    """Execute one allowlisted candidate without model involvement."""

    with deps.executionLock:
        if candidate_id in deps.evaluations:
            return deps.evaluations[candidate_id]
        if candidate_id not in deps.candidates:
            return ParameterCandidateEvaluation(
                candidateId=candidate_id,
                status="failed",
                error=(
                    f"Unknown candidate id {candidate_id!r}; allowed ids are "
                    f"{sorted(deps.candidates)}"
                ),
            )
        if len(deps.executionOrder) >= deps.maxCandidates:
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
        if candidate.useHarmony and not deps.batchColumns:
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
            pca_ref = store.run_pca(
                deps.normalized,
                from_assay=deps.fromAssay,
                dims=candidate.dimensions,
                feat_scaling=True,
                show_elbow_plot=False,
                update_state=False,
                invalidate_cache=False,
            )
            artifacts["pca"] = ArtifactRecord.from_ref(pca_ref)

            coordinates_ref = pca_ref
            if candidate.useHarmony:
                coordinates_ref = store.run_harmony(
                    list(deps.batchColumns),
                    reduction=pca_ref,
                    from_assay=deps.fromAssay,
                    update_state=False,
                    invalidate_cache=False,
                )
                artifacts["harmony"] = ArtifactRecord.from_ref(coordinates_ref)

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

            neighbors_ref = store.query_neighbors(
                ann_index=ann_ref,
                from_assay=deps.fromAssay,
                coordinates=coordinates_ref,
                k=candidate.neighborsK,
                update_state=False,
                invalidate_cache=False,
            )
            artifacts["neighbors"] = ArtifactRecord.from_ref(neighbors_ref)

            graph_ref = store.build_connectivity_map(
                neighbors=neighbors_ref,
                from_assay=deps.fromAssay,
                local_connectivity=1.0,
                bandwidth=1.5,
                update_state=False,
                invalidate_cache=False,
            )
            artifacts["connectivityMap"] = ArtifactRecord.from_ref(graph_ref)

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

            cluster_group = store.load_artifact(cluster_ref)
            cluster_data = cluster_group["values"]
            cluster_values = np.asarray(cluster_data[:])
            if cluster_values.ndim != 1 or len(cluster_values) == 0:
                raise ValueError(
                    "Cluster artifact must contain one non-empty label vector"
                )
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

            try:
                separability = store.metric_cluster_separability(
                    pca_ref,
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
                            evidence_ids.append(
                                f"candidate:{candidate_id}:{evidence_name}"
                            )
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
                        evidence_ids.append(
                            f"candidate:{candidate_id}:batchMixing:{column}"
                        )
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
                    warnings.append(
                        f"Graph connectivity for {column!r} unavailable: {exc}"
                    )
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
                metrics=metrics,
                evidenceIds=evidence_ids,
                eligibilityReasons=eligibility_reasons,
                warnings=warnings,
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

        state_after = None
        if hasattr(store, "get_assay_state"):
            state_after = store.get_assay_state(deps.fromAssay)
        if state_after != state_before:
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
        paired_modes: dict[tuple[int, float, int], set[bool]] = {}
        for candidate in parent_candidates:
            parameter_key = (
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
    dimensions = [candidate.dimensions for candidate in initial_candidates]
    resolutions = [candidate.leidenResolution for candidate in initial_candidates]
    neighbor_counts = [candidate.neighborsK for candidate in initial_candidates]
    dimension_bounds = (min(dimensions), max(dimensions))
    resolution_bounds = (min(resolutions), max(resolutions))
    neighbor_bounds = (min(neighbor_counts), max(neighbor_counts))
    known_signatures = {
        (
            candidate.dimensions,
            candidate.leidenResolution,
            candidate.neighborsK,
            candidate.useHarmony,
        )
        for candidate in initial_candidates
    }
    proposed_ids: set[str] = set()
    proposed_signatures: set[tuple[int, float, int, bool]] = set()
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
        }
    )


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
            request_limit=3,
            tool_call_limit=1,
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
            config=self.config,
        )


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
    config: AgentRunConfig | None = None,
) -> ParameterTuningReport:
    """Run the bounded parameter tuning agent against an existing DataStore."""

    if max_candidates < 1:
        raise ValueError("max_candidates must be at least one")
    if max_refined_candidates < 0:
        raise ValueError("max_refined_candidates must be non-negative")
    if min_cluster_cells < 1:
        raise ValueError("min_cluster_cells must be at least one")
    resolved_cell_key = cell_key
    resolved_batch_columns = list(batch_columns)
    resolved_preservation_columns = list(preservation_columns)
    if experimental_handoff is not None:
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
        if (
            experimental_handoff.batchAction == "skip"
            and experimental_handoff.batchColumns
        ):
            raise ValueError("A skip handoff must not contain batch columns")
        if experimental_handoff.batchAction == "evaluateHarmony":
            expected_coefficients = set(experimental_handoff.coefficientsOfInterest)
            safe_coefficients = {
                item.coefficient
                for item in experimental_handoff.batchSafety
                if item.status == "safe"
                and item.batchColumns == canonical_batch_columns
            }
            if (
                not expected_coefficients
                or not canonical_batch_columns
                or (safe_coefficients != expected_coefficients)
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
        resolved_cell_key = experimental_handoff.cellKey
        resolved_batch_columns = canonical_batch_columns
        resolved_preservation_columns = list(experimental_handoff.preservationColumns)
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
    run_config = (config or AgentRunConfig()).with_limits(
        request_limit=3,
        tool_call_limit=1,
        output_token_limit=32768,
        timeout_seconds=600.0,
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

    harmony_authorized = bool(resolved_batch_columns) and (
        experimental_handoff is None
        or experimental_handoff.batchAction == "evaluateHarmony"
    )
    total_candidate_limit = len(candidate_values) + max_refined_candidates
    deps = ParameterTuningDependencies(
        store=store,
        normalized=normalized,
        fromAssay=from_assay,
        cellKey=resolved_cell_key,
        candidates=candidate_map,
        candidatePhases={candidate_id: "initial" for candidate_id in candidate_map},
        batchColumns=tuple(resolved_batch_columns),
        preservationColumns=tuple(resolved_preservation_columns),
        harmonyAuthorized=harmony_authorized,
        maxCandidates=total_candidate_limit,
        minClusterCells=min_cluster_cells,
    )

    initial_candidate_ids = list(candidate_map)
    for candidate_id in initial_candidate_ids:
        execute_parameter_candidate(deps, candidate_id)
    initial_evaluations = [
        deps.evaluations[candidate_id] for candidate_id in initial_candidate_ids
    ]

    if max_refined_candidates == 0:
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
        planning_execution = run_agent_sync(
            model=model,
            output_type=ParameterSearchPlan,
            system_prompt=parameter_search_system_prompt(),
            user_prompt=parameter_search_prompt(
                from_assay=from_assay,
                cell_key=resolved_cell_key,
                evaluations=initial_evaluations,
                batch_columns=resolved_batch_columns,
                preservation_columns=resolved_preservation_columns,
                harmony_authorized=harmony_authorized,
                max_refined_candidates=max_refined_candidates,
            ),
            deps_type=ParameterTuningDependencies,
            deps=deps,
            config=run_config,
            name="parameter_search_planning",
            output_validator=lambda proposed_plan: validate_parameter_search_plan(
                proposed_plan,
                deps,
                initial_candidate_ids=initial_candidate_ids,
                max_refined_candidates=max_refined_candidates,
            ),
        )
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

    for candidate in plan.candidates:
        deps.candidates[candidate.candidateId] = candidate
        deps.candidatePhases[candidate.candidateId] = "refined"
        execute_parameter_candidate(deps, candidate.candidateId)

    evaluations = [
        deps.evaluations[candidate_id]
        for candidate_id in deps.executionOrder
        if candidate_id in deps.evaluations
    ]
    selection_execution = run_agent_sync(
        model=model,
        output_type=ParameterTuningReport,
        system_prompt=parameter_tuning_system_prompt(min_cluster_cells),
        user_prompt=parameter_tuning_prompt(
            from_assay=from_assay,
            cell_key=resolved_cell_key,
            evaluations=evaluations,
            batch_columns=resolved_batch_columns,
            preservation_columns=resolved_preservation_columns,
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
    if not isinstance(selection_execution.output, ParameterTuningReport):
        raise TypeError("Parameter tuning agent returned an unexpected output type")
    report = validate_parameter_tuning_report(
        selection_execution.output,
        deps,
        search_plan=plan,
    )
    return report.model_copy(update={"runInfo": selection_execution.runInfo})


__all__ = [
    "ArtifactRecord",
    "build_initial_parameter_candidates",
    "CandidateComparison",
    "execute_parameter_candidate",
    "ParameterCandidate",
    "ParameterCandidateEvaluation",
    "ParameterMetrics",
    "ParameterSearchPlan",
    "ParameterTuningAgent",
    "ParameterTuningDependencies",
    "ParameterTuningNeedsInput",
    "ParameterTuningReport",
    "evaluate_parameter_candidate",
    "get_default_parameter_candidates",
    "parameter_search_prompt",
    "parameter_search_system_prompt",
    "parameter_tuning_prompt",
    "parameter_tuning_system_prompt",
    "tune_parameters",
    "validate_parameter_search_plan",
    "validate_parameter_tuning_report",
]
