"""Grounded biological interpretation of Scarf cluster results."""

import math
from collections import Counter, defaultdict
from collections.abc import Mapping
from textwrap import dedent
from typing import Any, Literal

import numpy as np
from pydantic import Field

from .config import CONFIG, AgentRunConfig
from .config.agent_exec import run_agent_sync
from .tools import artifact_reference, core_artifact_reference
from .types import (
    AgentDataModel,
    AgentRunInfo,
    ArtifactReferenceModel,
    ExperimentalBiologyHandoff,
    StageStatus,
    TuningBiologyHandoff,
)

try:
    from pydantic_ai import ModelRetry, RunContext
except ImportError as exc:
    from .config._deps import AGENT_INSTALL_HINT

    raise ImportError(AGENT_INSTALL_HINT) from exc

__all__ = [
    "BiologicalContext",
    "BiologicalInterpretationAgent",
    "BiologicalInterpretationNeedsInput",
    "BiologicalInterpretationReport",
    "ClusterCompositionEvidence",
    "ClusterInterpretation",
    "ClusterMarkerBatchEvidence",
    "ClusterMarkerEvidence",
    "ConditionClusterSummary",
    "FollowUpRecommendation",
    "MarkerFeature",
    "TreatmentObservation",
    "inspect_cluster_composition",
    "inspect_cluster_markers_batch",
    "inspect_cluster_markers",
    "validate_biological_interpretation_report",
]

type InterpretationConfidence = Literal["low", "medium", "high"]
type TreatmentDirection = Literal["higher", "lower", "equal"]


class BiologicalContext(AgentDataModel):
    """Caller-supplied facts that constrain biological interpretation."""

    organism: str = ""
    tissue: str = ""
    cellTypeReferences: list[str] = Field(default_factory=list)
    experimentalDetails: list[str] = Field(default_factory=list)
    treatmentQuestion: str = ""

    @classmethod
    def get_example(cls) -> "BiologicalContext":
        return cls(
            organism="Homo sapiens",
            tissue="lung",
            cellTypeReferences=["alveolar macrophage", "T cell"],
            experimentalDetails=["drug and vehicle groups"],
            treatmentQuestion="Which populations respond selectively to treatment?",
        )


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

    @classmethod
    def get_example(cls) -> "ConditionClusterSummary":
        return cls(
            condition="treated",
            clusterId="3",
            nSamples=4,
            meanFraction=0.18,
            minFraction=0.12,
            maxFraction=0.25,
            cellCount=180,
            evidenceId="composition:RNA_cluster:condition:treated:cluster:3",
        )


class ClusterCompositionEvidence(AgentDataModel):
    """Bounded deterministic evidence about cluster sizes and conditions."""

    clusterColumn: str = ""
    clusterArtifact: ArtifactReferenceModel | None = None
    cellKey: str = ""
    totalCells: int = 0
    clusterCounts: dict[str, int] = Field(default_factory=dict)
    sampleColumn: str | None = None
    conditionColumn: str | None = None
    conditionSummaries: list[ConditionClusterSummary] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @classmethod
    def get_example(cls) -> "ClusterCompositionEvidence":
        summary = ConditionClusterSummary.get_example()
        reference_summary = ConditionClusterSummary(
            condition="control",
            clusterId=summary.clusterId,
            nSamples=4,
            meanFraction=0.11,
            minFraction=0.08,
            maxFraction=0.15,
            cellCount=110,
            evidenceId="composition:RNA_cluster:condition:control:cluster:3",
        )
        return cls(
            clusterColumn="RNA_cluster",
            clusterArtifact=ArtifactReferenceModel(
                assay="RNA",
                kind="cluster_labels",
                artifactId="b" * 64,
            ),
            cellKey="I",
            totalCells=1000,
            clusterCounts={"0": 520, "1": 300, "3": 180},
            sampleColumn="sample",
            conditionColumn="treatment",
            conditionSummaries=[reference_summary, summary],
            evidenceIds=[
                "composition:RNA_cluster:counts",
                reference_summary.evidenceId,
                summary.evidenceId,
            ],
        )


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

    @classmethod
    def get_example(cls) -> "MarkerFeature":
        return cls(
            featureId="ENSG00000173372",
            featureName="C1QA",
            featureIndex=123,
            score=0.83,
            foldChange=3.4,
            fractionExpressed=0.76,
            fractionExpressedRest=0.18,
            auc=0.91,
            adjustedPvalue=0.001,
        )


class ClusterMarkerEvidence(AgentDataModel):
    """Bounded markers for one exact cluster label."""

    clusterId: str = ""
    markers: list[MarkerFeature] = Field(default_factory=list)
    markerArtifact: ArtifactReferenceModel | None = None
    evidenceId: str = ""
    warnings: list[str] = Field(default_factory=list)

    @classmethod
    def get_example(cls) -> "ClusterMarkerEvidence":
        return cls(
            clusterId="3",
            markers=[MarkerFeature.get_example()],
            markerArtifact=ArtifactReferenceModel(
                assay="RNA",
                kind="marker_table",
                artifactId="a" * 64,
            ),
            evidenceId="markers:RNA_cluster:cluster:3",
        )


class ClusterMarkerBatchEvidence(AgentDataModel):
    """Markers for all model-selected clusters returned by one tool call."""

    clusters: list[ClusterMarkerEvidence] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "ClusterMarkerBatchEvidence":
        return cls()

    @classmethod
    def get_example(cls) -> "ClusterMarkerBatchEvidence":
        cluster = ClusterMarkerEvidence.get_example()
        return cls(clusters=[cluster], evidenceIds=[cluster.evidenceId])


class ClusterInterpretation(AgentDataModel):
    """One evidence-linked cluster interpretation or hypothesis."""

    clusterId: str = ""
    proposedIdentity: str = "unresolved"
    identityIsHypothesis: bool = True
    confidence: InterpretationConfidence = "low"
    rationale: str = ""
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_example(cls) -> "ClusterInterpretation":
        return cls(
            clusterId="3",
            proposedIdentity="alveolar macrophage-like",
            identityIsHypothesis=True,
            confidence="medium",
            rationale="Observed marker pattern is consistent with the proposed identity.",
            evidenceIds=["markers:RNA_cluster:cluster:3"],
        )


class TreatmentObservation(AgentDataModel):
    """Descriptive treatment observation with no unsupported causal claim."""

    clusterId: str = ""
    referenceCondition: str = ""
    comparisonCondition: str = ""
    direction: TreatmentDirection = "equal"
    observation: str = ""
    isDescriptiveOnly: Literal[True] = True
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_example(cls) -> "TreatmentObservation":
        return cls(
            clusterId="3",
            referenceCondition="control",
            comparisonCondition="treated",
            direction="higher",
            observation="Cluster 3 has a higher mean fraction in treated samples.",
            evidenceIds=[
                "composition:RNA_cluster:condition:control:cluster:3",
                "composition:RNA_cluster:condition:treated:cluster:3",
            ],
        )


class FollowUpRecommendation(AgentDataModel):
    """A bounded next analysis tied to an observed uncertainty."""

    question: str = ""
    operation: str = ""
    rationale: str = ""
    requiredInputs: list[str] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_example(cls) -> "FollowUpRecommendation":
        return cls(
            question="Is the abundance difference reproducible across donors?",
            operation="sample-level differential abundance",
            rationale="Current evidence is descriptive and requires independent replicates.",
            requiredInputs=["sample", "condition", "donor"],
            evidenceIds=[
                "composition:RNA_cluster:condition:control:cluster:3",
                "composition:RNA_cluster:condition:treated:cluster:3",
            ],
        )


class BiologicalInterpretationNeedsInput(AgentDataModel):
    question: str = ""
    requiredInputs: list[str] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_example(cls) -> "BiologicalInterpretationNeedsInput":
        return cls(
            question="Provide an exact marker artifact or authorize marker search.",
            requiredInputs=["markerArtifact"],
        )


class BiologicalInterpretationReport(AgentDataModel):
    """Structured, evidence-grounded biological review."""

    status: StageStatus = "needsInput"
    clusterInterpretations: list[ClusterInterpretation] = Field(default_factory=list)
    treatmentObservations: list[TreatmentObservation] = Field(default_factory=list)
    followUps: list[FollowUpRecommendation] = Field(default_factory=list)
    clusterArtifact: ArtifactReferenceModel | None = None
    markerArtifact: ArtifactReferenceModel | None = None
    evidenceIds: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    stopReason: str = ""
    needsInput: BiologicalInterpretationNeedsInput | None = None
    runInfo: AgentRunInfo = Field(default_factory=AgentRunInfo)

    @classmethod
    def get_example(cls) -> "BiologicalInterpretationReport":
        interpretation = ClusterInterpretation.get_example()
        observation = TreatmentObservation.get_example()
        follow_up = FollowUpRecommendation.get_example()
        return cls(
            status="done",
            clusterInterpretations=[interpretation],
            treatmentObservations=[observation],
            followUps=[follow_up],
            clusterArtifact=ClusterCompositionEvidence.get_example().clusterArtifact,
            markerArtifact=ClusterMarkerEvidence.get_example().markerArtifact,
            evidenceIds=sorted(
                {
                    *interpretation.evidenceIds,
                    *observation.evidenceIds,
                    *follow_up.evidenceIds,
                }
            ),
            limitations=[
                "Cell identities remain hypotheses until independently validated."
            ],
            stopReason="The requested clusters were reviewed.",
        )


class BiologicalInterpretationDependencies(AgentDataModel):
    """Runtime state available only to biological interpretation tools."""

    store: Any = Field(default=None, exclude=True)
    clusterColumn: str = ""
    cluster: Any = Field(default=None, exclude=True)
    cellKey: str = "I"
    fromAssay: str | None = None
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
    conditionEvidence: dict[str, ConditionClusterSummary] = Field(
        default_factory=dict,
        exclude=True,
    )
    designHandoff: ExperimentalBiologyHandoff | None = Field(
        default=None,
        exclude=True,
    )

    @classmethod
    def get_example(cls) -> "BiologicalInterpretationDependencies":
        return cls(
            clusterColumn="RNA_cluster",
            cluster=object(),
            fromAssay="RNA",
            sampleColumn="sample",
            conditionColumn="treatment",
        )


_SYSTEM_PROMPT = dedent(
    """
        You are Scarf's Biological Interpretation Agent. Use only the supplied
        tools and caller context. Call inspect_cluster_composition exactly once.
        Then select every cluster you intend to interpret and call
        inspect_cluster_markers_batch exactly once with all selected cluster IDs.
        Tool calls execute Scarf operations, so wait for their results before
        drawing conclusions. Do not split marker inspection across calls.

        Treat cell identities as hypotheses unless the caller supplied a trusted
        label. Do not invent genes, cell types, statistics, artifact identifiers,
        or evidence identifiers. Cite only evidenceIds returned by tools. For each
        cluster interpretation, copy the exact non-empty marker evidenceId returned
        for that cluster into its evidenceIds. Do not interpret a cluster whose
        marker evidenceId is empty. Cluster abundance summaries are descriptive,
        not tests of significance or causal effects. Treatment observations must
        compare two returned sample-level condition summaries for the same cluster.
        Marker p-values describe cluster-versus-rest marker specificity, not
        condition effects. Keep treatment content out of cluster identity
        interpretations. Recommend a named follow-up operation when replication, a
        covariate, or an exact artifact is missing. Do not write exploratory code,
        use a shell, access files, or call arbitrary Scarf methods. Return only
        fields defined by the structured output schema.
    """
).strip()


def _string_value(value: Any) -> str:
    return str(value.item() if isinstance(value, np.generic) else value)


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _check_column(store: Any, column: str, label: str) -> None:
    if column not in set(store.cells.columns):
        raise ValueError(f"{label} {column!r} is not present in cell metadata")


async def inspect_cluster_composition(
    ctx: RunContext[BiologicalInterpretationDependencies],
) -> ClusterCompositionEvidence:
    """Inspect bounded cluster and condition composition without identifiers."""
    deps = ctx.deps
    _check_column(deps.store, deps.clusterColumn, "cluster column")
    _check_column(deps.store, deps.cellKey, "cell key")
    if deps.sampleColumn is not None:
        _check_column(deps.store, deps.sampleColumn, "sample column")
    if deps.conditionColumn is not None:
        _check_column(deps.store, deps.conditionColumn, "condition column")

    if deps.cluster is None:
        raise ValueError("An exact cluster artifact is required")
    deps.cluster = core_artifact_reference(deps.cluster)
    cluster_artifact = artifact_reference(deps.cluster)
    if cluster_artifact.kind not in {"cluster_labels", "cluster_cut"}:
        raise ValueError(
            "cluster must identify a cluster_labels or cluster_cut artifact"
        )
    if (
        deps.fromAssay is not None
        and cluster_artifact.scope == "assay"
        and cluster_artifact.assay != deps.fromAssay
    ):
        raise ValueError("cluster artifact belongs to a different assay")
    if hasattr(deps.store, "inspect_artifact"):
        status = deps.store.inspect_artifact(deps.cluster)
        if not getattr(status, "exists", True):
            raise ValueError("cluster artifact does not exist")
        if not getattr(status, "complete", True):
            raise ValueError("cluster artifact is incomplete")
        inputs = getattr(status, "inputs", None) or {}
        raw_selection = inputs.get("cell_selection")
        if raw_selection is not None and hasattr(deps.store, "zw"):
            from ..graph.state import validate_cell_selection_artifact
            from ..storage.refs import ArtifactRef

            validate_cell_selection_artifact(
                deps.store.zw,
                ArtifactRef.from_dict(raw_selection),
                deps.cellKey,
            )
    cluster_group = deps.store.load_artifact(deps.cluster)
    value_name = "labels" if cluster_artifact.kind == "cluster_cut" else "values"
    if value_name not in cluster_group:
        raise ValueError(
            f"cluster artifact does not contain its {value_name!r} label vector"
        )
    cluster_values = np.asarray(cluster_group[value_name][:])
    active_cells = np.asarray(deps.store.cells.fetch(deps.cellKey, key=deps.cellKey))
    if cluster_values.ndim != 1 or len(cluster_values) != len(active_cells):
        raise ValueError(
            "cluster artifact labels do not align with the active cell selection"
        )
    if len(cluster_values) == 0:
        raise ValueError(f"cell key {deps.cellKey!r} selects no cells")
    counts = Counter(_string_value(value) for value in cluster_values)
    ordered_clusters = sorted(counts, key=lambda value: (-counts[value], value))
    retained_clusters = ordered_clusters[: deps.maxClusters]
    deps.clusterValues = {
        _string_value(value): value.item() if isinstance(value, np.generic) else value
        for value in cluster_values
        if _string_value(value) in retained_clusters
    }
    evidence_prefix = f"composition:{cluster_artifact.artifactId}:{deps.clusterColumn}"
    count_evidence = f"{evidence_prefix}:counts"
    deps.evidenceIds.add(count_evidence)
    warnings: list[str] = []
    if len(ordered_clusters) > deps.maxClusters:
        warnings.append(f"Only the {deps.maxClusters} largest clusters were returned.")

    condition_summaries: list[ConditionClusterSummary] = []
    if deps.conditionColumn is not None:
        condition_values = np.asarray(
            deps.store.cells.fetch(deps.conditionColumn, key=deps.cellKey)
        )
        if len(condition_values) != len(cluster_values):
            raise ValueError("condition and cluster columns are not aligned")
        n_conditions = len({_string_value(value) for value in condition_values})
        if n_conditions > CONFIG._MAX_CONDITIONS:
            warnings.append(
                f"Only the first {CONFIG._MAX_CONDITIONS} conditions were returned."
            )
        if deps.sampleColumn is not None:
            sample_values = np.asarray(
                deps.store.cells.fetch(deps.sampleColumn, key=deps.cellKey)
            )
            if len(sample_values) != len(cluster_values):
                raise ValueError("sample and cluster columns are not aligned")
            summaries = _sample_condition_summaries(
                sample_values=sample_values,
                condition_values=condition_values,
                cluster_values=cluster_values,
                retained_clusters=retained_clusters,
                evidence_prefix=evidence_prefix,
            )
        else:
            summaries = _cell_condition_summaries(
                condition_values=condition_values,
                cluster_values=cluster_values,
                retained_clusters=retained_clusters,
                evidence_prefix=evidence_prefix,
            )
            warnings.append(
                "No sample column was supplied; condition fractions are cell-level summaries."
            )
        condition_summaries = summaries[
            : CONFIG._MAX_CONDITIONS * len(retained_clusters)
        ]
        deps.evidenceIds.update(summary.evidenceId for summary in condition_summaries)
        deps.conditionEvidence.update(
            {summary.evidenceId: summary for summary in condition_summaries}
        )

    return ClusterCompositionEvidence(
        clusterColumn=deps.clusterColumn,
        clusterArtifact=cluster_artifact,
        cellKey=deps.cellKey,
        totalCells=len(cluster_values),
        clusterCounts={cluster: counts[cluster] for cluster in retained_clusters},
        sampleColumn=deps.sampleColumn,
        conditionColumn=deps.conditionColumn,
        conditionSummaries=condition_summaries,
        evidenceIds=sorted(deps.evidenceIds),
        warnings=warnings,
    )


def _sample_condition_summaries(
    *,
    sample_values: np.ndarray,
    condition_values: np.ndarray,
    cluster_values: np.ndarray,
    retained_clusters: list[str],
    evidence_prefix: str,
) -> list[ConditionClusterSummary]:
    sample_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    sample_totals: Counter[tuple[str, str]] = Counter()
    sample_conditions: dict[str, set[str]] = defaultdict(set)
    for sample, condition, cluster in zip(
        sample_values,
        condition_values,
        cluster_values,
        strict=True,
    ):
        condition_label = _string_value(condition)
        sample_label = _string_value(sample)
        sample_conditions[sample_label].add(condition_label)
        key = (condition_label, sample_label)
        sample_totals[key] += 1
        sample_counts[key][_string_value(cluster)] += 1
    conflicting_samples = sorted(
        sample
        for sample, conditions in sample_conditions.items()
        if len(conditions) > 1
    )
    if conflicting_samples:
        raise ValueError(
            "each sample must map to exactly one condition; "
            f"{len(conflicting_samples)} samples map to multiple conditions"
        )

    fractions: dict[tuple[str, str], list[float]] = defaultdict(list)
    cell_counts: Counter[tuple[str, str]] = Counter()
    for key, total in sample_totals.items():
        condition, _sample = key
        for cluster in retained_clusters:
            count = sample_counts[key][cluster]
            fractions[(condition, cluster)].append(count / total)
            cell_counts[(condition, cluster)] += count

    output: list[ConditionClusterSummary] = []
    for condition, cluster in sorted(fractions):
        values = fractions[(condition, cluster)]
        evidence_id = f"{evidence_prefix}:condition:{condition}:cluster:{cluster}"
        output.append(
            ConditionClusterSummary(
                condition=condition,
                clusterId=cluster,
                nSamples=len(values),
                meanFraction=float(np.mean(values)),
                minFraction=float(np.min(values)),
                maxFraction=float(np.max(values)),
                cellCount=cell_counts[(condition, cluster)],
                evidenceId=evidence_id,
            )
        )
    return output


def _cell_condition_summaries(
    *,
    condition_values: np.ndarray,
    cluster_values: np.ndarray,
    retained_clusters: list[str],
    evidence_prefix: str,
) -> list[ConditionClusterSummary]:
    totals: Counter[str] = Counter(_string_value(value) for value in condition_values)
    counts: Counter[tuple[str, str]] = Counter()
    for condition, cluster in zip(condition_values, cluster_values, strict=True):
        counts[(_string_value(condition), _string_value(cluster))] += 1
    output: list[ConditionClusterSummary] = []
    for condition in sorted(totals):
        for cluster in retained_clusters:
            count = counts[(condition, cluster)]
            fraction = count / totals[condition]
            evidence_id = f"{evidence_prefix}:condition:{condition}:cluster:{cluster}"
            output.append(
                ConditionClusterSummary(
                    condition=condition,
                    clusterId=cluster,
                    meanFraction=fraction,
                    minFraction=fraction,
                    maxFraction=fraction,
                    cellCount=count,
                    evidenceId=evidence_id,
                )
            )
    return output


async def inspect_cluster_markers(
    ctx: RunContext[BiologicalInterpretationDependencies],
    cluster_id: str,
) -> ClusterMarkerEvidence:
    """Load markers for one observed cluster, optionally creating one artifact."""
    deps = ctx.deps
    if not deps.clusterValues:
        raise ModelRetry("Call inspect_cluster_composition before inspecting markers.")
    if cluster_id not in deps.clusterValues:
        raise ModelRetry(f"cluster_id must be one of {sorted(deps.clusterValues)}")
    if deps.marker is None:
        if not deps.allowMarkerSearch:
            return ClusterMarkerEvidence(
                clusterId=cluster_id,
                evidenceId="",
                warnings=[
                    "No exact marker artifact was supplied and marker search was not authorized."
                ],
            )
        if deps.markerFeatures is None:
            return ClusterMarkerEvidence(
                clusterId=cluster_id,
                evidenceId="",
                warnings=["Marker search requires an exact feature selection."],
            )
        deps.marker = deps.store.run_marker_search(
            from_assay=deps.fromAssay,
            group_key=deps.clusterColumn,
            cell_key=deps.cellKey,
            features=deps.markerFeatures,
            skip_save=False,
        )
        if not hasattr(deps.marker, "artifact_id"):
            raise RuntimeError("marker search did not return an artifact reference")
    deps.marker = core_artifact_reference(deps.marker)

    marker_artifact = artifact_reference(deps.marker)
    if marker_artifact.kind != "marker_table":
        raise ModelRetry("marker must identify a marker_table artifact")
    if deps.fromAssay is not None and marker_artifact.assay != deps.fromAssay:
        raise ModelRetry("marker artifact belongs to a different assay")
    if hasattr(deps.store, "inspect_artifact"):
        marker_status = deps.store.inspect_artifact(deps.marker)
        if not getattr(marker_status, "exists", True):
            raise ModelRetry("marker artifact does not exist")
        if not getattr(marker_status, "complete", True):
            raise ModelRetry("marker artifact is incomplete")
        marker_inputs = getattr(marker_status, "inputs", None) or {}
        stored_clusters = marker_inputs.get("clusters")
        if isinstance(stored_clusters, Mapping) and isinstance(
            stored_clusters.get("artifact"),
            Mapping,
        ):
            stored_clusters = stored_clusters["artifact"]
        expected_cluster = artifact_reference(deps.cluster)
        if (
            not isinstance(stored_clusters, Mapping)
            or stored_clusters.get("artifact_id") != expected_cluster.artifactId
            or stored_clusters.get("kind") != expected_cluster.kind
            or stored_clusters.get("scope") != expected_cluster.scope
            or stored_clusters.get("assay") != expected_cluster.assay
        ):
            raise ModelRetry(
                "marker artifact is not linked to the exact cluster artifact"
            )

    frame = deps.store.get_markers(
        from_assay=deps.fromAssay,
        cell_key=deps.cellKey,
        group_key=deps.clusterColumn,
        group_id=deps.clusterValues[cluster_id],
        min_score=deps.markerMinScore,
        min_frac_exp=deps.markerMinFraction,
        marker=deps.marker,
    )
    if "score" in frame.columns:
        frame = frame.sort_values("score", ascending=False, na_position="last")
    markers = [
        _marker_feature(row)
        for row in frame.head(min(deps.maxMarkers, CONFIG._MAX_MARKERS)).to_dict(
            "records"
        )
    ]
    cluster_artifact = artifact_reference(deps.cluster)
    evidence_id = (
        f"markers:{marker_artifact.artifactId}:clusters:"
        f"{cluster_artifact.artifactId}:cluster:{cluster_id}"
    )
    if markers:
        deps.evidenceIds.add(evidence_id)
        deps.markerEvidenceIds[cluster_id] = evidence_id
    return ClusterMarkerEvidence(
        clusterId=cluster_id,
        markers=markers,
        markerArtifact=marker_artifact,
        evidenceId=evidence_id if markers else "",
        warnings=[] if markers else ["No markers passed the requested thresholds."],
    )


async def inspect_cluster_markers_batch(
    ctx: RunContext[BiologicalInterpretationDependencies],
    cluster_ids: list[str],
) -> ClusterMarkerBatchEvidence:
    """Inspect every selected cluster in one bounded model tool call."""
    if not ctx.deps.clusterValues:
        raise ModelRetry("Call inspect_cluster_composition before inspecting markers.")
    if not cluster_ids:
        raise ModelRetry("cluster_ids must contain at least one observed cluster")
    if len(cluster_ids) > ctx.deps.maxClusters:
        raise ModelRetry(
            f"cluster_ids may contain at most {ctx.deps.maxClusters} values"
        )
    if len(set(cluster_ids)) != len(cluster_ids):
        raise ModelRetry("cluster_ids must not contain duplicates")

    clusters = [
        await inspect_cluster_markers(ctx, cluster_id=cluster_id)
        for cluster_id in cluster_ids
    ]
    evidence_ids = [cluster.evidenceId for cluster in clusters if cluster.evidenceId]
    warnings = [
        f"Cluster {cluster.clusterId}: {warning}"
        for cluster in clusters
        for warning in cluster.warnings
    ]
    return ClusterMarkerBatchEvidence(
        clusters=clusters,
        evidenceIds=evidence_ids,
        warnings=warnings,
    )


def _marker_feature(row: dict[str, Any]) -> MarkerFeature:
    raw_index = _finite_float(row.get("feature_index"))
    return MarkerFeature(
        featureId=str(row.get("feature_id", "")),
        featureName=str(row.get("feature_name", "")),
        featureIndex=int(raw_index) if raw_index is not None else None,
        score=_finite_float(row.get("score")),
        foldChange=_finite_float(row.get("fold_change")),
        fractionExpressed=_finite_float(row.get("frac_exp")),
        fractionExpressedRest=_finite_float(row.get("frac_exp_rest")),
        mean=_finite_float(row.get("mean")),
        meanRest=_finite_float(row.get("mean_rest")),
        auc=_finite_float(row.get("auc")),
        adjustedPvalue=_finite_float(row.get("p_value_adjusted")),
    )


def validate_biological_interpretation_report(
    report: BiologicalInterpretationReport,
    deps: BiologicalInterpretationDependencies,
) -> BiologicalInterpretationReport:
    """Reject invented evidence, clusters, or completed marker-free reviews."""
    if not deps.clusterValues:
        raise ModelRetry("Call inspect_cluster_composition before returning a report.")
    expected_cluster_artifact = artifact_reference(deps.cluster)
    if (
        report.clusterArtifact is not None
        and report.clusterArtifact != expected_cluster_artifact
    ):
        raise ModelRetry("Report clusterArtifact does not match the inspected artifact")
    if deps.marker is not None:
        expected_marker_artifact = artifact_reference(deps.marker)
        if (
            report.markerArtifact is not None
            and report.markerArtifact != expected_marker_artifact
        ):
            raise ModelRetry(
                "Report markerArtifact does not match the inspected artifact"
            )

    cited = set(report.evidenceIds)
    for interpretation in report.clusterInterpretations:
        cited.update(interpretation.evidenceIds)
    for observation in report.treatmentObservations:
        cited.update(observation.evidenceIds)
    for follow_up in report.followUps:
        cited.update(follow_up.evidenceIds)
    if report.needsInput is not None:
        cited.update(report.needsInput.evidenceIds)
    unknown = cited.difference(deps.evidenceIds)
    if unknown:
        raise ModelRetry(f"Unknown evidenceIds: {sorted(unknown)}")
    interpreted_clusters = {item.clusterId for item in report.clusterInterpretations}
    observed_clusters = {item.clusterId for item in report.treatmentObservations}
    unknown_clusters = (interpreted_clusters | observed_clusters).difference(
        deps.clusterValues
    )
    if unknown_clusters:
        raise ModelRetry(f"Unknown cluster ids: {sorted(unknown_clusters)}")
    canonical_interpretations: list[ClusterInterpretation] = []
    omitted_interpretation_clusters: list[str] = []
    for interpretation in report.clusterInterpretations:
        marker_id = deps.markerEvidenceIds.get(interpretation.clusterId)
        if marker_id is None:
            omitted_interpretation_clusters.append(interpretation.clusterId)
            continue
        non_marker_evidence = sorted(set(interpretation.evidenceIds) - {marker_id})
        if non_marker_evidence:
            raise ModelRetry(
                "Cluster identity interpretations may cite only their exact marker "
                f"evidence: {non_marker_evidence}"
            )
        canonical_interpretations.append(
            interpretation.model_copy(update={"evidenceIds": [marker_id]})
        )

    if report.treatmentObservations and deps.conditionColumn is None:
        raise ModelRetry("Treatment observations require a condition column.")
    if report.treatmentObservations and deps.sampleColumn is None:
        raise ModelRetry(
            "Treatment observations require sample-level composition summaries."
        )
    canonical_observations: list[TreatmentObservation] = []
    for observation in report.treatmentObservations:
        if not observation.isDescriptiveOnly:
            raise ModelRetry("Treatment observations must remain descriptive.")
        if len(observation.evidenceIds) != 2 or len(set(observation.evidenceIds)) != 2:
            raise ModelRetry(
                "Every treatment observation must cite exactly two distinct "
                "condition summaries."
            )
        if any(
            evidence_id not in deps.conditionEvidence
            for evidence_id in observation.evidenceIds
        ):
            raise ModelRetry(
                "Treatment observations may cite only condition composition evidence."
            )
        summaries = [
            deps.conditionEvidence[evidence_id]
            for evidence_id in observation.evidenceIds
        ]
        if any(summary.clusterId != observation.clusterId for summary in summaries):
            raise ModelRetry(
                "Every treatment observation must cite condition summaries for "
                "its exact cluster."
            )
        if (
            not observation.referenceCondition
            or not observation.comparisonCondition
            or observation.referenceCondition == observation.comparisonCondition
        ):
            raise ModelRetry(
                "Treatment observations require two distinct named conditions."
            )
        summaries_by_condition = {summary.condition: summary for summary in summaries}
        expected_conditions = {
            observation.referenceCondition,
            observation.comparisonCondition,
        }
        if set(summaries_by_condition) != expected_conditions:
            raise ModelRetry(
                "Treatment observation conditions must match the two cited "
                "condition summaries."
            )
        if any(summary.nSamples < 2 for summary in summaries):
            raise ModelRetry(
                "Sample-level treatment observations require at least two samples "
                "in every cited condition."
            )
        reference = summaries_by_condition[observation.referenceCondition]
        comparison = summaries_by_condition[observation.comparisonCondition]
        if math.isclose(
            comparison.meanFraction,
            reference.meanFraction,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            expected_direction: TreatmentDirection = "equal"
        elif comparison.meanFraction > reference.meanFraction:
            expected_direction = "higher"
        else:
            expected_direction = "lower"
        if observation.direction != expected_direction:
            raise ModelRetry(
                "Treatment observation direction does not match the cited mean "
                "sample-level fractions."
            )
        if expected_direction == "equal":
            canonical_text = (
                f"Cluster {observation.clusterId} has equal mean sample-level "
                f"fractions in {comparison.condition} and {reference.condition} "
                f"({comparison.meanFraction:.6g}); this is descriptive only."
            )
        else:
            canonical_text = (
                f"Cluster {observation.clusterId} has a {expected_direction} mean "
                f"sample-level fraction in {comparison.condition} "
                f"({comparison.meanFraction:.6g}) than in {reference.condition} "
                f"({reference.meanFraction:.6g}); this is descriptive only."
            )
        canonical_observations.append(
            observation.model_copy(update={"observation": canonical_text})
        )
    if report.status == "done" and not canonical_interpretations:
        raise ModelRetry(
            "A done report must contain at least one cluster interpretation with "
            "non-empty marker evidence."
        )
    limitations = list(report.limitations)
    if omitted_interpretation_clusters:
        omitted_clusters = ", ".join(sorted(set(omitted_interpretation_clusters)))
        marker_limitation = (
            "Cluster identity interpretations without non-empty marker evidence "
            f"were omitted for clusters: {omitted_clusters}."
        )
        if marker_limitation not in limitations:
            limitations.append(marker_limitation)
    if canonical_observations:
        descriptive_limitation = (
            "Condition-level cluster fractions are descriptive summaries, not "
            "tests of significance or causal treatment effects."
        )
        if descriptive_limitation not in limitations:
            limitations.append(descriptive_limitation)
        handoff = deps.designHandoff
        if handoff is not None and (
            handoff.coefficientScope != "betweenUnit"
            or handoff.estimability.get("status") != "ok"
            or handoff.estimability.get("coefficientEstimable") is not True
        ):
            design_limitation = (
                "Experimental design evidence does not establish an estimable "
                "between-unit condition contrast."
            )
            if design_limitation not in limitations:
                limitations.append(design_limitation)
    return report.model_copy(
        update={
            "clusterInterpretations": canonical_interpretations,
            "treatmentObservations": canonical_observations,
            "evidenceIds": sorted(
                {
                    *report.evidenceIds,
                    *(
                        evidence_id
                        for interpretation in canonical_interpretations
                        for evidence_id in interpretation.evidenceIds
                    ),
                }
            ),
            "limitations": limitations,
            "clusterArtifact": expected_cluster_artifact,
            "markerArtifact": (
                artifact_reference(deps.marker) if deps.marker is not None else None
            ),
        }
    )


class BiologicalInterpretationAgent:
    """Run a bounded biological review through explicit Scarf tools."""

    def __init__(
        self,
        model: Any,
        *,
        config: AgentRunConfig | None = None,
    ) -> None:
        self.model = model
        self.config = (config or AgentRunConfig()).with_limits(
            request_limit=5,
            tool_call_limit=2,
            output_token_limit=32768,
            timeout_seconds=600.0,
        )

    def run(
        self,
        store: Any,
        *,
        cluster_column: str | None = None,
        cluster: Any = None,
        biological_context: BiologicalContext | None = None,
        cell_key: str = "I",
        from_assay: str | None = None,
        sample_column: str | None = None,
        condition_column: str | None = None,
        tuning_handoff: TuningBiologyHandoff | None = None,
        experimental_handoff: ExperimentalBiologyHandoff | None = None,
        marker: Any = None,
        marker_features: Any = None,
        allow_marker_search: bool = False,
        max_clusters: int = 12,
        max_markers: int = 10,
        marker_min_score: float = 0.25,
        marker_min_fraction: float = 0.2,
    ) -> BiologicalInterpretationReport:
        """Interpret cluster results while exposing only bounded tools to the model."""
        if tuning_handoff is not None:
            if tuning_handoff.clusterArtifact is None:
                raise ValueError("tuning_handoff lacks a cluster artifact")
            if (
                cluster_column is not None
                and cluster_column != tuning_handoff.clusterColumn
            ):
                raise ValueError("cluster_column conflicts with tuning_handoff")
            if cluster is not None and (
                artifact_reference(cluster) != tuning_handoff.clusterArtifact
            ):
                raise ValueError("cluster conflicts with tuning_handoff")
            if from_assay is not None and from_assay != tuning_handoff.fromAssay:
                raise ValueError("from_assay conflicts with tuning_handoff")
            if cell_key != "I" and cell_key != tuning_handoff.cellKey:
                raise ValueError("cell_key conflicts with tuning_handoff")
            cluster_column = tuning_handoff.clusterColumn
            cluster = tuning_handoff.clusterArtifact
            from_assay = tuning_handoff.fromAssay
            cell_key = tuning_handoff.cellKey
        if experimental_handoff is not None:
            if tuning_handoff is not None and cell_key != experimental_handoff.cellKey:
                raise ValueError(
                    "Experimental and tuning handoffs use different cell keys"
                )
            if (
                tuning_handoff is None
                and cell_key != "I"
                and cell_key != experimental_handoff.cellKey
            ):
                raise ValueError("cell_key conflicts with experimental_handoff")
            if (
                condition_column is not None
                and condition_column != experimental_handoff.conditionColumn
            ):
                raise ValueError("condition_column conflicts with experimental_handoff")
            if (
                sample_column is not None
                and sample_column != experimental_handoff.observationUnit
            ):
                raise ValueError("sample_column conflicts with experimental_handoff")
            condition_column = experimental_handoff.conditionColumn
            sample_column = experimental_handoff.observationUnit
            cell_key = experimental_handoff.cellKey
        if not cluster_column:
            raise ValueError("cluster_column must be non-empty")
        if cluster is None:
            raise ValueError("cluster must identify an exact cluster artifact")
        if not 1 <= max_clusters <= CONFIG._MAX_CLUSTERS:
            raise ValueError(
                f"max_clusters must be between 1 and {CONFIG._MAX_CLUSTERS}"
            )
        if not 1 <= max_markers <= CONFIG._MAX_MARKERS:
            raise ValueError(f"max_markers must be between 1 and {CONFIG._MAX_MARKERS}")
        if not 0 < marker_min_score <= 1:
            raise ValueError("marker_min_score must be greater than 0 and at most 1")
        if not 0 <= marker_min_fraction <= 1:
            raise ValueError("marker_min_fraction must be between 0 and 1")
        if allow_marker_search and marker is None and marker_features is None:
            raise ValueError(
                "marker_features is required when marker search is authorized"
            )
        cluster = core_artifact_reference(cluster)
        marker = core_artifact_reference(marker)
        cluster_artifact = artifact_reference(cluster)
        if cluster_artifact.kind not in {"cluster_labels", "cluster_cut"}:
            raise ValueError(
                "cluster must identify a cluster_labels or cluster_cut artifact"
            )
        if (
            from_assay is not None
            and cluster_artifact.scope == "assay"
            and cluster_artifact.assay != from_assay
        ):
            raise ValueError("cluster belongs to a different assay")
        resolved_assay = from_assay or cluster_artifact.assay
        context = biological_context or BiologicalContext()
        deps = BiologicalInterpretationDependencies(
            store=store,
            clusterColumn=cluster_column,
            cluster=cluster,
            cellKey=cell_key,
            fromAssay=resolved_assay,
            sampleColumn=sample_column,
            conditionColumn=condition_column,
            designHandoff=experimental_handoff,
            marker=marker,
            markerFeatures=marker_features,
            allowMarkerSearch=allow_marker_search,
            maxClusters=max_clusters,
            maxMarkers=max_markers,
            markerMinScore=marker_min_score,
            markerMinFraction=marker_min_fraction,
        )
        marker_state = "provided" if marker is not None else "not provided"
        user_prompt = (
            dedent(
                """
                Review the cluster results named {cluster_column} for cell selection
                {cell_key}. The exact cluster artifact is {cluster_artifact}. The exact
                marker artifact is {marker_state}; creating a marker artifact is
                authorized={allow_marker_search}. Review no more
                than {max_clusters} clusters and return no more than {max_markers}
                markers per tool call.

                Caller biological context:
                {biological_context}

                Experimental design context:
                {experimental_context}

                Call inspect_cluster_composition once. Then send every cluster you
                intend to interpret in one inspect_cluster_markers_batch call. If
                markers cannot be inspected, return needsInput and state the exact
                missing input.
                """
            )
            .strip()
            .format(
                cluster_column=cluster_column,
                cell_key=cell_key,
                cluster_artifact=cluster_artifact.model_dump_json(),
                marker_state=marker_state,
                allow_marker_search=allow_marker_search,
                max_clusters=max_clusters,
                max_markers=max_markers,
                biological_context=context.model_dump_json(),
                experimental_context=(
                    experimental_handoff.model_dump_json()
                    if experimental_handoff is not None
                    else "not provided"
                ),
            )
        )
        execution = run_agent_sync(
            model=self.model,
            output_type=BiologicalInterpretationReport,
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            tools=(inspect_cluster_composition, inspect_cluster_markers_batch),
            deps_type=BiologicalInterpretationDependencies,
            deps=deps,
            config=self.config,
            name="biological_interpretation",
            output_validator=lambda report: validate_biological_interpretation_report(
                report,
                deps,
            ),
        )
        report = validate_biological_interpretation_report(execution.output, deps)
        report.runInfo = execution.runInfo
        return report
