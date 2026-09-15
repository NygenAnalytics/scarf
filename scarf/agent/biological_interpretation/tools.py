"""Bounded evidence tools for biological interpretation."""

from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np

from ...metadata.rows import read_metadata_missing_rows, read_metadata_rows
from ...storage.refs import ArtifactRef
from ...storage.selections import read_stored_selection_indices
from ...utils.logging import logger
from ..tools import artifact_reference, core_artifact_reference
from .contracts import (
    _MAX_CONDITIONS,
    _MAX_MARKERS,
    BiologicalInterpretationDependencies,
    ClusterCompositionEvidence,
    ClusterMarkerBatchEvidence,
    ClusterMarkerEvidence,
    ConditionClusterSummary,
    MarkerFeature,
)

try:
    from pydantic_ai import ModelRetry, RunContext
except ImportError as exc:
    from .._deps import AGENT_INSTALL_HINT

    raise ImportError(AGENT_INSTALL_HINT) from exc


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
    if deps.compositionEvidence is not None:
        logger.info("Reused completed cluster composition inspection")
        return deps.compositionEvidence
    logger.info(
        f"Inspecting cluster composition from artifact "
        f"{getattr(deps.cluster, 'artifact_id', '')!r}; "
        f"max_clusters={deps.maxClusters}"
    )
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
        deps.graphAssay is not None
        and cluster_artifact.scope == "assay"
        and cluster_artifact.assay != deps.graphAssay
    ):
        raise ValueError("cluster artifact belongs to a different assay")
    if cluster_artifact.scope == "datastore" and cluster_artifact.assay is not None:
        raise ValueError("datastore-scoped cluster artifacts must not name an assay")
    status = deps.store.inspect_artifact(deps.cluster)
    if not getattr(status, "exists", True):
        raise ValueError("cluster artifact does not exist")
    if not getattr(status, "complete", False):
        raise ValueError("cluster artifact is incomplete")
    inputs = getattr(status, "inputs", None) or {}
    raw_selection = inputs.get("cell_selection")
    if not isinstance(raw_selection, Mapping):
        raise ValueError("cluster artifact has no cell-selection input")
    cell_selection = ArtifactRef.from_dict(dict(raw_selection))
    if (
        cell_selection.scope != "datastore"
        or cell_selection.kind != "cell_selection"
        or cell_selection.assay is not None
    ):
        raise ValueError("cluster artifact has an invalid cell-selection input")
    if (
        deps.cellSelection is not None
        and core_artifact_reference(deps.cellSelection) != cell_selection
    ):
        raise ValueError(
            "cluster artifact cell selection conflicts with the prepared selection"
        )
    cell_indices = read_stored_selection_indices(
        deps.store.zw,
        cell_selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    ).astype(np.int64, copy=False)
    deps.cellSelection = cell_selection
    deps.cellIndices = cell_indices
    cluster_group = deps.store.load_artifact(deps.cluster)
    value_name = "labels" if cluster_artifact.kind == "cluster_cut" else "values"
    if value_name not in cluster_group:
        raise ValueError(
            f"cluster artifact does not contain its {value_name!r} label vector"
        )
    cluster_values = np.asarray(cluster_group[value_name][:])
    if cluster_values.ndim != 1 or len(cluster_values) != len(cell_indices):
        raise ValueError("cluster artifact labels do not align with its cell selection")
    if len(cluster_values) == 0:
        raise ValueError("cluster artifact selects no cells")
    counts = Counter(_string_value(value) for value in cluster_values)
    ordered_clusters = sorted(counts, key=lambda value: (-counts[value], value))
    retained_clusters = ordered_clusters[: deps.maxClusters]
    deps.clusterValues = {
        _string_value(value): value.item() if isinstance(value, np.generic) else value
        for value in cluster_values
        if _string_value(value) in retained_clusters
    }
    evidence_prefix = f"composition:{cluster_artifact.artifactId}"
    count_evidence = f"{evidence_prefix}:counts"
    deps.evidenceIds.add(count_evidence)
    warnings: list[str] = []
    if len(ordered_clusters) > deps.maxClusters:
        warnings.append(f"Only the {deps.maxClusters} largest clusters were returned.")

    condition_summaries: list[ConditionClusterSummary] = []
    if deps.conditionColumn is not None:
        condition_values = read_metadata_rows(
            deps.store.cells,
            deps.conditionColumn,
            cell_indices,
        )
        if len(condition_values) != len(cluster_values):
            raise ValueError("condition and cluster columns are not aligned")
        condition_missing = read_metadata_missing_rows(
            deps.store.cells,
            deps.conditionColumn,
            cell_indices,
        )
        if condition_missing is not None and np.any(condition_missing):
            raise ValueError("condition column contains missing selected values")
        n_conditions = len({_string_value(value) for value in condition_values})
        if n_conditions > _MAX_CONDITIONS:
            warnings.append(
                f"Only the first {_MAX_CONDITIONS} conditions were returned."
            )
        if deps.sampleColumn is not None:
            sample_values = read_metadata_rows(
                deps.store.cells,
                deps.sampleColumn,
                cell_indices,
            )
            if len(sample_values) != len(cluster_values):
                raise ValueError("sample and cluster columns are not aligned")
            sample_missing = read_metadata_missing_rows(
                deps.store.cells,
                deps.sampleColumn,
                cell_indices,
            )
            if sample_missing is not None and np.any(sample_missing):
                raise ValueError("sample column contains missing selected values")
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
        condition_summaries = summaries[: _MAX_CONDITIONS * len(retained_clusters)]
        deps.evidenceIds.update(summary.evidenceId for summary in condition_summaries)
        deps.conditionEvidence.update(
            {summary.evidenceId: summary for summary in condition_summaries}
        )

    deps.toolCalls.append("inspect_cluster_composition")
    evidence = ClusterCompositionEvidence(
        clusterArtifact=cluster_artifact,
        cellSelection=artifact_reference(cell_selection),
        totalCells=len(cluster_values),
        clusterCounts={cluster: counts[cluster] for cluster in retained_clusters},
        sampleColumn=deps.sampleColumn,
        conditionColumn=deps.conditionColumn,
        conditionSummaries=condition_summaries,
        evidenceIds=sorted(deps.evidenceIds),
        warnings=warnings,
    )
    deps.compositionEvidence = evidence
    logger.info(
        f"Completed cluster composition inspection: cells={evidence.totalCells}, "
        f"clusters={len(evidence.clusterCounts)}, "
        f"condition_summaries={len(evidence.conditionSummaries)}, "
        f"warnings={len(evidence.warnings)}"
    )
    return evidence


def _sample_condition_summaries(
    *,
    sample_values: np.ndarray,
    condition_values: np.ndarray,
    cluster_values: np.ndarray,
    retained_clusters: list[str],
    evidence_prefix: str,
) -> list[ConditionClusterSummary]:
    """Aggregate each condition-unit pair without exposing unit identifiers."""
    sample_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    sample_totals: Counter[tuple[str, str]] = Counter()
    for sample, condition, cluster in zip(
        sample_values,
        condition_values,
        cluster_values,
        strict=True,
    ):
        condition_label = _string_value(condition)
        sample_label = _string_value(sample)
        key = (condition_label, sample_label)
        sample_totals[key] += 1
        sample_counts[key][_string_value(cluster)] += 1

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
    logger.debug(f"Inspecting markers for cluster {cluster_id!r}")
    if not deps.clusterValues:
        raise ModelRetry("Call inspect_cluster_composition before inspecting markers.")
    if cluster_id not in deps.clusterValues:
        raise ModelRetry(f"cluster_id must be one of {sorted(deps.clusterValues)}")
    cached = deps.markerEvidence.get(cluster_id)
    if cached is not None:
        logger.debug(f"Reused cached markers for cluster {cluster_id!r}")
        return cached
    if deps.marker is None:
        if not deps.allowMarkerSearch:
            logger.warning(
                f"Markers for cluster {cluster_id!r} are unavailable because no "
                "marker artifact was supplied or authorized"
            )
            return ClusterMarkerEvidence(
                clusterId=cluster_id,
                evidenceId="",
                warnings=[
                    "No exact marker artifact was supplied and marker search was not authorized."
                ],
            )
        if deps.markerFeatures is None:
            logger.warning(
                f"Markers for cluster {cluster_id!r} are unavailable because no "
                "feature selection was supplied"
            )
            return ClusterMarkerEvidence(
                clusterId=cluster_id,
                evidenceId="",
                warnings=["Marker search requires an exact feature selection."],
            )
        logger.info(
            f"Creating one marker artifact for assay {deps.markerAssay!r} "
            f"from cluster artifact {deps.cluster.artifact_id!r}"
        )
        deps.marker = deps.store.run_marker_search(
            deps.cluster,
            features=deps.markerFeatures,
        )
        if not hasattr(deps.marker, "artifact_id"):
            raise RuntimeError("marker search did not return an artifact reference")
    deps.marker = core_artifact_reference(deps.marker)

    marker_artifact = artifact_reference(deps.marker)
    if marker_artifact.kind != "marker_table":
        raise ModelRetry("marker must identify a marker_table artifact")
    if deps.markerAssay is not None and marker_artifact.assay != deps.markerAssay:
        raise ModelRetry("marker artifact belongs to a different assay")
    if hasattr(deps.store, "inspect_artifact"):
        marker_status = deps.store.inspect_artifact(deps.marker)
        if not getattr(marker_status, "exists", True):
            raise ModelRetry("marker artifact does not exist")
        if not getattr(marker_status, "complete", False):
            raise ModelRetry("marker artifact is incomplete")
        marker_inputs = getattr(marker_status, "inputs", None) or {}
        stored_clusters = marker_inputs.get("clusters")
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
        deps.marker,
        group_id=deps.clusterValues[cluster_id],
        min_score=deps.markerMinScore,
        min_frac_exp=deps.markerMinFraction,
    )
    if "score" in frame.columns:
        frame = frame.sort_values("score", ascending=False, na_position="last")
    markers = [
        _marker_feature(row)
        for row in frame.head(min(deps.maxMarkers, _MAX_MARKERS)).to_dict("records")
    ]
    cluster_artifact = artifact_reference(deps.cluster)
    evidence_id = (
        f"markers:{marker_artifact.artifactId}:clusters:"
        f"{cluster_artifact.artifactId}:cluster:{cluster_id}"
    )
    if markers:
        deps.evidenceIds.add(evidence_id)
        deps.markerEvidenceIds[cluster_id] = evidence_id
    evidence = ClusterMarkerEvidence(
        clusterId=cluster_id,
        markers=markers,
        markerArtifact=marker_artifact,
        evidenceId=evidence_id if markers else "",
        warnings=[] if markers else ["No markers passed the requested thresholds."],
    )
    deps.markerEvidence[cluster_id] = evidence
    logger.debug(
        f"Completed marker inspection for cluster {cluster_id!r}: "
        f"markers={len(markers)}"
    )
    return evidence


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
    if ctx.deps.markerBatch is not None:
        if cluster_ids != ctx.deps.markerBatchClusterIds:
            raise ModelRetry(
                "Marker inspection already completed. Use the returned evidence "
                "and do not request a different cluster batch."
            )
        logger.info("Reused completed cluster marker batch")
        return ctx.deps.markerBatch

    logger.info(f"Inspecting markers for {len(cluster_ids)} cluster(s) in one batch")
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
    ctx.deps.toolCalls.append("inspect_cluster_markers_batch")
    evidence = ClusterMarkerBatchEvidence(
        clusters=clusters,
        evidenceIds=evidence_ids,
        warnings=warnings,
    )
    ctx.deps.markerBatch = evidence
    ctx.deps.markerBatchClusterIds = list(cluster_ids)
    logger.info(
        f"Completed marker batch inspection: clusters={len(clusters)}, "
        f"clusters_with_markers={sum(bool(cluster.markers) for cluster in clusters)}, "
        f"evidence_records={len(evidence_ids)}"
    )
    return evidence


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
