"""Validation and dependency preparation for biological interpretation."""

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from ...storage.refs import ArtifactRef
from ...storage.selections import read_stored_selection_indices
from ...utils.logging import logger
from ..tools import artifact_reference, core_artifact_reference
from ..types import AgentRunInfo, ExperimentalBiologyHandoff, TuningBiologyHandoff
from .contracts import (
    _MAX_CLUSTERS,
    _MAX_MARKERS,
    BiologicalInterpretationDependencies,
    BiologicalInterpretationReport,
    ClusterInterpretation,
    TreatmentDirection,
    TreatmentObservation,
)

try:
    from pydantic_ai import (
        ModelRetry,
        RunContext,
        UnexpectedModelBehavior,
        UsageLimitExceeded,
    )
    from pydantic_ai.tools import ToolDefinition
except ImportError as exc:
    from .._deps import AGENT_INSTALL_HINT

    raise ImportError(AGENT_INSTALL_HINT) from exc


def _prepare_biological_interpretation_tool(
    ctx: RunContext[BiologicalInterpretationDependencies],
    tool_definition: ToolDefinition,
) -> ToolDefinition | None:
    """Expose composition and marker batches once and in the required order."""
    completed_calls = set(ctx.deps.toolCalls)
    if tool_definition.name == "inspect_cluster_composition":
        return None if tool_definition.name in completed_calls else tool_definition
    if tool_definition.name == "inspect_cluster_markers_batch":
        if (
            "inspect_cluster_composition" not in completed_calls
            or tool_definition.name in completed_calls
        ):
            return None
        return tool_definition
    return tool_definition


def _canonicalize_cluster_interpretations(
    report: BiologicalInterpretationReport,
    deps: BiologicalInterpretationDependencies,
) -> tuple[list[ClusterInterpretation], list[str]]:
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
            interpretation.model_copy(
                update={
                    "evidenceIds": [marker_id],
                    "identityIsHypothesis": True,
                    **(
                        {
                            "confidence": "low",
                        }
                        if deps.markerAssayType == "ATAC"
                        else {}
                    ),
                }
            )
        )
    return canonical_interpretations, omitted_interpretation_clusters


def _canonicalize_treatment_observations(
    report: BiologicalInterpretationReport,
    deps: BiologicalInterpretationDependencies,
) -> list[TreatmentObservation]:
    if report.treatmentObservations and deps.conditionColumn is None:
        raise ModelRetry("Treatment observations require a condition column.")
    if report.treatmentObservations and deps.sampleColumn is None:
        raise ModelRetry(
            "Treatment observations require independent-unit composition summaries."
        )
    if report.treatmentObservations:
        handoff = deps.designHandoff
        if (
            handoff is None
            or not handoff.conditionColumn
            or handoff.conditionColumn != deps.conditionColumn
            or not handoff.independentUnit
            or handoff.independentUnit != deps.sampleColumn
            or handoff.coefficientScope != "betweenUnit"
            or handoff.estimability.get("status") != "ok"
            or handoff.estimability.get("coefficientEstimable") is not True
        ):
            raise ModelRetry(
                "Treatment observations require an explicit condition, aggregation "
                "at the independent unit, a between-unit coefficient, and an "
                "estimable experimental contrast."
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
                "independent-unit fractions."
            )
        if expected_direction == "equal":
            canonical_text = (
                f"Cluster {observation.clusterId} has equal mean independent-unit "
                f"fractions in {comparison.condition} and {reference.condition} "
                f"({comparison.meanFraction:.6g}); this is descriptive only."
            )
        else:
            canonical_text = (
                f"Cluster {observation.clusterId} has a {expected_direction} mean "
                f"independent-unit fraction in {comparison.condition} "
                f"({comparison.meanFraction:.6g}) than in {reference.condition} "
                f"({reference.meanFraction:.6g}); this is descriptive only."
            )
        canonical_observations.append(
            observation.model_copy(update={"observation": canonical_text})
        )
    return canonical_observations


def validate_biological_interpretation_report(
    report: BiologicalInterpretationReport,
    deps: BiologicalInterpretationDependencies,
) -> BiologicalInterpretationReport:
    """Reject invented evidence, clusters, or completed marker-free reviews."""
    if not deps.clusterValues:
        raise ModelRetry("Call inspect_cluster_composition before returning a report.")
    if report.status == "failed":
        raise ModelRetry(
            "Do not return failed for biological uncertainty; return needsInput "
            "with one concrete question instead."
        )
    if report.status == "needsInput" and (
        report.needsInput is None or not report.needsInput.question.strip()
    ):
        raise ModelRetry("A needsInput report requires one concrete input question.")
    if report.status != "needsInput" and report.needsInput is not None:
        raise ModelRetry("Only a needsInput report may include an input question.")
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
    canonical_interpretations, omitted_interpretation_clusters = (
        _canonicalize_cluster_interpretations(report, deps)
    )
    canonical_observations = _canonicalize_treatment_observations(report, deps)
    if report.status == "done" and not canonical_interpretations:
        raise ModelRetry(
            "A done report must contain at least one cluster interpretation with "
            "non-empty marker evidence."
        )
    limitations = list(report.limitations)
    if deps.markerAssayType == "ATAC":
        atac_limitation = (
            "ATAC peak markers are descriptive, so all cell identities remain "
            "low-confidence hypotheses."
        )
        if atac_limitation not in limitations:
            limitations.append(atac_limitation)
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
            "Independent-unit cluster fractions are descriptive summaries, not "
            "tests of significance or causal treatment effects."
        )
        if descriptive_limitation not in limitations:
            limitations.append(descriptive_limitation)
    validated = report.model_copy(
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
            "graphAssay": deps.graphAssay,
            "markerAssay": deps.markerAssay,
        }
    )
    logger.debug(
        f"Validated biological interpretation report: status={validated.status}, "
        f"cluster_interpretations={len(validated.clusterInterpretations)}, "
        f"treatment_observations={len(validated.treatmentObservations)}, "
        f"omitted_interpretations={len(omitted_interpretation_clusters)}"
    )
    return validated


def fallback_biological_interpretation_report(
    deps: BiologicalInterpretationDependencies,
    *,
    error: UnexpectedModelBehavior | UsageLimitExceeded,
    model_name: str,
) -> BiologicalInterpretationReport:
    """Keep measured evidence without claiming completed model interpretation."""
    if not deps.clusterValues:
        raise error
    from ..config.agent_exec import describe_agent_error

    error_detail = describe_agent_error(error)
    report = BiologicalInterpretationReport(
        status="failed",
        clusterArtifact=artifact_reference(deps.cluster)
        if deps.cluster is not None
        else None,
        markerArtifact=artifact_reference(deps.marker)
        if deps.marker is not None
        else None,
        graphAssay=deps.graphAssay,
        markerAssay=deps.markerAssay,
        evidenceIds=sorted({*deps.evidenceIds, *deps.markerEvidenceIds.values()}),
        limitations=[
            "Measured composition and marker evidence remain available. No biological "
            "interpretation or treatment observation was accepted after model failure.",
            error_detail,
        ],
        stopReason="The model's biological interpretation could not be validated.",
        runInfo=getattr(
            error,
            "agent_run_info",
            AgentRunInfo(
                agentName="biological_interpretation_failed", modelName=model_name
            ),
        ),
    )
    logger.warning(
        f"Biological interpretation failed; observed evidence was retained: {error_detail}"
    )
    return report


def _prepare_biological_interpretation_dependencies(
    store: Any,
    *,
    cluster: Any,
    from_assay: str | None,
    graph_assay: str | None,
    marker_assay_type: str | None,
    sample_column: str | None,
    condition_column: str | None,
    tuning_handoff: TuningBiologyHandoff | None,
    experimental_handoff: ExperimentalBiologyHandoff | None,
    marker: Any,
    marker_features: Any,
    allow_marker_search: bool,
    max_clusters: int,
    max_markers: int,
    marker_min_score: float,
    marker_min_fraction: float,
) -> BiologicalInterpretationDependencies:
    expected_selections: list[ArtifactRef] = []
    if tuning_handoff is not None:
        if tuning_handoff.clusterArtifact is None:
            raise ValueError("tuning_handoff lacks a cluster artifact")
        tuning_selection = core_artifact_reference(tuning_handoff.cellSelection)
        if not isinstance(tuning_selection, ArtifactRef):
            raise ValueError("tuning_handoff lacks an exact cell selection")
        expected_selections.append(tuning_selection)
        if cluster is not None and (
            artifact_reference(cluster) != tuning_handoff.clusterArtifact
        ):
            raise ValueError("cluster conflicts with tuning_handoff")
        if from_assay is not None and from_assay != tuning_handoff.fromAssay:
            raise ValueError("from_assay conflicts with tuning_handoff")
        if graph_assay is not None and graph_assay != tuning_handoff.graphAssay:
            raise ValueError("graph_assay conflicts with tuning_handoff")
        cluster = tuning_handoff.clusterArtifact
        from_assay = tuning_handoff.fromAssay
        graph_assay = tuning_handoff.graphAssay
    if experimental_handoff is not None:
        experimental_selection = core_artifact_reference(
            experimental_handoff.cellSelection
        )
        if not isinstance(experimental_selection, ArtifactRef):
            raise ValueError("experimental_handoff lacks an exact cell selection")
        expected_selections.append(experimental_selection)
        if len(expected_selections) == 2 and (
            expected_selections[0] != expected_selections[1]
        ):
            raise ValueError(
                "Experimental and tuning handoffs use different cell selections"
            )
        if (
            condition_column is not None
            and condition_column != experimental_handoff.conditionColumn
        ):
            raise ValueError("condition_column conflicts with experimental_handoff")
        aggregation_unit = (
            experimental_handoff.independentUnit or experimental_handoff.observationUnit
        )
        if sample_column is not None and sample_column != aggregation_unit:
            raise ValueError("sample_column conflicts with experimental_handoff")
        condition_column = experimental_handoff.conditionColumn
        sample_column = aggregation_unit
    if cluster is None:
        raise ValueError("cluster must identify an exact cluster artifact")
    if not 1 <= max_clusters <= _MAX_CLUSTERS:
        raise ValueError(f"max_clusters must be between 1 and {_MAX_CLUSTERS}")
    if not 1 <= max_markers <= _MAX_MARKERS:
        raise ValueError(f"max_markers must be between 1 and {_MAX_MARKERS}")
    if not 0 < marker_min_score <= 1:
        raise ValueError("marker_min_score must be greater than 0 and at most 1")
    if not 0 <= marker_min_fraction <= 1:
        raise ValueError("marker_min_fraction must be between 0 and 1")
    if allow_marker_search and marker is None and marker_features is None:
        raise ValueError("marker_features is required when marker search is authorized")

    cluster = core_artifact_reference(cluster)
    marker = core_artifact_reference(marker)
    marker_features = core_artifact_reference(marker_features)
    if not isinstance(cluster, ArtifactRef):
        raise TypeError("cluster must be an ArtifactRef")
    if marker is not None and (
        not isinstance(marker, ArtifactRef) or marker.kind != "marker_table"
    ):
        raise TypeError("marker must be a marker_table ArtifactRef")
    if marker_features is not None and (
        not isinstance(marker_features, ArtifactRef)
        or marker_features.kind != "feature_selection"
    ):
        raise TypeError("marker_features must be a feature_selection ArtifactRef")
    cluster_artifact = artifact_reference(cluster)
    if cluster_artifact.kind not in {"cluster_labels", "cluster_cut"}:
        raise ValueError(
            "cluster must identify a cluster_labels or cluster_cut artifact"
        )
    if cluster_artifact.scope == "datastore" and cluster_artifact.assay is not None:
        raise ValueError("datastore-scoped cluster artifacts must not name an assay")
    if (
        tuning_handoff is not None
        and cluster_artifact.scope == "datastore"
        and not tuning_handoff.markerAssay
    ):
        raise ValueError(
            "Integrated tuning handoffs must explicitly identify markerAssay"
        )
    resolved_graph_assay = graph_assay or cluster_artifact.assay
    if (
        resolved_graph_assay is not None
        and cluster_artifact.scope == "assay"
        and cluster_artifact.assay != resolved_graph_assay
    ):
        raise ValueError("cluster belongs to a different assay")
    resolved_marker_assay = (
        tuning_handoff.markerAssay or cluster_artifact.assay
        if tuning_handoff is not None
        else (
            marker.assay
            if isinstance(marker, ArtifactRef)
            else (
                marker_features.assay
                if isinstance(marker_features, ArtifactRef)
                else from_assay or cluster_artifact.assay
            )
        )
    )
    if cluster_artifact.scope == "datastore" and not resolved_marker_assay:
        raise ValueError(
            "from_assay is required to resolve markers for integrated clusters"
        )
    if isinstance(marker, ArtifactRef) and marker.assay != resolved_marker_assay:
        raise ValueError("marker artifact belongs to a different marker assay")
    if (
        isinstance(marker_features, ArtifactRef)
        and marker_features.assay != resolved_marker_assay
    ):
        raise ValueError("marker feature selection belongs to a different assay")

    cluster_status = store.inspect_artifact(cluster)
    if not getattr(cluster_status, "exists", True):
        raise ValueError("cluster artifact does not exist")
    if not getattr(cluster_status, "complete", False):
        raise ValueError("cluster artifact is incomplete")
    raw_selection = (getattr(cluster_status, "inputs", None) or {}).get(
        "cell_selection"
    )
    if not isinstance(raw_selection, Mapping):
        raise ValueError("cluster artifact has no cell-selection input")
    cell_selection = ArtifactRef.from_dict(dict(raw_selection))
    if (
        cell_selection.scope != "datastore"
        or cell_selection.kind != "cell_selection"
        or cell_selection.assay is not None
    ):
        raise ValueError("cluster artifact has an invalid cell-selection input")
    if any(selection != cell_selection for selection in expected_selections):
        raise ValueError("handoff cell selection conflicts with cluster")
    cell_indices = read_stored_selection_indices(
        store.zw,
        cell_selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    ).astype(np.int64, copy=False)
    if isinstance(marker, ArtifactRef):
        marker_status = store.inspect_artifact(marker)
        if not getattr(marker_status, "exists", True):
            raise ValueError("marker artifact does not exist")
        if not getattr(marker_status, "complete", False):
            raise ValueError("marker artifact is incomplete")
        marker_inputs = getattr(marker_status, "inputs", None) or {}
        stored_clusters = marker_inputs.get("clusters")
        expected_cluster = artifact_reference(cluster)
        if (
            not isinstance(stored_clusters, Mapping)
            or stored_clusters.get("artifact_id") != expected_cluster.artifactId
            or stored_clusters.get("kind") != expected_cluster.kind
            or stored_clusters.get("scope") != expected_cluster.scope
            or stored_clusters.get("assay") != expected_cluster.assay
        ):
            raise ValueError(
                "marker artifact is not linked to the exact cluster artifact"
            )
    return BiologicalInterpretationDependencies(
        store=store,
        cluster=cluster,
        cellSelection=cell_selection,
        cellIndices=cell_indices,
        fromAssay=from_assay or cluster_artifact.assay or resolved_marker_assay,
        graphAssay=resolved_graph_assay,
        markerAssay=resolved_marker_assay,
        markerAssayType=marker_assay_type,
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
