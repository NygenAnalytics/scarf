from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ...storage.refs import ArtifactRef
from ...utils.logging import logger
from ..tools import core_artifact_reference
from .contracts import (
    FinalGraphSelection,
    IntegrationCandidateEvaluation,
    ParameterCandidate,
    ParameterCandidateEvaluation,
    ParameterMetrics,
    ParameterTuningReport,
)


def _finite_metric(
    objectives: dict[str, tuple[float, int, str]],
    name: str,
    value: float | None,
    *,
    direction: int,
    evidence_class: str,
) -> None:
    if value is not None and np.isfinite(value):
        objectives[name] = (float(value), direction, evidence_class)


def _candidate_objectives(
    metrics: ParameterMetrics,
) -> dict[str, tuple[float, int, str]]:
    objectives: dict[str, tuple[float, int, str]] = {}
    for name, value in (
        ("minClusterFraction", metrics.minClusterFraction),
        ("graphSilhouetteMedian", metrics.graphSilhouetteMedian),
        ("membershipStrengthMean", metrics.membershipStrengthMean),
        ("membershipStrengthP10", metrics.membershipStrengthP10),
        ("clusterConnectivity", metrics.clusterConnectivity),
    ):
        _finite_metric(
            objectives,
            name,
            value,
            direction=1,
            evidence_class="geometric",
        )
    for name, value in (
        ("seedStability", metrics.seedStability),
        ("subsampleStability", metrics.subsampleStability),
    ):
        _finite_metric(
            objectives,
            name,
            value,
            direction=1,
            evidence_class="resamplingStability",
        )
    for name, value in (
        ("markerCoherence", metrics.markerCoherence),
        ("markerSpecificityMedian", metrics.markerSpecificityMedian),
    ):
        _finite_metric(
            objectives,
            name,
            value,
            direction=1,
            evidence_class="markerCoherence",
        )
    _finite_metric(
        objectives,
        "crossUnitSupport",
        metrics.crossUnitSupport,
        direction=1,
        evidence_class="crossUnitSupport",
    )
    _finite_metric(
        objectives,
        "doubletHighScoreConcentration",
        metrics.doubletHighScoreConcentration,
        direction=-1,
        evidence_class="qualityControl",
    )
    for column, value in metrics.technicalAssociation.items():
        _finite_metric(
            objectives,
            f"technicalAssociation:{column}",
            value,
            direction=-1,
            evidence_class="technical",
        )
    for column, value in metrics.batchMixing.items():
        _finite_metric(
            objectives,
            f"batchMixing:{column}",
            value,
            direction=1,
            evidence_class="batchRemoval",
        )
    for column, values in metrics.biologicalPreservation.items():
        for name, value in values.items():
            _finite_metric(
                objectives,
                f"biologicalPreservation:{column}:{name}",
                value,
                direction=1,
                evidence_class="protectedVariablePreservation",
            )
    return objectives


def _single_varied_parameter(
    left: ParameterCandidate,
    right: ParameterCandidate,
) -> str | None:
    if (
        left.reductionMethod != right.reductionMethod
        or left.useHarmony != right.useHarmony
    ):
        return None
    varied = [
        name
        for name in ("dimensions", "neighborsK", "leidenResolution")
        if getattr(left, name) != getattr(right, name)
    ]
    return varied[0] if len(varied) == 1 else None


def _dominance_metrics(
    left: ParameterMetrics,
    right: ParameterMetrics,
    *,
    tolerance: float,
) -> list[str]:
    left_objectives = _candidate_objectives(left)
    right_objectives = _candidate_objectives(right)
    if not left_objectives or set(left_objectives) != set(right_objectives):
        return []
    classes = {value[2] for value in left_objectives.values()}
    if len(classes) < 2:
        return []
    strict: list[str] = []
    for name in sorted(left_objectives):
        left_value, direction, _evidence_class = left_objectives[name]
        right_value = right_objectives[name][0]
        difference = direction * (left_value - right_value)
        if difference < -tolerance:
            return []
        if difference > tolerance:
            strict.append(name)
    return strict


def annotate_candidate_dominance(
    evaluations: Sequence[ParameterCandidateEvaluation],
    *,
    tolerance: float = 0.02,
) -> tuple[ParameterCandidateEvaluation, ...]:
    """Attach conservative pairwise Pareto evidence to comparable candidates."""

    values = list(evaluations)
    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError("Dominance tolerance must be finite and non-negative")
    completed = [value for value in values if value.status == "done" and value.eligible]
    dominated_by: dict[str, list[str]] = {value.candidateId: [] for value in completed}
    dominates: dict[str, list[str]] = {value.candidateId: [] for value in completed}
    metrics_by_id: dict[str, dict[str, list[str]]] = {
        value.candidateId: {} for value in completed
    }
    comparable: set[str] = set()
    for left in completed:
        for right in completed:
            if left.candidateId == right.candidateId or (
                _single_varied_parameter(left.parameters, right.parameters) is None
            ):
                continue
            comparable.add(left.candidateId)
            strict = _dominance_metrics(
                left.metrics,
                right.metrics,
                tolerance=tolerance,
            )
            if not strict:
                continue
            dominates[left.candidateId].append(right.candidateId)
            dominated_by[right.candidateId].append(left.candidateId)
            metrics_by_id[left.candidateId][f"dominates:{right.candidateId}"] = strict
            metrics_by_id[right.candidateId][f"dominatedBy:{left.candidateId}"] = strict

    annotated: list[ParameterCandidateEvaluation] = []
    for evaluation in values:
        if evaluation.candidateId not in dominated_by:
            annotated.append(evaluation)
            continue
        candidate_id = evaluation.candidateId
        candidate_dominators = sorted(set(dominated_by[candidate_id]))
        candidate_dominates = sorted(set(dominates[candidate_id]))
        updated_metrics = evaluation.metrics.model_copy(
            update={
                "paretoOptimal": (
                    not candidate_dominators if candidate_id in comparable else None
                ),
                "dominatedByCandidateIds": candidate_dominators,
                "dominatesCandidateIds": candidate_dominates,
                "dominanceMetrics": metrics_by_id[candidate_id],
            }
        )
        prefix = f"candidate:{candidate_id}:"
        retained_evidence = [
            evidence_id
            for evidence_id in evaluation.evidenceIds
            if not (
                evidence_id == f"{prefix}paretoDominance"
                or evidence_id.startswith(f"{prefix}dominatedBy:")
                or evidence_id.startswith(f"{prefix}dominates:")
            )
        ]
        dominance_evidence = (
            [f"{prefix}paretoDominance"] if candidate_id in comparable else []
        )
        dominance_evidence.extend(
            f"{prefix}dominatedBy:{other}" for other in candidate_dominators
        )
        dominance_evidence.extend(
            f"{prefix}dominates:{other}" for other in candidate_dominates
        )
        annotated.append(
            evaluation.model_copy(
                update={
                    "metrics": updated_metrics,
                    "evidenceIds": [
                        *retained_evidence,
                        *dominance_evidence,
                    ],
                }
            )
        )
    return tuple(annotated)


def harmony_acceptance_gate(
    native: ParameterCandidateEvaluation | None,
    harmony: ParameterCandidateEvaluation | None,
    *,
    batch_columns: Sequence[str],
    protected_columns: Sequence[str],
    independent_unit_columns: Sequence[str] = (),
    tolerance: float = 0.05,
    require_doublet_evidence: bool = False,
) -> tuple[bool, list[str]]:
    """Require matched batch improvement without material biological loss."""

    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError("Harmony gate tolerance must be finite and non-negative")
    reasons: list[str] = []
    if native is None or harmony is None:
        return False, ["Matched native and Harmony candidates are unavailable."]
    if native.status != "done" or not native.eligible:
        reasons.append("The matched native candidate is not an eligible execution.")
    if harmony.status != "done" or not harmony.eligible:
        reasons.append("The matched Harmony candidate is not an eligible execution.")
    if native.parameters.useHarmony or not harmony.parameters.useHarmony:
        reasons.append("Candidates do not have native and Harmony correction modes.")
    native_parameters = native.parameters.model_dump(
        mode="json",
        exclude={"candidateId", "useHarmony"},
    )
    harmony_parameters = harmony.parameters.model_dump(
        mode="json",
        exclude={"candidateId", "useHarmony"},
    )
    if native_parameters != harmony_parameters:
        reasons.append("Native and Harmony candidate parameters are not matched.")
    if core_artifact_reference(native.cellSelection) != core_artifact_reference(
        harmony.cellSelection
    ):
        reasons.append("Native and Harmony candidates use different cell selections.")

    columns = list(dict.fromkeys(batch_columns))
    if not columns:
        reasons.append("No approved batch metric was supplied.")
    batch_deltas: dict[str, float] = {}
    for column in columns:
        native_score = native.metrics.batchMixing.get(column)
        harmony_score = harmony.metrics.batchMixing.get(column)
        if native_score is None or harmony_score is None:
            reasons.append(f"Batch comparison is missing for {column!r}.")
            continue
        batch_deltas[column] = harmony_score - native_score
    if columns and len(batch_deltas) == len(columns):
        if not any(delta > tolerance for delta in batch_deltas.values()):
            reasons.append(
                "Harmony did not improve an approved batch metric beyond tolerance."
            )
        if any(delta < -tolerance for delta in batch_deltas.values()):
            reasons.append("Harmony materially worsened an approved batch metric.")

    for column in dict.fromkeys(protected_columns):
        native_scores = native.metrics.biologicalPreservation.get(column)
        harmony_scores = harmony.metrics.biologicalPreservation.get(column)
        if not native_scores or not harmony_scores:
            reasons.append(f"Protected comparison is missing for {column!r}.")
            continue
        missing_metrics = sorted(
            {"clisi", "graphConnectivity"} - (set(native_scores) & set(harmony_scores))
        )
        if missing_metrics:
            reasons.append(
                f"Required protected metrics {missing_metrics} are missing for {column!r}."
            )
            continue
        if set(native_scores) != set(harmony_scores):
            reasons.append(f"Protected metrics do not align for {column!r}.")
            continue
        if any(
            harmony_scores[name] < native_scores[name] - tolerance
            for name in native_scores
        ):
            reasons.append(
                f"Harmony materially degraded protected evidence for {column!r}."
            )

    if independent_unit_columns:
        if (
            native.metrics.crossUnitSupport is None
            or harmony.metrics.crossUnitSupport is None
        ):
            reasons.append("Cross-unit support comparison is missing.")
        elif (
            harmony.metrics.crossUnitSupport
            < native.metrics.crossUnitSupport - tolerance
        ):
            reasons.append("Harmony materially degraded cross-unit support.")

    if (
        native.metrics.markerCoherence is None
        or harmony.metrics.markerCoherence is None
    ):
        reasons.append("Marker-coherence comparison is missing.")
    elif harmony.metrics.markerCoherence < native.metrics.markerCoherence - tolerance:
        reasons.append("Harmony materially degraded marker coherence.")

    for label, native_value, harmony_value in (
        (
            "marker specificity",
            native.metrics.markerSpecificityMedian,
            harmony.metrics.markerSpecificityMedian,
        ),
        (
            "cluster connectivity",
            native.metrics.clusterConnectivity,
            harmony.metrics.clusterConnectivity,
        ),
        (
            "membership strength",
            native.metrics.membershipStrengthMean,
            harmony.metrics.membershipStrengthMean,
        ),
    ):
        if native_value is None and harmony_value is None:
            continue
        if native_value is None or harmony_value is None:
            reasons.append(f"Matched {label} comparison is missing.")
        elif harmony_value < native_value - tolerance:
            reasons.append(f"Harmony materially degraded {label}.")

    native_doublet = native.metrics.doubletHighScoreConcentration
    harmony_doublet = harmony.metrics.doubletHighScoreConcentration
    if (
        require_doublet_evidence
        or native_doublet is not None
        or harmony_doublet is not None
    ):
        if native_doublet is None or harmony_doublet is None:
            reasons.append("Matched doublet-concentration comparison is missing.")
        elif harmony_doublet > native_doublet + tolerance:
            reasons.append("Harmony materially increased doublet concentration.")
    return not reasons, reasons


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
    report_cell_selection = core_artifact_reference(report.cellSelection)
    if not isinstance(report_cell_selection, ArtifactRef):
        raise ValueError("Parameter tuning report lacks an exact cell selection")
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
        if selected.clusterArtifact is None:
            raise ValueError("Recommended integration lacks an exact cluster artifact")
        if core_artifact_reference(selected.cellSelection) != report_cell_selection:
            raise ValueError("Recommended integration uses a different cell selection")
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
            or "clusters" not in selected_native.artifacts
        ):
            raise ValueError("Primary native recommendation lacks exact clusters")
        if (
            core_artifact_reference(selected_native.cellSelection)
            != report_cell_selection
        ):
            raise ValueError(
                "Recommended native candidate uses a different cell selection"
            )
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


def promote_parameter_candidate(
    store: Any,
    *,
    report: ParameterTuningReport,
    normalized: Any,
    identity_feature_limit: int = 64,
) -> ParameterCandidateEvaluation:
    """Resolve and verify the exact selected native branch without replaying it."""

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
    if "clusters" not in evaluation.artifacts:
        raise ValueError("Recommended candidate lacks an exact cluster artifact")
    normalized_ref = core_artifact_reference(normalized)
    if (
        not isinstance(normalized_ref, ArtifactRef)
        or normalized_ref.kind != "normalized"
        or normalized_ref.assay != report.fromAssay
    ):
        raise ValueError(
            "normalized must identify the report's exact normalized assay artifact"
        )
    status = store.inspect_artifact(normalized_ref)
    if not getattr(status, "exists", True) or not getattr(status, "complete", False):
        raise ValueError("normalized artifact is unavailable or incomplete")
    raw_selection = (getattr(status, "inputs", None) or {}).get("cell_selection")
    if not isinstance(raw_selection, Mapping):
        raise ValueError("normalized artifact has no cell-selection input")
    normalized_selection = ArtifactRef.from_dict(dict(raw_selection))
    if normalized_selection != core_artifact_reference(evaluation.cellSelection):
        raise ValueError(
            "Recommended candidate does not match normalized artifact lineage"
        )
    if normalized_selection != core_artifact_reference(report.cellSelection):
        raise ValueError(
            "Parameter tuning report does not match normalized artifact lineage"
        )
    if identity_feature_limit < 2:
        raise ValueError("identity_feature_limit must be at least two")
    logger.info(
        f"Resolved parameter candidate {evaluation.candidateId!r} for assay "
        f"{report.fromAssay!r} without replay"
    )
    return evaluation
