from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ...storage.refs import ArtifactRef
from ...utils.logging import logger
from ..tools import core_artifact_reference
from .contracts import (
    ParameterCandidateEvaluation,
    ParameterTuningReport,
)


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
    native_assay: str | None = None,
) -> ParameterTuningReport:
    """Attach the executor-selected native cluster branch."""

    logger.debug(f"Finalizing parameter graph selection: marker_assay={marker_assay!r}")
    if report.status != "done":
        raise ValueError("Parameter tuning must be done before final graph selection")
    if not marker_assay:
        raise ValueError("marker_assay must be non-empty")
    report_cell_selection = core_artifact_reference(report.cellSelection)
    if not isinstance(report_cell_selection, ArtifactRef):
        raise ValueError("Parameter tuning report lacks an exact cell selection")
    if marker_assay != report.fromAssay:
        raise ValueError(f"Unknown marker assay {marker_assay!r}")
    selected_assay = native_assay or report.fromAssay
    if selected_assay != report.fromAssay or report.recommendedCandidateId is None:
        raise ValueError("Selected assay lacks a native tuning recommendation")
    selected_native = next(
        (
            item
            for item in report.evaluations
            if item.candidateId == report.recommendedCandidateId
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
    if core_artifact_reference(selected_native.cellSelection) != report_cell_selection:
        raise ValueError("Recommended native candidate uses a different cell selection")
    finalized = report.model_copy(
        update={
            "totalCandidates": len(report.evaluations),
            "finalClusterColumn": selected_native.clusterColumn,
            "finalClusterArtifact": selected_native.artifacts["clusters"],
            "graphAssay": selected_assay,
            "markerAssay": marker_assay,
        }
    )
    logger.info(
        f"Finalized parameter graph selection: graph={selected_assay!r}, "
        f"marker_assay={marker_assay!r}, "
        f"cluster_column={selected_native.clusterColumn!r}"
    )
    return finalized


def promote_parameter_candidate(
    store: Any,
    *,
    report: ParameterTuningReport,
    normalized: Any,
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
    logger.info(
        f"Resolved parameter candidate {evaluation.candidateId!r} for assay "
        f"{report.fromAssay!r} without replay"
    )
    return evaluation
