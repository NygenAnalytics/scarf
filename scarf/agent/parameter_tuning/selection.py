from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ...storage.refs import ArtifactRef
from ...utils.logging import logger
from .._deps import AGENT_INSTALL_HINT
from ..config import AgentRunConfig
from ..config.agent_exec import run_agent_sync
from ..tools import artifact_reference, core_artifact_reference
from ..types import AgentRunInfo, StageStatus
from .contracts import (
    _CANDIDATE_ID,
    ArtifactRecord,
    FinalGraphNeedsInput,
    FinalGraphSelection,
    IntegrationCandidateEvaluation,
    ParameterCandidate,
    ParameterCandidateEvaluation,
    ParameterMetrics,
    ParameterSearchPlan,
    ParameterSearchStatus,
    ParameterTuningBatchSearchPlan,
    ParameterTuningDependencies,
    ParameterTuningNeedsInput,
    ParameterTuningReport,
)
from .execution import _final_graph_options
from .prompts import final_graph_selection_prompt, final_graph_selection_system_prompt

try:
    from pydantic_ai import UnexpectedModelBehavior, UsageLimitExceeded
except ImportError as exc:
    raise ImportError(AGENT_INSTALL_HINT) from exc


def final_graph_options(
    report: ParameterTuningReport,
    integration_evaluations: Sequence[IntegrationCandidateEvaluation],
) -> dict[str, dict[str, Any]]:
    """Return the exact eligible graph options and option-scoped evidence."""

    return _final_graph_options(report, integration_evaluations)


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
    if deps.harmonyAuthorized and any(
        candidate.useHarmony for candidate in plan.candidates
    ):
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
        if not _CANDIDATE_ID.fullmatch(candidate.candidateId):
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


def parameter_evidence_classes(evidence_ids: Sequence[str]) -> frozenset[str]:
    """Infer stable scientific evidence classes from executor evidence IDs."""

    classes: set[str] = set()
    for evidence_id in evidence_ids:
        token = evidence_id.casefold()
        if (
            "seedstability" in token
            or "subsamplestability" in token
            or token.endswith(":stability")
        ):
            classes.add("resamplingStability")
        elif "marker" in token:
            classes.add("markerCoherence")
        elif "crossunitsupport" in token or "unitsupport" in token:
            classes.add("crossUnitSupport")
        elif (
            "protected" in token
            or "clisi" in token
            or ("graphconnectivity" in token and "clusterconnectivity" not in token)
        ):
            classes.add("protectedVariablePreservation")
        elif "doublet" in token:
            classes.add("qualityControl")
        elif "technical" in token or "batchmixing" in token:
            classes.add("technical")
        elif any(
            value in token
            for value in (
                "clusterconnectivity",
                "clusters",
                "geometry",
                "membershipstrength",
                "neighbor",
                "paretodominance",
                "silhouette",
            )
        ):
            classes.add("geometric")
    return frozenset(classes)


def require_dominated_candidate_evidence(
    selected: ParameterCandidateEvaluation,
    evidence_ids: Sequence[str],
    *,
    context: str,
) -> None:
    """Require two independent non-geometric classes for a dominated choice."""

    if not selected.metrics.dominatedByCandidateIds:
        return
    independent = parameter_evidence_classes(evidence_ids).intersection(
        {
            "markerCoherence",
            "resamplingStability",
            "crossUnitSupport",
            "protectedVariablePreservation",
            "qualityControl",
        }
    )
    if len(independent) < 2:
        raise ValueError(
            f"{context} selects a Pareto-dominated candidate and must cite at "
            "least two independent non-geometric evidence classes"
        )


def validate_parameter_tuning_report(
    report: ParameterTuningReport,
    deps: ParameterTuningDependencies,
    *,
    search_plan: ParameterSearchPlan | None = None,
) -> ParameterTuningReport:
    """Ground the model report in candidate executions recorded by the tool."""

    evaluations = list(
        annotate_candidate_dominance(
            [
                deps.evaluations[candidate_id]
                for candidate_id in deps.executionOrder
                if candidate_id in deps.evaluations
            ]
        )
    )
    evaluations_by_id = {
        evaluation.candidateId: evaluation for evaluation in evaluations
    }
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
    selected_evaluation: ParameterCandidateEvaluation | None = None
    if report.recommendedCandidateId is not None:
        selected = evaluations_by_id.get(report.recommendedCandidateId)
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
        selected_evaluation = selected
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
            if (
                selected_evaluation is not None
                and comparison.candidateId
                in selected_evaluation.metrics.dominatedByCandidateIds
                and _single_varied_parameter(
                    selected_evaluation.parameters,
                    evaluations_by_id[comparison.candidateId].parameters,
                )
                in {"neighborsK", "leidenResolution"}
            ):
                require_dominated_candidate_evidence(
                    selected_evaluation,
                    comparison.evidenceIds,
                    context=(f"The comparison with {comparison.candidateId!r}"),
                )

    graph_partition_dominators = (
        [
            candidate_id
            for candidate_id in selected_evaluation.metrics.dominatedByCandidateIds
            if candidate_id in evaluations_by_id
            and _single_varied_parameter(
                selected_evaluation.parameters,
                evaluations_by_id[candidate_id].parameters,
            )
            in {"neighborsK", "leidenResolution"}
        ]
        if selected_evaluation is not None
        else []
    )
    if (
        report.status == "done"
        and selected_evaluation is not None
        and graph_partition_dominators
    ):
        selection_evidence = [
            *report.evidenceIds,
            *(
                evidence_id
                for comparison in report.comparisons
                for evidence_id in comparison.evidenceIds
            ),
        ]
        require_dominated_candidate_evidence(
            selected_evaluation,
            selection_evidence,
            context="The tuning recommendation",
        )

    return report.model_copy(
        update={
            "fromAssay": deps.fromAssay,
            "cellSelection": artifact_reference(deps.cellSelection),
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
    cell_selections = [
        core_artifact_reference(assay_report.cellSelection)
        for assay_report in validated_reports.values()
    ]
    if (
        not cell_selections
        or not isinstance(cell_selections[0], ArtifactRef)
        or any(selection != cell_selections[0] for selection in cell_selections[1:])
    ):
        raise ValueError("Every assay report must use the same exact cell selection")
    return report.model_copy(
        update={
            "status": status,
            "fromAssay": primary_assay,
            "cellSelection": primary.cellSelection,
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


def pending_parameter_tuning_report(
    deps: ParameterTuningDependencies,
    *,
    search_plan: ParameterSearchPlan,
    agent_name: str,
) -> ParameterTuningReport:
    """Pause when structured selection is unavailable.

    Completed executor evidence is retained for an exact human resume, but it is
    never converted into an implicit scientific recommendation.
    """

    evaluations = [
        deps.evaluations[candidate_id]
        for candidate_id in deps.executionOrder
        if candidate_id in deps.evaluations
    ]
    successful = [item for item in evaluations if item.status == "done"]
    eligible = [item for item in successful if item.eligible]
    logger.warning(
        f"Parameter tuning for assay {deps.fromAssay!r} requires input after "
        f"model exhaustion: completed={len(successful)}, eligible={len(eligible)}"
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
            "The bounded structured selection was unavailable. Executor evidence "
            "is complete enough to resume, but it cannot choose a scientific "
            "alternative by itself."
        ),
        evidenceIds=known_evidence,
        limitations=["No candidate was selected merely to complete the workflow."],
        stopReason="The bounded screen completed without a valid decision.",
        needsInput=ParameterTuningNeedsInput(
            question=(
                "Select one eligible executed candidate and provide a scientific "
                "rationale tied to the cited evidence."
            ),
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


def pending_parameter_tuning_batch_report(
    dependencies: Mapping[str, ParameterTuningDependencies],
    *,
    search_plans: Mapping[str, ParameterSearchPlan],
    primary_assay: str,
) -> ParameterTuningReport:
    """Build one grounded pause over completed assay screens."""

    logger.warning(
        f"Pausing parameter tuning after model exhaustion for "
        f"{len(dependencies)} assay(s)"
    )
    assay_reports = {
        assay: pending_parameter_tuning_report(
            deps,
            search_plan=search_plans[assay],
            agent_name="parameter_tuning_batch_needs_input",
        )
        for assay, deps in dependencies.items()
    }
    aggregate = ParameterTuningReport(
        status="needsInput",
        assayReports=assay_reports,
        rationale=(
            "Structured model selection was unavailable. All completed evidence "
            "was retained without choosing a branch."
        ),
        evidenceIds=list(
            dict.fromkeys(
                evidence_id
                for assay_report in assay_reports.values()
                for evidence_id in assay_report.evidenceIds
            )
        ),
        limitations=["No assay candidate was selected merely to finish the workflow."],
        stopReason="The bounded native screens completed without valid decisions.",
        runInfo=AgentRunInfo(agentName="parameter_tuning_batch_needs_input"),
    )
    logger.warning(
        f"Parameter tuning batch pause status={aggregate.status}; "
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
        except (UnexpectedModelBehavior, UsageLimitExceeded) as exc:
            option_ids = sorted(options)
            logger.warning(
                "Final graph selection model run failed within its bounds "
                f"({type(exc).__name__}); "
                f"requesting input for {len(option_ids)} eligible options"
            )
            selection = validate_final_graph_selection(
                FinalGraphSelection(
                    status="needsInput",
                    markerAssay=marker_assay,
                    confidence="low",
                    rationale=(
                        "The bounded structured final-graph selection was unavailable."
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
                    runInfo=getattr(
                        exc,
                        "agent_run_info",
                        AgentRunInfo(
                            agentName="parameter_tuning_final_graph_needs_input"
                        ),
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
