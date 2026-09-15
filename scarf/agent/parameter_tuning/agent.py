from collections.abc import Mapping, Sequence
from typing import Any

from ...storage.refs import ArtifactRef
from ...utils.logging import logger
from .._deps import AGENT_INSTALL_HINT
from ..config import AgentRunConfig
from ..config.agent_exec import run_agent_sync
from ..tools import artifact_reference, core_artifact_reference
from ..types import AgentRunInfo, ExperimentalTuningHandoff
from .contracts import (
    _CANDIDATE_ID,
    IntegrationCandidateEvaluation,
    ParameterCandidate,
    ParameterCandidateEvaluation,
    ParameterSearchPlan,
    ParameterTuningAssayInput,
    ParameterTuningBatchSearchPlan,
    ParameterTuningDependencies,
    ParameterTuningReport,
)
from .execution import execute_parameter_candidate, normalized_artifact_shape
from .prompts import (
    build_initial_parameter_candidates,
    get_default_parameter_candidates,
    parameter_batch_search_prompt,
    parameter_batch_search_system_prompt,
    parameter_batch_selection_prompt,
    parameter_batch_selection_system_prompt,
    parameter_search_prompt,
    parameter_search_system_prompt,
    parameter_tuning_prompt,
    parameter_tuning_system_prompt,
)
from .selection import (
    annotate_candidate_dominance,
    pending_parameter_tuning_batch_report,
    pending_parameter_tuning_report,
    promote_parameter_candidate,
    select_final_parameter_graph,
    validate_parameter_batch_search_plan,
    validate_parameter_search_plan,
    validate_parameter_tuning_batch_report,
    validate_parameter_tuning_report,
)

try:
    from pydantic_ai import UnexpectedModelBehavior, UsageLimitExceeded
except ImportError as exc:
    raise ImportError(AGENT_INSTALL_HINT) from exc


_MAX_CANDIDATES_OFFERED = 25


def _resolve_experimental_tuning_handoff(
    *,
    normalized_cell_selection: ArtifactRef,
    batch_columns: Sequence[str],
    preservation_columns: Sequence[str],
    experimental_handoff: ExperimentalTuningHandoff | None,
) -> tuple[ArtifactRef, list[str], list[str]]:
    resolved_batch_columns = list(batch_columns)
    resolved_preservation_columns = list(preservation_columns)
    if experimental_handoff is None:
        return (
            normalized_cell_selection,
            resolved_batch_columns,
            resolved_preservation_columns,
        )

    handoff_batch_columns = list(experimental_handoff.batchColumns)
    canonical_batch_columns = sorted(set(handoff_batch_columns))
    if len(canonical_batch_columns) != len(handoff_batch_columns):
        raise ValueError("experimental_handoff batch columns must be unique")
    handoff_cell_selection = core_artifact_reference(experimental_handoff.cellSelection)
    if not isinstance(handoff_cell_selection, ArtifactRef):
        raise ValueError("experimental_handoff lacks an exact cell selection")
    if handoff_cell_selection != normalized_cell_selection:
        raise ValueError("normalized selection conflicts with experimental_handoff")
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
        normalized_cell_selection,
        canonical_batch_columns,
        list(experimental_handoff.preservationColumns),
    )


def prepare_parameter_tuning_dependencies(
    store: Any,
    *,
    normalized: ArtifactRef,
    candidates: Sequence[ParameterCandidate] | None = None,
    batch_columns: Sequence[str] = (),
    preservation_columns: Sequence[str] = (),
    experimental_handoff: ExperimentalTuningHandoff | None = None,
    max_candidates: int = 5,
    max_refined_candidates: int = 0,
    allow_harmony_refinement: bool = True,
    pair_harmony_candidates: bool | None = None,
    min_cluster_cells: int = 20,
    identity_feature_limit: int = 64,
) -> tuple[ParameterTuningDependencies, list[str]]:
    """Validate one assay request and construct branch-safe dependencies."""

    if max_candidates < 1:
        raise ValueError("max_candidates must be at least one")
    if max_refined_candidates < 0:
        raise ValueError("max_refined_candidates must be non-negative")
    if pair_harmony_candidates is not None and not isinstance(
        pair_harmony_candidates,
        bool,
    ):
        raise TypeError("pair_harmony_candidates must be a boolean or None")
    if min_cluster_cells < 1:
        raise ValueError("min_cluster_cells must be at least one")
    if identity_feature_limit < 2:
        raise ValueError("identity_feature_limit must be at least two")
    normalized = core_artifact_reference(normalized)
    if not isinstance(normalized, ArtifactRef) or normalized.kind != "normalized":
        raise TypeError("normalized must be a normalized ArtifactRef")
    if normalized.assay is None:
        raise ValueError("normalized artifact has no assay")
    normalized_status = store.inspect_artifact(normalized)
    if not getattr(normalized_status, "exists", True):
        raise ValueError("normalized artifact does not exist")
    if not getattr(normalized_status, "complete", False):
        raise ValueError("normalized artifact is incomplete")
    raw_cell_selection = (getattr(normalized_status, "inputs", None) or {}).get(
        "cell_selection"
    )
    if not isinstance(raw_cell_selection, Mapping):
        raise ValueError("normalized artifact has no cell-selection input")
    normalized_cell_selection = ArtifactRef.from_dict(dict(raw_cell_selection))
    if (
        normalized_cell_selection.scope != "datastore"
        or normalized_cell_selection.kind != "cell_selection"
        or normalized_cell_selection.assay is not None
    ):
        raise ValueError("normalized artifact has an invalid cell-selection input")
    from_assay = normalized.assay
    (
        resolved_cell_selection,
        resolved_batch_columns,
        resolved_preservation_columns,
    ) = _resolve_experimental_tuning_handoff(
        normalized_cell_selection=normalized_cell_selection,
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
        (
            experimental_handoff is not None
            and experimental_handoff.batchAction == "evaluateHarmony"
        )
        if pair_harmony_candidates is None
        else pair_harmony_candidates
    )
    candidate_values = build_initial_parameter_candidates(
        seed_candidates,
        pair_harmony=pair_harmony,
    )
    if len(candidate_values) + max_refined_candidates > _MAX_CANDIDATES_OFFERED:
        raise ValueError(
            "Initial and refined candidates may contain at most "
            f"{_MAX_CANDIDATES_OFFERED} values"
        )
    candidate_map: dict[str, ParameterCandidate] = {}
    for candidate in candidate_values:
        if not _CANDIDATE_ID.fullmatch(candidate.candidateId):
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
        cellSelection=resolved_cell_selection,
        normalizedShape=normalized_shape,
        fromAssay=from_assay,
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
        """Resolve the exact selected native branch without mutating state."""

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
    _refresh_candidate_dominance(deps)


def _refresh_candidate_dominance(deps: ParameterTuningDependencies) -> None:
    ordered = [
        deps.evaluations[candidate_id]
        for candidate_id in deps.executionOrder
        if candidate_id in deps.evaluations
    ]
    deps.evaluations.update(
        {
            evaluation.candidateId: evaluation
            for evaluation in annotate_candidate_dominance(ordered)
        }
    )


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
    _refresh_candidate_dominance(deps)


def execute_parameter_search_plan(
    deps: ParameterTuningDependencies,
    plan: ParameterSearchPlan,
    *,
    initial_candidate_ids: Sequence[str],
    max_refined_candidates: int,
) -> tuple[ParameterSearchPlan, tuple[ParameterCandidateEvaluation, ...]]:
    """Validate and execute one already-proposed bounded refinement plan."""

    validated = validate_parameter_search_plan(
        plan,
        deps,
        initial_candidate_ids=initial_candidate_ids,
        max_refined_candidates=max_refined_candidates,
    )
    _register_refined_parameter_candidates(deps, validated.candidates)
    return (
        validated,
        tuple(
            deps.evaluations[candidate.candidateId]
            for candidate in validated.candidates
        ),
    )


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
    planned_total = sum(item.maxCandidates for item in assay_inputs)
    if planned_total > max_total_candidates:
        raise ValueError(
            f"Batched tuning requests {planned_total} candidate branches; "
            f"the global limit is {max_total_candidates}"
        )
    normalized_refs = [
        core_artifact_reference(item.normalized) for item in assay_inputs
    ]
    if any(
        not isinstance(ref, ArtifactRef)
        or ref.kind != "normalized"
        or ref.assay is None
        for ref in normalized_refs
    ):
        raise TypeError(
            "Every batched tuning input requires an assay-scoped normalized artifact"
        )
    assay_names = [ref.assay for ref in normalized_refs]
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
    for item, assay_name in zip(assay_inputs, assay_names, strict=True):
        deps, candidate_ids = prepare_parameter_tuning_dependencies(
            store,
            normalized=item.normalized,
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
        dependencies[assay_name] = deps
        initial_ids[assay_name] = candidate_ids
        max_refined_by_assay[assay_name] = item.maxRefinedCandidates
    cell_selections = {deps.cellSelection for deps in dependencies.values()}
    if len(cell_selections) != 1:
        raise ValueError("Batched tuning inputs must use the same cell selection")
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
        except (UnexpectedModelBehavior, UsageLimitExceeded) as exc:
            logger.warning(
                "Batched parameter refinement model run failed within its bounds "
                f"({type(exc).__name__}); pausing without a refinement decision"
            )
            failed_info = getattr(
                exc,
                "agent_run_info",
                AgentRunInfo(agentName="parameter_batch_search_planning_needs_input"),
            )
            failed_plans = {
                assay: ParameterSearchPlan(
                    status="complete",
                    basedOnCandidateIds=[
                        next(
                            (
                                candidate_id
                                for candidate_id in initial_ids[assay]
                                if dependencies[assay].evaluations[candidate_id].status
                                == "done"
                                and dependencies[assay]
                                .evaluations[candidate_id]
                                .eligible
                            ),
                            initial_ids[assay][0],
                        )
                    ],
                    rationale=(
                        "The required bounded refinement review was unavailable."
                    ),
                    evidenceIds=sorted(
                        {
                            evidence_id
                            for candidate_id in initial_ids[assay]
                            for evidence_id in dependencies[assay]
                            .evaluations[candidate_id]
                            .evidenceIds
                        }
                    ),
                    stoppingCriteria=[
                        "Obtain a grounded refinement disposition before selection."
                    ],
                    runInfo=failed_info,
                )
                for assay in assay_names
            }
            batch_plan = ParameterTuningBatchSearchPlan(
                assayPlans=failed_plans,
                runInfo=failed_info,
            )
            return pending_parameter_tuning_batch_report(
                dependencies,
                search_plans=batch_plan.assayPlans,
                primary_assay=resolved_primary,
            ).model_copy(update={"runInfo": failed_info})
        else:
            if not isinstance(
                planning_execution.output, ParameterTuningBatchSearchPlan
            ):
                raise TypeError("Batched parameter planner returned an unexpected type")
            validated_batch_plan = validate_parameter_batch_search_plan(
                planning_execution.output,
                dependencies,
                initial_candidate_ids=initial_ids,
                max_refined_by_assay=max_refined_by_assay,
            )
            batch_plan = validated_batch_plan.model_copy(
                update={
                    "assayPlans": {
                        assay: plan.model_copy(
                            update={"runInfo": planning_execution.runInfo}
                        )
                        for assay, plan in validated_batch_plan.assayPlans.items()
                    },
                    "runInfo": planning_execution.runInfo,
                }
            )
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
    except (UnexpectedModelBehavior, UsageLimitExceeded) as exc:
        logger.warning(
            "Batched parameter selection model run failed within its bounds "
            f"({type(exc).__name__}); "
            "returning needsInput with the completed executor evidence"
        )
        return pending_parameter_tuning_batch_report(
            dependencies,
            search_plans=batch_plan.assayPlans,
            primary_assay=resolved_primary,
        ).model_copy(
            update={
                "runInfo": getattr(
                    exc,
                    "agent_run_info",
                    AgentRunInfo(agentName="parameter_tuning_batch_needs_input"),
                )
            }
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

    deps, initial_candidate_ids = prepare_parameter_tuning_dependencies(
        store,
        normalized=normalized,
        candidates=candidates,
        batch_columns=batch_columns,
        preservation_columns=preservation_columns,
        experimental_handoff=experimental_handoff,
        max_candidates=max_candidates,
        max_refined_candidates=max_refined_candidates,
        min_cluster_cells=min_cluster_cells,
        identity_feature_limit=identity_feature_limit,
    )
    from_assay = deps.fromAssay
    cell_selection = artifact_reference(deps.cellSelection)
    logger.info(
        f"Starting parameter tuning for assay {from_assay!r}; "
        f"candidate_limit={max_candidates}, "
        f"refinement_limit={max_refined_candidates}"
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
                    cell_selection=cell_selection,
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
        except (UnexpectedModelBehavior, UsageLimitExceeded) as exc:
            logger.warning(
                f"Parameter refinement planning for assay {from_assay!r} "
                f"failed within its model-run bounds ({type(exc).__name__}); "
                "pausing without a refinement decision"
            )
            successful_parent = next(
                (
                    evaluation
                    for evaluation in initial_evaluations
                    if evaluation.status == "done" and evaluation.eligible
                ),
                initial_evaluations[0],
            )
            failed_plan = ParameterSearchPlan(
                status="complete",
                basedOnCandidateIds=[successful_parent.candidateId],
                rationale=("The required bounded refinement review was unavailable."),
                evidenceIds=sorted(
                    {
                        evidence_id
                        for evaluation in initial_evaluations
                        for evidence_id in evaluation.evidenceIds
                    }
                ),
                stoppingCriteria=[
                    "Obtain a grounded refinement disposition before selection."
                ],
                runInfo=getattr(
                    exc,
                    "agent_run_info",
                    AgentRunInfo(agentName="parameter_search_planning_needs_input"),
                ),
            )
            return pending_parameter_tuning_report(
                deps,
                search_plan=failed_plan,
                agent_name="parameter_search_planning_needs_input",
            ).model_copy(update={"runInfo": failed_plan.runInfo})
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
                cell_selection=cell_selection,
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
    except (UnexpectedModelBehavior, UsageLimitExceeded) as exc:
        logger.warning(
            f"Parameter selection for assay {from_assay!r} failed within its "
            f"model-run bounds ({type(exc).__name__}); returning needsInput "
            "with the completed executor evidence"
        )
        return pending_parameter_tuning_report(
            deps,
            search_plan=plan,
            agent_name="parameter_tuning_needs_input",
        ).model_copy(
            update={
                "runInfo": getattr(
                    exc,
                    "agent_run_info",
                    AgentRunInfo(agentName="parameter_tuning_needs_input"),
                )
            }
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
