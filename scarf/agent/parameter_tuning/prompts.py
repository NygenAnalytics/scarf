import json
from collections.abc import Mapping, Sequence
from textwrap import dedent
from typing import Any

from ..types import ArtifactReferenceModel
from .contracts import (
    IntegrationCandidateEvaluation,
    ParameterCandidate,
    ParameterCandidateEvaluation,
    ParameterSearchPlan,
    ParameterTuningDependencies,
    ParameterTuningReport,
    _default_parameter_candidates,
)
from .execution import _final_graph_options


def get_default_parameter_candidates() -> list[ParameterCandidate]:
    """Return a small one-factor candidate set around Scarf defaults."""

    return _default_parameter_candidates()


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
        you must not propose or modify batch columns. When proposing any Harmony
        refinement, base it on one matched corrected and uncorrected initial pair
        with otherwise identical parameters.

        Cite only evidenceIds from the initial evaluations. Identify the successful
        initial candidates that motivate refinement, state focused objectives, and
        provide concrete stopping criteria. Do not invent metrics, artifacts, or
        candidate ids. Treat pcaSilhouette, macroF1, and weightedF1 only as PCA
        cluster-separability metrics. Biological preservation evidence exists only
        in a non-empty biologicalPreservation map. Check every exact value before
        stating a ranking or trend, and keep narrative fields as plain prose
        without serialized JSON.
        """
    ).strip()


def parameter_evaluation_payload(
    evaluation: ParameterCandidateEvaluation,
) -> dict[str, Any]:
    """Return only candidate evidence needed for planning and selection."""
    metrics = evaluation.metrics.model_dump(mode="json")
    loading_items = list(evaluation.metrics.topLoadingGenes.items())
    bounded_loading_items = [
        *loading_items[:10],
        *(loading_items[-3:] if len(loading_items) > 13 else loading_items[10:]),
    ]
    metrics["topLoadingGenes"] = {
        component: genes[:10] for component, genes in bounded_loading_items
    }
    metrics["topMarkerGenes"] = {
        cluster: genes[:10]
        for cluster, genes in list(evaluation.metrics.topMarkerGenes.items())[:30]
    }
    return {
        "candidateId": evaluation.candidateId,
        "phase": evaluation.phase,
        "harmonyBatchColumns": evaluation.harmonyBatchColumns,
        "status": evaluation.status,
        "eligible": evaluation.eligible,
        "parameters": evaluation.parameters.model_dump(mode="json"),
        "effectiveDimensions": evaluation.effectiveDimensions,
        "metrics": metrics,
        "evidenceIds": evaluation.evidenceIds,
        "eligibilityReasons": evaluation.eligibilityReasons,
        "warnings": [warning[:500] for warning in evaluation.warnings[:10]],
        "error": evaluation.error[:500] if evaluation.error is not None else None,
    }


def parameter_search_prompt(
    *,
    from_assay: str,
    cell_selection: ArtifactReferenceModel,
    evaluations: Sequence[ParameterCandidateEvaluation],
    batch_columns: Sequence[str],
    preservation_columns: Sequence[str],
    harmony_authorized: bool,
    max_refined_candidates: int,
) -> str:
    """Build the planning prompt from deterministic initial evaluations."""

    evaluation_payload = [
        parameter_evaluation_payload(evaluation) for evaluation in evaluations
    ]
    correction_modes = ["none", "harmony"] if harmony_authorized else ["none"]
    return (
        dedent(
            """
        Inspect the completed initial screen for assay {from_assay} and exact
        cell-selection artifact {cell_selection}.

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
            cell_selection=cell_selection.artifactId,
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
        evidence for parameter quality. Treat pcaSilhouette, macroF1, and
        weightedF1 only as PCA cluster-separability metrics. Biological
        preservation evidence exists only in a non-empty biologicalPreservation
        map. A candidate with non-empty dominatedByCandidateIds is Pareto
        dominated. Selecting a dominated graph or resolution requires at least
        two independent non-geometric evidence classes that explain the
        tradeoff. Do not call any metric highest, lowest, improved, degraded, or
        monotonic without checking its exact value across every relevant
        candidate. Narrative fields contain plain prose only and must not contain
        serialized JSON keys or objects. When multiple candidates complete,
        return one comparison for every non-selected successful candidate. Each
        comparison must cite evidence from both the selected candidate and that
        comparator. Return only model-owned selection fields. Leave evaluations,
        selectedArtifacts, searchPlan, assayReports, integration fields, final
        graph fields, and runInfo out of the response because validation fills
        them from executor state. Return a concise structured report.
        """
        )
        .strip()
        .format(min_cluster_cells=min_cluster_cells)
    )


def parameter_tuning_prompt(
    *,
    from_assay: str,
    cell_selection: ArtifactReferenceModel,
    evaluations: Sequence[ParameterCandidateEvaluation],
    batch_columns: Sequence[str],
    preservation_columns: Sequence[str],
    search_plan: ParameterSearchPlan,
) -> str:
    """Build the final selection prompt from completed evaluations."""

    evaluation_payload = [
        parameter_evaluation_payload(evaluation) for evaluation in evaluations
    ]
    return (
        dedent(
            """
        Select a completed candidate for assay {from_assay} and exact
        cell-selection artifact {cell_selection}.

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
            cell_selection=cell_selection.artifactId,
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
                parameter_evaluation_payload(deps.evaluations[candidate_id])
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
            appearance is not evidence. Treat pcaSilhouette, macroF1, and
            weightedF1 only as PCA cluster-separability metrics; biological
            preservation exists only when biologicalPreservation is non-empty.
            Check all exact values before making ranking or trend claims, and keep
            narrative fields as plain prose without serialized JSON. Inside each
            assay report, return only
            model-owned selection, rationale, comparison, trade-off, limitation,
            evidence, and stop fields. Leave evaluations, selectedArtifacts,
            searchPlan, nested assayReports, integration fields, final graph fields,
            and runInfo out of the response because validation fills them from
            executor state.
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
                parameter_evaluation_payload(deps.evaluations[candidate_id])
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
            Select an option and explain it; Scarf attaches its exact graph,
            assay, and execution identities. Do not return those derived fields.
            """
        )
        .strip()
        .format()
    )


def final_graph_selection_prompt(
    *,
    report: ParameterTuningReport,
    integration_evaluations: Sequence[IntegrationCandidateEvaluation],
    marker_assay: str,
) -> str:
    """Build the selection prompt from executor-grounded final graph options."""

    options = _final_graph_options(report, integration_evaluations)
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
