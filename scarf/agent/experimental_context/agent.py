"""Experimental-context agent prompts and execution."""

import json
from collections.abc import Mapping, Sequence
from textwrap import dedent
from typing import TYPE_CHECKING, Any

from ...graph.feature_projection import graph_cell_selection
from ...metadata.selection import resolve_cell_aligned_artifact
from ...storage.refs import ArtifactRef
from ...utils.logging import logger
from .._deps import AGENT_INSTALL_HINT
from ..config import AgentRunConfig
from ..config.agent_exec import run_agent_sync
from ..tools import artifact_reference, core_artifact_reference
from ..types import StageStatus
from .characterization import _SelectionBoundCells, characterize_covariates
from .contracts import (
    CellQcPlan,
    ExperimentalContextDecision,
    ExperimentalContextDependencies,
    ExperimentalContextResult,
    NamedArtifactSource,
)
from .qc_evidence import (
    _derive_missing_percentage_artifacts,
    _hto_artifact_map,
    _qc_driver,
    _source_ref,
)
from .tools import (
    _prepare_experimental_context_tool,
    analyze_experimental_design,
    capture_repair_inputs,
    compact_context_evidence,
    contrast_plans_from_characterization,
    inspect_cell_covariates,
    inspect_context_evidence,
    model_evidence_tool,
    restore_context_evidence,
    score_current_representation,
)
from .validation import (
    failed_experimental_context_result,
    validate_experimental_context,
)

if TYPE_CHECKING:
    from ...datastore.pipeline_run import PipelineRun

try:
    from pydantic_ai import Tool, UnexpectedModelBehavior
except ImportError as exc:
    raise ImportError(AGENT_INSTALL_HINT) from exc


class ExperimentalContextAgent:
    """A narrow agent for study design and batch-correction planning."""

    def __init__(
        self,
        model: Any,
        *,
        config: AgentRunConfig | None = None,
    ) -> None:
        self.model = model
        self.config = (config or AgentRunConfig()).with_limits(
            request_limit=10,
            tool_call_limit=10,
            output_token_limit=32768,
            timeout_seconds=600.0,
        )
        self.system_prompt = (
            dedent(
                """
            You are Scarf's Experimental Context Agent. Work only through the
            provided read-only tools and return the structured decision schema.

            Call inspect_cell_covariates exactly once unless its committed evidence
            is already supplied on resume. Then call
            analyze_experimental_design with all explicit domains, all biological
            coefficients, every unit of inference, and the complete exact batch
            column set. Nominate up to eight comparisons that explain the study
            objective: single variables, two-column joint effects, or associations
            within categorical strata. Each comparison must have a distinct
            response and either one or two explanatoryColumns with conditionedOn
            null, or exactly one explanatory column and one distinct categorical
            conditionedOn column. These are the at most three measured columns;
            observationUnit and independentUnit are separate unit fields and do
            not count toward that limit. Never repeat response among explanatory
            or conditioning columns. Never append unit identifiers to explanatory
            columns just to identify replication. Two explanatory columns plus a
            conditioning column are unsupported. One response, one explanatory
            column and one conditioning column are supported for descriptive
            design coverage; association support depends on the measured units.
            Separate simpler comparisons do not establish an unsupported joint
            conditional finding. If a tool rejects a proposal, correct
            the named fields while preserving its scientific question or record
            the unsupported requirement explicitly. You may make one
            follow-up call with at most four new or revised comparisons after
            reading the first evidence. Never split the exact batch-column set.
            A donor can carry biological variation and also be the explicitly
            declared independent unit; these roles do not conflict. Keep its
            biological protection when specifying its unit role. Do not replace
            independent donors with cells or samples to obtain a supported result.
            Protect objective-relevant pairs of categorical biological variables
            with protectCombination. The tools report unsupported designs,
            missingness, replication, and sparse strata; do not turn these into
            negative findings. Explain unsupported requested comparisons in the
            final rationale; a proposed comparison is not a completed analysis.
            Set each proposal's purpose to designCoverage for measured counts,
            crossing, replication, or pairing; association when an association
            coefficient is essential; effectEstimation for an explicitly requested
            biological effect, which this workflow cannot deliver. Copy an exact
            objectiveQuote from the supplied study text and mark explicit objective
            questions essential. Do not downgrade an essential association or effect
            question to descriptive coverage to obtain completion. The returned
            evidenceRequirements and evidenceCoverage enforce this distinction.
            For repeated donors and incomplete pairing, inspect descriptiveDesign:
            it retains observation counts, distinct donors, group support and paired
            coverage without collapsing a donor to its first condition. A computed
            descriptiveDesign answers a designCoverage question even if the separate
            association method is unsupported. Preserve that association limitation
            in the rationale; do not ask the user to resolve an optional association
            before continuing descriptive population discovery. Unsupported
            association methods do not establish non-identifiability. Only measured
            rank and estimability for the exact tested design support that claim.
            Use the single follow-up round to resolve missing design evidence.
            If essential evidence remains unsupported, ask for clarification or
            abstain. Optional questions must remain explicit limitations.
            Continuous conditioning and expression hypothesis
            testing are unsupported. You may call
            score_current_representation at most once when an exact supplied graph
            can add evidence. Pass batch_columns as a JSON array, including a
            singleton. When the study explicitly identifies physical capture, pass
            capture_proposal in analyze_experimental_design, preferably in its first
            call. This existing tool validates capture provenance; no separate
            capture tool or additional user confirmation is required for an explicit
            supported study statement. Capture proposals must name an exact observed
            column and quote the study statement identifying it as a physical capture.
            An optional reference pool also needs an exact quote identifying the
            observed reference captures. Sample uniqueness is not capture proof.
            Leave unresolved capture provenance explicit. Validated tools own
            capture identities and protected combinations; do not copy them into
            the final decision. Nominate any new combination through a design tool.
            If capture was omitted from two completed design rounds, the same tool
            permits only a capture provenance repair with identical domains,
            coefficients, units and batch columns, and no proposals. This does not
            authorize another design comparison round.

            The tools return bounded cell-QC profiles projected against the exact
            shared cell selection. Do not choose a profile or return cellQc.
            Summaries retain adverse findings, missingness, protected group loss,
            replication and correction constraints. Use inspect_context_evidence
            to inspect one exact saved policy/capture or design record when its
            details are needed. Copy the supplied section and record_id from
            confounding details; do not construct an evidence ID for this lookup.
            Reuse returned details instead of requesting the same record again.
            Omitted donor examples and detailed thresholds
            remain available; do not interpret their omission as passing evidence.
            A later audited checkpoint compares the Scarf default with eligible
            alternatives and selects one exact policy. Never author
            or alter numeric quality bounds. RNA is the preferred QC driver and
            ATAC is the fallback. ADT and HTO never drive automatic cell filtering.
            An exact HTO identity artifact may be used as grouping evidence. It is
            not a live metadata column and does not make HTO a QC driver.

            A batch column must be categorical and technical. Never use donor,
            sample, observation-unit, independent-unit, biological, cluster, or
            embedding columns as Harmony batch columns. A biological coefficient
            that is not estimable with the exact proposed batch columns makes
            correction unsafe. A sample or library identifier is not automatically
            technical. When no exact observed column is both categorical and
            technical, pass batch_columns=[] and recommend skipping Harmony. Every
            observation and independent unit must be an exact observed column name
            or null.
            LISI evaluates a representation; it does not identify which metadata
            column is a batch. Recommend evaluateHarmony, not application, because
            Parameter Tuning must compare exact uncorrected and corrected artifacts.
            Current Harmony preservation metrics support categorical biology.
            A continuous protected variable has unavailable preservation evidence.
            Keep this limitation explicit; it does not prove correction unnecessary.
            Later matched acceptance must reject unavailable required protection,
            and unresolved essential evidence requires input or abstention.

            Cite only evidenceIds returned by tools. Ask for input when study
            design cannot be resolved. The study objective is authoritative: use
            it to identify protected biological variables and the intended unit
            of inference. Study-design explanations are in scope; biological
            expression hypothesis testing, effect inference and causal claims are not.
            Never propose Python, shell commands,
            direct Zarr access, or any datastore mutation. Every rationale and
            question must be plain prose. Never place serialized JSON, schema
            field names, or sibling output fields inside a narrative string.
            Return only fields defined by the structured output schema.
                """
            )
            .strip()
            .format()
        )

    def run(
        self,
        store: Any,
        *,
        study_context: str | None = None,
        study_objective: str | None = None,
        cell_selection: ArtifactRef | None = None,
        directions: Mapping[str, Any] | None = None,
        run: "PipelineRun | None" = None,
        neighbors: ArtifactRef | None = None,
        connectivity_map: ArtifactRef | None = None,
        quality_metric_artifacts: Sequence[NamedArtifactSource] = (),
        hto_identity_artifacts: Sequence[NamedArtifactSource] = (),
        qc_assay: str | None = None,
        checkpoint_read: Any = None,
        checkpoint_write: Any = None,
        on_attempt: Any = None,
        previous_context: ExperimentalContextResult | None = None,
    ) -> ExperimentalContextResult:
        """Inspect one datastore and return a validated experimental-context report."""
        study_context = (study_context or "").strip()
        study_objective = (study_objective or "").strip()
        direction_map = dict(directions or {})
        if run is not None:
            if (
                cell_selection is not None
                or neighbors is not None
                or connectivity_map is not None
            ):
                raise ValueError(
                    "run is mutually exclusive with explicit artifact inputs"
                )
            if getattr(run, "_owner", store) is not store:
                raise ValueError("run must be opened from this datastore")
            neighbors = run["neighbors"]
            cell_selection = run["analysis_cell_selection"]
            connectivity_map = (
                run["connectivity_map"] if "connectivity_map" in run else None
            )
        cell_selection = core_artifact_reference(cell_selection)
        neighbors = core_artifact_reference(neighbors)
        connectivity_map = core_artifact_reference(connectivity_map)
        if not isinstance(cell_selection, ArtifactRef) or (
            cell_selection.kind != "cell_selection"
            or cell_selection.scope != "datastore"
        ):
            raise TypeError(
                "cell_selection must be a datastore cell_selection ArtifactRef"
            )
        if neighbors is not None:
            if not isinstance(neighbors, ArtifactRef) or neighbors.kind != "neighbors":
                raise TypeError("neighbors must be a neighbors ArtifactRef")
            if graph_cell_selection(store.zw, neighbors) != cell_selection:
                raise ValueError(
                    "neighbors and metadata must use the same cell selection"
                )
        if connectivity_map is not None:
            if not isinstance(
                connectivity_map, ArtifactRef
            ) or connectivity_map.kind not in {
                "connectivity_map",
                "integrated_graph",
            }:
                raise TypeError(
                    "connectivity_map must be a connectivity graph ArtifactRef"
                )
            if graph_cell_selection(store.zw, connectivity_map) != cell_selection:
                raise ValueError(
                    "neighbors and connectivity_map must use the same cell selection"
                )
        quality_sources = _derive_missing_percentage_artifacts(
            store,
            cell_selection=cell_selection,
            driver=_qc_driver(store, qc_assay),
            quality_sources=quality_metric_artifacts,
        )
        hto_sources = list(hto_identity_artifacts)
        source_names: set[str] = set()
        for sources, expected_kind in (
            (quality_sources, "quality_metric"),
            (hto_sources, "hto_identity"),
        ):
            for source in sources:
                artifact = _source_ref(source, expected_kind=expected_kind)
                if source.name in source_names:
                    raise ValueError(
                        "Experimental Context artifact source names must be unique"
                    )
                source_names.add(source.name)
                resolve_cell_aligned_artifact(
                    store.zw,
                    artifact,
                    cell_selection=cell_selection,
                    expected_kind=expected_kind,
                )
        directed_qc = direction_map.get("cellQc")
        directed_qc_map = dict(directed_qc) if isinstance(directed_qc, Mapping) else {}
        if "cellKey" in directed_qc_map:
            raise ValueError(
                "cellQc.cellKey is unsupported; use the exact cell_selection input"
            )
        logger.info(
            "Experimental Context Agent started: "
            f"cellSelection={cell_selection.artifact_id}, "
            f"directions={len(direction_map)}, "
            f"qualityMetrics={len(quality_sources)}, "
            f"htoIdentities={len(hto_sources)}, "
            f"studyContextProvided={bool(study_context)}, "
            f"studyObjectiveProvided={bool(study_objective)}"
        )
        deps = ExperimentalContextDependencies(
            store=store,
            qcAssay=qc_assay,
            cells=_SelectionBoundCells(
                store.zw,
                store.cells,
                cell_selection,
                artifacts={
                    source.name: _source_ref(
                        source,
                        expected_kind="hto_identity",
                    )
                    for source in hto_sources
                },
            ),
            neighbors=neighbors,
            connectivityMap=connectivity_map,
            cellSelection=cell_selection,
            studyContext=study_context,
            studyObjective=study_objective,
            directions=direction_map,
            qualityMetricArtifacts=quality_sources,
            htoIdentityArtifacts=hto_sources,
            checkpointRead=checkpoint_read,
            checkpointWrite=checkpoint_write,
        )
        if checkpoint_read is not None:
            saved_result = checkpoint_read("result")
            if saved_result is not None:
                report = ExperimentalContextResult.model_validate(
                    saved_result["report"]
                )
                if report.cellSelection != artifact_reference(cell_selection):
                    raise ValueError(
                        "Committed context decision has a different cell selection"
                    )
                return report
        restored = restore_context_evidence(deps)
        if not restored and previous_context is not None:
            # A completed older stage remains immutable. Its measurements seed
            # an explicitly requested evidence revision under new stage inputs.
            deps.characterization = previous_context.characterization
            deps.comparisons = list(previous_context.characterization.comparisons)
            deps.captureProposal = previous_context.characterization.captureProvenance
            deps.protectedCombinations = list(
                previous_context.decision.protectedCombinations
            )
            deps.qcProfiles = {
                item.profileId: item for item in previous_context.qcProfiles
            }
            deps.qcMetricSources = list(previous_context.qcMetricSources)
            deps.qcSourceConcordance = list(previous_context.qcSourceConcordance)
            deps.batchSafety = {
                item.evidenceId: item for item in previous_context.batchSafety
            }
            deps.contrastPlans = {
                item.coefficient: item for item in previous_context.contrastPlans
            }
            deps.evidenceIds.update(previous_context.decision.evidenceIds)
            deps.evidenceIds.update(
                item.evidenceId for item in previous_context.batchSafety
            )
            deps.toolCalls = ["inspect_cell_covariates", "analyze_experimental_design"]
            deps.designRounds = min(
                2,
                max(
                    1,
                    sum(
                        item.toolName == "analyze_experimental_design"
                        for item in previous_context.runInfo.toolCalls
                    ),
                ),
            )
            restored = True
        user_prompt = (
            dedent(
                """
                Characterize this experiment's metadata and decide whether Harmony
                should be evaluated. Return cell-QC candidates as tool evidence;
                do not return cellQc; the later filtering checkpoint owns it.

                Study context: {study_context}
                Study objective: {study_objective}
                Exact cell-selection artifact: {cell_selection}
                Exact quality-metric artifacts: {quality_metrics}
                Exact HTO identity artifacts: {hto_identities}
                Caller directions: {directions}
                """
            )
            .strip()
            .format(
                study_context=study_context or "not provided",
                study_objective=study_objective or "not provided",
                cell_selection=cell_selection.artifact_id,
                quality_metrics=json.dumps(
                    [source.model_dump(mode="json") for source in quality_sources],
                    sort_keys=True,
                ),
                hto_identities=json.dumps(
                    [source.model_dump(mode="json") for source in hto_sources],
                    sort_keys=True,
                ),
                directions=json.dumps(direction_map, sort_keys=True, default=str),
            )
        )
        if restored:
            from .contracts import CovariateEvidence
            from .requirements import requested_design_questions

            assert deps.characterization is not None

            user_prompt += (
                "\nCommitted evidence already measured; do not repeat completed tools:\n"
                + json.dumps(
                    compact_context_evidence(
                        CovariateEvidence(
                            characterization=deps.characterization,
                            batchSafety=list(deps.batchSafety.values()),
                            qcProfiles=list(deps.qcProfiles.values()),
                            qcMetricSources=deps.qcMetricSources,
                            qcSourceConcordance=deps.qcSourceConcordance,
                            contrastPlans=list(deps.contrastPlans.values()),
                            evidenceIds=sorted(deps.evidenceIds),
                        )
                    ),
                    sort_keys=True,
                )
                + f"\nCompleted design rounds: {deps.designRounds} of 2."
            )
            user_prompt += (
                "\nExplicit requested questions requiring matched evidence: "
                + json.dumps(
                    requested_design_questions(
                        study_context,
                        study_objective,
                        [row["name"] for row in deps.characterization.columns],
                    )
                )
            )
            repair_inputs = capture_repair_inputs(deps)
            if repair_inputs is not None:
                user_prompt += (
                    "\nCapture provenance can be repaired with these exact saved "
                    "analyze_experimental_design inputs. Add only capture_proposal "
                    "with an exact supporting study quote: "
                    + json.dumps(repair_inputs, sort_keys=True)
                )
        if previous_context is not None and previous_context.status == "needsInput":
            from .requirements import objective_evidence

            requirements, coverage = (
                objective_evidence(
                    study_context=study_context,
                    study_objective=study_objective,
                    experimental_result=previous_context.model_copy(
                        update={
                            "characterization": deps.characterization,
                            "batchSafety": list(deps.batchSafety.values()),
                        }
                    ),
                )
                if study_objective
                else ([], [])
            )
            user_prompt += (
                "\nCurrent objective requirements and their measured support: "
                + json.dumps(
                    {
                        "evidenceRequirements": [
                            item.model_dump(mode="json") for item in requirements
                        ],
                        "evidenceCoverage": [
                            item.model_dump(mode="json") for item in coverage
                        ],
                    }
                )
                + "\nThe previous interpretation stopped with these unresolved questions: "
                + json.dumps(previous_context.decision.needsInput)
                + "\nReassess each against the original study text and committed evidence. "
                "Use the existing design tool for an omitted capture proposal. "
                "Distinguish computed descriptive support from unavailable association "
                "methods. Keep genuinely essential unresolved questions; do not copy "
                "a previous blocker when supplied provenance or measured evidence answers it."
            )
        try:
            execution = run_agent_sync(
                model=self.model,
                output_type=ExperimentalContextDecision,
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
                tools=(
                    Tool(
                        model_evidence_tool(inspect_cell_covariates),
                        prepare=_prepare_experimental_context_tool,
                        sequential=self.config.sequentialTools,
                        timeout=self.config.timeoutSeconds,
                    ),
                    Tool(
                        model_evidence_tool(analyze_experimental_design),
                        max_retries=3,
                        prepare=_prepare_experimental_context_tool,
                        sequential=self.config.sequentialTools,
                        timeout=self.config.timeoutSeconds,
                    ),
                    Tool(
                        inspect_context_evidence,
                        sequential=self.config.sequentialTools,
                        timeout=self.config.timeoutSeconds,
                    ),
                    Tool(
                        score_current_representation,
                        prepare=_prepare_experimental_context_tool,
                        sequential=self.config.sequentialTools,
                        timeout=self.config.timeoutSeconds,
                    ),
                ),
                deps_type=ExperimentalContextDependencies,
                deps=deps,
                config=self.config,
                name="experimental_context",
                on_attempt=on_attempt,
                output_validator=lambda decision: validate_experimental_context(
                    decision,
                    deps,
                ),
            )
        except UnexpectedModelBehavior as exc:
            model_name = getattr(self.model, "model_name", type(self.model).__name__)
            return failed_experimental_context_result(
                deps,
                error=exc,
                model_name=str(model_name),
            )
        decision = ExperimentalContextDecision.model_validate(execution.output)
        run_info = execution.runInfo
        characterization = deps.characterization
        if characterization is None:
            characterization = characterize_covariates(
                store,
                cellSelection=cell_selection,
                studyContext=(f"{study_context}\nStudy objective: {study_objective}"),
                model=None,
                directions=direction_map,
                groupingArtifacts=_hto_artifact_map(deps),
            )
        if characterization.status == "failed":
            status: StageStatus = "failed"
        elif decision.needsInput or decision.batchCorrection.action == "needsInput":
            status = "needsInput"
        else:
            status = "done"
        logger.info(
            "Experimental Context Agent completed: "
            f"status={status}, qcProfiles={len(deps.qcProfiles)}, "
            f"batchCorrection={decision.batchCorrection.action}, "
            f"coefficients={len(decision.coefficientsOfInterest)}, "
            f"toolCalls={len(deps.toolCalls)}, evidence={len(deps.evidenceIds)}"
        )
        contrast_plans = list(deps.contrastPlans.values())
        if not contrast_plans:
            contrast_plans = contrast_plans_from_characterization(characterization)
        report = ExperimentalContextResult(
            status=status,
            decision=decision,
            characterization=characterization,
            cellSelection=artifact_reference(cell_selection),
            cellQc=CellQcPlan.get_blank(),
            qcProfiles=list(deps.qcProfiles.values()),
            qcMetricSources=deps.qcMetricSources,
            qcSourceConcordance=deps.qcSourceConcordance,
            contrastPlans=contrast_plans,
            qualityMetricArtifacts=deps.qualityMetricArtifacts,
            htoIdentityColumns=deps.htoIdentityColumns,
            htoIdentityArtifacts=deps.htoIdentityArtifacts,
            batchSafety=list(deps.batchSafety.values()),
            currentRepresentation=deps.currentRepresentation,
            notes=[
                *characterization.notes,
                *decision.needsInput,
                *(
                    [
                        "Matched preservation evidence is unavailable for continuous variables: "
                        + ", ".join(decision.unsupportedProtection)
                        + ". A safe design does not by itself authorize correction."
                    ]
                    if decision.unsupportedProtection
                    else []
                ),
            ],
            runInfo=run_info,
        )
        if checkpoint_write is not None:
            checkpoint_write("result", {"report": report.model_dump(mode="json")})
        return report
