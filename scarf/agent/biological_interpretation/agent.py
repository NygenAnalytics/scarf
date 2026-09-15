"""Biological interpretation prompt and agent runner."""

from textwrap import dedent
from typing import Any

from ...storage.refs import ArtifactRef
from ...utils.logging import logger
from ..config import AgentRunConfig
from ..config.agent_exec import run_agent_sync
from ..tools import artifact_reference
from ..types import (
    ArtifactReferenceModel,
    ExperimentalBiologyHandoff,
    TuningBiologyHandoff,
)
from .contracts import (
    BiologicalContext,
    BiologicalInterpretationDependencies,
    BiologicalInterpretationReport,
)
from .tools import inspect_cluster_composition, inspect_cluster_markers_batch
from .validation import (
    _prepare_biological_interpretation_dependencies,
    _prepare_biological_interpretation_tool,
    fallback_biological_interpretation_report,
    validate_biological_interpretation_report,
)

try:
    from pydantic_ai import Tool, UnexpectedModelBehavior, UsageLimitExceeded
except ImportError as exc:
    from .._deps import AGENT_INSTALL_HINT

    raise ImportError(AGENT_INSTALL_HINT) from exc


_SYSTEM_PROMPT = dedent(
    """
        You are Scarf's Biological Interpretation Agent. Use only the supplied
        tools and caller context. Call inspect_cluster_composition exactly once.
        Then select every cluster you intend to interpret and call
        inspect_cluster_markers_batch exactly once with all selected cluster IDs.
        Tool calls execute Scarf operations, so wait for their results before
        drawing conclusions. Do not split marker inspection across calls.

        This API does not provide trusted per-cluster identities. Treat every cell
        identity as a hypothesis and always set identityIsHypothesis=true. Caller
        cell-type references are context, not assignments to clusters. Prefer
        proposedIdentity="unresolved" when the returned markers do not ground a
        specific hypothesis. Do not invent genes, cell types, statistics, artifact
        identifiers, or evidence identifiers. Cite only evidenceIds returned by
        tools. For each cluster interpretation, copy the exact non-empty marker
        evidenceId returned for that cluster into its evidenceIds. Do not interpret
        a cluster whose marker evidenceId is empty. Cluster abundance summaries are
        descriptive, not tests of significance or causal effects. Treatment observations must
        compare two returned independent-unit condition summaries for the same
        cluster. Independent units may occur in more than one condition in paired
        or repeated-measure designs. Return treatmentObservations empty unless the
        exact experimental handoff confirms independent-unit aggregation, a
        between-unit coefficient, an estimable coefficient, and at least two
        independent units in each cited condition.
        Marker p-values describe cluster-versus-rest marker specificity, not
        condition effects. Keep treatment content out of cluster identity
        interpretations. Never return status=failed for uncertainty or weak
        evidence. Return needsInput with one concrete question instead. Recommend a
        named follow-up operation when replication, a covariate, or an exact
        artifact is missing. Do not write exploratory code, use a shell, access
        files, or call arbitrary Scarf methods. Return only fields defined by the
        structured output schema.
    """
).strip()


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
            request_limit=8,
            tool_call_limit=5,
            output_token_limit=32768,
            timeout_seconds=600.0,
        )

    def run(
        self,
        store: Any,
        *,
        cluster: ArtifactRef | ArtifactReferenceModel | None = None,
        biological_context: BiologicalContext | None = None,
        from_assay: str | None = None,
        graph_assay: str | None = None,
        marker_assay_type: str | None = None,
        sample_column: str | None = None,
        condition_column: str | None = None,
        tuning_handoff: TuningBiologyHandoff | None = None,
        experimental_handoff: ExperimentalBiologyHandoff | None = None,
        marker: ArtifactRef | ArtifactReferenceModel | None = None,
        marker_features: ArtifactRef | ArtifactReferenceModel | None = None,
        allow_marker_search: bool = False,
        max_clusters: int = 12,
        max_markers: int = 10,
        marker_min_score: float = 0.25,
        marker_min_fraction: float = 0.2,
    ) -> BiologicalInterpretationReport:
        """Interpret cluster results while exposing only bounded tools to the model."""
        deps = _prepare_biological_interpretation_dependencies(
            store,
            cluster=cluster,
            from_assay=from_assay,
            graph_assay=graph_assay,
            marker_assay_type=marker_assay_type,
            sample_column=sample_column,
            condition_column=condition_column,
            tuning_handoff=tuning_handoff,
            experimental_handoff=experimental_handoff,
            marker=marker,
            marker_features=marker_features,
            allow_marker_search=allow_marker_search,
            max_clusters=max_clusters,
            max_markers=max_markers,
            marker_min_score=marker_min_score,
            marker_min_fraction=marker_min_fraction,
        )
        logger.info(
            f"Starting biological interpretation: "
            f"cluster_artifact={deps.cluster.artifact_id!r}, "
            f"graph_assay={deps.graphAssay!r}, marker_assay={deps.markerAssay!r}, "
            f"marker_artifact_supplied={deps.marker is not None}, "
            f"marker_search_authorized={deps.allowMarkerSearch}"
        )
        context = biological_context or BiologicalContext()
        cluster_artifact = artifact_reference(deps.cluster)
        marker_state = "provided" if deps.marker is not None else "not provided"
        treatment_eligible = bool(
            experimental_handoff is not None
            and experimental_handoff.conditionColumn
            and experimental_handoff.independentUnit
            and experimental_handoff.coefficientScope == "betweenUnit"
            and experimental_handoff.estimability.get("status") == "ok"
            and experimental_handoff.estimability.get("coefficientEstimable") is True
        )
        user_prompt = (
            dedent(
                """
                Review the exact cluster artifact {cluster_artifact} over
                cell-selection artifact {cell_selection}. The graph owner is
                {graph_assay}; markers are resolved from {marker_assay}. The exact
                marker artifact is {marker_state}; creating a marker artifact is
                authorized={allow_marker_search}. Review no more
                than {max_clusters} clusters. The tool returns no more than
                {max_markers} markers per cluster. Treatment observations are
                eligible from the supplied design={treatment_eligible}.

                Caller biological context:
                {biological_context}

                Experimental design context:
                {experimental_context}

                Call inspect_cluster_composition once. Copy every returned cluster
                ID exactly and send the complete unique list in one
                inspect_cluster_markers_batch call. Interpret only clusters with a
                non-empty returned marker evidenceId. Use proposedIdentity="unresolved"
                when markers do not ground a specific hypothesis, and always set
                identityIsHypothesis=true. Caller cell-type references are not
                cluster labels. If marker evidence is empty, return needsInput with
                one populated question. If treatment eligibility is false, return
                treatmentObservations=[]. Never return status=failed for biological
                uncertainty; use needsInput with a concrete question. Leave artifact
                fields null or copy only exact tool-returned values. Each tool is
                removed after it succeeds, so request every cluster in that one
                marker batch.
                """
            )
            .strip()
            .format(
                graph_assay=deps.graphAssay or "datastore integration",
                marker_assay=deps.markerAssay,
                cluster_artifact=cluster_artifact.model_dump_json(),
                cell_selection=deps.cellSelection.artifact_id,
                marker_state=marker_state,
                allow_marker_search=allow_marker_search,
                max_clusters=max_clusters,
                max_markers=max_markers,
                treatment_eligible=str(treatment_eligible).lower(),
                biological_context=context.model_dump_json(),
                experimental_context=(
                    experimental_handoff.model_dump_json()
                    if experimental_handoff is not None
                    else "not provided"
                ),
            )
        )
        logger.info(
            f"Requesting biological interpretation for at most "
            f"{deps.maxClusters} clusters"
        )
        try:
            execution = run_agent_sync(
                model=self.model,
                output_type=BiologicalInterpretationReport,
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                tools=(
                    Tool(
                        inspect_cluster_composition,
                        prepare=_prepare_biological_interpretation_tool,
                        sequential=self.config.sequentialTools,
                        timeout=self.config.timeoutSeconds,
                    ),
                    Tool(
                        inspect_cluster_markers_batch,
                        max_retries=1,
                        prepare=_prepare_biological_interpretation_tool,
                        sequential=self.config.sequentialTools,
                        timeout=self.config.timeoutSeconds,
                    ),
                ),
                deps_type=BiologicalInterpretationDependencies,
                deps=deps,
                config=self.config,
                name="biological_interpretation",
                output_validator=lambda report: (
                    validate_biological_interpretation_report(
                        report,
                        deps,
                    )
                ),
            )
        except (UnexpectedModelBehavior, UsageLimitExceeded) as exc:
            if not deps.clusterValues:
                raise
            model_name = getattr(self.model, "model_name", type(self.model).__name__)
            return fallback_biological_interpretation_report(
                deps,
                error=exc,
                model_name=str(model_name),
            )
        report = BiologicalInterpretationReport.model_validate(execution.output)
        report = validate_biological_interpretation_report(report, deps)
        report.runInfo = execution.runInfo
        logger.info(
            f"Completed biological interpretation: status={report.status}, "
            f"interpreted_clusters={len(report.clusterInterpretations)}, "
            f"treatment_observations={len(report.treatmentObservations)}, "
            f"follow_ups={len(report.followUps)}, tool_calls={len(deps.toolCalls)}"
        )
        return report
