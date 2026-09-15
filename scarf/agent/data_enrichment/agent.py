"""Data enrichment prompt and agent runner."""

from collections.abc import Callable, Sequence
from pathlib import Path
from textwrap import dedent
from typing import Any

from ...utils.logging import logger
from .._deps import AGENT_INSTALL_HINT
from ..config import AgentRunConfig
from ..config.agent_exec import run_agent_sync
from ..types import AgentRunInfo
from .contracts import (
    DataEnrichmentContext,
    DataEnrichmentDependencies,
    DataEnrichmentReport,
)
from .tools import (
    _prepare_data_enrichment_tool,
    find_present_features_batch,
    inspect_assay_features_batch,
)
from .validation import (
    _SUPPORTED_SPECIES,
    failed_data_enrichment_report,
    validate_data_enrichment_report,
)

try:
    from pydantic_ai import Tool, UnexpectedModelBehavior, UsageLimitExceeded
except ImportError as exc:
    raise ImportError(AGENT_INSTALL_HINT) from exc

_SYSTEM_PROMPT = (
    dedent(
        """
        You are Scarf's Data Enrichment Agent. Work only through the supplied
        read-only tools. Inspect every requested assay before making a decision.
        Use gene identifiers and names together with the supplied organism hint,
        tissue references, cell-type references, and experimental details when
        species evidence is ambiguous. Supported species keys are: {supported_species}.

        Call inspect_assay_features_batch once for all requested assays. Never
        invent a feature. If individual features are needed, collect all proposed
        names across assays and call find_present_features_batch once before
        placing them in a policy. Do not call feature lookup when no individual
        feature decision is needed. Absent or ambiguous lookup results must never
        enter a policy. If inspection resolves a supported species, copy that exact
        species key. Use caller organism context only when inspection leaves the
        species unknown. Use excludeFamilies only to nominate one conditional
        representation-sensitivity bundle from observed families with
        defaultExclude=true. It is not an instruction to remove those families.
        Never nominate a family with defaultExclude=false.
        The defaultFeatureInventory is separate deterministic evidence for Scarf's
        exact default HVG blacklist. It is evidence only, not an automatic
        exclusion or a source of policy nominations. Keep marker eligibility
        broader than any graph-feature exclusion.

        Persisted assay types determine modality routes; never infer a route from
        an assay label. The validator fills assay type, modality eligibility, ADT
        controls, HTO tags, ATAC-coordinate status, inspections, tool calls, and
        report-level evidence. Do not return those derived fields.
        Treat Ensembl release misses as unresolved,
        not artificial. Mitochondrial, ribosomal, and histone families may be
        sensitivity candidates. Sex-linked and cell-cycle families are protected
        by default. Marker testing retains conditional biological families.

        Structure studyContextSummary using only verbatim spans from the supplied
        study paragraph, study objective, or exact caller references. Do not
        paraphrase, infer, or
        invent an organism, tissue, cell type, experiment, hypothesis, or analysis
        intent. Empty optional hint lists do not mean that the paragraph lacks
        those references. Select at most 12 objective-relevant spans per category,
        each at most 240 characters. These lists are excerpts, not an exhaustive
        replacement for the full context and objective that Scarf preserves and
        passes downstream. The validator binds the original paragraph and exact
        caller references. Return a bounded
        report with citations copied from tool or context evidence IDs.
        Do not write code, mutate the datastore, or request arbitrary Scarf calls.
        """
    )
    .strip()
    .format(supported_species=", ".join(sorted(_SUPPORTED_SPECIES)))
)


def _persisted_assay_types(store: Any, assays: Sequence[str]) -> dict[str, str]:
    """Read exact persisted assay types through the public datastore summary."""
    summary_method = getattr(store, "summary", None)
    if not callable(summary_method):
        return {}
    summary = summary_method()
    requested = set(assays)
    return {
        str(item.name): str(item.assay_type)
        for item in getattr(summary, "assays", ())
        if str(item.name) in requested
    }


class DataEnrichmentAgent:
    """A small read-only tool agent for feature and organism enrichment."""

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
        context: DataEnrichmentContext | None = None,
        assays: Sequence[str] | None = None,
        cache_dir: Path | str | None = None,
        allow_download: bool = False,
        on_attempt: Callable[[AgentRunInfo], None] | None = None,
    ) -> DataEnrichmentReport:
        """Run the bounded tool loop without mutating the supplied datastore."""
        available_assays = [str(value) for value in store.assay_names]
        selected_assays = (
            [str(value) for value in assays] if assays is not None else available_assays
        )
        unknown_assays = sorted(set(selected_assays) - set(available_assays))
        if unknown_assays:
            raise ValueError(f"unknown assays: {unknown_assays}")
        if not selected_assays:
            raise ValueError("at least one assay is required")

        logger.info(
            "Data Enrichment Agent started: "
            f"assays={len(selected_assays)}, allowDownload={allow_download}"
        )

        enrichment_context = context or DataEnrichmentContext.get_blank()
        evidence_ids: set[str] = set()
        if enrichment_context.studyContext:
            evidence_ids.add("context:study")
        if enrichment_context.studyObjective:
            evidence_ids.add("context:objective")
        if enrichment_context.organismHint:
            evidence_ids.add("context:organism")
        evidence_ids.update(
            f"context:tissue:{index}"
            for index, _value in enumerate(enrichment_context.tissueReferences)
        )
        evidence_ids.update(
            f"context:cellType:{index}"
            for index, _value in enumerate(enrichment_context.cellTypeReferences)
        )
        evidence_ids.update(
            f"context:experiment:{index}"
            for index, _value in enumerate(enrichment_context.experimentalDetails)
        )
        deps = DataEnrichmentDependencies(
            store=store,
            context=enrichment_context,
            assays=selected_assays,
            assayTypes=_persisted_assay_types(store, selected_assays),
            cacheDir=Path(cache_dir) if cache_dir is not None else None,
            allowDownload=allow_download,
            evidenceIds=evidence_ids,
        )
        user_prompt = (
            dedent(
                """
                Enrich the feature policy for assays: {assays}.
                Study context: {study_context}
                Study objective: {study_objective}
                Organism hint: {organism_hint}
                Tissue references: {tissue_references}
                Cell-type references: {cell_type_references}
                Experimental details: {experimental_details}

                Call inspect_assay_features_batch exactly once to inspect every
                assay together. If a policy needs individual features, collect all
                proposed names for all assays and call
                find_present_features_batch exactly once. Do not call a singular
                assay tool or split lookups across calls. A batched tool is removed
                after it succeeds, so use each call to request all required data.
                If no policy needs an individual feature, do not call feature lookup
                and keep excludeFeatures, protectFeatures, and artificialFeatures
                empty. Return exactly one policy for every requested assay. Copy a
                resolved inspection species exactly; otherwise use unknown unless
                exact caller context supports a species. Exclude only observed
                defaultExclude=true families and protect every observed
                defaultExclude=false family.
                Populate studyContextSummary only with exact verbatim spans from
                the paragraph or caller references. Empty optional hint fields do
                not erase references present in the paragraph. Before returning,
                select at most 12 verbatim references per category, each no more
                than 240 characters, prioritizing the objective. These bounded
                excerpts do not replace the preserved full context. Do not return
                inspections, modality-derived fields, exact controls and tags,
                toolCalls, runInfo, or report evidence; validation attaches them.
                """
            )
            .strip()
            .format(
                assays=", ".join(selected_assays),
                study_context=enrichment_context.studyContext or "not provided",
                study_objective=enrichment_context.studyObjective or "not provided",
                organism_hint=enrichment_context.organismHint or "not provided",
                tissue_references=", ".join(enrichment_context.tissueReferences)
                or "not provided",
                cell_type_references=", ".join(enrichment_context.cellTypeReferences)
                or "not provided",
                experimental_details=", ".join(enrichment_context.experimentalDetails)
                or "not provided",
            )
        )
        try:
            execution = run_agent_sync(
                model=self.model,
                output_type=DataEnrichmentReport,
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                tools=[
                    Tool(
                        inspect_assay_features_batch,
                        max_retries=1,
                        prepare=_prepare_data_enrichment_tool,
                        sequential=self.config.sequentialTools,
                        timeout=self.config.timeoutSeconds,
                    ),
                    Tool(
                        find_present_features_batch,
                        max_retries=1,
                        prepare=_prepare_data_enrichment_tool,
                        sequential=self.config.sequentialTools,
                        timeout=self.config.timeoutSeconds,
                    ),
                ],
                deps_type=DataEnrichmentDependencies,
                deps=deps,
                config=self.config,
                name="data_enrichment",
                on_attempt=on_attempt,
                output_validator=lambda report: validate_data_enrichment_report(
                    deps,
                    report,
                ),
            )
        except (UnexpectedModelBehavior, UsageLimitExceeded) as exc:
            model_name = getattr(self.model, "model_name", type(self.model).__name__)
            return failed_data_enrichment_report(
                deps, error=exc, model_name=str(model_name)
            )
        report = DataEnrichmentReport.model_validate(execution.output)
        report = validate_data_enrichment_report(deps, report)
        report.runInfo = execution.runInfo
        logger.info(
            "Data Enrichment Agent completed: "
            f"status={report.status}, policies={len(report.policies)}, "
            f"toolCalls={len(report.toolCalls)}, evidence={len(report.evidenceIds)}"
        )
        return report
