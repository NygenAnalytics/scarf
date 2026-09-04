"""Read-only feature and organism enrichment agent."""

import re
from collections.abc import Sequence
from pathlib import Path
from textwrap import dedent
from typing import Any, Literal

from ..features.gene_reference import species_registry
from ..utils.logging import logger
from .characterize_features import characterize_features
from .config import CONFIG, AgentRunConfig
from .config._deps import AGENT_INSTALL_HINT
from .config.agent_exec import run_agent_sync
from .tools import bounded_list
from .types import AgentDataModel, AgentRunInfo, StageStatus

try:
    from pydantic import ConfigDict, Field, model_validator
    from pydantic_ai import (
        ModelRetry,
        RunContext,
        Tool,
        UnexpectedModelBehavior,
        UsageLimitExceeded,
    )
    from pydantic_ai.tools import ToolDefinition
except ImportError as exc:
    raise ImportError(AGENT_INSTALL_HINT) from exc

__all__ = [
    "AdtControlEvidence",
    "AssayFeatureInspection",
    "AssayFeatureInspectionBatch",
    "AssayModalityEvidence",
    "AtacCoordinateEvidence",
    "DataEnrichmentAgent",
    "DataEnrichmentContext",
    "DataEnrichmentDependencies",
    "DataEnrichmentReport",
    "DataEnrichmentToolCall",
    "ExogenousFeatureEvidence",
    "FeatureFamilyEvidence",
    "FeatureLookupResult",
    "FeatureLookupBatch",
    "FeatureMatch",
    "FeatureReference",
    "FeatureSelectionPolicy",
    "HtoTagEvidence",
    "StudyContextSummary",
    "find_present_features",
    "find_present_features_batch",
    "inspect_assay_features",
    "inspect_assay_features_batch",
    "validate_data_enrichment_report",
]

_SUPPORTED_SPECIES = species_registry()
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
        species unknown. Exclude only observed families with defaultExclude=true,
        and never exclude a family with defaultExclude=false.

        Persisted assay types determine modality routes; never infer a route from
        an assay label. The validator fills assay type, modality eligibility, ADT
        controls, HTO tags, ATAC-coordinate status, inspections, tool calls, and
        report-level evidence. Leave those derived fields at their defaults instead
        of copying them into the output. Treat Ensembl release misses as unresolved,
        not artificial. Mitochondrial, ribosomal, and histone families may be
        exclusion candidates. Sex-linked and cell-cycle families are protected by
        default in this initial implementation.

        Structure studyContextSummary using only verbatim spans from the supplied
        study paragraph or exact caller references. Do not paraphrase, infer, or
        invent an organism, tissue, cell type, experiment, hypothesis, or analysis
        intent. Empty optional hint lists do not mean that the paragraph lacks
        those references. When a category is explicitly present in the paragraph,
        include its exact span in the corresponding summary list. The validator
        binds the original paragraph and exact caller references. Return a bounded
        report with citations copied from tool or context evidence IDs.
        Do not write code, mutate the datastore, or request arbitrary Scarf calls.
        """
    )
    .strip()
    .format(supported_species=", ".join(sorted(_SUPPORTED_SPECIES)))
)


class DataEnrichmentContext(AgentDataModel):
    """Study evidence that may help resolve organism and feature policy."""

    studyContext: str = ""
    organismHint: str = ""
    tissueReferences: list[str] = Field(default_factory=list)
    cellTypeReferences: list[str] = Field(default_factory=list)
    experimentalDetails: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "DataEnrichmentContext":
        return cls()

    @classmethod
    def get_example(cls) -> "DataEnrichmentContext":
        return cls(
            studyContext="Single-cell profiling of treated lung tissue",
            organismHint="human",
            tissueReferences=["lung"],
            cellTypeReferences=["alveolar macrophage", "T cell"],
            experimentalDetails=["CRISPR perturbation", "10x 3 prime RNA-seq"],
        )


class StudyContextSummary(AgentDataModel):
    """Verbatim, evidence-backed references extracted from the study context."""

    studyContext: str = ""
    organismReferences: list[str] = Field(default_factory=list)
    tissueReferences: list[str] = Field(default_factory=list)
    cellTypeReferences: list[str] = Field(default_factory=list)
    experimentalReferences: list[str] = Field(default_factory=list)
    hypothesisReferences: list[str] = Field(default_factory=list)
    analysisIntentReferences: list[str] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "StudyContextSummary":
        return cls()

    @classmethod
    def get_example(cls) -> "StudyContextSummary":
        return cls(
            studyContext=(
                "Single-cell profiling of treated human lung tests whether "
                "treatment changes alveolar macrophage states."
            ),
            organismReferences=["human"],
            tissueReferences=["lung"],
            cellTypeReferences=["alveolar macrophage"],
            experimentalReferences=["treated"],
            hypothesisReferences=["treatment changes alveolar macrophage states"],
            analysisIntentReferences=["Single-cell profiling"],
            evidenceIds=["context:study"],
        )


class AdtControlEvidence(AgentDataModel):
    """One exact observed ADT feature carrying an explicit control token."""

    featureId: str
    featureName: str
    matchedToken: Literal["control", "isotype"]
    evidenceId: str

    @classmethod
    def get_blank(cls) -> "AdtControlEvidence":
        return cls(
            featureId="",
            featureName="",
            matchedToken="control",
            evidenceId="",
        )

    @classmethod
    def get_example(cls) -> "AdtControlEvidence":
        return cls(
            featureId="Mouse-IgG1-Control",
            featureName="Mouse IgG1 isotype control",
            matchedToken="isotype",
            evidenceId="assay:ADT:adtControl:Mouse-IgG1-Control",
        )


class HtoTagEvidence(AgentDataModel):
    """One exact feature from an assay persisted with the HTO type."""

    featureId: str
    featureName: str
    evidenceId: str

    @classmethod
    def get_blank(cls) -> "HtoTagEvidence":
        return cls(featureId="", featureName="", evidenceId="")

    @classmethod
    def get_example(cls) -> "HtoTagEvidence":
        return cls(
            featureId="HTO-1",
            featureName="Sample tag 1",
            evidenceId="assay:HTO:htoTag:HTO-1",
        )


class AtacCoordinateEvidence(AgentDataModel):
    """Validation evidence for exact ATAC feature IDs as genomic intervals."""

    status: Literal["notApplicable", "valid", "partial", "invalid"] = "notApplicable"
    coordinateColumn: Literal["ids"] = "ids"
    coordinateFormat: str = "chrom:start-end"
    totalFeatures: int = 0
    validFeatures: int = 0
    invalidExamples: list[str] = Field(default_factory=list)
    validExamples: list[str] = Field(default_factory=list)
    genomeBuild: Literal["unknown"] = "unknown"
    evidenceId: str = ""

    @classmethod
    def get_blank(cls) -> "AtacCoordinateEvidence":
        return cls()

    @classmethod
    def get_example(cls) -> "AtacCoordinateEvidence":
        return cls(
            status="valid",
            totalFeatures=2,
            validFeatures=2,
            validExamples=["chr1:100-200", "chr2:300-450"],
            evidenceId="assay:ATAC:atacCoordinates",
        )


class AssayModalityEvidence(AgentDataModel):
    """Bounded deterministic routing evidence for one persisted assay type."""

    assayType: str = "Assay"
    modality: Literal["RNA", "ATAC", "ADT", "HTO", "unsupported"] = "unsupported"
    typeSource: Literal["persisted", "assayClass", "unknown"] = "unknown"
    graphEligible: bool = False
    markerEligible: bool = False
    demultiplexEligible: bool = False
    adtControls: list[AdtControlEvidence] = Field(default_factory=list)
    htoTags: list[HtoTagEvidence] = Field(default_factory=list)
    atacCoordinates: AtacCoordinateEvidence = Field(
        default_factory=AtacCoordinateEvidence.get_blank
    )
    totalObservedFeatures: int = 0
    reportedFeatures: int = 0
    truncated: bool = False
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "AssayModalityEvidence":
        return cls()

    @classmethod
    def get_example(cls) -> "AssayModalityEvidence":
        control = AdtControlEvidence.get_example()
        return cls(
            assayType="ADT",
            modality="ADT",
            typeSource="persisted",
            graphEligible=True,
            markerEligible=True,
            adtControls=[control],
            totalObservedFeatures=20,
            reportedFeatures=1,
            evidenceIds=["assay:ADT:modality", control.evidenceId],
        )


class FeatureFamilyEvidence(AgentDataModel):
    """One observed feature family from deterministic Scarf analysis."""

    family: str
    species: str = "unknown"
    method: str = ""
    count: int = 0
    examples: list[str] = Field(default_factory=list)
    defaultExclude: bool | None = None
    skipped: str | None = None
    catalogSuspect: str | None = None
    catalogSize: int | None = None
    catalogJoinRate: float | None = None
    catalogJoined: int | None = None
    evidenceId: str

    @classmethod
    def get_blank(cls) -> "FeatureFamilyEvidence":
        return cls(family="", evidenceId="")

    @classmethod
    def get_example(cls) -> "FeatureFamilyEvidence":
        return cls(
            family="mitochondrial",
            species="homo_sapiens",
            method="chromosome",
            count=2,
            examples=["MT-CO1", "MT-CYB"],
            defaultExclude=True,
            evidenceId="assay:RNA:family:mitochondrial",
        )


class ExogenousFeatureEvidence(AgentDataModel):
    """One bounded candidate for an artificial or exogenous feature."""

    featureId: str
    featureName: str
    score: int = 0
    classification: str = "unresolved"
    evidenceId: str

    @classmethod
    def get_blank(cls) -> "ExogenousFeatureEvidence":
        return cls(featureId="", featureName="", evidenceId="")

    @classmethod
    def get_example(cls) -> "ExogenousFeatureEvidence":
        return cls(
            featureId="ERCC-00002",
            featureName="ERCC-00002",
            score=4,
            classification="potentialExogenous",
            evidenceId="assay:RNA:exogenous:ERCC-00002",
        )


class AssayFeatureInspection(AgentDataModel):
    """Bounded read-only inspection returned to the model."""

    assay: str
    assayKind: str = ""
    identity: dict[str, Any] = Field(default_factory=dict)
    species: str = "unknown"
    speciesMethod: str | None = None
    speciesReason: str = ""
    families: list[FeatureFamilyEvidence] = Field(default_factory=list)
    exogenous: list[ExogenousFeatureEvidence] = Field(default_factory=list)
    modalityEvidence: AssayModalityEvidence = Field(
        default_factory=AssayModalityEvidence.get_blank
    )
    notes: list[str] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "AssayFeatureInspection":
        return cls(assay="")

    @classmethod
    def get_example(cls) -> "AssayFeatureInspection":
        family = FeatureFamilyEvidence.get_example()
        modality = AssayModalityEvidence(
            assayType="RNA",
            modality="RNA",
            typeSource="persisted",
            graphEligible=True,
            markerEligible=True,
            totalObservedFeatures=20_000,
            evidenceIds=["assay:RNA:modality"],
        )
        return cls(
            assay="RNA",
            assayKind="RNAassay",
            identity={"nFeatures": 20_000, "nDuplicateIds": 0},
            species="homo_sapiens",
            speciesMethod="ensemblPrefix",
            speciesReason="Most feature IDs carry the ENSG prefix",
            families=[family],
            modalityEvidence=modality,
            evidenceIds=[
                "assay:RNA:identity",
                "assay:RNA:species",
                family.evidenceId,
                *modality.evidenceIds,
            ],
        )


class AssayFeatureInspectionBatch(AgentDataModel):
    """All requested assay inspections returned by one model tool call."""

    inspections: list[AssayFeatureInspection] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "AssayFeatureInspectionBatch":
        return cls()

    @classmethod
    def get_example(cls) -> "AssayFeatureInspectionBatch":
        inspection = AssayFeatureInspection.get_example()
        return cls(
            inspections=[inspection],
            evidenceIds=list(inspection.evidenceIds),
        )


class FeatureReference(AgentDataModel):
    """An exact feature identifier and name observed in one assay."""

    featureId: str
    featureName: str

    @classmethod
    def get_blank(cls) -> "FeatureReference":
        return cls(featureId="", featureName="")

    @classmethod
    def get_example(cls) -> "FeatureReference":
        return cls(featureId="ENSG00000198727", featureName="MT-CYB")


class FeatureMatch(AgentDataModel):
    """Resolution of one proposed feature against an assay."""

    query: str
    status: Literal["present", "ambiguous", "absent"]
    matches: list[FeatureReference] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "FeatureMatch":
        return cls(query="", status="absent")

    @classmethod
    def get_example(cls) -> "FeatureMatch":
        return cls(
            query="MT-CYB",
            status="present",
            matches=[FeatureReference.get_example()],
            evidenceIds=["assay:RNA:feature:ENSG00000198727"],
        )


class FeatureLookupResult(AgentDataModel):
    """Bounded result from exact feature lookup."""

    assay: str
    results: list[FeatureMatch] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "FeatureLookupResult":
        return cls(assay="")

    @classmethod
    def get_example(cls) -> "FeatureLookupResult":
        match = FeatureMatch.get_example()
        return cls(
            assay="RNA",
            results=[match],
            evidenceIds=list(match.evidenceIds),
        )


class FeatureLookupBatch(AgentDataModel):
    """Exact feature lookups for every requested assay in one tool result."""

    lookups: list[FeatureLookupResult] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "FeatureLookupBatch":
        return cls()

    @classmethod
    def get_example(cls) -> "FeatureLookupBatch":
        lookup = FeatureLookupResult.get_example()
        return cls(lookups=[lookup], evidenceIds=list(lookup.evidenceIds))


class FeatureSelectionPolicy(AgentDataModel):
    """Grounded feature policy proposed for one assay."""

    assay: str
    species: str = "unknown"
    organismName: str = "unknown"
    speciesConfidence: Literal["high", "medium", "low", "unknown"] = "unknown"
    speciesRationale: str = ""
    excludeFamilies: list[str] = Field(default_factory=list)
    protectFamilies: list[str] = Field(default_factory=list)
    excludeFeatures: list[str] = Field(default_factory=list)
    protectFeatures: list[str] = Field(default_factory=list)
    artificialFeatures: list[str] = Field(default_factory=list)
    tissueReferences: list[str] = Field(default_factory=list)
    cellTypeReferences: list[str] = Field(default_factory=list)
    experimentalReferences: list[str] = Field(default_factory=list)
    assayType: str = "Assay"
    assayModality: Literal["RNA", "ATAC", "ADT", "HTO", "unsupported"] = "unsupported"
    graphEligible: bool = False
    markerEligible: bool = False
    demultiplexEligible: bool = False
    exactControlFeatures: list[FeatureReference] = Field(default_factory=list)
    exactTagFeatures: list[FeatureReference] = Field(default_factory=list)
    peakCoordinateStatus: Literal["notApplicable", "valid", "partial", "invalid"] = (
        "notApplicable"
    )
    rationale: str = ""
    evidenceIds: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_non_conflicting_policy(self) -> "FeatureSelectionPolicy":
        family_overlap = set(self.excludeFamilies) & set(self.protectFamilies)
        feature_overlap = set(self.excludeFeatures) & set(self.protectFeatures)
        if family_overlap:
            raise ValueError(
                "feature families cannot be both excluded and protected: "
                f"{sorted(family_overlap)}"
            )
        if feature_overlap:
            raise ValueError(
                "features cannot be both excluded and protected: "
                f"{sorted(feature_overlap)}"
            )
        return self

    @classmethod
    def get_blank(cls) -> "FeatureSelectionPolicy":
        return cls(assay="")

    @classmethod
    def get_example(cls) -> "FeatureSelectionPolicy":
        return cls(
            assay="RNA",
            species="homo_sapiens",
            organismName="human",
            speciesConfidence="high",
            speciesRationale="Gene IDs and study context agree",
            excludeFamilies=["mitochondrial", "ribosomal"],
            protectFamilies=["cellCycle", "sex"],
            artificialFeatures=["ERCC-00002"],
            tissueReferences=["lung"],
            cellTypeReferences=["alveolar macrophage"],
            experimentalReferences=["ERCC spike-in"],
            assayType="RNA",
            assayModality="RNA",
            graphEligible=True,
            markerEligible=True,
            rationale="Use technical families for feature-selection exclusions",
            evidenceIds=["assay:RNA:species", "assay:RNA:family:mitochondrial"],
        )


class DataEnrichmentToolCall(AgentDataModel):
    """Compact audit record for one read-only model tool call."""

    name: str
    assay: str
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "DataEnrichmentToolCall":
        return cls(name="", assay="")

    @classmethod
    def get_example(cls) -> "DataEnrichmentToolCall":
        return cls(
            name="inspect_assay_features",
            assay="RNA",
            evidenceIds=["assay:RNA:identity", "assay:RNA:species"],
        )


class DataEnrichmentReport(AgentDataModel):
    """Final grounded report from :class:`DataEnrichmentAgent`."""

    status: StageStatus
    policies: list[FeatureSelectionPolicy] = Field(default_factory=list)
    inspections: list[AssayFeatureInspection] = Field(default_factory=list)
    studyContextSummary: StudyContextSummary = Field(
        default_factory=StudyContextSummary.get_blank
    )
    unresolvedQuestions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)
    toolCalls: list[DataEnrichmentToolCall] = Field(default_factory=list)
    runInfo: AgentRunInfo = Field(default_factory=AgentRunInfo)

    @model_validator(mode="after")
    def validate_status(self) -> "DataEnrichmentReport":
        if self.status == "done" and not self.policies:
            raise ValueError("done reports require at least one feature policy")
        if self.status == "needsInput" and not self.unresolvedQuestions:
            raise ValueError("needsInput reports require an unresolved question")
        if self.status == "failed" and not self.limitations:
            raise ValueError("failed reports require a limitation")
        return self

    @classmethod
    def get_blank(cls) -> "DataEnrichmentReport":
        return cls(status="failed", limitations=["No agent result was produced"])

    @classmethod
    def get_example(cls) -> "DataEnrichmentReport":
        policy = FeatureSelectionPolicy.get_example()
        inspection = AssayFeatureInspection.get_example()
        return cls(
            status="done",
            policies=[policy],
            inspections=[inspection],
            studyContextSummary=StudyContextSummary.get_example(),
            evidenceIds=list(policy.evidenceIds),
            toolCalls=[DataEnrichmentToolCall.get_example()],
            runInfo=AgentRunInfo.get_example(),
        )


class DataEnrichmentDependencies(AgentDataModel):
    """Hidden runtime state supplied to read-only enrichment tools."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    store: Any = Field(default=None, exclude=True)
    context: DataEnrichmentContext = Field(
        default_factory=DataEnrichmentContext.get_blank
    )
    assays: list[str] = Field(default_factory=list)
    assayTypes: dict[str, str] = Field(default_factory=dict)
    cacheDir: Path | None = None
    allowDownload: bool = False
    evidenceIds: set[str] = Field(default_factory=set)
    inspections: dict[str, AssayFeatureInspection] = Field(default_factory=dict)
    confirmedFeatures: dict[str, set[str]] = Field(default_factory=dict)
    lookupBatch: FeatureLookupBatch | None = Field(default=None, exclude=True)
    lookupQueries: dict[str, list[str]] = Field(default_factory=dict, exclude=True)
    toolCalls: list[DataEnrichmentToolCall] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "DataEnrichmentDependencies":
        return cls()

    @classmethod
    def get_example(cls) -> "DataEnrichmentDependencies":
        return cls(
            context=DataEnrichmentContext.get_example(),
            assays=["RNA"],
            cacheDir=Path("/tmp/scarf-gene-reference"),
            allowDownload=False,
            evidenceIds={"context:organism", "context:tissue:0"},
        )


def _prepare_data_enrichment_tool(
    ctx: RunContext[DataEnrichmentDependencies],
    tool_definition: ToolDefinition,
) -> ToolDefinition | None:
    """Expose each batched enrichment tool only while its work is pending."""
    deps = ctx.deps
    completed_calls = {call.name for call in deps.toolCalls}
    inspection_complete = "inspect_assay_features_batch" in completed_calls

    if tool_definition.name == "inspect_assay_features_batch":
        return None if inspection_complete else tool_definition
    if tool_definition.name == "find_present_features_batch":
        if not inspection_complete or tool_definition.name in completed_calls:
            return None
        return tool_definition
    return tool_definition


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


def _assay_modality(
    assay_type: str | None,
    assay_kind: str,
) -> tuple[
    Literal["RNA", "ATAC", "ADT", "HTO", "unsupported"],
    str,
    Literal["persisted", "assayClass", "unknown"],
]:
    """Map persisted types to supported routes, with a mock-store class fallback."""
    if assay_type is not None:
        if assay_type == "RNA":
            return "RNA", assay_type, "persisted"
        if assay_type == "ATAC":
            return "ATAC", assay_type, "persisted"
        if assay_type == "ADT":
            return "ADT", assay_type, "persisted"
        if assay_type == "HTO":
            return "HTO", assay_type, "persisted"
        return "unsupported", assay_type, "persisted"
    if assay_kind == "RNAassay":
        return "RNA", assay_kind, "assayClass"
    if assay_kind == "ATACassay":
        return "ATAC", assay_kind, "assayClass"
    if assay_kind:
        return "unsupported", assay_kind, "assayClass"
    return "unsupported", "Assay", "unknown"


def _feature_tokens(*values: str) -> set[str]:
    """Return literal alphanumeric tokens without accepting generated patterns."""
    text = " ".join(values).casefold()
    normalized = "".join(
        character if character.isalnum() else " " for character in text
    )
    return set(normalized.split())


def _valid_peak_coordinate(value: str) -> bool:
    """Validate the documented ``chrom:start-end`` representation exactly."""
    chromosome, separator, interval = value.partition(":")
    if not separator or not chromosome:
        return False
    start_text, separator, end_text = interval.partition("-")
    if not separator or not start_text or not end_text:
        return False
    try:
        start = int(start_text)
        end = int(end_text)
    except ValueError:
        return False
    return start >= 0 and end > start


def _inspect_adt_features(
    assay_name: str,
    feature_rows: list[tuple[str, str]],
) -> list[AdtControlEvidence]:
    """Return control candidates from exact observed ADT features."""
    candidates: list[AdtControlEvidence] = []
    for feature_id, feature_name in feature_rows:
        tokens = _feature_tokens(feature_id, feature_name)
        matched_token: Literal["control", "isotype"] | None = None
        if "isotype" in tokens:
            matched_token = "isotype"
        elif "control" in tokens:
            matched_token = "control"
        if matched_token is None:
            continue
        candidates.append(
            AdtControlEvidence(
                featureId=feature_id,
                featureName=feature_name,
                matchedToken=matched_token,
                evidenceId=f"assay:{assay_name}:adtControl:{feature_id}",
            )
        )
    return candidates


def _inspect_hto_features(
    assay_name: str,
    feature_rows: list[tuple[str, str]],
) -> list[HtoTagEvidence]:
    """Return HTO tag evidence in exact observed order."""
    return [
        HtoTagEvidence(
            featureId=feature_id,
            featureName=feature_name,
            evidenceId=f"assay:{assay_name}:htoTag:{feature_id}",
        )
        for feature_id, feature_name in feature_rows
    ]


def _inspect_atac_features(
    assay_name: str,
    feature_ids: list[str],
) -> AtacCoordinateEvidence:
    """Validate exact observed ATAC coordinates without inferring a build."""
    valid_ids = [value for value in feature_ids if _valid_peak_coordinate(value)]
    invalid_ids = [value for value in feature_ids if not _valid_peak_coordinate(value)]
    if not feature_ids or not valid_ids:
        coordinate_status: Literal["valid", "partial", "invalid"] = "invalid"
    elif invalid_ids:
        coordinate_status = "partial"
    else:
        coordinate_status = "valid"
    return AtacCoordinateEvidence(
        status=coordinate_status,
        totalFeatures=len(feature_ids),
        validFeatures=len(valid_ids),
        validExamples=bounded_list(valid_ids, limit=5),
        invalidExamples=bounded_list(invalid_ids, limit=5),
        evidenceId=f"assay:{assay_name}:atacCoordinates",
    )


def _inspect_modality_features(
    *,
    assay_name: str,
    assay: Any,
    assay_type: str | None,
    assay_kind: str,
    identity: dict[str, Any],
) -> AssayModalityEvidence:
    """Build bounded modality evidence from exact observed feature metadata."""
    modality, resolved_type, type_source = _assay_modality(assay_type, assay_kind)
    modality_evidence_id = f"assay:{assay_name}:modality"
    total_features = int(identity.get("nFeatures", 0))
    evidence_ids = [modality_evidence_id]
    graph_eligible = modality in {"RNA", "ATAC", "ADT"}
    marker_eligible = graph_eligible

    adt_controls: list[AdtControlEvidence] = []
    hto_tags: list[HtoTagEvidence] = []
    atac_coordinates = AtacCoordinateEvidence.get_blank()
    reported_features = 0
    truncated = False

    if modality in {"ADT", "HTO", "ATAC"}:
        feature_ids = [str(value) for value in assay.feats.fetch_all("ids")]
        total_features = len(feature_ids)
    else:
        feature_ids = []

    if modality in {"ADT", "HTO"}:
        feature_names = [str(value) for value in assay.feats.fetch_all("names")]
        feature_rows = list(zip(feature_ids, feature_names, strict=True))
    else:
        feature_rows = []

    if modality == "ADT":
        control_candidates = _inspect_adt_features(assay_name, feature_rows)
        adt_controls = bounded_list(
            control_candidates,
            limit=CONFIG._MAX_FEATURE_QUERIES,
        )
        evidence_ids.extend(item.evidenceId for item in adt_controls)
        reported_features = len(adt_controls)
        truncated = len(control_candidates) > len(adt_controls)

    if modality == "HTO":
        limited_rows = bounded_list(feature_rows, limit=CONFIG._MAX_FEATURE_QUERIES)
        hto_tags = _inspect_hto_features(assay_name, limited_rows)
        evidence_ids.extend(item.evidenceId for item in hto_tags)
        reported_features = len(hto_tags)
        truncated = len(feature_rows) > len(limited_rows)

    if modality == "ATAC":
        atac_coordinates = _inspect_atac_features(
            assay_name,
            feature_ids,
        )
        evidence_ids.append(atac_coordinates.evidenceId)
        reported_features = len(atac_coordinates.validExamples) + len(
            atac_coordinates.invalidExamples
        )
        truncated = len(feature_ids) > reported_features

    return AssayModalityEvidence(
        assayType=resolved_type,
        modality=modality,
        typeSource=type_source,
        graphEligible=graph_eligible,
        markerEligible=marker_eligible,
        demultiplexEligible=modality == "HTO",
        adtControls=adt_controls,
        htoTags=hto_tags,
        atacCoordinates=atac_coordinates,
        totalObservedFeatures=total_features,
        reportedFeatures=reported_features,
        truncated=truncated,
        evidenceIds=evidence_ids,
    )


async def inspect_assay_features(
    ctx: RunContext[DataEnrichmentDependencies],
    assay_name: str,
) -> AssayFeatureInspection:
    """Inspect feature identity, species evidence, families, and exogenous cues."""
    deps = ctx.deps
    if deps.store is None:
        raise ModelRetry("The datastore is unavailable")
    if assay_name not in deps.assays:
        raise ModelRetry(
            f"assay_name must be one of the requested assays: {deps.assays}"
        )
    cached = deps.inspections.get(assay_name)
    if cached is not None:
        logger.debug(
            f"Data Enrichment reused cached inspection for assay {assay_name!r}"
        )
        return cached

    characterization = characterize_features(
        deps.store,
        studyContext=deps.context.studyContext,
        model=None,
        assays=[assay_name],
        cacheDir=deps.cacheDir,
        allowDownload=deps.allowDownload,
    )
    if characterization.status != "done" or not characterization.assays:
        logger.warning(f"Data Enrichment could not characterize assay {assay_name!r}")
        detail = "; ".join(characterization.notes) or "feature inspection failed"
        raise ModelRetry(detail)

    record = characterization.assays[0]
    family_evidence: list[FeatureFamilyEvidence] = []
    evidence_ids = [f"assay:{assay_name}:identity", f"assay:{assay_name}:species"]
    for family in record.get("families", []):
        family_name = str(family.get("family", ""))
        evidence_id = f"assay:{assay_name}:family:{family_name}"
        family_evidence.append(
            FeatureFamilyEvidence(
                family=family_name,
                species=str(family.get("species", record.get("species", "unknown"))),
                method=str(family.get("method", "")),
                count=int(family.get("count", 0)),
                examples=[str(value) for value in family.get("examples", [])],
                defaultExclude=family.get("defaultExclude"),
                skipped=family.get("skipped"),
                catalogSuspect=family.get("catalogSuspect"),
                catalogSize=family.get("catalogSize"),
                catalogJoinRate=family.get("catalogJoinRate"),
                catalogJoined=family.get("catalogJoined"),
                evidenceId=evidence_id,
            )
        )
        evidence_ids.append(evidence_id)

    exogenous_evidence: list[ExogenousFeatureEvidence] = []
    for item in record.get("exogenous", []):
        feature_id = str(item.get("id", ""))
        evidence_id = f"assay:{assay_name}:exogenous:{feature_id}"
        exogenous_evidence.append(
            ExogenousFeatureEvidence(
                featureId=feature_id,
                featureName=str(item.get("name", "")),
                score=int(item.get("score", 0)),
                classification=str(item.get("class", "unresolved")),
                evidenceId=evidence_id,
            )
        )
        evidence_ids.append(evidence_id)

    resolution = record.get("speciesResolution") or {}
    assay = deps.store.get_assay(assay_name)
    identity = dict(record.get("identity") or {})
    modality_evidence = _inspect_modality_features(
        assay_name=assay_name,
        assay=assay,
        assay_type=deps.assayTypes.get(assay_name),
        assay_kind=str(record.get("assayKind", "")),
        identity=identity,
    )
    evidence_ids.extend(modality_evidence.evidenceIds)
    inspection = AssayFeatureInspection(
        assay=assay_name,
        assayKind=str(record.get("assayKind", "")),
        identity=identity,
        species=str(record.get("species", "unknown")),
        speciesMethod=record.get("speciesMethod"),
        speciesReason=str(resolution.get("reason", "")),
        families=family_evidence,
        exogenous=exogenous_evidence,
        modalityEvidence=modality_evidence,
        notes=[str(value) for value in record.get("notes", [])],
        evidenceIds=evidence_ids,
    )
    deps.inspections[assay_name] = inspection
    deps.evidenceIds.update(evidence_ids)
    deps.toolCalls.append(
        DataEnrichmentToolCall(
            name="inspect_assay_features",
            assay=assay_name,
            evidenceIds=evidence_ids,
        )
    )
    logger.debug(
        "Data Enrichment inspected "
        f"assay={assay_name!r}, modality={modality_evidence.modality}, "
        f"species={inspection.species}, families={len(family_evidence)}, "
        f"exogenous={len(exogenous_evidence)}, evidence={len(evidence_ids)}"
    )
    return inspection


async def inspect_assay_features_batch(
    ctx: RunContext[DataEnrichmentDependencies],
) -> AssayFeatureInspectionBatch:
    """Inspect every requested assay and return one bounded tool result."""
    deps = ctx.deps
    if not deps.assays:
        raise ModelRetry("No assays were requested")
    if any(
        call.name == "inspect_assay_features_batch" for call in deps.toolCalls
    ) and all(assay_name in deps.inspections for assay_name in deps.assays):
        inspections = [deps.inspections[assay_name] for assay_name in deps.assays]
        evidence_ids = list(
            dict.fromkeys(
                evidence_id
                for inspection in inspections
                for evidence_id in inspection.evidenceIds
            )
        )
        logger.info("Data Enrichment reused the completed feature inspection batch")
        return AssayFeatureInspectionBatch(
            inspections=inspections,
            evidenceIds=evidence_ids,
        )
    logger.info(
        f"Data Enrichment feature inspection started for {len(deps.assays)} assays"
    )
    start = len(deps.toolCalls)
    try:
        inspections = [
            await inspect_assay_features(ctx, assay_name=assay_name)
            for assay_name in deps.assays
        ]
    except Exception:
        del deps.toolCalls[start:]
        raise
    del deps.toolCalls[start:]
    evidence_ids = list(
        dict.fromkeys(
            evidence_id
            for inspection in inspections
            for evidence_id in inspection.evidenceIds
        )
    )
    deps.toolCalls.append(
        DataEnrichmentToolCall(
            name="inspect_assay_features_batch",
            assay=",".join(deps.assays),
            evidenceIds=evidence_ids,
        )
    )
    supported_routes = sum(
        inspection.modalityEvidence.modality != "unsupported"
        for inspection in inspections
    )
    logger.info(
        "Data Enrichment feature inspection completed: "
        f"assays={len(inspections)}, supportedRoutes={supported_routes}, "
        f"evidence={len(evidence_ids)}"
    )
    return AssayFeatureInspectionBatch(
        inspections=inspections,
        evidenceIds=evidence_ids,
    )


async def find_present_features(
    ctx: RunContext[DataEnrichmentDependencies],
    assay_name: str,
    queries: list[str],
) -> FeatureLookupResult:
    """Resolve a bounded list of gene IDs or names against one exact assay."""
    deps = ctx.deps
    if deps.store is None:
        raise ModelRetry("The datastore is unavailable")
    if assay_name not in deps.assays:
        raise ModelRetry(
            f"assay_name must be one of the requested assays: {deps.assays}"
        )
    clean_queries = list(
        dict.fromkeys(value.strip() for value in queries if value.strip())
    )
    if not clean_queries or len(clean_queries) > CONFIG._MAX_FEATURE_QUERIES:
        raise ModelRetry(
            f"queries must contain between 1 and {CONFIG._MAX_FEATURE_QUERIES} values"
        )

    assay = deps.store.get_assay(assay_name)
    feature_ids = [str(value) for value in assay.feats.fetch_all("ids")]
    feature_names = [str(value) for value in assay.feats.fetch_all("names")]
    rows = list(zip(feature_ids, feature_names, strict=True))
    results: list[FeatureMatch] = []
    result_evidence_ids: list[str] = []
    confirmed = deps.confirmedFeatures.setdefault(assay_name, set())

    for query in clean_queries:
        exact = [row for row in rows if query in row]
        candidates = exact
        if not candidates:
            folded = query.casefold()
            candidates = [
                row
                for row in rows
                if folded == row[0].casefold() or folded == row[1].casefold()
            ]
        unique_candidates = bounded_list(
            dict.fromkeys(candidates),
            limit=10,
        )
        references = [
            FeatureReference(featureId=feature_id, featureName=feature_name)
            for feature_id, feature_name in unique_candidates
        ]
        evidence_ids = [
            f"assay:{assay_name}:feature:{reference.featureId}"
            for reference in references
        ]
        if len(references) == 1:
            status: Literal["present", "ambiguous", "absent"] = "present"
            confirmed.update({references[0].featureId, references[0].featureName})
            deps.evidenceIds.update(evidence_ids)
            result_evidence_ids.extend(evidence_ids)
        elif references:
            status = "ambiguous"
        else:
            status = "absent"
        results.append(
            FeatureMatch(
                query=query,
                status=status,
                matches=references,
                evidenceIds=evidence_ids if status == "present" else [],
            )
        )

    result = FeatureLookupResult(
        assay=assay_name,
        results=results,
        evidenceIds=list(dict.fromkeys(result_evidence_ids)),
    )
    deps.toolCalls.append(
        DataEnrichmentToolCall(
            name="find_present_features",
            assay=assay_name,
            evidenceIds=result.evidenceIds,
        )
    )
    return result


async def find_present_features_batch(
    ctx: RunContext[DataEnrichmentDependencies],
    queries_by_assay: dict[str, list[str]],
) -> FeatureLookupBatch:
    """Resolve all proposed individual features through one model tool call."""
    deps = ctx.deps
    unknown_assays = sorted(set(queries_by_assay) - set(deps.assays))
    if unknown_assays:
        raise ModelRetry(f"Unknown requested assays: {unknown_assays}")
    if not queries_by_assay:
        raise ModelRetry("queries_by_assay must contain at least one assay")
    clean_queries_by_assay = {
        assay_name: list(
            dict.fromkeys(value.strip() for value in queries if value.strip())
        )
        for assay_name, queries in queries_by_assay.items()
    }
    empty_assays = sorted(
        assay_name
        for assay_name, queries in clean_queries_by_assay.items()
        if not queries
    )
    if empty_assays:
        raise ModelRetry(f"Feature-query batches cannot be empty: {empty_assays}")
    query_count = sum(len(queries) for queries in clean_queries_by_assay.values())
    if query_count > CONFIG._MAX_FEATURE_QUERIES:
        raise ModelRetry(
            "The batch may contain at most "
            f"{CONFIG._MAX_FEATURE_QUERIES} feature queries in total"
        )
    if deps.lookupBatch is not None:
        if clean_queries_by_assay != deps.lookupQueries:
            raise ModelRetry(
                "Feature lookup already completed. Use only the returned lookup "
                "evidence and do not request a different batch."
            )
        logger.info("Data Enrichment reused the completed feature lookup batch")
        return deps.lookupBatch

    logger.info(
        "Data Enrichment feature lookup started: "
        f"assays={len(clean_queries_by_assay)}, queries={query_count}"
    )

    start = len(deps.toolCalls)
    lookups = [
        await find_present_features(
            ctx,
            assay_name=assay_name,
            queries=clean_queries_by_assay[assay_name],
        )
        for assay_name in deps.assays
        if assay_name in clean_queries_by_assay
    ]
    del deps.toolCalls[start:]
    evidence_ids = list(
        dict.fromkeys(
            evidence_id for lookup in lookups for evidence_id in lookup.evidenceIds
        )
    )
    deps.toolCalls.append(
        DataEnrichmentToolCall(
            name="find_present_features_batch",
            assay=",".join(queries_by_assay),
            evidenceIds=evidence_ids,
        )
    )
    result_counts = {"present": 0, "ambiguous": 0, "absent": 0}
    for lookup in lookups:
        for result in lookup.results:
            result_counts[result.status] += 1
    logger.info(
        "Data Enrichment feature lookup completed: "
        f"present={result_counts['present']}, "
        f"ambiguous={result_counts['ambiguous']}, "
        f"absent={result_counts['absent']}, evidence={len(evidence_ids)}"
    )
    batch = FeatureLookupBatch(lookups=lookups, evidenceIds=evidence_ids)
    deps.lookupBatch = batch
    deps.lookupQueries = clean_queries_by_assay
    return batch


def _ground_study_context_summary(
    context: DataEnrichmentContext,
    proposed: StudyContextSummary,
) -> StudyContextSummary:
    """Bind structured context references to exact caller text."""
    original_context = context.studyContext
    organism_references = [context.organismHint] if context.organismHint else []
    for species in _SUPPORTED_SPECIES.values():
        match = re.search(
            rf"\b{re.escape(species.label)}\b",
            original_context,
            flags=re.IGNORECASE,
        )
        if match is not None:
            organism_references.append(match.group(0))
    field_sources = {
        "organismReferences": organism_references,
        "tissueReferences": list(context.tissueReferences),
        "cellTypeReferences": list(context.cellTypeReferences),
        "experimentalReferences": list(context.experimentalDetails),
        "hypothesisReferences": [],
        "analysisIntentReferences": [],
    }
    grounded: dict[str, list[str]] = {}
    for field_name, supplied_values in field_sources.items():
        exact_supplied = [value.strip() for value in supplied_values if value.strip()]
        proposed_values = list(getattr(proposed, field_name))
        combined = list(
            dict.fromkeys(
                value.strip()
                for value in [*exact_supplied, *proposed_values]
                if value.strip()
            )
        )
        if len(combined) > 12:
            raise ValueError(
                f"studyContextSummary.{field_name} may contain at most 12 values"
            )
        supplied = set(exact_supplied)
        invalid = [
            value
            for value in combined
            if value not in supplied and value not in original_context
        ]
        if invalid:
            raise ValueError(
                f"Study-context references must be verbatim caller text: {invalid}"
            )
        oversized = [value for value in combined if len(value) > 240]
        if oversized:
            raise ValueError("Study-context references may not exceed 240 characters")
        grounded[field_name] = combined

    evidence_ids: list[str] = []
    if original_context:
        evidence_ids.append("context:study")
    if context.organismHint:
        evidence_ids.append("context:organism")
    evidence_ids.extend(
        f"context:tissue:{index}"
        for index, _value in enumerate(context.tissueReferences)
    )
    evidence_ids.extend(
        f"context:cellType:{index}"
        for index, _value in enumerate(context.cellTypeReferences)
    )
    evidence_ids.extend(
        f"context:experiment:{index}"
        for index, _value in enumerate(context.experimentalDetails)
    )
    return StudyContextSummary(
        studyContext=original_context,
        **grounded,
        evidenceIds=evidence_ids,
    )


def _validate_feature_policy(
    deps: DataEnrichmentDependencies,
    policy: FeatureSelectionPolicy,
    grounded_context: StudyContextSummary,
) -> None:
    """Ground and validate one feature policy in deterministic order."""
    supported_species = {*_SUPPORTED_SPECIES, "unknown"}
    if policy.species not in supported_species:
        raise ValueError(
            f"unsupported species {policy.species!r}; choose a supported key or unknown"
        )
    if not policy.evidenceIds:
        raise ValueError(f"policy for assay {policy.assay!r} requires evidence IDs")
    policy.organismName = (
        _SUPPORTED_SPECIES[policy.species].label
        if policy.species in _SUPPORTED_SPECIES
        else "unknown"
    )
    inspection = deps.inspections.get(policy.assay)
    if inspection is None:
        raise ValueError(f"assay {policy.assay!r} was not inspected")
    modality = inspection.modalityEvidence
    policy.assayType = modality.assayType
    policy.assayModality = modality.modality
    policy.graphEligible = modality.graphEligible
    policy.markerEligible = modality.markerEligible
    policy.demultiplexEligible = modality.demultiplexEligible
    policy.exactControlFeatures = [
        FeatureReference(
            featureId=item.featureId,
            featureName=item.featureName,
        )
        for item in modality.adtControls
    ]
    policy.exactTagFeatures = [
        FeatureReference(
            featureId=item.featureId,
            featureName=item.featureName,
        )
        for item in modality.htoTags
    ]
    policy.peakCoordinateStatus = modality.atacCoordinates.status
    policy.evidenceIds = list(
        dict.fromkeys([*policy.evidenceIds, *modality.evidenceIds])
    )
    if (
        inspection.species in _SUPPORTED_SPECIES
        and policy.species != inspection.species
    ):
        raise ValueError(
            f"policy species {policy.species!r} conflicts with inspected "
            f"species {inspection.species!r}"
        )
    if (
        inspection.species == "unknown"
        and policy.species != "unknown"
        and not any(
            evidence_id.startswith("context:") for evidence_id in policy.evidenceIds
        )
    ):
        raise ValueError(
            "A context-derived species decision must cite context evidence"
        )
    observed_families = {item.family for item in inspection.families}
    cited_families = set(policy.excludeFamilies) | set(policy.protectFamilies)
    unknown_families = cited_families - observed_families
    if unknown_families:
        raise ValueError(
            f"policy cites unobserved families: {sorted(unknown_families)}"
        )
    protected_defaults = {
        item.family for item in inspection.families if item.defaultExclude is False
    }
    excluded_protected = sorted(
        set(policy.excludeFamilies).intersection(protected_defaults)
    )
    if excluded_protected:
        raise ValueError(
            "The initial enrichment policy cannot exclude families that "
            f"deterministic evidence protects by default: {excluded_protected}"
        )
    confirmed = deps.confirmedFeatures.get(policy.assay, set())
    cited_features = {
        *policy.excludeFeatures,
        *policy.protectFeatures,
        *policy.artificialFeatures,
    }
    unknown_features = cited_features - confirmed
    if unknown_features:
        raise ValueError(
            "Call find_present_features_batch before citing individual features: "
            f"{sorted(unknown_features)}"
        )
    exogenous_evidence = {
        value: item.evidenceId
        for item in inspection.exogenous
        for value in (item.featureId, item.featureName)
    }
    unsupported_artificial: list[str] = []
    for feature in policy.artificialFeatures:
        evidence_id = exogenous_evidence.get(feature)
        if evidence_id is not None and evidence_id in policy.evidenceIds:
            continue
        matching_context_ids = {
            f"context:experiment:{index}"
            for index, detail in enumerate(deps.context.experimentalDetails)
            if feature.casefold() in detail.casefold()
        }
        if matching_context_ids.intersection(policy.evidenceIds):
            continue
        unsupported_artificial.append(feature)
    if unsupported_artificial:
        raise ValueError(
            "Artificial features require their exogenous evidence ID or a "
            "feature-specific experimental-context evidence ID: "
            f"{sorted(unsupported_artificial)}"
        )
    unknown_evidence = set(policy.evidenceIds) - deps.evidenceIds
    if unknown_evidence:
        raise ValueError(
            f"policy cites unknown evidence IDs: {sorted(unknown_evidence)}"
        )
    policy.tissueReferences = list(grounded_context.tissueReferences)
    policy.cellTypeReferences = list(grounded_context.cellTypeReferences)
    policy.experimentalReferences = list(grounded_context.experimentalReferences)


def validate_data_enrichment_report(
    deps: DataEnrichmentDependencies,
    report: DataEnrichmentReport,
) -> DataEnrichmentReport:
    """Ground an agent report in inspected assays, context, and exact lookups."""
    if not deps.inspections:
        raise ValueError("Inspect every requested assay before returning the report")

    requested = set(deps.assays)
    reported = {policy.assay for policy in report.policies}
    if len(reported) != len(report.policies):
        raise ValueError("reports may contain only one policy for each assay")
    if not reported.issubset(requested):
        raise ValueError(
            f"policies cite assays outside the requested set: {sorted(reported - requested)}"
        )
    if report.status == "done" and reported != requested:
        raise ValueError(
            f"done reports require one policy for every requested assay: {deps.assays}"
        )

    grounded_context = _ground_study_context_summary(
        deps.context,
        report.studyContextSummary,
    )
    for policy in report.policies:
        _validate_feature_policy(deps, policy, grounded_context)

    report.studyContextSummary = grounded_context
    report.inspections = [deps.inspections[name] for name in deps.assays]
    report.toolCalls = list(deps.toolCalls)
    report.evidenceIds = list(
        dict.fromkeys(
            evidence_id
            for evidence_id in [
                *report.studyContextSummary.evidenceIds,
                *(
                    evidence_id
                    for policy in report.policies
                    for evidence_id in policy.evidenceIds
                ),
            ]
        )
    )
    logger.debug(
        "Data Enrichment report validated: "
        f"status={report.status}, policies={len(report.policies)}, "
        f"inspections={len(report.inspections)}, "
        f"toolCalls={len(report.toolCalls)}, evidence={len(report.evidenceIds)}"
    )
    return report


def fallback_data_enrichment_report(
    deps: DataEnrichmentDependencies,
    *,
    error: UnexpectedModelBehavior | UsageLimitExceeded,
    model_name: str,
) -> DataEnrichmentReport:
    """Build a conservative policy from completed deterministic inspections."""
    if set(deps.inspections) != set(deps.assays):
        raise error
    policies: list[FeatureSelectionPolicy] = []
    for assay_name in deps.assays:
        inspection = deps.inspections[assay_name]
        species = "unknown"
        species_confidence: Literal["high", "medium", "low", "unknown"] = "unknown"
        species_rationale = (
            inspection.speciesReason
            or "Deterministic feature inspection did not resolve a species."
        )
        policy_evidence = [f"assay:{assay_name}:species"]
        if inspection.species in _SUPPORTED_SPECIES:
            species = inspection.species
            species_confidence = (
                "high" if inspection.speciesMethod == "ensemblPrefix" else "medium"
            )
        else:
            organism_hint = deps.context.organismHint.strip().casefold()
            for key, specification in _SUPPORTED_SPECIES.items():
                if organism_hint in {key.casefold(), specification.label.casefold()}:
                    species = key
                    species_confidence = "medium"
                    species_rationale = (
                        "Exact caller organism hint resolved an otherwise unknown "
                        "feature-based species."
                    )
                    policy_evidence.append("context:organism")
                    break
        excluded_families = [
            family
            for family in inspection.families
            if family.defaultExclude is True and family.count > 0
        ]
        protected_families = [
            family for family in inspection.families if family.defaultExclude is False
        ]
        policy_evidence.extend(
            family.evidenceId for family in [*excluded_families, *protected_families]
        )
        policies.append(
            FeatureSelectionPolicy(
                assay=assay_name,
                species=species,
                speciesConfidence=species_confidence,
                speciesRationale=species_rationale,
                excludeFamilies=[family.family for family in excluded_families],
                protectFamilies=[family.family for family in protected_families],
                rationale=(
                    "Retained only deterministic family defaults after structured "
                    "model output was unavailable."
                ),
                evidenceIds=list(dict.fromkeys(policy_evidence)),
            )
        )
    error_detail = str(error).replace("\n", " ").strip()[:500]
    report = DataEnrichmentReport(
        status="done",
        policies=policies,
        studyContextSummary=StudyContextSummary.get_blank(),
        limitations=[
            "Structured enrichment output was unavailable; the fallback omitted "
            "all model-selected individual and artificial features.",
            "Free-text context extraction may be incomplete because only exact "
            "caller fields and deterministic organism mentions were retained.",
            error_detail,
        ],
        runInfo=AgentRunInfo(
            agentName="data_enrichment_fallback",
            modelName=model_name,
        ),
    )
    validated = validate_data_enrichment_report(deps, report)
    logger.warning(
        "Data Enrichment used its conservative fallback: "
        f"assays={len(validated.policies)}, evidence={len(validated.evidenceIds)}, "
        f"reason={error_detail}"
    )
    return validated


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
                verify that every explicit organism, tissue, cell population,
                experiment, hypothesis, and analysis intent has been placed in its
                corresponding summary list. Leave inspections, modality-derived
                fields, exact controls and tags, toolCalls, and report evidence at
                their defaults because validation fills them from exact tool state.
                """
            )
            .strip()
            .format(
                assays=", ".join(selected_assays),
                study_context=enrichment_context.studyContext or "not provided",
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
                output_validator=lambda report: validate_data_enrichment_report(
                    deps,
                    report,
                ),
            )
        except (UnexpectedModelBehavior, UsageLimitExceeded) as exc:
            if set(deps.inspections) != set(deps.assays):
                raise
            model_name = getattr(self.model, "model_name", type(self.model).__name__)
            return fallback_data_enrichment_report(
                deps,
                error=exc,
                model_name=str(model_name),
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
