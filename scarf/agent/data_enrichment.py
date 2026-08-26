"""Read-only feature and organism enrichment agent."""

from collections.abc import Sequence
from pathlib import Path
from textwrap import dedent
from typing import Any, Literal

from ..features.gene_reference import species_registry
from .characterize_features import characterize_features
from .config import CONFIG, AgentRunConfig
from .config._deps import AGENT_INSTALL_HINT
from .config.agent_exec import run_agent_sync
from .tools import bounded_list
from .types import AgentDataModel, AgentRunInfo, StageStatus

try:
    from pydantic import ConfigDict, Field, model_validator
    from pydantic_ai import ModelRetry, RunContext
except ImportError as exc:
    raise ImportError(AGENT_INSTALL_HINT) from exc

__all__ = [
    "AssayFeatureInspection",
    "DataEnrichmentAgent",
    "DataEnrichmentContext",
    "DataEnrichmentDependencies",
    "DataEnrichmentReport",
    "DataEnrichmentToolCall",
    "ExogenousFeatureEvidence",
    "FeatureFamilyEvidence",
    "FeatureLookupResult",
    "FeatureMatch",
    "FeatureReference",
    "FeatureSelectionPolicy",
    "find_present_features",
    "inspect_assay_features",
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

        Never invent a feature. Call find_present_features before placing an
        individual feature in a policy. Treat Ensembl release misses as unresolved,
        not artificial. Mitochondrial, ribosomal, and histone families may be
        exclusion candidates. Sex-linked and cell-cycle families are protected by
        default in this initial implementation. Return a bounded report with
        citations copied from tool or context evidence IDs.
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
    notes: list[str] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "AssayFeatureInspection":
        return cls(assay="")

    @classmethod
    def get_example(cls) -> "AssayFeatureInspection":
        family = FeatureFamilyEvidence.get_example()
        return cls(
            assay="RNA",
            assayKind="RNAassay",
            identity={"nFeatures": 20_000, "nDuplicateIds": 0},
            species="homo_sapiens",
            speciesMethod="ensemblPrefix",
            speciesReason="Most feature IDs carry the ENSG prefix",
            families=[family],
            evidenceIds=["assay:RNA:identity", "assay:RNA:species", family.evidenceId],
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
    cacheDir: Path | None = None
    allowDownload: bool = False
    evidenceIds: set[str] = Field(default_factory=set)
    inspections: dict[str, AssayFeatureInspection] = Field(default_factory=dict)
    confirmedFeatures: dict[str, set[str]] = Field(default_factory=dict)
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

    characterization = characterize_features(
        deps.store,
        studyContext=deps.context.studyContext,
        model=None,
        assays=[assay_name],
        cacheDir=deps.cacheDir,
        allowDownload=deps.allowDownload,
    )
    if characterization.status != "done" or not characterization.assays:
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
    inspection = AssayFeatureInspection(
        assay=assay_name,
        assayKind=str(record.get("assayKind", "")),
        identity=dict(record.get("identity") or {}),
        species=str(record.get("species", "unknown")),
        speciesMethod=record.get("speciesMethod"),
        speciesReason=str(resolution.get("reason", "")),
        families=family_evidence,
        exogenous=exogenous_evidence,
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
    return inspection


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
    clean_queries = [value.strip() for value in queries if value.strip()]
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


def validate_data_enrichment_report(
    deps: DataEnrichmentDependencies,
    report: DataEnrichmentReport,
) -> DataEnrichmentReport:
    """Ground an agent report in inspected assays, context, and exact lookups."""
    if not deps.inspections:
        raise ValueError("Inspect every requested assay before returning the report")

    requested = set(deps.assays)
    reported = {policy.assay for policy in report.policies}
    if not reported.issubset(requested):
        raise ValueError(
            f"policies cite assays outside the requested set: {sorted(reported - requested)}"
        )
    if report.status == "done" and reported != requested:
        raise ValueError(
            f"done reports require one policy for every requested assay: {deps.assays}"
        )

    supported_species = {*_SUPPORTED_SPECIES, "unknown"}
    for policy in report.policies:
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
                "Call find_present_features before citing individual features: "
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
        unknown_tissues = set(policy.tissueReferences) - set(
            deps.context.tissueReferences
        )
        unknown_cell_types = set(policy.cellTypeReferences) - set(
            deps.context.cellTypeReferences
        )
        unknown_experiments = set(policy.experimentalReferences) - set(
            deps.context.experimentalDetails
        )
        if unknown_tissues or unknown_cell_types or unknown_experiments:
            raise ValueError(
                "policy context references must be copied from the supplied context: "
                f"tissues={sorted(unknown_tissues)}, "
                f"cellTypes={sorted(unknown_cell_types)}, "
                f"experiments={sorted(unknown_experiments)}"
            )

    report.inspections = [deps.inspections[name] for name in deps.assays]
    report.toolCalls = list(deps.toolCalls)
    report.evidenceIds = list(
        dict.fromkeys(
            evidence_id
            for policy in report.policies
            for evidence_id in policy.evidenceIds
        )
    )
    return report


class DataEnrichmentAgent:
    """A small read-only tool agent for feature and organism enrichment."""

    def __init__(
        self,
        model: Any,
        *,
        config: AgentRunConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or AgentRunConfig(
            requestLimit=8,
            toolCallLimit=12,
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

                Start by inspecting every assay. Resolve individual feature names
                through the lookup tool before returning them in a policy.
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
        execution = run_agent_sync(
            model=self.model,
            output_type=DataEnrichmentReport,
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            tools=[inspect_assay_features, find_present_features],
            deps_type=DataEnrichmentDependencies,
            deps=deps,
            config=self.config,
            name="data_enrichment",
            output_validator=lambda report: validate_data_enrichment_report(
                deps,
                report,
            ),
        )
        report = DataEnrichmentReport.model_validate(execution.output)
        report = validate_data_enrichment_report(deps, report)
        report.runInfo = execution.runInfo
        return report
