"""Serializable contracts for data enrichment."""

from pathlib import Path
from typing import Any, Literal

from .._deps import AGENT_INSTALL_HINT
from ..types import AgentDataModel, AgentRunInfo, StageStatus

try:
    from pydantic import ConfigDict, Field, model_validator
    from pydantic.json_schema import SkipJsonSchema
except ImportError as exc:
    raise ImportError(AGENT_INSTALL_HINT) from exc


class DataEnrichmentContext(AgentDataModel):
    """Study evidence that may help resolve organism and feature policy."""

    studyContext: str = ""
    studyObjective: str = ""
    organismHint: str = ""
    tissueReferences: list[str] = Field(default_factory=list)
    cellTypeReferences: list[str] = Field(default_factory=list)
    experimentalDetails: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "DataEnrichmentContext":
        return cls()


class StudyContextSummary(AgentDataModel):
    """Verbatim, evidence-backed references extracted from the study context."""

    studyContext: SkipJsonSchema[str] = ""
    studyObjective: SkipJsonSchema[str] = ""
    organismReferences: list[str] = Field(default_factory=list)
    tissueReferences: list[str] = Field(default_factory=list)
    cellTypeReferences: list[str] = Field(default_factory=list)
    experimentalReferences: list[str] = Field(default_factory=list)
    hypothesisReferences: list[str] = Field(default_factory=list)
    analysisIntentReferences: list[str] = Field(default_factory=list)
    evidenceIds: SkipJsonSchema[list[str]] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "StudyContextSummary":
        return cls()


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


class HtoTagEvidence(AgentDataModel):
    """One exact feature from an assay persisted with the HTO type."""

    featureId: str
    featureName: str
    evidenceId: str

    @classmethod
    def get_blank(cls) -> "HtoTagEvidence":
        return cls(featureId="", featureName="", evidenceId="")


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


class DefaultHvgFamilyEvidence(AgentDataModel):
    """One case-insensitive family within Scarf's default HVG blacklist."""

    family: str = ""
    pattern: str = ""
    caseInsensitive: Literal[True] = True
    count: int = 0
    examples: list[str] = Field(default_factory=list)
    evidenceId: str = ""

    @classmethod
    def get_blank(cls) -> "DefaultHvgFamilyEvidence":
        return cls()


class RnaFeatureInventoryEvidence(AgentDataModel):
    """Exact name-column matches for Scarf's default HVG blacklist."""

    source: Literal["scarfDefaultHvgBlacklist"] = "scarfDefaultHvgBlacklist"
    policyEffect: Literal["evidenceOnly"] = "evidenceOnly"
    featureColumn: Literal["names"] = "names"
    totalFeatures: int = 0
    blacklist: str = ""
    matchCount: int = 0
    examples: list[str] = Field(default_factory=list)
    families: list[DefaultHvgFamilyEvidence] = Field(default_factory=list)
    evidenceId: str = ""
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "RnaFeatureInventoryEvidence":
        return cls()


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


class AssayFeatureInspection(AgentDataModel):
    """Bounded read-only inspection returned to the model."""

    assay: str
    assayKind: str = ""
    identity: dict[str, Any] = Field(default_factory=dict)
    species: str = "unknown"
    speciesMethod: str | None = None
    speciesReason: str = ""
    families: list[FeatureFamilyEvidence] = Field(default_factory=list)
    defaultFeatureInventory: RnaFeatureInventoryEvidence | None = None
    exogenous: list[ExogenousFeatureEvidence] = Field(default_factory=list)
    modalityEvidence: AssayModalityEvidence = Field(
        default_factory=AssayModalityEvidence.get_blank
    )
    notes: list[str] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "AssayFeatureInspection":
        return cls(assay="")


class AssayFeatureInspectionBatch(AgentDataModel):
    """All requested assay inspections returned by one model tool call."""

    inspections: list[AssayFeatureInspection] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "AssayFeatureInspectionBatch":
        return cls()


class FeatureReference(AgentDataModel):
    """An exact feature identifier and name observed in one assay."""

    featureId: str
    featureName: str

    @classmethod
    def get_blank(cls) -> "FeatureReference":
        return cls(featureId="", featureName="")


class FeatureMatch(AgentDataModel):
    """Resolution of one proposed feature against an assay."""

    query: str
    status: Literal["present", "ambiguous", "absent"]
    matches: list[FeatureReference] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "FeatureMatch":
        return cls(query="", status="absent")


class FeatureLookupResult(AgentDataModel):
    """Bounded result from exact feature lookup."""

    assay: str
    results: list[FeatureMatch] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "FeatureLookupResult":
        return cls(assay="")


class FeatureLookupBatch(AgentDataModel):
    """Exact feature lookups for every requested assay in one tool result."""

    lookups: list[FeatureLookupResult] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "FeatureLookupBatch":
        return cls()


class FeatureSelectionPolicy(AgentDataModel):
    """Grounded feature policy proposed for one assay."""

    assay: str
    species: str = "unknown"
    organismName: SkipJsonSchema[str] = "unknown"
    speciesConfidence: Literal["high", "medium", "low", "unknown"] = "unknown"
    speciesRationale: str = ""
    excludeFamilies: list[str] = Field(default_factory=list)
    protectFamilies: list[str] = Field(default_factory=list)
    excludeFeatures: list[str] = Field(default_factory=list)
    protectFeatures: list[str] = Field(default_factory=list)
    artificialFeatures: list[str] = Field(default_factory=list)
    tissueReferences: SkipJsonSchema[list[str]] = Field(default_factory=list)
    cellTypeReferences: SkipJsonSchema[list[str]] = Field(default_factory=list)
    experimentalReferences: SkipJsonSchema[list[str]] = Field(default_factory=list)
    assayType: SkipJsonSchema[str] = "Assay"
    assayModality: SkipJsonSchema[
        Literal["RNA", "ATAC", "ADT", "HTO", "unsupported"]
    ] = "unsupported"
    graphEligible: SkipJsonSchema[bool] = False
    markerEligible: SkipJsonSchema[bool] = False
    demultiplexEligible: SkipJsonSchema[bool] = False
    exactControlFeatures: SkipJsonSchema[list[FeatureReference]] = Field(
        default_factory=list
    )
    exactTagFeatures: SkipJsonSchema[list[FeatureReference]] = Field(
        default_factory=list
    )
    peakCoordinateStatus: SkipJsonSchema[
        Literal["notApplicable", "valid", "partial", "invalid"]
    ] = "notApplicable"
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


class DataEnrichmentToolCall(AgentDataModel):
    """Compact audit record for one read-only model tool call."""

    name: str
    assay: str
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "DataEnrichmentToolCall":
        return cls(name="", assay="")


class DataEnrichmentReport(AgentDataModel):
    """Final grounded report from :class:`DataEnrichmentAgent`."""

    status: StageStatus
    policies: list[FeatureSelectionPolicy] = Field(default_factory=list)
    inspections: SkipJsonSchema[list[AssayFeatureInspection]] = Field(
        default_factory=list
    )
    studyContextSummary: StudyContextSummary = Field(
        default_factory=StudyContextSummary.get_blank
    )
    unresolvedQuestions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    evidenceIds: SkipJsonSchema[list[str]] = Field(default_factory=list)
    toolCalls: SkipJsonSchema[list[DataEnrichmentToolCall]] = Field(
        default_factory=list
    )
    runInfo: SkipJsonSchema[AgentRunInfo] = Field(default_factory=AgentRunInfo)

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
