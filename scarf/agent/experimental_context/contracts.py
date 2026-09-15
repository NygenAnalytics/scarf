"""Experimental-context contracts and handoffs."""

import math
from typing import Any, Literal

from .._deps import AGENT_INSTALL_HINT
from ..cell_quality.profiles import QcMetricRole, RegisteredCellQcProfile
from ..types import (
    AgentDataModel,
    AgentRunInfo,
    ArtifactReferenceModel,
    BatchCorrectionAction,
    BatchSafetyEvidence,
    ExperimentalBiologyHandoff,
    ExperimentalTuningHandoff,
    StageStatus,
)

try:
    from pydantic import ConfigDict, Field, model_validator
    from pydantic.json_schema import SkipJsonSchema
except ImportError as exc:
    raise ImportError(AGENT_INSTALL_HINT) from exc

type ColumnDomain = Literal["biological", "technical", "design", "ignore", "unknown"]
type IntegrationMetric = Literal[
    "iLISI",
    "cLISI",
    "graphConnectivity",
    "proportionalBatchMixing",
]
type CellQcAction = Literal[
    "skip",
    "globalGaussian",
    "sampleMad",
    "registeredMad",
]
type LegacyCellQcAction = Literal["skip", "globalGaussian", "sampleMad"]
type CellQcDriverType = Literal["RNA", "ATAC"]


type ContrastTest = Literal["mann_whitney", "kruskal_wallis", "wilcoxon"]
type ContrastSampleStatistic = Literal["mean", "median", "fraction"]
type ContrastStatus = Literal["licensed", "blocked", "needsInput"]


class CovariateProposal(AgentDataModel):
    """Compare at most three distinct measured columns, without expression tests.

    Use one response with either one or two explanatory columns and no conditioning,
    or one response with one explanatory column and one categorical conditioning
    column. Observation and independent units are separate and do not count toward
    this limit. Two explanatory columns plus conditioning are unsupported; one
    explanatory column within a categorical stratum is an offered comparison.
    """

    response: str = Field(
        description=(
            "Exact observed outcome column. It must differ from every explanatory "
            "and conditioning column; do not compare a column with itself."
        )
    )
    explanatoryColumns: list[str] = Field(
        min_length=1,
        max_length=2,
        description=(
            "One or two distinct observed columns, each different from response. "
            "With two explanatory columns, conditionedOn must be null. With a "
            "conditioning column, supply exactly one explanatory column. Unit "
            "identifiers belong in observationUnit/independentUnit unless they "
            "are themselves the explicitly requested scientific comparison."
        ),
    )
    conditionedOn: str | None = Field(
        default=None,
        description=(
            "Optional exact categorical column defining strata, different from "
            "response and the single explanatory column. Set null for a two-column "
            "joint explanation. Continuous conditioning is unsupported."
        ),
    )
    observationUnit: str = Field(
        description=(
            "Exact observed column identifying the observation unit, such as a "
            "sample. This unit field does not count toward the three measured "
            "columns; it does not automatically belong in explanatoryColumns."
        )
    )
    independentUnit: str | None = Field(
        default=None,
        description=(
            "Exact observed independent-unit column, such as donor, or null to "
            "use observationUnit. It may equal observationUnit and does not count "
            "toward the three measured columns. Preserve repeated-donor identity."
        ),
    )
    rationale: str = Field(min_length=1)
    protectCombination: bool = False
    purpose: Literal["designCoverage", "association", "effectEstimation"] = (
        "designCoverage"
    )
    essential: bool = True
    objectiveQuote: str = ""

    @model_validator(mode="after")
    def validate_columns(self) -> "CovariateProposal":
        columns = [self.response, *self.explanatoryColumns]
        if self.conditionedOn is not None:
            columns.append(self.conditionedOn)
        repeated = sorted({name for name in columns if columns.count(name) > 1})
        if repeated:
            raise ValueError(
                "Comparison columns must be distinct; repeated columns "
                f"{repeated!r} occur in response={self.response!r}, "
                f"explanatoryColumns={self.explanatoryColumns!r}, "
                f"conditionedOn={self.conditionedOn!r}. A column cannot explain "
                "itself or also define its conditioning strata. Choose the actual "
                "distinct measured columns. Observation and independent units may "
                "share names and are not part of this uniqueness check."
            )
        if len(columns) > 3:
            raise ValueError(
                "A comparison requires at most three distinct measured columns; "
                f"received response={self.response!r}, "
                f"explanatoryColumns={self.explanatoryColumns!r}, "
                f"conditionedOn={self.conditionedOn!r}. With two explanatory "
                "columns set conditionedOn=null for a joint comparison. A "
                "within-stratum comparison requires one explanatory column. "
                "Observation and independent units do not count toward this "
                "limit. A joint explanation within strata is unsupported; do not "
                "claim that separate simpler comparisons answer that question."
            )
        if any(not value.strip() for value in [*columns, self.observationUnit]):
            raise ValueError(
                "Comparison columns and observation unit must be non-empty"
            )
        if self.protectCombination and len(self.explanatoryColumns) != 2:
            raise ValueError("A protected combination requires two explanatory columns")
        return self


class CovariateComparison(AgentDataModel):
    proposal: CovariateProposal
    status: Literal["computed", "unsupported"]
    evidence: dict[str, Any] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    evidenceId: str


class DesignEvidenceRequirement(AgentDataModel):
    """One objective question whose completion is checked against measured evidence."""

    requirementId: str = Field(min_length=1)
    question: str = Field(min_length=1)
    objectiveQuote: str = Field(min_length=1)
    kind: Literal["studyDesign", "designCoverage", "association", "effectEstimation"]
    columns: list[str] = Field(default_factory=list)
    observationUnit: str | None = None
    independentUnit: str | None = None
    essential: bool = True


class DesignEvidenceCoverage(AgentDataModel):
    """Measured answer to a requirement, without assigning a second workflow status."""

    requirementId: str
    status: Literal["computed", "nonIdentifiable", "unsupported", "failed"]
    evidenceIds: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class CaptureProposal(AgentDataModel):
    """An exact capture column and optional references supported by study prose."""

    column: str
    provenanceQuote: str = Field(min_length=1)
    referenceCaptures: list[str] = Field(default_factory=list, max_length=32)
    referenceProvenanceQuote: str = ""


class CovariateCharacterization(AgentDataModel):
    status: StageStatus
    cellSelection: ArtifactReferenceModel | None = None
    auditLog: list[dict[str, Any]] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    decisions: list[dict[str, Any]] = Field(default_factory=list)
    columns: list[dict[str, Any]] = Field(default_factory=list)
    coefficients: list[dict[str, Any]] = Field(default_factory=list)
    technicalNesting: list[dict[str, Any]] = Field(default_factory=list)
    confounding: list[dict[str, Any]] = Field(default_factory=list)
    unitLevelCounts: list[dict[str, Any]] = Field(default_factory=list)
    groupImbalance: list[dict[str, Any]] = Field(default_factory=list)
    missingness: list[dict[str, Any]] = Field(default_factory=list)
    designStructures: list[dict[str, Any]] = Field(default_factory=list)
    pairedCoverage: list[dict[str, Any]] = Field(default_factory=list)
    coefficientEstimability: list[dict[str, Any]] = Field(default_factory=list)
    comparisons: list[CovariateComparison] = Field(default_factory=list)
    captureProvenance: CaptureProposal | None = None

    @classmethod
    def get_blank(cls) -> "CovariateCharacterization":
        return cls(status="failed")


class InferenceUnit(AgentDataModel):
    """Observation and independent units for one biological coefficient."""

    observationUnit: str | None = None
    independentUnit: str | None = None

    @classmethod
    def get_blank(cls) -> "InferenceUnit":
        return cls()


class BatchCorrectionPlan(AgentDataModel):
    """A grounded recommendation about whether Harmony should be evaluated."""

    action: BatchCorrectionAction
    batchColumns: list[str] = Field(default_factory=list)
    preserveColumns: list[str] = Field(default_factory=list)
    metricsRequired: list[IntegrationMetric] = Field(default_factory=list)
    rationale: str = ""
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "BatchCorrectionPlan":
        return cls(action="needsInput")


class NamedArtifactSource(AgentDataModel):
    """One semantic name bound to an exact immutable artifact."""

    name: str = ""
    artifact: ArtifactReferenceModel = Field(default_factory=ArtifactReferenceModel)

    @model_validator(mode="after")
    def validate_source(self) -> "NamedArtifactSource":
        if self.name != self.name.strip():
            raise ValueError("Artifact source names cannot have surrounding whitespace")
        if bool(self.name.strip()) != bool(self.artifact.artifactId):
            raise ValueError("A named artifact source requires both name and artifact")
        return self

    @classmethod
    def get_blank(cls) -> "NamedArtifactSource":
        return cls()


class QcMetricSourceEvidence(AgentDataModel):
    """One source-specific quality metric on the exact active cells."""

    sourceId: str = ""
    metricName: str = ""
    metricRole: QcMetricRole = "diagnostic"
    assay: str | None = None
    sourceType: Literal["metadataColumn", "artifact"] = "metadataColumn"
    origin: Literal[
        "ingestionMetadata",
        "derivedArtifact",
        "externalArtifact",
    ] = "ingestionMetadata"
    executionName: str = ""
    metadataColumn: str | None = None
    artifact: ArtifactReferenceModel | None = None
    cellSelection: ArtifactReferenceModel | None = None
    inputArtifacts: list[ArtifactReferenceModel] = Field(default_factory=list)
    provenanceOperation: str | None = None
    valuesFingerprint: str = ""
    activeCells: int = 0
    missingCells: int = 0
    missingCellsByCapture: dict[str, int] = Field(default_factory=dict)
    usableForFiltering: bool = False
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_source(self) -> "QcMetricSourceEvidence":
        if (
            not self.sourceId
            and not self.metricName
            and self.metadataColumn is None
            and self.artifact is None
        ):
            return self
        if self.sourceType == "metadataColumn":
            if self.metadataColumn is None or self.artifact is not None:
                raise ValueError(
                    "A metadata QC source requires only metadataColumn provenance"
                )
            if self.origin != "ingestionMetadata":
                raise ValueError(
                    "Metadata QC sources must use ingestionMetadata origin"
                )
        elif self.artifact is None or self.metadataColumn is not None:
            raise ValueError("An artifact QC source requires only artifact provenance")
        if self.activeCells < 0 or not 0 <= self.missingCells <= self.activeCells:
            raise ValueError("QC source active and missing counts are inconsistent")
        if self.usableForFiltering and self.missingCells:
            raise ValueError("A QC source with missing values cannot drive filtering")
        return self


class QcSourceConcordance(AgentDataModel):
    """Observed agreement between imported and derived forms of one metric."""

    metricRole: QcMetricRole = "diagnostic"
    leftSourceId: str = ""
    rightSourceId: str = ""
    comparedCells: int = 0
    missingCells: int = 0
    meanAbsoluteDifference: float | None = None
    maximumAbsoluteDifference: float | None = None
    pearsonCorrelation: float | None = None
    exactlyEqual: bool = False
    numericallyClose: bool = False
    evidenceId: str = ""


class CaptureFailureEvidence(AgentDataModel):
    """Multi-axis capture anomaly plus exclusion-safety inputs."""

    capture: str = ""
    activeCells: int = 0
    retainedCells: int = 0
    retainedFraction: float = 0.0
    adverseAxes: list[QcMetricRole] = Field(default_factory=list)
    independentAdverseAxes: int = 0
    metricMissingFractions: dict[str, float] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    wholeCaptureFailure: bool = False
    conditionAndUnitSafety: list[dict[str, Any]] = Field(default_factory=list)
    preservesConditionCoverage: bool = False
    preservesIndependentUnitCoverage: bool = False
    exclusionEligible: bool = False
    doubletEvidenceIds: list[str] = Field(default_factory=list)
    evidenceId: str = ""

    @model_validator(mode="after")
    def validate_failure(self) -> "CaptureFailureEvidence":
        if self.independentAdverseAxes != len(set(self.adverseAxes)):
            raise ValueError("Capture failure axis count must match its unique axes")
        if self.wholeCaptureFailure != (self.independentAdverseAxes >= 2):
            raise ValueError(
                "Whole-capture failure requires at least two independent QC axes"
            )
        if self.exclusionEligible and (
            not self.wholeCaptureFailure
            or not self.preservesConditionCoverage
            or not self.preservesIndependentUnitCoverage
        ):
            raise ValueError(
                "Capture exclusion requires failure and preserved design coverage"
            )
        return self


class ContrastPlan(AgentDataModel):
    """One deterministic sample-aware statistical-testing license."""

    coefficient: str = ""
    groupOrder: list[str | int | float | bool] = Field(default_factory=list)
    sampleBy: str | None = None
    pairBy: str | None = None
    test: ContrastTest | None = None
    sampleStatistic: ContrastSampleStatistic = "mean"
    expressionCutoff: float = 0.0
    status: ContrastStatus = "blocked"
    betweenUnitDesign: bool = False
    replicationPassed: bool = False
    estimabilityPassed: bool = False
    pairedCoveragePassed: bool | None = None
    replication: dict[str, Any] = Field(default_factory=dict)
    estimability: dict[str, Any] = Field(default_factory=dict)
    pairedCoverage: dict[str, Any] = Field(default_factory=dict)
    blockedReasons: list[str] = Field(default_factory=list)
    evidenceId: str = ""
    evidenceIds: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_contrast(self) -> "ContrastPlan":
        if self.coefficient != self.coefficient.strip():
            raise ValueError(
                "Contrast coefficient cannot contain surrounding whitespace"
            )
        if self.sampleBy is not None and (
            not self.sampleBy.strip() or self.sampleBy != self.sampleBy.strip()
        ):
            raise ValueError("Contrast sampleBy must be a non-empty trimmed name")
        if self.pairBy is not None and (
            not self.pairBy.strip() or self.pairBy != self.pairBy.strip()
        ):
            raise ValueError("Contrast pairBy must be a non-empty trimmed name")
        group_keys = [(type(value).__name__, repr(value)) for value in self.groupOrder]
        if len(group_keys) != len(set(group_keys)):
            raise ValueError("Contrast groupOrder must contain unique values")
        if any(
            isinstance(value, float) and not math.isfinite(value)
            for value in self.groupOrder
        ):
            raise ValueError("Contrast groupOrder cannot contain non-finite values")
        if not math.isfinite(self.expressionCutoff):
            raise ValueError("Contrast expressionCutoff must be finite")
        if self.sampleStatistic != "fraction" and self.expressionCutoff != 0.0:
            raise ValueError(
                "Contrast expressionCutoff is only used with fraction summaries"
            )
        if self.test == "mann_whitney" and len(self.groupOrder) != 2:
            raise ValueError("mann_whitney requires exactly two ordered groups")
        if self.test == "kruskal_wallis" and len(self.groupOrder) < 3:
            raise ValueError("kruskal_wallis requires at least three ordered groups")
        if self.test == "wilcoxon":
            if len(self.groupOrder) != 2 or self.pairBy is None:
                raise ValueError(
                    "wilcoxon requires exactly two groups and an explicit pairBy"
                )
        elif self.pairBy is not None and self.test is not None:
            raise ValueError("A paired contrast must use the wilcoxon test")
        if self.status == "licensed":
            if (
                not self.coefficient
                or self.sampleBy is None
                or self.test is None
                or self.blockedReasons
                or not self.betweenUnitDesign
                or not self.replicationPassed
                or not self.estimabilityPassed
                or (self.pairBy is not None and self.pairedCoveragePassed is not True)
            ):
                raise ValueError(
                    "A licensed contrast requires resolved design, replication, "
                    "estimability, and paired coverage"
                )
        elif not self.blockedReasons:
            raise ValueError("A non-licensed contrast requires blockedReasons")
        return self

    @classmethod
    def get_blank(cls) -> "ContrastPlan":
        return cls(blockedReasons=["unresolvedContrast"])


def _validate_qc_sources(
    *,
    action: CellQcAction,
    attributes: list[str],
    artifact_metrics: list[NamedArtifactSource],
    sample_column: str | None,
    sample_artifact: NamedArtifactSource | None,
    registered_profile: RegisteredCellQcProfile | None = None,
    allow_metric_name_collisions: bool = False,
) -> None:
    if len(attributes) != len(set(attributes)):
        raise ValueError("Cell-QC metadata attributes must be unique")
    if any(
        not attribute.strip() or attribute != attribute.strip()
        for attribute in attributes
    ):
        raise ValueError(
            "Cell-QC metadata attributes cannot be blank or have surrounding whitespace"
        )
    artifact_names = [source.name for source in artifact_metrics]
    if len(artifact_names) != len(set(artifact_names)):
        raise ValueError("Cell-QC artifact metric names must be unique")
    if not allow_metric_name_collisions and set(attributes) & set(artifact_names):
        raise ValueError(
            "Cell-QC metadata and artifact metric names collide; explicitly "
            "validated multi-source evidence is required"
        )
    if any(source.artifact.kind != "quality_metric" for source in artifact_metrics):
        raise ValueError(
            "Cell-QC artifactMetrics must reference quality_metric artifacts"
        )
    if sample_column is not None and sample_artifact is not None:
        raise ValueError(
            "Cell-QC sampleColumn and sampleArtifact are mutually exclusive"
        )
    if sample_column is not None and (
        not sample_column.strip() or sample_column != sample_column.strip()
    ):
        raise ValueError(
            "Cell-QC sampleColumn cannot be blank or have surrounding whitespace"
        )
    if sample_artifact is not None and sample_artifact.artifact.kind != "hto_identity":
        raise ValueError(
            "Cell-QC sampleArtifact must reference an hto_identity artifact"
        )
    if sample_artifact is not None and sample_artifact.name in artifact_names:
        raise ValueError("Cell-QC sample and metric artifact names must be distinct")
    if registered_profile is not None:
        if registered_profile == "retainWithFlags":
            if action != "skip":
                raise ValueError(
                    "retainWithFlags must use the non-filtering skip action"
                )
            if sample_column is not None or sample_artifact is not None:
                raise ValueError("retainWithFlags cannot include a capture source")
            return
        if action != "registeredMad":
            raise ValueError(f"{registered_profile} must use the registeredMad action")
        capture_profile = registered_profile in {
            "captureMad5",
            "captureMad3Sensitivity",
            "pooledReferenceMad5",
        }
        has_one_capture_source = (sample_column is None) != (sample_artifact is None)
        if capture_profile and not has_one_capture_source:
            raise ValueError(
                f"{registered_profile} requires exactly one proven capture source"
            )
        if not capture_profile and (
            sample_column is not None or sample_artifact is not None
        ):
            raise ValueError(f"{registered_profile} cannot include a capture source")
        if not attributes and not artifact_metrics:
            raise ValueError("Registered MAD filtering requires at least one metric")
        return
    if action == "registeredMad":
        raise ValueError("registeredMad requires a registeredProfile")
    if action == "skip" and (attributes or artifact_metrics):
        raise ValueError("skip cannot include Cell-QC metrics")
    if action != "skip" and not attributes and not artifact_metrics:
        raise ValueError("Cell-QC filtering requires at least one metric")
    if action == "sampleMad" and (sample_column is None) == (sample_artifact is None):
        raise ValueError(
            "sampleMad requires exactly one sampleColumn or sampleArtifact"
        )
    if action != "sampleMad" and (
        sample_column is not None or sample_artifact is not None
    ):
        raise ValueError("Only sampleMad can include a sample source")


class CellQcProfileEvidence(AgentDataModel):
    """Projected retention for one registered or legacy cell-QC profile."""

    profileId: str = ""
    action: CellQcAction = "skip"
    registeredProfile: RegisteredCellQcProfile | None = None
    driverAssay: str | None = None
    driverAssayType: CellQcDriverType | None = None
    sampleColumn: str | None = None
    sampleArtifact: NamedArtifactSource | None = None
    captureColumn: str | None = None
    captureArtifact: NamedArtifactSource | None = None
    attributes: list[str] = Field(default_factory=list)
    artifactMetrics: list[NamedArtifactSource] = Field(default_factory=list)
    metricSources: list[QcMetricSourceEvidence] = Field(default_factory=list)
    sourceConcordance: list[QcSourceConcordance] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    resolvedBounds: dict[str, Any] | list[dict[str, Any]] = Field(default_factory=dict)
    activeCells: int = 0
    retainedCells: int = 0
    retainedFraction: float = 0.0
    activeCellsByCapture: dict[str, int] = Field(default_factory=dict)
    sampleRetainedCells: dict[str, int] = Field(default_factory=dict)
    retainedCellsByColumn: dict[str, dict[str, int]] = Field(default_factory=dict)
    retainedCellsByCombination: dict[str, dict[str, int]] = Field(default_factory=dict)
    unsafeRetentionGroups: list[str] = Field(default_factory=list)
    flaggedCells: dict[str, int] = Field(default_factory=dict)
    metricFlaggedCells: dict[str, dict[str, int]] = Field(default_factory=dict)
    failedCaptureCandidates: list[str] = Field(default_factory=list)
    captureFailureEvidence: list[CaptureFailureEvidence] = Field(default_factory=list)
    excludableCaptureCandidates: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    evidenceId: str = ""

    @model_validator(mode="after")
    def validate_sources(self) -> "CellQcProfileEvidence":
        _validate_qc_sources(
            action=self.action,
            attributes=self.attributes,
            artifact_metrics=self.artifactMetrics,
            sample_column=self.sampleColumn,
            sample_artifact=self.sampleArtifact,
            registered_profile=self.registeredProfile,
            allow_metric_name_collisions=True,
        )
        if self.captureColumn is not None and self.captureArtifact is not None:
            raise ValueError(
                "Cell-QC captureColumn and captureArtifact are mutually exclusive"
            )
        if (
            self.captureArtifact is not None
            and self.captureArtifact.artifact.kind != "hto_identity"
        ):
            raise ValueError(
                "Cell-QC captureArtifact must reference an hto_identity artifact"
            )
        failures = {item.capture: item for item in self.captureFailureEvidence}
        if len(failures) != len(self.captureFailureEvidence):
            raise ValueError("Cell-QC capture failure evidence must be unique")
        expected_failed = sorted(
            capture for capture, item in failures.items() if item.wholeCaptureFailure
        )
        if failures and sorted(self.failedCaptureCandidates) != expected_failed:
            raise ValueError(
                "Cell-QC failed captures must match their multi-axis evidence"
            )
        expected_excludable = sorted(
            capture for capture, item in failures.items() if item.exclusionEligible
        )
        if failures and sorted(self.excludableCaptureCandidates) != expected_excludable:
            raise ValueError(
                "Cell-QC excludable captures must match design-safety evidence"
            )
        return self

    @classmethod
    def get_blank(cls) -> "CellQcProfileEvidence":
        return cls()


class CellQcPlan(AgentDataModel):
    """A validated selection from the bounded cell-QC profiles."""

    action: CellQcAction = "skip"
    registeredProfile: RegisteredCellQcProfile | None = None
    profileId: str = ""
    driverAssay: str | None = None
    driverAssayType: CellQcDriverType | None = None
    sampleColumn: str | None = None
    sampleArtifact: NamedArtifactSource | None = None
    attributes: list[str] = Field(default_factory=list)
    artifactMetrics: list[NamedArtifactSource] = Field(default_factory=list)
    rationale: str = ""
    evidenceIds: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_sources(self) -> "CellQcPlan":
        _validate_qc_sources(
            action=self.action,
            attributes=self.attributes,
            artifact_metrics=self.artifactMetrics,
            sample_column=self.sampleColumn,
            sample_artifact=self.sampleArtifact,
            registered_profile=self.registeredProfile,
            allow_metric_name_collisions=True,
        )
        return self

    @classmethod
    def get_blank(cls) -> "CellQcPlan":
        return cls()


class ExperimentalContextDecision(AgentDataModel):
    """Model-authored choices that are revalidated against the datastore."""

    columnDomains: dict[str, ColumnDomain] = Field(default_factory=dict)
    coefficientsOfInterest: list[str] = Field(default_factory=list)
    unitsOfInference: dict[str, InferenceUnit] = Field(default_factory=dict)
    protectedCombinations: SkipJsonSchema[list[list[str]]] = Field(default_factory=list)
    physicalCaptureColumn: SkipJsonSchema[str | None] = None
    pooledReferenceCaptures: SkipJsonSchema[list[str]] = Field(default_factory=list)
    unsupportedProtection: SkipJsonSchema[list[str]] = Field(default_factory=list)
    batchCorrection: BatchCorrectionPlan = Field(
        default_factory=BatchCorrectionPlan.get_blank
    )
    cellQc: SkipJsonSchema[CellQcPlan] = Field(default_factory=CellQcPlan.get_blank)
    rationale: str = ""
    evidenceIds: list[str] = Field(default_factory=list)
    needsInput: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "ExperimentalContextDecision":
        return cls()


class RepresentationEvaluation(AgentDataModel):
    """Bounded integration metrics for one exact graph representation."""

    available: bool = False
    assay: str | None = None
    cellSelection: ArtifactReferenceModel | None = None
    neighbors: ArtifactReferenceModel | None = None
    connectivityMap: ArtifactReferenceModel | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "RepresentationEvaluation":
        return cls()


class CovariateEvidence(AgentDataModel):
    """One deterministic covariate characterization returned by a tool."""

    characterization: CovariateCharacterization = Field(
        default_factory=lambda: CovariateCharacterization(status="needsInput")
    )
    batchSafety: list[BatchSafetyEvidence] = Field(default_factory=list)
    qcProfiles: list[CellQcProfileEvidence] = Field(default_factory=list)
    qcMetricSources: list[QcMetricSourceEvidence] = Field(default_factory=list)
    qcSourceConcordance: list[QcSourceConcordance] = Field(default_factory=list)
    contrastPlans: list[ContrastPlan] = Field(default_factory=list)
    htoIdentityColumns: list[str] = Field(default_factory=list)
    htoIdentityArtifacts: list[NamedArtifactSource] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)
    evidenceRequirements: list[DesignEvidenceRequirement] = Field(default_factory=list)
    evidenceCoverage: list[DesignEvidenceCoverage] = Field(default_factory=list)


class ExperimentalContextResult(AgentDataModel):
    """Canonical experimental-context report returned to the caller."""

    status: StageStatus
    decision: ExperimentalContextDecision
    characterization: CovariateCharacterization
    cellSelection: ArtifactReferenceModel | None = None
    cellQc: CellQcPlan = Field(default_factory=CellQcPlan.get_blank)
    qcProfiles: list[CellQcProfileEvidence] = Field(default_factory=list)
    qcMetricSources: list[QcMetricSourceEvidence] = Field(default_factory=list)
    qcSourceConcordance: list[QcSourceConcordance] = Field(default_factory=list)
    contrastPlans: list[ContrastPlan] = Field(default_factory=list)
    qualityMetricArtifacts: list[NamedArtifactSource] = Field(default_factory=list)
    htoIdentityColumns: list[str] = Field(default_factory=list)
    htoIdentityArtifacts: list[NamedArtifactSource] = Field(default_factory=list)
    batchSafety: list[BatchSafetyEvidence] = Field(default_factory=list)
    currentRepresentation: RepresentationEvaluation = Field(
        default_factory=RepresentationEvaluation.get_blank
    )
    notes: list[str] = Field(default_factory=list)
    runInfo: AgentRunInfo = Field(default_factory=AgentRunInfo)

    @classmethod
    def get_blank(cls) -> "ExperimentalContextResult":
        return cls(
            status="needsInput",
            decision=ExperimentalContextDecision.get_blank(),
            characterization=CovariateCharacterization(status="needsInput"),
        )

    def to_parameter_tuning_handoff(self) -> ExperimentalTuningHandoff:
        """Return validated integration inputs for Parameter Tuning."""
        if self.status != "done":
            raise ValueError(
                "Experimental Context must be done before creating a tuning handoff"
            )
        if self.cellSelection is None:
            raise ValueError("Experimental Context result lacks a cell selection")
        plan = self.decision.batchCorrection
        batch_columns = sorted(plan.batchColumns)
        safety = sorted(
            (
                item
                for item in self.batchSafety
                if item.batchColumns == batch_columns
                and item.coefficient in self.decision.coefficientsOfInterest
            ),
            key=lambda item: item.coefficient,
        )
        if plan.action in {"evaluateHarmony", "unsafe"}:
            expected = set(self.decision.coefficientsOfInterest)
            if {item.coefficient for item in safety} != expected:
                raise ValueError(
                    "Experimental Context result lacks exact batch safety evidence"
                )
            if any(item.evidenceId not in plan.evidenceIds for item in safety):
                raise ValueError(
                    "Batch-correction plan does not cite its exact safety evidence"
                )
            if plan.action == "evaluateHarmony" and any(
                item.status != "safe" for item in safety
            ):
                raise ValueError("Harmony plan contains non-safe batch evidence")
            if plan.action == "unsafe" and (
                any(item.status == "notComputed" for item in safety)
                or not any(item.status == "unsafe" for item in safety)
            ):
                raise ValueError("Unsafe plan lacks exact unsafe batch evidence")
        return ExperimentalTuningHandoff(
            cellSelection=self.cellSelection,
            batchAction=plan.action,
            batchColumns=batch_columns,
            preservationColumns=list(plan.preserveColumns),
            coefficientsOfInterest=list(self.decision.coefficientsOfInterest),
            batchSafety=safety,
            evidenceIds=sorted({*self.decision.evidenceIds, *plan.evidenceIds}),
        )

    def to_biological_handoff(
        self,
        coefficient: str | None = None,
    ) -> ExperimentalBiologyHandoff:
        """Return one explicitly resolved biological coefficient."""
        if self.status != "done":
            raise ValueError(
                "Experimental Context must be done before creating a biology handoff"
            )
        if self.cellSelection is None:
            raise ValueError("Experimental Context result lacks a cell selection")
        coefficients = list(self.decision.coefficientsOfInterest)
        if coefficient is None:
            if len(coefficients) != 1:
                raise ValueError(
                    "Select one coefficient explicitly for biological interpretation"
                )
            coefficient = coefficients[0]
        if coefficient not in coefficients:
            raise ValueError(f"Unknown coefficient of interest {coefficient!r}")
        records = {
            record.get("name"): record
            for record in self.characterization.coefficients
            if isinstance(record.get("name"), str)
        }
        record = records.get(coefficient)
        if record is None:
            raise ValueError(f"Missing characterization for {coefficient!r}")
        reports = {
            report.get("coefficient"): report
            for report in self.characterization.confounding
            if isinstance(report.get("coefficient"), str)
        }
        report = reports.get(coefficient)
        known_evidence = characterization_evidence(self.characterization)
        relevant_evidence = {
            f"column:{coefficient}",
            f"coefficient:{coefficient}",
            f"estimability:{coefficient}",
            *(
                evidence_id
                for evidence_id in known_evidence
                if evidence_id.startswith(f"confounding:{coefficient}:")
            ),
        }
        for unit_name in (
            record.get("observationUnit"),
            record.get("independentUnit"),
        ):
            if isinstance(unit_name, str):
                relevant_evidence.add(f"column:{unit_name}")
        return ExperimentalBiologyHandoff(
            cellSelection=self.cellSelection,
            conditionColumn=coefficient,
            observationUnit=record.get("observationUnit"),
            independentUnit=record.get("independentUnit"),
            coefficientScope=str(record.get("scope", "")),
            estimability=dict(report.get("estimability") or {}) if report else {},
            evidenceIds=sorted(relevant_evidence.intersection(known_evidence)),
        )


class ExperimentalContextDependencies(AgentDataModel):
    """Runtime-only state shared by the agent's read-only tools."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    store: Any = Field(default=None, exclude=True)
    qcAssay: str | None = Field(default=None, exclude=True)
    cells: Any = Field(default=None, exclude=True)
    neighbors: Any = Field(default=None, exclude=True)
    connectivityMap: Any = Field(default=None, exclude=True)
    cellSelection: Any = Field(default=None, exclude=True)
    studyContext: str = ""
    studyObjective: str = ""
    directions: dict[str, Any] = Field(default_factory=dict)
    characterizationInputs: dict[str, Any] = Field(default_factory=dict, exclude=True)
    inventoryData: dict[str, Any] = Field(default_factory=dict, exclude=True)
    qcDesignData: Any = Field(default=None, exclude=True)
    checkpointRead: Any = Field(default=None, exclude=True)
    checkpointWrite: Any = Field(default=None, exclude=True)
    evidenceIds: set[str] = Field(default_factory=set)
    characterization: CovariateCharacterization | None = None
    designRounds: int = 0
    comparisons: list[CovariateComparison] = Field(default_factory=list)
    protectedCombinations: list[list[str]] = Field(default_factory=list)
    captureProposal: CaptureProposal | None = None
    batchSafety: dict[str, BatchSafetyEvidence] = Field(default_factory=dict)
    qcProfiles: dict[str, CellQcProfileEvidence] = Field(default_factory=dict)
    qcMetricSources: list[QcMetricSourceEvidence] = Field(default_factory=list)
    qcSourceConcordance: list[QcSourceConcordance] = Field(default_factory=list)
    contrastPlans: dict[str, ContrastPlan] = Field(default_factory=dict)
    htoIdentityColumns: list[str] = Field(default_factory=list)
    qualityMetricArtifacts: list[NamedArtifactSource] = Field(default_factory=list)
    htoIdentityArtifacts: list[NamedArtifactSource] = Field(default_factory=list)
    currentRepresentation: RepresentationEvaluation = Field(
        default_factory=RepresentationEvaluation.get_blank
    )
    toolCalls: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "ExperimentalContextDependencies":
        return cls()


def characterization_evidence(
    characterization: CovariateCharacterization,
) -> set[str]:
    """Build stable evidence IDs from one deterministic characterization."""
    evidence_ids = {
        f"column:{record['name']}"
        for record in characterization.columns
        if isinstance(record.get("name"), str)
    }
    for record in characterization.coefficients:
        coefficient = record.get("name")
        if isinstance(coefficient, str):
            evidence_ids.add(f"coefficient:{coefficient}")
    for report in characterization.confounding:
        coefficient = report.get("coefficient")
        if not isinstance(coefficient, str):
            continue
        evidence_ids.add(f"estimability:{coefficient}")
        for pair in report.get("pairs", []):
            technical = pair.get("technical")
            if isinstance(technical, str):
                evidence_ids.add(f"confounding:{coefficient}:{technical}")
    evidence_ids.update(item.evidenceId for item in characterization.comparisons)
    if characterization.captureProvenance is not None:
        evidence_ids.add(
            f"captureProvenance:{characterization.captureProvenance.column}"
        )
    return evidence_ids
