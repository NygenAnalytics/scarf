"""Tool-driven experimental-design and batch-correction assessment."""

import json
import math
from collections.abc import Mapping, Sequence
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ..graph.feature_projection import graph_cell_selection
from ..metadata.queries import reduce_observation_units
from ..metadata.selection import resolve_cell_aligned_artifact
from ..metrics.association import coefficient_estimability
from ..quality_control.filtering import (
    _sample_aware_mad_mask,
    gaussian_quantile_bounds,
)
from ..storage.refs import ArtifactRef
from ..storage.selections import read_stored_selection_mask
from ..utils.logging import logger
from .characterize_covariates import (
    CovariateCharacterization,
    _SelectionBoundCells,
    characterize_covariates,
)
from .config import AgentRunConfig
from .config._deps import AGENT_INSTALL_HINT
from .config.agent_exec import run_agent_sync
from .tools import artifact_reference, core_artifact_reference
from .types import (
    AgentDataModel,
    AgentRunInfo,
    ArtifactReferenceModel,
    BatchCorrectionAction,
    BatchSafetyEvidence,
    BatchSafetyStatus,
    ExperimentalBiologyHandoff,
    ExperimentalTuningHandoff,
    StageStatus,
)

if TYPE_CHECKING:
    from ..datastore.pipeline_run import PipelineRun

try:
    from pydantic import ConfigDict, Field, model_validator
    from pydantic_ai import ModelRetry, RunContext, Tool, UnexpectedModelBehavior
    from pydantic_ai.tools import ToolDefinition
except ImportError as exc:
    raise ImportError(AGENT_INSTALL_HINT) from exc

__all__ = [
    "BatchCorrectionPlan",
    "BatchSafetyEvidence",
    "CellQcPlan",
    "CellQcProfileEvidence",
    "CovariateEvidence",
    "ExperimentalContextAgent",
    "ExperimentalContextDecision",
    "ExperimentalContextDependencies",
    "ExperimentalContextResult",
    "InferenceUnit",
    "NamedArtifactSource",
    "RepresentationEvaluation",
    "analyze_experimental_design",
    "inspect_cell_covariates",
    "score_current_representation",
    "validate_experimental_context",
]

type ColumnDomain = Literal["biological", "technical", "design", "ignore", "unknown"]
type IntegrationMetric = Literal[
    "iLISI",
    "cLISI",
    "graphConnectivity",
    "proportionalBatchMixing",
]
type CellQcAction = Literal["skip", "globalGaussian", "sampleMad"]
type CellQcDriverType = Literal["RNA", "ATAC"]

_CONTEXT_LIMIT = 1200
_MAX_QC_SAMPLE_PROFILES = 4
_MAX_SAMPLE_RETENTION_ITEMS = 20


class InferenceUnit(AgentDataModel):
    """Observation and independent units for one biological coefficient."""

    observationUnit: str | None = None
    independentUnit: str | None = None

    @classmethod
    def get_blank(cls) -> "InferenceUnit":
        return cls()

    @classmethod
    def get_example(cls) -> "InferenceUnit":
        return cls(observationUnit="sample", independentUnit="donor")


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

    @classmethod
    def get_example(cls) -> "BatchCorrectionPlan":
        return cls(
            action="evaluateHarmony",
            batchColumns=["batch"],
            preserveColumns=["cell_type", "treatment"],
            metricsRequired=[
                "iLISI",
                "cLISI",
                "graphConnectivity",
            ],
            rationale=(
                "Batch is technical and crossed with treatment, so compare an exact "
                "Harmony candidate while protecting biological labels."
            ),
            evidenceIds=[
                "column:batch",
                "estimability:treatment",
                "batchEstimability:treatment:batch",
            ],
        )


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

    @classmethod
    def get_example(cls) -> "NamedArtifactSource":
        return cls(
            name="RNA_percentMito",
            artifact=ArtifactReferenceModel(
                assay="RNA",
                kind="quality_metric",
                artifactId="1" * 64,
            ),
        )


def _validate_qc_sources(
    *,
    action: CellQcAction,
    attributes: list[str],
    artifact_metrics: list[NamedArtifactSource],
    sample_column: str | None,
    sample_artifact: NamedArtifactSource | None,
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
    if any(source.artifact.kind != "quality_metric" for source in artifact_metrics):
        raise ValueError(
            "Cell-QC artifactMetrics must reference quality_metric artifacts"
        )
    collisions = sorted(set(attributes).intersection(artifact_names))
    if collisions:
        raise ValueError(
            f"Cell-QC metadata and artifact metric names collide: {collisions}"
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
    """Projected retention for one executor-supported cell-QC profile."""

    profileId: str = ""
    action: CellQcAction = "skip"
    driverAssay: str | None = None
    driverAssayType: CellQcDriverType | None = None
    sampleColumn: str | None = None
    sampleArtifact: NamedArtifactSource | None = None
    attributes: list[str] = Field(default_factory=list)
    artifactMetrics: list[NamedArtifactSource] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    activeCells: int = 0
    retainedCells: int = 0
    retainedFraction: float = 0.0
    sampleRetainedCells: dict[str, int] = Field(default_factory=dict)
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
        )
        return self

    @classmethod
    def get_blank(cls) -> "CellQcProfileEvidence":
        return cls()

    @classmethod
    def get_example(cls) -> "CellQcProfileEvidence":
        return cls(
            profileId="cellQc:RNA:RNA:globalGaussian:0.01:0.99",
            action="globalGaussian",
            driverAssay="RNA",
            driverAssayType="RNA",
            attributes=["RNA_nCounts", "RNA_nFeatures"],
            artifactMetrics=[NamedArtifactSource.get_example()],
            parameters={"minP": 0.01, "maxP": 0.99},
            activeCells=100,
            retainedCells=96,
            retainedFraction=0.96,
            evidenceId=("qcProfile:cellQc:RNA:RNA:globalGaussian:0.01:0.99"),
        )


class CellQcPlan(AgentDataModel):
    """A validated selection from the bounded cell-QC profiles."""

    action: CellQcAction = "skip"
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
        )
        return self

    @classmethod
    def get_blank(cls) -> "CellQcPlan":
        return cls()

    @classmethod
    def get_example(cls) -> "CellQcPlan":
        evidence = CellQcProfileEvidence.get_example()
        return cls(
            action=evidence.action,
            profileId=evidence.profileId,
            driverAssay=evidence.driverAssay,
            driverAssayType=evidence.driverAssayType,
            sampleColumn=evidence.sampleColumn,
            sampleArtifact=evidence.sampleArtifact,
            attributes=evidence.attributes,
            artifactMetrics=evidence.artifactMetrics,
            rationale="Use the bounded global profile for the RNA assay.",
            evidenceIds=[evidence.evidenceId],
        )


class ExperimentalContextDecision(AgentDataModel):
    """Model-authored choices that are revalidated against the datastore."""

    columnDomains: dict[str, ColumnDomain] = Field(default_factory=dict)
    coefficientsOfInterest: list[str] = Field(default_factory=list)
    unitsOfInference: dict[str, InferenceUnit] = Field(default_factory=dict)
    batchCorrection: BatchCorrectionPlan = Field(
        default_factory=BatchCorrectionPlan.get_blank
    )
    cellQc: CellQcPlan = Field(default_factory=CellQcPlan.get_blank)
    rationale: str = ""
    evidenceIds: list[str] = Field(default_factory=list)
    needsInput: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "ExperimentalContextDecision":
        return cls()

    @classmethod
    def get_example(cls) -> "ExperimentalContextDecision":
        return cls(
            columnDomains={
                "batch": "technical",
                "sample": "design",
                "donor": "design",
                "treatment": "biological",
            },
            coefficientsOfInterest=["treatment"],
            unitsOfInference={"treatment": InferenceUnit.get_example()},
            batchCorrection=BatchCorrectionPlan.get_example(),
            cellQc=CellQcPlan.get_example(),
            rationale="Treatment is the primary between-sample contrast.",
            evidenceIds=[
                "column:batch",
                "column:donor",
                "column:sample",
                "column:treatment",
            ],
        )


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

    @classmethod
    def get_example(cls) -> "RepresentationEvaluation":
        return cls(
            available=True,
            assay="RNA",
            cellSelection=ArtifactReferenceModel(
                scope="datastore",
                kind="cell_selection",
                artifactId="c" * 64,
            ),
            neighbors=ArtifactReferenceModel(
                assay="RNA",
                kind="neighbors",
                artifactId="a" * 64,
            ),
            connectivityMap=ArtifactReferenceModel(
                assay="RNA",
                kind="connectivity_map",
                artifactId="b" * 64,
            ),
            metrics={"iLISI:batch": 0.71, "cLISI:cell_type": 0.94},
            evidenceIds=[
                "metric:iLISI:batch:assay:RNA:neighbors:example-neighbors",
                "metric:cLISI:cell_type:assay:RNA:neighbors:example-neighbors",
            ],
        )


class CovariateEvidence(AgentDataModel):
    """One deterministic covariate characterization returned by a tool."""

    characterization: CovariateCharacterization = Field(
        default_factory=lambda: CovariateCharacterization(status="needsInput")
    )
    batchSafety: list[BatchSafetyEvidence] = Field(default_factory=list)
    qcProfiles: list[CellQcProfileEvidence] = Field(default_factory=list)
    htoIdentityColumns: list[str] = Field(default_factory=list)
    htoIdentityArtifacts: list[NamedArtifactSource] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_example(cls) -> "CovariateEvidence":
        return cls(
            characterization=CovariateCharacterization(
                status="done",
                notes=["Example deterministic covariate characterization"],
            ),
            qcProfiles=[CellQcProfileEvidence.get_example()],
            htoIdentityColumns=["sample_id"],
            htoIdentityArtifacts=[
                NamedArtifactSource(
                    name="HTO_htoIdentity",
                    artifact=ArtifactReferenceModel(
                        assay="HTO",
                        kind="hto_identity",
                        artifactId="2" * 64,
                    ),
                )
            ],
            evidenceIds=[
                "column:batch",
                CellQcProfileEvidence.get_example().evidenceId,
                "htoIdentity:sample_id",
                f"htoIdentityArtifact:HTO_htoIdentity:{'2' * 64}",
            ],
        )


class ExperimentalContextResult(AgentDataModel):
    """Canonical experimental-context report returned to the caller."""

    status: StageStatus
    decision: ExperimentalContextDecision
    characterization: CovariateCharacterization
    cellSelection: ArtifactReferenceModel | None = None
    cellQc: CellQcPlan = Field(default_factory=CellQcPlan.get_blank)
    qcProfiles: list[CellQcProfileEvidence] = Field(default_factory=list)
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

    @classmethod
    def get_example(cls) -> "ExperimentalContextResult":
        representation = RepresentationEvaluation.get_example()
        return cls(
            status="done",
            decision=ExperimentalContextDecision.get_example(),
            characterization=CovariateCharacterization(
                status="done",
                notes=["Example deterministic design characterization"],
            ),
            cellSelection=representation.cellSelection,
            cellQc=CellQcPlan.get_example(),
            qcProfiles=[CellQcProfileEvidence.get_example()],
            qualityMetricArtifacts=[NamedArtifactSource.get_example()],
            htoIdentityColumns=["sample_id"],
            htoIdentityArtifacts=[
                NamedArtifactSource(
                    name="HTO_htoIdentity",
                    artifact=ArtifactReferenceModel(
                        assay="HTO",
                        kind="hto_identity",
                        artifactId="2" * 64,
                    ),
                )
            ],
            batchSafety=[BatchSafetyEvidence.get_example()],
            currentRepresentation=representation,
            runInfo=AgentRunInfo.get_example(),
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
    cells: Any = Field(default=None, exclude=True)
    neighbors: Any = Field(default=None, exclude=True)
    connectivityMap: Any = Field(default=None, exclude=True)
    cellSelection: Any = Field(default=None, exclude=True)
    studyContext: str = ""
    directions: dict[str, Any] = Field(default_factory=dict)
    evidenceIds: set[str] = Field(default_factory=set)
    characterization: CovariateCharacterization | None = None
    batchSafety: dict[str, BatchSafetyEvidence] = Field(default_factory=dict)
    qcProfiles: dict[str, CellQcProfileEvidence] = Field(default_factory=dict)
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

    @classmethod
    def get_example(cls) -> "ExperimentalContextDependencies":
        return cls(
            studyContext="Case-control study with samples nested in donors.",
            directions={"columnDomains": {"batch": "technical"}},
        )


def _prepare_experimental_context_tool(
    ctx: RunContext[ExperimentalContextDependencies],
    tool_definition: ToolDefinition,
) -> ToolDefinition | None:
    """Expose each context tool once and in its required dependency order."""
    completed_calls = set(ctx.deps.toolCalls)
    if tool_definition.name == "inspect_cell_covariates":
        return None if tool_definition.name in completed_calls else tool_definition
    if tool_definition.name == "analyze_experimental_design":
        if (
            "inspect_cell_covariates" not in completed_calls
            or tool_definition.name in completed_calls
        ):
            return None
        return tool_definition
    if tool_definition.name == "score_current_representation":
        if (
            "analyze_experimental_design" not in completed_calls
            or tool_definition.name in completed_calls
        ):
            return None
        characterization = ctx.deps.characterization
        if characterization is not None and not any(
            record.get("domain") == "technical" and record.get("kind") == "categorical"
            for record in characterization.columns
        ):
            return None
        return tool_definition
    return tool_definition


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
    return evidence_ids


def _persisted_assay_type(store: Any, assay_name: str) -> str:
    """Read one persisted assay type without inferring modality from features."""
    root = getattr(store, "zw", None)
    attrs = getattr(root, "attrs", {})
    raw_types = attrs.get("assayTypes", {}) if isinstance(attrs, Mapping) else {}
    if isinstance(raw_types, Mapping):
        assay_type = raw_types.get(assay_name)
        if isinstance(assay_type, str):
            return assay_type
    return assay_name if assay_name in {"RNA", "ATAC", "ADT", "HTO"} else "Assay"


def _qc_driver(store: Any) -> tuple[str, CellQcDriverType] | None:
    """Choose the first RNA assay, otherwise the first ATAC assay."""
    assay_names = [str(name) for name in getattr(store, "assay_names", [])]
    for assay_type in ("RNA", "ATAC"):
        for assay_name in assay_names:
            if _persisted_assay_type(store, assay_name) == assay_type:
                return assay_name, assay_type
    return None


def _hto_identity_columns(deps: ExperimentalContextDependencies) -> list[str]:
    """Return explicitly supplied imported HTO identity metadata columns."""
    requested: list[str] = []
    directed_many = deps.directions.get("htoIdentityColumns")
    if isinstance(directed_many, list | tuple):
        requested.extend(str(value) for value in directed_many)
    directed_one = deps.directions.get("htoIdentityColumn")
    if isinstance(directed_one, str):
        requested.append(directed_one)
    available = set(deps.store.cells.columns)
    return list(dict.fromkeys(name for name in requested if name in available))


def _cell_selection_ref(deps: ExperimentalContextDependencies) -> ArtifactRef:
    selection = core_artifact_reference(deps.cellSelection)
    if not isinstance(selection, ArtifactRef):
        raise ValueError("cellSelection must identify an exact artifact")
    if selection.kind != "cell_selection" or selection.scope != "datastore":
        raise ValueError("cellSelection must identify a datastore cell selection")
    return selection


def _active_cell_count(deps: ExperimentalContextDependencies) -> int:
    selection = _cell_selection_ref(deps)
    active = read_stored_selection_mask(
        deps.store.zw,
        selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    if active.ndim != 1 or active.shape[0] != deps.store.cells.N:
        raise ValueError(
            "cellSelection must contain an aligned boolean selection vector"
        )
    return int(active.sum())


def _source_ref(
    source: NamedArtifactSource,
    *,
    expected_kind: str,
) -> ArtifactRef:
    if not isinstance(source, NamedArtifactSource):
        raise TypeError("Artifact sources must be NamedArtifactSource values")
    if not source.name.strip():
        raise ValueError("Artifact sources require a non-empty semantic name")
    artifact = core_artifact_reference(source.artifact)
    if not isinstance(artifact, ArtifactRef) or artifact.kind != expected_kind:
        raise ValueError(
            f"Artifact source {source.name!r} must reference {expected_kind!r}"
        )
    return artifact


def _artifact_evidence_id(source: NamedArtifactSource) -> str:
    return f"htoIdentityArtifact:{source.name}:{source.artifact.artifactId}"


def _hto_artifact_map(
    deps: ExperimentalContextDependencies,
) -> dict[str, ArtifactRef]:
    artifacts: dict[str, ArtifactRef] = {}
    for source in deps.htoIdentityArtifacts:
        if source.name in artifacts:
            raise ValueError("HTO identity artifact names must be unique")
        artifacts[source.name] = _source_ref(
            source,
            expected_kind="hto_identity",
        )
    return artifacts


def _resolved_artifact_values(
    deps: ExperimentalContextDependencies,
    source: NamedArtifactSource,
    *,
    expected_kind: str,
) -> np.ndarray:
    resolved = resolve_cell_aligned_artifact(
        deps.store.zw,
        _source_ref(source, expected_kind=expected_kind),
        cell_selection=_cell_selection_ref(deps),
        expected_kind=expected_kind,
    )
    return np.asarray(resolved.values)


def _qc_attributes(store: Any, assay_name: str, assay_type: str) -> list[str]:
    del assay_type
    suffixes = ["nCounts", "nFeatures"]
    available = set(store.cells.columns)
    return [
        f"{assay_name}_{suffix}"
        for suffix in suffixes
        if f"{assay_name}_{suffix}" in available
    ]


def _qc_sample_columns(
    deps: ExperimentalContextDependencies,
    characterization: CovariateCharacterization | None,
) -> list[str]:
    requested: list[str] = []
    directed = deps.directions.get("cellQc")
    if isinstance(directed, Mapping):
        sample_column = directed.get("sampleColumn")
        if isinstance(sample_column, str):
            requested.append(sample_column)
    if characterization is not None:
        for record in characterization.coefficients:
            observation_unit = record.get("observationUnit")
            if isinstance(observation_unit, str):
                requested.append(observation_unit)
    requested.extend(deps.htoIdentityColumns)
    available = set(deps.store.cells.columns)
    return list(
        dict.fromkeys(name for name in requested if name in available and name != "I")
    )[:_MAX_QC_SAMPLE_PROFILES]


def _qc_profile_id(
    action: CellQcAction,
    *,
    driver: tuple[str, CellQcDriverType] | None,
    sample_column: str | None = None,
    sample_artifact: NamedArtifactSource | None = None,
) -> str:
    assay_name, assay_type = driver or ("none", "none")
    suffix = {
        "skip": "skip",
        "globalGaussian": "globalGaussian:0.01:0.99",
        "sampleMad": (
            f"sampleMad:metadata:{sample_column}:3:20"
            if sample_artifact is None
            else (
                f"sampleMad:artifact:{sample_artifact.name}:"
                f"{sample_artifact.artifact.artifactId}:3:20"
            )
        ),
    }[action]
    return f"cellQc:{assay_type}:{assay_name}:{suffix}"


def _global_qc_profile(
    deps: ExperimentalContextDependencies,
    driver: tuple[str, CellQcDriverType],
    active: np.ndarray,
    active_cells: int,
    values_by_attr: dict[str, np.ndarray],
    metadata_attributes: list[str],
    artifact_metrics: list[NamedArtifactSource],
    attribute_notes: list[str],
) -> CellQcProfileEvidence | None:
    """Build the bounded global Gaussian QC profile when bounds are valid."""
    resolved_bounds: dict[str, dict[str, float]] = {}
    global_keep = active.copy()
    selected_names: list[str] = []
    for attribute, values in values_by_attr.items():
        if float(np.std(values)) == 0.0:
            attribute_notes.append(f"Ignored constant QC metric {attribute!r}")
            continue
        low, high = gaussian_quantile_bounds(values, 0.01, 0.99)
        if not np.isfinite([low, high]).all():
            attribute_notes.append(
                f"Ignored QC column {attribute!r} with non-finite Gaussian bounds"
            )
            continue
        resolved_bounds[attribute] = {"low": low, "high": high}
        selected_names.append(attribute)
        global_keep &= (values > low) & (values < high)
    if not selected_names:
        return None
    retained_cells = int(global_keep.sum())
    profile_id = _qc_profile_id(
        "globalGaussian",
        driver=driver,
    )
    selected = set(selected_names)
    return CellQcProfileEvidence(
        profileId=profile_id,
        action="globalGaussian",
        driverAssay=driver[0],
        driverAssayType=driver[1],
        attributes=[name for name in metadata_attributes if name in selected],
        artifactMetrics=[
            source for source in artifact_metrics if source.name in selected
        ],
        parameters={
            "minP": 0.01,
            "maxP": 0.99,
            "resolvedBounds": resolved_bounds,
        },
        activeCells=active_cells,
        retainedCells=retained_cells,
        retainedFraction=retained_cells / active_cells,
        notes=attribute_notes,
        evidenceId=f"qcProfile:{profile_id}",
    )


def _sample_qc_profiles(
    deps: ExperimentalContextDependencies,
    characterization: CovariateCharacterization | None,
    driver: tuple[str, CellQcDriverType],
    active: np.ndarray,
    active_cells: int,
    values_by_attr: dict[str, np.ndarray],
    metadata_attributes: list[str],
    artifact_metrics: list[NamedArtifactSource],
) -> list[CellQcProfileEvidence]:
    """Build bounded sample-aware MAD profiles from exact sample sources."""
    attributes = list(values_by_attr)
    profiles: list[CellQcProfileEvidence] = []
    sample_sources: list[tuple[str | None, NamedArtifactSource | None]] = [
        (None, source) for source in deps.htoIdentityArtifacts
    ]
    sample_sources.extend(
        (column, None) for column in _qc_sample_columns(deps, characterization)
    )
    for sample_column, sample_artifact in sample_sources[:_MAX_QC_SAMPLE_PROFILES]:
        if not attributes:
            break
        artifact_labels = (
            None
            if sample_artifact is None
            else _resolved_artifact_values(
                deps,
                sample_artifact,
                expected_kind="hto_identity",
            )
        )
        try:
            sample_labels = (
                np.asarray(deps.cells.fetch(sample_column))
                if sample_column is not None
                else np.asarray(artifact_labels)
            )
            keep, provenance = _sample_aware_mad_mask(
                values_by_attr=values_by_attr,
                sample_labels=sample_labels,
                active=active,
                n_mads=3.0,
                min_cells_per_sample=20,
                attrs=attributes,
            )
        except (TypeError, ValueError):
            continue
        retained_mask = active & keep
        retained_cells = int(retained_mask.sum())
        sample_retention: dict[str, int] = {}
        seen: set[object] = set()
        for label in sample_labels[active]:
            value = label.item() if isinstance(label, np.generic) else label
            if value in seen:
                continue
            seen.add(value)
            key = value.decode("utf-8") if isinstance(value, bytes) else str(value)
            sample_retention[key] = int(
                (retained_mask & (sample_labels == label)).sum()
            )
        notes = list(provenance["warnings"])
        if len(sample_retention) > _MAX_SAMPLE_RETENTION_ITEMS:
            notes.append(
                "Per-sample retention was truncated to the first "
                f"{_MAX_SAMPLE_RETENTION_ITEMS} samples"
            )
            sample_retention = dict(
                list(sample_retention.items())[:_MAX_SAMPLE_RETENTION_ITEMS]
            )
        profile_id = _qc_profile_id(
            "sampleMad",
            driver=driver,
            sample_column=sample_column,
            sample_artifact=sample_artifact,
        )
        profiles.append(
            CellQcProfileEvidence(
                profileId=profile_id,
                action="sampleMad",
                driverAssay=driver[0],
                driverAssayType=driver[1],
                sampleColumn=sample_column,
                sampleArtifact=sample_artifact,
                attributes=list(metadata_attributes),
                artifactMetrics=list(artifact_metrics),
                parameters={
                    "nMads": 3.0,
                    "minCellsPerSample": 20,
                    "nSamples": len(provenance["sample_sizes"]),
                    "nSkippedSamples": len(provenance["skip_reasons"]),
                },
                activeCells=active_cells,
                retainedCells=retained_cells,
                retainedFraction=retained_cells / active_cells,
                sampleRetainedCells=sample_retention,
                notes=notes,
                evidenceId=f"qcProfile:{profile_id}",
            )
        )
    return profiles


def _offered_qc_profiles(
    deps: ExperimentalContextDependencies,
    characterization: CovariateCharacterization | None = None,
) -> list[CellQcProfileEvidence]:
    """Project bounded QC profiles against the exact shared cell selection."""
    active_cells = _active_cell_count(deps)
    active = np.ones(active_cells, dtype=bool)
    driver = _qc_driver(deps.store)
    driver_assay = driver[0] if driver is not None else None
    driver_type = driver[1] if driver is not None else None
    skip_id = _qc_profile_id(
        "skip",
        driver=driver,
    )
    skip_notes = (
        []
        if driver is not None
        else ["No RNA or ATAC assay is eligible to drive automatic cell QC"]
    )
    profiles = [
        CellQcProfileEvidence(
            profileId=skip_id,
            action="skip",
            driverAssay=driver_assay,
            driverAssayType=driver_type,
            activeCells=active_cells,
            retainedCells=active_cells,
            retainedFraction=1.0 if active_cells else 0.0,
            notes=skip_notes,
            evidenceId=f"qcProfile:{skip_id}",
        )
    ]
    if driver is None or active_cells == 0:
        deps.qcProfiles = {profile.profileId: profile for profile in profiles}
        return profiles

    driver_assay, driver_type = driver
    metadata_attributes = _qc_attributes(deps.store, driver_assay, driver_type)
    values_by_attr: dict[str, np.ndarray] = {}
    attribute_notes: list[str] = []
    for attribute in metadata_attributes:
        try:
            values = np.asarray(
                deps.cells.fetch(attribute),
                dtype=float,
            )
        except (TypeError, ValueError):
            attribute_notes.append(f"Ignored non-numeric QC column {attribute!r}")
            continue
        if values.ndim != 1 or values.shape != active.shape:
            attribute_notes.append(f"Ignored unaligned QC column {attribute!r}")
            continue
        if not np.isfinite(values).all():
            attribute_notes.append(f"Ignored non-finite QC column {attribute!r}")
            continue
        values_by_attr[attribute] = values
    valid_metadata_attributes = [
        attribute for attribute in metadata_attributes if attribute in values_by_attr
    ]
    artifact_metrics: list[NamedArtifactSource] = []
    for source in deps.qualityMetricArtifacts:
        artifact = _source_ref(source, expected_kind="quality_metric")
        if artifact.assay != driver_assay:
            continue
        if source.name in values_by_attr:
            raise ValueError(
                f"QC artifact name {source.name!r} collides with a metadata metric"
            )
        values = np.asarray(
            _resolved_artifact_values(
                deps,
                source,
                expected_kind="quality_metric",
            ),
            dtype=float,
        )
        if values.ndim != 1 or values.shape != active.shape:
            raise ValueError(
                f"QC artifact {source.name!r} does not align with cellSelection"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"QC artifact {source.name!r} contains non-finite values")
        values_by_attr[source.name] = values
        artifact_metrics.append(source)

    global_profile = _global_qc_profile(
        deps,
        driver,
        active,
        active_cells,
        values_by_attr,
        valid_metadata_attributes,
        artifact_metrics,
        attribute_notes,
    )
    if global_profile is not None:
        profiles.append(global_profile)
    profiles.extend(
        _sample_qc_profiles(
            deps,
            characterization,
            driver,
            active,
            active_cells,
            values_by_attr,
            valid_metadata_attributes,
            artifact_metrics,
        )
    )

    deps.qcProfiles = {profile.profileId: profile for profile in profiles}
    return profiles


async def inspect_cell_covariates(
    ctx: RunContext[ExperimentalContextDependencies],
) -> CovariateEvidence:
    """Inspect cell metadata without making model-driven choices or writing data."""
    logger.info(
        "Experimental Context covariate inspection started: "
        f"cellSelection={ctx.deps.cellSelection.artifact_id}"
    )
    ctx.deps.htoIdentityColumns = _hto_identity_columns(ctx.deps)
    characterization = characterize_covariates(
        ctx.deps.store,
        cellSelection=ctx.deps.cellSelection,
        studyContext=ctx.deps.studyContext,
        model=None,
        directions=ctx.deps.directions,
        groupingArtifacts=_hto_artifact_map(ctx.deps),
    )
    ctx.deps.characterization = characterization
    qc_profiles = _offered_qc_profiles(ctx.deps)
    evidence_ids = characterization_evidence(characterization)
    evidence_ids.update(profile.evidenceId for profile in qc_profiles)
    evidence_ids.update(
        f"htoIdentity:{column}" for column in ctx.deps.htoIdentityColumns
    )
    evidence_ids.update(
        _artifact_evidence_id(source) for source in ctx.deps.htoIdentityArtifacts
    )
    ctx.deps.evidenceIds.update(evidence_ids)
    ctx.deps.toolCalls.append("inspect_cell_covariates")
    logger.info(
        "Experimental Context covariate inspection completed: "
        f"status={characterization.status}, "
        f"columns={len(characterization.columns)}, "
        f"coefficients={len(characterization.coefficients)}, "
        f"qcProfiles={len(qc_profiles)}, "
        f"htoIdentities={len(ctx.deps.htoIdentityColumns)}, "
        f"evidence={len(evidence_ids)}"
    )
    return CovariateEvidence(
        characterization=characterization,
        qcProfiles=qc_profiles,
        htoIdentityColumns=ctx.deps.htoIdentityColumns,
        htoIdentityArtifacts=ctx.deps.htoIdentityArtifacts,
        evidenceIds=sorted(evidence_ids),
    )


async def analyze_experimental_design(
    ctx: RunContext[ExperimentalContextDependencies],
    column_domains: dict[str, ColumnDomain],
    coefficients_of_interest: list[str],
    units_of_inference: dict[str, InferenceUnit],
    batch_columns: list[str] | str | None = None,
) -> CovariateEvidence:
    """Validate proposed domains and inference units and compute confounding.

    Args:
        ctx: Pydantic AI run context containing the existing datastore.
        column_domains: Domain assignment for each metadata column under review.
        coefficients_of_interest: Biological columns representing study contrasts.
        units_of_inference: Observation and independent units for each coefficient.
        batch_columns: Exact technical columns proposed for Harmony evaluation.
            A single column may be supplied as either a string or a one-item list.
    """
    proposed_batch_count = (
        1 if isinstance(batch_columns, str) else len(batch_columns or [])
    )
    logger.info(
        "Experimental Context design analysis started: "
        f"domains={len(column_domains)}, "
        f"coefficients={len(coefficients_of_interest)}, "
        f"inferenceUnits={len(units_of_inference)}, "
        f"batchColumns={proposed_batch_count}"
    )
    directions = dict(ctx.deps.directions)
    directed_domains = dict(column_domains)
    directed_domains.update(dict(directions.get("columnDomains") or {}))
    directions["columnDomains"] = directed_domains
    directed_coefficients = list(
        dict.fromkeys(
            [
                *coefficients_of_interest,
                *(directions.get("coefficientsOfInterest") or []),
            ]
        )
    )
    directions["coefficientsOfInterest"] = directed_coefficients
    directed_units = {
        name: unit.model_dump(exclude_none=True)
        for name, unit in units_of_inference.items()
    }
    directed_units.update(dict(directions.get("unitsOfInference") or {}))
    directions["unitsOfInference"] = directed_units

    proposed_batch_columns = (
        [batch_columns] if isinstance(batch_columns, str) else list(batch_columns or [])
    )
    canonical_batch_columns = sorted(set(proposed_batch_columns))
    if len(canonical_batch_columns) != len(proposed_batch_columns):
        logger.warning(
            "Experimental Context rejected duplicate proposed batch columns: "
            f"{proposed_batch_columns[:20]}"
        )
        raise ModelRetry("Proposed batch columns must be unique")
    inspected_records = {
        record.get("name"): record
        for record in (
            ctx.deps.characterization.columns
            if ctx.deps.characterization is not None
            else []
        )
        if isinstance(record.get("name"), str)
    }
    if ctx.deps.characterization is not None:
        for batch_column in canonical_batch_columns:
            inspected = inspected_records.get(batch_column)
            if inspected is None:
                logger.warning(
                    "Experimental Context rejected unknown proposed batch column "
                    f"before design recomputation: {batch_column!r}"
                )
                raise ModelRetry(f"Unknown batch column {batch_column!r}")
            proposed_domain = directed_domains.get(
                batch_column,
                inspected.get("domain"),
            )
            if proposed_domain != "technical":
                logger.warning(
                    "Experimental Context rejected proposed batch column before "
                    f"design recomputation: {batch_column!r}, "
                    f"domain={proposed_domain!r}, required='technical'"
                )
                raise ModelRetry(
                    f"Batch column {batch_column!r} must be classified as technical"
                )
            if inspected.get("kind") != "categorical":
                logger.warning(
                    "Experimental Context rejected proposed batch column before "
                    f"design recomputation: {batch_column!r}, "
                    f"kind={inspected.get('kind')!r}, required='categorical'"
                )
                raise ModelRetry(
                    f"Batch column {batch_column!r} must be categorical for Harmony"
                )

    characterization = characterize_covariates(
        ctx.deps.store,
        cellSelection=ctx.deps.cellSelection,
        studyContext=ctx.deps.studyContext,
        model=None,
        directions=directions,
        groupingArtifacts=_hto_artifact_map(ctx.deps),
    )
    if characterization.status == "failed":
        rejection = "; ".join(characterization.notes).strip()
        logger.warning(
            "Experimental Context design characterization rejected the proposed "
            f"directions: {rejection[:1000]}; "
            f"domainColumns={sorted(column_domains)[:50]}, "
            f"coefficients={coefficients_of_interest[:50]}, "
            f"inferenceUnits={sorted(units_of_inference)[:50]}"
        )
        raise ModelRetry("; ".join(characterization.notes))

    # Retain the validated deterministic work even when the proposed Harmony
    # columns below are rejected. A bounded fallback can then continue without
    # rescanning the metadata or accepting an unsafe model choice.
    ctx.deps.characterization = characterization
    if not ctx.deps.htoIdentityColumns:
        ctx.deps.htoIdentityColumns = _hto_identity_columns(ctx.deps)
    qc_profiles = _offered_qc_profiles(ctx.deps, characterization)
    evidence_ids = characterization_evidence(characterization)
    evidence_ids.update(profile.evidenceId for profile in qc_profiles)
    evidence_ids.update(
        f"htoIdentity:{column}" for column in ctx.deps.htoIdentityColumns
    )
    evidence_ids.update(
        _artifact_evidence_id(source) for source in ctx.deps.htoIdentityArtifacts
    )
    ctx.deps.evidenceIds.update(evidence_ids)

    column_records = {
        record.get("name"): record
        for record in characterization.columns
        if isinstance(record.get("name"), str)
    }
    for batch_column in canonical_batch_columns:
        record = column_records.get(batch_column)
        if record is None:
            logger.warning(
                "Experimental Context rejected unknown proposed batch column: "
                f"{batch_column!r}"
            )
            raise ModelRetry(f"Unknown batch column {batch_column!r}")
        if record.get("domain") != "technical":
            logger.warning(
                "Experimental Context rejected proposed batch column "
                f"{batch_column!r}: domain={record.get('domain')!r}, "
                "required='technical'"
            )
            raise ModelRetry(
                f"Batch column {batch_column!r} must be classified as technical"
            )
        if record.get("kind") != "categorical":
            logger.warning(
                "Experimental Context rejected proposed batch column "
                f"{batch_column!r}: kind={record.get('kind')!r}, "
                "required='categorical'"
            )
            raise ModelRetry(
                f"Batch column {batch_column!r} must be categorical for Harmony"
            )

    coefficient_records = {
        record.get("name"): record
        for record in characterization.coefficients
        if isinstance(record.get("name"), str)
    }
    confounding_reports = {
        report.get("coefficient"): report
        for report in characterization.confounding
        if isinstance(report.get("coefficient"), str)
    }
    batch_safety: list[BatchSafetyEvidence] = []
    for coefficient in directed_coefficients:
        if not canonical_batch_columns:
            break
        coefficient_record = coefficient_records.get(coefficient)
        report = confounding_reports.get(coefficient)
        coefficient_kind = (
            coefficient_record.get("kind") if coefficient_record is not None else None
        )
        if coefficient_kind not in {"categorical", "continuous"}:
            coefficient_kind = None
        observation_unit = (
            report.get("observationUnit")
            if report is not None
            else (
                coefficient_record.get("observationUnit")
                if coefficient_record is not None
                else None
            )
        )
        unit_constant = {
            pair.get("technical")
            for pair in (report.get("pairs", []) if report is not None else [])
            if isinstance(pair.get("technical"), str)
        }
        effective_batch_columns = [
            name for name in canonical_batch_columns if name in unit_constant
        ]
        estimability: dict[str, Any]
        if (
            coefficient_record is None
            or coefficient_record.get("scope") != "betweenUnit"
            or report is None
            or not isinstance(observation_unit, str)
            or coefficient_kind is None
        ):
            estimability = {
                "status": "notComputed",
                "reason": "unresolvedCoefficientDesign",
            }
        else:
            try:
                design = reduce_observation_units(
                    ctx.deps.cells,
                    observation_unit,
                    [coefficient, *effective_batch_columns],
                    cell_key="I",
                )
                estimability = coefficient_estimability(
                    design[coefficient].to_numpy(),
                    coefficientKind=coefficient_kind,
                    technicals={
                        name: design[name].to_numpy()
                        for name in effective_batch_columns
                    },
                    technicalKinds={
                        name: column_records[name]["kind"]
                        for name in effective_batch_columns
                    },
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.debug(
                    "Experimental Context batch estimability was not computed: "
                    f"errorType={type(exc).__name__}"
                )
                estimability = {
                    "status": "notComputed",
                    "reason": type(exc).__name__,
                }
        if estimability.get("status") != "ok":
            safety_status: BatchSafetyStatus = "notComputed"
        elif estimability.get("coefficientEstimable") is True and not bool(
            estimability.get("rankDeficient")
        ):
            safety_status = "safe"
        else:
            safety_status = "unsafe"
        batch_token = ",".join(canonical_batch_columns)
        safety = BatchSafetyEvidence(
            coefficient=coefficient,
            coefficientKind=coefficient_kind,
            observationUnit=(
                observation_unit if isinstance(observation_unit, str) else None
            ),
            batchColumns=canonical_batch_columns,
            unitConstantBatchColumns=effective_batch_columns,
            status=safety_status,
            estimability=estimability,
            evidenceId=f"batchEstimability:{coefficient}:{batch_token}",
        )
        batch_safety.append(safety)
        ctx.deps.batchSafety[safety.evidenceId] = safety

    evidence_ids.update(item.evidenceId for item in batch_safety)
    ctx.deps.evidenceIds.update(evidence_ids)
    ctx.deps.toolCalls.append("analyze_experimental_design")
    safety_counts = {
        status: sum(item.status == status for item in batch_safety)
        for status in ("safe", "unsafe", "notComputed")
    }
    logger.info(
        "Experimental Context design analysis completed: "
        f"status={characterization.status}, "
        f"batchSafetySafe={safety_counts['safe']}, "
        f"batchSafetyUnsafe={safety_counts['unsafe']}, "
        f"batchSafetyNotComputed={safety_counts['notComputed']}, "
        f"qcProfiles={len(qc_profiles)}, evidence={len(evidence_ids)}"
    )
    return CovariateEvidence(
        characterization=characterization,
        batchSafety=batch_safety,
        qcProfiles=qc_profiles,
        htoIdentityColumns=ctx.deps.htoIdentityColumns,
        htoIdentityArtifacts=ctx.deps.htoIdentityArtifacts,
        evidenceIds=sorted(evidence_ids),
    )


async def score_current_representation(
    ctx: RunContext[ExperimentalContextDependencies],
    batch_column: str,
    biological_column: str | None = None,
) -> RepresentationEvaluation:
    """Score one explicitly supplied graph without changing datastore state.

    Args:
        ctx: Pydantic AI run context containing the existing datastore.
        batch_column: Categorical technical column used to assess batch mixing.
        biological_column: Optional biological label used to assess preservation.
    """
    logger.info(
        "Experimental Context representation scoring started: "
        f"graphSupplied={ctx.deps.neighbors is not None}, "
        f"biologicalLabelSpecified={biological_column is not None}"
    )
    store = ctx.deps.store
    available_columns = set(store.cells.columns)
    if batch_column not in available_columns:
        raise ModelRetry(f"Unknown batch column {batch_column!r}")
    if biological_column is not None and biological_column not in available_columns:
        raise ModelRetry(f"Unknown biological column {biological_column!r}")
    characterization = ctx.deps.characterization
    if characterization is not None:
        batch_record = next(
            (
                record
                for record in characterization.columns
                if record.get("name") == batch_column
            ),
            None,
        )
        if (
            batch_record is None
            or batch_record.get("domain") != "technical"
            or batch_record.get("kind") != "categorical"
        ):
            raise ModelRetry(
                "Representation scoring requires a characterized categorical "
                "technical batch column"
            )

    neighbors = core_artifact_reference(ctx.deps.neighbors)
    connectivity = core_artifact_reference(ctx.deps.connectivityMap)
    if neighbors is None:
        evaluation = RepresentationEvaluation(
            cellSelection=(
                artifact_reference(ctx.deps.cellSelection)
                if ctx.deps.cellSelection is not None
                else None
            ),
            notes=["No exact neighbors artifact was supplied"],
        )
        ctx.deps.currentRepresentation = evaluation
        ctx.deps.toolCalls.append("score_current_representation")
        logger.info(
            "Experimental Context representation scoring skipped: "
            "no current neighbors artifact"
        )
        return evaluation
    if not isinstance(neighbors, ArtifactRef) or neighbors.kind != "neighbors":
        raise ModelRetry("neighbors must identify an exact neighbors artifact")
    if connectivity is not None and (
        not isinstance(connectivity, ArtifactRef)
        or connectivity.kind not in {"connectivity_map", "integrated_graph"}
    ):
        raise ModelRetry(
            "connectivity_map must identify an exact connectivity graph artifact"
        )

    metrics: dict[str, float] = {}
    notes: list[str] = []
    evidence_ids: list[str] = []
    neighbor_route = f"assay:{neighbors.assay}:neighbors:{neighbors.artifact_id}"
    try:
        value = float(store.metric_ilisi(batch_column, neighbors))
        if math.isfinite(value):
            metrics[f"iLISI:{batch_column}"] = value
            evidence_ids.append(f"metric:iLISI:{batch_column}:{neighbor_route}")
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        logger.debug(
            "Experimental Context iLISI scoring was unavailable: "
            f"errorType={type(exc).__name__}"
        )
        notes.append(f"iLISI could not be scored: {exc}")
    try:
        value = float(store.metric_proportional_batch_mixing(batch_column, neighbors))
        if math.isfinite(value):
            metrics[f"proportionalBatchMixing:{batch_column}"] = value
            evidence_ids.append(
                f"metric:proportionalBatchMixing:{batch_column}:{neighbor_route}"
            )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        logger.debug(
            "Experimental Context batch-mixing scoring was unavailable: "
            f"errorType={type(exc).__name__}"
        )
        notes.append(f"Proportional batch mixing could not be scored: {exc}")
    if biological_column is not None:
        try:
            value = float(store.metric_clisi(biological_column, neighbors))
            if math.isfinite(value):
                metrics[f"cLISI:{biological_column}"] = value
                evidence_ids.append(
                    f"metric:cLISI:{biological_column}:{neighbor_route}"
                )
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            logger.debug(
                "Experimental Context cLISI scoring was unavailable: "
                f"errorType={type(exc).__name__}"
            )
            notes.append(f"cLISI could not be scored: {exc}")
        if connectivity is not None:
            try:
                value = float(
                    store.metric_graph_connectivity(biological_column, connectivity)
                )
                if math.isfinite(value):
                    metrics[f"graphConnectivity:{biological_column}"] = value
                    evidence_ids.append(
                        "metric:graphConnectivity:"
                        f"{biological_column}:assay:{connectivity.assay}:connectivity:"
                        f"{connectivity.artifact_id}"
                    )
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                logger.debug(
                    "Experimental Context connectivity scoring was unavailable: "
                    f"errorType={type(exc).__name__}"
                )
                notes.append(f"Graph connectivity could not be scored: {exc}")

    evaluation = RepresentationEvaluation(
        available=bool(metrics),
        assay=neighbors.assay,
        cellSelection=(
            artifact_reference(ctx.deps.cellSelection)
            if ctx.deps.cellSelection is not None
            else None
        ),
        neighbors=artifact_reference(neighbors),
        connectivityMap=(
            artifact_reference(connectivity) if connectivity is not None else None
        ),
        metrics=metrics,
        notes=notes,
        evidenceIds=evidence_ids,
    )
    ctx.deps.currentRepresentation = evaluation
    ctx.deps.evidenceIds.update(evidence_ids)
    ctx.deps.toolCalls.append("score_current_representation")
    logger.info(
        "Experimental Context representation scoring completed: "
        f"available={evaluation.available}, metrics={len(metrics)}, "
        f"notes={len(notes)}, evidence={len(evidence_ids)}"
    )
    return evaluation


def _canonical_cell_qc_plan(
    plan: CellQcPlan,
    deps: ExperimentalContextDependencies,
    characterization: CovariateCharacterization,
) -> CellQcPlan:
    """Resolve one exact offered profile and reject model-authored parameters."""
    if not deps.qcProfiles:
        _offered_qc_profiles(deps, characterization)
    directed = deps.directions.get("cellQc")
    direction_map = dict(directed) if isinstance(directed, Mapping) else {}
    directed_profile_id = direction_map.get("profileId")
    if directed_profile_id is not None and not isinstance(directed_profile_id, str):
        raise ModelRetry("cellQc.profileId direction must be a string")

    has_directed_selector = any(
        key in direction_map
        for key in ("profileId", "action", "sampleColumn", "sampleArtifactName")
    )
    selected_id = directed_profile_id or (
        "" if has_directed_selector else plan.profileId
    )
    if not selected_id:
        requested_action = direction_map.get("action")
        requested_sample = direction_map.get("sampleColumn")
        requested_sample_artifact = direction_map.get("sampleArtifactName")
        if requested_sample is not None and requested_sample_artifact is not None:
            raise ModelRetry(
                "cellQc directions cannot select both sampleColumn and "
                "sampleArtifactName"
            )
        if requested_sample_artifact is not None and not isinstance(
            requested_sample_artifact, str
        ):
            raise ModelRetry("cellQc.sampleArtifactName must be a string")
        if requested_action is not None and requested_action not in {
            "skip",
            "globalGaussian",
            "sampleMad",
        }:
            raise ModelRetry(f"Unsupported cellQc.action {requested_action!r}")
        matches = [
            profile
            for profile in deps.qcProfiles.values()
            if (requested_action is None or profile.action == requested_action)
            and (requested_sample is None or profile.sampleColumn == requested_sample)
            and (
                requested_sample_artifact is None
                or (
                    profile.sampleArtifact is not None
                    and profile.sampleArtifact.name == requested_sample_artifact
                )
            )
        ]
        if requested_action is not None:
            if len(matches) != 1:
                raise ModelRetry(
                    "cellQc directions must identify exactly one offered profile"
                )
            selected_id = matches[0].profileId
        else:
            global_profiles = [
                profile
                for profile in deps.qcProfiles.values()
                if profile.action == "globalGaussian"
            ]
            if global_profiles:
                selected_id = global_profiles[0].profileId
            else:
                selected_id = next(
                    profile.profileId
                    for profile in deps.qcProfiles.values()
                    if profile.action == "skip"
                )

    profile = deps.qcProfiles.get(selected_id)
    if profile is None:
        raise ModelRetry(
            f"Cell-QC profile {selected_id!r} was not offered by the evidence tool"
        )
    model_selected = bool(plan.profileId) and not has_directed_selector
    if model_selected:
        expected_fields = {
            "action": profile.action,
            "driverAssay": profile.driverAssay,
            "driverAssayType": profile.driverAssayType,
            "sampleColumn": profile.sampleColumn,
            "sampleArtifact": profile.sampleArtifact,
            "attributes": profile.attributes,
            "artifactMetrics": profile.artifactMetrics,
        }
        mismatches = [
            name
            for name, expected in expected_fields.items()
            if getattr(plan, name) != expected
        ]
        if mismatches:
            raise ModelRetry(
                "Cell-QC plan must copy the selected offered profile exactly: "
                f"{mismatches}"
            )
        if profile.evidenceId not in plan.evidenceIds:
            raise ModelRetry(
                "Cell-QC plan must cite its exact profile retention evidence"
            )
    rationale = plan.rationale.strip()
    if not rationale:
        rationale = (
            "Selected the caller-directed bounded cell-QC profile."
            if direction_map
            else "Selected the bounded default cell-QC profile."
        )
    cited_evidence = plan.evidenceIds if model_selected else []
    return CellQcPlan(
        action=profile.action,
        profileId=profile.profileId,
        driverAssay=profile.driverAssay,
        driverAssayType=profile.driverAssayType,
        sampleColumn=profile.sampleColumn,
        sampleArtifact=profile.sampleArtifact,
        attributes=profile.attributes,
        artifactMetrics=profile.artifactMetrics,
        rationale=rationale,
        evidenceIds=sorted({*cited_evidence, profile.evidenceId}),
    )


def _validate_batch_correction_plan(
    decision: ExperimentalContextDecision,
    deps: ExperimentalContextDependencies,
    characterization: CovariateCharacterization,
    requested_coefficients: set[str],
    units_of_inference: dict[str, dict[str, Any]],
    cell_qc_plan: CellQcPlan,
    records: dict[str, dict[str, Any]],
    coefficient_records: dict[str, dict[str, Any]],
) -> None:
    """Validate one batch plan against exact design, safety, and metric evidence."""
    confounding_reports = {
        report.get("coefficient"): report
        for report in characterization.confounding
        if isinstance(report.get("coefficient"), str)
    }
    plan = decision.batchCorrection
    unknown_columns = sorted(set(decision.columnDomains) - set(records))
    if unknown_columns:
        raise ModelRetry(f"Unknown column domain assignments: {unknown_columns}")
    unit_columns = {
        unit_name
        for unit in units_of_inference.values()
        for unit_name in (
            unit.get("observationUnit"),
            unit.get("independentUnit"),
        )
        if isinstance(unit_name, str)
    }
    if plan.action == "evaluateHarmony" and not plan.batchColumns:
        raise ModelRetry("evaluateHarmony requires at least one batch column")
    if plan.action == "unsafe" and not plan.batchColumns:
        raise ModelRetry("unsafe requires the exact batch columns that were assessed")
    if plan.action == "skip" and plan.batchColumns:
        raise ModelRetry("skip must not include batch columns")
    if plan.action == "needsInput" and not decision.needsInput:
        raise ModelRetry("needsInput action requires at least one concrete question")
    if len(set(plan.batchColumns)) != len(plan.batchColumns):
        raise ModelRetry("Batch columns must be unique")

    for batch_column in plan.batchColumns:
        record = records.get(batch_column)
        if record is None:
            raise ModelRetry(f"Unknown batch column {batch_column!r}")
        if record.get("domain") != "technical":
            raise ModelRetry(
                f"Batch column {batch_column!r} must be classified as technical"
            )
        if record.get("kind") != "categorical":
            raise ModelRetry(
                f"Batch column {batch_column!r} must be categorical for Harmony"
            )
        if batch_column in requested_coefficients or batch_column in unit_columns:
            raise ModelRetry(
                f"Batch column {batch_column!r} cannot be a coefficient or unit of inference"
            )

    if plan.action == "evaluateHarmony":
        mixing_metrics = {"iLISI", "proportionalBatchMixing"}
        preservation_metrics = {"cLISI", "graphConnectivity"}
        if not mixing_metrics.intersection(plan.metricsRequired):
            raise ModelRetry(
                "evaluateHarmony requires iLISI or proportionalBatchMixing"
            )
        if plan.preserveColumns and not preservation_metrics.intersection(
            plan.metricsRequired
        ):
            raise ModelRetry(
                "evaluateHarmony requires cLISI or graphConnectivity for preservation"
            )
        missing_preserve = sorted(requested_coefficients - set(plan.preserveColumns))
        if missing_preserve:
            raise ModelRetry(
                "preserveColumns must include every coefficient of interest: "
                f"{missing_preserve}"
            )
        unresolved_coefficients = sorted(
            coefficient
            for coefficient in requested_coefficients
            if coefficient_records[coefficient].get("scope") != "betweenUnit"
            or coefficient not in confounding_reports
        )
        if unresolved_coefficients:
            raise ModelRetry(
                "evaluateHarmony requires a between-unit coefficient with a "
                "matching estimability report; use needsInput or unsafe for: "
                f"{unresolved_coefficients}"
            )
        for preserve_column in plan.preserveColumns:
            record = records.get(preserve_column)
            if record is None:
                raise ModelRetry(f"Unknown preservation column {preserve_column!r}")
            if record.get("domain") != "biological":
                raise ModelRetry(
                    f"Preservation column {preserve_column!r} must be biological"
                )
            if record.get("kind") != "categorical":
                raise ModelRetry(
                    f"Preservation column {preserve_column!r} must be categorical"
                )

    matched_safety: list[BatchSafetyEvidence] = []
    if plan.action in {"evaluateHarmony", "unsafe"}:
        canonical_batch_columns = sorted(plan.batchColumns)
        for coefficient in sorted(requested_coefficients):
            coefficient_record = coefficient_records[coefficient]
            report = confounding_reports.get(coefficient)
            observation_unit = (
                report.get("observationUnit")
                if report is not None
                else coefficient_record.get("observationUnit")
            )
            unit_constant = {
                pair.get("technical")
                for pair in (report.get("pairs", []) if report is not None else [])
                if isinstance(pair.get("technical"), str)
            }
            expected_effective = [
                name for name in canonical_batch_columns if name in unit_constant
            ]
            candidates = [
                item
                for item in deps.batchSafety.values()
                if item.coefficient == coefficient
                and item.coefficientKind == coefficient_record.get("kind")
                and item.observationUnit == observation_unit
                and item.batchColumns == canonical_batch_columns
                and item.unitConstantBatchColumns == expected_effective
            ]
            if len(candidates) != 1:
                raise ModelRetry(
                    "Call analyze_experimental_design with the exact proposed batch "
                    f"columns before returning a recommendation for {coefficient!r}"
                )
            matched_safety.append(candidates[0])
        missing_safety_evidence = sorted(
            item.evidenceId
            for item in matched_safety
            if item.evidenceId not in plan.evidenceIds
        )
        if missing_safety_evidence:
            raise ModelRetry(
                "Batch-correction recommendations must cite exact batch "
                f"estimability evidence: {missing_safety_evidence}"
            )
        not_computed = [
            item.coefficient for item in matched_safety if item.status == "notComputed"
        ]
        if not_computed:
            raise ModelRetry(
                "Batch estimability could not be computed; use action='needsInput' "
                f"for: {sorted(not_computed)}"
            )
        unsafe_coefficients = [
            item.coefficient for item in matched_safety if item.status == "unsafe"
        ]
        if plan.action == "evaluateHarmony" and unsafe_coefficients:
            raise ModelRetry(
                "Batch correction is unsafe because the biological coefficient is "
                "not estimable after the exact proposed batch columns; use "
                f"action='unsafe' for: {sorted(unsafe_coefficients)}"
            )
        if plan.action == "unsafe" and not unsafe_coefficients:
            raise ModelRetry(
                "The exact proposed batch columns were estimable for every "
                "coefficient; use action='evaluateHarmony' or 'skip'"
            )

    cited_ids = [
        *decision.evidenceIds,
        *plan.evidenceIds,
        *cell_qc_plan.evidenceIds,
    ]
    unknown_evidence = sorted(set(cited_ids) - deps.evidenceIds)
    if unknown_evidence:
        raise ModelRetry(f"Unknown evidence IDs: {unknown_evidence}")
    if plan.action in {"evaluateHarmony", "skip", "unsafe"} and not plan.evidenceIds:
        raise ModelRetry("Batch-correction recommendations require evidence IDs")
    current_metric_evidence = set(deps.currentRepresentation.evidenceIds)
    stale_metric_evidence = sorted(
        evidence_id
        for evidence_id in cited_ids
        if evidence_id.startswith("metric:")
        and evidence_id not in current_metric_evidence
    )
    if stale_metric_evidence:
        raise ModelRetry(
            "Metric evidence must come from the returned exact representation: "
            f"{stale_metric_evidence}"
        )


def validate_experimental_context(
    decision: ExperimentalContextDecision,
    deps: ExperimentalContextDependencies,
) -> ExperimentalContextDecision:
    """Recompute and validate every model-authored design choice."""
    narrative_fields = {
        "rationale": decision.rationale,
        "batchCorrection.rationale": decision.batchCorrection.rationale,
        "cellQc.rationale": decision.cellQc.rationale,
        **{
            f"needsInput[{index}]": question
            for index, question in enumerate(decision.needsInput)
        },
    }
    serialized_field_markers = (
        '"evidenceIds":',
        '"needsInput":',
        '"runInfo":',
        '"batchCorrection":',
        '"cellQc":',
    )
    invalid_narratives = [
        name
        for name, value in narrative_fields.items()
        if any(
            marker in value.replace('\\"', '"') for marker in serialized_field_markers
        )
    ]
    if invalid_narratives:
        raise ModelRetry(
            "Narrative fields must contain plain prose without serialized sibling "
            f"fields: {invalid_narratives}"
        )
    directions = dict(deps.directions)
    column_domains = dict(decision.columnDomains)
    column_domains.update(dict(directions.get("columnDomains") or {}))
    directions["columnDomains"] = column_domains
    directions["coefficientsOfInterest"] = list(
        dict.fromkeys(
            [
                *decision.coefficientsOfInterest,
                *(directions.get("coefficientsOfInterest") or []),
            ]
        )
    )
    units_of_inference = {
        name: unit.model_dump(exclude_none=True)
        for name, unit in decision.unitsOfInference.items()
    }
    units_of_inference.update(dict(directions.get("unitsOfInference") or {}))
    directions["unitsOfInference"] = units_of_inference

    characterization = characterize_covariates(
        deps.store,
        cellSelection=deps.cellSelection,
        studyContext=deps.studyContext,
        model=None,
        directions=directions,
        groupingArtifacts=_hto_artifact_map(deps),
    )
    if characterization.status == "failed":
        raise ModelRetry("; ".join(characterization.notes))
    deps.characterization = characterization
    deps.evidenceIds.update(characterization_evidence(characterization))

    if "inspect_cell_covariates" not in deps.toolCalls:
        raise ModelRetry("Call inspect_cell_covariates before returning a decision")
    if "analyze_experimental_design" not in deps.toolCalls:
        raise ModelRetry("Call analyze_experimental_design before returning a decision")

    cell_qc_plan = _canonical_cell_qc_plan(
        decision.cellQc,
        deps,
        characterization,
    )
    deps.evidenceIds.update(profile.evidenceId for profile in deps.qcProfiles.values())

    requested_coefficients = set(directions["coefficientsOfInterest"])
    characterized_coefficients = {
        record.get("name") for record in characterization.coefficients
    }
    missing_coefficients = sorted(
        name
        for name in requested_coefficients
        if name not in characterized_coefficients
    )
    if missing_coefficients:
        raise ModelRetry(
            "Coefficients of interest must be classified as biological: "
            f"{missing_coefficients}"
        )

    coefficient_records: dict[str, dict[str, Any]] = {}
    for record in characterization.coefficients:
        name = record.get("name")
        if isinstance(name, str):
            coefficient_records[name] = record
    records: dict[str, dict[str, Any]] = {}
    for record in characterization.columns:
        name = record.get("name")
        if isinstance(name, str):
            records[name] = record
    _validate_batch_correction_plan(
        decision,
        deps,
        characterization,
        requested_coefficients,
        units_of_inference,
        cell_qc_plan,
        records,
        coefficient_records,
    )
    canonical_domains = {
        name: records[name]["domain"]
        for name in column_domains
        if name in records
        and records[name].get("domain")
        in {
            "biological",
            "technical",
            "design",
            "ignore",
            "unknown",
        }
    }
    canonical_units = {
        coefficient: InferenceUnit(
            observationUnit=coefficient_records[coefficient].get("observationUnit"),
            independentUnit=coefficient_records[coefficient].get("independentUnit"),
        )
        for coefficient in directions["coefficientsOfInterest"]
        if coefficient in coefficient_records
    }
    validated = decision.model_copy(
        update={
            "columnDomains": canonical_domains,
            "coefficientsOfInterest": list(directions["coefficientsOfInterest"]),
            "unitsOfInference": canonical_units,
            "cellQc": cell_qc_plan,
        }
    )
    logger.debug(
        "Experimental Context decision validated: "
        f"domains={len(validated.columnDomains)}, "
        f"coefficients={len(validated.coefficientsOfInterest)}, "
        f"cellQc={validated.cellQc.action}, "
        f"batchCorrection={validated.batchCorrection.action}, "
        f"needsInput={len(validated.needsInput)}"
    )
    return validated


def fallback_experimental_context_result(
    deps: ExperimentalContextDependencies,
    *,
    error: UnexpectedModelBehavior,
    model_name: str,
) -> ExperimentalContextResult:
    """Continue conservatively when the model exhausts its correction budget."""
    characterization = deps.characterization
    if characterization is None:
        characterization = characterize_covariates(
            deps.store,
            cellSelection=deps.cellSelection,
            studyContext=deps.studyContext,
            model=None,
            directions=deps.directions,
            groupingArtifacts=_hto_artifact_map(deps),
        )
        deps.characterization = characterization
    if not deps.htoIdentityColumns:
        deps.htoIdentityColumns = _hto_identity_columns(deps)
    qc_profiles = list(deps.qcProfiles.values())
    if not qc_profiles:
        qc_profiles = _offered_qc_profiles(deps, characterization)
    cell_qc = _canonical_cell_qc_plan(
        CellQcPlan.get_blank(),
        deps,
        characterization,
    )
    evidence_ids = characterization_evidence(characterization)
    evidence_ids.update(profile.evidenceId for profile in qc_profiles)
    evidence_ids.update(f"htoIdentity:{column}" for column in deps.htoIdentityColumns)
    evidence_ids.update(
        _artifact_evidence_id(source) for source in deps.htoIdentityArtifacts
    )
    deps.evidenceIds.update(evidence_ids)
    column_domains = {
        str(record["name"]): record["domain"]
        for record in characterization.columns
        if isinstance(record.get("name"), str)
        and record.get("domain")
        in {"biological", "technical", "design", "ignore", "unknown"}
    }
    coefficient_records = {
        str(record["name"]): record
        for record in characterization.coefficients
        if isinstance(record.get("name"), str)
    }
    coefficients = list(coefficient_records)
    units = {
        coefficient: InferenceUnit(
            observationUnit=record.get("observationUnit"),
            independentUnit=record.get("independentUnit"),
        )
        for coefficient, record in coefficient_records.items()
    }
    batch_evidence = sorted(
        f"column:{name}"
        for name, domain in column_domains.items()
        if domain == "technical"
    )
    if not batch_evidence:
        batch_evidence = sorted(
            evidence_id
            for evidence_id in evidence_ids
            if evidence_id.startswith("column:")
        )[:1]
    limitation = (
        "The model exhausted its bounded correction budget while proposing the "
        "experimental design. Harmony was skipped because no model proposal was "
        "accepted as a categorical technical batch design."
    )
    decision = ExperimentalContextDecision(
        columnDomains=column_domains,
        coefficientsOfInterest=coefficients,
        unitsOfInference=units,
        batchCorrection=BatchCorrectionPlan(
            action="skip",
            rationale=(
                "Use the native representation because bounded validation did not "
                "authorize a safe Harmony batch column."
            ),
            evidenceIds=batch_evidence,
        ),
        cellQc=cell_qc,
        rationale=(
            "Retained deterministic metadata characterization and the exact bounded "
            "cell-QC profile, while declining an unvalidated batch-correction choice."
        ),
        evidenceIds=sorted(evidence_ids),
    )
    error_detail = str(error).replace("\n", " ").strip()[:500]
    status: StageStatus = "failed" if characterization.status == "failed" else "done"
    logger.warning(
        "Experimental Context used its conservative fallback: "
        f"status={status}, cellQc={cell_qc.action}, coefficients={len(coefficients)}, "
        f"reason={error_detail}"
    )
    return ExperimentalContextResult(
        status=status,
        decision=decision,
        characterization=characterization,
        cellSelection=artifact_reference(deps.cellSelection),
        cellQc=cell_qc,
        qcProfiles=qc_profiles,
        qualityMetricArtifacts=deps.qualityMetricArtifacts,
        htoIdentityColumns=deps.htoIdentityColumns,
        htoIdentityArtifacts=deps.htoIdentityArtifacts,
        batchSafety=list(deps.batchSafety.values()),
        currentRepresentation=deps.currentRepresentation,
        notes=[*characterization.notes, limitation, error_detail],
        runInfo=AgentRunInfo(
            agentName="experimental_context_fallback",
            modelName=model_name,
        ),
    )


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
            request_limit=9,
            tool_call_limit=5,
            output_token_limit=32768,
            timeout_seconds=600.0,
        )
        self.system_prompt = (
            dedent(
                """
            You are Scarf's Experimental Context Agent. Work only through the
            provided read-only tools and return the structured decision schema.

            Call inspect_cell_covariates exactly once. Then call
            analyze_experimental_design exactly once with all explicit domains,
            all biological coefficients, every unit of inference, and the complete
            exact batch-column set being considered. You may call
            score_current_representation at most once when an exact supplied graph
            can add evidence. Do not split metadata, coefficients, or batch columns across
            calls, and do not repeat a tool call. Pass batch_columns as a JSON array,
            including when the array contains exactly one column. Each tool is
            removed after it succeeds, so include the complete decision context in
            its single call.

            The tools return bounded cell-QC profiles projected against the exact
            shared cell selection. Select one returned profileId, copy its action,
            driver assay name and type, metadata attributes, artifact metrics, and
            sample source exactly, and cite its evidenceId. RNA is the preferred
            QC driver and ATAC is the fallback. ADT and HTO never drive automatic
            cell filtering. An exact HTO identity artifact may be used as sample
            or grouping evidence. It is not a live metadata column and does not
            make HTO a QC driver.

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

            Cite only evidenceIds returned by tools. Ask for input when study
            design cannot be resolved. Never propose Python, shell commands,
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
        cell_selection: ArtifactRef | None = None,
        directions: Mapping[str, Any] | None = None,
        run: "PipelineRun | None" = None,
        neighbors: ArtifactRef | None = None,
        connectivity_map: ArtifactRef | None = None,
        quality_metric_artifacts: Sequence[NamedArtifactSource] = (),
        hto_identity_artifacts: Sequence[NamedArtifactSource] = (),
    ) -> ExperimentalContextResult:
        """Inspect one datastore and return a validated experimental-context report."""
        study_context = (study_context or "").strip()
        if len(study_context) > _CONTEXT_LIMIT:
            study_context = study_context[: _CONTEXT_LIMIT - 3] + "..."
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
        quality_sources = list(quality_metric_artifacts)
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
            f"studyContextProvided={bool(study_context)}"
        )
        deps = ExperimentalContextDependencies(
            store=store,
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
            directions=direction_map,
            qualityMetricArtifacts=quality_sources,
            htoIdentityArtifacts=hto_sources,
        )
        user_prompt = (
            dedent(
                """
                Characterize this experiment's metadata, select one offered cell-QC
                profile, and decide whether Harmony should be evaluated.

                Study context: {study_context}
                Exact cell-selection artifact: {cell_selection}
                Exact quality-metric artifacts: {quality_metrics}
                Exact HTO identity artifacts: {hto_identities}
                Caller directions: {directions}
                """
            )
            .strip()
            .format(
                study_context=study_context or "not provided",
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
        try:
            execution = run_agent_sync(
                model=self.model,
                output_type=ExperimentalContextDecision,
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
                tools=(
                    Tool(
                        inspect_cell_covariates,
                        prepare=_prepare_experimental_context_tool,
                        sequential=self.config.sequentialTools,
                        timeout=self.config.timeoutSeconds,
                    ),
                    Tool(
                        analyze_experimental_design,
                        max_retries=1,
                        prepare=_prepare_experimental_context_tool,
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
                output_validator=lambda decision: validate_experimental_context(
                    decision,
                    deps,
                ),
            )
        except UnexpectedModelBehavior as exc:
            model_name = getattr(self.model, "model_name", type(self.model).__name__)
            return fallback_experimental_context_result(
                deps,
                error=exc,
                model_name=str(model_name),
            )
        decision = ExperimentalContextDecision.model_validate(execution.output)
        characterization = deps.characterization
        if characterization is None:
            characterization = characterize_covariates(
                store,
                cellSelection=cell_selection,
                studyContext=study_context,
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
            f"status={status}, cellQc={decision.cellQc.action}, "
            f"batchCorrection={decision.batchCorrection.action}, "
            f"coefficients={len(decision.coefficientsOfInterest)}, "
            f"toolCalls={len(deps.toolCalls)}, evidence={len(deps.evidenceIds)}"
        )
        return ExperimentalContextResult(
            status=status,
            decision=decision,
            characterization=characterization,
            cellSelection=artifact_reference(cell_selection),
            cellQc=decision.cellQc,
            qcProfiles=list(deps.qcProfiles.values()),
            qualityMetricArtifacts=deps.qualityMetricArtifacts,
            htoIdentityColumns=deps.htoIdentityColumns,
            htoIdentityArtifacts=deps.htoIdentityArtifacts,
            batchSafety=list(deps.batchSafety.values()),
            currentRepresentation=deps.currentRepresentation,
            notes=[*characterization.notes, *decision.needsInput],
            runInfo=execution.runInfo,
        )
