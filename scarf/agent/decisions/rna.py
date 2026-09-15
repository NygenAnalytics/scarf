"""Deterministic RNA decision definitions and executor-owned option payloads."""

from collections.abc import Mapping, Sequence
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from ..cell_quality.profiles import CellQualityProfile
from ..types import AgentDataModel
from .kernel import (
    DecisionOption,
    DecisionRecord,
    DecisionSpec,
    DecisionStatus,
    DeterministicDecisionAuditor,
    EvidenceBundle,
    VerificationCheck,
)

type RnaDecisionCheckpoint = Literal["qcGrouping", "cellQuality"]
type QcGroupingMode = Literal["global", "physicalCapture", "pooledReference"]


class RnaDecisionGateError(ValueError):
    """Raised when deterministic evidence forbids constructing an option set."""


class RnaDecisionCompilationError(ValueError):
    """Raised when an unverified decision cannot compile to an executor payload."""


class RnaRegistryModel(AgentDataModel):
    """Base for immutable, closed RNA registry contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class CellQualityExecutorPayload(RnaRegistryModel):
    """Exact cell-quality thresholds owned by the deterministic executor."""

    operation: Literal["cellQualityProfile"] = "cellQualityProfile"
    profile: CellQualityProfile
    lowerCountMad: float | None = Field(default=None, ge=0, strict=True)
    lowerFeatureMad: float | None = Field(default=None, ge=0, strict=True)
    upperMitoMad: float | None = Field(default=None, ge=0, strict=True)
    groupByCapture: bool = Field(strict=True)
    pooledReference: bool = Field(strict=True)
    sensitivityOnly: bool = Field(strict=True)
    flagHighCounts: Literal[True] = True
    flagHighFeatures: Literal[True] = True

    @model_validator(mode="after")
    def validate_profile(self) -> "CellQualityExecutorPayload":
        thresholds = (self.lowerCountMad, self.lowerFeatureMad, self.upperMitoMad)
        if self.profile == "retainWithFlags":
            if any(value is not None for value in thresholds):
                raise ValueError("retainWithFlags cannot define removal thresholds")
            if self.groupByCapture or self.pooledReference or self.sensitivityOnly:
                raise ValueError("retainWithFlags cannot enable filtering modes")
            return self
        if self.profile in {"coreGlobalGaussian", "coreSampleMad3"}:
            if any(value is not None for value in thresholds):
                raise ValueError(
                    "Core profiles use exact core bounds, not one-sided MAD fields"
                )
            if self.groupByCapture != (self.profile == "coreSampleMad3"):
                raise ValueError("The core sample profile requires capture grouping")
            if self.pooledReference or self.sensitivityOnly:
                raise ValueError(
                    "Core profiles cannot enable registered filtering modes"
                )
            return self
        if any(value is None for value in thresholds):
            raise ValueError("Filtering profiles require all three MAD thresholds")
        if self.profile.startswith("captureMad") != self.groupByCapture:
            raise ValueError("Capture profiles and groupByCapture must agree")
        if (self.profile == "pooledReferenceMad5") != self.pooledReference:
            raise ValueError("pooledReferenceMad5 and pooledReference must agree")
        if (self.profile == "captureMad3Sensitivity") != self.sensitivityOnly:
            raise ValueError("captureMad3Sensitivity and sensitivityOnly must agree")
        return self


class QcGroupingExecutorPayload(RnaRegistryModel):
    """Exact population used to estimate registered cell-quality thresholds."""

    operation: Literal["qcGrouping"] = "qcGrouping"
    groupingMode: QcGroupingMode


class NoExecutionPayload(RnaRegistryModel):
    """Typed terminal or pause outcome with no analytical operation."""

    operation: Literal["noExecution"] = "noExecution"
    reasonCode: Literal[
        "needsInput",
        "scientificAbstention",
    ]


type RnaOptionPayload = Annotated[
    QcGroupingExecutorPayload | CellQualityExecutorPayload | NoExecutionPayload,
    Field(discriminator="operation"),
]


class RnaExecutorOption(RnaRegistryModel):
    """Executor-only payload keyed by the semantic option shown to an agent."""

    checkpoint: RnaDecisionCheckpoint
    optionId: str
    payload: RnaOptionPayload

    @field_validator("optionId")
    @classmethod
    def validate_option_id(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError(
                "optionId must be non-empty without surrounding whitespace"
            )
        return value


class RnaDecisionDefinition(RnaRegistryModel):
    """Agent-visible decision spec paired with executor-only payloads."""

    checkpoint: RnaDecisionCheckpoint
    spec: DecisionSpec
    executorOptions: list[RnaExecutorOption] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_exact_registry(self) -> "RnaDecisionDefinition":
        if self.spec.checkpoint != self.checkpoint:
            raise ValueError(
                "DecisionSpec checkpoint must match the registry checkpoint"
            )
        visible_ids = [option.optionId for option in self.spec.options]
        executor_ids = [option.optionId for option in self.executorOptions]
        if len(executor_ids) != len(set(executor_ids)):
            raise ValueError("executorOptions must not contain duplicate option IDs")
        if executor_ids != visible_ids:
            raise ValueError(
                "executorOptions must exactly match the ordered visible option IDs"
            )
        if any(option.checkpoint != self.checkpoint for option in self.executorOptions):
            raise ValueError("Every executor option must match the registry checkpoint")
        return self

    def executor_option(self, option_id: str) -> RnaExecutorOption:
        """Return one exact registered executor option."""
        for option in self.executorOptions:
            if option.optionId == option_id:
                return option
        raise KeyError(f"Unknown option ID for {self.spec.decisionId}: {option_id}")


class CompiledRnaDecision(RnaRegistryModel):
    """Verified executor handoff kept separate from agent-authored records."""

    decisionRecordId: str
    decisionId: str
    selectedOptionId: str
    status: DecisionStatus
    executorPayload: RnaOptionPayload
    checks: list[VerificationCheck]


def compile_rna_decision(
    definition: RnaDecisionDefinition,
    evidence: EvidenceBundle,
    record: DecisionRecord,
    *,
    created_at_ns: int = 0,
) -> CompiledRnaDecision:
    """Audit an exact selection and resolve its executor-owned payload."""
    verification = DeterministicDecisionAuditor.audit(
        definition.spec,
        evidence,
        record,
        created_at_ns=created_at_ns,
    )
    failed_checks = [
        check.checkId for check in verification if check.status == "failed"
    ]
    if failed_checks:
        raise RnaDecisionCompilationError(
            "Decision failed deterministic verification: " + ", ".join(failed_checks)
        )
    executor_option = definition.executor_option(record.selectedOptionId)
    return CompiledRnaDecision(
        decisionRecordId=record.recordId,
        decisionId=record.decisionId,
        selectedOptionId=record.selectedOptionId,
        status=record.status,
        executorPayload=executor_option.payload,
        checks=verification,
    )


def _defer_option(
    checkpoint: RnaDecisionCheckpoint, option_id: str
) -> tuple[DecisionOption, RnaExecutorOption]:
    return (
        DecisionOption(
            optionId=option_id,
            status="defer",
            label="Request input",
            description="Pause because the available evidence cannot identify a choice.",
        ),
        RnaExecutorOption(
            checkpoint=checkpoint,
            optionId=option_id,
            payload=NoExecutionPayload(reasonCode="needsInput"),
        ),
    )


def _build_definition(
    *,
    checkpoint: RnaDecisionCheckpoint,
    decision_id: str,
    evidence_bundle_id: str,
    question: str,
    visible_options: list[DecisionOption],
    executor_options: list[RnaExecutorOption],
    baseline_option_id: str | None,
) -> RnaDecisionDefinition:
    return RnaDecisionDefinition(
        checkpoint=checkpoint,
        spec=DecisionSpec(
            decisionId=decision_id,
            definitionVersion=1,
            checkpoint=checkpoint,
            question=question,
            evidenceBundleId=evidence_bundle_id,
            options=visible_options,
            baselineOptionId=baseline_option_id,
        ),
        executorOptions=executor_options,
    )


def require_option_evidence(
    definition: RnaDecisionDefinition,
    requirements: Mapping[str, Sequence[str]],
) -> RnaDecisionDefinition:
    """Bind exact evidence IDs to the registered options they support."""
    options = definition.spec.option_by_id()
    unknown = sorted(set(requirements) - set(options))
    if unknown:
        raise ValueError(f"Evidence requirements name unknown options: {unknown}")
    updated: list[DecisionOption] = []
    for option in definition.spec.options:
        raw_ids = requirements.get(option.optionId, ())
        if isinstance(raw_ids, str | bytes):
            raise TypeError("Option evidence requirements must be sequences of IDs")
        updated.append(
            option.model_copy(
                update={"requiredEvidenceIds": list(dict.fromkeys(raw_ids))}
            )
        )
    spec = definition.spec.model_copy(update={"options": updated})
    return definition.model_copy(update={"spec": spec})


def build_qc_grouping_decision(
    *,
    evidence_bundle_id: str,
    physical_capture_eligible: bool,
    pooled_reference_eligible: bool,
) -> RnaDecisionDefinition:
    """Build only quality-threshold grouping modes licensed by design evidence."""
    if pooled_reference_eligible and not physical_capture_eligible:
        raise RnaDecisionGateError(
            "Pooled-reference QC requires eligible physical captures"
        )
    rows: list[tuple[str, str, str, QcGroupingMode]] = [
        (
            "qcGrouping:global",
            "Global reference",
            "Estimate registered quality boundaries across the selected dataset.",
            "global",
        )
    ]
    if physical_capture_eligible:
        rows.append(
            (
                "qcGrouping:physicalCapture",
                "Physical-capture references",
                "Estimate registered boundaries separately within proven captures.",
                "physicalCapture",
            )
        )
    if pooled_reference_eligible:
        rows.append(
            (
                "qcGrouping:pooledReference",
                "Comparable pooled reference",
                "Estimate registered boundaries from the exact comparable captures.",
                "pooledReference",
            )
        )
    visible = [
        DecisionOption(
            optionId=option_id,
            status="apply",
            label=label,
            description=description,
            requiredEvidenceClasses=["qualityControl", "design"],
        )
        for option_id, label, description, _mode in rows
    ]
    executor = [
        RnaExecutorOption(
            checkpoint="qcGrouping",
            optionId=option_id,
            payload=QcGroupingExecutorPayload(groupingMode=mode),
        )
        for option_id, _label, _description, mode in rows
    ]
    defer_visible, defer_executor = _defer_option("qcGrouping", "qcGrouping:defer")
    visible.append(defer_visible)
    executor.append(defer_executor)
    return _build_definition(
        checkpoint="qcGrouping",
        decision_id="qcGrouping",
        evidence_bundle_id=evidence_bundle_id,
        question="Which reference population should define cell-quality boundaries?",
        visible_options=visible,
        executor_options=executor,
        baseline_option_id="qcGrouping:global",
    )


def build_cell_quality_decision(
    *,
    evidence_bundle_id: str,
    available_profiles: Sequence[CellQualityProfile],
) -> RnaDecisionDefinition:
    """Build exactly the cell-quality profiles licensed by grouping evidence."""
    if len(available_profiles) != len(set(available_profiles)):
        raise RnaDecisionGateError("Cell-quality profiles must not contain duplicates")
    if not available_profiles:
        raise RnaDecisionGateError("At least one cell-quality profile is required")
    definitions: dict[
        CellQualityProfile,
        tuple[str, DecisionStatus, str, str, CellQualityExecutorPayload],
    ] = {
        "retainWithFlags": (
            "cellQuality:retainWithFlags",
            "skip",
            "Retain with flags",
            "Preserve the published cell set and retain diagnostic quality flags.",
            CellQualityExecutorPayload(
                profile="retainWithFlags",
                groupByCapture=False,
                pooledReference=False,
                sensitivityOnly=False,
            ),
        ),
        "coreGlobalGaussian": (
            "cellQuality:coreGlobalGaussian",
            "apply",
            "Scarf default global filter",
            "Use Scarf's default global Gaussian quantiles (0.01 and 0.99).",
            CellQualityExecutorPayload(
                profile="coreGlobalGaussian",
                groupByCapture=False,
                pooledReference=False,
                sensitivityOnly=False,
            ),
        ),
        "coreSampleMad3": (
            "cellQuality:coreSampleMad3",
            "apply",
            "Scarf sample-aware default",
            "Use Scarf's exact sample-aware filter with three scaled MADs within proven physical captures.",
            CellQualityExecutorPayload(
                profile="coreSampleMad3",
                groupByCapture=True,
                pooledReference=False,
                sensitivityOnly=False,
            ),
        ),
        "globalMad5": (
            "cellQuality:globalMad5",
            "apply",
            "Global lenient filter",
            "Apply one-sided global five-MAD quality boundaries.",
            CellQualityExecutorPayload(
                profile="globalMad5",
                lowerCountMad=5.0,
                lowerFeatureMad=5.0,
                upperMitoMad=5.0,
                groupByCapture=False,
                pooledReference=False,
                sensitivityOnly=False,
            ),
        ),
        "captureMad5": (
            "cellQuality:captureMad5",
            "apply",
            "Capture-aware lenient filter",
            "Apply one-sided five-MAD boundaries within physical captures.",
            CellQualityExecutorPayload(
                profile="captureMad5",
                lowerCountMad=5.0,
                lowerFeatureMad=5.0,
                upperMitoMad=5.0,
                groupByCapture=True,
                pooledReference=False,
                sensitivityOnly=False,
            ),
        ),
        "captureMad3Sensitivity": (
            "cellQuality:captureMad3Sensitivity",
            "apply",
            "Capture sensitivity branch",
            "Evaluate stricter three-MAD capture boundaries as sensitivity evidence.",
            CellQualityExecutorPayload(
                profile="captureMad3Sensitivity",
                lowerCountMad=3.0,
                lowerFeatureMad=3.0,
                upperMitoMad=3.0,
                groupByCapture=True,
                pooledReference=False,
                sensitivityOnly=True,
            ),
        ),
        "pooledReferenceMad5": (
            "cellQuality:pooledReferenceMad5",
            "apply",
            "Pooled-reference filter",
            "Apply five-MAD boundaries from comparable pooled reference captures.",
            CellQualityExecutorPayload(
                profile="pooledReferenceMad5",
                lowerCountMad=5.0,
                lowerFeatureMad=5.0,
                upperMitoMad=5.0,
                groupByCapture=False,
                pooledReference=True,
                sensitivityOnly=False,
            ),
        ),
    }
    profiles = [definitions[profile] for profile in available_profiles]

    visible = [
        DecisionOption(
            optionId=option_id,
            status=status,
            label=label,
            description=description,
            requiredEvidenceClasses=["qualityControl"],
        )
        for option_id, status, label, description, _payload in profiles
    ]
    executor = [
        RnaExecutorOption(checkpoint="cellQuality", optionId=option_id, payload=payload)
        for option_id, _status, _label, _description, payload in profiles
    ]
    defer_visible, defer_executor = _defer_option("cellQuality", "cellQuality:defer")
    visible.append(defer_visible)
    executor.append(defer_executor)
    return _build_definition(
        checkpoint="cellQuality",
        decision_id="cellQuality",
        evidence_bundle_id=evidence_bundle_id,
        question="Which cell-quality policy preserves valid biology?",
        visible_options=visible,
        executor_options=executor,
        baseline_option_id=(
            "cellQuality:coreGlobalGaussian"
            if "coreGlobalGaussian" in available_profiles
            else "cellQuality:retainWithFlags"
            if "retainWithFlags" in available_profiles
            else profiles[0][0]
        ),
    )


__all__ = [
    "CellQualityExecutorPayload",
    "QcGroupingExecutorPayload",
    "CompiledRnaDecision",
    "NoExecutionPayload",
    "RnaDecisionCheckpoint",
    "RnaDecisionCompilationError",
    "RnaDecisionDefinition",
    "RnaDecisionGateError",
    "RnaExecutorOption",
    "RnaOptionPayload",
    "build_cell_quality_decision",
    "build_qc_grouping_decision",
    "compile_rna_decision",
    "require_option_evidence",
]
