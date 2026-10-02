"""Closed inputs and small decisions for the bounded RNA procedure."""

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AgentModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class Study(AgentModel):
    """Supplied facts, never inferred experimental permissions."""

    context: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    organism: str | None = None
    tissue: str | None = None
    technicalBatchColumns: list[str] = Field(default_factory=list)
    protectedColumns: list[str] = Field(default_factory=list)
    sampleColumn: str | None = None
    captureColumn: str | None = None
    excludedColumns: list[str] = Field(default_factory=list)
    featureExclusions: list[str] = Field(default_factory=list)
    referenceFiles: list[str] = Field(default_factory=list)
    batchCorrectionEvidence: str | None = None

    @field_validator("context", "objective")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Study text must not be blank")
        return value


class AnalysisConfig(AgentModel):
    """Scientific policy frozen for an analysis."""

    assay: str | None = None
    workspace: str | None = None
    cellKey: str = "I"
    qcPolicy: Literal["retain", "gentleMad5", "manual"] = "retain"
    qcBounds: dict[str, tuple[float | None, float | None]] = Field(default_factory=dict)
    interactionMode: Literal["strict", "lenient"] = "lenient"
    hvgCount: int = Field(default=1000, ge=2)
    pcaDims: int = Field(default=21, ge=2)
    neighborsK: int = Field(default=11, ge=2)
    resolutions: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25)
    maxCandidates: int = Field(default=5, ge=1, le=5)
    maxFinalists: int = Field(default=2, ge=1, le=2)
    scoreDoublets: bool = False
    randomSeed: int = Field(default=4444, ge=0)

    @model_validator(mode="after")
    def validate_policy(self) -> "AnalysisConfig":
        if not self.resolutions or len(self.resolutions) > 4:
            raise ValueError("Provide one to four Leiden resolutions")
        if len(set(self.resolutions)) != len(self.resolutions) or any(
            not math.isfinite(value) or value <= 0 for value in self.resolutions
        ):
            raise ValueError(
                "Leiden resolutions must be distinct positive finite values"
            )
        if (self.qcPolicy == "manual") != bool(self.qcBounds):
            raise ValueError("qcBounds must be supplied only for manual QC")
        for name, (low, high) in self.qcBounds.items():
            if not name or any(
                value is not None and not math.isfinite(value) for value in (low, high)
            ):
                raise ValueError("QC bounds must have names and finite values")
            if low is not None and high is not None and low > high:
                raise ValueError("QC lower bound must not exceed upper bound")
        return self


class RuntimeConfig(AgentModel):
    """Operational settings that may change on explicit resume."""

    maxRequests: int = Field(default=30, ge=1)
    maxRequestsPerDecision: int = Field(default=3, ge=1, le=3)
    maxPromptBytes: int = Field(default=65536, ge=1024)
    maxOutputTokens: int = Field(default=4096, ge=128)
    decisionTimeout: float = Field(default=120.0, gt=0, allow_inf_nan=False)
    nthreads: int = Field(default=4, ge=1)
    memBudget: str = "0.5"
    modelSettings: dict[str, Any] = Field(default_factory=dict)

    @field_validator("modelSettings")
    @classmethod
    def no_secrets(cls, value: dict[str, Any]) -> dict[str, Any]:
        def inspect(item: Any) -> None:
            if isinstance(item, dict):
                for key, child in item.items():
                    normalized = str(key).lower().replace("-", "_")
                    if any(
                        word in normalized
                        for word in (
                            "token",
                            "secret",
                            "password",
                            "header",
                            "api_key",
                            "apikey",
                            "authorization",
                            "cookie",
                        )
                    ) and normalized not in {"max_tokens", "max_output_tokens"}:
                        raise ValueError(
                            "Configure credentials on the provider, not in saved settings"
                        )
                    inspect(child)
            elif isinstance(item, list | tuple):
                for child in item:
                    inspect(child)

        inspect(value)
        if {"max_tokens", "max_output_tokens", "timeout"} & value.keys():
            raise ValueError(
                "Use the explicit runtime output-token and deadline limits"
            )
        return value


class Candidate(AgentModel):
    candidateId: str
    parentId: str | None = None
    hvgCount: int = Field(ge=2)
    pcaDims: int = Field(ge=2)
    neighborsK: int = Field(ge=2)
    useHarmony: bool = False


class ContextDecision(AgentModel):
    columnRoles: dict[
        str, Literal["technical", "protected", "sample", "capture", "ignore"]
    ] = Field(
        default_factory=dict,
        description="Optional new roles. Supplied roles persist when omitted. The technical role is restricted to study.technicalBatchColumns; QC metrics are not authorized batches.",
    )
    excludeFeatures: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)
    evidenceIds: list[str] = Field(default_factory=list)
    question: str | None = None
    deferralReason: (
        Literal[
            "uncertainMetadata",
            "ambiguousSelection",
            "missingEssentialInput",
            "unsupportedObjective",
        ]
        | None
    ) = None


class Choice(AgentModel):
    action: Literal["experiment", "shortlist", "choose", "defer"]
    optionIds: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)
    evidenceIds: list[str] = Field(default_factory=list)
    question: str | None = None
    deferralReason: (
        Literal[
            "uncertainMetadata",
            "ambiguousSelection",
            "missingEssentialInput",
            "unsupportedObjective",
        ]
        | None
    ) = None
    acceptableOptionIds: list[str] = Field(default_factory=list)


class ClusterAnnotation(AgentModel):
    clusterId: str
    identity: str = "unassigned"
    confidence: Literal["low", "medium", "high"] = "low"
    supportingMarkers: list[str] = Field(default_factory=list)
    contradictingMarkers: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)


class AnnotationDecision(AgentModel):
    annotations: list[ClusterAnnotation]


class NeedsInput(ValueError):
    """Missing essential facts; the workflow saves the question before stopping."""

    def __init__(self, question: str, *, field: str | None = None) -> None:
        self.question = question
        self.field = field
        super().__init__(question)


class AnalysisInputError(ValueError):
    """An agent-owned validation failure whose explanation is safe to record."""
