"""Typed decision contracts and deterministic validation for agent workflows.

The models in this module deliberately separate deterministic option construction
from agent output. An agent may select one offered option and cite observed
evidence, but it cannot add operations or numeric execution parameters.
"""

import hashlib
import re
from collections.abc import Iterable
from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from .. import record_io
from ..types import AgentDataModel, ArtifactReferenceModel

type DecisionStatus = Literal["apply", "skip", "defer", "abstain"]
type DecisionSource = Literal["rule", "agent", "human"]
type DecisionConfidence = Literal["low", "medium", "high", "notApplicable"]
type EvidenceClass = Literal[
    "geometric",
    "markerCoherence",
    "resamplingStability",
    "crossUnitSupport",
    "protectedVariablePreservation",
    "qualityControl",
    "technical",
    "design",
    "provenance",
    "batchRemoval",
    "biologicalConservation",
    "other",
]
type VerificationStatus = Literal["passed", "failed", "inconclusive"]
type DecisionWorkflowStatus = Literal[
    "running",
    "completed",
    "needsInput",
    "abstained",
    "failed",
]
type ProtectedVariableEffectStatus = Literal[
    "preserved",
    "degraded",
    "improved",
    "notEvaluated",
]

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_NON_GEOMETRIC_OVERRIDE_CLASSES: frozenset[EvidenceClass] = frozenset(
    {
        "markerCoherence",
        "resamplingStability",
        "crossUnitSupport",
        "protectedVariablePreservation",
    }
)


def _validate_identifier(value: str, field_name: str) -> str:
    if _ID_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{field_name} must be a non-empty stable identifier containing only "
            "letters, digits, '.', '_', ':', '/', or '-'"
        )
    return value


def _validate_unique(values: list[str], field_name: str) -> list[str]:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return values


def _default_decision_sources() -> list[DecisionSource]:
    return ["rule", "agent", "human"]


class DecisionKernelModel(AgentDataModel):
    """Base for immutable, closed decision-kernel contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class DecisionEvidence(DecisionKernelModel):
    """One bounded observed fact available to a decision."""

    evidenceId: str
    evidenceClass: EvidenceClass
    summary: str = Field(min_length=1, max_length=2000)
    artifactReferences: list[ArtifactReferenceModel] = Field(default_factory=list)

    @field_validator("evidenceId")
    @classmethod
    def validate_evidence_id(cls, value: str) -> str:
        return _validate_identifier(value, "evidenceId")

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("summary must not contain surrounding whitespace")
        return value

    @model_validator(mode="after")
    def validate_artifact_references(self) -> "DecisionEvidence":
        identities: list[tuple[str, str | None, str, str]] = []
        for reference in self.artifactReferences:
            if not reference.kind or not reference.artifactId:
                raise ValueError("artifactReferences require kind and artifactId")
            identities.append(
                (
                    reference.scope,
                    reference.assay,
                    reference.kind,
                    reference.artifactId,
                )
            )
        if len(identities) != len(set(identities)):
            raise ValueError("artifactReferences must not contain duplicates")
        return self


class EvidenceBundle(DecisionKernelModel):
    """The complete immutable evidence inventory offered for one decision."""

    formatVersion: Literal[1] = 1
    bundleId: str
    decisionId: str
    evidence: list[DecisionEvidence] = Field(default_factory=list)
    contentSha256: str | None = None

    @field_validator("bundleId", "decisionId")
    @classmethod
    def validate_ids(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "identifier")
        return _validate_identifier(value, field_name)

    @field_validator("contentSha256")
    @classmethod
    def validate_content_sha256(cls, value: str | None) -> str | None:
        if value is not None and _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("contentSha256 must be a lowercase SHA-256 digest")
        return value

    @model_validator(mode="after")
    def validate_evidence_ids(self) -> "EvidenceBundle":
        _validate_unique(
            [item.evidenceId for item in self.evidence],
            "EvidenceBundle.evidence IDs",
        )
        if self.contentSha256 is not None:
            payload = self.model_dump(mode="json", exclude={"contentSha256"})
            expected = hashlib.sha256(
                record_io.canonical_json_bytes(payload)
            ).hexdigest()
            if self.contentSha256 != expected:
                raise ValueError("contentSha256 does not match the evidence bundle")
        return self

    def evidence_by_id(self) -> dict[str, DecisionEvidence]:
        """Return evidence indexed by its stable identifier."""
        return {item.evidenceId: item for item in self.evidence}

    def with_content_sha256(self) -> "EvidenceBundle":
        """Return this bundle with its canonical content identity."""
        payload = self.model_dump(mode="json", exclude={"contentSha256"})
        digest = hashlib.sha256(record_io.canonical_json_bytes(payload)).hexdigest()
        return EvidenceBundle.model_validate({**payload, "contentSha256": digest})


class DecisionOption(DecisionKernelModel):
    """One pre-registered option whose execution details live outside the model."""

    optionId: str
    status: DecisionStatus
    label: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=2000)
    requiredEvidenceClasses: list[EvidenceClass] = Field(default_factory=list)
    requiredEvidenceIds: list[str] = Field(default_factory=list)

    @field_validator("optionId")
    @classmethod
    def validate_option_id(cls, value: str) -> str:
        return _validate_identifier(value, "optionId")

    @field_validator("label", "description")
    @classmethod
    def validate_text(cls, value: str, info: object) -> str:
        if value != value.strip():
            field_name = getattr(info, "field_name", "text")
            raise ValueError(f"{field_name} must not contain surrounding whitespace")
        return value

    @field_validator("requiredEvidenceClasses")
    @classmethod
    def validate_required_classes(
        cls, value: list[EvidenceClass]
    ) -> list[EvidenceClass]:
        if len(value) != len(set(value)):
            raise ValueError("requiredEvidenceClasses must not contain duplicates")
        return value

    @field_validator("requiredEvidenceIds")
    @classmethod
    def validate_required_evidence_ids(cls, value: list[str]) -> list[str]:
        for evidence_id in value:
            _validate_identifier(evidence_id, "requiredEvidenceIds item")
        return _validate_unique(value, "requiredEvidenceIds")


class DecisionSpec(DecisionKernelModel):
    """Authoritative closed option set for one atomic decision."""

    decisionId: str
    definitionVersion: int = Field(ge=1, strict=True)
    checkpoint: str
    question: str = Field(min_length=1, max_length=2000)
    evidenceBundleId: str
    options: list[DecisionOption] = Field(min_length=1)
    baselineOptionId: str | None = None
    metricPreferredOptionId: str | None = None
    requireIndependentOverrideEvidence: bool = Field(default=False, strict=True)
    allowedSources: list[DecisionSource] = Field(
        default_factory=_default_decision_sources
    )

    @field_validator("decisionId", "checkpoint", "evidenceBundleId")
    @classmethod
    def validate_ids(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "identifier")
        return _validate_identifier(value, field_name)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("question must not contain surrounding whitespace")
        return value

    @field_validator("allowedSources")
    @classmethod
    def validate_allowed_sources(
        cls, value: list[DecisionSource]
    ) -> list[DecisionSource]:
        if not value:
            raise ValueError("allowedSources must not be empty")
        if len(value) != len(set(value)):
            raise ValueError("allowedSources must not contain duplicates")
        return value

    @model_validator(mode="after")
    def validate_option_set(self) -> "DecisionSpec":
        option_ids = [option.optionId for option in self.options]
        _validate_unique(option_ids, "DecisionSpec option IDs")
        if (
            self.baselineOptionId is not None
            and self.baselineOptionId not in option_ids
        ):
            raise ValueError("baselineOptionId must reference an offered option")
        if (
            self.metricPreferredOptionId is not None
            and self.metricPreferredOptionId not in option_ids
        ):
            raise ValueError("metricPreferredOptionId must reference an offered option")
        if (
            self.requireIndependentOverrideEvidence
            and self.metricPreferredOptionId is None
        ):
            raise ValueError(
                "requireIndependentOverrideEvidence requires metricPreferredOptionId"
            )
        return self

    def option_by_id(self) -> dict[str, DecisionOption]:
        """Return offered options indexed by their stable identifier."""
        return {option.optionId: option for option in self.options}


class ProtectedVariableEffect(DecisionKernelModel):
    """Observed effect of a choice on one objective-protected variable."""

    variable: str
    status: ProtectedVariableEffectStatus
    evidenceIds: list[str] = Field(default_factory=list)
    summary: str = Field(min_length=1, max_length=1000)

    @field_validator("variable")
    @classmethod
    def validate_variable(cls, value: str) -> str:
        return _validate_identifier(value, "variable")

    @field_validator("evidenceIds")
    @classmethod
    def validate_evidence_ids(cls, value: list[str]) -> list[str]:
        for evidence_id in value:
            _validate_identifier(evidence_id, "evidenceIds item")
        return _validate_unique(value, "ProtectedVariableEffect.evidenceIds")

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("summary must not contain surrounding whitespace")
        return value


class DecisionSelection(DecisionKernelModel):
    """The bounded choice an agent or human may return."""

    selectedOptionId: str
    evidenceIds: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1, max_length=4000)
    confidence: DecisionConfidence = "notApplicable"
    protectedVariableEffects: list[ProtectedVariableEffect] = Field(
        default_factory=list
    )
    overrideOfOptionId: str | None = None
    overrideEvidenceIds: list[str] = Field(default_factory=list)

    @field_validator("selectedOptionId", "overrideOfOptionId")
    @classmethod
    def validate_ids(cls, value: str | None, info: object) -> str | None:
        if value is None:
            return None
        field_name = getattr(info, "field_name", "identifier")
        return _validate_identifier(value, field_name)

    @field_validator("evidenceIds", "overrideEvidenceIds")
    @classmethod
    def validate_id_lists(cls, value: list[str], info: object) -> list[str]:
        field_name = getattr(info, "field_name", "identifiers")
        for item in value:
            _validate_identifier(item, f"{field_name} item")
        return _validate_unique(value, field_name)

    @field_validator("rationale")
    @classmethod
    def validate_rationale(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("rationale must not contain surrounding whitespace")
        return value

    @model_validator(mode="after")
    def validate_override(self) -> "DecisionSelection":
        if not set(self.overrideEvidenceIds).issubset(self.evidenceIds):
            raise ValueError("overrideEvidenceIds must be included in evidenceIds")
        if self.overrideOfOptionId is None and self.overrideEvidenceIds:
            raise ValueError("overrideEvidenceIds require overrideOfOptionId")
        if self.overrideOfOptionId == self.selectedOptionId:
            raise ValueError("overrideOfOptionId must differ from selectedOptionId")
        return self


class DecisionRecord(DecisionKernelModel):
    """One durable rule, agent, or human selection from an exact option set."""

    recordId: str
    decisionId: str
    definitionVersion: int = Field(ge=1, strict=True)
    evidenceBundleId: str
    evidenceBundleSha256: str
    offeredOptionIds: list[str] = Field(min_length=1)
    availableEvidenceIds: list[str] = Field(default_factory=list)
    selectedOptionId: str
    status: DecisionStatus
    source: DecisionSource
    evidenceIds: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1, max_length=4000)
    confidence: DecisionConfidence = "notApplicable"
    protectedVariableEffects: list[ProtectedVariableEffect] = Field(
        default_factory=list
    )
    overrideOfOptionId: str | None = None
    overrideEvidenceIds: list[str] = Field(default_factory=list)
    promptSha256: str | None = None
    modelName: str | None = None
    softwareSha256: str | None = None
    createdAtNs: int = Field(default=0, ge=0, strict=True)

    @field_validator(
        "recordId",
        "decisionId",
        "evidenceBundleId",
        "selectedOptionId",
        "overrideOfOptionId",
    )
    @classmethod
    def validate_ids(cls, value: str | None, info: object) -> str | None:
        if value is None:
            return None
        field_name = getattr(info, "field_name", "identifier")
        return _validate_identifier(value, field_name)

    @field_validator(
        "offeredOptionIds",
        "availableEvidenceIds",
        "evidenceIds",
        "overrideEvidenceIds",
    )
    @classmethod
    def validate_id_lists(cls, value: list[str], info: object) -> list[str]:
        field_name = getattr(info, "field_name", "identifiers")
        for item in value:
            _validate_identifier(item, f"{field_name} item")
        return _validate_unique(value, field_name)

    @field_validator("rationale")
    @classmethod
    def validate_rationale(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("rationale must not contain surrounding whitespace")
        return value

    @field_validator(
        "evidenceBundleSha256",
        "promptSha256",
        "softwareSha256",
    )
    @classmethod
    def validate_sha256(cls, value: str | None, info: object) -> str | None:
        if value is not None and _SHA256_PATTERN.fullmatch(value) is None:
            field_name = getattr(info, "field_name", "digest")
            raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
        return value

    @field_validator("modelName")
    @classmethod
    def validate_model_name(cls, value: str | None) -> str | None:
        if value is not None and (not value.strip() or value != value.strip()):
            raise ValueError(
                "modelName must be non-empty without surrounding whitespace"
            )
        return value

    @model_validator(mode="after")
    def validate_references(self) -> "DecisionRecord":
        offered = set(self.offeredOptionIds)
        available = set(self.availableEvidenceIds)
        used = set(self.evidenceIds)
        override = set(self.overrideEvidenceIds)
        if self.selectedOptionId not in offered:
            raise ValueError("selectedOptionId must reference an offered option")
        if not used.issubset(available):
            raise ValueError("evidenceIds must reference only available evidence")
        if not override.issubset(used):
            raise ValueError("overrideEvidenceIds must be included in evidenceIds")
        protected_ids = {
            evidence_id
            for effect in self.protectedVariableEffects
            for evidence_id in effect.evidenceIds
        }
        if not protected_ids.issubset(available):
            raise ValueError(
                "protectedVariableEffects must reference only available evidence"
            )
        if self.overrideOfOptionId is None and self.overrideEvidenceIds:
            raise ValueError("overrideEvidenceIds require overrideOfOptionId")
        if self.overrideOfOptionId is not None:
            if self.overrideOfOptionId not in offered:
                raise ValueError("overrideOfOptionId must reference an offered option")
            if self.overrideOfOptionId == self.selectedOptionId:
                raise ValueError("overrideOfOptionId must differ from selectedOptionId")
        return self


class VerificationCheck(DecisionKernelModel):
    """One deterministic invariant evaluated by the auditor."""

    checkId: str
    status: VerificationStatus
    summary: str = Field(min_length=1, max_length=2000)
    evidenceIds: list[str] = Field(default_factory=list)

    @field_validator("checkId")
    @classmethod
    def validate_check_id(cls, value: str) -> str:
        return _validate_identifier(value, "checkId")

    @field_validator("evidenceIds")
    @classmethod
    def validate_evidence_ids(cls, value: list[str]) -> list[str]:
        for evidence_id in value:
            _validate_identifier(evidence_id, "evidenceIds item")
        return _validate_unique(value, "VerificationCheck.evidenceIds")

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("summary must not contain surrounding whitespace")
        return value


class DeterministicDecisionAuditor:
    """Cross-check a decision against its authoritative spec and evidence bundle."""

    @classmethod
    def audit(
        cls,
        spec: DecisionSpec,
        evidence: EvidenceBundle,
        record: DecisionRecord,
        *,
        created_at_ns: int = 0,
    ) -> list[VerificationCheck]:
        """Return a deterministic verification without repairing invalid output."""
        checks: list[VerificationCheck] = []
        evidence_sha256 = (
            evidence.contentSha256
            if evidence.contentSha256 is not None
            else evidence.with_content_sha256().contentSha256
        )

        def add_check(
            check_id: str,
            passed: bool,
            pass_summary: str,
            fail_summary: str,
            evidence_ids: Iterable[str] = (),
        ) -> None:
            checks.append(
                VerificationCheck(
                    checkId=check_id,
                    status="passed" if passed else "failed",
                    summary=pass_summary if passed else fail_summary,
                    evidenceIds=list(evidence_ids),
                )
            )

        add_check(
            "decisionIdentity",
            record.decisionId == spec.decisionId
            and record.definitionVersion == spec.definitionVersion
            and evidence.decisionId == spec.decisionId
            and record.evidenceBundleId == spec.evidenceBundleId
            and evidence.bundleId == spec.evidenceBundleId
            and record.evidenceBundleSha256 == evidence_sha256,
            "Decision, definition, and evidence bundle identities agree.",
            "Decision, definition, or evidence bundle identity does not agree.",
        )

        offered_option_ids = [option.optionId for option in spec.options]
        add_check(
            "exactOptionInventory",
            record.offeredOptionIds == offered_option_ids,
            "The durable record contains the exact offered option inventory.",
            "The durable record does not contain the exact offered option inventory.",
        )

        available_evidence_ids = [item.evidenceId for item in evidence.evidence]
        add_check(
            "exactEvidenceInventory",
            record.availableEvidenceIds == available_evidence_ids,
            "The durable record contains the exact evidence inventory.",
            "The durable record does not contain the exact evidence inventory.",
        )

        option = spec.option_by_id().get(record.selectedOptionId)
        add_check(
            "selectedOption",
            option is not None and option.status == record.status,
            "The selected option and decision status agree.",
            "The selected option is unavailable or its status does not agree.",
        )

        add_check(
            "decisionSource",
            record.source in spec.allowedSources,
            "The decision source is allowed by the definition.",
            "The decision source is not allowed by the definition.",
        )

        evidence_by_id = evidence.evidence_by_id()
        cited_ids = set(record.evidenceIds)
        required_classes = set(option.requiredEvidenceClasses) if option else set()
        required_ids = set(option.requiredEvidenceIds) if option else set()
        cited_classes = {
            evidence_by_id[evidence_id].evidenceClass
            for evidence_id in cited_ids
            if evidence_id in evidence_by_id
        }
        required_evidence_ok = (
            cited_ids.issubset(evidence_by_id)
            and required_ids.issubset(cited_ids)
            and required_classes.issubset(cited_classes)
        )
        add_check(
            "requiredEvidence",
            required_evidence_ok,
            "All cited and required evidence is present.",
            "Cited evidence is unavailable or a required evidence class is missing.",
            record.evidenceIds,
        )

        protected_evidence_ids = {
            evidence_id
            for effect in record.protectedVariableEffects
            for evidence_id in effect.evidenceIds
        }
        protected_ok = not any(
            effect.status == "degraded" for effect in record.protectedVariableEffects
        ) and protected_evidence_ids.issubset(cited_ids)
        add_check(
            "protectedVariablePreservation",
            protected_ok,
            "No cited protected variable is degraded.",
            "A protected variable degraded or its evidence was not cited.",
            sorted(protected_evidence_ids),
        )

        metric_override = (
            spec.requireIndependentOverrideEvidence
            and spec.metricPreferredOptionId is not None
            and record.status in {"apply", "skip"}
            and record.selectedOptionId != spec.metricPreferredOptionId
        )
        if metric_override:
            override_classes = {
                evidence_by_id[evidence_id].evidenceClass
                for evidence_id in record.overrideEvidenceIds
                if evidence_id in evidence_by_id
            }
            qualifying_classes = override_classes & _NON_GEOMETRIC_OVERRIDE_CLASSES
            override_ok = (
                record.overrideOfOptionId == spec.metricPreferredOptionId
                and set(record.overrideEvidenceIds).issubset(record.evidenceIds)
                and (
                    not required_ids
                    or set(record.overrideEvidenceIds).issubset(required_ids)
                )
                and len(qualifying_classes) >= 2
            )
            add_check(
                "independentOverrideEvidence",
                override_ok,
                "The override cites at least two independent non-geometric evidence classes.",
                "The override requires two independent non-geometric evidence classes.",
                record.overrideEvidenceIds,
            )
        else:
            add_check(
                "independentOverrideEvidence",
                record.overrideOfOptionId is None and not record.overrideEvidenceIds,
                "No metric override evidence is required.",
                "Override fields were supplied without an eligible metric override.",
                record.overrideEvidenceIds,
            )

        return checks


__all__ = [
    "DecisionConfidence",
    "DecisionEvidence",
    "DecisionOption",
    "DecisionRecord",
    "DecisionSelection",
    "DecisionSource",
    "DecisionSpec",
    "DecisionStatus",
    "DecisionWorkflowStatus",
    "DeterministicDecisionAuditor",
    "EvidenceBundle",
    "EvidenceClass",
    "ProtectedVariableEffect",
    "ProtectedVariableEffectStatus",
    "VerificationCheck",
    "VerificationStatus",
]
