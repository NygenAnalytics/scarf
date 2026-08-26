"""Grounded structured decisions over evidence IDs."""

from collections.abc import Sequence
from textwrap import dedent
from typing import Any

from .config.agent_exec import run_agent_sync
from .types import Decision, EvidenceItem

_SYSTEM_PROMPT = dedent(
    """
        Choose exactly one evidence id that answers the question. selectedId
        must be copied exactly from the id= values. Do not put labels or
        summaries in selectedId. evidenceIds must include selectedId and only
        provided ids. Keep the rationale to one short sentence. Return only
        fields defined by the structured output schema.
        """
).strip()


class DecisionValidationError(ValueError):
    """Raised when a model decision cites unknown or invalid evidence."""


def _coerce_evidence_id(evidence_id: str, allowed: set[str]) -> str:
    """Map a model-emitted id onto an allowed evidence id when unambiguous.

    Live models often echo prompt scaffolding such as ``id=domain:biological``
    instead of the bare id. Accept that when exactly one allowed id is embedded.
    """
    if evidence_id in allowed:
        return evidence_id
    stripped = evidence_id.strip()
    if stripped.startswith("id="):
        stripped = stripped[3:].strip()
        if stripped in allowed:
            return stripped
    matches = [allowed_id for allowed_id in allowed if allowed_id in evidence_id]
    if len(matches) == 1:
        return matches[0]
    return evidence_id


def validate_decision(
    decision: Decision,
    evidence: Sequence[EvidenceItem],
) -> Decision:
    allowed = {item.id for item in evidence}
    if not allowed:
        raise DecisionValidationError("evidence must contain at least one item")
    selected_id = _coerce_evidence_id(decision.selectedId, allowed)
    evidence_ids = [
        _coerce_evidence_id(evidence_id, allowed)
        for evidence_id in decision.evidenceIds
    ]
    if selected_id not in evidence_ids and selected_id in allowed:
        evidence_ids = [selected_id, *evidence_ids]
    if selected_id != decision.selectedId or evidence_ids != list(decision.evidenceIds):
        decision = Decision(
            selectedId=selected_id,
            rationale=decision.rationale,
            evidenceIds=evidence_ids,
        )
    if decision.selectedId not in allowed:
        raise DecisionValidationError(
            f"selectedId {decision.selectedId!r} is not in evidence ids {sorted(allowed)}"
        )
    unknown = [
        evidence_id
        for evidence_id in decision.evidenceIds
        if evidence_id not in allowed
    ]
    if unknown:
        raise DecisionValidationError(
            f"evidenceIds cite unknown ids {unknown}; allowed {sorted(allowed)}"
        )
    if decision.selectedId not in decision.evidenceIds:
        raise DecisionValidationError(
            f"evidenceIds must include selectedId {decision.selectedId!r}"
        )
    return decision


def _format_user_prompt(question: str, evidence: Sequence[EvidenceItem]) -> str:
    evidence_lines = "\n".join(
        f"- id={item.id} | label={item.label} | summary={item.summary}"
        for item in evidence
    )
    return (
        dedent(
            """
            {question}

            Choose selectedId from these exact id values:
            {evidence_lines}

            Allowed selectedId values: {allowed_ids}
            """
        )
        .strip()
        .format(
            question=question.strip(),
            evidence_lines=evidence_lines,
            allowed_ids=", ".join(item.id for item in evidence),
        )
    )


def decide(
    *,
    model: Any,
    question: str,
    evidence: Sequence[EvidenceItem],
    system_prompt: str = _SYSTEM_PROMPT,
) -> Decision:
    """Ask the model to choose among evidence IDs and validate citations."""
    if not question.strip():
        raise ValueError("question must be non-empty")
    if not evidence:
        raise ValueError("evidence must contain at least one item")
    seen: set[str] = set()
    duplicates: set[str] = set()
    for item in evidence:
        if item.id in seen:
            duplicates.add(item.id)
        else:
            seen.add(item.id)
    if duplicates:
        raise ValueError(
            f"evidence ids must be unique; duplicates: {sorted(duplicates)}"
        )

    execution = run_agent_sync(
        model=model,
        output_type=Decision,
        system_prompt=system_prompt,
        user_prompt=_format_user_prompt(question, evidence),
        name="decision",
    )
    return validate_decision(execution.output, evidence)
