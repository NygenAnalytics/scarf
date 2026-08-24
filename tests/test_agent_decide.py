"""Unit tests for grounded decide()."""

from collections.abc import Sequence

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.function import AgentInfo
from pydantic_ai.models.test import TestModel

from scarf.agent import DecisionValidationError, EvidenceItem, decide
from scarf.agent.decide import validate_decision
from scarf.agent.types import Decision


def _evidence() -> list[EvidenceItem]:
    return [
        EvidenceItem(id="matrix:X", label="X", summary="float, mostly non-integer"),
        EvidenceItem(id="matrix:raw/X", label="raw/X", summary="integer-like"),
    ]


def _function_model(decision: Decision) -> FunctionModel:
    def reply(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool = info.output_tools[0]
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=tool.name,
                    args=decision.model_dump(),
                )
            ]
        )

    return FunctionModel(reply)


def test_decide_with_function_model_returns_valid_decision() -> None:
    expected = Decision(
        selectedId="matrix:raw/X",
        rationale="integer-like counts",
        evidenceIds=["matrix:raw/X"],
    )
    result = decide(
        model=_function_model(expected),
        question="Which matrix looks like raw counts?",
        evidence=_evidence(),
    )
    assert result == expected


def test_decide_with_test_model_returns_schema_valid_decision() -> None:
    result = decide(
        model=TestModel(
            custom_output_args={
                "selectedId": "matrix:raw/X",
                "rationale": "integer-like",
                "evidenceIds": ["matrix:raw/X"],
            }
        ),
        question="Which matrix looks like raw counts?",
        evidence=_evidence(),
    )
    assert result.selectedId == "matrix:raw/X"
    assert result.evidenceIds == ["matrix:raw/X"]
    validate_decision(result, _evidence())


def test_validate_decision_coerces_selected_id_embedded_in_line() -> None:
    decision = Decision(
        selectedId="id=matrix:raw/X | label=raw/X | summary=integer-like",
        rationale="integer-like",
        evidenceIds=[],
    )
    result = validate_decision(decision, _evidence())
    assert result.selectedId == "matrix:raw/X"
    assert result.evidenceIds == ["matrix:raw/X"]


def test_validate_decision_coerces_id_equals_prefix_in_evidence_ids() -> None:
    decision = Decision(
        selectedId="id=matrix:raw/X",
        rationale="echoed prompt scaffolding",
        evidenceIds=["id=matrix:raw/X"],
    )
    result = validate_decision(decision, _evidence())
    assert result.selectedId == "matrix:raw/X"
    assert result.evidenceIds == ["matrix:raw/X"]


def test_decide_rejects_unknown_selected_id() -> None:
    bad = Decision(
        selectedId="matrix:missing",
        rationale="guess",
        evidenceIds=["matrix:missing"],
    )
    with pytest.raises(DecisionValidationError, match="selectedId"):
        decide(
            model=_function_model(bad),
            question="Which matrix looks like raw counts?",
            evidence=_evidence(),
        )


def test_validate_decision_rejects_unknown_evidence_ids() -> None:
    decision = Decision(
        selectedId="matrix:X",
        rationale="ok",
        evidenceIds=["matrix:X", "matrix:ghost"],
    )
    with pytest.raises(DecisionValidationError, match="unknown ids"):
        validate_decision(decision, _evidence())


def test_decide_rejects_empty_evidence() -> None:
    with pytest.raises(ValueError, match="evidence"):
        decide(
            model=TestModel(),
            question="Which matrix looks like raw counts?",
            evidence=[],
        )


def test_decide_rejects_duplicate_evidence_ids() -> None:
    evidence: Sequence[EvidenceItem] = [
        EvidenceItem(id="matrix:X", label="X", summary="a"),
        EvidenceItem(id="matrix:X", label="X2", summary="b"),
    ]
    with pytest.raises(ValueError, match="unique"):
        decide(
            model=TestModel(),
            question="Which matrix looks like raw counts?",
            evidence=evidence,
        )
