"""Provider-independent checks on bounded images, repair feedback, and failures."""

from typing import Any

import pytest
from pydantic_ai import ModelRetry, UnexpectedModelBehavior
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import BinaryContent, ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from scarf.agent.config import agent_exec
from scarf.agent.decisions.selection import decide
from scarf.agent.types import EvidenceItem
from tests.test_agent_attempt_audit import CountDecision


@pytest.mark.parametrize(
    ("prompt", "images", "reason"),
    [
        ("", [("a", b"a", "image/png")], "non-empty string"),
        ("Review", [], "at least one"),
        ("Review", [(str(i), b"a", "image/png") for i in range(9)], "maximum"),
        ("Review", [(" a", b"a", "image/png")], "trimmed"),
        ("Review", [("a", b"a", "image/png")] * 2, "unique"),
        ("Review", [("a", b"a", "image/gif")], "unsupported media"),
        ("Review", [("a", b"", "image/png")], "non-empty bytes"),
        ("Review", [("a", b"abcde", "image/png")], "exceeds 4 bytes"),
        ("Review", [(str(i), b"abcd", "image/png") for i in range(3)], "total byte"),
    ],
)
def test_visual_prompt_rejects_unbounded_or_ambiguous_evidence(
    monkeypatch: Any, prompt: str, images: list[tuple[str, bytes, Any]], reason: str
) -> None:
    monkeypatch.setattr(agent_exec, "_MAX_VISUAL_EVIDENCE_ITEM_BYTES", 4)
    monkeypatch.setattr(agent_exec, "_MAX_VISUAL_EVIDENCE_TOTAL_BYTES", 8)
    with pytest.raises(ValueError, match=reason):
        agent_exec.build_visual_evidence_prompt(
            prompt, [agent_exec.ImageEvidence(*value) for value in images]
        )


def test_visual_prompt_preserves_exact_supplied_image_bytes() -> None:
    image = agent_exec.ImageEvidence("observed-loadings", b"observed")
    prompt = agent_exec.build_visual_evidence_prompt("Assess these loadings.", [image])
    assert prompt[0] == "Assess these loadings."
    assert isinstance(prompt[1], BinaryContent)
    assert prompt[1].data == image.data
    assert prompt[1].identifier == image.identifier


@pytest.mark.parametrize("unsupported", [True, False])
def test_image_rejection_keeps_failed_usage_and_other_provider_failures_are_not_hidden(
    unsupported: bool,
) -> None:
    original = ModelHTTPError(
        400 if unsupported else 401,
        "test-model",
        {"message": "Image input is not supported" if unsupported else "Unauthorized"},
    )

    async def respond(_messages: Any, _info: Any) -> ModelResponse:
        raise original

    attempts = []
    with pytest.raises(
        agent_exec.ImageInputUnsupportedError if unsupported else ModelHTTPError
    ) as caught:
        agent_exec.run_agent_sync(
            model=FunctionModel(respond),
            output_type=CountDecision,
            system_prompt="Assess only measured evidence.",
            user_prompt=agent_exec.build_visual_evidence_prompt(
                "Review the loading plot.",
                [agent_exec.ImageEvidence("loadings", b"image")],
            ),
            on_attempt=attempts.append,
        )
    assert attempts == [caught.value.agent_run_info]
    assert attempts[0].usage.availability == "unavailable"
    assert attempts[0].status == "failed"
    assert "ModelHTTPError" in attempts[0].error
    if unsupported:
        assert caught.value.__cause__ is original
    else:
        assert caught.value is original


def test_explicit_async_model_retry_is_audited_once_before_repair() -> None:
    calls = 0

    async def respond(_messages: Any, info: Any) -> ModelResponse:
        nonlocal calls
        calls += 1
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, {"value": calls})],
            usage=RequestUsage(input_tokens=3, output_tokens=1),
        )

    async def validate(output: CountDecision) -> CountDecision:
        if output.value != 2:
            raise ModelRetry("The completed comparison requires count 2")
        return output

    result = agent_exec.run_agent_sync(
        model=FunctionModel(respond),
        output_type=CountDecision,
        system_prompt="Use measured evidence.",
        user_prompt="Interpret the count.",
        output_validator=validate,
    )
    assert result.output.value == 2
    assert len(result.runInfo.validationRetries) == 1
    assert result.runInfo.validationRetries[0].response == {"value": 1}
    assert result.runInfo.usage.requests == 2


def test_generic_decision_does_not_reclassify_unrelated_provider_failure() -> None:
    original = UnexpectedModelBehavior("Provider returned an empty response")

    async def respond(_messages: Any, _info: Any) -> ModelResponse:
        raise original

    with pytest.raises(UnexpectedModelBehavior) as caught:
        decide(
            model=FunctionModel(respond),
            question="Select measured evidence.",
            evidence=[
                EvidenceItem(id="measured", label="Measured", summary="Observed")
            ],
        )
    assert caught.value is original
    assert caught.value.agent_run_info.status == "failed"


def test_error_detail_handles_implicit_context_cycles_and_bounded_notes() -> None:
    cause = ValueError("Exact failed measurement")
    error = RuntimeError("Could not finish interpretation")
    error.__context__ = cause
    cause.__context__ = error
    error.add_note("The failed evidence remains saved")
    detail = agent_exec.describe_agent_error(error)
    assert detail.count("Could not finish") == 1
    assert "Exact failed measurement" in detail
    assert "evidence remains saved" in detail
    assert len(agent_exec.describe_agent_error(error, limit=20)) == 20
