"""Model attempts retain measured usage and rejected decisions without execution."""

import asyncio
from typing import Any

import pytest
from pydantic_ai import UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
)
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from scarf.agent.config import AgentRunConfig
from scarf.agent.config.agent_exec import run_agent_async, run_agent_sync
from scarf.agent.decisions.selection import decide
from scarf.agent.types import AgentDataModel, AgentRunInfo, AgentUsageInfo, EvidenceItem


class CountDecision(AgentDataModel):
    value: int


@pytest.mark.parametrize("host", ["sync", "async", "notebook"])
def test_repaired_attempt_preserves_usage_and_rejection(host: str) -> None:
    requests = 0
    attempts: list[AgentRunInfo] = []

    async def respond(messages: Any, info: Any) -> ModelResponse:
        nonlocal requests
        requests += 1
        if requests == 2:
            assert any(
                isinstance(part, RetryPromptPart)
                and "exact measured count is 2" in str(part.content)
                for message in messages
                for part in message.parts
            )
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, {"value": requests})],
            usage=RequestUsage(input_tokens=11, output_tokens=3),
        )

    def validate(output: CountDecision) -> CountDecision:
        if output.value != 2:
            raise ValueError("The exact measured count is 2")
        return output

    kwargs = dict(
        model=FunctionModel(respond),
        output_type=CountDecision,
        system_prompt="Use measured counts.",
        user_prompt="Choose the measured count.",
        output_validator=validate,
        on_attempt=attempts.append,
    )
    if host == "async":
        result = asyncio.run(run_agent_async(**kwargs))
    elif host == "notebook":

        async def notebook() -> Any:
            return run_agent_sync(**kwargs)

        result = asyncio.run(notebook())
    else:
        result = run_agent_sync(**kwargs)
    assert requests == 2
    assert attempts == [result.runInfo]
    assert result.runInfo.status == "done"
    assert result.runInfo.usage.inputTokens == 22
    assert result.runInfo.usage.outputTokens == 6
    assert result.runInfo.usage.availability == "reported"
    assert len(result.runInfo.validationRetries) == 1
    assert result.runInfo.validationRetries[0].response == {"value": 1}
    assert result.runInfo.validationRetries[0].requestIndex == 1


def test_exhausted_attempt_keeps_exact_error_usage_and_every_rejection() -> None:
    attempts: list[AgentRunInfo] = []

    async def respond(_messages: Any, info: Any) -> ModelResponse:
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, {"value": 1})],
            usage=RequestUsage(input_tokens=7, output_tokens=2),
        )

    def reject(_output: CountDecision) -> CountDecision:
        raise ValueError("Missing the measured alternative's marker advantage")

    with pytest.raises(UnexpectedModelBehavior) as caught:
        run_agent_sync(
            model=FunctionModel(respond),
            output_type=CountDecision,
            system_prompt="Assess evidence.",
            user_prompt="Select only supported settings.",
            config=AgentRunConfig(retries=1),
            output_validator=reject,
            on_attempt=attempts.append,
        )
    info = caught.value.agent_run_info
    assert attempts == [info]
    assert info.status == "failed"
    assert info.usage.requests == 2
    assert info.usage.inputTokens == 14
    assert info.usage.outputTokens == 4
    assert info.usage.availability == "reported"
    assert len(info.validationRetries) == 2
    assert [row.requestIndex for row in info.validationRetries] == [1, 2]
    assert all(item.response == {"value": 1} for item in info.validationRetries)
    assert "marker advantage" in info.error


def test_failed_request_does_not_invent_provider_usage() -> None:
    attempts: list[AgentRunInfo] = []

    async def fail(_messages: Any, _info: Any) -> ModelResponse:
        raise RuntimeError("Provider unavailable")

    with pytest.raises(RuntimeError, match="Provider unavailable") as caught:
        run_agent_sync(
            model=FunctionModel(fail),
            output_type=CountDecision,
            system_prompt="Assess evidence.",
            user_prompt="Choose a count.",
            on_attempt=attempts.append,
        )
    assert attempts == [caught.value.agent_run_info]
    assert attempts[0].usage.availability == "unavailable"
    assert attempts[0].validationRetries == []


@pytest.mark.parametrize("provider_fails", [False, True])
def test_journal_callback_failure_preserves_model_outcome(provider_fails: bool) -> None:
    from scarf.agent.config.agent_exec import describe_agent_error

    calls = []

    async def respond(_messages: Any, info: Any) -> ModelResponse:
        if provider_fails:
            raise RuntimeError("Original provider outage")
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, {"value": 2})],
            usage=RequestUsage(input_tokens=11, output_tokens=2),
        )

    def failed_journal(info: AgentRunInfo) -> None:
        calls.append(info)
        raise OSError("Test history store is unavailable")

    with pytest.raises(RuntimeError if provider_fails else OSError) as caught:
        run_agent_sync(
            model=FunctionModel(respond),
            output_type=CountDecision,
            system_prompt="Interpret saved evidence.",
            user_prompt="Choose the measured count.",
            on_attempt=failed_journal,
        )
    assert len(calls) == 1
    assert caught.value.agent_run_info == calls[0]
    assert calls[0].status == ("failed" if provider_fails else "done")
    assert calls[0].runId
    detail = describe_agent_error(caught.value)
    assert "history store is unavailable" in detail
    if provider_fails:
        assert "Original provider outage" in detail
    else:
        assert "model completed" in detail


def test_provider_outage_retains_partial_usage_and_completed_tool_evidence() -> None:
    requests = 0
    operations = 0

    async def measure() -> int:
        nonlocal operations
        operations += 1
        return 2

    async def respond(_messages: Any, _info: Any) -> ModelResponse:
        nonlocal requests
        requests += 1
        if requests > 1:
            raise RuntimeError("Provider unavailable after the measured tool result")
        return ModelResponse(
            parts=[ToolCallPart("measure", {})],
            usage=RequestUsage(input_tokens=11, output_tokens=2),
        )

    with pytest.raises(RuntimeError, match="after the measured") as caught:
        run_agent_sync(
            model=FunctionModel(respond),
            output_type=CountDecision,
            system_prompt="Measure once before interpreting.",
            user_prompt="Assess the measured count.",
            tools=[measure],
        )
    info = caught.value.agent_run_info
    assert operations == 1
    assert requests == 2
    assert info.usage.requests == 2  # Include the failed provider request.
    assert info.usage.inputTokens == 11
    assert info.usage.availability == "partial"
    assert len(info.providerFailures) == 1
    assert info.providerFailures[0].requestIndex == 2
    assert info.providerFailures[0].retryDelaySeconds is None
    assert [call.toolName for call in info.toolCalls] == ["measure"]


def test_cancelled_attempt_retains_known_execution_information() -> None:
    attempts: list[AgentRunInfo] = []

    async def respond(_messages: Any, _info: Any) -> ModelResponse:
        raise asyncio.CancelledError("Analysis interrupted")

    with pytest.raises(asyncio.CancelledError) as caught:
        asyncio.run(
            run_agent_async(
                model=FunctionModel(respond),
                output_type=CountDecision,
                system_prompt="Assess evidence.",
                user_prompt="Select a count.",
                on_attempt=attempts.append,
            )
        )
    assert attempts == [caught.value.agent_run_info]
    assert attempts[0].errorType == "CancelledError"
    assert attempts[0].usage.availability == "unavailable"


def test_old_run_information_serializes_without_added_default_fields() -> None:
    payload = {
        "agentName": "prior",
        "modelName": "model",
        "runId": "id",
        "durationSeconds": 12.0,
        "usage": {
            "inputTokens": 1,
            "outputTokens": 2,
            "totalTokens": 3,
            "requests": 1,
            "toolCalls": 0,
        },
        "toolCalls": [],
    }
    assert AgentRunInfo.model_validate(payload).model_dump(mode="json") == payload
    assert "availability" not in AgentUsageInfo().model_dump(mode="json")


def test_schema_repair_does_not_repeat_a_successful_tool_operation() -> None:
    operations = 0
    requests = 0

    async def measure(value: int) -> int:
        nonlocal operations
        operations += 1
        return value

    async def respond(_messages: Any, info: Any) -> ModelResponse:
        nonlocal requests
        requests += 1
        if requests <= 2:
            part = ToolCallPart("measure", {"value": "invalid" if requests == 1 else 2})
        else:
            part = ToolCallPart(info.output_tools[0].name, {"value": 2})
        return ModelResponse(
            parts=[part], usage=RequestUsage(input_tokens=5, output_tokens=1)
        )

    result = run_agent_sync(
        model=FunctionModel(respond),
        output_type=CountDecision,
        system_prompt="Use one successful measured value.",
        user_prompt="Measure the value.",
        tools=[measure],
    )
    assert operations == 1
    assert result.output.value == 2
    assert requests == 3
    assert len(result.runInfo.validationRetries) == 1
    assert result.runInfo.validationRetries[0].source == "tool"
    assert result.runInfo.validationRetries[0].response == {"value": "invalid"}


@pytest.mark.parametrize("content", ["none", "text", "thinking", "both"])
def test_failed_schema_attempt_retains_the_last_invalid_response(content: str) -> None:
    attempts: list[AgentRunInfo] = []

    async def respond(_messages: Any, info: Any) -> ModelResponse:
        parts: list[Any] = []
        if content in {"text", "both"}:
            parts.append(TextPart("Here is the requested structured result."))
        if content in {"thinking", "both"}:
            parts.append(ThinkingPart("Checking the observed count."))
        parts.append(ToolCallPart(info.output_tools[0].name, {"value": "invalid"}))
        return ModelResponse(
            parts=parts, usage=RequestUsage(input_tokens=5, output_tokens=2)
        )

    with pytest.raises(UnexpectedModelBehavior) as caught:
        run_agent_sync(
            model=FunctionModel(respond),
            output_type=CountDecision,
            system_prompt="Use measured counts.",
            user_prompt="Choose a count.",
            config=AgentRunConfig(retries=1),
            on_attempt=attempts.append,
        )
    info = caught.value.agent_run_info
    assert attempts == [info]
    assert len(info.validationRetries) == 2
    assert all(row.response == {"value": "invalid"} for row in info.validationRetries)
    assert [row.requestIndex for row in info.validationRetries] == [1, 2]
    assert info.usage.inputTokens == 10
    assert info.usage.requests == 2
    assert info.status == "failed"


def test_exhausted_schema_with_multiple_calls_keeps_error_without_guessing_attribution() -> (
    None
):
    async def respond(_messages: Any, info: Any) -> ModelResponse:
        return ModelResponse(
            parts=[
                TextPart("Two proposed results."),
                ToolCallPart(
                    info.output_tools[0].name,
                    {"value": "first-invalid"},
                    tool_call_id="first",
                ),
                ToolCallPart(
                    info.output_tools[0].name,
                    {"value": "second-invalid"},
                    tool_call_id="second",
                ),
            ],
            usage=RequestUsage(input_tokens=5, output_tokens=2),
        )

    with pytest.raises(UnexpectedModelBehavior) as caught:
        run_agent_sync(
            model=FunctionModel(respond),
            output_type=CountDecision,
            system_prompt="Use one measured count.",
            user_prompt="Choose a count.",
            config=AgentRunConfig(retries=0),
        )
    info = caught.value.agent_run_info
    assert "ValidationError" in info.error
    assert info.validationRetries == []
    assert info.status == "failed"
    assert info.usage.requests == 1
    assert info.usage.inputTokens == 5


def test_exact_evidence_id_errors_repair_inside_the_model_loop() -> None:
    requests = 0

    async def respond(messages: Any, info: Any) -> ModelResponse:
        nonlocal requests
        requests += 1
        if requests == 2:
            assert any(
                isinstance(part, RetryPromptPart)
                and "not in evidence ids" in str(part.content)
                for message in messages
                for part in message.parts
            )
        identifier = "unrelated:matrix:raw/X" if requests == 1 else "matrix:raw/X"
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {
                        "selectedId": identifier,
                        "rationale": "The observed raw matrix contains integer counts.",
                        "evidenceIds": [identifier],
                    },
                )
            ]
        )

    result = decide(
        model=FunctionModel(respond),
        question="Which measured matrix contains raw counts?",
        evidence=[
            EvidenceItem(
                id="matrix:raw/X", label="Raw counts", summary="Integer counts"
            )
        ],
    )
    assert requests == 2
    assert result.selectedId == "matrix:raw/X"
