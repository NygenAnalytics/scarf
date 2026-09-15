"""Transport throttling replays one request without repeating scientific tools."""

import asyncio
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import httpx
import pytest
from pydantic_ai.exceptions import ModelHTTPError, UsageLimitExceeded
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from scarf.agent.config import AgentRunConfig, agent_exec
from scarf.agent.types import AgentDataModel, AgentRunInfo


class Choice(AgentDataModel):
    value: int


@pytest.fixture
def recorded_waits(monkeypatch):
    waits = []
    original_sleep = asyncio.sleep

    async def sleep(delay):
        waits.append(delay)
        await original_sleep(0)

    monkeypatch.setattr(agent_exec.asyncio, "sleep", sleep)
    return waits


def _response(info, value=2):
    return ModelResponse(
        parts=[ToolCallPart(info.output_tools[0].name, {"value": value})],
        usage=RequestUsage(input_tokens=11, output_tokens=3),
    )


@pytest.mark.parametrize("host", ["sync", "async", "notebook"])
def test_rate_limit_retries_same_request_without_reexecuting_tools(
    recorded_waits, host
):
    calls = []
    tools = []
    attempts = []

    async def measure():
        tools.append("measured")
        return {"measured": 2}

    async def respond(messages, info):
        calls.append(messages)
        if len(calls) == 1:
            return ModelResponse(
                parts=[ToolCallPart("measure", {}, tool_call_id="measurement")],
                usage=RequestUsage(input_tokens=7, output_tokens=2),
            )
        if len(calls) in {2, 3}:
            raise ModelHTTPError(429, "test", "Too Many Requests")
        return _response(info)

    kwargs = dict(
        model=FunctionModel(respond),
        output_type=Choice,
        system_prompt="Use the completed measurement.",
        user_prompt="Measure once and select.",
        tools=[measure],
        on_attempt=attempts.append,
    )
    if host == "sync":
        result = agent_exec.run_agent_sync(**kwargs)
    elif host == "async":
        result = asyncio.run(agent_exec.run_agent_async(**kwargs))
    else:

        async def notebook():
            return agent_exec.run_agent_sync(**kwargs)

        result = asyncio.run(notebook())

    assert result.output.value == 2
    assert len(calls) == 4
    assert calls[1] is calls[2] is calls[3]
    assert tools == ["measured"]
    assert recorded_waits == [15, 30]
    assert attempts == [result.runInfo]
    assert result.runInfo.usage.requests == 4
    assert result.runInfo.usage.toolCalls == 1
    assert result.runInfo.usage.inputTokens == 18
    assert result.runInfo.usage.outputTokens == 5
    assert result.runInfo.usage.availability == "partial"
    assert [item.requestIndex for item in result.runInfo.providerFailures] == [2, 3]
    assert all(item.statusCode == 429 for item in result.runInfo.providerFailures)
    assert result.runInfo.validationRetries == []


@pytest.mark.parametrize(
    "status, body",
    [
        (429, {"error": {"code": "insufficient_quota"}}),
        (429, "You exceeded your current quota"),
        (429, "Insufficient credits"),
        (429, "Billing limit reached"),
        (429, "Daily limit reached"),
        (401, "Unauthorized"),
        (400, "This model does not support multimodal inputs"),
        (503, "Service unavailable"),
    ],
)
def test_nontransient_provider_errors_are_recorded_without_retry(
    recorded_waits, status, body
):
    error = ModelHTTPError(status, "test", body)
    requests = []

    async def fail(messages, info):
        requests.append(messages)
        raise error

    with pytest.raises(ModelHTTPError) as caught:
        agent_exec.run_agent_sync(
            model=FunctionModel(fail),
            output_type=Choice,
            system_prompt="Use evidence.",
            user_prompt="Assess evidence.",
        )
    assert caught.value is error
    assert len(requests) == 1
    assert recorded_waits == []
    saved = error.agent_run_info
    assert saved.usage.requests == 1
    assert saved.usage.availability == "unavailable"
    assert len(saved.providerFailures) == 1
    assert saved.providerFailures[0].retryDelaySeconds is None
    assert saved.validationRetries == []


@pytest.mark.parametrize(
    "source", ["direct", "cause", "date", "past", "zero", "invalid"]
)
def test_retry_after_headers_are_preserved(recorded_waits, source):
    calls = []
    wait = (
        format_datetime(
            datetime.now(timezone.utc)
            + timedelta(seconds=40 if source == "date" else -40)
        )
        if source in {"date", "past"}
        else "0"
        if source == "zero"
        else "invalid"
        if source == "invalid"
        else "7"
    )
    error = ModelHTTPError(
        429,
        "test",
        "Too Many Requests",
        headers={"Retry-After": wait} if source != "cause" else None,
    )
    if source == "cause":
        response = httpx.Response(429, headers={"Retry-After": wait})
        original = RuntimeError("Throttled request")
        original.response = response
        error.__cause__ = original

    async def respond(messages, info):
        calls.append(messages)
        if len(calls) == 1:
            raise error
        return _response(info)

    result = agent_exec.run_agent_sync(
        model=FunctionModel(respond),
        output_type=Choice,
        system_prompt="Use evidence.",
        user_prompt="Assess evidence.",
    )
    assert len(calls) == 2
    if source == "date":
        assert 30 < recorded_waits[0] <= 40
    elif source in {"past", "zero"}:
        assert recorded_waits == [0]
    else:
        assert recorded_waits == [15 if source == "invalid" else 7]
    assert result.runInfo.providerFailures[0].retryDelaySeconds == recorded_waits[0]


@pytest.mark.parametrize(
    "header, request_limit, expected_calls, expected_waits",
    [
        (None, 10, 4, [15, 30, 60]),
        ("80", 10, 2, [80]),
        ("121", 10, 1, []),
        (None, 2, 2, [15]),
    ],
)
def test_retries_respect_attempt_wait_and_request_limits(
    recorded_waits, header, request_limit, expected_calls, expected_waits
):
    calls = []
    error = ModelHTTPError(
        429,
        "test",
        "Too Many Requests",
        headers={"Retry-After": header} if header else None,
    )

    async def fail(messages, info):
        calls.append(messages)
        raise error

    with pytest.raises(ModelHTTPError) as caught:
        agent_exec.run_agent_sync(
            model=FunctionModel(fail),
            output_type=Choice,
            system_prompt="Use evidence.",
            user_prompt="Assess evidence.",
            config=AgentRunConfig(requestLimit=request_limit),
        )
    assert caught.value is error
    assert len(calls) == expected_calls
    assert recorded_waits == expected_waits
    info = error.agent_run_info
    assert info.usage.requests == expected_calls
    assert len(info.providerFailures) == expected_calls
    assert info.providerFailures[-1].retryDelaySeconds is None
    assert info.usage.availability == "unavailable"


def test_rate_limit_wait_cancels_without_another_request(monkeypatch):
    entered_wait = asyncio.Event()
    attempts = []
    requests = []

    async def wait(delay):
        entered_wait.set()
        await asyncio.Future()

    async def fail(messages, info):
        requests.append(messages)
        raise ModelHTTPError(429, "test", "Too Many Requests")

    monkeypatch.setattr(agent_exec.asyncio, "sleep", wait)

    async def interrupt():
        task = asyncio.create_task(
            agent_exec.run_agent_async(
                model=FunctionModel(fail),
                output_type=Choice,
                system_prompt="Use evidence.",
                user_prompt="Assess evidence.",
                on_attempt=attempts.append,
            )
        )
        await entered_wait.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(interrupt())
    assert len(requests) == len(attempts) == 1
    assert attempts[0].status == "failed"
    assert attempts[0].usage.requests == 1
    assert attempts[0].usage.availability == "unavailable"
    assert attempts[0].providerFailures[0].statusCode == 429


def test_transport_retries_do_not_replace_semantic_repair(recorded_waits):
    calls = []

    async def respond(messages, info):
        calls.append(messages)
        if len(calls) == 1:
            raise ModelHTTPError(429, "test", "Too Many Requests")
        return _response(info, 1 if len(calls) == 2 else 2)

    def validate(choice):
        if choice.value != 2:
            raise ValueError("Saved measurement requires value 2")
        return choice

    result = agent_exec.run_agent_sync(
        model=FunctionModel(respond),
        output_type=Choice,
        system_prompt="Use evidence.",
        user_prompt="Assess evidence.",
        output_validator=validate,
    )
    assert result.output.value == 2
    assert result.runInfo.usage.requests == 3
    assert len(result.runInfo.validationRetries) == 1
    assert result.runInfo.validationRetries[0].requestIndex == 2
    assert len(result.runInfo.providerFailures) == 1
    assert result.runInfo.providerFailures[0].requestIndex == 1
    assert recorded_waits == [15]


def test_failed_requests_consume_allowance_before_the_next_agent_step(recorded_waits):
    calls = []
    operations = []

    async def measure():
        operations.append("measured")
        return 2

    async def respond(messages, info):
        calls.append(messages)
        if len(calls) == 1:
            raise ModelHTTPError(429, "test", "Too Many Requests")
        return ModelResponse(
            parts=[ToolCallPart("measure", {})],
            usage=RequestUsage(input_tokens=11, output_tokens=3),
        )

    with pytest.raises(
        UsageLimitExceeded, match="includes failed provider requests"
    ) as caught:
        agent_exec.run_agent_sync(
            model=FunctionModel(respond),
            output_type=Choice,
            system_prompt="Use evidence.",
            user_prompt="Measure once, then decide.",
            tools=[measure],
            config=AgentRunConfig(requestLimit=2),
        )
    assert len(calls) == 2
    assert operations == ["measured"]
    assert caught.value.agent_run_info.usage.requests == 2
    assert len(caught.value.agent_run_info.providerFailures) == 1


def test_total_wait_allowance_is_shared_across_agent_steps(recorded_waits):
    calls = []
    operations = []

    async def measure():
        operations.append("measured")
        return 2

    async def respond(messages, info):
        calls.append(messages)
        if len(calls) != 2:
            raise ModelHTTPError(
                429, "test", "Too Many Requests", headers={"Retry-After": "80"}
            )
        return ModelResponse(
            parts=[ToolCallPart("measure", {})],
            usage=RequestUsage(input_tokens=11, output_tokens=3),
        )

    with pytest.raises(ModelHTTPError) as caught:
        agent_exec.run_agent_sync(
            model=FunctionModel(respond),
            output_type=Choice,
            system_prompt="Use evidence.",
            user_prompt="Measure once, then decide.",
            tools=[measure],
        )
    assert len(calls) == 3
    assert operations == ["measured"]
    assert recorded_waits == [80]
    assert caught.value.agent_run_info.usage.requests == 3
    assert [
        row.retryDelaySeconds for row in caught.value.agent_run_info.providerFailures
    ] == [80, None]


def test_old_run_serialization_does_not_gain_empty_provider_records():
    saved = AgentRunInfo().model_dump(mode="json")
    assert "providerFailures" not in saved
    assert AgentRunInfo.model_validate(saved).model_dump(mode="json") == saved


def test_schema_retry_uses_actual_request_ordinal_after_transient_quota(recorded_waits):
    calls = []

    async def respond(messages, info):
        calls.append(messages)
        if len(calls) == 1:
            raise ModelHTTPError(429, "test", "Per-minute quota exceeded")
        return _response(info, "invalid" if len(calls) == 2 else 2)

    result = agent_exec.run_agent_sync(
        model=FunctionModel(respond),
        output_type=Choice,
        system_prompt="Use evidence.",
        user_prompt="Assess evidence.",
    )
    assert result.output.value == 2
    assert result.runInfo.usage.requests == 3
    assert len(result.runInfo.validationRetries) == 1
    assert result.runInfo.validationRetries[0].source == "schema"
    assert result.runInfo.validationRetries[0].requestIndex == 2
    assert result.runInfo.providerFailures[0].requestIndex == 1
    assert recorded_waits == [15]
