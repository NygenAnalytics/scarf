"""Tests for shared Scarf agent execution and configuration."""

import asyncio
import json
import threading

import httpx
import pydantic_ai.providers.openai as openai_provider_module
from openai import AsyncOpenAI
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.providers.openai import OpenAIProvider

from scarf.agent import CovariateCharacterization, FeatureCharacterization, IngestResult
from scarf.agent.config.agent_exec import _model_name, run_agent, run_agent_sync
from scarf.agent.config import AgentRunConfig, get_model_settings, get_usage_limits
from scarf.agent.tools import artifact_reference, core_artifact_reference
from scarf.agent.types import (
    AgentDataModel,
    AgentExecutionResult,
    AgentRunInfo,
    AgentUsageInfo,
    ArtifactReferenceModel,
    Decision,
    EvidenceItem,
    NeedsInput,
    StageResult,
    ToolCallInfo,
)


class ExampleOutput(AgentDataModel):
    value: str = ""

    @classmethod
    def get_example(cls) -> "ExampleOutput":
        return cls(value="complete")


def test_shared_models_have_blank_and_example_constructors() -> None:
    models = (
        AgentRunConfig,
        CovariateCharacterization,
        FeatureCharacterization,
        IngestResult,
        AgentExecutionResult,
        AgentRunInfo,
        AgentUsageInfo,
        ArtifactReferenceModel,
        Decision,
        EvidenceItem,
        NeedsInput,
        StageResult,
        ToolCallInfo,
        ExampleOutput,
    )
    for model in models:
        assert isinstance(model.get_blank(), model)
        assert isinstance(model.get_example(), model)
        assert all("_" not in field_name for field_name in model.model_fields)


def test_four_agent_objects_are_public() -> None:
    import scarf.agent as agent_package

    assert agent_package.DataEnrichmentAgent.__name__ == "DataEnrichmentAgent"
    assert agent_package.ExperimentalContextAgent.__name__ == "ExperimentalContextAgent"
    assert agent_package.ParameterTuningAgent.__name__ == "ParameterTuningAgent"
    assert (
        agent_package.BiologicalInterpretationAgent.__name__
        == "BiologicalInterpretationAgent"
    )


def test_model_settings_disable_thinking_across_provider_shapes() -> None:
    settings = get_model_settings(
        AgentRunConfig(
            timeoutSeconds=42,
            sequentialTools=True,
            extraModelSettings={"max_tokens": 123},
        )
    )

    assert settings["thinking"] is False
    assert settings["parallel_tool_calls"] is False
    assert settings["timeout"] == 42
    assert settings["temperature"] == 0
    assert settings["max_tokens"] == 123
    assert "gtemperature" not in settings
    assert "openai_reasoning_effort" not in settings
    assert settings["extra_body"] == {
        "thinking": {"type": "disabled"},
        "reasoning_effort": "none",
        "chat_template_kwargs": {"thinking": False},
        "reasoning": {"enabled": False},
    }

    for profile in (
        "auto",
        "unified",
        "ollama",
        "chatTemplate",
        "thinkingBody",
        "reasoningBody",
    ):
        assert (
            get_model_settings(AgentRunConfig(thinkingOffProfile=profile))["extra_body"]
            == settings["extra_body"]
        )
    assert get_model_settings(
        AgentRunConfig(extraModelSettings={"extra_body": {"custom": False}}),
        model="ollama:qwen3",
    )["extra_body"] == {"custom": False}


def test_agent_run_config_clamps_per_stage_limits() -> None:
    original = AgentRunConfig(
        requestLimit=128,
        toolCallLimit=64,
        outputTokenLimit=None,
        timeoutSeconds=1800,
        extraModelSettings={"service_tier": "default"},
    )

    bounded = original.with_limits(
        request_limit=6,
        tool_call_limit=4,
        output_token_limit=4096,
        timeout_seconds=600,
    )

    assert bounded.requestLimit == 6
    assert bounded.toolCallLimit == 4
    assert bounded.outputTokenLimit == 4096
    assert bounded.timeoutSeconds == 600
    assert bounded.extraModelSettings == {"service_tier": "default"}
    assert original.requestLimit == 128
    assert original.outputTokenLimit is None

    already_stricter = AgentRunConfig(
        requestLimit=2,
        toolCallLimit=1,
        outputTokenLimit=512,
        timeoutSeconds=30,
    ).with_limits(
        request_limit=6,
        tool_call_limit=4,
        output_token_limit=4096,
        timeout_seconds=600,
    )
    assert already_stricter.requestLimit == 2
    assert already_stricter.toolCallLimit == 1
    assert already_stricter.outputTokenLimit == 512
    assert already_stricter.timeoutSeconds == 30


def test_baseten_request_disables_reasoning_at_wire_level() -> None:
    bodies: list[dict[str, object]] = []

    async def execute() -> dict[str, object]:
        async def handle(request: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "done"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            openai_client = AsyncOpenAI(
                api_key="test-key",
                base_url="https://inference.baseten.co/v1",
                http_client=client,
            )
            model = OpenAIChatModel(
                "deepseek-ai/DeepSeek-V4-Flash-0731",
                provider=OpenAIProvider(openai_client=openai_client),
            )
            settings = get_model_settings(model=model)
            await model.request(
                [ModelRequest(parts=[UserPromptPart("hello")])],
                settings,
                ModelRequestParameters(),
            )
            return settings

    settings = asyncio.run(execute())

    assert settings["openai_reasoning_effort"] == "none"
    assert len(bodies) == 1
    assert bodies[0]["reasoning_effort"] == "none"
    assert bodies[0]["temperature"] == 0
    assert bodies[0]["max_completion_tokens"] == 32768


def test_model_name_preserves_string_model_identifier() -> None:
    assert _model_name("openai:gpt-5-mini") == "openai:gpt-5-mini"


def test_agent_artifact_models_round_trip_to_exact_core_references() -> None:
    model = ArtifactReferenceModel(
        assay="RNA",
        kind="cluster_labels",
        artifactId="b" * 64,
    )

    core_ref = core_artifact_reference(model)

    assert core_ref.kind == "cluster_labels"
    assert core_ref.artifact_id == "b" * 64
    assert artifact_reference(core_ref) == model


def test_usage_limits_are_derived_from_public_config() -> None:
    limits = get_usage_limits(
        AgentRunConfig(
            requestLimit=3,
            toolCallLimit=4,
            inputTokenLimit=100,
            outputTokenLimit=200,
            totalTokenLimit=250,
        )
    )

    assert limits.request_limit == 3
    assert limits.tool_calls_limit == 4
    assert limits.input_tokens_limit == 100
    assert limits.output_tokens_limit == 200
    assert limits.total_tokens_limit == 250


def test_sync_runner_records_only_function_tool_calls() -> None:
    async def inspect_value() -> dict[str, str]:
        return {"status": "observed"}

    def reply(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        returns = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if not returns:
            return ModelResponse(
                parts=[ToolCallPart(tool_name="inspect_value", args="{}")]
            )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args=ExampleOutput.get_example().model_dump(),
                )
            ]
        )

    result = run_agent_sync(
        model=FunctionModel(reply),
        output_type=ExampleOutput,
        system_prompt="Use the tool.",
        user_prompt="Inspect the value.",
        tools=[inspect_value],
        name="example-agent",
    )

    assert result.output == ExampleOutput.get_example()
    assert result.runInfo.agentName == "example-agent"
    assert result.runInfo.usage.toolCalls == 1
    assert [call.toolName for call in result.runInfo.toolCalls] == ["inspect_value"]


def test_sync_runner_does_not_attribute_history_tool_calls_to_current_run() -> None:
    async def inspect_value() -> dict[str, str]:
        return {"status": "observed"}

    async def second_reply(
        _messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args={"value": "second"},
                )
            ]
        )

    history = [
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="inspect_value",
                    args="{}",
                    tool_call_id="history-call",
                )
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="inspect_value",
                    content={"status": "observed"},
                    tool_call_id="history-call",
                )
            ]
        ),
    ]
    result = run_agent_sync(
        model=FunctionModel(second_reply),
        output_type=ExampleOutput,
        system_prompt="Return structured output.",
        user_prompt="Return now.",
        tools=[inspect_value],
        message_history=history,
    )

    assert result.runInfo.usage.toolCalls == 0
    assert result.runInfo.toolCalls == []


def test_sync_runner_works_when_an_event_loop_is_already_running() -> None:
    async def inspect_value() -> dict[str, str]:
        return {"status": "observed"}

    def reply(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        returns = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if not returns:
            return ModelResponse(
                parts=[ToolCallPart(tool_name="inspect_value", args="{}")]
            )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args=ExampleOutput.get_example().model_dump(),
                )
            ]
        )

    async def call_from_running_loop() -> object:
        return run_agent_sync(
            model=FunctionModel(reply),
            output_type=ExampleOutput,
            system_prompt="Use the tool.",
            user_prompt="Inspect the value.",
            tools=[inspect_value],
            name="notebook-host",
        )

    result = asyncio.run(call_from_running_loop())

    assert result.output == ExampleOutput.get_example()
    assert result.runInfo.agentName == "notebook-host"
    assert [call.toolName for call in result.runInfo.toolCalls] == ["inspect_value"]


def test_sync_runner_closes_and_reopens_owned_client_between_notebook_calls(
    monkeypatch,
) -> None:
    clients: list[httpx.AsyncClient] = []
    request_loops: list[asyncio.AbstractEventLoop] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        request_loops.append(asyncio.get_running_loop())
        payload = json.loads(request.content)
        output_tool_name = payload["tools"][-1]["function"]["name"]
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-scarf-test",
                "object": "chat.completion",
                "created": 0,
                "model": "mock-openai-compatible",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-result",
                                    "type": "function",
                                    "function": {
                                        "name": output_tool_name,
                                        "arguments": json.dumps(
                                            ExampleOutput.get_example().model_dump()
                                        ),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )

    def make_client(*_args, **_kwargs) -> httpx.AsyncClient:
        client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        clients.append(client)
        return client

    monkeypatch.setattr(
        openai_provider_module,
        "create_async_http_client",
        make_client,
    )
    model = OpenAIChatModel(
        "mock-openai-compatible",
        provider=OpenAIProvider(
            base_url="https://example.test/v1",
            api_key="test-key",
        ),
    )

    async def call_twice_from_running_loop() -> list[AgentExecutionResult]:
        return [
            run_agent_sync(
                model=model,
                output_type=ExampleOutput,
                system_prompt="Return structured output.",
                user_prompt="Return now.",
            )
            for _ in range(2)
        ]

    results = asyncio.run(call_twice_from_running_loop())

    assert [result.output for result in results] == [
        ExampleOutput.get_example(),
        ExampleOutput.get_example(),
    ]
    assert len(clients) == 2
    assert all(client.is_closed for client in clients)
    assert len(request_loops) == 2
    assert request_loops[0] is not request_loops[1]


def test_async_runner_returns_structured_output() -> None:
    result = asyncio.run(
        run_agent(
            model=TestModel(custom_output_args={"value": "async"}),
            output_type=ExampleOutput,
            system_prompt="Return structured output.",
            user_prompt="Return now.",
            name="async-example-agent",
        )
    )

    assert result.output == ExampleOutput(value="async")
    assert result.runInfo.agentName == "async-example-agent"
    assert result.runInfo.usage.toolCalls == 0


def test_async_runner_does_not_run_sync_function_model_on_event_loop() -> None:
    callback_thread = 0
    event_loop_thread = threading.get_ident()

    def reply(
        _messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal callback_thread
        callback_thread = threading.get_ident()
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args=ExampleOutput.get_example().model_dump(),
                )
            ]
        )

    result = asyncio.run(
        run_agent(
            model=FunctionModel(reply),
            output_type=ExampleOutput,
            system_prompt="Return structured output.",
            user_prompt="Return now.",
        )
    )

    assert result.output == ExampleOutput.get_example()
    assert callback_thread != event_loop_thread


def test_runner_converts_grounding_errors_into_model_retries() -> None:
    requests = 0

    async def reply(
        _messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal requests
        requests += 1
        value = "invalid" if requests == 1 else "grounded"
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args={"value": value},
                )
            ]
        )

    def validate_output(output: ExampleOutput) -> ExampleOutput:
        if output.value != "grounded":
            raise ValueError("output is not grounded")
        return output

    result = run_agent_sync(
        model=FunctionModel(reply),
        output_type=ExampleOutput,
        system_prompt="Return grounded output.",
        user_prompt="Return now.",
        output_validator=validate_output,
    )

    assert result.output == ExampleOutput(value="grounded")
    assert requests == 2
