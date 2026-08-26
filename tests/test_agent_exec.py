"""Tests for shared Scarf agent execution and configuration."""

import asyncio
import threading

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

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
    assert settings["max_tokens"] == 123
    assert "extra_body" not in settings

    assert get_model_settings(model="ollama:qwen3")["extra_body"] == {"think": False}
    assert get_model_settings(AgentRunConfig(thinkingOffProfile="chatTemplate"))[
        "extra_body"
    ] == {"chat_template_kwargs": {"thinking": False}}
    assert get_model_settings(AgentRunConfig(thinkingOffProfile="thinkingBody"))[
        "extra_body"
    ] == {"thinking": {"type": "disabled"}}
    assert get_model_settings(AgentRunConfig(thinkingOffProfile="reasoningBody"))[
        "extra_body"
    ] == {"reasoning": {"enabled": False}}
    assert get_model_settings(
        AgentRunConfig(extraModelSettings={"extra_body": {"custom": False}}),
        model="ollama:qwen3",
    )["extra_body"] == {"custom": False}


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

    async def reply(
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
