"""Common bounded execution for the four Scarf domain agents."""

import asyncio
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from inspect import isawaitable, iscoroutinefunction
from typing import Any

from ...utils.logging import logger
from ..types import (
    AgentExecutionResult,
    AgentRunInfo,
    AgentUsageInfo,
    ToolCallInfo,
)
from . import AgentRunConfig, get_model_settings, get_usage_limits
from ._deps import require_pydantic_ai

__all__ = ["run_agent", "run_agent_sync"]


def _tool_definitions(
    tools: Sequence[Callable[..., Any] | Any],
    config: AgentRunConfig,
) -> list[Any]:
    from pydantic_ai import Tool

    definitions: list[Any] = []
    for tool in tools:
        if isinstance(tool, Tool):
            definitions.append(tool)
        else:
            definitions.append(
                Tool(
                    tool,
                    sequential=config.sequentialTools,
                    timeout=config.timeoutSeconds,
                )
            )
    return definitions


def _model_name(model: Any) -> str:
    if isinstance(model, str):
        return model
    name = getattr(model, "model_name", None)
    if isinstance(name, str):
        return name
    return type(model).__name__


def _normalize_model(model: Any) -> Any:
    """Avoid worker-thread deadlocks for synchronous test model callbacks."""
    from pydantic_ai.models.function import FunctionModel

    if sys.version_info < (3, 14):
        return model
    function = getattr(model, "function", None)
    if not isinstance(model, FunctionModel) or function is None:
        return model
    if iscoroutinefunction(function):
        return model

    async def async_function(messages: Any, info: Any) -> Any:
        response = function(messages, info)
        if isawaitable(response):
            return await response
        return response

    stream_function = model.stream_function
    if stream_function is None:
        return FunctionModel(
            async_function,
            model_name=model.model_name,
            profile=model.profile,
            settings=model.settings,
        )
    return FunctionModel(
        async_function,
        stream_function=stream_function,
        model_name=model.model_name,
        profile=model.profile,
        settings=model.settings,
    )


def _tool_names(tools: Sequence[Callable[..., Any] | Any]) -> set[str]:
    names: set[str] = set()
    for tool in tools:
        name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
        if isinstance(name, str):
            names.add(name)
    return names


def _tool_calls(
    messages: Sequence[Any],
    *,
    allowed_names: set[str],
) -> list[ToolCallInfo]:
    from pydantic_ai.messages import ToolCallPart

    calls: list[ToolCallInfo] = []
    for message in messages:
        for part in getattr(message, "parts", ()):
            if not isinstance(part, ToolCallPart):
                continue
            if part.tool_name not in allowed_names:
                continue
            calls.append(
                ToolCallInfo(
                    toolName=part.tool_name,
                    callId=part.tool_call_id or "",
                    arguments=part.args_as_dict(),
                )
            )
    return calls


def _usage_info(usage: Any, *, tool_calls: int) -> AgentUsageInfo:
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    total_tokens = int(
        getattr(usage, "total_tokens", input_tokens + output_tokens)
        or input_tokens + output_tokens
    )
    return AgentUsageInfo(
        inputTokens=input_tokens,
        outputTokens=output_tokens,
        totalTokens=total_tokens,
        requests=int(getattr(usage, "requests", 0) or 0),
        toolCalls=tool_calls,
    )


def _build_agent(
    *,
    model: Any,
    output_type: Any,
    system_prompt: str,
    tools: Sequence[Callable[..., Any] | Any],
    deps_type: type[Any] | None,
    config: AgentRunConfig,
    name: str | None,
    output_validator: Callable[[Any], Any] | None,
    normalize_sync_function_model: bool,
) -> Any:
    require_pydantic_ai()
    from pydantic_ai import Agent

    agent = Agent(
        _normalize_model(model) if normalize_sync_function_model else model,
        output_type=output_type,
        system_prompt=system_prompt,
        deps_type=deps_type or object,
        name=name,
        model_settings=get_model_settings(config, model=model),
        retries=config.retries,
        tools=_tool_definitions(tools, config),
        tool_timeout=config.timeoutSeconds,
    )
    if output_validator is not None:

        @agent.output_validator
        async def validate_output(output: Any) -> Any:
            from pydantic_ai import ModelRetry

            try:
                validated = output_validator(output)
                if isawaitable(validated):
                    return await validated
                return validated
            except ModelRetry as exc:
                logger.warning(
                    f"Agent {name or 'unnamed'} requested a structured-output "
                    f"retry: {str(exc)[:500]}"
                )
                raise
            except (TypeError, ValueError) as exc:
                logger.warning(
                    f"Agent {name or 'unnamed'} rejected structured output: "
                    f"{str(exc)[:500]}"
                )
                raise ModelRetry(str(exc)) from exc

    return agent


def _execution_result(
    *,
    result: Any,
    model: Any,
    name: str | None,
    started: float,
    tools: Sequence[Callable[..., Any] | Any],
) -> AgentExecutionResult:
    messages = result.new_messages()
    calls = _tool_calls(messages, allowed_names=_tool_names(tools))
    execution = AgentExecutionResult(
        output=result.output,
        runInfo=AgentRunInfo(
            agentName=name or "",
            modelName=_model_name(model),
            runId=str(getattr(result, "run_id", "")),
            durationSeconds=time.monotonic() - started,
            usage=_usage_info(result.usage, tool_calls=len(calls)),
            toolCalls=calls,
        ),
    )
    usage = execution.runInfo.usage
    logger.info(
        f"Agent {name or 'unnamed'} completed in "
        f"{execution.runInfo.durationSeconds:.2f}s: requests={usage.requests}, "
        f"tool_calls={usage.toolCalls}, input_tokens={usage.inputTokens}, "
        f"output_tokens={usage.outputTokens}"
    )
    return execution


def run_agent_sync(
    *,
    model: Any,
    output_type: Any,
    system_prompt: str,
    user_prompt: str,
    tools: Sequence[Callable[..., Any] | Any] = (),
    deps_type: type[Any] | None = None,
    deps: Any = None,
    config: AgentRunConfig | None = None,
    name: str | None = None,
    output_validator: Callable[[Any], Any] | None = None,
    message_history: Sequence[Any] = (),
) -> AgentExecutionResult:
    """Run one synchronous agent loop and return its bounded audit record.

    Jupyter and other hosts already have a running event loop. Pydantic AI's
    asynchronous runner cannot drive that loop synchronously, so this hops to
    a worker thread in that case. Entering the agent context ensures provider
    HTTP clients are closed on that worker's event loop before it exits.
    """

    async def execute() -> AgentExecutionResult:
        run_config = config or AgentRunConfig()
        agent_name = name or "unnamed"
        usage_limits = get_usage_limits(run_config)
        logger.info(
            f"Starting agent {agent_name}: model={_model_name(model)}, "
            f"tools={len(tools)}, request_limit={run_config.requestLimit}, "
            f"tool_call_limit={run_config.toolCallLimit}, retries={run_config.retries}, "
            f"per_response_output_limit={run_config.outputTokenLimit}, "
            f"run_output_limit={usage_limits.output_tokens_limit}"
        )
        agent = _build_agent(
            model=model,
            output_type=output_type,
            system_prompt=system_prompt,
            tools=tools,
            deps_type=deps_type,
            config=run_config,
            name=name,
            output_validator=output_validator,
            normalize_sync_function_model=True,
        )
        started = time.monotonic()
        try:
            async with agent:
                result = await agent.run(
                    user_prompt,
                    deps=deps,
                    message_history=message_history,
                    usage_limits=usage_limits,
                )
        except Exception as exc:
            error_detail = str(exc).replace("\n", " ").strip()[:500]
            cause = exc.__cause__
            if cause is not None and cause is not exc:
                cause_detail = str(cause).replace("\n", " ").strip()[:500]
                error_detail = (
                    f"{error_detail}; caused by {type(cause).__name__}: {cause_detail}"
                )
            logger.error(
                f"Agent {agent_name} failed after {time.monotonic() - started:.2f}s: "
                f"{type(exc).__name__}: {error_detail}"
            )
            raise
        return _execution_result(
            result=result,
            model=model,
            name=name,
            started=started,
            tools=tools,
        )

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(execute())

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, execute()).result()


async def run_agent(
    *,
    model: Any,
    output_type: Any,
    system_prompt: str,
    user_prompt: str,
    tools: Sequence[Callable[..., Any] | Any] = (),
    deps_type: type[Any] | None = None,
    deps: Any = None,
    config: AgentRunConfig | None = None,
    name: str | None = None,
    output_validator: Callable[[Any], Any] | None = None,
    message_history: Sequence[Any] = (),
) -> AgentExecutionResult:
    """Run one asynchronous agent loop and return its bounded audit record."""
    run_config = config or AgentRunConfig()
    agent_name = name or "unnamed"
    usage_limits = get_usage_limits(run_config)
    logger.info(
        f"Starting agent {agent_name}: model={_model_name(model)}, "
        f"tools={len(tools)}, request_limit={run_config.requestLimit}, "
        f"tool_call_limit={run_config.toolCallLimit}, retries={run_config.retries}, "
        f"per_response_output_limit={run_config.outputTokenLimit}, "
        f"run_output_limit={usage_limits.output_tokens_limit}"
    )
    agent = _build_agent(
        model=model,
        output_type=output_type,
        system_prompt=system_prompt,
        tools=tools,
        deps_type=deps_type,
        config=run_config,
        name=name,
        output_validator=output_validator,
        normalize_sync_function_model=False,
    )
    started = time.monotonic()
    try:
        async with agent:
            result = await agent.run(
                user_prompt,
                deps=deps,
                message_history=message_history,
                usage_limits=usage_limits,
            )
    except Exception as exc:
        error_detail = str(exc).replace("\n", " ").strip()[:500]
        cause = exc.__cause__
        if cause is not None and cause is not exc:
            cause_detail = str(cause).replace("\n", " ").strip()[:500]
            error_detail = (
                f"{error_detail}; caused by {type(cause).__name__}: {cause_detail}"
            )
        logger.error(
            f"Agent {agent_name} failed after {time.monotonic() - started:.2f}s: "
            f"{type(exc).__name__}: {error_detail}"
        )
        raise
    return _execution_result(
        result=result,
        model=model,
        name=name,
        started=started,
        tools=tools,
    )
