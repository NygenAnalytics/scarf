"""Common bounded execution for the four Scarf domain agents."""

import asyncio
import json
import math
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from inspect import isawaitable, iscoroutinefunction
from threading import Event, Lock
from typing import TYPE_CHECKING, Any, Literal

from ...utils.logging import logger
from .._deps import require_pydantic_ai
from ..types import (
    AgentExecutionResult,
    AgentProviderFailure,
    AgentRunInfo,
    AgentUsageInfo,
    AgentValidationRetry,
    ToolCallInfo,
)
from . import AgentRunConfig, get_model_settings, get_usage_limits

if TYPE_CHECKING:
    from pydantic_ai.messages import UserContent
else:
    UserContent = Any

type AgentUserPrompt = str | Sequence[UserContent]
type ImageMediaType = Literal["image/png", "image/jpeg", "image/webp"]

_MAX_VISUAL_EVIDENCE_ITEMS = 8
_MAX_VISUAL_EVIDENCE_ITEM_BYTES = 4 * 1024 * 1024
_MAX_VISUAL_EVIDENCE_TOTAL_BYTES = 16 * 1024 * 1024
_RATE_LIMIT_DELAYS = (15.0, 30.0, 60.0)
_RATE_LIMIT_WAIT_LIMIT = 120.0

__all__ = [
    "AgentUserPrompt",
    "build_visual_evidence_prompt",
    "ImageEvidence",
    "ImageInputUnsupportedError",
    "ImageMediaType",
    "run_agent",
    "run_agent_async",
    "run_agent_sync",
]


@dataclass(frozen=True, slots=True)
class ImageEvidence:
    """One in-memory image supplied to a bounded agent comparison."""

    identifier: str
    data: bytes
    media_type: ImageMediaType = "image/png"


class ImageInputUnsupportedError(RuntimeError):
    """The configured model or provider rejected image input."""


@dataclass
class _ProviderRequestAudit:
    requests: int = 0
    waited: float = 0.0
    failures: list[AgentProviderFailure] = field(default_factory=list)
    response_requests: list[int] = field(default_factory=list)


def _rate_limit_delay(error: Exception, retry: int) -> float | None:
    """Retry temporary throttling, preserving provider timing and hard quotas."""
    from pydantic_ai.exceptions import ModelHTTPError

    if not isinstance(error, ModelHTTPError) or error.status_code != 429:
        return None
    detail = describe_agent_error(error).casefold()
    if any(
        marker in detail
        for marker in (
            "insufficient_quota",
            "insufficient quota",
            "exceeded your current quota",
            "insufficient credit",
            "insufficient balance",
            "billing",
            "payment required",
            "spending limit",
            "daily limit",
            "monthly limit",
        )
    ):
        return None
    if retry >= len(_RATE_LIMIT_DELAYS):
        return None
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen and len(seen) < 8:
        seen.add(id(current))
        headers = getattr(current, "headers", None) or getattr(
            getattr(current, "response", None), "headers", None
        )
        if headers:
            parsed = ModelHTTPError(429, error.model_name, headers=headers).retry_after
            if parsed is not None and math.isfinite(parsed):
                return parsed
        current = current.__cause__ or (
            current.__context__ if not current.__suppress_context__ else None
        )
    return _RATE_LIMIT_DELAYS[retry]


def _rate_limited_model(
    model: Any,
    *,
    config: AgentRunConfig,
    audit: _ProviderRequestAudit,
    name: str | None,
) -> Any:
    """Retry only a failed request; completed agent tools are never replayed."""
    from pydantic_ai.exceptions import UsageLimitExceeded
    from pydantic_ai.models.wrapper import WrapperModel

    class RateLimitedModel(WrapperModel):
        async def request(
            self, messages: Any, model_settings: Any, model_request_parameters: Any
        ) -> Any:
            retry = 0
            while True:
                if audit.requests >= config.requestLimit:
                    raise UsageLimitExceeded(
                        "The model request limit includes failed provider requests; "
                        f"all {config.requestLimit} available requests were used"
                    )
                audit.requests += 1
                try:
                    response = await self.wrapped.request(
                        messages, model_settings, model_request_parameters
                    )
                    audit.response_requests.append(audit.requests)
                    return response
                except Exception as exc:
                    delay = _rate_limit_delay(exc, retry)
                    if delay is not None and (
                        audit.requests >= config.requestLimit
                        or audit.waited + delay > _RATE_LIMIT_WAIT_LIMIT
                    ):
                        delay = None
                    audit.failures.append(
                        AgentProviderFailure(
                            requestIndex=audit.requests,
                            statusCode=getattr(exc, "status_code", None),
                            error=describe_agent_error(exc),
                            retryDelaySeconds=delay,
                        )
                    )
                    if delay is None:
                        raise
                    logger.warning(
                        f"Agent {name or 'unnamed'}: provider rate limit; retrying "
                        f"the same request in {delay:g}s "
                        f"({retry + 1}/{len(_RATE_LIMIT_DELAYS)} retries). "
                        "Completed tools and scientific evidence are preserved."
                    )
                    audit.waited += delay
                    await asyncio.sleep(delay)
                    retry += 1

    return RateLimitedModel(model)


def describe_agent_error(error: BaseException, *, limit: int = 8000) -> str:
    """Describe the concrete cause chain without traceback or response payloads."""
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen and len(parts) < 8:
        seen.add(id(current))
        detail = " ".join(str(current).split())
        notes = getattr(current, "__notes__", ())
        if notes:
            detail += "; " + "; ".join(str(note) for note in notes)
        parts.append(f"{type(current).__name__}: {detail[:2000]}")
        current = current.__cause__ or (
            current.__context__ if not current.__suppress_context__ else None
        )
    return "; caused by ".join(parts)[:limit]


def _image_input_is_unsupported(exc: Exception) -> bool:
    from pydantic_ai.exceptions import ModelHTTPError, UserError

    if isinstance(exc, UserError):
        detail = str(exc).casefold()
    elif isinstance(exc, ModelHTTPError) and exc.status_code in {400, 415, 422}:
        detail = str(exc.body).casefold()
    else:
        return False
    return any(
        marker in detail
        for marker in (
            "binary content is not supported",
            "binary input is not supported",
            "does not support binary content",
            "does not support multimodal",
            "doesn't support multimodal",
            "image content is not supported",
            "multimodal input is not supported",
            "image input is not supported",
            "image inputs are not supported",
            "images are not supported",
            "does not support image",
            "unsupported image input",
            "only text input",
        )
    )


def build_visual_evidence_prompt(
    prompt: str,
    images: Sequence[ImageEvidence],
) -> tuple[UserContent, ...]:
    """Build bounded Pydantic AI image content without creating report files."""

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Visual evidence prompt must be a non-empty string")
    values = tuple(images)
    if not values:
        raise ValueError("Visual evidence requires at least one image")
    if len(values) > _MAX_VISUAL_EVIDENCE_ITEMS:
        raise ValueError(
            "Visual evidence exceeds the maximum of "
            f"{_MAX_VISUAL_EVIDENCE_ITEMS} images"
        )
    identifiers = [value.identifier for value in values]
    if any(not value or value != value.strip() for value in identifiers):
        raise ValueError("Visual evidence identifiers must be non-empty and trimmed")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Visual evidence identifiers must be unique")
    total_bytes = 0
    for value in values:
        if value.media_type not in {"image/png", "image/jpeg", "image/webp"}:
            raise ValueError(
                f"Visual evidence image {value.identifier!r} has unsupported media type"
            )
        if not isinstance(value.data, bytes) or not value.data:
            raise ValueError("Visual evidence images must contain non-empty bytes")
        if len(value.data) > _MAX_VISUAL_EVIDENCE_ITEM_BYTES:
            raise ValueError(
                f"Visual evidence image {value.identifier!r} exceeds "
                f"{_MAX_VISUAL_EVIDENCE_ITEM_BYTES} bytes"
            )
        total_bytes += len(value.data)
    if total_bytes > _MAX_VISUAL_EVIDENCE_TOTAL_BYTES:
        raise ValueError(
            "Visual evidence exceeds the total byte limit of "
            f"{_MAX_VISUAL_EVIDENCE_TOTAL_BYTES}"
        )

    require_pydantic_ai()
    from pydantic_ai.messages import BinaryContent

    return (
        prompt,
        *(
            BinaryContent(
                data=value.data,
                media_type=value.media_type,
                identifier=value.identifier,
            )
            for value in values
        ),
    )


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
            try:
                arguments = part.args_as_dict()
            except (TypeError, ValueError):
                arguments = {"unparsedArguments": str(part.args)}
            calls.append(
                ToolCallInfo(
                    toolName=part.tool_name,
                    callId=part.tool_call_id or "",
                    arguments=arguments,
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
    validation_failures: list[AgentValidationRetry] | None = None,
    provider_audit: _ProviderRequestAudit | None = None,
) -> Any:
    require_pydantic_ai()
    from pydantic_ai import Agent, RunContext

    normalized_model = (
        _normalize_model(model) if normalize_sync_function_model else model
    )
    if provider_audit is not None:
        normalized_model = _rate_limited_model(
            normalized_model, config=config, audit=provider_audit, name=name
        )
    agent = Agent(
        normalized_model,
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
        async def validate_output(context: RunContext[Any], output: Any) -> Any:
            from pydantic_ai import ModelRetry

            submitted = (
                output.model_dump(mode="json")
                if hasattr(output, "model_dump")
                else str(output)
            )
            try:
                validated = output_validator(output)
                if isawaitable(validated):
                    return await validated
                return validated
            except ModelRetry as exc:
                if validation_failures is not None:
                    validation_failures.append(
                        AgentValidationRetry(
                            source="output",
                            requestIndex=(
                                provider_audit.requests
                                if provider_audit is not None
                                else context.usage.requests
                            ),
                            message=str(exc),
                            response=submitted,
                        )
                    )
                logger.warning(
                    f"Agent {name or 'unnamed'} requested a structured-output "
                    f"retry: {str(exc)[:500]}"
                )
                raise
            except (TypeError, ValueError) as exc:
                if validation_failures is not None:
                    validation_failures.append(
                        AgentValidationRetry(
                            source="output",
                            requestIndex=(
                                provider_audit.requests
                                if provider_audit is not None
                                else context.usage.requests
                            ),
                            message=str(exc),
                            response=submitted,
                        )
                    )
                logger.warning(
                    f"Agent {name or 'unnamed'} rejected structured output: "
                    f"{str(exc)[:500]}"
                )
                raise ModelRetry(str(exc)) from exc

    return agent


def _run_info(
    *,
    messages: Sequence[Any],
    usage: Any,
    model: Any,
    name: str | None,
    started: float,
    tools: Sequence[Callable[..., Any] | Any],
    validation_failures: Sequence[AgentValidationRetry],
    error: BaseException | None = None,
    provider_audit: _ProviderRequestAudit | None = None,
) -> AgentRunInfo:
    from pydantic import ValidationError
    from pydantic_ai.messages import ModelResponse, RetryPromptPart, ToolCallPart

    calls = _tool_calls(messages, allowed_names=_tool_names(tools))
    reported_usage = _usage_info(usage, tool_calls=len(calls))
    if provider_audit is not None:
        reported_usage.requests = provider_audit.requests
    responses = [message for message in messages if isinstance(message, ModelResponse)]
    measured = [message for message in responses if message.usage.has_values()]
    reported_usage.availability = (
        "reported"
        if measured
        and not (provider_audit is not None and provider_audit.failures)
        and len(measured) == len(responses) == reported_usage.requests
        and not (
            error is not None
            and messages
            and not isinstance(messages[-1], ModelResponse)
        )
        and all(
            getattr(message, "state", "complete") == "complete" for message in responses
        )
        else "partial"
        if measured
        else "unavailable"
    )
    retries = list(validation_failures)
    semantic_messages = {retry.message for retry in retries}
    pending: dict[str, Any] = {}
    tool_names = _tool_names(tools)
    request_index = 0
    response_index = 0
    for message in messages:
        if isinstance(message, ModelResponse):
            request_index = (
                provider_audit.response_requests[response_index]
                if provider_audit is not None
                and response_index < len(provider_audit.response_requests)
                else request_index + 1
            )
            response_index += 1
        for part in getattr(message, "parts", ()):
            if isinstance(part, ToolCallPart):
                pending[part.tool_call_id] = part.args
            elif isinstance(part, RetryPromptPart):
                detail = (
                    part.content
                    if isinstance(part.content, str)
                    else json.dumps(part.content, default=str, sort_keys=True)
                )
                if detail not in semantic_messages:
                    retries.append(
                        AgentValidationRetry(
                            source="tool" if part.tool_name in tool_names else "schema",
                            requestIndex=request_index,
                            message=detail,
                            response=pending.get(part.tool_call_id),
                        )
                    )
    error_detail = describe_agent_error(error) if error is not None else None
    if (
        error is not None
        and isinstance(error.__cause__, ValidationError)
        and messages
        and isinstance(messages[-1], ModelResponse)
    ):
        # The SDK does not append feedback after the last schema retry is exhausted.
        calls_in_response = [
            part for part in messages[-1].parts if isinstance(part, ToolCallPart)
        ]
        if len(calls_in_response) == 1:
            rejected = calls_in_response[0]
            retries.append(
                AgentValidationRetry(
                    source="tool" if rejected.tool_name in tool_names else "schema",
                    requestIndex=request_index,
                    message=str(error.__cause__),
                    response=rejected.args,
                )
            )
    return AgentRunInfo(
        agentName=name or "unnamed",
        modelName=_model_name(model),
        runId=next(
            (
                str(message.run_id)
                for message in reversed(messages)
                if getattr(message, "run_id", None)
            ),
            uuid.uuid4().hex,
        ),
        durationSeconds=time.monotonic() - started,
        usage=reported_usage,
        toolCalls=calls,
        status="failed" if error is not None else "done",
        validationRetries=sorted(retries, key=lambda retry: retry.requestIndex),
        providerFailures=provider_audit.failures if provider_audit is not None else [],
        errorType=type(error).__name__ if error is not None else None,
        error=error_detail,
    )


async def _execute_agent(
    *,
    model: Any,
    output_type: Any,
    system_prompt: str,
    user_prompt: AgentUserPrompt,
    tools: Sequence[Callable[..., Any] | Any],
    deps_type: type[Any] | None,
    deps: Any,
    config: AgentRunConfig | None,
    name: str | None,
    output_validator: Callable[[Any], Any] | None,
    message_history: Sequence[Any],
    on_attempt: Callable[[AgentRunInfo], None] | None,
    normalize_sync_function_model: bool,
    cancel_requested: Event | None = None,
) -> AgentExecutionResult:
    from pydantic_ai import capture_run_messages
    from pydantic_ai.usage import RunUsage

    run_config = config or AgentRunConfig()
    agent_name = name or "unnamed"
    usage_limits = get_usage_limits(run_config)
    usage = RunUsage()
    failures: list[AgentValidationRetry] = []
    provider_audit = _ProviderRequestAudit()
    started = time.monotonic()
    logger.debug(
        f"Starting agent {agent_name}: model={_model_name(model)}, "
        f"tools={len(tools)}, request_limit={run_config.requestLimit}, "
        f"tool_call_limit={run_config.toolCallLimit}, retries={run_config.retries}, "
        f"per_response_output_limit={run_config.outputTokenLimit}, "
        f"run_output_limit={usage_limits.output_tokens_limit}"
    )
    with capture_run_messages() as messages:
        try:
            if cancel_requested is not None and cancel_requested.is_set():
                raise asyncio.CancelledError(
                    "Analysis interrupted before model execution"
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
                normalize_sync_function_model=normalize_sync_function_model,
                validation_failures=failures,
                provider_audit=provider_audit,
            )
            async with agent:
                try:
                    result = await agent.run(
                        user_prompt,
                        deps=deps,
                        message_history=message_history,
                        usage_limits=usage_limits,
                        usage=usage,
                    )
                except Exception as exc:
                    if not isinstance(user_prompt, str) and _image_input_is_unsupported(
                        exc
                    ):
                        raise ImageInputUnsupportedError(
                            "The configured model does not accept image input"
                        ) from exc
                    raise
        except (Exception, asyncio.CancelledError) as exc:
            info = _run_info(
                messages=messages[len(message_history) :],
                usage=usage,
                model=model,
                name=name,
                started=started,
                tools=tools,
                validation_failures=failures,
                error=exc,
                provider_audit=provider_audit,
            )
            setattr(exc, "agent_run_info", info)
            logger.error(
                f"Agent {agent_name} failed after {info.durationSeconds:.2f}s: "
                f"{info.error}"
            )
            if on_attempt is not None:
                try:
                    on_attempt(info)
                except Exception as callback_error:
                    exc.add_note(
                        f"Saving agent execution evidence also failed: {type(callback_error).__name__}: {callback_error}"
                    )
            raise
        info = _run_info(
            messages=result.new_messages(),
            usage=result.usage,
            model=model,
            name=name,
            started=started,
            tools=tools,
            validation_failures=failures,
            provider_audit=provider_audit,
        )
        if on_attempt is not None:
            try:
                on_attempt(info)
            except Exception as exc:
                setattr(exc, "agent_run_info", info)
                exc.add_note(
                    "The model completed, but saving its execution evidence failed."
                )
                raise
        logger.debug(
            f"Agent {agent_name} completed in {info.durationSeconds:.2f}s: "
            f"requests={info.usage.requests}, tool_calls={info.usage.toolCalls}, "
            f"input_tokens={info.usage.inputTokens}, output_tokens={info.usage.outputTokens}, "
            f"usage={info.usage.availability}"
        )
        return AgentExecutionResult(output=result.output, runInfo=info)


def run_agent_sync(
    *,
    model: Any,
    output_type: Any,
    system_prompt: str,
    user_prompt: AgentUserPrompt,
    tools: Sequence[Callable[..., Any] | Any] = (),
    deps_type: type[Any] | None = None,
    deps: Any = None,
    config: AgentRunConfig | None = None,
    name: str | None = None,
    output_validator: Callable[[Any], Any] | None = None,
    message_history: Sequence[Any] = (),
    on_attempt: Callable[[AgentRunInfo], None] | None = None,
) -> AgentExecutionResult:
    """Run one synchronous agent loop and return its bounded audit record.

    Jupyter and other hosts already have a running event loop. Pydantic AI's
    asynchronous runner cannot drive that loop synchronously, so this hops to
    a worker thread in that case. Entering the agent context ensures provider
    HTTP clients are closed on that worker's event loop before it exits.

    ``on_attempt`` receives invocation evidence once, including failure usage and
    rejected responses. It runs on the agent's event-loop thread. A failed run
    re-raises its original exception with this evidence in ``agent_run_info``.
    """

    cancel_requested = Event()
    worker_lock = Lock()
    worker: tuple[asyncio.AbstractEventLoop, asyncio.Task[Any]] | None = None

    async def execute(*, notebook_worker: bool = False) -> AgentExecutionResult:
        nonlocal worker
        if notebook_worker:
            task = asyncio.current_task()
            assert task is not None
            with worker_lock:
                worker = (asyncio.get_running_loop(), task)
        return await _execute_agent(
            model=model,
            output_type=output_type,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=tools,
            deps_type=deps_type,
            deps=deps,
            config=config,
            name=name,
            output_validator=output_validator,
            normalize_sync_function_model=True,
            message_history=message_history,
            on_attempt=on_attempt,
            cancel_requested=cancel_requested if notebook_worker else None,
        )

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(execute())

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, execute(notebook_worker=True))
        try:
            return future.result()
        except KeyboardInterrupt as interrupted:
            cancel_requested.set()
            with worker_lock:
                running = worker
            if running is not None:
                loop, task = running
                try:
                    loop.call_soon_threadsafe(task.cancel, "Analysis interrupted")
                except RuntimeError as cancellation_error:
                    # The worker may have finished and closed its loop already.
                    interrupted.add_note(
                        "Cancellation raced with worker shutdown: "
                        + describe_agent_error(cancellation_error)
                    )
            try:
                completed = future.result()
            except BaseException as worker_error:
                info = getattr(worker_error, "agent_run_info", None)
                if info is not None:
                    setattr(interrupted, "agent_run_info", info)
                interrupted.add_note(
                    "Worker shutdown reported: " + describe_agent_error(worker_error)
                )
            else:
                setattr(interrupted, "agent_run_info", completed.runInfo)
                interrupted.add_note(
                    "The model invocation completed before cancellation was delivered."
                )
            raise


async def run_agent_async(
    *,
    model: Any,
    output_type: Any,
    system_prompt: str,
    user_prompt: AgentUserPrompt,
    tools: Sequence[Callable[..., Any] | Any] = (),
    deps_type: type[Any] | None = None,
    deps: Any = None,
    config: AgentRunConfig | None = None,
    name: str | None = None,
    output_validator: Callable[[Any], Any] | None = None,
    message_history: Sequence[Any] = (),
    on_attempt: Callable[[AgentRunInfo], None] | None = None,
) -> AgentExecutionResult:
    """Run one asynchronous agent loop and return its bounded audit record."""
    return await _execute_agent(
        model=model,
        output_type=output_type,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        tools=tools,
        deps_type=deps_type,
        deps=deps,
        config=config,
        name=name,
        output_validator=output_validator,
        normalize_sync_function_model=False,
        message_history=message_history,
        on_attempt=on_attempt,
    )


run_agent = run_agent_async
