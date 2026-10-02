"""Audited structured decisions with explicit request and repair limits.

The output tool only describes a Pydantic value. It never executes scientific
code. SDK-internal retries are outside the observed-request accounting boundary.
"""

import asyncio
import hashlib
import importlib.metadata
import json
import time
import uuid
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Any, TypeVar, cast

import httpx
from pydantic import BaseModel, ValidationError
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestParameters, infer_model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition

from .models import RuntimeConfig
from .prompts import BASE_INSTRUCTIONS
from .records import RecordError

if TYPE_CHECKING:
    from .records import RunRecords

DecisionT = TypeVar("DecisionT", bound=BaseModel)


class ProviderError(RuntimeError):
    """A model decision could not complete within the recorded contract."""


class DecisionValidationError(ProviderError):
    """The bounded semantic repair did not produce an acceptable decision."""


class RequestBudgetExceeded(ProviderError):
    """No further observed model requests are admitted."""


class PromptTooLarge(ProviderError):
    """A serialized request exceeds its byte limit before provider execution."""


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def model_identity(model: Model | str) -> dict[str, str]:
    """Record model labels without serializing clients, endpoints or credentials."""
    if isinstance(model, str):
        return {"model": model}
    return {"model": model.model_name, "provider": model.system}


def visible_response(response: ModelResponse) -> dict[str, Any]:
    """Allowlist public content, omitting reasoning and opaque provider fields."""
    parts: list[dict[str, Any]] = []
    for part in response.parts:
        if isinstance(part, TextPart):
            parts.append({"kind": "text", "text": part.content})
        elif isinstance(part, ToolCallPart):
            parts.append(
                {"kind": "tool", "name": part.tool_name, "arguments": part.args}
            )
    usage = response.usage
    measured_usage = None
    if usage.has_values():
        measured_usage = {
            "inputTokens": usage.input_tokens,
            "outputTokens": usage.output_tokens,
            "cacheReadTokens": usage.cache_read_tokens,
            "cacheWriteTokens": usage.cache_write_tokens,
        }
    return {
        "parts": parts,
        "usage": measured_usage,
        "finishReason": response.finish_reason,
    }


def _parse(visible: dict[str, Any], output_type: type[DecisionT]) -> DecisionT:
    tools = [part for part in visible["parts"] if part["kind"] == "tool"]
    if len(tools) != 1 or tools[0]["name"] != "decision":
        raise ValueError(
            "Return exactly one decision output tool call; no other tools are available"
        )
    arguments = tools[0]["arguments"]
    if isinstance(arguments, str):
        return output_type.model_validate_json(arguments)
    return output_type.model_validate(arguments)


def _errors(exc: ValueError) -> list[str]:
    if isinstance(exc, ValidationError):
        return [
            f"{'.'.join(str(part) for part in item['loc']) or 'decision'}: {item['msg']}"
            for item in exc.errors(
                include_url=False, include_context=False, include_input=False
            )
        ]
    return [str(exc)]


def _transient(exc: Exception) -> bool:
    if isinstance(exc, ModelHTTPError):
        if exc.status_code == 429:
            body = exc.body
            if not isinstance(body, dict):
                return False
            error = body.get("error", body)
            if not isinstance(error, dict):
                return False
            codes = {
                str(error.get("code", "")).lower(),
                str(error.get("type", "")).lower(),
            }
            if codes & {
                "insufficient_quota",
                "billing_hard_limit_reached",
                "quota_exceeded",
            }:
                return False
            return bool(
                codes & {"rate_limit_exceeded", "rate_limit_error", "too_many_requests"}
            )
        return exc.status_code in {408, 425, 500, 502, 503, 504}
    if isinstance(
        exc, httpx.TimeoutException | httpx.NetworkError | httpx.RemoteProtocolError
    ):
        return True
    # The optional OpenAI provider can expose these before wrapping HTTP errors.
    try:
        from openai import APIConnectionError
    except ImportError:
        return False
    return isinstance(exc, APIConnectionError)


def _error_record(exc: BaseException) -> dict[str, Any]:
    # Raw transport messages may contain URLs, request headers or response bodies.
    result: dict[str, Any] = {"errorType": type(exc).__name__}
    if isinstance(exc, ModelHTTPError):
        result["httpStatus"] = exc.status_code
    return result


def _versions() -> dict[str, str]:
    result = {}
    for package in ("scarf", "pydantic-ai-slim", "pydantic"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "unknown"
    return result


def _disabled_reasoning_body() -> dict[str, Any]:
    return {
        "thinking": {"type": "disabled"},
        "reasoning_effort": "none",
        "chat_template_kwargs": {"thinking": False},
        "reasoning": {"enabled": False},
    }


def _settings(model: Model, runtime: RuntimeConfig) -> ModelSettings:
    overrides = dict(runtime.modelSettings)
    existing = model.settings or {}
    body: dict[str, Any] = {}
    for supplied in (existing, overrides):
        supplied_body = supplied.get("extra_body")
        if supplied_body is not None:
            if not isinstance(supplied_body, Mapping):
                raise ProviderError("extra_body must be a mapping for agent requests")
            body.update(deepcopy(supplied_body))
    body.update(_disabled_reasoning_body())
    overrides.update(
        extra_body=body,
        thinking=False,
        openai_reasoning_effort="none",
        # OpenRouter otherwise replaces extra_body.reasoning from its unified
        # thinking translation, even when thinking is disabled.
        openrouter_reasoning={"enabled": False},
    )
    # Explicit native controls take precedence over unified thinking in the
    # installed SDK. Neutralize supplied conflicts without changing callers.
    native_disabled: dict[str, Any] = {
        "anthropic_thinking": {"type": "disabled"},
        "google_thinking_config": {"thinking_budget": 0},
        "groq_reasoning_effort": "none",
        "groq_reasoning_format": "hidden",
        "xai_reasoning_effort": "none",
        "snowflake_reasoning": {"enabled": False},
    }
    for key, value in native_disabled.items():
        if key in existing or key in overrides:
            overrides[key] = value
    additional = overrides.get(
        "bedrock_additional_model_requests_fields",
        existing.get("bedrock_additional_model_requests_fields"),
    )
    if isinstance(additional, Mapping):
        overrides["bedrock_additional_model_requests_fields"] = {
            key: deepcopy(value)
            for key, value in additional.items()
            if key not in {"thinking", "reasoning_effort", "reasoning_config"}
        }
    # Preserve a caller's smaller configured output budget, and all their model
    # settings, unless the runtime limit was explicitly supplied.
    if "maxOutputTokens" in runtime.model_fields_set or not existing.get("max_tokens"):
        overrides["max_tokens"] = runtime.maxOutputTokens
    elif existing["max_tokens"] > runtime.maxOutputTokens:
        overrides["max_tokens"] = runtime.maxOutputTokens
    return cast(ModelSettings, overrides)


def _safe_settings(model: Model, overrides: ModelSettings) -> dict[str, Any]:
    # Provider-specific settings can contain clients or custom headers. Keep a
    # narrow reproducibility allowlist, never a repr of arbitrary configuration.
    allowed = {
        "temperature",
        "top_p",
        "max_tokens",
        "seed",
        "frequency_penalty",
        "presence_penalty",
        "parallel_tool_calls",
        "openai_reasoning_effort",
        "openai_text_verbosity",
        "thinking",
    }
    merged = {**(model.settings or {}), **overrides}
    safe: dict[str, Any] = {
        key: value
        for key, value in merged.items()
        if key in allowed and isinstance(value, str | int | float | bool | type(None))
    }
    # Only these agent-controlled fields are public provenance. Arbitrary
    # caller body fields may contain provider-private values or credentials.
    safe["extra_body"] = _disabled_reasoning_body()
    return safe


def _matches(event: dict[str, Any], decision_id: str, identity: str) -> bool:
    return (
        event.get("decisionId") == decision_id
        and event.get("identityDigest") == identity
    )


def _read_response(
    records: "RunRecords", request: dict[str, Any]
) -> dict[str, Any] | None:
    try:
        return cast(dict[str, Any], records.read_json(request["responsePath"]))
    except FileNotFoundError:
        return None
    except RecordError as exc:
        if isinstance(exc.__cause__, FileNotFoundError):
            return None
        raise


def _validate_response(
    visible: dict[str, Any],
    output_type: type[DecisionT],
    validate: Callable[[DecisionT], None],
) -> tuple[DecisionT | None, list[str]]:
    try:
        output = _parse(visible, output_type)
        validate(output)
    except ValueError as exc:
        errors = _errors(exc)
        if visible.get("finishReason") == "length":
            errors.insert(
                0,
                "The provider stopped at its output-token limit (finishReason=length) without a valid decision. "
                "Return one complete, concise decision tool call satisfying the supplied schema and validation rules; "
                "shorten the rationale and omit extra prose.",
            )
        return None, errors
    return output, []


async def decide(
    model: Model | str,
    *,
    decision_id: str,
    stage: str,
    evidence: dict[str, Any],
    output_type: type[DecisionT],
    validate: Callable[[DecisionT], None],
    records: "RunRecords",
    runtime: RuntimeConfig,
    instructions: str,
) -> DecisionT:
    """Produce or recover one decision; persist requests before contacting a model.

    A process crash after a visible response was written can be resumed without
    another provider call. A request with no persisted response still consumes an
    observed slot because the provider may have received it.
    """
    prompt = BASE_INSTRUCTIONS + "\n" + instructions
    schema = output_type.model_json_schema()
    identity = _digest(
        {"stage": stage, "evidence": evidence, "prompt": prompt, "schema": schema}
    )
    provenance = {
        "decisionId": decision_id,
        "stage": stage,
        "identityDigest": identity,
        "evidenceDigest": _digest(evidence),
        "promptDigest": _digest(prompt),
        "schemaDigest": _digest(schema),
    }
    events = records.events()
    invocation_id = next(
        (
            event["invocationId"]
            for event in reversed(events)
            if event["kind"] == "invocationStarted"
        ),
        "standalone",
    )
    provenance["invocationId"] = invocation_id
    related = [event for event in events if event.get("decisionId") == decision_id]
    if any(event.get("identityDigest") != identity for event in related):
        raise ProviderError(
            "Saved decision evidence, instructions or schema changed; start a new run"
        )
    for event in related:
        if event["kind"] == "decisionAccepted":
            accepted = output_type.model_validate(event["output"])
            validate(accepted)
            return accepted

    feedback: list[str] = []
    previous_bad: set[str] = set()
    rejected = 0
    transport_failures = 0
    requests = [event for event in related if event["kind"] == "modelRequest"]
    for request in requests:
        call_id = request["callId"]
        current_invocation = request.get("invocationId", "standalone") == invocation_id
        visible = _read_response(records, request)
        if visible is not None:
            output, errors = _validate_response(visible, output_type, validate)
            if output is not None:
                records.append(
                    "decisionAccepted",
                    **provenance,
                    callId=call_id,
                    output=output.model_dump(mode="json"),
                    recovered=True,
                )
                return output
            fingerprint = _digest(visible["parts"])
            if not any(
                event["kind"] == "decisionRejected" and event.get("callId") == call_id
                for event in related
            ):
                records.append(
                    "decisionRejected",
                    **provenance,
                    callId=call_id,
                    errors=errors,
                    responseDigest=fingerprint,
                )
            if not current_invocation:
                continue
            feedback = errors
            rejected += 1
            if fingerprint in previous_bad:
                raise DecisionValidationError(
                    "The model repeated an identical invalid response"
                )
            previous_bad.add(fingerprint)
        else:
            failures = [
                event
                for event in related
                if event["kind"] == "modelFailure" and event.get("callId") == call_id
            ]
            if not current_invocation:
                continue
            if failures and not failures[-1]["transient"]:
                raise ProviderError("Saved provider failure requires explicit resume")
            transport_failures += 1
            if not failures:
                records.append(
                    "modelFailure",
                    **provenance,
                    callId=call_id,
                    transient=True,
                    errorType="InterruptedRequest",
                )

    resolved = infer_model(model)
    settings = _settings(resolved, runtime)
    parameters = ModelRequestParameters(
        output_mode="tool",
        output_tools=[
            ToolDefinition(
                name="decision", parameters_json_schema=schema, kind="output"
            )
        ],
        allow_text_output=False,
    )
    deadline = time.monotonic() + runtime.decisionTimeout
    try:
        async with asyncio.timeout(runtime.decisionTimeout):
            while True:
                if rejected > 1:
                    raise DecisionValidationError(
                        "The semantic repair was rejected: " + "; ".join(feedback)
                    )
                if transport_failures > 1:
                    raise ProviderError(
                        "The single transient transport retry was exhausted"
                    )
                events = records.events()
                count = sum(event["kind"] == "modelRequest" for event in events)
                local_count = sum(
                    event["kind"] == "modelRequest"
                    and _matches(event, decision_id, identity)
                    and event.get("invocationId", "standalone") == invocation_id
                    for event in events
                )
                if (
                    count >= runtime.maxRequests
                    or local_count >= runtime.maxRequestsPerDecision
                ):
                    raise RequestBudgetExceeded(
                        "Observed model request limit exhausted"
                    )

                payload = {"stage": stage, "evidence": evidence}
                if feedback:
                    payload["validationErrors"] = feedback
                user_prompt = _json(payload)
                request_record = {
                    **provenance,
                    "model": model_identity(resolved),
                    "instructions": prompt,
                    "userPrompt": user_prompt,
                    "schema": schema,
                    "settings": _safe_settings(resolved, settings),
                    "softwareVersions": _versions(),
                }
                if len(_json(request_record).encode()) > runtime.maxPromptBytes:
                    raise PromptTooLarge(
                        "Serialized prompt, schema and settings exceed maxPromptBytes"
                    )
                call_id = f"{count + 1:06d}-{uuid.uuid4().hex[:12]}"
                request_path = f"calls/{call_id}.request.json"
                response_path = f"calls/{call_id}.response.json"
                records.write_json(request_path, request_record)
                records.append(
                    "modelRequest",
                    **provenance,
                    callId=call_id,
                    requestPath=request_path,
                    responsePath=response_path,
                )
                started = time.monotonic()
                try:
                    response = await resolved.request(
                        [
                            ModelRequest(
                                parts=[
                                    SystemPromptPart(prompt),
                                    UserPromptPart(user_prompt),
                                ]
                            )
                        ],
                        settings,
                        parameters,
                    )
                except asyncio.CancelledError as exc:
                    records.append(
                        "modelFailure",
                        **provenance,
                        callId=call_id,
                        transient=True,
                        **_error_record(exc),
                    )
                    raise
                except Exception as exc:
                    transient = _transient(exc)
                    records.append(
                        "modelFailure",
                        **provenance,
                        callId=call_id,
                        transient=transient,
                        **_error_record(exc),
                    )
                    if not transient:
                        raise ProviderError(
                            f"Provider request failed ({type(exc).__name__}); inspect recorded failure type"
                        ) from None
                    transport_failures += 1
                    if (
                        transport_failures <= 1
                        and isinstance(exc, ModelHTTPError)
                        and (delay := exc.retry_after) is not None
                    ):
                        if delay >= deadline - time.monotonic():
                            raise ProviderError(
                                "Retry-After exceeds the remaining decision deadline"
                            ) from None
                        await asyncio.sleep(delay)
                    continue
                visible = visible_response(response)
                # Persist before parsing or validation, allowing recovery even if
                # a validator crashes or a subsequent journal write fails.
                records.write_json(response_path, visible)
                records.append(
                    "modelResponse",
                    **provenance,
                    callId=call_id,
                    responsePath=response_path,
                    elapsedSeconds=time.monotonic() - started,
                    usage=visible["usage"],
                )
                output, feedback = _validate_response(visible, output_type, validate)
                if output is not None:
                    records.append(
                        "decisionAccepted",
                        **provenance,
                        callId=call_id,
                        output=output.model_dump(mode="json"),
                        recovered=False,
                    )
                    return output
                fingerprint = _digest(visible["parts"])
                records.append(
                    "decisionRejected",
                    **provenance,
                    callId=call_id,
                    errors=feedback,
                    responseDigest=fingerprint,
                )
                rejected += 1
                if fingerprint in previous_bad:
                    raise DecisionValidationError(
                        "The model repeated an identical invalid response"
                    )
                previous_bad.add(fingerprint)
    except TimeoutError:
        raise ProviderError("Decision deadline exceeded") from None


def replay_decisions(
    records: "RunRecords",
    *,
    schemas: dict[str, type[BaseModel]],
    validators: dict[str, Callable[[BaseModel, dict[str, Any]], None]],
) -> list[dict[str, Any]]:
    """Revalidate accepted outputs against their exact saved evidence offline.

    Both registries are keyed by stage. A missing registry entry is an error,
    rather than silently declaring an unvalidated stage successfully replayed.
    """
    events = records.events()
    requests = {
        event["callId"]: event for event in events if event["kind"] == "modelRequest"
    }
    results = []
    for event in events:
        if event["kind"] != "decisionAccepted":
            continue
        request = records.read_json(requests[event["callId"]]["requestPath"])
        stage = event["stage"]
        schema = schemas[stage]
        if _digest(schema.model_json_schema()) != event["schemaDigest"]:
            raise ProviderError("Replay schema differs from the frozen decision schema")
        payload = json.loads(request["userPrompt"])
        evidence = payload["evidence"]
        if (
            _digest(evidence) != event["evidenceDigest"]
            or _digest(request["instructions"]) != event["promptDigest"]
        ):
            raise ProviderError("Replay evidence or prompt digest does not match")
        output = schema.model_validate(event["output"])
        validators[stage](output, evidence)
        results.append(
            {"decisionId": event["decisionId"], "stage": stage, "valid": True}
        )
    return results
