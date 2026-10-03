"""Offline acceptance tests for audited, bounded structured decisions."""

import asyncio
import inspect
import importlib.metadata
import json
from copy import deepcopy
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, ConfigDict
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelResponse, TextPart, ThinkingPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from scarf.agent.models import RuntimeConfig
from scarf.agent.provider import (
    DecisionValidationError,
    PromptTooLarge,
    ProviderError,
    RequestBudgetExceeded,
    decide,
    model_identity,
    replay_decisions,
    visible_response,
)
from scarf.agent.records import RunRecords


DISABLED_REASONING_BODY = {
    "thinking": {"type": "disabled"},
    "reasoning_effort": "none",
    "chat_template_kwargs": {"thinking": False},
    "reasoning": {"enabled": False},
}


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    choice: str
    evidenceIds: list[str]


def scripted_model(function: Any, **kwargs: Any) -> Any:
    async def execute(messages: Any, info: Any) -> Any:
        value = function(messages, info)
        return await value if inspect.isawaitable(value) else value

    return FunctionModel(execute, **kwargs)


@pytest.fixture
def records(tmp_path: Any) -> Any:
    return RunRecords.create(tmp_path / "analysis", {"procedure": "test"})


def validate(output: Any) -> Any:
    errors = []
    if output.choice not in {"baseline", "stop"}:
        errors.append("choice is not a registered option")
    if any(value != "evidence-1" for value in output.evidenceIds):
        errors.append("unknown evidence identifier")
    if errors:
        raise ValueError("; ".join(errors))


def response(choice: Any = "baseline", evidence_ids: Any = None) -> Any:
    return ModelResponse(
        parts=[
            ToolCallPart(
                "decision",
                {"choice": choice, "evidenceIds": evidence_ids or ["evidence-1"]},
            )
        ]
    )


def run_decision(model: Any, records: Any, **kwargs: Any) -> Any:
    return asyncio.run(
        decide(
            model,
            decision_id="choose-1",
            stage="select",
            evidence=kwargs.pop("evidence", {"evidence-1": {"score": 0.5}}),
            output_type=Decision,
            validate=kwargs.pop("validate", validate),
            records=records,
            runtime=kwargs.pop("runtime", RuntimeConfig()),
            instructions="Choose a measured option.",
            **kwargs,
        )
    )


def test_request_saved_before_provider_and_accepted_replay(records: Any) -> None:
    observed = []

    def model(messages: Any, info: Any) -> Any:
        assert records.events()[-1]["kind"] == "modelRequest"
        event = records.events()[-1]
        saved = records.read_json(event["requestPath"])
        assert json.loads(saved["userPrompt"])["evidence"]["evidence-1"]["score"] == 0.5
        assert info.function_tools == []
        assert [tool.name for tool in info.output_tools] == ["decision"]
        assert not info.allow_text_output
        observed.append(info)
        return response()

    client = scripted_model(model, settings={"temperature": 0.2, "max_tokens": 1024})
    assert run_decision(client, records).choice == "baseline"
    assert run_decision(client, records).choice == "baseline"
    assert len(observed) == 1
    assert observed[0].model_settings["temperature"] == 0.2
    assert observed[0].model_settings["max_tokens"] == 1024
    assert [event["kind"] for event in records.events()] == [
        "modelRequest",
        "modelResponse",
        "decisionAccepted",
    ]


def test_repairs_combined_errors_once(records: Any) -> None:
    prompts = []

    def model(messages: Any, info: Any) -> Any:
        prompts.append(messages[0].parts[-1].content)
        return response("invented", ["fiction"]) if len(prompts) == 1 else response()

    assert run_decision(scripted_model(model), records).choice == "baseline"
    assert "not a registered option" in prompts[1]
    assert "unknown evidence identifier" in prompts[1]
    assert sum(event["kind"] == "modelRequest" for event in records.events()) == 2


def test_structural_errors_are_aggregated(records: Any) -> None:
    prompts = []

    def model(messages: Any, info: Any) -> Any:
        prompts.append(messages[0].parts[-1].content)
        return (
            ModelResponse(parts=[ToolCallPart("decision", {})])
            if len(prompts) == 1
            else response()
        )

    run_decision(scripted_model(model), records)
    errors = json.loads(prompts[-1])["validationErrors"]
    assert len(errors) == 2
    assert any("choice" in error for error in errors)
    assert any("evidenceIds" in error for error in errors)


@pytest.mark.parametrize(
    "parts",
    [[], [ToolCallPart("decision", '{"choice":"base')]],
)
def test_truncated_response_gets_actionable_feedback_with_unchanged_limits(
    records: Any, parts: list[Any]
) -> None:
    prompts = []
    settings = []

    def model(messages: Any, info: Any) -> Any:
        prompts.append(json.loads(messages[0].parts[-1].content))
        settings.append(dict(info.model_settings))
        if len(prompts) == 1:
            return ModelResponse(
                parts=parts,
                finish_reason="length",
                usage=RequestUsage(output_tokens=4096),
            )
        return response()

    assert run_decision(scripted_model(model), records).choice == "baseline"
    feedback = prompts[1]["validationErrors"]
    assert "output-token limit" in feedback[0]
    assert "finishReason=length" in feedback[0]
    assert "complete, concise decision tool call" in feedback[0]
    assert len(feedback) >= 2  # The original structural failure is still available.
    assert len(settings) == 2
    assert settings[0] == settings[1]
    assert settings[0]["max_tokens"] == 4096
    rejected = records.latest("decisionRejected")
    assert rejected["errors"] == feedback
    saved_request = next(
        event for event in records.events() if event["kind"] == "modelRequest"
    )
    visible = records.read_json(saved_request["responsePath"])
    assert visible["finishReason"] == "length"
    assert visible["usage"]["outputTokens"] == 4096


def test_repeated_truncation_remains_a_bounded_operational_failure(
    records: Any,
) -> None:
    def model(messages: Any, info: Any) -> Any:
        return ModelResponse(
            parts=[], finish_reason="length", usage=RequestUsage(output_tokens=4096)
        )

    with pytest.raises(DecisionValidationError, match="identical invalid"):
        run_decision(scripted_model(model), records)
    assert sum(event["kind"] == "modelRequest" for event in records.events()) == 2
    assert not any(event["kind"] == "decisionAccepted" for event in records.events())
    assert all(
        "output-token limit" in event["errors"][0]
        for event in records.events()
        if event["kind"] == "decisionRejected"
    )


def test_valid_decision_is_not_discarded_only_for_length_finish_reason(
    records: Any,
) -> None:
    def model(messages: Any, info: Any) -> Any:
        valid = response()
        valid.finish_reason = "length"
        return valid

    assert run_decision(scripted_model(model), records).choice == "baseline"
    assert sum(event["kind"] == "modelRequest" for event in records.events()) == 1
    assert not any(event["kind"] == "decisionRejected" for event in records.events())


def test_stops_repeated_invalid_response(records: Any) -> None:
    with pytest.raises(DecisionValidationError, match="identical invalid"):
        run_decision(
            scripted_model(lambda messages, info: response("invented")), records
        )
    assert sum(event["kind"] == "modelRequest" for event in records.events()) == 2


def test_transport_then_semantic_repair_uses_three_requests(records: Any) -> None:
    attempts = 0

    def model(messages: Any, info: Any) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError(
                "do not persist https://private.invalid/?key=credential"
            )
        return response("invented") if attempts == 2 else response()

    assert run_decision(scripted_model(model), records).choice == "baseline"
    assert attempts == 3
    assert "private.invalid" not in json.dumps(records.events())
    assert "credential" not in json.dumps(records.events())


@pytest.mark.parametrize("status", [400, 401, 403, 429])
def test_auth_and_quota_are_not_retried(records: Any, status: Any) -> None:
    attempts = 0

    def model(messages: Any, info: Any) -> Any:
        nonlocal attempts
        attempts += 1
        raise ModelHTTPError(status, "fixture", body={"secret": "private"})

    with pytest.raises(ProviderError, match="Provider request failed"):
        run_decision(scripted_model(model), records)
    assert attempts == 1
    assert "private" not in json.dumps(records.events())


def test_typed_temporary_rate_limit_retries(records: Any) -> None:
    attempts = 0

    def model(messages: Any, info: Any) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ModelHTTPError(
                429,
                "fixture",
                body={"error": {"code": "rate_limit_exceeded"}},
                headers={"retry-after": "0"},
            )
        return response()

    assert run_decision(scripted_model(model), records).choice == "baseline"
    assert attempts == 2


def test_hard_quota_overrides_rate_limit_type(records: Any) -> None:
    def model(messages: Any, info: Any) -> Any:
        raise ModelHTTPError(
            429,
            "fixture",
            body={"error": {"code": "insufficient_quota", "type": "rate_limit_error"}},
        )

    with pytest.raises(ProviderError, match="Provider request failed"):
        run_decision(scripted_model(model), records)
    assert sum(event["kind"] == "modelRequest" for event in records.events()) == 1


def test_retry_after_cannot_exceed_remaining_deadline(records: Any) -> None:
    def model(messages: Any, info: Any) -> Any:
        raise ModelHTTPError(503, "fixture", headers={"retry-after": "60"})

    with pytest.raises(ProviderError, match="remaining decision deadline"):
        run_decision(
            scripted_model(model), records, runtime=RuntimeConfig(decisionTimeout=1)
        )
    assert sum(event["kind"] == "modelRequest" for event in records.events()) == 1


def test_request_limit_counts_failures(records: Any) -> None:
    attempts = 0

    def model(messages: Any, info: Any) -> Any:
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectError("unavailable")

    with pytest.raises(RequestBudgetExceeded):
        run_decision(
            scripted_model(model), records, runtime=RuntimeConfig(maxRequests=1)
        )
    assert attempts == 1


def test_explicit_resume_resets_local_but_not_global_budget(records: Any) -> None:
    records.append("invocationStarted", invocationId="first")
    with pytest.raises(DecisionValidationError, match="identical invalid response"):
        run_decision(
            scripted_model(lambda messages, info: response("invented")), records
        )
    records.append("invocationStarted", invocationId="second")
    assert (
        run_decision(
            scripted_model(lambda messages, info: response()),
            records,
            runtime=RuntimeConfig(maxRequests=3),
        ).choice
        == "baseline"
    )
    assert sum(event["kind"] == "modelRequest" for event in records.events()) == 3


def test_preflight_prompt_limit_never_calls_model(records: Any) -> None:
    def model(messages: Any, info: Any) -> Any:
        pytest.fail("An oversized request reached the model")

    with pytest.raises(PromptTooLarge):
        run_decision(
            scripted_model(model),
            records,
            evidence={"large": "a" * 10000},
            runtime=RuntimeConfig(maxPromptBytes=1024),
        )
    assert records.events() == []


def test_persisted_visible_response_recovers_after_validation_crash(
    records: Any,
) -> None:
    def crash(output: Any) -> Any:
        raise RuntimeError("simulated interruption in validator")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_decision(
            scripted_model(lambda messages, info: response()), records, validate=crash
        )

    def no_provider(messages: Any, info: Any) -> Any:
        pytest.fail("Saved visible output should be validated without a new call")

    assert run_decision(scripted_model(no_provider), records).choice == "baseline"
    assert records.events()[-1]["recovered"] is True
    assert sum(event["kind"] == "modelRequest" for event in records.events()) == 1


def test_audit_failure_prevents_model_call(records: Any, monkeypatch: Any) -> None:
    def fail_event(kind: Any, **payload: Any) -> Any:
        raise OSError("cannot persist request")

    monkeypatch.setattr(records, "append", fail_event)
    with pytest.raises(OSError, match="cannot persist"):
        run_decision(
            scripted_model(lambda messages, info: pytest.fail("unaudited request")),
            records,
        )


def test_changed_evidence_refuses_replay(records: Any) -> None:
    client = scripted_model(lambda messages, info: response())
    run_decision(client, records)
    with pytest.raises(ProviderError, match="changed"):
        run_decision(client, records, evidence={"evidence-1": {"score": 0.7}})


def test_visible_only_sanitizer_and_unknown_usage() -> None:
    saved = visible_response(
        ModelResponse(
            parts=[
                ThinkingPart("hidden chain", signature="opaque signature"),
                TextPart("visible", provider_details={"secret": "private"}),
                ToolCallPart(
                    "decision",
                    {"choice": "baseline"},
                    provider_details={"token": "private"},
                ),
            ],
            provider_url="https://private.invalid",
            provider_details={"token": "private"},
            metadata={"client": "private"},
        )
    )
    assert saved["usage"] is None
    assert len(saved["parts"]) == 2
    serialized = json.dumps(saved)
    assert "private" not in serialized
    assert "hidden" not in serialized
    assert "signature" not in serialized


def test_deadline_persists_cancellation(records: Any) -> None:
    async def model(messages: Any, info: Any) -> Any:
        await asyncio.sleep(10)
        return response()

    with pytest.raises(ProviderError, match="deadline"):
        run_decision(
            scripted_model(model), records, runtime=RuntimeConfig(decisionTimeout=0.02)
        )
    assert records.events()[-1]["kind"] == "modelFailure"
    assert records.events()[-1]["errorType"] == "CancelledError"


def test_external_cancellation_propagates(records: Any) -> None:
    async def model(messages: Any, info: Any) -> Any:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        run_decision(scripted_model(model), records)
    assert records.events()[-1]["errorType"] == "CancelledError"


def test_offline_replay_uses_saved_evidence(records: Any) -> None:
    run_decision(scripted_model(lambda messages, info: response()), records)
    inspected = []

    def replay_validate(output: Any, evidence: Any) -> Any:
        validate(output)
        inspected.append(evidence)

    result = replay_decisions(
        records, schemas={"select": Decision}, validators={"select": replay_validate}
    )
    assert result == [{"decisionId": "choose-1", "stage": "select", "valid": True}]
    assert inspected == [{"evidence-1": {"score": 0.5}}]


@pytest.mark.parametrize(
    "parts",
    [
        [TextPart('{"choice":"baseline","evidenceIds":["evidence-1"]}')],
        [ToolCallPart("shell", {"command": "false"})],
        [
            ToolCallPart("decision", {"choice": "baseline", "evidenceIds": []}),
            ToolCallPart("decision", {"choice": "stop", "evidenceIds": []}),
        ],
    ],
)
def test_output_tool_contract_is_repaired_without_executing_tools(
    records: Any, parts: Any
) -> None:
    prompts = []

    def model(messages: Any, info: Any) -> Any:
        prompts.append(json.loads(messages[0].parts[-1].content))
        assert info.function_tools == []
        return ModelResponse(parts=parts) if len(prompts) == 1 else response()

    assert run_decision(scripted_model(model), records).choice == "baseline"
    assert len(prompts) == 2
    assert "exactly one decision output tool call" in prompts[1]["validationErrors"][0]


def test_provider_accepts_json_string_tool_arguments(records: Any) -> None:
    model = scripted_model(
        lambda messages, info: ModelResponse(
            parts=[
                ToolCallPart(
                    "decision",
                    '{"choice":"baseline","evidenceIds":["evidence-1"]}',
                )
            ]
        )
    )
    assert run_decision(model, records).choice == "baseline"


@pytest.mark.parametrize("body", ["rate limit", {"error": "rate limit"}])
def test_unstructured_rate_limit_body_is_not_assumed_transient(
    records: Any, body: Any
) -> None:
    def model(messages: Any, info: Any) -> Any:
        raise ModelHTTPError(429, "fixture", body=body)

    with pytest.raises(ProviderError, match="Provider request failed"):
        run_decision(scripted_model(model), records)
    assert len([e for e in records.events() if e["kind"] == "modelRequest"]) == 1
    assert records.events()[-1]["transient"] is False


def test_unclassified_provider_error_is_sanitized_and_requires_explicit_resume(
    records: Any,
) -> None:
    attempts = 0

    def model(messages: Any, info: Any) -> Any:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("authorization=secret at https://private.invalid")

    client = scripted_model(model)
    with pytest.raises(ProviderError, match="Provider request failed"):
        run_decision(client, records)
    with pytest.raises(ProviderError, match="requires explicit resume"):
        run_decision(client, records)
    assert attempts == 1
    assert "secret" not in json.dumps(records.events())
    records.append("invocationStarted", invocationId="explicit-resume")
    assert (
        run_decision(scripted_model(lambda messages, info: response()), records).choice
        == "baseline"
    )


def test_interrupted_request_without_outcome_consumes_a_retry_slot(
    records: Any,
) -> None:
    class ProcessInterrupted(BaseException):
        pass

    def crash(messages: Any, info: Any) -> Any:
        raise ProcessInterrupted

    with pytest.raises(ProcessInterrupted):
        run_decision(scripted_model(crash), records)
    assert [event["kind"] for event in records.events()] == ["modelRequest"]
    assert (
        run_decision(scripted_model(lambda messages, info: response()), records).choice
        == "baseline"
    )
    failures = [e for e in records.events() if e["kind"] == "modelFailure"]
    assert len(failures) == 1
    assert failures[0]["errorType"] == "InterruptedRequest"
    assert failures[0]["transient"] is True
    assert len([e for e in records.events() if e["kind"] == "modelRequest"]) == 2


def test_saved_repeated_rejections_stop_before_calling_provider(records: Any) -> None:
    client = scripted_model(lambda messages, info: response("invented"))
    with pytest.raises(DecisionValidationError, match="identical invalid"):
        run_decision(client, records)
    previous = records.events()
    with pytest.raises(DecisionValidationError, match="identical invalid"):
        run_decision(
            scripted_model(lambda messages, info: pytest.fail("No retries remain")),
            records,
        )
    assert records.events() == previous


def test_different_invalid_repairs_exhaust_semantic_budget(records: Any) -> None:
    attempts = 0

    def model(messages: Any, info: Any) -> Any:
        nonlocal attempts
        attempts += 1
        return response(f"unregistered-{attempts}")

    with pytest.raises(DecisionValidationError, match="semantic repair was rejected"):
        run_decision(scripted_model(model), records)
    assert attempts == 2


def test_transport_retry_exhaustion_is_not_reset_by_implicit_reentry(
    records: Any,
) -> None:
    attempts = 0

    def model(messages: Any, info: Any) -> Any:
        nonlocal attempts
        attempts += 1
        raise httpx.ReadError("disconnected")

    client = scripted_model(model)
    with pytest.raises(ProviderError, match="transport retry was exhausted"):
        run_decision(client, records)
    assert attempts == 2
    with pytest.raises(ProviderError, match="transport retry was exhausted"):
        run_decision(client, records)
    assert attempts == 2


def test_saved_invalid_response_is_rejected_after_journal_interruption(
    records: Any, monkeypatch: Any
) -> None:
    append = records.append

    def fail_rejection(kind: str, **payload: Any) -> Any:
        if kind == "decisionRejected":
            raise OSError("journal publication interrupted")
        return append(kind, **payload)

    with monkeypatch.context() as patch:
        patch.setattr(records, "append", fail_rejection)
        with pytest.raises(OSError, match="publication interrupted"):
            run_decision(
                scripted_model(lambda messages, info: response("invented")), records
            )
    observed = []

    def repaired(messages: Any, info: Any) -> Any:
        observed.append(json.loads(messages[0].parts[-1].content))
        return response()

    assert run_decision(scripted_model(repaired), records).choice == "baseline"
    assert "registered option" in observed[0]["validationErrors"][0]
    assert len([e for e in records.events() if e["kind"] == "decisionRejected"]) == 1


def test_corrupt_saved_response_is_not_treated_as_an_unanswered_request(
    records: Any,
) -> None:
    def fail_validation(output: Any) -> None:
        raise RuntimeError("stopped after response publication")

    with pytest.raises(RuntimeError, match="stopped after response"):
        run_decision(
            scripted_model(lambda messages, info: response()),
            records,
            validate=fail_validation,
        )
    event = next(e for e in records.events() if e["kind"] == "modelRequest")
    (records.path / event["responsePath"]).write_text("{truncated")
    from scarf.agent.records import RecordError

    with pytest.raises(RecordError, match="Cannot read record"):
        run_decision(
            scripted_model(lambda messages, info: pytest.fail("Corruption must stop")),
            records,
        )


@pytest.mark.parametrize(
    ("configured", "runtime", "expected"),
    [
        (8192, RuntimeConfig(), 4096),
        (128, RuntimeConfig(maxOutputTokens=512), 512),
    ],
)
def test_effective_output_limit_and_saved_settings_are_explicit(
    records: Any, configured: int, runtime: RuntimeConfig, expected: int
) -> None:
    observed = []

    def model(messages: Any, info: Any) -> Any:
        observed.append(info.model_settings)
        return response()

    client = scripted_model(
        model,
        settings={
            "max_tokens": configured,
            "temperature": 0.1,
            "extra_headers": {"Authorization": "secret"},
            "thinking": {"providerPrivate": "secret"},
        },
    )
    run_decision(client, records, runtime=runtime)
    assert observed[0]["max_tokens"] == expected
    request = records.read_json(records.events()[0]["requestPath"])
    assert request["settings"] == {
        "max_tokens": expected,
        "temperature": 0.1,
        "thinking": False,
        "openai_reasoning_effort": "none",
        "extra_body": DISABLED_REASONING_BODY,
    }
    assert "secret" not in json.dumps(request)


@pytest.mark.parametrize("stage", ["context", "explore", "assess", "annotate"])
def test_all_decisions_disable_reasoning_through_repair_and_transport_retry(
    records: Any, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    supplied_settings = {
        "thinking": "high",
        "openai_reasoning_effort": "high",
        "openrouter_reasoning": {"enabled": True, "effort": "high"},
        "temperature": 0.2,
        "seed": 42,
        "extra_body": {
            "thinking": {"type": "enabled", "budget": 500},
            "reasoning_effort": "high",
            "chat_template_kwargs": {"thinking": True},
            "reasoning": {"enabled": True},
            "callerPrivate": {"value": "do-not-record"},
            "shared": "model",
        },
    }
    runtime = RuntimeConfig(
        modelSettings={
            "thinking": True,
            "openai_reasoning_effort": "low",
            "temperature": 0.3,
            "extra_body": {
                "thinking": {"type": "enabled"},
                "reasoning_effort": "low",
                "chat_template_kwargs": {"thinking": True},
                "reasoning": {"enabled": True, "effort": "high"},
                "runtimeOption": [1, 2],
                "shared": "runtime",
            },
        }
    )
    before_settings = deepcopy(supplied_settings)
    before_runtime = runtime.model_dump()
    observed = []

    def execute(messages: Any, info: Any) -> Any:
        if len(observed) == 1:
            return response("invalid")
        if len(observed) == 2:
            raise httpx.ConnectError("temporary transport failure")
        return response()

    client = scripted_model(execute, settings=supplied_settings)
    original_request = client.request

    async def request(messages: Any, settings: Any, parameters: Any) -> Any:
        observed.append(deepcopy(settings))
        return await original_request(messages, settings, parameters)

    monkeypatch.setattr(client, "request", request)
    result = asyncio.run(
        decide(
            client,
            decision_id=f"{stage}-reasoning-policy",
            stage=stage,
            evidence={"evidence-1": {"score": 0.5}},
            output_type=Decision,
            validate=validate,
            records=records,
            runtime=runtime,
            instructions="Choose a measured option.",
        )
    )
    assert result.choice == "baseline"
    assert len(observed) == 3
    for settings in observed:
        assert settings["thinking"] is False
        assert settings["openai_reasoning_effort"] == "none"
        assert settings["openrouter_reasoning"] == {"enabled": False}
        assert settings["temperature"] == 0.3
        assert settings["extra_body"] == {
            **DISABLED_REASONING_BODY,
            "callerPrivate": {"value": "do-not-record"},
            "runtimeOption": [1, 2],
            "shared": "runtime",
        }
    assert supplied_settings == before_settings
    assert client.settings == before_settings
    assert runtime.model_dump() == before_runtime
    requests = [e for e in records.events() if e["kind"] == "modelRequest"]
    assert len(requests) == 3
    for event in requests:
        saved = records.read_json(event["requestPath"])
        assert saved["settings"]["extra_body"] == DISABLED_REASONING_BODY
        assert saved["settings"]["thinking"] is False
        assert saved["settings"]["seed"] == 42
        assert "do-not-record" not in json.dumps(saved)
        assert "runtimeOption" not in json.dumps(saved)


@pytest.mark.parametrize("settings_location", ["model", "runtime"])
def test_explicit_native_reasoning_settings_cannot_override_agent_policy(
    records: Any, monkeypatch: pytest.MonkeyPatch, settings_location: str
) -> None:
    supplied = {
        "anthropic_thinking": {"type": "enabled"},
        "google_thinking_config": {"thinking_level": "HIGH"},
        "groq_reasoning_effort": "high",
        "groq_reasoning_format": "raw",
        "xai_reasoning_effort": "high",
        "snowflake_reasoning": {"effort": "high"},
        "bedrock_additional_model_requests_fields": {
            "thinking": {"type": "enabled"},
            "reasoning_effort": "high",
            "reasoning_config": "high",
            "top_k": 4,
        },
    }
    original = deepcopy(supplied)
    client = scripted_model(
        lambda messages, info: response(),
        settings=supplied if settings_location == "model" else {},
    )
    runtime = RuntimeConfig(
        modelSettings=supplied if settings_location == "runtime" else {}
    )
    original_request = client.request

    async def request(messages: Any, settings: Any, parameters: Any) -> Any:
        assert settings["anthropic_thinking"] == {"type": "disabled"}
        assert settings["google_thinking_config"] == {"thinking_budget": 0}
        assert settings["groq_reasoning_effort"] == "none"
        assert settings["groq_reasoning_format"] == "hidden"
        assert settings["xai_reasoning_effort"] == "none"
        assert settings["snowflake_reasoning"] == {"enabled": False}
        assert settings["bedrock_additional_model_requests_fields"] == {"top_k": 4}
        return await original_request(messages, settings, parameters)

    monkeypatch.setattr(client, "request", request)
    assert run_decision(client, records, runtime=runtime).choice == "baseline"
    assert supplied == original


@pytest.mark.parametrize("route", ["openai", "openrouter"])
def test_http_request_contains_exact_reasoning_disable_body(
    records: Any, route: str
) -> None:
    from pydantic_ai.models import override_allow_model_requests
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    payloads = []

    def handle(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        payloads.append(payload)
        assert {key: payload[key] for key in DISABLED_REASONING_BODY} == (
            DISABLED_REASONING_BODY
        )
        assert payload["temperature"] == 0.3
        assert payload["callerOption"] == "retained"
        assert payload["runtimeOption"] == "retained"
        return httpx.Response(
            200,
            json={
                "id": "offline-response",
                "object": "chat.completion",
                "created": 1,
                "model": "test-model",
                "provider": "offline-test-provider",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "offline-tool-call",
                                    "type": "function",
                                    "function": {
                                        "name": "decision",
                                        "arguments": json.dumps(
                                            {
                                                "choice": "baseline",
                                                "evidenceIds": ["evidence-1"],
                                            }
                                        ),
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 20,
                },
            },
        )

    async def execute() -> Any:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            provider_type = OpenAIProvider if route == "openai" else OpenRouterProvider
            model_type = OpenAIChatModel if route == "openai" else OpenRouterModel
            provider = provider_type(
                api_key="offline-test-only",
                http_client=http,
                **(
                    {"base_url": "https://model.invalid/v1"}
                    if route == "openai"
                    else {}
                ),
            )
            client = model_type(
                "test-model" if route == "openai" else "openai/test-model",
                provider=provider,
                settings={
                    "thinking": "high",
                    "openai_reasoning_effort": "high",
                    "openrouter_reasoning": {"enabled": True, "effort": "high"},
                    "extra_body": {
                        "thinking": {"type": "enabled"},
                        "reasoning_effort": "high",
                        "chat_template_kwargs": {"thinking": True},
                        "reasoning": {"enabled": True},
                        "callerOption": "retained",
                    },
                },
            )
            with override_allow_model_requests(True):
                return await decide(
                    client,
                    decision_id="wire-test",
                    stage="explore",
                    evidence={"evidence-1": {"score": 0.5}},
                    output_type=Decision,
                    validate=validate,
                    records=records,
                    runtime=RuntimeConfig(
                        modelSettings={
                            "openai_reasoning_effort": "low",
                            "temperature": 0.3,
                            "extra_body": {"runtimeOption": "retained"},
                        }
                    ),
                    instructions="Choose a measured option.",
                )

    assert asyncio.run(execute()).choice == "baseline"
    assert len(payloads) == 1


@pytest.mark.parametrize("body", [[], "invalid"])
def test_non_mapping_extra_body_stops_without_sending_request(
    records: Any, body: Any
) -> None:
    client = scripted_model(
        lambda messages, info: pytest.fail("Invalid settings must not be sent"),
        settings={"extra_body": body},
    )
    with pytest.raises(ProviderError, match="extra_body must be a mapping"):
        run_decision(client, records)
    assert records.events() == []


def test_missing_optional_distribution_version_is_saved_as_unknown(
    records: Any, monkeypatch: Any
) -> None:
    version = importlib.metadata.version

    def available_version(package: str) -> str:
        if package == "pydantic-ai-slim":
            raise importlib.metadata.PackageNotFoundError(package)
        return version(package)

    monkeypatch.setattr(importlib.metadata, "version", available_version)
    run_decision(scripted_model(lambda messages, info: response()), records)
    request = records.read_json(records.events()[0]["requestPath"])
    assert request["softwareVersions"]["pydantic-ai-slim"] == "unknown"
    assert request["softwareVersions"]["pydantic"] != "unknown"


def test_string_model_identity_does_not_require_provider_initialization() -> None:
    assert model_identity("provider:configured-model") == {
        "model": "provider:configured-model"
    }


def test_offline_replay_rejects_replacement_output_schema(records: Any) -> None:
    class ReplacementDecision(Decision):
        rationale: str

    run_decision(scripted_model(lambda messages, info: response()), records)
    with pytest.raises(ProviderError, match="Replay schema differs"):
        replay_decisions(
            records,
            schemas={"select": ReplacementDecision},
            validators={"select": lambda output, evidence: validate(output)},
        )


@pytest.mark.parametrize("altered", ["evidence", "instructions"])
def test_offline_replay_rejects_modified_prompt_or_evidence(
    records: Any, altered: str
) -> None:
    run_decision(scripted_model(lambda messages, info: response()), records)
    path = records.path / records.events()[0]["requestPath"]
    request = json.loads(path.read_text())
    if altered == "evidence":
        payload = json.loads(request["userPrompt"])
        payload["evidence"]["evidence-1"]["score"] = 0.9
        request["userPrompt"] = json.dumps(payload)
    else:
        request["instructions"] = "Different scientific procedure"
    path.write_text(json.dumps(request))
    with pytest.raises(ProviderError, match="Replay evidence or prompt digest"):
        replay_decisions(
            records,
            schemas={"select": Decision},
            validators={"select": lambda output, evidence: validate(output)},
        )


def test_unclassified_errors_are_not_retried_without_the_optional_openai_sdk(
    records: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    # A None entry makes "from openai import ..." raise ImportError.
    monkeypatch.setitem(sys.modules, "openai", None)
    attempts = 0

    def model(messages: Any, info: Any) -> Any:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("private transport detail")

    with pytest.raises(
        ProviderError, match=r"Provider request failed \(RuntimeError\)"
    ):
        run_decision(scripted_model(model), records)
    assert attempts == 1
    failure = records.latest("modelFailure")
    assert failure["transient"] is False
    assert failure["errorType"] == "RuntimeError"
    assert "private" not in json.dumps(records.events())


def test_openai_connection_errors_use_the_single_transient_retry(
    records: Any,
) -> None:
    openai = pytest.importorskip("openai")
    attempts = 0

    def model(messages: Any, info: Any) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise openai.APIConnectionError(
                message="private connection detail",
                request=httpx.Request("POST", "https://model.invalid/v1"),
            )
        return response()

    assert run_decision(scripted_model(model), records).choice == "baseline"
    assert attempts == 2
    failure = records.latest("modelFailure")
    assert failure["transient"] is True
    assert failure["errorType"] == "APIConnectionError"
    assert "private" not in json.dumps(records.events())
