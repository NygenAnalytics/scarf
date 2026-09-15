"""Notebook interrupts cancel real worker tasks without waiting for model output."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import Any

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from scarf.agent.config import agent_exec
from scarf.agent.types import AgentDataModel


class Choice(AgentDataModel):
    value: int


@pytest.mark.parametrize("callback_fails", [False, True])
def test_notebook_interrupt_cancels_pending_request_and_keeps_partial_usage(
    monkeypatch: Any,
    callback_fails: bool,
) -> None:
    waiting = Event()
    cancelled = Event()
    calls = {"requests": 0, "measurements": 0, "completedResponses": 0}
    attempts = []
    original_interrupt = KeyboardInterrupt("Notebook interrupted")

    async def measure() -> int:
        calls["measurements"] += 1
        return 2

    async def respond(_messages: Any, _info: Any) -> ModelResponse:
        calls["requests"] += 1
        if calls["requests"] == 1:
            calls["completedResponses"] += 1
            return ModelResponse(
                parts=[ToolCallPart("measure", {})],
                usage=RequestUsage(input_tokens=11, output_tokens=2),
            )
        waiting.set()
        try:
            await asyncio.Event().wait()
            raise AssertionError("A pending request must not produce a completion")
        except asyncio.CancelledError:
            cancelled.set()
            raise

    def save_attempt(info: Any) -> None:
        attempts.append(info)
        if callback_fails:
            raise RuntimeError("Saving cancellation evidence failed")

    class InterruptingPool(ThreadPoolExecutor):
        def submit(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
            future = super().submit(fn, *args, **kwargs)
            original_result = future.result
            first = True

            def result(timeout: float | None = None) -> Any:
                nonlocal first
                if first:
                    first = False
                    assert waiting.wait(10), "The simulated provider did not start"
                    raise original_interrupt
                return original_result(timeout=timeout)

            future.result = result
            return future

    monkeypatch.setattr(agent_exec, "ThreadPoolExecutor", InterruptingPool)

    async def notebook() -> BaseException:
        with pytest.raises(KeyboardInterrupt) as caught:
            agent_exec.run_agent_sync(
                model=FunctionModel(respond),
                output_type=Choice,
                system_prompt="Interpret a measured count.",
                user_prompt="Measure once, then decide.",
                tools=[measure],
                on_attempt=save_attempt,
            )
        return caught.value

    interrupted = asyncio.run(notebook())
    assert interrupted is original_interrupt
    assert cancelled.is_set()
    assert calls == {"requests": 2, "measurements": 1, "completedResponses": 1}
    assert len(attempts) == 1
    assert interrupted.agent_run_info is attempts[0]
    assert attempts[0].status == "failed"
    assert attempts[0].usage.availability == "partial"
    assert attempts[0].usage.inputTokens == 11
    assert attempts[0].usage.requests == 2
    assert attempts[0].errorType == "CancelledError"
    if callback_fails:
        assert "Saving cancellation evidence failed" in agent_exec.describe_agent_error(
            interrupted
        )


def test_notebook_interrupt_before_worker_ready_never_starts_a_provider_request(
    monkeypatch: Any,
) -> None:
    ready = Event()
    release = Event()
    attempts = []
    requests = []
    original_interrupt = KeyboardInterrupt("Interrupted before worker startup")

    async def respond(_messages: Any, _info: Any) -> ModelResponse:
        requests.append("unexpected")
        raise AssertionError("The interrupted invocation must not contact the provider")

    class StartingPool(ThreadPoolExecutor):
        def submit(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
            def start() -> Any:
                ready.set()
                assert release.wait(10), "Worker startup was not released for cleanup"
                return fn(*args, **kwargs)

            future = super().submit(start)
            original_result = future.result
            first = True

            def result(timeout: float | None = None) -> Any:
                nonlocal first
                if first:
                    first = False
                    assert ready.wait(10), "The worker thread did not start"
                    raise original_interrupt
                release.set()
                return original_result(timeout=timeout)

            future.result = result
            return future

    monkeypatch.setattr(agent_exec, "ThreadPoolExecutor", StartingPool)

    async def notebook() -> BaseException:
        with pytest.raises(KeyboardInterrupt) as caught:
            agent_exec.run_agent_sync(
                model=FunctionModel(respond),
                output_type=Choice,
                system_prompt="Interpret evidence.",
                user_prompt="Assess the observed count.",
                on_attempt=attempts.append,
            )
        return caught.value

    interrupted = asyncio.run(notebook())
    assert interrupted is original_interrupt
    assert requests == []
    assert len(attempts) == 1
    assert interrupted.agent_run_info is attempts[0]
    assert attempts[0].status == "failed"
    assert attempts[0].errorType == "CancelledError"
    assert attempts[0].usage.availability == "unavailable"
    assert "before model execution" in attempts[0].error


def test_notebook_interrupt_racing_with_completed_worker_preserves_actual_usage(
    monkeypatch: Any,
) -> None:
    attempts = []
    original_interrupt = KeyboardInterrupt("Interrupted after completion")

    async def respond(_messages: Any, info: Any) -> ModelResponse:
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, {"value": 1})],
            usage=RequestUsage(input_tokens=7, output_tokens=2),
        )

    class CompletedPool(ThreadPoolExecutor):
        def submit(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
            future = super().submit(fn, *args, **kwargs)
            original_result = future.result
            first = True

            def result(timeout: float | None = None) -> Any:
                nonlocal first
                completed = original_result(timeout=timeout)
                if first:
                    first = False
                    raise original_interrupt
                return completed

            future.result = result
            return future

    monkeypatch.setattr(agent_exec, "ThreadPoolExecutor", CompletedPool)

    async def notebook() -> BaseException:
        with pytest.raises(KeyboardInterrupt) as caught:
            agent_exec.run_agent_sync(
                model=FunctionModel(respond),
                output_type=Choice,
                system_prompt="Interpret evidence.",
                user_prompt="Use the measured count.",
                on_attempt=attempts.append,
            )
        return caught.value

    interrupted = asyncio.run(notebook())
    assert interrupted is original_interrupt
    assert len(attempts) == 1
    assert attempts[0].status == "done"
    assert attempts[0].usage.inputTokens == 7
    assert interrupted.agent_run_info is attempts[0]
    assert "completed before cancellation" in agent_exec.describe_agent_error(
        interrupted
    )
