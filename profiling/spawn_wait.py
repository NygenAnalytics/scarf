"""Spawn-safe waits: never use Function.remote() for long profiling work."""

import time
from collections.abc import Callable
from typing import Any

from profiling.config import ProfilingConfig, StageName
from profiling.r2 import get_json
from profiling.results import result_exists

# How often orchestrators poll R2 / call status. Short polls keep heartbeats alive.
DEFAULT_POLL_SECONDS = 20.0
# Extra grace after stage timeout for scheduling + result upload.
DEFAULT_GRACE_SECONDS = 600.0
# Re-spawn a stage this many times if the Modal call dies without an R2 result.
DEFAULT_STAGE_SPAWN_ATTEMPTS = 3


def await_function_call(
    call: Any,
    *,
    pollSeconds: float = DEFAULT_POLL_SECONDS,
    deadlineSeconds: float,
) -> Any:
    """Wait on a spawned FunctionCall with short get() polls (not .remote())."""
    if pollSeconds <= 0:
        raise ValueError("pollSeconds must be positive")
    if deadlineSeconds <= 0:
        raise ValueError("deadlineSeconds must be positive")
    deadline = time.monotonic() + deadlineSeconds
    while time.monotonic() < deadline:
        remaining = max(0.1, deadline - time.monotonic())
        timeout = min(pollSeconds, remaining)
        try:
            return call.get(timeout=timeout)
        except TimeoutError:
            continue
    raise TimeoutError(f"Spawned call did not finish within {deadlineSeconds:.0f}s")


def await_many_function_calls(
    calls: list[Any],
    *,
    pollSeconds: float = DEFAULT_POLL_SECONDS,
    deadlineSeconds: float,
) -> list[Any]:
    """Wait on many spawned calls with short interleaved get() polls."""
    if not calls:
        return []
    if pollSeconds <= 0:
        raise ValueError("pollSeconds must be positive")
    if deadlineSeconds <= 0:
        raise ValueError("deadlineSeconds must be positive")
    deadline = time.monotonic() + deadlineSeconds
    pending = list(enumerate(calls))
    results: list[Any | None] = [None] * len(calls)
    while pending:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Spawned calls did not finish within {deadlineSeconds:.0f}s "
                f"({len(pending)} still pending)"
            )
        still_pending: list[tuple[int, Any]] = []
        for index, call in pending:
            remaining = max(0.1, deadline - time.monotonic())
            timeout = min(pollSeconds, remaining)
            try:
                results[index] = call.get(timeout=timeout)
            except TimeoutError:
                still_pending.append((index, call))
        pending = still_pending
    return [item for item in results]


def await_stage_result(
    config: ProfilingConfig,
    nRows: int,
    stage: StageName,
    call: Any,
    *,
    pollSeconds: float = DEFAULT_POLL_SECONDS,
    deadlineSeconds: float,
    onPoll: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Prefer durable R2 result JSON; fall back to the call return value."""
    if pollSeconds <= 0:
        raise ValueError("pollSeconds must be positive")
    if deadlineSeconds <= 0:
        raise ValueError("deadlineSeconds must be positive")
    deadline = time.monotonic() + deadlineSeconds
    last_error: BaseException | None = None

    while time.monotonic() < deadline:
        if result_exists(config, nRows, stage):
            return get_json(config.resultUri(nRows, stage))
        if onPoll is not None:
            onPoll()
        remaining = max(0.1, deadline - time.monotonic())
        timeout = min(pollSeconds, remaining)
        try:
            payload = call.get(timeout=timeout)
        except TimeoutError:
            continue
        except Exception as exc:  # noqa: BLE001 - Modal surfaces many failure types
            last_error = exc
            # Call may have died after writing R2; check once more.
            if result_exists(config, nRows, stage):
                return get_json(config.resultUri(nRows, stage))
            raise

        if result_exists(config, nRows, stage):
            return get_json(config.resultUri(nRows, stage))
        if isinstance(payload, dict):
            return payload
        raise TypeError(
            f"Stage {stage} returned non-dict payload: {type(payload).__name__}"
        )

    if result_exists(config, nRows, stage):
        return get_json(config.resultUri(nRows, stage))
    if last_error is not None:
        raise TimeoutError(
            f"Timed out waiting for {stage} after call error: {last_error}"
        ) from last_error
    raise TimeoutError(
        f"Timed out waiting for {stage} result at {config.resultUri(nRows, stage)}"
    )
