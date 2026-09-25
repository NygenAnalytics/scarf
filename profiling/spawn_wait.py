"""Spawn-safe waits: never use Function.remote() for long profiling work."""

import time
from typing import Any

from profiling.config import ProfilingConfig, StageName
from profiling.results import load_result

# How often orchestrators poll R2 / call status. Short polls keep heartbeats alive.
DEFAULT_POLL_SECONDS = 20.0
# Extra grace after stage timeout for scheduling + result upload.
DEFAULT_GRACE_SECONDS = 600.0

# Modal sometimes surfaces failed calls as empty TimeoutError from get(timeout=...).
# Detect terminal statuses via the call graph instead of spinning until deadline.
_TERMINAL_FAILURE_STATUSES = frozenset(
    {"FAILURE", "INIT_FAILURE", "TERMINATED", "TIMEOUT"}
)


def _call_id(call: Any) -> str | None:
    # object_id is a property that raises until hydrated; getattr default masks that.
    if hasattr(call, "hydrate"):
        try:
            call.hydrate()
        except Exception:  # noqa: BLE001 - best-effort
            pass
    try:
        object_id = call.object_id
    except Exception:  # noqa: BLE001 - unhydrated / closed client
        object_id = None
    if object_id:
        return object_id
    return getattr(call, "call_id", None)


def _input_status_name(call: Any) -> str | None:
    """Return InputStatus name for this call, if the call graph exposes it."""
    call_id = _call_id(call)
    if not call_id or not hasattr(call, "get_call_graph"):
        return None
    try:
        graph = call.get_call_graph()
    except Exception:  # noqa: BLE001 - best-effort probe
        return None
    stack = list(graph or [])
    while stack:
        node = stack.pop()
        if getattr(node, "function_call_id", None) == call_id:
            status = getattr(node, "status", None)
            if status is None:
                return None
            return getattr(status, "name", str(status))
        stack.extend(getattr(node, "children", None) or [])
    return None


def _raise_if_terminal_failure(call: Any) -> None:
    status = _input_status_name(call)
    if status in _TERMINAL_FAILURE_STATUSES:
        raise RuntimeError(
            f"Spawned call {_call_id(call) or '<unknown>'} ended with status={status}"
        )


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
                _raise_if_terminal_failure(call)
                still_pending.append((index, call))
        pending = still_pending
    return [item for item in results]


def await_function_call(
    call: Any,
    *,
    pollSeconds: float = DEFAULT_POLL_SECONDS,
    deadlineSeconds: float,
) -> Any:
    """Wait on a spawned FunctionCall with short get() polls (not .remote())."""
    return await_many_function_calls(
        [call], pollSeconds=pollSeconds, deadlineSeconds=deadlineSeconds
    )[0]


def await_stage_result(
    config: ProfilingConfig,
    nRows: int,
    stage: StageName,
    call: Any,
    *,
    submissionId: str,
    pollSeconds: float = DEFAULT_POLL_SECONDS,
    deadlineSeconds: float,
) -> dict[str, Any]:
    """Wait for a spawned stage, preferring its durable result JSON on R2.

    The JSON is read before every short poll and after a call error, because a
    call can fail or stall after it persisted its result.
    """
    deadline = time.monotonic() + deadlineSeconds
    while (
        stored := load_result(config, nRows, stage, submissionId=submissionId)
    ) is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Timed out waiting for {stage} result at "
                f"{config.resultUri(nRows, stage)}"
            )
        try:
            payload = await_function_call(
                call,
                pollSeconds=pollSeconds,
                deadlineSeconds=min(pollSeconds, remaining),
            )
        except TimeoutError:
            continue
        except Exception:
            if (
                stored := load_result(config, nRows, stage, submissionId=submissionId)
            ) is not None:
                return stored
            raise
        if not isinstance(payload, dict):
            raise TypeError(
                f"Stage {stage} returned non-dict payload: {type(payload).__name__}"
            )
        if payload.get("submissionId") != submissionId:
            raise ValueError(f"Stage {stage} returned a result from another submission")
        return payload
    return stored
