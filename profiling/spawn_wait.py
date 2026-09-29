"""Spawn-safe waits: never use Function.remote() for long profiling work."""

import time
from typing import Any

from profiling.config import ProfilingConfig, StageName
from profiling.modal_support import await_function_call
from profiling.results import load_result

# How often orchestrators poll R2 / call status. Short polls keep heartbeats alive.
DEFAULT_POLL_SECONDS = 20.0
# Extra grace after stage timeout for scheduling + result upload.
DEFAULT_GRACE_SECONDS = 600.0


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
