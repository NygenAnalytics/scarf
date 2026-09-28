"""Modal names and spawn-safe waits shared by the profiling and documentation apps.

The documentation image copies this file without the rest of ``profiling``, so it must not
import ``modal`` or another profiling module. Calls are duck-typed ``modal.FunctionCall``
objects.
"""

import time
from typing import Any

MODAL_ENVIRONMENT_NAME = "scarf_profiling"
MODAL_APP_NAME = "scarf-profiling"

# Modal can surface a failed input as an empty TimeoutError from get(timeout=...), so
# waits read the call graph instead of spinning until their deadline.
TERMINAL_FAILURE_STATUSES = frozenset(
    {"FAILURE", "INIT_FAILURE", "TERMINATED", "TIMEOUT"}
)


def call_id(call: Any) -> str | None:
    """Return the Modal object ID of ``call``, or None when it is unavailable."""
    # object_id is a property that raises until the call is hydrated.
    if hasattr(call, "hydrate"):
        try:
            call.hydrate()
        except Exception:  # noqa: BLE001 - best-effort
            pass
    try:
        object_id = call.object_id
    except Exception:  # noqa: BLE001 - unhydrated call or closed client
        return None
    return str(object_id) if object_id else None


def input_status_name(call: Any) -> str | None:
    """Return the input status name of ``call`` from its call graph, if exposed."""
    identifier = call_id(call)
    if identifier is None or not hasattr(call, "get_call_graph"):
        return None
    try:
        graph = call.get_call_graph()
    except Exception:  # noqa: BLE001 - best-effort probe
        return None
    stack = list(graph or [])
    while stack:
        node = stack.pop()
        if getattr(node, "function_call_id", None) == identifier:
            status = getattr(node, "status", None)
            return None if status is None else str(getattr(status, "name", status))
        stack.extend(getattr(node, "children", None) or [])
    return None


def raise_if_terminal_failure(call: Any) -> None:
    status = input_status_name(call)
    if status in TERMINAL_FAILURE_STATUSES:
        raise RuntimeError(
            f"Spawned call {call_id(call) or '<unknown>'} ended with status={status}"
        )


def await_function_calls(
    calls: list[Any],
    *,
    pollSeconds: float,
    deadlineSeconds: float,
) -> list[Any]:
    """Wait on spawned calls with short interleaved ``get`` polls, never ``.remote()``."""
    if not calls:
        return []
    if pollSeconds <= 0:
        raise ValueError("pollSeconds must be positive")
    if deadlineSeconds <= 0:
        raise ValueError("deadlineSeconds must be positive")
    deadline = time.monotonic() + deadlineSeconds
    pending = list(enumerate(calls))
    results: list[Any] = [None] * len(calls)
    while pending:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Spawned calls did not finish within {deadlineSeconds:.0f}s "
                f"({len(pending)} still pending)"
            )
        still_pending: list[tuple[int, Any]] = []
        for index, call in pending:
            remaining = max(0.1, deadline - time.monotonic())
            try:
                results[index] = call.get(timeout=min(pollSeconds, remaining))
            except TimeoutError:
                raise_if_terminal_failure(call)
                still_pending.append((index, call))
        pending = still_pending
    return results


def await_function_call(
    call: Any,
    *,
    pollSeconds: float,
    deadlineSeconds: float,
) -> Any:
    return await_function_calls(
        [call], pollSeconds=pollSeconds, deadlineSeconds=deadlineSeconds
    )[0]
