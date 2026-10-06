import os
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

import scarf.utils.shutdown as shutdown_module
from scarf.utils.shutdown import (
    ShutdownRequested,
    ShutdownToken,
    TemporarySignalGuard,
    current_shutdown_token,
    shutdown_checkpoint,
    shutdown_scope,
)


def test_shutdown_scope_raises_only_at_a_checkpoint_and_resets_context() -> None:
    token = ShutdownToken()
    with shutdown_scope(token):
        assert current_shutdown_token() is token
        assert token.request(reason="stop work")
        assert not token.request(reason="second request")
        with pytest.raises(ShutdownRequested, match="stop work") as caught:
            shutdown_checkpoint()
        assert caught.value.request.reason == "stop work"

    assert current_shutdown_token() is None


def test_signal_guard_respects_ignored_signals_and_restores_handlers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(threading, "current_thread", threading.main_thread)
    installed: dict[int, object] = {}
    prior = {
        int(signal.SIGTERM): signal.SIG_IGN,
        int(signal.SIGINT): signal.default_int_handler,
        int(signal.SIGHUP): signal.SIG_DFL,
    }
    monkeypatch.setattr(signal, "getsignal", lambda number: prior[int(number)])
    monkeypatch.setattr(
        signal,
        "signal",
        lambda number, handler: installed.__setitem__(int(number), handler),
    )

    token = ShutdownToken()
    with TemporarySignalGuard(token) as guard:
        assert guard.available
        assert int(signal.SIGTERM) not in installed
        handler = installed[int(signal.SIGINT)]
        assert callable(handler)
        handler(int(signal.SIGINT), None)
        assert token.requested
        assert token.request_record is not None
        assert token.request_record.signal_name == "SIGINT"

    assert installed[int(signal.SIGINT)] is signal.default_int_handler
    assert installed[int(signal.SIGHUP)] == signal.SIG_DFL


def test_second_signal_escalates_to_the_prior_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(threading, "current_thread", threading.main_thread)
    installed: dict[int, object] = {}
    propagated: list[int] = []

    def prior_handler(signum: int, _frame: object) -> None:
        propagated.append(signum)

    monkeypatch.setattr(signal, "getsignal", lambda _number: prior_handler)
    monkeypatch.setattr(
        signal,
        "signal",
        lambda number, handler: installed.__setitem__(int(number), handler),
    )
    token = ShutdownToken()
    with TemporarySignalGuard(token):
        handler = installed[int(signal.SIGTERM)]
        assert callable(handler)
        handler(int(signal.SIGTERM), None)
        assert propagated == []
        handler(int(signal.SIGTERM), None)
        assert propagated == [int(signal.SIGTERM)]


def test_signal_guard_reports_non_main_thread_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = object()
    main = object()
    monkeypatch.setattr(
        "scarf.utils.shutdown.threading.current_thread", lambda: current
    )
    monkeypatch.setattr("scarf.utils.shutdown.threading.main_thread", lambda: main)

    with TemporarySignalGuard(ShutdownToken()) as guard:
        assert not guard.available
        assert guard.unavailable_reason == "signal handlers require the main thread"


def test_signal_guard_leaves_handlers_installed_outside_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(threading, "current_thread", threading.main_thread)
    installed: dict[int, object] = {}
    prior = {
        int(signal.SIGTERM): signal.SIG_DFL,
        int(signal.SIGINT): signal.default_int_handler,
        int(signal.SIGHUP): None,
    }
    monkeypatch.setattr(signal, "getsignal", lambda number: prior[int(number)])
    monkeypatch.setattr(
        signal,
        "signal",
        lambda number, handler: installed.__setitem__(int(number), handler),
    )

    with TemporarySignalGuard(ShutdownToken()) as guard:
        assert guard.available
        assert int(signal.SIGHUP) not in installed
        assert callable(installed[int(signal.SIGTERM)])

    assert installed[int(signal.SIGTERM)] == signal.SIG_DFL
    assert installed[int(signal.SIGINT)] is signal.default_int_handler
    assert int(signal.SIGHUP) not in installed


def test_token_requests_need_a_reason_and_name_unknown_signals() -> None:
    token = ShutdownToken()
    for reason in ("", None, 3):
        with pytest.raises(TypeError, match="reason must be a non-empty string"):
            token.request(reason=reason)  # type: ignore[arg-type]
    assert not token.requested

    assert token.request(reason="operator stop", signal_number=1_000)
    record = token.request_record
    assert record is not None
    assert (record.reason, record.signal_number, record.signal_name) == (
        "operator stop",
        1_000,
        "SIGNAL_1000",
    )
    assert record.requested_at_ns > 0


def test_propagate_continues_with_the_prior_signal_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raised: list[int] = []
    monkeypatch.setattr(signal, "raise_signal", raised.append)

    # Nothing was requested, so there is nothing to continue.
    ShutdownToken().propagate()

    called: list[tuple[int, object]] = []
    frame = object()
    callable_prior = ShutdownToken()
    callable_prior.request(
        reason="received SIGTERM",
        signal_number=int(signal.SIGTERM),
        previous_handler=lambda number, current: called.append((number, current)),
        frame=frame,  # type: ignore[arg-type]
    )
    with pytest.raises(ShutdownRequested, match="received SIGTERM"):
        callable_prior.propagate()
    assert called == [(int(signal.SIGTERM), frame)]

    ignored = ShutdownToken()
    ignored.request(
        reason="received SIGINT",
        signal_number=int(signal.SIGINT),
        previous_handler=signal.SIG_IGN,
    )
    with pytest.raises(ShutdownRequested, match="received SIGINT"):
        ignored.propagate()
    assert raised == []

    default = ShutdownToken()
    default.request(
        reason="received SIGHUP",
        signal_number=int(signal.SIGHUP),
        previous_handler=signal.SIG_DFL,
    )
    with pytest.raises(ShutdownRequested, match="received SIGHUP"):
        default.propagate()
    # The default action is delivered again before the interruption is raised.
    assert raised == [int(signal.SIGHUP)]


def test_scopes_and_guards_accept_only_shutdown_tokens() -> None:
    with pytest.raises(TypeError, match="token must be a ShutdownToken"):
        with shutdown_scope(object()):  # type: ignore[arg-type]
            pass
    with pytest.raises(TypeError, match="token must be a ShutdownToken"):
        TemporarySignalGuard(object())  # type: ignore[arg-type]
    assert current_shutdown_token() is None


def test_signal_guard_reports_when_no_signal_can_be_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(threading, "current_thread", threading.main_thread)
    installed: dict[int, object] = {}
    monkeypatch.setattr(signal, "getsignal", lambda _number: signal.SIG_IGN)
    monkeypatch.setattr(
        signal,
        "signal",
        lambda number, handler: installed.__setitem__(int(number), handler),
    )

    with TemporarySignalGuard(ShutdownToken()) as guard:
        assert not guard.available
        assert (
            guard.unavailable_reason == "no catchable termination signals are available"
        )
    assert installed == {}


def test_second_signal_with_a_default_prior_delivers_it_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(threading, "current_thread", threading.main_thread)
    installed: dict[int, list[object]] = {}
    raised: list[int] = []
    monkeypatch.setattr(signal, "getsignal", lambda _number: signal.SIG_DFL)
    monkeypatch.setattr(
        signal,
        "signal",
        lambda number, handler: installed.setdefault(int(number), []).append(handler),
    )
    monkeypatch.setattr(signal, "raise_signal", raised.append)
    token = ShutdownToken()

    with TemporarySignalGuard(token):
        handler = installed[int(signal.SIGTERM)][-1]
        assert callable(handler)
        handler(int(signal.SIGTERM), None)
        assert raised == []
        handler(int(signal.SIGTERM), None)
        # The second signal restores the default handler and delivers again.
        assert installed[int(signal.SIGTERM)][-1] == signal.SIG_DFL
        assert raised == [int(signal.SIGTERM)]
    assert token.request_record is not None
    assert token.request_record.signal_name == "SIGTERM"


def test_signal_during_a_request_cannot_deadlock_the_same_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(threading, "current_thread", threading.main_thread)
    installed: dict[int, object] = {}
    escalated: list[int] = []

    def prior_handler(signum: int, _frame: object) -> None:
        escalated.append(signum)

    monkeypatch.setattr(signal, "getsignal", lambda _number: prior_handler)
    monkeypatch.setattr(
        signal,
        "signal",
        lambda number, handler: installed.__setitem__(int(number), handler),
    )
    token = ShutdownToken()
    real_time_ns = time.time_ns
    nested: list[bool] = []

    def time_ns_with_a_second_signal() -> int:
        # A second SIGINT arrives while the first one is being recorded, as
        # CPython can run a pending handler inside the interrupted handler.
        if not nested:
            nested.append(True)
            handler = installed[int(signal.SIGINT)]
            assert callable(handler)
            handler(int(signal.SIGINT), None)
        return real_time_ns()

    # Only the shutdown module sees the patched clock.
    monkeypatch.setattr(
        shutdown_module, "time", SimpleNamespace(time_ns=time_ns_with_a_second_signal)
    )
    with TemporarySignalGuard(token):
        handler = installed[int(signal.SIGINT)]
        assert callable(handler)
        worker = threading.Thread(
            target=handler, args=(int(signal.SIGINT), None), daemon=True
        )
        worker.start()
        worker.join(timeout=5)
        assert not worker.is_alive(), "a nested signal deadlocked the token"

    # One signal requested shutdown and the other escalated to the prior handler.
    assert escalated == [int(signal.SIGINT)]
    assert token.requested
    assert token.request_record is not None
    assert token.request_record.signal_name == "SIGINT"
    with pytest.raises(ShutdownRequested, match="received SIGINT"):
        token.checkpoint()


def test_concurrent_requests_record_exactly_one_winner() -> None:
    token = ShutdownToken()
    start = threading.Barrier(8)
    results: list[bool] = []
    results_lock = threading.Lock()

    def request(index: int) -> None:
        start.wait()
        first = token.request(reason=f"request {index}")
        with results_lock:
            results.append(first)

    workers = [
        threading.Thread(target=request, args=(index,), daemon=True)
        for index in range(8)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=5)
    assert sorted(results) == [False] * 7 + [True]
    record = token.request_record
    assert record is not None and record.reason.startswith("request ")


def test_signal_guard_restores_installed_handlers_when_installation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(threading, "current_thread", threading.main_thread)
    calls: list[tuple[int, object]] = []

    def prior_handler(_signum: int, _frame: object) -> None:
        return None

    def install(number: int, handler: object) -> None:
        if int(number) == int(signal.SIGINT) and handler is not prior_handler:
            raise OSError("cannot install a SIGINT handler")
        calls.append((int(number), handler))

    monkeypatch.setattr(signal, "getsignal", lambda _number: prior_handler)
    monkeypatch.setattr(signal, "signal", install)
    guard = TemporarySignalGuard(ShutdownToken())

    with pytest.raises(OSError, match="cannot install a SIGINT handler"):
        guard.__enter__()

    # SIGTERM was installed before SIGINT failed, so both get their prior
    # handler back.
    assert [number for number, _ in calls] == [
        int(signal.SIGTERM),
        int(signal.SIGTERM),
        int(signal.SIGINT),
    ]
    assert callable(calls[0][1]) and calls[0][1] is not prior_handler
    assert calls[1][1] is prior_handler and calls[2][1] is prior_handler
    assert not guard.available
    guard.__exit__(None, None, None)
    assert len(calls) == 3


_SIGNAL_LOOP_CHILD = """
import sys
import time

from scarf.utils.shutdown import ShutdownRequested, ShutdownToken, TemporarySignalGuard

token = ShutdownToken()
with TemporarySignalGuard(token) as guard:
    assert guard.available
    print("READY", flush=True)
    deadline = time.monotonic() + 20
    try:
        # Reading the token dominates the loop, so the signal usually arrives
        # while a reader is inside the token.
        while time.monotonic() < deadline:
            for _ in range(1000):
                token.checkpoint()
                token.requested
    except ShutdownRequested as error:
        print("REQUESTED", error.request.signal_name, flush=True)
        sys.exit(0)
sys.exit(3)
"""


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal delivery")
@pytest.mark.skipif(
    sys.version_info < (3, 14),
    reason="a single signal deadlocked a locking token only on Python 3.14",
)
def test_sigint_while_polling_the_token_never_deadlocks() -> None:
    environment = os.environ | {"OMP_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1"}
    for trial in range(10):
        child = subprocess.Popen(
            [sys.executable, "-c", _SIGNAL_LOOP_CHILD],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        )
        try:
            assert child.stdout is not None
            assert child.stdout.readline().strip() == "READY"
            # Let the loop take and release the token many times first.
            time.sleep(0.05)
            child.send_signal(signal.SIGINT)
            stdout, stderr = child.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.communicate()
            pytest.fail(f"trial {trial}: SIGINT deadlocked the polling loop")
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate()
        assert child.returncode == 0, (trial, stdout, stderr)
        assert stdout.strip() == "REQUESTED SIGINT", (trial, stdout, stderr)
