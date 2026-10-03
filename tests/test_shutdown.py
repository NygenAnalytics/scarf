import signal
import threading

import pytest

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
