import signal

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
