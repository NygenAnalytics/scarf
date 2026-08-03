"""Behavioral tests for process helpers."""

import subprocess
from unittest.mock import MagicMock

import pytest

from scarf.utils.process import process_rss_mb, rss_peak_tracker, system_call


def test_system_call_forwards_stdout_to_logger(monkeypatch):
    lines = [b"hello\n", b""]
    process = MagicMock()
    process.stdout.readline.side_effect = lines
    process.poll.side_effect = [None, 0, 0]

    logged: list[object] = []
    launched: list[object] = []

    def fake_popen(args, **kwargs):
        launched.append((args, kwargs))
        return process

    monkeypatch.setattr("scarf.utils.process.subprocess.Popen", fake_popen)
    monkeypatch.setattr(
        "scarf.utils.process.logger.debug",
        lambda message: logged.append(message),
    )

    system_call("echo hello")

    assert launched == [(["echo", "hello"], {"stdout": subprocess.PIPE})]
    assert logged == [b"hello"]
    assert process.poll.call_count >= 2


def test_process_rss_mb_reads_proc_status(tmp_path, monkeypatch):
    status = tmp_path / "status"
    status.write_text("Name:\tpython\nVmRSS:\t2048 kB\n")
    monkeypatch.setattr("builtins.open", lambda *_a, **_k: status.open())
    assert process_rss_mb() == pytest.approx(2.0)


def test_process_rss_mb_falls_back_when_proc_unavailable(monkeypatch):
    monkeypatch.setattr(
        "builtins.open",
        MagicMock(side_effect=OSError("no /proc")),
    )
    usage = MagicMock()
    usage.ru_maxrss = 4096
    monkeypatch.setattr(
        "scarf.utils.process.resource.getrusage",
        lambda *_a, **_k: usage,
    )
    assert process_rss_mb() == pytest.approx(4.0)


def test_rss_peak_tracker_keeps_running_maximum(monkeypatch):
    readings = iter([10.0, 12.0, 11.0, 15.0, 14.0])
    monkeypatch.setattr(
        "scarf.utils.process.process_rss_mb",
        lambda: next(readings),
    )

    class ImmediateThread:
        def __init__(self, target, daemon=False):
            self._target = target

        def start(self) -> None:
            self._target()

        def join(self, timeout=None) -> None:
            return None

    class ImmediateEvent:
        def __init__(self) -> None:
            self._waits = 0

        def wait(self, timeout=None) -> bool:
            self._waits += 1
            # Allow three sample-loop iterations, then stop.
            return self._waits > 3

        def set(self) -> None:
            return None

    monkeypatch.setattr("scarf.utils.process.threading.Thread", ImmediateThread)
    monkeypatch.setattr("scarf.utils.process.threading.Event", ImmediateEvent)

    with rss_peak_tracker(interval_s=0.01) as peak:
        # Initial 10, then samples 12 / 11 / 15 -> peak 15.
        assert peak() == 15.0

    # Final sample is 14; peak must remain the high-water mark.
    assert peak() == 15.0
