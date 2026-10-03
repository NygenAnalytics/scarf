"""Behavioral tests for process helpers."""

import os
import io
import sys
from unittest.mock import MagicMock
from pathlib import Path

import pytest

from scarf.utils.process import (
    process_rss_mb,
    read_process_tree_rss_bytes,
    sample_process_tree_rss,
    suppress_native_output,
)


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


def test_process_tree_rss_sums_only_root_and_descendants(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    statuses = {
        100: "PPid:\t1\nVmRSS:\t10 kB\n",
        101: "PPid:\t100\nVmRSS:\t20 kB\n",
        102: "PPid:\t101\nVmRSS:\t30 kB\n",
        200: "PPid:\t1\nVmRSS:\t999 kB\n",
    }
    for pid, contents in statuses.items():
        directory = proc_root / str(pid)
        directory.mkdir(parents=True)
        (directory / "status").write_text(contents)

    assert read_process_tree_rss_bytes(100, proc_root=proc_root) == 60 * 1024


def test_process_tree_rss_skips_malformed_status_lines() -> None:
    statuses = {
        100: "Name:\tpython\nno separator here\nPPid:\t1\nVmRSS:\t10 kB\n",
        # A parent that is not a number and an unreadable amount are ignored.
        101: "PPid:\tnot-a-pid\nVmRSS:\tlots kB\n",
        102: "PPid:\t100\nVmRSS:\t2 mB\nVmRSS:\n",
        103: "PPid:\t100\nVmRSS:\t5 pages\n",
    }

    def read_text(path: Path) -> str:
        return statuses[int(path.parent.name)]

    total = read_process_tree_rss_bytes(
        100, read_text=read_text, list_pids=lambda _root: list(statuses)
    )
    # 10 kB of the root and 2 MiB of its child; 103 reports no known unit.
    assert total == 10 * 1024 + 2 * 1024**2
    assert (
        read_process_tree_rss_bytes(
            101, read_text=read_text, list_pids=lambda _root: list(statuses)
        )
        is None
    )


def test_process_tree_rss_survives_cycles_and_unreadable_roots() -> None:
    # Each process names the other as its parent, as a recycled PID can.
    statuses = {
        100: "PPid:\t101\nVmRSS:\t1 kB\n",
        101: "PPid:\t100\nVmRSS:\t2 kB\n",
    }

    def read_text(path: Path) -> str:
        pid = int(path.parent.name)
        if pid not in statuses:
            raise OSError("process exited")
        return statuses[pid]

    def listing(_root: Path) -> list[int]:
        return [100, 101]

    assert (
        read_process_tree_rss_bytes(100, read_text=read_text, list_pids=listing)
        == 3 * 1024
    )
    # A root that exited has no record, so nothing is measured.
    assert (
        read_process_tree_rss_bytes(555, read_text=read_text, list_pids=listing) is None
    )
    for invalid in (0, -4, True, "100"):
        assert (
            read_process_tree_rss_bytes(invalid, read_text=read_text, list_pids=listing)  # type: ignore[arg-type]
            is None
        )


def test_process_tree_sampler_reports_unavailable_measurements() -> None:
    # The sampling interval outlasts the test, so exactly the first and the
    # closing samples are taken.
    with sample_process_tree_rss(
        interval_seconds=3600.0,
        root_pid=123,
        reader=lambda _pid: None,
    ) as measurement:
        observed = measurement()

    assert observed.baseline_bytes is None
    assert observed.peak_bytes is None
    assert observed.incremental_peak_bytes is None
    assert observed.sample_count == observed.sampling_error_count == 1
    assert observed.sample_interval_seconds == 3600.0
    assert observed.unavailable_reason == "process-tree RSS is unavailable"
    final = measurement()
    assert final.sample_count == final.sampling_error_count == 2


def test_process_tree_sampler_preserves_baseline_and_sampled_peak() -> None:
    readings = iter((100, 140))

    with sample_process_tree_rss(
        interval_seconds=3600.0,
        root_pid=123,
        reader=lambda _pid: next(readings),
    ) as measurement:
        first = measurement()
        assert first.baseline_bytes == 100
        assert first.peak_bytes == 100

    final = measurement()
    assert final.baseline_bytes == 100
    assert final.peak_bytes == 140
    assert final.incremental_peak_bytes == 40
    assert (final.sample_count, final.sampling_error_count) == (2, 0)
    assert final.unavailable_reason is None


@pytest.mark.parametrize("bad_reading", [RuntimeError("proc unavailable"), -1, True])
def test_process_tree_sampler_counts_failed_and_invalid_readings(bad_reading) -> None:
    readings = iter((bad_reading, 300))

    def reader(_pid: int) -> int:
        reading = next(readings)
        if isinstance(reading, Exception):
            raise reading
        return reading

    with sample_process_tree_rss(
        interval_seconds=3600.0, root_pid=123, reader=reader
    ) as measurement:
        pass

    final = measurement()
    # The failed first reading leaves the baseline to the next valid one.
    assert (final.sample_count, final.sampling_error_count) == (2, 1)
    assert (final.baseline_bytes, final.peak_bytes) == (300, 300)
    assert final.incremental_peak_bytes == 0


@pytest.mark.parametrize("interval", [0.0, -1.0])
def test_process_tree_sampler_requires_a_positive_interval(interval) -> None:
    with pytest.raises(ValueError, match="interval_seconds must be positive"):
        with sample_process_tree_rss(interval_seconds=interval):
            pass


def _standard_descriptors() -> dict[int, tuple[int, int]]:
    """Return the device and inode behind standard output and error."""
    return {fd: (os.fstat(fd).st_dev, os.fstat(fd).st_ino) for fd in (1, 2)}


def _null_descriptors() -> dict[int, tuple[int, int]]:
    """Return what ``_standard_descriptors`` reports while both are discarded."""
    status = os.stat(os.devnull)
    return {fd: (status.st_dev, status.st_ino) for fd in (1, 2)}


def test_suppress_native_output_discards_descriptor_writes_and_restores(capfd):
    import ctypes

    libc = ctypes.CDLL(None)
    before = _standard_descriptors()
    # capfd replaces sys.stdout, so the original streams exercise the flushes
    # that keep buffered text on the correct side of the redirection.
    stdout = sys.__stdout__
    stderr = sys.__stderr__
    assert stdout is not None and stderr is not None
    stdout.write("python-before\n")

    with suppress_native_output():
        assert _standard_descriptors() == _null_descriptors()
        stdout.write("python-inside\n")
        stderr.write("python-error-inside\n")
        libc.printf(b"native-inside\n")
        os.write(1, b"descriptor-inside\n")
        os.write(2, b"descriptor-error-inside\n")

    assert _standard_descriptors() == before
    with open(os.devnull, "w") as later:
        assert later.fileno() > 2
    os.write(1, b"descriptor-after\n")
    captured = capfd.readouterr()
    assert captured.out == "python-before\ndescriptor-after\n"
    assert captured.err == ""


def test_suppress_native_output_restores_descriptors_after_errors(capfd):
    before = _standard_descriptors()

    with pytest.raises(RuntimeError, match="native failure"):
        with suppress_native_output():
            os.write(2, b"hidden\n")
            raise RuntimeError("native failure")

    assert _standard_descriptors() == before
    os.write(2, b"visible\n")
    assert capfd.readouterr().err == "visible\n"


@pytest.mark.parametrize("missing_libc", [False, True])
def test_suppress_native_output_handles_closed_streams(
    capfd, monkeypatch, missing_libc
):
    import ctypes

    stream = io.TextIOWrapper(io.BytesIO())
    stream.close()
    before = _standard_descriptors()

    def unavailable_libc(*_args):
        raise OSError("C standard library unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(sys, "stdout", stream)
        patch.setattr(sys, "stderr", stream)
        patch.setattr(sys, "__stdout__", None)
        patch.setattr(sys, "__stderr__", None)
        if missing_libc:
            patch.setattr(ctypes, "CDLL", unavailable_libc)
        with suppress_native_output():
            assert _standard_descriptors() == _null_descriptors()
            os.write(1, b"hidden\n")
            os.write(2, b"hidden error\n")

    assert _standard_descriptors() == before
    os.write(1, b"visible\n")
    os.write(2, b"visible error\n")
    captured = capfd.readouterr()
    assert captured.out == "visible\n"
    assert captured.err == "visible error\n"
