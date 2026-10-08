"""Behavioral tests for process helpers."""

import os
import io
import sys
import threading
from pathlib import Path

import pytest

import scarf.utils.process as process_module
from scarf.utils.process import (
    process_rss_mb,
    read_process_tree_rss_bytes,
    rss_text,
    sample_process_tree_rss,
    suppress_native_output,
)


def _fake_proc_root(tmp_path: Path, status: bytes) -> Path:
    proc_root = tmp_path / "proc"
    (proc_root / "self").mkdir(parents=True)
    (proc_root / "self" / "status").write_bytes(status)
    return proc_root


def test_process_rss_mb_reads_proc_status(tmp_path, monkeypatch):
    # A process name truncated inside a multibyte character is not UTF-8.
    proc_root = _fake_proc_root(tmp_path, b"Name:\tpy\xe6\x97\nVmRSS:\t2048 kB\n")
    monkeypatch.setattr(process_module, "_PROC_ROOT", proc_root)
    assert process_rss_mb() == pytest.approx(2.0)
    assert rss_text() == "2 MiB"


@pytest.mark.parametrize("status", [None, b"Name:\tpython\nVmSize:\t2048 kB\n"])
def test_process_rss_mb_is_none_without_proc_vmrss(tmp_path, monkeypatch, status):
    # macOS and Windows have no /proc. Scarf reports nothing rather than the
    # peak resident size from getrusage, which macOS reports in bytes.
    proc_root = (
        tmp_path / "missing" if status is None else _fake_proc_root(tmp_path, status)
    )
    monkeypatch.setattr(process_module, "_PROC_ROOT", proc_root)
    assert process_rss_mb() is None
    assert rss_text() == "n/a"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux reports VmRSS in /proc")
def test_process_rss_mb_reports_current_resident_memory_on_linux():
    rss = process_rss_mb()
    assert rss is not None and rss > 0
    amount, unit = rss_text().split(" ")
    assert unit == "MiB" and int(amount) > 0


def test_rna_feature_stats_log_rss_without_proc(tmp_path, monkeypatch):
    import numpy as np

    from scarf.assay import RNAassay
    from scarf.metadata import MetaData
    from scarf.utils.logging import logger
    from tests.test_counts_t import _memory_root, _write_small_assay

    root = _memory_root()
    values = np.array([[4, 0, 1], [3, 2, 1], [0, 5, 1]], dtype=np.uint32)
    _write_small_assay(root, workspace=None, values=values)
    cells = MetaData(root["cellData"])
    cells.insert("RNA_nCounts", values.sum(axis=1).astype(np.float64), overwrite=True)
    assay = RNAassay(root, "RNA", cells, workspace=None, nthreads=1)
    assay.sf = 1000
    monkeypatch.setattr(process_module, "_PROC_ROOT", tmp_path / "missing")
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(message), level="DEBUG")
    try:
        assay._streaming_feature_stats(np.arange(3), np.arange(3))
    finally:
        logger.remove(sink)

    bands = [message for message in messages if "feature stats band" in message]
    assert bands
    assert all(message.rstrip().endswith("rss n/a") for message in bands), bands


def test_process_tree_sampler_starts_no_thread_without_proc(tmp_path, monkeypatch):
    monkeypatch.setattr(process_module, "_PROC_ROOT", tmp_path / "missing")
    started: list[str] = []
    real_start = threading.Thread.start

    def recording_start(thread: threading.Thread) -> None:
        started.append(thread.name)
        real_start(thread)

    monkeypatch.setattr(threading.Thread, "start", recording_start)
    with sample_process_tree_rss(interval_seconds=0.01) as measurement:
        during = measurement()
    final = measurement()

    assert "scarf-pipeline-rss" not in started
    for observed in (during, final):
        assert (observed.baseline_bytes, observed.peak_bytes) == (None, None)
        assert observed.incremental_peak_bytes is None
        assert (observed.sample_count, observed.sampling_error_count) == (0, 0)
        assert observed.sample_interval_seconds == 0.01
        assert observed.unavailable_reason == (
            "process-tree RSS requires the Linux /proc filesystem"
        )


def test_process_tree_sampler_reads_the_patched_proc_root(tmp_path, monkeypatch):
    proc_root = _fake_proc_root(tmp_path, b"PPid:\t1\nVmRSS:\t4 kB\n")
    pid_status = proc_root / str(os.getpid()) / "status"
    pid_status.parent.mkdir()
    pid_status.write_bytes(b"PPid:\t1\nVmRSS:\t8 kB\n")
    monkeypatch.setattr(process_module, "_PROC_ROOT", proc_root)

    with sample_process_tree_rss(interval_seconds=3600.0) as measurement:
        pass

    final = measurement()
    assert (final.baseline_bytes, final.peak_bytes) == (8 * 1024, 8 * 1024)
    assert (final.sample_count, final.sampling_error_count) == (2, 0)
    assert final.unavailable_reason is None


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
