import resource
import shlex
import subprocess
import threading
import os
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .logging import logger


type _ReadText = Callable[[Path], str]
type _ListPids = Callable[[Path], Iterable[int]]


def _default_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _default_list_pids(proc_root: Path) -> list[int]:
    return [
        int(entry.name)
        for entry in proc_root.iterdir()
        if entry.name.isdigit() and entry.is_dir()
    ]


def _nonnegative_int(value: str) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _optional_read(path: Path, reader: _ReadText) -> str | None:
    try:
        return reader(path)
    except (OSError, RuntimeError):
        return None


def _parse_proc_status(value: str) -> tuple[int | None, int | None]:
    parent_pid: int | None = None
    rss_bytes: int | None = None
    for line in value.splitlines():
        key, separator, raw_value = line.partition(":")
        if not separator:
            continue
        parts = raw_value.split()
        if not parts:
            continue
        if key == "PPid":
            parent_pid = _nonnegative_int(parts[0])
        elif key == "VmRSS":
            amount = _nonnegative_int(parts[0])
            if amount is None:
                continue
            unit = parts[1].lower() if len(parts) > 1 else "b"
            scale = {"b": 1, "kb": 1024, "mb": 1024**2}.get(unit)
            if scale is not None:
                rss_bytes = amount * scale
    return parent_pid, rss_bytes


def read_process_tree_rss_bytes(
    root_pid: int,
    *,
    proc_root: str | Path = "/proc",
    read_text: _ReadText | None = None,
    list_pids: _ListPids | None = None,
) -> int | None:
    """Return sampled RSS for one process and all discoverable descendants."""

    if isinstance(root_pid, bool) or not isinstance(root_pid, int) or root_pid <= 0:
        return None
    resolved_root = Path(proc_root)
    text_reader = _default_read_text if read_text is None else read_text
    pid_reader = _default_list_pids if list_pids is None else list_pids
    try:
        pids = {int(pid) for pid in pid_reader(resolved_root)}
    except Exception:
        return None
    pids.add(root_pid)

    records: dict[int, tuple[int | None, int | None]] = {}
    children: dict[int, set[int]] = {}
    for pid in pids:
        status = _optional_read(resolved_root / str(pid) / "status", text_reader)
        if status is None:
            continue
        parent_pid, rss_bytes = _parse_proc_status(status)
        records[pid] = (parent_pid, rss_bytes)
        if parent_pid is not None:
            children.setdefault(parent_pid, set()).add(pid)

    tree_pids: set[int] = set()
    pending = [root_pid]
    while pending:
        pid = pending.pop()
        if pid in tree_pids:
            continue
        tree_pids.add(pid)
        pending.extend(children.get(pid, ()))

    values: list[int] = []
    for pid in tree_pids:
        if pid not in records:
            continue
        value = records[pid][1]
        if value is not None:
            values.append(value)
    return None if not values else sum(values)


@dataclass(frozen=True, slots=True)
class ProcessTreeRssMeasurement:
    baseline_bytes: int | None
    peak_bytes: int | None
    incremental_peak_bytes: int | None
    sample_interval_seconds: float
    sample_count: int
    sampling_error_count: int
    unavailable_reason: str | None

    def to_stage_metrics(self) -> dict[str, int | float | str | None]:
        return {
            "rssBaselineBytes": self.baseline_bytes,
            "rssPeakBytes": self.peak_bytes,
            "rssIncrementalPeakBytes": self.incremental_peak_bytes,
            "sampleIntervalSeconds": self.sample_interval_seconds,
            "sampleCount": self.sample_count,
            "samplingErrorCount": self.sampling_error_count,
            "rssUnavailableReason": self.unavailable_reason,
        }


@contextmanager
def sample_process_tree_rss(
    *,
    interval_seconds: float = 0.1,
    root_pid: int | None = None,
    reader: Callable[[int], int | None] = read_process_tree_rss_bytes,
) -> Iterator[Callable[[], ProcessTreeRssMeasurement]]:
    """Sample process-tree RSS; reported peaks are lower-bound observations."""

    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    pid = os.getpid() if root_pid is None else root_pid
    values: list[int] = []
    sample_count = 0
    error_count = 0
    lock = threading.Lock()
    stop = threading.Event()

    def sample() -> None:
        nonlocal sample_count, error_count
        try:
            value = reader(pid)
        except Exception:
            value = None
        with lock:
            sample_count += 1
            if value is None:
                error_count += 1
            elif isinstance(value, bool) or value < 0:
                error_count += 1
            else:
                values.append(int(value))

    sample()

    def sample_loop() -> None:
        while not stop.wait(interval_seconds):
            sample()

    thread = threading.Thread(
        target=sample_loop,
        name="scarf-pipeline-rss",
        daemon=True,
    )
    thread.start()

    def measurement() -> ProcessTreeRssMeasurement:
        with lock:
            baseline = values[0] if values else None
            peak = max(values) if values else None
            count = sample_count
            errors = error_count
        return ProcessTreeRssMeasurement(
            baseline_bytes=baseline,
            peak_bytes=peak,
            incremental_peak_bytes=(
                None if baseline is None or peak is None else max(0, peak - baseline)
            ),
            sample_interval_seconds=float(interval_seconds),
            sample_count=count,
            sampling_error_count=errors,
            unavailable_reason=(
                None if peak is not None else "process-tree RSS is unavailable"
            ),
        )

    try:
        yield measurement
    finally:
        stop.set()
        thread.join(timeout=max(2.0, interval_seconds * 4))
        sample()


def system_call(command: str) -> None:
    """Run a command and forward its output to Scarf's logger."""
    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)
    while True:
        output = process.stdout.readline()  # type: ignore[union-attr]
        if process.poll() is not None:
            break
        if output:
            logger.debug(output.strip())
    process.poll()


def process_rss_mb() -> float:
    """Return this process's resident memory in MiB."""
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


@contextmanager
def rss_peak_tracker(
    interval_s: float = 0.25,
) -> Iterator[Callable[[], float]]:
    """Track peak resident memory while the context is active."""
    peak = process_rss_mb()
    stop = threading.Event()

    def sample_loop() -> None:
        nonlocal peak
        while not stop.wait(interval_s):
            peak = max(peak, process_rss_mb())

    thread = threading.Thread(target=sample_loop, daemon=True)
    thread.start()
    try:
        yield lambda: peak
    finally:
        stop.set()
        thread.join(timeout=2.0)
        peak = max(peak, process_rss_mb())
