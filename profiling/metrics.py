import math
import os
import resource
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from io import TextIOBase
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self

from scarf.utils.process import (
    read_process_tree_rss_bytes as _read_process_tree_rss_bytes,
)

type PeakScope = Literal["operation", "containerLifetime", "unavailable"]
type LimitValue = int | Literal["max"] | None
type _ReadText = Callable[[Path], str]
type _WriteText = Callable[[Path, str], None]
type _ListPids = Callable[[Path], Iterable[int]]


@dataclass(frozen=True, slots=True)
class ResourceMeasurement:
    operationBaselineBytes: int | None = None
    operationPeakBytes: int | None = None
    operationIncrementalPeakBytes: int | None = None
    operationPeakSource: str | None = None
    processTreeRssBaselineBytes: int | None = None
    processTreeRssPeakBytes: int | None = None
    processTreeRssIncrementalPeakBytes: int | None = None
    processTreeRssAfterBytes: int | None = None
    cgroupMemoryCurrentBaselineBytes: int | None = None
    cgroupMemoryCurrentPeakBytes: int | None = None
    cgroupMemoryCurrentAfterBytes: int | None = None
    cgroupMemoryPeakBytes: int | None = None
    cgroupMemoryPeakScope: PeakScope = "unavailable"
    memoryMaxBytes: LimitValue = None
    memoryEventsDelta: dict[str, int] | None = None
    cpuQuotaCores: float | None = None


@dataclass(frozen=True, slots=True)
class StageTimings:
    inputSetupSeconds: float | None = None
    measuredOperationSeconds: float | None = None
    validationPersistenceSeconds: float | None = None
    wholeFunctionSeconds: float | None = None


@dataclass(frozen=True, slots=True)
class _Sample:
    processTreeRssBytes: int | None
    cgroupMemoryCurrentBytes: int | None


def _default_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _default_write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def _default_list_pids(procRoot: Path) -> list[int]:
    return [
        int(entry.name)
        for entry in procRoot.iterdir()
        if entry.name.isdigit() and entry.is_dir()
    ]


def _optional_read(path: Path, readText: _ReadText) -> str | None:
    try:
        value = readText(path)
    except Exception:
        return None
    return value if isinstance(value, str) else None


def _optional_write(path: Path, value: str, writeText: _WriteText) -> bool:
    try:
        writeText(path, value)
    except Exception:
        return False
    return True


def _nonnegative_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value.strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _limit_value(value: str | None) -> LimitValue:
    if value is None:
        return None
    value = value.strip()
    if value == "max":
        return "max"
    return _nonnegative_int(value)


def _camel_case(value: str) -> str:
    first, *remaining = value.split("_")
    return first + "".join(part[:1].upper() + part[1:] for part in remaining)


def parse_memory_events(value: str) -> dict[str, int]:
    events: dict[str, int] = {}
    for line in value.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        count = _nonnegative_int(parts[1])
        if count is not None:
            events[_camel_case(parts[0])] = count
    return events


def _memory_event_delta(
    before: dict[str, int] | None,
    after: dict[str, int] | None,
) -> dict[str, int] | None:
    if before is None or after is None:
        return None
    return {
        key: after.get(key, 0) - before.get(key, 0)
        for key in sorted(before.keys() | after.keys())
    }


def read_process_tree_rss_bytes(
    rootPid: int,
    *,
    procRoot: str | Path = "/proc",
    readText: _ReadText | None = None,
    listPids: _ListPids | None = None,
) -> int | None:
    return _read_process_tree_rss_bytes(
        rootPid,
        proc_root=procRoot,
        read_text=readText,
        list_pids=listPids,
    )


def _relative_cgroup_path(value: str) -> tuple[str, ...] | None:
    for line in value.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3 or parts[0] != "0" or parts[1]:
            continue
        path_parts = tuple(part for part in parts[2].split("/") if part)
        if any(part in {".", ".."} for part in path_parts):
            return None
        return path_parts
    return None


def _relative_cgroup_v1_memory_path(value: str) -> tuple[str, ...] | None:
    for line in value.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        controllers = {item.strip() for item in parts[1].split(",") if item.strip()}
        if "memory" not in controllers:
            continue
        path_parts = tuple(part for part in parts[2].split("/") if part)
        if any(part in {".", ".."} for part in path_parts):
            return None
        return ("memory",) + path_parts
    return None


_V1_MEMORY_FILES = {
    "memory.current": "memory.usage_in_bytes",
    "memory.peak": "memory.max_usage_in_bytes",
    "memory.max": "memory.limit_in_bytes",
}


class ResourceSampler:
    def __init__(
        self,
        *,
        sampleIntervalSeconds: float = 0.1,
        rootPid: int | None = None,
        procRoot: str | Path = "/proc",
        cgroupRoot: str | Path = "/sys/fs/cgroup",
        cgroupPath: str | Path | None = None,
        resetCgroupPeak: bool = True,
        readText: _ReadText | None = None,
        writeText: _WriteText | None = None,
        listPids: _ListPids | None = None,
    ) -> None:
        if isinstance(sampleIntervalSeconds, bool):
            raise ValueError("sampleIntervalSeconds must be a positive finite number")
        interval = float(sampleIntervalSeconds)
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("sampleIntervalSeconds must be a positive finite number")
        selected_pid = os.getpid() if rootPid is None else rootPid
        if isinstance(selected_pid, bool) or selected_pid <= 0:
            raise ValueError("rootPid must be a positive integer")

        self.sampleIntervalSeconds = interval
        self.rootPid = selected_pid
        self.procRoot = Path(procRoot)
        self.cgroupRoot = Path(cgroupRoot)
        self._explicitCgroupPath = None if cgroupPath is None else Path(cgroupPath)
        self.resetCgroupPeak = resetCgroupPeak
        self._readText = _default_read_text if readText is None else readText
        self._writeText = _default_write_text if writeText is None else writeText
        self._listPids = _default_list_pids if listPids is None else listPids
        self._usesDefaultTextIo = readText is None and writeText is None

        self._stateLock = threading.Lock()
        self._lifecycleLock = threading.Lock()
        self._running = False
        self._generation = 0
        self._stopEvent: threading.Event | None = None
        self._thread: threading.Thread | None = None
        self._peakHandle: TextIOBase | None = None
        self._reset_values()

    def _reset_values(self) -> None:
        self._result: ResourceMeasurement | None = None
        self._sampleCount = 0
        self._cgroupPath: Path | None = None
        self._cgroupVersion: Literal[1, 2] | None = None
        self._processTreeBaselineRssBytes: int | None = None
        self._processTreePeakRssBytes: int | None = None
        self._cgroupMemoryCurrentBaselineBytes: int | None = None
        self._cgroupMemoryCurrentPeakBytes: int | None = None
        self._memoryEventsBefore: dict[str, int] | None = None
        self._memoryMaxBytes: LimitValue = None
        self._cpuQuotaCores: float | None = None
        self._cgroupMemoryPeakInitialBytes: int | None = None
        self._cgroupMemoryPeakScope: PeakScope = "unavailable"

    @property
    def isRunning(self) -> bool:
        with self._stateLock:
            return self._running

    @property
    def sampleCount(self) -> int:
        with self._stateLock:
            return self._sampleCount

    @property
    def result(self) -> ResourceMeasurement | None:
        with self._stateLock:
            return self._result

    def _cgroup_file_name(self, name: str) -> str:
        if self._cgroupVersion == 1:
            return _V1_MEMORY_FILES.get(name, name)
        return name

    def _resolve_cgroup_path(self) -> Path | None:
        if self._explicitCgroupPath is not None:
            # Explicit paths are treated as cgroup v2 unless v1 memory files exist.
            if (
                _optional_read(
                    self._explicitCgroupPath / "memory.usage_in_bytes",
                    self._readText,
                )
                is not None
            ):
                self._cgroupVersion = 1
            else:
                self._cgroupVersion = 2
            return self._explicitCgroupPath
        candidates = (
            self.procRoot / str(self.rootPid) / "cgroup",
            self.procRoot / "self" / "cgroup",
        )
        for candidate in candidates:
            value = _optional_read(candidate, self._readText)
            if value is None:
                continue
            relative = _relative_cgroup_path(value)
            if relative is not None:
                path = self.cgroupRoot.joinpath(*relative)
                if (
                    _optional_read(path / "memory.current", self._readText) is not None
                    or _optional_read(path / "cgroup.controllers", self._readText)
                    is not None
                ):
                    self._cgroupVersion = 2
                    return path
            relative_v1 = _relative_cgroup_v1_memory_path(value)
            if relative_v1 is not None:
                path = self.cgroupRoot.joinpath(*relative_v1)
                if (
                    _optional_read(path / "memory.usage_in_bytes", self._readText)
                    is not None
                ):
                    self._cgroupVersion = 1
                    return path
                flat = self.cgroupRoot / "memory"
                if (
                    _optional_read(flat / "memory.usage_in_bytes", self._readText)
                    is not None
                ):
                    self._cgroupVersion = 1
                    return flat
        probes_v2 = ("cgroup.controllers", "memory.current", "cpu.max")
        if any(
            _optional_read(self.cgroupRoot / name, self._readText) is not None
            for name in probes_v2
        ):
            self._cgroupVersion = 2
            return self.cgroupRoot
        if (
            _optional_read(
                self.cgroupRoot / "memory" / "memory.usage_in_bytes",
                self._readText,
            )
            is not None
        ):
            self._cgroupVersion = 1
            return self.cgroupRoot / "memory"
        return None

    def _read_cgroup_int(self, name: str) -> int | None:
        if self._cgroupPath is None:
            return None
        return _nonnegative_int(
            _optional_read(
                self._cgroupPath / self._cgroup_file_name(name),
                self._readText,
            )
        )

    def _read_cgroup_limit(self, name: str) -> LimitValue:
        if self._cgroupPath is None:
            return None
        return _limit_value(
            _optional_read(
                self._cgroupPath / self._cgroup_file_name(name),
                self._readText,
            )
        )

    def _read_memory_events(self) -> dict[str, int] | None:
        if self._cgroupPath is None or self._cgroupVersion == 1:
            return None
        value = _optional_read(self._cgroupPath / "memory.events", self._readText)
        return None if value is None else parse_memory_events(value)

    def _read_cpu_quota_cores(self) -> float | None:
        if self._cgroupPath is None:
            return None
        value = _optional_read(self._cgroupPath / "cpu.max", self._readText)
        parts = [] if value is None else value.split()
        if len(parts) < 2:
            return None
        quota = _limit_value(parts[0])
        period = _nonnegative_int(parts[1])
        if not isinstance(quota, int) or not period:
            return None
        return quota / period

    @staticmethod
    def _verified_peak_reset(
        initialPeak: int,
        resetPeak: int | None,
        currentBefore: int | None,
        currentAfter: int | None,
    ) -> bool:
        if resetPeak is None or resetPeak >= initialPeak:
            return False
        if resetPeak == 0:
            return True
        current = currentAfter if currentAfter is not None else currentBefore
        return current is None or resetPeak >= current

    @staticmethod
    def _read_peak_handle(handle: TextIOBase) -> int | None:
        try:
            handle.seek(0)
            return _nonnegative_int(handle.read())
        except Exception:
            return None

    @staticmethod
    def _write_peak_handle(handle: TextIOBase, value: str) -> bool:
        try:
            handle.seek(0)
            handle.write(value)
            handle.flush()
            try:
                handle.truncate()
            except OSError:
                pass
        except Exception:
            return False
        return True

    def _peak_reset_values(self, current: int | None) -> tuple[str, ...]:
        if current in (None, 0):
            return ("0",)
        return ("0", str(current))

    def _prepare_cgroup_peak(self) -> tuple[PeakScope, int | None]:
        if self._cgroupPath is None:
            return "unavailable", None
        path = self._cgroupPath / self._cgroup_file_name("memory.peak")
        initial_peak = _nonnegative_int(_optional_read(path, self._readText))
        if initial_peak is None:
            return "unavailable", None
        if not self.resetCgroupPeak:
            return "containerLifetime", initial_peak
        # cgroup v1 max_usage_in_bytes is usually not resettable mid-run.
        if self._cgroupVersion == 1:
            return "containerLifetime", initial_peak
        current_before = self._read_cgroup_int("memory.current")

        if self._usesDefaultTextIo:
            try:
                handle = path.open("r+", encoding="utf-8")
            except Exception:
                return "containerLifetime", initial_peak
            handle_initial = self._read_peak_handle(handle)
            if handle_initial is not None:
                initial_peak = handle_initial
            for reset_value in self._peak_reset_values(current_before):
                if not self._write_peak_handle(handle, reset_value):
                    continue
                current_after = self._read_cgroup_int("memory.current")
                reset_peak = self._read_peak_handle(handle)
                if self._verified_peak_reset(
                    initial_peak,
                    reset_peak,
                    current_before,
                    current_after,
                ):
                    self._peakHandle = handle
                    return "operation", initial_peak
            try:
                handle.close()
            except Exception:
                pass
            return "containerLifetime", initial_peak

        for reset_value in self._peak_reset_values(current_before):
            if not _optional_write(path, reset_value, self._writeText):
                continue
            current_after = self._read_cgroup_int("memory.current")
            reset_peak = _nonnegative_int(_optional_read(path, self._readText))
            if self._verified_peak_reset(
                initial_peak,
                reset_peak,
                current_before,
                current_after,
            ):
                return "operation", initial_peak
        return "containerLifetime", initial_peak

    def _read_final_cgroup_peak(self) -> int | None:
        if self._cgroupMemoryPeakScope == "unavailable":
            return None
        peak_name = self._cgroup_file_name("memory.peak")
        if self._cgroupMemoryPeakScope == "operation":
            if self._peakHandle is not None:
                return self._read_peak_handle(self._peakHandle)
            if self._cgroupPath is None:
                return None
            return _nonnegative_int(
                _optional_read(
                    self._cgroupPath / peak_name,
                    self._readText,
                )
            )
        if self._cgroupPath is None:
            return self._cgroupMemoryPeakInitialBytes
        final_peak = _nonnegative_int(
            _optional_read(self._cgroupPath / peak_name, self._readText)
        )
        if final_peak is None:
            return self._cgroupMemoryPeakInitialBytes
        if self._cgroupMemoryPeakInitialBytes is None:
            return final_peak
        return max(final_peak, self._cgroupMemoryPeakInitialBytes)

    def _close_peak_handle(self) -> None:
        handle, self._peakHandle = self._peakHandle, None
        if handle is None:
            return
        try:
            handle.close()
        except Exception:
            pass

    def _collect_sample(self) -> _Sample:
        process_rss = read_process_tree_rss_bytes(
            self.rootPid,
            procRoot=self.procRoot,
            readText=self._readText,
            listPids=self._listPids,
        )
        return _Sample(
            processTreeRssBytes=process_rss,
            cgroupMemoryCurrentBytes=self._read_cgroup_int("memory.current"),
        )

    def _sample_and_record(
        self,
        *,
        isBaseline: bool,
        force: bool,
        generation: int,
    ) -> _Sample:
        try:
            sample = self._collect_sample()
        except Exception:
            sample = _Sample(None, None)
        with self._stateLock:
            if generation != self._generation:
                return sample
            if not force and not self._running:
                return sample
            self._sampleCount += 1
            if isBaseline:
                self._processTreeBaselineRssBytes = sample.processTreeRssBytes
                self._cgroupMemoryCurrentBaselineBytes = sample.cgroupMemoryCurrentBytes
            if sample.processTreeRssBytes is not None:
                current = self._processTreePeakRssBytes
                self._processTreePeakRssBytes = max(
                    sample.processTreeRssBytes,
                    current if current is not None else sample.processTreeRssBytes,
                )
            if sample.cgroupMemoryCurrentBytes is not None:
                current = self._cgroupMemoryCurrentPeakBytes
                self._cgroupMemoryCurrentPeakBytes = max(
                    sample.cgroupMemoryCurrentBytes,
                    (
                        current
                        if current is not None
                        else sample.cgroupMemoryCurrentBytes
                    ),
                )
        return sample

    def _thread_main(
        self,
        stopEvent: threading.Event,
        readyEvent: threading.Event,
        generation: int,
    ) -> None:
        readyEvent.wait()
        if stopEvent.is_set():
            return
        while not stopEvent.wait(self.sampleIntervalSeconds):
            self._sample_and_record(
                isBaseline=False,
                force=False,
                generation=generation,
            )

    def start(self) -> Self:
        with self._lifecycleLock:
            with self._stateLock:
                if self._running:
                    return self
            self._close_peak_handle()
            with self._stateLock:
                self._reset_values()
                self._generation += 1
                generation = self._generation
                self._running = True
                stop_event = threading.Event()
                self._stopEvent = stop_event
                self._thread = None

            self._cgroupPath = self._resolve_cgroup_path()
            self._memoryEventsBefore = self._read_memory_events()
            self._memoryMaxBytes = self._read_cgroup_limit("memory.max")
            self._cpuQuotaCores = self._read_cpu_quota_cores()

            ready_event = threading.Event()
            thread = threading.Thread(
                target=self._thread_main,
                args=(stop_event, ready_event, generation),
                name=f"resource-sampler-{self.rootPid}",
                daemon=True,
            )
            try:
                thread.start()
            except Exception:
                pass  # Without a thread, only the start and stop samples remain.
            else:
                with self._stateLock:
                    self._thread = thread

            peak_scope, initial_peak = self._prepare_cgroup_peak()
            self._cgroupMemoryPeakScope = peak_scope
            self._cgroupMemoryPeakInitialBytes = initial_peak
            self._sample_and_record(
                isBaseline=True,
                force=True,
                generation=generation,
            )
            ready_event.set()
        return self

    def sample(self) -> None:
        with self._stateLock:
            generation = self._generation
            running = self._running
        if not running:
            return
        self._sample_and_record(
            isBaseline=False,
            force=False,
            generation=generation,
        )

    def _build_result(
        self,
        finalSample: _Sample,
        memoryEventsAfter: dict[str, int] | None,
        cgroupMemoryPeakBytes: int | None,
    ) -> ResourceMeasurement:
        with self._stateLock:
            process_baseline = self._processTreeBaselineRssBytes
            process_peak = self._processTreePeakRssBytes
            cgroup_baseline = self._cgroupMemoryCurrentBaselineBytes
            cgroup_current_peak = self._cgroupMemoryCurrentPeakBytes

        operation_baseline = cgroup_baseline
        operation_peak = cgroup_current_peak
        operation_source: str | None = (
            "cgroupMemoryCurrent" if cgroup_baseline is not None else None
        )
        if (
            cgroup_baseline is not None
            and self._cgroupMemoryPeakScope == "operation"
            and cgroupMemoryPeakBytes is not None
            and (operation_peak is None or cgroupMemoryPeakBytes >= operation_peak)
        ):
            operation_peak = cgroupMemoryPeakBytes
            operation_source = "cgroupMemoryPeak"
        if cgroup_baseline is None:
            operation_baseline = process_baseline
            operation_peak = process_peak
            operation_source = (
                "processTreeRss" if process_baseline is not None else None
            )
        incremental_peak = (
            max(0, operation_peak - operation_baseline)
            if operation_peak is not None and operation_baseline is not None
            else None
        )
        process_incremental_peak = (
            max(0, process_peak - process_baseline)
            if process_peak is not None and process_baseline is not None
            else None
        )

        return ResourceMeasurement(
            operationBaselineBytes=operation_baseline,
            operationPeakBytes=operation_peak,
            operationIncrementalPeakBytes=incremental_peak,
            operationPeakSource=operation_source,
            processTreeRssBaselineBytes=process_baseline,
            processTreeRssPeakBytes=process_peak,
            processTreeRssIncrementalPeakBytes=process_incremental_peak,
            processTreeRssAfterBytes=finalSample.processTreeRssBytes,
            cgroupMemoryCurrentBaselineBytes=cgroup_baseline,
            cgroupMemoryCurrentPeakBytes=cgroup_current_peak,
            cgroupMemoryCurrentAfterBytes=(finalSample.cgroupMemoryCurrentBytes),
            cgroupMemoryPeakBytes=cgroupMemoryPeakBytes,
            cgroupMemoryPeakScope=self._cgroupMemoryPeakScope,
            memoryMaxBytes=self._memoryMaxBytes,
            memoryEventsDelta=_memory_event_delta(
                self._memoryEventsBefore,
                memoryEventsAfter,
            ),
            cpuQuotaCores=self._cpuQuotaCores,
        )

    def stop(self) -> ResourceMeasurement:
        with self._lifecycleLock:
            with self._stateLock:
                if not self._running:
                    if self._result is None:
                        self._result = ResourceMeasurement()
                    return self._result
                self._running = False
                generation = self._generation
                stop_event = self._stopEvent
                thread = self._thread

            if stop_event is not None:
                stop_event.set()
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=max(0.25, min(2.0, self.sampleIntervalSeconds * 2)))

            final_sample = self._sample_and_record(
                isBaseline=False,
                force=True,
                generation=generation,
            )
            memory_events_after = self._read_memory_events()
            cgroup_peak = self._read_final_cgroup_peak()
            self._close_peak_handle()
            result = self._build_result(
                final_sample,
                memory_events_after,
                cgroup_peak,
            )
            with self._stateLock:
                self._result = result
                self._stopEvent = None
                self._thread = None
            return result

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(
        self,
        excType: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        try:
            self.stop()
        except Exception:
            pass
        return False


class StageTimer:
    def __init__(self, *, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._active = False
        self._wholeStarted: float | None = None
        self._wholeElapsed: float | None = None
        self._inputSetupElapsed = 0.0
        self._operationElapsed = 0.0
        self._validationPersistenceElapsed = 0.0
        self._hasInputSetup = False
        self._hasOperation = False
        self._hasValidationPersistence = False

    def __enter__(self) -> Self:
        if self._active:
            raise RuntimeError("StageTimer is already active")
        self._active = True
        self._wholeStarted = self._clock()
        self._wholeElapsed = None
        self._inputSetupElapsed = 0.0
        self._operationElapsed = 0.0
        self._validationPersistenceElapsed = 0.0
        self._hasInputSetup = False
        self._hasOperation = False
        self._hasValidationPersistence = False
        return self

    def __exit__(
        self,
        excType: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if self._wholeStarted is not None:
            self._wholeElapsed = max(0.0, self._clock() - self._wholeStarted)
        self._wholeStarted = None
        self._active = False
        return False

    @contextmanager
    def inputSetup(self) -> Iterator[None]:
        started = self._clock()
        try:
            yield
        finally:
            self._inputSetupElapsed += max(0.0, self._clock() - started)
            self._hasInputSetup = True

    @contextmanager
    def operation(self) -> Iterator[None]:
        started = self._clock()
        try:
            yield
        finally:
            self._operationElapsed += max(0.0, self._clock() - started)
            self._hasOperation = True

    @contextmanager
    def validationPersistence(self) -> Iterator[None]:
        started = self._clock()
        try:
            yield
        finally:
            self._validationPersistenceElapsed += max(
                0.0,
                self._clock() - started,
            )
            self._hasValidationPersistence = True

    @property
    def result(self) -> StageTimings:
        return StageTimings(
            inputSetupSeconds=(
                self._inputSetupElapsed if self._hasInputSetup else None
            ),
            measuredOperationSeconds=(
                self._operationElapsed if self._hasOperation else None
            ),
            validationPersistenceSeconds=(
                self._validationPersistenceElapsed
                if self._hasValidationPersistence
                else None
            ),
            wholeFunctionSeconds=self._wholeElapsed,
        )


def child_cpu_seconds() -> float:
    """Return finished-child user plus system CPU seconds for this process."""
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return float(usage.ru_utime) + float(usage.ru_stime)


def stage_utilization(result: Mapping[str, Any]) -> dict[str, float | None]:
    """Summarize how much of its CPU and memory allowance a stage result used.

    CPU cores are ``(processCpuSeconds + childCpuSeconds) / wholeFunctionSeconds``,
    reported as a percentage of ``modalCpuLimit`` and of ``workers``. Peak cgroup
    and RSS bytes are reported as a percentage of the container memory limit
    (``modalMemoryMb``) and of ``scarfMemoryBudget``. A value is None when an
    input was not recorded.
    """

    def percent(value: float | None, limit: float | None) -> float | None:
        return None if value is None or not limit else round(100 * value / limit, 1)

    wall = result.get("wholeFunctionSeconds")
    cpu = result.get("processCpuSeconds")
    cores = None
    if wall and cpu is not None:
        cores = (cpu + (result.get("childCpuSeconds") or 0.0)) / wall
    container_mb = result.get("modalMemoryMb")
    container = container_mb * 1024**2 if container_mb else None
    budget = result.get("scarfMemoryBudget")
    summary = {
        "cpuCores": None if cores is None else round(cores, 2),
        "cpuPercentOfLimit": percent(cores, result.get("modalCpuLimit")),
        "cpuPercentOfWorkers": percent(cores, result.get("workers")),
    }
    for name in ("peakCgroup", "peakRss"):
        peak = result.get(f"{name}Bytes")
        summary[f"{name}PercentOfContainer"] = percent(peak, container)
        summary[f"{name}PercentOfBudget"] = percent(peak, budget)
    return summary


__all__ = [
    "LimitValue",
    "PeakScope",
    "ResourceMeasurement",
    "ResourceSampler",
    "StageTimer",
    "StageTimings",
    "child_cpu_seconds",
    "parse_memory_events",
    "read_process_tree_rss_bytes",
    "stage_utilization",
]
