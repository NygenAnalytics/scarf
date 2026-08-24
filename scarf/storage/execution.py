"""One planner and one execution report for Scarf storage work."""

import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from ..utils.logging import logger
from .budget import ResourceBudget
from .io_policy import DEFAULT_STORAGE_IO_POLICY, StorageIoPolicy

AUTO_READ_WIDTH_MULTIPLIER = 8

__all__ = [
    "AUTO_READ_WIDTH_MULTIPLIER",
    "ExecutionReport",
    "OperationPlan",
    "WorkShape",
    "auto_read_width",
    "clear_execution_reports",
    "detect_external_thread_caps",
    "execution_report_scope",
    "execution_reports_by_kind",
    "last_execution_report",
    "plan_operation",
    "record_execution_report",
    "recorded_execution_reports",
]

_LAST_REPORT: "ExecutionReport | None" = None
_REPORTS: list["ExecutionReport"] = []
_SCOPES: list[list["ExecutionReport"]] = []


@dataclass(frozen=True, slots=True)
class WorkShape:
    """Independent on-disk units for one operation."""

    nUnits: int
    unitBytes: int
    residentBytes: int = 0
    scratchBytes: int = 0
    decodeBytes: int = 0
    innerReadBytes: int = 0
    maxInnerReads: int | None = None
    ordered: bool = False
    writes: bool = False
    chunksPerShard: int = 1


@dataclass(frozen=True, slots=True)
class OperationPlan:
    """Resolved read, compute, and write limits for one operation."""

    readWorkers: int
    computeWorkers: int
    writeWorkers: int
    threadsPerComputeWorker: int
    ioConcurrency: int
    prefetch: int
    innerReads: int
    unitBytes: int
    reservedBytes: int
    reductionReason: str | None
    requestedWorkers: int
    requestedMemoryBytes: int
    requestedReadWorkers: int | None
    requestedComputeWorkers: int | None
    requestedWriteWorkers: int | None
    chunksPerShard: int
    ordered: bool
    writes: bool

    def as_metrics(self) -> dict[str, Any]:
        return {
            "requestedWorkers": self.requestedWorkers,
            "requestedMemoryBytes": self.requestedMemoryBytes,
            "requestedReadWorkers": self.requestedReadWorkers,
            "requestedComputeWorkers": self.requestedComputeWorkers,
            "requestedWriteWorkers": self.requestedWriteWorkers,
            "effectiveReadWorkers": self.readWorkers,
            "effectiveComputeWorkers": self.computeWorkers,
            "effectiveWriteWorkers": self.writeWorkers,
            "threadsPerComputeWorker": self.threadsPerComputeWorker,
            "ioConcurrency": self.ioConcurrency,
            "prefetch": self.prefetch,
            "innerReads": self.innerReads,
            "unitBytes": self.unitBytes,
            "reservedBytes": self.reservedBytes,
            "reductionReason": self.reductionReason,
            "chunksPerShard": self.chunksPerShard,
            "kind": "plan",
        }


@dataclass(slots=True)
class ExecutionReport:
    """Requested limits, actual mix, and wait times for one operation."""

    plan: OperationPlan
    unitKind: str
    actualReadWorkers: int
    actualComputeWorkers: int
    actualWriteWorkers: int
    fetchSeconds: float = 0.0
    computeSeconds: float = 0.0
    writeSeconds: float = 0.0
    readerWaitSeconds: float = 0.0
    computeWaitSeconds: float = 0.0
    unitsCompleted: int = 0
    peakHeldBytes: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def as_metrics(self) -> dict[str, Any]:
        payload = self.plan.as_metrics()
        payload.update(
            {
                "unitKind": self.unitKind,
                "actualReadWorkers": self.actualReadWorkers,
                "actualComputeWorkers": self.actualComputeWorkers,
                "actualWriteWorkers": self.actualWriteWorkers,
                "fetchSeconds": self.fetchSeconds,
                "computeSeconds": self.computeSeconds,
                "writeSeconds": self.writeSeconds,
                "readerWaitSeconds": self.readerWaitSeconds,
                "computeWaitSeconds": self.computeWaitSeconds,
                "unitsCompleted": self.unitsCompleted,
                "peakHeldBytes": self.peakHeldBytes,
                "kind": "observed",
            }
        )
        payload.update(self.extra)
        return payload

    def log_line(self) -> str:
        reason = self.plan.reductionReason or "full request used"
        return (
            f"execution {self.unitKind}: "
            f"read={self.actualReadWorkers}/{self.plan.readWorkers} "
            f"compute={self.actualComputeWorkers}/{self.plan.computeWorkers} "
            f"write={self.actualWriteWorkers}/{self.plan.writeWorkers} "
            f"fetch={self.fetchSeconds:.2f}s compute={self.computeSeconds:.2f}s "
            f"readerWait={self.readerWaitSeconds:.2f}s "
            f"computeWait={self.computeWaitSeconds:.2f}s "
            f"held={self.peakHeldBytes} reason={reason}"
        )


def detect_external_thread_caps() -> dict[str, int]:
    """Return process environment thread ceilings that can shrink a plan."""
    caps: dict[str, int] = {}
    for name in (
        "NUMBA_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        raw = os.environ.get(name)
        if raw is None or raw == "":
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if value > 0:
            caps[name] = value
    return caps


def last_execution_report() -> ExecutionReport | None:
    """Return the most recent recorded execution report, if any."""
    return _LAST_REPORT


def recorded_execution_reports() -> tuple[ExecutionReport, ...]:
    """Return every report recorded since the last clear."""
    return tuple(_REPORTS)


def clear_execution_reports() -> None:
    """Drop the process-wide report collection."""
    global _LAST_REPORT
    _LAST_REPORT = None
    _REPORTS.clear()


@contextmanager
def execution_report_scope() -> Iterator[list[ExecutionReport]]:
    """Collect reports recorded inside this block, keyed later by unit kind."""
    collected: list[ExecutionReport] = []
    _SCOPES.append(collected)
    try:
        yield collected
    finally:
        if _SCOPES and _SCOPES[-1] is collected:
            _SCOPES.pop()


def execution_reports_by_kind(
    reports: Sequence[ExecutionReport],
) -> dict[str, list[dict[str, Any]]]:
    """Group report metrics by ``unitKind``, preserving record order."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for report in reports:
        grouped.setdefault(report.unitKind, []).append(report.as_metrics())
    return grouped


def record_execution_report(report: ExecutionReport) -> ExecutionReport:
    """Retain the report, append it to active scopes, and emit one log line."""
    global _LAST_REPORT
    if "externalLimits" not in report.extra:
        report.extra["externalLimits"] = detect_external_thread_caps()
    _LAST_REPORT = report
    _REPORTS.append(report)
    for scope in _SCOPES:
        scope.append(report)
    logger.info(report.log_line())
    return report


def _positive_width(value: int | None, *, name: str) -> int | None:
    if value is None:
        return None
    resolved = int(value)
    if resolved < 1:
        raise ValueError(f"{name} must be positive when set")
    return resolved


def auto_read_width(workers: int) -> int:
    """Request eight read lanes per compute worker before memory admission."""
    return max(1, int(workers) * AUTO_READ_WIDTH_MULTIPLIER)


def _nested_read_mix(
    *,
    available: int,
    per_unit: int,
    inner_read: int,
    outer_limit: int,
    inner_limit: int,
    minimum_outer: int,
) -> tuple[int, int]:
    """Maximize bounded nested I/O without starving the compute width."""
    feasible_outer = min(
        max(1, int(outer_limit)),
        available // max(1, per_unit + inner_read),
    )
    if feasible_outer < 1:
        raise MemoryError(
            f"One live unit plus one inner read needs about "
            f"{per_unit + inner_read} bytes, but only {available} bytes are available"
        )
    outer_floor = min(max(1, int(minimum_outer)), feasible_outer)
    resolved_inner_limit = max(1, int(inner_limit))
    candidates = {outer_floor, feasible_outer}
    for inner_reads in range(1, resolved_inner_limit + 1):
        outer = min(
            feasible_outer,
            available // max(1, per_unit + inner_reads * inner_read),
        )
        if outer >= outer_floor:
            candidates.add(outer)

    def resolve(outer: int) -> tuple[int, int]:
        leftover = available - outer * per_unit
        inner_reads = min(
            resolved_inner_limit,
            leftover // max(1, outer * inner_read),
        )
        return outer, max(1, inner_reads)

    return max(
        (resolve(outer) for outer in candidates),
        key=lambda widths: (widths[0] * widths[1], widths[0]),
    )


def plan_operation(
    resources: ResourceBudget,
    shape: WorkShape,
    policy: StorageIoPolicy | None = None,
) -> OperationPlan:
    """Resolve read, compute, and write workers from one budget and policy.

    ``nthreads`` / ``resources.workers`` is compute width. Read width is
    planner-owned and memory-bounded; it is not capped by compute width.
    """
    resolved = policy or DEFAULT_STORAGE_IO_POLICY
    requested_read = _positive_width(resolved.readWorkers, name="readWorkers")
    requested_compute = _positive_width(resolved.computeWorkers, name="computeWorkers")
    requested_write = _positive_width(resolved.writeWorkers, name="writeWorkers")

    n_units = max(0, int(shape.nUnits))
    unit_bytes = max(1, int(shape.unitBytes))
    resident = max(0, int(shape.residentBytes))
    scratch = max(0, int(shape.scratchBytes))
    decode = max(0, int(shape.decodeBytes))
    inner_read = max(0, int(shape.innerReadBytes))
    max_inner_reads = (
        None if shape.maxInnerReads is None else max(1, int(shape.maxInnerReads))
    )
    chunks = max(1, int(shape.chunksPerShard))
    workers = max(1, int(resources.workers))
    available = int(resources.memoryBytes) - resident - scratch
    if available <= 0:
        raise MemoryError(
            f"Resident data needs {resident + scratch} bytes, but the operation "
            f"limit is {resources.memoryBytes} bytes"
        )

    per_unit = unit_bytes + decode
    if per_unit > available:
        raise MemoryError(
            f"One unit needs about {per_unit} bytes in addition to "
            f"{resident + scratch} reserved bytes, but the operation limit is "
            f"{resources.memoryBytes} bytes"
        )
    memory_live = max(1, available // per_unit)
    units = max(1, n_units) if n_units else 1
    read_auto = min(units, memory_live, auto_read_width(workers))

    if shape.ordered:
        compute_auto = 1
    else:
        compute_auto = min(units, workers, memory_live)
    compute_workers = (
        min(requested_compute, units, memory_live, workers)
        if requested_compute is not None
        else compute_auto
    )
    compute_workers = max(1, compute_workers)

    write_auto = min(units, workers, memory_live) if shape.writes else 1
    write_workers = (
        min(requested_write, units, memory_live, workers)
        if requested_write is not None
        else write_auto
    )
    write_workers = max(1, write_workers)

    read_limit = (
        min(requested_read, units, memory_live)
        if requested_read is not None
        else read_auto
    )
    read_limit = max(1, read_limit)

    if inner_read > 0:
        inner_limit = chunks
        if max_inner_reads is not None:
            inner_limit = min(inner_limit, max_inner_reads)
        if requested_read is not None:
            inner_limit = min(inner_limit, requested_read)
        minimum_outer = max(
            compute_workers,
            write_workers if shape.writes else 1,
        )
        read_workers, inner_reads = _nested_read_mix(
            available=available,
            per_unit=per_unit,
            inner_read=inner_read,
            outer_limit=read_limit,
            inner_limit=inner_limit,
            minimum_outer=minimum_outer,
        )
    else:
        read_workers = read_limit
        inner_reads = 1
    if shape.writes:
        write_workers = min(write_workers, read_workers)

    active = max(compute_workers, write_workers if shape.writes else 1, 1)
    threads = max(1, workers // active)
    while active * threads > workers:
        threads -= 1
    threads = max(1, threads)

    io_concurrency = (
        inner_reads if inner_read > 0 else min(chunks, max(1, read_workers))
    )
    prefetch = read_workers

    reasons: list[str] = []
    requested_read_baseline = requested_read or read_auto
    requested_compute_baseline = requested_compute or (1 if shape.ordered else workers)
    requested_write_baseline = requested_write or (workers if shape.writes else 1)
    if read_workers < requested_read_baseline:
        if read_workers == memory_live:
            reasons.append(
                f"{read_workers} readers used because each unit needs {per_unit} bytes"
            )
        elif read_workers == units:
            reasons.append(
                f"{read_workers} readers used because there are {n_units} units"
            )
        elif inner_read > 0:
            reasons.append(
                f"{read_workers} readers used to reserve {inner_reads} inner reads"
            )
        else:
            reasons.append(f"{read_workers} readers used")
    if compute_workers < requested_compute_baseline:
        if shape.ordered:
            reasons.append("1 compute worker used because results accumulate in order")
        elif compute_workers == memory_live:
            reasons.append(
                f"{compute_workers} compute workers used because each unit "
                f"needs {per_unit} bytes"
            )
        elif compute_workers == units:
            reasons.append(
                f"{compute_workers} compute workers used because there are "
                f"{n_units} units"
            )
        else:
            reasons.append(f"{compute_workers} compute workers used")
    if shape.writes and write_workers < requested_write_baseline:
        if write_workers == memory_live:
            reasons.append(
                f"{write_workers} writers used because each unit needs {per_unit} bytes"
            )
        elif write_workers == units:
            reasons.append(
                f"{write_workers} writers used because there are {n_units} units"
            )
        else:
            reasons.append(f"{write_workers} writers used")
    reduction = "; ".join(reasons) if reasons else None

    live_slots = read_workers
    if shape.writes:
        live_slots = max(live_slots, write_workers)
    reserved = (
        resident
        + scratch
        + live_slots * per_unit
        + read_workers * inner_read * inner_reads
    )
    while reserved > int(resources.memoryBytes) and inner_reads > 1:
        inner_reads -= 1
        reserved = (
            resident
            + scratch
            + live_slots * per_unit
            + read_workers * inner_read * inner_reads
        )
    if reserved > int(resources.memoryBytes):
        raise MemoryError(
            f"Planned reservation is {reserved} bytes, but the operation "
            f"limit is {resources.memoryBytes} bytes"
        )
    return OperationPlan(
        readWorkers=read_workers,
        computeWorkers=compute_workers,
        writeWorkers=write_workers,
        threadsPerComputeWorker=threads,
        ioConcurrency=max(1, io_concurrency),
        prefetch=max(1, prefetch),
        innerReads=max(1, inner_reads),
        unitBytes=per_unit,
        reservedBytes=reserved,
        reductionReason=reduction,
        requestedWorkers=workers,
        requestedMemoryBytes=int(resources.memoryBytes),
        requestedReadWorkers=requested_read,
        requestedComputeWorkers=requested_compute,
        requestedWriteWorkers=requested_write,
        chunksPerShard=chunks,
        ordered=bool(shape.ordered),
        writes=bool(shape.writes),
    )
