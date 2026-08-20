"""Resource limits used by one storage or analysis operation."""

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

_FALLBACK_MEMORY_BYTES = 8 * 1024 * 1024 * 1024
_SUFFIXES = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
_MIN_RAW_BYTES = 1024 * 1024
_CGROUP_UNLIMITED_THRESHOLD = 1 << 60


@dataclass(frozen=True)
class ResourceBudget:
    """Memory and CPU limits for one operation.

    Attributes:
        memoryBytes: Planning budget in bytes. Not a hard process RSS cap.
        workers: Maximum compute-worker budget.
    """

    memoryBytes: int
    workers: int


def _read_int(path: str) -> int | None:
    try:
        value = Path(path).read_text().strip()
    except OSError:
        return None
    if not value or value == "max":
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    if parsed <= 0 or parsed >= _CGROUP_UNLIMITED_THRESHOLD:
        return None
    return parsed


def _physical_memory_bytes() -> int:
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        total_pages = os.sysconf("SC_PHYS_PAGES")
        if page_size > 0 and total_pages > 0:
            return int(page_size) * int(total_pages)
    except (ValueError, OSError, AttributeError):
        pass
    return _FALLBACK_MEMORY_BYTES


def _process_cgroup_entry(controller: str) -> tuple[list[str], str] | None:
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        controllers = fields[1].split(",") if fields[1] else []
        if (not controllers and controller == "") or controller in controllers:
            return controllers, fields[2].lstrip("/")
    return None


def _process_cgroup_path(controller: str) -> str | None:
    entry = _process_cgroup_entry(controller)
    return None if entry is None else entry[1]


def _unified_cgroup_files(name: str) -> list[str]:
    path = _process_cgroup_path("")
    parts = Path(path).parts if path else ()
    candidates = [
        str(Path("/sys/fs/cgroup", *parts[:depth], name))
        for depth in range(len(parts), -1, -1)
    ]
    return list(dict.fromkeys(candidates))


def _legacy_cgroup_files(controller: str, name: str) -> list[str]:
    entry = _process_cgroup_entry(controller)
    controllers, path = ([], "") if entry is None else entry
    parts = Path(path).parts if path else ()
    mount_names = list(dict.fromkeys([",".join(controllers), controller]))
    candidates = [
        str(Path("/sys/fs/cgroup", mount_name, *parts[:depth], name))
        for mount_name in mount_names
        if mount_name
        for depth in range(len(parts), -1, -1)
    ]
    return list(dict.fromkeys(candidates))


def _cgroup_memory_bytes() -> int | None:
    paths = _unified_cgroup_files("memory.max")
    paths.extend(_legacy_cgroup_files("memory", "memory.limit_in_bytes"))
    limits = [value for path in paths if (value := _read_int(path)) is not None]
    return min(limits) if limits else None


def detect_total_memory_bytes() -> int:
    """Return the effective physical or cgroup memory limit."""
    physical = _physical_memory_bytes()
    cgroup = _cgroup_memory_bytes()
    return physical if cgroup is None else min(physical, cgroup)


def _cpu_quota_workers() -> int | None:
    limits: list[int] = []
    for path in _unified_cgroup_files("cpu.max"):
        try:
            quota_text, period_text = Path(path).read_text().strip().split()
        except (OSError, ValueError):
            continue
        if quota_text != "max":
            try:
                quota = int(quota_text)
                period = int(period_text)
            except ValueError:
                continue
            if quota > 0 and period > 0:
                limits.append(max(1, quota // period))

    for legacy_quota_path in _legacy_cgroup_files("cpu", "cpu.cfs_quota_us"):
        legacy_period_path = str(Path(legacy_quota_path).with_name("cpu.cfs_period_us"))
        legacy_quota = _read_int(legacy_quota_path)
        legacy_period = _read_int(legacy_period_path)
        if legacy_quota is not None and legacy_period is not None:
            limits.append(max(1, legacy_quota // legacy_period))
    return min(limits) if limits else None


def detect_workers() -> int:
    limits = [max(1, os.cpu_count() or 1)]
    try:
        limits.append(max(1, len(os.sched_getaffinity(0))))
    except (AttributeError, OSError):
        pass
    quota = _cpu_quota_workers()
    if quota is not None:
        limits.append(quota)
    return min(limits)


def _parse_memory_spec(spec: str | int | float) -> int:
    guidance = (
        "Use raw bytes, a suffixed size like '8G'/'512M', "
        "or a fraction strictly between 0 and 1 like '0.6'."
    )
    if isinstance(spec, bool):
        raise ValueError(f"Invalid memory spec: {spec!r}. {guidance}")
    if isinstance(spec, int):
        if spec <= 0:
            raise ValueError(f"Memory budget must be positive, got {spec!r}.")
        return spec

    text = str(spec).strip()
    if not text:
        raise ValueError("Empty memory spec. " + guidance)
    suffix = text[-1].upper()
    if suffix in _SUFFIXES:
        try:
            value = float(text[:-1])
        except ValueError as exc:
            raise ValueError(f"Invalid memory spec: {spec!r}. {guidance}") from exc
        if value <= 0:
            raise ValueError(f"Memory budget must be positive, got {spec!r}.")
        return max(1, int(value * _SUFFIXES[suffix]))

    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(f"Invalid memory spec: {spec!r}. {guidance}") from exc
    if number <= 0:
        raise ValueError(f"Memory budget must be positive, got {spec!r}.")
    if number < 1:
        return max(1, int(number * detect_total_memory_bytes()))
    if number < _MIN_RAW_BYTES:
        raise ValueError(
            f"Ambiguous memory spec {spec!r}: a bare number >= 1 is read as bytes, "
            f"which is implausibly small. {guidance}"
        )
    return int(number)


def resolve_budget(
    memory: int | str | None = None,
    workers: int | None = None,
) -> ResourceBudget:
    """Build a :class:`ResourceBudget`, auto-detecting unset fields."""
    if memory is None:
        env_mem = os.environ.get("SCARF_MEM_BUDGET")
        if env_mem:
            memory_bytes = _parse_memory_spec(env_mem)
        else:
            memory_bytes = detect_total_memory_bytes()
    else:
        memory_bytes = _parse_memory_spec(memory)

    if workers is None:
        env_workers = os.environ.get("SCARF_WORKERS")
        if env_workers:
            try:
                worker_count = int(env_workers)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid SCARF_WORKERS={env_workers!r}; expected an integer."
                ) from exc
        else:
            worker_count = detect_workers()
    else:
        worker_count = int(workers)

    return ResourceBudget(
        memoryBytes=max(1, memory_bytes),
        workers=max(1, worker_count),
    )


def admitted_worker_count(
    resources: ResourceBudget,
    *,
    taskBytes: int,
    residentBytes: int = 0,
    requested: int | None = None,
) -> int:
    """Bound concurrent tasks by CPU and live byte estimates."""
    workers, _ = admitted_worker_split(
        resources,
        nTasks=resources.workers if requested is None else requested,
        taskBytes=lambda _: taskBytes,
        residentBytes=residentBytes,
        requested=requested,
    )
    return workers


def admitted_worker_split(
    resources: ResourceBudget,
    *,
    nTasks: int,
    taskBytes: Callable[[int], int],
    residentBytes: int = 0,
    requested: int | None = None,
) -> tuple[int, int]:
    """Split worker slots between outer tasks and each task's inner work."""
    cpu = min(
        resources.workers,
        resources.workers if requested is None else max(1, int(requested)),
    )
    tasks = max(1, int(nTasks))
    resident = max(0, int(residentBytes))
    available = resources.memoryBytes - resident
    if available <= 0:
        raise MemoryError(
            f"Resident data needs about {resident} bytes, but the operation "
            f"limit is {resources.memoryBytes} bytes"
        )

    for outer in range(min(cpu, tasks), 0, -1):
        for inner in range(max(1, cpu // outer), 0, -1):
            per_task = max(1, int(taskBytes(inner)))
            if outer * per_task <= available:
                return outer, inner

    one_task = max(1, int(taskBytes(1)))
    raise MemoryError(
        f"One task needs about {one_task} bytes in addition to {resident} "
        f"resident bytes, but the operation limit is {resources.memoryBytes} bytes"
    )


DEFAULT_READ_AHEAD_BLOCKS = 2
"""Read depth a public streaming API uses when the caller has not planned one."""


@dataclass(frozen=True, slots=True)
class StreamAdmission:
    """Read depth and per-read concurrency admitted for one block stream."""

    outerWorkers: int
    ioConcurrency: int


def admit_stream(
    resources: ResourceBudget,
    *,
    nBlocks: int,
    blockBytes: int,
    decodeBytes: int = 0,
    residentBytes: int = 0,
    requested: int | None = None,
) -> StreamAdmission:
    """Admit read depth for a block stream that also decodes stored chunks.

    Each in-flight read owns its block buffer plus as many decoded chunks as its
    own concurrency allows, so ``decodeBytes`` is charged per concurrent decode
    rather than once per read.
    """
    block = max(1, int(blockBytes))
    decode = max(0, int(decodeBytes))
    outer, inner = admitted_worker_split(
        resources,
        nTasks=nBlocks,
        taskBytes=lambda concurrency: block + concurrency * decode,
        residentBytes=residentBytes,
        requested=requested,
    )
    return StreamAdmission(outerWorkers=outer, ioConcurrency=inner)
