"""Process-wide resource budget for predictable memory use.

A single :class:`ResourceBudget` (total memory bytes, worker count, working
copies) is the source of truth. Write-time chunk and shard geometry is derived
from ``memoryBytes // workingCopies``; ``workers`` sets read concurrency and
async IO parallelism. Once files are written, reads follow the on-disk chunk
and shard geometry with no additional memory heuristics.
"""

import os
from dataclasses import dataclass

_DEFAULT_MEMORY_FRACTION = 0.6
_DEFAULT_WORKING_COPIES = 4
_FALLBACK_MEMORY_BYTES = 8 * 1024 * 1024 * 1024
_SUFFIXES = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
_MIN_RAW_BYTES = 1024 * 1024


@dataclass(frozen=True)
class ResourceBudget:
    """Memory and worker budget for write-time geometry and read concurrency."""

    memoryBytes: int
    workers: int
    workingCopies: int = _DEFAULT_WORKING_COPIES


def detect_available_memory_bytes() -> int:
    """Best-effort available system memory in bytes."""
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        avail_pages = os.sysconf("SC_AVPHYS_PAGES")
        if page_size > 0 and avail_pages > 0:
            return int(page_size) * int(avail_pages)
    except (ValueError, OSError, AttributeError):
        pass
    return _FALLBACK_MEMORY_BYTES


def detect_workers() -> int:
    return max(1, os.cpu_count() or 1)


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
        return max(1, int(number * detect_available_memory_bytes()))
    if number < _MIN_RAW_BYTES:
        raise ValueError(
            f"Ambiguous memory spec {spec!r}: a bare number >= 1 is read as bytes, "
            f"which is implausibly small. {guidance}"
        )
    return int(number)


def _resolve_working_copies(working_copies: int | None = None) -> int:
    if working_copies is not None:
        copies = int(working_copies)
        if copies <= 0:
            raise ValueError(f"workingCopies must be positive, got {copies!r}.")
        return copies
    env = os.environ.get("SCARF_WORKING_COPIES")
    if env:
        try:
            copies = int(env)
        except ValueError as exc:
            raise ValueError(
                f"Invalid SCARF_WORKING_COPIES={env!r}; expected an integer."
            ) from exc
        if copies <= 0:
            raise ValueError(f"SCARF_WORKING_COPIES must be positive, got {copies!r}.")
        return copies
    return _DEFAULT_WORKING_COPIES


def resolve_budget(
    memory: int | str | None = None,
    workers: int | None = None,
    *,
    working_copies: int | None = None,
) -> ResourceBudget:
    """Build a :class:`ResourceBudget`, auto-detecting unset fields."""
    if memory is None:
        env_mem = os.environ.get("SCARF_MEM_BUDGET")
        if env_mem:
            memory_bytes = _parse_memory_spec(env_mem)
        else:
            memory_bytes = int(
                detect_available_memory_bytes() * _DEFAULT_MEMORY_FRACTION
            )
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
        workingCopies=_resolve_working_copies(working_copies),
    )


_activeBudget: ResourceBudget | None = None
_cachedDefault: ResourceBudget | None = None


def set_resource_budget(budget: ResourceBudget | None) -> None:
    """Install the process-wide active budget (None resets to auto)."""
    global _activeBudget, _cachedDefault
    _activeBudget = budget
    _cachedDefault = None


def get_resource_budget() -> ResourceBudget:
    """Return the active budget, lazily resolving (and caching) a default."""
    global _cachedDefault
    if _activeBudget is not None:
        return _activeBudget
    if _cachedDefault is None:
        _cachedDefault = resolve_budget()
    return _cachedDefault


def worker_prefetch_depth(
    requested: int | None = None,
    budget: ResourceBudget | None = None,
) -> int:
    """Read-ahead depth for parallel block reads, capped by worker count.

    Defaults to one in-flight block per worker. A consumer may ``request`` a
    smaller depth; the worker count is always the ceiling.
    """
    budget = budget or get_resource_budget()
    workers = max(1, budget.workers)
    if requested is None:
        return workers
    return max(1, min(int(requested), workers))


# Returns flatten beyond ~4-8 shards processed at once (benchmarked on local and
# R2); across-shard depth is the dominant lever, so this caps it while memory
# (workingCopies) is the real ceiling for very large data.
ACROSS_SHARD_CAP = 8


@dataclass(frozen=True)
class ShardPlan:
    """How to split a worker budget for shard-parallel processing.

    ``across`` is the number of shards processed concurrently (the dominant
    lever). ``ioConcurrency`` feeds Zarr's ``async.concurrency`` for the
    duration of the op, and ``withinBlockThreads`` bounds BLAS/OpenMP threads
    per shard. The product ``across * max(ioConcurrency, withinBlockThreads)``
    stays near ``workers`` so nested concurrency never fans out to ``workers``
    squared in-flight requests.
    """

    across: int
    ioConcurrency: int
    withinBlockThreads: int


def shard_parallelism(
    workers: int | None = None,
    n_shards: int | None = None,
    *,
    budget: ResourceBudget | None = None,
) -> ShardPlan:
    """Derive a :class:`ShardPlan` from the resource budget.

    Spends the budget on across-shard depth (capped by ``ACROSS_SHARD_CAP``,
    ``workingCopies`` and the shard count) and keeps within-block BLAS threads
    at one, matching the benchmark finding that across-shard parallelism is the
    lever and extra BLAS threads add little. The remaining budget becomes the
    Zarr IO concurrency for the op.
    """
    budget = budget or get_resource_budget()
    workers = budget.workers if workers is None else max(1, int(workers))
    caps = [workers, ACROSS_SHARD_CAP, max(1, budget.workingCopies)]
    if n_shards is not None:
        caps.append(max(1, int(n_shards)))
    across = max(1, min(caps))
    remainder = max(1, workers // across)
    return ShardPlan(
        across=across,
        ioConcurrency=remainder,
        withinBlockThreads=1,
    )
