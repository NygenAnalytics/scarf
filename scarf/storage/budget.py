"""Process-wide resource budget for predictable memory use.

A single :class:`ResourceBudget` (total memory bytes, worker count, working
copies) is the source of truth. Write-time chunk and shard geometry is derived
from ``memoryBytes // workingCopies``; ``workers`` sets read concurrency and
async IO parallelism. Once files are written, reads follow the on-disk chunk
and shard geometry with no additional memory heuristics.
"""

import os
from dataclasses import dataclass

_DEFAULT_WORKING_COPIES = 8
_FALLBACK_MEMORY_BYTES = 8 * 1024 * 1024 * 1024
_SUFFIXES = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
_MIN_RAW_BYTES = 1024 * 1024


@dataclass(frozen=True)
class ResourceBudget:
    """Memory and worker budget for write-time geometry and read concurrency."""

    memoryBytes: int
    workers: int
    workingCopies: int = _DEFAULT_WORKING_COPIES


def detect_total_memory_bytes() -> int:
    """Best-effort total physical system memory in bytes.

    Uses total (installed) memory rather than currently free memory so the
    derived write-time geometry is a property of the machine, not of transient
    load at the moment the process happens to run.
    """
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
        return max(1, int(number * detect_total_memory_bytes()))
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
    """Read-ahead depth for parallel block reads, capped by ``READ_AHEAD``.

    Matches the shard read-ahead model: at most ``READ_AHEAD`` blocks are kept in
    flight so IO overlaps compute without inflating peak memory. A consumer may
    ``request`` a smaller depth; ``READ_AHEAD`` (bounded by ``workingCopies``) is
    the ceiling.
    """
    budget = budget or get_resource_budget()
    ceiling = max(1, min(READ_AHEAD, budget.workingCopies))
    if requested is None:
        return ceiling
    return max(1, min(int(requested), ceiling))


# Shards are processed in order with this shallow read-ahead so the next band
# downloads while the current one is being processed. A deeper queue only
# inflates peak memory (more resident bands) without improving throughput, since
# the whole worker budget is already spent parallelising IO and compute *within*
# each shard.
READ_AHEAD = 2


@dataclass(frozen=True)
class ShardPlan:
    """How to run a shard-parallel op.

    Shards are processed in order with a shallow read-ahead: up to ``readAhead``
    bands may be in flight so IO overlaps compute. The worker budget is spent on
    ``ioConcurrency`` for Zarr's ``async.concurrency`` (parallel inner-chunk IO
    within a shard); ``withinBlockThreads`` stays at 1 so BLAS/OpenMP reductions
    stay single-threaded and bit-identical regardless of the worker count. Peak
    resident bands is bounded by ``readAhead`` (kept at or below
    ``workingCopies``), so memory stays within the budget the geometry was sized
    against.
    """

    readAhead: int
    ioConcurrency: int
    withinBlockThreads: int


def shard_parallelism(
    workers: int | None = None,
    n_shards: int | None = None,
    *,
    budget: ResourceBudget | None = None,
) -> ShardPlan:
    """Derive a :class:`ShardPlan` from the resource budget.

    Spends the worker budget on parallel inner-chunk IO within a shard and
    pipelines a shallow ``READ_AHEAD`` of shards so the next band downloads while
    the current one is processed. BLAS threads are pinned to one so results stay
    bit-identical across worker counts (extra BLAS threads were benchmarked to
    add little). Read-ahead is bounded by ``workingCopies`` and the shard count
    so peak resident memory stays within budget.
    """
    budget = budget or get_resource_budget()
    workers = budget.workers if workers is None else max(1, int(workers))
    caps = [READ_AHEAD, max(1, budget.workingCopies)]
    if n_shards is not None:
        caps.append(max(1, int(n_shards)))
    read_ahead = max(1, min(caps))
    return ShardPlan(
        readAhead=read_ahead,
        ioConcurrency=workers,
        withinBlockThreads=1,
    )
