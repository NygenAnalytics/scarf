"""Process-wide resource budget for predictable, worker-scaled memory use.

A single :class:`ResourceBudget` (total memory bytes and worker count) is the
source of truth for every default that drives peak memory: streaming block
size, async concurrency, prefetch depth, and tiled reductions. The intent is
that peak memory stays close to ``workers * perWorkerTileBytes`` so adding a
worker raises memory predictably while improving runtime.
"""

import os
from dataclasses import dataclass

_DEFAULT_MEMORY_FRACTION = 0.6
_FALLBACK_MEMORY_BYTES = 8 * 1024 * 1024 * 1024
_SUFFIXES = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
# Numeric specs below this (without a unit suffix) are almost certainly a
# mistake (e.g. "1.0" meant as a fraction), so we reject them with guidance.
_MIN_RAW_BYTES = 1024 * 1024
# Headroom multiplier for streaming reductions: the raw band plus its float64
# normalized copy and a transient. Used to size tiles against the budget.
DEFAULT_TEMP_FACTOR = 3


@dataclass(frozen=True)
class ResourceBudget:
    """Memory and worker budget that bounds streaming and concurrency."""

    memoryBytes: int
    workers: int

    @property
    def perWorkerBytes(self) -> int:
        return max(1, self.memoryBytes // max(1, self.workers))


def detect_available_memory_bytes() -> int:
    """Best-effort available system memory in bytes.

    Prefers ``/proc/meminfo`` ``MemAvailable`` (Linux/WSL), then sysconf, then
    a conservative fallback so the budget is always defined.
    """
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
    except ValueError, OSError, AttributeError:
        pass
    return _FALLBACK_MEMORY_BYTES


def detect_workers() -> int:
    return max(1, os.cpu_count() or 1)


def _parse_memory_spec(spec: str | int | float) -> int:
    """Parse a memory spec into bytes.

    Accepts raw byte counts (int), suffixed sizes such as ``"8G"`` or
    ``"512M"``, and fractions of available memory strictly between 0 and 1
    (e.g. ``"0.6"``). A bare numeric ``>= 1`` is treated as raw bytes; values
    that are clearly too small to be a real budget (and likely a mistaken
    fraction such as ``"1.0"``) are rejected with guidance.
    """
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


def resolve_budget(
    memory: int | str | None = None,
    workers: int | None = None,
) -> ResourceBudget:
    """Build a :class:`ResourceBudget`, auto-detecting unset fields.

    ``memory`` falls back to ``SCARF_MEM_BUDGET`` then to
    ``_DEFAULT_MEMORY_FRACTION`` of available memory. ``workers`` falls back to
    ``SCARF_WORKERS`` then to the detected CPU count.
    """
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
        memoryBytes=max(1, memory_bytes), workers=max(1, worker_count)
    )


_activeBudget: ResourceBudget | None = None
_cachedDefault: ResourceBudget | None = None


def set_resource_budget(budget: ResourceBudget | None) -> None:
    """Install the process-wide active budget (None resets to auto)."""
    global _activeBudget, _cachedDefault
    _activeBudget = budget
    # Invalidate the lazily resolved default so a reset re-detects fresh.
    _cachedDefault = None


def get_resource_budget() -> ResourceBudget:
    """Return the active budget, lazily resolving (and caching) a default."""
    global _cachedDefault
    if _activeBudget is not None:
        return _activeBudget
    if _cachedDefault is None:
        _cachedDefault = resolve_budget()
    return _cachedDefault


def tile_rows_for_width(
    n_cols: int,
    itemsize: int,
    budget: ResourceBudget | None = None,
    *,
    temp_factor: int = DEFAULT_TEMP_FACTOR,
    chunk_rows: int = 1,
    n_rows: int | None = None,
) -> int:
    """Rows per tile so one in-flight tile fits a worker's memory slice.

    The result is aligned down to a multiple of ``chunk_rows`` (never below
    one chunk) and capped at ``n_rows`` when provided. ``temp_factor`` reserves
    headroom for the normalized output and transient copies.
    """
    budget = budget or get_resource_budget()
    n_cols = max(1, int(n_cols))
    itemsize = max(1, int(itemsize))
    chunk_rows = max(1, int(chunk_rows))
    denom = n_cols * itemsize * max(1, temp_factor)
    rows = budget.perWorkerBytes // denom
    rows = max(chunk_rows, (rows // chunk_rows) * chunk_rows)
    if n_rows is not None:
        rows = min(rows, max(1, int(n_rows)))
    return int(max(1, rows))


def concurrency_for_chunk(
    chunk_bytes: int,
    budget: ResourceBudget | None = None,
    *,
    factor: int = 2,
    floor: int = 4,
) -> int:
    """Max concurrent chunk decompressions that fit the memory budget."""
    budget = budget or get_resource_budget()
    chunk_bytes = max(1, int(chunk_bytes))
    cap = budget.memoryBytes // (chunk_bytes * max(1, factor))
    return int(max(floor, cap))


def bounded_prefetch(
    band_bytes: int,
    budget: ResourceBudget | None = None,
    *,
    requested: int | None = None,
) -> int:
    """Read-ahead depth so in-flight bands fit the per-worker memory slice.

    Bounds the number of concurrently prefetched read bands by both the worker
    count and how many bands of ``band_bytes`` fit in ``perWorkerBytes``. This
    is sized from the actual read band rather than a precomputed layout, so it
    stays correct even when the on-disk geometry differs from the layout used
    for new writes.
    """
    budget = budget or get_resource_budget()
    band_bytes = max(1, int(band_bytes))
    fit = budget.perWorkerBytes // band_bytes
    ceiling = (
        budget.workers if requested is None else min(int(requested), budget.workers)
    )
    return int(max(1, min(ceiling, fit)))
