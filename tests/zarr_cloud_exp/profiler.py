import gc
import resource
import time
import tracemalloc
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


def rss_mb() -> float:
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


@dataclass
class ProfileResult:
    seconds: float = 0.0
    ok: bool = True
    error: str | None = None
    memoryMb: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seconds": self.seconds,
            "ok": self.ok,
            "error": self.error,
            "memoryMb": self.memoryMb,
            "details": self.details,
        }


@contextmanager
def profile() -> Iterator[ProfileResult]:
    gc.collect()
    tracemalloc.start()
    rss_before = rss_mb()
    started = time.perf_counter()
    result = ProfileResult()
    try:
        yield result
    except Exception as exc:
        result.ok = False
        result.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _, traced_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss_after = rss_mb()
        result.seconds = round(time.perf_counter() - started, 3)
        result.memoryMb = {
            "rssBeforeMb": round(rss_before, 1),
            "rssAfterMb": round(rss_after, 1),
            "rssDeltaMb": round(rss_after - rss_before, 1),
            "tracedPeakMb": round(traced_peak / (1024 * 1024), 1),
        }
