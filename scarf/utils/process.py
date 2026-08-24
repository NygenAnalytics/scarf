import resource
import shlex
import subprocess
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from .logging import logger


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
