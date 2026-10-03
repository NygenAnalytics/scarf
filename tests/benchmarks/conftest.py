"""Pytest wiring for the scaling micro-benchmarks.

By default every benchmark runs once at its smoke size and checks its
output, so the suite keeps the benchmarks working without timing anything.
``SCARF_RUN_BENCHMARKS=1`` times each ladder, prints a projection table, and
fails a benchmark whose slowdown projects to a meaningful delay. Timed runs
need their own process: pass ``-n 0``.
"""

import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from .harness import (
    BASELINE_PATH,
    BASELINE_VARIABLE,
    DEFAULT_OUTPUT_PATH,
    OUTPUT_VARIABLE,
    RUN_VARIABLE,
    SLOWDOWN_VARIABLE,
    THREADS_VARIABLE,
    UPDATE_VARIABLE,
    Ladder,
    Result,
    Thresholds,
    Verdict,
    calibrate,
    combine,
    compare,
    load_baseline,
    measure,
    report_lines,
    speed_factor,
    write_results,
)


@dataclass
class BenchmarkSession:
    """State shared by the benchmarks of one pytest session."""

    mode: str
    baseline_path: Path
    output_path: Path
    baseline: dict | None = None
    calibration: dict[str, float] = field(default_factory=dict)
    factor: float = 1.0
    results: dict[str, Result] = field(default_factory=dict)
    verdicts: dict[str, Verdict] = field(default_factory=dict)

    @property
    def timed(self) -> bool:
        return self.mode != "smoke"


_REPORTED: list[BenchmarkSession] = []


def _mode() -> str:
    if os.environ.get(RUN_VARIABLE) != "1":
        return "smoke"
    return "update" if os.environ.get(UPDATE_VARIABLE) == "1" else "compare"


def _thresholds(overrides: Thresholds | None) -> Thresholds:
    thresholds = overrides or Thresholds()
    slowdown = os.environ.get(SLOWDOWN_VARIABLE)
    if slowdown is None:
        return thresholds
    return Thresholds(
        slowdown=float(slowdown),
        exponent_increase=thresholds.exponent_increase,
        meaningful_delay=thresholds.meaningful_delay,
    )


@pytest.fixture(scope="session")
def benchmark_session() -> Iterator[BenchmarkSession]:
    session = BenchmarkSession(
        mode=_mode(),
        baseline_path=Path(os.environ.get(BASELINE_VARIABLE, BASELINE_PATH)),
        output_path=Path(os.environ.get(OUTPUT_VARIABLE, DEFAULT_OUTPUT_PATH)),
    )
    if not session.timed:
        yield session
        return
    if os.environ.get("PYTEST_XDIST_WORKER"):
        pytest.fail(
            "Timed benchmarks need a dedicated process; rerun them with -n 0",
            pytrace=False,
        )
    import numba

    previous_threads = numba.get_num_threads()
    requested = int(os.environ.get(THREADS_VARIABLE, "1"))
    numba.set_num_threads(min(requested, numba.config.NUMBA_NUM_THREADS))
    session.baseline = load_baseline(session.baseline_path)
    session.calibration = calibrate()
    if session.baseline is not None and session.mode == "compare":
        session.factor = speed_factor(
            session.calibration, session.baseline.get("calibration", {})
        )
    _REPORTED.append(session)
    try:
        yield session
    finally:
        numba.set_num_threads(previous_threads)
        if session.results:
            write_results(session.output_path, session.results, session.calibration)
            if session.mode == "update":
                write_results(
                    BASELINE_PATH,
                    session.results,
                    session.calibration,
                    merge_with=load_baseline(BASELINE_PATH),
                )


class Bench:
    """Run one benchmark in the session's mode.

    ``make(size)`` builds inputs outside the timed region and returns the
    call to time. ``check(size, value)`` validates the value of that call in
    every mode, so a benchmark is also a test of the operation it times.
    """

    def __init__(self, session: BenchmarkSession) -> None:
        self._session = session

    @property
    def timed(self) -> bool:
        return self._session.timed

    def __call__(
        self,
        name: str,
        make: Callable[[int], Callable[[], object]],
        ladder: Ladder,
        *,
        check: Callable[[int, object], None] | None = None,
        fresh: bool = False,
        thresholds: Thresholds | None = None,
        min_repeats: int = 3,
    ) -> Result | None:
        session = self._session
        if not session.timed:
            value = make(ladder.smoke)()
            if check is not None:
                check(ladder.smoke, value)
            return None
        if check is not None and ladder.model != "fixed":
            check(ladder.sizes[0], make(ladder.sizes[0])())
        result = measure(name, make, ladder, fresh=fresh, min_repeats=min_repeats)
        if session.mode == "compare":
            entry = (session.baseline or {}).get("benchmarks", {}).get(name)
            limits = _thresholds(thresholds)
            verdict = compare(result, entry, factor=session.factor, thresholds=limits)
            if verdict.failed:
                # Confirm a regression before failing: measure again and keep
                # the fastest time at every size, so transient noise cannot
                # fail a benchmark on its own.
                again = measure(
                    name, make, ladder, fresh=fresh, min_repeats=min_repeats
                )
                result = combine(result, again)
                verdict = compare(
                    result, entry, factor=session.factor, thresholds=limits
                )
            session.verdicts[name] = verdict
            session.results[name] = result
            if verdict.failed:
                pytest.fail(
                    f"{name} regressed: " + "; ".join(verdict.reasons), pytrace=False
                )
        session.results[name] = result
        return result


@pytest.fixture
def bench(benchmark_session: BenchmarkSession) -> Bench:
    return Bench(benchmark_session)


def pytest_terminal_summary(terminalreporter) -> None:
    for session in _REPORTED:
        if not session.results:
            continue
        terminalreporter.section("scarf scaling benchmarks")
        for line in report_lines(session.results, session.verdicts, session.factor):
            terminalreporter.write_line(line)
        terminalreporter.write_line(f"results written to {session.output_path}")
        if session.mode == "update":
            terminalreporter.write_line(f"baseline updated at {BASELINE_PATH}")
    _REPORTED.clear()
