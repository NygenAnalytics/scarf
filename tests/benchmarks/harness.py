"""Scaling micro-benchmarks that project small timings onto production sizes.

A benchmark times one operation over a ladder of input sizes, keeping the
fastest repeat at each size, and fits the power law
``seconds = coefficient * size ** exponent`` to those times. Anchored at the
largest measured size, the fit projects the operation onto production sizes
such as one and ten million cells. A stage whose fixed overhead hides its
per-cell work at small sizes instead fits ``seconds = fixed + rate * size``,
which separates the two. A fixed-cost benchmark, such as the import and
compilation work of a new process, is measured at one size and projects as a
constant.

A comparison with a saved baseline reports a regression only when two things
hold: the slowdown exceeds timing noise, and the delay it projects at
production size is long enough to matter. A change of a few milliseconds is
therefore a regression when the operation scales it into minutes, and noise
when it does not.
"""

import gc
import json
import math
import os
import platform
import statistics
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

BASELINE_PATH = Path(__file__).with_name("baseline.json")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = REPOSITORY_ROOT / "build" / "benchmarks" / "latest.json"
PRODUCTION_SIZES = (1_000_000, 10_000_000)
BASELINE_SCHEMA = 1
MODELS = ("power", "linear", "fixed")

# Environment switches, in the style of SCARF_RUN_VISUAL_REGRESSION.
RUN_VARIABLE = "SCARF_RUN_BENCHMARKS"
UPDATE_VARIABLE = "SCARF_BENCHMARK_UPDATE"
BASELINE_VARIABLE = "SCARF_BENCHMARK_BASELINE"
OUTPUT_VARIABLE = "SCARF_BENCHMARK_OUTPUT"
SLOWDOWN_VARIABLE = "SCARF_BENCHMARK_SLOWDOWN"
THREADS_VARIABLE = "SCARF_BENCHMARK_THREADS"


@dataclass(frozen=True)
class Ladder:
    """The sizes one benchmark measures and the sizes it projects onto.

    ``work`` scales the size-dependent cost onto the production
    configuration when the benchmark runs a reduced amount of it, such as 30
    of the 300 UMAP epochs. ``smoke`` is the single size run without timing.
    ``model`` is ``"power"`` for a power law, ``"linear"`` for a fixed
    overhead plus a per-unit rate, or ``"fixed"`` for a constant cost
    measured at one size.
    """

    sizes: tuple[int, ...]
    smoke: int
    unit: str = "cells"
    targets: tuple[int, ...] = PRODUCTION_SIZES
    work: float = 1.0
    model: str = "power"

    def __post_init__(self) -> None:
        if self.model not in MODELS:
            raise ValueError(f"The model must be one of {MODELS}")
        if self.model == "fixed" and len(self.sizes) != 1:
            raise ValueError("A fixed-cost ladder measures exactly one size")
        if self.model != "fixed" and len(self.sizes) < 2:
            raise ValueError("A ladder needs at least two sizes to fit scaling")
        if any(size <= 0 for size in self.sizes) or any(
            later <= earlier for earlier, later in zip(self.sizes, self.sizes[1:])
        ):
            raise ValueError("Ladder sizes must be positive and increasing")
        if self.smoke <= 0:
            raise ValueError("The smoke size must be positive")
        if not self.targets or any(target <= 0 for target in self.targets):
            raise ValueError("Projection targets must be positive")
        if not math.isfinite(self.work) or self.work <= 0:
            raise ValueError("The work multiplier must be finite and positive")


@dataclass(frozen=True)
class Timing:
    """Seconds taken at one ladder size, over ``repeats`` timed calls."""

    size: int
    best: float
    median: float
    repeats: int


@dataclass(frozen=True)
class Fit:
    """``seconds = intercept + coefficient * size ** exponent``.

    A power law has no intercept; a linear fit has exponent one.
    """

    exponent: float
    coefficient: float
    r_squared: float
    intercept: float = 0.0


@dataclass(frozen=True)
class Thresholds:
    """When a slower measurement counts as a regression.

    ``slowdown`` is the tolerated ratio of calibrated times, above timing
    noise: at the first production target for a reliable linear fit, and at
    the largest measured size otherwise. ``exponent_increase`` is the
    tolerated growth of a power law's exponent. ``meaningful_delay`` is the
    projected number of seconds, at the first production target, below which
    a change is not worth failing on; larger targets scale it in proportion,
    so 2 s at one million cells means 20 s at ten million. A slowdown counts
    when its delay is meaningful at any target, which catches a superlinear
    cost that is still small at the first. ``meaningful_overhead`` is the same for
    the fixed overhead of each call of a linear fit. ``min_r_squared`` is the
    fit quality that both the result and its baseline need before their
    scaling is judged; a noisier fit is judged only at the largest size.
    """

    slowdown: float = 1.3
    exponent_increase: float = 0.25
    meaningful_delay: float = 2.0
    meaningful_overhead: float = 0.5
    min_r_squared: float = 0.97

    def __post_init__(self) -> None:
        if not math.isfinite(self.slowdown) or self.slowdown <= 1.0:
            raise ValueError("The slowdown threshold must be greater than one")
        if not math.isfinite(self.exponent_increase) or self.exponent_increase <= 0:
            raise ValueError("The exponent threshold must be positive")
        if not math.isfinite(self.meaningful_delay) or self.meaningful_delay < 0:
            raise ValueError("The meaningful delay must be non-negative")
        if not math.isfinite(self.meaningful_overhead) or self.meaningful_overhead < 0:
            raise ValueError("The meaningful overhead must be non-negative")
        if not 0.0 <= self.min_r_squared <= 1.0:
            raise ValueError("The minimum r-squared must be between zero and one")


@dataclass(frozen=True)
class Result:
    """The measured ladder of one benchmark and its production projections."""

    name: str
    ladder: Ladder
    timings: tuple[Timing, ...]
    fit: Fit
    projections: Mapping[int, float]

    @property
    def largest(self) -> Timing:
        return self.timings[-1]

    def to_json(self) -> dict[str, Any]:
        return {
            "unit": self.ladder.unit,
            "work": self.ladder.work,
            "model": self.ladder.model,
            "sizes": list(self.ladder.sizes),
            "best": [timing.best for timing in self.timings],
            "median": [timing.median for timing in self.timings],
            "repeats": [timing.repeats for timing in self.timings],
            "exponent": self.fit.exponent,
            "coefficient": self.fit.coefficient,
            "intercept": self.fit.intercept,
            "rSquared": self.fit.r_squared,
            "projections": {
                str(target): seconds for target, seconds in self.projections.items()
            },
        }


@dataclass(frozen=True)
class Verdict:
    """How one result compares with its baseline entry."""

    status: str
    ratio: float | None = None
    exponent_change: float | None = None
    delays: Mapping[int, float] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def failed(self) -> bool:
        return self.status == "regression"


def _fit_inputs(
    sizes: Sequence[float], seconds: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    n = np.asarray(sizes, dtype=np.float64)
    t = np.asarray(seconds, dtype=np.float64)
    if n.shape != t.shape or n.ndim != 1 or n.size < 2:
        raise ValueError("A fit needs matching sizes and times at two sizes or more")
    if not (np.all(np.isfinite(n) & (n > 0)) and np.all(np.isfinite(t) & (t > 0))):
        raise ValueError("Sizes and times must be finite and positive")
    if np.ptp(n) == 0:
        raise ValueError("A fit needs at least two distinct sizes")
    return n, t


def _r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    spread = float(np.sum((observed - observed.mean()) ** 2))
    if spread == 0:
        return 1.0
    return 1.0 - float(np.sum((observed - predicted) ** 2)) / spread


def fit_power_law(sizes: Sequence[float], seconds: Sequence[float]) -> Fit:
    """Fit ``seconds = coefficient * size ** exponent`` in log space."""
    n, t = _fit_inputs(sizes, seconds)
    x, y = np.log(n), np.log(t)
    exponent, intercept = np.polyfit(x, y, 1)
    r_squared = _r_squared(y, exponent * x + intercept)
    return Fit(float(exponent), float(math.exp(intercept)), r_squared)


def fit_linear(sizes: Sequence[float], seconds: Sequence[float]) -> Fit:
    """Fit ``seconds = intercept + coefficient * size`` with non-negative terms.

    Residuals are weighted by the inverse of each time, so every size counts
    by its relative error. A negative rate, which only noise can produce,
    becomes a constant cost. A negative intercept means there is no overhead
    to separate, so the rate is refitted through the origin by absolute
    error, which the largest sizes dominate as they do at production size.
    """
    n, t = _fit_inputs(sizes, seconds)
    weights = 1.0 / t
    design = np.column_stack([np.ones_like(n), n]) * weights[:, None]
    (intercept, rate), *_ = np.linalg.lstsq(design, t * weights, rcond=None)
    if rate < 0:
        intercept, rate = float(np.sum(weights**2 * t) / np.sum(weights**2)), 0.0
    elif intercept < 0:
        intercept, rate = 0.0, float(np.sum(n * t) / np.sum(n * n))
    predicted = intercept + rate * n
    return Fit(1.0, float(rate), _r_squared(t, predicted), float(intercept))


def project(timing: Timing, exponent: float, target: int, work: float = 1.0) -> float:
    """Project the best time at ``timing.size`` onto ``target`` by a power law.

    The projection is anchored at the measurement, not at the fit's
    intercept, and a negative exponent, which only noise can produce, is
    treated as constant time.
    """
    return work * timing.best * (target / timing.size) ** max(0.0, exponent)


def project_linear(fit: Fit, target: int, work: float = 1.0) -> float:
    """Project a linear fit: its fixed overhead plus ``work`` times the rate."""
    return fit.intercept + work * fit.coefficient * target


def _projections(ladder: Ladder, fit: Fit, largest: Timing) -> dict[int, float]:
    if ladder.model == "linear":
        return {
            target: project_linear(fit, target, ladder.work)
            for target in ladder.targets
        }
    return {
        target: project(largest, fit.exponent, target, ladder.work)
        for target in ladder.targets
    }


def _time_once(fn: Callable[[], object], clock: Callable[[], int]) -> float:
    """Return the seconds one call takes, with garbage collection paused."""
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        start = clock()
        fn()
        elapsed = clock() - start
    finally:
        if was_enabled:
            gc.enable()
    return elapsed / 1e9


def _needs_more(
    samples: list[float], min_repeats: int, max_repeats: int, min_total: float
) -> bool:
    return len(samples) < max_repeats and (
        len(samples) < min_repeats or sum(samples) < min_total
    )


def best_time(
    fn: Callable[[], object],
    *,
    min_repeats: int = 3,
    max_repeats: int = 15,
    min_total: float = 0.2,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> tuple[float, float, int]:
    """Time ``fn`` repeatedly; return the best and median seconds and the repeats.

    Repeats continue until ``min_total`` timed seconds accumulate, within
    ``[min_repeats, max_repeats]``. Earlier garbage is collected once up
    front, and collection is paused inside each timed call, so no collection
    is billed to a call; collections between calls are not timed.
    """
    if min_repeats < 1 or max_repeats < min_repeats:
        raise ValueError("Repeats must satisfy 1 <= min_repeats <= max_repeats")
    samples: list[float] = []
    gc.collect()
    while _needs_more(samples, min_repeats, max_repeats, min_total):
        samples.append(_time_once(fn, clock))
    return min(samples), statistics.median(samples), len(samples)


def measure(
    name: str,
    make: Callable[[int], Callable[[], object]],
    ladder: Ladder,
    *,
    fresh: bool = False,
    min_repeats: int = 3,
    max_repeats: int = 15,
    min_total: float = 0.2,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> Result:
    """Measure ``make(size)()`` over the ladder and project it.

    ``make`` builds the inputs of one size outside the timed region and
    returns the call to time. With ``fresh``, inputs are rebuilt before every
    repeat, for operations that consume or mutate them, and each size is
    timed ``min_repeats`` times. The first call at the smallest size is
    discarded: it pays in-process compilation and imports, and it fills
    on-disk caches that a fixed per-process cost should find.

    Repeats are interleaved across sizes, in rounds of alternating order, so
    a change in machine load while the ladder runs slows every size alike
    instead of bending the fitted scaling.
    """
    if min_repeats < 1 or max_repeats < min_repeats:
        raise ValueError("Repeats must satisfy 1 <= min_repeats <= max_repeats")
    make(ladder.sizes[0])()
    calls = {} if fresh else {size: make(size) for size in ladder.sizes}
    samples: dict[int, list[float]] = {size: [] for size in ladder.sizes}
    if fresh:
        min_total, max_repeats = 0.0, min_repeats
    order = list(ladder.sizes)
    # Collection is paused inside every timed call, so earlier garbage needs
    # collecting only once, not before every round.
    gc.collect()
    while any(
        _needs_more(samples[size], min_repeats, max_repeats, min_total)
        for size in order
    ):
        for size in order:
            if _needs_more(samples[size], min_repeats, max_repeats, min_total):
                call = make(size) if fresh else calls[size]
                samples[size].append(_time_once(call, clock))
        order.reverse()
    timings = [
        Timing(
            size,
            min(samples[size]),
            statistics.median(samples[size]),
            len(samples[size]),
        )
        for size in ladder.sizes
    ]
    return refit(name, ladder, timings)


def refit(name: str, ladder: Ladder, timings: Sequence[Timing]) -> Result:
    """Fit the ladder's model to ``timings`` and project it."""
    best = [timing.best for timing in timings]
    if ladder.model == "fixed":
        fit = Fit(0.0, best[0], 1.0)
    elif ladder.model == "linear":
        fit = fit_linear(ladder.sizes, best)
    else:
        fit = fit_power_law(ladder.sizes, best)
    return Result(
        name, ladder, tuple(timings), fit, _projections(ladder, fit, timings[-1])
    )


def combine(first: Result, second: Result) -> Result:
    """Keep the fastest of two measurements of a ladder at every size.

    A real slowdown persists in both; noise that slowed one of them does
    not survive the minimum.
    """
    if first.ladder != second.ladder:
        raise ValueError("Only measurements of the same ladder can be combined")
    timings = [
        Timing(
            one.size,
            min(one.best, two.best),
            min(one.median, two.median),
            one.repeats + two.repeats,
        )
        for one, two in zip(first.timings, second.timings, strict=True)
    ]
    return refit(first.name, first.ladder, timings)


def calibration_workloads() -> dict[str, Callable[[], object]]:
    """Fixed workloads whose times describe the speed of the current machine.

    They cover the three kinds of work Scarf benchmarks mix: a compiled
    NumPy kernel, a memory-bound sparse product, and interpreted Python.
    """
    from scipy.sparse import random as sparse_random

    rng = np.random.default_rng(0)
    values = rng.random(1 << 20)
    matrix = sparse_random(
        20_000, 2_000, density=0.05, format="csr", random_state=np.random.default_rng(1)
    )
    vector = rng.random(2_000)

    def interpreted() -> int:
        total = 0
        for index in range(200_000):
            total += index * index
        return total

    return {
        "numpySort": lambda: np.sort(values),
        "sparseProduct": lambda: matrix @ vector,
        "interpreted": interpreted,
    }


def calibrate() -> dict[str, float]:
    """Best seconds of each calibration workload on this machine."""
    calibration = {}
    for name, workload in calibration_workloads().items():
        workload()
        best, _median, _repeats = best_time(workload, min_repeats=7, max_repeats=7)
        calibration[name] = best
    return calibration


def speed_factor(current: Mapping[str, float], reference: Mapping[str, float]) -> float:
    """Geometric mean of current over reference calibration times.

    A factor above one means this machine is slower than the one that wrote
    the reference, so the reference times are scaled up by it.
    """
    ratios = [
        current[name] / reference[name]
        for name in reference
        if name in current and current[name] > 0 and reference[name] > 0
    ]
    if not ratios:
        return 1.0
    return math.exp(sum(math.log(ratio) for ratio in ratios) / len(ratios))


def compare(
    result: Result,
    baseline: Mapping[str, Any] | None,
    *,
    factor: float = 1.0,
    thresholds: Thresholds = Thresholds(),
) -> Verdict:
    """Compare a result with its baseline entry after machine calibration."""
    if baseline is None:
        return Verdict("new", reasons=("no baseline entry",))
    ladder = result.ladder
    if (
        list(baseline.get("sizes", ())) != list(ladder.sizes)
        or baseline.get("unit") != ladder.unit
        or float(baseline.get("work", 1.0)) != ladder.work
        or baseline.get("model") != ladder.model
    ):
        return Verdict("stale", reasons=("the ladder changed; update the baseline",))
    first, last = ladder.targets[0], ladder.targets[-1]
    unit = ladder.unit
    reference_exponent = float(baseline["exponent"])
    reference_best = float(baseline["best"][-1]) * factor
    fit_quality = min(result.fit.r_squared, float(baseline.get("rSquared", 1.0)))
    reliable = fit_quality >= thresholds.min_r_squared
    notes = []
    if not reliable:
        notes.append(
            f"fit r-squared {fit_quality:.3f} is below {thresholds.min_r_squared}: "
            "judged at the largest size only, scaling not judged"
        )
    if ladder.model == "linear":
        reference_fit = Fit(
            1.0,
            float(baseline["coefficient"]) * factor,
            1.0,
            float(baseline["intercept"]) * factor,
        )
        reference = _projections(ladder, reference_fit, result.largest)
    else:
        reference = _projections(
            ladder,
            Fit(reference_exponent, 1.0, 1.0),
            Timing(result.largest.size, reference_best, reference_best, 1),
        )
    if ladder.model == "linear" and reliable:
        ratio = result.projections[first] / max(reference[first], 1e-12)
        where = f"at {format_size(first)} {unit}"
    else:
        ratio = result.largest.best / reference_best
        where = f"at {format_size(result.largest.size)} {unit}"
    exponent_change = result.fit.exponent - reference_exponent
    delays = {
        target: result.projections[target] - reference[target]
        for target in ladder.targets
    }
    meaningful = {
        target: thresholds.meaningful_delay * target / first
        for target in ladder.targets
    }
    reasons = []
    exceeded = [
        target for target in ladder.targets if delays[target] > meaningful[target]
    ]
    if ratio > thresholds.slowdown and exceeded:
        reasons.append(
            f"{ratio:.2f}x slower {where}; "
            f"{format_seconds(delays[exceeded[0]], signed=True)} projected at "
            f"{format_size(exceeded[0])} {unit}"
        )
    if ladder.model == "linear" and reliable:
        overhead = result.fit.intercept
        reference_overhead = float(baseline["intercept"]) * factor
        if (
            overhead > thresholds.slowdown * reference_overhead
            and overhead - reference_overhead > thresholds.meaningful_overhead
        ):
            reasons.append(
                f"fixed overhead per call {format_seconds(reference_overhead)} -> "
                f"{format_seconds(overhead)}"
            )
    elif (
        ladder.model == "power"
        and reliable
        and exponent_change > thresholds.exponent_increase
        and delays[last] > meaningful[last]
    ):
        reasons.append(
            f"scaling exponent {reference_exponent:.2f} -> "
            f"{result.fit.exponent:.2f}; {format_seconds(delays[last], signed=True)} "
            f"projected at {format_size(last)} {unit}"
        )
    if reasons:
        status = "regression"
    elif ratio < 1.0 / thresholds.slowdown:
        status = "faster"
    else:
        status = "ok"
    return Verdict(status, ratio, exponent_change, delays, tuple(reasons), tuple(notes))


def format_size(size: float) -> str:
    """Render a size compactly, such as 16k or 10M."""
    for divisor, suffix in ((1_000_000, "M"), (1_000, "k")):
        if size >= divisor:
            text = f"{size / divisor:.1f}".rstrip("0").rstrip(".")
            return f"{text}{suffix}"
    return str(int(size))


def format_seconds(seconds: float, *, signed: bool = False) -> str:
    """Render seconds as ms, s, min, or h with three significant digits."""
    sign = ""
    if signed:
        sign = "+" if seconds >= 0 else "-"
    value = abs(seconds)
    if value < 1.0:
        text = f"{value * 1e3:.3g} ms"
    elif value < 120.0:
        text = f"{value:.3g} s"
    elif value < 7_200.0:
        text = f"{value / 60.0:.3g} min"
    else:
        text = f"{value / 3_600.0:.3g} h"
    return sign + text


def machine_description() -> dict[str, Any]:
    """Describe the interpreter, libraries, and revision behind a measurement."""
    import numba
    import scipy

    try:
        revision = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpuCount": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numba": numba.__version__,
        "revision": revision,
    }


def load_baseline(path: Path) -> dict[str, Any] | None:
    """Read a baseline file, or return ``None`` when there is none."""
    if not path.is_file():
        return None
    document = json.loads(path.read_text())
    if document.get("schema") != BASELINE_SCHEMA:
        raise ValueError(
            f"{path} has benchmark schema {document.get('schema')!r}; "
            f"expected {BASELINE_SCHEMA}. Regenerate it with {UPDATE_VARIABLE}=1."
        )
    return document


def write_results(
    path: Path,
    results: Mapping[str, Result],
    calibration: Mapping[str, float],
    *,
    merge_with: Mapping[str, Any] | None = None,
    machine: Mapping[str, Any] | None = None,
) -> None:
    """Write measured results; entries of ``merge_with`` not re-measured are kept."""
    benchmarks: dict[str, Any] = {}
    if merge_with is not None:
        benchmarks.update(merge_with.get("benchmarks", {}))
    benchmarks.update({name: result.to_json() for name, result in results.items()})
    document = {
        "schema": BASELINE_SCHEMA,
        "machine": dict(machine) if machine is not None else machine_description(),
        "calibration": dict(calibration),
        "benchmarks": dict(sorted(benchmarks.items())),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n")


def _projection_text(result: Result, verdict: Verdict) -> str:
    unit = result.ladder.unit
    cells = []
    if result.ladder.model == "linear":
        rate = format_seconds(result.fit.coefficient * 1_000 * result.ladder.work)
        cells.append(
            f"fixed {format_seconds(result.fit.intercept)} + {rate} per 1k {unit}"
        )
    for target, seconds in result.projections.items():
        text = f"{format_size(target)} {unit}: {format_seconds(seconds)}"
        if target in verdict.delays:
            text += f" ({format_seconds(verdict.delays[target], signed=True)})"
        cells.append(text)
    return ", ".join(cells)


def report_lines(
    results: Mapping[str, Result],
    verdicts: Mapping[str, Verdict],
    factor: float,
) -> list[str]:
    """Render one row per benchmark: timing, scaling, verdict, and projections.

    Projections are single-thread seconds, followed by the delay they add
    over the calibrated baseline when there is one.
    """
    lines = [
        f"machine speed factor vs baseline: {factor:.2f}; projections assume "
        "one thread",
        f"{'benchmark':<34} {'largest':>18} {'exp':>5} {'vs base':>8}  "
        f"{'verdict':<10} projections (change vs baseline)",
    ]
    for name, result in sorted(results.items()):
        verdict = verdicts.get(name, Verdict("measured"))
        largest = (
            f"{format_seconds(result.largest.best)} @ "
            f"{format_size(result.largest.size)}"
        )
        exponent = {
            "fixed": "fixed",
            "linear": "lin",
        }.get(result.ladder.model, f"{result.fit.exponent:.2f}")
        ratio = "" if verdict.ratio is None else f"{verdict.ratio:.2f}x"
        lines.append(
            f"{name:<34} {largest:>18} {exponent:>5} {ratio:>8}  "
            f"{verdict.status:<10} {_projection_text(result, verdict)}"
        )
        lines.extend(f"    {reason}" for reason in verdict.reasons)
        lines.extend(f"    note: {note}" for note in verdict.notes)
    return lines
