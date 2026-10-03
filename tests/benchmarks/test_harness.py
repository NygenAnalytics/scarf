"""Deterministic tests of the benchmark harness's fitting, timing, and verdicts."""

import gc
import json
import math
from pathlib import Path

import numpy as np
import pytest

from . import harness
from .conftest import Bench, BenchmarkSession
from .harness import Fit, Ladder, Result, Thresholds, Timing, Verdict

LADDER = Ladder(sizes=(1_000, 2_000, 4_000, 8_000), smoke=100)


STAGE = Ladder(sizes=(1_000, 2_000, 4_000, 8_000), smoke=100, model="linear")


def _result(best: list[float], exponent: float, ladder: Ladder = LADDER) -> Result:
    timings = tuple(
        Timing(size, seconds, seconds, 3) for size, seconds in zip(ladder.sizes, best)
    )
    projections = {
        target: harness.project(timings[-1], exponent, target, ladder.work)
        for target in ladder.targets
    }
    return Result("demo", ladder, timings, Fit(exponent, 1.0, 1.0), projections)


def _stage(overhead: float, rate: float, ladder: Ladder = STAGE) -> Result:
    """A linear-model result whose times follow ``overhead + rate * size``."""
    timings = tuple(
        Timing(size, overhead + rate * size, overhead + rate * size, 3)
        for size in ladder.sizes
    )
    fit = Fit(1.0, rate, 1.0, overhead)
    projections = {
        target: harness.project_linear(fit, target, ladder.work)
        for target in ladder.targets
    }
    return Result("stage", ladder, timings, fit, projections)


def _baseline(best: list[float], exponent: float, ladder: Ladder = LADDER) -> dict:
    return _result(best, exponent, ladder).to_json()


def test_fit_power_law_recovers_an_exact_power_law() -> None:
    sizes = np.array([1_000, 2_000, 4_000, 8_000], dtype=float)
    fit = harness.fit_power_law(sizes, 2e-6 * sizes**1.5)

    assert fit.exponent == pytest.approx(1.5, abs=1e-12)
    assert fit.coefficient == pytest.approx(2e-6, rel=1e-9)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-12)


def test_fit_power_law_reports_how_well_the_law_fits() -> None:
    sizes = [1_000, 2_000, 4_000, 8_000]
    # A fixed overhead bends the curve, so a power law fits it imperfectly.
    fit = harness.fit_power_law(sizes, [0.5 + size * 1e-5 for size in sizes])

    assert 0.0 < fit.exponent < 1.0
    assert 0.9 < fit.r_squared < 1.0


@pytest.mark.parametrize(
    ("sizes", "seconds", "message"),
    [
        ([1_000], [0.1], "two sizes or more"),
        ([1_000, 2_000], [0.1], "two sizes or more"),
        ([1_000, 1_000], [0.1, 0.2], "two distinct sizes"),
        ([1_000, 2_000], [0.1, 0.0], "finite and positive"),
        ([0, 2_000], [0.1, 0.2], "finite and positive"),
    ],
)
def test_fit_power_law_rejects_unusable_measurements(sizes, seconds, message) -> None:
    with np.errstate(divide="ignore"), pytest.raises(ValueError, match=message):
        harness.fit_power_law(sizes, seconds)


def test_fit_linear_separates_fixed_overhead_from_the_per_unit_rate() -> None:
    sizes = [4_000, 8_000, 16_000, 32_000]
    fit = harness.fit_linear(sizes, [0.3 + 2e-5 * size for size in sizes])

    assert fit.intercept == pytest.approx(0.3, rel=1e-9)
    assert fit.coefficient == pytest.approx(2e-5, rel=1e-9)
    assert (fit.exponent, fit.r_squared) == (1.0, pytest.approx(1.0))


def test_fit_linear_never_returns_negative_terms() -> None:
    sizes = [4_000, 8_000, 16_000, 32_000]
    # Times that fall with size have no measurable per-unit cost.
    falling = harness.fit_linear(sizes, [0.4, 0.3, 0.3, 0.2])
    assert falling.coefficient == 0.0
    assert 0.2 < falling.intercept < 0.4
    # A steep line through a negative intercept is refitted through the origin.
    steep = harness.fit_linear(sizes, [0.01, 0.06, 0.16, 0.36])
    assert steep.intercept == 0.0
    assert steep.coefficient == pytest.approx(1.1e-5, rel=0.1)


def test_projection_is_anchored_at_the_largest_measurement() -> None:
    timing = Timing(16_000, 0.1, 0.12, 3)

    assert harness.project(timing, 1.0, 1_000_000) == pytest.approx(6.25)
    assert harness.project(timing, 2.0, 32_000) == pytest.approx(0.4)
    assert harness.project(timing, 1.0, 1_000_000, work=10.0) == pytest.approx(62.5)
    # Only noise can make time fall with size; that projects as constant.
    assert harness.project(timing, -0.3, 1_000_000) == pytest.approx(0.1)


def test_best_time_keeps_the_fastest_repeat_and_pauses_collection() -> None:
    ticks = iter([0, 300, 1_000, 1_100, 2_000, 2_200, 3_000, 3_250])
    states = []

    best, median, repeats = harness.best_time(
        lambda: states.append(gc.isenabled()),
        min_repeats=4,
        max_repeats=4,
        clock=lambda: next(ticks),
    )

    assert (best, median, repeats) == (100e-9, 225e-9, 4)
    assert states == [False] * 4
    assert gc.isenabled()


def test_best_time_repeats_until_the_minimum_total_accumulates() -> None:
    now = [0]

    def clock() -> int:
        now[0] += 50_000_000
        return now[0]

    # Every call takes 50 ms by the fake clock, so 0.2 s takes four calls.
    _best, _median, repeats = harness.best_time(
        lambda: None, min_repeats=1, max_repeats=10, min_total=0.2, clock=clock
    )
    assert repeats == 4


@pytest.mark.parametrize(("low", "high"), [(0, 3), (4, 3)])
def test_best_time_rejects_impossible_repeat_bounds(low: int, high: int) -> None:
    with pytest.raises(ValueError, match="min_repeats"):
        harness.best_time(lambda: None, min_repeats=low, max_repeats=high)


def test_measure_warms_up_then_fits_and_projects_the_ladder() -> None:
    built = []
    milliseconds = {1_000: 10, 2_000: 20, 4_000: 40, 8_000: 80}
    now = [0]

    def make(size: int):
        built.append(size)

        def call() -> None:
            now[0] += milliseconds[size] * 1_000_000

        return call

    result = harness.measure(
        "linear", make, LADDER, min_repeats=3, clock=lambda: now[0]
    )

    # The warm-up builds the smallest size once before the timed ladder.
    assert built == [1_000, 1_000, 2_000, 4_000, 8_000]
    # Repeats stop once 0.2 s accumulate, between 3 and 15 calls.
    assert [timing.repeats for timing in result.timings] == [15, 10, 5, 3]
    assert [timing.best for timing in result.timings] == pytest.approx(
        [0.01, 0.02, 0.04, 0.08]
    )
    assert result.fit.exponent == pytest.approx(1.0, abs=1e-9)
    assert result.projections == {
        1_000_000: pytest.approx(10.0),
        10_000_000: pytest.approx(100.0),
    }


def test_measure_rebuilds_inputs_before_each_fresh_repeat() -> None:
    built = []

    def make(size: int):
        built.append(size)
        return lambda: None

    result = harness.measure("fresh", make, LADDER, fresh=True, min_repeats=2)

    # After the warm-up, rounds alternate direction across the ladder.
    assert built == [1_000, 1_000, 2_000, 4_000, 8_000, 8_000, 4_000, 2_000, 1_000]
    assert [timing.repeats for timing in result.timings] == [2, 2, 2, 2]


def test_measure_projects_a_fixed_cost_as_a_constant() -> None:
    ladder = Ladder(sizes=(1,), smoke=1, unit="process", targets=(1, 20), model="fixed")
    ticks = iter(range(0, 10_000_000_000, 1_000_000_000))

    result = harness.measure(
        "cold", lambda _size: lambda: None, ladder, clock=lambda: next(ticks)
    )

    assert result.fit == Fit(0.0, 1.0, 1.0)
    assert result.projections == {1: pytest.approx(1.0), 20: pytest.approx(1.0)}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sizes": (1_000,)}, "at least two sizes"),
        ({"sizes": (2_000, 1_000)}, "positive and increasing"),
        ({"sizes": (0, 1_000)}, "positive and increasing"),
        ({"sizes": (1, 2), "model": "fixed"}, "exactly one size"),
        ({"sizes": (1, 2), "model": "cubic"}, "model must be one of"),
        ({"sizes": (1_000, 2_000), "smoke": 0}, "smoke size"),
        ({"sizes": (1_000, 2_000), "targets": ()}, "targets must be positive"),
        ({"sizes": (1_000, 2_000), "work": math.inf}, "work multiplier"),
    ],
)
def test_ladder_rejects_inconsistent_settings(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Ladder(**{"smoke": 10, **kwargs})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"slowdown": 1.0}, "greater than one"),
        ({"exponent_increase": 0.0}, "exponent threshold"),
        ({"meaningful_delay": -1.0}, "non-negative"),
        ({"meaningful_overhead": -1.0}, "overhead must be non-negative"),
        ({"min_r_squared": 1.5}, "between zero and one"),
    ],
)
def test_thresholds_reject_settings_that_cannot_separate_noise(
    kwargs: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Thresholds(**kwargs)


def test_speed_factor_is_the_geometric_mean_of_shared_workloads() -> None:
    current = {"numpySort": 0.2, "sparseProduct": 0.05, "interpreted": 9.0}
    reference = {"numpySort": 0.1, "sparseProduct": 0.1, "missing": 1.0}

    assert harness.speed_factor(current, reference) == pytest.approx(1.0)
    assert harness.speed_factor(current, {"numpySort": 0.05}) == pytest.approx(4.0)
    assert harness.speed_factor(current, {}) == 1.0


def test_calibration_times_every_workload() -> None:
    calibration = harness.calibrate()

    assert set(calibration) == {"numpySort", "sparseProduct", "interpreted"}
    assert all(0 < seconds < 5 for seconds in calibration.values())


def test_compare_without_or_with_a_changed_baseline_does_not_judge() -> None:
    result = _result([0.01, 0.02, 0.04, 0.08], 1.0)
    changed = _baseline(
        [0.01, 0.02, 0.04], 1.0, Ladder(sizes=(1_000, 2_000, 4_000), smoke=100)
    )
    refactored = {**_baseline([0.01, 0.02, 0.04, 0.08], 1.0), "work": 10.0}

    assert harness.compare(result, None).status == "new"
    assert harness.compare(result, changed).status == "stale"
    assert harness.compare(result, refactored).status == "stale"


def test_compare_flags_a_slowdown_that_projects_to_a_meaningful_delay() -> None:
    baseline = _baseline([0.01, 0.02, 0.04, 0.08], 1.0)
    # 0.08 s at 8k cells is 10 s at 1M; 1.5x slower adds 5 s there.
    verdict = harness.compare(_result([0.015, 0.03, 0.06, 0.12], 1.0), baseline)

    assert verdict.status == "regression" and verdict.failed
    assert verdict.ratio == pytest.approx(1.5)
    assert verdict.delays[1_000_000] == pytest.approx(5.0)
    assert verdict.delays[10_000_000] == pytest.approx(50.0)
    assert verdict.reasons == ("1.50x slower at 8k cells; +5 s projected at 1M cells",)


def test_compare_tolerates_a_slowdown_too_small_to_matter_at_scale() -> None:
    # A constant-time call doubles from 1 ms to 2 ms: noise-sized at any scale.
    baseline = _baseline([0.001] * 4, 0.0)
    verdict = harness.compare(_result([0.002] * 4, 0.0), baseline)

    assert verdict.status == "ok"
    assert verdict.ratio == pytest.approx(2.0)
    assert verdict.delays[10_000_000] == pytest.approx(0.001)


def test_compare_flags_a_scaling_regression_hidden_at_small_sizes() -> None:
    baseline = _baseline([0.01, 0.02, 0.04, 0.08], 1.0)
    # Equal time at 8k cells, but the exponent grew from 1 to 1.4.
    verdict = harness.compare(_result([0.0045, 0.012, 0.03, 0.08], 1.4), baseline)

    assert verdict.status == "regression"
    assert verdict.ratio == pytest.approx(1.0)
    assert verdict.exponent_change == pytest.approx(0.4)
    assert verdict.reasons[0].startswith("scaling exponent 1.00 -> 1.40; +")
    assert verdict.reasons[0].endswith("projected at 10M cells")


def test_compare_scales_the_baseline_by_the_machine_speed_factor() -> None:
    baseline = _baseline([0.01, 0.02, 0.04, 0.08], 1.0)
    slower_machine = _result([0.02, 0.04, 0.08, 0.16], 1.0)

    assert harness.compare(slower_machine, baseline).status == "regression"
    calibrated = harness.compare(slower_machine, baseline, factor=2.0)
    assert calibrated.status == "ok"
    assert calibrated.ratio == pytest.approx(1.0)
    assert harness.compare(
        _result([0.005, 0.01, 0.02, 0.04], 1.0), baseline
    ).status == ("faster")


def test_compare_linear_flags_a_per_cell_slowdown_at_production_size() -> None:
    baseline = _stage(0.2, 1e-5).to_json()
    # The per-cell rate grows 1.5x: +5 s at 1M cells, +50 s at 10M.
    verdict = harness.compare(_stage(0.2, 1.5e-5), baseline)

    assert verdict.status == "regression"
    assert verdict.ratio == pytest.approx(15.2 / 10.2)
    assert verdict.delays == {
        1_000_000: pytest.approx(5.0),
        10_000_000: pytest.approx(50.0),
    }
    assert verdict.reasons == ("1.49x slower at 1M cells; +5 s projected at 1M cells",)


def test_compare_linear_judges_fixed_overhead_on_its_own_threshold() -> None:
    baseline = _stage(0.2, 1e-5).to_json()

    slower_calls = harness.compare(_stage(1.0, 1e-5), baseline)
    assert slower_calls.status == "regression"
    assert slower_calls.reasons == ("fixed overhead per call 200 ms -> 1 s",)
    # Doubling a 20 ms overhead is noise next to a stage's per-cell work.
    small = harness.compare(_stage(0.04, 1e-5), _stage(0.02, 1e-5).to_json())
    assert small.status == "ok"


def test_compare_marks_a_changed_model_as_stale() -> None:
    power = _baseline([0.01, 0.02, 0.04, 0.08], 1.0)

    assert harness.compare(_stage(0.2, 1e-5), power).status == "stale"


def test_formatting_uses_compact_units() -> None:
    assert [harness.format_size(size) for size in (900, 1_000, 16_000, 1_500_000)] == [
        "900",
        "1k",
        "16k",
        "1.5M",
    ]
    assert harness.format_seconds(0.00123) == "1.23 ms"
    assert harness.format_seconds(42.0) == "42 s"
    assert harness.format_seconds(600.0) == "10 min"
    assert harness.format_seconds(9_000.0) == "2.5 h"
    assert harness.format_seconds(-0.5, signed=True) == "-500 ms"
    assert harness.format_seconds(3.0, signed=True) == "+3 s"


def test_results_round_trip_and_merge_into_a_baseline(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    machine = {"processor": "test"}
    harness.write_results(
        path, {"kept": _result([0.01] * 4, 0.0)}, {"numpySort": 0.1}, machine=machine
    )
    harness.write_results(
        path,
        {"added": _result([0.02, 0.04, 0.08, 0.16], 1.0)},
        {"numpySort": 0.2},
        merge_with=harness.load_baseline(path),
        machine=machine,
    )

    document = harness.load_baseline(path)
    assert document is not None
    assert sorted(document["benchmarks"]) == ["added", "kept"]
    assert document["calibration"] == {"numpySort": 0.2}
    assert document["benchmarks"]["added"]["best"] == [0.02, 0.04, 0.08, 0.16]
    assert document["benchmarks"]["added"]["projections"]["1000000"] == pytest.approx(
        20.0
    )
    assert harness.load_baseline(tmp_path / "missing.json") is None

    path.write_text(json.dumps({**document, "schema": 0}))
    with pytest.raises(ValueError, match="schema 0"):
        harness.load_baseline(path)


def test_report_lists_projections_with_their_change_from_baseline() -> None:
    result = _result([0.015, 0.03, 0.06, 0.12], 1.0)
    verdict = harness.compare(result, _baseline([0.01, 0.02, 0.04, 0.08], 1.0))
    fixed = Ladder(sizes=(1,), smoke=1, unit="process", targets=(1,), model="fixed")
    cold = _result([9.0], 0.0, fixed)

    stage = _stage(0.2, 1e-5)

    lines = harness.report_lines(
        {"demo": result, "cold": cold, "stage": stage}, {"demo": verdict}, 1.25
    )

    assert lines[0] == (
        "machine speed factor vs baseline: 1.25; projections assume one thread"
    )
    stage_row = next(line for line in lines if line.startswith("stage"))
    assert "lin" in stage_row
    assert "fixed 200 ms + 10 ms per 1k cells, 1M cells: 10.2 s" in stage_row
    demo = next(line for line in lines if line.startswith("demo"))
    assert "120 ms @ 8k" in demo and "1.50x" in demo and "regression" in demo
    assert "1M cells: 15 s (+5 s), 10M cells: 2.5 min (+50 s)" in demo
    cold_row = next(line for line in lines if line.startswith("cold"))
    assert (
        "fixed" in cold_row and "measured" in cold_row and "1 process: 9 s" in cold_row
    )
    assert "    1.50x slower at 8k cells; +5 s projected at 1M cells" in lines


def test_smoke_mode_runs_and_checks_the_smallest_input_without_timing() -> None:
    session = BenchmarkSession("smoke", Path("unused"), Path("unused"))
    seen = []

    result = Bench(session)(
        "demo",
        lambda size: lambda: size * 2,
        LADDER,
        check=lambda size, value: seen.append((size, value)),
    )

    assert result is None
    assert seen == [(100, 200)]
    assert session.results == {}


def test_compare_mode_fails_a_benchmark_that_regressed() -> None:
    ladder = Ladder(sizes=(1_000, 2_000), smoke=10, work=1e9)
    baseline = {
        "benchmarks": {"demo": {**_baseline([1e-12, 2e-12], 1.0, ladder)}},
    }
    session = BenchmarkSession(
        "compare", Path("unused"), Path("unused"), baseline=baseline
    )

    with pytest.raises(pytest.fail.Exception, match="demo regressed: "):
        Bench(session)("demo", lambda size: lambda: sum(range(size)), ladder)
    assert session.verdicts["demo"].status == "regression"
    assert isinstance(session.verdicts["demo"], Verdict)


def test_timed_mode_checks_the_smallest_size_and_records_the_result() -> None:
    session = BenchmarkSession("update", Path("unused"), Path("unused"))
    checked = []
    ladder = Ladder(sizes=(1_000, 2_000), smoke=10, model="linear")

    result = Bench(session)(
        "demo",
        lambda size: lambda: sum(range(size)),
        ladder,
        check=lambda size, value: checked.append((size, value)),
        min_repeats=1,
    )

    assert checked == [(1_000, sum(range(1_000)))]
    assert result is session.results["demo"]
    assert [timing.size for timing in result.timings] == [1_000, 2_000]
    # Update mode records without judging.
    assert session.verdicts == {}


def test_compare_does_not_judge_the_scaling_of_a_noisy_fit() -> None:
    baseline = _baseline([0.01, 0.02, 0.04, 0.08], 1.0)
    clean = _result([0.0045, 0.012, 0.03, 0.08], 1.4)
    noisy = Result(
        clean.name, clean.ladder, clean.timings, Fit(1.4, 1.0, 0.94), clean.projections
    )

    # The same exponent growth fails with a clean fit; a noisy fit is judged
    # only at the largest size, where the times agree.
    assert harness.compare(clean, baseline).status == "regression"
    verdict = harness.compare(noisy, baseline)
    assert verdict.status == "ok"
    assert verdict.reasons == ()
    assert verdict.notes == (
        "fit r-squared 0.940 is below 0.97: judged at the largest size only, "
        "scaling not judged",
    )


def test_compare_judges_a_noisy_linear_fit_at_the_largest_size() -> None:
    baseline = _stage(0.2, 1e-5).to_json()
    clean = _stage(0.2, 1.5e-5)
    noisy = Result(
        clean.name,
        clean.ladder,
        clean.timings,
        Fit(1.0, 1.5e-5, 0.5, 0.2),
        clean.projections,
    )

    verdict = harness.compare(noisy, baseline)

    # 0.32 s against 0.28 s at 8k cells is within the slowdown tolerance,
    # although the clean fit's projection at 1M cells fails.
    assert harness.compare(clean, baseline).status == "regression"
    assert verdict.ratio == pytest.approx(0.32 / 0.28)
    assert verdict.status == "ok"
    assert verdict.notes[0].startswith("fit r-squared 0.500 is below 0.97")


def test_measure_spreads_a_load_change_across_every_size() -> None:
    # After the warm-up, the machine slows 3x from the eleventh call on.
    work = {1_000: 1, 2_000: 2, 4_000: 4, 8_000: 8}

    def drifting_ladder():
        now, calls = [0], [0]

        def make(size: int):
            def call() -> None:
                calls[0] += 1
                slowdown = 3 if calls[0] > 10 else 1
                now[0] += work[size] * slowdown * 10_000_000

            return call

        return make, lambda: now[0]

    # Measured size after size, every repeat of the largest size is slowed,
    # which bends the exponent upward.
    make, clock = drifting_ladder()
    make(1_000)()
    sequential = [
        harness.best_time(make(size), min_repeats=3, max_repeats=3, clock=clock)[0]
        for size in LADDER.sizes
    ]
    assert sequential == pytest.approx([0.01, 0.02, 0.04, 0.24])
    assert harness.fit_power_law(LADDER.sizes, sequential).exponent > 1.4

    # Interleaved rounds give every size an unslowed repeat.
    make, clock = drifting_ladder()
    result = harness.measure(
        "drift", make, LADDER, min_repeats=3, max_repeats=3, clock=clock
    )
    assert [timing.best for timing in result.timings] == pytest.approx(
        [0.01, 0.02, 0.04, 0.08]
    )
    assert result.fit.exponent == pytest.approx(1.0)


def test_combine_keeps_the_fastest_time_at_every_size_and_refits() -> None:
    noisy = _result([0.01, 0.02, 0.04, 0.24], 1.6)
    clean = _result([0.012, 0.019, 0.05, 0.08], 1.0)

    combined = harness.combine(noisy, clean)

    assert [timing.best for timing in combined.timings] == [0.01, 0.019, 0.04, 0.08]
    assert [timing.repeats for timing in combined.timings] == [6, 6, 6, 6]
    assert combined.fit == harness.fit_power_law(
        LADDER.sizes, [0.01, 0.019, 0.04, 0.08]
    )
    with pytest.raises(ValueError, match="same ladder"):
        harness.combine(noisy, _stage(0.2, 1e-5))


def test_compare_mode_confirms_a_regression_before_failing(monkeypatch) -> None:
    from . import conftest

    ladder = Ladder(sizes=(1_000, 2_000), smoke=10)
    baseline = {"benchmarks": {"demo": _baseline([0.01, 0.02], 1.0, ladder)}}
    measured = iter(
        [
            _result([0.01, 0.06], 2.6, ladder),  # a noisy first measurement
            _result([0.011, 0.021], 1.0, ladder),  # the confirmation
        ]
    )
    monkeypatch.setattr(conftest, "measure", lambda *_args, **_kwargs: next(measured))
    session = BenchmarkSession(
        "compare", Path("unused"), Path("unused"), baseline=baseline
    )

    result = Bench(session)("demo", lambda size: lambda: None, ladder)

    # The slow point did not survive the minimum over both measurements.
    assert [timing.best for timing in result.timings] == [0.01, 0.021]
    assert session.verdicts["demo"].status == "ok"
    assert session.results["demo"] is result


def test_compare_mode_fails_a_regression_that_persists(monkeypatch) -> None:
    from . import conftest

    ladder = Ladder(sizes=(1_000, 2_000), smoke=10, work=1e4)
    baseline = {"benchmarks": {"demo": _baseline([0.01, 0.02], 1.0, ladder)}}
    slow = _result([0.02, 0.04], 1.0, ladder)
    calls = []

    def measure(*_args, **_kwargs):
        calls.append(1)
        return slow

    monkeypatch.setattr(conftest, "measure", measure)
    session = BenchmarkSession(
        "compare", Path("unused"), Path("unused"), baseline=baseline
    )

    with pytest.raises(pytest.fail.Exception, match="demo regressed: 2.00x slower"):
        Bench(session)("demo", lambda size: lambda: None, ladder)
    assert len(calls) == 2
    assert session.verdicts["demo"].status == "regression"


def test_compare_flags_a_superlinear_cost_that_is_small_at_the_first_target() -> None:
    # Linear baseline at 7 ms for 8k cells. The current run is 1.4x slower at
    # 8k cells with exponent 1.15, as an added n ** 1.5 cost would make it:
    # +1.65 s at 1M cells, under 2 s, but +27 s at 10M cells, above the
    # 20 s that 2 s per million cells allows there. Its exponent grew by
    # only 0.15, under the exponent threshold.
    baseline = _baseline([0.000875, 0.00175, 0.0035, 0.007], 1.0)
    current = _result([0.000897, 0.00199, 0.00441, 0.0098], 1.15)

    verdict = harness.compare(current, baseline)

    delay = 0.0098 * 1_250**1.15 - 0.007 * 1_250
    assert verdict.delays[1_000_000] == pytest.approx(0.0098 * 125**1.15 - 0.875)
    assert verdict.delays[1_000_000] < 2.0
    assert verdict.delays[10_000_000] == pytest.approx(delay)
    assert verdict.status == "regression"
    assert verdict.reasons == (
        f"1.40x slower at 8k cells; {harness.format_seconds(delay, signed=True)} "
        "projected at 10M cells",
    )


def test_compare_scales_the_meaningful_delay_with_the_target() -> None:
    baseline_ms = [0.006, 0.012, 0.024, 0.048]
    # 1.4x slower and linear: +2.4 s at 1M cells is meaningful there.
    slower = harness.compare(
        _result([value * 1.4 for value in baseline_ms], 1.0),
        _baseline(baseline_ms, 1.0),
    )
    assert slower.reasons == ("1.40x slower at 8k cells; +2.4 s projected at 1M cells",)
    # The same 1.4x on a kernel a quarter as costly adds +0.6 s at 1M and
    # +6 s at 10M cells, under both thresholds.
    cheap = [value / 4 for value in baseline_ms]
    small = harness.compare(
        _result([value * 1.4 for value in cheap], 1.0), _baseline(cheap, 1.0)
    )
    assert small.delays[10_000_000] == pytest.approx(6.0)
    assert small.status == "ok"
