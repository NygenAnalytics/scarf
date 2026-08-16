from profiling.config import SELECTED_STAGE_ORDER, validate_requested_stages
from profiling.lattice_experiment import (
    run_phase1_local_checks,
    run_phase2_local_checks,
    run_phase3_local_checks,
    select_phase3_branch,
)
from scarf.storage.async_execution import reset_zarr_runtime_for_tests


def setup_function() -> None:
    reset_zarr_runtime_for_tests()


def teardown_function() -> None:
    reset_zarr_runtime_for_tests()


def test_phase1_local_scaled_pair_is_deterministic() -> None:
    result = run_phase1_local_checks()
    assert "scaled-pair-metadata" in result["checks"]
    counts_chunks = result["scaledPlan"]["countsChunks"]
    counts_shards = result["scaledPlan"]["countsShards"]
    counts_t_chunks = result["scaledPlan"]["countsTChunks"]
    counts_t_shards = result["scaledPlan"]["countsTShards"]
    assert counts_shards[0] % counts_chunks[0] == 0
    assert counts_shards[1] % counts_chunks[1] == 0
    assert counts_t_shards[0] % counts_t_chunks[0] == 0
    assert counts_t_shards[1] % counts_t_chunks[1] == 0
    assert result["canonical10k"]["countsShards"][1] == 45_525


def test_phase2_local_transpose_is_exact() -> None:
    result = run_phase2_local_checks()
    assert "exact-transpose" in result["checks"]
    assert result["checksum"]
    assert result["writer"]["heldLedgerBytes"] == 0
    assert result["failureWriter"]["heldLedgerBytes"] == 0


def test_phase3_local_variants_match_the_transpose() -> None:
    result = run_phase3_local_checks(reps=1)
    assert "transpose-equality" in result["checks"]
    checksums = {payload["checksum"] for payload in result["summaries"].values()}
    assert len(checksums) == 1
    assert result["selectionGroups"]["current"] >= 1
    assert result["selectionGroups"]["candidate"] >= 1
    assert result["branch"] == "E"
    assert result["reason"] == "insufficient-repetitions"


def test_phase3_branch_a_requires_material_gain() -> None:
    summaries = {
        "currentWholeStrip": {
            "reps": 3,
            "writeMedianSeconds": 10.0,
            "markerMedianSeconds": 10.0,
            "hvgMedianSeconds": 10.0,
            "peakMemoryBytes": 100,
        },
        "currentBounded": {
            "reps": 3,
            "writeMedianSeconds": 10.0,
            "markerMedianSeconds": 10.0,
            "hvgMedianSeconds": 10.0,
            "peakMemoryBytes": 100,
        },
        "candidateBounded": {
            "reps": 3,
            "writeMedianSeconds": 7.0,
            "markerMedianSeconds": 8.0,
            "hvgMedianSeconds": 10.0,
            "peakMemoryBytes": 95,
        },
    }
    branch, reason = select_phase3_branch(summaries)
    assert branch == "A"
    assert reason == "candidate-layout-gain"


def test_phase3_selects_b_when_layout_adds_no_bounded_consumer_gain() -> None:
    summaries = {
        "currentWholeStrip": {
            "reps": 3,
            "writeMedianSeconds": 10.0,
            "markerMedianSeconds": 10.0,
            "hvgMedianSeconds": 10.0,
            "peakMemoryBytes": 100,
        },
        "currentBounded": {
            "reps": 3,
            "writeMedianSeconds": 10.0,
            "markerMedianSeconds": 7.0,
            "hvgMedianSeconds": 10.0,
            "peakMemoryBytes": 100,
        },
        "candidateBounded": {
            "reps": 3,
            "writeMedianSeconds": 10.0,
            "markerMedianSeconds": 7.0,
            "hvgMedianSeconds": 10.0,
            "peakMemoryBytes": 100,
        },
    }
    branch, reason = select_phase3_branch(summaries)
    assert branch == "B"
    assert reason == "bounded-current-layout-gain"


def test_phase3_hvg_improvement_is_not_treated_as_a_regression() -> None:
    summaries = {
        "currentWholeStrip": {
            "reps": 3,
            "writeMedianSeconds": 10.0,
            "markerMedianSeconds": 10.0,
            "hvgMedianSeconds": 10.0,
            "peakMemoryBytes": 100,
            "hvgPeakMemoryBytes": 100,
        },
        "currentBounded": {
            "reps": 3,
            "writeMedianSeconds": 10.0,
            "markerMedianSeconds": 10.0,
            "hvgMedianSeconds": 10.0,
            "peakMemoryBytes": 100,
            "hvgPeakMemoryBytes": 100,
        },
        "candidateBounded": {
            "reps": 3,
            "writeMedianSeconds": 7.0,
            "markerMedianSeconds": 10.0,
            "hvgMedianSeconds": 7.0,
            "peakMemoryBytes": 100,
            "hvgPeakMemoryBytes": 70,
        },
    }
    branch, reason = select_phase3_branch(summaries)
    assert branch == "A"
    assert reason == "candidate-layout-gain"


def test_phase3_branch_c_when_writer_admission_fails() -> None:
    summaries = {
        "currentWholeStrip": {
            "reps": 3,
            "writeMedianSeconds": 10.0,
            "markerMedianSeconds": 10.0,
            "hvgMedianSeconds": 10.0,
            "peakMemoryBytes": 100,
        },
        "currentBounded": {
            "reps": 3,
            "writeMedianSeconds": 10.0,
            "markerMedianSeconds": 10.0,
            "hvgMedianSeconds": 10.0,
            "peakMemoryBytes": 100,
        },
        "candidateBounded": {
            "reps": 3,
            "writeMedianSeconds": 10.0,
            "markerMedianSeconds": 7.0,
            "hvgMedianSeconds": 10.0,
            "peakMemoryBytes": 95,
            "writerAdmissionFailed": True,
        },
    }
    branch, reason = select_phase3_branch(summaries)
    assert branch == "C"
    assert reason == "writer-admission-failed-consumer-gain"


def test_phase3_high_variance_is_inconclusive() -> None:
    summaries = {
        name: {
            "reps": 3,
            "writeMedianSeconds": 10.0,
            "markerMedianSeconds": 10.0,
            "hvgMedianSeconds": 10.0,
            "peakMemoryBytes": 100,
            "highVariance": name == "candidateBounded",
        }
        for name in (
            "currentWholeStrip",
            "currentBounded",
            "candidateBounded",
        )
    }
    branch, reason = select_phase3_branch(summaries)
    assert branch == "E"
    assert reason == "high-variance"


def test_phase4_keeps_current_product_layout() -> None:
    from profiling.lattice_experiment import run_phase4_local_checks

    result = run_phase4_local_checks()
    assert result["productBranch"] == "current"
    assert "non-a-keeps-current" in result["checks"]


def test_phase5_local_consumers_match_the_transpose() -> None:
    from profiling.lattice_experiment import run_phase5_local_checks

    result = run_phase5_local_checks()
    assert "unsorted-cell-order" in result["checks"]
    assert result["groups"] >= 1


def test_phase6_local_cluster_contract() -> None:
    from profiling.lattice_experiment import run_phase6_local_checks

    result = run_phase6_local_checks()
    assert result["groupCount"] == 2


def test_selected_stage_order_rejects_missing_prerequisites() -> None:
    validate_requested_stages(SELECTED_STAGE_ORDER)
    try:
        validate_requested_stages(("importClusters",))
    except ValueError as exc:
        assert "requires" in str(exc)
    else:
        raise AssertionError("expected missing prerequisite to fail")
