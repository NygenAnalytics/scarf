from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import zarr
from obstore.store import MemoryStore

from profiling import modal_app
from profiling.config import ProfilingConfig, load_profiling_config
from profiling.phase_checks import (
    CLUSTER_SOURCES_PATH,
    Phase3VariantResult,
    PhaseWorkerResult,
    ScaleBatchValidationResult,
    _phase3_comparable_summary,
    _validate_pca_reduction,
    build_scale_comparison_schedule,
    claim_phase,
    claim_scale_comparison,
    finalize_scale_comparison,
    finalize_phase0,
    inspect_h5ad_manifest,
    reconcile_cluster_inventory,
    render_cluster_sources_toml,
    resume_phase_claim,
    run_current_path_baselines,
    run_phase0_local_checks,
    run_phase0_reopen_body,
    run_phase2_worker_body,
    run_phase3_variant_body,
    synthetic_count_values,
    validate_scale_comparison_prerequisites,
    validate_phase3_prerequisites,
    write_worker_result,
)
from profiling.recording_store import RecordingMemoryStore
from scarf.storage.feature_stream import load_feature_strip

_EXAMPLE_CONFIG = Path(__file__).parents[1] / "profiling" / "config.example.toml"


def _config(*, runTag: str = "phase0-test") -> ProfilingConfig:
    return load_profiling_config(_EXAMPLE_CONFIG).model_copy(update={"runTag": runTag})


def _memory_r2(monkeypatch: pytest.MonkeyPatch) -> MemoryStore:
    store = MemoryStore()

    def fake_open(uri: str):
        key = uri.split("://", 1)[1].split("/", 1)[1]
        return store, key

    monkeypatch.setattr("profiling.r2.open_r2_object", fake_open)
    monkeypatch.setattr(
        "profiling.phase_checks.object_exists", lambda uri: _exists(store, uri)
    )
    monkeypatch.setattr(
        "profiling.phase_checks.put_json_if_absent", _put_if_absent(store)
    )
    monkeypatch.setattr("profiling.phase_checks.get_json", lambda uri: _get(store, uri))
    return store


def _key(uri: str) -> str:
    return uri.split("://", 1)[1].split("/", 1)[1]


def _exists(store: MemoryStore, uri: str) -> bool:
    try:
        store.head(_key(uri))
    except FileNotFoundError:
        return False
    return True


def _put(store: MemoryStore, uri: str, value: dict[str, Any]) -> None:
    from profiling.r2 import _encode_json

    store.put(_key(uri), _encode_json(value))


def _put_if_absent(store: MemoryStore):
    def inner(uri: str, value: dict[str, Any]) -> bool:
        if _exists(store, uri):
            return False
        _put(store, uri, value)
        return True

    return inner


def _get(store: MemoryStore, uri: str) -> dict[str, Any]:
    import json

    body = bytes(store.get(_key(uri)).bytes())
    payload = json.loads(body.decode())
    assert isinstance(payload, dict)
    return payload


def test_phase0_local_zarr_and_current_path_baselines() -> None:
    result = run_phase0_local_checks()
    assert "outer-block-index" in result["checks"]
    assert "async-roundtrip" in result["checks"]
    assert result["currentPathBaselines"]["wholeStripShape"][1] == 221
    assert result["readGroups"]["oneInnerChunk"]["store"]["gets"] >= 1


def test_current_whole_strip_loads_the_full_cell_axis() -> None:
    values = synthetic_count_values()
    baselines = run_current_path_baselines(values)
    store = RecordingMemoryStore()
    from profiling.phase_checks import fill_synthetic_pair

    _counts, counts_t = fill_synthetic_pair(store)
    shard = load_feature_strip(counts_t, 0)
    assert shard.values.shape == (
        min(int(counts_t.chunks[0]), values.shape[1]),
        values.shape[0],
    )
    assert baselines["featureStreamBlocks"] >= 1
    assert baselines["rowStreamSlices"] >= 1


def test_recording_store_tracks_ranges_writes_and_bytes() -> None:
    store = RecordingMemoryStore()
    counts, _counts_t = __import__(
        "profiling.phase_checks", fromlist=["fill_synthetic_pair"]
    ).fill_synthetic_pair(store)
    store.reset()
    _ = np.asarray(counts[0:50, 0:20])
    summary = store.probe.summary()
    assert summary.gets >= 1
    assert summary.transferredBytes > 0
    assert summary.maxInFlight >= 1


def test_claim_refuses_duplicates_and_conflicting_destinations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _memory_r2(monkeypatch)
    config = _config()
    claim = claim_phase(config, "phase0")
    assert claim.phase == "phase0"
    with pytest.raises(RuntimeError, match="Duplicate phase claim"):
        claim_phase(config, "phase0")


def test_resume_phase_claim_loads_matching_durable_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _memory_r2(monkeypatch)
    config = _config()
    original = claim_phase(config, "phase0")

    resumed = resume_phase_claim(config, "phase0")

    assert resumed == original


def test_worker_result_is_create_only_and_reopen_reads_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _memory_r2(monkeypatch)
    config = _config()
    claim_phase(config, "phase0")
    worker = PhaseWorkerResult(
        phase="phase0",
        hypothesis="H0",
        status="ok",
        checks=("outer-block-index",),
        observations={"kind": "observed"},
    )
    write_worker_result(config, worker)
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        write_worker_result(config, worker)
    reopen = run_phase0_reopen_body(config)
    assert reopen.status == "ok"
    assert reopen.validated is True
    final = finalize_phase0(worker, reopen)
    assert final.decision == "accept"
    assert final.nextPhase == "phase1"


def test_finalize_rejects_terminal_worker_failure() -> None:
    worker = PhaseWorkerResult(
        phase="phase0",
        hypothesis="H0",
        status="error",
        checks=(),
        error="boom",
    )
    from profiling.phase_checks import PhaseReopenResult

    reopen_result = PhaseReopenResult(
        phase="phase0",
        hypothesis="H0",
        status="error",
        validated=False,
        checks=(),
        error="boom",
    )
    final = finalize_phase0(worker, reopen_result)
    assert final.decision == "reject"
    assert final.nextPhase is None


def test_h5ad_manifest_records_dtype_and_row_digest(tmp_path: Path) -> None:
    from profiling.datasets import write_fixture_h5ad

    path = tmp_path / "tiny.h5ad"
    write_fixture_h5ad(path, nRows=8, nColumns=6, avgNnzPerRow=3)
    manifest = inspect_h5ad_manifest(
        path, objectMeta={"eTag": "abc", "size": path.stat().st_size}
    )
    assert manifest["nRows"] == 8
    assert manifest["nColumns"] == 6
    assert manifest["orderedRowDigest"]
    assert manifest["inferredStorageDtype"]
    assert manifest["eTag"] == "abc"


def test_cluster_source_toml_is_explicit_and_ignored() -> None:
    inventory = {
        "10000": {
            "candidates": [
                {
                    "status": "candidate",
                    "uri": "s3://bucket/stores/run/10000.zarr",
                    "labelColumn": "RNA_leiden_cluster",
                }
            ]
        }
    }
    text = render_cluster_sources_toml(inventory)
    assert "nRows = 10000" in text
    assert "s3://bucket/stores/run/10000.zarr" in text
    assert CLUSTER_SOURCES_PATH.name == "cluster_sources.toml"


def test_cluster_inventory_requires_exact_input_identity() -> None:
    inventory = {
        "100000": {
            "h7": "accepted",
            "candidates": [
                {
                    "status": "candidate",
                    "orderedCellDigest": "wrong",
                    "orderedFeatureDigest": "features",
                    "complete": True,
                }
            ],
        }
    }
    manifests = {
        "100000": {
            "orderedRowDigest": "cells",
            "orderedFeatureDigest": "features",
        }
    }
    reconciled = reconcile_cluster_inventory(inventory, manifests)
    candidate = reconciled["100000"]["candidates"][0]
    assert candidate["status"] == "input-identity-mismatch"
    assert reconciled["100000"]["h7"] == "blocked"


def test_phase3_prerequisites_require_durable_input_and_cluster_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    manifest = {
        "status": "ok",
        "nRows": 100_000,
        "nColumns": 45_525,
        "inferredStorageDtype": "uint16",
        "sha256": "input-sha",
    }
    inventory = {
        "100000": {
            "h7": "accepted",
            "candidates": [
                {
                    "status": "candidate",
                    "uri": "s3://bucket/store.zarr",
                    "labelColumn": "RNA_leiden_cluster",
                }
            ],
        }
    }
    monkeypatch.setattr(
        "profiling.phase_checks.get_json",
        lambda uri: manifest if "inputs/" in uri else inventory,
    )

    evidence = validate_phase3_prerequisites(config)
    assert evidence["inputSha256"] == "input-sha"
    assert evidence["clusterSourceUri"] == "s3://bucket/store.zarr"


def test_phase3_prerequisites_reject_missing_validated_cluster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    manifest = {
        "status": "ok",
        "nRows": 100_000,
        "nColumns": 45_525,
        "inferredStorageDtype": "uint16",
        "sha256": "input-sha",
    }
    inventory = {"100000": {"h7": "blocked", "candidates": []}}
    monkeypatch.setattr(
        "profiling.phase_checks.get_json",
        lambda uri: manifest if "inputs/" in uri else inventory,
    )

    with pytest.raises(RuntimeError, match="validated 100k cluster"):
        validate_phase3_prerequisites(config)


def test_scale_schedule_runs_and_validates_the_pilot_before_continuation() -> None:
    schedule = build_scale_comparison_schedule()

    assert len(schedule) == 9
    assert schedule[0] == {
        "repetition": 0,
        "variant": "currentWholeStrip",
        "order": 0,
        "batch": "pilot",
    }
    assert {item["variant"] for item in schedule[:3]} == {
        "currentWholeStrip",
        "currentBounded",
        "candidateBounded",
    }
    assert all(item["batch"] == "continuation" for item in schedule[3:])


def test_scale_prerequisites_use_durable_one_million_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(runTag="scale-1m")
    manifest = {
        "status": "ok",
        "nRows": 1_000_000,
        "nColumns": 45_525,
        "inferredStorageDtype": "uint16",
        "sha256": "input-sha",
    }
    inventory = {
        "1000000": {
            "h7": "accepted",
            "candidates": [
                {
                    "status": "candidate",
                    "uri": "s3://bucket/store.zarr",
                    "labelColumn": "RNA_leiden_cluster",
                }
            ],
        }
    }
    monkeypatch.setattr(
        "profiling.phase_checks.get_json",
        lambda uri: manifest if "inputs/" in uri else inventory,
    )

    evidence = validate_scale_comparison_prerequisites(
        config,
        nRows=1_000_000,
        evidenceRunTag="phase-evidence",
    )

    assert evidence["inputSha256"] == "input-sha"
    assert evidence["nRows"] == 1_000_000
    assert evidence["clusterSource"]["nRows"] == 1_000_000


def test_scale_variant_body_routes_every_stage_to_one_million(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(runTag="scale-1m")
    calls: list[dict[str, Any]] = []
    written_json: list[str] = []

    class _StageResult:
        status = "ok"
        error = None

        def to_json(self) -> dict[str, Any]:
            return {
                "status": "ok",
                "seconds": 1.0,
                "details": {},
            }

    def fake_run_stage(stage: str, **kwargs: Any) -> _StageResult:
        calls.append({"stage": stage, **kwargs})
        return _StageResult()

    def fake_download(_uri: str, path: Path) -> None:
        path.write_bytes(b"one-million")

    monkeypatch.setattr("profiling.stages.run_stage", fake_run_stage)
    monkeypatch.setattr("profiling.r2.download_file", fake_download)
    monkeypatch.setattr(
        "profiling.phase_checks._phase3_workflow",
        lambda current, _variant, *, nRows: current.workflow.model_copy(
            update={"featureConsume": "wholeStrip"}
        ),
    )
    monkeypatch.setattr(
        "profiling.phase_checks.collect_phase3_outputs",
        lambda *_args, **_kwargs: (
            {"orderedCellDigest": "cells"},
            {"hvgs": np.array([True, False])},
        ),
    )
    monkeypatch.setattr("profiling.phase_checks.object_exists", lambda _uri: False)
    monkeypatch.setattr(
        "profiling.phase_checks.put_json_if_absent",
        lambda uri, _value: written_json.append(uri) is None,
    )
    monkeypatch.setattr(
        "profiling.phase_checks.put_bytes_if_absent",
        lambda _uri, _value: True,
    )

    result = run_phase3_variant_body(
        config,
        0,
        "currentWholeStrip",
        tmp_path,
        nRows=1_000_000,
        namespace="scale",
    )

    assert result.status == "ok", result.error
    assert result.storeUri.endswith("/1000000.zarr")
    assert result.setup["nRows"] == 1_000_000
    assert len(calls) == 10
    assert all(item["nRows"] == 1_000_000 for item in calls)
    assert (
        config.scaleVariantResultUri(
            1_000_000,
            0,
            "currentWholeStrip",
        )
        in written_json
    )


def test_pca_reduction_gate_accepts_roundoff_but_rejects_material_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_group = zarr.open_group(store=zarr.storage.MemoryStore(), mode="w")
    actual_group = zarr.open_group(store=zarr.storage.MemoryStore(), mode="w")
    reference_group.create_array(
        "data",
        data=np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
    )
    actual_group.create_array(
        "data",
        data=np.array([[2e-7, 1.0], [2.0, 3.0]], dtype=np.float32),
    )
    reference_group.create_array(
        "loadings",
        data=np.array([[0.25, 0.5], [0.75, 1.0]], dtype=np.float64),
    )
    actual_group.create_array(
        "loadings",
        data=np.array([[0.25 + 2e-14, 0.5], [0.75, 1.0]], dtype=np.float64),
    )
    groups = {
        "s3://bucket/reference.zarr": reference_group,
        "s3://bucket/actual.zarr": actual_group,
    }
    monkeypatch.setattr(
        "profiling.phase_checks._open_comparison_reduction",
        lambda uri, *_args: groups[uri],
    )
    monkeypatch.setattr(
        "profiling.phase_checks._phase3_workflow",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    reference = Phase3VariantResult(
        repetition=0,
        variant="currentWholeStrip",
        runTag="reference",
        storeUri="s3://bucket/reference.zarr",
        status="ok",
    )
    actual = Phase3VariantResult(
        repetition=0,
        variant="currentBounded",
        runTag="actual",
        storeUri="s3://bucket/actual.zarr",
        status="ok",
    )

    result = _validate_pca_reduction(
        _config(),
        reference=reference,
        actual=actual,
        nRows=1_000_000,
    )

    assert result["arrays"]["data"]["unequalValues"] == 1
    assert result["arrays"]["loadings"]["unequalValues"] == 1
    actual_group["data"][0, 0] = np.float32(1e-3)
    with pytest.raises(AssertionError, match="differs beyond tolerance"):
        _validate_pca_reduction(
            _config(),
            reference=reference,
            actual=actual,
            nRows=1_000_000,
        )


def test_comparable_summary_keeps_markers_exact_but_not_pca_bytes() -> None:
    reference = {
        "arrayDigests": {"hvgs": "hvg"},
        "artifacts": {
            "reduction": [{"digest": "pca-a"}],
            "marker_table": [{"digest": "markers"}],
        },
    }
    pca_roundoff = {
        "arrayDigests": {"hvgs": "hvg"},
        "artifacts": {
            "reduction": [{"digest": "pca-b"}],
            "marker_table": [{"digest": "markers"}],
        },
    }
    marker_change = {
        "arrayDigests": {"hvgs": "hvg"},
        "artifacts": {
            "reduction": [{"digest": "pca-a"}],
            "marker_table": [{"digest": "changed"}],
        },
    }

    assert _phase3_comparable_summary(reference) == _phase3_comparable_summary(
        pca_roundoff
    )
    assert _phase3_comparable_summary(reference) != _phase3_comparable_summary(
        marker_change
    )


def test_scale_claim_resumes_only_matching_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _memory_r2(monkeypatch)
    config = _config(runTag="scale-1m")
    original = claim_scale_comparison(
        config,
        nRows=1_000_000,
        evidenceRunTag="phase-evidence",
        baselineRunTag="phase-evidence",
    )

    resumed = claim_scale_comparison(
        config,
        nRows=1_000_000,
        evidenceRunTag="phase-evidence",
        baselineRunTag="phase-evidence",
    )

    assert resumed == original
    with pytest.raises(RuntimeError, match="does not match"):
        claim_scale_comparison(
            config,
            nRows=1_000_000,
            evidenceRunTag="other-evidence",
            baselineRunTag="phase-evidence",
        )


def _scale_variant_result(
    repetition: int,
    variant: str,
) -> Phase3VariantResult:
    stage = {
        "seconds": 10.0 + repetition,
        "operationIncrementalPeakBytes": 1_000,
        "details": {
            "shape": [45_525, 1_000_000],
            "dtype": "uint16",
            "storeOperations": {"readRequestedBytes": 100},
        },
    }
    return Phase3VariantResult.model_validate(
        {
            "repetition": repetition,
            "variant": variant,
            "runTag": f"scale-r{repetition}-{variant}",
            "storeUri": f"s3://bucket/scale-r{repetition}-{variant}.zarr",
            "status": "ok",
            "stages": {
                "writeCountsT": dict(stage),
                "findMarkers": dict(stage),
                "markHvgs": dict(stage),
                "runPca": dict(stage),
            },
            "setup": {},
            "outputs": {},
        }
    )


def test_scale_final_reports_context_without_automatic_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _memory_r2(monkeypatch)
    config = _config(runTag="scale-1m")
    pilot = ScaleBatchValidationResult(
        nRows=1_000_000,
        batch="pilot",
        repetitionStart=0,
        repetitionEnd=1,
        status="ok",
        validated=True,
        checks=("exact-transpose",),
    )
    continuation = ScaleBatchValidationResult(
        nRows=1_000_000,
        batch="continuation",
        repetitionStart=1,
        repetitionEnd=3,
        status="ok",
        validated=True,
        checks=("exact-transpose",),
    )
    _put(
        store,
        config.scaleBatchValidationUri(1_000_000, "pilot"),
        pilot.model_dump(mode="json"),
    )
    _put(
        store,
        config.scaleBatchValidationUri(1_000_000, "continuation"),
        continuation.model_dump(mode="json"),
    )
    variants = ("currentWholeStrip", "currentBounded", "candidateBounded")
    for repetition in range(3):
        for variant in variants:
            result = _scale_variant_result(repetition, variant)
            _put(
                store,
                config.scaleVariantResultUri(1_000_000, repetition, variant),
                result.model_dump(mode="json"),
            )
    baseline_summaries = {
        variant: {
            "writeMedianSeconds": 2.0,
            "markerMedianSeconds": 2.0,
            "hvgMedianSeconds": 2.0,
            "pcaMedianSeconds": 2.0,
            "peakMemoryBytes": 500,
            "hvgPeakMemoryBytes": 500,
            "usefulToRequestedBytes": 2.0,
        }
        for variant in variants
    }
    baseline_config = config.model_copy(update={"runTag": "phase-evidence"})
    baseline = PhaseWorkerResult(
        phase="phase3",
        hypothesis="H3",
        status="ok",
        checks=(),
        observations={"summaries": baseline_summaries},
    )
    _put(
        store,
        baseline_config.phaseWorkerResultUri("phase3"),
        baseline.model_dump(mode="json"),
    )

    final = finalize_scale_comparison(
        config,
        nRows=1_000_000,
        repetitions=3,
        baselineRunTag="phase-evidence",
    )

    assert final.status == "ok"
    assert final.conclusion == "measurement-complete"
    assert final.requiresReview is True
    assert final.scalingContext["_comparison"]["kind"] == "descriptive"
    assert (
        final.scalingContext["candidateBounded"]["writeMedianSeconds"]["ratio"] == 5.5
    )


def test_phase2_remote_gate_records_sweeps_and_controlled_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scarf.storage.async_execution import reset_zarr_runtime_for_tests

    reset_zarr_runtime_for_tests()
    stores: dict[str, zarr.storage.MemoryStore] = {}

    def make_store(uri: str, **_kwargs: Any) -> zarr.storage.MemoryStore:
        return stores.setdefault(uri, zarr.storage.MemoryStore())

    monkeypatch.setattr("scarf.storage.stores.make_store", make_store)
    monkeypatch.setattr("profiling.r2.storage_options", lambda _uri: {})

    result = run_phase2_worker_body(
        _config(),
        tmp_path,
        includeRemote=True,
    )

    assert result.status == "ok", result.error
    remote = result.observations["remote"]
    assert "fullShard" in remote["readGroupWidths"]
    assert len(remote["outerConcurrencySweeps"]) == 2
    assert result.observations["remoteFailure"]["complete"] is False
    assert result.observations["remoteFailure"]["writer"]["heldLedgerBytes"] == 0


def test_phase_check_cli_spawns_coordinator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    captured: dict[str, Any] = {}

    class _Target:
        def with_options(self, **options: Any) -> "_Target":
            captured["options"] = options
            return self

        def spawn(self, *args: Any) -> Any:
            captured["spawnArgs"] = args
            return SimpleNamespace(object_id="fc-phase0")

    monkeypatch.setattr(modal_app, "_load_config", lambda _path: config)
    monkeypatch.setattr(
        modal_app,
        "orchestrator_function_options",
        lambda *_args, **_kwargs: {"retries": 0, "memory": (2048, 4096)},
    )
    monkeypatch.setattr(modal_app, "phase_check_coordinator_job", _Target())
    monkeypatch.setattr(
        modal_app,
        "_print_spawned",
        lambda label, call: captured.update(label=label, call=call),
    )

    modal_app.main(
        "phase-check",
        "--config",
        "unused.toml",
        "--phase",
        "phase0",
        "--ephemeral",
    )

    assert captured["label"] == "phase_check_coordinator_job phase0"
    payload, phase = captured["spawnArgs"]
    assert payload["runTag"] == "phase0-test"
    assert phase == "phase0"
    assert captured["options"]["retries"] == 0


def test_phase_check_cli_passes_explicit_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    captured: dict[str, Any] = {}

    class _Target:
        def with_options(self, **_options: Any) -> "_Target":
            return self

        def spawn(self, *args: Any) -> Any:
            captured["spawnArgs"] = args
            return SimpleNamespace(object_id="fc-phase0-resume")

    monkeypatch.setattr(modal_app, "_load_config", lambda _path: config)
    monkeypatch.setattr(
        modal_app,
        "orchestrator_function_options",
        lambda *_args, **_kwargs: {"retries": 0},
    )
    monkeypatch.setattr(modal_app, "phase_check_coordinator_job", _Target())
    monkeypatch.setattr(modal_app, "_print_spawned", lambda *_args: None)

    modal_app.main(
        "phase-check",
        "--config",
        "unused.toml",
        "--phase",
        "phase0",
        "--ephemeral",
        "--resume",
    )

    _payload, phase, resume = captured["spawnArgs"]
    assert phase == "phase0"
    assert resume is True


def test_scale_check_cli_spawns_ephemeral_coordinator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(runTag="scale-1m")
    captured: dict[str, Any] = {}

    class _Target:
        def with_options(self, **options: Any) -> "_Target":
            captured["options"] = options
            return self

        def spawn(self, *args: Any) -> Any:
            captured["spawnArgs"] = args
            return SimpleNamespace(object_id="fc-scale-1m")

    monkeypatch.setattr(modal_app, "_load_config", lambda _path: config)
    monkeypatch.setattr(
        modal_app,
        "orchestrator_function_options",
        lambda *_args, **_kwargs: {"retries": 0},
    )
    monkeypatch.setattr(modal_app, "scale_comparison_coordinator_job", _Target())
    monkeypatch.setattr(modal_app, "_print_spawned", lambda *_args: None)

    modal_app.main(
        "scale-check",
        "--config",
        "unused.toml",
        "--size",
        "1000000",
        "--evidence-run-tag",
        "phase-evidence",
        "--baseline-run-tag",
        "phase-evidence",
        "--ephemeral",
    )

    _payload, n_rows, evidence_tag, baseline_tag = captured["spawnArgs"]
    assert n_rows == 1_000_000
    assert evidence_tag == "phase-evidence"
    assert baseline_tag == "phase-evidence"
    assert captured["options"]["retries"] == 0


def test_scale_call_receipt_recovers_existing_modal_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(runTag="scale-1m")
    recovered = object()
    monkeypatch.setattr(modal_app, "object_exists", lambda _uri: True)
    monkeypatch.setattr(
        modal_app,
        "get_json",
        lambda _uri: {"functionCallId": "fc-existing"},
    )
    monkeypatch.setattr(
        modal_app.modal.FunctionCall,
        "from_id",
        classmethod(
            lambda _cls, call_id: recovered if call_id == "fc-existing" else None
        ),
    )

    call = modal_app._spawn_or_recover_scale_call(
        config,
        nRows=1_000_000,
        operation="variant-r0-currentWholeStrip",
        spawn=lambda: pytest.fail("existing call must not be spawned again"),
    )

    assert call is recovered


def test_phase3_finalize_records_the_measured_branch() -> None:
    from profiling.phase_checks import (
        PhaseReopenResult,
        PhaseWorkerResult,
        finalize_phase3,
    )

    worker = PhaseWorkerResult(
        phase="phase3",
        hypothesis="H3",
        status="ok",
        checks=("transpose-equality",),
        observations={
            "local": {"branch": "B", "reason": "bounded-current-layout-gain"}
        },
    )
    reopen = PhaseReopenResult(
        phase="phase3",
        hypothesis="H3",
        status="ok",
        validated=True,
        checks=("branch-recorded",),
    )
    final = finalize_phase3(worker, reopen)
    assert final.branch == "B"
    assert final.nextPhase == "phase4"
