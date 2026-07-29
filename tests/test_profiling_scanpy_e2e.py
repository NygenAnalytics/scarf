from pathlib import Path
from typing import Any

import pytest

from profiling import scanpy_modal_app
from profiling import r2 as r2_mod
from profiling import scanpy_stages as stages_mod
from profiling.metrics import ResourceMeasurement
from profiling.scanpy_config import (
    SCANPY_STAGE_ORDER,
    ScanpyProfilingConfig,
    ScanpyStageName,
    load_scanpy_profiling_config,
)
from profiling.scanpy_stages import (
    ScanpyPipelineState,
    ScanpyStageRunResult,
    run_scanpy_e2e_funnel_body,
)

_EXAMPLE_CONFIG = (
    Path(__file__).parents[1] / "profiling" / "config.scanpy.example.toml"
)


def _config(*, runTag: str = "scanpy-e2e-test") -> ScanpyProfilingConfig:
    return load_scanpy_profiling_config(_EXAMPLE_CONFIG).model_copy(
        update={"runTag": runTag}
    )


class _Sampler:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.started = False

    def start(self) -> "_Sampler":
        self.started = True
        return self

    def stop(self) -> ResourceMeasurement:
        assert self.started
        return ResourceMeasurement(
            sampleCount=2,
            sampleIntervalSeconds=0.1,
            operationBaselineBytes=100,
            operationPeakBytes=400,
            operationIncrementalPeakBytes=300,
            operationPeakSource="cgroupMemoryCurrent",
            processTreeRssBaselineBytes=90,
            processTreeRssPeakBytes=350,
            processTreeRssIncrementalPeakBytes=260,
            processTreeRssAfterBytes=120,
            cgroupMemoryCurrentBaselineBytes=100,
            cgroupMemoryCurrentPeakBytes=400,
            cgroupMemoryCurrentAfterBytes=130,
            cgroupMemoryPeakScope="operation",
        )


def _stage_result(
    stage: ScanpyStageName,
    *,
    status: str = "ok",
    error: str | None = None,
) -> ScanpyStageRunResult:
    return ScanpyStageRunResult(
        stage=stage,
        nRows=10_000,
        status=status,
        seconds=1.0,
        peakRssBytes=350,
        peakCgroupBytes=400,
        modalMemoryMb=65_536,
        error=error,
        details={"ok": True},
    )


def _patch_funnel_io(
    monkeypatch: pytest.MonkeyPatch,
    *,
    funnel_payloads: list[dict[str, Any]],
) -> None:
    monkeypatch.setattr(stages_mod, "ResourceSampler", _Sampler)
    monkeypatch.setattr(r2_mod, "put_json_if_absent", lambda *_a, **_k: True)
    monkeypatch.setattr(
        r2_mod,
        "download_file",
        lambda _uri, destination: destination.write_bytes(b"h5ad"),
    )
    monkeypatch.setattr(
        stages_mod,
        "write_scanpy_result",
        lambda config, result: config.resultUri(result.nRows, result.stage),
    )
    monkeypatch.setattr(
        stages_mod,
        "write_scanpy_funnel_result",
        lambda _config, _n_rows, payload: (
            funnel_payloads.append(payload) or "s3://example/funnel.json"
        ),
    )


def test_scanpy_example_config_loads() -> None:
    config = _config()
    assert config.resources.modalCpuLimit == 8.0
    assert config.resources.modalMemoryLimitMb == 65_536
    assert config.workflow.nTopGenes == 2000
    assert config.workflow.nComps == 50
    assert config.workflow.targetSum == 10_000.0
    assert config.dask.nWorkers == 1
    assert config.dask.threadsPerWorker == 1
    assert config.dask.sparseChunkSize == 20_000
    assert config.dask.memoryPerWorker == "50GB"


def test_scanpy_e2e_funnel_runs_all_stages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config()
    funnel_payloads: list[dict[str, Any]] = []
    calls: list[ScanpyStageName] = []
    _patch_funnel_io(monkeypatch, funnel_payloads=funnel_payloads)

    def run_stage(stage: ScanpyStageName, **_kwargs: Any) -> ScanpyStageRunResult:
        calls.append(stage)
        return _stage_result(stage)

    monkeypatch.setattr(stages_mod, "run_scanpy_stage", run_stage)

    summary = run_scanpy_e2e_funnel_body(config, 10_000, workDir=tmp_path / "work")

    assert calls == list(SCANPY_STAGE_ORDER)
    assert summary["status"] == "ok"
    assert summary["stack"] == "scanpy"
    assert summary["completedStages"] == list(SCANPY_STAGE_ORDER)
    assert summary["modalResources"]["modalMemoryLimitMb"] == 65_536
    assert summary["modalResources"]["modalCpuLimit"] == 8.0
    assert summary["comparisonNotes"]["hvgCount"] == 2000
    assert summary["comparisonNotes"]["pcaComps"] == 50
    assert summary["comparisonNotes"]["normalizeTargetSum"] == 1e4
    assert funnel_payloads == [summary]


def test_scanpy_e2e_stops_on_stage_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config(runTag="scanpy-fail")
    funnel_payloads: list[dict[str, Any]] = []
    calls: list[ScanpyStageName] = []
    _patch_funnel_io(monkeypatch, funnel_payloads=funnel_payloads)

    def run_stage(stage: ScanpyStageName, **_kwargs: Any) -> ScanpyStageRunResult:
        calls.append(stage)
        if stage == "runPca":
            return _stage_result(stage, status="error", error="RuntimeError: boom")
        return _stage_result(stage)

    monkeypatch.setattr(stages_mod, "run_scanpy_stage", run_stage)

    summary = run_scanpy_e2e_funnel_body(config, 10_000, workDir=tmp_path / "work")

    expected = list(SCANPY_STAGE_ORDER[: SCANPY_STAGE_ORDER.index("runPca") + 1])
    assert calls == expected
    assert summary["status"] == "error"
    assert summary["failedStage"] == "runPca"
    assert summary["completedStages"] == expected[:-1]
    assert funnel_payloads[0]["error"] == "RuntimeError: boom"


def test_scanpy_e2e_entry_rejects_conflicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    monkeypatch.setattr(
        scanpy_modal_app,
        "_e2e_conflicting_uris",
        lambda *_a, **_k: ["s3://bucket/results/claim.json"],
    )
    with pytest.raises(FileExistsError, match="fresh runTag"):
        scanpy_modal_app.run_scanpy_e2e_entry(
            config.model_dump(mode="python"),
            10_000,
        )


def test_pipeline_state_close_is_idempotent() -> None:
    state = ScanpyPipelineState()
    state.close()
    state.close()
