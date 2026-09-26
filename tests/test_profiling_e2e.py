from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from profiling import modal_app, stages
from profiling.config import (
    CORE_STAGE_ORDER,
    ProfilingConfig,
    StageName,
    load_profiling_config,
)
from profiling.metrics import ResourceMeasurement
from profiling.stages import StageRunResult
from scarf.storage import ArtifactRef

_EXAMPLE_CONFIG = Path(__file__).parents[1] / "profiling" / "config.example.toml"
_HVG_REF = ArtifactRef(
    scope="assay",
    assay="RNA",
    kind="feature_selection",
    artifact_id="a" * 64,
)
_CELL_REF = ArtifactRef(
    scope="datastore",
    assay=None,
    kind="cell_selection",
    artifact_id="b" * 64,
)
_CLUSTER_REF = ArtifactRef(
    scope="assay",
    assay="RNA",
    kind="cluster_labels",
    artifact_id="c" * 64,
)


def _config(*, runTag: str = "e2e-test") -> ProfilingConfig:
    return load_profiling_config(_EXAMPLE_CONFIG).model_copy(update={"runTag": runTag})


def test_load_stage_input_refs_from_prior_stage_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    refs = {
        "filterCells": _CELL_REF,
        "markHvgs": _HVG_REF,
    }

    def load_result(_config: ProfilingConfig, _rows: int, stage: str):
        return {
            "status": "ok",
            "details": {"artifact": refs[stage].to_dict()},
        }

    monkeypatch.setattr(modal_app, "load_result", load_result)

    assert modal_app._load_stage_input_refs(
        config,
        10_000,
        "runNormalization",
    ) == {"cells": _CELL_REF, "features": _HVG_REF}


def test_load_stage_input_refs_rejects_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(modal_app, "load_result", lambda *_args: None)

    with pytest.raises(ValueError, match="filterCells stage result is unavailable"):
        modal_app._load_stage_input_refs(
            _config(),
            10_000,
            "runNormalization",
        )


def test_load_stage_input_refs_uses_bound_imported_cluster_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    loaded_stages: list[str] = []

    def load_result(_config: ProfilingConfig, _rows: int, stage: str):
        loaded_stages.append(stage)
        return {
            "status": "ok",
            "details": {"artifact": _CLUSTER_REF.to_dict()},
        }

    monkeypatch.setattr(modal_app, "load_result", load_result)
    workflow = config.workflow.model_copy(
        update={
            "clusterSourceUri": "s3://bucket/source.zarr",
            "clusterSourceArtifactId": "d" * 64,
        }
    )

    assert modal_app._load_stage_input_refs(
        config,
        10_000,
        "findMarkers",
        workflow=workflow,
    ) == {"clusters": _CLUSTER_REF}
    assert loaded_stages == ["importClusters"]


class _Sampler:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.started = False

    def start(self) -> "_Sampler":
        self.started = True
        return self

    def stop(self) -> ResourceMeasurement:
        assert self.started
        return ResourceMeasurement(
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
            memoryMaxBytes=4096,
            memoryEventsDelta={"oomKill": 0},
            cpuQuotaCores=16.0,
        )


def _stage_result(
    stage: StageName,
    *,
    status: str = "ok",
    error: str | None = None,
) -> StageRunResult:
    return StageRunResult(
        stage=stage,
        nRows=10_000,
        status=status,
        seconds=1.0,
        peakRssBytes=350,
        peakCgroupBytes=400,
        modalMemoryMb=4096,
        scarfMemoryBudget=2 * 1024**3,
        storeUri="s3://bucket/store.zarr",
        error=error,
        submissionId="testsubmission",
        utilization={"cpuPercentOfLimit": 50.0},
    )


def _run_e2e(config: ProfilingConfig) -> dict[str, Any]:
    return modal_app.run_funnel_job.local(
        config.model_dump(mode="python"),
        10_000,
        "testsubmission",
        "r2",
        list(CORE_STAGE_ORDER),
    )


def _mock_e2e_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[list[dict[str, Any]], list[StageRunResult]]:
    funnel_payloads: list[dict[str, Any]] = []
    stage_results: list[StageRunResult] = []
    monkeypatch.setattr(modal_app, "_WORK", tmp_path)
    monkeypatch.setattr(
        modal_app, "_e2e_conflicting_uris", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(modal_app, "ResourceSampler", _Sampler)
    monkeypatch.setattr(modal_app, "put_json_if_absent", lambda *_args: True)

    def download(_uri: str, destination: Path) -> None:
        destination.write_bytes(b"h5ad")

    def write_stage(_config: ProfilingConfig, result: StageRunResult) -> str:
        stage_results.append(result)
        return _config.resultUri(result.nRows, result.stage)

    def write_funnel(
        _config: ProfilingConfig,
        _n_rows: int,
        payload: dict[str, Any],
    ) -> str:
        funnel_payloads.append(payload)
        return _config.funnelResultUri(_n_rows)

    monkeypatch.setattr(modal_app, "download_file", download)
    monkeypatch.setattr(modal_app, "write_result", write_stage)
    monkeypatch.setattr(modal_app, "write_funnel_result", write_funnel)
    return funnel_payloads, stage_results


def test_e2e_funnel_runs_graph_construction_core_once_on_r2(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config()
    funnel_payloads, stage_results = _mock_e2e_dependencies(monkeypatch, tmp_path)
    calls: list[tuple[StageName, dict[str, Any]]] = []

    def run_stage(stage: StageName, **kwargs: Any) -> StageRunResult:
        calls.append((stage, kwargs))
        return _stage_result(stage)

    monkeypatch.setattr(modal_app, "run_stage", run_stage)
    monkeypatch.setattr(
        modal_app,
        "result_exists",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("e2e must not skip stage results")
        ),
    )

    summary = _run_e2e(config)

    assert sorted(stage for stage, _kwargs in calls) == sorted(CORE_STAGE_ORDER)
    assert all(
        kwargs["storeUri"] == config.storeUri(10_000) for _stage, kwargs in calls
    )
    assert calls[0][0] == "createStore"
    assert calls[0][1]["localH5adPath"].is_file()
    assert all(kwargs["localH5adPath"] is None for _stage, kwargs in calls[1:])
    assert {
        stage for stage, kwargs in calls if not kwargs["recordStoreOperations"]
    } == set(modal_app.BACKGROUND_OVERLAPS)
    assert all(kwargs["containerMemoryMb"] == 147_456 for _stage, kwargs in calls)
    assert all(kwargs["containerCpuRequest"] == 16.0 for _stage, kwargs in calls)
    assert all(kwargs["containerCpuLimit"] == 16.0 for _stage, kwargs in calls)
    assert all(kwargs["resetCgroupPeak"] is False for _stage, kwargs in calls)
    assert all(kwargs["storageIo"] is None for _stage, kwargs in calls)
    assert all(kwargs["countMatrix"] is None for _stage, kwargs in calls)
    assert all(isinstance(kwargs["session"], dict) for _stage, kwargs in calls)
    # Background stages are recorded when the first stage that needs them starts.
    assert [result.stage for result in stage_results] == [
        *CORE_STAGE_ORDER[: CORE_STAGE_ORDER.index("runUmap")],
        "runLeiden",
        "runUmap",
        "findMarkers",
    ]
    assert summary["status"] == "ok"
    assert summary["storeBackend"] == "r2"
    assert summary["funnelSeconds"] is not None
    assert summary["completedStages"] == list(CORE_STAGE_ORDER)
    assert summary["peakRssBytes"] == 350
    assert summary["peakCgroupBytes"] == 400
    assert summary["rssAfterBytes"] == 120
    assert summary["cgroupCurrentAfterBytes"] == 130
    assert summary["memoryMaxBytes"] == 4096
    assert summary["memoryEventsDelta"] == {"oomKill": 0}
    assert summary["cpuQuotaCores"] == 16.0
    assert summary["modalResources"] == {
        "modalMemoryRequestMb": 131_072,
        "modalMemoryLimitMb": 147_456,
        "modalCpuRequest": 16.0,
        "modalCpuLimit": 16.0,
        "ephemeralDiskMb": 524_288,
        "timeoutSeconds": 86_400,
    }
    assert summary["utilization"] == [
        {"stage": stage, "cpuPercentOfLimit": 50.0} for stage in CORE_STAGE_ORDER
    ]
    assert funnel_payloads == [summary]


def test_reopen_stage_preserves_initialized_store_in_shared_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    initialized_store = object()
    reopened_store = object()
    session: dict[str, Any] = {"store": initialized_store}
    opened: list[tuple[str, bool]] = []

    def open_datastore(
        store_uri: str,
        _workflow: Any,
        _resources: Any,
        *,
        initialize: bool,
        **_kwargs: Any,
    ) -> Any:
        opened.append((store_uri, initialize))
        return reopened_store

    monkeypatch.setattr(stages, "_open_datastore", open_datastore)
    monkeypatch.setattr(stages, "install_stage_zarr_runtime", lambda: None)
    monkeypatch.setattr(stages, "ResourceSampler", _Sampler)
    monkeypatch.setattr(
        "profiling.metrics.child_cpu_seconds",
        lambda: 0.0,
    )
    monkeypatch.setattr(
        "profiling.provenance.collect_run_provenance",
        lambda **_kwargs: {},
    )

    result = stages.run_stage(
        "reopenStore",
        nRows=10_000,
        storeUri=config.storeUri(10_000),
        workflow=config.workflow,
        resources=config.resourcesFor("reopenStore"),
        recordStoreOperations=False,
        session=session,
        submissionId="testsubmission",
    )

    assert result.status == "ok"
    assert opened == [(config.storeUri(10_000), False)]
    assert session["store"] is initialized_store


def test_forced_initialize_rejects_prepared_data_before_changes(tmp_path):
    import numpy as np
    import zarr

    path = str(tmp_path / "prepared.zarr")
    root = zarr.open_group(path, mode="w")
    root.create_group("RNA").attrs["prepared"] = True
    root.create_group("cellData").create_array(
        "RNA_nCounts", data=np.array([10.0, 20.0])
    )
    config = _config()
    result = stages.run_stage(
        "initializeStore",
        nRows=2,
        storeUri=path,
        workflow=config.workflow,
        resources=config.resourcesFor("initializeStore"),
        recordStoreOperations=False,
        invalidateCache=True,
        submissionId="testsubmission",
    )
    assert result.status == "error"
    assert "fresh runTag" in result.error
    np.testing.assert_array_equal(root["cellData/RNA_nCounts"][:], [10.0, 20.0])


def test_e2e_funnel_forwards_storage_io(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from profiling.config import StorageIoConfig

    policy = StorageIoConfig(
        readWorkers=4,
        writeWorkers=2,
        computeWorkers=1,
    )
    config = _config(runTag="e2e-policy").model_copy(update={"storageIo": policy})
    _mock_e2e_dependencies(monkeypatch, tmp_path)
    calls: list[dict[str, Any]] = []

    def run_stage(stage: StageName, **kwargs: Any) -> StageRunResult:
        calls.append(kwargs)
        return _stage_result(stage)

    monkeypatch.setattr(modal_app, "run_stage", run_stage)

    summary = _run_e2e(config)

    assert summary["status"] == "ok"
    assert calls
    assert all(kwargs["storageIo"] == policy for kwargs in calls)


def test_e2e_funnel_stops_and_persists_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config(runTag="e2e-failure")
    funnel_payloads, stage_results = _mock_e2e_dependencies(monkeypatch, tmp_path)
    calls: list[StageName] = []

    def run_stage(stage: StageName, **_kwargs: Any) -> StageRunResult:
        calls.append(stage)
        if stage == "runPca":
            return _stage_result(stage, status="error", error="RuntimeError: failed")
        return _stage_result(stage)

    monkeypatch.setattr(modal_app, "run_stage", run_stage)

    summary = _run_e2e(config)

    expected = list(CORE_STAGE_ORDER[: CORE_STAGE_ORDER.index("runPca") + 1])
    assert calls == expected
    assert [result.stage for result in stage_results] == expected
    assert summary["status"] == "error"
    assert summary["failedStage"] == "runPca"
    assert summary["completedStages"] == expected[:-1]
    assert funnel_payloads == [summary]


def test_e2e_funnel_rejects_empty_or_reused_run_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="non-empty runTag"):
        _run_e2e(_config(runTag=""))

    config = _config(runTag="used")
    conflict = f"{config.storeUri(10_000)}/zarr.json"
    monkeypatch.setattr(
        modal_app,
        "_e2e_conflicting_uris",
        lambda *_args, **_kwargs: [conflict],
    )
    with pytest.raises(FileExistsError, match="fresh runTag"):
        _run_e2e(config)

    monkeypatch.setattr(
        modal_app, "_e2e_conflicting_uris", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(modal_app, "put_json_if_absent", lambda *_args: False)
    with pytest.raises(FileExistsError, match="claimed concurrently"):
        _run_e2e(_config(runTag="racing"))


def test_e2e_freshness_checks_store_and_all_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    checked: list[str] = []

    def exists(uri: str) -> bool:
        checked.append(uri)
        return False

    monkeypatch.setattr(modal_app, "object_exists", exists)

    assert modal_app._e2e_conflicting_uris(config, 10_000) == []
    assert f"{config.storeUri(10_000)}/zarr.json" in checked
    assert config.e2eClaimUri() in checked
    assert config.funnelResultUri(10_000) in checked
    assert {config.resultUri(10_000, stage) for stage in CORE_STAGE_ORDER}.issubset(
        checked
    )


def test_e2e_modal_options_use_max_resources_and_disable_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    captured: dict[str, Any] = {}

    def options(
        passed_config: ProfilingConfig,
        resources: Any,
        *,
        maxContainers: int,
        retries: int,
    ) -> dict[str, Any]:
        captured.update(
            config=passed_config,
            resources=resources,
            maxContainers=maxContainers,
            retries=retries,
        )
        return {}

    monkeypatch.setattr(modal_app, "modal_function_options", options)

    result = modal_app._e2e_function_options(config)

    assert captured["config"] is config
    assert captured["maxContainers"] == 1
    assert captured["retries"] == 0
    assert result["memory"] == (131_072, 147_456)
    assert result["cpu"] == (16.0, 16.0)
    assert result["timeout"] == 86_400


def test_e2e_rejects_unavailable_dynamic_ephemeral_disk() -> None:
    config = _config()
    resources = dict(config.stageResources)
    resources["createStore"] = resources["createStore"].model_copy(
        update={"ephemeralDiskMb": 524_289}
    )
    config = config.model_copy(update={"stageResources": resources})

    with pytest.raises(ValueError, match="dynamic ephemeral_disk override"):
        modal_app._e2e_resource_envelope(config)


def test_run_e2e_cli_spawns_deployed_function_and_returns(
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
            return SimpleNamespace(object_id="fc-e2e")

    monkeypatch.setattr(modal_app, "_load_config", lambda _path: config)
    monkeypatch.setattr(
        modal_app,
        "_e2e_function_options",
        lambda _config, _stages: {"timeout": 86_400, "retries": 0},
    )

    def deployed(_config: ProfilingConfig, name: str) -> _Target:
        captured["functionName"] = name
        return _Target()

    monkeypatch.setattr(modal_app, "_deployed_function", deployed)
    monkeypatch.setattr(
        modal_app,
        "_print_spawned",
        lambda label, call: captured.update(label=label, call=call),
    )

    modal_app.main(
        "run-e2e",
        "--config",
        "unused.toml",
        "--size",
        "10000",
    )

    assert captured["functionName"] == "run_funnel_job"
    assert captured["options"] == {"timeout": 86_400, "retries": 0}
    payload, n_rows, submission_id, backend, funnel_stages = captured["spawnArgs"]
    assert submission_id
    assert payload["runTag"] == "e2e-test"
    assert n_rows == 10_000
    assert (backend, funnel_stages) == ("r2", list(CORE_STAGE_ORDER))
    assert captured["label"] == "run_funnel_job r2 10000"

    modal_app.main(
        "run-local",
        "--config",
        "unused.toml",
        "--size",
        "10000",
        "--stages",
        "createStore",
        "writeCountsT",
    )
    *_head, backend, funnel_stages = captured["spawnArgs"]
    assert (backend, funnel_stages) == ("local", ["createStore", "writeCountsT"])
    assert captured["label"] == "run_funnel_job local 10000"


def test_targeted_run_requires_force_to_overwrite_an_existing_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    captured: dict[str, Any] = {}

    class _Call:
        object_id = "fc-force"

        def get(self, *, timeout: float) -> dict[str, str]:
            captured["waitTimeout"] = timeout
            return {"status": "ok"}

    class _Target:
        def with_options(self, **options: Any) -> "_Target":
            captured["options"] = options
            return self

        def spawn(self, *args: Any) -> Any:
            captured["spawnArgs"] = args
            return _Call()

    monkeypatch.setattr(modal_app, "_load_config", lambda _path: config)
    monkeypatch.setattr(modal_app, "load_result", lambda *_args: {"status": "ok"})
    monkeypatch.setattr(
        modal_app,
        "modal_function_options",
        lambda *_args, **_kwargs: {"retries": 0},
    )
    monkeypatch.setattr(modal_app, "run_stage_job", _Target())
    monkeypatch.setattr(modal_app, "_print_spawned", lambda *_args: None)

    base_args = (
        "run",
        "--config",
        "unused.toml",
        "--size",
        "10000",
        "--stage",
        "findMarkers",
        "--ephemeral",
    )
    modal_app.main(*base_args)
    assert "spawnArgs" not in captured
    assert "waitTimeout" not in captured

    modal_app.main(*base_args, "--force")
    payload, n_rows, stage, submission_id, force = captured["spawnArgs"]
    assert submission_id
    assert payload["runTag"] == "e2e-test"
    assert n_rows == 10_000
    assert stage == "findMarkers"
    assert force is True
    assert 0 < captured["waitTimeout"] <= 20


def test_targeted_run_stops_on_an_existing_error_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    monkeypatch.setattr(modal_app, "_load_config", lambda _path: config)
    monkeypatch.setattr(modal_app, "load_result", lambda *_args: {"status": "error"})

    def unexpected_launch(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("an existing error result must not be recomputed")

    monkeypatch.setattr(modal_app, "_launch", unexpected_launch)

    with pytest.raises(SystemExit) as stopped:
        modal_app.main(
            "run",
            "--config",
            "unused.toml",
            "--size",
            "10000",
            "--stage",
            "findMarkers",
        )
    assert stopped.value.code == 1


def test_stage_zarr_runtime_preserves_active_process_ceiling():
    from scarf.storage.async_execution import ensure_zarr_host_ceiling
    from profiling.stages import install_stage_zarr_runtime

    before = ensure_zarr_host_ceiling()
    install_stage_zarr_runtime()
    assert ensure_zarr_host_ceiling() == before


def test_shared_session_counts_store_operations_per_stage(tmp_path: Path) -> None:
    from profiling.config import StageResources, WorkflowParameters
    from profiling.datasets import write_fixture_h5ad

    h5ad = write_fixture_h5ad(tmp_path / "cells.h5ad", nRows=200, nColumns=60)
    workflow = WorkflowParameters(
        filterAttrs=("RNA_nCounts", "RNA_nFeatures"),
        minFeaturesPerCell=1,
    )
    resources = StageResources(
        modalMemoryRequestMb=4096,
        modalMemoryLimitMb=4096,
        modalCpuRequest=1.0,
        modalCpuLimit=1.0,
        scarfMemoryBudget=2 * 1024**3,
        workers=1,
        timeoutSeconds=600,
        ephemeralDiskMb=524_288,
    )
    session: dict[str, Any] = {}
    operations: dict[str, dict[str, int]] = {}
    for stage in (
        "createStore",
        "writeCountsT",
        "initializeStore",
        "reopenStore",
        "filterCells",
    ):
        result = stages.run_stage(
            stage,
            submissionId="testsubmission",
            nRows=200,
            storeUri=str(tmp_path / "store.zarr"),
            workflow=workflow,
            resources=resources,
            localH5adPath=h5ad.localPath if stage == "createStore" else None,
            sampleIntervalSeconds=0.01,
            session=session,
        )
        assert result.status == "ok", result.error
        assert result.details is not None
        operations[stage] = result.details["storeOperations"]
        if stage == "initializeStore":
            initialized_store = session["store"]

    # filterCells reuses the store that initializeStore opened and still counts.
    assert session["store"] is initialized_store
    assert operations["filterCells"]["gets"] > 0
    # The local store is wrapped, so creating it records writes.
    assert operations["createStore"]["sets"] > 0
    # The shared probe restarts per stage instead of accumulating.
    assert operations["reopenStore"]["sets"] < operations["writeCountsT"]["sets"]
    assert all("keysTouched" not in counts for counts in operations.values())


def test_funnel_overlaps_background_stages_like_the_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import threading

    config = _config(runTag="e2e-overlap")
    _mock_e2e_dependencies(monkeypatch, tmp_path)
    # Each background stage returns only once the stage it overlaps has run.
    partners = {"runUmap": "runLeiden"}
    ran = {stage: threading.Event() for stage in partners.values()}
    threads: dict[StageName, threading.Thread] = {}
    finished: list[StageName] = []

    def run_stage(stage: StageName, **_kwargs: Any) -> StageRunResult:
        threads[stage] = threading.current_thread()
        if stage in ran:
            ran[stage].set()
        if stage in partners and not ran[partners[stage]].wait(30):
            raise TimeoutError(f"{partners[stage]} did not run beside {stage}")
        finished.append(stage)
        return _stage_result(stage)

    monkeypatch.setattr(modal_app, "run_stage", run_stage)

    summary = _run_e2e(config)

    assert summary["status"] == "ok"
    assert summary["completedStages"] == list(CORE_STAGE_ORDER)
    for background, partner in partners.items():
        assert threads[background] is not threading.current_thread()
        outcomes = {item["stage"]: item for item in summary["outcomes"]}
        assert partner in outcomes[background]["concurrentStages"]
        assert background in outcomes[partner]["concurrentStages"]
    # Budget-planned findMarkers waits for UMAP, and UMAP waits for its input.
    assert finished.index("runUmap") < finished.index("findMarkers")
    assert finished.index("buildEmbeddingInitialization") < finished.index("runUmap")
