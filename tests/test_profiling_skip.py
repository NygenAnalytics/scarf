from pathlib import Path

import pytest

from profiling.config import (
    CORE_STAGE_ORDER,
    SELECTED_STAGE_ORDER,
    ClusterSourceRef,
    CountMatrixConfig,
    WorkflowParameters,
    _normalize_raw_config,
    bind_cluster_source,
    load_profiling_config,
)
from profiling.results import result_exists
from profiling.stages import StageRunResult

_EXAMPLE_CONFIG = Path(__file__).parents[1] / "profiling" / "config.example.toml"


def test_example_config_loads():
    config = load_profiling_config(_EXAMPLE_CONFIG)
    assert config.modalEnvironmentName == "scarf_profiling"
    assert set(CORE_STAGE_ORDER) <= set(config.stageResources)
    assert config.effectiveStages == CORE_STAGE_ORDER
    assert "writeCountsT" in CORE_STAGE_ORDER
    assert CORE_STAGE_ORDER[-2:] == ("runLeiden", "findMarkers")
    assert config.datasetUri(10_000).endswith("/10000.h5ad")
    assert config.resultUri(10_000, "createStore").endswith(
        "/results/10000/createStore.json"
    )
    assert config.funnelResultUri(10_000).endswith("/results/10000/funnel.json")
    assert config.e2eClaimUri().endswith("/results/e2e-claim.json")
    leiden = config.resourcesFor("runLeiden")
    assert leiden.modalMemoryLimitMb == 32_768
    assert leiden.modalCpuLimit == 2.0
    assert config.workflow.topN == 1000
    assert config.workflow.dims == 21
    assert config.workflow.k == 11


def test_run_tag_isolates_store_and_result_uris():
    # layouts/ is gitignored, so derive runTag settings from the committed example.
    config = load_profiling_config(_EXAMPLE_CONFIG).model_copy(
        update={
            "runTag": "chunk256m",
            "countMatrix": CountMatrixConfig(
                unitBytes=256 * 1024 * 1024,
                chunkBytes=128 * 1024 * 1024,
            ),
        }
    )
    assert config.runTag == "chunk256m"
    assert config.countMatrix is not None
    assert config.countMatrix.unitBytes == 256 * 1024 * 1024
    assert config.storeUri(100_000).endswith("/stores/chunk256m/100000.zarr")
    assert config.resultUri(100_000, "markHvgs").endswith(
        "/results/chunk256m/100000/markHvgs.json"
    )


def test_fixed_resource_map_expands_the_current_funnel():
    fixed = {"placeholder": "fixed"}
    normalized = _normalize_raw_config(
        {
            "fixedResources": fixed,
        }
    )

    assert normalized["stageResources"] == {stage: fixed for stage in CORE_STAGE_ORDER}


def test_workflow_defaults_are_algorithmic_not_output_aliases():
    workflow = WorkflowParameters()
    assert workflow.topN == 1000
    assert workflow.dims == 21
    assert workflow.k == 11
    assert not hasattr(workflow, "hvgLabel")
    assert not hasattr(workflow, "umapLabel")
    assert not hasattr(workflow, "leidenLabel")
    assert not hasattr(workflow, "markerFeatures")
    assert not hasattr(workflow, "clusterLabelColumn")


def test_cluster_source_requires_an_explicit_artifact_id() -> None:
    with pytest.raises(ValueError, match="must be set together"):
        WorkflowParameters(clusterSourceUri="s3://bucket/source.zarr")
    with pytest.raises(ValueError, match="64-character lowercase hex"):
        WorkflowParameters(
            clusterSourceUri="s3://bucket/source.zarr",
            clusterSourceArtifactId="not-an-artifact-id",
        )


def test_bound_cluster_source_preserves_the_explicit_artifact() -> None:
    config = load_profiling_config(_EXAMPLE_CONFIG).model_copy(
        update={
            "clusterSources": (
                ClusterSourceRef(
                    nRows=10_000,
                    storeUri="s3://bucket/source.zarr",
                    artifactId="d" * 64,
                ),
            )
        }
    )

    workflow = bind_cluster_source(config, 10_000)

    assert workflow.clusterSourceUri == "s3://bucket/source.zarr"
    assert workflow.clusterSourceArtifactId == "d" * 64


def test_existing_error_result_is_terminal(monkeypatch) -> None:
    config = load_profiling_config(_EXAMPLE_CONFIG)
    payload = {
        "stage": "createStore",
        "nRows": 10_000,
        "status": "error",
        "error": "boom",
    }
    monkeypatch.setattr(
        "profiling.results.object_exists",
        lambda uri: uri.endswith("/results/10000/createStore.json"),
    )
    monkeypatch.setattr("profiling.results.get_json", lambda uri: payload)
    from profiling.results import existing_error_result

    failed = existing_error_result(config, 10_000, "createStore")
    assert failed == payload
    assert existing_error_result(config, 10_000, "filterCells") is None


def test_result_exists_skips_when_object_present(monkeypatch):
    config = load_profiling_config(_EXAMPLE_CONFIG)
    monkeypatch.setattr(
        "profiling.results.object_exists",
        lambda uri: uri.endswith("/results/10000/createStore.json"),
    )
    assert result_exists(config, 10_000, "createStore") is True
    assert result_exists(config, 10_000, "filterCells") is False


def test_stage_run_result_json_shape():
    result = StageRunResult(
        stage="reopenStore",
        nRows=10_000,
        status="ok",
        seconds=1.25,
        peakRssBytes=1024,
        peakCgroupBytes=2048,
        modalMemoryMb=20480,
        modalCpuRequest=4.0,
        modalCpuLimit=4.0,
        scarfMemoryBudget=12884901888,
        storeUri="s3://bucket/stores/10000.zarr",
        rssBaselineBytes=512,
        rssIncrementalPeakBytes=512,
        rssAfterBytes=768,
        cgroupCurrentBaselineBytes=1024,
        cgroupCurrentPeakBytes=2048,
        cgroupCurrentAfterBytes=1536,
        operationBaselineBytes=1024,
        operationIncrementalPeakBytes=1024,
        operationPeakSource="cgroupMemoryCurrent",
        cgroupPeakScope="operation",
    )
    payload = result.to_json()
    assert payload["stage"] == "reopenStore"
    assert payload["nRows"] == 10_000
    assert payload["status"] == "ok"
    assert payload["seconds"] == 1.25
    assert payload["modalCpuRequest"] == 4.0
    assert payload["modalCpuLimit"] == 4.0
    assert payload["rssBaselineBytes"] == 512
    assert payload["rssIncrementalPeakBytes"] == 512
    assert payload["rssAfterBytes"] == 768
    assert payload["cgroupCurrentAfterBytes"] == 1536
    assert payload["operationPeakSource"] == "cgroupMemoryCurrent"
    assert payload["cgroupPeakScope"] == "operation"


def test_selected_stage_graph_is_available_and_rejects_gaps() -> None:
    from profiling.config import ProfilingConfig

    config = load_profiling_config(_EXAMPLE_CONFIG)
    selected = ProfilingConfig.model_validate(
        {**config.model_dump(mode="python"), "stages": SELECTED_STAGE_ORDER}
    )
    assert selected.effectiveStages == SELECTED_STAGE_ORDER
    payload = config.model_dump(mode="python")
    payload["stages"] = ("filterCells", "importClusters")
    with pytest.raises(ValueError, match="requires"):
        ProfilingConfig.model_validate(payload)


def test_partial_storage_io_is_rejected() -> None:
    from profiling.config import ProfilingConfig

    payload = load_profiling_config(_EXAMPLE_CONFIG).model_dump(mode="python")
    payload["storageIo"] = {"readWorkers": 0}
    with pytest.raises(Exception):
        ProfilingConfig.model_validate(payload)
