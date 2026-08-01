from pathlib import Path

from profiling.config import (
    CORE_STAGE_ORDER,
    StorageLayout,
    WorkflowParameters,
    _normalize_raw_config,
    load_profiling_config,
)
from profiling.results import result_exists
from profiling.stages import StageRunResult

_EXAMPLE_CONFIG = Path(__file__).parents[1] / "profiling" / "config.example.toml"


def test_example_config_loads():
    config = load_profiling_config(_EXAMPLE_CONFIG)
    assert config.modalEnvironmentName == "scarf_profiling"
    assert set(config.stageResources) == set(CORE_STAGE_ORDER)
    assert config.effectiveStages == CORE_STAGE_ORDER
    assert "writeCountsT" in CORE_STAGE_ORDER
    assert "runClustering" in CORE_STAGE_ORDER
    assert config.datasetUri(10_000).endswith("/10000.h5ad")
    assert config.resultUri(10_000, "createStore").endswith(
        "/results/10000/createStore.json"
    )
    assert config.funnelResultUri(10_000).endswith("/results/10000/funnel.json")
    assert config.e2eClaimUri().endswith("/results/e2e-claim.json")
    leiden = config.resourcesFor("runLeiden")
    assert leiden.modalMemoryLimitMb == 32_768
    assert leiden.modalCpuLimit == 2.0


def test_run_tag_isolates_store_and_result_uris():
    # layouts/ is gitignored, so derive runTag settings from the committed example.
    config = load_profiling_config(_EXAMPLE_CONFIG).model_copy(
        update={
            "runTag": "chunk256m",
            "storageLayout": StorageLayout(targetChunkBytes=256 * 1024 * 1024),
        }
    )
    assert config.runTag == "chunk256m"
    assert config.storageLayout.targetChunkBytes == 256 * 1024 * 1024
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


def test_marker_group_key_matches_leiden_column():
    workflow = WorkflowParameters()
    assert workflow.resolvedMarkerGroupKey == "RNA_leiden_cluster"
    assert workflow.resolvedHvgKey == "I__hvgs"


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
