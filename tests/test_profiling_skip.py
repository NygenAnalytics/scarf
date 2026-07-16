from pathlib import Path

from profiling.config import (
    STAGE_ORDER,
    StorageLayout,
    WorkflowParameters,
    load_profiling_config,
)
from profiling.results import result_exists
from profiling.stages import StageRunResult

_EXAMPLE_CONFIG = Path(__file__).parents[1] / "profiling" / "config.example.toml"


def test_example_config_loads():
    config = load_profiling_config(_EXAMPLE_CONFIG)
    assert config.modalEnvironmentName == "scarf_profiling"
    assert set(config.stageResources) == set(STAGE_ORDER)
    assert config.datasetUri(10_000).endswith("/10000.h5ad")
    assert config.resultUri(10_000, "createStore").endswith(
        "/results/10000/createStore.json"
    )


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
        scarfMemoryBudget=12884901888,
        storeUri="s3://bucket/stores/10000.zarr",
    )
    payload = result.to_json()
    assert payload["stage"] == "reopenStore"
    assert payload["nRows"] == 10_000
    assert payload["status"] == "ok"
    assert payload["seconds"] == 1.25
