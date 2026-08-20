import json
from pathlib import Path

import pytest

from profiling import paris_worker
from profiling.config import StageResources, WorkflowParameters
from profiling.stages import _run_paris_in_subprocess, run_stage


def _resources() -> StageResources:
    return StageResources(
        modalMemoryRequestMb=16_384,
        modalMemoryLimitMb=16_384,
        modalCpuRequest=2.0,
        modalCpuLimit=2.0,
        scarfMemoryBudget=12 * 1024**3,
        workers=2,
        timeoutSeconds=82_800,
        ephemeralDiskMb=524_288,
    )


class _Store:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.arguments: dict[str, object] | None = None

    def run_paris_clustering(self, **arguments: object) -> None:
        self.arguments = arguments
        if self.error is not None:
            raise self.error


def _request(
    tmpPath: Path,
    *,
    workflow: WorkflowParameters | None = None,
    invalidateCache: bool = False,
) -> tuple[Path, Path]:
    status_path = tmpPath / "status.json"
    request_path = tmpPath / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "storeUri": "s3://bucket/store.zarr",
                "workflow": (workflow or WorkflowParameters()).model_dump(mode="json"),
                "resources": _resources().model_dump(mode="json"),
                "statusPath": str(status_path),
                "invalidateCache": invalidateCache,
            }
        ),
        encoding="utf-8",
    )
    return request_path, status_path


def test_worker_auto_cut_uses_minimum_cluster_size(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = _Store()

    def fake_open(
        storeUri: str,
        workflow: WorkflowParameters,
        resources: StageResources,
        *,
        initialize: bool,
    ) -> _Store:
        assert storeUri == "s3://bucket/store.zarr"
        assert initialize is False
        return store

    monkeypatch.setattr(paris_worker, "_open_datastore", fake_open)
    request_path, status_path = _request(
        tmp_path,
        workflow=WorkflowParameters(
            parisNClusters="auto",
            parisMinClusterSize=7,
        ),
    )

    paris_worker.run_paris_worker(request_path)

    assert store.arguments == {
        "from_assay": "RNA",
        "cell_key": "I",
        "feat_key": "hvgs",
        "label": "paris_cluster",
        "n_clusters": "auto",
        "min_cluster_size": 7,
    }
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["status"] == "ok"
    assert status["error"] is None
    assert status["inputSetupSeconds"] >= 0
    assert status["operationSeconds"] >= 0
    assert status["wholeWorkerSeconds"] >= (
        status["inputSetupSeconds"] + status["operationSeconds"]
    )


def test_worker_straight_cut_uses_n_clusters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = _Store()
    monkeypatch.setattr(
        paris_worker,
        "_open_datastore",
        lambda *_args, **_kwargs: store,
    )
    request_path, status_path = _request(
        tmp_path,
        workflow=WorkflowParameters(parisNClusters=12),
        invalidateCache=True,
    )

    paris_worker.run_paris_worker(request_path)

    assert store.arguments == {
        "from_assay": "RNA",
        "cell_key": "I",
        "feat_key": "hvgs",
        "label": "paris_cluster",
        "n_clusters": 12,
        "force_recalc": True,
    }
    assert json.loads(status_path.read_text(encoding="utf-8"))["status"] == "ok"


def test_parent_starts_paris_worker_module(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []

    class Process:
        pid = 789

        def __init__(self, command: list[str]) -> None:
            commands.append(command)
            self.command = command

        def wait(self, timeout: float | None = None) -> int:
            request_path = Path(self.command[-1])
            request = json.loads(request_path.read_text(encoding="utf-8"))
            Path(request["statusPath"]).write_text(
                json.dumps({"status": "ok", "error": None}),
                encoding="utf-8",
            )
            return 0

    monkeypatch.setattr("profiling.stages.subprocess.Popen", Process)

    _run_paris_in_subprocess(
        storeUri="s3://bucket/store.zarr",
        workflow=WorkflowParameters(),
        resources=_resources(),
        workDir=tmp_path,
        invalidateCache=True,
    )

    assert commands[0][1:3] == ["-m", "profiling.paris_worker"]
    request = json.loads((tmp_path / "request.json").read_text(encoding="utf-8"))
    assert request["invalidateCache"] is True


def test_run_stage_routes_paris_to_child(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    called: dict[str, object] = {}

    def fake_child(**arguments: object) -> None:
        called.update(arguments)

    def unexpected_open(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("parent opened the datastore")

    monkeypatch.setattr("profiling.stages._run_paris_in_subprocess", fake_child)
    monkeypatch.setattr("profiling.stages._open_datastore", unexpected_open)

    result = run_stage(
        "runClustering",
        nRows=5_000_000,
        storeUri="s3://bucket/store.zarr",
        workflow=WorkflowParameters(),
        resources=_resources(),
        workDir=tmp_path,
        sampleIntervalSeconds=0.01,
        invalidateCache=True,
    )

    assert result.status == "ok"
    assert called["storeUri"] == "s3://bucket/store.zarr"
    assert called["workDir"] == tmp_path
    assert called["invalidateCache"] is True
