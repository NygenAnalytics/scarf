import json
import subprocess
from pathlib import Path

import pytest

from profiling import leiden_worker
from profiling.config import StageResources, WorkflowParameters
from profiling.stages import (
    _monitor_child_process,
    _run_leiden_in_subprocess,
    run_stage,
)


def _resources() -> StageResources:
    return StageResources(
        modalMemoryRequestMb=32_768,
        modalMemoryLimitMb=32_768,
        modalCpuRequest=2.0,
        modalCpuLimit=2.0,
        scarfMemoryBudget=24 * 1024**3,
        workers=2,
        timeoutSeconds=82_800,
        ephemeralDiskMb=524_288,
    )


class _Store:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.arguments: dict[str, object] | None = None

    def run_leiden_clustering(self, **arguments: object) -> None:
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


def test_worker_runs_leiden(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = _Store()
    opened: dict[str, object] = {}

    def fake_open(
        storeUri: str,
        workflow: WorkflowParameters,
        resources: StageResources,
        *,
        initialize: bool,
    ) -> _Store:
        opened.update(
            storeUri=storeUri,
            workflow=workflow,
            resources=resources,
            initialize=initialize,
        )
        return store

    monkeypatch.setattr(leiden_worker, "_open_datastore", fake_open)
    request_path, status_path = _request(tmp_path, invalidateCache=True)

    leiden_worker.run_leiden_worker(request_path)

    assert opened["storeUri"] == "s3://bucket/store.zarr"
    assert opened["initialize"] is False
    assert store.arguments == {
        "from_assay": "RNA",
        "cell_key": "I",
        "feat_key": "hvgs",
        "resolution": 1.0,
        "backend": "igraph",
        "label": "leiden_cluster",
        "random_seed": 4444,
        "invalidate_cache": True,
    }
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["status"] == "ok"
    assert status["error"] is None
    assert status["inputSetupSeconds"] >= 0
    assert status["operationSeconds"] >= 0
    assert status["wholeWorkerSeconds"] >= (
        status["inputSetupSeconds"] + status["operationSeconds"]
    )


def test_worker_records_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = _Store(error=ValueError("bad graph"))
    monkeypatch.setattr(
        leiden_worker,
        "_open_datastore",
        lambda *_args, **_kwargs: store,
    )
    request_path, status_path = _request(tmp_path)

    with pytest.raises(ValueError, match="bad graph"):
        leiden_worker.run_leiden_worker(request_path)

    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["status"] == "error"
    assert status["error"] == "ValueError: bad graph"
    assert status["inputSetupSeconds"] >= 0
    assert status["operationSeconds"] >= 0
    assert status["wholeWorkerSeconds"] >= status["operationSeconds"]


def test_monitor_warns_without_terminating(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class Process:
        pid = 123
        waitCalls = 0

        def wait(self, timeout: float | None = None) -> int:
            self.waitCalls += 1
            if self.waitCalls < 3:
                if timeout is None:
                    raise AssertionError("monitor wait must use a timeout")
                raise subprocess.TimeoutExpired("leiden", timeout)
            return 0

    clock = iter((0.0, 30.0, 1_801.0))
    monkeypatch.setattr("profiling.stages.time.monotonic", lambda: next(clock))
    process = Process()

    assert (
        _monitor_child_process(
            process,  # type: ignore[arg-type]
            stageLabel="runLeiden",
            warningSeconds=1_800.0,
            pollSeconds=30.0,
        )
        == 0
    )

    output = capsys.readouterr().out
    assert process.waitCalls == 3
    assert output.count("WARNING") == 1
    assert "continuing" in output


def test_parent_starts_worker_module(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []

    class Process:
        pid = 456

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

    _run_leiden_in_subprocess(
        storeUri="s3://bucket/store.zarr",
        workflow=WorkflowParameters(),
        resources=_resources(),
        workDir=tmp_path,
        invalidateCache=True,
    )

    assert commands[0][1:3] == ["-m", "profiling.leiden_worker"]
    request = json.loads((tmp_path / "request.json").read_text(encoding="utf-8"))
    assert request["storeUri"] == "s3://bucket/store.zarr"
    assert request["invalidateCache"] is True
    assert request["workflow"]["leidenBackend"] == "igraph"


def test_run_stage_routes_leiden_to_child(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    called: dict[str, object] = {}

    def fake_child(**arguments: object) -> dict[str, object]:
        called.update(arguments)
        return {
            "inputSetupSeconds": 0.5,
            "operationSeconds": 1.5,
            "wholeWorkerSeconds": 2.25,
            "childCpuSeconds": 1.1,
            "processCpuSeconds": 1.05,
        }

    def unexpected_open(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("parent opened the datastore")

    monkeypatch.setattr("profiling.stages._run_leiden_in_subprocess", fake_child)
    monkeypatch.setattr("profiling.stages._open_datastore", unexpected_open)

    result = run_stage(
        "runLeiden",
        nRows=5_000_000,
        storeUri="s3://bucket/store.zarr",
        workflow=WorkflowParameters(),
        resources=_resources(),
        workDir=tmp_path,
        sampleIntervalSeconds=0.01,
        invalidateCache=True,
    )

    assert result.status == "ok"
    assert result.inputSetupSeconds == 0.5
    assert result.seconds == 1.5
    assert result.details is not None
    assert result.details["workerWholeSeconds"] == 2.25
    assert result.details["workerProcessCpuSeconds"] == 1.05
    assert result.childCpuSeconds == 1.1
    assert called["storeUri"] == "s3://bucket/store.zarr"
    assert called["workDir"] == tmp_path
    assert called["invalidateCache"] is True
