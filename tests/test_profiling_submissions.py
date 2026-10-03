from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
from obstore.store import MemoryStore

from profiling import modal_app, r2, results
from profiling.config import CORE_STAGE_ORDER
from profiling.spawn_wait import await_stage_result
from tests.test_profiling_e2e import _config, _stage_result


@pytest.fixture
def object_store(monkeypatch):
    store = MemoryStore()
    monkeypatch.setattr(
        r2, "open_r2_object", lambda uri: (store, urlsplit(uri).path.lstrip("/"))
    )
    return store


def test_submission_claim_has_one_winner(object_store):
    config = _config()
    barrier = Barrier(8)

    def claim():
        barrier.wait()
        try:
            results.claim_submission(config, 10_000, "writeCountsT", "submission")
        except FileExistsError:
            return False
        return True

    with ThreadPoolExecutor(8) as pool:
        assert sum(pool.map(lambda _: claim(), range(8))) == 1


def test_duplicate_worker_cannot_execute_or_replace_owner_result(
    object_store, monkeypatch, tmp_path
):
    config = _config()
    started = Event()
    finish = Event()
    calls = []
    monkeypatch.setattr(modal_app, "_WORK", tmp_path)

    def run(stage, **kwargs):
        calls.append(kwargs["submissionId"])
        started.set()
        assert finish.wait(10)
        return _stage_result(stage)

    monkeypatch.setattr(modal_app, "run_stage", run)
    with ThreadPoolExecutor(1) as pool:
        owner = pool.submit(
            modal_app.run_stage_job.local,
            config.model_dump(),
            10_000,
            "initializeStore",
            "testsubmission",
            True,
        )
        try:
            assert started.wait(10)
            with pytest.raises(FileExistsError, match="already claimed"):
                modal_app.run_stage_job.local(
                    config.model_dump(),
                    10_000,
                    "initializeStore",
                    "testsubmission",
                    True,
                )
        finally:
            finish.set()
        payload = owner.result(timeout=10)
    assert calls == ["testsubmission"]
    assert (
        results.load_result(
            config, 10_000, "initializeStore", submissionId="testsubmission"
        )
        == payload
    )
    assert (
        modal_app.run_stage_job.local(
            config.model_dump(), 10_000, "initializeStore", "testsubmission", True
        )
        == payload
    )
    assert calls == ["testsubmission"]


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        pytest.param(RuntimeError("worker failed"), "^worker failed$", id="error"),
        # Modal reports a failed input as an empty TimeoutError.
        pytest.param(
            TimeoutError(),
            "^Spawned call failed-call ended with status=FAILURE$",
            id="failed-input",
        ),
    ],
)
def test_recovery_never_accepts_a_previous_submission(object_store, failure, message):
    config = _config()
    r2.put_json(
        config.resultUri(10_000, "initializeStore"),
        {"submissionId": "previous", "status": "ok"},
    )

    class FailedCall:
        object_id = "failed-call"

        def get(self, timeout):
            raise failure

        def get_call_graph(self):
            from types import SimpleNamespace

            return [
                SimpleNamespace(
                    function_call_id=self.object_id, status="FAILURE", children=[]
                )
            ]

    with pytest.raises(RuntimeError, match=message):
        await_stage_result(
            config,
            10_000,
            "initializeStore",
            FailedCall(),
            submissionId="current",
            deadlineSeconds=1,
        )


def test_wait_uses_matching_completion_only(object_store):
    config = _config()
    r2.put_json(
        config.resultUri(10_000, "initializeStore"),
        {"submissionId": "previous", "status": "ok"},
    )

    class CompletedCall:
        def get(self, timeout):
            return {"submissionId": "current", "status": "error"}

    assert await_stage_result(
        config,
        10_000,
        "initializeStore",
        CompletedCall(),
        submissionId="current",
        deadlineSeconds=1,
    ) == {"submissionId": "current", "status": "error"}


def test_fresh_local_run_rejects_old_results_before_deleting_data(
    object_store, monkeypatch, tmp_path
):
    config = _config()
    work = tmp_path / f"local-{config.runTag}-10000"
    work.mkdir()
    saved = work / "existing-data"
    saved.write_text("retain")
    monkeypatch.setattr(modal_app, "_WORK", tmp_path)
    r2.put_json(
        config.resultUri(10_000, "createStore"),
        {"submissionId": "previous", "status": "ok"},
    )
    with pytest.raises(FileExistsError, match="fresh runTag"):
        modal_app.run_funnel_job.local(
            config.model_dump(), 10_000, "current", "local", ["createStore"]
        )
    assert saved.read_text() == "retain"


def test_claims_stay_at_funnel_and_job_level(object_store, monkeypatch, tmp_path):
    config = _config()
    monkeypatch.setattr(modal_app, "_WORK", tmp_path)

    def download(_uri, destination):
        Path(destination).write_bytes(b"h5ad")
        return r2.ObjectDownload(fileBytes=4, eTag=None)

    monkeypatch.setattr(modal_app, "download_file", download)
    monkeypatch.setattr(
        modal_app, "run_stage", lambda stage, **_kwargs: _stage_result(stage)
    )
    summary = modal_app.run_funnel_job.local(
        config.model_dump(mode="python"),
        10_000,
        "testsubmission",
        "r2",
        list(CORE_STAGE_ORDER),
    )
    assert summary["status"] == "ok"

    class SizeCoordinator:
        def with_options(self, **_options):
            return self

        def spawn(self, *_args):
            return SimpleNamespace(get=lambda timeout: {"stopped": False})

    monkeypatch.setattr(modal_app, "run_size_jobs", SizeCoordinator())
    monkeypatch.setattr(
        modal_app, "orchestrator_function_options", lambda *_args, **_kwargs: {}
    )
    # The funnel holds this runTag, so a stage fan-out on it is refused.
    with pytest.raises(FileExistsError, match="held by an e2e funnel"):
        modal_app.run_all_jobs.local(config.model_dump(mode="python"), "testsubmission")

    keys = [item["path"] for batch in object_store.list() for item in batch]
    assert any(key.endswith("/e2e-claim.json") for key in keys)
    assert not [key for key in keys if ".submissions/" in key or ".claim.json" in key]
    assert not [key for key in keys if "/0/" in key]


def _run_stage_job(config, stage: str, submission: str, *args):
    return modal_app.run_stage_job.local(
        config.model_dump(mode="python"), 10_000, stage, submission, *args
    )


def test_second_submission_cannot_run_a_claimed_stage(
    object_store, monkeypatch, tmp_path
):
    config = _config(runTag="shared")
    monkeypatch.setattr(modal_app, "_WORK", tmp_path)
    started = Event()
    finish = Event()
    calls = []

    def run(stage, **kwargs):
        calls.append(kwargs["submissionId"])
        started.set()
        assert finish.wait(10)
        return _stage_result(stage)

    monkeypatch.setattr(modal_app, "run_stage", run)
    with ThreadPoolExecutor(1) as pool:
        owner = pool.submit(_run_stage_job, config, "initializeStore", "first")
        try:
            assert started.wait(10)
            with pytest.raises(FileExistsError, match="claimed by submission first"):
                _run_stage_job(config, "initializeStore", "second")
        finally:
            finish.set()
        owner.result(timeout=10)
    assert calls == ["first"]

    # The owner released its claim, so a forced rerun may take the stage.
    monkeypatch.setattr(
        modal_app, "run_stage", lambda stage, **_kwargs: _stage_result(stage)
    )
    assert _run_stage_job(config, "initializeStore", "third", True)["status"] == "ok"
    keys = [item["path"] for batch in object_store.list() for item in batch]
    assert not [key for key in keys if key.endswith(".claim.json")]


def test_stage_job_refuses_a_run_tag_held_by_a_funnel(
    object_store, monkeypatch, tmp_path
):
    config = _config(runTag="funnel-owned")
    monkeypatch.setattr(modal_app, "_WORK", tmp_path)
    monkeypatch.setattr(
        modal_app,
        "run_stage",
        lambda *_args, **_kwargs: pytest.fail("a funnel-owned stage must not run"),
    )
    r2.put_json(config.e2eClaimUri(), {"runTag": config.runTag})

    with pytest.raises(FileExistsError, match="held by an e2e funnel"):
        _run_stage_job(config, "filterCells", "late")
    keys = [item["path"] for batch in object_store.list() for item in batch]
    assert not [key for key in keys if key.endswith(".claim.json")]


def test_create_store_refuses_an_existing_store_unless_forced(
    object_store, monkeypatch, tmp_path
):
    config = _config(runTag="existing-store")
    monkeypatch.setattr(modal_app, "_WORK", tmp_path)
    object_store.put(urlsplit(config.datasetUri(10_000)).path.lstrip("/"), b"h5ad")
    object_store.put(
        urlsplit(f"{config.storeUri(10_000)}/zarr.json").path.lstrip("/"), b"{}"
    )
    runs = []
    monkeypatch.setattr(
        modal_app,
        "run_stage",
        lambda stage, **kwargs: runs.append(kwargs) or _stage_result(stage),
    )

    with pytest.raises(FileExistsError, match="would replace the existing store"):
        _run_stage_job(config, "createStore", "unforced")
    assert runs == []

    payload = _run_stage_job(config, "createStore", "forced", True)
    assert runs and runs[0]["invalidateCache"] is True
    assert payload["datasetUri"] == config.datasetUri(10_000)
    assert payload["datasetBytes"] == 4
    assert payload["datasetETag"]


def test_stage_job_refuses_an_override_for_non_consume_stages(
    object_store, monkeypatch, tmp_path
):
    config = _config(runTag="consume-ab").model_copy(
        update={"storeUriOverride": "s3://bucket/existing.zarr"}
    )
    monkeypatch.setattr(modal_app, "_WORK", tmp_path)
    monkeypatch.setattr(
        modal_app,
        "run_stage",
        lambda *_args, **_kwargs: pytest.fail("an overridden store must not change"),
    )

    with pytest.raises(
        ValueError, match="only for consume stages; refusing createStore"
    ):
        _run_stage_job(config, "createStore", "wipe")
