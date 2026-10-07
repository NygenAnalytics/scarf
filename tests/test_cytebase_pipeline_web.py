"""Offline tests for the Cytebase development HTTP API served by the Modal app."""

import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from tests.fixtures_cytebase import (
    COLLECTION_ID,
    CYTEBASE_ID,
    NOW,
    FakeProgressStore,
    dataset_record,
)

modal = pytest.importorskip("modal")
FastAPI = pytest.importorskip("fastapi").FastAPI
TestClient = pytest.importorskip("fastapi.testclient").TestClient

pytestmark = pytest.mark.usefixtures("cytebase_offline")

OTHER_COLLECTION_ID = "77777777-7777-4777-8777-777777777777"
ADAMS_ID = "adams_2021_lung_airway_55555555"
BLOOD_ID = "doe_2023_blood_atlas_66666666"
RECORD_PATH = f"datasets/{CYTEBASE_ID}/dataset.json"
PROGRESS_KEY = f"fc-run:{CYTEBASE_ID}:process"
SELECTORS = {
    "cytebaseIds": None,
    "collectionId": None,
    "collectionIds": None,
    "force": False,
    "approvedDeletionPaths": [],
}
LIVE = {
    "runId": "fc-run",
    "callId": "fc-worker",
    "stage": "downloading",
    "attempt": 2,
    "heartbeatAt": "2026-01-02T03:05:00+00:00",
    "progress": {"completed": 512, "total": 1024, "unit": "bytes", "percent": 50.0},
}
API_TOKEN = "offline-test-api-token-0123456789abcdef"
AUTH = {"Cytebase-Token": API_TOKEN}
UNAUTHORIZED = {"detail": "Missing or invalid Cytebase-Token header"}
NOT_STARTED = {
    "detail": "No job with this ID was started by this API, or its record "
    "expired after 7 days without a poll"
}


def _unexpected_lookup(call_id: str) -> None:
    raise AssertionError(f"Unexpected Modal function call lookup: {call_id}")


def _poll_outcome(monkeypatch, outcome: object) -> list[tuple[str, float | None]]:
    """Make ``FunctionCall.from_id(...).get`` return or raise ``outcome``."""
    polls: list[tuple[str, float | None]] = []

    def from_id(call_id: str) -> SimpleNamespace:
        def get(timeout: float | None = None) -> object:
            polls.append((call_id, timeout))
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        return SimpleNamespace(get=get)

    monkeypatch.setattr(modal, "FunctionCall", SimpleNamespace(from_id=from_id))
    return polls


def _normalized(record: dict) -> dict:
    from scarf.cytebase.pipeline.models import DatasetRecord

    return DatasetRecord.model_validate(record).model_dump(mode="json")


def _durable_status(body: dict) -> dict:
    """The response status without ``updatedAt``, after checking that timestamp."""
    status = dict(body["status"])
    assert datetime.fromisoformat(status.pop("updatedAt")) == datetime.fromisoformat(
        NOW
    )
    return status


def _registered(hub, *records: dict) -> None:
    for record in records:
        hub.put(f"datasets/{record['cytebaseId']}/dataset.json", record)


def _recorded_jobs(runs: FakeProgressStore) -> dict[str, str]:
    """Map each call ID that the API recorded in the run store to its action."""
    jobs = {}
    for key, value in runs.values.items():
        if key.startswith("api:"):
            assert set(value) == {"action", "submittedAt"}
            assert datetime.fromisoformat(value["submittedAt"]).tzinfo is not None
            jobs[key.removeprefix("api:")] = value["action"]
    return jobs


@pytest.fixture
def progress(pipeline_app, monkeypatch) -> FakeProgressStore:
    store = FakeProgressStore()
    monkeypatch.setattr(pipeline_app, "progress_store", store)
    return store


@pytest.fixture
def runs(pipeline_app, monkeypatch) -> FakeProgressStore:
    store = FakeProgressStore()
    monkeypatch.setattr(pipeline_app, "run_store", store)
    return store


@pytest.fixture
def started_job(runs) -> str:
    """The ID of a call that the API recorded when it started it."""
    runs.put("api:fc-run", {"action": "process", "submittedAt": NOW})
    return "fc-run"


@pytest.fixture
def spawned(
    pipeline_app, fake_hub, progress, runs, monkeypatch
) -> list[tuple[str, dict]]:
    """Replace every remote dependency of the routes and record submitted jobs."""
    calls: list[tuple[str, dict]] = []

    def spawn(action: str, payload: dict) -> SimpleNamespace:
        calls.append((action, payload))
        return SimpleNamespace(object_id=f"fc-{action}")

    monkeypatch.setattr(pipeline_app, "run_pipeline", SimpleNamespace(spawn=spawn))
    monkeypatch.setattr(pipeline_app, "_storage", fake_hub.bucket)
    monkeypatch.setattr(
        modal, "FunctionCall", SimpleNamespace(from_id=_unexpected_lookup)
    )
    return calls


@pytest.fixture(scope="module")
def shared_web_app(pipeline_app):
    """One API for the module; its routes look up the patched modules per call."""
    return pipeline_app.create_web_app(API_TOKEN)


@pytest.fixture(scope="module")
def authorized_client(shared_web_app):
    with TestClient(shared_web_app, headers=AUTH) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def anonymous_client(shared_web_app):
    with TestClient(shared_web_app) as test_client:
        yield test_client


@pytest.fixture
def client(authorized_client, spawned):
    """A client that sends the API token with every request."""
    return authorized_client


@pytest.fixture
def anonymous(anonymous_client, spawned):
    """A client of the same API that sends only the headers a test passes."""
    return anonymous_client


def test_health_reports_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


_REJECTED_TOKEN_HEADERS = {
    "missing": [],
    "empty": [("Cytebase-Token", "")],
    "wrong": [("Cytebase-Token", "x" * len(API_TOKEN))],
    "prefix": [("Cytebase-Token", API_TOKEN[:-1])],
    "extended": [("Cytebase-Token", f"{API_TOKEN}x")],
    "case-changed": [("Cytebase-Token", API_TOKEN.upper())],
    "repeated": [("Cytebase-Token", API_TOKEN), ("Cytebase-Token", "x")],
    # Modal's proxy reads these; neither carries the API token.
    "bearer": [("Authorization", f"Bearer {API_TOKEN}")],
    "proxy-token-only": [("Modal-Key", "wk-offline"), ("Modal-Secret", "ws-offline")],
}


# The check runs before routing, so every rejected header is sent to one route
# that would start work, and every other route is sent no header.
@pytest.mark.parametrize(
    ("method", "path", "body", "headers"),
    [
        *(
            pytest.param(
                "POST",
                "/datasets/process",
                {"cytebaseIds": [CYTEBASE_ID], "force": True},
                headers,
                id=f"process-{name}",
            )
            for name, headers in _REJECTED_TOKEN_HEADERS.items()
        ),
        pytest.param("GET", "/health", None, [], id="health"),
        pytest.param("GET", "/collections", None, [], id="collections"),
        pytest.param(
            "POST",
            "/collections/register",
            {"collectionIds": [COLLECTION_ID]},
            [],
            id="register",
        ),
        pytest.param(
            "POST", "/datasets/process", b'{"cytebaseIds": [', [], id="bad-json"
        ),
        pytest.param("POST", "/catalog/build", None, [], id="catalog"),
        pytest.param("GET", "/jobs/fc-run", None, [], id="job"),
        pytest.param(
            "GET",
            f"/datasets/{CYTEBASE_ID}?includeReplacementPaths=true",
            None,
            [],
            id="dataset",
        ),
        pytest.param("GET", "/openapi.json", None, [], id="schema"),
        pytest.param("GET", "/missing", None, [], id="unknown-path"),
        pytest.param("DELETE", "/catalog/build", None, [], id="unsupported-method"),
    ],
)
def test_requests_without_the_api_token_are_rejected_before_routing(
    pipeline_app,
    anonymous,
    spawned,
    runs,
    started_job,
    fake_hub,
    monkeypatch,
    method,
    path,
    body,
    headers,
):
    storage, listings = [], []
    monkeypatch.setattr(pipeline_app, "_storage", lambda: storage.append(True))
    monkeypatch.setattr(
        "scarf.cytebase.pipeline.catalog.list_collection_ids",
        lambda: listings.append(True),
    )
    polls = _poll_outcome(monkeypatch, {"state": "completed"})
    _registered(fake_hub, dataset_record())
    if body is not None:
        headers = [*headers, ("Content-Type", "application/json")]
        body = body if isinstance(body, bytes) else json.dumps(body).encode()

    response = anonymous.request(method, path, headers=headers, content=body)

    assert response.status_code == 401
    assert response.json() == UNAUTHORIZED
    assert response.headers["www-authenticate"] == "Cytebase-Token"
    assert spawned == []
    assert fake_hub.calls == []
    assert storage == listings == polls == []
    assert [key for key, _ in runs.puts] == [f"api:{started_job}"]


def test_websockets_without_the_api_token_are_closed(anonymous):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect) as closed:
        with anonymous.websocket_connect("/health"):
            pass
    assert closed.value.code == 1008


@pytest.mark.parametrize(
    "token",
    [
        pytest.param("x" * 31, id="short"),
        pytest.param("x" * 32 + "\n", id="trailing-newline"),
        pytest.param("x" * 16 + " " + "x" * 16, id="inner-space"),
        pytest.param("é" * 32, id="non-ascii"),
    ],
)
def test_create_web_app_rejects_tokens_that_cannot_guard_the_api(pipeline_app, token):
    with pytest.raises(ValueError, match="at least 32 visible ASCII characters"):
        pipeline_app.create_web_app(token)


def test_a_token_of_32_visible_characters_guards_the_api(pipeline_app, spawned):
    token = "!" + "x" * 30 + "~"  # The first and last visible ASCII characters.
    with TestClient(pipeline_app.create_web_app(token)) as test_client:
        accepted = test_client.get("/health", headers={"Cytebase-Token": token})
        rejected = test_client.get("/health", headers={"Cytebase-Token": token[1:]})
    assert accepted.json() == {"ok": True}
    assert rejected.status_code == 401


@pytest.mark.parametrize("path", ["/openapi.json", "/docs", "/redoc"])
def test_schema_and_interactive_docs_are_not_served(client, spawned, path):
    response = client.get(path)
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}
    assert spawned == []


def test_deployed_endpoint_requires_modal_proxy_auth(pipeline_app):
    # Modal checks the Modal-Key and Modal-Secret headers of a proxy auth token
    # before a request reaches the app. The deployed function keeps the webhook
    # configuration that it sends at deployment, which is read here offline.
    from modal._utils.async_utils import synchronizer
    from modal_proto import api_pb2

    web_function = synchronizer._translate_in(pipeline_app.web_app)
    config = web_function._webhook_config
    assert config is not None
    assert config.type == api_pb2.WEBHOOK_TYPE_ASGI_APP
    assert config.requires_proxy_auth is True


def test_only_the_deployed_web_function_receives_the_api_secret(pipeline_app):
    # Each function keeps the secrets it sends at deployment, read here offline.
    from modal._utils.async_utils import synchronizer

    def secrets(function) -> list:
        return list(synchronizer._translate_in(function)._spec.secrets)

    api_secret = synchronizer._translate_in(pipeline_app.api_secret)
    env_secret = synchronizer._translate_in(pipeline_app.secret)
    assert secrets(pipeline_app.web_app) == [env_secret, api_secret]
    for worker in (
        pipeline_app.run_pipeline,
        pipeline_app.build_catalog,
        pipeline_app.process_dataset,
    ):
        assert secrets(worker) == [env_secret]


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/health"),
        ("PUT", "/health"),
        ("GET", "/collections/register"),
        ("DELETE", "/catalog/build"),
        ("POST", f"/datasets/{CYTEBASE_ID}"),
    ],
)
def test_unsupported_methods_are_reported_as_missing_endpoints(
    client, spawned, method, path
):
    response = client.request(method, path)
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}
    assert "allow" not in response.headers
    assert spawned == []


def test_register_submits_canonical_collection_ids(client, spawned, runs):
    response = client.post(
        "/collections/register",
        json={"collectionIds": [COLLECTION_ID.upper(), OTHER_COLLECTION_ID]},
    )
    assert response.status_code == 202
    assert response.json() == {"callId": "fc-register"}
    assert spawned == [
        ("register", {"collectionIds": [COLLECTION_ID, OTHER_COLLECTION_ID]})
    ]
    assert _recorded_jobs(runs) == {"fc-register": "register"}


@pytest.mark.parametrize(
    ("body", "error_type"),
    [
        pytest.param({}, "missing", id="missing"),
        pytest.param({"collectionIds": []}, "too_short", id="empty"),
        pytest.param({"collectionIds": ["lung"]}, "uuid_parsing", id="not-uuid"),
        pytest.param(
            {"collectionIds": [COLLECTION_ID], "force": True},
            "extra_forbidden",
            id="extra-field",
        ),
    ],
)
def test_register_rejects_invalid_requests(client, spawned, runs, body, error_type):
    response = client.post("/collections/register", json=body)
    assert response.status_code == 422
    assert [error["type"] for error in response.json()["detail"]] == [error_type]
    assert spawned == []
    assert runs.puts == []


def test_collections_lists_public_collection_ids(client, spawned, monkeypatch):
    listings = []

    def list_collection_ids() -> list[str]:
        listings.append(True)
        return [COLLECTION_ID, OTHER_COLLECTION_ID]

    monkeypatch.setattr(
        "scarf.cytebase.pipeline.catalog.list_collection_ids", list_collection_ids
    )
    response = client.get("/collections")
    assert response.status_code == 200
    assert response.json() == {"collectionIds": [COLLECTION_ID, OTHER_COLLECTION_ID]}
    assert listings == [True]
    assert spawned == []


def test_process_submits_the_request_and_links_each_selected_dataset(
    client, spawned, runs, fake_hub
):
    _registered(fake_hub, dataset_record())
    request = {
        "cytebaseIds": [CYTEBASE_ID, BLOOD_ID, CYTEBASE_ID],
        "force": True,
        "approvedDeletionPaths": [f"datasets/{CYTEBASE_ID}/scarf_ingest.json"],
    }
    response = client.post("/datasets/process", json=request)
    assert response.status_code == 202
    assert response.json() == {
        "callId": "fc-process",
        "datasets": [
            {"cytebaseId": CYTEBASE_ID, "statusUrl": f"/datasets/{CYTEBASE_ID}"},
            {"cytebaseId": BLOOD_ID, "statusUrl": f"/datasets/{BLOOD_ID}"},
        ],
    }
    assert spawned == [("process", SELECTORS | request)]
    assert _recorded_jobs(runs) == {"fc-process": "process"}


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        pytest.param(
            {"collectionId": COLLECTION_ID}, [ADAMS_ID, CYTEBASE_ID], id="collection"
        ),
        pytest.param(
            {"collectionIds": [OTHER_COLLECTION_ID]}, [BLOOD_ID], id="collections"
        ),
        pytest.param(
            {"collectionIds": [OTHER_COLLECTION_ID, COLLECTION_ID]},
            [ADAMS_ID, BLOOD_ID, CYTEBASE_ID],
            id="all-collections",
        ),
    ],
)
def test_process_expands_collection_selectors_from_registered_records(
    client, spawned, fake_hub, selector, expected
):
    _registered(
        fake_hub,
        dataset_record(),
        dataset_record(
            cytebaseId=ADAMS_ID, datasetId="55555555-5555-4555-8555-555555555555"
        ),
        dataset_record(
            cytebaseId=BLOOD_ID,
            datasetId="66666666-6666-4666-8666-666666666666",
            collectionId=OTHER_COLLECTION_ID,
        ),
    )
    response = client.post("/datasets/process", json=selector)
    assert response.status_code == 202
    assert response.json() == {
        "callId": "fc-process",
        "datasets": [
            {"cytebaseId": key, "statusUrl": f"/datasets/{key}"} for key in expected
        ],
    }
    assert spawned == [("process", SELECTORS | selector)]


@pytest.mark.parametrize(
    ("records", "body", "detail"),
    [
        pytest.param(
            [],
            {"cytebaseIds": [CYTEBASE_ID, "Lung-Atlas"]},
            "ValueError: Cytebase IDs use 1 to 80 lowercase letters, digits, "
            "or underscores",
            id="invalid-id",
        ),
        pytest.param(
            [dataset_record(), dataset_record(cytebaseId=ADAMS_ID)],
            {"collectionId": COLLECTION_ID},
            "ValueError: Multiple registered directories use the same CELLxGENE "
            "dataset ID",
            id="duplicate-dataset",
        ),
    ],
)
def test_process_rejects_selections_that_cannot_be_resolved(
    client, spawned, runs, fake_hub, records, body, detail
):
    _registered(fake_hub, *records)
    response = client.post("/datasets/process", json=body)
    assert response.status_code == 422
    assert response.json() == {"detail": detail}
    assert spawned == []
    assert runs.puts == []


@pytest.mark.parametrize(
    ("body", "error_type", "message"),
    [
        pytest.param(
            {"cytebaseIds": [CYTEBASE_ID], "collectionId": COLLECTION_ID},
            "value_error",
            "Supply exactly one of cytebaseIds, collectionId, or collectionIds",
            id="two-selectors",
        ),
        pytest.param(
            {"force": True},
            "value_error",
            "Supply exactly one of cytebaseIds, collectionId, or collectionIds",
            id="no-selector",
        ),
        pytest.param({"cytebaseIds": []}, "too_short", "", id="empty-ids"),
        pytest.param(
            {"collectionId": COLLECTION_ID, "dryRun": True},
            "extra_forbidden",
            "",
            id="extra-field",
        ),
    ],
)
def test_process_rejects_invalid_requests(
    client, spawned, runs, body, error_type, message
):
    response = client.post("/datasets/process", json=body)
    assert response.status_code == 422
    [error] = response.json()["detail"]
    assert error["type"] == error_type
    assert message in error["msg"]
    assert spawned == []
    assert runs.puts == []


def test_catalog_build_submits_a_catalog_job(client, spawned, runs):
    response = client.post("/catalog/build")
    assert response.status_code == 202
    assert response.json() == {"callId": "fc-catalog"}
    assert spawned == [("catalog", {})]
    assert _recorded_jobs(runs) == {"fc-catalog": "catalog"}


def test_submit_names_a_started_job_that_it_could_not_record(
    pipeline_app, client, spawned, monkeypatch
):
    monkeypatch.setattr(pipeline_app, "run_store", FakeProgressStore(fail=True))
    response = client.post("/catalog/build")
    assert response.status_code == 500
    assert response.json() == {
        "detail": "Job fc-catalog started, but this API could not record it, so "
        "/jobs will not report it: RuntimeError: progress store unavailable"
    }
    assert spawned == [("catalog", {})]


@pytest.mark.parametrize(
    "call_id",
    [
        pytest.param("fc-unknown", id="unknown"),
        pytest.param("fc-orchestrator", id="orchestrator-run"),
        pytest.param("current", id="run-store-key"),
    ],
)
def test_jobs_never_look_up_calls_this_api_did_not_start(
    client, runs, started_job, monkeypatch, call_id
):
    # The orchestrator records its run in the same store under another key;
    # FunctionCall would return it, so it is not polled.
    runs.put("current", {"runId": "fc-orchestrator", "state": "running"})
    polls = _poll_outcome(monkeypatch, {"state": "completed"})
    response = client.get(f"/jobs/{call_id}")
    assert response.status_code == 404
    assert response.json() == NOT_STARTED
    assert polls == []


def test_job_returns_the_finished_result_without_waiting(
    client, started_job, monkeypatch
):
    result = {"callId": "fc-run", "state": "completed", "successes": [CYTEBASE_ID]}
    polls = _poll_outcome(monkeypatch, result)
    response = client.get("/jobs/fc-run")
    assert response.status_code == 200
    assert response.json() == result
    assert polls == [("fc-run", 0)]


@pytest.mark.parametrize(
    ("error", "status_code", "body"),
    [
        pytest.param(
            modal.exception.OutputExpiredError(),
            404,
            {"detail": "Job result is missing or expired"},
            id="expired",
        ),
        pytest.param(
            modal.exception.NotFoundError("No such call"),
            404,
            {"detail": "Job result is missing or expired"},
            id="not-found",
        ),
        pytest.param(
            modal.exception.InvalidError("Malformed call ID"),
            404,
            {"detail": "Job result is missing or expired"},
            id="invalid",
        ),
        pytest.param(
            modal.exception.FunctionTimeoutError("Function timed out"),
            500,
            {"detail": "FunctionTimeoutError: Function timed out"},
            id="worker-timeout",
        ),
        pytest.param(TimeoutError(), 202, {"status": "pending"}, id="pending"),
        pytest.param(
            RuntimeError("worker crashed"),
            500,
            {"detail": "RuntimeError: worker crashed"},
            id="worker-error",
        ),
    ],
)
def test_job_maps_unfinished_and_failed_calls_to_http_responses(
    client, started_job, monkeypatch, error, status_code, body
):
    polls = _poll_outcome(monkeypatch, error)
    response = client.get("/jobs/fc-run")
    assert response.status_code == status_code
    assert response.json() == body
    assert polls == [("fc-run", 0)]


def test_job_errors_redact_credentials(client, started_job, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "plainsecret123")
    _poll_outcome(
        monkeypatch, RuntimeError("upload with plainsecret123 and hf_abcDEF123 failed")
    )
    response = client.get("/jobs/fc-run")
    assert response.status_code == 500
    assert response.json() == {
        "detail": "RuntimeError: upload with [redacted] and [redacted] failed"
    }


def test_dataset_returns_the_record_and_its_durable_status(client, fake_hub, progress):
    record = dataset_record(
        status="ready",
        stage="process",
        stageOutcome="succeeded",
        runId="fc-run",
        callId="fc-worker",
        attempt=1,
    )
    _registered(fake_hub, record)
    # A heartbeat left behind by the finished worker must not override the record.
    progress.put(PROGRESS_KEY, LIVE)
    response = client.get(f"/datasets/{CYTEBASE_ID}")
    assert response.status_code == 200
    body = response.json()
    assert body["dataset"] == _normalized(record)
    assert _durable_status(body) == {
        "status": "ready",
        "stage": "process",
        "stageOutcome": "succeeded",
        "runId": "fc-run",
        "callId": "fc-worker",
        "attempt": 1,
        "error": None,
        "needsInput": None,
    }
    assert "replacementPaths" not in body


@pytest.mark.parametrize(
    ("stored", "live"),
    [
        pytest.param({}, {}, id="no-heartbeat"),
        pytest.param(
            {f"fc-old:{CYTEBASE_ID}:process": LIVE}, {}, id="earlier-run-ignored"
        ),
        pytest.param(
            {PROGRESS_KEY: LIVE},
            {
                "stage": "downloading",
                "heartbeatAt": LIVE["heartbeatAt"],
                "progress": LIVE["progress"],
            },
            id="live-progress",
        ),
    ],
)
def test_dataset_merges_live_progress_while_running(
    client, fake_hub, progress, stored, live
):
    record = dataset_record(
        status="processing",
        stage="process",
        stageOutcome="running",
        runId="fc-run",
        callId="fc-worker",
        attempt=2,
    )
    _registered(fake_hub, record)
    for key, value in stored.items():
        progress.put(key, value)
    response = client.get(f"/datasets/{CYTEBASE_ID}")
    assert response.status_code == 200
    body = response.json()
    assert body["dataset"] == _normalized(record)
    durable = {
        "status": "processing",
        "stage": "process",
        "stageOutcome": "running",
        "runId": "fc-run",
        "callId": "fc-worker",
        "attempt": 2,
        "error": None,
        "needsInput": None,
    }
    assert _durable_status(body) == durable | live


def test_dataset_lists_generated_paths_that_need_replacement_approval(client, fake_hub):
    _registered(fake_hub, dataset_record(status="ready"))
    prefix = f"datasets/{CYTEBASE_ID}"
    generated = [
        f"{prefix}/data.zarr/zarr.json",
        f"{prefix}/metadata/obs.parquet",
        f"{prefix}/scarf_ingest.json",
    ]
    for path in [
        *generated,
        f"{prefix}/cellxgene/dataset.json",
        f"datasets/{BLOOD_ID}/data.zarr/zarr.json",
    ]:
        fake_hub.put(path, "{}")
    response = client.get(
        f"/datasets/{CYTEBASE_ID}", params={"includeReplacementPaths": "true"}
    )
    assert response.status_code == 200
    assert response.json()["replacementPaths"] == generated


@pytest.mark.parametrize(
    ("cytebase_id", "stored", "status_code", "detail"),
    [
        pytest.param(
            CYTEBASE_ID,
            None,
            404,
            f"FileNotFoundError: Dataset {CYTEBASE_ID} is not registered",
            id="unregistered",
        ),
        pytest.param(
            "Lung-Atlas",
            None,
            422,
            "ValueError: Cytebase IDs use 1 to 80 lowercase letters, digits, "
            "or underscores",
            id="invalid-id",
        ),
        pytest.param(
            CYTEBASE_ID,
            dataset_record(cytebaseId=BLOOD_ID),
            422,
            "ValueError: Dataset identity does not match its directory",
            id="identity-mismatch",
        ),
        pytest.param(
            CYTEBASE_ID,
            "[]",
            422,
            f"ValueError: Expected a JSON object at {RECORD_PATH}",
            id="not-an-object",
        ),
        pytest.param(
            CYTEBASE_ID,
            {"cytebaseId": CYTEBASE_ID},
            422,
            "ValidationError: ",
            id="invalid-record",
        ),
    ],
)
def test_dataset_reports_missing_and_invalid_records(
    client, fake_hub, cytebase_id, stored, status_code, detail
):
    if stored is not None:
        fake_hub.put(RECORD_PATH, stored)
    response = client.get(f"/datasets/{cytebase_id}")
    assert response.status_code == status_code
    assert response.json()["detail"].startswith(detail)


def test_web_app_serves_the_development_api(pipeline_app, spawned, runs, monkeypatch):
    monkeypatch.setenv("CYTEBASE_API_TOKEN", API_TOKEN)
    web = pipeline_app.web_app.local()
    assert isinstance(web, FastAPI)
    assert web.title == "Cytebase pipeline (development)"
    with TestClient(web) as test_client:
        assert test_client.get("/health").json() == UNAUTHORIZED
        assert test_client.get("/health", headers=AUTH).json() == {"ok": True}
        submitted = test_client.post("/catalog/build", headers=AUTH)
    assert submitted.json() == {"callId": "fc-catalog"}
    assert spawned == [("catalog", {})]
    assert _recorded_jobs(runs) == {"fc-catalog": "catalog"}


@pytest.mark.parametrize(
    ("token", "error", "message"),
    [
        pytest.param(
            None,
            RuntimeError,
            "Missing required environment variable: CYTEBASE_API_TOKEN",
            id="unset",
        ),
        pytest.param(
            "",
            RuntimeError,
            "Missing required environment variable: CYTEBASE_API_TOKEN",
            id="empty",
        ),
        pytest.param(
            f"{API_TOKEN}\n",
            ValueError,
            "at least 32 visible ASCII characters",
            id="trailing-newline",
        ),
    ],
)
def test_web_app_does_not_start_without_a_valid_api_token(
    pipeline_app, monkeypatch, token, error, message
):
    if token is not None:
        monkeypatch.setenv("CYTEBASE_API_TOKEN", token)
    with pytest.raises(error, match=message):
        pipeline_app.web_app.local()
