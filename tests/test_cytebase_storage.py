"""Offline tests for the shared Cytebase bucket helpers."""

import threading
from concurrent.futures import CancelledError
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from types import SimpleNamespace

import httpx
import pytest
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError

from scarf.cytebase import _storage
from tests.fixtures_cytebase import BUCKET_ID, bucket_file

pytestmark = pytest.mark.usefixtures("cytebase_offline")

NOW = 1_700_000_000.0


def _response(status: int, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(
        status,
        headers=headers or {},
        request=httpx.Request("GET", "https://bucket.invalid/object"),
    )


def _status_error(
    status: int, headers: dict[str, str] | None = None
) -> httpx.HTTPStatusError:
    response = _response(status, headers)
    return httpx.HTTPStatusError(
        f"HTTP {status}", request=response.request, response=response
    )


def _failing(errors: list[Exception], result: object = "done"):
    calls = []

    def operation():
        calls.append(len(calls))
        if errors:
            raise errors.pop(0)
        return result

    return operation, calls


def test_dataset_prefix_accepts_catalog_ids():
    assert _storage.dataset_prefix("lung_2024_a1") == "datasets/lung_2024_a1"


@pytest.mark.parametrize("cytebase_id", ["Upper", "with-dash", "", "a" * 81, "a/b"])
def test_dataset_prefix_rejects_other_ids(cytebase_id):
    with pytest.raises(ValueError, match="lowercase letters"):
        _storage.dataset_prefix(cytebase_id)


def test_json_bytes_keeps_unicode_and_rejects_nan():
    assert _storage.json_bytes({"label": "café"}) == (
        '{\n  "label": "café"\n}\n'.encode()
    )
    with pytest.raises(ValueError):
        _storage.json_bytes({"value": float("nan")})


def test_error_message_redacts_environment_stored_and_literal_tokens(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "env-secret")
    monkeypatch.setattr(_storage, "get_token", lambda: "stored-secret")
    error = RuntimeError("env-secret stored-secret hf_AbC123 kept")
    assert _storage.error_message(error) == (
        "RuntimeError: [redacted] [redacted] [redacted] kept"
    )


def test_error_message_without_tokens():
    assert _storage.error_message(ValueError("plain")) == "ValueError: plain"


@pytest.mark.parametrize(
    ("path", "prefix", "expected"),
    [
        ("datasets/a/dataset.json", False, "datasets/a/dataset.json"),
        ("datasets/", True, "datasets/"),
        ("datasets", True, "datasets"),
        ("", True, ""),
        ("/", True, ""),
    ],
)
def test_path_accepts_exact_relative_paths(path, prefix, expected):
    assert _storage._path(path, prefix=prefix) == expected


@pytest.mark.parametrize(
    "path",
    ["", "a//b", "./a", "../a", "/a", "a/", "a*", "a?b", "a[0]", "a\\b"],
)
def test_path_rejects_ambiguous_paths(path):
    with pytest.raises(ValueError, match="exact, relative bucket object path"):
        _storage._path(path)


def test_bucket_uses_explicit_name_and_token():
    bucket = _storage.Bucket("ns/name", token="secret")
    assert bucket.bucket_id == "ns/name"
    assert bucket.root == "hf://buckets/ns/name"
    assert bucket.token == "secret"
    assert bucket.progress is None


def test_bucket_reads_environment_and_normalizes_uri(monkeypatch):
    monkeypatch.setenv("CYTEBASE_BUCKET", "  hf://buckets/ns/name.v2/  ")
    bucket = _storage.Bucket()
    assert bucket.bucket_id == "ns/name.v2"
    assert bucket.token is False


def test_bucket_uses_stored_token_when_not_supplied(monkeypatch):
    monkeypatch.setattr(_storage, "get_token", lambda: "stored")
    assert _storage.Bucket("ns/name").token == "stored"
    assert _storage.Bucket("ns/name", token=False).token is False


@pytest.mark.parametrize("bucket", [None, "noslash", "ns/na me", "hf://buckets/"])
def test_bucket_requires_a_namespace_and_name(bucket):
    with pytest.raises(ValueError, match="Supply bucket="):
        _storage.Bucket(bucket)


def test_bucket_round_trips_objects(fake_hub, tmp_path):
    bucket = fake_hub.bucket(token="secret")
    bucket.write_json("datasets/a/dataset.json", {"cytebaseId": "a"})
    source = tmp_path / "local.bin"
    source.write_bytes(b"payload")
    bucket.upload([(source, "datasets/a/object.bin")])
    assert bucket.read_json("datasets/a/dataset.json") == {"cytebaseId": "a"}
    assert bucket.read_bytes("datasets/a/object.bin") == b"payload"
    destination = tmp_path / "nested" / "copy.bin"
    bucket.download("datasets/a/object.bin", destination)
    assert destination.read_bytes() == b"payload"
    assert (
        "batch_bucket_files",
        BUCKET_ID,
        ["datasets/a/object.bin"],
        [],
        "secret",
    ) in (fake_hub.calls)


def test_bucket_reports_missing_objects_and_rejects_non_objects(fake_hub):
    bucket = fake_hub.bucket()
    assert bucket.read_bytes("missing.json") is None
    assert bucket.read_json("missing.json") is None
    fake_hub.put("list.json", "[1, 2]")
    with pytest.raises(ValueError, match="Expected a JSON object at list.json"):
        bucket.read_json("list.json")
    with pytest.raises(EntryNotFoundError):
        bucket.download("missing.json", fake_hub.root / "unused")


def test_bucket_skips_empty_batches(fake_hub):
    bucket = fake_hub.bucket()
    bucket.upload([])
    bucket.delete_exact([])
    assert fake_hub.calls == []


def test_list_files_filters_by_exact_prefix(fake_hub):
    for path in (
        "datasets/a/dataset.json",
        "datasets/a/data.zarr/zarr.json",
        "datasets/ab/dataset.json",
        "catalog/cytebase.duckdb",
    ):
        fake_hub.put(path, b"x")
    bucket = fake_hub.bucket()
    expected = ["datasets/a/data.zarr/zarr.json", "datasets/a/dataset.json"]
    assert bucket.list_files("datasets/a/") == expected
    assert bucket.list_files("datasets/a") == expected
    assert bucket.list_files("datasets/a/dataset.json") == ["datasets/a/dataset.json"]
    assert bucket.list_files("") == sorted(fake_hub.files())
    assert bucket.list_files("datasets/missing/") == []


def test_list_files_raises_when_listing_fails_part_way(fake_hub, monkeypatch):
    def partial_listing(bucket_id, prefix=None, *, recursive=None, token=None):
        yield bucket_file("datasets/a/dataset.json", 1)
        raise EntryNotFoundError("listing ended early")

    monkeypatch.setattr(_storage, "list_bucket_tree", partial_listing)
    with pytest.raises(EntryNotFoundError, match="ended early"):
        fake_hub.bucket().list_files("datasets/")


def test_list_files_rejects_unsafe_remote_names(fake_hub, monkeypatch):
    monkeypatch.setattr(
        _storage,
        "list_bucket_tree",
        lambda *args, **kwargs: [bucket_file("datasets/a/*.json", 1)],
    )
    with pytest.raises(ValueError, match="exact, relative"):
        fake_hub.bucket().list_files("datasets/")


def test_sync_store_and_delete_exact(fake_hub, tmp_path):
    store = tmp_path / "data.zarr"
    (store / "RNA").mkdir(parents=True)
    (store / "zarr.json").write_text("{}")
    (store / "RNA" / "zarr.json").write_text("{}")
    local = fake_hub.bucket()
    local.sync_store(store, "lung")
    remote = _storage.Bucket(BUCKET_ID, token=False)
    remote.sync_store(store, "other")
    assert "datasets/lung/data.zarr/RNA/zarr.json" in fake_hub.files()
    assert "datasets/other/data.zarr/zarr.json" in fake_hub.files()
    local.delete_exact(
        ["datasets/lung/data.zarr/zarr.json", "datasets/lung/data.zarr/zarr.json"]
    )
    assert fake_hub.calls[-1] == (
        "batch_bucket_files",
        BUCKET_ID,
        [],
        ["datasets/lung/data.zarr/zarr.json"],
        False,
    )
    assert "datasets/lung/data.zarr/zarr.json" not in fake_hub.files()


def test_retry_returns_the_first_success(recorded_sleeps):
    operation, calls = _failing([])
    assert _storage.retry(operation) == "done"
    assert calls == [0]
    assert recorded_sleeps == []


@pytest.mark.parametrize(
    "error", [_status_error(404), ValueError("bad input"), EntryNotFoundError("gone")]
)
def test_retry_raises_non_transient_errors_immediately(recorded_sleeps, error):
    operation, calls = _failing([error])
    with pytest.raises(type(error)):
        _storage.retry(operation)
    assert calls == [0]
    assert recorded_sleeps == []


def test_retry_backs_off_and_reports_transient_errors(recorded_sleeps):
    progress = []
    operation, calls = _failing(
        [
            httpx.ConnectError("reset"),
            _status_error(503),
            HfHubHTTPError("busy", response=_response(500)),
        ]
    )
    result = _storage.retry(
        operation, lambda stage, **kwargs: progress.append((stage, kwargs["message"]))
    )
    assert result == "done"
    assert calls == [0, 1, 2, 3]
    assert recorded_sleeps == [2.0, 4.0, 8.0]
    assert progress == [
        ("retrying_transfer", "Retry 1/3 after 2 seconds"),
        ("retrying_transfer", "Retry 2/3 after 4 seconds"),
        ("retrying_transfer", "Retry 3/3 after 8 seconds"),
    ]


def test_retry_gives_up_after_three_retries(recorded_sleeps):
    operation, calls = _failing([_status_error(502) for _ in range(4)])
    with pytest.raises(httpx.HTTPStatusError):
        _storage.retry(operation)
    assert calls == [0, 1, 2, 3]
    assert recorded_sleeps == [2.0, 4.0, 8.0]


def test_retry_reports_rate_limits(recorded_sleeps):
    progress = []
    operation, _ = _failing([_status_error(429, {"Retry-After": "7"})])
    _storage.retry(operation, lambda stage, **kwargs: progress.append(stage))
    assert progress == ["waiting_for_rate_limit"]
    assert recorded_sleeps == [7.0]


def test_retry_leaves_long_server_waits_to_the_caller(recorded_sleeps):
    operation, calls = _failing([_status_error(503, {"Retry-After": "301"})])
    with pytest.raises(httpx.HTTPStatusError):
        _storage.retry(operation)
    assert calls == [0]
    assert recorded_sleeps == []


def test_retry_stops_before_the_first_attempt(recorded_sleeps):
    stop = threading.Event()
    stop.set()
    operation, calls = _failing([])
    with pytest.raises(CancelledError, match="Transfer cancelled"):
        _storage.retry(operation, stop_event=stop)
    assert calls == []


def test_retry_stops_during_backoff(recorded_sleeps):
    stop = threading.Event()

    def operation():
        stop.set()
        raise httpx.ReadTimeout("slow")

    with pytest.raises(CancelledError) as raised:
        _storage.retry(operation, stop_event=stop)
    assert isinstance(raised.value.__cause__, httpx.ReadTimeout)
    assert recorded_sleeps == []


def test_retry_waits_on_the_stop_event_between_attempts(recorded_sleeps):
    waits = []
    stop = SimpleNamespace(is_set=lambda: False, wait=lambda delay: waits.append(delay))
    operation, calls = _failing([httpx.ConnectError("reset")])
    assert _storage.retry(operation, stop_event=stop) == "done"
    assert waits == [2.0]
    assert calls == [0, 1]


def test_retry_delay_without_a_response_uses_exponential_backoff():
    assert _storage._retry_delay(httpx.ConnectError("reset"), 0) == 2.0
    assert _storage._retry_delay(httpx.ConnectError("reset"), 2) == 8.0


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"Retry-After": "10"}, 10.0),
        ({"Retry-After": "1"}, 2.0),
        ({"Retry-After": "soon"}, 2.0),
        ({"RateLimit-Reset": str(int(NOW) + 100)}, 100.0),
        ({"RateLimit-Reset": "30"}, 30.0),
        ({"X-RateLimit-Reset": "25"}, 25.0),
        ({"RateLimit-Reset": "later"}, 2.0),
        ({"RateLimit": '"api";r=0;t=55'}, 55.0),
        ({"RateLimit": '"api";r=3;t=55'}, 2.0),
    ],
)
def test_retry_delay_honours_server_headers(recorded_sleeps, headers, expected):
    assert _storage._retry_delay(_status_error(503, headers), 0) == expected


@pytest.mark.parametrize("aware", [True, False])
def test_retry_delay_accepts_http_dates(aware):
    moment = datetime.now(UTC) + timedelta(seconds=120)
    header = format_datetime(
        moment if aware else moment.replace(tzinfo=None), usegmt=aware
    )
    delay = _storage._retry_delay(_status_error(503, {"Retry-After": header}), 0)
    assert 100 < delay <= 120
