from pathlib import Path

from profiling.r2 import download_file, join_uri, put_json, put_json_if_absent
from obstore.store import MemoryStore


def test_join_uri():
    assert (
        join_uri("s3://bucket/prefix", "10000.h5ad") == "s3://bucket/prefix/10000.h5ad"
    )
    assert join_uri("s3://bucket/prefix/", "/a/", "b") == "s3://bucket/prefix/a/b"


def test_memory_store_put_get_roundtrip(monkeypatch):
    store = MemoryStore()

    def fake_open(_uri: str):
        return store, "results/10000/createStore.json"

    monkeypatch.setattr("profiling.r2.open_r2_object", fake_open)
    put_json("s3://bucket/results/10000/createStore.json", {"status": "ok"})
    body = bytes(store.get("results/10000/createStore.json").bytes())
    assert b'"status":"ok"' in body


def test_put_json_if_absent_claims_once(monkeypatch):
    store = MemoryStore()

    def fake_open(_uri: str):
        return store, "results/e2e-claim.json"

    monkeypatch.setattr("profiling.r2.open_r2_object", fake_open)
    uri = "s3://bucket/results/e2e-claim.json"

    assert put_json_if_absent(uri, {"runTag": "first"}) is True
    assert put_json_if_absent(uri, {"runTag": "second"}) is False
    body = bytes(store.get("results/e2e-claim.json").bytes())
    assert b'"runTag":"first"' in body


def test_download_file_writes_concurrent_ranges(tmp_path: Path, monkeypatch) -> None:
    payload = b"abcdefghijklmnop"
    store = MemoryStore()
    store.put("data.bin", payload)

    def fake_open(_uri: str):
        return store, "data.bin"

    monkeypatch.setattr("profiling.r2.open_r2_object", fake_open)
    destination = tmp_path / "out.bin"
    result = download_file(
        "s3://bucket/data.bin",
        destination,
        chunkBytes=4,
        maxWorkers=4,
    )
    assert result.fileBytes == len(payload)
    assert destination.read_bytes() == payload
