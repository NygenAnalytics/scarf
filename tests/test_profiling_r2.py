from profiling.r2 import join_uri, put_json
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
