from pathlib import Path

from profiling.r2 import download_file, put_json, put_json_if_absent
from obstore.store import MemoryStore


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


def test_download_file_rejects_an_object_replaced_after_head(
    tmp_path: Path, monkeypatch
) -> None:
    import pytest
    from obstore.exceptions import PreconditionError

    store = MemoryStore()
    store.put("data.bin", bytes(range(1, 201)) * 5)

    class ReplacedAfterHead:
        def head(self, key):
            meta = store.head(key)
            store.put(key, bytes(range(1, 201)) * 4 + bytes(range(1, 151)))
            return meta

        def get(self, key, *, options):
            return store.get(key, options=options)

    monkeypatch.setattr(
        "profiling.r2.open_r2_object", lambda _uri: (ReplacedAfterHead(), "data.bin")
    )
    destination = tmp_path / "out.bin"

    with pytest.raises(PreconditionError):
        download_file("s3://bucket/data.bin", destination, chunkBytes=100, maxWorkers=1)
    assert not destination.exists()
    assert not list(tmp_path.iterdir())


def test_download_file_rejects_a_short_range(tmp_path: Path, monkeypatch) -> None:
    import pytest

    payload = bytes(range(1, 101)) * 10

    class ShortRanges:
        def head(self, _key):
            return {"size": len(payload), "e_tag": "etag"}

        def get(self, _key, *, options):
            start, end = options["range"]
            assert options["if_match"] == "etag"
            data = payload[start:end]
            return type("Result", (), {"bytes": lambda self: data[: len(data) // 2]})()

    monkeypatch.setattr(
        "profiling.r2.open_r2_object", lambda _uri: (ShortRanges(), "data.bin")
    )

    with pytest.raises(RuntimeError, match="returned 50 bytes, expected 100"):
        download_file(
            "s3://bucket/data.bin", tmp_path / "out.bin", chunkBytes=100, maxWorkers=1
        )
    assert not list(tmp_path.iterdir())


def test_create_only_upload_never_replaces_an_object(tmp_path: Path, monkeypatch):
    import pytest

    from profiling.r2 import delete_object, upload_file

    store = MemoryStore()
    store.put("datasets/10000.h5ad", b"prepared sample")
    monkeypatch.setattr(
        "profiling.r2.open_r2_object",
        lambda uri: (store, uri.removeprefix("s3://bucket/")),
    )
    fixture = tmp_path / "fixture.h5ad"
    fixture.write_bytes(b"fixture")

    with pytest.raises(FileExistsError, match="Refusing to replace"):
        upload_file(fixture, "s3://bucket/datasets/10000.h5ad", createOnly=True)
    assert bytes(store.get("datasets/10000.h5ad").bytes()) == b"prepared sample"

    upload_file(fixture, "s3://bucket/fixtures/10000.h5ad", createOnly=True)
    assert bytes(store.get("fixtures/10000.h5ad").bytes()) == b"fixture"
    delete_object("s3://bucket/fixtures/10000.h5ad")
    assert not [item for batch in store.list("fixtures") for item in batch]
