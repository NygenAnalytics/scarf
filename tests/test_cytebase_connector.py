"""Offline tests for opening and mounting published Cytebase stores."""

import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest
import zarr
from zarr.abc.store import RangeByteRequest
from zarr.core.buffer import default_buffer_prototype
from zarr.storage import FsspecStore

from scarf.cytebase import connector
from tests.fixtures_cytebase import (
    COLLECTION_ID,
    CYTEBASE_ID,
    DATASET_ID,
    NOW,
    VERSION_ID,
)

pytestmark = pytest.mark.usefixtures("cytebase_offline")

ROOT = "/bucket"
URI = f"{ROOT}/datasets/{CYTEBASE_ID}/data.zarr"
CHECKSUM = "a" * 64
RECORD_PATH = f"datasets/{CYTEBASE_ID}/dataset.json"


def _record() -> dict:
    return {
        "cytebaseId": CYTEBASE_ID,
        "status": "ready",
        "processedVersionId": VERSION_ID,
        "latestVersionId": VERSION_ID,
        "datasetId": DATASET_ID.upper(),
        "collectionId": COLLECTION_ID,
        "zarrUri": URI,
        "inspection": {
            "sourceSha256": CHECKSUM,
            "datasetId": DATASET_ID,
            "collectionId": COLLECTION_ID,
            "datasetVersionId": VERSION_ID,
            "nObs": 6,
            "nVars": 5,
        },
        "buildReceipt": {
            "datasetVersionId": VERSION_ID,
            "sourceSha256": CHECKSUM,
            "zarrUri": URI,
            "verifiedAt": NOW,
            "verification": {
                "countsTMatches": True,
                "sourceSampleMatches": True,
                "nObs": 6,
                "nVars": 5,
                "countsTShape": [5, 6],
                "countsDtype": "<i4",
            },
        },
    }


def _storage(record: dict | None) -> SimpleNamespace:
    return SimpleNamespace(root=ROOT, token=False, read_json=lambda path: record)


def _mutated(change) -> dict:
    record = _record()
    change(record)
    return record


def _identity_of(ready) -> dict:
    return connector._identity(ready.bucket, ready.cytebase_id)


def _file_contents(directory) -> dict:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


def test_identity_returns_the_verified_build():
    assert connector._identity(_storage(_record()), CYTEBASE_ID) == {
        "cytebaseId": CYTEBASE_ID,
        "datasetId": DATASET_ID,
        "processedVersionId": VERSION_ID,
        "sourceSha256": CHECKSUM,
        "zarrUri": URI,
        "verifiedAt": NOW,
        "nObs": 6,
        "nVars": 5,
        "countsDtype": "int32",
    }


def test_identity_requires_a_registered_record_at_its_own_path():
    with pytest.raises(KeyError, match="No dataset is registered"):
        connector._identity(_storage(None), CYTEBASE_ID)
    with pytest.raises(ValueError, match="identity does not match its path"):
        connector._identity(
            _storage(_mutated(lambda r: r.update(cytebaseId="other"))), CYTEBASE_ID
        )


@pytest.mark.parametrize(
    "change",
    [
        {"status": "processing"},
        {"processedVersionId": None},
        {"processedVersionId": "55555555-5555-4555-8555-555555555555"},
    ],
)
def test_identity_requires_the_registered_version_to_be_ready(change):
    with pytest.raises(RuntimeError, match="is not ready for its registered version"):
        connector._identity(_storage(_mutated(lambda r: r.update(change))), CYTEBASE_ID)


@pytest.mark.parametrize("field", ["inspection", "buildReceipt"])
def test_identity_requires_inspection_and_receipt(field):
    record = _mutated(lambda r: r.update({field: None}))
    with pytest.raises(ValueError, match="no verified inspection and build receipt"):
        connector._identity(_storage(record), CYTEBASE_ID)


def _set(path: str, value):
    def change(record: dict) -> None:
        *parents, leaf = path.split(".")
        target = record
        for parent in parents:
            target = target[parent]
        target[leaf] = value

    return change


@pytest.mark.parametrize(
    "change",
    [
        _set("inspection.sourceSha256", "A" * 64),
        _set("inspection.sourceSha256", None),
        _set("inspection.datasetId", "55555555-5555-4555-8555-555555555555"),
        _set("inspection.collectionId", "other"),
        _set("inspection.datasetVersionId", "other"),
        _set("buildReceipt.datasetVersionId", "other"),
        _set("buildReceipt.sourceSha256", "b" * 64),
        _set("buildReceipt.zarrUri", "/elsewhere"),
        _set("zarrUri", "/elsewhere"),
        _set("buildReceipt.verifiedAt", ""),
        _set("buildReceipt.verification", None),
        _set("buildReceipt.verification.countsTMatches", False),
        _set("buildReceipt.verification.sourceSampleMatches", False),
        _set("buildReceipt.verification.nObs", 7),
        _set("buildReceipt.verification.nVars", 4),
        _set("buildReceipt.verification.countsTShape", [6, 5]),
        _set("buildReceipt.verification.countsDtype", 4),
    ],
)
def test_identity_rejects_provenance_that_does_not_match(change):
    with pytest.raises(ValueError, match="provenance does not match"):
        connector._identity(_storage(_mutated(change)), CYTEBASE_ID)


@pytest.mark.parametrize("dtype", ["bool", "<U4", "complex64"])
def test_identity_requires_real_numeric_counts(dtype):
    record = _mutated(_set("buildReceipt.verification.countsDtype", dtype))
    with pytest.raises(ValueError, match="real numeric dtype"):
        connector._identity(_storage(record), CYTEBASE_ID)


def test_options_apply_cytebase_defaults():
    assert connector._options({}, mode="r") == {
        "min_features_per_cell": -1,
        "default_assay": "RNA",
    }
    assert connector._options(
        {"zarr_mode": "r+", "min_features_per_cell": 10, "nthreads": 2}, mode="r+"
    ) == {"min_features_per_cell": 10, "default_assay": "RNA", "nthreads": 2}


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"zarr_loc": "x"}, TypeError, "zarr_loc is resolved by Catalog"),
        ({"storage_options": {}}, TypeError, "storage_options is resolved"),
        ({"zarr_mode": "r+"}, ValueError, "requires zarr_mode='r'"),
        ({"min_features_per_cell": 10}, ValueError, "min_features_per_cell=-1"),
    ],
)
def test_options_reject_catalog_managed_arguments(options, error, message):
    with pytest.raises(error, match=message):
        connector._options(options, mode="r")


def _opened(**changes) -> SimpleNamespace:
    values = {
        "cells": 6,
        "feats": 5,
        "counts": ((6, 5), np.int32),
        "counts_t": ((5, 6), np.int32),
    } | changes
    counts_t = values["counts_t"]
    assay = SimpleNamespace(
        feats=SimpleNamespace(N=values["feats"]),
        rawData=SimpleNamespace(shape=values["counts"][0], dtype=values["counts"][1]),
        rawDataT=None
        if counts_t is None
        else SimpleNamespace(shape=counts_t[0], dtype=counts_t[1]),
    )
    return SimpleNamespace(
        cells=SimpleNamespace(N=values["cells"]), get_assay=lambda name: assay
    )


EXPECTED = {"nObs": 6, "nVars": 5, "countsDtype": "int32"}


def test_verify_opened_accepts_matching_arrays():
    connector._verify_opened(_opened(), EXPECTED)


@pytest.mark.parametrize(
    "changes",
    [
        {"cells": 7},
        {"feats": 4},
        {"counts": ((6, 4), np.int32)},
        {"counts_t": None},
        {"counts_t": ((6, 5), np.int32)},
        {"counts": ((6, 5), np.float32)},
        {"counts_t": ((5, 6), np.float32)},
    ],
)
def test_verify_opened_rejects_mismatched_arrays(changes):
    with pytest.raises(ValueError, match="differs from the committed build receipt"):
        connector._verify_opened(_opened(**changes), EXPECTED)


class _Store:
    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


def test_close_closes_separate_matrix_stores_once():
    root = _Store()
    matrix = _Store()
    connector._close(SimpleNamespace(z=SimpleNamespace(store=root)))
    connector._close(
        SimpleNamespace(
            z=SimpleNamespace(store=root), _matrix_z=SimpleNamespace(store=root)
        )
    )
    connector._close(
        SimpleNamespace(
            z=SimpleNamespace(store=root), _matrix_z=SimpleNamespace(store=matrix)
        )
    )
    assert (root.closed, matrix.closed) == (3, 1)


def test_read_store_removes_repeated_listing_entries(tmp_path, monkeypatch):
    async def repeated(self, prefix):
        for name in ["a", "b", "a", "c", "b"]:
            yield name

    monkeypatch.setattr(FsspecStore, "list_dir", repeated)
    store = connector._HfReadStore.from_url(
        str(tmp_path), read_only=True, storage_options={"token": False}
    )

    async def names() -> list[str]:
        return [name async for name in store.list_dir("")]

    assert asyncio.run(names()) == ["a", "b", "c"]


@pytest.mark.parametrize("value", [b'{"node_type": "group"}', None])
@pytest.mark.parametrize("key", ["zarr.json", "RNA/.zattrs"])
def test_read_store_caches_metadata_per_open(tmp_path, monkeypatch, key, value):
    calls = []
    published = value
    prototype = default_buffer_prototype()

    async def read(self, key, prototype, byte_range=None):
        calls.append(key)
        return None if published is None else prototype.buffer.from_bytes(published)

    monkeypatch.setattr(FsspecStore, "get", read)
    first = connector._HfReadStore.from_url(str(tmp_path), read_only=True)
    second = connector._HfReadStore.from_url(str(tmp_path), read_only=True)

    async def check():
        nonlocal published
        initial = await first.get(key, prototype)
        published = b'{"node_type": "group", "attributes": {"new": true}}'
        cached = await first.get(key, prototype)
        assert (None if initial is None else initial.to_bytes()) == value
        assert (None if cached is None else cached.to_bytes()) == value
        assert (await second.get(key, prototype)).to_bytes() == published
        first.close()
        assert (await first.get(key, prototype)).to_bytes() == published

    try:
        asyncio.run(check())
        assert calls == [key, key, key]
    finally:
        first.close()
        second.close()


def test_read_store_shares_fetches_without_serializing_other_keys(
    tmp_path, monkeypatch
):
    calls = []
    prototype = default_buffer_prototype()
    store = connector._HfReadStore.from_url(str(tmp_path), read_only=True)

    async def check():
        started, release = asyncio.Event(), asyncio.Event()

        async def read(self, key, prototype, byte_range=None):
            calls.append(key)
            if len(calls) == 2:
                started.set()
            await release.wait()
            return prototype.buffer.from_bytes(b"metadata")

        monkeypatch.setattr(FsspecStore, "get", read)
        cancelled = asyncio.create_task(store.get("zarr.json", prototype))
        shared = asyncio.create_task(store.get("zarr.json", prototype))
        other = asyncio.create_task(store.get("RNA/zarr.json", prototype))
        await asyncio.wait_for(started.wait(), timeout=5)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        release.set()
        results = await asyncio.gather(shared, other)
        assert [result.to_bytes() for result in results] == [b"metadata"] * 2
        assert (await store.get("zarr.json", prototype)).to_bytes() == b"metadata"

    try:
        asyncio.run(check())
        assert sorted(calls) == ["RNA/zarr.json", "zarr.json"]
    finally:
        store.close()


def test_read_store_retries_failed_metadata_fetches(tmp_path, monkeypatch):
    calls = []
    prototype = default_buffer_prototype()

    async def read(self, key, prototype, byte_range=None):
        calls.append(key)
        if len(calls) == 1:
            raise OSError("Temporary transfer failure")
        return prototype.buffer.from_bytes(b"metadata")

    monkeypatch.setattr(FsspecStore, "get", read)
    store = connector._HfReadStore.from_url(str(tmp_path), read_only=True)

    async def check():
        with pytest.raises(OSError, match="Temporary transfer failure"):
            await store.get("zarr.json", prototype)
        for _ in range(2):
            assert (await store.get("zarr.json", prototype)).to_bytes() == b"metadata"

    try:
        asyncio.run(check())
        assert calls == ["zarr.json", "zarr.json"]
    finally:
        store.close()


def test_read_store_handles_failure_after_its_reader_is_cancelled(
    tmp_path, monkeypatch
):
    prototype = default_buffer_prototype()
    store = connector._HfReadStore.from_url(str(tmp_path), read_only=True)
    errors = []

    async def check():
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda loop, context: errors.append(context))
        started, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def read(self, key, prototype, byte_range=None):
            if not started.is_set():
                started.set()
                await release.wait()
                loop.call_soon(finished.set)
                raise OSError("Abandoned transfer failed")
            return prototype.buffer.from_bytes(b"metadata")

        monkeypatch.setattr(FsspecStore, "get", read)
        reader = asyncio.create_task(store.get("zarr.json", prototype))
        await asyncio.wait_for(started.wait(), timeout=5)
        reader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reader
        release.set()
        await asyncio.wait_for(finished.wait(), timeout=5)
        assert (await store.get("zarr.json", prototype)).to_bytes() == b"metadata"

    try:
        asyncio.run(check())
        assert errors == []
    finally:
        store.close()


@pytest.mark.parametrize(
    ("key", "byte_range", "read_only"),
    [
        ("RNA/counts/c/0/0", None, True),
        ("RNA/zarr.json", RangeByteRequest(0, 4), True),
        ("zarr.json", None, False),
    ],
)
def test_read_store_does_not_cache_chunks_ranges_or_writable_metadata(
    tmp_path, monkeypatch, key, byte_range, read_only
):
    calls = []
    prototype = default_buffer_prototype()

    async def read(self, key, prototype, byte_range=None):
        calls.append((key, byte_range))
        return prototype.buffer.from_bytes(str(len(calls)).encode())

    monkeypatch.setattr(FsspecStore, "get", read)
    store = connector._HfReadStore.from_url(str(tmp_path), read_only=read_only)

    async def check():
        first = await store.get(key, prototype, byte_range)
        second = await store.get(key, prototype, byte_range)
        assert first.to_bytes() == b"1"
        assert second.to_bytes() == b"2"

    try:
        asyncio.run(check())
        assert calls == [(key, byte_range), (key, byte_range)]
    finally:
        store.close()


def test_open_dataset_reads_the_published_store_without_writing(ready_dataset):
    before = _file_contents(ready_dataset.store)
    datastore = connector.open_dataset(ready_dataset.bucket, CYTEBASE_ID)
    try:
        assert datastore.zw.read_only
        assert datastore.cells.N == 6
        assert {"cell_type", "donor_id", "is_primary_data"} <= set(
            datastore.cells.columns
        )
        assert np.dtype(datastore.RNA.rawData.dtype) == np.int32
    finally:
        connector._close(datastore)
    assert _file_contents(ready_dataset.store) == before


def test_open_dataset_closes_the_store_when_the_datastore_fails(
    ready_dataset, monkeypatch
):
    created = []
    original = connector._HfReadStore.from_url

    def from_url(*args, **kwargs):
        created.append(original(*args, **kwargs))
        return created[-1]

    monkeypatch.setattr(connector._HfReadStore, "from_url", from_url)
    with pytest.raises(TypeError, match="bogus_option"):
        connector.open_dataset(ready_dataset.bucket, CYTEBASE_ID, bogus_option=True)
    assert len(created) == 1
    assert not created[0]._is_open


def test_open_dataset_rejects_a_receipt_that_disagrees_with_the_store(
    ready_dataset, fake_hub
):
    record = fake_hub.read_json(RECORD_PATH)
    record["inspection"]["nObs"] = 7
    record["buildReceipt"]["verification"] |= {"nObs": 7, "countsTShape": [5, 7]}
    fake_hub.put(RECORD_PATH, record)
    with pytest.raises(ValueError, match="differs from the committed build receipt"):
        connector.open_dataset(ready_dataset.bucket, CYTEBASE_ID)


def _change_record_on_read(hub, read_number: int, change) -> list[str]:
    reads: list[str] = []

    def hook(bucket_id, remote):
        if remote == RECORD_PATH:
            reads.append(remote)
            if len(reads) == read_number:
                change()

    hub.hooks["download_bucket_files"] = [hook]
    return reads


def _republish(hub):
    def change() -> None:
        record = hub.read_json(RECORD_PATH)
        record["buildReceipt"]["verifiedAt"] = "2027-01-01T00:00:00+00:00"
        hub.put(RECORD_PATH, record)

    return change


def test_open_dataset_fails_when_publication_changes_while_opening(
    ready_dataset, fake_hub
):
    reads = _change_record_on_read(fake_hub, 2, _republish(fake_hub))
    with pytest.raises(RuntimeError, match="changed while opening"):
        connector.open_dataset(ready_dataset.bucket, CYTEBASE_ID)
    assert len(reads) == 2


def test_mount_dataset_creates_then_reopens_a_pinned_analysis(ready_dataset, tmp_path):
    target = tmp_path / "analysis.zarr"
    sidecar = tmp_path / "analysis.zarr.cytebase.json"
    datastore = connector.mount_dataset(ready_dataset.bucket, CYTEBASE_ID, target)
    try:
        assert not datastore.zw.read_only
        assert datastore.cells.N == 6
    finally:
        connector._close(datastore)
    assert json.loads(sidecar.read_text()) == _identity_of(ready_dataset)
    reopened = connector.mount_dataset(
        ready_dataset.bucket, CYTEBASE_ID, str(target), zarr_mode="r+"
    )
    try:
        assert reopened.cells.N == 6
    finally:
        connector._close(reopened)


def test_mount_dataset_requires_a_local_destination(ready_dataset):
    with pytest.raises(ValueError, match="require a local destination"):
        connector.mount_dataset(ready_dataset.bucket, CYTEBASE_ID, "s3://bucket/x")


def test_mount_dataset_refuses_unreceipted_destinations(ready_dataset, tmp_path):
    existing = tmp_path / "existing.zarr"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="without a Cytebase mount receipt"):
        connector.mount_dataset(ready_dataset.bucket, CYTEBASE_ID, existing)
    orphan = tmp_path / "orphan.zarr"
    (tmp_path / "orphan.zarr.cytebase.json").write_text("{}")
    with pytest.raises(FileExistsError, match="receipt already exists"):
        connector.mount_dataset(ready_dataset.bucket, CYTEBASE_ID, orphan)


def test_mount_dataset_refuses_a_receipt_for_another_build(ready_dataset, tmp_path):
    target = tmp_path / "analysis.zarr"
    target.mkdir()
    (tmp_path / "analysis.zarr.cytebase.json").write_text('{"cytebaseId": "old"}')
    with pytest.raises(ValueError, match="refers to a different dataset build"):
        connector.mount_dataset(ready_dataset.bucket, CYTEBASE_ID, target)


@pytest.mark.parametrize("attributes", [{}, {"matrixSource": {"location": "/other"}}])
def test_mount_dataset_refuses_a_foreign_matrix_source(
    ready_dataset, tmp_path, attributes
):
    target = tmp_path / "analysis.zarr"
    zarr.open_group(str(target), mode="w", attributes=attributes)
    (tmp_path / "analysis.zarr.cytebase.json").write_text(
        json.dumps(_identity_of(ready_dataset))
    )
    with pytest.raises(ValueError, match="does not match the Cytebase receipt"):
        connector.mount_dataset(ready_dataset.bucket, CYTEBASE_ID, target)


def test_mount_dataset_closes_the_mount_when_publication_changes(
    ready_dataset, fake_hub, tmp_path
):
    _change_record_on_read(fake_hub, 2, _republish(fake_hub))
    with pytest.raises(RuntimeError, match="changed while opening"):
        connector.mount_dataset(
            ready_dataset.bucket, CYTEBASE_ID, tmp_path / "analysis.zarr"
        )
    assert not (tmp_path / "analysis.zarr.cytebase.json").exists()


def test_mount_dataset_does_not_overwrite_a_receipt_created_meanwhile(
    ready_dataset, fake_hub, tmp_path
):
    sidecar = tmp_path / "analysis.zarr.cytebase.json"
    _change_record_on_read(fake_hub, 2, lambda: sidecar.write_text("claimed"))
    with pytest.raises(FileExistsError):
        connector.mount_dataset(
            ready_dataset.bucket, CYTEBASE_ID, tmp_path / "analysis.zarr"
        )
    assert sidecar.read_text() == "claimed"
