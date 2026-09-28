"""Offline tests for the local CELLxGENE collection inventory."""

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

pytest.importorskip("duckdb")
pytest.importorskip("natsort")

from scarf.cytebase.pipeline import inventory  # noqa: E402
from scarf.cytebase.pipeline.selection import SELECTION_SOURCES  # noqa: E402
from tests.fixtures_cytebase import (  # noqa: E402
    BUCKET_ID,
    COLLECTION_ID,
    CYTEBASE_ID,
    DATASET_ID,
    NEW_VERSION_ID,
    NOW,
    SOURCE_URL,
    VERSION_ID,
    cellxgene_collection,
    cellxgene_dataset,
    dataset_record,
    publish_catalog_rows,
)

pytestmark = pytest.mark.usefixtures("cytebase_offline")

OTHER_COLLECTION_ID = "55555555-5555-4555-8555-555555555555"
OLD_VERSION_ID = "66666666-6666-4666-8666-666666666666"
SECOND_DATASET_ID = "77777777-7777-4777-8777-777777777777"
THIRD_DATASET_ID = "88888888-8888-4888-8888-888888888888"
FOURTH_DATASET_ID = "99999999-9999-4999-8999-999999999999"
BLOOD_ID = "doe_2023_blood_atlas_77777777"
COLON_ID = "lee_2022_colon_atlas_88888888"
ZARR_URI = "/published/lung/data.zarr"
RNA = {"label": "10x 3' v3", "ontology_term_id": "EFO:0009922"}
ATAC = {"label": "scATAC-seq", "ontology_term_id": "EFO:0010891"}
UNREVIEWED = {"label": "unreviewed assay", "ontology_term_id": "EFO:9999999"}
SECONDARY = {"primary_cell_count": 0, "is_primary_data": [False]}
READY_ROW = {
    "dataset_id": DATASET_ID,
    "cytebase_id": CYTEBASE_ID,
    "latest_version_id": VERSION_ID,
    "processed_version_id": VERSION_ID,
    "status": "ready",
    "zarr_uri": ZARR_URI,
}
UNREGISTERED = {"state": "notRegistered", "ready": False}


def _uuid(number: int) -> str:
    return str(UUID(int=number))


def _dataset(*, drop: tuple[str, ...] = (), **overrides: object) -> dict:
    """A CELLxGENE dataset with ``overrides`` applied and ``drop`` keys removed."""
    dataset = cellxgene_dataset(**overrides)
    for key in drop:
        del dataset[key]
    return dataset


def _h5ad(**fields: object) -> list[dict]:
    return [{"filetype": "H5AD", "url": SOURCE_URL} | fields]


def _serve(monkeypatch, responses: dict[str, dict | Exception]) -> list[str]:
    """Answer collection requests from ``responses`` instead of CELLxGENE."""
    requested: list[str] = []

    def fetch(collection_id: str) -> tuple[bytes, dict]:
        requested.append(collection_id)
        response = responses[collection_id]
        if isinstance(response, Exception):
            raise response
        return json.dumps(response).encode(), response

    monkeypatch.setattr(inventory, "list_collection_ids", lambda: list(responses))
    monkeypatch.setattr(inventory, "fetch_collection", fetch)
    return requested


def _statuses(snapshot: dict) -> dict[str, str]:
    return {key: row["status"] for key, row in snapshot["collections"].items()}


def _saved(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def clock(monkeypatch) -> str:
    monkeypatch.setattr(inventory, "_now", lambda: NOW)
    return NOW


def test_timestamps_are_utc_iso_strings():
    before = datetime.now(UTC)
    stamp = datetime.fromisoformat(inventory._now())
    assert stamp.utcoffset() == timedelta(0)
    assert before <= stamp <= datetime.now(UTC)


def test_write_snapshot_creates_folders_and_saves_readable_json(tmp_path):
    path = tmp_path / "inventory" / "latest" / "collections.json"
    snapshot = {"name": "Célula atlas", "counts": [1, None], "nested": {"ok": True}}
    inventory._write_snapshot(path, snapshot)
    assert path.read_text(encoding="utf-8") == (
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n"
    )
    assert list(path.parent.iterdir()) == [path]


def test_write_snapshot_syncs_the_complete_file_before_replacing(monkeypatch, tmp_path):
    folder = tmp_path / "inventory"
    folder.mkdir()
    path = folder / "collections.json"
    path.write_text("previous\n", encoding="utf-8")
    snapshot = {"collections": {COLLECTION_ID: {"status": "complete"}}}
    synced = []

    def fsync(descriptor: int) -> None:
        (temporary,) = [
            item for item in folder.iterdir() if item.name.startswith(".inventory-")
        ]
        assert os.fstat(descriptor).st_ino == temporary.stat().st_ino
        synced.append(
            (temporary.read_text(encoding="utf-8"), path.read_text(encoding="utf-8"))
        )
        os.fsync(descriptor)

    monkeypatch.setattr(inventory, "os", SimpleNamespace(fsync=fsync))
    inventory._write_snapshot(path, snapshot)
    expected = json.dumps(snapshot, indent=2) + "\n"
    assert synced == [(expected, "previous\n")]
    assert path.read_text(encoding="utf-8") == expected
    assert list(folder.iterdir()) == [path]


@pytest.mark.parametrize(
    ("value", "error", "match"),
    [
        pytest.param(float("nan"), ValueError, "not JSON compliant", id="nan"),
        pytest.param(float("inf"), ValueError, "not JSON compliant", id="infinity"),
        pytest.param({"set"}, TypeError, "not JSON serializable", id="set"),
    ],
)
def test_failed_write_keeps_the_previous_snapshot(tmp_path, value, error, match):
    folder = tmp_path / "inventory"
    folder.mkdir()
    path = folder / "collections.json"
    path.write_text('{"previous": true}\n', encoding="utf-8")
    with pytest.raises(error, match=match):
        inventory._write_snapshot(path, {"collections": {}, "bad": value})
    assert path.read_text(encoding="utf-8") == '{"previous": true}\n'
    assert list(folder.iterdir()) == [path]


@pytest.mark.parametrize(
    ("changes", "ready"),
    [
        pytest.param({}, True, id="processed-current-version"),
        pytest.param(
            {"processed_version_id": OLD_VERSION_ID}, False, id="processed-older"
        ),
        pytest.param({"processed_version_id": None}, False, id="unprocessed"),
        pytest.param({"status": "processing"}, False, id="processing"),
        pytest.param({"zarr_uri": None}, False, id="no-store"),
        pytest.param({"zarr_uri": ""}, False, id="blank-store"),
    ],
)
def test_current_version_is_ready_only_with_its_processed_store(changes, ready):
    row = READY_ROW | changes
    status = inventory._catalog_status(cellxgene_dataset(), {DATASET_ID: row})
    assert status["ready"] is ready
    assert (status["status"], status["zarrUri"]) == (row["status"], row["zarr_uri"])


@pytest.mark.parametrize(
    "dataset_id",
    [
        pytest.param(FOURTH_DATASET_ID, id="unknown"),
        pytest.param(None, id="null"),
        pytest.param(123, id="number"),
        pytest.param([DATASET_ID], id="unhashable-list"),
    ],
)
def test_datasets_missing_from_the_catalog_are_not_registered(dataset_id):
    dataset = cellxgene_dataset(dataset_id=dataset_id)
    assert inventory._catalog_status(dataset, {DATASET_ID: READY_ROW}) == (UNREGISTERED)


def test_dataset_row_describes_a_selected_dataset():
    dataset = cellxgene_dataset(raw_data_location="raw.X")
    assert inventory._dataset_row(cellxgene_collection([dataset]), dataset, None) == {
        "datasetId": DATASET_ID,
        "datasetVersionId": VERSION_ID,
        "title": "Healthy lung scRNA-seq atlas",
        "organisms": [{"label": "Homo sapiens", "ontology_term_id": "NCBITaxon:9606"}],
        "assays": [RNA],
        "cellCount": 6,
        "primaryCellCount": 4,
        "isPrimaryData": [True, False],
        "h5adUrl": SOURCE_URL,
        "h5adBytes": 1234,
        "rawDataLocation": "raw.X",
        "selection": "selected",
        "reason": "Contains primary cells and reviewed RNA assays",
        "primary": True,
    }


@pytest.mark.parametrize(
    ("overrides", "drop", "reason"),
    [
        pytest.param(
            {"assets": None},
            (),
            "Missing or invalid assets list; review the H5AD download asset",
            id="assets-null",
        ),
        pytest.param(
            {"assets": ["x"]},
            (),
            "Missing or invalid assets list; review the H5AD download asset",
            id="asset-not-object",
        ),
        pytest.param(
            {"assets": _h5ad(filesize="abc")},
            (),
            "Invalid H5AD filesize; expected a nonnegative integer or no value",
            id="filesize-text",
        ),
        pytest.param(
            {},
            ("dataset_version_id",),
            "Missing or invalid dataset_version_id; review the CELLxGENE record",
            id="version-missing",
        ),
        pytest.param(
            {"assets": [{"filetype": "H5AD", "filesize": 5}]},
            (),
            "Missing or invalid H5AD HTTP URL; review the download asset",
            id="url-missing",
        ),
        pytest.param(
            {"assets": _h5ad() + _h5ad()},
            (),
            "Expected exactly one H5AD download asset; review the CELLxGENE record",
            id="two-h5ad-assets",
        ),
    ],
)
def test_dataset_row_flags_unusable_downloads_for_review(overrides, drop, reason):
    dataset = _dataset(drop=drop, **overrides)
    row = inventory._dataset_row(cellxgene_collection([dataset]), dataset, None)
    assert (row["selection"], row["reason"]) == ("needsReview", reason)
    assert (row["h5adUrl"], row["h5adBytes"]) == (None, None)
    assert row["datasetId"] == DATASET_ID


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        pytest.param({"assay": [ATAC]}, "Known non-RNA assays: EFO:0010891", id="atac"),
        pytest.param(SECONDARY, "All cells are secondary", id="secondary"),
    ],
)
def test_dataset_row_keeps_skipped_datasets_without_a_download(overrides, reason):
    dataset = cellxgene_dataset(assets=None, **overrides)
    row = inventory._dataset_row(cellxgene_collection([dataset]), dataset, None)
    assert (row["selection"], row["reason"]) == ("skipped", reason)
    assert (row["h5adUrl"], row["h5adBytes"]) == (None, None)


def test_dataset_row_lists_missing_organisms_and_assays_as_empty():
    dataset = cellxgene_dataset(organism=None, assay=None)
    row = inventory._dataset_row(cellxgene_collection([dataset]), dataset, None)
    assert (row["organisms"], row["assays"]) == ([], [])
    assert (row["selection"], row["reason"]) == (
        "needsReview",
        "Missing assay metadata; review RNA content",
    )


def test_collection_row_summarizes_one_collection(clock):
    datasets = [
        cellxgene_dataset(),
        cellxgene_dataset(dataset_id=SECOND_DATASET_ID, **SECONDARY),
    ]
    collection = cellxgene_collection(
        datasets, collection_url="https://cellxgene.example.org/lung"
    )
    row = inventory._collection_row(collection, None)
    assert {key: value for key, value in row.items() if key != "datasets"} == {
        "collectionId": COLLECTION_ID,
        "name": "Human lung atlas",
        "doi": "10.1000/lung",
        "url": "https://cellxgene.example.org/lung",
        "status": "complete",
        "updatedAt": NOW,
        "selection": "ready",
    }
    assert {key: value["selection"] for key, value in row["datasets"].items()} == {
        DATASET_ID: "selected",
        SECOND_DATASET_ID: "skipped",
    }
    assert "catalog" not in row["datasets"][DATASET_ID]


@pytest.mark.parametrize("url", ["missing", None, ""])
def test_collection_row_links_to_cellxgene_without_a_collection_url(url):
    collection = cellxgene_collection(collection_url=url)
    if url == "missing":
        del collection["collection_url"]
    assert inventory._collection_row(collection, None)["url"] == (
        f"https://cellxgene.cziscience.com/collections/{COLLECTION_ID}"
    )


@pytest.mark.parametrize(
    ("kinds", "selection"),
    [
        pytest.param(["selected"], "ready", id="selected"),
        pytest.param(["skipped", "selected"], "ready", id="skipped-and-selected"),
        pytest.param(
            ["selected", "needsReview", "skipped"], "needsReview", id="review-wins"
        ),
        pytest.param(["skipped", "skipped"], "noSelectedDatasets", id="all-skipped"),
        pytest.param([], "noSelectedDatasets", id="empty"),
    ],
)
def test_collection_selection_combines_dataset_decisions(kinds, selection):
    overrides = {
        "selected": {},
        "skipped": SECONDARY,
        "needsReview": {"assay": [UNREVIEWED]},
    }
    datasets = [
        cellxgene_dataset(dataset_id=_uuid(index), **overrides[kind])
        for index, kind in enumerate(kinds, start=1)
    ]
    row = inventory._collection_row(cellxgene_collection(datasets), None)
    assert row["selection"] == selection
    assert [dataset["selection"] for dataset in row["datasets"].values()] == kinds


@pytest.mark.parametrize(
    ("datasets", "match"),
    [
        pytest.param(None, "Collection response has no dataset list", id="null"),
        pytest.param(
            {DATASET_ID: cellxgene_dataset()},
            "Collection response has no dataset list",
            id="object",
        ),
        pytest.param(
            [cellxgene_dataset(), "x"],
            "Collection dataset entry 1 is not an object",
            id="entry-not-object",
        ),
        pytest.param(
            [cellxgene_dataset(), cellxgene_dataset(title="Copy")],
            f"Collection repeats dataset {DATASET_ID}",
            id="repeated-id",
        ),
    ],
)
def test_collection_row_rejects_malformed_dataset_lists(datasets, match):
    collection = cellxgene_collection() | {"datasets": datasets}
    with pytest.raises(ValueError, match=match):
        inventory._collection_row(collection, None)


def test_collection_row_keys_datasets_without_a_usable_id_by_position():
    datasets = [
        _dataset(drop=("dataset_id",)),
        cellxgene_dataset(dataset_id=""),
        cellxgene_dataset(dataset_id=123),
        cellxgene_dataset(),
    ]
    row = inventory._collection_row(cellxgene_collection(datasets), None)
    assert list(row["datasets"]) == [
        "invalid-id-0",
        "invalid-id-1",
        "invalid-id-2",
        DATASET_ID,
    ]
    assert {row["datasets"][f"invalid-id-{index}"]["reason"] for index in range(3)} == {
        "Missing or invalid stable dataset_id"
    }
    assert row["selection"] == "needsReview"


def test_an_empty_catalog_marks_every_dataset_unregistered():
    row = inventory._collection_row(cellxgene_collection(), {})
    assert row["datasets"][DATASET_ID]["catalog"] == UNREGISTERED


def test_summary_counts_collections_and_selected_datasets(clock):
    ready, empty, review, failed = _uuid(0xA), _uuid(0xB), _uuid(0xC), _uuid(0xE)
    pending = [_uuid(0xF), _uuid(0xD)]
    big = _uuid(106)
    rows = {
        ready: [
            cellxgene_dataset(dataset_id=_uuid(101)),
            cellxgene_dataset(dataset_id=_uuid(102), assets=_h5ad()),
            cellxgene_dataset(
                dataset_id=_uuid(103),
                cell_count=1_000_000,
                primary_cell_count=1_000_000,
                is_primary_data=[True],
                assets=_h5ad(filesize=10),
            ),
            cellxgene_dataset(dataset_id=_uuid(104), **SECONDARY),
        ],
        empty: [cellxgene_dataset(dataset_id=_uuid(105), assay=[ATAC])],
        review: [
            cellxgene_dataset(
                dataset_id=big,
                cell_count=2_000_000,
                primary_cell_count=2_000_000,
                is_primary_data=[True],
                assets=_h5ad(filesize=100),
            ),
            cellxgene_dataset(dataset_id=_uuid(107), assay=[UNREVIEWED]),
            cellxgene_dataset(
                dataset_id=_uuid(108), primary_cell_count=None, is_primary_data=None
            ),
        ],
    }
    complete = {
        key: inventory._collection_row(
            cellxgene_collection(datasets, collection_id=key), None
        )
        for key, datasets in rows.items()
    }
    snapshot = {
        "collections": {
            review: complete[review],
            pending[0]: {
                "collectionId": pending[0],
                "status": "pending",
                "datasets": {},
            },
            ready: complete[ready],
            failed: {
                "collectionId": failed,
                "status": "failed",
                "updatedAt": NOW,
                "error": "RuntimeError: unavailable",
                "datasets": {},
            },
            empty: complete[empty],
            pending[1]: {
                "collectionId": pending[1],
                "status": "pending",
                "datasets": {},
            },
        }
    }

    inventory._summarize(snapshot)

    assert snapshot["registrationReadyCollectionIds"] == [ready]
    assert snapshot["needsReviewCollectionIds"] == [review]
    assert snapshot["failedCollectionIds"] == [failed]
    assert snapshot["pendingCollectionIds"] == [_uuid(0xD), _uuid(0xF)]
    assert snapshot["updatedAt"] == NOW
    assert snapshot["summary"] == {
        "collectionCount": 6,
        "completedCollectionCount": 3,
        "failedCollectionCount": 1,
        "pendingCollectionCount": 2,
        "registrationReadyCollectionCount": 1,
        "needsReviewCollectionCount": 1,
        "datasetCount": 8,
        "primaryDatasetCount": 6,
        "unknownPrimaryDatasetCount": 1,
        "selectedDatasetCount": 4,
        "skippedDatasetCount": 2,
        "needsReviewDatasetCount": 2,
        # Selected datasets in collections needing review still count here.
        "selectedSourceBytes": 1234 + 10 + 100,
        "selectedUnknownSizeCount": 1,
        "selectedDatasetsOverMillionCells": [
            {"collectionId": review, "datasetId": big, "cellCount": 2_000_000}
        ],
    }


def test_inventory_saves_progress_after_every_collection(monkeypatch, tmp_path, clock):
    first, second = COLLECTION_ID, OTHER_COLLECTION_ID
    responses = {
        first: cellxgene_collection(),
        second: cellxgene_collection(
            [cellxgene_dataset(dataset_id=SECOND_DATASET_ID, **SECONDARY)],
            collection_id=second,
        ),
    }
    output = tmp_path / "inventory" / "collections.json"
    saved: list[dict] = []
    status_when_requested: dict[str, str] = {}
    second_saved = threading.Event()
    write = inventory._write_snapshot

    def record_write(path: Path, snapshot: dict) -> None:
        write(path, snapshot)
        saved.append(_saved(path))
        if _statuses(saved[-1])[second] == "complete":
            second_saved.set()

    def fetch(collection_id: str) -> tuple[bytes, dict]:
        status_when_requested[collection_id] = _statuses(_saved(output))[collection_id]
        if collection_id == first:
            # Finish the first listed collection only after the second is saved.
            assert second_saved.wait(5)
        return b"{}", responses[collection_id]

    monkeypatch.setattr(inventory, "list_collection_ids", lambda: [first, second])
    monkeypatch.setattr(inventory, "fetch_collection", fetch)
    monkeypatch.setattr(inventory, "_write_snapshot", record_write)

    result = inventory.build_inventory(output)

    assert status_when_requested == {first: "pending", second: "pending"}
    assert [(_statuses(state), state["completedAt"]) for state in saved] == [
        ({first: "pending", second: "pending"}, None),
        ({first: "pending", second: "complete"}, None),
        ({first: "complete", second: "complete"}, None),
        ({first: "complete", second: "complete"}, NOW),
    ]
    assert result == saved[-1] == _saved(output)
    assert list(result["collections"]) == [first, second]
    assert result["startedAt"] == NOW
    assert result["selectionSources"] == SELECTION_SOURCES
    assert result["catalogSnapshot"] is None
    assert result["registrationReadyCollectionIds"] == [first]
    assert result["collections"][second]["selection"] == "noSelectedDatasets"
    assert "catalog" not in result["collections"][first]["datasets"][DATASET_ID]


@pytest.mark.parametrize(
    ("response", "error"),
    [
        pytest.param(
            RuntimeError("CELLxGENE unavailable"),
            "RuntimeError: CELLxGENE unavailable",
            id="request-failed",
        ),
        pytest.param(
            RuntimeError("token hf_Secret123 rejected"),
            "RuntimeError: token [redacted] rejected",
            id="token-redacted",
        ),
        pytest.param(
            cellxgene_collection(),
            "ValueError: Collection identity differs from the requested ID",
            id="other-collection",
        ),
        pytest.param(
            cellxgene_collection(collection_id="not-a-uuid"),
            "ValueError: badly formed hexadecimal UUID string",
            id="invalid-collection-id",
        ),
        pytest.param(
            cellxgene_collection({}, collection_id=OTHER_COLLECTION_ID),
            "ValueError: Collection response has no dataset list",
            id="no-dataset-list",
        ),
    ],
)
def test_inventory_records_a_failed_collection_and_continues(
    monkeypatch, tmp_path, clock, response, error
):
    output = tmp_path / "collections.json"
    _serve(
        monkeypatch,
        {COLLECTION_ID: cellxgene_collection(), OTHER_COLLECTION_ID: response},
    )

    result = inventory.build_inventory(output)

    assert result["collections"][OTHER_COLLECTION_ID] == {
        "collectionId": OTHER_COLLECTION_ID,
        "status": "failed",
        "updatedAt": NOW,
        "error": error,
        "datasets": {},
    }
    assert result["collections"][COLLECTION_ID]["status"] == "complete"
    assert result["failedCollectionIds"] == [OTHER_COLLECTION_ID]
    assert result["registrationReadyCollectionIds"] == [COLLECTION_ID]
    assert result["completedAt"] == NOW
    assert _saved(output) == result


def test_inventory_requests_at_most_four_collections_at_once(monkeypatch, tmp_path):
    ids = [_uuid(number) for number in range(1, 7)]
    first_wave = threading.Barrier(4, timeout=5)
    lock = threading.Lock()
    calls = active = peak = 0

    def fetch(collection_id: str) -> tuple[bytes, dict]:
        nonlocal calls, active, peak
        with lock:
            calls += 1
            active += 1
            peak = max(peak, active)
            waits = calls <= 4
        try:
            if waits:
                # Four requests must be in flight together before any returns.
                first_wave.wait()
            return b"{}", cellxgene_collection(collection_id=collection_id)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(inventory, "list_collection_ids", lambda: ids)
    monkeypatch.setattr(inventory, "fetch_collection", fetch)

    result = inventory.build_inventory(tmp_path / "collections.json")

    assert result["failedCollectionIds"] == []
    assert result["summary"]["completedCollectionCount"] == 6
    assert (calls, peak) == (6, 4)


def test_interrupted_inventory_keeps_progress_and_skips_queued_requests(
    monkeypatch, tmp_path
):
    ids = [_uuid(number) for number in range(1, 7)]
    first, fifth = ids[0], ids[4]
    output = tmp_path / "inventory" / "collections.json"
    started = {collection_id: threading.Event() for collection_id in ids}
    release = threading.Event()
    requested: list[str] = []

    def fetch(collection_id: str) -> tuple[bytes, dict]:
        requested.append(collection_id)
        started[collection_id].set()
        if collection_id != first:
            assert release.wait(5)
        return b"{}", cellxgene_collection(collection_id=collection_id)

    class HoldingPool(ThreadPoolExecutor):
        """Hold active requests until the interrupted scan has cancelled the rest."""

        def shutdown(self, *args: object, **kwargs: object) -> None:
            release.set()
            super().shutdown(*args, **kwargs)

    write = inventory._write_snapshot
    writes: list[Path] = []

    def interrupt_after_first_progress(path: Path, snapshot: dict) -> None:
        writes.append(path)
        if len(writes) == 2:
            # The worker freed by the first collection runs the fifth request,
            # so all four workers are busy while the sixth is still queued.
            assert started[fifth].wait(5)
        write(path, snapshot)
        if len(writes) == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(inventory, "list_collection_ids", lambda: ids)
    monkeypatch.setattr(inventory, "fetch_collection", fetch)
    monkeypatch.setattr(inventory, "ThreadPoolExecutor", HoldingPool)
    monkeypatch.setattr(inventory, "_write_snapshot", interrupt_after_first_progress)

    with pytest.raises(KeyboardInterrupt):
        inventory.build_inventory(output)

    assert sorted(requested) == ids[:5]
    saved = _saved(output)
    assert saved["completedAt"] is None
    assert _statuses(saved) == {first: "complete"} | dict.fromkeys(ids[1:], "pending")
    assert saved["pendingCollectionIds"] == ids[1:]


def test_inventory_reports_catalog_status_from_the_bucket(
    fake_hub, monkeypatch, tmp_path, clock
):
    publish_catalog_rows(
        fake_hub,
        [
            dataset_record(
                status="ready", processedVersionId=VERSION_ID, zarrUri=ZARR_URI
            ),
            dataset_record(
                cytebaseId=BLOOD_ID,
                datasetId=SECOND_DATASET_ID,
                status="ready",
                processedVersionId=VERSION_ID,
                zarrUri="/published/blood/data.zarr",
            ),
            dataset_record(cytebaseId=COLON_ID, datasetId=THIRD_DATASET_ID),
        ],
    )
    datasets = [
        cellxgene_dataset(),
        cellxgene_dataset(
            dataset_id=SECOND_DATASET_ID, dataset_version_id=NEW_VERSION_ID
        ),
        cellxgene_dataset(dataset_id=THIRD_DATASET_ID),
        cellxgene_dataset(dataset_id=FOURTH_DATASET_ID),
    ]
    _serve(monkeypatch, {COLLECTION_ID: cellxgene_collection(datasets)})
    output = tmp_path / "collections.json"

    result = inventory.build_inventory(output, bucket=BUCKET_ID)

    rows = result["collections"][COLLECTION_ID]["datasets"]
    assert {key: row["catalog"] for key, row in rows.items()} == {
        DATASET_ID: {
            "state": "ready",
            "ready": True,
            "cytebaseId": CYTEBASE_ID,
            "registeredVersionId": VERSION_ID,
            "processedVersionId": VERSION_ID,
            "status": "ready",
            "zarrUri": ZARR_URI,
        },
        SECOND_DATASET_ID: {
            "state": "updateAvailable",
            "ready": False,
            "cytebaseId": BLOOD_ID,
            "registeredVersionId": VERSION_ID,
            "processedVersionId": VERSION_ID,
            "status": "ready",
            "zarrUri": "/published/blood/data.zarr",
        },
        THIRD_DATASET_ID: {
            "state": "registered",
            "ready": False,
            "cytebaseId": COLON_ID,
            "registeredVersionId": VERSION_ID,
            "processedVersionId": None,
            "status": "registered",
            "zarrUri": None,
        },
        FOURTH_DATASET_ID: UNREGISTERED,
    }
    assert result["catalogSnapshot"]["checkedAt"] == NOW
    assert "not live progress" in result["catalogSnapshot"]["description"]
    assert _saved(output) == result


@pytest.mark.parametrize(
    ("records", "error", "match"),
    [
        pytest.param(
            None,
            RuntimeError,
            "Missing catalog/cytebase.duckdb.sha256",
            id="no-catalog",
        ),
        pytest.param(
            [
                dataset_record(),
                dataset_record(cytebaseId="smith_2024_healthy_lung_copy_22222222"),
            ],
            ValueError,
            "Catalog snapshot repeats a stable dataset ID",
            id="repeated-dataset-id",
        ),
    ],
)
def test_inventory_stops_on_catalog_failures_before_writing(
    fake_hub, monkeypatch, tmp_path, records, error, match
):
    if records is not None:
        publish_catalog_rows(fake_hub, records)
    requested = _serve(monkeypatch, {COLLECTION_ID: cellxgene_collection()})
    listed = []
    monkeypatch.setattr(
        inventory, "list_collection_ids", lambda: listed.append(True) or []
    )
    output = tmp_path / "collections.json"
    output.write_text('{"previous": true}\n', encoding="utf-8")

    with pytest.raises(error, match=match):
        inventory.build_inventory(output, bucket=BUCKET_ID)

    assert output.read_text(encoding="utf-8") == '{"previous": true}\n'
    assert (listed, requested) == ([], [])


def test_inventory_ignores_the_configured_bucket_unless_one_is_given(
    fake_hub, monkeypatch, tmp_path
):
    publish_catalog_rows(
        fake_hub,
        [
            dataset_record(
                status="ready", processedVersionId=VERSION_ID, zarrUri=ZARR_URI
            )
        ],
    )
    monkeypatch.setenv("CYTEBASE_BUCKET", BUCKET_ID)
    _serve(monkeypatch, {COLLECTION_ID: cellxgene_collection()})

    result = inventory.build_inventory(tmp_path / "collections.json")

    assert fake_hub.calls == []
    assert result["catalogSnapshot"] is None
    assert "catalog" not in result["collections"][COLLECTION_ID]["datasets"][DATASET_ID]
