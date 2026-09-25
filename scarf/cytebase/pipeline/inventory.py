"""Local, refreshable discovery snapshots of public CELLxGENE collections."""

import json
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from uuid import UUID

from .._storage import error_message
from .catalog import fetch_collection, list_collection_ids, source_metadata
from .selection import SELECTION_SOURCES, classify_dataset


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_snapshot(path: Path, snapshot: dict) -> None:
    """Replace one local JSON file only after the complete next snapshot is saved."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".inventory-",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(snapshot, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _catalog_snapshot(bucket: str) -> tuple[dict, dict]:
    from ..catalog import Catalog

    rows = Catalog(bucket=bucket).query(
        "SELECT dataset_id, cytebase_id, latest_version_id, processed_version_id, "
        "status, zarr_uri FROM datasets"
    )
    by_id = {row["dataset_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Catalog snapshot repeats a stable dataset ID")
    return by_id, {
        "checkedAt": _now(),
        "description": (
            "Verified cached DuckDB snapshot, not live progress. Use the dataset "
            "status endpoints for active jobs."
        ),
    }


def _catalog_status(dataset: dict, known: dict) -> dict:
    key = dataset.get("dataset_id")
    row = known.get(key) if isinstance(key, str) else None
    if row is None:
        return {"state": "notRegistered", "ready": False}
    version = dataset.get("dataset_version_id")
    current = row["latest_version_id"] == version
    ready = (
        current
        and row["processed_version_id"] == version
        and row["status"] == "ready"
        and bool(row["zarr_uri"])
    )
    return {
        "state": "updateAvailable" if not current else row["status"],
        "ready": ready,
        "cytebaseId": row["cytebase_id"],
        "registeredVersionId": row["latest_version_id"],
        "processedVersionId": row["processed_version_id"],
        "status": row["status"],
        "zarrUri": row["zarr_uri"],
    }


def _dataset_row(collection: dict, dataset: dict, known: dict | None) -> dict:
    decision = classify_dataset(dataset)
    source_url = None
    source_bytes = None
    try:
        source = source_metadata(collection, dataset)
    except (AttributeError, KeyError, TypeError, ValueError):
        # Selection records invalid assets as needsReview for primary RNA datasets.
        # Secondary and non-RNA records remain useful without a downloadable asset.
        pass
    else:
        source_url = source.source_url
        source_bytes = source.source_bytes
    row = {
        "datasetId": dataset.get("dataset_id"),
        "datasetVersionId": dataset.get("dataset_version_id"),
        "title": dataset.get("title"),
        "organisms": dataset.get("organism") or [],
        "assays": dataset.get("assay") or [],
        "cellCount": dataset.get("cell_count"),
        "primaryCellCount": dataset.get("primary_cell_count"),
        "isPrimaryData": dataset.get("is_primary_data"),
        "h5adUrl": source_url,
        "h5adBytes": source_bytes,
        "rawDataLocation": dataset.get("raw_data_location"),
        **decision,
    }
    if known is not None:
        row["catalog"] = _catalog_status(dataset, known)
    return row


def _collection_row(collection: dict, known: dict | None) -> dict:
    datasets = collection.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError("Collection response has no dataset list")
    rows = {}
    for index, dataset in enumerate(datasets):
        if not isinstance(dataset, dict):
            raise ValueError(f"Collection dataset entry {index} is not an object")
        key = dataset.get("dataset_id")
        if not isinstance(key, str) or not key:
            key = f"invalid-id-{index}"
        if key in rows:
            raise ValueError(f"Collection repeats dataset {key}")
        rows[key] = _dataset_row(collection, dataset, known)
    decisions = {row["selection"] for row in rows.values()}
    selection = (
        "needsReview"
        if "needsReview" in decisions
        else "ready"
        if "selected" in decisions
        else "noSelectedDatasets"
    )
    return {
        "collectionId": collection["collection_id"],
        "name": collection.get("name"),
        "doi": collection.get("doi"),
        "url": collection.get("collection_url")
        or f"https://cellxgene.cziscience.com/collections/{collection['collection_id']}",
        "status": "complete",
        "updatedAt": _now(),
        "selection": selection,
        "datasets": rows,
    }


def _summarize(snapshot: dict) -> None:
    collections = snapshot["collections"]
    datasets = [
        (key, dataset)
        for key, collection in collections.items()
        for dataset in collection.get("datasets", {}).values()
    ]
    selected = [(key, row) for key, row in datasets if row["selection"] == "selected"]
    filters: tuple[tuple[str, Callable[[dict[str, Any]], bool]], ...] = (
        ("registrationReadyCollectionIds", lambda row: row.get("selection") == "ready"),
        ("needsReviewCollectionIds", lambda row: row.get("selection") == "needsReview"),
        ("failedCollectionIds", lambda row: row["status"] == "failed"),
        ("pendingCollectionIds", lambda row: row["status"] == "pending"),
    )
    for field, predicate in filters:
        snapshot[field] = sorted(
            key for key, row in collections.items() if predicate(row)
        )
    snapshot["summary"] = {
        "collectionCount": len(collections),
        "completedCollectionCount": sum(
            row["status"] == "complete" for row in collections.values()
        ),
        "failedCollectionCount": len(snapshot["failedCollectionIds"]),
        "pendingCollectionCount": len(snapshot["pendingCollectionIds"]),
        "registrationReadyCollectionCount": len(
            snapshot["registrationReadyCollectionIds"]
        ),
        "needsReviewCollectionCount": len(snapshot["needsReviewCollectionIds"]),
        "datasetCount": len(datasets),
        "primaryDatasetCount": sum(row["primary"] is True for _, row in datasets),
        "unknownPrimaryDatasetCount": sum(
            row["primary"] is None for _, row in datasets
        ),
        "selectedDatasetCount": len(selected),
        "skippedDatasetCount": sum(
            row["selection"] == "skipped" for _, row in datasets
        ),
        "needsReviewDatasetCount": sum(
            row["selection"] == "needsReview" for _, row in datasets
        ),
        "selectedSourceBytes": sum(
            row["h5adBytes"] for _, row in selected if row["h5adBytes"] is not None
        ),
        "selectedUnknownSizeCount": sum(
            row["h5adBytes"] is None for _, row in selected
        ),
        "selectedDatasetsOverMillionCells": [
            {
                "collectionId": key,
                "datasetId": row["datasetId"],
                "cellCount": row["cellCount"],
            }
            for key, row in selected
            if isinstance(row["cellCount"], int) and row["cellCount"] > 1_000_000
        ],
    }
    snapshot["updatedAt"] = _now()


def build_inventory(output: Path, *, bucket: str | None = None) -> dict:
    """Refresh public metadata using four requests, never submit pipeline jobs.

    The optional bucket is read only through the checksum-verified local catalog.
    Catalog authentication or snapshot failures propagate before writing output.
    """
    known, catalog_snapshot = (
        _catalog_snapshot(bucket) if bucket is not None else (None, None)
    )
    ids = list_collection_ids()
    snapshot: dict[str, Any] = {
        "startedAt": _now(),
        "completedAt": None,
        "selectionSources": SELECTION_SOURCES,
        "catalogSnapshot": catalog_snapshot,
        "collections": {
            key: {"collectionId": key, "status": "pending", "datasets": {}}
            for key in ids
        },
    }
    _summarize(snapshot)
    _write_snapshot(output, snapshot)
    with ThreadPoolExecutor(
        max_workers=4, thread_name_prefix="cytebase-inventory"
    ) as pool:
        futures = {pool.submit(fetch_collection, key): key for key in ids}
        try:
            for future in as_completed(futures):
                key = futures[future]
                try:
                    _, collection = future.result()
                    if str(UUID(collection["collection_id"])) != key:
                        raise ValueError(
                            "Collection identity differs from the requested ID"
                        )
                    snapshot["collections"][key] = _collection_row(collection, known)
                except Exception as error:
                    snapshot["collections"][key] = {
                        "collectionId": key,
                        "status": "failed",
                        "updatedAt": _now(),
                        "error": error_message(error),
                        "datasets": {},
                    }
                _summarize(snapshot)
                _write_snapshot(output, snapshot)
        finally:
            # An interrupted scan waits only for active calls, not the queued census.
            for future in futures:
                future.cancel()
    snapshot["completedAt"] = _now()
    _summarize(snapshot)
    _write_snapshot(output, snapshot)
    return snapshot
