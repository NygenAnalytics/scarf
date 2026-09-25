"""Offline end-to-end test from CELLxGENE registration to Scarf analysis.

Only network edges are faked: the CELLxGENE curation response and H5AD download,
Hugging Face bucket transfers (one local directory per bucket), and Modal calls,
which run the real worker bodies in-process. Registration, inspection,
conversion, publication, verification, catalog publication, and every SDK call
run real code, so the SDK must accept exactly what the pipeline publishes.
"""

import hashlib
import json
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import matplotlib

matplotlib.use("Agg")

import httpx
import numpy as np
import pytest

from tests.fixtures_cytebase import (
    BUCKET_ID,
    CELL_TYPES,
    CITATION,
    COLLECTION_ID,
    COUNTS,
    CYTEBASE_ID,
    DATASET_ID,
    DONORS,
    GENES,
    PIPELINE_VERSION,
    SOURCE_URL,
    UMAP,
    VERSION_ID,
    ModalHarness,
    NetworkAccessBlocked,
    cellxgene_collection,
    cellxgene_dataset,
    source_details,
    write_h5ad,
)

pytest.importorskip("modal")
pytest.importorskip("fastapi")
pytest.importorskip("duckdb")
pytest.importorskip("natsort")

pytestmark = pytest.mark.usefixtures("cytebase_offline")

COLLECTION_API_URL = (
    f"https://api.cellxgene.cziscience.com/curation/v1/collections/{COLLECTION_ID}"
)
COLLECTION_PAGE = f"https://cellxgene.cziscience.com/collections/{COLLECTION_ID}"
PREFIX = f"datasets/{CYTEBASE_ID}"
RECORD_PATH = f"{PREFIX}/dataset.json"
INGEST_PATH = f"{PREFIX}/scarf_ingest.json"
RUN_PATH = "_internal/pipeline.json"
PROCESS_KEY = f"{CYTEBASE_ID}:process"
TITLE = "Healthy lung scRNA-seq atlas"
GROUPS = ["a", "b", "a", "b", "a", "b"]
DESCRIPTION = "\n".join(
    [
        f"### {TITLE}",
        "",
        f"- **Cytebase ID:** `{CYTEBASE_ID}`",
        "- **Citation:**",
        "    - Publication: https://doi.org/10.1000/lung",
        f"    - Dataset Version: {SOURCE_URL}",
        "    - curated and distributed by CZ CELLxGENE Discover in Collection: "
        + COLLECTION_PAGE,
        "- **DOI:** 10.1000/lung",
        f"- **CELLxGENE:** {COLLECTION_PAGE}",
        "- **Explorer:** https://cellxgene.cziscience.com/e/lung.cxg/",
        "- **Size:** 6 cells, 5 genes",
        "- **Status:** ready",
        "- **Organisms:** Homo sapiens",
        "- **Assays:** 10x 3' v3",
        "- **Tissues:** lung",
        "- **Diseases:** normal",
        "- **Suspension:** cell",
        "- **Source embeddings:** X_umap",
    ]
)
# Worker progress milestones in order. The fake download reports "downloaded".
PROGRESS_MILESTONES = [
    "process",
    "preflight",
    "downloaded",
    "inspecting",
    "converting",
    "uploading_store",
    "verifying_store",
    "cleaning_local",
]


@dataclass
class FakeCellxgene:
    """Serve one collection response and one H5AD download, and nothing else."""

    source: Path
    size: int
    sha256: str
    collection: dict[str, Any]
    body: bytes
    requests: list[str] = field(default_factory=list)
    downloads: list[tuple[str, int | None]] = field(default_factory=list)

    def get(self, url: str) -> httpx.Response:
        self.requests.append(url)
        if url != COLLECTION_API_URL:
            raise NetworkAccessBlocked(f"Unexpected CELLxGENE request: {url}")
        return httpx.Response(200, content=self.body, request=httpx.Request("GET", url))

    def download_h5ad(
        self,
        url: str,
        destination: Path,
        expected_bytes: int | None = None,
        *,
        progress: Callable | None = None,
        timings: dict[str, float] | None = None,
    ) -> tuple[int, str]:
        self.downloads.append((url, expected_bytes))
        if url != SOURCE_URL or expected_bytes != self.size:
            raise NetworkAccessBlocked(
                f"Unexpected H5AD download: {url} ({expected_bytes} bytes)"
            )
        destination = Path(destination)
        if destination.exists():
            raise FileExistsError(f"Download destination exists: {destination}")
        shutil.copyfile(self.source, destination)
        if timings is not None:
            timings.update(downloadTransferSeconds=0.0, downloadHashSeconds=0.0)
        if progress is not None:
            progress(
                "downloaded",
                completed=self.size,
                total=self.size,
                downloadedBytes=self.size,
                unit="bytes",
                phase="complete",
                message="Local H5AD downloaded",
            )
        return source_details(destination)


@pytest.fixture
def cellxgene(monkeypatch, tmp_path) -> FakeCellxgene:
    """Replace the CELLxGENE network edges with a local H5AD and its collection."""
    from scarf.cytebase.pipeline import catalog, download

    source = write_h5ad(tmp_path / "source.h5ad")
    size, checksum = source_details(source)
    asset = {"filetype": "H5AD", "url": SOURCE_URL, "filesize": size}
    collection = cellxgene_collection([cellxgene_dataset(assets=[asset])])
    fake = FakeCellxgene(
        source, size, checksum, collection, json.dumps(collection).encode()
    )
    monkeypatch.setattr(catalog, "_get", fake.get)
    # The dataset worker imports download_h5ad from this module when it runs.
    monkeypatch.setattr(download, "download_h5ad", fake.download_h5ad)
    return fake


@pytest.fixture
def worker_tmp(monkeypatch, tmp_path) -> Path:
    """Root default temporary directories here so worker cleanup can be checked."""
    root = tmp_path / "worker-tmp"
    root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    return root


def _pipeline_name(collection: dict[str, Any]) -> str:
    """Name the collection's only dataset with the pipeline's naming rules."""
    from scarf.cytebase.pipeline import catalog

    (dataset,) = collection["datasets"]
    parts = catalog._name_parts(collection, catalog._facets(dataset), dataset)
    return catalog._assign_name(parts, UUID(dataset["dataset_id"]), {})


def _pick(values: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {key: values[key] for key in keys}


def _without_times(run: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in run.items() if not key.endswith("At")}


def _in_order(expected: list[str], actual: list[str]) -> bool:
    """Whether ``expected`` occurs in ``actual`` in order, allowing other items."""
    remaining = iter(actual)
    return all(item in remaining for item in expected)


def _files(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _check_catalog_publication(harness: ModalHarness, result: dict[str, Any]) -> None:
    """The database, its checksum sidecar, and the catalog worker result agree."""
    hub = harness.hub
    digest = hashlib.sha256(hub.read("catalog/cytebase.duckdb")).hexdigest()
    assert hub.read("catalog/cytebase.duckdb.sha256") == (
        f"{digest}  cytebase.duckdb\n".encode()
    )
    assert _pick(
        result, "status", "datasets", "collections", "catalogUri", "catalogSha256"
    ) == {
        "status": "done",
        "datasets": 1,
        "collections": 1,
        "catalogUri": f"{harness.bucket.root}/catalog/cytebase.duckdb",
        "catalogSha256": digest,
    }


def _check_registration(
    harness: ModalHarness, cellxgene: FakeCellxgene, result: dict[str, Any]
) -> None:
    hub = harness.hub
    assert result["state"] == "completed", result
    assert result["error"] is None
    assert result["datasets"] == [{"cytebaseId": CYTEBASE_ID, "status": "registered"}]
    assert (result["successes"], result["failures"]) == ([COLLECTION_ID], [])
    assert result["catalog"]["failedCollections"] == []
    _check_catalog_publication(harness, result["catalog"])
    assert cellxgene.requests == [COLLECTION_API_URL]
    assert cellxgene.downloads == []

    # Source metadata is kept byte for byte beside a readable summary.
    assert hub.read(f"{PREFIX}/cellxgene/collection.json") == cellxgene.body
    source_dataset = cellxgene.collection["datasets"][0]
    assert hub.read_json(f"{PREFIX}/cellxgene/dataset.json") == source_dataset
    readme = hub.read(f"{PREFIX}/README.md").decode()
    assert readme.startswith(f"# {TITLE}\n\n{CITATION}\n")
    assert f"Dataset ID: `{DATASET_ID}`\n" in readme
    assert f"Source H5AD: {SOURCE_URL}\n" in readme

    record = hub.read_json(RECORD_PATH)
    assert _pick(
        record,
        "cytebaseId",
        "datasetId",
        "collectionId",
        "latestVersionId",
        "processedVersionId",
        "status",
        "sourceUrl",
        "sourceBytes",
        "zarrUri",
        "pipelineVersion",
    ) == {
        "cytebaseId": CYTEBASE_ID,
        "datasetId": DATASET_ID,
        "collectionId": COLLECTION_ID,
        "latestVersionId": VERSION_ID,
        "processedVersionId": None,
        "status": "registered",
        "sourceUrl": SOURCE_URL,
        "sourceBytes": cellxgene.size,
        "zarrUri": None,
        "pipelineVersion": PIPELINE_VERSION,
    }
    assert _without_times(hub.read_json(RUN_PATH)) == {
        "runId": "fc-register",
        "callId": "fc-register",
        "action": "register",
        "state": "completed",
        "children": {
            "catalog": {
                "stage": "catalog",
                "callId": "fc-catalog-1",
                "state": "succeeded",
            }
        },
    }


def _check_processing(
    harness: ModalHarness,
    cellxgene: FakeCellxgene,
    result: dict[str, Any],
    worker_tmp: Path,
) -> dict[str, Any]:
    """Check one successful dataset worker and return its committed record."""
    hub = harness.hub
    zarr_uri = f"{harness.bucket.root}/{PREFIX}/data.zarr"
    assert result["state"] == "completed", result
    assert result["error"] is None
    assert result["datasets"] == [
        {"outcome": "succeeded", "cytebaseId": CYTEBASE_ID, "status": "ready"}
    ]
    assert (result["successes"], result["failures"]) == ([CYTEBASE_ID], [])
    _check_catalog_publication(harness, result["catalog"])
    # One download; processing does not fetch the collection metadata again.
    assert cellxgene.requests == [COLLECTION_API_URL]
    assert cellxgene.downloads == [(SOURCE_URL, cellxgene.size)]

    record = hub.read_json(RECORD_PATH)
    assert _pick(
        record,
        "status",
        "stage",
        "stageOutcome",
        "runId",
        "callId",
        "attempt",
        "error",
        "needsInput",
        "latestVersionId",
        "processedVersionId",
        "zarrUri",
        "h5adUri",
        "sourceBytes",
        "cellCount",
        "primaryCellCount",
        "nGenes",
        "pipelineVersion",
    ) == {
        "status": "ready",
        "stage": "process",
        "stageOutcome": "succeeded",
        "runId": "fc-process",
        "callId": "fc-process-1",
        "attempt": 1,
        "error": None,
        "needsInput": None,
        "latestVersionId": VERSION_ID,
        "processedVersionId": VERSION_ID,
        "zarrUri": zarr_uri,
        "h5adUri": None,
        "sourceBytes": cellxgene.size,
        "cellCount": 6,
        "primaryCellCount": 4,
        "nGenes": 5,
        "pipelineVersion": PIPELINE_VERSION,
    }
    (version,) = record["versions"]
    assert _pick(version, "datasetVersionId", "sourceSha256", "processedAt") == {
        "datasetVersionId": VERSION_ID,
        "sourceSha256": cellxgene.sha256,
        "processedAt": record["processedAt"],
    }
    # Every stage is timed, including the downloader's own measurements.
    assert set(record["timings"]) == {
        "downloadTransferSeconds",
        "downloadHashSeconds",
        "downloadSeconds",
        "inspectSeconds",
        "convertSeconds",
        "uploadSeconds",
        "verifySeconds",
        "cleanupSeconds",
        "processSeconds",
    }
    assert _pick(
        record["inspection"],
        "collectionId",
        "datasetId",
        "datasetVersionId",
        "sourceUrl",
        "sourceBytes",
        "sourceSha256",
        "nObs",
        "nVars",
        "countsLocation",
        "countsDtype",
        "featureNameKey",
        "isPrimaryDataCounts",
        "embeddings",
    ) == {
        "collectionId": COLLECTION_ID,
        "datasetId": DATASET_ID,
        "datasetVersionId": VERSION_ID,
        "sourceUrl": SOURCE_URL,
        "sourceBytes": cellxgene.size,
        "sourceSha256": cellxgene.sha256,
        "nObs": 6,
        "nVars": 5,
        "countsLocation": "X",
        "countsDtype": "int32",
        "featureNameKey": "feature_name",
        "isPrimaryDataCounts": {"true": 4, "false": 2},
        "embeddings": ["X_umap"],
    }

    # The receipt records a verified comparison of the published store.
    receipt = record["buildReceipt"]
    verification = receipt["verification"]
    assert _pick(receipt, "datasetVersionId", "sourceSha256", "zarrUri") == {
        "datasetVersionId": VERSION_ID,
        "sourceSha256": cellxgene.sha256,
        "zarrUri": zarr_uri,
    }
    assert datetime.fromisoformat(receipt["verifiedAt"]) == datetime.fromisoformat(
        record["processedAt"]
    )
    assert _pick(
        verification,
        "nObs",
        "nVars",
        "countsDtype",
        "countsTShape",
        "countsTComplete",
        "countsTMatches",
        "sourceSampleMatches",
        "countsBlock",
    ) == {
        "nObs": 6,
        "nVars": 5,
        "countsDtype": "int32",
        "countsTShape": [5, 6],
        "countsTComplete": True,
        "countsTMatches": True,
        "sourceSampleMatches": True,
        "countsBlock": COUNTS[:3].tolist(),
    }
    ingest = hub.read_json(INGEST_PATH)
    assert _pick(
        ingest, "status", "zarrPath", "sourceSha256", "verification", "completedAt"
    ) == {
        "status": "done",
        "zarrPath": zarr_uri,
        "sourceSha256": cellxgene.sha256,
        "verification": verification,
        "completedAt": receipt["verifiedAt"],
    }
    assert _pick(ingest["conversion"], "matrixKey", "embeddingRoles") == {
        "matrixKey": "X",
        "embeddingRoles": {"X_umap": "umap"},
    }

    # Only metadata, provenance, and the Zarr store are published; no H5AD.
    assert [path for path in hub.files() if "/data.zarr/" not in path] == sorted(
        [
            RUN_PATH,
            "catalog/cytebase.duckdb",
            "catalog/cytebase.duckdb.sha256",
            f"{PREFIX}/README.md",
            f"{PREFIX}/cellxgene/collection.json",
            f"{PREFIX}/cellxgene/dataset.json",
            RECORD_PATH,
            INGEST_PATH,
        ]
    )
    assert hub.path(f"{PREFIX}/data.zarr/zarr.json").is_file()
    # Worker, bucket-read, and catalog temporary directories are all removed.
    assert sorted(worker_tmp.glob("cytebase-*")) == []

    assert _without_times(hub.read_json(RUN_PATH)) == {
        "runId": "fc-process",
        "callId": "fc-process",
        "action": "process",
        "state": "completed",
        "children": {
            "catalog": {
                "stage": "catalog",
                "callId": "fc-catalog-3",
                "state": "succeeded",
            },
            PROCESS_KEY: {
                "cytebaseId": CYTEBASE_ID,
                "datasetVersionId": VERSION_ID,
                "stage": "process",
                "callId": "fc-process-1",
                "state": "succeeded",
            },
        },
    }
    assert harness.process_dataset.spawned == [
        (
            CYTEBASE_ID,
            "fc-process",
            {"collectionId": COLLECTION_ID, "approvedDeletionPaths": []},
        )
    ]
    # Processing refreshes the catalog, then updates it from the committed record.
    assert harness.build_catalog.spawned == [
        ({"collectionIds": [COLLECTION_ID]}, "fc-register"),
        ({"updates": []}, "fc-process"),
        ({"updates": [record]}, "fc-process"),
    ]

    puts = harness.progress_store.puts
    assert {key for key, _ in puts} == {f"fc-process:{CYTEBASE_ID}:process"}
    assert {
        (value["runId"], value["callId"], value["attempt"]) for _, value in puts
    } == {("fc-process", "fc-process-1", 1)}
    stages = [value["stage"] for _, value in puts]
    assert _in_order(PROGRESS_MILESTONES, stages), stages
    assert stages[-1] == "cleaning_local"
    return record


def _check_skipped_rerun(
    harness: ModalHarness,
    cellxgene: FakeCellxgene,
    result: dict[str, Any],
    ready: dict[str, Any],
) -> None:
    hub = harness.hub
    assert result["state"] == "completed", result
    assert result["datasets"] == [
        {
            "outcome": "skipped",
            "message": "The registered version already has a ready Scarf store",
            "cytebaseId": CYTEBASE_ID,
            "status": "ready",
        }
    ]
    assert cellxgene.downloads == [(SOURCE_URL, cellxgene.size)]
    run = hub.read_json(RUN_PATH)
    assert (run["runId"], run["state"]) == ("fc-again", "completed")
    assert run["children"][PROCESS_KEY] == {
        "cytebaseId": CYTEBASE_ID,
        "datasetVersionId": VERSION_ID,
        "stage": "process",
        "callId": "fc-process-2",
        "state": "skipped",
    }
    record = hub.read_json(RECORD_PATH)
    assert _pick(record, "status", "stageOutcome", "runId", "callId", "attempt") == {
        "status": "ready",
        "stageOutcome": "skipped",
        "runId": "fc-again",
        "callId": "fc-process-2",
        "attempt": 2,
    }
    build = (
        "processedVersionId",
        "processedAt",
        "zarrUri",
        "inspection",
        "buildReceipt",
    )
    assert _pick(record, *build) == _pick(ready, *build)


def _check_plot(result: Any, panel: str, legend: str, layout: dict) -> None:
    from scarf.plotting import PlotResult

    try:
        assert isinstance(result, PlotResult)
        assert list(result.axes) == [panel]
        assert [spec.kind for spec in result.legends] == [legend]
        assert result.provenance.n_cells == 6
        assert result.provenance.extras["layout"] == layout
    finally:
        result.close()


def test_cellxgene_collection_becomes_an_openable_cytebase_dataset(
    modal_harness, cellxgene, worker_tmp, tmp_path
):
    from scarf.cytebase import Catalog, connector

    hub = modal_harness.hub
    zarr_uri = f"{modal_harness.bucket.root}/{PREFIX}/data.zarr"
    assert _pipeline_name(cellxgene.collection) == CYTEBASE_ID

    # 1. Registration fetches metadata only; the SDK lists but cannot open it.
    registered = modal_harness.run(
        "register", {"collectionIds": [COLLECTION_ID]}, run_id="fc-register"
    )
    _check_registration(modal_harness, cellxgene, registered)
    catalog = Catalog(BUCKET_ID, token=False)
    assert catalog.find_datasets() == []
    (row,) = catalog.find_datasets(ready_only=False)
    assert _pick(row, "cytebase_id", "status", "zarr_uri") == {
        "cytebase_id": CYTEBASE_ID,
        "status": "registered",
        "zarr_uri": None,
    }
    assert catalog.query(
        "SELECT collection_id, name, first_author, year, n_datasets_total, "
        "n_datasets_main, skipped_dataset_ids FROM collections"
    ) == [
        {
            "collection_id": COLLECTION_ID,
            "name": "Human lung atlas",
            "first_author": "Smith",
            "year": 2024,
            "n_datasets_total": 1,
            "n_datasets_main": 1,
            "skipped_dataset_ids": [],
        }
    ]
    with pytest.raises(RuntimeError, match="not ready for its registered version"):
        catalog.dataset(CYTEBASE_ID).open()

    # 2. Processing downloads, converts, verifies, and publishes the store.
    processed = modal_harness.run(
        "process", {"collectionId": COLLECTION_ID}, run_id="fc-process"
    )
    record = _check_processing(modal_harness, cellxgene, processed, worker_tmp)
    umap_ref = hub.read_json(INGEST_PATH)["importedArtifacts"]["embeddings"]["X_umap"]
    published = hub.path(f"{PREFIX}/data.zarr")
    published_files = _files(published)

    # 3. The same catalog object refreshes and now finds the ready dataset.
    (row,) = catalog.find_datasets()
    assert _pick(
        row,
        "cytebase_id",
        "dataset_id",
        "collection_id",
        "latest_version_id",
        "processed_version_id",
        "status",
        "zarr_uri",
        "title",
        "citation",
        "first_author",
        "year",
        "cell_count",
        "primary_cell_count",
        "n_genes",
        "tissue_labels",
        "tissue_ids",
        "cell_type_labels",
        "processed_at",
        "pipeline_version",
    ) == {
        "cytebase_id": CYTEBASE_ID,
        "dataset_id": DATASET_ID,
        "collection_id": COLLECTION_ID,
        "latest_version_id": VERSION_ID,
        "processed_version_id": VERSION_ID,
        "status": "ready",
        "zarr_uri": zarr_uri,
        "title": TITLE,
        "citation": CITATION,
        "first_author": "Smith",
        "year": 2024,
        "cell_count": 6,
        "primary_cell_count": 4,
        "n_genes": 5,
        "tissue_labels": ["lung"],
        "tissue_ids": ["UBERON:0002048"],
        "cell_type_labels": ["B cell", "T cell"],
        "processed_at": datetime.fromisoformat(record["processedAt"]),
        "pipeline_version": PIPELINE_VERSION,
    }
    assert [row["cytebase_id"] for row in catalog.search("lung")] == [CYTEBASE_ID]
    assert [
        (term["label"], term["term_id"], term["n_datasets"])
        for term in catalog.list_terms("tissue")
    ] == [("lung", "UBERON:0002048", 1)]
    dataset = catalog.dataset(CYTEBASE_ID)
    assert dataset.describe() == DESCRIPTION

    target = tmp_path / "analysis.zarr"
    opened = []
    try:
        # 4. Read-only access returns exactly the source counts and annotations.
        datastore = dataset.open()
        opened.append(datastore)
        assert datastore.zw.read_only
        assert (datastore.cells.N, datastore.RNA.feats.N) == (6, 5)
        np.testing.assert_array_equal(np.asarray(datastore.RNA.rawData[:]), COUNTS)
        np.testing.assert_array_equal(np.asarray(datastore.RNA.rawDataT[:]), COUNTS.T)
        assert datastore.RNA.feats.fetch_all("names").tolist() == GENES
        cells = dataset.cell_metadata(["cell_type", "donor_id"])
        assert cells["cell_type"].tolist() == CELL_TYPES
        assert cells["donor_id"].tolist() == DONORS
        embeddings = dataset.embeddings()
        assert list(embeddings) == ["X_umap"]
        assert embeddings["X_umap"].to_dict() == umap_ref
        coordinates = dataset.embedding_coordinates()
        assert list(coordinates.columns) == ["umap_1", "umap_2"]
        assert coordinates.index.tolist() == [f"cell{i}" for i in range(6)]
        np.testing.assert_allclose(coordinates.to_numpy(), UMAP)
        _check_plot(
            dataset.plot_embedding(color_by="cell_type", show=False),
            "cell_type",
            "categorical",
            umap_ref,
        )
        _check_plot(
            dataset.plot_embedding(color_by=["CD3E"], show=False),
            "CD3E",
            "colorbar",
            umap_ref,
        )

        # 5. A writable local mount is pinned to the verified build.
        mounted = catalog.mount_dataset(CYTEBASE_ID, target)
        opened.append(mounted)
        assert not mounted.zw.read_only
        mounted.cells.insert("e2e_group", np.array(GROUPS))
        connector._close(opened.pop())
        sidecar = tmp_path / "analysis.zarr.cytebase.json"
        assert json.loads(sidecar.read_text()) == {
            "cytebaseId": CYTEBASE_ID,
            "datasetId": DATASET_ID,
            "processedVersionId": VERSION_ID,
            "sourceSha256": cellxgene.sha256,
            "zarrUri": zarr_uri,
            "verifiedAt": record["buildReceipt"]["verifiedAt"],
            "nObs": 6,
            "nVars": 5,
            "countsDtype": "int32",
        }

        # 6. Rerunning the ready dataset skips it and keeps the verified build.
        again = modal_harness.run(
            "process", {"cytebaseIds": [CYTEBASE_ID]}, run_id="fc-again"
        )
        _check_skipped_rerun(modal_harness, cellxgene, again, record)
        assert [row["cytebase_id"] for row in catalog.find_datasets()] == [CYTEBASE_ID]

        # 7. The local analysis reopens over the same published counts.
        reopened = dataset.mount(target)
        opened.append(reopened)
        assert reopened.cells.fetch_all("e2e_group").tolist() == GROUPS
        np.testing.assert_array_equal(np.asarray(reopened.RNA.rawData[:]), COUNTS)
    finally:
        for datastore in reversed(opened):
            connector._close(datastore)
    # Reading, plotting, mounting, and the rerun never changed the published store.
    assert _files(published) == published_files
