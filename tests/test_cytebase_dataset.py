"""Tests for the dataset handle returned by ``Catalog.dataset``."""

from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from scarf.cytebase.dataset import CytebaseDataset
from tests.fixtures_cytebase import (
    BUCKET_ID,
    CELL_TYPES,
    CITATION,
    CYTEBASE_ID,
    DONORS,
    UMAP,
    publish_catalog_rows,
)

pytestmark = pytest.mark.usefixtures("cytebase_offline")

RECORD_PATH = f"datasets/{CYTEBASE_ID}/dataset.json"
FULL_ROW = {
    "cytebase_id": CYTEBASE_ID,
    "title": "Healthy lung atlas",
    "citation": CITATION,
    "doi": "10.1000/lung",
    "cellxgene_url": "https://cellxgene.example/collection",
    "explorer_url": "https://cellxgene.example/explorer",
    "cell_count": 1234,
    "n_genes": 5,
    "status": "ready",
    "organism_labels": ["Homo sapiens"],
    "assay_labels": ["10x 3' v3"],
    "tissue_labels": ["lung", "trachea"],
    "disease_labels": ["normal"],
    "suspension_types": ["cell"],
}


def _catalog(storage=None, **methods) -> SimpleNamespace:
    return SimpleNamespace(_storage=storage, **methods)


def _record_reads(hub) -> int:
    return sum(
        1
        for call in hub.calls
        if call[0] == "download_bucket_files" and RECORD_PATH in call[2]
    )


def test_properties_come_from_the_catalog_row():
    dataset = CytebaseDataset(_catalog(), {"cytebase_id": 7, "cell_count": 3})
    assert dataset.id == "7"
    assert (dataset.title, dataset.citation, dataset.cell_count) == (None, None, 3)
    assert repr(dataset) == "CytebaseDataset('7')"
    dataset.row["cell_count"] = 4
    assert dataset.cell_count == 4


def test_record_is_read_once_and_must_exist(fake_hub):
    fake_hub.put(RECORD_PATH, {"inspection": {"embeddings": ["X_umap"]}})
    dataset = CytebaseDataset(_catalog(fake_hub.bucket()), FULL_ROW)
    assert dataset.record() == {"inspection": {"embeddings": ["X_umap"]}}
    assert dataset.source_embeddings() == ["X_umap"]
    assert _record_reads(fake_hub) == 1
    missing = CytebaseDataset(_catalog(fake_hub.bucket()), {"cytebase_id": "missing"})
    with pytest.raises(KeyError, match="No dataset record is published for 'missing'"):
        missing.record()


@pytest.mark.parametrize(
    "record", [{}, {"inspection": None}, {"inspection": {"embeddings": None}}]
)
def test_source_embeddings_default_to_empty(record):
    dataset = CytebaseDataset(_catalog(), FULL_ROW)
    dataset._record = record
    assert dataset.source_embeddings() == []


def test_describe_summarizes_the_dataset_with_a_split_citation():
    dataset = CytebaseDataset(_catalog(), FULL_ROW)
    dataset._record = {"inspection": {"embeddings": ["X_pca", "X_umap"]}}
    assert dataset.describe() == "\n".join(
        [
            "### Healthy lung atlas",
            "",
            f"- **Cytebase ID:** `{CYTEBASE_ID}`",
            "- **Citation:**",
            "    - Publication: https://doi.org/10.1000/lung",
            "    - Dataset Version: https://datasets.example.org/source.h5ad",
            "    - curated and distributed by CZ CELLxGENE Discover in Collection: "
            "https://cellxgene.cziscience.com/collections/"
            "11111111-1111-4111-8111-111111111111",
            "- **DOI:** 10.1000/lung",
            "- **CELLxGENE:** https://cellxgene.example/collection",
            "- **Explorer:** https://cellxgene.example/explorer",
            "- **Size:** 1,234 cells, 5 genes",
            "- **Status:** ready",
            "- **Organisms:** Homo sapiens",
            "- **Assays:** 10x 3' v3",
            "- **Tissues:** lung, trachea",
            "- **Diseases:** normal",
            "- **Suspension:** cell",
            "- **Source embeddings:** X_pca, X_umap",
        ]
    )
    assert dataset._repr_markdown_() == dataset.describe()


def test_describe_keeps_other_citations_on_one_line_and_skips_missing_fields(
    fake_hub,
):
    row = {
        "cytebase_id": "lung_b",
        "citation": "Smith et al. (2024) Lung Journal",
        "cell_count": 6,
        "status": "registered",
        "tissue_labels": [],
    }
    dataset = CytebaseDataset(_catalog(fake_hub.bucket()), row)
    assert dataset.describe() == "\n".join(
        [
            "### lung_b",
            "",
            "- **Cytebase ID:** `lung_b`",
            "- **Citation:** Smith et al. (2024) Lung Journal",
            "- **Size:** 6 cells",
            "- **Status:** registered",
        ]
    )


def test_open_reuses_the_datastore_and_rejects_new_options(tmp_path):
    opened = []
    mounted = []
    catalog = _catalog(
        open_dataset=lambda cytebase_id, **options: (
            opened.append((cytebase_id, options)) or "datastore"
        ),
        mount_dataset=lambda cytebase_id, at, **options: (
            mounted.append((cytebase_id, at, options)) or "mount"
        ),
    )
    dataset = CytebaseDataset(catalog, FULL_ROW)
    assert dataset.open(nthreads=2) == "datastore"
    assert dataset.open() == "datastore"
    assert opened == [(CYTEBASE_ID, {"nthreads": 2})]
    with pytest.raises(ValueError, match="already open"):
        dataset.open(nthreads=1)
    assert dataset.mount(tmp_path / "a.zarr", nthreads=1) == "mount"
    assert mounted == [(CYTEBASE_ID, tmp_path / "a.zarr", {"nthreads": 1})]


def test_embeddings_keep_keys_with_exactly_one_imported_artifact():
    requests = []
    refs = {"X_umap": ["umap"], "X_pca": [], "X_tsne": ["a", "b"]}

    def list_artifacts(**kwargs):
        requests.append(kwargs)
        return refs[kwargs["parameters"]["dimreduc_key"]]

    datastore = SimpleNamespace(list_artifacts=list_artifacts)
    dataset = CytebaseDataset(
        _catalog(open_dataset=lambda cytebase_id: datastore), FULL_ROW
    )
    dataset._record = {"inspection": {"embeddings": ["X_umap", "X_pca", "X_tsne"]}}
    assert dataset.embeddings() == {"X_umap": "umap"}
    assert requests[0] == {
        "kind": "embedding",
        "from_assay": "RNA",
        "operation": "import_dimreduc",
        "complete_only": True,
        "parameters": {"dimreduc_key": "X_umap"},
    }
    assert dataset.embedding() == "umap"
    with pytest.raises(KeyError, match="'X_pca' was not imported .* X_umap"):
        dataset.embedding("X_pca")
    dataset._record = {}
    with pytest.raises(KeyError, match="available: none"):
        dataset.embedding()


@pytest.fixture
def published_dataset(ready_dataset, fake_hub):
    from scarf.cytebase import Catalog
    from scarf.cytebase.connector import _close

    publish_catalog_rows(fake_hub, [ready_dataset.record])
    dataset = Catalog(BUCKET_ID, token=False).dataset(CYTEBASE_ID)
    yield dataset
    if dataset._datastore is not None:
        _close(dataset._datastore)


def test_published_dataset_exposes_metadata_and_coordinates(published_dataset):
    dataset = published_dataset
    assert "**Status:** ready" in dataset.describe()
    metadata = dataset.cell_metadata(["cell_type", "donor_id"])
    assert list(metadata.columns) == ["cell_type", "donor_id"]
    assert metadata["cell_type"].tolist() == CELL_TYPES
    assert metadata["donor_id"].tolist() == DONORS
    assert {"ids", "is_primary_data", "RNA_nCounts"} <= set(dataset.cell_metadata())
    assert list(dataset.embeddings()) == ["X_umap"]
    coordinates = dataset.embedding_coordinates()
    assert list(coordinates.columns) == ["umap_1", "umap_2"]
    assert coordinates.index.tolist() == [f"cell{i}" for i in range(6)]
    np.testing.assert_allclose(coordinates.to_numpy(), UMAP)


def test_published_dataset_plots_annotations_and_genes(published_dataset):
    pytest.importorskip("matplotlib")
    for color_by in ("cell_type", ["CD3E", "LYZ"]):
        result = published_dataset.plot_embedding(color_by=color_by, show=False)
        try:
            assert result.figure is not None
        finally:
            result.close()


def test_embedding_coordinates_require_a_matching_cell_selection(
    published_dataset, monkeypatch
):
    datastore = published_dataset.open()
    status = datastore.inspect_artifact(published_dataset.embedding())
    monkeypatch.setattr(
        datastore,
        "inspect_artifact",
        lambda ref: SimpleNamespace(inputs={}, parameters=status.parameters),
    )
    with pytest.raises(ValueError, match="has no cell-selection input"):
        published_dataset.embedding_coordinates()
    monkeypatch.setattr(datastore, "inspect_artifact", lambda ref: status)
    monkeypatch.setattr(
        datastore,
        "load_artifact",
        lambda ref: {"values": zarr.array(np.zeros((2, 2), dtype=np.float32))},
    )
    with pytest.raises(ValueError, match="does not match its cell selection"):
        published_dataset.embedding_coordinates()
