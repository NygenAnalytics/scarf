"""Tests for the metadata-only entry returned by ``Catalog.dataset``."""

from types import SimpleNamespace

import pytest

from scarf.cytebase.entry import DatasetEntry
from tests.fixtures_cytebase import CITATION, CYTEBASE_ID

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
    entry = DatasetEntry(_catalog(), {"cytebase_id": 7, "cell_count": 3})
    assert entry.id == "7"
    assert (entry.title, entry.citation, entry.cell_count) == (None, None, 3)
    assert repr(entry) == "DatasetEntry('7')"
    entry.row["cell_count"] = 4
    assert entry.cell_count == 4


def test_record_is_read_once_and_must_exist(fake_hub):
    fake_hub.put(RECORD_PATH, {"inspection": {"embeddings": ["X_umap"]}})
    entry = DatasetEntry(_catalog(fake_hub.bucket()), FULL_ROW)
    assert entry.record() == {"inspection": {"embeddings": ["X_umap"]}}
    assert entry.source_embeddings() == ["X_umap"]
    assert _record_reads(fake_hub) == 1
    missing = DatasetEntry(_catalog(fake_hub.bucket()), {"cytebase_id": "missing"})
    with pytest.raises(KeyError, match="No dataset record is published for 'missing'"):
        missing.record()


@pytest.mark.parametrize(
    "record", [{}, {"inspection": None}, {"inspection": {"embeddings": None}}]
)
def test_source_embeddings_default_to_empty(record):
    entry = DatasetEntry(_catalog(), FULL_ROW)
    entry._record = record
    assert entry.source_embeddings() == []


def test_describe_summarizes_the_dataset_with_a_split_citation():
    entry = DatasetEntry(_catalog(), FULL_ROW)
    entry._record = {"inspection": {"embeddings": ["X_pca", "X_umap"]}}
    assert entry.describe() == "\n".join(
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
    assert entry._repr_markdown_() == entry.describe()


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
    entry = DatasetEntry(_catalog(fake_hub.bucket()), row)
    assert entry.describe() == "\n".join(
        [
            "### lung_b",
            "",
            "- **Cytebase ID:** `lung_b`",
            "- **Citation:** Smith et al. (2024) Lung Journal",
            "- **Size:** 6 cells",
            "- **Status:** registered",
        ]
    )


def test_entry_discovery_never_opens_a_datastore(fake_hub):
    def unexpected_open(*args, **kwargs):
        raise AssertionError("Dataset entries must not open or mount DataStores")

    fake_hub.put(RECORD_PATH, {"inspection": {"embeddings": ["X_umap"]}})
    catalog = _catalog(
        fake_hub.bucket(),
        open_datastore=unexpected_open,
        mount_datastore=unexpected_open,
    )
    entry = DatasetEntry(catalog, FULL_ROW)
    assert (entry.id, entry.title, entry.citation, entry.cell_count) == (
        CYTEBASE_ID,
        FULL_ROW["title"],
        CITATION,
        FULL_ROW["cell_count"],
    )
    assert entry.source_embeddings() == ["X_umap"]
    assert entry._repr_markdown_() == entry.describe()
    assert "**Source embeddings:** X_umap" in entry.describe()
    assert _record_reads(fake_hub) == 1
    for attribute in (
        "_datastore",
        "open",
        "mount",
        "cell_metadata",
        "embeddings",
        "embedding",
        "embedding_coordinates",
        "plot_embedding",
    ):
        assert not hasattr(entry, attribute)
