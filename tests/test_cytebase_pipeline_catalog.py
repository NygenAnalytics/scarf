"""Offline tests for CELLxGENE registration and DuckDB catalog publication."""

import hashlib
import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import httpx
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("natsort")

import duckdb

from scarf.cytebase.pipeline import catalog
from scarf.cytebase.pipeline.models import DatasetRecord, ProcessRequest
from tests.fixtures_cytebase import (
    BUCKET_ID,
    CITATION,
    COLLECTION_ID,
    CYTEBASE_ID,
    DATASET_ID,
    NEW_VERSION_ID,
    NOW,
    PIPELINE_VERSION,
    SOURCE_URL,
    VERSION_ID,
    FakeHub,
    cellxgene_collection,
    cellxgene_dataset,
    dataset_record,
    noop,
    publish_catalog_rows,
)

pytestmark = pytest.mark.usefixtures("cytebase_offline")

API = "https://api.cellxgene.cziscience.com/curation/v1"
DB_PATH = "catalog/cytebase.duckdb"
HASH_PATH = "catalog/cytebase.duckdb.sha256"
REGISTERED = datetime.fromisoformat(NOW)
LATER = datetime(2026, 3, 4, 5, 6, 7, tzinfo=UTC)

OTHER_COLLECTION = "66666666-6666-4666-8666-666666666666"
REVIEW_COLLECTION = "77777777-7777-4777-8777-777777777777"
LEAK_COLLECTION = "88888888-8888-4888-8888-888888888888"
BRAIN_COLLECTION = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
# Shares the first eight hex digits of DATASET_ID, so its short name collides.
SIBLING_DATASET = "22222222-9999-4999-8999-999999999999"
SIBLING_ID = "smith_2024_healthy_lung_scrna_atlas_222222229999"
BLOOD_DATASET = "55555555-5555-4555-8555-555555555555"
BLOOD_ID = "doe_2023_blood_atlas_55555555"
ATAC_DATASET = "99999999-9999-4999-8999-999999999999"
SECONDARY_DATASET = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
UNREVIEWED_ASSAY = [{"label": "mystery assay", "ontology_term_id": "EFO:9999999"}]
ATAC_ASSAY = [{"label": "scATAC-seq", "ontology_term_id": "EFO:0010891"}]
NAME_PARTS = ["smith", "2024", "healthy_lung_scrna_atlas"]
H5AD_ASSET = {"filetype": "H5AD", "url": SOURCE_URL, "filesize": 1234}


def _record(**overrides: Any) -> DatasetRecord:
    return DatasetRecord.model_validate(dataset_record(**overrides))


def _tissues(*labels: str) -> dict[str, list[dict[str, str | None]]]:
    """The default record facets with unannotated ``tissue`` labels."""
    return dataset_record()["facets"] | {
        "tissue": [{"termId": None, "label": label} for label in labels]
    }


def _secondary(**overrides: Any) -> dict[str, Any]:
    return cellxgene_dataset(primary_cell_count=0, is_primary_data=[False], **overrides)


def _collection_row(
    collection_id: str, name: str, registered_at: datetime
) -> dict[str, Any]:
    return {
        "collection_id": collection_id,
        "name": name,
        "description": None,
        "doi": "10.1000/lung",
        "first_author": "Smith",
        "year": 2024,
        "journal": None,
        "consortia": ["Lung Network"],
        "n_datasets_total": 2,
        "n_datasets_main": 1,
        "skipped_dataset_ids": [SECONDARY_DATASET],
        "skipped_dataset_reasons": {SECONDARY_DATASET: "All cells are secondary"},
        "registered_at": registered_at,
    }


def _prepare(
    collections: list[dict[str, Any]],
    existing: list[DatasetRecord] | None = None,
    now: datetime | None = LATER,
) -> tuple[list[DatasetRecord], list[dict], list[tuple[bytes, str]]]:
    return catalog.prepare_registration(
        [(json.dumps(collection).encode(), collection) for collection in collections],
        existing or [],
        pipeline_version=PIPELINE_VERSION,
        now=now,
    )


def _response(url: str, status: int = 200, **body: Any) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("GET", url), **body)


def _patch_get(
    monkeypatch, answer: Callable[[str], httpx.Response]
) -> list[tuple[str, dict[str, Any]]]:
    calls = []

    def get(url: str, **kwargs: Any) -> httpx.Response:
        calls.append((url, kwargs))
        return answer(url)

    monkeypatch.setattr("scarf.cytebase.pipeline.catalog.httpx.get", get)
    return calls


def _serve_collections(monkeypatch, answers: dict[str, bytes | Exception]) -> list[str]:
    """Answer collection requests by ID and record which IDs were fetched."""
    requested = []

    def answer(url: str) -> httpx.Response:
        collection_id = url.removeprefix(f"{API}/collections/")
        requested.append(collection_id)
        body = answers[collection_id]
        if isinstance(body, Exception):
            raise body
        return _response(url, content=body)

    _patch_get(monkeypatch, answer)
    return requested


def _register(hub: FakeHub, *records: dict[str, Any]) -> None:
    for record in records:
        hub.put(f"datasets/{record['cytebaseId']}/dataset.json", record)


def _uploads(hub: FakeHub) -> list[list[str]]:
    return [call[2] for call in hub.calls if call[0] == "batch_bucket_files"]


def _tables(hub: FakeHub) -> dict[str, list[dict[str, Any]]]:
    order = {
        "datasets": "cytebase_id",
        "collections": "collection_id",
        "dataset_terms": "cytebase_id, facet, label_rank, term_id",
    }
    tables = {}
    with duckdb.connect(str(hub.path(DB_PATH)), read_only=True) as database:
        for name, columns in order.items():
            result = database.execute(f"SELECT * FROM {name} ORDER BY {columns}")
            names = [column[0] for column in result.description]
            tables[name] = [
                dict(zip(names, row, strict=True)) for row in result.fetchall()
            ]
    return tables


def _tissue_ranks(tables: dict[str, list[dict[str, Any]]]) -> list[tuple]:
    return [
        (row["cytebase_id"], row["label"], row["label_rank"])
        for row in tables["dataset_terms"]
        if row["facet"] == "tissue"
    ]


@pytest.fixture
def registered(fake_hub) -> FakeHub:
    """Two lung datasets in one collection and a blood dataset in another."""
    _register(
        fake_hub,
        dataset_record(),
        dataset_record(cytebaseId=SIBLING_ID, datasetId=SIBLING_DATASET),
        dataset_record(
            cytebaseId=BLOOD_ID,
            datasetId=BLOOD_DATASET,
            collectionId=OTHER_COLLECTION,
            title="Blood atlas",
        ),
    )
    return fake_hub


# CELLxGENE curation API


def test_fetch_collection_returns_the_response_bytes_and_collection(monkeypatch):
    collection = cellxgene_collection()
    raw = json.dumps(collection, separators=(",", ":")).encode()
    calls = _patch_get(monkeypatch, lambda url: _response(url, content=raw))
    assert catalog.fetch_collection(COLLECTION_ID) == (raw, collection)
    assert calls == [
        (
            f"{API}/collections/{COLLECTION_ID}",
            {"timeout": 30, "follow_redirects": True},
        )
    ]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ([cellxgene_collection()], "must be a JSON object"),
        (
            cellxgene_collection(collection_id=OTHER_COLLECTION),
            "different collection identity",
        ),
        ({"name": "Unidentified"}, "different collection identity"),
    ],
)
def test_fetch_collection_rejects_unexpected_bodies(monkeypatch, body, message):
    _patch_get(monkeypatch, lambda url: _response(url, json=body))
    with pytest.raises(ValueError, match=message):
        catalog.fetch_collection(COLLECTION_ID)


def test_get_retries_transient_server_errors(monkeypatch, recorded_sleeps):
    statuses = iter([503, 200])
    calls = _patch_get(
        monkeypatch, lambda url: _response(url, next(statuses), json={"ok": True})
    )
    response = catalog._get(f"{API}/collections")
    assert (response.status_code, response.json()) == (200, {"ok": True})
    assert [url for url, _ in calls] == [f"{API}/collections"] * 2
    assert recorded_sleeps == [2.0]


def test_get_raises_client_errors_without_retrying(monkeypatch, recorded_sleeps):
    calls = _patch_get(monkeypatch, lambda url: _response(url, 404))
    with pytest.raises(httpx.HTTPStatusError, match="404 Not Found"):
        catalog._get(f"{API}/collections/{COLLECTION_ID}")
    assert len(calls) == 1
    assert recorded_sleeps == []


def test_list_collection_ids_normalizes_deduplicates_and_sorts(monkeypatch):
    rows = [
        {"collection_id": "AbCdEf01-2345-4678-9aBc-DeF012345678"},
        {"collection_id": OTHER_COLLECTION},
        {"collection_id": COLLECTION_ID.replace("-", "")},
        {"collection_id": "{" + COLLECTION_ID + "}"},
        {"collection_id": COLLECTION_ID},
    ]
    calls = _patch_get(monkeypatch, lambda url: _response(url, json=rows))
    assert catalog.list_collection_ids() == [
        COLLECTION_ID,
        OTHER_COLLECTION,
        "abcdef01-2345-4678-9abc-def012345678",
    ]
    assert [url for url, _ in calls] == [f"{API}/collections?visibility=PUBLIC"]


def test_list_collection_ids_requires_an_array(monkeypatch):
    _patch_get(monkeypatch, lambda url: _response(url, json={"collections": []}))
    with pytest.raises(ValueError, match="must be an array"):
        catalog.list_collection_ids()


# Source metadata and manifest fields


def test_source_metadata_selects_the_single_h5ad_asset():
    collection = cellxgene_collection()
    dataset = cellxgene_dataset(
        assets=[
            {"filetype": "RDS", "url": "https://datasets.example.org/a.rds"},
            {"filetype": "h5ad", "url": SOURCE_URL, "filesize": 1234},
        ]
    )
    assert catalog.source_metadata(collection, dataset) == catalog.SourceMetadata(
        collection=collection,
        dataset=dataset,
        source_url=SOURCE_URL,
        source_bytes=1234,
    )


def test_source_metadata_allows_an_unknown_file_size():
    dataset = cellxgene_dataset(assets=[H5AD_ASSET | {"filesize": None}])
    assert catalog.source_metadata({}, dataset).source_bytes is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"dataset_id": None}, "missing its stable or version ID"),
        ({"dataset_version_id": ""}, "missing its stable or version ID"),
        ({"assets": []}, "one H5AD download asset"),
        ({"assets": [{"filetype": "RDS", "url": SOURCE_URL}]}, "one H5AD download"),
        ({"assets": [H5AD_ASSET, H5AD_ASSET]}, "one H5AD download asset"),
        ({"assets": [{"filetype": "H5AD", "filesize": 1}]}, "one H5AD download"),
    ],
)
def test_source_metadata_rejects_incomplete_datasets(overrides, message):
    with pytest.raises(ValueError, match=message):
        catalog.source_metadata(cellxgene_collection(), cellxgene_dataset(**overrides))


@pytest.mark.parametrize(
    ("publisher", "citation"),
    [
        (
            {
                "authors": [{"family": "Smith", "name": "Ignored"}, {"name": "Team"}],
                "published_year": 2024,
                "journal": "Lung Journal",
            },
            "Smith et al. (2024) Lung Journal",
        ),
        (
            {"authors": [{"name": "Lung Team"}], "published_year": 2024},
            "Lung Team (2024)",
        ),
        (
            {"authors": [{"name": "Lung Team"}, {"family": "Smith"}], "journal": "J"},
            "Lung Team et al. J",
        ),
        ({"published_year": 2020, "journal": "Lung Journal"}, "(2020) Lung Journal"),
        ({"authors": [{"given": "Ada"}]}, ""),
        (None, ""),
    ],
)
def test_publication_citation_formats_author_year_and_journal(publisher, citation):
    assert catalog._publication_citation({"publisher_metadata": publisher}) == citation


def test_metadata_manifest_fields_prefer_dataset_values():
    source = catalog.source_metadata(cellxgene_collection(), cellxgene_dataset())
    assert catalog.metadata_manifest_fields(source) == {
        "title": "Healthy lung scRNA-seq atlas",
        "citation": CITATION,
        "doi": "10.1000/lung",
        "schemaVersion": "5.3.0",
        "organism": "Homo sapiens",
        "metadataSource": "curation_api",
    }


def test_metadata_manifest_fields_fall_back_to_collection_values():
    dataset = cellxgene_dataset(
        title=None,
        citation="",
        organism=[
            {"label": "Homo sapiens"},
            {"ontology_term_id": "NCBITaxon:7955"},
            {"label": ""},
            {"label": "Mus musculus"},
        ],
    )
    source = catalog.SourceMetadata(cellxgene_collection(), dataset, SOURCE_URL, None)
    fields = catalog.metadata_manifest_fields(source)
    assert fields["title"] == "Human lung atlas"
    assert fields["citation"] == "Smith et al. (2024) Lung Journal"
    assert fields["organism"] == "Homo sapiens, Mus musculus"


def test_metadata_manifest_fields_drop_missing_values():
    source = catalog.SourceMetadata({}, {"organism": None}, SOURCE_URL, None)
    assert catalog.metadata_manifest_fields(source) == {
        "metadataSource": "curation_api"
    }


# Names and facets


@pytest.mark.parametrize(
    ("text", "slug"),
    [
        ("Müller-Lüdenscheidt", "muller_ludenscheidt"),
        ("  --Hello, World!--  ", "hello_world"),
        ("a" * 30, "a" * 24),
        ("abcdefghijklmnopqrstuvw xyz", "abcdefghijklmnopqrstuvw"),
        ("東京", ""),
    ],
)
def test_slug_is_short_lowercase_ascii(text, slug):
    assert catalog._slug(text) == slug


@pytest.mark.parametrize(
    ("publisher", "expected"),
    [
        (
            {"authors": [{"family": "García-Márquez"}], "published_year": 2020},
            ("García-Márquez", 2020, "garciamarquez"),
        ),
        (
            {
                "authors": [{"name": "The Tabula Sapiens Consortium"}],
                "published_year": "2022",
            },
            ("The Tabula Sapiens Consortium", 2022, "tabula"),
        ),
        ({"authors": [{"name": "A An The Lab"}]}, ("A An The Lab", None, "lab")),
        (
            {"authors": [{"family": "", "name": "Human Cell Atlas"}]},
            ("Human Cell Atlas", None, "human"),
        ),
        ({"authors": [{"name": "The"}]}, ("The", None, "unpub")),
        ({"authors": [{"name": "!!!"}]}, ("!!!", None, "unpub")),
        ({"authors": [{"family": "東京"}]}, ("東京", None, "unpub")),
        ({"authors": [{"name": "東京 Lab"}]}, ("東京 Lab", None, "unpub")),
        ({}, (None, None, "unpub")),
        (None, (None, None, "unpub")),
    ],
)
def test_publication_derives_author_year_and_name_slug(publisher, expected):
    assert catalog._publication({"publisher_metadata": publisher}) == expected


def test_facets_deduplicate_and_sort_terms():
    dataset = {
        "organism": [
            {"label": "Homo sapiens", "ontology_term_id": "NCBITaxon:9606"},
            {"label": "Homo sapiens", "ontology_term_id": "NCBITaxon:9606"},
        ],
        "cell_type": [
            {"label": "T cell", "ontology_term_id": "CL:0000084"},
            {"label": "B cell", "ontology_term_id": "CL:0000236"},
            {"label": "B cell", "ontology_term_id": "CL:0000001"},
            {"label": "B cell"},
        ],
        "tissue": None,
        "organ": [{"label": "lung", "ontology_term_id": "UBERON:0002048"}],
        "suspension_type": ["nucleus", "cell", "cell"],
    }
    facets = {
        facet: [term.model_dump() for term in terms]
        for facet, terms in catalog._facets(dataset).items()
    }
    assert facets == {
        "organism": [{"termId": "NCBITaxon:9606", "label": "Homo sapiens"}],
        "assay": [],
        "tissue": [],
        "disease": [],
        "cell_type": [
            {"termId": None, "label": "B cell"},
            {"termId": "CL:0000001", "label": "B cell"},
            {"termId": "CL:0000236", "label": "B cell"},
            {"termId": "CL:0000084", "label": "T cell"},
        ],
        "sex": [],
        "development_stage": [],
        "organ": [],
        "suspension_type": [
            {"termId": None, "label": "cell"},
            {"termId": None, "label": "nucleus"},
        ],
    }
    assert set(facets) == set(catalog.FACETS)


@pytest.mark.parametrize(
    ("title", "slug"),
    [
        ("Healthy lung scRNA-seq atlas", "healthy_lung_scrna_atlas"),
        ("snRNA seq of the human brain", "snrna_human_brain"),
        ("scRNA_seq", "scrna"),
        ("scrnaseq: T cells in the gut", "scrna_t_cells_gut"),
        ("Single-cell RNA-seq", "single_cell_rna_seq"),
        ("Café à la carte", "cafe_la_carte"),
        ("The of and", "dataset"),
        ("", "dataset"),
    ],
)
def test_title_slug_keeps_descriptive_words(title, slug):
    assert catalog._title_slug(title) == slug


@pytest.mark.parametrize(
    ("organisms", "organism_part"),
    [
        (["Homo sapiens"], []),
        (["Mus musculus"], ["mouse"]),
        (["Danio rerio"], ["danio_rerio"]),
        (["Mus musculus", "Homo sapiens"], ["multispecies"]),
        ([], ["unknown"]),
    ],
)
def test_name_parts_name_non_human_organisms(organisms, organism_part):
    dataset = cellxgene_dataset(
        organism=[
            {"label": label, "ontology_term_id": f"NCBITaxon:{index}"}
            for index, label in enumerate(organisms)
        ]
    )
    parts = catalog._name_parts(
        cellxgene_collection(), catalog._facets(dataset), dataset
    )
    assert parts == ["smith", "2024", *organism_part, "healthy_lung_scrna_atlas"]


@pytest.mark.parametrize(
    ("collection_overrides", "title", "parts"),
    [
        ({"publisher_metadata": None}, "Lung atlas", ["unpub", "0000", "lung_atlas"]),
        ({}, None, ["smith", "2024", "human_lung_atlas"]),
        ({"name": None}, None, ["smith", "2024", "dataset"]),
    ],
)
def test_name_parts_fall_back_for_missing_metadata(collection_overrides, title, parts):
    dataset = cellxgene_dataset(title=title)
    collection = cellxgene_collection(**collection_overrides)
    assert catalog._name_parts(collection, catalog._facets(dataset), dataset) == parts


@pytest.mark.parametrize(
    ("part", "limit", "shortened"),
    [
        ("healthy_lung", 12, "healthy_lung"),
        ("abcdef", 2, "ab"),
        ("one_two_three_four_five_six", 12, "one_six"),
        ("ab_cd_efghij_kl_mn", 11, "ab_cd_kl_mn"),
        ("mouse_hippocampus_slide_seq_puck_200115_08", 32, "mouse_puck_200115_08"),
        ("abcdefghijklmnop", 7, "abc_nop"),
    ],
)
def test_shorten_name_part_keeps_whole_words_from_both_ends(part, limit, shortened):
    assert catalog._shorten_name_part(part, limit) == shortened
    assert len(shortened) <= limit


@pytest.mark.parametrize(
    ("parts", "name"),
    [
        (["a" * 30, "b" * 40], f"{'a' * 30}_{'b' * 40}_12345678"),
        (["a" * 30, "b" * 41], f"{'a' * 30}_{'b' * 19}_{'b' * 20}_12345678"),
        (
            ["a" * 50, "b" * 50],
            f"{'a' * 17}_{'a' * 17}_{'b' * 17}_{'b' * 17}_12345678",
        ),
    ],
)
def test_name_with_suffix_shortens_the_longest_parts_to_80_characters(parts, name):
    assert catalog._name_with_suffix(parts, "12345678") == name
    assert len(name) <= 80


def test_assign_name_lengthens_the_suffix_on_collisions():
    first = UUID(DATASET_ID)
    second = UUID(SIBLING_DATASET)
    third = UUID("22222222-9999-4aaa-8aaa-aaaaaaaaaaaa")
    reserved: dict[str, UUID] = {}
    names = [
        catalog._assign_name(NAME_PARTS, dataset_id, reserved)
        for dataset_id in (first, second, third, first)
    ]
    longest = "smith_2024_healthy_lung_scrna_atlas_2222222299994aaa8aaaaaaaaaaaaaaa"
    assert names == [CYTEBASE_ID, SIBLING_ID, longest, CYTEBASE_ID]
    assert reserved == {CYTEBASE_ID: first, SIBLING_ID: second, longest: third}


def test_assign_name_fails_when_every_suffix_is_taken():
    dataset_id = UUID(DATASET_ID)
    other = UUID(SIBLING_DATASET)
    reserved = {
        catalog._name_with_suffix(NAME_PARTS, dataset_id.hex[:length]): other
        for length in (8, 12, 32)
    }
    with pytest.raises(
        ValueError, match=f"unique Cytebase name for dataset {DATASET_ID}"
    ):
        catalog._assign_name(NAME_PARTS, dataset_id, reserved)


def test_readme_describes_the_dataset_and_its_collection():
    readme = catalog._readme(_record(), cellxgene_collection()).decode()
    assert readme == "\n".join(
        [
            "# Healthy lung scRNA-seq atlas",
            "",
            CITATION,
            "",
            "DOI: https://doi.org/10.1000/lung",
            "",
            f"Collection: https://cellxgene.cziscience.com/collections/{COLLECTION_ID}",
            f"Dataset ID: `{DATASET_ID}`",
            f"Latest dataset version: `{VERSION_ID}`",
            f"Source H5AD: {SOURCE_URL}",
            "CELLxGENE Explorer: https://cellxgene.cziscience.com/e/lung.cxg/",
            "",
            "## Metadata",
            "",
            "- organism: Homo sapiens",
            "- assay: 10x 3' v3",
            "- tissue: lung",
            "- disease: normal",
            "- cell_type: B cell, T cell",
            "- sex: female",
            "- development_stage: adult stage",
            "- suspension_type: cell",
            "",
            "## Abstract",
            "",
            "Single-cell profiles of healthy human lung.",
            "",
        ]
    )


def test_readme_omits_missing_sections():
    record = _record(
        title=None,
        citation=None,
        doi=None,
        explorerUrl=None,
        facets={"organ": [], "tissue": []},
    )
    assert catalog._readme(record, {"description": ""}).decode() == "\n".join(
        [
            f"# {CYTEBASE_ID}",
            "",
            f"Collection: https://cellxgene.cziscience.com/collections/{COLLECTION_ID}",
            f"Dataset ID: `{DATASET_ID}`",
            f"Latest dataset version: `{VERSION_ID}`",
            f"Source H5AD: {SOURCE_URL}",
            "",
            "## Metadata",
            "",
            "",
        ]
    )


# Registration


def test_prepare_registration_builds_a_new_registration():
    title = "Healthy lung scRNA-seq atlas (Zürich)"
    name = "smith_2024_healthy_lung_scrna_atlas_zurich_22222222"
    dataset = cellxgene_dataset(title=title)
    collection = cellxgene_collection([dataset])
    raw = json.dumps(collection, separators=(",", ":")).encode()
    records, rows, uploads = catalog.prepare_registration(
        [(raw, collection)], [], pipeline_version=PIPELINE_VERSION, now=LATER
    )
    expected = _record(
        cytebaseId=name,
        title=title,
        versions=[{"datasetVersionId": VERSION_ID, "seenAt": LATER}],
        registeredAt=LATER,
        updatedAt=LATER,
    )
    assert [record.model_dump(mode="json") for record in records] == [
        expected.model_dump(mode="json")
    ]
    assert rows == [
        {
            "collection_id": COLLECTION_ID,
            "name": "Human lung atlas",
            "description": "Single-cell profiles of healthy human lung.",
            "doi": "10.1000/lung",
            "first_author": "Smith",
            "year": 2024,
            "journal": "Lung Journal",
            "consortia": ["Lung Network"],
            "n_datasets_total": 1,
            "n_datasets_main": 1,
            "skipped_dataset_ids": [],
            "skipped_dataset_reasons": {},
            "registered_at": LATER,
        }
    ]
    assert [path for _, path in uploads] == [
        f"datasets/{name}/cellxgene/collection.json",
        f"datasets/{name}/cellxgene/dataset.json",
        f"datasets/{name}/README.md",
    ]
    assert uploads[0][0] == raw
    assert json.loads(uploads[1][0]) == dataset
    assert uploads[1][0].startswith(b'{\n  "dataset_id": ')
    assert "Zürich".encode() in uploads[1][0]
    assert uploads[2][0].startswith(f"# {title}\n".encode())


def test_prepare_registration_uses_the_current_time_by_default():
    before = datetime.now(UTC)
    records, rows, _ = _prepare([cellxgene_collection()], now=None)
    after = datetime.now(UTC)
    assert before <= records[0].registeredAt == records[0].updatedAt <= after
    assert rows[0]["registered_at"] == records[0].registeredAt


def test_prepare_registration_tolerates_missing_collection_metadata():
    collection = cellxgene_collection(
        collection_url=None, consortia=None, publisher_metadata=None
    )
    records, rows, _ = _prepare([collection])
    assert records[0].cellxgeneUrl == (
        f"https://cellxgene.cziscience.com/collections/{COLLECTION_ID}"
    )
    assert records[0].cytebaseId == "unpub_0000_healthy_lung_scrna_atlas_22222222"
    assert (records[0].firstAuthor, records[0].year) == (None, None)
    assert records[0].citation == CITATION
    assert (rows[0]["consortia"], rows[0]["journal"]) == ([], None)


@pytest.mark.parametrize(
    ("previous", "version_id", "status"),
    [
        ({"status": "ready", "processedVersionId": VERSION_ID}, VERSION_ID, "ready"),
        (
            {"status": "ready", "processedVersionId": VERSION_ID},
            NEW_VERSION_ID,
            "update_available",
        ),
        ({"status": "registered"}, NEW_VERSION_ID, "registered"),
        ({"status": "failed"}, NEW_VERSION_ID, "registered"),
        ({"status": "failed"}, VERSION_ID, "failed"),
    ],
    ids=[
        "ready-same-version",
        "ready-new-version",
        "registered-new-version",
        "failed-new-version",
        "failed-same-version",
    ],
)
def test_prepare_registration_updates_an_existing_registration(
    previous, version_id, status
):
    existing = _record(
        cytebaseId="legacy_lung_name",
        zarrUri="/published/lung/data.zarr",
        pipelineVersion="old-pipeline",
        **previous,
    )
    dataset = cellxgene_dataset(dataset_version_id=version_id, title="Revised atlas")
    records, rows, uploads = _prepare([cellxgene_collection([dataset])], [existing])
    (record,) = records
    assert record.cytebaseId == "legacy_lung_name"
    assert (record.title, record.status) == ("Revised atlas", status)
    assert record.latestVersionId == UUID(version_id)
    seen = [(UUID(VERSION_ID), REGISTERED)]
    if version_id == NEW_VERSION_ID:
        seen.append((UUID(NEW_VERSION_ID), LATER))
    assert [(v.datasetVersionId, v.seenAt) for v in record.versions] == seen
    assert (record.registeredAt, record.updatedAt) == (REGISTERED, LATER)
    assert record.pipelineVersion == PIPELINE_VERSION
    assert record.zarrUri == "/published/lung/data.zarr"
    assert record.processedVersionId == existing.processedVersionId
    assert rows[0]["registered_at"] == REGISTERED
    assert {path.split("/")[1] for _, path in uploads} == {"legacy_lung_name"}


def test_prepare_registration_keeps_the_earliest_collection_registration():
    existing = [
        _record(registeredAt="2025-05-01T00:00:00+00:00"),
        _record(
            cytebaseId=SIBLING_ID,
            datasetId=SIBLING_DATASET,
            registeredAt="2025-01-01T00:00:00+00:00",
        ),
        _record(
            cytebaseId=BLOOD_ID,
            datasetId=BLOOD_DATASET,
            collectionId=OTHER_COLLECTION,
            registeredAt="2024-01-01T00:00:00+00:00",
        ),
    ]
    _, rows, _ = _prepare([cellxgene_collection()], existing)
    assert rows[0]["registered_at"] == datetime(2025, 1, 1, tzinfo=UTC)


def test_prepare_registration_avoids_names_held_by_other_datasets():
    holder = _record(datasetId=SIBLING_DATASET)
    records, _, uploads = _prepare([cellxgene_collection()], [holder])
    name = "smith_2024_healthy_lung_scrna_atlas_222222222222"
    assert records[0].cytebaseId == name
    assert uploads[0][1] == f"datasets/{name}/cellxgene/collection.json"


def test_prepare_registration_orders_collections_and_datasets_by_id():
    lung = cellxgene_collection(
        [cellxgene_dataset(dataset_id=SIBLING_DATASET), cellxgene_dataset()]
    )
    blood = cellxgene_collection(
        [cellxgene_dataset(dataset_id=BLOOD_DATASET, title="Blood atlas")],
        collection_id=OTHER_COLLECTION,
    )
    records, rows, uploads = _prepare([blood, lung])
    assert [row["collection_id"] for row in rows] == [COLLECTION_ID, OTHER_COLLECTION]
    assert [record.cytebaseId for record in records] == [
        CYTEBASE_ID,
        SIBLING_ID,
        "smith_2024_blood_atlas_55555555",
    ]
    assert len(uploads) == 9


def test_prepare_registration_records_skipped_datasets():
    collection = cellxgene_collection(
        [
            cellxgene_dataset(dataset_id=ATAC_DATASET, assay=ATAC_ASSAY),
            cellxgene_dataset(),
            _secondary(dataset_id=SECONDARY_DATASET),
        ]
    )
    records, rows, uploads = _prepare([collection])
    assert [record.cytebaseId for record in records] == [CYTEBASE_ID]
    assert len(uploads) == 3
    assert (rows[0]["n_datasets_total"], rows[0]["n_datasets_main"]) == (3, 2)
    assert rows[0]["skipped_dataset_ids"] == [ATAC_DATASET, SECONDARY_DATASET]
    assert rows[0]["skipped_dataset_reasons"] == {
        ATAC_DATASET: "Known non-RNA assays: EFO:0010891",
        SECONDARY_DATASET: "All cells are secondary",
    }


@pytest.mark.parametrize(
    "duplicate",
    [{"cytebaseId": "renamed_lung"}, {"datasetId": SIBLING_DATASET}],
    ids=["same-dataset", "same-name"],
)
def test_prepare_registration_rejects_duplicate_existing_registrations(duplicate):
    with pytest.raises(ValueError, match="duplicate IDs or names"):
        _prepare([cellxgene_collection()], [_record(), _record(**duplicate)])


def test_prepare_registration_rejects_a_repeated_collection():
    repeated = cellxgene_collection(collection_id=COLLECTION_ID.replace("-", ""))
    with pytest.raises(ValueError, match=f"supplied more than once: {COLLECTION_ID}"):
        _prepare([cellxgene_collection(), repeated])


@pytest.mark.parametrize("datasets", [None, {DATASET_ID: {}}, "datasets"])
def test_prepare_registration_requires_a_dataset_list(datasets):
    collection = cellxgene_collection() | {"datasets": datasets}
    with pytest.raises(ValueError, match=f"Collection {COLLECTION_ID} has no dataset"):
        _prepare([collection])


@pytest.mark.parametrize(
    ("dataset", "problem"),
    [
        (
            "not a dataset",
            "dataset unknown needs review: Dataset metadata must be an object",
        ),
        (
            {k: v for k, v in cellxgene_dataset().items() if k != "dataset_id"},
            "dataset unknown needs review: Missing or invalid stable dataset_id",
        ),
        (
            cellxgene_dataset(assay=UNREVIEWED_ASSAY),
            f"dataset {DATASET_ID} needs review: Unreviewed assay IDs: EFO:9999999; "
            "review RNA content",
        ),
    ],
    ids=["not-an-object", "missing-id", "unreviewed-assay"],
)
def test_prepare_registration_rejects_collections_needing_review(dataset, problem):
    collection = cellxgene_collection(
        [cellxgene_dataset(dataset_id=SIBLING_DATASET), dataset]
    )
    message = (
        f"Collection {COLLECTION_ID}, {problem}. "
        "Resolve its metadata before registering this collection."
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        _prepare([collection])


def test_prepare_registration_rejects_repeated_datasets():
    collection = cellxgene_collection(
        [cellxgene_dataset(), cellxgene_dataset(dataset_version_id=NEW_VERSION_ID)]
    )
    with pytest.raises(ValueError, match=f"Collection repeats dataset {DATASET_ID}"):
        _prepare([collection])


@pytest.mark.parametrize(
    ("collections", "existing"),
    [
        ([cellxgene_collection()], [{"collectionId": OTHER_COLLECTION}]),
        (
            [
                cellxgene_collection(),
                cellxgene_collection(collection_id=OTHER_COLLECTION),
            ],
            [],
        ),
    ],
    ids=["registered-elsewhere", "supplied-twice"],
)
def test_prepare_registration_rejects_a_dataset_in_two_collections(
    collections, existing
):
    with pytest.raises(
        ValueError, match=f"{DATASET_ID} occurs in multiple collections"
    ):
        _prepare(collections, [_record(**overrides) for overrides in existing])


def test_prepare_registration_refuses_to_skip_a_registered_dataset_now_secondary():
    collection = cellxgene_collection([_secondary()])
    with pytest.raises(ValueError, match=f"{DATASET_ID} is now all-secondary"):
        _prepare([collection], [_record()])


# Catalog rows and database


def test_dataset_row_maps_record_fields_to_catalog_columns():
    record = _record(
        status="ready",
        processedVersionId=VERSION_ID,
        zarrUri="/published/lung/data.zarr",
        h5adUri="/published/lung/source.h5ad",
        processedAt="2026-02-03T04:05:06+00:00",
        facets={
            "tissue": [
                {"termId": "UBERON:0002048", "label": "lung"},
                {"termId": "UBERON:0008952", "label": "lung"},
                {"termId": None, "label": "airway"},
            ],
            "sex": [{"termId": "PATO:0000383", "label": "female"}],
            "suspension_type": [{"termId": None, "label": "nucleus"}],
        },
    )
    row = catalog._dataset_row(record)
    assert list(row) == [column for column, _ in catalog._SCHEMAS["datasets"]]
    assert row == {
        "cytebase_id": CYTEBASE_ID,
        "dataset_id": DATASET_ID,
        "collection_id": COLLECTION_ID,
        "latest_version_id": VERSION_ID,
        "processed_version_id": VERSION_ID,
        "title": "Healthy lung scRNA-seq atlas",
        "citation": CITATION,
        "doi": "10.1000/lung",
        "first_author": "Smith",
        "year": 2024,
        "organism_labels": [],
        "organism_ids": [],
        "assay_labels": [],
        "assay_ids": [],
        "tissue_labels": ["lung", "airway"],
        "tissue_ids": ["UBERON:0002048", "UBERON:0008952"],
        "organ_labels": [],
        "organ_ids": [],
        "disease_labels": [],
        "disease_ids": [],
        "cell_type_labels": [],
        "cell_type_ids": [],
        "sex_labels": ["female"],
        "development_stage_labels": [],
        "suspension_types": ["nucleus"],
        "cell_count": 6,
        "primary_cell_count": 4,
        "n_genes": 5,
        "schema_version": "5.3.0",
        "status": "ready",
        "zarr_uri": "/published/lung/data.zarr",
        "h5ad_uri": "/published/lung/source.h5ad",
        "cellxgene_url": f"https://cellxgene.cziscience.com/collections/{COLLECTION_ID}",
        "explorer_url": "https://cellxgene.cziscience.com/e/lung.cxg/",
        "registered_at": "2026-01-02T03:04:05Z",
        "processed_at": "2026-02-03T04:05:06Z",
        "pipeline_version": PIPELINE_VERSION,
    }


def test_term_rows_rank_labels_in_natural_order():
    first = _record(
        cytebaseId="b_lung",
        facets={
            "suspension_type": [{"label": "cell"}],
            "tissue": [
                {"termId": "UBERON:2", "label": "colon 10"},
                {"termId": None, "label": "lung"},
            ],
        },
    )
    second = _record(
        cytebaseId="a_lung",
        facets={
            "tissue": [
                {"termId": "UBERON:9", "label": "lung"},
                {"termId": None, "label": "lung"},
                {"termId": "UBERON:1", "label": "colon 2"},
                {"termId": "UBERON:1", "label": "colon 2"},
            ]
        },
    )

    def term(cytebase_id, facet, term_id, label, rank):
        return {
            "cytebase_id": cytebase_id,
            "facet": facet,
            "term_id": term_id,
            "label": label,
            "label_rank": rank,
        }

    assert catalog._term_rows([first, second]) == [
        term("a_lung", "tissue", "UBERON:1", "colon 2", 1),
        term("a_lung", "tissue", None, "lung", 3),
        term("a_lung", "tissue", "UBERON:9", "lung", 3),
        term("b_lung", "tissue", "UBERON:2", "colon 10", 2),
        term("b_lung", "tissue", None, "lung", 3),
        term("b_lung", "suspension_type", None, "cell", 1),
    ]


def test_write_catalog_creates_empty_tables_and_a_checksum_sidecar(tmp_path):
    paths = catalog._write_catalog({name: [] for name in catalog._SCHEMAS}, tmp_path)
    assert paths == {
        "cytebase.duckdb": tmp_path / "cytebase.duckdb",
        "cytebase.duckdb.sha256": tmp_path / "cytebase.duckdb.sha256",
    }
    digest = hashlib.sha256(paths["cytebase.duckdb"].read_bytes()).hexdigest()
    sidecar = paths["cytebase.duckdb.sha256"].read_text(encoding="ascii")
    assert sidecar == f"{digest}  cytebase.duckdb\n"
    with duckdb.connect(str(paths["cytebase.duckdb"]), read_only=True) as database:
        for name, schema in catalog._SCHEMAS.items():
            assert database.execute(f"SELECT count(*) FROM {name}").fetchone() == (0,)
            columns = database.execute(f"DESCRIBE {name}").fetchall()
            assert [column[0] for column in columns] == [c for c, _ in schema]


def test_write_catalog_fills_missing_list_and_map_columns(tmp_path):
    rows = {
        "datasets": [{"cytebase_id": CYTEBASE_ID}],
        "collections": [{"collection_id": COLLECTION_ID}],
        "dataset_terms": [],
    }
    paths = catalog._write_catalog(rows, tmp_path)
    with duckdb.connect(str(paths["cytebase.duckdb"]), read_only=True) as database:
        datasets = database.execute(
            "SELECT title, tissue_labels, suspension_types, registered_at FROM datasets"
        ).fetchall()
        collections = database.execute(
            "SELECT name, consortia, skipped_dataset_ids, skipped_dataset_reasons "
            "FROM collections"
        ).fetchall()
    assert datasets == [(None, [], [], None)]
    assert collections == [(None, [], [], {})]


def test_write_catalog_refuses_to_replace_a_database(tmp_path):
    (tmp_path / "cytebase.duckdb").write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="Catalog database already exists"):
        catalog._write_catalog({name: [] for name in catalog._SCHEMAS}, tmp_path)
    assert (tmp_path / "cytebase.duckdb").read_bytes() == b"existing"
    assert not (tmp_path / "cytebase.duckdb.sha256").exists()


def test_materialize_returns_rows_as_column_dictionaries():
    with duckdb.connect() as database:
        result = database.execute(
            "SELECT * FROM (VALUES (1, 'lung'), (2, NULL)) AS terms(rank, label) "
            "ORDER BY rank"
        )
        assert catalog._materialize(result) == [
            {"rank": 1, "label": "lung"},
            {"rank": 2, "label": None},
        ]
        assert catalog._materialize(database.execute("SELECT 1 AS n WHERE false")) == []


# Registered records


def test_load_record_reads_a_registered_dataset(fake_hub):
    _register(fake_hub, dataset_record())
    assert catalog.load_record(fake_hub.bucket(), CYTEBASE_ID) == _record()


def test_load_record_requires_a_registration(fake_hub):
    with pytest.raises(
        FileNotFoundError, match="Dataset missing_lung is not registered"
    ):
        catalog.load_record(fake_hub.bucket(), "missing_lung")


def test_load_record_rejects_a_record_saved_under_another_name(fake_hub):
    fake_hub.put("datasets/copied_lung/dataset.json", dataset_record())
    with pytest.raises(ValueError, match="identity does not match its directory"):
        catalog.load_record(fake_hub.bucket(), "copied_lung")


def test_load_record_rejects_invalid_ids(fake_hub):
    with pytest.raises(ValueError, match="lowercase letters"):
        catalog.load_record(fake_hub.bucket(), "../lung")
    assert fake_hub.calls == []


def test_list_records_reads_registered_directories_shallowly(registered):
    registered.put(f"datasets/{CYTEBASE_ID}/data.zarr/zarr.json", "{}")
    registered.put("datasets/README.md", "not a dataset")
    records = catalog.list_records(registered.bucket())
    assert [record.cytebaseId for record in records] == [
        BLOOD_ID,
        CYTEBASE_ID,
        SIBLING_ID,
    ]
    assert records[1] == _record()
    listings = [call for call in registered.calls if call[0] == "list_bucket_tree"]
    assert listings == [("list_bucket_tree", BUCKET_ID, "datasets/", False, False)]


def test_list_records_without_registrations_is_empty(fake_hub):
    assert catalog.list_records(fake_hub.bucket()) == []


def test_list_records_rejects_two_directories_for_one_dataset(registered):
    _register(registered, dataset_record(cytebaseId="lung_copy"))
    with pytest.raises(ValueError, match="use the same CELLxGENE dataset ID"):
        catalog.list_records(registered.bucket())


def test_select_dataset_ids_validates_and_deduplicates_named_datasets(fake_hub):
    request = ProcessRequest(cytebaseIds=["lung_b", "lung_a", "lung_b"])
    assert catalog.select_dataset_ids(request, fake_hub.bucket()) == [
        "lung_b",
        "lung_a",
    ]
    with pytest.raises(ValueError, match="lowercase letters"):
        catalog.select_dataset_ids(
            ProcessRequest(cytebaseIds=["lung_a", "Lung-B"]), fake_hub.bucket()
        )
    assert fake_hub.calls == []


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        ({"collectionId": COLLECTION_ID}, [CYTEBASE_ID, SIBLING_ID]),
        (
            {"collectionIds": [OTHER_COLLECTION, COLLECTION_ID]},
            [BLOOD_ID, CYTEBASE_ID, SIBLING_ID],
        ),
        ({"collectionIds": [BRAIN_COLLECTION]}, []),
    ],
)
def test_select_dataset_ids_by_collection(registered, selector, expected):
    request = ProcessRequest.model_validate(selector)
    assert catalog.select_dataset_ids(request, registered.bucket()) == expected


# Publication


def test_publish_catalog_first_run_rebuilds_from_registered_records(fake_hub):
    _register(
        fake_hub,
        dataset_record(),
        dataset_record(
            cytebaseId=BLOOD_ID,
            datasetId=BLOOD_DATASET,
            collectionId=OTHER_COLLECTION,
            title="Blood atlas",
            facets=_tissues("blood"),
        ),
    )
    bucket = fake_hub.bucket()
    lung = _collection_row(COLLECTION_ID, "Human lung atlas", LATER)
    update = _record(title="Renamed lung atlas", status="ready")
    result = catalog.publish_catalog(bucket, [update], [lung])

    digest = hashlib.sha256(fake_hub.read(DB_PATH)).hexdigest()
    assert result == {
        "status": "done",
        "datasets": 2,
        "collections": 1,
        "catalogUri": f"{bucket.root}/{DB_PATH}",
        "catalogSha256": digest,
    }
    assert fake_hub.read(HASH_PATH) == f"{digest}  cytebase.duckdb\n".encode()
    assert _uploads(fake_hub) == [[DB_PATH], [HASH_PATH]]
    tables = _tables(fake_hub)
    assert [
        (row["cytebase_id"], row["title"], row["status"]) for row in tables["datasets"]
    ] == [
        (BLOOD_ID, "Blood atlas", "registered"),
        (CYTEBASE_ID, "Renamed lung atlas", "ready"),
    ]
    assert tables["datasets"][1]["registered_at"] == REGISTERED
    assert tables["collections"] == [lung]
    assert _tissue_ranks(tables) == [(BLOOD_ID, "blood", 1), (CYTEBASE_ID, "lung", 2)]


def test_publish_catalog_merges_updates_into_the_previous_catalog(fake_hub):
    lung = dataset_record(facets=_tissues("colon 10"))
    blood = dataset_record(
        cytebaseId=BLOOD_ID,
        datasetId=BLOOD_DATASET,
        collectionId=OTHER_COLLECTION,
        title="Blood atlas",
        facets=_tissues("colon 9"),
    )
    publish_catalog_rows(
        fake_hub,
        [lung, blood],
        [
            _collection_row(COLLECTION_ID, "Human lung atlas", REGISTERED),
            _collection_row(OTHER_COLLECTION, "Human blood atlas", REGISTERED),
        ],
    )
    before = _tables(fake_hub)
    assert _tissue_ranks(before) == [
        (BLOOD_ID, "colon 9", 1),
        (CYTEBASE_ID, "colon 10", 2),
    ]
    fake_hub.calls.clear()

    renamed = _collection_row(COLLECTION_ID, "Lung atlas, revised", LATER)
    brain = _collection_row(BRAIN_COLLECTION, "Human brain atlas", LATER)
    update = _record(title="Lung atlas v2", facets=_tissues("colon 1"))
    result = catalog.publish_catalog(fake_hub.bucket(), [update], [renamed, brain])

    # A previous snapshot is merged without listing the bucket again.
    assert [call[0] for call in fake_hub.calls] == [
        "download_bucket_files",
        "batch_bucket_files",
        "batch_bucket_files",
    ]
    assert _uploads(fake_hub) == [[DB_PATH], [HASH_PATH]]
    assert (result["datasets"], result["collections"]) == (2, 3)
    assert result["catalogSha256"] == hashlib.sha256(fake_hub.read(DB_PATH)).hexdigest()
    after = _tables(fake_hub)
    assert after["datasets"][0] == before["datasets"][0]
    assert (after["datasets"][1]["title"], after["datasets"][1]["tissue_labels"]) == (
        "Lung atlas v2",
        ["colon 1"],
    )
    assert after["collections"] == [
        renamed | {"registered_at": REGISTERED},
        before["collections"][1],
        brain,
    ]
    assert _tissue_ranks(after) == [
        (BLOOD_ID, "colon 9", 2),
        (CYTEBASE_ID, "colon 1", 1),
    ]


def test_run_catalog_registers_collections_and_records_failures(fake_hub, monkeypatch):
    monkeypatch.setenv("CYTEBASE_PIPELINE_VERSION", PIPELINE_VERSION)
    lung_raw = json.dumps(cellxgene_collection(), indent=1).encode()
    review = cellxgene_collection(
        [cellxgene_dataset(dataset_id=BLOOD_DATASET, assay=UNREVIEWED_ASSAY)],
        collection_id=REVIEW_COLLECTION,
    )
    requested = _serve_collections(
        monkeypatch,
        {
            COLLECTION_ID: lung_raw,
            REVIEW_COLLECTION: json.dumps(review).encode(),
            LEAK_COLLECTION: RuntimeError("proxy echoed Bearer hf_LeakedToken123"),
        },
    )
    checks = []
    before = datetime.now(UTC)
    result = catalog.run_catalog(
        {
            "collectionIds": [
                LEAK_COLLECTION,
                REVIEW_COLLECTION,
                COLLECTION_ID.replace("-", ""),
                COLLECTION_ID,
            ]
        },
        fake_hub.bucket(),
        lambda: checks.append(len(_uploads(fake_hub))),
    )
    after = datetime.now(UTC)

    assert requested == [COLLECTION_ID, REVIEW_COLLECTION, LEAK_COLLECTION]
    assert result["failedCollections"] == [
        {
            "collectionId": REVIEW_COLLECTION,
            "error": (
                f"ValueError: Collection {REVIEW_COLLECTION}, dataset {BLOOD_DATASET} "
                "needs review: Unreviewed assay IDs: EFO:9999999; review RNA content. "
                "Resolve its metadata before registering this collection."
            ),
        },
        {
            "collectionId": LEAK_COLLECTION,
            "error": "RuntimeError: proxy echoed Bearer [redacted]",
        },
    ]
    assert result["registeredDatasets"] == [
        {"cytebaseId": CYTEBASE_ID, "status": "registered"}
    ]
    (row,) = result["registeredCollections"]
    assert row["collection_id"] == COLLECTION_ID
    assert before <= row["registered_at"] <= after
    # Ownership is confirmed before each write and before publication.
    assert checks == [0, 0, 1, 2, 2, 2]
    assert fake_hub.files() == sorted(
        [
            DB_PATH,
            HASH_PATH,
            f"datasets/{CYTEBASE_ID}/README.md",
            f"datasets/{CYTEBASE_ID}/cellxgene/collection.json",
            f"datasets/{CYTEBASE_ID}/cellxgene/dataset.json",
            f"datasets/{CYTEBASE_ID}/dataset.json",
        ]
    )
    assert (
        fake_hub.read(f"datasets/{CYTEBASE_ID}/cellxgene/collection.json") == lung_raw
    )
    saved = fake_hub.read_json(f"datasets/{CYTEBASE_ID}/dataset.json")
    assert saved["pipelineVersion"] == PIPELINE_VERSION
    assert result["status"] == "done"
    assert (result["datasets"], result["collections"]) == (1, 1)
    tables = _tables(fake_hub)
    assert [row["cytebase_id"] for row in tables["datasets"]] == [CYTEBASE_ID]
    assert [row["collection_id"] for row in tables["collections"]] == [COLLECTION_ID]


def test_run_catalog_reserves_names_across_collections(fake_hub, monkeypatch):
    monkeypatch.setenv("CYTEBASE_PIPELINE_VERSION", PIPELINE_VERSION)
    collections = {
        COLLECTION_ID: cellxgene_collection(),
        OTHER_COLLECTION: cellxgene_collection(
            [cellxgene_dataset(dataset_id=SIBLING_DATASET)],
            collection_id=OTHER_COLLECTION,
        ),
        BRAIN_COLLECTION: cellxgene_collection(
            [_secondary(dataset_id=SECONDARY_DATASET)],
            collection_id=BRAIN_COLLECTION,
        ),
    }
    _serve_collections(
        monkeypatch,
        {key: json.dumps(value).encode() for key, value in collections.items()},
    )
    result = catalog.run_catalog(
        {"collectionIds": list(reversed(collections))}, fake_hub.bucket(), noop
    )
    assert result["failedCollections"] == []
    assert result["registeredDatasets"] == [
        {"cytebaseId": CYTEBASE_ID, "status": "registered"},
        {"cytebaseId": SIBLING_ID, "status": "registered"},
    ]
    assert [
        (row["collection_id"], row["n_datasets_main"], row["skipped_dataset_ids"])
        for row in result["registeredCollections"]
    ] == [
        (COLLECTION_ID, 1, []),
        (OTHER_COLLECTION, 1, []),
        (BRAIN_COLLECTION, 0, [SECONDARY_DATASET]),
    ]
    assert (result["datasets"], result["collections"]) == (2, 3)
    directories = {
        path.split("/")[1] for path in fake_hub.files() if path.startswith("datasets/")
    }
    assert directories == {CYTEBASE_ID, SIBLING_ID}


def test_run_catalog_updates_existing_registrations(fake_hub, monkeypatch):
    monkeypatch.setenv("CYTEBASE_PIPELINE_VERSION", PIPELINE_VERSION)
    _register(
        fake_hub,
        dataset_record(
            cytebaseId="legacy_lung_name",
            status="ready",
            processedVersionId=VERSION_ID,
            zarrUri="/published/lung/data.zarr",
        ),
    )
    collection = cellxgene_collection(
        [cellxgene_dataset(dataset_version_id=NEW_VERSION_ID)]
    )
    _serve_collections(monkeypatch, {COLLECTION_ID: json.dumps(collection).encode()})
    result = catalog.run_catalog(
        {"collectionIds": [COLLECTION_ID]}, fake_hub.bucket(), noop
    )
    assert result["registeredDatasets"] == [
        {"cytebaseId": "legacy_lung_name", "status": "update_available"}
    ]
    saved = fake_hub.read_json("datasets/legacy_lung_name/dataset.json")
    assert saved["latestVersionId"] == NEW_VERSION_ID
    assert saved["zarrUri"] == "/published/lung/data.zarr"
    (row,) = _tables(fake_hub)["datasets"]
    assert row["cytebase_id"] == "legacy_lung_name"
    assert row["status"] == "update_available"


def test_run_catalog_requires_a_pipeline_version_to_register(fake_hub, monkeypatch):
    requested = _serve_collections(monkeypatch, {})
    with pytest.raises(KeyError, match="CYTEBASE_PIPELINE_VERSION"):
        catalog.run_catalog({"collectionIds": [COLLECTION_ID]}, fake_hub.bucket(), noop)
    assert requested == []
    assert fake_hub.files() == []


def test_run_catalog_stops_when_ownership_is_lost(fake_hub, monkeypatch):
    monkeypatch.setenv("CYTEBASE_PIPELINE_VERSION", PIPELINE_VERSION)
    _serve_collections(
        monkeypatch, {COLLECTION_ID: json.dumps(cellxgene_collection()).encode()}
    )
    checks = []

    def assert_owner():
        checks.append(len(checks))
        if len(checks) == 2:
            raise RuntimeError("Another catalog worker owns this run")

    with pytest.raises(RuntimeError, match="Another catalog worker owns this run"):
        catalog.run_catalog(
            {"collectionIds": [COLLECTION_ID]}, fake_hub.bucket(), assert_owner
        )
    assert checks == [0, 1]
    assert fake_hub.files() == []


def test_run_catalog_publishes_supplied_updates(fake_hub):
    publish_catalog_rows(fake_hub, [dataset_record()])
    checks = []
    update = dataset_record(
        status="ready",
        processedVersionId=VERSION_ID,
        zarrUri="/published/lung/data.zarr",
    )
    result = catalog.run_catalog(
        {"updates": [update]},
        fake_hub.bucket(),
        lambda: checks.append(len(_uploads(fake_hub))),
    )
    assert checks == [0]
    assert set(result) == {
        "status",
        "datasets",
        "collections",
        "catalogUri",
        "catalogSha256",
    }
    assert fake_hub.files() == [DB_PATH, HASH_PATH]
    (row,) = _tables(fake_hub)["datasets"]
    assert (row["status"], row["zarr_uri"]) == ("ready", "/published/lung/data.zarr")


def test_run_catalog_republishes_every_registered_record(registered):
    checks = []
    result = catalog.run_catalog({}, registered.bucket(), lambda: checks.append(1))
    assert checks == [1]
    assert (result["datasets"], result["collections"]) == (3, 0)
    assert "registeredDatasets" not in result
    assert [row["cytebase_id"] for row in _tables(registered)["datasets"]] == [
        BLOOD_ID,
        CYTEBASE_ID,
        SIBLING_ID,
    ]
