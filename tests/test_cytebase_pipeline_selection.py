"""Offline tests for Cytebase primary-RNA selection and the pipeline data models."""

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from pydantic import ValidationError

from scarf.cytebase.pipeline.build import inspect_file
from scarf.cytebase.pipeline.models import (
    DatasetRecord,
    DatasetVersion,
    FacetTerm,
    Manifest,
    ProcessRequest,
    RegisterRequest,
)
from scarf.cytebase.pipeline.selection import (
    _CENSUS_RNA,
    _NON_RNA,
    _REVIEWED_RNA,
    SELECTION_SOURCES,
    _asset_issue,
    classify_dataset,
    is_main_dataset,
)
from tests.fixtures_cytebase import (
    BUCKET_ID,
    COLLECTION_ID,
    CYTEBASE_ID,
    DATASET_ID,
    SOURCE_URL,
    VERSION_ID,
    cellxgene_dataset,
    dataset_record,
    full_manifest,
    write_h5ad,
)

pytestmark = pytest.mark.usefixtures("cytebase_offline")

CENSUS_RNA = "EFO:0009922"
REVIEWED_RNA = "EFO:0008913"
ATAC = "EFO:0010891"
TENX_ATAC = "EFO:0030007"
UNKNOWN = "EFO:9999999"
SELECTED = "Contains primary cells and reviewed RNA assays"
BAD_PRIMARY = "Invalid primary_cell_count"
BAD_TOTAL = "Invalid cell_count"
BAD_FLAGS = "Invalid is_primary_data list"
NO_PRIMARY = "Primary-data metadata is missing"
CONTRADICTS = "Primary-data metadata contradicts itself"
BAD_VERSION = "Missing or invalid dataset_version_id; review the CELLxGENE record"
BAD_ASSETS = "Missing or invalid assets list; review the H5AD download asset"
NOT_ONE_H5AD = "Expected exactly one H5AD download asset; review the CELLxGENE record"
BAD_URL = "Missing or invalid H5AD HTTP URL; review the download asset"
UNPARSABLE_URL = "Invalid H5AD HTTP URL; review the download asset"
BAD_SIZE = "Invalid H5AD filesize; expected a nonnegative integer or no value"
BAD_ID = "Missing or invalid stable dataset_id"
NO_ASSAYS = "Missing assay metadata; review RNA content"
BAD_TERMS = "Missing or invalid assay ontology term IDs"
NOW_UTC = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
PROVENANCE = {
    "collectionId",
    "datasetId",
    "datasetVersionId",
    "sourceUrl",
    "sourceBytes",
    "sourceSha256",
    "metadataSource",
    "ingestedAt",
    "pipelineVersion",
}
MISSING = object()


def _assay(term_id: Any, label: str = "assay") -> dict[str, Any]:
    return {"label": label, "ontology_term_id": term_id}


def _dataset(**overrides: Any) -> dict[str, Any]:
    """A curation API dataset with fields replaced, or removed when ``MISSING``."""
    dataset = cellxgene_dataset()
    for key, value in overrides.items():
        if value is MISSING:
            del dataset[key]
        else:
            dataset[key] = value
    return dataset


def _primary_metadata(count: Any, cells: Any, flags: Any) -> dict[str, Any]:
    """Primary-data fields for one dataset; ``None`` leaves a field out."""
    fields = {
        "primary_cell_count": count,
        "cell_count": cells,
        "is_primary_data": flags,
    }
    return {"dataset_id": DATASET_ID} | {
        key: value for key, value in fields.items() if value is not None
    }


def _decision(selection: str, reason: str, primary: bool | None) -> dict[str, Any]:
    return {"selection": selection, "reason": reason, "primary": primary}


def _errors(model: Any, payload: Any) -> list[tuple[tuple, str]]:
    with pytest.raises(ValidationError) as caught:
        model.model_validate(payload)
    return [(error["loc"], error["type"]) for error in caught.value.errors()]


@pytest.fixture(scope="module")
def h5ad_source(tmp_path_factory) -> Path:
    return write_h5ad(tmp_path_factory.mktemp("selection") / "source.h5ad")


@pytest.mark.parametrize(
    ("count", "cells", "flags", "expected"),
    [
        pytest.param(4, 6, [True, False], True, id="mixed"),
        pytest.param(6, 6, [True], True, id="all-primary"),
        pytest.param(0, 6, [False], False, id="all-secondary"),
        pytest.param(3, None, None, True, id="count-only"),
        pytest.param(0, None, None, False, id="zero-count-only"),
        pytest.param(0, 0, None, False, id="empty-dataset"),
        pytest.param(None, None, [True], True, id="flags-only-primary"),
        pytest.param(None, 6, [False], False, id="flags-only-secondary"),
        pytest.param(None, 6, [True, False], True, id="flags-only-mixed"),
        pytest.param(4, None, [True], True, id="all-primary-without-total"),
        pytest.param(6, None, [True, False], True, id="mixed-without-total"),
        pytest.param(2, 6, [], True, id="empty-flags-defer-to-count"),
    ],
)
def test_is_main_dataset_keeps_datasets_with_any_primary_cells(
    count, cells, flags, expected
):
    assert is_main_dataset(_primary_metadata(count, cells, flags)) is expected


@pytest.mark.parametrize(
    ("count", "cells", "flags", "message"),
    [
        pytest.param(-1, 6, None, BAD_PRIMARY, id="negative-count"),
        pytest.param(4.0, 6, None, BAD_PRIMARY, id="float-count"),
        pytest.param(True, 6, None, BAD_PRIMARY, id="bool-count"),
        pytest.param("4", 6, None, BAD_PRIMARY, id="text-count"),
        pytest.param(4, -6, None, BAD_TOTAL, id="negative-total"),
        pytest.param(4, 6.0, None, BAD_TOTAL, id="float-total"),
        pytest.param(4, True, None, BAD_TOTAL, id="bool-total"),
        pytest.param(4, 6, "True", BAD_FLAGS, id="text-flags"),
        pytest.param(4, 6, (True, False), BAD_FLAGS, id="tuple-flags"),
        pytest.param(4, 6, [1, 0], BAD_FLAGS, id="int-flags"),
        pytest.param(4, 6, [True, None], BAD_FLAGS, id="null-flag"),
        pytest.param(None, None, None, NO_PRIMARY, id="none"),
        pytest.param(None, 6, None, NO_PRIMARY, id="total-only"),
        pytest.param(None, 6, [], NO_PRIMARY, id="empty-flags"),
        pytest.param(7, 6, None, CONTRADICTS, id="count-above-total"),
        pytest.param(0, 6, [True, False], CONTRADICTS, id="flags-primary-count-zero"),
        pytest.param(3, 6, [False], CONTRADICTS, id="flags-secondary-count-positive"),
        pytest.param(4, 6, [True], CONTRADICTS, id="all-primary-count-below-total"),
        pytest.param(6, 6, [True, False], CONTRADICTS, id="mixed-count-equals-total"),
    ],
)
def test_is_main_dataset_rejects_invalid_primary_metadata(count, cells, flags, message):
    with pytest.raises(ValueError, match=f"^{message} for dataset {DATASET_ID}$"):
        is_main_dataset(_primary_metadata(count, cells, flags))


def test_is_main_dataset_names_an_unidentified_dataset():
    with pytest.raises(ValueError, match="missing for dataset unknown$"):
        is_main_dataset({"cell_count": 6})


@pytest.mark.parametrize(
    "assets",
    [
        pytest.param(
            [{"filetype": "H5AD", "url": SOURCE_URL, "filesize": 1234}], id="https"
        ),
        pytest.param(
            [{"filetype": "h5ad", "url": "http://datasets.example.org/a.h5ad"}],
            id="lowercase-type-http-unknown-size",
        ),
        pytest.param(
            [{"filetype": "H5AD", "url": SOURCE_URL, "filesize": 0}], id="empty-file"
        ),
        pytest.param(
            [{"filetype": "H5AD", "url": SOURCE_URL, "filesize": None}],
            id="null-size",
        ),
        pytest.param(
            [
                {
                    "filetype": "H5AD",
                    "url": "https://datasets.example.org:8443/a.h5ad?download=1",
                }
            ],
            id="explicit-port",
        ),
        pytest.param(
            [{"filetype": "H5AD", "url": "https://[2001:db8::1]/a.h5ad"}],
            id="ipv6-host",
        ),
        pytest.param(
            [
                {"filetype": "RDS", "url": "not a url"},
                {"filetype": "H5AD", "url": SOURCE_URL},
            ],
            id="ignores-other-assets",
        ),
    ],
)
def test_asset_issue_accepts_one_downloadable_h5ad(assets):
    assert _asset_issue(_dataset(assets=assets)) is None


@pytest.mark.parametrize(
    ("overrides", "issue"),
    [
        pytest.param({"dataset_version_id": MISSING}, BAD_VERSION, id="no-version"),
        pytest.param({"dataset_version_id": None}, BAD_VERSION, id="null-version"),
        pytest.param({"dataset_version_id": 3}, BAD_VERSION, id="int-version"),
        pytest.param({"dataset_version_id": "v1"}, BAD_VERSION, id="text-version"),
        pytest.param({"assets": MISSING}, BAD_ASSETS, id="no-assets"),
        pytest.param(
            {"assets": {"filetype": "H5AD", "url": SOURCE_URL}},
            BAD_ASSETS,
            id="asset-object",
        ),
        pytest.param({"assets": ["H5AD"]}, BAD_ASSETS, id="asset-text"),
        pytest.param({"assets": []}, NOT_ONE_H5AD, id="empty-assets"),
        pytest.param(
            {"assets": [{"filetype": "RDS", "url": SOURCE_URL}]},
            NOT_ONE_H5AD,
            id="rds-only",
        ),
        pytest.param({"assets": [{"url": SOURCE_URL}]}, NOT_ONE_H5AD, id="no-type"),
        pytest.param(
            {
                "assets": [
                    {"filetype": "H5AD", "url": SOURCE_URL},
                    {"filetype": "h5ad", "url": SOURCE_URL},
                ]
            },
            NOT_ONE_H5AD,
            id="two-h5ad",
        ),
    ],
)
def test_asset_issue_requires_a_version_and_one_h5ad_asset(overrides, issue):
    assert _asset_issue(_dataset(**overrides)) == issue


@pytest.mark.parametrize(
    ("url", "issue"),
    [
        pytest.param(MISSING, BAD_URL, id="no-url"),
        pytest.param(None, BAD_URL, id="null-url"),
        pytest.param(123, BAD_URL, id="int-url"),
        pytest.param("", BAD_URL, id="empty-url"),
        pytest.param("ftp://datasets.example.org/a.h5ad", BAD_URL, id="ftp"),
        pytest.param("//datasets.example.org/a.h5ad", BAD_URL, id="no-scheme"),
        pytest.param("https:///a.h5ad", BAD_URL, id="no-host"),
        pytest.param(
            "https://reader@datasets.example.org/a.h5ad", BAD_URL, id="username"
        ),
        pytest.param(
            "https://reader:secret@datasets.example.org/a.h5ad",
            BAD_URL,
            id="credentials",
        ),
        pytest.param(
            "https://datasets.example.org/a b.h5ad", BAD_URL, id="inner-space"
        ),
        # urlsplit silently strips these characters, so the raw URL is checked.
        pytest.param(
            " https://datasets.example.org/a.h5ad", BAD_URL, id="leading-space"
        ),
        pytest.param("https://datasets.example.org/a\tb.h5ad", BAD_URL, id="tab"),
        pytest.param(
            "https://datasets.example.org:port/a.h5ad", UNPARSABLE_URL, id="text-port"
        ),
        pytest.param(
            "https://datasets.example.org:70000/a.h5ad",
            UNPARSABLE_URL,
            id="port-out-of-range",
        ),
        pytest.param("https://[::1/a.h5ad", UNPARSABLE_URL, id="broken-ipv6"),
    ],
)
def test_asset_issue_requires_a_plain_http_url(url, issue):
    asset = {"filetype": "H5AD"} | ({} if url is MISSING else {"url": url})
    assert _asset_issue(_dataset(assets=[asset])) == issue


@pytest.mark.parametrize("size", [-1, 1.5, "1234", True])
def test_asset_issue_rejects_invalid_filesize(size):
    asset = {"filetype": "H5AD", "url": SOURCE_URL, "filesize": size}
    assert _asset_issue(_dataset(assets=[asset])) == BAD_SIZE


@pytest.mark.parametrize(
    "assays",
    [
        pytest.param([_assay(CENSUS_RNA)], id="census-rna"),
        pytest.param([_assay(REVIEWED_RNA)], id="reviewed-rna"),
        pytest.param([_assay("EFO:0030059")], id="multiome-includes-rna"),
        pytest.param(
            [_assay(CENSUS_RNA), _assay(REVIEWED_RNA), _assay(CENSUS_RNA)],
            id="several-rna-with-duplicate",
        ),
    ],
)
def test_classify_dataset_selects_primary_rna(assays):
    assert classify_dataset(_dataset(assay=assays)) == _decision(
        "selected", SELECTED, True
    )


def test_classify_dataset_trusts_term_ids_over_labels():
    rna_label = _dataset(assay=[_assay(UNKNOWN, "10x 3' v3")])
    atac_label = _dataset(assay=[_assay(CENSUS_RNA, "scATAC-seq")])
    assert classify_dataset(rna_label)["selection"] == "needsReview"
    assert classify_dataset(atac_label)["selection"] == "selected"


@pytest.mark.parametrize(
    ("overrides", "reason", "primary"),
    [
        pytest.param(
            {"primary_cell_count": 0, "is_primary_data": [False]},
            "All cells are secondary",
            False,
            id="secondary",
        ),
        pytest.param(
            {
                "primary_cell_count": 0,
                "is_primary_data": [False],
                "assay": MISSING,
                "assets": MISSING,
            },
            "All cells are secondary",
            False,
            id="secondary-before-assay-review",
        ),
        pytest.param(
            {"assay": [_assay(TENX_ATAC), _assay(ATAC)]},
            f"Known non-RNA assays: {ATAC}, {TENX_ATAC}",
            True,
            id="non-rna",
        ),
        pytest.param(
            {"assay": [_assay(ATAC)], "assets": []},
            f"Known non-RNA assays: {ATAC}",
            True,
            id="non-rna-before-asset-review",
        ),
    ],
)
def test_classify_dataset_skips_secondary_and_non_rna_datasets(
    overrides, reason, primary
):
    assert classify_dataset(_dataset(**overrides)) == _decision(
        "skipped", reason, primary
    )


@pytest.mark.parametrize(
    ("overrides", "reason", "primary"),
    [
        pytest.param(
            {"primary_cell_count": MISSING, "is_primary_data": MISSING},
            f"{NO_PRIMARY} for dataset {DATASET_ID}",
            None,
            id="no-primary-metadata",
        ),
        pytest.param(
            {"primary_cell_count": 7},
            f"{CONTRADICTS} for dataset {DATASET_ID}",
            None,
            id="contradictory-primary-metadata",
        ),
        pytest.param(
            {"cell_count": "6"},
            f"{BAD_TOTAL} for dataset {DATASET_ID}",
            None,
            id="invalid-cell-count",
        ),
        pytest.param({"dataset_id": MISSING}, BAD_ID, True, id="no-dataset-id"),
        pytest.param({"dataset_id": None}, BAD_ID, True, id="null-dataset-id"),
        pytest.param({"dataset_id": 7}, BAD_ID, True, id="int-dataset-id"),
        pytest.param({"dataset_id": "lung-atlas"}, BAD_ID, True, id="text-dataset-id"),
        pytest.param(
            {
                "dataset_id": "lung-atlas",
                "primary_cell_count": 0,
                "is_primary_data": [False],
            },
            BAD_ID,
            False,
            id="dataset-id-before-secondary-skip",
        ),
        pytest.param({"assay": MISSING}, NO_ASSAYS, True, id="no-assays"),
        pytest.param({"assay": []}, NO_ASSAYS, True, id="empty-assays"),
        pytest.param({"assay": _assay(CENSUS_RNA)}, NO_ASSAYS, True, id="assay-object"),
        pytest.param({"assay": [CENSUS_RNA]}, BAD_TERMS, True, id="assay-text"),
        pytest.param(
            {"assay": [{"label": "10x 3' v3"}]}, BAD_TERMS, True, id="label-only"
        ),
        pytest.param({"assay": [_assay(None)]}, BAD_TERMS, True, id="null-term"),
        pytest.param({"assay": [_assay(9922)]}, BAD_TERMS, True, id="int-term"),
        pytest.param({"assay": [_assay("")]}, BAD_TERMS, True, id="empty-term"),
        pytest.param(
            {"assay": [_assay(CENSUS_RNA), _assay(None)]},
            BAD_TERMS,
            True,
            id="one-bad-term",
        ),
        pytest.param(
            {"assay": [_assay(UNKNOWN)]},
            f"Unreviewed assay IDs: {UNKNOWN}; review RNA content",
            True,
            id="unknown-assay",
        ),
        pytest.param(
            {
                "assay": [
                    _assay(UNKNOWN),
                    _assay(CENSUS_RNA),
                    _assay(ATAC),
                    _assay("EFO:0000001"),
                ]
            },
            f"Unreviewed assay IDs: EFO:0000001, {UNKNOWN}; review RNA content",
            True,
            id="unknown-before-conflict",
        ),
        pytest.param(
            {"assay": [_assay(ATAC), _assay(CENSUS_RNA)]},
            f"Conflicting RNA and non-RNA assay IDs: {CENSUS_RNA}, {ATAC}; "
            "review which measurement the H5AD contains",
            True,
            id="rna-and-non-rna",
        ),
        pytest.param({"assets": []}, NOT_ONE_H5AD, True, id="rna-without-h5ad"),
        pytest.param(
            {"dataset_version_id": "v2"}, BAD_VERSION, True, id="rna-without-version"
        ),
        pytest.param(
            {
                "assets": [
                    {
                        "filetype": "H5AD",
                        "url": "https://reader:secret@datasets.example.org/a.h5ad",
                    }
                ]
            },
            BAD_URL,
            True,
            id="rna-with-credentialed-url",
        ),
    ],
)
def test_classify_dataset_requests_review_with_a_reason(overrides, reason, primary):
    assert classify_dataset(_dataset(**overrides)) == _decision(
        "needsReview", reason, primary
    )


@pytest.mark.parametrize("dataset", [None, [cellxgene_dataset()], "dataset"])
def test_classify_dataset_requires_an_object(dataset):
    assert classify_dataset(dataset) == _decision(
        "needsReview", "Dataset metadata must be an object", None
    )


def test_every_pinned_assay_id_is_classified_by_its_list():
    rna = _CENSUS_RNA | _REVIEWED_RNA.keys()
    non_rna = set(_NON_RNA)
    for term_id in rna | non_rna:
        assert re.fullmatch(r"EFO:\d{7}", term_id), term_id
    outcomes = {
        term_id: classify_dataset(_dataset(assay=[_assay(term_id)]))["selection"]
        for term_id in rna | non_rna
    }
    assert {key for key, value in outcomes.items() if value == "selected"} == rna
    assert {key for key, value in outcomes.items() if value == "skipped"} == non_rna


def test_census_selection_source_is_pinned_to_its_commit():
    census = SELECTION_SOURCES[0]
    assert re.fullmatch(r"[0-9a-f]{40}", census["commit"])
    assert f"/{census['commit']}/" in census["url"]
    assert re.fullmatch(r"[0-9a-f]{64}", census["sha256"])


def test_register_request_parses_collection_ids():
    request = RegisterRequest.model_validate({"collectionIds": [COLLECTION_ID.upper()]})
    assert request.collectionIds == [UUID(COLLECTION_ID)]


@pytest.mark.parametrize(
    ("payload", "errors"),
    [
        pytest.param({}, [(("collectionIds",), "missing")], id="missing"),
        pytest.param(
            {"collectionIds": []}, [(("collectionIds",), "too_short")], id="empty"
        ),
        pytest.param(
            {"collectionIds": ["lung"]},
            [(("collectionIds", 0), "uuid_parsing")],
            id="not-a-uuid",
        ),
        pytest.param(
            {"collectionIds": [COLLECTION_ID], "force": True},
            [(("force",), "extra_forbidden")],
            id="extra-field",
        ),
    ],
)
def test_register_request_rejects_invalid_payloads(payload, errors):
    assert _errors(RegisterRequest, payload) == errors


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"cytebaseIds": [CYTEBASE_ID]}, id="datasets"),
        pytest.param({"collectionId": COLLECTION_ID}, id="collection"),
        pytest.param(
            {
                "collectionIds": [COLLECTION_ID],
                "force": True,
                "approvedDeletionPaths": [f"datasets/{CYTEBASE_ID}/data.zarr"],
            },
            id="collections-with-options",
        ),
        pytest.param(
            {"cytebaseIds": None, "collectionId": COLLECTION_ID, "collectionIds": None},
            id="explicit-nulls-are-absent",
        ),
    ],
)
def test_process_request_accepts_exactly_one_selector(payload):
    request = ProcessRequest.model_validate(payload)
    assert request.model_dump(mode="json") == {
        "cytebaseIds": None,
        "collectionId": None,
        "collectionIds": None,
        "force": False,
        "approvedDeletionPaths": [],
    } | {key: value for key, value in payload.items() if value is not None}


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({}, id="none"),
        pytest.param({"force": True}, id="options-only"),
        pytest.param(
            {"cytebaseIds": [CYTEBASE_ID], "collectionId": COLLECTION_ID}, id="two"
        ),
        pytest.param(
            {"collectionId": COLLECTION_ID, "collectionIds": [COLLECTION_ID]},
            id="collection-and-collections",
        ),
        pytest.param(
            {
                "cytebaseIds": [CYTEBASE_ID],
                "collectionId": COLLECTION_ID,
                "collectionIds": [COLLECTION_ID],
            },
            id="all",
        ),
    ],
)
def test_process_request_rejects_other_selector_counts(payload):
    with pytest.raises(
        ValidationError,
        match="Supply exactly one of cytebaseIds, collectionId, or collectionIds",
    ):
        ProcessRequest.model_validate(payload)


@pytest.mark.parametrize(
    ("payload", "errors"),
    [
        pytest.param(
            {"cytebaseIds": []}, [(("cytebaseIds",), "too_short")], id="no-datasets"
        ),
        pytest.param(
            {"collectionIds": []},
            [(("collectionIds",), "too_short")],
            id="no-collections",
        ),
        pytest.param(
            {"collectionId": "lung"},
            [(("collectionId",), "uuid_parsing")],
            id="collection-not-a-uuid",
        ),
        pytest.param(
            {"collectionIds": ["lung"]},
            [(("collectionIds", 0), "uuid_parsing")],
            id="collections-not-uuids",
        ),
        pytest.param(
            {"cytebaseIds": [CYTEBASE_ID], "bucket": BUCKET_ID},
            [(("bucket",), "extra_forbidden")],
            id="extra-field",
        ),
        pytest.param(
            {"cytebaseIds": [CYTEBASE_ID], "force": "maybe"},
            [(("force",), "bool_parsing")],
            id="force-not-bool",
        ),
    ],
)
def test_process_request_rejects_invalid_fields(payload, errors):
    assert _errors(ProcessRequest, payload) == errors


def test_dataset_record_parses_registered_json():
    record = DatasetRecord.model_validate(dataset_record())
    assert record.cytebaseId == CYTEBASE_ID
    assert record.datasetId == UUID(DATASET_ID)
    assert record.versions == [
        DatasetVersion(datasetVersionId=UUID(VERSION_ID), seenAt=NOW_UTC)
    ]
    assert record.facets["suspension_type"] == [FacetTerm(label="cell")]
    assert record.registeredAt == NOW_UTC
    assert (record.status, record.attempt, record.inspection) == ("registered", 0, None)
    assert record.timings == {}
    assert DatasetRecord.model_validate(record.model_dump(mode="json")) == record


@pytest.mark.parametrize("cytebase_id", ["a", "_", "0", "a" * 80, CYTEBASE_ID])
def test_dataset_record_accepts_catalog_ids(cytebase_id):
    record = DatasetRecord.model_validate(dataset_record(cytebaseId=cytebase_id))
    assert record.cytebaseId == cytebase_id


@pytest.mark.parametrize(
    "cytebase_id",
    ["", "a" * 81, "Lung", "lung-2024", "lung 2024", "a/b", "../a", "lung\n", "é"],
)
def test_dataset_record_rejects_unsafe_ids(cytebase_id):
    assert _errors(DatasetRecord, dataset_record(cytebaseId=cytebase_id)) == [
        (("cytebaseId",), "string_pattern_mismatch")
    ]


@pytest.mark.parametrize(
    ("overrides", "errors"),
    [
        pytest.param(
            {"attempt": -1}, [(("attempt",), "greater_than_equal")], id="attempt"
        ),
        pytest.param({"status": "done"}, [(("status",), "literal_error")], id="status"),
        pytest.param(
            {"versions": [{"datasetVersionId": "v1", "seenAt": NOW_UTC}]},
            [(("versions", 0, "datasetVersionId"), "uuid_parsing")],
            id="version-id",
        ),
        pytest.param(
            {"facets": {"tissue": [{"termId": "UBERON:0002048"}]}},
            [(("facets", "tissue", 0, "label"), "missing")],
            id="facet-label",
        ),
        pytest.param(
            {"timings": {"download": "fast"}},
            [(("timings", "download"), "float_parsing")],
            id="timing",
        ),
    ],
)
def test_dataset_record_rejects_invalid_fields(overrides, errors):
    assert _errors(DatasetRecord, dataset_record(**overrides)) == errors


def test_dataset_record_validates_its_inspection(h5ad_source):
    manifest = full_manifest(h5ad_source)
    record = DatasetRecord.model_validate(
        dataset_record(status="processing", attempt=2, inspection=manifest)
    )
    assert isinstance(record.inspection, Manifest)
    assert (record.inspection.countsLocation, record.attempt) == ("X", 2)
    broken = manifest | {"countsLocation": "layers/normalized"}
    assert _errors(DatasetRecord, dataset_record(inspection=broken)) == [
        (("inspection", "countsLocation"), "literal_error")
    ]


def test_manifest_keeps_every_inspection_field(h5ad_source):
    payload = full_manifest(h5ad_source)
    manifest = Manifest.model_validate(payload)
    assert manifest.datasetVersionId == UUID(VERSION_ID)
    assert manifest.ingestedAt == NOW_UTC
    assert manifest.model_dump(mode="json") == payload | {
        "doi": None,
        "ingestedAt": "2026-01-02T03:04:05Z",
    }


def test_manifest_requires_source_provenance(h5ad_source):
    errors = _errors(Manifest, inspect_file(h5ad_source)["manifest"])
    assert {loc for loc, _ in errors} == {(field,) for field in PROVENANCE}
    assert {kind for _, kind in errors} == {"missing"}


def test_manifest_accepts_a_needs_input_inspection(h5ad_source):
    manifest = Manifest.model_validate(full_manifest(h5ad_source, "raw.X"))
    assert (manifest.countsLocation, manifest.featureAttrsKey) == ("none", None)
    assert manifest.countsValidationMode is None
    assert manifest.selectionNeedsInput["options"] == ["raw/X"]


@pytest.mark.parametrize(
    ("overrides", "errors"),
    [
        pytest.param(
            {"countsLocation": "layers/normalized"},
            [(("countsLocation",), "literal_error")],
            id="counts-location",
        ),
        pytest.param(
            {"countsSelectionSource": "label"},
            [(("countsSelectionSource",), "literal_error")],
            id="selection-source",
        ),
        pytest.param(
            {"countsValidationMode": "full_matrix"},
            [(("countsValidationMode",), "literal_error")],
            id="validation-mode",
        ),
        pytest.param(
            {"metadataSource": "inspection"},
            [(("metadataSource",), "literal_error")],
            id="metadata-source",
        ),
        pytest.param(
            {"datasetId": "lung"}, [(("datasetId",), "uuid_parsing")], id="dataset-id"
        ),
        pytest.param(
            {"ingestedAt": "yesterday"},
            [(("ingestedAt",), "datetime_from_date_parsing")],
            id="ingested-at",
        ),
        pytest.param(
            {"isPrimaryDataCounts": {"true": "many"}},
            [(("isPrimaryDataCounts", "true"), "int_parsing")],
            id="primary-counts",
        ),
    ],
)
def test_manifest_rejects_values_outside_its_contract(h5ad_source, overrides, errors):
    assert _errors(Manifest, full_manifest(h5ad_source) | overrides) == errors
