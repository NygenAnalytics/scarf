"""Offline tests for converting, verifying, and publishing Cytebase Scarf stores."""

import dataclasses
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import pytest
import zarr
from zarr.storage import LocalStore

import scarf
from scarf.cytebase.pipeline import build
from scarf.cytebase.pipeline.models import DatasetRecord, Manifest
from tests.fixtures_cytebase import (
    COLLECTION_ID,
    COUNTS,
    CYTEBASE_ID,
    DATASET_ID,
    NEW_VERSION_ID,
    NOW,
    PIPELINE_VERSION,
    SOURCE_URL,
    VERSION_ID,
    CytebaseBuild,
    FakeHub,
    dataset_record,
    full_manifest,
    noop,
    source_details,
    write_h5ad,
)

pytestmark = pytest.mark.usefixtures("cytebase_offline")

PREFIX = f"datasets/{CYTEBASE_ID}"
RECORD_PATH = f"{PREFIX}/dataset.json"
INGEST_PATH = f"{PREFIX}/scarf_ingest.json"
LIST = ("list", f"{PREFIX}/")
QUESTION = {"question": "Which matrix holds raw counts?", "options": ["X", "raw/X"]}


def _unexpected(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("This step must not run")


def _source(tmp_path: Path, **options: Any) -> tuple[Path, dict[str, Any]]:
    source = write_h5ad(tmp_path / "source.h5ad", **options)
    return source, full_manifest(source)


def _patch_inspection(monkeypatch, source: Path, **changes: Any) -> None:
    inspection = dataclasses.replace(
        scarf.inspect_h5ad(str(source), matrix_key="X"), **changes
    )
    monkeypatch.setattr(scarf, "inspect_h5ad", lambda path, *, matrix_key: inspection)


def _manifest_json(built: CytebaseBuild) -> dict[str, Any]:
    return built.manifest().model_dump(mode="json")


def _copy_store(built: CytebaseBuild, tmp_path: Path, name: str = "data.zarr") -> Path:
    return Path(shutil.copytree(built.store, tmp_path / name))


def _set_first_count(store: Path, path: str, value: int = 99) -> None:
    zarr.open_array(str(store / path), mode="r+")[0, 0] = value


def _store_files(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


def _track_stores(monkeypatch) -> list[LocalStore]:
    """Open local stores through objects the test can inspect after closing."""
    opened: list[LocalStore] = []

    def make_store(location, *, storage_options=None, read_only=False):
        opened.append(LocalStore(location, read_only=read_only))
        return opened[-1]

    monkeypatch.setattr(build, "make_store", make_store)
    return opened


@dataclass(frozen=True)
class DenseBuild:
    store: Path
    manifest: dict[str, Any]
    converted: dict[str, Any]
    stages: list[tuple[str, dict[str, Any]]]


@pytest.fixture(scope="module")
def dense_build(tmp_path_factory) -> DenseBuild:
    """Convert a dense float32 H5AD without embeddings once for this module."""
    root = tmp_path_factory.mktemp("cytebase-dense")
    source = write_h5ad(
        root / "source.h5ad", COUNTS.astype(np.float32), encoding="dense", umap=False
    )
    manifest = full_manifest(source)
    stages: list[tuple[str, dict[str, Any]]] = []
    converted = build.convert_local(
        source,
        root / "data.zarr",
        manifest,
        progress=lambda stage, **kwargs: stages.append((stage, kwargs)),
    )
    return DenseBuild(root / "data.zarr", manifest, converted, stages)


# convert_local


def test_convert_local_validates_the_manifest_before_reading_the_source(tmp_path):
    _, manifest = _source(tmp_path)
    del manifest["selectionNeedsInput"]
    with pytest.raises(ValueError, match="missing selectionNeedsInput"):
        build.convert_local(tmp_path / "absent.h5ad", tmp_path / "data.zarr", manifest)
    assert not (tmp_path / "data.zarr").exists()


def test_convert_local_never_writes_into_an_existing_destination(tmp_path, monkeypatch):
    source, manifest = _source(tmp_path)
    destination = tmp_path / "data.zarr"
    destination.mkdir()
    (destination / "keep.txt").write_text("previous output")
    monkeypatch.setattr(scarf, "inspect_h5ad", _unexpected)
    with pytest.raises(FileExistsError, match="Scarf destination already exists"):
        build.convert_local(source, destination, manifest)
    assert [path.name for path in destination.iterdir()] == ["keep.txt"]


def _append_byte(path: Path) -> None:
    with path.open("ab") as handle:
        handle.write(b"\0")


def _flip_last_byte(path: Path) -> None:
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    path.write_bytes(bytes(data))


@pytest.mark.parametrize(
    ("change", "message", "stages"),
    [
        (_append_byte, "Source byte size does not match", []),
        (_flip_last_byte, "Source SHA-256 does not match", ["verifying_source"]),
    ],
    ids=["size", "checksum"],
)
def test_convert_local_rejects_a_source_changed_after_inspection(
    tmp_path, monkeypatch, change, message, stages
):
    source, manifest = _source(tmp_path)
    change(source)
    monkeypatch.setattr(scarf, "inspect_h5ad", _unexpected)
    seen: list[str] = []
    with pytest.raises(ValueError, match=message):
        build.convert_local(
            source,
            tmp_path / "data.zarr",
            manifest,
            progress=lambda stage, **kwargs: seen.append(stage),
        )
    assert seen == stages
    assert not (tmp_path / "data.zarr").exists()


def test_convert_local_returns_the_inspection_question_without_converting(
    tmp_path, monkeypatch
):
    source = write_h5ad(tmp_path / "source.h5ad")
    manifest = full_manifest(source, raw_data_location="raw.X")
    monkeypatch.setattr(scarf, "inspect_h5ad", _unexpected)
    result = build.convert_local(str(source), str(tmp_path / "data.zarr"), manifest)
    assert result["status"] == "needsInput"
    assert result["needsInput"] == manifest["selectionNeedsInput"]
    assert "CELLxGENE specifies raw.X" in result["needsInput"]["question"]
    assert (result["sourceBytes"], result["sourceSha256"]) == source_details(source)
    assert result["zarrPath"] == str(tmp_path / "data.zarr")
    assert {
        key: result["conversion"][key]
        for key in ("matrixKey", "apiRawDataLocation", "countsSelectionSource")
    } == {
        "matrixKey": "none",
        "apiRawDataLocation": "raw.X",
        "countsSelectionSource": "curation_api",
    }
    assert not (tmp_path / "data.zarr").exists()


@pytest.mark.parametrize(
    "error", [ValueError("no matrix"), KeyError("var"), TypeError("bad shape")]
)
def test_convert_local_asks_for_input_when_scarf_cannot_inspect_the_counts(
    tmp_path, monkeypatch, error
):
    source, manifest = _source(tmp_path)
    requested = []

    def inspect(path, *, matrix_key):
        requested.append((path, matrix_key))
        raise error

    monkeypatch.setattr(scarf, "inspect_h5ad", inspect)
    result = build.convert_local(source, tmp_path / "data.zarr", manifest)
    assert requested == [(str(source), "X")]
    assert result["status"] == "needsInput"
    assert result["needsInput"] == {
        "question": f"Scarf cannot inspect selected counts: {error}. "
        "Confirm the matrix and feature metadata.",
        "options": ["X"],
    }
    assert not (tmp_path / "data.zarr").exists()


@pytest.mark.parametrize(
    ("inspection_changes", "manifest_changes", "question", "options"),
    [
        (
            {"matrixEncoding": "csc"},
            {},
            "Scarf materializes CSC inputs",
            ["CSR H5AD", "Dense H5AD"],
        ),
        (
            {"featureIdsKey": "gene_ids"},
            {},
            "disagree on feature IDs",
            ["gene_ids", "_index"],
        ),
        (
            {"featureAttrsKey": "raw/var"},
            {},
            "disagree on the selected feature table",
            ["raw/var", "var"],
        ),
        (
            {},
            {"featureNameKey": "gene_symbols"},
            "feature-name column is missing",
            ["_index", "feature_name"],
        ),
    ],
    ids=["csc", "feature-ids", "feature-table", "feature-names"],
)
def test_convert_local_asks_for_input_when_scarf_disagrees_with_the_manifest(
    tmp_path, monkeypatch, inspection_changes, manifest_changes, question, options
):
    source, manifest = _source(tmp_path)
    _patch_inspection(monkeypatch, source, **inspection_changes)
    result = build.convert_local(
        source, tmp_path / "data.zarr", manifest | manifest_changes
    )
    assert result["status"] == "needsInput"
    assert question in result["needsInput"]["question"]
    assert result["needsInput"]["options"] == options
    assert not (tmp_path / "data.zarr").exists()


@pytest.mark.parametrize(
    ("inspection_changes", "manifest_changes", "message"),
    [
        ({"nCells": 7}, {}, "Source dimensions do not match"),
        ({"nFeatures": 4}, {}, "Source dimensions do not match"),
        ({}, {"countsDtype": "float64"}, "Source dtype does not match"),
    ],
    ids=["cells", "features", "dtype"],
)
def test_convert_local_rejects_a_source_that_drifts_from_the_manifest(
    tmp_path, monkeypatch, inspection_changes, manifest_changes, message
):
    source, manifest = _source(tmp_path)
    _patch_inspection(monkeypatch, source, **inspection_changes)
    with pytest.raises(ValueError, match=message):
        build.convert_local(source, tmp_path / "data.zarr", manifest | manifest_changes)
    assert not (tmp_path / "data.zarr").exists()


class _Closable:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize(
    ("failing_step", "opened"),
    [("writer", set()), ("dump", {"writer"}), ("summary", {"writer", "datastore"})],
)
def test_convert_local_closes_open_handles_when_a_step_fails(
    tmp_path, monkeypatch, failing_step, opened
):
    source, manifest = _source(tmp_path)
    readers = []
    stores: dict[str, _Closable] = {}
    original = scarf.H5adReader.from_inspect

    def from_inspect(inspection, **overrides):
        readers.append(original(inspection, **overrides))
        return readers[-1]

    def step(name: str) -> None:
        if name == failing_step:
            raise RuntimeError(f"{name} failed")

    class Writer:
        def __init__(self, reader, **options):
            step("writer")
            self.z = SimpleNamespace(store=stores.setdefault("writer", _Closable()))

        def dump(self):
            step("dump")

    class Datastore:
        def __init__(self, location, **options):
            self.z = SimpleNamespace(store=stores.setdefault("datastore", _Closable()))

        def summary(self):
            step("summary")

    monkeypatch.setattr(scarf.H5adReader, "from_inspect", from_inspect)
    monkeypatch.setattr(scarf, "H5adToZarr", Writer)
    monkeypatch.setattr(scarf, "DataStore", Datastore)
    with pytest.raises(RuntimeError, match=f"{failing_step} failed"):
        build.convert_local(source, tmp_path / "data.zarr", manifest)
    assert len(readers) == 1
    assert not readers[0].h5
    assert set(stores) == opened
    assert all(store.closed for store in stores.values())


def test_convert_local_preserves_dense_counts_without_embeddings(dense_build):
    converted = dense_build.converted
    assert converted["status"] == "done"
    assert converted["zarrPath"] == str(dense_build.store)
    assert converted["conversion"]["matrixKey"] == "X"
    assert converted["conversion"]["embeddingRoles"] == {}
    assert converted["conversion"]["scarfSuggestedFeatureNameKey"] == "feature_name"
    assert converted["importedArtifacts"]["embeddings"] == {}
    assert converted["importedArtifacts"]["clusters"] == {}
    assert converted["importedArtifacts"]["cellSelection"]["kind"] == "cell_selection"
    assert converted["qcSummary"]["total_cells"] == 6
    assert converted["qcSummary"]["active_cells"] == 6
    assert [stage for stage, _ in dense_build.stages] == [
        "verifying_source",
        "converting",
        "initializing_qc",
    ]
    assert "preserving source dtype float32" in dense_build.stages[1][1]["message"]
    verification = build.verify_store(str(dense_build.store), dense_build.manifest)
    assert verification["countsDtype"] == "float32"
    assert verification["countsBlock"] == COUNTS[:3].tolist()


# build_local


def test_build_local_inspects_and_converts_a_registered_source(cytebase_build):
    manifest = cytebase_build.manifest()
    converted = cytebase_build.converted()
    assert manifest.countsLocation == "X"
    assert manifest.isPrimaryDataCounts == {"true": 4, "false": 2}
    assert (manifest.sourceBytes, manifest.sourceSha256) == (
        cytebase_build.size,
        cytebase_build.sha256,
    )
    assert manifest.organism == "Homo sapiens"
    assert manifest.metadataSource == "curation_api"
    assert converted["status"] == "done"
    assert converted["zarrPath"] == str(cytebase_build.store)
    assert {
        key: converted[key]
        for key in (
            "cytebaseId",
            "collectionId",
            "datasetId",
            "datasetVersionId",
            "pipelineVersion",
            "sourcePath",
        )
    } == {
        "cytebaseId": CYTEBASE_ID,
        "collectionId": COLLECTION_ID,
        "datasetId": DATASET_ID,
        "datasetVersionId": VERSION_ID,
        "pipelineVersion": PIPELINE_VERSION,
        "sourcePath": SOURCE_URL,
    }
    assert set(converted["inspection"]) == {"h5ad_keys", "obs_summary", "uns"}
    assert converted["conversion"]["embeddingRoles"] == {"X_umap": "umap"}
    umap = converted["importedArtifacts"]["embeddings"]["X_umap"]
    assert (umap["kind"], umap["assay"]) == ("embedding", "RNA")
    embedding = cytebase_build.store / "RNA" / "artifacts" / "embedding"
    assert (embedding / umap["artifact_id"]).is_dir()
    assert converted["qcSummary"]["active_cells"] == 6
    assert set(cytebase_build.record().timings) == {"inspectSeconds", "convertSeconds"}


def test_build_local_prefers_registration_metadata(tmp_path, monkeypatch):
    source = write_h5ad(tmp_path / "source.h5ad")
    size, checksum = source_details(source)
    organisms = [
        {"termId": "NCBITaxon:9606", "label": "Homo sapiens"},
        {"termId": "NCBITaxon:10090", "label": "Mus musculus"},
    ]
    record = DatasetRecord.model_validate(
        dataset_record(
            title="Registered title",
            primaryCellCount=None,
            facets={"organism": organisms},
        )
    )
    calls = []

    def convert(source_path, destination, manifest, *, progress):
        calls.append((source_path, destination, manifest, progress))
        return {"status": "needsInput", "needsInput": QUESTION}

    monkeypatch.setattr(build, "convert_local", convert)
    stages: list[str] = []

    def progress(stage, **kwargs):
        stages.append(stage)

    manifest, converted = build.build_local(
        record,
        source,
        tmp_path / "data.zarr",
        {"raw_data_location": "X"},
        size,
        checksum,
        progress,
    )
    assert calls == [
        (source, tmp_path / "data.zarr", manifest.model_dump(mode="json"), progress)
    ]
    assert manifest.title == "Registered title"
    assert manifest.organism == "Homo sapiens, Mus musculus"
    assert (manifest.apiRawDataLocation, manifest.countsSelectionSource) == (
        "X",
        "curation_api",
    )
    assert (manifest.sourceBytes, manifest.sourceSha256) == (size, checksum)
    assert str(manifest.datasetVersionId) == VERSION_ID
    assert converted["status"] == "needsInput"
    assert converted["needsInput"] == QUESTION
    assert converted["cytebaseId"] == CYTEBASE_ID
    assert set(converted["inspection"]) == {"h5ad_keys", "obs_summary", "uns"}
    assert stages[0] == "inspecting"
    assert set(record.timings) == {"inspectSeconds", "convertSeconds"}


@pytest.mark.parametrize(
    ("annotations", "primary_cell_count"),
    [(False, None), (True, 3)],
    ids=["no-primary-cells", "registered-count-differs"],
)
def test_build_local_rejects_a_primary_cell_count_disagreement(
    tmp_path, monkeypatch, annotations, primary_cell_count
):
    source = write_h5ad(tmp_path / "source.h5ad", annotations=annotations)
    record = DatasetRecord.model_validate(
        dataset_record(primaryCellCount=primary_cell_count)
    )
    monkeypatch.setattr(build, "convert_local", _unexpected)
    with pytest.raises(ValueError, match="primary-cell count disagrees"):
        build.build_local(
            record,
            source,
            tmp_path / "data.zarr",
            {},
            *source_details(source),
            noop,
        )
    assert record.timings == {}


def test_build_local_requires_a_finished_conversion(tmp_path, monkeypatch):
    source = write_h5ad(tmp_path / "source.h5ad")
    record = DatasetRecord.model_validate(dataset_record())
    monkeypatch.setattr(
        build, "convert_local", lambda *args, **kwargs: {"status": "failed"}
    )
    with pytest.raises(RuntimeError, match="Scarf conversion did not complete"):
        build.build_local(
            record,
            source,
            tmp_path / "data.zarr",
            {},
            *source_details(source),
            noop,
        )


# verify_store and _open_verified_arrays


@pytest.mark.parametrize("with_local", [False, True])
def test_verify_store_reports_metadata_and_closes_stores(
    cytebase_build, tmp_path, monkeypatch, with_local
):
    published = _copy_store(cytebase_build, tmp_path)
    opened = _track_stores(monkeypatch)
    local = str(cytebase_build.store) if with_local else None
    result = build.verify_store(
        str(published),
        _manifest_json(cytebase_build),
        {"token": False},
        local_location=local,
    )
    layout = zarr.open_group(str(published / "RNA"), mode="r").attrs[
        "scarf:countMatrixLayout"
    ]
    assert result == {
        "nObs": 6,
        "nVars": 5,
        "countsDtype": "int32",
        "countsTShape": [5, 6],
        "countsTComplete": True,
        "layoutFingerprint": layout["fingerprint"],
        "countsTMatches": True,
        "sourceSampleMatches": True,
        "countsBlock": COUNTS[:3].tolist(),
    }
    expected = [published, cytebase_build.store] if with_local else [published]
    assert [Path(store.root) for store in opened] == expected
    assert not any(store._is_open for store in opened)


@pytest.mark.parametrize(
    "location", ["hf://buckets/test/cytebase/data.zarr", "s3://bucket/data.zarr"]
)
def test_verify_store_requires_the_local_store_for_remote_locations(
    cytebase_build, monkeypatch, location
):
    opened = _track_stores(monkeypatch)
    with pytest.raises(ValueError, match="Remote verification requires the local"):
        build.verify_store(location, _manifest_json(cytebase_build))
    assert opened == []


@pytest.mark.parametrize("change", [{"nObs": 7}, {"nVars": 4}])
def test_verify_store_rejects_dimensions_that_differ_from_the_manifest(
    cytebase_build, change
):
    with pytest.raises(ValueError, match="count dimensions do not match the source"):
        build.verify_store(
            str(cytebase_build.store), _manifest_json(cytebase_build) | change
        )


def _remove_rna(store: Path) -> None:
    shutil.rmtree(store / "RNA")


def _replace_counts_with_group(store: Path) -> None:
    shutil.rmtree(store / "RNA" / "counts")
    zarr.open_group(str(store / "RNA"), mode="r+").create_group("counts")


def _rewrite_as_zarr_v2(store: Path) -> None:
    shutil.rmtree(store)
    rna = zarr.open_group(str(store), mode="w", zarr_format=2).create_group("RNA")
    rna.create_array("counts", shape=(6, 5), dtype="int32")
    rna.create_array("countsT", shape=(5, 6), dtype="int32")


def _retype(*names: str, dtype: str):
    def change(store: Path) -> None:
        rna = zarr.open_group(str(store / "RNA"), mode="r+")
        for name in names:
            array = rna[name]
            rna.create_array(
                name,
                shape=array.shape,
                dtype=dtype,
                attributes=dict(array.attrs),
                overwrite=True,
            )

    return change


def _set_complete(value: object):
    def change(store: Path) -> None:
        attributes = zarr.open_array(str(store / "RNA" / "countsT"), mode="r+").attrs
        if value is None:
            del attributes["complete"]
        else:
            attributes["complete"] = value

    return change


def _resize_ids(path: str, size: int):
    def change(store: Path) -> None:
        parent, name = path.rsplit("/", 1)
        zarr.open_group(str(store / parent), mode="r+").create_array(
            name, shape=(size,), dtype="int32", overwrite=True
        )

    return change


def _drop_layout(store: Path) -> None:
    del zarr.open_group(str(store / "RNA"), mode="r+").attrs["scarf:countMatrixLayout"]


@pytest.mark.parametrize(
    ("damage", "error", "message"),
    [
        (_remove_rna, KeyError, "RNA"),
        (_replace_counts_with_group, ValueError, "requires RNA/counts and RNA/countsT"),
        (_rewrite_as_zarr_v2, ValueError, "paired Zarr v3 layout"),
        (_retype("countsT", dtype="int64"), ValueError, "matching numeric dtypes"),
        (_retype("counts", "countsT", dtype="bool"), ValueError, "numeric dtypes"),
        (_set_complete(None), ValueError, "countsT is incomplete"),
        (_set_complete("true"), ValueError, "countsT is incomplete"),
        (_resize_ids("cellData/ids", 5), ValueError, "cellData/ids dimensions"),
        (
            _resize_ids("RNA/featureData/ids", 6),
            ValueError,
            "RNA/featureData/ids dimensions",
        ),
        (_drop_layout, ValueError, "count matrix layout metadata is missing"),
    ],
    ids=[
        "missing-assay",
        "counts-group",
        "zarr-v2",
        "dtype-mismatch",
        "non-numeric",
        "complete-missing",
        "complete-not-true",
        "cell-ids",
        "feature-ids",
        "layout-missing",
    ],
)
def test_verify_store_rejects_damaged_stores_and_closes_them(
    cytebase_build, tmp_path, monkeypatch, damage, error, message
):
    store = _copy_store(cytebase_build, tmp_path)
    damage(store)
    opened = _track_stores(monkeypatch)
    with pytest.raises(error, match=message):
        build.verify_store(str(store), _manifest_json(cytebase_build))
    assert len(opened) == 1
    assert not opened[0]._is_open


def test_verify_store_rejects_counts_t_that_disagrees_with_counts(
    cytebase_build, tmp_path, monkeypatch
):
    store = _copy_store(cytebase_build, tmp_path)
    _set_first_count(store, "RNA/countsT")
    opened = _track_stores(monkeypatch)
    with pytest.raises(ValueError, match="counts and countsT samples do not agree"):
        build.verify_store(str(store), _manifest_json(cytebase_build))
    assert not opened[0]._is_open


def test_verify_store_rejects_published_metadata_that_differs_from_local(
    cytebase_build, dense_build, monkeypatch
):
    opened = _track_stores(monkeypatch)
    with pytest.raises(ValueError, match="Published Scarf metadata differs"):
        build.verify_store(
            str(dense_build.store),
            _manifest_json(cytebase_build),
            local_location=str(cytebase_build.store),
        )
    assert len(opened) == 2
    assert not any(store._is_open for store in opened)


@pytest.mark.parametrize(
    ("published_edits", "local_edits"),
    [(("RNA/counts", "RNA/countsT"), ()), ((), ("RNA/countsT",))],
    ids=["published-values", "local-counts-t"],
)
def test_verify_store_rejects_published_counts_that_differ_from_local(
    cytebase_build, tmp_path, monkeypatch, published_edits, local_edits
):
    published = _copy_store(cytebase_build, tmp_path, "published.zarr")
    local = _copy_store(cytebase_build, tmp_path, "local.zarr")
    for path in published_edits:
        _set_first_count(published, path)
    for path in local_edits:
        _set_first_count(local, path)
    opened = _track_stores(monkeypatch)
    with pytest.raises(ValueError, match="counts differ from the local store sample"):
        build.verify_store(
            str(published), _manifest_json(cytebase_build), local_location=str(local)
        )
    assert len(opened) == 2
    assert not any(store._is_open for store in opened)


# replacement_paths and _source_details


def test_replacement_paths_lists_only_generated_output_of_the_dataset():
    listed = [
        f"{PREFIX}/metadata/obs.parquet",
        f"{PREFIX}/data.zarr/zarr.json",
        f"{PREFIX}/data.zarr/zarr.json",
        f"{PREFIX}/scarf_ingest.json",
        f"{PREFIX}/dataset.json",
        f"{PREFIX}/cellxgene/dataset.json",
        f"{PREFIX}/data.zarr.backup/zarr.json",
        f"{PREFIX}/metadata",
        f"{PREFIX}/scarf_ingest.json.bak",
        f"{PREFIX}_v2/data.zarr/zarr.json",
        "datasets/other/scarf_ingest.json",
        "catalog/cytebase.duckdb",
    ]
    prefixes = []

    def list_files(prefix: str) -> list[str]:
        prefixes.append(prefix)
        return listed

    record = DatasetRecord.model_validate(dataset_record())
    paths = build.replacement_paths(record, SimpleNamespace(list_files=list_files))
    assert paths == [
        f"{PREFIX}/data.zarr/zarr.json",
        f"{PREFIX}/metadata/obs.parquet",
        f"{PREFIX}/scarf_ingest.json",
    ]
    assert prefixes == [f"{PREFIX}/"]


def test_source_details_update_only_the_latest_version():
    record = DatasetRecord.model_validate(
        dataset_record(
            latestVersionId=NEW_VERSION_ID,
            versions=[
                {"datasetVersionId": VERSION_ID, "seenAt": NOW, "sourceSha256": "old"},
                {"datasetVersionId": NEW_VERSION_ID, "seenAt": NOW},
            ],
        )
    )
    build._source_details(record, 42, "b" * 64)
    assert record.sourceBytes == 42
    assert [version.sourceSha256 for version in record.versions] == ["old", "b" * 64]


# publish_store


@dataclass
class Publisher:
    """Run ``publish_store`` on the fake hub, recording progress and ownership checks."""

    hub: FakeHub
    built: CytebaseBuild
    stages: list[str] = field(default_factory=list)
    owner_checks: list[int] = field(default_factory=list)

    def __call__(
        self,
        record: DatasetRecord,
        *,
        request: dict | None = None,
        store: Path | None = None,
        manifest: Manifest | None = None,
        converted: dict | None = None,
    ) -> dict:
        return build.publish_store(
            record,
            {} if request is None else request,
            self.hub.bucket(),
            self.built.store if store is None else store,
            self.built.manifest() if manifest is None else manifest,
            self.built.converted() if converted is None else converted,
            lambda stage, **kwargs: self.stages.append(stage),
            lambda: self.owner_checks.append(len(self.hub.calls)),
        )


@pytest.fixture
def publish(fake_hub, cytebase_build) -> Publisher:
    return Publisher(fake_hub, cytebase_build)


def _operations(hub: FakeHub) -> list[tuple[str, ...]]:
    """Summarize bucket calls as listings, writes, deletions and store uploads."""
    operations: list[tuple[str, ...]] = []
    for name, *arguments in hub.calls:
        if name == "list_bucket_tree":
            operations.append(("list", arguments[1]))
        elif name == "sync_bucket":
            operations.append(("sync", arguments[1]))
        elif name == "batch_bucket_files":
            _, added, deleted, _ = arguments
            operations.append(("write", *added) if added else ("delete", *deleted))
        else:
            operations.append((name,))
    return operations


def _assert_writes_follow_ownership_checks(hub: FakeHub, owner_checks: list[int]):
    writes = [
        index for index, call in enumerate(hub.calls) if call[0] != "list_bucket_tree"
    ]
    assert writes
    assert set(writes) <= set(owner_checks)


def _bucket_record(hub: FakeHub) -> DatasetRecord:
    return DatasetRecord.model_validate(hub.read_json(RECORD_PATH))


def _bucket_contents(hub: FakeHub) -> dict[str, bytes]:
    return {path: hub.read(path) for path in hub.files()}


def _needs_input_result(built: CytebaseBuild) -> dict[str, Any]:
    converted = built.converted()
    for key in ("qcSummary", "importedArtifacts"):
        del converted[key]
    return converted | {"status": "needsInput", "needsInput": QUESTION}


def _moment(value: str) -> datetime:
    return datetime.fromisoformat(value)


def test_publish_store_publishes_a_verified_ready_dataset(
    fake_hub, publish, cytebase_build
):
    zarr_uri = f"{fake_hub.bucket().root}/{PREFIX}/data.zarr"
    at_upload = []
    fake_hub.hooks["sync_bucket"] = [
        lambda source, dest: at_upload.append(
            (fake_hub.read_json(RECORD_PATH), fake_hub.files())
        )
    ]
    record = cytebase_build.record()
    assert publish(record) == {"outcome": "succeeded"}

    [(committed, files)] = at_upload
    assert (committed["status"], committed["zarrUri"]) == ("processing", None)
    assert files == [RECORD_PATH]
    assert _operations(fake_hub) == [
        LIST,
        ("write", RECORD_PATH),
        LIST,
        ("sync", zarr_uri),
        ("write", INGEST_PATH),
        ("write", RECORD_PATH),
    ]
    _assert_writes_follow_ownership_checks(fake_hub, publish.owner_checks)
    assert publish.stages == ["uploading_store", "verifying_store"]
    assert _store_files(Path(zarr_uri)) == _store_files(cytebase_build.store)

    ready = fake_hub.read_json(RECORD_PATH)
    assert (ready["status"], ready["stageOutcome"]) == ("ready", "succeeded")
    assert ready["zarrUri"] == zarr_uri
    assert ready["h5adUri"] is None
    assert ready["needsInput"] is None
    assert ready["processedVersionId"] == VERSION_ID
    assert (ready["cellCount"], ready["nGenes"], ready["primaryCellCount"]) == (6, 5, 4)
    assert ready["sourceBytes"] == cytebase_build.size
    assert ready["inspection"] == _manifest_json(cytebase_build)
    receipt = ready["buildReceipt"]
    assert {key: receipt[key] for key in ("datasetVersionId", "sourceSha256")} == {
        "datasetVersionId": VERSION_ID,
        "sourceSha256": cytebase_build.sha256,
    }
    assert receipt["zarrUri"] == zarr_uri
    assert receipt["verification"]["countsTMatches"] is True
    assert receipt["verification"]["countsBlock"] == COUNTS[:3].tolist()
    completed = _moment(receipt["verifiedAt"])
    assert _moment(ready["processedAt"]) == completed
    assert _moment(ready["updatedAt"]) == completed
    [version] = ready["versions"]
    assert _moment(version["processedAt"]) == completed
    assert version["sourceSha256"] == cytebase_build.sha256
    assert {"uploadSeconds", "verifySeconds"} <= set(ready["timings"])

    ingest = fake_hub.read_json(INGEST_PATH)
    assert ingest["status"] == "done"
    assert ingest["zarrPath"] == zarr_uri
    assert ingest["verification"] == receipt["verification"]
    assert ingest["completedAt"] == receipt["verifiedAt"]
    assert record.model_dump(mode="json") == ready


def test_publish_store_records_a_needs_input_result_without_uploading(
    fake_hub, publish, cytebase_build
):
    record = cytebase_build.record()
    result = publish(record, converted=_needs_input_result(cytebase_build))
    assert result == {
        "outcome": "needsInput",
        "message": "Counts conversion needs a user decision",
    }
    assert fake_hub.files() == [INGEST_PATH]
    ingest = fake_hub.read_json(INGEST_PATH)
    assert (ingest["status"], ingest["needsInput"]) == ("needsInput", QUESTION)
    assert _operations(fake_hub) == [LIST, ("write", INGEST_PATH)]
    _assert_writes_follow_ownership_checks(fake_hub, publish.owner_checks)
    assert publish.stages == ["uploading_metadata"]
    assert (record.status, record.needsInput) == ("needsInput", QUESTION)
    assert record.inspection == cytebase_build.manifest()
    assert record.sourceBytes == cytebase_build.size
    assert record.versions[0].sourceSha256 == cytebase_build.sha256
    assert (record.zarrUri, record.processedVersionId) == (None, None)


def test_publish_store_keeps_existing_output_when_input_is_needed(
    ready_dataset, fake_hub, publish, cytebase_build
):
    before = _bucket_contents(fake_hub)
    fake_hub.calls.clear()
    record = _bucket_record(fake_hub)
    result = publish(record, converted=_needs_input_result(cytebase_build))
    assert result["outcome"] == "needsInput"
    assert _bucket_contents(fake_hub) == before
    assert _operations(fake_hub) == [LIST]
    assert publish.stages == []
    assert (record.status, record.needsInput) == ("needsInput", QUESTION)


@pytest.mark.parametrize("approve_store", [False, True])
def test_publish_store_requires_approval_for_every_generated_path(
    ready_dataset, fake_hub, publish, approve_store
):
    fake_hub.put(f"{PREFIX}/metadata/obs.parquet", b"old")
    record = _bucket_record(fake_hub)
    generated = build.replacement_paths(record, ready_dataset.bucket)
    store_paths = [
        path for path in generated if path.startswith(f"{PREFIX}/data.zarr/")
    ]
    approved = store_paths if approve_store else []
    before = _bucket_contents(fake_hub)
    fake_hub.calls.clear()
    result = publish(record, request={"approvedDeletionPaths": approved})
    assert result == {
        "outcome": "needsApproval",
        "message": "Review these exact generated paths and resubmit with "
        "approvedDeletionPaths",
        "deletionPaths": sorted(set(generated) - set(approved)),
    }
    if approve_store:
        assert result["deletionPaths"] == [
            f"{PREFIX}/metadata/obs.parquet",
            INGEST_PATH,
        ]
    assert _operations(fake_hub) == [LIST]
    assert publish.owner_checks == [0]
    assert _bucket_contents(fake_hub) == before
    assert record == _bucket_record(fake_hub)


def test_publish_store_stops_when_generated_output_appears_during_publication(
    fake_hub, publish, cytebase_build
):
    late = f"{PREFIX}/metadata/late.parquet"
    listings = []

    def add_late_output(bucket_id, prefix):
        listings.append(prefix)
        if len(listings) == 2:
            fake_hub.put(late, b"late")

    fake_hub.hooks["list_bucket_tree"] = [add_late_output]
    record = cytebase_build.record()
    assert publish(record) == {
        "outcome": "needsApproval",
        "message": "Generated output changed during publication; review these "
        "exact paths and resubmit with approvedDeletionPaths",
        "deletionPaths": [late],
    }
    committed = fake_hub.read_json(RECORD_PATH)
    assert committed["status"] == "processing"
    assert (committed["zarrUri"], committed["buildReceipt"]) == (None, None)
    assert fake_hub.files() == [RECORD_PATH, late]
    assert _operations(fake_hub) == [LIST, ("write", RECORD_PATH), LIST]
    assert publish.stages == []
    assert record.status == "processing"


def test_publish_store_replaces_an_older_ready_build_after_withdrawing_it(
    ready_dataset, fake_hub, publish, cytebase_build
):
    stale = [f"{PREFIX}/data.zarr/stale/zarr.json", f"{PREFIX}/metadata/obs.parquet"]
    for path in stale:
        fake_hub.put(path, b"stale")
    source_metadata = f"{PREFIX}/cellxgene/dataset.json"
    fake_hub.put(source_metadata, {"dataset_id": DATASET_ID})
    # Registration has since recorded a newer source version.
    registered = fake_hub.read_json(RECORD_PATH)
    [older] = registered["versions"]
    registered["latestVersionId"] = NEW_VERSION_ID
    registered["versions"].append({"datasetVersionId": NEW_VERSION_ID, "seenAt": NOW})
    record = DatasetRecord.model_validate(registered)
    manifest = cytebase_build.manifest().model_copy(
        update={"datasetVersionId": record.latestVersionId}
    )
    approved = build.replacement_paths(record, ready_dataset.bucket)
    assert set(stale) | {INGEST_PATH} < set(approved)
    at_delete = []

    def record_state_before_deletion(bucket_id, added, deleted):
        if deleted:
            at_delete.append(fake_hub.read_json(RECORD_PATH))

    fake_hub.hooks["batch_bucket_files"] = [record_state_before_deletion]
    fake_hub.calls.clear()

    result = publish(
        record, request={"approvedDeletionPaths": approved}, manifest=manifest
    )
    assert result == {"outcome": "succeeded"}
    ready = fake_hub.read_json(RECORD_PATH)
    assert ready["processedVersionId"] == NEW_VERSION_ID
    assert ready["buildReceipt"]["datasetVersionId"] == NEW_VERSION_ID
    assert ready["versions"][0] == older
    assert _moment(ready["versions"][1]["processedAt"]) == _moment(ready["processedAt"])
    [withdrawn] = at_delete
    assert withdrawn["status"] == "processing"
    assert (withdrawn["zarrUri"], withdrawn["processedVersionId"]) == (None, None)
    assert withdrawn["buildReceipt"] is None
    assert _operations(fake_hub) == [
        LIST,
        ("write", RECORD_PATH),
        LIST,
        ("delete", *approved),
        ("sync", str(ready_dataset.store)),
        ("write", INGEST_PATH),
        ("write", RECORD_PATH),
    ]
    _assert_writes_follow_ownership_checks(fake_hub, publish.owner_checks)
    assert publish.stages == ["replacing", "uploading_store", "verifying_store"]
    files = fake_hub.files()
    assert not set(stale) & set(files)
    assert source_metadata in files
    assert _store_files(ready_dataset.store) == _store_files(cytebase_build.store)
    assert _bucket_record(fake_hub).status == "ready"


def test_publish_store_never_marks_an_unverified_upload_ready(
    fake_hub, publish, cytebase_build, tmp_path
):
    store = _copy_store(cytebase_build, tmp_path)
    _set_first_count(store, "RNA/countsT")
    record = cytebase_build.record()
    with pytest.raises(ValueError, match="counts and countsT samples do not agree"):
        publish(record, store=store)
    committed = fake_hub.read_json(RECORD_PATH)
    assert (committed["status"], committed["zarrUri"]) == ("processing", None)
    assert INGEST_PATH not in fake_hub.files()
    assert (record.status, record.buildReceipt) == ("processing", None)
    assert publish.stages == ["uploading_store", "verifying_store"]


def test_publish_store_retries_transient_verification_errors(
    fake_hub, publish, cytebase_build, monkeypatch, recorded_sleeps
):
    attempts = []
    verify = build.verify_store

    def flaky_verify(location, manifest, **options):
        attempts.append(options)
        if len(attempts) == 1:
            raise httpx.ReadTimeout("published store is not visible yet")
        return verify(location, manifest, **options)

    monkeypatch.setattr(build, "verify_store", flaky_verify)
    assert publish(cytebase_build.record()) == {"outcome": "succeeded"}
    assert attempts == 2 * [
        {
            "storage_options": {"token": False, "skip_instance_cache": True},
            "local_location": str(cytebase_build.store),
        }
    ]
    assert recorded_sleeps == [2.0]
    assert publish.stages == ["uploading_store", "verifying_store", "retrying_transfer"]
    assert fake_hub.read_json(RECORD_PATH)["status"] == "ready"
