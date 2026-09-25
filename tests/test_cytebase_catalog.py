"""Offline tests for the Cytebase catalog cache and queries."""

import shutil
import sys
from datetime import datetime
from importlib.metadata import requires
from pathlib import Path
from types import SimpleNamespace

import pytest

from scarf.cytebase import catalog as catalog_module
from scarf.cytebase.catalog import Catalog
from tests.fixtures_cytebase import (
    BUCKET_ID,
    CYTEBASE_ID,
    DATASET_ID,
    NOW,
    VERSION_ID,
    dataset_record,
    publish_catalog_rows,
)

pytest.importorskip("duckdb")
pytest.importorskip("natsort")
pytestmark = pytest.mark.usefixtures("cytebase_offline")

HASH_PATH = "catalog/cytebase.duckdb.sha256"
DB_PATH = "catalog/cytebase.duckdb"
BLOOD_ID = "doe_2023_blood_atlas_55555555"
COLON_ID = "lee_2022_colon_atlas_77777777"


def _terms(*labels: str) -> list[dict[str, str | None]]:
    return [{"termId": None, "label": label} for label in labels]


def _records() -> list[dict]:
    ready = dataset_record(
        status="ready",
        processedVersionId=VERSION_ID,
        zarrUri="/published/lung/data.zarr",
        cellCount=100,
    )
    registered = dataset_record(
        cytebaseId=BLOOD_ID,
        datasetId="55555555-5555-4555-8555-555555555555",
        title="Blood atlas",
        citation=None,
        firstAuthor="Doe",
        cellCount=50,
        facets=dataset_record()["facets"]
        | {"tissue": _terms("blood"), "disease": _terms("COVID-19")},
    )
    stale = dataset_record(
        cytebaseId=COLON_ID,
        datasetId="77777777-7777-4777-8777-777777777777",
        title="Colon atlas",
        firstAuthor="Lee",
        status="update_available",
        processedVersionId="66666666-6666-4666-8666-666666666666",
        zarrUri="/published/colon/data.zarr",
        cellCount=None,
        facets=dataset_record()["facets"]
        | {"tissue": _terms("colon 10", "colon 2"), "disease": _terms("normal")},
    )
    return [ready, registered, stale]


def _downloads(hub, remote: str) -> int:
    return sum(
        1
        for call in hub.calls
        if call[0] == "download_bucket_files" and remote in call[2]
    )


@pytest.fixture
def published(fake_hub):
    publish_catalog_rows(fake_hub, _records())
    return fake_hub


@pytest.fixture
def catalog(published) -> Catalog:
    return Catalog(BUCKET_ID, token=False)


def test_cytebase_extra_declares_pytz_for_duckdb_timestamps():
    extra = [req for req in requires("scarf") or [] if 'extra == "cytebase"' in req]
    assert any(req.startswith("pytz") for req in extra)
    assert any(req.startswith("duckdb") for req in extra)


@pytest.mark.parametrize(
    ("explicit", "environment", "expected"),
    [
        (None, None, "Nygen/cytebase"),
        (None, "example/environment", "example/environment"),
        ("example/explicit", "example/environment", "example/explicit"),
        ("hf://buckets/example/explicit/", None, "example/explicit"),
    ],
)
def test_catalog_bucket_precedence(monkeypatch, explicit, environment, expected):
    monkeypatch.delenv("CYTEBASE_BUCKET", raising=False)
    if environment is not None:
        monkeypatch.setenv("CYTEBASE_BUCKET", environment)
    opened = []
    monkeypatch.setattr(catalog_module, "_cached_catalog", opened.append)

    catalog = Catalog(bucket=explicit)

    assert opened == [catalog._storage]
    assert catalog._storage.bucket_id == expected
    assert catalog._storage.token is False


@pytest.mark.parametrize("explicit", [None, ""])
def test_catalog_rejects_empty_bucket_configuration(monkeypatch, explicit):
    monkeypatch.setenv("CYTEBASE_BUCKET", "")
    with pytest.raises(ValueError, match="Supply bucket="):
        Catalog(bucket=explicit)


def test_cache_directory_uses_the_home_directory(cytebase_offline):
    assert catalog_module._cache_directory() == cytebase_offline / ".scarf"


def test_cache_directory_on_windows(monkeypatch, tmp_path, cytebase_offline):
    monkeypatch.setattr(catalog_module, "sys", SimpleNamespace(platform="win32"))
    assert catalog_module._cache_directory() == (
        cytebase_offline / "AppData" / "Local" / "scarf"
    )
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    assert catalog_module._cache_directory() == tmp_path / "local" / "scarf"


@pytest.mark.parametrize(
    "raw",
    [
        b"a" * 64 + b"  cytebase.duckdb",
        b"a" * 64 + b"  cytebase.duckdb\n",
        b"A" * 64 + b"  cytebase.duckdb\r\n",
    ],
)
def test_parse_hash_accepts_sha256_sidecars(raw):
    assert catalog_module._parse_hash(raw) == "a" * 64


@pytest.mark.parametrize(
    "raw",
    [b"a" * 64, b"a" * 63 + b"  cytebase.duckdb", b"a" * 64 + b" cytebase.duckdb"],
)
def test_parse_hash_rejects_other_text(raw):
    with pytest.raises(ValueError, match="SHA-256 digest"):
        catalog_module._parse_hash(raw)


def test_remote_hash_requires_a_published_sidecar(fake_hub):
    bucket = fake_hub.bucket()
    with pytest.raises(RuntimeError, match="Missing catalog/cytebase.duckdb.sha256"):
        catalog_module._remote_hash(bucket)
    fake_hub.put(HASH_PATH, "not a checksum")
    with pytest.raises(RuntimeError, match="Malformed") as raised:
        catalog_module._remote_hash(bucket)
    assert isinstance(raised.value.__cause__, ValueError)


def test_catalog_downloads_once_and_reuses_the_verified_copy(published):
    catalog = Catalog(BUCKET_ID, token=False)
    local = Path.home() / ".scarf" / "cytebase.duckdb"
    assert local.read_bytes() == published.read(DB_PATH)
    assert local.with_suffix(".duckdb.sha256").read_bytes() == published.read(HASH_PATH)
    assert local.stat().st_mode & 0o777 == 0o600
    assert len(catalog.find_datasets(ready_only=False)) == 3
    assert _downloads(published, DB_PATH) == 1
    assert _downloads(published, HASH_PATH) == 3


def test_catalog_refreshes_when_the_published_catalog_changes(published):
    catalog = Catalog(BUCKET_ID, token=False)
    assert catalog.dataset(CYTEBASE_ID).title == "Healthy lung scRNA-seq atlas"
    publish_catalog_rows(published, [dataset_record(title="Renamed atlas")])
    assert catalog.dataset(CYTEBASE_ID).title == "Renamed atlas"
    assert _downloads(published, DB_PATH) == 2


@pytest.mark.parametrize("sidecar", [None, b"garbage"])
def test_catalog_repairs_missing_or_damaged_local_sidecars(published, sidecar):
    Catalog(BUCKET_ID, token=False)
    local_sidecar = Path.home() / ".scarf" / "cytebase.duckdb.sha256"
    if sidecar is None:
        local_sidecar.unlink()
    else:
        local_sidecar.write_bytes(sidecar)
    Catalog(BUCKET_ID, token=False)
    assert local_sidecar.read_bytes() == published.read(HASH_PATH)
    assert _downloads(published, DB_PATH) == 1


def test_verified_cache_rejects_missing_vanishing_and_changed_files(
    tmp_path, monkeypatch
):
    database = tmp_path / "cytebase.duckdb"
    assert not catalog_module._verified_cache(database, "0" * 64)
    database.write_bytes(b"catalog")
    assert not catalog_module._verified_cache(database, "0" * 64)

    def vanished(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr(catalog_module, "_file_hash", vanished)
    assert not catalog_module._verified_cache(database, "0" * 64)


def test_cached_catalog_retries_when_publication_changes_mid_download(
    published, recorded_sleeps
):
    replacement = dataset_record(title="Republished atlas")
    changed = []

    def republish(bucket_id, remote):
        if remote == DB_PATH and not changed:
            changed.append(remote)
            publish_catalog_rows(published, [replacement])

    published.hooks["download_bucket_files"] = [republish]
    catalog = Catalog(BUCKET_ID, token=False)
    assert recorded_sleeps == [1]
    assert catalog.dataset(CYTEBASE_ID).title == "Republished atlas"


def test_cached_catalog_gives_up_after_three_mismatched_downloads(
    published, recorded_sleeps
):
    published.put(HASH_PATH, "b" * 64 + "  cytebase.duckdb\n")
    with pytest.raises(RuntimeError, match="did not match its published SHA-256"):
        Catalog(BUCKET_ID, token=False)
    assert recorded_sleeps == [1, 2]
    assert _downloads(published, DB_PATH) == 3
    assert list((Path.home() / ".scarf").glob(".catalog-*")) == []


def _replace_raising(monkeypatch, prefix: str, before=None):
    original = Path.replace

    def replace(self, target):
        if self.name.startswith(prefix):
            if before is not None:
                before(self, Path(target))
            raise PermissionError(f"{target} is open elsewhere")
        return original(self, target)

    monkeypatch.setattr(Path, "replace", replace)


def test_save_hash_accepts_an_identical_concurrent_sidecar(tmp_path, monkeypatch):
    sidecar = tmp_path / "cytebase.duckdb.sha256"
    sidecar.write_text("c" * 64 + "  cytebase.duckdb\n")
    _replace_raising(monkeypatch, ".checksum-")
    catalog_module._save_hash(sidecar, "c" * 64)
    assert list(tmp_path.glob(".checksum-*")) == []


@pytest.mark.parametrize("existing", [None, "garbage", "d" * 64 + "  cytebase.duckdb"])
def test_save_hash_reraises_when_the_sidecar_differs(tmp_path, monkeypatch, existing):
    sidecar = tmp_path / "cytebase.duckdb.sha256"
    if existing is not None:
        sidecar.write_text(existing)
    _replace_raising(monkeypatch, ".checksum-")
    with pytest.raises(PermissionError):
        catalog_module._save_hash(sidecar, "c" * 64)
    assert list(tmp_path.glob(".checksum-*")) == []


def _catalog_file(directory: Path) -> tuple[Path, str]:
    source = directory / "source.duckdb"
    source.write_bytes(b"verified catalog bytes")
    return source, catalog_module._file_hash(source)


def test_install_keeps_a_copy_another_process_already_verified(tmp_path):
    source, digest = _catalog_file(tmp_path)
    destination = tmp_path / "cytebase.duckdb"
    shutil.copyfile(source, destination)
    temporary = tmp_path / ".catalog-new.download"
    temporary.write_bytes(b"unused")
    catalog_module._install(temporary, destination, digest)
    assert temporary.exists()
    assert destination.read_bytes() == b"verified catalog bytes"


def test_install_accepts_a_verified_copy_after_a_refused_replacement(
    tmp_path, monkeypatch
):
    source, digest = _catalog_file(tmp_path)
    destination = tmp_path / "cytebase.duckdb"
    temporary = tmp_path / ".catalog-new.download"
    shutil.copyfile(source, temporary)
    _replace_raising(
        monkeypatch,
        ".catalog-",
        before=lambda path, target: shutil.copyfile(path, target),
    )
    catalog_module._install(temporary, destination, digest)
    assert destination.read_bytes() == b"verified catalog bytes"


def test_install_reports_a_catalog_it_cannot_refresh(tmp_path, monkeypatch):
    source, digest = _catalog_file(tmp_path)
    temporary = tmp_path / ".catalog-new.download"
    shutil.copyfile(source, temporary)
    _replace_raising(monkeypatch, ".catalog-")
    with pytest.raises(RuntimeError, match="Cannot refresh catalog cache"):
        catalog_module._install(temporary, tmp_path / "cytebase.duckdb", digest)


def test_connect_catalog_requires_the_cytebase_extra(catalog, monkeypatch):
    monkeypatch.setitem(sys.modules, "duckdb", None)
    with pytest.raises(ImportError, match=r"scarf\[cytebase\] extra"):
        catalog.connect_catalog()


def test_query_returns_rows_and_timezone_aware_timestamps(catalog):
    rows = catalog.query(
        "SELECT cytebase_id, registered_at FROM datasets WHERE cytebase_id = ?",
        [CYTEBASE_ID],
        max_cell_chars=None,
    )
    assert rows[0]["cytebase_id"] == CYTEBASE_ID
    assert rows[0]["registered_at"] == datetime.fromisoformat(NOW)
    named = catalog.query(
        "SELECT count(*) AS n FROM datasets WHERE status = $status",
        {"status": "registered"},
    )
    assert named == [{"n": 1}]


def test_find_datasets_returns_ready_datasets_by_default(catalog):
    ready = catalog.find_datasets()
    assert [row["cytebase_id"] for row in ready] == [CYTEBASE_ID]
    assert ready[0]["dataset_id"] == DATASET_ID
    assert ready.to_markdown().splitlines()[0] == (
        "| Cytebase ID | Status | Cells | Genes | Tissues | Diseases |"
    )


def test_find_datasets_filters_exact_labels(catalog):
    everything = catalog.find_datasets(ready_only=False)
    assert [row["cytebase_id"] for row in everything] == [
        CYTEBASE_ID,
        BLOOD_ID,
        COLON_ID,
    ]
    lung = catalog.find_datasets(ready_only=False, tissue="lung")
    assert [row["cytebase_id"] for row in lung] == [CYTEBASE_ID]
    either = catalog.find_datasets(
        ready_only=False, disease=["COVID-19", "missing"], sex="female"
    )
    assert [row["cytebase_id"] for row in either] == [BLOOD_ID]
    assert catalog.find_datasets(tissue="blood") == []


def test_find_datasets_rejects_unknown_facets_and_labels(catalog):
    with pytest.raises(ValueError, match="Unknown facet 'color'"):
        catalog.find_datasets(color="red")
    for labels in (("lung",), ["lung", 1]):
        with pytest.raises(TypeError, match="strings or lists of strings"):
            catalog.find_datasets(tissue=labels)


def test_search_matches_every_word_ignoring_case(catalog):
    assert [row["cytebase_id"] for row in catalog.search("LUNG smith")] == [CYTEBASE_ID]
    assert catalog.search("blood") == []
    assert [
        row["cytebase_id"] for row in catalog.search("covid", ready_only=False)
    ] == [BLOOD_ID]
    everything = catalog.search("atlas", ready_only=False, limit=None)
    assert len(everything) == 3
    assert len(catalog.search("atlas", ready_only=False, limit=2)) == 2
    assert everything.to_markdown(max_rows=0).splitlines()[0] == (
        "| Cytebase ID | Title | Cells | Tissues | Diseases |"
    )


@pytest.mark.parametrize("text", ["", "   ", 5])
def test_search_requires_words(catalog, text):
    with pytest.raises(ValueError, match="at least one search word"):
        catalog.search(text)


@pytest.mark.parametrize("limit", [True, 0, 1.5])
def test_search_requires_a_positive_limit(catalog, limit):
    with pytest.raises(ValueError, match="limit must be a positive integer"):
        catalog.search("atlas", limit=limit)


def test_dataset_returns_a_handle_or_raises(catalog):
    dataset = catalog.dataset(BLOOD_ID)
    assert dataset.id == BLOOD_ID
    assert dataset.row["title"] == "Blood atlas"
    with pytest.raises(KeyError, match="No catalog dataset is registered as 'x'"):
        catalog.dataset("x")


def test_list_terms_counts_datasets_in_natural_order(catalog):
    tissues = catalog.list_terms("tissue")
    assert [(row["label"], row["n_datasets"]) for row in tissues] == [
        ("blood", 1),
        ("colon 2", 1),
        ("colon 10", 1),
        ("lung", 1),
    ]
    assert tissues.to_markdown().splitlines()[0] == "| Label | Ontology ID | Datasets |"
    everything = catalog.list_terms(max_cell_chars=None)
    assert {row["facet"] for row in everything} >= {"tissue", "disease", "cell_type"}
    assert everything.to_markdown().splitlines()[0].startswith("| Facet | Label |")
    with pytest.raises(ValueError, match="Unknown facet 'color'"):
        catalog.list_terms("color")


def test_open_and_mount_recheck_the_catalog_then_delegate(
    catalog, published, monkeypatch, tmp_path
):
    from scarf.cytebase import connector

    calls = []
    monkeypatch.setattr(
        connector,
        "open_dataset",
        lambda storage, cytebase_id, **options: (
            calls.append(("open", storage.bucket_id, cytebase_id, options)) or "opened"
        ),
    )
    monkeypatch.setattr(
        connector,
        "mount_dataset",
        lambda storage, cytebase_id, at, **options: (
            calls.append(("mount", storage.bucket_id, cytebase_id, at, options))
            or "mounted"
        ),
    )
    checks = _downloads(published, HASH_PATH)
    assert catalog.open_dataset(CYTEBASE_ID, nthreads=1) == "opened"
    assert catalog.mount_dataset(CYTEBASE_ID, tmp_path / "a.zarr") == "mounted"
    assert calls == [
        ("open", BUCKET_ID, CYTEBASE_ID, {"nthreads": 1}),
        ("mount", BUCKET_ID, CYTEBASE_ID, tmp_path / "a.zarr", {}),
    ]
    assert _downloads(published, HASH_PATH) == checks + 2
