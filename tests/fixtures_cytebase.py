"""Offline fakes and builders shared by the Cytebase SDK and pipeline tests.

Nothing here is autouse. Cytebase test modules opt in with
``pytestmark = pytest.mark.usefixtures("cytebase_offline")``, which blocks
non-loopback network access, isolates credentials and the home directory, turns
unexpected retry backoff into a failure, and keeps unpatched Modal calls from
authenticating.
"""

import asyncio
import copy
import functools
import hashlib
import ipaddress
import json
import os
import shutil
import socket
import time
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from huggingface_hub import BucketFile, BucketFolder
from huggingface_hub.errors import EntryNotFoundError

if TYPE_CHECKING:
    from scarf.cytebase._storage import Bucket

BUCKET_ID = "test/cytebase"
COLLECTION_ID = "11111111-1111-4111-8111-111111111111"
DATASET_ID = "22222222-2222-4222-8222-222222222222"
VERSION_ID = "33333333-3333-4333-8333-333333333333"
NEW_VERSION_ID = "44444444-4444-4444-8444-444444444444"
CYTEBASE_ID = "smith_2024_healthy_lung_scrna_atlas_22222222"
SOURCE_URL = "https://datasets.example.org/source.h5ad"
PIPELINE_VERSION = "test-pipeline"
NOW = "2026-01-02T03:04:05+00:00"
CITATION = (
    "Publication: https://doi.org/10.1000/lung Dataset Version: "
    f"{SOURCE_URL} curated and distributed by CZ CELLxGENE Discover in "
    f"Collection: https://cellxgene.cziscience.com/collections/{COLLECTION_ID}"
)
GENES = ["CD3E", "MS4A1", "LYZ", "ACTB", "GAPDH"]
COUNTS = np.array(
    [
        [1, 0, 2, 5, 3],
        [0, 4, 0, 6, 1],
        [3, 0, 1, 7, 2],
        [0, 2, 0, 4, 5],
        [2, 1, 0, 3, 1],
        [0, 1, 6, 2, 1],
    ],
    dtype=np.int32,
)
CELL_TYPES = ["T cell", "B cell", "T cell", "monocyte", "B cell", "monocyte"]
DONORS = ["D1", "D1", "D2", "D2", "D3", "D3"]
PRIMARY = [True, True, True, True, False, False]
UMAP = np.array(
    [[0.0, 1.0], [1.0, 0.5], [2.0, 2.5], [3.0, 1.5], [4.0, 3.0], [5.0, 0.0]],
    dtype=np.float32,
)


class NetworkAccessBlocked(BaseException):
    """Escapes ``retry()`` and ``except Exception`` so network use fails loudly."""


def _loopback(host: object) -> bool:
    if host in (None, "localhost"):
        return True
    try:
        return ipaddress.ip_address(str(host).split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def _unexpected_sleep(seconds: float) -> None:
    raise AssertionError(f"Unexpected retry backoff of {seconds} seconds")


def noop(*args: Any, **kwargs: Any) -> None:
    """A progress or ownership callback that accepts anything."""


@pytest.fixture
def cytebase_offline(monkeypatch, tmp_path):
    """Block remote traffic and isolate credentials, home, sleeps and Modal."""
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def check(sock: socket.socket, address: Any) -> None:
        if sock.family in (socket.AF_INET, socket.AF_INET6) and not _loopback(
            address[0]
        ):
            raise NetworkAccessBlocked(f"Offline test tried to connect to {address!r}")

    def connect(sock: socket.socket, address: Any) -> None:
        check(sock, address)
        real_connect(sock, address)

    def connect_ex(sock: socket.socket, address: Any) -> int:
        check(sock, address)
        return real_connect_ex(sock, address)

    def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if not _loopback(host):
            raise NetworkAccessBlocked(f"Offline test tried to resolve {host!r}")
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)

    for name in (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "CYTEBASE_BUCKET",
        "CYTEBASE_PIPELINE_VERSION",
        "CYTEBASE_DOWNLOAD_CONNECTIONS",
        "CYTEBASE_PROCESS_CONTAINERS",
        "SCARF_CYTEBASE_LOCAL",
        "LOCALAPPDATA",
        "MODAL_IS_REMOTE",
        "MODAL_PROFILE",
    ):
        monkeypatch.delenv(name, raising=False)
    # Keep matplotlib's existing font cache while HOME points elsewhere.
    real_home = Path.home()
    for name, default in (
        ("XDG_CACHE_HOME", real_home / ".cache"),
        ("XDG_CONFIG_HOME", real_home / ".config"),
    ):
        if not os.environ.get(name):
            monkeypatch.setenv(name, str(default))
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    # Empty tokens override ~/.modal.toml, so a missed Modal call fails locally.
    monkeypatch.setenv("MODAL_TOKEN_ID", "")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "")

    from scarf.cytebase import _storage, catalog

    monkeypatch.setattr(_storage, "get_token", lambda: None)
    tripwire = SimpleNamespace(sleep=_unexpected_sleep, time=time.time)
    monkeypatch.setattr(_storage, "time", tripwire)
    monkeypatch.setattr(catalog, "time", tripwire)
    return home


@pytest.fixture
def recorded_sleeps(cytebase_offline, monkeypatch) -> list[float]:
    """Record retry backoff instead of sleeping; the clock is fixed."""
    from scarf.cytebase import _storage, catalog

    sleeps: list[float] = []
    clock = SimpleNamespace(sleep=sleeps.append, time=lambda: 1_700_000_000.0)
    monkeypatch.setattr(_storage, "time", clock)
    monkeypatch.setattr(catalog, "time", clock)
    return sleeps


def bucket_file(path: str, size: int) -> BucketFile:
    return BucketFile(
        type="file",
        path=path,
        size=size,
        xetHash="test-hash",
        mtime=None,
        uploadedAt=None,
    )


def bucket_folder(path: str) -> BucketFolder:
    return BucketFolder(type="directory", path=path, uploadedAt=None)


class FakeHub:
    """Directory-backed stand-in for the Hugging Face bucket functions.

    Each bucket is a directory under ``root``. While installed by ``fake_hub``,
    every ``Bucket`` is rooted in its directory, so store URIs are local paths.
    ``hooks[name]`` callbacks run before the named operation with its arguments.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls: list[tuple[Any, ...]] = []
        self.hooks: dict[str, list[Callable[..., None]]] = {}

    def directory(self, bucket_id: str = BUCKET_ID) -> Path:
        path = self.root / bucket_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def path(self, remote: str, bucket_id: str = BUCKET_ID) -> Path:
        return self.directory(bucket_id) / remote

    def put(
        self, remote: str, data: bytes | str | dict, bucket_id: str = BUCKET_ID
    ) -> None:
        if isinstance(data, dict):
            data = json.dumps(data)
        target = self.path(remote, bucket_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_bytes(data.encode() if isinstance(data, str) else data)
        os.replace(temporary, target)

    def read(self, remote: str, bucket_id: str = BUCKET_ID) -> bytes:
        return self.path(remote, bucket_id).read_bytes()

    def read_json(self, remote: str, bucket_id: str = BUCKET_ID) -> Any:
        return json.loads(self.read(remote, bucket_id))

    def files(self, bucket_id: str = BUCKET_ID) -> list[str]:
        base = self.directory(bucket_id)
        return sorted(
            path.relative_to(base).as_posix()
            for path in base.rglob("*")
            if path.is_file()
        )

    def bucket(self, bucket_id: str = BUCKET_ID, *, token: str | bool = False):
        from scarf.cytebase._storage import Bucket

        bucket = Bucket(bucket_id, token=token)
        bucket.root = str(self.directory(bucket_id))
        return bucket

    def _run_hooks(self, name: str, *args: Any) -> None:
        for hook in self.hooks.get(name, []):
            hook(*args)

    def list_bucket_tree(
        self,
        bucket_id: str,
        prefix: str | None = None,
        *,
        recursive: bool | None = None,
        token: str | bool | None = None,
    ) -> list[BucketFile | BucketFolder]:
        self.calls.append(("list_bucket_tree", bucket_id, prefix, recursive, token))
        self._run_hooks("list_bucket_tree", bucket_id, prefix)
        base = self.directory(bucket_id)
        if recursive:
            # Object stores match prefixes as strings, including sibling names.
            entries = [
                path
                for path in sorted(base.rglob("*"))
                if path.relative_to(base).as_posix().startswith(prefix or "")
            ]
        else:
            start = base / prefix.rstrip("/") if prefix else base
            entries = sorted(start.iterdir()) if start.is_dir() else []
        if not entries:
            raise EntryNotFoundError(f"No bucket entries under {prefix}")
        return [
            bucket_file(path.relative_to(base).as_posix(), path.stat().st_size)
            if path.is_file()
            else bucket_folder(path.relative_to(base).as_posix())
            for path in entries
        ]

    def download_bucket_files(
        self,
        bucket_id: str,
        files: list[tuple[str, Path]],
        *,
        raise_on_missing_files: bool = False,
        token: str | bool | None = None,
    ) -> None:
        remotes = [remote for remote, _ in files]
        self.calls.append(("download_bucket_files", bucket_id, remotes, token))
        for remote in remotes:
            self._run_hooks("download_bucket_files", bucket_id, remote)
        missing = [
            remote for remote in remotes if not self.path(remote, bucket_id).is_file()
        ]
        if missing and raise_on_missing_files:
            raise EntryNotFoundError(f"Missing bucket files: {', '.join(missing)}")
        for remote, destination in files:
            source = self.path(remote, bucket_id)
            if source.is_file():
                Path(destination).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)

    def batch_bucket_files(
        self,
        bucket_id: str,
        *,
        add: list[tuple[bytes | Path, str]] | None = None,
        copy: list | None = None,
        delete: list[str] | None = None,
        token: str | bool | None = None,
    ) -> None:
        added = [remote for _, remote in add or []]
        self.calls.append(
            ("batch_bucket_files", bucket_id, added, list(delete or []), token)
        )
        self._run_hooks("batch_bucket_files", bucket_id, added, list(delete or []))
        for source, remote in add or []:
            data = source if isinstance(source, bytes) else Path(source).read_bytes()
            self.put(remote, data, bucket_id)
        for remote in delete or []:
            self.path(remote, bucket_id).unlink(missing_ok=True)

    def sync_bucket(
        self,
        source: str,
        dest: str,
        *,
        delete: bool = False,
        token: str | bool | None = None,
        **kwargs: Any,
    ) -> None:
        self.calls.append(("sync_bucket", source, dest, delete, token))
        self._run_hooks("sync_bucket", source, dest)
        if dest.startswith("hf://buckets/"):
            namespace, name, *rest = dest.removeprefix("hf://buckets/").split("/")
            target = self.directory(f"{namespace}/{name}").joinpath(*rest)
        else:
            target = Path(dest)
        shutil.copytree(source, target, dirs_exist_ok=True)


@pytest.fixture
def fake_hub(cytebase_offline, monkeypatch, tmp_path) -> FakeHub:
    """Route every Hugging Face bucket call through local directories."""
    from scarf.cytebase import _storage

    hub = FakeHub(tmp_path / "hub")
    for name in (
        "list_bucket_tree",
        "download_bucket_files",
        "batch_bucket_files",
        "sync_bucket",
    ):
        monkeypatch.setattr(_storage, name, getattr(hub, name))
    original_init = _storage.Bucket.__init__

    @functools.wraps(original_init)
    def local_root(bucket, *args: Any, **kwargs: Any) -> None:
        original_init(bucket, *args, **kwargs)
        bucket.root = str(hub.directory(bucket.bucket_id))

    monkeypatch.setattr(_storage.Bucket, "__init__", local_root)
    try:
        from scarf.cytebase.pipeline import catalog
    except ImportError:  # The pipeline needs the cytebase dependency group.
        return hub
    monkeypatch.setattr(catalog, "list_bucket_tree", hub.list_bucket_tree)
    return hub


def cellxgene_dataset(**overrides: Any) -> dict[str, Any]:
    """A CELLxGENE curation API dataset that registration selects."""
    dataset = {
        "dataset_id": DATASET_ID,
        "dataset_version_id": VERSION_ID,
        "title": "Healthy lung scRNA-seq atlas",
        "citation": CITATION,
        "schema_version": "5.3.0",
        "cell_count": 6,
        "primary_cell_count": 4,
        "is_primary_data": [True, False],
        "feature_count": 5,
        "explorer_url": "https://cellxgene.cziscience.com/e/lung.cxg/",
        "assay": [{"label": "10x 3' v3", "ontology_term_id": "EFO:0009922"}],
        "organism": [{"label": "Homo sapiens", "ontology_term_id": "NCBITaxon:9606"}],
        "tissue": [{"label": "lung", "ontology_term_id": "UBERON:0002048"}],
        "disease": [{"label": "normal", "ontology_term_id": "PATO:0000461"}],
        "cell_type": [
            {"label": "T cell", "ontology_term_id": "CL:0000084"},
            {"label": "B cell", "ontology_term_id": "CL:0000236"},
        ],
        "sex": [{"label": "female", "ontology_term_id": "PATO:0000383"}],
        "development_stage": [
            {"label": "adult stage", "ontology_term_id": "HsapDv:0000087"}
        ],
        "suspension_type": ["cell"],
        "assets": [{"filetype": "H5AD", "url": SOURCE_URL, "filesize": 1234}],
    }
    return dataset | overrides


def cellxgene_collection(
    datasets: list[Any] | None = None, **overrides: Any
) -> dict[str, Any]:
    """A CELLxGENE curation API collection holding ``datasets``."""
    collection = {
        "collection_id": COLLECTION_ID,
        "name": "Human lung atlas",
        "description": "Single-cell profiles of healthy human lung.",
        "doi": "10.1000/lung",
        "collection_url": f"https://cellxgene.cziscience.com/collections/{COLLECTION_ID}",
        "consortia": ["Lung Network"],
        "publisher_metadata": {
            "authors": [{"family": "Smith", "given": "Ada"}, {"name": "Lung Team"}],
            "published_year": 2024,
            "journal": "Lung Journal",
        },
        "datasets": [cellxgene_dataset()] if datasets is None else datasets,
    }
    return collection | overrides


def _terms(*pairs: tuple[str | None, str]) -> list[dict[str, Any]]:
    return [{"termId": term_id, "label": label} for term_id, label in pairs]


def dataset_record(**overrides: Any) -> dict[str, Any]:
    """JSON for a registered ``DatasetRecord`` matching ``cellxgene_dataset()``."""
    record = {
        "cytebaseId": CYTEBASE_ID,
        "datasetId": DATASET_ID,
        "collectionId": COLLECTION_ID,
        "latestVersionId": VERSION_ID,
        "processedVersionId": None,
        "versions": [{"datasetVersionId": VERSION_ID, "seenAt": NOW}],
        "title": "Healthy lung scRNA-seq atlas",
        "citation": CITATION,
        "doi": "10.1000/lung",
        "firstAuthor": "Smith",
        "year": 2024,
        "facets": {
            "organism": _terms(("NCBITaxon:9606", "Homo sapiens")),
            "assay": _terms(("EFO:0009922", "10x 3' v3")),
            "tissue": _terms(("UBERON:0002048", "lung")),
            "disease": _terms(("PATO:0000461", "normal")),
            "cell_type": _terms(("CL:0000236", "B cell"), ("CL:0000084", "T cell")),
            "sex": _terms(("PATO:0000383", "female")),
            "development_stage": _terms(("HsapDv:0000087", "adult stage")),
            "organ": [],
            "suspension_type": _terms((None, "cell")),
        },
        "cellCount": 6,
        "primaryCellCount": 4,
        "nGenes": 5,
        "schemaVersion": "5.3.0",
        "status": "registered",
        "sourceUrl": SOURCE_URL,
        "sourceBytes": 1234,
        "cellxgeneUrl": f"https://cellxgene.cziscience.com/collections/{COLLECTION_ID}",
        "explorerUrl": "https://cellxgene.cziscience.com/e/lung.cxg/",
        "registeredAt": NOW,
        "updatedAt": NOW,
        "pipelineVersion": PIPELINE_VERSION,
    }
    return record | overrides


def _write_matrix(parent: Any, name: str, values: np.ndarray, encoding: str) -> None:
    if encoding == "dense":
        parent.create_dataset(name, data=values)
        return
    from scipy.sparse import csc_matrix, csr_matrix

    matrix = csr_matrix(values) if encoding == "csr" else csc_matrix(values)
    group = parent.create_group(name)
    group.attrs["encoding-type"] = f"{encoding}_matrix"
    group.attrs["shape"] = values.shape
    group.create_dataset("data", data=matrix.data)
    group.create_dataset("indices", data=matrix.indices)
    group.create_dataset("indptr", data=matrix.indptr)


def write_categorical(parent: Any, name: str, values: list[str | None]) -> None:
    categories = sorted({value for value in values if value is not None})
    column = parent.create_group(name)
    column.attrs["encoding-type"] = "categorical"
    column.create_dataset(
        "codes",
        data=np.array(
            [-1 if value is None else categories.index(value) for value in values],
            dtype=np.int8,
        ),
    )
    column.create_dataset(
        "categories", data=np.array([value.encode() for value in categories])
    )


def _write_names(parent: Any, name: str, values: list[str]) -> None:
    parent.create_dataset(name, data=np.array([value.encode() for value in values]))


def write_h5ad(
    path: Path,
    counts: np.ndarray = COUNTS,
    *,
    encoding: str = "csr",
    raw_counts: np.ndarray | None = None,
    layers: dict[str, np.ndarray] | None = None,
    annotations: bool = True,
    umap: bool = True,
    uns: bool = True,
) -> Path:
    """Write a small CELLxGENE-style H5AD file.

    ``counts`` goes to ``X``. With ``raw_counts``, ``raw/X`` and ``raw/var`` are
    added using as many genes as ``raw_counts`` has columns.
    """
    import h5py

    n_cells, n_genes = counts.shape
    with h5py.File(path, "w") as h5:
        _write_matrix(h5, "X", counts, encoding)
        obs = h5.create_group("obs")
        _write_names(obs, "_index", [f"cell{i}" for i in range(n_cells)])
        if annotations:
            primary = (PRIMARY * n_cells)[:n_cells]
            obs.create_dataset("is_primary_data", data=np.array(primary, dtype=bool))
            write_categorical(obs, "cell_type", (CELL_TYPES * n_cells)[:n_cells])
            write_categorical(obs, "donor_id", (DONORS * n_cells)[:n_cells])
            obs.create_dataset("n_genes", data=np.arange(n_cells, dtype=np.int64))
        var = h5.create_group("var")
        _write_names(var, "_index", [f"ENSG{i:011d}" for i in range(n_genes)])
        _write_names(var, "feature_name", (GENES * n_genes)[:n_genes])
        if raw_counts is not None:
            raw_genes = raw_counts.shape[1]
            raw = h5.create_group("raw")
            _write_matrix(raw, "X", raw_counts, encoding)
            raw_var = raw.create_group("var")
            _write_names(raw_var, "_index", [f"ENSG{i:011d}" for i in range(raw_genes)])
            _write_names(
                raw_var, "feature_name", [f"GENE{i}" for i in range(raw_genes)]
            )
        for layer, values in (layers or {}).items():
            _write_matrix(h5.require_group("layers"), layer, values, encoding)
        if umap:
            h5.create_group("obsm").create_dataset(
                "X_umap", data=np.resize(UMAP, (n_cells, 2))
            )
        if uns:
            group = h5.create_group("uns")
            group.create_dataset("title", data="Healthy lung scRNA-seq atlas")
            group.create_dataset("schema_version", data="5.3.0")
            group.create_dataset("citation", data=CITATION)
    return path


def source_details(path: Path) -> tuple[int, str]:
    data = Path(path).read_bytes()
    return len(data), hashlib.sha256(data).hexdigest()


def full_manifest(path: Path, raw_data_location: str | None = None) -> dict[str, Any]:
    """The complete manifest ``convert_local`` requires for ``path``."""
    from scarf.cytebase.pipeline.build import inspect_file

    size, checksum = source_details(path)
    return inspect_file(path, raw_data_location)["manifest"] | {
        "collectionId": COLLECTION_ID,
        "datasetId": DATASET_ID,
        "datasetVersionId": VERSION_ID,
        "sourceUrl": SOURCE_URL,
        "sourceBytes": size,
        "sourceSha256": checksum,
        "metadataSource": "curation_api",
        "ingestedAt": NOW,
        "pipelineVersion": PIPELINE_VERSION,
    }


@dataclass(frozen=True)
class CytebaseBuild:
    """One real ``build_local`` run; accessors return copies that tests may mutate."""

    source: Path
    store: Path
    size: int
    sha256: str
    record_json: dict[str, Any]
    manifest_json: dict[str, Any]
    converted_json: dict[str, Any]

    def record(self):
        from scarf.cytebase.pipeline.models import DatasetRecord

        return DatasetRecord.model_validate(copy.deepcopy(self.record_json))

    def manifest(self):
        from scarf.cytebase.pipeline.models import Manifest

        return Manifest.model_validate(copy.deepcopy(self.manifest_json))

    def converted(self) -> dict[str, Any]:
        return copy.deepcopy(self.converted_json)


@pytest.fixture(scope="session")
def cytebase_build(tmp_path_factory) -> CytebaseBuild:
    """Inspect and convert the default H5AD once per worker."""
    from scarf.cytebase.pipeline.build import build_local
    from scarf.cytebase.pipeline.models import DatasetRecord

    root = tmp_path_factory.mktemp("cytebase-build")
    source = write_h5ad(root / "source.h5ad")
    size, checksum = source_details(source)
    record = DatasetRecord.model_validate(dataset_record())
    manifest, converted = build_local(
        record, source, root / "data.zarr", cellxgene_dataset(), size, checksum, noop
    )
    assert converted["status"] == "done", converted
    return CytebaseBuild(
        source=source,
        store=root / "data.zarr",
        size=size,
        sha256=checksum,
        record_json=record.model_dump(mode="json"),
        manifest_json=manifest.model_dump(mode="json"),
        converted_json=converted,
    )


@dataclass(frozen=True)
class ReadyDataset:
    bucket: "Bucket"
    cytebase_id: str
    record: dict[str, Any]
    store: Path


@pytest.fixture
def ready_dataset(fake_hub, cytebase_build) -> ReadyDataset:
    """Publish the shared build into the hub with the real ``publish_store``."""
    from scarf.cytebase.pipeline.build import publish_store

    bucket = fake_hub.bucket()
    result = publish_store(
        cytebase_build.record(),
        {},
        bucket,
        cytebase_build.store,
        cytebase_build.manifest(),
        cytebase_build.converted(),
        noop,
        noop,
    )
    assert result == {"outcome": "succeeded"}
    record = bucket.read_json(f"datasets/{CYTEBASE_ID}/dataset.json")
    return ReadyDataset(bucket, CYTEBASE_ID, record, Path(record["zarrUri"]))


def publish_catalog_rows(
    hub: FakeHub,
    records: list[dict[str, Any]],
    collections: list[dict[str, Any]] | None = None,
    *,
    bucket_id: str = BUCKET_ID,
) -> str:
    """Write a real catalog database for ``records`` into the hub; return its hash."""
    from scarf.cytebase.pipeline.catalog import _dataset_row, _term_rows, _write_catalog
    from scarf.cytebase.pipeline.models import DatasetRecord

    models = [DatasetRecord.model_validate(record) for record in records]
    with TemporaryDirectory() as directory:
        paths = _write_catalog(
            {
                "datasets": [_dataset_row(model) for model in models],
                "collections": collections or [],
                "dataset_terms": _term_rows(models),
            },
            Path(directory),
        )
        hub.put(
            "catalog/cytebase.duckdb", paths["cytebase.duckdb"].read_bytes(), bucket_id
        )
        sidecar = paths["cytebase.duckdb.sha256"].read_text()
    hub.put("catalog/cytebase.duckdb.sha256", sidecar, bucket_id)
    return sidecar.split()[0]


# Modal harness: every Modal-decorated function runs in-process through .local().
CALL_ID: ContextVar[str | None] = ContextVar("cytebase_test_call_id", default=None)


class FakeProgressStore:
    """In-memory replacement for the deployed ``modal.Dict``."""

    def __init__(self, fail: bool = False) -> None:
        self.values: dict[str, Any] = {}
        self.puts: list[tuple[str, Any]] = []
        self.fail = fail

    def put(self, key: str, value: Any) -> None:
        if self.fail:
            raise RuntimeError("progress store unavailable")
        self.puts.append((key, copy.deepcopy(value)))
        self.values[key] = copy.deepcopy(value)

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)


class FakeFunction:
    """Implements ``spawn.aio`` and ``get.aio`` by running ``target`` in a thread.

    The child's call ID is visible through ``modal.current_function_call_id``
    while it runs, so ownership checks behave as they do in Modal.
    """

    def __init__(
        self,
        target: Callable[..., Any],
        name: str,
        *,
        spawn_error: BaseException | None = None,
        get_errors: list[BaseException | None] | None = None,
    ) -> None:
        self.target = target
        self.name = name
        self.spawned: list[tuple[Any, ...]] = []
        self._spawn_error = spawn_error
        self._get_errors = list(get_errors or [])
        self.spawn = SimpleNamespace(aio=self._spawn)

    async def _spawn(self, *args: Any) -> SimpleNamespace:
        if self._spawn_error is not None:
            raise self._spawn_error
        self.spawned.append(args)
        object_id = f"fc-{self.name}-{len(self.spawned)}"
        error = self._get_errors.pop(0) if self._get_errors else None

        async def get() -> Any:
            if error is not None:
                raise error
            token = CALL_ID.set(object_id)
            try:
                return await asyncio.to_thread(self.target, *args)
            finally:
                CALL_ID.reset(token)

        return SimpleNamespace(object_id=object_id, get=SimpleNamespace(aio=get))


@pytest.fixture(scope="module")
def pipeline_app():
    """Import the Modal app module only in tests that need it."""
    pytest.importorskip("modal")
    pytest.importorskip("fastapi")
    from scarf.cytebase.pipeline import app

    return app


@dataclass
class ModalHarness:
    app: Any
    hub: FakeHub
    bucket: "Bucket"
    progress_store: FakeProgressStore
    build_catalog: FakeFunction
    process_dataset: FakeFunction

    def run(self, action: str, request: dict[str, Any], *, run_id: str) -> Any:
        token = CALL_ID.set(run_id)
        try:
            return asyncio.run(self.app.run_pipeline.local(action, request))
        finally:
            CALL_ID.reset(token)


@pytest.fixture
def modal_harness(pipeline_app, fake_hub, monkeypatch) -> ModalHarness:
    """Run the orchestrator offline with real workers and a local bucket."""
    import modal

    monkeypatch.setenv("HF_TOKEN", "hf_offlineTestToken")
    monkeypatch.setenv("CYTEBASE_BUCKET", BUCKET_ID)
    monkeypatch.setenv("CYTEBASE_PIPELINE_VERSION", PIPELINE_VERSION)
    monkeypatch.setattr(modal, "current_function_call_id", CALL_ID.get)
    store = FakeProgressStore()
    monkeypatch.setattr(pipeline_app, "progress_store", store)
    build_catalog = FakeFunction(pipeline_app.build_catalog.local, "catalog")
    process_dataset = FakeFunction(pipeline_app.process_dataset.local, "process")
    monkeypatch.setattr(pipeline_app, "build_catalog", build_catalog)
    monkeypatch.setattr(pipeline_app, "process_dataset", process_dataset)
    return ModalHarness(
        pipeline_app,
        fake_hub,
        fake_hub.bucket(),
        store,
        build_catalog,
        process_dataset,
    )
