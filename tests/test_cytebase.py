import io
import tarfile

from huggingface_hub import BucketFile, BucketFolder
import pytest


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
    return BucketFolder(
        type="directory",
        path=path,
        uploadedAt=None,
    )


def test_list_repositories_is_sorted_and_anonymous(monkeypatch):
    from scarf import cytebase

    calls = []

    def fake_list_bucket_tree(bucket_id, prefix=None, *, recursive=None, token=None):
        calls.append((bucket_id, prefix, recursive, token))
        return [
            bucket_folder("scarf_docs"),
            bucket_file("README.md", 3),
            bucket_folder("cellxgene"),
        ]

    monkeypatch.setattr(cytebase, "list_bucket_tree", fake_list_bucket_tree)

    assert cytebase.list_repositories() == ["cellxgene", "scarf_docs"]
    assert calls == [("Nygen/cytebase", None, False, False)]


def test_connect_returns_repository(monkeypatch):
    from scarf import cytebase

    monkeypatch.setattr(
        cytebase,
        "list_repositories",
        lambda: ["cellxgene", "scarf_docs"],
    )

    repository = cytebase.connect("scarf_docs")

    assert repository == cytebase.Repository("scarf_docs")


def test_connect_rejects_unknown_repository(monkeypatch):
    from scarf import cytebase

    monkeypatch.setattr(cytebase, "list_repositories", lambda: ["scarf_docs"])

    with pytest.raises(KeyError, match="Available repositories"):
        cytebase.connect("missing")


@pytest.mark.parametrize(
    "name",
    ["", ".", "..", "../outside", "/outside", r"..\outside"],
)
def test_connect_rejects_invalid_repository_name(name):
    from scarf import cytebase

    with pytest.raises(ValueError, match="Invalid repository name"):
        cytebase.Repository(name)


def test_repository_lists_datasets(monkeypatch):
    from scarf import cytebase

    calls = []

    def fake_list_bucket_tree(bucket_id, prefix=None, *, recursive=None, token=None):
        calls.append((bucket_id, prefix, recursive, token))
        return [
            bucket_folder("scarf_docs/zeta"),
            bucket_file("scarf_docs/index.json", 2),
            bucket_folder("scarf_docs/alpha"),
        ]

    monkeypatch.setattr(cytebase, "list_bucket_tree", fake_list_bucket_tree)

    assert cytebase.Repository("scarf_docs").list_datasets() == ["alpha", "zeta"]
    assert calls == [("Nygen/cytebase", "scarf_docs", False, False)]


def test_repository_lists_relative_files(monkeypatch):
    from scarf import cytebase

    calls = []

    def fake_list_bucket_tree(bucket_id, prefix=None, *, recursive=None, token=None):
        calls.append((bucket_id, prefix, recursive, token))
        return [
            bucket_file("scarf_docs/alpha/matrix.mtx.gz", 4),
            bucket_file("scarf_docs/alphabet/other.bin", 3),
            bucket_file("scarf_docs/alpha/barcodes.tsv.gz", 2),
        ]

    monkeypatch.setattr(cytebase, "list_bucket_tree", fake_list_bucket_tree)

    files = cytebase.Repository("scarf_docs").list_files(
        "alpha",
        recursive=False,
    )

    assert files == ["alpha/barcodes.tsv.gz", "alpha/matrix.mtx.gz"]
    assert calls == [("Nygen/cytebase", "scarf_docs/alpha", False, False)]


def test_repository_downloads_file_anonymously(monkeypatch, tmp_path):
    from scarf import cytebase

    file = bucket_file("scarf_docs/alpha/data.bin", 7)
    monkeypatch.setattr(
        cytebase,
        "_bucket_files",
        lambda repository, path, recursive: [file],
    )
    calls = []

    def fake_download(bucket_id, files, *, raise_on_missing_files=False, token=None):
        calls.append((bucket_id, files, raise_on_missing_files, token))
        files[0][1].write_bytes(b"payload")

    monkeypatch.setattr(cytebase, "download_bucket_files", fake_download)

    downloaded = cytebase.Repository("scarf_docs").download(
        "alpha/data.bin",
        tmp_path,
    )

    destination = tmp_path / "alpha" / "data.bin"
    assert downloaded == [destination]
    assert destination.read_bytes() == b"payload"
    assert calls[0][0] == "Nygen/cytebase"
    assert calls[0][1][0][0] is file
    assert calls[0][2:] == (False, False)
    assert not any(
        path.name.startswith(".cytebase-download-") for path in tmp_path.iterdir()
    )


def test_download_dataset_excludes_zarr_archive_by_default(monkeypatch, tmp_path):
    from scarf import cytebase

    source = bucket_file("scarf_docs/alpha/data.h5", 6)
    archive = bucket_file("scarf_docs/alpha/data.zarr.tar.gz", 12)
    monkeypatch.setattr(
        cytebase,
        "_bucket_files",
        lambda repository, path, recursive: [archive, source],
    )
    requested = []

    def fake_download(bucket_id, files, *, raise_on_missing_files=False, token=None):
        requested.extend(file.path for file, _ in files)
        for _, destination in files:
            destination.write_bytes(b"source")

    monkeypatch.setattr(cytebase, "download_bucket_files", fake_download)

    dataset_path = cytebase.Repository("scarf_docs").download_dataset(
        "alpha",
        tmp_path,
    )

    assert dataset_path == tmp_path / "alpha"
    assert requested == ["scarf_docs/alpha/data.h5"]
    assert (dataset_path / "data.h5").read_bytes() == b"source"
    assert not (dataset_path / "data.zarr.tar.gz").exists()


def test_download_dataset_copies_local_catalog_zarr(monkeypatch, tmp_path):
    from scarf import cytebase

    local_root = tmp_path / "catalog"
    store = local_root / "alpha" / "data.zarr"
    store.mkdir(parents=True)
    (store / "zarr.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv(cytebase._LOCAL_CATALOG_ENV, str(local_root))

    def fail_bucket(*_args, **_kwargs):
        raise AssertionError("local catalog should not list the remote bucket")

    monkeypatch.setattr(cytebase, "_bucket_files", fail_bucket)

    destination = tmp_path / "downloads"
    dataset_path = cytebase.Repository("scarf_docs").download_dataset(
        "alpha",
        destination,
        zarr=True,
    )

    copied = dataset_path / "data.zarr" / "zarr.json"
    assert dataset_path == destination / "alpha"
    assert copied.read_text(encoding="utf-8") == "{}"
    assert (store / "zarr.json").read_text(encoding="utf-8") == "{}"


def test_download_dataset_selects_and_extracts_zarr_archive(monkeypatch, tmp_path):
    from scarf import cytebase

    archive_buffer = io.BytesIO()
    payload = b'{"zarr_format": 3, "node_type": "group"}'
    with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
        member = tarfile.TarInfo("data.zarr/zarr.json")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    archive_bytes = archive_buffer.getvalue()
    source = bucket_file("scarf_docs/alpha/data.h5", 6)
    zarr_archive = bucket_file(
        "scarf_docs/alpha/data.zarr.tar.gz",
        len(archive_bytes),
    )
    monkeypatch.setattr(
        cytebase,
        "_bucket_files",
        lambda repository, path, recursive: [source, zarr_archive],
    )

    def fake_download(bucket_id, files, *, raise_on_missing_files=False, token=None):
        assert [file.path for file, _ in files] == ["scarf_docs/alpha/data.zarr.tar.gz"]
        files[0][1].write_bytes(archive_bytes)

    monkeypatch.setattr(cytebase, "download_bucket_files", fake_download)

    dataset_path = cytebase.Repository("scarf_docs").download_dataset(
        "alpha",
        tmp_path,
        zarr=True,
    )

    assert (dataset_path / "data.zarr.tar.gz").read_bytes() == archive_bytes
    assert (dataset_path / "data.zarr" / "zarr.json").read_bytes() == payload


def test_download_dataset_reports_missing_zarr(monkeypatch, tmp_path):
    from scarf import cytebase

    monkeypatch.setattr(
        cytebase,
        "_bucket_files",
        lambda repository, path, recursive: [
            bucket_file("scarf_docs/alpha/data.h5", 6)
        ],
    )

    with pytest.raises(FileNotFoundError, match="No Zarr archive"):
        cytebase.Repository("scarf_docs").download_dataset(
            "alpha",
            tmp_path,
            zarr=True,
        )


def test_download_dataset_reports_unknown_name(monkeypatch, tmp_path):
    from scarf import cytebase

    monkeypatch.setattr(cytebase, "_bucket_files", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        cytebase.Repository,
        "list_datasets",
        lambda self: ["alpha", "beta"],
    )

    with pytest.raises(KeyError, match="Available datasets"):
        cytebase.Repository("scarf_docs").download_dataset("missing", tmp_path)


@pytest.mark.parametrize(
    "path",
    ["", ".", "..", "../outside", "/outside", r"..\outside", "a//b"],
)
def test_repository_rejects_invalid_remote_paths(path, tmp_path):
    from scarf import cytebase

    with pytest.raises(ValueError, match="Invalid download path"):
        cytebase.Repository("scarf_docs").download(path, tmp_path)


def test_download_preserves_existing_file_when_transfer_fails(monkeypatch, tmp_path):
    from scarf import cytebase

    file = bucket_file("scarf_docs/alpha/data.bin", 3)
    monkeypatch.setattr(
        cytebase,
        "_bucket_files",
        lambda repository, path, recursive: [file],
    )
    destination = tmp_path / "alpha" / "data.bin"
    destination.parent.mkdir()
    destination.write_bytes(b"existing")

    def fail_download(bucket_id, files, *, raise_on_missing_files=False, token=None):
        files[0][1].write_bytes(b"new")
        raise RuntimeError("transfer failed")

    monkeypatch.setattr(cytebase, "download_bucket_files", fail_download)

    with pytest.raises(RuntimeError, match="transfer failed"):
        cytebase.Repository("scarf_docs").download("alpha/data.bin", tmp_path)

    assert destination.read_bytes() == b"existing"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["alpha"]


def test_download_rejects_size_mismatch_and_preserves_existing(
    monkeypatch,
    tmp_path,
):
    from scarf import cytebase

    file = bucket_file("scarf_docs/alpha/data.bin", 4)
    monkeypatch.setattr(
        cytebase,
        "_bucket_files",
        lambda repository, path, recursive: [file],
    )
    destination = tmp_path / "alpha" / "data.bin"
    destination.parent.mkdir()
    destination.write_bytes(b"existing")

    def incomplete_download(
        bucket_id,
        files,
        *,
        raise_on_missing_files=False,
        token=None,
    ):
        files[0][1].write_bytes(b"new")

    monkeypatch.setattr(cytebase, "download_bucket_files", incomplete_download)

    with pytest.raises(OSError, match="Downloaded 3 bytes"):
        cytebase.Repository("scarf_docs").download("alpha/data.bin", tmp_path)

    assert destination.read_bytes() == b"existing"


def test_download_rejects_unsafe_tar_member(monkeypatch, tmp_path):
    from scarf import cytebase

    archive_buffer = io.BytesIO()
    payload = b"unsafe"
    with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
        member = tarfile.TarInfo("../outside.txt")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    archive_bytes = archive_buffer.getvalue()
    file = bucket_file(
        "scarf_docs/alpha/data.tar.gz",
        len(archive_bytes),
    )
    monkeypatch.setattr(
        cytebase,
        "_bucket_files",
        lambda repository, path, recursive: [file],
    )

    def fake_download(bucket_id, files, *, raise_on_missing_files=False, token=None):
        files[0][1].write_bytes(archive_bytes)

    monkeypatch.setattr(cytebase, "download_bucket_files", fake_download)

    with pytest.raises(ValueError, match="Unsafe tar archive member"):
        cytebase.Repository("scarf_docs").download("alpha/data.tar.gz", tmp_path)

    assert not (tmp_path / "outside.txt").exists()


def test_open_zarr_uses_anonymous_read_only_store(monkeypatch):
    from scarf import cytebase
    import zarr
    from zarr.storage import FsspecStore

    calls = []
    store = object()
    group = object()

    def fake_from_url(url, storage_options=None, read_only=False):
        calls.append(("store", url, storage_options, read_only))
        return store

    def fake_open_group(*, store, mode):
        calls.append(("group", store, mode))
        return group

    monkeypatch.setattr(FsspecStore, "from_url", fake_from_url)
    monkeypatch.setattr(zarr, "open_group", fake_open_group)

    result = cytebase.Repository("cellxgene").open_zarr("atlas/data.zarr")

    assert result is group
    assert calls == [
        (
            "store",
            "hf://buckets/Nygen/cytebase/cellxgene/atlas/data.zarr",
            {"token": False},
            True,
        ),
        ("group", store, "r"),
    ]


@pytest.mark.integration
def test_live_bucket_catalog_is_public():
    from scarf import cytebase

    assert "scarf_docs" in cytebase.list_repositories()
    repository = cytebase.connect("scarf_docs")
    datasets = set(repository.list_datasets())
    assert {
        "annotations",
        "bastidas-ponce_4K_pancreas-d15_rnaseq",
        "kang_14K_ifnb-pbmc_rnaseq",
        "kang_14K_ifnb-pbmc_rnaseq_legacy_master",
        "kang_15K_pbmc_rnaseq",
        "kang_15K_pbmc_rnaseq_legacy_master",
        "kang_29K_ctrl-ifnb_pbmc_rnaseq",
        "swanson_7K_pbmc_teaseq",
        "tenx_10K_pbmc-v1_atacseq",
        "tenx_5K_pbmc_rnaseq",
        "tenx_8K_pbmc_citeseq",
    } <= datasets
    assert "tenx_5K_pbmc_rnaseq_legacy_master" in datasets
    files = repository.list_files("tenx_5K_pbmc_rnaseq")
    assert {
        "tenx_5K_pbmc_rnaseq/data.h5",
        "tenx_5K_pbmc_rnaseq/data.zarr.tar.gz",
        "tenx_5K_pbmc_rnaseq/manifest.json",
    } <= set(files)
    assert any(name.startswith("tenx_5K_pbmc_rnaseq/data.zarr/") for name in files)


@pytest.mark.integration
def test_live_zarr_archive_download(tmp_path, monkeypatch):
    from scarf import cytebase

    monkeypatch.delenv("SCARF_CYTEBASE_LOCAL", raising=False)
    dataset_path = cytebase.connect("scarf_docs").download_dataset(
        "tenx_5K_pbmc_rnaseq",
        tmp_path,
        zarr=True,
    )

    assert (dataset_path / "data.zarr.tar.gz").is_file()
    assert (dataset_path / "data.zarr").is_dir()
