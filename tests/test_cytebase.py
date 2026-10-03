import io
import os
import tarfile
from pathlib import Path

import pytest

from tests.fixtures_cytebase import bucket_file, bucket_folder


@pytest.fixture(autouse=True)
def _offline_unless_live(request):
    """Keep every test except the explicit live checks off the network."""
    if request.node.get_closest_marker("integration") is None:
        request.getfixturevalue("cytebase_offline")


def _write_tar(path: Path, members: list[tuple[str, str, bytes | str | None]]) -> Path:
    """Write a gzip tar archive from ``(name, kind, payload)`` member specs.

    ``kind`` is ``file`` (payload is the content), ``symlink`` or ``hardlink``
    (payload is the link name), ``fifo``, or ``chardev``.
    """
    with tarfile.open(path, "w:gz") as archive:
        for name, kind, payload in members:
            member = tarfile.TarInfo(name)
            if kind == "file":
                assert isinstance(payload, bytes)
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
                continue
            member.type = {
                "symlink": tarfile.SYMTYPE,
                "hardlink": tarfile.LNKTYPE,
                "fifo": tarfile.FIFOTYPE,
                "chardev": tarfile.CHRTYPE,
            }[kind]
            if payload is not None:
                member.linkname = str(payload)
            archive.addfile(member)
    return path


def _tree(root: Path) -> list[str]:
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))


@pytest.mark.parametrize("member_name", ["notes.txt", "other.zarr/zarr.json"])
def test_archive_cannot_replace_unrelated_siblings(tmp_path, member_name):
    from scarf.cytebase import _extract_and_replace

    notes = tmp_path / "notes.txt"
    notes.write_text("keep this")
    output = tmp_path / "data.zarr"
    output.mkdir()
    (output / "zarr.json").write_text("original")
    staged = tmp_path / "download"
    with tarfile.open(staged, "w:gz") as archive:
        for name in ["data.zarr/zarr.json", member_name]:
            member = tarfile.TarInfo(name)
            member.size = 7
            archive.addfile(member, io.BytesIO(b"replace"))
    destination = tmp_path / "data.zarr.tar.gz"
    with pytest.raises(ValueError, match="must contain only the directory"):
        _extract_and_replace(staged, destination)
    assert notes.read_text() == "keep this"
    assert (output / "zarr.json").read_text() == "original"
    assert not destination.exists()


def test_archive_replaces_only_its_named_output_directory(tmp_path):
    from scarf.cytebase import _extract_and_replace

    output = tmp_path / "data.zarr"
    output.mkdir()
    (output / "old").write_text("old data")
    notes = tmp_path / "notes.txt"
    notes.write_text("keep this")
    staged = tmp_path / "download"
    with tarfile.open(staged, "w:gz") as archive:
        member = tarfile.TarInfo("data.zarr/zarr.json")
        member.size = 2
        archive.addfile(member, io.BytesIO(b"{}"))
    destination = tmp_path / "data.zarr.tar.gz"
    _extract_and_replace(staged, destination)
    assert (output / "zarr.json").read_text() == "{}"
    assert not (output / "old").exists()
    assert notes.read_text() == "keep this"
    assert destination.is_file()


def test_archive_output_directory_cannot_be_a_symlink(tmp_path):
    from scarf.cytebase import _extract_and_replace

    staged = tmp_path / "download"
    with tarfile.open(staged, "w:gz") as archive:
        member = tarfile.TarInfo("data.zarr")
        member.type = tarfile.SYMTYPE
        member.linkname = "."
        archive.addfile(member)
    with pytest.raises(ValueError, match="must contain only the directory"):
        _extract_and_replace(staged, tmp_path / "data.zarr.tar.gz")
    assert not (tmp_path / "data.zarr").exists()


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

    monkeypatch.setattr(cytebase, "list_repositories", lambda: ["cellxgene", "docs"])

    with pytest.raises(KeyError) as raised:
        cytebase.connect("missing")
    assert raised.value.args[0] == (
        "'missing' is not a Cytebase repository. Available repositories:\n"
        "cellxgene\ndocs"
    )


@pytest.mark.parametrize(
    "name",
    ["", ".", "..", "../outside", "/outside", r"..\outside", "a/b"],
)
def test_connect_rejects_invalid_repository_name(monkeypatch, name):
    from scarf import cytebase

    def unexpected_listing():
        raise AssertionError("an invalid name must be rejected before listing")

    monkeypatch.setattr(cytebase, "list_repositories", unexpected_listing)
    with pytest.raises(ValueError, match="Invalid repository name"):
        cytebase.connect(name)
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
            # Object stores match prefixes as strings, so a sibling repository
            # and deeper folders can appear in a non-recursive listing.
            bucket_folder("scarf_docs_v2"),
            bucket_folder("scarf_docs_v2/beta"),
            bucket_folder("scarf_docs/alpha/raw"),
        ]

    monkeypatch.setattr(cytebase, "list_bucket_tree", fake_list_bucket_tree)

    assert cytebase.Repository("scarf_docs").list_datasets() == ["alpha", "zeta"]
    assert calls == [("Nygen/cytebase", "scarf_docs", False, False)]


def test_repository_file_listing_is_scoped_and_sorted(monkeypatch):
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

    files = cytebase._bucket_files("scarf_docs", "alpha", recursive=False)

    assert [file.path for file in files] == [
        "scarf_docs/alpha/barcodes.tsv.gz",
        "scarf_docs/alpha/matrix.mtx.gz",
    ]
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

    downloads = tmp_path / "downloads"
    downloaded = cytebase.Repository("scarf_docs").download(
        "alpha/data.bin",
        downloads,
    )

    destination = downloads / "alpha" / "data.bin"
    assert downloaded == [destination]
    assert destination.read_bytes() == b"payload"
    assert calls[0][0] == "Nygen/cytebase"
    assert calls[0][1][0][0] is file
    assert calls[0][2:] == (False, False)
    # The staging directory is gone and nothing else was written.
    assert _tree(downloads) == ["alpha", "alpha/data.bin"]


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


def test_local_catalog_copy_replaces_only_the_zarr_store(monkeypatch, tmp_path):
    from scarf import cytebase

    store = tmp_path / "catalog" / "alpha" / "data.zarr"
    store.mkdir(parents=True)
    (store / "zarr.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv(cytebase._LOCAL_CATALOG_ENV, str(tmp_path / "catalog"))

    def fail_bucket(*_args, **_kwargs):
        raise AssertionError("local catalog should not list the remote bucket")

    monkeypatch.setattr(cytebase, "_bucket_files", fail_bucket)
    dataset = tmp_path / "downloads" / "alpha"
    (dataset / "data.zarr").mkdir(parents=True)
    (dataset / "data.zarr" / "stale.json").write_text("old", encoding="utf-8")
    (dataset / "data.h5").write_bytes(b"raw counts")

    dataset_path = cytebase.Repository("scarf_docs").download_dataset(
        "alpha",
        tmp_path / "downloads",
        zarr=True,
    )

    assert dataset_path == dataset
    assert sorted(path.name for path in dataset.iterdir()) == ["data.h5", "data.zarr"]
    assert (dataset / "data.h5").read_bytes() == b"raw counts"
    assert sorted(path.name for path in (dataset / "data.zarr").iterdir()) == [
        "zarr.json"
    ]


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

    with pytest.raises(
        FileNotFoundError, match="^No Zarr archive is available for 'alpha'$"
    ):
        cytebase.Repository("scarf_docs").download_dataset(
            "alpha",
            tmp_path / "downloads",
            zarr=True,
        )
    assert not (tmp_path / "downloads").exists()


def test_download_dataset_reports_unknown_name(monkeypatch, tmp_path):
    from scarf import cytebase

    monkeypatch.setattr(cytebase, "_bucket_files", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        cytebase.Repository,
        "list_datasets",
        lambda self: ["alpha", "beta"],
    )

    with pytest.raises(KeyError) as raised:
        cytebase.Repository("scarf_docs").download_dataset("missing", tmp_path)
    assert raised.value.args[0] == (
        "'missing' is not in repository 'scarf_docs'. Available datasets:\nalpha\nbeta"
    )


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
    downloads = tmp_path / "downloads"
    destination = downloads / "alpha" / "data.bin"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"existing")

    def fail_download(bucket_id, files, *, raise_on_missing_files=False, token=None):
        files[0][1].write_bytes(b"new")
        raise RuntimeError("transfer failed")

    monkeypatch.setattr(cytebase, "download_bucket_files", fail_download)

    with pytest.raises(RuntimeError, match="transfer failed"):
        cytebase.Repository("scarf_docs").download("alpha/data.bin", downloads)

    assert destination.read_bytes() == b"existing"
    assert _tree(downloads) == ["alpha", "alpha/data.bin"]


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
    downloads = tmp_path / "downloads"
    destination = downloads / "alpha" / "data.bin"
    destination.parent.mkdir(parents=True)
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

    with pytest.raises(
        OSError,
        match="^Downloaded 3 bytes for 'scarf_docs/alpha/data.bin', expected 4$",
    ):
        cytebase.Repository("scarf_docs").download("alpha/data.bin", downloads)

    assert destination.read_bytes() == b"existing"
    assert _tree(downloads) == ["alpha", "alpha/data.bin"]


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

    downloads = tmp_path / "downloads"
    with pytest.raises(ValueError, match="^Unsafe tar archive member: ../outside.txt$"):
        cytebase.Repository("scarf_docs").download("alpha/data.tar.gz", downloads)

    assert not (tmp_path / "outside.txt").exists()
    assert _tree(downloads) == ["alpha"]


def test_download_of_a_directory_skips_folders_and_sibling_prefixes(
    monkeypatch, tmp_path
):
    from scarf import cytebase

    calls = []

    def fake_list_bucket_tree(bucket_id, prefix=None, *, recursive=None, token=None):
        calls.append((bucket_id, prefix, recursive, token))
        return [
            bucket_folder("scarf_docs/alpha/raw"),
            bucket_file("scarf_docs/alpha/raw/matrix.mtx", 3),
            bucket_file("scarf_docs/alpha/barcodes.tsv", 2),
            # A string prefix match, not a file under alpha/.
            bucket_file("scarf_docs/alphabet/other.bin", 1),
        ]

    requested = []

    def fake_download(bucket_id, files, *, raise_on_missing_files=False, token=None):
        for file, destination in files:
            requested.append(file.path)
            destination.write_bytes(b"x" * file.size)

    monkeypatch.setattr(cytebase, "list_bucket_tree", fake_list_bucket_tree)
    monkeypatch.setattr(cytebase, "download_bucket_files", fake_download)
    downloads = tmp_path / "downloads"

    downloaded = cytebase.Repository("scarf_docs").download("alpha", downloads)

    assert calls == [("Nygen/cytebase", "scarf_docs/alpha", True, False)]
    assert requested == [
        "scarf_docs/alpha/barcodes.tsv",
        "scarf_docs/alpha/raw/matrix.mtx",
    ]
    assert downloaded == [
        downloads / "alpha" / "barcodes.tsv",
        downloads / "alpha" / "raw" / "matrix.mtx",
    ]
    assert (downloads / "alpha" / "raw" / "matrix.mtx").read_bytes() == b"xxx"
    assert _tree(downloads) == [
        "alpha",
        "alpha/barcodes.tsv",
        "alpha/raw",
        "alpha/raw/matrix.mtx",
    ]


def test_download_reports_a_path_without_files(monkeypatch, tmp_path):
    from scarf import cytebase

    monkeypatch.setattr(
        cytebase,
        "list_bucket_tree",
        lambda *args, **kwargs: [bucket_folder("scarf_docs/alpha/empty")],
    )

    with pytest.raises(
        FileNotFoundError, match="^No Cytebase files found at scarf_docs/alpha/empty$"
    ):
        cytebase.Repository("scarf_docs").download("alpha/empty", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_download_rejects_a_listing_that_escapes_the_destination(monkeypatch, tmp_path):
    from scarf import cytebase

    monkeypatch.setattr(
        cytebase,
        "list_bucket_tree",
        lambda *args, **kwargs: [bucket_file("scarf_docs/alpha/../../escape", 1)],
    )
    monkeypatch.setattr(cytebase, "download_bucket_files", _unexpected_download)

    with pytest.raises(
        ValueError, match="^Invalid bucket file path: 'alpha/../../escape'$"
    ):
        cytebase.Repository("scarf_docs").download("alpha", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def _unexpected_download(*_args, **_kwargs):
    raise AssertionError("nothing may be downloaded")


@pytest.mark.parametrize("path", ["other/alpha/data.bin", "scarf_docs_v2/data.bin"])
def test_bucket_file_paths_must_stay_inside_their_repository(path):
    from scarf import cytebase

    # Listings are filtered by the requested prefix first, so this guards the
    # helper rather than a path that Repository methods reach today.
    with pytest.raises(
        ValueError, match=f"^File is outside repository 'scarf_docs': '{path}'$"
    ):
        cytebase._relative_file_path("scarf_docs", bucket_file(path, 1))


def test_download_rejects_files_the_transfer_skipped(monkeypatch, tmp_path):
    from scarf import cytebase

    file = bucket_file("scarf_docs/alpha/data.bin", 3)
    monkeypatch.setattr(
        cytebase,
        "_bucket_files",
        lambda repository, path, recursive: [file],
    )
    downloads = tmp_path / "downloads"
    destination = downloads / "alpha" / "data.bin"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"old")
    # The transfer returns without writing the file it was asked for.
    monkeypatch.setattr(
        cytebase, "download_bucket_files", lambda bucket_id, files, **kwargs: None
    )

    with pytest.raises(
        FileNotFoundError,
        match="^Cytebase did not download 'scarf_docs/alpha/data.bin'$",
    ):
        cytebase.Repository("scarf_docs").download("alpha/data.bin", downloads)

    assert destination.read_bytes() == b"old"
    assert _tree(downloads) == ["alpha", "alpha/data.bin"]


@pytest.mark.parametrize("previous", [True, False], ids=["replace", "first"])
def test_failed_archive_install_restores_the_previous_download(
    monkeypatch, tmp_path, previous
):
    from scarf import cytebase

    dataset = tmp_path / "downloads" / "alpha"
    dataset.mkdir(parents=True)
    if previous:
        (dataset / "data.zarr").mkdir()
        (dataset / "data.zarr" / "old.json").write_text("old store")
        (dataset / "data.zarr.tar.gz").write_bytes(b"old archive")
    before = _tree(dataset)
    archive_bytes = _write_tar(
        tmp_path / "new.tar.gz", [("data.zarr/zarr.json", "file", b"{}")]
    ).read_bytes()
    remote = bucket_file("scarf_docs/alpha/data.zarr.tar.gz", len(archive_bytes))
    monkeypatch.setattr(
        cytebase, "_bucket_files", lambda repository, path, recursive: [remote]
    )

    def fake_download(bucket_id, files, *, raise_on_missing_files=False, token=None):
        files[0][1].write_bytes(archive_bytes)

    monkeypatch.setattr(cytebase, "download_bucket_files", fake_download)
    real_replace = os.replace

    def replace(source, destination):
        # The extracted store is already in place when the archive move fails.
        if ".cytebase-download-" in str(source):
            raise OSError("No space left on device")
        real_replace(source, destination)

    monkeypatch.setattr(cytebase.os, "replace", replace)

    with pytest.raises(OSError, match="^No space left on device$"):
        cytebase.Repository("scarf_docs").download_dataset(
            "alpha", tmp_path / "downloads", zarr=True
        )

    assert _tree(tmp_path / "downloads") == ["alpha", *(f"alpha/{p}" for p in before)]
    if previous:
        assert (dataset / "data.zarr" / "old.json").read_text() == "old store"
        assert (dataset / "data.zarr.tar.gz").read_bytes() == b"old archive"


@pytest.mark.parametrize(
    ("member", "message"),
    [
        pytest.param(
            ("data.zarr/link", "symlink", "/etc/passwd"),
            "Unsafe tar archive link: data.zarr/link",
            id="absolute-symlink",
        ),
        pytest.param(
            ("data.zarr/link", "symlink", "../../outside"),
            "Unsafe tar archive link: data.zarr/link",
            id="parent-symlink",
        ),
        pytest.param(
            ("data.zarr/hard", "hardlink", "../outside"),
            "Unsafe tar archive link: data.zarr/hard",
            id="parent-hardlink",
        ),
        pytest.param(
            ("data.zarr/pipe", "fifo", None),
            "Unsupported tar archive member: data.zarr/pipe",
            id="fifo",
        ),
        pytest.param(
            ("data.zarr/tty", "chardev", None),
            "Unsupported tar archive member: data.zarr/tty",
            id="device",
        ),
    ],
)
def test_archive_rejects_unsafe_links_and_special_members(tmp_path, member, message):
    from scarf.cytebase import _extract_and_replace

    work = tmp_path / "work"
    output = work / "data.zarr"
    output.mkdir(parents=True)
    (output / "zarr.json").write_text("original")
    staged = _write_tar(
        work / "download", [("data.zarr/zarr.json", "file", b"{}"), member]
    )

    with pytest.raises(ValueError, match=f"^{message}$"):
        _extract_and_replace(staged, work / "data.zarr.tar.gz")

    assert (output / "zarr.json").read_text() == "original"
    assert _tree(work) == ["data.zarr", "data.zarr/zarr.json", "download"]


@pytest.mark.parametrize(
    ("member", "message"),
    [
        (("escape/file", "file", b"x"), "Unsafe tar archive member: escape/file"),
        (
            ("data/hard", "hardlink", "escape/file"),
            "Unsafe tar archive link: data/hard",
        ),
    ],
)
def test_tar_members_must_resolve_inside_the_staging_directory(
    tmp_path, member, message
):
    from scarf.cytebase import _safe_tar_members

    # Downloads extract into a fresh directory, where a relative name without
    # ".." stays inside; an existing symlink is what could carry it outside.
    staging = tmp_path / "staging"
    staging.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (staging / "escape").symlink_to(outside, target_is_directory=True)
    path = _write_tar(tmp_path / "archive.tar.gz", [member])

    with tarfile.open(path, "r:gz") as archive:
        with pytest.raises(ValueError, match=f"^{message}$"):
            _safe_tar_members(archive, staging)
    assert list(outside.iterdir()) == []


def test_archive_cannot_replace_its_own_download_path(tmp_path):
    from scarf.cytebase import _extract_and_replace

    work = tmp_path / "work"
    work.mkdir()
    staged = _write_tar(work / "download", [("data.zarr/zarr.json", "file", b"{}")])

    # Without a .tar.gz suffix, the archive's directory would take the
    # archive's own destination.
    with pytest.raises(
        ValueError,
        match="^Tar archive member conflicts with an internal download path$",
    ):
        _extract_and_replace(staged, work / "data.zarr")
    assert _tree(work) == ["download"]


def test_remove_path_unlinks_symlinks_without_following_them(tmp_path):
    from scarf.cytebase import _remove_path

    target = tmp_path / "target"
    target.mkdir()
    (target / "keep.txt").write_text("keep")
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    nested = tmp_path / "nested" / "inner"
    nested.mkdir(parents=True)
    (nested / "data.bin").write_bytes(b"x")

    _remove_path(link)
    _remove_path(tmp_path / "nested")
    _remove_path(tmp_path / "missing")

    assert not link.is_symlink()
    assert (target / "keep.txt").read_text() == "keep"
    assert not (tmp_path / "nested").exists()


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
    files = [
        cytebase._relative_file_path("scarf_docs", file).as_posix()
        for file in cytebase._bucket_files(
            "scarf_docs", "tenx_5K_pbmc_rnaseq", recursive=True
        )
    ]
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


def test_lazy_exports_import_and_cache_sdk_objects(monkeypatch):
    from scarf import cytebase
    from scarf.cytebase import _embeddings
    from scarf.cytebase.catalog import Catalog
    from scarf.cytebase.entry import DatasetEntry

    exports = {
        "Catalog": Catalog,
        "DatasetEntry": DatasetEntry,
        "embeddings": _embeddings.embeddings,
        "embedding": _embeddings.embedding,
        "embedding_coordinates": _embeddings.embedding_coordinates,
    }
    for name, value in exports.items():
        monkeypatch.delitem(vars(cytebase), name, raising=False)
        assert getattr(cytebase, name) is value
        assert vars(cytebase)[name] is value


def test_removed_dataset_wrapper_is_not_exported():
    from importlib.util import find_spec

    from scarf import cytebase

    assert find_spec("scarf.cytebase.dataset") is None
    assert "CytebaseDataset" not in cytebase.__all__
    assert "CytebaseDataset" not in dir(cytebase)
    with pytest.raises(AttributeError, match="has no attribute 'CytebaseDataset'"):
        cytebase.CytebaseDataset


def test_unknown_module_attributes_raise():
    from scarf import cytebase

    with pytest.raises(AttributeError, match="has no attribute 'missing'"):
        cytebase.missing


def test_dir_lists_lazy_exports():
    from scarf import cytebase

    assert {
        "Catalog",
        "DatasetEntry",
        "Repository",
        "connect",
        "embeddings",
        "embedding",
        "embedding_coordinates",
    } <= set(dir(cytebase))
