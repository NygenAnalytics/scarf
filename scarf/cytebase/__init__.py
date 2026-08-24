"""Browse and download public data from Cytebase."""

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import tarfile
import tempfile
from typing import TYPE_CHECKING

from huggingface_hub import (
    BucketFile,
    BucketFolder,
    download_bucket_files,
    list_bucket_tree,
)

if TYPE_CHECKING:
    import zarr

__all__ = ["Repository", "connect", "list_repositories"]

_BUCKET_ID = "Nygen/cytebase"
_ZARR_ARCHIVE_SUFFIX = ".zarr.tar.gz"


def _safe_name(name: str, *, kind: str) -> str:
    posix_name = PurePosixPath(name)
    windows_name = PureWindowsPath(name)
    if (
        not name
        or name in {".", ".."}
        or posix_name.name != name
        or windows_name.name != name
    ):
        raise ValueError(f"Invalid {kind}: {name!r}")
    return name


def _safe_relative_path(path: str, *, kind: str) -> PurePosixPath:
    if not path or path.startswith("/") or "\\" in path:
        raise ValueError(f"Invalid {kind}: {path!r}")
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Invalid {kind}: {path!r}")
    return PurePosixPath(*parts)


def _remote_path(repository: str, path: str | None = None) -> str:
    if path is None:
        return repository
    relative_path = _safe_relative_path(path, kind="remote path")
    return f"{repository}/{relative_path.as_posix()}"


def _relative_file_path(repository: str, file: BucketFile) -> PurePosixPath:
    repository_prefix = f"{repository}/"
    if not file.path.startswith(repository_prefix):
        raise ValueError(f"File is outside repository {repository!r}: {file.path!r}")
    return _safe_relative_path(
        file.path.removeprefix(repository_prefix),
        kind="bucket file path",
    )


def _bucket_files(
    repository: str,
    path: str | None = None,
    *,
    recursive: bool,
) -> list[BucketFile]:
    prefix = _remote_path(repository, path)
    files = []
    for item in list_bucket_tree(
        _BUCKET_ID,
        prefix=prefix,
        recursive=recursive,
        token=False,
    ):
        if not isinstance(item, BucketFile):
            continue
        if item.path != prefix and not item.path.startswith(f"{prefix}/"):
            continue
        _relative_file_path(repository, item)
        files.append(item)
    return sorted(files, key=lambda file: file.path)


def list_repositories() -> list[str]:
    """Return the top-level repositories in Cytebase."""
    repositories = []
    for item in list_bucket_tree(_BUCKET_ID, recursive=False, token=False):
        if isinstance(item, BucketFolder) and "/" not in item.path:
            repositories.append(_safe_name(item.path, kind="repository name"))
    return sorted(repositories)


def connect(repository: str) -> "Repository":
    """Connect to a public Cytebase repository."""
    repository = _safe_name(repository, kind="repository name")
    available = list_repositories()
    if repository not in available:
        choices = "\n".join(available)
        raise KeyError(
            f"{repository!r} is not a Cytebase repository. "
            f"Available repositories:\n{choices}"
        )
    return Repository(repository)


@dataclass(frozen=True, slots=True)
class Repository:
    """Read files from one top-level Cytebase repository."""

    name: str

    def __post_init__(self) -> None:
        _safe_name(self.name, kind="repository name")

    def list_datasets(self) -> list[str]:
        """Return immediate dataset directories in this repository."""
        prefix = f"{self.name}/"
        datasets = []
        for item in list_bucket_tree(
            _BUCKET_ID,
            prefix=self.name,
            recursive=False,
            token=False,
        ):
            if not isinstance(item, BucketFolder):
                continue
            if not item.path.startswith(prefix):
                continue
            relative_path = item.path.removeprefix(prefix)
            if "/" not in relative_path:
                datasets.append(_safe_name(relative_path, kind="dataset name"))
        return sorted(datasets)

    def list_files(
        self,
        path: str | None = None,
        *,
        recursive: bool = True,
    ) -> list[str]:
        """Return file paths relative to this repository."""
        return [
            _relative_file_path(self.name, file).as_posix()
            for file in _bucket_files(self.name, path, recursive=recursive)
        ]

    def download(
        self,
        path: str,
        destination: str | Path = ".",
    ) -> list[Path]:
        """Download a file or directory while preserving its repository path."""
        _safe_relative_path(path, kind="download path")
        files = _bucket_files(self.name, path, recursive=True)
        if not files:
            raise FileNotFoundError(f"No Cytebase files found at {self.name}/{path}")
        return _download_files(self.name, files, Path(destination))

    def download_dataset(
        self,
        name: str,
        destination: str | Path = ".",
        *,
        zarr: bool = False,
    ) -> Path:
        """Download one dataset, optionally selecting its Zarr archive."""
        name = _safe_name(name, kind="dataset name")
        files = _bucket_files(self.name, name, recursive=True)
        if not files:
            choices = "\n".join(self.list_datasets())
            raise KeyError(
                f"{name!r} is not in repository {self.name!r}. "
                f"Available datasets:\n{choices}"
            )

        if zarr:
            selected = [
                file for file in files if file.path.endswith(_ZARR_ARCHIVE_SUFFIX)
            ]
            if not selected:
                raise FileNotFoundError(f"No Zarr archive is available for {name!r}")
        else:
            selected = [
                file for file in files if not file.path.endswith(_ZARR_ARCHIVE_SUFFIX)
            ]

        destination = Path(destination).absolute()
        _download_files(self.name, selected, destination)
        return destination / name

    def open_zarr(self, path: str) -> "zarr.Group":
        """Open an unpacked Zarr group for anonymous read-only access."""
        relative_path = _safe_relative_path(path, kind="Zarr path")
        uri = f"hf://buckets/{_BUCKET_ID}/{self.name}/{relative_path.as_posix()}"

        import zarr
        from zarr.storage import FsspecStore

        store = FsspecStore.from_url(
            uri,
            storage_options={"token": False},
            read_only=True,
        )
        return zarr.open_group(store=store, mode="r")


def _download_files(
    repository: str,
    files: list[BucketFile],
    destination: Path,
) -> list[Path]:
    destination = destination.absolute()
    destination.mkdir(parents=True, exist_ok=True)
    staging_path = Path(
        tempfile.mkdtemp(
            dir=destination,
            prefix=".cytebase-download-",
        )
    )
    downloads = []

    try:
        for file in files:
            relative_path = _relative_file_path(repository, file)
            staged_file = staging_path.joinpath(*relative_path.parts)
            staged_file.parent.mkdir(parents=True, exist_ok=True)
            final_file = destination.joinpath(*relative_path.parts)
            downloads.append((file, staged_file, final_file))

        download_bucket_files(
            _BUCKET_ID,
            files=[(file, staged_file) for file, staged_file, _ in downloads],
            token=False,
        )

        for file, staged_file, _ in downloads:
            if not staged_file.is_file():
                raise FileNotFoundError(f"Cytebase did not download {file.path!r}")
            if staged_file.stat().st_size != file.size:
                raise OSError(
                    f"Downloaded {staged_file.stat().st_size} bytes for {file.path!r}, "
                    f"expected {file.size}"
                )

        for _, staged_file, final_file in downloads:
            final_file.parent.mkdir(parents=True, exist_ok=True)
            if final_file.name.endswith(".tar.gz"):
                _extract_and_replace(staged_file, final_file)
            else:
                os.replace(staged_file, final_file)
    finally:
        shutil.rmtree(staging_path, ignore_errors=True)

    return [final_file for _, _, final_file in downloads]


def _safe_tar_members(
    archive: tarfile.TarFile,
    staging_path: Path,
) -> list[tarfile.TarInfo]:
    staging_root = staging_path.resolve()
    members = archive.getmembers()

    for member in members:
        member_path = PurePosixPath(member.name)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError(f"Unsafe tar archive member: {member.name}")

        target = (staging_root / Path(*member_path.parts)).resolve()
        if not target.is_relative_to(staging_root):
            raise ValueError(f"Unsafe tar archive member: {member.name}")

        if member.issym() or member.islnk():
            link_path = PurePosixPath(member.linkname)
            if link_path.is_absolute() or ".." in link_path.parts:
                raise ValueError(f"Unsafe tar archive link: {member.name}")

            link_root = target.parent if member.issym() else staging_root
            link_target = (link_root / Path(*link_path.parts)).resolve()
            if not link_target.is_relative_to(staging_root):
                raise ValueError(f"Unsafe tar archive link: {member.name}")
        elif not member.isfile() and not member.isdir():
            raise ValueError(f"Unsupported tar archive member: {member.name}")

    return members


def _path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _replace_staged_paths(
    staged_paths: list[tuple[Path, Path]],
    backup_path: Path,
) -> None:
    replacements: list[tuple[Path, Path | None]] = []

    try:
        for index, (source, destination) in enumerate(staged_paths):
            backup = backup_path / str(index)
            previous = backup if _path_exists(destination) else None
            if previous is not None:
                os.replace(destination, previous)

            try:
                os.replace(source, destination)
            except BaseException:
                if previous is not None:
                    os.replace(previous, destination)
                raise
            replacements.append((destination, previous))
    except BaseException:
        for destination, previous in reversed(replacements):
            _remove_path(destination)
            if previous is not None:
                os.replace(previous, destination)
        raise


def _extract_and_replace(
    archive_path: Path,
    destination: Path,
) -> None:
    staging_path = Path(
        tempfile.mkdtemp(
            dir=destination.parent,
            prefix=f".{destination.name}.extract-",
        )
    )
    backup_path = Path(
        tempfile.mkdtemp(
            dir=destination.parent,
            prefix=f".{destination.name}.backup-",
        )
    )

    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            members = _safe_tar_members(archive, staging_path)
            archive.extractall(staging_path, members=members, filter="data")

        extracted_paths = sorted(staging_path.iterdir(), key=lambda path: path.name)
        extracted_replacements = [
            (path, destination.parent / path.name) for path in extracted_paths
        ]
        reserved_paths = {destination, archive_path, staging_path, backup_path}
        if any(
            extracted_destination in reserved_paths
            for _, extracted_destination in extracted_replacements
        ):
            raise ValueError(
                "Tar archive member conflicts with an internal download path"
            )

        _replace_staged_paths(
            [
                *extracted_replacements,
                (archive_path, destination),
            ],
            backup_path,
        )
    finally:
        shutil.rmtree(staging_path, ignore_errors=True)
        shutil.rmtree(backup_path, ignore_errors=True)
