"""Download datasets published for Scarf examples."""

from collections.abc import Callable
import io
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import tarfile
import tempfile
import time
from typing import Any

import pandas as pd

from ..utils.logging import logger
from ..utils.progress import tqdmbar

__all__ = ["show_available_datasets", "fetch_dataset"]

type JsonDict = dict[str, Any]
type DatasetEntry = tuple[str, str]
type DatasetsMap = dict[str, DatasetEntry]
type FileMap = dict[str, str]

_REQUEST_TIMEOUT = (5.0, 30.0)
_MAX_REQUEST_ATTEMPTS = 5
_RETRY_DELAY_SECONDS = 1.0
_DOWNLOAD_CHUNK_SIZE = 10_000_000


class _IncompleteDownloadError(OSError):
    pass


def _is_retryable_status(status_code: int | None) -> bool:
    return status_code == 429 or (status_code is not None and 500 <= status_code < 600)


def _get_with_retries[T](
    url: str,
    *,
    stream: bool,
    consume: Callable[[Any], T],
) -> T:
    import requests

    for attempt in range(_MAX_REQUEST_ATTEMPTS):
        response = None
        try:
            response = requests.get(url, stream=stream, timeout=_REQUEST_TIMEOUT)
            response.raise_for_status()
            return consume(response)
        except requests.exceptions.InvalidJSONError:
            raise
        except requests.exceptions.HTTPError:
            status_code = response.status_code if response is not None else None
            if (
                not _is_retryable_status(status_code)
                or attempt == _MAX_REQUEST_ATTEMPTS - 1
            ):
                raise
        except requests.exceptions.RequestException:
            if attempt == _MAX_REQUEST_ATTEMPTS - 1:
                raise
        except _IncompleteDownloadError:
            if attempt == _MAX_REQUEST_ATTEMPTS - 1:
                raise
        finally:
            if response is not None:
                response.close()
        time.sleep(_RETRY_DELAY_SECONDS * (2**attempt))

    raise RuntimeError("OSF request retry loop exited unexpectedly")


def _response_json(response: Any) -> Any:
    return response.json()


def _response_text(response: Any) -> str:
    return str(response.text)


class OSFdownloader:
    """Download datasets from an OSF project via the OSF API."""

    def __init__(self, project_id: str) -> None:
        self.projectId = project_id
        self.url = f"https://api.osf.io/v2/nodes/{self.projectId}/files/"
        self.storages = self._populate_storages()
        self.datasets, self.sourceFile = self._populate_datasets()
        self.sources = self._populate_sources()

    def get_json(self, storage: str, endpoint: str, url: str | None) -> Any:
        if endpoint != "":
            endpoint = endpoint + "/"
        if url is None:
            url = self.url + f"{storage}/{endpoint}"
        return _get_with_retries(
            url,
            stream=False,
            consume=_response_json,
        )

    def _get_all_pages(
        self,
        storage: str,
        endpoint: str,
        initial_url: str | None,
    ) -> list[JsonDict]:
        data: list[JsonDict] = []
        url = initial_url
        seen_urls = {initial_url} if initial_url is not None else set()

        while True:
            response = self.get_json(storage, endpoint, url)
            if not isinstance(response, dict):
                raise ValueError("Malformed OSF pagination response")

            page_data = response.get("data")
            links = response.get("links")
            if (
                not isinstance(page_data, list)
                or not all(isinstance(node, dict) for node in page_data)
                or not isinstance(links, dict)
                or "next" not in links
            ):
                raise ValueError("Malformed OSF pagination response")

            next_url = links["next"]
            if next_url is not None and not isinstance(next_url, str):
                raise ValueError("Malformed OSF pagination link")

            data.extend(page_data)
            if next_url is None:
                return data
            if next_url in seen_urls:
                raise ValueError("OSF pagination contains a cycle")

            seen_urls.add(next_url)
            url = next_url

    def get_all_pages(self, storage: str, endpoint: str = "") -> list[JsonDict]:
        return self._get_all_pages(storage, endpoint, None)

    def _populate_storages(self) -> list[str]:
        storages = []
        for node in self._get_all_pages("", "", self.url):
            attributes = node.get("attributes")
            relationships = node.get("relationships")
            if not isinstance(attributes, dict) or not isinstance(relationships, dict):
                raise ValueError("Malformed OSF storage response")

            provider = attributes.get("provider")
            root_folder = relationships.get("root_folder")
            if isinstance(provider, str) and isinstance(root_folder, dict):
                storages.append(provider)

        if not storages:
            raise ValueError("OSF project has no connected storage providers")
        return storages

    @staticmethod
    def _process_path(node: JsonDict) -> str:
        return str(node["attributes"]["path"]).rstrip("/").lstrip("/")

    def _populate_datasets(self) -> tuple[DatasetsMap, str]:
        datasets = {}
        source_filename = ""
        for storage in self.storages:
            for node in self.get_all_pages(storage):
                path = self._process_path(node)
                if node["attributes"]["name"] == "sources":
                    source_filename = path
                    continue
                datasets[node["attributes"]["name"]] = (path, storage)
        return datasets, source_filename

    def _populate_sources(self) -> dict[str, Any]:
        try:
            source_filename = self._get_files_for_node(
                "osfstorage",
                self.sourceFile,
            )["sources.csv"]
        except KeyError as error:
            raise KeyError("OSF dataset sources file was not found") from error

        source_text = _get_with_retries(
            source_filename,
            stream=False,
            consume=_response_text,
        )
        return dict(pd.read_csv(io.StringIO(source_text)).set_index("id").to_dict())

    def show_datasets(self) -> None:
        print("\n".join(sorted(self.datasets.keys())))

    def _get_files_for_node(self, storage: str, file_id: str) -> FileMap:
        base_url = f"https://files.de-1.osf.io/v1/resources/{self.projectId}/providers/"
        files = {}
        for node in self.get_all_pages(storage, file_id):
            path = self._process_path(node)
            files[node["attributes"]["name"]] = base_url + f"{storage}/{path}"
        return files

    def get_dataset_file_ids(self, dataset_name: str) -> FileMap:
        if dataset_name not in self.datasets:
            available = "\n".join(sorted(self.datasets.keys()))
            raise KeyError(
                f"ERROR: {dataset_name} was not found. "
                f"Please choose one of the following:\n{available}"
            )
        file_id, storage = self.datasets[dataset_name]
        return self._get_files_for_node(storage, file_id)


osfd: OSFdownloader | None = None


def _content_length(response: Any) -> int | None:
    value = response.headers.get("content-length")
    if value is None:
        return None

    try:
        length = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid Content-Length response header") from error
    if length < 0:
        raise ValueError("Invalid Content-Length response header")
    return length


def _stream_to_file(response: Any, partial_path: Path, seq_counter: str) -> None:
    expected_length = _content_length(response)
    total = (
        None
        if expected_length is None
        else (expected_length + _DOWNLOAD_CHUNK_SIZE - 1) // _DOWNLOAD_CHUNK_SIZE
    )
    bytes_written = 0

    with partial_path.open("wb") as handle:
        for chunk in tqdmbar(
            response.iter_content(chunk_size=_DOWNLOAD_CHUNK_SIZE),
            total=total,
            desc=f"Downloading {seq_counter}",
        ):
            if chunk:
                handle.write(chunk)
                bytes_written += len(chunk)

    if expected_length is not None and bytes_written != expected_length:
        raise _IncompleteDownloadError(
            f"Downloaded {bytes_written} bytes, expected {expected_length}"
        )


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
    partial_path: Path,
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
        with tarfile.open(partial_path, "r:gz") as archive:
            members = _safe_tar_members(archive, staging_path)
            archive.extractall(staging_path, members=members, filter="data")

        extracted_paths = sorted(staging_path.iterdir(), key=lambda path: path.name)
        extracted_replacements = [
            (path, destination.parent / path.name) for path in extracted_paths
        ]
        reserved_paths = {destination, partial_path, staging_path, backup_path}
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
                (partial_path, destination),
            ],
            backup_path,
        )
    finally:
        shutil.rmtree(staging_path, ignore_errors=True)
        shutil.rmtree(backup_path, ignore_errors=True)


def handle_download(url: str, out_fn: str, seq_counter: str = "") -> None:
    """Download one dataset file and extract tar archives."""
    destination = Path(out_fn).absolute()
    partial_path = destination.with_name(f"{destination.name}.partial")
    partial_path.unlink(missing_ok=True)

    try:
        _get_with_retries(
            url,
            stream=True,
            consume=lambda response: _stream_to_file(
                response,
                partial_path,
                seq_counter,
            ),
        )
        if destination.name.endswith(".tar.gz"):
            _extract_and_replace(partial_path, destination)
        else:
            os.replace(partial_path, destination)
    finally:
        partial_path.unlink(missing_ok=True)

    logger.debug(f"Download finished! File saved here: {out_fn}")


def show_available_datasets() -> None:
    """Print datasets available from the Scarf OSF project."""
    global osfd
    if osfd is None:
        osfd = OSFdownloader("zeupv")
    osfd.show_datasets()


def _safe_child_path(parent: Path, name: str, *, kind: str) -> Path:
    posix_name = PurePosixPath(name)
    windows_name = PureWindowsPath(name)
    if (
        not name
        or name in {".", ".."}
        or posix_name.name != name
        or windows_name.name != name
    ):
        raise ValueError(f"Unsafe dataset {kind}: {name!r}")
    return parent / name


def fetch_dataset(
    dataset_name: str,
    save_path: str = ".",
    as_zarr: bool = False,
) -> None:
    """Download one Scarf example dataset."""
    zarr_extension = ".zarr.tar.gz"

    def has_zarr(entry: FileMap) -> bool:
        return any(filename.endswith(zarr_extension) for filename in entry)

    def get_zarr_entry(entry: FileMap) -> tuple[str, str]:
        for filename in entry:
            if filename.endswith(zarr_extension):
                return filename, entry[filename]
        raise KeyError("No zarr entry found in dataset files")

    global osfd
    if osfd is None:
        osfd = OSFdownloader("zeupv")

    files = osfd.get_dataset_file_ids(dataset_name)
    if as_zarr and not has_zarr(files):
        logger.error(f"Zarr file does not exist for {dataset_name}. Nothing downloaded")
        return None

    save_root = Path(save_path).absolute()
    save_directory = _safe_child_path(
        save_root,
        dataset_name,
        kind="name",
    )
    save_directory.mkdir(parents=True, exist_ok=True)

    if as_zarr:
        save_name, url = get_zarr_entry(files)
        output_path = _safe_child_path(
            save_directory,
            save_name,
            kind="filename",
        )
        handle_download(url, str(output_path))
        return None

    valid_files = [
        filename for filename in files if not filename.endswith(zarr_extension)
    ]
    for index, filename in enumerate(valid_files, start=1):
        output_path = _safe_child_path(
            save_directory,
            filename,
            kind="filename",
        )
        handle_download(
            files[filename],
            str(output_path),
            f"{index}/{len(valid_files)}",
        )
    return None
