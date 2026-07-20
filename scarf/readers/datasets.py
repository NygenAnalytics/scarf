"""Download datasets published for Scarf examples."""

import io
import os
import tarfile
import time
from json import JSONDecodeError
from typing import Any

import pandas as pd

from ..utils.logging import logger
from ..utils.progress import tqdmbar

__all__ = ["show_available_datasets", "fetch_dataset"]

type JsonDict = dict[str, Any]
type DatasetEntry = tuple[str, str]
type DatasetsMap = dict[str, DatasetEntry]
type FileMap = dict[str, str]


class OSFdownloader:
    """Download datasets from an OSF project via the OSF API."""

    def __init__(self, project_id: str) -> None:
        self.projectId = project_id
        self.storages = ["osfstorage", "figshare"]
        self.url = f"https://api.osf.io/v2/nodes/{self.projectId}/files/"
        self.datasets, self.sourceFile = self._populate_datasets()
        self.sources = self._populate_sources()

    def get_json(self, storage: str, endpoint: str, url: str | None) -> Any:
        import requests

        if endpoint != "":
            endpoint = endpoint + "/"
        if url is None:
            url = self.url + f"{storage}/{endpoint}"
        return requests.get(url).json()

    def get_all_pages(self, storage: str, endpoint: str = "") -> list[JsonDict]:
        data = []
        url = None
        n_attempts = 0
        while n_attempts < 5:
            try:
                response = self.get_json(storage, endpoint, url)
            except JSONDecodeError:
                time.sleep(1)
                n_attempts += 1
                continue
            if "data" in response:
                data.extend(response["data"])
                url = response["links"]["next"]
                if url is None:
                    break
            else:
                n_attempts += 1
                time.sleep(1)
        return data

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
        import requests

        n_attempts = 0
        while n_attempts < 5:
            try:
                source_filename = self._get_files_for_node(
                    "osfstorage",
                    self.sourceFile,
                )["sources.csv"]
                return dict(
                    pd.read_csv(io.StringIO(requests.get(source_filename).text))
                    .set_index("id")
                    .to_dict()
                )
            except KeyError:
                time.sleep(1)
                n_attempts += 1
        raise KeyError("Failed to load dataset sources after 5 attempts")

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


def handle_download(url: str, out_fn: str, seq_counter: str = "") -> None:
    """Download one dataset file and extract tar archives."""
    import requests
    from pathlib import Path

    chunk_size = int(1e7)
    response = requests.head(url, allow_redirects=True)
    size = int(response.headers.get("content-length", -1))
    size = size // chunk_size + 1

    response = requests.get(url, stream=True)
    with open(out_fn, "wb") as handle:
        for chunk in tqdmbar(
            response.iter_content(chunk_size=chunk_size),
            total=size,
            desc=f"Downloading {seq_counter}",
        ):
            if chunk:
                handle.write(chunk)
    logger.debug(f"Download finished! File saved here: {out_fn}")
    if out_fn.endswith("tar.gz"):
        archive = tarfile.open(out_fn, "r:gz")
        archive.extractall(str(Path(out_fn).parent.absolute()))
        archive.close()


def show_available_datasets() -> None:
    """Print datasets available from the Scarf OSF project."""
    global osfd
    if osfd is None:
        osfd = OSFdownloader("zeupv")
    osfd.show_datasets()


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

    save_directory = os.path.join(save_path, dataset_name)
    if not os.path.isdir(save_directory):
        os.makedirs(save_directory)

    if as_zarr:
        save_name, url = get_zarr_entry(files)
        output_path = os.path.abspath(os.path.join(save_directory, save_name))
        handle_download(url, output_path)
        return None

    valid_files = [
        filename for filename in files if not filename.endswith(zarr_extension)
    ]
    for index, filename in enumerate(valid_files, start=1):
        output_path = os.path.abspath(os.path.join(save_directory, filename))
        handle_download(
            files[filename],
            output_path,
            f"{index}/{len(valid_files)}",
        )
    return None
