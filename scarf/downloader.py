"""
Used to download datasets included in Scarf.

Classes:
    OSFdownloader: A class for downloading datasets from OSF.

- Methods:
    - handle_download: carry out the download of a specified dataset
    - show_available_datasets: list datasets offered through Scarf
    - fetch_dataset: Downloads datasets from online repositories and saves them in as-is format
"""

import io
import os
import tarfile
import time
from json import JSONDecodeError

import pandas as pd

from .utils import logger, tqdmbar

__all__ = ["show_available_datasets", "fetch_dataset"]


class OSFdownloader:
    """Download datasets from an OSF project via the OSF API.

    Attributes:
        projectId: OSF node identifier.
        storages: Storage providers to search.
        url: OSF files API endpoint.
        datasets: Parsed dataset metadata table.
        sourceFile: Raw OSF file listing JSON.
        sources: Mapping of dataset names to download URLs.
    """

    def __init__(self, project_id):
        self.projectId = project_id
        self.storages = ["osfstorage", "figshare"]
        self.url = f"https://api.osf.io/v2/nodes/{self.projectId}/files/"
        self.datasets, self.sourceFile = self._populate_datasets()
        self.sources = self._populate_sources()

    def get_json(self, storage, endpoint, url):
        import requests

        if endpoint != "":
            endpoint = endpoint + "/"
        if url is None:
            url = self.url + f"{storage}/{endpoint}"
        return requests.get(url).json()

    def get_all_pages(self, storage, endpoint=""):
        data = []
        url = None
        n_attempts = 0
        while n_attempts < 5:
            try:
                r = self.get_json(storage, endpoint, url)
            except JSONDecodeError:
                time.sleep(1)
                n_attempts += 1
                continue
            if "data" in r:
                data.extend(r["data"])
                url = r["links"]["next"]
                if url is None:
                    break
            else:
                n_attempts += 1
                time.sleep(1)
        return data

    @staticmethod
    def _process_path(node):
        return node["attributes"]["path"].rstrip("/").lstrip("/")

    def _populate_datasets(self):
        datasets = {}
        source_fn = ""
        for storage in self.storages:
            for i in self.get_all_pages(storage):
                path = self._process_path(i)
                if i["attributes"]["name"] == "sources":
                    source_fn = path
                    continue
                datasets[i["attributes"]["name"]] = (path, storage)
        return datasets, source_fn

    def _populate_sources(self):
        import requests

        n_attempts = 0
        while n_attempts < 5:
            try:
                source_fn = self._get_files_for_node("osfstorage", self.sourceFile)[
                    "sources.csv"
                ]
                return (
                    pd.read_csv(io.StringIO(requests.get(source_fn).text))
                    .set_index("id")
                    .to_dict()
                )
            except KeyError:
                time.sleep(1)
                n_attempts += 1

    def show_datasets(self):
        print("\n".join(sorted(self.datasets.keys())))

    def _get_files_for_node(self, storage, file_id):
        base_url = f"https://files.de-1.osf.io/v1/resources/{self.projectId}/providers/"
        ret_val = {}
        for i in self.get_all_pages(storage, file_id):
            path = self._process_path(i)
            ret_val[i["attributes"]["name"]] = base_url + f"{storage}/{path}"
        return ret_val

    def get_dataset_file_ids(self, dataset_name):
        if dataset_name not in self.datasets:
            raise KeyError(
                f"ERROR: {dataset_name} was not found. "
                f"Please choose one of the following:\n{self.show_datasets()}"
            )
        file_id, storage = self.datasets[dataset_name]
        return self._get_files_for_node(storage, file_id)


osfd = None


def handle_download(url: str, out_fn: str, seq_counter: str = "") -> None:
    """
    Carry out the download of a specified dataset.

    Args:
        url: URL of file to be downloaded
        out_fn: Absolute path to the file where the downloaded data is to be saved
        seq_counter: show the sequence number of the download

    Returns: None

    """

    import requests
    from pathlib import Path

    chunk_size = int(1e7)
    r = requests.head(url, allow_redirects=True)
    size = int(r.headers.get("content-length", -1))
    size = size // chunk_size + 1

    r = requests.get(url, stream=True)
    with open(out_fn, "wb") as handle:
        for chunk in tqdmbar(
            r.iter_content(chunk_size=chunk_size),
            total=size,
            desc=f"Downloading {seq_counter}",
        ):
            if chunk:
                handle.write(chunk)
    logger.debug(f"Download finished! File saved here: {out_fn}")
    if out_fn.endswith("tar.gz"):
        tar = tarfile.open(out_fn, "r:gz")
        tar.extractall(str(Path(out_fn).parent.absolute()))
        tar.close()


def show_available_datasets() -> None:
    """
    List datasets offered through Scarf.

    Prints the list of datasets.

    Returns:
        None
    """
    global osfd
    if osfd is None:
        osfd = OSFdownloader("zeupv")
    osfd.show_datasets()


def fetch_dataset(
    dataset_name: str, save_path: str = ".", as_zarr: bool = False
) -> None:
    """
    Downloads datasets from online repositories and saves them in as-is format.

    Args:
        dataset_name: Name of the dataset
        save_path: Save location without name of the file
        as_zarr: If True, then a Zarr format file, if available, is downloaded instead

    Returns:
        None
    """

    zarr_ext = ".zarr.tar.gz"

    def has_zarr(entry):
        for e in entry:
            if e.endswith(zarr_ext):
                return True
        return False

    def get_zarr_entry(entry):
        for e in entry:
            if e.endswith(zarr_ext):
                return e, entry[e]
        return False, False

    global osfd
    if osfd is None:
        osfd = OSFdownloader("zeupv")

    files = osfd.get_dataset_file_ids(dataset_name)

    if as_zarr:
        if has_zarr(files) is False:
            logger.error(
                f"Zarr file does not exist for {dataset_name}. Nothing downloaded"
            )
            return None

    save_dir = os.path.join(save_path, dataset_name)
    if os.path.isdir(save_dir) is False:
        os.makedirs(save_dir)

    if as_zarr:
        sp, url = get_zarr_entry(files)
        sp = os.path.abspath(os.path.join(save_dir, sp))
        handle_download(url, sp)
    else:
        valid_files = [x for x in files if not x.endswith(zarr_ext)]
        for n, i in enumerate(valid_files, start=1):
            sp = os.path.abspath(os.path.join(save_dir, i))
            handle_download(files[i], sp, f"{n}/{len(valid_files)}")
