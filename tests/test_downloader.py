import io
from json import JSONDecodeError
import tarfile

import pytest

from . import full_path, remove


@pytest.fixture
def mock_osf_downloader(monkeypatch):
    from scarf import downloader

    class FakeOSFdownloader:
        projectId = "zeupv"
        datasets = {
            **{f"dataset_{i}": (f"path/{i}", "osfstorage") for i in range(20)},
            "tenx_5K_pbmc_rnaseq": ("tenx_5K", "osfstorage"),
        }
        sourceFile = "sources"

        def show_datasets(self):
            print("\n".join(sorted(self.datasets.keys())))

        def get_dataset_file_ids(self, dataset_name):
            if dataset_name not in self.datasets:
                raise KeyError(dataset_name)
            if dataset_name == "tenx_5K_pbmc_rnaseq":
                return {"tenx_5K_pbmc.zarr.tar.gz": "http://example.test/zarr.tar.gz"}
            return {"data.h5ad": "http://example.test/data.h5ad"}

    fake = FakeOSFdownloader()
    monkeypatch.setattr(downloader, "osfd", fake)
    return fake


def test_downloader(mock_osf_downloader):
    assert len(mock_osf_downloader.datasets) > 15


def test_osf_downloader_initializes_and_lists_sorted(monkeypatch, capsys):
    from scarf.downloader import OSFdownloader

    datasets = {
        "zeta": ("zeta-id", "figshare"),
        "alpha": ("alpha-id", "osfstorage"),
    }
    monkeypatch.setattr(
        OSFdownloader,
        "_populate_datasets",
        lambda self: (datasets, "sources-id"),
    )
    monkeypatch.setattr(
        OSFdownloader,
        "_populate_sources",
        lambda self: {"url": {"alpha": "https://example.test/alpha"}},
    )

    client = OSFdownloader("project-id")

    assert client.projectId == "project-id"
    assert client.storages == ["osfstorage", "figshare"]
    assert client.datasets == datasets
    assert client.sourceFile == "sources-id"
    client.show_datasets()
    assert capsys.readouterr().out == "alpha\nzeta\n"


def test_osf_downloader_builds_api_urls(monkeypatch):
    import requests

    from scarf.downloader import OSFdownloader

    requested_urls = []

    class Response:
        def json(self):
            return {"data": []}

    def fake_get(url):
        requested_urls.append(url)
        return Response()

    monkeypatch.setattr(requests, "get", fake_get)
    client = object.__new__(OSFdownloader)
    client.url = "https://api.example.test/files/"

    assert client.get_json("osfstorage", "folder", None) == {"data": []}
    assert client.get_json("figshare", "", "https://next.example.test") == {"data": []}
    assert requested_urls == [
        "https://api.example.test/files/osfstorage/folder/",
        "https://next.example.test",
    ]


def test_osf_downloader_retries_and_paginates(monkeypatch):
    from scarf.downloader import OSFdownloader

    client = object.__new__(OSFdownloader)
    responses = iter(
        [
            JSONDecodeError("invalid response", "", 0),
            {},
            {"data": [{"id": "first"}], "links": {"next": "next-page"}},
            {"data": [{"id": "second"}], "links": {"next": None}},
        ]
    )
    calls = []
    sleeps = []

    def fake_get_json(storage, endpoint, url):
        calls.append((storage, endpoint, url))
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(client, "get_json", fake_get_json)
    monkeypatch.setattr("scarf.downloader.time.sleep", sleeps.append)

    assert client.get_all_pages("osfstorage", "folder") == [
        {"id": "first"},
        {"id": "second"},
    ]
    assert calls == [
        ("osfstorage", "folder", None),
        ("osfstorage", "folder", None),
        ("osfstorage", "folder", None),
        ("osfstorage", "folder", "next-page"),
    ]
    assert sleeps == [1, 1]


def test_osf_downloader_discovers_datasets_and_files(monkeypatch):
    from scarf.downloader import OSFdownloader

    client = object.__new__(OSFdownloader)
    client.projectId = "project-id"
    client.storages = ["osfstorage", "figshare"]

    def node(name, path):
        return {"attributes": {"name": name, "path": path}}

    def fake_get_all_pages(storage, endpoint=""):
        if endpoint == "":
            if storage == "osfstorage":
                return [
                    node("sources", "/sources-id/"),
                    node("alpha", "/alpha-id/"),
                ]
            return [node("beta", "/beta-id/")]
        assert storage == "osfstorage"
        assert endpoint == "alpha-id"
        return [
            node("counts.mtx.gz", "/files/counts-id/"),
            node("barcodes.tsv.gz", "/files/barcodes-id/"),
        ]

    monkeypatch.setattr(client, "get_all_pages", fake_get_all_pages)

    datasets, source_file = client._populate_datasets()
    assert datasets == {
        "alpha": ("alpha-id", "osfstorage"),
        "beta": ("beta-id", "figshare"),
    }
    assert source_file == "sources-id"

    client.datasets = datasets
    expected_files = {
        "counts.mtx.gz": (
            "https://files.de-1.osf.io/v1/resources/project-id/providers/"
            "osfstorage/files/counts-id"
        ),
        "barcodes.tsv.gz": (
            "https://files.de-1.osf.io/v1/resources/project-id/providers/"
            "osfstorage/files/barcodes-id"
        ),
    }
    assert client.get_dataset_file_ids("alpha") == expected_files

    with pytest.raises(KeyError, match="missing was not found") as error:
        client.get_dataset_file_ids("missing")
    assert "alpha\nbeta" in error.value.args[0]


def test_osf_downloader_populates_sources(monkeypatch):
    import requests

    from scarf.downloader import OSFdownloader

    client = object.__new__(OSFdownloader)
    client.sourceFile = "sources-id"
    monkeypatch.setattr(
        client,
        "_get_files_for_node",
        lambda storage, file_id: {"sources.csv": "https://example.test/sources.csv"},
    )

    class Response:
        text = (
            "id,title,url\n"
            "alpha,Alpha dataset,https://example.test/alpha\n"
            "beta,Beta dataset,https://example.test/beta\n"
        )

    requested_urls = []

    def fake_get(url):
        requested_urls.append(url)
        return Response()

    monkeypatch.setattr(requests, "get", fake_get)

    assert client._populate_sources() == {
        "title": {
            "alpha": "Alpha dataset",
            "beta": "Beta dataset",
        },
        "url": {
            "alpha": "https://example.test/alpha",
            "beta": "https://example.test/beta",
        },
    }
    assert requested_urls == ["https://example.test/sources.csv"]


def test_osf_downloader_source_retries_are_bounded(monkeypatch):
    from scarf.downloader import OSFdownloader

    client = object.__new__(OSFdownloader)
    client.sourceFile = "sources-id"
    attempts = []
    sleeps = []

    def missing_sources(storage, file_id):
        attempts.append((storage, file_id))
        return {}

    monkeypatch.setattr(client, "_get_files_for_node", missing_sources)
    monkeypatch.setattr("scarf.downloader.time.sleep", sleeps.append)

    with pytest.raises(KeyError, match="after 5 attempts"):
        client._populate_sources()
    assert attempts == [("osfstorage", "sources-id")] * 5
    assert sleeps == [1] * 5


def test_show_available_datasets(mock_osf_downloader):
    from scarf.downloader import show_available_datasets

    show_available_datasets()


def test_fetch_dataset(bastidas_ponce_data):
    import os

    assert os.path.isfile(bastidas_ponce_data)


def test_downloader_as_zarr(mock_osf_downloader, monkeypatch, tmp_path):
    from scarf import downloader

    downloaded = []

    def fake_handle_download(url, out_fn, seq_counter=""):
        downloaded.append((url, out_fn))
        with open(out_fn, "wb") as handle:
            handle.write(b"test")

    monkeypatch.setattr(downloader, "handle_download", fake_handle_download)

    sample = "tenx_5K_pbmc_rnaseq"
    save_root = str(tmp_path)
    downloader.fetch_dataset(sample, as_zarr=True, save_path=save_root)
    assert len(downloaded) == 1
    assert downloaded[0][0] == "http://example.test/zarr.tar.gz"
    assert downloaded[0][1].endswith("tenx_5K_pbmc.zarr.tar.gz")


def test_fetch_dataset_downloads_each_non_zarr_file(monkeypatch, tmp_path):
    from scarf import downloader

    class MultiFileDownloader:
        def get_dataset_file_ids(self, dataset_name):
            assert dataset_name == "tiny_dataset"
            return {
                "matrix.mtx.gz": "https://example.test/matrix",
                "features.tsv.gz": "https://example.test/features",
                "tiny_dataset.zarr.tar.gz": "https://example.test/zarr",
            }

    downloads = []

    def fake_handle_download(url, out_fn, seq_counter=""):
        downloads.append((url, out_fn, seq_counter))

    monkeypatch.setattr(downloader, "osfd", MultiFileDownloader())
    monkeypatch.setattr(downloader, "handle_download", fake_handle_download)

    downloader.fetch_dataset("tiny_dataset", save_path=str(tmp_path))

    output_dir = tmp_path / "tiny_dataset"
    assert output_dir.is_dir()
    assert downloads == [
        (
            "https://example.test/matrix",
            str((output_dir / "matrix.mtx.gz").absolute()),
            "1/2",
        ),
        (
            "https://example.test/features",
            str((output_dir / "features.tsv.gz").absolute()),
            "2/2",
        ),
    ]


def test_handle_download_streams_nonempty_chunks(monkeypatch, tmp_path):
    import requests

    from scarf import downloader

    calls = []
    progress = {}

    class HeadResponse:
        headers = {"content-length": "20000001"}

    class DownloadResponse:
        def iter_content(self, chunk_size):
            calls.append(("iter_content", chunk_size))
            return iter([b"first", b"", b"-second"])

    def fake_head(url, allow_redirects):
        calls.append(("head", url, allow_redirects))
        return HeadResponse()

    def fake_get(url, stream):
        calls.append(("get", url, stream))
        return DownloadResponse()

    def fake_tqdm(iterable, **kwargs):
        progress.update(kwargs)
        return iterable

    monkeypatch.setattr(requests, "head", fake_head)
    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(downloader, "tqdmbar", fake_tqdm)
    output = tmp_path / "data.bin"

    downloader.handle_download(
        "https://example.test/data",
        str(output),
        seq_counter="2/3",
    )

    assert output.read_bytes() == b"first-second"
    assert calls == [
        ("head", "https://example.test/data", True),
        ("get", "https://example.test/data", True),
        ("iter_content", 10_000_000),
    ]
    assert progress == {"total": 3, "desc": "Downloading 2/3"}


def test_handle_download_extracts_tar_archive(monkeypatch, tmp_path):
    import requests

    from scarf import downloader

    archive = io.BytesIO()
    payload = b"archive contents"
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        member = tarfile.TarInfo("nested/data.txt")
        member.size = len(payload)
        tar.addfile(member, io.BytesIO(payload))
    archive_bytes = archive.getvalue()

    class HeadResponse:
        headers = {"content-length": str(len(archive_bytes))}

    class DownloadResponse:
        def iter_content(self, chunk_size):
            assert chunk_size == 10_000_000
            return iter([archive_bytes])

    monkeypatch.setattr(
        requests,
        "head",
        lambda url, allow_redirects: HeadResponse(),
    )
    monkeypatch.setattr(
        requests,
        "get",
        lambda url, stream: DownloadResponse(),
    )
    monkeypatch.setattr(downloader, "tqdmbar", lambda iterable, **kwargs: iterable)
    output = tmp_path / "bundle.tar.gz"

    downloader.handle_download("https://example.test/bundle", str(output))

    assert output.read_bytes() == archive_bytes
    assert (tmp_path / "nested" / "data.txt").read_bytes() == payload


def test_handle_download_propagates_request_errors(monkeypatch, tmp_path):
    import requests

    from scarf import downloader

    class HeadResponse:
        headers = {}

    monkeypatch.setattr(
        requests,
        "head",
        lambda url, allow_redirects: HeadResponse(),
    )

    def fail_get(url, stream):
        raise RuntimeError("request failed")

    monkeypatch.setattr(requests, "get", fail_get)
    output = tmp_path / "missing.bin"

    with pytest.raises(RuntimeError, match="request failed"):
        downloader.handle_download("https://example.test/missing", str(output))
    assert not output.exists()


def test_fetch_dataset_unknown_name_raises(mock_osf_downloader):
    from scarf.downloader import fetch_dataset

    with pytest.raises(KeyError):
        fetch_dataset("this_dataset_does_not_exist", save_path=".")


def test_fetch_dataset_as_zarr_missing_file(mock_osf_downloader, tmp_path):
    from scarf.downloader import fetch_dataset

    fetch_dataset("dataset_0", as_zarr=True, save_path=str(tmp_path))
    assert not any(tmp_path.iterdir())


@pytest.mark.integration
def test_downloader_live_osf():
    from scarf.downloader import OSFdownloader

    osfd = OSFdownloader("zeupv")
    assert len(osfd.datasets) > 15


@pytest.mark.integration
def test_fetch_dataset_live():
    from scarf.downloader import fetch_dataset

    sample = "tenx_5K_pbmc_rnaseq"
    fetch_dataset(sample, as_zarr=True, save_path=full_path(None))
    remove(full_path(sample))
