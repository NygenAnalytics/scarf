import io
import tarfile

import pytest

from . import full_path, remove


class MockResponse:
    def __init__(
        self,
        *,
        status_code=200,
        json_data=None,
        text="",
        headers=None,
        chunks=(),
    ):
        self.status_code = status_code
        self.json_data = json_data
        self.text = text
        self.headers = headers or {}
        self.chunks = chunks
        self.raise_calls = 0
        self.closed = False

    def raise_for_status(self):
        import requests

        self.raise_calls += 1
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"HTTP {self.status_code}",
                response=self,
            )

    def json(self):
        if isinstance(self.json_data, BaseException):
            raise self.json_data
        return self.json_data

    def iter_content(self, chunk_size):
        self.chunk_size = chunk_size
        for chunk in self.chunks:
            if isinstance(chunk, BaseException):
                raise chunk
            yield chunk

    def close(self):
        self.closed = True


@pytest.fixture
def mock_osf_downloader(monkeypatch):
    from scarf.readers import datasets as downloader

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
    from scarf.readers.datasets import OSFdownloader

    datasets = {
        "zeta": ("zeta-id", "figshare"),
        "alpha": ("alpha-id", "osfstorage"),
    }
    monkeypatch.setattr(
        OSFdownloader,
        "_populate_storages",
        lambda self: ["osfstorage", "figshare"],
    )
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


def test_osf_downloader_uses_only_connected_storages(monkeypatch):
    from scarf.readers.datasets import OSFdownloader

    client = object.__new__(OSFdownloader)
    client.url = "https://api.example.test/files/"
    nodes = [
        {
            "attributes": {"provider": "osfstorage"},
            "relationships": {"root_folder": {"links": {}}},
        },
        {
            "attributes": {"provider": "figshare"},
            "relationships": {},
        },
    ]
    calls = []

    def fake_get_all_pages(storage, endpoint, initial_url):
        calls.append((storage, endpoint, initial_url))
        return nodes

    monkeypatch.setattr(client, "_get_all_pages", fake_get_all_pages)

    assert client._populate_storages() == ["osfstorage"]
    assert calls == [("", "", client.url)]


def test_osf_downloader_builds_api_urls(monkeypatch):
    from scarf.readers.datasets import OSFdownloader

    calls = []
    responses = []

    def fake_get(url, *, stream, timeout):
        calls.append((url, stream, timeout))
        response = MockResponse(json_data={"data": []})
        responses.append(response)
        return response

    import requests

    monkeypatch.setattr(requests, "get", fake_get)
    client = object.__new__(OSFdownloader)
    client.url = "https://api.example.test/files/"

    assert client.get_json("osfstorage", "folder", None) == {"data": []}
    assert client.get_json("figshare", "", "https://next.example.test") == {"data": []}
    assert calls == [
        (
            "https://api.example.test/files/osfstorage/folder/",
            False,
            (5.0, 30.0),
        ),
        ("https://next.example.test", False, (5.0, 30.0)),
    ]
    assert all(response.raise_calls == 1 for response in responses)
    assert all(response.closed for response in responses)


def test_osf_downloader_retries_transport_errors(monkeypatch):
    import requests

    from scarf.readers.datasets import OSFdownloader

    client = object.__new__(OSFdownloader)
    calls = []
    sleeps = []
    response = MockResponse(json_data={"data": []})

    def fake_get(url, *, stream, timeout):
        calls.append((url, stream, timeout))
        if len(calls) == 1:
            raise requests.ConnectionError("connection dropped")
        return response

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr("scarf.readers.datasets.time.sleep", sleeps.append)
    client.url = "https://api.example.test/files/"

    assert client.get_json("osfstorage", "", None) == {"data": []}
    assert calls == [
        ("https://api.example.test/files/osfstorage/", False, (5.0, 30.0)),
        ("https://api.example.test/files/osfstorage/", False, (5.0, 30.0)),
    ]
    assert sleeps == [1.0]
    assert response.closed


@pytest.mark.parametrize("status_code", [429, 500, 503])
def test_osf_downloader_retries_retryable_responses(
    monkeypatch,
    status_code,
):
    import requests

    from scarf.readers.datasets import OSFdownloader

    failed = MockResponse(status_code=status_code)
    succeeded = MockResponse(json_data={"data": []})
    responses = iter([failed, succeeded])
    sleeps = []
    calls = []

    def fake_get(url, *, stream, timeout):
        calls.append((url, stream, timeout))
        return next(responses)

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr("scarf.readers.datasets.time.sleep", sleeps.append)
    client = object.__new__(OSFdownloader)
    client.url = "https://api.example.test/files/"

    assert client.get_json("osfstorage", "", None) == {"data": []}
    assert len(calls) == 2
    assert sleeps == [1.0]
    assert failed.raise_calls == 1
    assert failed.closed
    assert succeeded.closed


@pytest.mark.parametrize("status_code", [400, 404])
def test_osf_downloader_does_not_retry_terminal_responses(
    monkeypatch,
    status_code,
):
    import requests

    from scarf.readers.datasets import OSFdownloader

    response = MockResponse(status_code=status_code)
    calls = []
    sleeps = []

    def fake_get(url, *, stream, timeout):
        calls.append((url, stream, timeout))
        return response

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr("scarf.readers.datasets.time.sleep", sleeps.append)
    client = object.__new__(OSFdownloader)
    client.url = "https://api.example.test/files/"

    with pytest.raises(requests.HTTPError, match=f"HTTP {status_code}"):
        client.get_json("osfstorage", "", None)
    assert len(calls) == 1
    assert sleeps == []
    assert response.closed


def test_osf_downloader_bounds_retryable_responses(monkeypatch):
    import requests

    from scarf.readers.datasets import OSFdownloader

    responses = [MockResponse(status_code=503) for _ in range(5)]
    response_iterator = iter(responses)
    sleeps = []
    calls = []

    def fake_get(url, *, stream, timeout):
        calls.append((url, stream, timeout))
        return next(response_iterator)

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr("scarf.readers.datasets.time.sleep", sleeps.append)
    client = object.__new__(OSFdownloader)
    client.url = "https://api.example.test/files/"

    with pytest.raises(requests.HTTPError, match="HTTP 503"):
        client.get_json("osfstorage", "", None)
    assert len(calls) == 5
    assert sleeps == [1.0, 2.0, 4.0, 8.0]
    assert all(response.closed for response in responses)


def test_osf_downloader_rejects_malformed_json_without_retry(monkeypatch):
    import requests

    from scarf.readers.datasets import OSFdownloader

    malformed = requests.exceptions.JSONDecodeError("invalid response", "", 0)
    response = MockResponse(json_data=malformed)
    calls = []
    sleeps = []

    def fake_get(url, *, stream, timeout):
        calls.append((url, stream, timeout))
        return response

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr("scarf.readers.datasets.time.sleep", sleeps.append)
    client = object.__new__(OSFdownloader)
    client.url = "https://api.example.test/files/"

    with pytest.raises(requests.exceptions.JSONDecodeError):
        client.get_json("osfstorage", "", None)
    assert len(calls) == 1
    assert sleeps == []
    assert response.closed


def test_osf_downloader_paginates_complete_responses(monkeypatch):
    from scarf.readers.datasets import OSFdownloader

    client = object.__new__(OSFdownloader)
    responses = iter(
        [
            {"data": [{"id": "first"}], "links": {"next": "next-page"}},
            {"data": [{"id": "second"}], "links": {"next": None}},
        ]
    )
    calls = []

    def fake_get_json(storage, endpoint, url):
        calls.append((storage, endpoint, url))
        return next(responses)

    monkeypatch.setattr(client, "get_json", fake_get_json)

    assert client.get_all_pages("osfstorage", "folder") == [
        {"id": "first"},
        {"id": "second"},
    ]
    assert calls == [
        ("osfstorage", "folder", None),
        ("osfstorage", "folder", "next-page"),
    ]


def test_osf_downloader_fails_malformed_later_page(monkeypatch):
    from scarf.readers.datasets import OSFdownloader

    client = object.__new__(OSFdownloader)
    responses = iter(
        [
            {"data": [{"id": "first"}], "links": {"next": "next-page"}},
            {"data": [{"id": "second"}]},
        ]
    )
    calls = []

    def fake_get_json(storage, endpoint, url):
        calls.append((storage, endpoint, url))
        return next(responses)

    monkeypatch.setattr(client, "get_json", fake_get_json)

    with pytest.raises(ValueError, match="Malformed OSF pagination response"):
        client.get_all_pages("osfstorage", "folder")
    assert calls == [
        ("osfstorage", "folder", None),
        ("osfstorage", "folder", "next-page"),
    ]


def test_osf_downloader_discovers_datasets_and_files(monkeypatch):
    from scarf.readers.datasets import OSFdownloader

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

    from scarf.readers.datasets import OSFdownloader

    client = object.__new__(OSFdownloader)
    client.sourceFile = "sources-id"
    monkeypatch.setattr(
        client,
        "_get_files_for_node",
        lambda storage, file_id: {"sources.csv": "https://example.test/sources.csv"},
    )

    response = MockResponse(
        text=(
            "id,title,url\n"
            "alpha,Alpha dataset,https://example.test/alpha\n"
            "beta,Beta dataset,https://example.test/beta\n"
        ),
    )

    calls = []

    def fake_get(url, *, stream, timeout):
        calls.append((url, stream, timeout))
        return response

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
    assert calls == [
        ("https://example.test/sources.csv", False, (5.0, 30.0)),
    ]
    assert response.raise_calls == 1
    assert response.closed


def test_osf_downloader_requires_source_file(monkeypatch):
    from scarf.readers.datasets import OSFdownloader

    client = object.__new__(OSFdownloader)
    client.sourceFile = "sources-id"
    attempts = []

    def missing_sources(storage, file_id):
        attempts.append((storage, file_id))
        return {}

    monkeypatch.setattr(client, "_get_files_for_node", missing_sources)

    with pytest.raises(KeyError, match="sources file was not found"):
        client._populate_sources()
    assert attempts == [("osfstorage", "sources-id")]


def test_show_available_datasets(mock_osf_downloader):
    from scarf.readers.datasets import show_available_datasets

    show_available_datasets()


def test_fetch_dataset(bastidas_ponce_data):
    import os

    assert os.path.isfile(bastidas_ponce_data)


def test_downloader_as_zarr(mock_osf_downloader, monkeypatch, tmp_path):
    from scarf.readers import datasets as downloader

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
    from scarf.readers import datasets as downloader

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


@pytest.mark.parametrize("dataset_name", ["../outside", "/outside", r"..\outside"])
def test_fetch_dataset_rejects_unsafe_dataset_name(
    monkeypatch,
    tmp_path,
    dataset_name,
):
    from scarf.readers import datasets as downloader

    class UnsafeCatalog:
        def get_dataset_file_ids(self, requested_name):
            assert requested_name == dataset_name
            return {"data.h5ad": "https://example.test/data"}

    monkeypatch.setattr(downloader, "osfd", UnsafeCatalog())
    monkeypatch.setattr(
        downloader,
        "handle_download",
        lambda *args, **kwargs: pytest.fail("unsafe destination reached downloader"),
    )

    with pytest.raises(ValueError, match="Unsafe dataset name"):
        downloader.fetch_dataset(dataset_name, save_path=str(tmp_path / "downloads"))

    assert not (tmp_path / "outside").exists()


@pytest.mark.parametrize(
    "filename", ["../outside.h5ad", "/outside.h5ad", r"..\outside.h5ad"]
)
def test_fetch_dataset_rejects_unsafe_filename(monkeypatch, tmp_path, filename):
    from scarf.readers import datasets as downloader

    class UnsafeCatalog:
        def get_dataset_file_ids(self, dataset_name):
            assert dataset_name == "safe"
            return {filename: "https://example.test/data"}

    monkeypatch.setattr(downloader, "osfd", UnsafeCatalog())
    monkeypatch.setattr(
        downloader,
        "handle_download",
        lambda *args, **kwargs: pytest.fail("unsafe destination reached downloader"),
    )

    with pytest.raises(ValueError, match="Unsafe dataset filename"):
        downloader.fetch_dataset("safe", save_path=str(tmp_path / "downloads"))

    assert not (tmp_path / "outside.h5ad").exists()


def test_handle_download_streams_nonempty_chunks(monkeypatch, tmp_path):
    import requests

    from scarf.readers import datasets as downloader

    calls = []
    progress = {}
    response = MockResponse(chunks=[b"first", b"", b"-second"])

    def fake_get(url, *, stream, timeout):
        calls.append((url, stream, timeout))
        return response

    def fake_tqdm(iterable, **kwargs):
        progress.update(kwargs)
        return iterable

    monkeypatch.setattr(
        requests,
        "head",
        lambda *args, **kwargs: pytest.fail("HEAD must not be used"),
    )
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
        ("https://example.test/data", True, (5.0, 30.0)),
    ]
    assert response.chunk_size == 10_000_000
    assert response.raise_calls == 1
    assert response.closed
    assert progress == {"total": None, "desc": "Downloading 2/3"}
    assert not (tmp_path / "data.bin.partial").exists()


def test_handle_download_extracts_tar_archive(monkeypatch, tmp_path):
    import requests

    from scarf.readers import datasets as downloader

    archive = io.BytesIO()
    payload = b"archive contents"
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        member = tarfile.TarInfo("nested/data.txt")
        member.size = len(payload)
        tar.addfile(member, io.BytesIO(payload))
    archive_bytes = archive.getvalue()

    response = MockResponse(
        headers={"content-length": str(len(archive_bytes))},
        chunks=[archive_bytes],
    )

    calls = []

    def fake_get(url, *, stream, timeout):
        calls.append((url, stream, timeout))
        return response

    monkeypatch.setattr(
        requests,
        "get",
        fake_get,
    )
    monkeypatch.setattr(downloader, "tqdmbar", lambda iterable, **kwargs: iterable)
    output = tmp_path / "bundle.tar.gz"

    downloader.handle_download("https://example.test/bundle", str(output))

    assert output.read_bytes() == archive_bytes
    assert (tmp_path / "nested" / "data.txt").read_bytes() == payload
    assert calls == [
        ("https://example.test/bundle", True, (5.0, 30.0)),
    ]
    assert response.chunk_size == 10_000_000
    assert response.closed
    assert not any(".extract-" in path.name for path in tmp_path.iterdir())
    assert not any(".backup-" in path.name for path in tmp_path.iterdir())


def test_handle_download_retries_interrupted_stream_and_preserves_destination(
    monkeypatch,
    tmp_path,
):
    import requests

    from scarf.readers import datasets as downloader

    responses = [
        MockResponse(
            headers={"content-length": "8"},
            chunks=[b"partial", requests.ConnectionError("stream interrupted")],
        )
        for _ in range(5)
    ]
    response_iterator = iter(responses)
    calls = []
    sleeps = []

    def fake_get(url, *, stream, timeout):
        calls.append((url, stream, timeout))
        return next(response_iterator)

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(downloader.time, "sleep", sleeps.append)
    monkeypatch.setattr(downloader, "tqdmbar", lambda iterable, **kwargs: iterable)
    output = tmp_path / "data.bin"
    output.write_bytes(b"existing")

    with pytest.raises(requests.ConnectionError, match="stream interrupted"):
        downloader.handle_download("https://example.test/data", str(output))

    assert len(calls) == 5
    assert sleeps == [1.0, 2.0, 4.0, 8.0]
    assert all(response.closed for response in responses)
    assert output.read_bytes() == b"existing"
    assert not (tmp_path / "data.bin.partial").exists()


def test_handle_download_rejects_length_mismatch_and_preserves_destination(
    monkeypatch,
    tmp_path,
):
    import requests

    from scarf.readers import datasets as downloader

    response = MockResponse(
        headers={"content-length": "4"},
        chunks=[b"new"],
    )
    calls = []
    sleeps = []

    def fake_get(url, *, stream, timeout):
        calls.append((url, stream, timeout))
        return response

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(downloader.time, "sleep", sleeps.append)
    monkeypatch.setattr(downloader, "tqdmbar", lambda iterable, **kwargs: iterable)
    output = tmp_path / "data.bin"
    output.write_bytes(b"existing")

    with pytest.raises(OSError, match="Downloaded 3 bytes, expected 4"):
        downloader.handle_download("https://example.test/data", str(output))

    assert len(calls) == 5
    assert sleeps == [1.0, 2.0, 4.0, 8.0]
    assert response.closed
    assert output.read_bytes() == b"existing"
    assert not (tmp_path / "data.bin.partial").exists()


def test_handle_download_rejects_unsafe_tar_member(monkeypatch, tmp_path):
    import requests

    from scarf.readers import datasets as downloader

    archive = io.BytesIO()
    payload = b"unsafe"
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        member = tarfile.TarInfo("../outside.txt")
        member.size = len(payload)
        tar.addfile(member, io.BytesIO(payload))
    archive_bytes = archive.getvalue()
    response = MockResponse(
        headers={"content-length": str(len(archive_bytes))},
        chunks=[archive_bytes],
    )

    def fake_get(url, *, stream, timeout):
        return response

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(downloader, "tqdmbar", lambda iterable, **kwargs: iterable)
    output = tmp_path / "bundle.tar.gz"
    output.write_bytes(b"existing archive")

    with pytest.raises(ValueError, match="Unsafe tar archive member"):
        downloader.handle_download("https://example.test/bundle", str(output))

    assert output.read_bytes() == b"existing archive"
    assert not (tmp_path / "outside.txt").exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["bundle.tar.gz"]


def test_fetch_dataset_unknown_name_raises(mock_osf_downloader):
    from scarf.readers.datasets import fetch_dataset

    with pytest.raises(KeyError):
        fetch_dataset("this_dataset_does_not_exist", save_path=".")


def test_fetch_dataset_as_zarr_missing_file(mock_osf_downloader, tmp_path):
    from scarf.readers.datasets import fetch_dataset

    fetch_dataset("dataset_0", as_zarr=True, save_path=str(tmp_path))
    assert not any(tmp_path.iterdir())


@pytest.mark.integration
def test_downloader_live_osf():
    from scarf.readers.datasets import OSFdownloader

    osfd = OSFdownloader("zeupv")
    assert len(osfd.datasets) > 15


@pytest.mark.integration
def test_fetch_dataset_live():
    from scarf.readers.datasets import fetch_dataset

    sample = "tenx_5K_pbmc_rnaseq"
    fetch_dataset(sample, as_zarr=True, save_path=full_path(None))
    remove(full_path(sample))
