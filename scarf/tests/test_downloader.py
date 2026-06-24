import pytest

from . import full_path, remove


@pytest.fixture
def mock_osf_downloader(monkeypatch):
    from .. import downloader

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


def test_show_available_datasets(mock_osf_downloader):
    from ..downloader import show_available_datasets

    show_available_datasets()


def test_fetch_dataset(bastidas_ponce_data):
    import os

    assert os.path.isfile(bastidas_ponce_data)


def test_downloader_as_zarr(mock_osf_downloader, monkeypatch, tmp_path):
    from .. import downloader

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


@pytest.mark.integration
def test_downloader_live_osf():
    from ..downloader import OSFdownloader

    osfd = OSFdownloader("zeupv")
    assert len(osfd.datasets) > 15


@pytest.mark.integration
def test_fetch_dataset_live():
    from ..downloader import fetch_dataset

    sample = "tenx_5K_pbmc_rnaseq"
    fetch_dataset(sample, as_zarr=True, save_path=full_path(None))
    remove(full_path(sample))
