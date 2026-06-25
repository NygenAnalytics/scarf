import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.merge import AssayMerge
from scarf.storage.zarr_store import is_local_zarr_path
from scarf.utils import load_zarr
from scarf.writers import CrToZarr, SubsetZarr


def test_is_local_zarr_path():
    assert is_local_zarr_path("/tmp/foo.zarr") is True
    assert is_local_zarr_path("s3://bucket/path") is False
    assert is_local_zarr_path("gs://bucket/path") is False
    assert is_local_zarr_path(MemoryStore()) is False


def test_subset_zarr_local_path_guard(tmp_path):
    existing = tmp_path / "out.zarr"
    existing.mkdir()
    subset = object.__new__(SubsetZarr)
    subset.overFn = False
    subset.storage_options = None
    with pytest.raises(ValueError, match="already exists"):
        SubsetZarr._check_files(subset, str(existing))


def test_subset_zarr_store_skips_local_guard():
    mem = MemoryStore()
    subset = object.__new__(SubsetZarr)
    subset.overFn = False
    subset.storage_options = None
    root = SubsetZarr._check_files(subset, mem)
    assert isinstance(root, zarr.Group)


def test_subset_zarr_remote_uri_skips_local_guard(monkeypatch):
    calls = []

    def fake_load_zarr(zarr_loc, mode, storage_options=None):
        calls.append((zarr_loc, mode, storage_options))
        return zarr.open_group(store=MemoryStore(), mode="w")

    monkeypatch.setattr("scarf.writers.load_zarr", fake_load_zarr)
    subset = object.__new__(SubsetZarr)
    subset.overFn = False
    subset.storage_options = {"access_key_id": "key"}
    SubsetZarr._check_files(subset, "s3://bucket/out.zarr")
    assert calls == [("s3://bucket/out.zarr", "w", {"access_key_id": "key"})]


def test_assay_merge_store_skips_local_exists_guard(monkeypatch):
    calls = []

    def fake_load_zarr(zarr_loc, mode, storage_options=None):
        calls.append((zarr_loc, mode, storage_options))
        if mode == "r":
            raise FileNotFoundError("missing")
        return zarr.open_group(store=MemoryStore(), mode="w")

    monkeypatch.setattr("scarf.merge.load_zarr", fake_load_zarr)
    merge = object.__new__(AssayMerge)
    merge.outWorkspace = None
    merge.storage_options = {"region": "us-east-1"}
    root = AssayMerge._use_existing_zarr(merge, MemoryStore(), "RNA", False)
    assert isinstance(root, zarr.Group)
    assert calls[-1] == (calls[-1][0], "w", {"region": "us-east-1"})


def test_crtozarr_forwards_storage_options(monkeypatch):
    captured = {}

    def fake_load_zarr(zarr_loc, mode, storage_options=None, synchronizer=None):
        captured["zarr_loc"] = zarr_loc
        captured["mode"] = mode
        captured["storage_options"] = storage_options
        return zarr.open_group(store=MemoryStore(), mode="w")

    monkeypatch.setattr("scarf.writers.load_zarr", fake_load_zarr)
    monkeypatch.setattr("scarf.writers.create_cell_data", lambda **kwargs: None)
    monkeypatch.setattr(
        "scarf.writers.create_zarr_count_assay", lambda **kwargs: None
    )

    class FakeCr:
        def cell_names(self):
            return ["c1"]

        @property
        def assayFeats(self):
            import pandas as pd

            return pd.DataFrame({"RNA": [0, 1]})

        @property
        def nCells(self):
            return 1

        def feature_ids(self, assay_name):
            return ["f1"]

        def feature_names(self, assay_name):
            return ["f1"]

    CrToZarr(
        FakeCr(),
        zarr_loc="s3://bucket/out.zarr",
        storage_options={"access_key_id": "id"},
    )
    assert captured["storage_options"] == {"access_key_id": "id"}


def test_load_zarr_forwards_storage_options_to_make_store(monkeypatch):
    captured = {}

    def fake_make_store(location, storage_options=None, read_only=False):
        captured["location"] = location
        captured["storage_options"] = storage_options
        captured["read_only"] = read_only
        return MemoryStore()

    monkeypatch.setattr("scarf.storage.zarr_store.make_store", fake_make_store)
    monkeypatch.setattr(
        "scarf.storage.zarr_store.configure_zarr_io_for_profile", lambda: None
    )
    monkeypatch.setattr("zarr.open_group", lambda **kwargs: object())
    load_zarr(
        "s3://bucket/path",
        mode="r",
        storage_options={"secret_access_key": "secret"},
    )
    assert captured["storage_options"] == {"secret_access_key": "secret"}
    assert captured["read_only"] is True
