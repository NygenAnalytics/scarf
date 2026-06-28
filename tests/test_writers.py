import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.writers import CrToZarr, SubsetZarr


def test_crtozarr(crh5_reader, tmp_path):
    from scarf.writers import CrToZarr

    fn = str(tmp_path / "dummy_1K_pbmc_citeseq.zarr")
    writer = CrToZarr(crh5_reader, zarr_loc=fn)
    writer.dump()


def test_crtozarr_fromdir(crdir_reader, tmp_path):
    from scarf.writers import CrToZarr

    fn = str(tmp_path / "1K_pbmc_citeseq_dir.zarr")
    writer = CrToZarr(crdir_reader, zarr_loc=fn)
    writer.dump()


def test_h5adtozarr(h5ad_reader, tmp_path):
    from scarf.writers import H5adToZarr

    fn = str(tmp_path / "bastidas.zarr")
    writer = H5adToZarr(h5ad_reader, zarr_loc=fn)
    writer.dump()


def test_loomtozarr(loom_reader, tmp_path):
    from scarf.writers import LoomToZarr

    fn = str(tmp_path / "sympathetic.zarr")
    writer = LoomToZarr(loom_reader, zarr_loc=fn)
    writer.dump()


def test_sparsetozarr(tmp_path):
    from scipy.sparse import csr_matrix

    from scarf.writers import SparseToZarr

    cols = [1, 3, 8, 2, 3, 1, 2, 8, 9]
    rows = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    data = [1, 10, 15, 10, 20, 2, 3, 1, 5]
    mat = (data, (rows, cols))
    mat = csr_matrix(mat, shape=(3, 10))

    fn = str(tmp_path / "dummy_sparse.zarr")

    writer = SparseToZarr(
        mat,
        zarr_loc=fn,
        cell_ids=[f"cell_{x}" for x in range(3)],
        feature_ids=[f"feat_{x}" for x in range(10)],
    )
    writer.dump()


def test_sparsetozarr_sharded_layout(tmp_path):
    import zarr
    from scipy.sparse import csr_matrix

    from scarf.writers import SparseToZarr

    n_cells, n_feats = 5000, 200
    rng = np.random.default_rng(0)
    rows = rng.integers(0, n_cells, size=50_000)
    cols = rng.integers(0, n_feats, size=50_000)
    data = np.ones(50_000, dtype=np.uint32)
    mat = csr_matrix((data, (rows, cols)), shape=(n_cells, n_feats))
    fn = str(tmp_path / "dummy_sparse_sharded.zarr")
    writer = SparseToZarr(
        mat,
        zarr_loc=fn,
        cell_ids=[f"cell_{x}" for x in range(n_cells)],
        feature_ids=[f"feat_{x}" for x in range(n_feats)],
    )
    writer.dump()
    store = zarr.open_group(fn, mode="r")
    counts = store["RNA/counts"]
    assert counts.shape == (n_cells, n_feats)
    assert counts.metadata.shards is not None
    assert int(counts[...].sum()) > 0


def test_v2_fixture_read_only(datastore):
    assert datastore.RNA.rawData.shape[0] > 0


def test_to_h5ad(datastore, tmp_path):
    from scarf.writers import to_h5ad

    fn = str(tmp_path / "test_1K_pbmc_citeseq.h5ad")
    to_h5ad(datastore.RNA, fn)


def test_to_mtx(datastore, tmp_path):
    from scarf.writers import to_mtx

    fn = str(tmp_path / "test_1K_pbmc_citeseq_dir")
    to_mtx(datastore.RNA, fn)


def test_zarr_subset(datastore, tmp_path):
    zarr_path = str(tmp_path / "subset.zarr")
    writer = SubsetZarr(
        zarr_loc=zarr_path, assays=[datastore.RNA], cell_idx=np.array([1, 10, 100, 500])
    )
    writer.dump()


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


def test_crtozarr_forwards_storage_options(monkeypatch):
    captured = {}

    def fake_load_zarr(zarr_loc, mode, storage_options=None, synchronizer=None):
        captured["zarr_loc"] = zarr_loc
        captured["mode"] = mode
        captured["storage_options"] = storage_options
        return zarr.open_group(store=MemoryStore(), mode="w")

    monkeypatch.setattr("scarf.writers.load_zarr", fake_load_zarr)
    monkeypatch.setattr("scarf.writers.create_cell_data", lambda **kwargs: None)
    monkeypatch.setattr("scarf.writers.create_zarr_count_assay", lambda **kwargs: None)

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
