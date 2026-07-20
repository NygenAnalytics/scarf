import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.readers import CSVReader
from scarf.writers import (
    CSVtoZarr,
    CrToZarr,
    SubsetZarr,
    bed_to_sparse_array,
    subset_assay_zarr,
)


class _FakeCells:
    def __init__(
        self, n_cells: int, columns: dict[str, np.ndarray] | None = None
    ) -> None:
        self.N = n_cells
        self._columns = columns or {}

    def fetch_all(self, key: str) -> np.ndarray:
        return self._columns[key]


class _FakeAssay:
    def __init__(
        self,
        name: str,
        n_cells: int,
        columns: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.name = name
        self.cells = _FakeCells(n_cells, columns)


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


def test_crtozarr_preserves_exact_counts_metadata_and_transpose():
    import pandas as pd
    from scipy.sparse import coo_matrix

    values = np.array(
        [[1, 0, 2], [0, 3, 0], [4, 5, 6]],
        dtype=np.uint16,
    )

    class ExactReader:
        nCells = values.shape[0]
        assayFeats = pd.DataFrame(
            {"RNA": ["Gene Expression", 0, values.shape[1], values.shape[1]]},
            index=["type", "start", "end", "nFeatures"],
        )

        def cell_names(self):
            return ["c1", "c2", "c3"]

        def feature_ids(self, assay_name):
            return ["f1", "f2", "f3"]

        def feature_names(self, assay_name):
            return ["g1", "g2", "g3"]

        def consume(self, batch_size, lines_in_mem):
            for start in range(0, self.nCells, batch_size):
                yield coo_matrix(values[start : start + batch_size])

    store = MemoryStore()
    writer = CrToZarr(
        ExactReader(),
        zarr_loc=store,
        dtype="uint16",
        chunk_size=(2, 2),
    )
    writer.dump(batch_size=2)

    root = zarr.open_group(store=store, mode="r")
    np.testing.assert_array_equal(root["RNA/counts"][:], values)
    np.testing.assert_array_equal(root["RNA/countsT"][:], values.T)
    assert root["RNA/countsT"].attrs["complete"] is True
    np.testing.assert_array_equal(root["cellData/ids"][:], ["c1", "c2", "c3"])
    np.testing.assert_array_equal(root["RNA/featureData/ids"][:], ["f1", "f2", "f3"])


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


def test_loomtozarr_preserves_exact_counts_and_transpose(tmp_path):
    import h5py

    from scarf.readers import LoomReader
    from scarf.writers import LoomToZarr

    values = np.array([[1, 0], [0, 2], [3, 4]], dtype=np.uint16)
    path = tmp_path / "exact.loom"
    with h5py.File(path, mode="w") as handle:
        handle.create_dataset("matrix", data=values.T)
        cells = handle.create_group("col_attrs")
        cells.create_dataset("obs_names", data=np.array([b"c1", b"c2", b"c3"]))
        features = handle.create_group("row_attrs")
        features.create_dataset("var_names", data=np.array([b"g1", b"g2"]))

    reader = LoomReader(str(path))
    store = MemoryStore()
    try:
        writer = LoomToZarr(reader, zarr_loc=store, chunk_size=(2, 2))
        writer.dump(batch_size=2)
    finally:
        reader.h5.close()

    root = zarr.open_group(store=store, mode="r")
    np.testing.assert_array_equal(root["RNA/counts"][:], values)
    np.testing.assert_array_equal(root["RNA/countsT"][:], values.T)
    assert root["RNA/countsT"].attrs["complete"] is True


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
    root = zarr.open_group(fn, mode="r")
    np.testing.assert_array_equal(root["RNA/counts"][:], mat.toarray())
    np.testing.assert_array_equal(root["RNA/countsT"][:], mat.toarray().T)
    assert root["RNA/countsT"].attrs["complete"] is True


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


def test_csv_to_zarr_round_trip(tmp_path):
    csv_path = tmp_path / "counts.csv"
    csv_path.write_text(
        "quality,geneA,geneB,geneC\n"
        "10,1,0,2\n"
        "20,0,3,0\n"
        "30,4,5,6\n"
        "40,7,0,8\n"
        "50,9,10,0\n",
        encoding="utf-8",
    )
    reader = CSVReader(
        str(csv_path),
        cell_data_cols=["quality"],
        batch_size=2,
    )
    store = MemoryStore()
    writer = CSVtoZarr(
        reader,
        zarr_loc=store,
        assay_name="RNA",
        dtype=np.dtype("uint16"),
    )

    writer.dump()

    root = zarr.open_group(store=store, mode="r")
    expected = np.array(
        [
            [1, 0, 2],
            [0, 3, 0],
            [4, 5, 6],
            [7, 0, 8],
            [9, 10, 0],
        ],
        dtype=np.uint16,
    )
    np.testing.assert_array_equal(root["RNA/counts"][:], expected)
    np.testing.assert_array_equal(root["RNA/countsT"][:], expected.T)
    assert root["RNA/countsT"].attrs["complete"] is True
    np.testing.assert_array_equal(
        root["RNA/featureData/ids"][:],
        np.array(["geneA", "geneB", "geneC"]),
    )
    np.testing.assert_array_equal(
        root["cellData/ids"][:],
        np.array(["cell_0", "cell_1", "cell_2", "cell_3", "cell_4"]),
    )
    np.testing.assert_array_equal(
        root["cellData/quality"][:],
        np.array([10, 20, 30, 40, 50]),
    )


def test_subset_assay_zarr_selects_ordered_rows_and_columns():
    store = MemoryStore()
    root = zarr.open_group(store=store, mode="w")
    source = root.create_array(
        "source",
        shape=(5, 4),
        chunks=(2, 2),
        dtype=np.uint16,
        fill_value=0,
    )
    values = np.arange(20, dtype=np.uint16).reshape(5, 4)
    source[:] = values
    cells = np.array([4, 1, 3])
    features = np.array([3, 0])

    result = subset_assay_zarr(
        store,
        in_grp="source",
        out_grp="selected",
        cells_idx=cells,
        feat_idx=features,
        chunk_size=(2, 2),
    )

    selected = root["selected"]
    assert result is None
    assert selected.dtype == np.dtype(np.uint32)
    np.testing.assert_array_equal(
        selected[:],
        values[np.ix_(cells, features)],
    )


def test_bed_to_sparse_array_bins_filters_and_drops_unknown_features(tmp_path):
    bed_path = tmp_path / "fragments.bed"
    bed_path.write_text(
        "# test fragments\n"
        "chr1\t0\t20\tcellB\t2\n"
        "chr1\t100\t120\tcellA\t3\n"
        "chr2\t0\t20\tcellB\t4\n"
        "chr1\t0\t10\tcellC\t5\n"
        "chr1\t10\t20\tcellA\t1\n"
        "chr2\t200\t220\tcellD\t1\n",
        encoding="utf-8",
    )

    matrix, cell_ids, feature_ids = bed_to_sparse_array(
        str(bed_path),
        bin_size=100,
        chrom_sizes={"chr1": 199, "chr2": 99},
        min_counts_per_cell=3,
        read_chunk_size=2,
        disable_tqdm=True,
    )

    assert cell_ids.tolist() == ["cellB", "cellA", "cellC"]
    assert feature_ids.tolist() == ["chr1_0", "chr1_1", "chr2_0"]
    np.testing.assert_array_equal(
        matrix.toarray(),
        np.array(
            [
                [2, 0, 4],
                [1, 3, 0],
                [5, 0, 0],
            ]
        ),
    )


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


def test_subset_zarr_rejects_invalid_assay_inputs():
    with pytest.raises(TypeError, match="should be a list"):
        SubsetZarr._check_assays("RNA")
    with pytest.raises(ValueError, match="actual assay objects"):
        SubsetZarr._check_assays([object()])
    with pytest.raises(ValueError, match="same numer of cells"):
        SubsetZarr._check_assays(
            [
                _FakeAssay("RNA", 3),
                _FakeAssay("ATAC", 4),
            ]
        )


def test_subset_zarr_requires_cell_key_or_indices():
    subset = object.__new__(SubsetZarr)
    subset.assays = [_FakeAssay("RNA", 3)]

    with pytest.raises(ValueError, match="cannot be None"):
        subset._check_idx(None, None)


@pytest.mark.parametrize(
    ("cell_idx", "message"),
    [
        (np.array([0.5]), "integer type"),
        (np.array([3]), "max value"),
    ],
)
def test_subset_zarr_rejects_invalid_explicit_indices(cell_idx, message):
    subset = object.__new__(SubsetZarr)
    subset.assays = [_FakeAssay("RNA", 3)]

    with pytest.raises(ValueError, match=message):
        subset._check_idx(None, cell_idx)


def test_subset_zarr_validates_cell_key():
    subset = object.__new__(SubsetZarr)
    subset.assays = [_FakeAssay("RNA", 3)]
    with pytest.raises(ValueError, match="was not found"):
        subset._check_idx("selected", None)

    subset.assays = [
        _FakeAssay("RNA", 3, {"selected": np.array([1, 0, 1])}),
    ]
    with pytest.raises(ValueError, match="not of boolean type"):
        subset._check_idx("selected", None)


def test_subset_zarr_resolves_consistent_cell_key():
    selected = np.array([True, False, True, False])
    subset = object.__new__(SubsetZarr)
    subset.assays = [
        _FakeAssay("RNA", 4, {"selected": selected}),
        _FakeAssay("ATAC", 4, {"selected": selected.copy()}),
    ]

    np.testing.assert_array_equal(
        subset._check_idx("selected", None),
        np.array([0, 2]),
    )


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


def test_h5adtozarr_forwards_storage_resources_and_chunk_controls(monkeypatch):
    from scarf.writers import H5adToZarr

    captured = {}

    def fake_load_zarr(zarr_loc, mode, storage_options=None):
        captured["store"] = (zarr_loc, mode, storage_options)
        return zarr.open_group(store=MemoryStore(), mode="w")

    def fake_budget(mem_budget, nthreads, working_copies):
        captured["budget"] = (mem_budget, nthreads, working_copies)

    def fake_create_count(**kwargs):
        captured["count"] = kwargs

    monkeypatch.setattr("scarf.writers.load_zarr", fake_load_zarr)
    monkeypatch.setattr("scarf.writers._apply_budget_override", fake_budget)
    monkeypatch.setattr("scarf.writers.create_zarr_count_assay", fake_create_count)
    monkeypatch.setattr(H5adToZarr, "_ini_cell_data", lambda self: None)
    monkeypatch.setattr(H5adToZarr, "_ini_feature_data", lambda self: None)

    class FakeH5ad:
        nCells = 1
        matrixDtype = np.dtype("uint16")

        def feat_ids(self):
            return np.array(["f1"])

        def feat_names(self):
            return np.array(["g1"])

    H5adToZarr(
        FakeH5ad(),
        zarr_loc="s3://bucket/out.zarr",
        storage_options={"access_key_id": "id"},
        mem_budget="2G",
        nthreads=3,
        working_copies=2,
        targetChunkBytes=4096,
        minFeatureChunk=16,
        maxFeatureChunk=128,
    )

    assert captured["store"] == (
        "s3://bucket/out.zarr",
        "w",
        {"access_key_id": "id"},
    )
    assert captured["budget"] == ("2G", 3, 2)
    assert captured["count"]["targetChunkBytes"] == 4096
    assert captured["count"]["minFeatureChunk"] == 16
    assert captured["count"]["maxFeatureChunk"] == 128
