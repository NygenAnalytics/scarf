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
        matrix_dtype = values.dtype
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

        def max_window_nnz(self, window_rows):
            width = min(window_rows, self.nCells)
            return max(
                np.count_nonzero(values[start : start + width])
                for start in range(self.nCells - width + 1)
            )

        def producer_staging_bytes(self, batch_size, lines_in_mem):
            return 0

    store = MemoryStore()
    writer = CrToZarr(
        ExactReader(),
        zarr_loc=store,
        dtype="uint16",
        targetChunkBytes=12,
        targetShardBytes=12,
    )
    writer.dump(batch_size=2)

    root = zarr.open_group(store=store, mode="r")
    np.testing.assert_array_equal(root["RNA/counts"][:], values)
    assert "countsT" not in root["RNA"]
    np.testing.assert_array_equal(root["cellData/ids"][:], ["c1", "c2", "c3"])
    np.testing.assert_array_equal(root["RNA/featureData/ids"][:], ["f1", "f2", "f3"])


def test_h5adtozarr(h5ad_reader, tmp_path):
    from scarf.writers import H5adToZarr

    fn = str(tmp_path / "bastidas.zarr")
    writer = H5adToZarr(h5ad_reader, zarr_loc=fn)
    writer.dump()


def test_h5adtozarr_splits_noncontiguous_feature_types():
    import tempfile
    from pathlib import Path

    import h5py
    from scipy.sparse import csr_matrix

    from scarf.readers import H5adReader
    from scarf.writers import H5adToZarr

    values = np.array(
        [
            [1, 2, 0, 3],
            [4, 0, 5, 0],
            [0, 6, 7, 8],
        ],
        dtype=np.uint16,
    )
    matrix = csr_matrix(values)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "multi.h5ad"
        with h5py.File(path, mode="w") as h5:
            sparse = h5.create_group("X")
            sparse.attrs["encoding-type"] = "csr_matrix"
            sparse.attrs["shape"] = matrix.shape
            sparse.create_dataset("data", data=matrix.data)
            sparse.create_dataset("indices", data=matrix.indices)
            sparse.create_dataset("indptr", data=matrix.indptr)

            obs = h5.create_group("obs")
            obs.create_dataset("_index", data=np.array([b"c1", b"c2", b"c3"]))
            obs.create_dataset("batch", data=np.array([b"A", b"A", b"B"]))

            var = h5.create_group("var")
            var.create_dataset(
                "_index",
                data=np.array([b"f1", b"a1", b"f2", b"a2"]),
            )
            var.create_dataset(
                "feature_name",
                data=np.array([b"g1", b"p1", b"g2", b"p2"]),
            )
            var.create_dataset(
                "feature_types",
                data=np.array(
                    [
                        b"Gene Expression",
                        b"Antibody Capture",
                        b"Gene Expression",
                        b"Antibody Capture",
                    ]
                ),
            )
            var.create_dataset(
                "chromosome",
                data=np.array([b"1", b"na", b"2", b"na"]),
            )

        reader = H5adReader(str(path), feature_name_key="feature_name")
        store = MemoryStore()
        try:
            assert tuple(
                reader.assay_feature_slices(
                    "feature_types",
                    {"Antibody Capture": "HTO"},
                )
            ) == ("RNA", "HTO")
            writer = H5adToZarr(
                reader,
                zarr_loc=store,
                assay_name="ignored",
                assay_split_key="feature_types",
                targetChunkBytes=8,
                targetShardBytes=8,
            )
            writer.dump(batch_size=2)
        finally:
            reader.h5.close()

    root = zarr.open_group(store=store, mode="r")
    assert set(root.group_keys()) == {"cellData", "RNA", "ADT"}
    np.testing.assert_array_equal(root["RNA/counts"][:], values[:, [0, 2]])
    np.testing.assert_array_equal(root["ADT/counts"][:], values[:, [1, 3]])
    assert "countsT" not in root["RNA"]
    assert "countsT" not in root["ADT"]
    np.testing.assert_array_equal(root["cellData/batch"][:], ["A", "A", "B"])
    np.testing.assert_array_equal(root["RNA/featureData/ids"][:], ["f1", "f2"])
    np.testing.assert_array_equal(root["ADT/featureData/ids"][:], ["a1", "a2"])
    np.testing.assert_array_equal(
        root["RNA/featureData/chromosome"][:],
        ["1", "2"],
    )
    np.testing.assert_array_equal(
        root["ADT/featureData/feature_types"][:],
        ["Antibody Capture", "Antibody Capture"],
    )


def _write_h5ad(
    path,
    values: np.ndarray,
    *,
    encoding: str = "csr",
    feature_types: list[bytes] | None = None,
):
    """Write a minimal AnnData file with the requested matrix encoding."""
    import h5py
    from scipy.sparse import csc_matrix, csr_matrix

    n_cells, n_feats = values.shape
    with h5py.File(path, mode="w") as h5:
        if encoding == "dense":
            h5.create_dataset("X", data=values)
        else:
            matrix = csr_matrix(values) if encoding == "csr" else csc_matrix(values)
            group = h5.create_group("X")
            group.attrs["encoding-type"] = f"{encoding}_matrix"
            group.attrs["shape"] = values.shape
            group.create_dataset("data", data=matrix.data)
            group.create_dataset("indices", data=matrix.indices)
            group.create_dataset("indptr", data=matrix.indptr)

        obs = h5.create_group("obs")
        obs.create_dataset(
            "_index",
            data=np.array([f"c{i}".encode() for i in range(n_cells)]),
        )
        var = h5.create_group("var")
        var.create_dataset(
            "_index",
            data=np.array([f"f{i}".encode() for i in range(n_feats)]),
        )
        var.create_dataset(
            "feature_name",
            data=np.array([f"g{i}".encode() for i in range(n_feats)]),
        )
        if feature_types is not None:
            var.create_dataset("feature_types", data=np.array(feature_types))
    return path


# Sizes a 12-row uint32 assay into (4, n_feats) row shards.
_SHARD_BAND_BUDGET = {
    "mem_budget": 1024**2,
    "nthreads": 4,
    "targetChunkBytes": 48,
    "targetShardBytes": 48,
}


def _band_counts(n_cells: int, n_feats: int) -> np.ndarray:
    rng = np.random.default_rng(7)
    values = rng.integers(0, 5, size=(n_cells, n_feats), dtype=np.uint32)
    values[4:8] = 0
    return values


@pytest.mark.parametrize("encoding", ["csr", "csc", "dense"])
def test_h5adtozarr_writes_shard_bands_for_every_encoding(tmp_path, encoding):
    from scarf.readers import H5adReader
    from scarf.storage.types import array_metadata_shards
    from scarf.writers import H5adToZarr

    values = _band_counts(12, 3)
    path = _write_h5ad(tmp_path / f"{encoding}.h5ad", values, encoding=encoding)
    reader = H5adReader(str(path), feature_name_key="feature_name")
    store = MemoryStore()
    try:
        H5adToZarr(reader, zarr_loc=store, **_SHARD_BAND_BUDGET).dump(batch_size=5)
    finally:
        reader.h5.close()

    root = zarr.open_group(store=store, mode="r")
    counts = root["RNA/counts"]
    assert array_metadata_shards(counts) == (4, 3)
    np.testing.assert_array_equal(counts[:], values)
    assert "countsT" not in root["RNA"]


@pytest.mark.parametrize(
    ("fractional", "expected_dtype"),
    [(False, np.dtype("uint16")), (True, np.dtype("float32"))],
)
def test_h5adtozarr_uses_smallest_lossless_dtype_for_float_counts(
    tmp_path,
    fractional,
    expected_dtype,
):
    from scarf.readers import H5adReader
    from scarf.writers import H5adToZarr

    values = _band_counts(12, 3).astype(np.float32)
    values[0, 0] = 1.5 if fractional else 3308
    path = _write_h5ad(tmp_path / "float_counts.h5ad", values, encoding="csr")
    reader = H5adReader(str(path), feature_name_key="feature_name")
    store = MemoryStore()
    try:
        H5adToZarr(
            reader,
            zarr_loc=store,
            mem_budget=1024**2,
            nthreads=2,
            targetChunkBytes=48,
            targetShardBytes=48,
        ).dump(batch_size=4)
    finally:
        reader.h5.close()

    root = zarr.open_group(store=store, mode="r")
    assert np.dtype(root["RNA/counts"].dtype) == expected_dtype
    np.testing.assert_array_equal(root["RNA/counts"][:], values)
    assert "countsT" not in root["RNA"]


@pytest.mark.parametrize("encoding", ["csr", "csc"])
def test_h5adtozarr_preserves_duplicate_coordinate_sums(tmp_path, encoding):
    import h5py

    from scarf.readers import H5adReader
    from scarf.writers import H5adToZarr

    path = _write_h5ad(
        tmp_path / f"duplicate_{encoding}.h5ad",
        np.zeros((1, 1), dtype=np.float32),
        encoding=encoding,
    )
    with h5py.File(path, mode="r+") as h5:
        group = h5["X"]
        for name in ("data", "indices", "indptr"):
            del group[name]
        group.create_dataset("data", data=np.array([200, 100], dtype=np.float32))
        group.create_dataset("indices", data=np.array([0, 0], dtype=np.int32))
        group.create_dataset("indptr", data=np.array([0, 2], dtype=np.int32))

    reader = H5adReader(str(path), feature_name_key="feature_name")
    store = MemoryStore()
    try:
        H5adToZarr(
            reader,
            zarr_loc=store,
            mem_budget=1024**2,
            nthreads=2,
            targetChunkBytes=16,
            targetShardBytes=16,
        ).dump(batch_size=1)
    finally:
        reader.h5.close()

    counts = zarr.open_group(store=store, mode="r")["RNA/counts"]
    assert counts.dtype == np.dtype("float32")
    assert counts[0, 0] == 300


@pytest.mark.parametrize("encoding", ["csr", "csc"])
def test_h5adtozarr_reduces_duplicates_before_explicit_dtype_cast(
    tmp_path,
    encoding,
):
    import h5py

    from scarf.readers import H5adReader
    from scarf.writers import H5adToZarr

    path = _write_h5ad(
        tmp_path / f"explicit_duplicate_{encoding}.h5ad",
        np.zeros((1, 1), dtype=np.float32),
        encoding=encoding,
    )
    with h5py.File(path, mode="r+") as h5:
        group = h5["X"]
        for name in ("data", "indices", "indptr"):
            del group[name]
        group.create_dataset(
            "data",
            data=np.array([100.5, 99.5], dtype=np.float32),
        )
        group.create_dataset("indices", data=np.array([0, 0], dtype=np.int32))
        group.create_dataset("indptr", data=np.array([0, 2], dtype=np.int32))

    reader = H5adReader(
        str(path),
        feature_name_key="feature_name",
        dtype="uint8",
    )
    store = MemoryStore()
    try:
        H5adToZarr(
            reader,
            zarr_loc=store,
            mem_budget=1024**2,
            nthreads=1,
            targetChunkBytes=16,
            targetShardBytes=16,
        ).dump(batch_size=1)
    finally:
        reader.h5.close()

    counts = zarr.open_group(store=store, mode="r")["RNA/counts"]
    assert counts.dtype == np.dtype("uint8")
    assert counts[0, 0] == 200


def test_h5adtozarr_reads_the_source_once_for_all_assays(tmp_path, monkeypatch):
    from scarf.readers import H5adReader
    from scarf.writers import H5adToZarr

    values = _band_counts(12, 4)
    # An empty leading band for the second assay must not shift its row offsets.
    values[:5, [1, 3]] = 0
    path = _write_h5ad(
        tmp_path / "multi.h5ad",
        values,
        feature_types=[
            b"Gene Expression",
            b"Antibody Capture",
            b"Gene Expression",
            b"Antibody Capture",
        ],
    )

    consumed: list[int] = []
    original = H5adReader.consume

    def spy(self, batch_size):
        consumed.append(batch_size)
        return original(self, batch_size)

    monkeypatch.setattr(H5adReader, "consume", spy)

    reader = H5adReader(str(path), feature_name_key="feature_name")
    store = MemoryStore()
    try:
        H5adToZarr(
            reader,
            zarr_loc=store,
            assay_split_key="feature_types",
            **_SHARD_BAND_BUDGET,
        ).dump(batch_size=5)
    finally:
        reader.h5.close()

    assert consumed == [5]
    root = zarr.open_group(store=store, mode="r")
    assert set(root.group_keys()) == {"cellData", "RNA", "ADT"}
    np.testing.assert_array_equal(root["RNA/counts"][:], values[:, [0, 2]])
    np.testing.assert_array_equal(root["ADT/counts"][:], values[:, [1, 3]])
    assert "countsT" not in root["RNA"]
    assert "countsT" not in root["ADT"]
    assert root["RNA/counts"].dtype == values.dtype


def test_h5adtozarr_small_assay_does_not_serialize_row_band_writes(
    tmp_path,
):
    from scarf.readers import H5adReader
    from scarf.writers import H5adToZarr
    from tests.store_probes import RecordingStore

    values = _band_counts(12, 4)
    path = _write_h5ad(
        tmp_path / "uneven_multi.h5ad",
        values,
        feature_types=[
            b"Gene Expression",
            b"Gene Expression",
            b"Gene Expression",
            b"Antibody Capture",
        ],
    )
    reader = H5adReader(str(path), feature_name_key="feature_name")
    store = RecordingStore(delay=0.01)
    try:
        writer = H5adToZarr(
            reader,
            zarr_loc=store,
            assay_split_key="feature_types",
            **_SHARD_BAND_BUDGET,
        )
        store.reset()
        writer.dump(batch_size=5)
    finally:
        reader.h5.close()

    assert store.max_in_flight > 1
    root = zarr.open_group(store=store, mode="r")
    np.testing.assert_array_equal(root["RNA/counts"][:], values[:, :3])
    np.testing.assert_array_equal(root["ADT/counts"][:], values[:, 3:])


def test_h5adtozarr_propagates_band_write_failure(
    tmp_path,
):
    from scarf.readers import H5adReader
    from scarf.writers import H5adToZarr
    from tests.store_probes import RecordingStore

    values = _band_counts(12, 3)
    path = _write_h5ad(tmp_path / "failing.h5ad", values)
    reader = H5adReader(str(path), feature_name_key="feature_name")
    store = RecordingStore(fail_on="RNA/counts/c/2/0")
    try:
        writer = H5adToZarr(reader, zarr_loc=store, **_SHARD_BAND_BUDGET)
        with pytest.raises(RuntimeError, match="injected write failure"):
            writer.dump(batch_size=5)
    finally:
        reader.h5.close()

    assert ("set", "RNA/counts/c/2/0") in store.ops


def test_h5adtozarr_counts_materialized_csr_as_resident_memory(tmp_path):
    from scarf.readers import H5adReader
    from scarf.writers import H5adToZarr

    values = (
        np.arange(400 * 400, dtype=np.uint32).reshape(400, 400) % 65_534 + 1
    ).astype(np.uint16)
    path = _write_h5ad(tmp_path / "resident_csc.h5ad", values, encoding="csc")
    reader = H5adReader(str(path), feature_name_key="feature_name")
    try:
        reader.infer_storage_dtype()
        conversion_peak = reader.csc_conversion_peak_bytes()
        writer = H5adToZarr(
            reader,
            zarr_loc=MemoryStore(),
            mem_budget=conversion_peak,
            nthreads=4,
        )
        assert reader.materialized_csr_bytes() > 0
        with pytest.raises(MemoryError, match="operation limit"):
            writer.dump(batch_size=values.shape[0])
    finally:
        reader.h5.close()


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
        writer = LoomToZarr(
            reader,
            zarr_loc=store,
            targetChunkBytes=8,
            targetShardBytes=8,
        )
        writer.dump(batch_size=2)
    finally:
        reader.h5.close()

    root = zarr.open_group(store=store, mode="r")
    np.testing.assert_array_equal(root["RNA/counts"][:], values)
    assert "countsT" not in root["RNA"]


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
    assert "countsT" not in root["RNA"]


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
    assert "countsT" not in root["RNA"]
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


def test_csv_to_zarr_writes_extra_cell_columns_into_workspace(tmp_path):
    csv_path = tmp_path / "counts.csv"
    csv_path.write_text(
        "quality,geneA,geneB\n1,1,2\n2,3,4\n3,5,6\n",
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
        workspace="run1",
        dtype=np.dtype("uint16"),
    )

    writer.dump()

    root = zarr.open_group(store=store, mode="r")
    assert "cellData" not in root
    np.testing.assert_array_equal(
        root["run1/cellData/quality"][:],
        np.array([1, 2, 3]),
    )
    np.testing.assert_array_equal(
        root["matrices/RNA/counts"][:],
        np.array([[1, 2], [3, 4], [5, 6]], dtype=np.uint16),
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
        targetChunkBytes=8,
        targetShardBytes=8,
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


@pytest.fixture
def export_assay_store(toy_crdir_writer, tmp_path):
    import shutil

    from scarf.datastore.datastore import DataStore

    destination = tmp_path / "export_toy.zarr"
    shutil.copytree(toy_crdir_writer, destination)
    return DataStore(
        str(destination),
        default_assay="RNA",
        min_features_per_cell=0,
        min_cells_per_feature=0,
        nthreads=1,
    )


def test_to_h5ad_preserves_counts_metadata_and_embeddings(export_assay_store, tmp_path):
    import h5py
    from scipy.sparse import csr_matrix

    from scarf.writers import to_h5ad

    assay = export_assay_store.RNA
    n_cells = assay.cells.N
    umap = np.column_stack(
        [
            np.linspace(0.0, 1.0, n_cells),
            np.linspace(2.0, 3.0, n_cells),
        ]
    )
    assay.cells.insert("RNA_UMAP1", umap[:, 0], overwrite=True)
    assay.cells.insert("RNA_UMAP2", umap[:, 1], overwrite=True)
    assay.cells.insert(
        "export_batch", np.array(["a", "b", "a"][:n_cells]), overwrite=True
    )

    path = tmp_path / "toy_export.h5ad"
    to_h5ad(assay, str(path), embeddings_cols=["UMAP"])

    expected = csr_matrix(assay.rawData.compute())
    with h5py.File(path, "r") as h5:
        shape = tuple(int(x) for x in h5["X"].attrs["shape"])
        assert shape == (assay.cells.N, assay.feats.N)
        exported = csr_matrix(
            (h5["X/data"][:], h5["X/indices"][:], h5["X/indptr"][:]),
            shape=shape,
        )
        np.testing.assert_array_equal(exported.toarray(), expected.toarray())
        np.testing.assert_array_equal(
            h5["obs/_index"].asstr()[:],
            assay.cells.fetch_all("ids").astype(str),
        )
        np.testing.assert_array_equal(
            h5["obs/export_batch"].asstr()[:],
            assay.cells.fetch_all("export_batch").astype(str),
        )
        np.testing.assert_array_equal(
            h5["var/_index"].asstr()[:],
            assay.feats.fetch_all("ids").astype(str),
        )
        np.testing.assert_array_equal(
            h5["var/gene_short_name"].asstr()[:],
            assay.feats.fetch_all("names").astype(str),
        )
        emb_cols = sorted(
            column for column in assay.cells.columns if column.startswith("RNA_UMAP")
        )
        assert emb_cols == ["RNA_UMAP1", "RNA_UMAP2"]
        np.testing.assert_allclose(h5["obsm/X_umap"][:], umap)
        assert "RNA_UMAP1" not in h5["obs"]
        assert "RNA_UMAP2" not in h5["obs"]


def test_to_mtx_preserves_counts_barcodes_and_features(export_assay_store, tmp_path):
    from scipy.io import mmread
    from scipy.sparse import csr_matrix

    from scarf.writers import to_mtx

    assay = export_assay_store.RNA
    out_dir = tmp_path / "toy_mtx"
    to_mtx(assay, str(out_dir), compress=False)

    exported = mmread(out_dir / "matrix.mtx").tocsr()
    expected = csr_matrix(assay.rawData.compute()).T.tocsr()
    assert exported.shape == (assay.feats.N, assay.cells.N)
    np.testing.assert_array_equal(exported.toarray(), expected.toarray())

    barcodes = (out_dir / "barcodes.tsv").read_text().splitlines()
    assert barcodes == list(assay.cells.fetch_all("ids").astype(str))

    features = [
        line.split("\t")
        for line in (out_dir / "genes.tsv").read_text().splitlines()
        if line
    ]
    assert [row[0] for row in features] == list(
        assay.feats.fetch_all("ids").astype(str)
    )
    assert [row[1] for row in features] == list(
        assay.feats.fetch_all("names").astype(str)
    )


def test_to_mtx_compress_writes_gzipped_matrix_market(export_assay_store, tmp_path):
    import gzip

    from scipy.io import mmread
    from scipy.sparse import csr_matrix

    from scarf.writers import to_mtx

    assay = export_assay_store.RNA
    out_dir = tmp_path / "toy_mtx_gz"
    to_mtx(assay, str(out_dir), compress=True)

    assert (out_dir / "matrix.mtx.gz").is_file()
    assert (out_dir / "barcodes.tsv.gz").is_file()
    assert (out_dir / "features.tsv.gz").is_file()

    exported = mmread(gzip.open(out_dir / "matrix.mtx.gz", "rt")).tocsr()
    expected = csr_matrix(assay.rawData.compute()).T.tocsr()
    np.testing.assert_array_equal(exported.toarray(), expected.toarray())

    with gzip.open(out_dir / "barcodes.tsv.gz", "rt") as handle:
        barcodes = [line.strip() for line in handle if line.strip()]
    assert barcodes == list(assay.cells.fetch_all("ids").astype(str))


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

    monkeypatch.setattr("scarf.writers.subset.load_zarr", fake_load_zarr)
    subset = object.__new__(SubsetZarr)
    subset.overFn = False
    subset.storage_options = {"access_key_id": "key"}
    SubsetZarr._check_files(subset, "s3://bucket/out.zarr")
    assert calls == [("s3://bucket/out.zarr", "w", {"access_key_id": "key"})]


def test_crtozarr_forwards_storage_options(monkeypatch):
    captured = {}

    def fake_load_zarr(zarr_loc, mode, storage_options=None):
        captured["zarr_loc"] = zarr_loc
        captured["mode"] = mode
        captured["storage_options"] = storage_options
        return zarr.open_group(store=MemoryStore(), mode="w")

    monkeypatch.setattr("scarf.storage.stores.load_zarr", fake_load_zarr)
    monkeypatch.setattr("scarf.storage.schema.create_cell_data", lambda **kwargs: None)
    monkeypatch.setattr(
        "scarf.storage.schema.create_zarr_count_assay",
        lambda **kwargs: None,
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


def test_h5adtozarr_applies_storage_resources_and_chunk_controls(tmp_path):
    from scarf.readers import H5adReader
    from scarf.storage.layout import count_array_spec
    from scarf.writers import H5adToZarr

    values = (np.arange(100 * 50, dtype=np.uint32).reshape(100, 50) % 1_000).astype(
        np.uint16
    )
    path = _write_h5ad(tmp_path / "layout_controls.h5ad", values)
    reader = H5adReader(str(path), feature_name_key="feature_name")
    try:
        writer = H5adToZarr(
            reader,
            zarr_loc=MemoryStore(),
            mem_budget="2G",
            nthreads=3,
            targetChunkBytes=4_096,
            targetShardBytes=20_480,
        )
    finally:
        reader.h5.close()

    expected = count_array_spec(
        100,
        50,
        dtype=np.uint16,
        profile="fast_local",
        targetChunkBytes=4_096,
        targetShardBytes=20_480,
    )
    counts = writer.z["RNA/counts"]
    assert writer.resources.memoryBytes == 2 * 1024**3
    assert writer.resources.workers == 3
    assert counts.chunks == expected.chunks
    assert counts.metadata.shards == expected.shards
