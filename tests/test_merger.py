import numpy as np
import pandas as pd
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.chunked import ChunkedArray
from scarf.merge import AssayMerge
from scarf.storage.budget import (
    ResourceBudget,
    get_resource_budget,
    set_resource_budget,
)
from scarf.storage.zarr_store import count_array_spec


class _MergeMeta:
    def __init__(self, **columns):
        self._columns = {key: np.asarray(value) for key, value in columns.items()}
        self.columns = list(self._columns)
        self.N = len(next(iter(self._columns.values())))

    def to_pandas_dataframe(self, columns):
        return pd.DataFrame({key: self._columns[key] for key in columns})

    def fetch_all(self, key):
        return self._columns[key]


class _MergeAssay:
    def __init__(
        self,
        name,
        counts,
        cell_ids,
        feature_ids,
        feature_names,
        block_size,
    ):
        self.name = name
        self.rawData = ChunkedArray.from_numpy(
            np.asarray(counts),
            block_size=block_size,
        )
        self.cells = _MergeMeta(
            ids=cell_ids,
            names=cell_ids,
            I=np.ones(len(cell_ids), dtype=bool),
        )
        self.feats = _MergeMeta(ids=feature_ids, names=feature_names)


@pytest.fixture
def tiny_merge_budget():
    previous = get_resource_budget()
    set_resource_budget(ResourceBudget(memoryBytes=96, workers=2, workingCopies=2))
    try:
        yield
    finally:
        set_resource_budget(previous)


def test_assay_merge(datastore, rna_raw_total, tmp_path):
    fn = str(tmp_path / "merged.zarr")
    writer = AssayMerge(
        zarr_path=fn,
        assays=[datastore.RNA, datastore.RNA],
        names=["self1", "self2"],
        merge_assay_name="RNA",
        prepend_text="",
        overwrite=True,
    )
    writer.dump()
    tmp = zarr.open(fn + "/RNA/counts")
    assert tmp.shape[0] == 2 * datastore.cells.N
    assert int(tmp[...].sum()) == rna_raw_total * 2


def test_assay_merge_maps_features_and_preserves_row_order(
    tmp_path,
    tiny_merge_budget,
):
    cell_ids = ["c0", "c1", "c2"]
    left = _MergeAssay(
        "RNA",
        [[1, 10], [2, 20], [0, 0]],
        cell_ids,
        ["id_a", "id_b"],
        ["A", "B"],
        block_size=1,
    )
    right = _MergeAssay(
        "RNA",
        [[30, 300], [40, 400], [50, 500]],
        cell_ids,
        ["id_b", "id_c"],
        ["B", "C"],
        block_size=3,
    )
    fn = str(tmp_path / "feature_order.zarr")
    writer = AssayMerge(
        zarr_path=fn,
        assays=[left, right],
        names=["left", "right"],
        merge_assay_name="RNA",
        prepend_text="",
        seed=3,
    )
    writer.dump()

    root = zarr.open_group(fn, mode="r")
    counts_array = root["RNA/counts"]
    counts = np.asarray(counts_array[:])
    merged_cell_ids = np.asarray(root["cellData/ids"][:]).astype(str)
    merged_feature_ids = np.asarray(root["RNA/featureData/ids"][:]).astype(str)

    np.testing.assert_array_equal(merged_feature_ids, ["id_a", "id_b", "id_c"])
    expected = {
        "left__c0": [1, 10, 0],
        "left__c1": [2, 20, 0],
        "left__c2": [0, 0, 0],
        "right__c0": [0, 30, 300],
        "right__c1": [0, 40, 400],
        "right__c2": [0, 50, 500],
    }
    for cell_id, row in zip(merged_cell_ids, counts, strict=True):
        np.testing.assert_array_equal(row, expected[cell_id])

    spec = count_array_spec(*counts_array.shape, dtype=counts_array.dtype)
    assert spec.shards is not None
    assert spec.shards[0] < counts_array.shape[0]
    assert counts_array.chunks == spec.chunks
    assert counts_array.metadata.shards == spec.shards
    assert counts_array.fill_value == 0


def test_dask_to_coo_sums_consolidated_features():
    merge = object.__new__(AssayMerge)
    merge.nFeats = 1

    result = AssayMerge._dask_to_coo(
        merge,
        np.array([[2, 3], [5, 7]]),
        np.array([0, 1]),
        np.array([0, 0]),
        1,
    )

    assert result.nnz == 2
    np.testing.assert_array_equal(result.toarray(), [[5], [12]])


def test_dataset_merge_2(datastore, rna_raw_total, assay2_raw_total, tmp_path):
    from scarf.merge import DatasetMerge

    fn = str(tmp_path / "merged.zarr")
    writer = DatasetMerge(
        zarr_path=fn,
        datasets=[datastore, datastore],
        names=["self1", "self2"],
        prepend_text="",
        overwrite=True,
    )
    writer.dump()
    rna_count = zarr.open(fn + "/RNA/counts")
    assay2_count = zarr.open(fn + "/assay2/counts")
    assert rna_count.shape[0] == 2 * datastore.cells.N
    assert assay2_count.shape[0] == 2 * datastore.cells.N
    assert int(rna_count[...].sum()) == rna_raw_total * 2
    assert int(assay2_count[...].sum()) == assay2_raw_total * 2


def test_dataset_merge_3(datastore, rna_raw_total, assay2_raw_total, tmp_path):
    from scarf.merge import DatasetMerge

    fn = str(tmp_path / "merged.zarr")
    writer = DatasetMerge(
        zarr_path=fn,
        datasets=[datastore, datastore, datastore],
        names=["self1", "self2", "self3"],
        prepend_text="",
        overwrite=True,
    )
    writer.dump()
    rna_count = zarr.open(fn + "/RNA/counts")
    assay2_count = zarr.open(fn + "/assay2/counts")
    assert rna_count.shape[0] == 3 * datastore.cells.N
    assert assay2_count.shape[0] == 3 * datastore.cells.N
    assert int(rna_count[...].sum()) == rna_raw_total * 3
    assert int(assay2_count[...].sum()) == assay2_raw_total * 3


def test_dataset_merge_cells(datastore, tmp_path):
    from scarf.datastore.datastore import DataStore
    from scarf.merge import DatasetMerge

    fn = str(tmp_path / "merged.zarr")
    writer = DatasetMerge(
        zarr_path=fn,
        datasets=[datastore, datastore],
        names=["self1", "self2"],
        prepend_text="orig",
        overwrite=True,
    )
    writer.dump()

    ds = DataStore(
        fn,
        default_assay="RNA",
    )

    df = ds.cells.to_pandas_dataframe(ds.cells.columns)
    df_diff = df[df["orig_RNA_nCounts"] != df["RNA_nCounts"]]
    assert len(df_diff) == 0


def test_assay_merge_rejects_duplicate_sample_names(datastore, tmp_path):
    fn = str(tmp_path / "merged_dup_names.zarr")
    with pytest.raises(ValueError, match="unique name"):
        AssayMerge(
            zarr_path=fn,
            assays=[datastore.RNA, datastore.RNA],
            names=["dup", "dup"],
            merge_assay_name="RNA",
            prepend_text="",
            overwrite=True,
        )


def test_assay_merge_rejects_existing_workspace_assay(
    datastore,
    rna_raw_total,
    tmp_path,
):
    fn = str(tmp_path / "workspace_merged.zarr")
    writer = AssayMerge(
        zarr_path=fn,
        assays=[datastore.RNA, datastore.RNA],
        names=["self1", "self2"],
        merge_assay_name="RNA",
        out_workspace="merged",
        prepend_text="",
    )
    writer.dump()

    root = zarr.open_group(fn, mode="r")
    counts = root["matrices/RNA/counts"]
    assert counts.shape[0] == 2 * datastore.cells.N
    assert int(counts[...].sum()) == 2 * rna_raw_total

    with pytest.raises(ValueError, match="already contains RNA assay"):
        AssayMerge(
            zarr_path=fn,
            assays=[datastore.RNA, datastore.RNA],
            names=["self1", "self2"],
            merge_assay_name="RNA",
            out_workspace="merged",
            prepend_text="",
        )


def test_assay_merge_validation_failure_does_not_overwrite_store(monkeypatch):
    calls = []
    root = zarr.open_group(store=MemoryStore(), mode="w")
    root.create_group("cellData")
    root.create_group("RNA")

    def fake_load_zarr(zarr_loc, mode, storage_options=None):
        calls.append(mode)
        if mode == "r":
            return root
        pytest.fail(f"Unexpected destructive open mode: {mode}")

    monkeypatch.setattr("scarf.merge.load_zarr", fake_load_zarr)
    merge = object.__new__(AssayMerge)
    merge.outWorkspace = None
    merge.storage_options = None

    with pytest.raises(ValueError, match="already contains RNA assay"):
        AssayMerge._use_existing_zarr(merge, MemoryStore(), "RNA", False)

    assert calls == ["r"]


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


def test_dummy_assay_holds_zero_counts(datastore):
    from scarf.merge import DummyAssay
    from scarf.writers import create_zarr_dataset

    mem = zarr.open_group(store=MemoryStore(), mode="w")
    dummy_array = create_zarr_dataset(
        mem,
        "counts",
        datastore.RNA.rawData.chunksize,
        datastore.RNA.rawData.dtype,
        (datastore.cells.N, datastore.RNA.feats.N),
    )
    dummy = DummyAssay(
        datastore,
        ChunkedArray(dummy_array, nthreads=1),
        datastore.RNA.feats,
        "RNA",
    )
    assert dummy.name == "RNA"
    assert int(dummy.rawData.compute().sum()) == 0
