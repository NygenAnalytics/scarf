import numpy as np
import pandas as pd
import pytest
import sys
import zarr
from zarr.storage import MemoryStore

from scarf.matrix import ChunkedArray
from scarf.merge import AssayMerge
from scarf.storage.budget import ResourceBudget
from scarf.storage.layout import count_array_spec


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


class _MergeDataStore:
    def __init__(self, assays):
        self._assays = {assay.name: assay for assay in assays}
        self.assay_names = list(self._assays)
        self.cells = assays[0].cells
        self.nthreads = 2
        self.memoryBytes = 1024**3
        self.resources = ResourceBudget(self.memoryBytes, self.nthreads)

    def get_assay(self, name):
        return self._assays[name]


def test_assay_merge_rejects_summary_before_truncating_destination():
    store = MemoryStore()
    root = zarr.open_group(store=store, mode="w")
    root.create_group("sentinel")

    with pytest.raises(ValueError, match=r"reserved for DataStore\.summary"):
        AssayMerge(
            zarr_path=store,
            assays=[],
            names=[],
            merge_assay_name="summary",
        )

    preserved = zarr.open_group(store=store, mode="r")
    assert set(preserved.group_keys()) == {"sentinel"}


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
    assert "countsT" not in root["RNA"]
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

    spec = count_array_spec(
        *counts_array.shape,
        dtype=counts_array.dtype,
        profile="fast_local",
    )
    assert spec.shards is not None
    assert counts_array.chunks == spec.chunks
    assert counts_array.metadata.shards == spec.shards
    assert counts_array.fill_value == 0


@pytest.mark.parametrize("dtype", [None, "uint16"])
def test_assay_merge_widens_before_consolidating_features(tmp_path, dtype):
    feature_ids = ["gene_0", "gene_1"]
    left = _MergeAssay(
        "RNA",
        np.array([[200, 100]], dtype=np.uint8),
        ["left"],
        feature_ids,
        feature_ids,
        block_size=1,
    )
    right = _MergeAssay(
        "RNA",
        np.array([[150, 150]], dtype=np.uint8),
        ["right"],
        feature_ids,
        feature_ids,
        block_size=1,
    )
    path = str(tmp_path / "consolidated_features.zarr")

    AssayMerge(
        zarr_path=path,
        assays=[left, right],
        names=["left", "right"],
        merge_assay_name="RNA",
        prepend_text="",
        dtype=dtype,
    ).dump()

    counts = zarr.open_group(path, mode="r")["RNA/counts"]
    assert counts.dtype == np.dtype("uint16")
    np.testing.assert_array_equal(counts[:], [[300], [300]])


def test_assay_merge_rejects_metadata_over_memory_budget(tmp_path):
    cell_ids = [f"cell-{index}-{'x' * 1000}" for index in range(100)]
    left = _MergeAssay(
        "RNA",
        np.ones((100, 1), dtype=np.uint8),
        cell_ids,
        ["gene-id"],
        ["gene"],
        block_size=25,
    )
    right = _MergeAssay(
        "RNA",
        np.ones((100, 1), dtype=np.uint8),
        cell_ids,
        ["gene-id"],
        ["gene"],
        block_size=25,
    )

    with pytest.raises(MemoryError, match="operation limit"):
        AssayMerge(
            zarr_path=str(tmp_path / "metadata_budget.zarr"),
            assays=[left, right],
            names=["left", "right"],
            merge_assay_name="RNA",
            prepend_text="",
            mem_budget="64K",
        )


def test_assay_merge_metadata_bytes_deduplicate_aliased_feature_maps():
    merge = object.__new__(AssayMerge)
    merge.mergedCells = pd.DataFrame()
    merge.mergedFeats = pd.DataFrame()
    merge.mergedFeats_map = pd.DataFrame()
    mapping = {"gene-id": "gene"}
    merge.featCollection = [mapping]
    merge.featCollection_map = [mapping]

    frame_bytes = sum(
        int(frame.memory_usage(index=True, deep=True).sum())
        for frame in (
            merge.mergedCells,
            merge.mergedFeats,
            merge.mergedFeats_map,
        )
    )
    expected = (
        frame_bytes
        + sys.getsizeof(merge.featCollection)
        + sys.getsizeof(merge.featCollection_map)
        + sys.getsizeof(mapping)
        + sys.getsizeof("gene-id")
        + sys.getsizeof("gene")
    )
    assert merge._metadata_resident_bytes() == expected


def test_assay_merge_keeps_source_metadata_aligned_after_permutation(
    tmp_path,
):
    left = _MergeAssay(
        "RNA",
        [[1], [2], [3]],
        ["l0", "l1", "l2"],
        ["gene"],
        ["Gene"],
        block_size=1,
    )
    right = _MergeAssay(
        "RNA",
        [[4], [5]],
        ["r0", "r1"],
        ["gene"],
        ["Gene"],
        block_size=1,
    )
    left.cells = _MergeMeta(
        ids=["l0", "l1", "l2"],
        names=["l0", "l1", "l2"],
        I=[True, False, True],
        cluster_labels=[10, 11, 12],
    )
    right.cells = _MergeMeta(
        ids=["r0", "r1"],
        names=["r0", "r1"],
        I=[False, True],
        cluster_labels=[20, 21],
    )

    fn = str(tmp_path / "aligned_metadata.zarr")
    AssayMerge(
        zarr_path=fn,
        assays=[left, right],
        names=["left", "right"],
        merge_assay_name="RNA",
        prepend_text="orig",
        reset_cell_filter=False,
        source_column="sample_id",
        seed=7,
    ).dump()

    root = zarr.open_group(fn, mode="r")
    ids = np.asarray(root["cellData/ids"][:]).astype(str)
    sources = np.asarray(root["cellData/sample_id"][:]).astype(str)
    clusters = np.asarray(root["cellData/orig_cluster_labels"][:])
    valid = np.asarray(root["cellData/I"][:], dtype=bool)
    observed = {
        cell_id: (source, int(cluster), bool(is_valid))
        for cell_id, source, cluster, is_valid in zip(
            ids,
            sources,
            clusters,
            valid,
            strict=True,
        )
    }

    assert observed == {
        "left__l0": ("left", 10, True),
        "left__l1": ("left", 11, False),
        "left__l2": ("left", 12, True),
        "right__r0": ("right", 20, False),
        "right__r1": ("right", 21, True),
    }


def test_assay_merge_rejects_source_column_conflict(tmp_path):
    assay = _MergeAssay(
        "RNA",
        [[1], [2]],
        ["c0", "c1"],
        ["gene"],
        ["Gene"],
        block_size=1,
    )
    assay.cells = _MergeMeta(
        ids=["c0", "c1"],
        names=["c0", "c1"],
        I=[True, True],
        sample_id=["existing", "existing"],
    )

    with pytest.raises(ValueError, match="conflicts with merged metadata"):
        AssayMerge(
            zarr_path=str(tmp_path / "source_conflict.zarr"),
            assays=[assay],
            names=["sample"],
            merge_assay_name="RNA",
            prepend_text="",
            source_column="sample_id",
        )


def test_assay_merge_preserves_order_across_source_block_sizes(tmp_path):
    cell_ids = ["c0", "c1", "c2", "c3"]
    left = _MergeAssay(
        "RNA",
        [[1, 10], [2, 20], [3, 30], [4, 40]],
        cell_ids,
        ["id_a", "id_b"],
        ["A", "B"],
        block_size=1,
    )
    right = _MergeAssay(
        "RNA",
        [[50, 500], [60, 600], [70, 700], [80, 800]],
        cell_ids,
        ["id_b", "id_c"],
        ["B", "C"],
        block_size=4,
    )
    fn = str(tmp_path / "out_of_order_prefetch.zarr")
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
    counts = np.asarray(root["RNA/counts"][:])
    merged_cell_ids = np.asarray(root["cellData/ids"][:]).astype(str)
    expected = {
        "left__c0": [1, 10, 0],
        "left__c1": [2, 20, 0],
        "left__c2": [3, 30, 0],
        "left__c3": [4, 40, 0],
        "right__c0": [0, 50, 500],
        "right__c1": [0, 60, 600],
        "right__c2": [0, 70, 700],
        "right__c3": [0, 80, 800],
    }
    for cell_id, row in zip(merged_cell_ids, counts, strict=True):
        np.testing.assert_array_equal(row, expected[cell_id])


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


def test_dataset_merge_shares_row_order_across_assay_chunk_sizes(
    tmp_path,
):
    from scarf.merge import DatasetMerge

    cell_ids = ["c0", "c1", "c2", "c3", "c4"]
    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]],
                cell_ids,
                ["rna_a", "rna_b"],
                ["RNA A", "RNA B"],
                block_size=3,
            ),
            _MergeAssay(
                "ADT",
                [[101, 110], [102, 120], [103, 130], [104, 140], [105, 150]],
                cell_ids,
                ["adt_a", "adt_b"],
                ["ADT A", "ADT B"],
                block_size=2,
            ),
        ]
    )
    right = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[6, 60], [7, 70], [8, 80], [9, 90], [10, 100]],
                cell_ids,
                ["rna_a", "rna_b"],
                ["RNA A", "RNA B"],
                block_size=3,
            ),
            _MergeAssay(
                "ADT",
                [[106, 160], [107, 170], [108, 180], [109, 190], [110, 200]],
                cell_ids,
                ["adt_a", "adt_b"],
                ["ADT A", "ADT B"],
                block_size=2,
            ),
        ]
    )
    fn = str(tmp_path / "mixed_assay_chunks.zarr")

    writer = DatasetMerge(
        datasets=[left, right],
        zarr_path=fn,
        names=["left", "right"],
        prepend_text="",
        seed=3,
    )
    writer.dump()

    assert writer.unique_assays == ["RNA", "ADT"]
    for generator in writer.merge_generators:
        assert (
            max(
                rows.size
                for blocks in generator.permutations_rows.values()
                for rows in blocks.values()
            )
            <= 2
        )
    root = zarr.open_group(fn, mode="r")
    merged_cell_ids = np.asarray(root["cellData/ids"][:]).astype(str)
    rna = np.asarray(root["RNA/counts"][:])
    adt = np.asarray(root["ADT/counts"][:])
    expected = {
        "left__c0": ([1, 10], [101, 110]),
        "left__c1": ([2, 20], [102, 120]),
        "left__c2": ([3, 30], [103, 130]),
        "left__c3": ([4, 40], [104, 140]),
        "left__c4": ([5, 50], [105, 150]),
        "right__c0": ([6, 60], [106, 160]),
        "right__c1": ([7, 70], [107, 170]),
        "right__c2": ([8, 80], [108, 180]),
        "right__c3": ([9, 90], [109, 190]),
        "right__c4": ([10, 100], [110, 200]),
    }
    for cell_id, rna_row, adt_row in zip(
        merged_cell_ids,
        rna,
        adt,
        strict=True,
    ):
        expected_rna, expected_adt = expected[cell_id]
        np.testing.assert_array_equal(rna_row, expected_rna)
        np.testing.assert_array_equal(adt_row, expected_adt)


def test_dataset_merge_shared_row_plan_handles_missing_assays(
    tmp_path,
):
    from scarf.merge import DatasetMerge

    cell_ids = ["c0", "c1", "c2"]
    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[1, 10], [2, 20], [3, 30]],
                cell_ids,
                ["rna_a", "rna_b"],
                ["RNA A", "RNA B"],
                block_size=3,
            )
        ]
    )
    right = _MergeDataStore(
        [
            _MergeAssay(
                "ADT",
                [[101, 110], [102, 120], [103, 130]],
                cell_ids,
                ["adt_a", "adt_b"],
                ["ADT A", "ADT B"],
                block_size=2,
            )
        ]
    )
    fn = str(tmp_path / "missing_assays.zarr")

    writer = DatasetMerge(
        datasets=[left, right],
        zarr_path=fn,
        names=["left", "right"],
        prepend_text="",
        seed=3,
    )
    writer.dump()

    root = zarr.open_group(fn, mode="r")
    merged_cell_ids = np.asarray(root["cellData/ids"][:]).astype(str)
    rna = np.asarray(root["RNA/counts"][:])
    adt = np.asarray(root["ADT/counts"][:])
    expected = {
        "left__c0": ([1, 10], [0, 0]),
        "left__c1": ([2, 20], [0, 0]),
        "left__c2": ([3, 30], [0, 0]),
        "right__c0": ([0, 0], [101, 110]),
        "right__c1": ([0, 0], [102, 120]),
        "right__c2": ([0, 0], [103, 130]),
    }
    for cell_id, rna_row, adt_row in zip(
        merged_cell_ids,
        rna,
        adt,
        strict=True,
    ):
        expected_rna, expected_adt = expected[cell_id]
        np.testing.assert_array_equal(rna_row, expected_rna)
        np.testing.assert_array_equal(adt_row, expected_adt)


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
        source_column="sample_id",
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
    assert df["sample_id"].value_counts().to_dict() == {
        "self1": datastore.cells.N,
        "self2": datastore.cells.N,
    }


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
    assert "countsT" not in root["matrices/RNA"]

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

    monkeypatch.setattr("scarf.merge.assays.load_zarr", fake_load_zarr)
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

    monkeypatch.setattr("scarf.merge.assays.load_zarr", fake_load_zarr)
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
