import numpy as np
import pandas as pd
import pytest
import zarr
from zarr.errors import GroupNotFoundError
from zarr.storage import MemoryStore

import scarf.merge.datasets as merge_datasets
from scarf.matrix import ChunkedArray
from scarf.metadata import MetaData
from scarf.merge import DataStoreMerge
from scarf.storage.budget import ResourceBudget
from scarf.storage.layout import count_array_spec


class _MergeMeta:
    def __init__(self, *, block_rows=None, **columns):
        self._columns = {key: np.asarray(value) for key, value in columns.items()}
        self.columns = list(self._columns)
        self.N = len(next(iter(self._columns.values())))
        self._blockRows = self.N if block_rows is None else int(block_rows)

    def to_pandas_dataframe(self, columns):
        return pd.DataFrame({key: self._columns[key] for key in columns})

    def fetch_all(self, key):
        return self._columns[key]

    def get_dtype(self, key):
        return self._columns[key].dtype

    def _get_array(self, key):
        return self._columns[key]

    def default_block_rows(self, column="I"):
        _ = column
        return max(1, min(self.N, self._blockRows))


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
            block_rows=block_size,
            ids=cell_ids,
            names=cell_ids,
            I=np.ones(len(cell_ids), dtype=bool),
        )
        self.feats = _MergeMeta(ids=feature_ids, names=feature_names)


class _MergeDataStore:
    def __init__(self, assays, *, zarr_loc="memory://test"):
        self._assays = {assay.name: assay for assay in assays}
        self.assay_names = list(self._assays)
        self.cells = assays[0].cells
        self.nthreads = 2
        self.memoryBytes = 1024**3
        self.resources = ResourceBudget(self.memoryBytes, self.nthreads)
        self.zarr_loc = zarr_loc
        self.workspace = None

    def get_assay(self, name):
        return self._assays[name]


class _NoFullReadMeta(_MergeMeta):
    def fetch_all(self, key):
        raise AssertionError(f"full metadata read attempted for {key}")


def _merge_two_rna(**kwargs):
    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[1, 10], [2, 20]],
                ["c0", "c1"],
                ["id_a", "id_b"],
                ["A", "B"],
                block_size=2,
            )
        ],
        zarr_loc="memory://left",
    )
    right = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[3, 30], [4, 40]],
                ["c0", "c1"],
                ["id_a", "id_b"],
                ["A", "B"],
                block_size=2,
            )
        ],
        zarr_loc="memory://right",
    )
    defaults = {
        "datasets": [left, right],
        "names": ["left", "right"],
        "prepend_text": "",
        "counts_t": "none",
        "overwrite": True,
        "seed": 0,
    }
    defaults.update(kwargs)
    return DataStoreMerge(**defaults)


def _two_assay_sources():
    cell_ids = ["c0", "c1"]
    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[1, 10], [2, 20]],
                cell_ids,
                ["rna_a", "rna_b"],
                ["RNA A", "RNA B"],
                block_size=2,
            ),
            _MergeAssay(
                "ADT",
                [[101, 110], [102, 120]],
                cell_ids,
                ["adt_a", "adt_b"],
                ["ADT A", "ADT B"],
                block_size=2,
            ),
        ],
        zarr_loc="memory://left",
    )
    right = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[3, 30], [4, 40]],
                cell_ids,
                ["rna_a", "rna_b"],
                ["RNA A", "RNA B"],
                block_size=2,
            ),
            _MergeAssay(
                "ADT",
                [[103, 130], [104, 140]],
                cell_ids,
                ["adt_a", "adt_b"],
                ["ADT A", "ADT B"],
                block_size=2,
            ),
        ],
        zarr_loc="memory://right",
    )
    return left, right


def test_dataset_merge_rejects_summary_before_truncating_destination():
    store = MemoryStore()
    root = zarr.open_group(store=store, mode="w")
    root.create_group("sentinel")
    left = _MergeDataStore(
        [
            _MergeAssay(
                "summary",
                [[1], [2]],
                ["c0", "c1"],
                ["id_a"],
                ["A"],
                block_size=2,
            )
        ],
        zarr_loc="memory://left",
    )
    right = _MergeDataStore(
        [
            _MergeAssay(
                "summary",
                [[3], [4]],
                ["c0", "c1"],
                ["id_a"],
                ["A"],
                block_size=2,
            )
        ],
        zarr_loc="memory://right",
    )
    with pytest.raises(ValueError, match=r"reserved for DataStore\.summary"):
        DataStoreMerge(
            datasets=[left, right],
            zarr_path=store,
            names=["left", "right"],
            counts_t="none",
        ).dump()
    preserved = zarr.open_group(store=store, mode="r")
    assert set(preserved.group_keys()) == {"sentinel"}


def test_dataset_merge(datastore, rna_raw_total, tmp_path):
    fn = str(tmp_path / "merged.zarr")
    writer = DataStoreMerge(
        datasets=[datastore, datastore],
        zarr_path=fn,
        names=["self1", "self2"],
        assays=["RNA"],
        prepend_text="",
        counts_t="none",
        overwrite=True,
    )
    writer.dump()
    rna_count = zarr.open(fn + "/RNA/counts")
    assert rna_count.shape[0] == 2 * datastore.cells.N
    assert int(rna_count[...].sum()) == rna_raw_total * 2


def test_dataset_merge_maps_features_and_preserves_row_order(tmp_path):
    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[1, 10], [2, 20], [3, 30]],
                ["c0", "c1", "c2"],
                ["id_a", "id_b"],
                ["A", "B"],
                block_size=2,
            )
        ],
        zarr_loc="memory://left",
    )
    right = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[4, 40], [5, 50], [6, 60]],
                ["c0", "c1", "c2"],
                ["id_b", "id_c"],
                ["B", "C"],
                block_size=2,
            )
        ],
        zarr_loc="memory://right",
    )
    fn = str(tmp_path / "feature_map.zarr")
    writer = DataStoreMerge(
        datasets=[left, right],
        zarr_path=fn,
        names=["left", "right"],
        prepend_text="",
        counts_t="none",
        seed=1,
        overwrite=True,
    )
    plan = writer.plan()
    assert plan.assays[0].nFeatures == 3
    assert plan.assays[0].featureOverlapFraction == pytest.approx(1 / 3)
    writer.dump()
    root = zarr.open_group(fn, mode="r")
    feat_ids = np.asarray(root["RNA/featureData/ids"][:]).astype(str)
    assert list(feat_ids) == ["id_a", "id_b", "id_c"]
    counts = np.asarray(root["RNA/counts"][:])
    cell_ids = np.asarray(root["cellData/ids"][:]).astype(str)
    expected = {
        "left__c0": [1, 10, 0],
        "left__c1": [2, 20, 0],
        "left__c2": [3, 30, 0],
        "right__c0": [0, 4, 40],
        "right__c1": [0, 5, 50],
        "right__c2": [0, 6, 60],
    }
    for cell_id, row in zip(cell_ids, counts, strict=True):
        np.testing.assert_array_equal(row, expected[cell_id])
    spec = count_array_spec(
        counts.shape[0],
        counts.shape[1],
        dtype=counts.dtype,
        profile="fast_local",
    )
    assert tuple(root["RNA/counts"].chunks) == spec.chunks
    assert "countsT" not in root["RNA"]


@pytest.mark.parametrize("dtype", [None, "uint16"])
def test_dataset_merge_widens_before_consolidating_features(tmp_path, dtype):
    feature_ids = ["gene_0", "gene_1"]
    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                np.array([[200, 100]], dtype=np.uint8),
                ["left"],
                feature_ids,
                feature_ids,
                block_size=1,
            )
        ],
        zarr_loc="memory://left",
    )
    right = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                np.array([[150, 150]], dtype=np.uint8),
                ["right"],
                feature_ids,
                feature_ids,
                block_size=1,
            )
        ],
        zarr_loc="memory://right",
    )
    path = str(tmp_path / "consolidated_features.zarr")
    DataStoreMerge(
        datasets=[left, right],
        zarr_path=path,
        names=["left", "right"],
        prepend_text="",
        dtype=dtype,
        counts_t="none",
        overwrite=True,
    ).dump()
    counts = zarr.open_group(path, mode="r")["RNA/counts"]
    assert counts.dtype == np.dtype("uint16")
    np.testing.assert_array_equal(counts[:], [[300], [300]])


def test_dataset_merge_keeps_source_metadata_aligned_after_permutation(tmp_path):
    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[1], [2], [3], [4]],
                ["c0", "c1", "c2", "c3"],
                ["id_a"],
                ["A"],
                block_size=2,
            )
        ],
        zarr_loc="memory://left",
    )
    left.cells = _MergeMeta(
        ids=["c0", "c1", "c2", "c3"],
        names=["c0", "c1", "c2", "c3"],
        I=np.array([True, False, True, True]),
        cluster_labels=np.array(["a", "b", "c", "d"]),
    )
    left._assays["RNA"].cells = left.cells
    right = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[5], [6], [7], [8]],
                ["c0", "c1", "c2", "c3"],
                ["id_a"],
                ["A"],
                block_size=2,
            )
        ],
        zarr_loc="memory://right",
    )
    right.cells = _MergeMeta(
        ids=["c0", "c1", "c2", "c3"],
        names=["c0", "c1", "c2", "c3"],
        I=np.array([True, True, False, True]),
        cluster_labels=np.array(["e", "f", "g", "h"]),
    )
    right._assays["RNA"].cells = right.cells
    fn = str(tmp_path / "meta.zarr")
    DataStoreMerge(
        datasets=[left, right],
        zarr_path=fn,
        names=["left", "right"],
        prepend_text="orig",
        reset_cell_filter=False,
        source_column="sample_id",
        counts_t="none",
        seed=3,
        overwrite=True,
    ).dump()
    root = zarr.open_group(fn, mode="r")
    ids = np.asarray(root["cellData/ids"][:]).astype(str)
    sample = np.asarray(root["cellData/sample_id"][:]).astype(str)
    labels = np.asarray(root["cellData/orig_cluster_labels"][:]).astype(str)
    included = np.asarray(root["cellData/I"][:])
    expected = {
        "left__c0": ("left", "a", True),
        "left__c1": ("left", "b", False),
        "left__c2": ("left", "c", True),
        "left__c3": ("left", "d", True),
        "right__c0": ("right", "e", True),
        "right__c1": ("right", "f", True),
        "right__c2": ("right", "g", False),
        "right__c3": ("right", "h", True),
    }
    for cell_id, sample_id, label, keep in zip(
        ids, sample, labels, included, strict=True
    ):
        assert (sample_id, label, bool(keep)) == expected[cell_id]


def test_dataset_merge_rejects_source_column_conflict(tmp_path):
    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[1], [2]],
                ["c0", "c1"],
                ["id_a"],
                ["A"],
                block_size=2,
            )
        ],
        zarr_loc="memory://left",
    )
    with pytest.raises(ValueError, match="source_column"):
        DataStoreMerge(
            datasets=[left, left],
            zarr_path=str(tmp_path / "conflict.zarr"),
            names=["left", "right"],
            source_column="ids",
            counts_t="none",
        ).plan()


def test_dataset_merge_preserves_order_across_source_block_sizes(tmp_path):
    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[1, 10], [2, 20], [3, 30], [4, 40]],
                ["c0", "c1", "c2", "c3"],
                ["id_a", "id_b"],
                ["A", "B"],
                block_size=3,
            )
        ],
        zarr_loc="memory://left",
    )
    right = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[5, 50], [6, 60], [7, 70], [8, 80]],
                ["c0", "c1", "c2", "c3"],
                ["id_a", "id_b"],
                ["A", "B"],
                block_size=2,
            )
        ],
        zarr_loc="memory://right",
    )
    fn = str(tmp_path / "blocks.zarr")
    DataStoreMerge(
        datasets=[left, right],
        zarr_path=fn,
        names=["left", "right"],
        prepend_text="",
        counts_t="none",
        seed=2,
        overwrite=True,
    ).dump()
    root = zarr.open_group(fn, mode="r")
    ids = np.asarray(root["cellData/ids"][:]).astype(str)
    counts = np.asarray(root["RNA/counts"][:])
    expected = {
        "left__c0": [1, 10],
        "left__c1": [2, 20],
        "left__c2": [3, 30],
        "left__c3": [4, 40],
        "right__c0": [5, 50],
        "right__c1": [6, 60],
        "right__c2": [7, 70],
        "right__c3": [8, 80],
    }
    for cell_id, row in zip(ids, counts, strict=True):
        np.testing.assert_array_equal(row, expected[cell_id])


def test_dataset_merge_shares_row_order_across_assay_chunk_sizes(tmp_path):
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
        ],
        zarr_loc="memory://left",
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
        ],
        zarr_loc="memory://right",
    )
    fn = str(tmp_path / "mixed_assay_chunks.zarr")
    writer = DataStoreMerge(
        datasets=[left, right],
        zarr_path=fn,
        names=["left", "right"],
        prepend_text="",
        counts_t="none",
        seed=3,
        overwrite=True,
    )
    writer.dump()
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
    for cell_id, rna_row, adt_row in zip(merged_cell_ids, rna, adt, strict=True):
        expected_rna, expected_adt = expected[cell_id]
        np.testing.assert_array_equal(rna_row, expected_rna)
        np.testing.assert_array_equal(adt_row, expected_adt)


def test_dataset_merge_shared_row_plan_handles_missing_assays(tmp_path):
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
        ],
        zarr_loc="memory://left",
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
        ],
        zarr_loc="memory://right",
    )
    fn = str(tmp_path / "missing_assays.zarr")
    DataStoreMerge(
        datasets=[left, right],
        zarr_path=fn,
        names=["left", "right"],
        prepend_text="",
        counts_t="none",
        seed=3,
        overwrite=True,
    ).dump()
    root = zarr.open_group(fn, mode="r")
    merged_cell_ids = np.asarray(root["cellData/ids"][:]).astype(str)
    rna = np.asarray(root["RNA/counts"][:])
    adt = np.asarray(root["ADT/counts"][:])
    rna_i = np.asarray(root["cellData/RNA_I"][:])
    adt_i = np.asarray(root["cellData/ADT_I"][:])
    expected = {
        "left__c0": ([1, 10], [0, 0], True, False),
        "left__c1": ([2, 20], [0, 0], True, False),
        "left__c2": ([3, 30], [0, 0], True, False),
        "right__c0": ([0, 0], [101, 110], False, True),
        "right__c1": ([0, 0], [102, 120], False, True),
        "right__c2": ([0, 0], [103, 130], False, True),
    }
    for cell_id, rna_row, adt_row, rna_flag, adt_flag in zip(
        merged_cell_ids,
        rna,
        adt,
        rna_i,
        adt_i,
        strict=True,
    ):
        expected_rna, expected_adt, expected_rna_i, expected_adt_i = expected[cell_id]
        np.testing.assert_array_equal(rna_row, expected_rna)
        np.testing.assert_array_equal(adt_row, expected_adt)
        assert bool(rna_flag) is expected_rna_i
        assert bool(adt_flag) is expected_adt_i
    assert root["cellData/RNA_I"].attrs["role"] == "assay_membership"
    assert root["cellData/ADT_I"].attrs["assay"] == "ADT"


def test_dataset_merge_missing_assay_policy_error(tmp_path):
    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[1], [2]],
                ["c0", "c1"],
                ["id_a"],
                ["A"],
                block_size=2,
            )
        ],
        zarr_loc="memory://left",
    )
    right = _MergeDataStore(
        [
            _MergeAssay(
                "ADT",
                [[3], [4]],
                ["c0", "c1"],
                ["id_b"],
                ["B"],
                block_size=2,
            )
        ],
        zarr_loc="memory://right",
    )
    with pytest.raises(ValueError, match="missing assay"):
        DataStoreMerge(
            datasets=[left, right],
            zarr_path=str(tmp_path / "missing_error.zarr"),
            names=["left", "right"],
            counts_t="none",
            missing_assay_policy="error",
        ).plan()


@pytest.mark.parametrize(
    ("n_sources", "names"),
    [
        (2, ["self1", "self2"]),
        (3, ["self1", "self2", "self3"]),
    ],
)
def test_dataset_merge_multi_source_totals(
    datastore,
    rna_raw_total,
    assay2_raw_total,
    tmp_path,
    n_sources,
    names,
):
    fn = str(tmp_path / f"merged_{n_sources}.zarr")
    writer = DataStoreMerge(
        zarr_path=fn,
        datasets=[datastore] * n_sources,
        names=names,
        prepend_text="",
        counts_t="none",
        overwrite=True,
    )
    writer.dump()
    rna_count = zarr.open(fn + "/RNA/counts")
    assay2_count = zarr.open(fn + "/assay2/counts")
    assert rna_count.shape[0] == n_sources * datastore.cells.N
    assert assay2_count.shape[0] == n_sources * datastore.cells.N
    assert int(rna_count[...].sum()) == rna_raw_total * n_sources
    assert int(assay2_count[...].sum()) == assay2_raw_total * n_sources


def test_dataset_merge_cells(datastore, tmp_path):
    from scarf.datastore.datastore import DataStore

    fn = str(tmp_path / "merged.zarr")
    writer = DataStoreMerge(
        zarr_path=fn,
        datasets=[datastore, datastore],
        names=["self1", "self2"],
        prepend_text="orig",
        source_column="sample_id",
        counts_t="none",
        overwrite=True,
    )
    writer.dump()
    ds = DataStore(fn, default_assay="RNA")
    df = ds.cells.to_pandas_dataframe(ds.cells.columns)
    df_diff = df[df["orig_RNA_nCounts"] != df["RNA_nCounts"]]
    assert len(df_diff) == 0
    assert df["sample_id"].value_counts().to_dict() == {
        "self1": datastore.cells.N,
        "self2": datastore.cells.N,
    }


def test_dataset_merge_rejects_duplicate_sample_names(datastore, tmp_path):
    fn = str(tmp_path / "merged_dup_names.zarr")
    with pytest.raises(ValueError, match="unique name"):
        DataStoreMerge(
            zarr_path=fn,
            datasets=[datastore, datastore],
            names=["dup", "dup"],
            counts_t="none",
            overwrite=True,
        )


def test_dataset_merge_requires_two_sources(tmp_path):
    source = _merge_two_rna(
        zarr_path=str(tmp_path / "unused.zarr"),
        overwrite=False,
    ).datasets[0]
    with pytest.raises(ValueError, match="at least two"):
        DataStoreMerge(
            datasets=[source],
            zarr_path=str(tmp_path / "single_source.zarr"),
            names=["only"],
            counts_t="none",
        )


def test_dataset_merge_row_order_ignores_unselected_assay_chunks(tmp_path):
    cell_ids = [f"c{i}" for i in range(6)]

    def sources(include_adt):
        stores = []
        for source_index, source_name in enumerate(("left", "right")):
            offset = 100 * source_index
            assays = [
                _MergeAssay(
                    "RNA",
                    [[offset + i] for i in range(6)],
                    cell_ids,
                    ["rna"],
                    ["RNA"],
                    block_size=2,
                )
            ]
            if include_adt:
                assays.append(
                    _MergeAssay(
                        "ADT",
                        [[offset + i] for i in range(6)],
                        cell_ids,
                        ["adt"],
                        ["ADT"],
                        block_size=1,
                    )
                )
            stores.append(
                _MergeDataStore(
                    assays,
                    zarr_loc=f"memory://{source_name}-{include_adt}",
                )
            )
        return stores

    paths = [
        str(tmp_path / "rna_only.zarr"),
        str(tmp_path / "rna_with_unselected_adt.zarr"),
    ]
    for path, include_adt in zip(paths, (False, True), strict=True):
        DataStoreMerge(
            datasets=sources(include_adt),
            zarr_path=path,
            names=["left", "right"],
            assays=["RNA"],
            prepend_text="",
            counts_t="none",
            seed=7,
        ).dump()

    ids_without_adt = np.asarray(
        zarr.open_group(paths[0], mode="r")["cellData/ids"][:],
    ).astype(str)
    ids_with_adt = np.asarray(
        zarr.open_group(paths[1], mode="r")["cellData/ids"][:],
    ).astype(str)
    np.testing.assert_array_equal(ids_with_adt, ids_without_adt)


def test_dataset_merge_workspace_and_counts_t(datastore, rna_raw_total, tmp_path):
    fn = str(tmp_path / "workspace_merged.zarr")
    writer = DataStoreMerge(
        zarr_path=fn,
        datasets=[datastore, datastore],
        names=["self1", "self2"],
        assays=["RNA"],
        out_workspace="merged",
        prepend_text="",
        counts_t="rna",
        overwrite=True,
    )
    writer.dump()
    root = zarr.open_group(fn, mode="r")
    counts = root["matrices/RNA/counts"]
    assert counts.shape[0] == 2 * datastore.cells.N
    assert int(counts[...].sum()) == 2 * rna_raw_total
    assert "countsT" in root["matrices/RNA"]
    assert root["matrices/RNA/countsT"].attrs["complete"] is True


def test_dataset_merge_plan_is_side_effect_free():
    store = MemoryStore()
    merger = _merge_two_rna(zarr_path=store)
    plan = merger.plan()
    assert plan.nCells == 4
    assert plan.assays[0].assayName == "RNA"
    assert plan.assays[0].featureOverlapFraction == 1.0
    with pytest.raises(GroupNotFoundError):
        zarr.open_group(store, mode="r")


def test_dataset_merge_idempotent_resume(tmp_path):
    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[1, 10], [2, 20]],
                ["c0", "c1"],
                ["id_a", "id_b"],
                ["A", "B"],
                block_size=2,
            )
        ],
        zarr_loc="memory://left",
    )
    right = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[3, 30], [4, 40]],
                ["c0", "c1"],
                ["id_a", "id_b"],
                ["A", "B"],
                block_size=2,
            )
        ],
        zarr_loc="memory://right",
    )
    fn = str(tmp_path / "resume.zarr")
    first = DataStoreMerge(
        datasets=[left, right],
        zarr_path=fn,
        names=["left", "right"],
        prepend_text="",
        counts_t="none",
        overwrite=True,
        seed=0,
    )
    first.dump()
    second = DataStoreMerge(
        datasets=[left, right],
        zarr_path=fn,
        names=["left", "right"],
        prepend_text="",
        counts_t="none",
        seed=0,
    )
    result = second.dump()
    assert all(component.action == "skip" for component in result.components)


def test_dataset_merge_metadata_layout_is_budget_independent_and_resumable(
    tmp_path,
    monkeypatch,
):
    import scarf.merge.metadata as merge_metadata
    import scarf.merge.row_plan as merge_row_plan

    def sources():
        cell_ids = [f"c{i}" for i in range(20)]
        left = _MergeDataStore(
            [
                _MergeAssay(
                    "RNA",
                    np.arange(20, dtype=np.uint16).reshape(20, 1),
                    cell_ids,
                    ["id_a"],
                    ["A"],
                    block_size=10,
                )
            ],
            zarr_loc="memory://left",
        )
        right = _MergeDataStore(
            [
                _MergeAssay(
                    "RNA",
                    np.arange(20, 40, dtype=np.uint16).reshape(20, 1),
                    cell_ids,
                    ["id_a"],
                    ["A"],
                    block_size=10,
                )
            ],
            zarr_loc="memory://right",
        )
        left.cells._columns["quality"] = np.arange(20, dtype=np.int16)
        left.cells.columns.append("quality")
        return left, right

    def merger(path, *, memory):
        left, right = sources()
        return DataStoreMerge(
            datasets=[left, right],
            zarr_path=str(path),
            names=["left", "right"],
            prepend_text="",
            counts_t="none",
            overwrite=False,
            seed=0,
            mem_budget=memory,
            nthreads=1,
        )

    original_peak = merge_metadata.CellMetadataPlan.peak_write_bytes_at

    def constrained_peak(self, rows, *, chunk_rows):
        _ = self, chunk_rows
        return 50_000 + max(1, int(rows)) * 5_000

    monkeypatch.setattr(
        merge_metadata.CellMetadataPlan,
        "peak_write_bytes_at",
        constrained_peak,
    )
    low_path = tmp_path / "low_budget_layout.zarr"
    low = merger(low_path, memory=100_000)
    low.plan()
    assert low._metadataPlan is not None
    low_width = low._metadataPlan.blockRows
    low.dump()

    monkeypatch.setattr(
        merge_metadata.CellMetadataPlan,
        "peak_write_bytes_at",
        original_peak,
    )
    high_path = tmp_path / "high_budget_layout.zarr"
    high = merger(high_path, memory=1024**3)
    high.plan()
    assert high._metadataPlan is not None
    high_width = high._metadataPlan.blockRows
    high.dump()
    assert low_width < high_width

    def cell_chunks(path):
        group = zarr.open_group(path, mode="r")["cellData"]
        return {
            name: tuple(int(value) for value in group[name].chunks)
            for name in group.array_keys()
        }

    low_chunks = cell_chunks(low_path)
    assert low_chunks == cell_chunks(high_path)
    assert set(low_chunks.values()) == {(10,)}

    identity_widths: list[int] = []
    original_identity_read = merge_row_plan.read_metadata_rows_chunkwise

    def tracking_identity_read(table, column, rows):
        identity_widths.append(int(np.asarray(rows).size))
        return original_identity_read(table, column, rows)

    monkeypatch.setattr(
        merge_metadata,
        "resolve_identity_validation_rows",
        lambda *args, **kwargs: 3,
    )
    monkeypatch.setattr(
        merge_row_plan,
        "read_metadata_rows_chunkwise",
        tracking_identity_read,
    )
    resumed = merger(low_path, memory=1024**3).plan()
    assert resumed.cellDataAction == "skip"
    assert resumed.canDump is True
    assert identity_widths
    assert max(identity_widths) <= 3


def test_dataset_merge_incomplete_store_is_rejected(tmp_path):
    from scarf.datastore.datastore import DataStore

    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[1], [2]],
                ["c0", "c1"],
                ["id_a"],
                ["A"],
                block_size=2,
            )
        ],
        zarr_loc="memory://left",
    )
    right = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[3], [4]],
                ["c0", "c1"],
                ["id_a"],
                ["A"],
                block_size=2,
            )
        ],
        zarr_loc="memory://right",
    )
    fn = str(tmp_path / "incomplete.zarr")
    DataStoreMerge(
        datasets=[left, right],
        zarr_path=fn,
        names=["left", "right"],
        counts_t="none",
        overwrite=True,
    ).dump()
    root = zarr.open_group(fn, mode="r+")
    root.attrs["scarf:import_complete"] = False
    with pytest.raises(RuntimeError, match="DataStoreMerge import is incomplete"):
        DataStore(fn, nthreads=1)


def test_dataset_merge_partial_metadata_uses_missing_mask(tmp_path):
    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[1], [2]],
                ["c0", "c1"],
                ["id_a"],
                ["A"],
                block_size=2,
            )
        ],
        zarr_loc="memory://left",
    )
    left.cells = _MergeMeta(
        ids=["c0", "c1"],
        names=["c0", "c1"],
        I=np.ones(2, dtype=bool),
        batch=np.array(["x", "y"]),
    )
    left._assays["RNA"].cells = left.cells
    right = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[3], [4]],
                ["c0", "c1"],
                ["id_a"],
                ["A"],
                block_size=2,
            )
        ],
        zarr_loc="memory://right",
    )
    fn = str(tmp_path / "partial_meta.zarr")
    DataStoreMerge(
        datasets=[left, right],
        zarr_path=fn,
        names=["left", "right"],
        prepend_text="",
        counts_t="none",
        overwrite=True,
    ).dump()
    root = zarr.open_group(fn, mode="r")
    assert "batch" in root["cellData"]
    assert "__scarf_missing__batch" in root["cellData"]


def test_dataset_merge_reads_cell_metadata_without_fetch_all(tmp_path):
    left, right = _merge_two_rna(
        zarr_path=str(tmp_path / "unused.zarr"),
        overwrite=False,
    ).datasets
    for store, label in ((left, "left"), (right, "right")):
        metadata = _NoFullReadMeta(
            ids=["c0", "c1"],
            names=["c0", "c1"],
            I=np.ones(2, dtype=bool),
            label=np.array([label, label]),
        )
        store.cells = metadata
        store.get_assay("RNA").cells = metadata

    path = str(tmp_path / "bounded_metadata.zarr")
    DataStoreMerge(
        datasets=[left, right],
        zarr_path=path,
        names=["left", "right"],
        prepend_text="",
        counts_t="none",
        seed=0,
    ).dump()

    root = zarr.open_group(path, mode="r")
    assert root["cellData/label"].shape == (4,)


def test_dataset_merge_preserves_existing_missing_masks(tmp_path):
    def metadata_table(prefix, missing):
        root = zarr.open_group(store=MemoryStore(), mode="w")
        values = {
            "ids": np.array(["c0", "c1"]),
            "names": np.array(["c0", "c1"]),
            "I": np.ones(2, dtype=bool),
            "batch": np.array([f"{prefix}0", f"{prefix}1"]),
        }
        for name, data in values.items():
            root.create_array(name, data=data, chunks=(2,))
        root.create_array(
            "__scarf_missing__batch",
            data=np.asarray(missing, dtype=bool),
            chunks=(2,),
        )
        root["batch"].attrs["missing_mask"] = "__scarf_missing__batch"
        return MetaData(root)

    merger = _merge_two_rna(
        zarr_path=str(tmp_path / "preserved_missing.zarr"),
        overwrite=False,
    )
    for store, metadata in zip(
        merger.datasets,
        (
            metadata_table("l", [False, True]),
            metadata_table("r", [True, False]),
        ),
        strict=True,
    ):
        store.cells = metadata
        store.get_assay("RNA").cells = metadata

    merger.dump()
    root = zarr.open_group(merger.zarr_path, mode="r")
    ids = np.asarray(root["cellData/ids"][:]).astype(str)
    missing = np.asarray(root["cellData/__scarf_missing__batch"][:], dtype=bool)
    expected = {
        "left__c0": False,
        "left__c1": True,
        "right__c0": True,
        "right__c1": False,
    }
    assert {
        cell_id: bool(is_missing)
        for cell_id, is_missing in zip(ids, missing, strict=True)
    } == expected


def test_dataset_merge_finalizes_complete_components_after_interruption(tmp_path):
    path = str(tmp_path / "finalize_only.zarr")
    _merge_two_rna(zarr_path=path, overwrite=False).dump()
    root = zarr.open_group(path, mode="r+")
    root.attrs["scarf:import_complete"] = False
    root.attrs["complete"] = False

    merger = _merge_two_rna(zarr_path=path, overwrite=False)
    plan = merger.plan()
    assert plan.canDump is True
    assert plan.willResume is True
    result = merger.dump()

    assert result.resumed is True
    assert all(component.action == "skip" for component in result.components)
    completed = zarr.open_group(path, mode="r")
    assert completed.attrs["scarf:import_complete"] is True
    assert completed.attrs["complete"] is True


def test_dataset_merge_overwrite_forces_scoped_rebuild(tmp_path):
    path = str(tmp_path / "forced_rebuild.zarr")
    first = _merge_two_rna(zarr_path=path, overwrite=False)
    first.dump()
    root = zarr.open_group(path, mode="r+")
    root.create_group("unrelated")

    first.datasets[0].get_assay("RNA").rawData = ChunkedArray.from_numpy(
        np.full((2, 2), 100, dtype=np.int64),
        block_size=2,
    )
    second = DataStoreMerge(
        datasets=first.datasets,
        zarr_path=path,
        names=["left", "right"],
        prepend_text="",
        counts_t="none",
        overwrite=True,
        seed=0,
    )
    result = second.dump()
    rebuilt = zarr.open_group(path, mode="r")

    assert [(component.name, component.action) for component in result.components] == [
        ("cellData", "write"),
        ("counts:RNA", "write"),
        ("countsT:RNA", "skip"),
    ]
    assert int(np.asarray(rebuilt["RNA/counts"][:]).sum()) == 477
    assert "unrelated" in rebuilt


def test_dataset_merge_instance_can_be_reused(tmp_path):
    path = str(tmp_path / "reuse_instance.zarr")
    merger = _merge_two_rna(zarr_path=path, overwrite=False)

    merger.dump()
    result = merger.dump()

    assert all(component.action == "skip" for component in result.components)


def test_dataset_merge_plan_reports_destination_conflict(tmp_path):
    path = str(tmp_path / "blocked_plan.zarr")
    _merge_two_rna(zarr_path=path, overwrite=False, seed=0).dump()
    blocked = _merge_two_rna(zarr_path=path, overwrite=False, seed=1)

    plan = blocked.plan()

    assert plan.canDump is False
    assert plan.blockedReason is not None
    assert "different configuration" in plan.blockedReason
    assert plan.cellDataAction == "blocked"
    assert plan.assays[0].countsAction == "blocked"
    with pytest.raises(ValueError, match="different configuration"):
        blocked.dump()


@pytest.mark.parametrize(
    ("case", "reason", "counts_t"),
    [
        ("cell_identity", "order of cells", "none"),
        ("counts_shape", "counts shape", "none"),
        ("counts_dtype", "counts dtype", "none"),
        ("counts_chunks", "counts chunks", "none"),
        ("counts_shards", "counts shards", "none"),
        ("feature_ids", "featureData/ids", "none"),
        ("feature_names", "featureData/names", "none"),
        ("feature_selection", "featureData/I", "none"),
        ("metadata_missing_chunks", "missing mask", "none"),
        ("counts_t_shape", "countsT shape", "all"),
        ("counts_t_dtype", "countsT dtype", "all"),
        ("counts_t_chunks", "countsT chunks", "all"),
        ("counts_t_shards", "must be unsharded", "all"),
        ("component_marker", "marked complete", "none"),
        ("import_source", "foreign import source", "none"),
        ("root_import_complete", "marked complete", "none"),
        ("root_complete", "marked complete", "none"),
    ],
)
def test_dataset_merge_plan_blocks_tampered_components(
    tmp_path,
    case,
    reason,
    counts_t,
):
    path = str(tmp_path / f"tampered_{case}.zarr")
    merge_kwargs = (
        {"targetChunkBytes": 16, "targetShardBytes": 32}
        if case
        in {
            "counts_shards",
            "counts_t_shards",
        }
        else {}
    )
    initial = _merge_two_rna(
        zarr_path=path,
        overwrite=False,
        counts_t=counts_t,
        **merge_kwargs,
    )
    if case == "metadata_missing_chunks":
        initial.datasets[0].cells._columns["quality"] = np.array([1, 2])
        initial.datasets[0].cells.columns.append("quality")
    initial.dump()
    root = zarr.open_group(path, mode="r+")
    if case == "cell_identity":
        ids = np.asarray(root["cellData/ids"][:]).astype(str)
        root["cellData/ids"][:] = ids[::-1]
    elif case == "counts_shape":
        del root["RNA/counts"]
        root["RNA"].create_array(
            "counts",
            shape=(4, 1),
            chunks=(4, 1),
            dtype=np.uint16,
        )
    elif case == "counts_dtype":
        values = np.asarray(root["RNA/counts"][:], dtype=np.float32)
        del root["RNA/counts"]
        root["RNA"].create_array("counts", data=values, chunks=(4, 2))
    elif case == "counts_chunks":
        values = np.asarray(root["RNA/counts"][:])
        del root["RNA/counts"]
        root["RNA"].create_array("counts", data=values, chunks=(2, 1))
    elif case == "counts_shards":
        counts = root["RNA/counts"]
        values = np.asarray(counts[:])
        chunks = tuple(int(value) for value in counts.chunks)
        del root["RNA/counts"]
        root["RNA"].create_array(
            "counts",
            data=values,
            chunks=chunks,
            shards=(chunks[0] * 2, chunks[1] * 2),
        )
        root["RNA"].attrs["complete"] = True
    elif case == "feature_ids":
        root["RNA/featureData/ids"][0] = "changed"
    elif case == "feature_names":
        root["RNA/featureData/names"][0] = "changed"
    elif case == "feature_selection":
        root["RNA/featureData/I"][0] = False
    elif case == "metadata_missing_chunks":
        missing_name = "__scarf_missing__quality"
        values = np.asarray(root[f"cellData/{missing_name}"][:], dtype=bool)
        del root[f"cellData/{missing_name}"]
        root["cellData"].create_array(
            missing_name,
            data=values,
            chunks=(1,),
        )
    elif case == "counts_t_shape":
        del root["RNA/countsT"]
        root["RNA"].create_array(
            "countsT",
            shape=(1, 4),
            chunks=(1, 4),
            dtype=np.uint16,
        )
        root["RNA/countsT"].attrs["complete"] = True
    elif case == "counts_t_dtype":
        values = np.asarray(root["RNA/countsT"][:], dtype=np.float32)
        del root["RNA/countsT"]
        root["RNA"].create_array("countsT", data=values, chunks=(2, 4))
        root["RNA/countsT"].attrs["complete"] = True
    elif case == "counts_t_chunks":
        values = np.asarray(root["RNA/countsT"][:])
        del root["RNA/countsT"]
        root["RNA"].create_array("countsT", data=values, chunks=(1, 2))
        root["RNA/countsT"].attrs["complete"] = True
    elif case == "counts_t_shards":
        counts_t_array = root["RNA/countsT"]
        values = np.asarray(counts_t_array[:])
        chunks = tuple(int(value) for value in counts_t_array.chunks)
        del root["RNA/countsT"]
        root["RNA"].create_array(
            "countsT",
            data=values,
            chunks=chunks,
            shards=(chunks[0] * 2, chunks[1]),
        )
        root["RNA/countsT"].attrs["complete"] = True
    elif case == "component_marker":
        root["RNA"].attrs["complete"] = False
    elif case == "import_source":
        root.attrs["scarf:import_source"] = "ForeignImporter"
    elif case == "root_import_complete":
        root["RNA"].attrs["complete"] = False
        root.attrs["scarf:import_complete"] = True
        root.attrs["complete"] = False
    else:
        root["RNA"].attrs["complete"] = False
        root.attrs["scarf:import_complete"] = False
        root.attrs["complete"] = True

    blocked_merger = _merge_two_rna(
        zarr_path=path,
        overwrite=False,
        counts_t=counts_t,
        **merge_kwargs,
    )
    if case == "metadata_missing_chunks":
        blocked_merger.datasets[0].cells._columns["quality"] = np.array([1, 2])
        blocked_merger.datasets[0].cells.columns.append("quality")
    blocked = blocked_merger.plan()
    assert blocked.canDump is False
    assert blocked.blockedReason is not None
    assert reason in blocked.blockedReason

    if case == "counts_shape":
        restarted = _merge_two_rna(zarr_path=path, overwrite=True)
        assert restarted.plan().canDump is True
        restarted.dump()
        assert zarr.open_group(path, mode="r")["RNA/counts"].shape == (4, 2)


def test_dataset_merge_manifest_does_not_persist_source_locations(tmp_path):
    path = str(tmp_path / "safe_manifest.zarr")
    merger = _merge_two_rna(zarr_path=path, overwrite=False)

    plan = merger.plan()
    assert "sourceLocations" not in plan.manifest
    merger.dump()
    stored = zarr.open_group(path, mode="r").attrs["scarf:merge_manifest"]
    assert "sourceLocations" not in stored
    assert stored["sourceFeatureCounts"] == {"RNA": [2, 2]}


def test_dataset_merge_plan_blocks_counts_t_for_zarr_v2(tmp_path):
    path = str(tmp_path / "zarr_v2.zarr")
    zarr.open_group(path, mode="w", zarr_format=2)
    merger = _merge_two_rna(
        zarr_path=path,
        overwrite=True,
        counts_t="all",
    )

    plan = merger.plan()

    assert plan.canDump is False
    assert plan.blockedReason is not None
    assert "Zarr v3" in plan.blockedReason
    with pytest.raises(ValueError, match="Zarr v3"):
        merger.dump()


@pytest.mark.parametrize(
    ("case", "counts_t", "error", "expected"),
    [
        (
            "cell_metadata",
            "none",
            "cellData interruption",
            {
                "cellData": "resume",
                "counts:RNA": "resume",
                "countsT:RNA": "skip",
            },
        ),
        (
            "counts",
            "none",
            "counts interruption",
            {
                "cellData": "skip",
                "counts:RNA": "resume",
                "countsT:RNA": "skip",
            },
        ),
        (
            "counts_t",
            "all",
            "countsT interruption",
            {
                "cellData": "skip",
                "counts:RNA": "skip",
                "countsT:RNA": "resume",
            },
        ),
        (
            "later_assay",
            "none",
            "later-assay interruption",
            {
                "counts:RNA": "skip",
                "counts:ADT": "resume",
            },
        ),
    ],
)
def test_dataset_merge_resumes_after_component_interruption(
    tmp_path,
    monkeypatch,
    case,
    counts_t,
    error,
    expected,
):
    path = str(tmp_path / f"resume_{case}.zarr")
    if case == "later_assay":
        left, right = _two_assay_sources()
        original = merge_datasets.write_assay_counts

        def fail_adt(root, assay_name, *args, **kwargs):
            if assay_name == "ADT":
                raise RuntimeError(f"simulated {error}")
            return original(root, assay_name, *args, **kwargs)

        monkeypatch.setattr(merge_datasets, "write_assay_counts", fail_adt)
        dump_kwargs = {
            "datasets": [left, right],
            "zarr_path": path,
            "names": ["left", "right"],
            "prepend_text": "",
            "counts_t": counts_t,
            "seed": 0,
        }
        with pytest.raises(RuntimeError, match=error):
            DataStoreMerge(**dump_kwargs).dump()
        monkeypatch.setattr(merge_datasets, "write_assay_counts", original)
        result = DataStoreMerge(**dump_kwargs).dump()
        actions = {component.name: component.action for component in result.components}
        for name, action in expected.items():
            assert actions[name] == action
        assert result.resumed is True
        return

    if case == "cell_metadata":
        original = merge_datasets.write_cell_metadata

        def fail_after_write(*args, **kwargs):
            group = original(*args, **kwargs)
            group.attrs["complete"] = False
            raise RuntimeError(f"simulated {error}")

        monkeypatch.setattr(merge_datasets, "write_cell_metadata", fail_after_write)
        restore = ("write_cell_metadata", original)
    elif case == "counts":
        original = merge_datasets.write_assay_counts

        def fail_counts(*args, **kwargs):
            raise RuntimeError(f"simulated {error}")

        monkeypatch.setattr(merge_datasets, "write_assay_counts", fail_counts)
        restore = ("write_assay_counts", original)
    else:
        original = merge_datasets.write_assay_counts_t

        def fail_after_counts_t(*args, **kwargs):
            counts_t_array = original(*args, **kwargs)
            assert counts_t_array is not None
            counts_t_array.attrs["complete"] = False
            raise RuntimeError(f"simulated {error}")

        monkeypatch.setattr(merge_datasets, "write_assay_counts_t", fail_after_counts_t)
        restore = ("write_assay_counts_t", original)

    with pytest.raises(RuntimeError, match=error):
        _merge_two_rna(
            zarr_path=path,
            overwrite=False,
            counts_t=counts_t,
        ).dump()
    monkeypatch.setattr(merge_datasets, restore[0], restore[1])

    if case == "cell_metadata":
        root = zarr.open_group(path, mode="r")
        assert root.attrs["scarf:import_complete"] is False

    result = _merge_two_rna(
        zarr_path=path,
        overwrite=False,
        counts_t=counts_t,
    ).dump()
    actions = {component.name: component.action for component in result.components}
    assert actions == expected
    assert result.resumed is True


def test_dataset_merge_rewrites_counts_t_when_counts_resume(tmp_path):
    path = str(tmp_path / "resume_counts_dependency.zarr")
    _merge_two_rna(
        zarr_path=path,
        overwrite=False,
        counts_t="all",
    ).dump()
    root = zarr.open_group(path, mode="r+")
    root.attrs["scarf:import_complete"] = False
    root.attrs["complete"] = False
    root["RNA"].attrs["complete"] = False
    root["RNA/counts"][:] = 0

    result = _merge_two_rna(
        zarr_path=path,
        overwrite=False,
        counts_t="all",
    ).dump()

    actions = {component.name: component.action for component in result.components}
    assert actions["counts:RNA"] == "resume"
    assert actions["countsT:RNA"] == "resume"
    completed = zarr.open_group(path, mode="r")
    np.testing.assert_array_equal(
        completed["RNA/countsT"][:],
        np.asarray(completed["RNA/counts"][:]).T,
    )


def test_dataset_merge_blocks_inconsistent_complete_marker(tmp_path):
    path = str(tmp_path / "inconsistent_complete.zarr")
    _merge_two_rna(zarr_path=path, overwrite=False).dump()
    root = zarr.open_group(path, mode="r+")
    root["RNA"].attrs["complete"] = False

    plan = _merge_two_rna(zarr_path=path, overwrite=False).plan()

    assert plan.canDump is False
    assert plan.blockedReason is not None
    assert "marked complete" in plan.blockedReason


def test_dataset_merge_workspace_overwrite_preserves_root_siblings(tmp_path):
    path = str(tmp_path / "workspace_preservation.zarr")
    first = _merge_two_rna(
        zarr_path=path,
        overwrite=False,
        out_workspace="merged",
    )
    first.dump()
    root = zarr.open_group(path, mode="r+")
    root.create_group("sentinel")

    _merge_two_rna(
        zarr_path=path,
        overwrite=True,
        out_workspace="merged",
    ).dump()

    completed = zarr.open_group(path, mode="r")
    assert "sentinel" in completed
    assert "merged/cellData" in completed
    assert "matrices/RNA/counts" in completed


def test_dataset_merge_overwrite_removes_old_assay_components(tmp_path):
    path = str(tmp_path / "removed_assay.zarr")
    left, right = _two_assay_sources()
    DataStoreMerge(
        datasets=[left, right],
        zarr_path=path,
        names=["left", "right"],
        prepend_text="",
        counts_t="none",
        seed=0,
    ).dump()

    DataStoreMerge(
        datasets=[left, right],
        zarr_path=path,
        names=["left", "right"],
        assays=["RNA"],
        prepend_text="",
        counts_t="none",
        overwrite=True,
        seed=0,
    ).dump()

    completed = zarr.open_group(path, mode="r")
    assert "RNA" in completed
    assert "ADT" not in completed


def test_dataset_merge_rejects_invalid_workspace_names(tmp_path):
    path = str(tmp_path / "invalid_workspace.zarr")
    for workspace in ("", "matrices", "cellData", "a/b", "artifacts"):
        with pytest.raises(ValueError):
            _merge_two_rna(zarr_path=path, out_workspace=workspace)


def test_dataset_merge_rejects_duplicate_assay_filter(tmp_path):
    path = str(tmp_path / "dup_assays.zarr")
    with pytest.raises(ValueError, match="duplicate assay"):
        _merge_two_rna(zarr_path=path, assays=["RNA", "RNA"])
    assert not (tmp_path / "dup_assays.zarr").exists()


def test_dataset_merge_rejects_source_shape_mismatch(tmp_path):
    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[1, 10], [2, 20]],
                ["c0", "c1"],
                ["id_a", "id_b"],
                ["A", "B"],
                block_size=2,
            )
        ],
        zarr_loc="memory://left",
    )
    right = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[3, 30]],
                ["c0", "c1"],
                ["id_a", "id_b"],
                ["A", "B"],
                block_size=2,
            )
        ],
        zarr_loc="memory://right",
    )
    path = str(tmp_path / "shape_mismatch.zarr")
    with pytest.raises(ValueError, match="rawData has 1 rows"):
        DataStoreMerge(
            datasets=[left, right],
            zarr_path=path,
            names=["left", "right"],
            counts_t="none",
        ).plan()
    assert not (tmp_path / "shape_mismatch.zarr").exists()


def test_dataset_merge_rejects_membership_source_column_collision(tmp_path):
    path = str(tmp_path / "membership_collision.zarr")
    with pytest.raises(ValueError, match="source_column"):
        _merge_two_rna(zarr_path=path, source_column="RNA_I").plan()
    assert not (tmp_path / "membership_collision.zarr").exists()


def test_dataset_merge_rejects_source_destination_alias(tmp_path):
    path = str(tmp_path / "alias.zarr")
    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[1, 10], [2, 20]],
                ["c0", "c1"],
                ["id_a", "id_b"],
                ["A", "B"],
                block_size=2,
            )
        ],
        zarr_loc=path,
    )
    right = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[3, 30], [4, 40]],
                ["c0", "c1"],
                ["id_a", "id_b"],
                ["A", "B"],
                block_size=2,
            )
        ],
        zarr_loc="memory://right",
    )
    merger = DataStoreMerge(
        datasets=[left, right],
        zarr_path=path,
        names=["left", "right"],
        counts_t="none",
    )
    plan = merger.plan()
    assert plan.canDump is False
    assert plan.blockedReason is not None
    assert "aliases source" in plan.blockedReason
    with pytest.raises(ValueError, match="aliases source"):
        merger.dump()


def test_dataset_merge_blocks_cross_workspace_matrix_overwrite(tmp_path):
    path = str(tmp_path / "cross_workspace.zarr")
    _merge_two_rna(
        zarr_path=path,
        overwrite=False,
        out_workspace="ws_a",
    ).dump()
    root = zarr.open_group(path, mode="r")
    before_counts = np.asarray(root["matrices/RNA/counts"][:]).copy()
    before_attrs = dict(root["ws_a"].attrs)

    blocked = _merge_two_rna(
        zarr_path=path,
        overwrite=True,
        out_workspace="ws_b",
    )
    plan = blocked.plan()
    assert plan.canDump is False
    assert plan.blockedReason is not None
    assert "claimed by workspace" in plan.blockedReason
    with pytest.raises(ValueError, match="claimed by workspace"):
        blocked.dump()

    after = zarr.open_group(path, mode="r")
    np.testing.assert_array_equal(after["matrices/RNA/counts"][:], before_counts)
    assert dict(after["ws_a"].attrs) == before_attrs
    assert "ws_b" not in after


def test_dataset_merge_blocks_orphaned_matrix_slot(tmp_path):
    path = str(tmp_path / "orphan_matrix.zarr")
    root = zarr.open_group(path, mode="w")
    matrices = root.create_group("matrices")
    rna = matrices.create_group("RNA")
    rna.create_array("counts", data=np.ones((2, 2), dtype=np.uint16))
    before = np.asarray(rna["counts"][:]).copy()

    blocked = _merge_two_rna(zarr_path=path, out_workspace="merged", overwrite=True)
    plan = blocked.plan()
    assert plan.canDump is False
    assert plan.blockedReason is not None
    assert "orphaned" in plan.blockedReason
    with pytest.raises(ValueError, match="orphaned"):
        blocked.dump()

    after = zarr.open_group(path, mode="r")
    np.testing.assert_array_equal(after["matrices/RNA/counts"][:], before)
    assert "merged" not in after


def test_dataset_merge_missing_modality_excluded_from_overlap(tmp_path):
    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[1, 10], [2, 20]],
                ["c0", "c1"],
                ["id_a", "id_b"],
                ["A", "B"],
                block_size=2,
            )
        ],
        zarr_loc="memory://left",
    )
    right = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[3, 30], [4, 40]],
                ["c0", "c1"],
                ["id_c", "id_d"],
                ["C", "D"],
                block_size=2,
            )
        ],
        zarr_loc="memory://right",
    )
    missing = _MergeDataStore(
        [
            _MergeAssay(
                "ADT",
                [[9], [8]],
                ["c0", "c1"],
                ["adt"],
                ["ADT"],
                block_size=2,
            )
        ],
        zarr_loc="memory://missing",
    )
    with pytest.raises(ValueError, match="No overlapping features"):
        DataStoreMerge(
            datasets=[left, right, missing],
            zarr_path=str(tmp_path / "disjoint_missing.zarr"),
            names=["left", "right", "missing"],
            assays=["RNA"],
            counts_t="none",
        ).plan()


def test_dataset_merge_missing_assay_plan_fields(tmp_path):
    cell_ids = ["c0", "c1"]
    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[1, 10], [2, 20]],
                cell_ids,
                ["rna_a", "rna_b"],
                ["RNA A", "RNA B"],
                block_size=2,
            )
        ],
        zarr_loc="memory://left",
    )
    right = _MergeDataStore(
        [
            _MergeAssay(
                "ADT",
                [[101], [102]],
                cell_ids,
                ["adt_a"],
                ["ADT A"],
                block_size=2,
            )
        ],
        zarr_loc="memory://right",
    )
    path = str(tmp_path / "missing_plan.zarr")
    merger = DataStoreMerge(
        datasets=[left, right],
        zarr_path=path,
        names=["left", "right"],
        prepend_text="",
        counts_t="none",
        seed=0,
    )
    plan = merger.plan()
    rna = next(item for item in plan.assays if item.assayName == "RNA")
    adt = next(item for item in plan.assays if item.assayName == "ADT")
    assert rna.sourcePresent == (True, False)
    assert rna.missingSources == ("right",)
    assert rna.nFeatures == 2
    assert rna.featureOverlapFraction == 1.0
    assert adt.sourcePresent == (False, True)
    assert adt.missingSources == ("left",)

    result = merger.dump()
    root = zarr.open_group(path, mode="r")
    assert result.nCells == 4
    for cell_id, rna_row, rna_flag in zip(
        np.asarray(root["cellData/ids"][:]).astype(str),
        np.asarray(root["RNA/counts"][:]),
        np.asarray(root["cellData/RNA_I"][:]),
        strict=True,
    ):
        if cell_id.startswith("right__"):
            np.testing.assert_array_equal(rna_row, [0, 0])
            assert bool(rna_flag) is False
        else:
            assert bool(rna_flag) is True


def test_dataset_merge_default_counts_t_follows_real_assay_type(
    datastore,
    tmp_path,
):
    rna_path = str(tmp_path / "default_rna.zarr")
    rna_plan = DataStoreMerge(
        datasets=[datastore, datastore],
        zarr_path=rna_path,
        names=["a", "b"],
        assays=["RNA"],
        prepend_text="",
        overwrite=True,
    ).plan()
    assert rna_plan.assays[0].writeCountsT is True

    adt_only = _MergeDataStore(
        [
            _MergeAssay(
                "ADT",
                [[1, 2], [3, 4]],
                ["c0", "c1"],
                ["a", "b"],
                ["A", "B"],
                block_size=2,
            )
        ],
        zarr_loc="memory://adt_left",
    )
    adt_right = _MergeDataStore(
        [
            _MergeAssay(
                "ADT",
                [[5, 6], [7, 8]],
                ["c0", "c1"],
                ["a", "b"],
                ["A", "B"],
                block_size=2,
            )
        ],
        zarr_loc="memory://adt_right",
    )
    adt_plan = DataStoreMerge(
        datasets=[adt_only, adt_right],
        zarr_path=str(tmp_path / "default_adt.zarr"),
        names=["left", "right"],
        prepend_text="",
        overwrite=True,
    ).plan()
    assert adt_plan.assays[0].writeCountsT is False

    mock_plan = _merge_two_rna(zarr_path=str(tmp_path / "mock_rna.zarr")).plan()
    assert mock_plan.assays[0].writeCountsT is False


def test_dataset_merge_resumes_after_partial_counts_band(tmp_path, monkeypatch):
    import scarf.merge.writer as merge_writer

    path = str(tmp_path / "partial_band.zarr")
    original = merge_writer.accumulate_sparse_to_shards

    def fail_after_persisting(dst, data_stream, **kwargs):
        rows = original(dst, data_stream, **kwargs)
        assert rows > 0
        assert int(np.asarray(dst[:]).sum()) > 0
        raise RuntimeError("simulated partial counts band")

    monkeypatch.setattr(
        merge_writer,
        "accumulate_sparse_to_shards",
        fail_after_persisting,
    )
    with pytest.raises(RuntimeError, match="partial counts band"):
        _merge_two_rna(
            zarr_path=path,
            overwrite=False,
            counts_t="all",
        ).dump()
    monkeypatch.setattr(
        merge_writer,
        "accumulate_sparse_to_shards",
        original,
    )

    interrupted = zarr.open_group(path, mode="r+")
    interrupted.create_group("sentinel")
    assert interrupted.attrs.get("scarf:import_complete") is not True
    assert interrupted["RNA"].attrs.get("complete") is not True

    result = _merge_two_rna(
        zarr_path=path,
        overwrite=False,
        counts_t="all",
    ).dump()
    actions = {component.name: component.action for component in result.components}
    assert actions["counts:RNA"] == "resume"
    assert actions["countsT:RNA"] == "resume"
    completed = zarr.open_group(path, mode="r")
    assert completed.attrs["scarf:import_complete"] is True
    assert completed.attrs["complete"] is True
    assert completed["RNA"].attrs["complete"] is True
    assert completed["RNA/countsT"].attrs["complete"] is True
    np.testing.assert_array_equal(
        completed["RNA/countsT"][:],
        np.asarray(completed["RNA/counts"][:]).T,
    )
    assert "sentinel" in completed
    assert int(np.asarray(completed["RNA/counts"][:]).sum()) == 110


def test_dataset_merge_metadata_admission_bounds_selection(tmp_path, monkeypatch):
    import scarf.merge.metadata as merge_metadata
    import scarf.merge.row_plan as merge_row_plan

    widths: list[int] = []
    original_meta = merge_metadata.read_metadata_rows_chunkwise

    def tracking_read(table, column, rows):
        widths.append(int(np.asarray(rows).size))
        return original_meta(table, column, rows)

    monkeypatch.setattr(
        merge_metadata,
        "read_metadata_rows_chunkwise",
        tracking_read,
    )
    monkeypatch.setattr(
        merge_row_plan,
        "read_metadata_rows_chunkwise",
        tracking_read,
    )
    path = str(tmp_path / "meta_admit.zarr")
    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                np.arange(20, dtype=np.int64).reshape(10, 2),
                [f"c{i}" for i in range(10)],
                ["id_a", "id_b"],
                ["A", "B"],
                block_size=3,
            )
        ],
        zarr_loc="memory://left",
    )
    right = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                np.arange(20, 40, dtype=np.int64).reshape(10, 2),
                [f"c{i}" for i in range(10)],
                ["id_a", "id_b"],
                ["A", "B"],
                block_size=4,
            )
        ],
        zarr_loc="memory://right",
    )
    DataStoreMerge(
        datasets=[left, right],
        zarr_path=path,
        names=["left", "right"],
        prepend_text="",
        counts_t="none",
        seed=0,
        overwrite=True,
    ).dump()
    assert widths
    assert max(widths) <= 4


def test_dataset_merge_schema_scan_uses_admitted_width_without_changing_schema(
    tmp_path,
    monkeypatch,
):
    import scarf.merge.metadata as merge_metadata

    cell_ids = [f"c{i}" for i in range(7)] + ["identifier_is_longest"]
    cell_names = [f"cell {i}" for i in range(7)] + ["longest cell name"]

    def source(name, offset):
        assay = _MergeAssay(
            "RNA",
            np.arange(offset, offset + 8, dtype=np.uint16).reshape(8, 1),
            cell_ids,
            ["id_a"],
            ["A"],
            block_size=4,
        )
        assay.cells._columns["names"] = np.asarray(cell_names)
        return _MergeDataStore([assay], zarr_loc=f"memory://{name}")

    widths: list[int] = []
    original_blocks = merge_metadata.iter_metadata_column_blocks

    def tracking_blocks(*args, **kwargs):
        for block in original_blocks(*args, **kwargs):
            widths.append(int(block.size))
            yield block

    def admitted_scan_rows(
        source_cell_tables,
        row_plan,
        resources,
        *,
        resident_bytes,
        preferred_rows,
    ):
        _ = source_cell_tables, row_plan, resources, resident_bytes
        assert preferred_rows == 4
        return 2

    monkeypatch.setattr(
        merge_metadata,
        "iter_metadata_column_blocks",
        tracking_blocks,
    )
    monkeypatch.setattr(
        merge_datasets,
        "resolve_metadata_schema_scan_rows",
        admitted_scan_rows,
    )
    path = tmp_path / "schema_scan.zarr"
    merger = DataStoreMerge(
        datasets=[source("left", 0), source("right", 8)],
        zarr_path=str(path),
        names=["left", "right"],
        prepend_text="",
        counts_t="none",
        overwrite=True,
        seed=0,
    )
    merger.plan()

    assert widths
    assert max(widths) == 2
    assert merger._metadataPlan is not None
    assert merger._metadataPlan.blockRows == 4
    specs = {spec.name: spec for spec in merger._metadataPlan.columns}
    assert specs["ids"].dtype == np.dtype(f"U{len('right') + 2 + len(cell_ids[-1])}")
    assert specs["names"].dtype == np.dtype(f"U{len(cell_names[-1])}")
    assert not path.exists()


def test_dataset_merge_metadata_admission_shrinks_under_budget(monkeypatch):
    import scarf.merge.metadata as merge_metadata
    from scarf.merge.row_plan import build_row_plan
    from scarf.storage.budget import ResourceBudget

    left = _MergeMeta(
        block_rows=10,
        ids=[f"c{i}" for i in range(20)],
        names=[f"c{i}" for i in range(20)],
        I=np.ones(20, dtype=bool),
    )
    right = _MergeMeta(
        block_rows=10,
        ids=[f"c{i}" for i in range(20)],
        names=[f"c{i}" for i in range(20)],
        I=np.ones(20, dtype=bool),
    )
    metadata_plan = merge_metadata.plan_cell_metadata(
        [left, right],
        ["left", "right"],
        prepend_text="",
        reset_cell_filter=True,
        source_column=None,
        membership_assays=["RNA"],
        block_rows=10,
    )
    row_plan = build_row_plan([20, 20], [10, 10], ["left", "right"], seed=0)
    assert metadata_plan.blockRows == 10

    def inflated(self, rows, *, chunk_rows):
        _ = chunk_rows
        return max(1, int(rows)) * 1_000

    monkeypatch.setattr(
        merge_metadata.CellMetadataPlan,
        "peak_write_bytes_at",
        inflated,
    )
    resources = ResourceBudget(4_500, 1)
    admitted = merge_metadata.resolve_metadata_segment_rows(
        metadata_plan,
        row_plan,
        resources,
        resident_bytes=2_000,
    )
    assert admitted < 10
    assert admitted >= 1

    updated = merge_metadata.admit_cell_metadata_plan(
        metadata_plan,
        row_plan,
        resources,
        resident_bytes=2_000,
    )
    assert updated.blockRows == admitted
    assert updated.columns == metadata_plan.columns


def test_dataset_merge_sparse_write_respects_admitted_batch_geometry(
    tmp_path,
    monkeypatch,
):
    import scarf.merge.writer as merge_writer

    path = str(tmp_path / "sparse_geometry.zarr")
    reference_path = str(tmp_path / "sparse_geometry_ref.zarr")
    n_left = 12
    n_right = 12
    n_feats = 8
    left_counts = np.arange(n_left * n_feats, dtype=np.uint16).reshape(n_left, n_feats)
    right_counts = (
        np.arange(n_right * n_feats, dtype=np.uint16).reshape(n_right, n_feats) + 100
    )
    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                left_counts,
                [f"c{i}" for i in range(n_left)],
                [f"id_{i}" for i in range(n_feats)],
                [f"G{i}" for i in range(n_feats)],
                block_size=4,
            )
        ],
        zarr_loc="memory://left",
    )
    right = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                right_counts,
                [f"c{i}" for i in range(n_right)],
                [f"id_{i}" for i in range(n_feats)],
                [f"G{i}" for i in range(n_feats)],
                block_size=3,
            )
        ],
        zarr_loc="memory://right",
    )
    left.cells._columns["RNA_nFeatures"] = np.full(n_left, n_feats, dtype=np.int64)
    left.cells.columns.append("RNA_nFeatures")
    right.cells._columns["RNA_nFeatures"] = np.full(n_right, n_feats, dtype=np.int64)
    right.cells.columns.append("RNA_nFeatures")

    DataStoreMerge(
        datasets=[left, right],
        zarr_path=reference_path,
        names=["left", "right"],
        prepend_text="",
        counts_t="none",
        seed=0,
        overwrite=True,
        mem_budget=256 * 1024,
        nthreads=1,
        targetChunkBytes=32,
        targetShardBytes=64,
    ).dump()

    nnz_widths: list[int] = []
    dense_rows: list[int] = []
    coo_rows: list[int] = []
    batch_rows: list[int] = []
    nnz_scan_rows: list[int] = []
    original_read = merge_writer.read_metadata_rows_chunkwise
    original_remap = merge_writer.remap_block_to_coo
    original_batch = merge_writer.resolve_sparse_import_batch
    original_requirements = merge_writer._merge_import_requirements

    def tracking_read(table, column, rows):
        if column.endswith("_nFeatures"):
            nnz_widths.append(int(np.asarray(rows).size))
        return original_read(table, column, rows)

    def tracking_remap(block, order_map, n_feats, n_threads, destination_dtype=None):
        result = original_remap(
            block,
            order_map,
            n_feats,
            n_threads,
            destination_dtype,
        )
        dense_rows.append(int(result.shape[0]))
        coo_rows.append(int(result.shape[0]))
        return result

    def tracking_batch(*args, **kwargs):
        plan = original_batch(*args, **kwargs)
        batch_rows.append(int(plan.batchRows))
        return plan

    def tracking_requirements(*args, **kwargs):
        requirements = original_requirements(*args, **kwargs)
        nnz_scan_rows.append(int(requirements.nnzScanRows))
        return requirements

    monkeypatch.setattr(
        merge_writer,
        "read_metadata_rows_chunkwise",
        tracking_read,
    )
    monkeypatch.setattr(merge_writer, "remap_block_to_coo", tracking_remap)
    monkeypatch.setattr(merge_writer, "resolve_sparse_import_batch", tracking_batch)
    monkeypatch.setattr(
        merge_writer,
        "_merge_import_requirements",
        tracking_requirements,
    )

    DataStoreMerge(
        datasets=[left, right],
        zarr_path=path,
        names=["left", "right"],
        prepend_text="",
        counts_t="none",
        seed=0,
        overwrite=True,
        mem_budget=256 * 1024,
        nthreads=1,
        targetChunkBytes=32,
        targetShardBytes=64,
    ).dump()

    assert batch_rows
    admitted = max(batch_rows)
    assert nnz_scan_rows
    assert max(nnz_scan_rows) >= 1
    assert nnz_widths
    assert max(nnz_widths) <= max(nnz_scan_rows)
    assert dense_rows
    assert coo_rows
    assert all(rows <= admitted for rows in dense_rows)
    assert all(rows <= admitted for rows in coo_rows)
    assert sum(coo_rows) == n_left + n_right
    reference = zarr.open_group(reference_path, mode="r")
    completed = zarr.open_group(path, mode="r")
    np.testing.assert_array_equal(
        completed["RNA/counts"][:],
        reference["RNA/counts"][:],
    )
    np.testing.assert_array_equal(
        completed["cellData/ids"][:],
        reference["cellData/ids"][:],
    )


def test_dataset_merge_producer_reserve_uses_source_feature_width():
    import scarf.merge.writer as merge_writer
    from scarf.merge.features import align_features
    from scarf.merge.row_plan import build_row_plan
    from scarf.storage.budget import ResourceBudget

    def assay(counts, feature_ids, feature_names, cell_ids):
        return _MergeAssay(
            "RNA",
            counts,
            cell_ids,
            feature_ids,
            feature_names,
            block_size=len(cell_ids),
        )

    cell_ids = ["c0", "c1", "c2", "c3"]
    # Partial overlap: union is wider than either source matrix.
    partial = [
        assay(
            np.ones((4, 3), dtype=np.uint16),
            ["a", "b", "c"],
            ["A", "B", "C"],
            cell_ids,
        ),
        assay(
            np.ones((4, 4), dtype=np.uint16) * 2,
            ["a", "d", "e", "f"],
            ["A", "D", "E", "F"],
            cell_ids,
        ),
    ]
    dense = [
        assay(
            np.ones((4, 5), dtype=np.uint16),
            ["a", "b", "c", "d", "e"],
            ["A", "B", "C", "D", "E"],
            cell_ids,
        ),
        assay(
            np.ones((4, 5), dtype=np.uint16) * 3,
            ["a", "b", "c", "d", "e"],
            ["A", "B", "C", "D", "E"],
            cell_ids,
        ),
    ]
    collapsed = [
        assay(
            np.ones((4, 2), dtype=np.uint16),
            ["gene_0", "gene_1"],
            ["gene_0", "gene_1"],
            cell_ids,
        ),
        assay(
            np.ones((4, 2), dtype=np.uint16) * 4,
            ["gene_0", "gene_1"],
            ["gene_0", "gene_1"],
            cell_ids,
        ),
    ]
    resources = ResourceBudget(1024**3, 1)

    def requirements_for(assays):
        alignment = align_features(assays, ["left", "right"])
        row_plan = build_row_plan(
            [4, 4],
            [4, 4],
            ["left", "right"],
            seed=0,
        )
        return alignment, merge_writer._merge_import_requirements(
            assays,
            row_plan,
            alignment,
            np.dtype(np.uint16),
            resources=resources,
        )

    alignment, requirements = requirements_for(partial)
    source_width = 4
    assert alignment.nFeats == 6
    assert alignment.nFeats > source_width
    value_bytes = np.dtype(np.uint16).itemsize
    index_bytes = 2 * np.dtype(np.int32).itemsize
    expected = 2 * source_width * value_bytes + 2 * source_width * (
        value_bytes + index_bytes
    )
    # Ignore backing decode, which is source-dependent and additive.
    assert requirements.extraProducerBytes(2) >= expected
    assert requirements.extraProducerBytes(2) < (
        2 * alignment.nFeats * value_bytes
        + 2 * alignment.nFeats * (value_bytes + index_bytes)
        + 1024**2
    )

    dense_alignment, dense_requirements = requirements_for(dense)
    assert dense_alignment.nFeats == 5
    dense_expected = 2 * 5 * value_bytes + 2 * 5 * (value_bytes + index_bytes)
    assert dense_requirements.extraProducerBytes(2) >= dense_expected
    assert dense_requirements.extraProducerBytes(
        4
    ) > dense_requirements.extraProducerBytes(2)

    collapsed_alignment, collapsed_requirements = requirements_for(collapsed)
    assert collapsed_alignment.nFeats == 1
    collapsed_expected = 2 * 2 * value_bytes + 2 * 2 * (value_bytes + index_bytes)
    destination_only = (
        2 * collapsed_alignment.nFeats * value_bytes
        + 2 * collapsed_alignment.nFeats * (value_bytes + index_bytes)
    )
    assert collapsed_requirements.extraProducerBytes(2) >= collapsed_expected
    assert collapsed_requirements.extraProducerBytes(2) > destination_only

    with pytest.raises(ValueError, match="No overlapping features"):
        align_features(
            [
                assay(
                    np.ones((4, 2), dtype=np.uint16),
                    ["a", "b"],
                    ["A", "B"],
                    cell_ids,
                ),
                assay(
                    np.ones((4, 2), dtype=np.uint16) * 2,
                    ["c", "d"],
                    ["C", "D"],
                    cell_ids,
                ),
            ],
            ["left", "right"],
        )


def test_dataset_merge_rejects_insufficient_metadata_budget(tmp_path):
    path = str(tmp_path / "meta_budget.zarr")
    merger = _merge_two_rna(
        zarr_path=path,
        mem_budget=64,
        nthreads=1,
    )
    with pytest.raises(MemoryError, match="schema discovery"):
        merger.plan()
    assert not (tmp_path / "meta_budget.zarr").exists()


def test_dataset_merge_rejects_insufficient_counts_budget(tmp_path):
    path = str(tmp_path / "counts_budget.zarr")
    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                np.ones((8, 32), dtype=np.uint16),
                [f"c{i}" for i in range(8)],
                [f"id_{i}" for i in range(32)],
                [f"G{i}" for i in range(32)],
                block_size=4,
            )
        ],
        zarr_loc="memory://left",
    )
    right = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                np.ones((8, 32), dtype=np.uint16) * 2,
                [f"c{i}" for i in range(8)],
                [f"id_{i}" for i in range(32)],
                [f"G{i}" for i in range(32)],
                block_size=4,
            )
        ],
        zarr_loc="memory://right",
    )
    merger = DataStoreMerge(
        datasets=[left, right],
        zarr_path=path,
        names=["left", "right"],
        prepend_text="",
        counts_t="none",
        mem_budget=2_000,
        nthreads=1,
        overwrite=True,
    )
    with pytest.raises(MemoryError):
        merger.plan()
    assert not (tmp_path / "counts_budget.zarr").exists()
