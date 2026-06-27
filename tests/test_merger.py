import pytest
import zarr


def test_assay_merge(datastore, rna_raw_total, tmp_path):
    from scarf.merge import AssayMerge

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
    from scarf.merge import AssayMerge

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


def test_dummy_assay_holds_zero_counts(datastore):
    import zarr
    from zarr.storage import MemoryStore

    from scarf.chunked import ChunkedArray
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
