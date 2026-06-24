import zarr


def test_assay_merge(datastore, rna_raw_total, tmp_path):
    from ..merge import AssayMerge

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
    from ..merge import DatasetMerge

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
    from ..merge import DatasetMerge

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
    from ..datastore.datastore import DataStore
    from ..merge import DatasetMerge

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
