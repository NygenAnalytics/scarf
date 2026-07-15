import numpy as np
import pytest


def _write_sparse_h5ad(path, values, *, encoding_type="csr_matrix"):
    import h5py
    from scipy.sparse import csr_matrix

    matrix = csr_matrix(values)
    with h5py.File(path, mode="w") as h5:
        sparse = h5.create_group("X")
        sparse.attrs["encoding-type"] = encoding_type
        sparse.attrs["encoding-version"] = "0.1.0"
        sparse.attrs["shape"] = matrix.shape
        sparse.create_dataset("data", data=matrix.data)
        sparse.create_dataset("indices", data=matrix.indices.astype(np.int64))
        sparse.create_dataset("indptr", data=matrix.indptr.astype(np.int64))

        obs = h5.create_group("obs")
        obs.create_dataset(
            "_index",
            data=np.array(
                [f"cell_{index}".encode() for index in range(matrix.shape[0])]
            ),
        )
        var = h5.create_group("var")
        var.create_dataset(
            "_index",
            data=np.array(
                [f"feature_{index}".encode() for index in range(matrix.shape[1])]
            ),
        )
        var.create_dataset(
            "feature_name",
            data=np.array(
                [f"gene_{index}".encode() for index in range(matrix.shape[1])]
            ),
        )
        h5.create_group("obsm")


def test_toy_crdir_assay_feats_table(toy_crdir_reader):
    assert np.all(
        toy_crdir_reader.assayFeats.columns
        == np.array(["RNA", "ADT", "RNA", "HTO", "RNA"])
    )
    assert np.all(
        toy_crdir_reader.assayFeats.values[1:]
        == [[0, 1, 3, 5, 6], [1, 3, 5, 6, 7], [1, 2, 2, 1, 1]]
    )


def test_toy_crdir_reader_cells_feats(toy_crdir_reader):
    assert toy_crdir_reader.nCells == 3
    assert toy_crdir_reader.nFeatures == 8
    assert toy_crdir_reader.cell_names() == ["b1", "b2", "b3"]
    assert toy_crdir_reader.feature_names() == [
        "g1",
        "a1",
        "a2",
        "g2",
        "g3",
        "h1",
        "g4",
        "a3",
    ]
    assert toy_crdir_reader.feature_ids() == [
        "g1",
        "a1",
        "a2",
        "g2",
        "g3",
        "h1",
        "g4",
        "a3",
    ]


def test_toy_crdir_reader_assay_subsets(toy_crdir_reader):
    assert toy_crdir_reader.feature_names("RNA") == ["g1", "g2", "g3", "g4"]
    assert toy_crdir_reader.feature_ids("ADT") == ["a1", "a2"]
    assert toy_crdir_reader.feature_names("HTO") == ["h1"]

    with pytest.raises(ValueError, match="Assay ID missing is not valid"):
        toy_crdir_reader.feature_names("missing")


def test_crdir_reader_filters_and_streams_selected_barcodes(tmp_path):
    from scarf.readers import CrDirReader

    (tmp_path / "features.tsv").write_text(
        "\n".join(
            [
                "f1\tg1\tGene Expression",
                "f2\tg2\tGene Expression",
                "f3\tg3\tGene Expression",
            ]
        )
        + "\n"
    )
    (tmp_path / "barcodes.tsv").write_text("b1\nb2\nb3\nb4\n")
    (tmp_path / "matrix.mtx").write_text(
        "\n".join(
            [
                "%%MatrixMarket matrix coordinate integer general",
                "% tiny deterministic matrix",
                "3 4 5",
                "1 1 2",
                "2 1 3",
                "1 2 6",
                "3 3 1",
                "2 4 4",
            ]
        )
        + "\n"
    )

    reader = CrDirReader(
        str(tmp_path),
        is_filtered=False,
        filtering_cutoff=4,
    )

    np.testing.assert_array_equal(reader.validBarcodeIdx, np.array([1, 2]))
    assert reader.nCells == 2
    assert reader.cell_names() == ["b1", "b2"]
    assert reader.read_header().iloc[0].to_dict() == {
        "nFeatures": 3,
        "nCells": 4,
        "nCounts": 5,
    }
    np.testing.assert_array_equal(
        reader._get_valid_barcodes(
            filtering_cutoff=4,
            batch_size=2,
            lines_in_mem=2,
        ),
        np.array([1, 2]),
    )

    chunks = list(reader.consume(batch_size=1, lines_in_mem=2, dtype=np.uint16))
    assert [chunk.shape for chunk in chunks] == [(1, 3), (1, 3)]
    assert all(chunk.dtype == np.uint16 for chunk in chunks)
    np.testing.assert_array_equal(chunks[0].toarray(), [[2, 3, 0]])
    np.testing.assert_array_equal(chunks[1].toarray(), [[6, 0, 0]])


def test_crdir_reader_supports_gzip_and_metadata_fallback(tmp_path):
    import gzip

    from scarf.readers import CrDirReader

    files = {
        "genes.tsv.gz": "f1\nf2\n",
        "barcodes.tsv.gz": "b1\nb2\n",
        "matrix.mtx.gz": "\n".join(
            [
                "%%MatrixMarket matrix coordinate integer general",
                "2 2 2",
                "1 1 7",
                "2 2 9",
            ]
        )
        + "\n",
    }
    for name, contents in files.items():
        with gzip.open(tmp_path / name, mode="wt") as handle:
            handle.write(contents)

    reader = CrDirReader(str(tmp_path))

    assert reader.feature_ids() == ["f1", "f2"]
    assert reader.feature_names() == ["f1", "f2"]
    assert reader.feature_types() == ["Gene Expression", "Gene Expression"]
    assert reader.cell_names() == ["b1", "b2"]
    with pytest.raises(ValueError, match="Dataset key must be provided"):
        reader._read_dataset()

    chunks = list(reader.consume(batch_size=2, lines_in_mem=1))
    assert len(chunks) == 1
    np.testing.assert_array_equal(chunks[0].toarray(), [[7, 0], [0, 9]])


def test_toy_crdir_empty(toy_crdir_empty):
    assert toy_crdir_empty.nCells == 0
    assert toy_crdir_empty.nFeatures == 4
    assert toy_crdir_empty.feature_names() == [
        "g1",
        "a1",
        "a2",
        "g2",
    ]
    assert toy_crdir_empty.feature_ids() == [
        "g1",
        "a1",
        "a2",
        "g2",
    ]
    # check for raise ValueError
    try:
        toy_crdir_empty.read_header()
    except ValueError:
        pass


def test_crh5reader(crh5_reader):
    assert crh5_reader.nCells == 892
    assert crh5_reader.nFeatures == 36611
    n_assay_feats = list(crh5_reader.assayFeats.T.nFeatures.values)
    assert n_assay_feats == [36601, 10]


def test_crh5reader_streams_counts(crh5_reader):
    streamed_rows = 0
    streamed_nnz = 0

    for chunk in crh5_reader.consume(batch_size=300):
        assert 0 < chunk.shape[0] <= 300
        assert chunk.shape[1] == crh5_reader.nFeatures
        streamed_rows += chunk.shape[0]
        streamed_nnz += chunk.nnz

    assert streamed_rows == crh5_reader.nCells
    assert streamed_nnz == crh5_reader.grp["data"].shape[0]
    assert len(crh5_reader.cell_names()) == crh5_reader.nCells


def test_crh5reader_filters_background_barcodes(crh5_reader):
    from scarf.readers import CrH5Reader

    reader = CrH5Reader(
        crh5_reader.h5obj.filename,
        is_filtered=False,
        filtering_cutoff=0,
    )
    try:
        indptr = reader.grp["indptr"][:]
        expected = np.flatnonzero(np.diff(indptr) > 0)
        np.testing.assert_array_equal(reader.validBarcodeIdx, expected)
        assert reader.cell_names() == list(
            np.asarray(crh5_reader.cell_names())[expected]
        )
    finally:
        reader.close()


def test_crdir_reader(crdir_reader):
    assert crdir_reader.nCells == 892
    assert crdir_reader.nFeatures == 36601  # Does not contain 10 ADTs


def test_h5ad_reader(h5ad_reader):
    assert h5ad_reader.nCells == 3696 == len(h5ad_reader.cell_ids())
    assert h5ad_reader.nFeatures == 27998 == len(h5ad_reader.feat_names())


def test_h5ad_reader_streams_sparse_matrix(h5ad_reader):
    streamed_rows = 0
    streamed_nnz = 0
    streamed_sum = 0.0

    for chunk in h5ad_reader.consume(batch_size=1000):
        assert 0 < chunk.shape[0] <= 1000
        assert chunk.shape[1] == h5ad_reader.nFeatures
        assert chunk.dtype == h5ad_reader.matrixDtype
        streamed_rows += chunk.shape[0]
        streamed_nnz += chunk.nnz
        streamed_sum += chunk.data.sum(dtype=np.float64)

    matrix_data = h5ad_reader.h5[h5ad_reader.matrixKey]["data"]
    assert streamed_rows == h5ad_reader.nCells
    assert streamed_nnz == matrix_data.shape[0]
    np.testing.assert_allclose(
        streamed_sum,
        matrix_data[:].sum(dtype=np.float64),
    )


@pytest.mark.parametrize(
    ("values", "batch_size"),
    [
        (
            np.array(
                [
                    [1, 0, 2],
                    [0, 0, 0],
                    [0, 3, 0],
                    [0, 0, 0],
                ],
                dtype=np.uint32,
            ),
            2,
        ),
        (
            np.array(
                [
                    [1, 0, 0],
                    [0, 2, 0],
                    [0, 0, 0],
                    [3, 0, 4],
                    [0, 5, 0],
                ],
                dtype=np.uint32,
            ),
            2,
        ),
    ],
)
def test_h5ad_reader_preserves_sparse_batches(tmp_path, values, batch_size):
    from scarf.readers import H5adReader

    file_name = tmp_path / "sparse.h5ad"
    _write_sparse_h5ad(file_name, values)
    reader = H5adReader(str(file_name), feature_name_key="feature_name")
    try:
        chunks = list(reader.consume(batch_size=batch_size))
        assert sum(chunk.shape[0] for chunk in chunks) == values.shape[0]
        assert sum(chunk.nnz for chunk in chunks) == np.count_nonzero(values)
        np.testing.assert_array_equal(
            np.vstack([chunk.toarray() for chunk in chunks]),
            values,
        )
    finally:
        reader.h5.close()


def test_h5ad_reader_rejects_csc_sparse_encoding(tmp_path):
    from scarf.readers import H5adReader

    file_name = tmp_path / "csc.h5ad"
    _write_sparse_h5ad(
        file_name,
        np.eye(3, dtype=np.uint32),
        encoding_type="csc_matrix",
    )

    with pytest.raises(ValueError, match="requires CSR encoding"):
        H5adReader(str(file_name), feature_name_key="feature_name")


def test_h5ad_to_zarr_preserves_exact_sparse_batch(tmp_path):
    import zarr

    from scarf.readers import H5adReader
    from scarf.writers import H5adToZarr

    values = np.array(
        [
            [1, 0, 2],
            [0, 0, 0],
            [0, 3, 0],
            [0, 0, 0],
        ],
        dtype=np.uint32,
    )
    file_name = tmp_path / "exact_batch.h5ad"
    zarr_path = tmp_path / "exact_batch.zarr"
    _write_sparse_h5ad(file_name, values)
    reader = H5adReader(str(file_name), feature_name_key="feature_name")
    try:
        writer = H5adToZarr(reader, zarr_loc=str(zarr_path))
        writer.dump(batch_size=2)
    finally:
        reader.h5.close()

    root = zarr.open_group(str(zarr_path), mode="r")
    np.testing.assert_array_equal(root["RNA/counts"][:], values)


def test_h5ad_reader_streams_cell_and_feature_metadata(h5ad_reader):
    cell_columns = dict(h5ad_reader.get_cell_columns())

    assert "index" not in cell_columns
    assert {
        "clusters_coarse",
        "clusters",
        "S_score",
        "G2M_score",
        "X_pca1",
        "X_pca50",
        "X_umap1",
        "X_umap2",
    } <= cell_columns.keys()
    assert all(
        values.shape == (h5ad_reader.nCells,) for values in cell_columns.values()
    )
    np.testing.assert_array_equal(
        cell_columns["clusters_coarse"][:3],
        np.array([b"Pre-endocrine", b"Ductal", b"Endocrine"]),
    )
    np.testing.assert_allclose(
        cell_columns["X_umap1"][:3],
        np.array([6.143066, -9.906417, 7.559791], dtype=np.float32),
    )

    feature_columns = dict(h5ad_reader.get_feat_columns())
    assert feature_columns.keys() == {"highly_variable_genes"}
    assert feature_columns["highly_variable_genes"].shape == (h5ad_reader.nFeatures,)
    np.testing.assert_array_equal(
        feature_columns["highly_variable_genes"][:3],
        np.array([b"False", b"True", b"True"]),
    )
    np.testing.assert_array_equal(
        h5ad_reader._replace_category_values(
            np.array([10_000]),
            "clusters",
            h5ad_reader.cellAttrsKey,
        ),
        np.array([10_000]),
    )
    assert h5ad_reader._check_exists("layers", "spliced")


def test_h5ad_reader_dense_matrix_and_group_metadata(tmp_path):
    import h5py

    from scarf.readers import H5adReader

    file_name = tmp_path / "dense.h5ad"
    with h5py.File(file_name, mode="w") as h5:
        h5.create_dataset(
            "X",
            data=np.array([[1, 0], [0, 2], [3, 4]], dtype=np.float32),
        )
        obs = h5.create_group("obs")
        obs.create_dataset("_index", data=np.array([b"c1", b"c2", b"c3"]))
        obs.create_dataset("batch", data=np.array([0, 1, 0], dtype=np.int8))
        obs_categories = obs.create_group("__categories")
        obs_categories.create_dataset("batch", data=np.array([b"A", b"B"]))

        var = h5.create_group("var")
        var.create_dataset("_index", data=np.array([b"f1", b"f2"]))
        feature_names = var.create_group("gene_short_name")
        feature_names.create_dataset("codes", data=np.array([1, 0]))
        feature_names.create_dataset(
            "categories",
            data=np.array([b"Gene A", b"Gene B"]),
        )
        var.create_dataset("chromosome", data=np.array([b"1", b"2"]))

        obsm = h5.create_group("obsm")
        obsm.create_dataset(
            "X_embed",
            data=np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
        )
        obsm.create_dataset(
            "bad_embed",
            data=np.array([[1, 2], [3, 4]], dtype=np.float32),
        )

    reader = H5adReader(str(file_name))
    try:
        assert reader.groupCodes == {"obs": 2, "var": 2, "obsm": 2, "X": 1}
        np.testing.assert_array_equal(reader.cell_ids(), [b"c1", b"c2", b"c3"])
        np.testing.assert_array_equal(reader.feat_ids(), [b"f1", b"f2"])
        np.testing.assert_array_equal(reader.feat_names(), [b"Gene B", b"Gene A"])

        cell_columns = dict(reader.get_cell_columns())
        assert cell_columns.keys() == {"batch", "X_embed1", "X_embed2"}
        np.testing.assert_array_equal(cell_columns["batch"], [b"A", b"B", b"A"])
        np.testing.assert_array_equal(cell_columns["X_embed2"], [2, 4, 6])

        feature_columns = dict(reader.get_feat_columns())
        assert feature_columns.keys() == {"chromosome"}
        np.testing.assert_array_equal(feature_columns["chromosome"], [b"1", b"2"])

        chunks = [chunk.toarray() for chunk in reader.consume(batch_size=2)]
        assert len(chunks) == 2
        np.testing.assert_array_equal(chunks[0], [[1, 0], [0, 2]])
        np.testing.assert_array_equal(chunks[1], [[3, 4]])
    finally:
        reader.h5.close()


def test_h5ad_reader_falls_back_without_metadata_groups(tmp_path):
    import h5py

    from scarf.readers import H5adReader

    file_name = tmp_path / "sparse_without_metadata.h5ad"
    with h5py.File(file_name, mode="w") as h5:
        matrix = h5.create_group("X")
        matrix.create_dataset("data", data=np.array([5, 8], dtype=np.int16))
        matrix.create_dataset("indices", data=np.array([0, 1]))
        matrix.create_dataset("indptr", data=np.array([0, 1, 2]))
        matrix.create_dataset("shape", data=np.array([2, 2]))

    reader = H5adReader(str(file_name))
    try:
        assert reader.nCells == reader.nFeatures == 2
        np.testing.assert_array_equal(reader.cell_ids(), ["cell_0", "cell_1"])
        np.testing.assert_array_equal(
            reader.feat_ids(),
            ["feature_0", "feature_1"],
        )
        np.testing.assert_array_equal(reader.feat_names(), reader.feat_ids())
        assert list(reader.get_cell_columns()) == []
        assert list(reader.get_feat_columns()) == []

        chunks = list(reader.consume(batch_size=3))
        assert len(chunks) == 1
        np.testing.assert_array_equal(chunks[0].toarray(), [[5, 0], [0, 8]])
    finally:
        reader.h5.close()


def test_loom_reader(loom_reader):
    assert loom_reader.nCells == 298 == len(loom_reader.cell_ids())
    assert loom_reader.nFeatures == 16892 == len(loom_reader.feature_names())


def test_loom_reader_streams_metadata_and_counts(loom_reader):
    cell_attributes = dict(loom_reader.get_cell_attrs())
    assert cell_attributes.keys() == {"Area", "Cell_cluster"}
    assert all(
        values.shape == (loom_reader.nCells,) for values in cell_attributes.values()
    )
    assert dict(loom_reader.get_feature_attrs()) == {}

    chunks = list(loom_reader.consume(batch_size=100))
    assert [chunk.shape for chunk in chunks] == [
        (100, loom_reader.nFeatures),
        (100, loom_reader.nFeatures),
        (98, loom_reader.nFeatures),
    ]
    assert sum(chunk.nnz for chunk in chunks) > 0
