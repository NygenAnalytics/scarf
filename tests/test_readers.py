import numpy as np
import pytest


def _write_sparse_group(parent, key, values, *, encoding_type="csr_matrix"):
    from scipy.sparse import csc_matrix, csr_matrix

    matrix_type = csc_matrix if encoding_type in {"csc", "csc_matrix"} else csr_matrix
    matrix = matrix_type(values)
    sparse = parent.create_group(key)
    sparse.attrs["encoding-type"] = encoding_type
    sparse.attrs["encoding-version"] = "0.1.0"
    sparse.attrs["shape"] = matrix.shape
    sparse.create_dataset("data", data=matrix.data)
    sparse.create_dataset("indices", data=matrix.indices.astype(np.int64))
    sparse.create_dataset("indptr", data=matrix.indptr.astype(np.int64))
    return matrix


def _write_sparse_h5ad(path, values, *, encoding_type="csr_matrix"):
    import h5py

    with h5py.File(path, mode="w") as h5:
        matrix = _write_sparse_group(
            h5,
            "X",
            values,
            encoding_type=encoding_type,
        )

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


def _write_cr_h5(path, values, *, legacy=False):
    import h5py
    from scipy.sparse import csr_matrix

    matrix = csr_matrix(values)
    with h5py.File(path, mode="w") as h5:
        group = h5.create_group("genome" if legacy else "matrix")
        group.create_dataset("data", data=matrix.data)
        group.create_dataset("indices", data=matrix.indices.astype(np.int64))
        group.create_dataset("indptr", data=matrix.indptr.astype(np.int64))
        group.create_dataset(
            "barcodes",
            data=np.array(
                [f"cell_{index}".encode() for index in range(values.shape[0])]
            ),
        )
        if legacy:
            group.create_dataset(
                "genes",
                data=np.array(
                    [f"feature_{index}".encode() for index in range(values.shape[1])]
                ),
            )
            group.create_dataset(
                "gene_names",
                data=np.array(
                    [f"gene_{index}".encode() for index in range(values.shape[1])]
                ),
            )
        else:
            features = group.create_group("features")
            features.create_dataset(
                "id",
                data=np.array(
                    [f"feature_{index}".encode() for index in range(values.shape[1])]
                ),
            )
            features.create_dataset(
                "name",
                data=np.array(
                    [f"gene_{index}".encode() for index in range(values.shape[1])]
                ),
            )
            features.create_dataset(
                "feature_type",
                data=np.array(
                    ["Gene Expression".encode() for _ in range(values.shape[1])]
                ),
            )


def test_toy_crdir_assay_feats_table(toy_crdir_reader):
    assert np.all(
        toy_crdir_reader.assayFeats.columns
        == np.array(["RNA", "ADT", "RNA", "HTO", "RNA", "ADT"])
    )
    assert np.all(
        toy_crdir_reader.assayFeats.values[1:]
        == [
            [0, 1, 3, 5, 6, 7],
            [1, 3, 5, 6, 7, 8],
            [1, 2, 2, 1, 1, 1],
        ]
    )
    assert toy_crdir_reader.assayFeats.loc["nFeatures"].sum() == 8


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
    assert toy_crdir_reader.feature_ids("ADT") == ["a1", "a2", "a3"]
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


def test_crdir_reader_splits_many_cells_from_one_input_chunk(tmp_path):
    from scarf.readers import CrDirReader

    (tmp_path / "features.tsv").write_text("f1\tg1\tGene Expression\n")
    (tmp_path / "barcodes.tsv").write_text("".join(f"b{index}\n" for index in range(5)))
    (tmp_path / "matrix.mtx").write_text(
        "\n".join(
            [
                "%%MatrixMarket matrix coordinate integer general",
                "1 5 5",
                *(f"1 {index + 1} {index + 1}" for index in range(5)),
            ]
        )
        + "\n"
    )

    reader = CrDirReader(str(tmp_path))
    chunks = list(reader.consume(batch_size=1, lines_in_mem=100))

    assert reader.producer_staging_bytes(1, 100) > (
        100 * 3 * np.dtype(np.int64).itemsize
    )
    assert [chunk.shape for chunk in chunks] == [(1, 1)] * 5
    np.testing.assert_array_equal(
        np.vstack([chunk.toarray() for chunk in chunks]),
        np.arange(1, 6).reshape(-1, 1),
    )


def test_crdir_reader_coalesces_duplicates_across_input_chunks(tmp_path):
    from scarf.readers import CrDirReader

    n_entries = 1_000
    (tmp_path / "features.tsv").write_text("f1\tg1\tGene Expression\n")
    (tmp_path / "barcodes.tsv").write_text("b1\n")
    (tmp_path / "matrix.mtx").write_text(
        "\n".join(
            [
                "%%MatrixMarket matrix coordinate integer general",
                f"1 1 {n_entries}",
                *("1 1 1" for _ in range(n_entries)),
            ]
        )
        + "\n"
    )

    reader = CrDirReader(str(tmp_path))
    chunks = list(
        reader.consume(
            batch_size=1,
            lines_in_mem=100,
            dtype=np.uint32,
        )
    )

    assert len(chunks) == 1
    assert chunks[0].nnz == 1
    np.testing.assert_array_equal(chunks[0].toarray(), [[n_entries]])


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

    indptr = crh5_reader.grp["indptr"]
    assert crh5_reader.producer_staging_bytes(300, 1) > (
        indptr.size * indptr.dtype.itemsize
    )
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


def test_crh5reader_preserves_filtered_values_dtype_and_batching(tmp_path):
    from scarf.readers import CrH5Reader

    values = np.array(
        [
            [1, 0, 2],
            [0, 0, 0],
            [3, 4, 0],
            [0, 5, 6],
        ],
        dtype=np.uint16,
    )
    path = tmp_path / "modern.h5"
    _write_cr_h5(path, values)
    reader = CrH5Reader(str(path), is_filtered=False, filtering_cutoff=0)
    try:
        chunks = list(reader.consume(batch_size=2))
        observed = np.concatenate([chunk.toarray() for chunk in chunks])
        np.testing.assert_array_equal(reader.validBarcodeIdx, [0, 2, 3])
        np.testing.assert_array_equal(observed, values[[0, 2, 3]])
        assert [chunk.shape[0] for chunk in chunks] == [2, 1]
        assert all(chunk.dtype == values.dtype for chunk in chunks)
    finally:
        reader.close()
    assert not reader.h5obj.id.valid


def test_crh5reader_preserves_legacy_layout_values(tmp_path):
    from scarf.readers import CrH5Reader

    values = np.array([[1, 0], [0, 2], [3, 4]], dtype=np.uint32)
    path = tmp_path / "legacy.h5"
    _write_cr_h5(path, values, legacy=True)
    reader = CrH5Reader(str(path))
    try:
        observed = np.concatenate(
            [chunk.toarray() for chunk in reader.consume(batch_size=2)]
        )
        np.testing.assert_array_equal(observed, values)
        assert reader.feature_ids() == ["feature_0", "feature_1"]
        assert reader.feature_names() == ["gene_0", "gene_1"]
        assert reader.feature_types() == ["Gene Expression", "Gene Expression"]
    finally:
        reader.close()


def test_crdir_reader(crdir_reader):
    assert crdir_reader.nCells == 892
    assert crdir_reader.nFeatures == 36601  # Does not contain 10 ADTs


def test_h5ad_reader(h5ad_reader):
    assert h5ad_reader.nCells == 3696 == len(h5ad_reader.cell_ids())
    assert h5ad_reader.nFeatures == 27998 == len(h5ad_reader.feat_names())


def test_inspect_h5ad_resolves_fixture_and_builds_reader(bastidas_ponce_data):
    from scarf.readers import H5adReader, inspect_h5ad

    inspection = inspect_h5ad(bastidas_ponce_data)

    assert inspection.matrixKey == "X"
    assert inspection.matrixEncoding == "csr"
    assert inspection.matrixCandidates == ("X", "layers/spliced", "layers/unspliced")
    assert inspection.featureAttrsKey == "var"
    assert inspection.cellIdsKey == "index"
    assert inspection.featureIdsKey == "index"
    assert inspection.layers == ("spliced", "unspliced")
    assert inspection.nCells == 3696
    assert inspection.nFeatures == 27998

    reader = H5adReader.from_inspect(inspection)
    try:
        assert reader.matrixKey == inspection.matrixKey
        assert reader.cellIdsKey == inspection.cellIdsKey
        assert reader.featIdsKey == inspection.featureIdsKey
        assert reader.nCells == inspection.nCells
        assert reader.nFeatures == inspection.nFeatures
    finally:
        reader.h5.close()


def test_inspect_h5ad_prefers_dimension_matched_raw_counts(tmp_path):
    import h5py

    from scarf.readers import H5adReader, inspect_h5ad

    file_name = tmp_path / "discovery.h5ad"
    with h5py.File(file_name, mode="w") as h5:
        _write_sparse_group(
            h5,
            "X",
            np.array([[0.1, 0.0], [0.0, 1.5]], dtype=np.float32),
        )
        _write_sparse_group(
            h5,
            "raw/X",
            np.array([[1, 0, 3], [0, 2, 0]], dtype=np.uint16),
        )
        layers = h5.create_group("layers")
        layers.create_dataset(
            "scaled",
            data=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        )

        obs = h5.create_group("obs")
        obs.create_dataset("barcode", data=np.array([b"cell-a", b"cell-b"]))
        obs.create_dataset("batch", data=np.array([0, 1], dtype=np.int8))
        categories = obs.create_group("categories")
        categories.create_dataset("batch", data=np.array([b"A", b"B"]))

        var = h5.create_group("var")
        var.create_dataset("gene_ids", data=np.array([b"v1", b"v2"]))

        raw_var = h5.create_group("raw/var")
        raw_var.create_dataset(
            "gene_ids",
            data=np.array([b"ENSG00000000001", b"ENSG00000000002", b"AB-1"]),
        )
        raw_var.create_dataset(
            "gene_symbol",
            data=np.array([b"GENE1", b"GENE2", b"CD3"]),
        )
        raw_var.create_dataset(
            "feature_types",
            data=np.array(
                [b"Gene Expression", b"Gene Expression", b"Antibody Capture"]
            ),
        )
        uns = h5.create_group("uns")
        uns.create_dataset("title", data=np.bytes_("Discovery fixture"))
        uns.create_dataset("citation", data=np.bytes_("Synthetic citation"))

    inspection = inspect_h5ad(str(file_name))

    assert inspection.matrixKey == "raw/X"
    assert inspection.matrixCandidates == ("raw/X", "layers/scaled", "X")
    assert inspection.featureAttrsKey == "raw/var"
    assert inspection.cellIdsKey == "barcode"
    assert inspection.featureIdsKey == "gene_ids"
    assert inspection.featureNameKey == "gene_symbol"
    assert inspection.categoryNamesKey == "categories"
    assert inspection.assaySplitKey == "feature_types"
    assert inspection.suggestedAssays == {"RNA": 2, "ADT": 1}
    assert inspection.layers == ("scaled",)
    assert inspection.title == "Discovery fixture"
    assert inspection.description == "Synthetic citation"
    assert inspection.to_reader_kwargs()["feature_ids_key"] == "gene_ids"

    reader = H5adReader.from_inspect(inspection)
    try:
        np.testing.assert_array_equal(
            reader.feat_ids(),
            [b"ENSG00000000001", b"ENSG00000000002", b"AB-1"],
        )
        np.testing.assert_array_equal(
            reader.feat_names(),
            [b"GENE1", b"GENE2", b"CD3"],
        )
        np.testing.assert_array_equal(
            np.vstack([chunk.toarray() for chunk in reader.consume(1)]),
            [[1, 0, 3], [0, 2, 0]],
        )
        np.testing.assert_array_equal(
            dict(reader.get_cell_columns())["batch"],
            [b"A", b"B"],
        )
    finally:
        reader.h5.close()


def test_inspect_h5ad_ignores_ensembl_biotype_feature_type(tmp_path):
    """CELLxGENE feature_type is gene biotype, not a modality split key."""
    import h5py
    from scipy import sparse

    from scarf.readers import inspect_h5ad

    file_name = tmp_path / "biotype.h5ad"
    matrix = sparse.csr_matrix(np.array([[1, 0], [0, 2]], dtype=np.int32))
    with h5py.File(file_name, "w") as h5:
        x = h5.create_group("X")
        x.create_dataset("data", data=matrix.data)
        x.create_dataset("indices", data=matrix.indices)
        x.create_dataset("indptr", data=matrix.indptr)
        x.attrs["encoding-type"] = "csr_matrix"
        x.attrs["encoding-version"] = "0.1.0"
        x.attrs["shape"] = matrix.shape
        obs = h5.create_group("obs")
        obs.create_dataset("_index", data=np.array([b"c1", b"c2"]))
        var = h5.create_group("var")
        var.create_dataset("_index", data=np.array([b"ENSG1", b"ENSG2"]))
        var.create_dataset(
            "feature_type",
            data=np.array([b"protein_coding", b"lncRNA"]),
        )

    inspection = inspect_h5ad(str(file_name))
    assert inspection.assaySplitKey is None
    assert inspection.suggestedAssays == {}


def test_inspect_h5ad_falls_back_from_mismatched_raw_var(tmp_path):
    import h5py

    from scarf.readers import inspect_h5ad

    file_name = tmp_path / "mismatched_raw_var.h5ad"
    with h5py.File(file_name, mode="w") as h5:
        _write_sparse_group(
            h5,
            "X",
            np.array([[1, 0], [0, 2]], dtype=np.uint16),
        )
        obs = h5.create_group("obs")
        obs.create_dataset("barcode", data=np.array([b"c1", b"c2"]))
        var = h5.create_group("var")
        var.create_dataset(
            "opaque_long",
            data=np.array([b"ENSG00000000001", b"ENSG00000000002"]),
        )
        var.create_dataset("opaque_short", data=np.array([b"G1", b"G2"]))
        raw_var = h5.create_group("raw/var")
        raw_var.create_dataset(
            "gene_ids",
            data=np.array([b"raw1", b"raw2", b"raw3"]),
        )

    inspection = inspect_h5ad(str(file_name))

    assert inspection.featureAttrsKey == "var"
    assert inspection.featureIdsKey == "opaque_long"
    assert inspection.featureNameKey == "opaque_short"


def test_inspect_h5ad_uses_index_attr_and_categorical_codes_for_lengths(tmp_path):
    import h5py

    from scarf.readers import inspect_h5ad

    file_name = tmp_path / "index_attr_lengths.h5ad"
    with h5py.File(file_name, mode="w") as h5:
        _write_sparse_group(
            h5,
            "X",
            np.array([[1, 0, 2], [0, 3, 0]], dtype=np.uint16),
        )
        obs = h5.create_group("obs")
        obs.attrs["_index"] = "barcode"
        barcode = obs.create_group("barcode")
        barcode.create_dataset("codes", data=np.array([0, 1], dtype=np.int8))
        barcode.create_dataset("categories", data=np.array([b"c0", b"c1"]))
        # A non-index column that would otherwise be the first length probe.
        obs.create_dataset("batch", data=np.array([0, 1], dtype=np.int8))

        var = h5.create_group("var")
        var.attrs["_index"] = "gene_ids"
        var.create_dataset(
            "gene_ids",
            data=np.array([b"g0", b"g1", b"g2"]),
        )
        var.create_dataset(
            "gene_symbol",
            data=np.array([b"A", b"B", b"C"]),
        )

    inspection = inspect_h5ad(str(file_name))
    assert inspection.nCells == 2
    assert inspection.nFeatures == 3
    assert inspection.cellIdsKey == "barcode"
    assert inspection.featureIdsKey == "gene_ids"
    assert inspection.featureNameKey == "gene_symbol"


def test_inspect_h5ad_infers_sparse_shape_without_stored_attrs(tmp_path):
    import h5py
    from scipy.sparse import csr_matrix

    from scarf.readers import inspect_h5ad

    values = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=np.uint16)
    matrix = csr_matrix(values)
    file_name = tmp_path / "inferred_shape.h5ad"
    with h5py.File(file_name, mode="w") as h5:
        sparse = h5.create_group("X")
        # No encoding-type / shape attrs: default CSR and infer from indptr/indices.
        sparse.create_dataset("data", data=matrix.data)
        sparse.create_dataset("indices", data=matrix.indices.astype(np.int64))
        sparse.create_dataset("indptr", data=matrix.indptr.astype(np.int64))
        obs = h5.create_group("obs")
        obs.create_dataset(
            "cell_id",
            data=np.array([b"a", b"b", b"c"]),
        )
        var = h5.create_group("var")
        var.create_dataset(
            "feature_id",
            data=np.array([b"f0", b"f1", b"f2"]),
        )

    inspection = inspect_h5ad(str(file_name))
    assert inspection.matrixEncoding == "csr"
    assert inspection.nCells == 3
    assert inspection.nFeatures == 3
    assert inspection.cellIdsKey == "cell_id"
    assert inspection.featureIdsKey == "feature_id"


def test_inspect_h5ad_falls_back_to_generated_ids_when_columns_are_not_unique(
    tmp_path,
):
    import h5py

    from scarf.readers import inspect_h5ad

    file_name = tmp_path / "non_unique_ids.h5ad"
    with h5py.File(file_name, mode="w") as h5:
        _write_sparse_group(
            h5,
            "X",
            np.array([[1, 0], [0, 2]], dtype=np.uint16),
        )
        obs = h5.create_group("obs")
        obs.create_dataset("batch", data=np.array([b"A", b"A"]))
        var = h5.create_group("var")
        var.create_dataset("score", data=np.array([1.0, 2.0]))

    inspection = inspect_h5ad(str(file_name))
    assert inspection.cellIdsKey == "_index"
    assert inspection.featureIdsKey == "_index"
    assert inspection.featureNameKey == "_index"


def test_inspect_h5ad_ignores_matrix_with_mismatched_obs_length(tmp_path):
    import h5py

    from scarf.readers import inspect_h5ad

    file_name = tmp_path / "obs_mismatch.h5ad"
    with h5py.File(file_name, mode="w") as h5:
        _write_sparse_group(
            h5,
            "X",
            np.array([[1, 0], [0, 2], [3, 0]], dtype=np.uint16),
        )
        layers = h5.create_group("layers")
        _write_sparse_group(
            layers,
            "counts",
            np.array([[1, 0], [0, 2]], dtype=np.uint16),
        )
        obs = h5.create_group("obs")
        obs.create_dataset("barcode", data=np.array([b"c0", b"c1"]))
        var = h5.create_group("var")
        var.create_dataset("gene_ids", data=np.array([b"g0", b"g1"]))

    inspection = inspect_h5ad(str(file_name))
    assert inspection.matrixKey == "layers/counts"
    assert inspection.nCells == 2
    assert inspection.nFeatures == 2


def test_inspect_h5ad_uses_generated_feature_ids_when_var_group_is_absent(tmp_path):
    import h5py

    from scarf.readers import inspect_h5ad

    file_name = tmp_path / "missing_var.h5ad"
    with h5py.File(file_name, mode="w") as h5:
        _write_sparse_group(
            h5,
            "X",
            np.array([[1, 0], [0, 2]], dtype=np.uint16),
        )
        obs = h5.create_group("obs")
        obs.create_dataset("barcode", data=np.array([b"c0", b"c1"]))

    inspection = inspect_h5ad(str(file_name))
    assert inspection.featureAttrsKey == "var"
    assert inspection.featureIdsKey == "_index"
    assert inspection.featureNameKey == "_index"
    assert inspection.nCells == 2
    assert inspection.nFeatures == 2


def test_inspect_h5ad_rejects_files_without_numeric_matrices(tmp_path):
    import h5py

    from scarf.readers import inspect_h5ad

    file_name = tmp_path / "no_matrix.h5ad"
    with h5py.File(file_name, mode="w") as h5:
        obs = h5.create_group("obs")
        obs.create_dataset("barcode", data=np.array([b"c0"]))
        var = h5.create_group("var")
        var.create_dataset("gene_ids", data=np.array([b"g0"]))

    with pytest.raises(ValueError, match="No sparse or numeric 2D matrix"):
        inspect_h5ad(str(file_name))


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
        expected_max_nnz = max(
            np.count_nonzero(values[start : start + batch_size])
            for start in range(
                0,
                max(1, values.shape[0] - batch_size + 1),
            )
        )
        assert reader.max_batch_nnz(batch_size) == expected_max_nnz
        chunks = list(reader.consume(batch_size=batch_size))
        assert sum(chunk.shape[0] for chunk in chunks) == values.shape[0]
        assert sum(chunk.nnz for chunk in chunks) == np.count_nonzero(values)
        np.testing.assert_array_equal(
            np.vstack([chunk.toarray() for chunk in chunks]),
            values,
        )
    finally:
        reader.h5.close()


def test_h5ad_reader_converts_csc_sparse_encoding(tmp_path):
    import h5py
    import zarr

    from scarf.readers import H5adReader
    from scarf.readers import inspect_h5ad
    from scarf.writers import H5adToZarr

    file_name = tmp_path / "csc.h5ad"
    zarr_path = tmp_path / "csc.zarr"
    values = np.array(
        [
            [1, 0, 2],
            [0, 3, 0],
            [4, 0, 5],
            [0, 6, 0],
        ],
        dtype=np.uint16,
    )
    _write_sparse_h5ad(
        file_name,
        values,
        encoding_type="csc_matrix",
    )
    with h5py.File(file_name, mode="r+") as h5:
        shape = h5["X"].attrs["shape"]
        del h5["X"].attrs["encoding-type"]
        del h5["X"].attrs["shape"]
        h5["X"].attrs["h5sparse_format"] = "csc"
        h5["X"].attrs["h5sparse_shape"] = shape

    inspection = inspect_h5ad(str(file_name))
    assert inspection.matrixEncoding == "csc"

    reader = H5adReader.from_inspect(inspection)
    try:
        chunks = list(reader.consume(batch_size=2))
        np.testing.assert_array_equal(
            np.vstack([chunk.toarray() for chunk in chunks]),
            values,
        )
        writer = H5adToZarr(reader, zarr_loc=str(zarr_path))
        writer.dump(batch_size=2)
    finally:
        reader.h5.close()

    root = zarr.open_group(str(zarr_path), mode="r")
    np.testing.assert_array_equal(root["RNA/counts"][:], values)
    assert "countsT" not in root["RNA"]


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
    assert "countsT" not in root["RNA"]


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
    # Legacy codes here are {-1, 0, 1} against categories [False, True]; the
    # -1 sentinel decodes to missing rather than wrapping to the last category.
    np.testing.assert_array_equal(
        feature_columns["highly_variable_genes"][:3],
        np.array([b"False", None, None], dtype=object),
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
    from loguru import logger

    from scarf.readers import H5adReader, inspect_h5ad

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
        state = obs.create_group("state")
        state.create_dataset("codes", data=np.array([0, 1, 0], dtype=np.int8))
        state.create_dataset("categories", data=np.array([b"cycling", b"resting"]))
        n_genes = obs.create_group("nGenes")
        n_genes.attrs["encoding-type"] = "nullable-integer"
        n_genes.create_dataset("values", data=np.array([5, 7, 0], dtype=np.int64))
        n_genes.create_dataset("mask", data=np.array([False, False, True]))

        var = h5.create_group("var")
        var.create_dataset("_index", data=np.array([b"f1", b"f2"]))
        feature_names = var.create_group("gene_short_name")
        feature_names.create_dataset("codes", data=np.array([1, 0]))
        feature_names.create_dataset(
            "categories",
            data=np.array([b"Gene A", b"Gene B"]),
        )
        var.create_dataset("chromosome", data=np.array([b"1", b"2"]))
        feature_type = var.create_group("feature_type")
        feature_type.create_dataset("codes", data=np.array([0, 1], dtype=np.int8))
        feature_type.create_dataset(
            "categories",
            data=np.array([b"Gene Expression", b"Antibody Capture"]),
        )
        highly_variable = var.create_group("highly_variable")
        highly_variable.attrs["encoding-type"] = "nullable-boolean"
        highly_variable.create_dataset("values", data=np.array([True, False]))
        highly_variable.create_dataset("mask", data=np.array([False, False]))
        reviewed = var.create_group("reviewed")
        reviewed.attrs["encoding-type"] = "nullable-boolean"
        reviewed.create_dataset("values", data=np.array([True, False]))
        reviewed.create_dataset("mask", data=np.array([False, True]))
        unreadable = var.create_group("per_cell_counts")
        unreadable.attrs["encoding-type"] = "dataframe"
        unreadable.create_dataset("column", data=np.array([1, 2]))

        obsm = h5.create_group("obsm")
        obsm.create_dataset(
            "X_embed",
            data=np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
        )
        obsm.create_dataset(
            "bad_embed",
            data=np.array([[1, 2], [3, 4]], dtype=np.float32),
        )
        _write_sparse_group(
            obsm,
            "sparse_embed",
            np.array([[1, 0], [0, 2], [3, 0]], dtype=np.float32),
        )

    inspection = inspect_h5ad(str(file_name))
    assert inspection.matrixKey == "X"
    assert inspection.matrixEncoding == "dense"

    reader = H5adReader(str(file_name))
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]), level="WARNING"
    )
    try:
        assert reader.groupCodes == {"obs": 2, "var": 2, "obsm": 2, "X": 1}
        np.testing.assert_array_equal(reader.cell_ids(), [b"c1", b"c2", b"c3"])
        np.testing.assert_array_equal(reader.feat_ids(), [b"f1", b"f2"])
        np.testing.assert_array_equal(reader.feat_names(), [b"Gene B", b"Gene A"])

        cell_columns = dict(reader.get_cell_columns())
        assert cell_columns.keys() == {
            "batch",
            "state",
            "nGenes",
            "X_embed1",
            "X_embed2",
        }
        np.testing.assert_array_equal(cell_columns["batch"], [b"A", b"B", b"A"])
        np.testing.assert_array_equal(
            cell_columns["state"],
            [b"cycling", b"resting", b"cycling"],
        )
        # Numeric nullable columns stay numeric by representing missing values
        # as NaN.
        assert cell_columns["nGenes"].dtype == np.dtype(np.float64)
        np.testing.assert_allclose(
            cell_columns["nGenes"],
            np.array([5, 7, np.nan]),
            equal_nan=True,
        )
        np.testing.assert_array_equal(cell_columns["X_embed2"], [2, 4, 6])

        feature_columns = dict(reader.get_feat_columns())
        assert feature_columns.keys() == {
            "chromosome",
            "feature_type",
            "highly_variable",
            "reviewed",
        }
        np.testing.assert_array_equal(feature_columns["chromosome"], [b"1", b"2"])
        np.testing.assert_array_equal(
            feature_columns["feature_type"],
            [b"Gene Expression", b"Antibody Capture"],
        )
        # Nothing is masked here, so the source dtype is preserved.
        assert feature_columns["highly_variable"].dtype == np.dtype(bool)
        np.testing.assert_array_equal(feature_columns["highly_variable"], [True, False])
        np.testing.assert_array_equal(
            feature_columns["reviewed"],
            np.array([True, None], dtype=object),
        )
        assert any(
            "per_cell_counts" in message and "dataframe" in message
            for message in messages
        )
        assert any("sparse_embed" in message for message in messages)
        assert not any("__categories" in message for message in messages)
        assert reader.feature_types("feature_type") == [
            "Gene Expression",
            "Antibody Capture",
        ]

        chunks = [chunk.toarray() for chunk in reader.consume(batch_size=2)]
        assert len(chunks) == 2
        np.testing.assert_array_equal(chunks[0], [[1, 0], [0, 2]])
        np.testing.assert_array_equal(chunks[1], [[3, 4]])
    finally:
        logger.remove(sink)
        reader.h5.close()


def test_h5ad_reader_sizes_axes_from_nullable_columns(tmp_path):
    import h5py

    from scarf.readers import H5adReader

    file_name = tmp_path / "nullable_only.h5ad"
    with h5py.File(file_name, mode="w") as h5:
        h5.create_dataset("X", data=np.ones((3, 2), dtype=np.float32))
        obs = h5.create_group("obs")
        n_genes = obs.create_group("nGenes")
        n_genes.attrs["encoding-type"] = "nullable-integer"
        n_genes.create_dataset("values", data=np.array([4, 5, 6], dtype=np.int64))
        n_genes.create_dataset("mask", data=np.array([False, False, False]))
        var = h5.create_group("var")
        # A mask that does not line up with the values cannot be applied.
        weight = var.create_group("weight")
        weight.attrs["encoding-type"] = "nullable-integer"
        weight.create_dataset("values", data=np.array([1, 2], dtype=np.int64))
        weight.create_dataset("mask", data=np.array([False]))

    reader = H5adReader(str(file_name))
    try:
        assert reader.nCells == 3
        assert reader.nFeatures == 2
        np.testing.assert_array_equal(reader.cell_ids(), ["cell_0", "cell_1", "cell_2"])
        cell_columns = dict(reader.get_cell_columns())
        assert cell_columns["nGenes"].dtype == np.dtype(np.int64)
        np.testing.assert_array_equal(cell_columns["nGenes"], [4, 5, 6])
        np.testing.assert_array_equal(
            dict(reader.get_feat_columns())["weight"],
            [1, 2],
        )
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


def test_csv_reader_preserves_batches_skipped_columns_and_cell_metadata(tmp_path):
    from scarf.readers import CSVReader

    path = tmp_path / "counts.csv"
    path.write_text("g1,g2,batch,drop\n1,2,a,10\n3,4,b,20\n5,6,c,30\n")
    reader = CSVReader(
        str(path),
        skip_cols=["drop"],
        cell_data_cols=["batch"],
        batch_size=2,
    )

    assert reader.nCells == 3
    assert reader.nFeatures == 2
    np.testing.assert_array_equal(reader.cell_ids(), ["cell_0", "cell_1", "cell_2"])
    np.testing.assert_array_equal(reader.feature_ids(), ["g1", "g2"])
    batches = list(reader.consume())
    np.testing.assert_array_equal(batches[0][0], [[1, 2], [3, 4]])
    np.testing.assert_array_equal(batches[0][1], [["a"], ["b"]])
    np.testing.assert_array_equal(batches[1][0], [[5, 6]])
    np.testing.assert_array_equal(batches[1][1], [["c"]])


def test_csv_reader_rejects_features_along_rows(tmp_path):
    from scarf.readers import CSVReader

    path = tmp_path / "counts.csv"
    path.write_text("g1,g2\n1,2\n")
    with pytest.raises(NotImplementedError, match="cells are along the rows"):
        CSVReader(str(path), rows_are_cells=False)


def test_csv_reader_rejects_non_mapping_pandas_kwargs(tmp_path):
    from scarf.readers import CSVReader

    path = tmp_path / "counts.csv"
    path.write_text("g1,g2\n1,2\n", encoding="utf-8")
    with pytest.raises(TypeError, match="pandas_kwargs must be a dictionary"):
        CSVReader(str(path), pandas_kwargs=["header"])
