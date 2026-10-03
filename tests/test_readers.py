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


def _assert_same_sparse(observed, expected) -> None:
    """Compare two sparse matrices entry by entry in canonical CSR form."""
    observed, expected = observed.tocsr(), expected.tocsr()
    assert observed.shape == expected.shape
    assert observed.dtype == expected.dtype
    for matrix in (observed, expected):
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
    np.testing.assert_array_equal(observed.indptr, expected.indptr)
    np.testing.assert_array_equal(observed.indices, expected.indices)
    np.testing.assert_array_equal(observed.data, expected.data)


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
                "3 4 6",
                "1 1 200",
                "2 1 101",
                "1 2 250",
                "3 2 60",
                "3 3 1",
                "2 4 300",
            ]
        )
        + "\n"
    )

    reader = CrDirReader(
        str(tmp_path),
        is_filtered=False,
        filtering_cutoff=300,
    )

    np.testing.assert_array_equal(reader.validBarcodeIdx, np.array([0, 1]))
    assert reader.nCells == 2
    assert reader.nFeatures == 3
    assert reader.matrixEntryCount == 6
    assert reader.cell_names() == ["b1", "b2"]
    # The background barcode's 300 is not kept, so its dtype is not needed.
    assert reader.count_value_ranges(0)[0].maximum == 250
    assert reader.matrix_dtype == np.uint8

    chunks = list(reader.consume(batch_size=1, lines_in_mem=2))
    assert [chunk.shape for chunk in chunks] == [(1, 3), (1, 3)]
    assert all(chunk.dtype == np.uint8 for chunk in chunks)
    np.testing.assert_array_equal(chunks[0].toarray(), [[200, 101, 0]])
    np.testing.assert_array_equal(chunks[1].toarray(), [[250, 0, 60]])


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
    chunks = list(reader.consume(batch_size=1, lines_in_mem=100))

    # The range holds the summed count, so it is stored in uint16, not uint8.
    assert reader.count_value_ranges(0)[0].maximum == n_entries
    assert len(chunks) == 1
    assert chunks[0].nnz == 1
    assert chunks[0].dtype == np.uint16
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
    assert list(toy_crdir_empty.consume(batch_size=10)) == []


def test_crh5reader(crh5_reader):
    from tests.test_writers import _read_cellranger_h5

    _counts, barcodes, features = _read_cellranger_h5(crh5_reader.h5obj.filename)
    assert crh5_reader.nCells == 892
    assert crh5_reader.nFeatures == 36611
    n_assay_feats = list(crh5_reader.assayFeats.T.nFeatures.values)
    assert n_assay_feats == [36601, 10]
    assert crh5_reader.cell_names() == barcodes.tolist()
    assert crh5_reader.feature_ids() == features["id"].tolist()
    assert crh5_reader.feature_names() == features["name"].tolist()
    assert crh5_reader.feature_types() == features["feature_type"].tolist()


def test_crh5reader_streams_counts(crh5_reader):
    from scipy.sparse import vstack

    from tests.test_writers import _read_cellranger_h5

    indptr = crh5_reader.grp["indptr"]
    assert crh5_reader.producer_staging_bytes(300, 1) > (
        indptr.size * indptr.dtype.itemsize
    )
    chunks = list(crh5_reader.consume(batch_size=300))
    assert [chunk.shape[0] for chunk in chunks] == [300, 300, 292]
    expected, _barcodes, _features = _read_cellranger_h5(crh5_reader.h5obj.filename)
    _assert_same_sparse(vstack(chunks, format="csr"), expected)


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


def test_crh5reader_opens_matrix_datasets_once_per_stream(tmp_path, monkeypatch):
    import h5py

    from scarf.readers import CrH5Reader

    # Background barcodes with counts below the cutoff split the cells into
    # runs of one barcode each.
    values = np.array([[5, 0, 5], [1, 0, 0]] * 4, dtype=np.uint16)
    path = tmp_path / "runs.h5"
    _write_cr_h5(path, values)
    reader = CrH5Reader(str(path), is_filtered=False, filtering_cutoff=2)
    opened: list[str] = []
    get_item = h5py.Group.__getitem__

    def counting_get_item(group, name):
        node = get_item(group, name)
        if isinstance(node, h5py.Dataset):
            opened.append(node.name)
        return node

    def stream_opens(batch_size):
        opened.clear()
        observed = np.concatenate(
            [chunk.toarray() for chunk in reader.consume(batch_size=batch_size)]
        )
        np.testing.assert_array_equal(observed, values[::2])
        return opened.count("/matrix/data"), opened.count("/matrix/indices")

    monkeypatch.setattr(h5py.Group, "__getitem__", counting_get_item)
    try:
        # Each open dataset has its own chunk cache, so the datasets are not
        # reopened for each batch or run.
        assert stream_opens(1) == stream_opens(4)
    finally:
        reader.close()


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


def test_crdir_reader(crdir_reader, mtx_dir):
    import gzip
    from pathlib import Path

    assert crdir_reader.nCells == 892
    assert crdir_reader.nFeatures == 36601  # Does not contain 10 ADTs
    with gzip.open(Path(mtx_dir) / "barcodes.tsv.gz", "rt") as handle:
        assert crdir_reader.cell_names() == handle.read().split()
    with gzip.open(Path(mtx_dir) / "features.tsv.gz", "rt") as handle:
        features = [line.rstrip("\n").split("\t") for line in handle]
    assert crdir_reader.feature_ids() == [row[0] for row in features]
    assert crdir_reader.feature_names() == [row[1] for row in features]


def test_h5ad_reader(h5ad_reader):
    import h5py

    assert h5ad_reader.nCells == 3696
    assert h5ad_reader.nFeatures == 27998
    with h5py.File(h5ad_reader.h5adFn, mode="r") as h5:
        cells = [value.decode() for value in h5["obs"]["index"]]
        features = [value.decode() for value in h5["var"]["index"]]
    assert _texts(h5ad_reader.cell_ids()) == cells
    assert _texts(h5ad_reader.feat_ids()) == features
    assert _texts(h5ad_reader.feat_names()) == features


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


def test_h5ad_sparse_groups_need_an_encoding_and_a_shape(tmp_path):
    import h5py
    from scipy.sparse import csr_matrix

    from scarf.readers import H5adReader, inspect_h5ad

    matrix = csr_matrix(np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=np.uint16))
    for missing_attribute in ("encoding-type", "shape"):
        file_name = tmp_path / f"without_{missing_attribute}.h5ad"
        with h5py.File(file_name, mode="w") as h5:
            sparse = h5.create_group("X")
            sparse.create_dataset("data", data=matrix.data)
            sparse.create_dataset("indices", data=matrix.indices.astype(np.int64))
            sparse.create_dataset("indptr", data=matrix.indptr.astype(np.int64))
            attributes = {"encoding-type": "csr_matrix", "shape": matrix.shape}
            del attributes[missing_attribute]
            sparse.attrs.update(attributes)
            h5.create_group("obs").create_dataset(
                "cell_id", data=np.array([b"a", b"b", b"c"])
            )

        with pytest.raises(ValueError, match="No sparse or numeric 2D matrix"):
            inspect_h5ad(str(file_name))
    with pytest.raises(ValueError, match="encoding `None` of `X` is not supported"):
        H5adReader(str(tmp_path / "without_encoding-type.h5ad"))


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
    import h5py
    from scipy.sparse import csr_matrix, vstack

    chunks = list(h5ad_reader.consume(batch_size=1000))
    assert [chunk.shape[0] for chunk in chunks] == [1000, 1000, 1000, 696]
    assert all(chunk.dtype == h5ad_reader.sourceMatrixDtype for chunk in chunks)
    with h5py.File(h5ad_reader.h5adFn, mode="r") as h5:
        expected = csr_matrix(
            (h5["X/data"][:], h5["X/indices"][:], h5["X/indptr"][:]),
            shape=(h5ad_reader.nCells, h5ad_reader.nFeatures),
        )
    _assert_same_sparse(vstack(chunks, format="csr"), expected)


@pytest.mark.parametrize("batch_size", [1, 2, 4, 5, 8, 9])
@pytest.mark.parametrize(
    "values",
    [
        np.array(
            [
                [1, 0, 2],
                [0, 0, 0],
                [0, 3, 0],
                [0, 0, 0],
            ],
            dtype=np.uint32,
        ),
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
        np.array(
            [[0, 0, 0]] * 5 + [[6, 0, 0], [0, 7, 0], [0, 0, 8]],
            dtype=np.uint32,
        ),
        np.zeros((8, 3), dtype=np.uint32),
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
        assert all(0 < chunk.shape[0] <= batch_size for chunk in chunks)
        assert sum(chunk.shape[0] for chunk in chunks) == values.shape[0]
        assert sum(chunk.nnz for chunk in chunks) == np.count_nonzero(values)
        np.testing.assert_array_equal(
            np.vstack([chunk.toarray() for chunk in chunks]),
            values,
        )
    finally:
        reader.h5.close()


def test_h5ad_reader_opens_matrix_datasets_once_per_stream(tmp_path, monkeypatch):
    import h5py

    from scarf.readers import H5adReader

    values = np.arange(24, dtype=np.uint32).reshape(8, 3) % 5
    file_name = tmp_path / "sparse.h5ad"
    _write_sparse_h5ad(file_name, values)
    reader = H5adReader(str(file_name), feature_name_key="feature_name")
    opened: list[str] = []
    get_item = h5py.Group.__getitem__

    def counting_get_item(group, name):
        node = get_item(group, name)
        if isinstance(node, h5py.Dataset):
            opened.append(node.name)
        return node

    def stream_opens(batch_size):
        opened.clear()
        chunks = list(reader.consume(batch_size=batch_size))
        np.testing.assert_array_equal(
            np.vstack([chunk.toarray() for chunk in chunks]), values
        )
        return opened.count("/X/data"), opened.count("/X/indices")

    monkeypatch.setattr(h5py.Group, "__getitem__", counting_get_item)
    try:
        # Each open dataset has its own chunk cache, so the datasets are not
        # reopened for each batch.
        assert stream_opens(1) == stream_opens(len(values))
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
    assert "countsT" in root["RNA"]
    assert root["RNA/countsT"].attrs["complete"] is True


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("leading_empty_cells", [0, 4])
def test_h5ad_to_zarr_preserves_exact_sparse_batch(
    tmp_path, batch_size, leading_empty_cells
):
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
    values = np.pad(values, ((leading_empty_cells, 0), (0, 0)))
    file_name = tmp_path / "exact_batch.h5ad"
    zarr_path = tmp_path / "exact_batch.zarr"
    _write_sparse_h5ad(file_name, values)
    reader = H5adReader(str(file_name), feature_name_key="feature_name")
    try:
        writer = H5adToZarr(reader, zarr_loc=str(zarr_path))
        writer.dump(batch_size=batch_size)
    finally:
        reader.h5.close()

    root = zarr.open_group(str(zarr_path), mode="r")
    np.testing.assert_array_equal(root["RNA/counts"][:], values)
    assert "countsT" in root["RNA"]
    assert root["RNA/countsT"].attrs["complete"] is True
    np.testing.assert_array_equal(root["RNA/countsT"][:], values.T)


def test_h5ad_reader_streams_cell_and_feature_metadata(h5ad_reader):
    import h5py

    def decoded(codes, categories):
        # AnnData 0.6 keeps the categories of a coded column in uns.
        return np.array(
            [categories[code] if code >= 0 else None for code in codes], dtype=object
        )

    with h5py.File(h5ad_reader.h5adFn, mode="r") as h5:
        obs = h5["obs"][:]
        var = h5["var"][:]
        uns = {
            key: h5["uns"][key][:] for key in h5["uns"] if key.endswith("_categories")
        }

    cell_columns = dict(h5ad_reader.get_cell_columns())
    assert cell_columns.keys() == {
        "clusters_coarse",
        "clusters",
        "S_score",
        "G2M_score",
    }
    for column in ("clusters_coarse", "clusters"):
        np.testing.assert_array_equal(
            cell_columns[column],
            decoded(obs[column], uns[f"{column}_categories"]),
        )
    np.testing.assert_array_equal(
        cell_columns["clusters_coarse"][:3],
        np.array([b"Pre-endocrine", b"Ductal", b"Endocrine"]),
    )
    for column in ("S_score", "G2M_score"):
        np.testing.assert_array_equal(cell_columns[column], obs[column])
    feature_columns = dict(h5ad_reader.get_feat_columns())
    assert feature_columns.keys() == {"highly_variable_genes"}
    # Legacy codes here are {-1, 0, 1} against categories [False, True]; the
    # -1 sentinel decodes to missing rather than wrapping to the last category.
    assert set(var["highly_variable_genes"]) == {-1, 0, 1}
    np.testing.assert_array_equal(
        feature_columns["highly_variable_genes"],
        decoded(var["highly_variable_genes"], uns["highly_variable_genes_categories"]),
    )
    np.testing.assert_array_equal(
        feature_columns["highly_variable_genes"][:3],
        np.array([b"False", None, None], dtype=object),
    )
    assert h5ad_reader._check_exists("layers", "spliced")


def test_h5ad_category_codes_outside_the_categories_are_missing():
    from scarf.readers._h5ad_columns import decode_categories

    values, missing = decode_categories(
        np.array([10_000, 0, -1]),
        np.array([b"a"]),
    )

    np.testing.assert_array_equal(values, np.array([None, b"a", None], dtype=object))
    np.testing.assert_array_equal(missing, [True, False, True])


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
        assert cell_columns.keys() == {"batch", "state", "nGenes"}
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
        assert not any(key.startswith("X_embed") for key in cell_columns)

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

    selected = H5adReader(
        str(file_name),
        embedding_roles={"X_embed": "umap"},
        cluster_keys=("state",),
    )
    try:
        assert selected.embeddingRoles == {"X_embed": "umap"}
        assert selected.clusterKeys == ("state",)
        assert set(dict(selected.get_cell_columns())) == {"batch", "nGenes"}
        np.testing.assert_array_equal(
            selected._cell_column_block("state", 1, 3)[0],
            [b"resting", b"cycling"],
        )
    finally:
        selected.h5.close()


def test_h5ad_reader_streams_compound_obs_cluster_fields(tmp_path):
    import h5py

    from scarf.readers import H5adReader

    file_name = tmp_path / "compound_obs.h5ad"
    values = np.array(
        [[1, 0], [0, 2], [3, 0], [0, 4]],
        dtype=np.uint16,
    )
    _write_sparse_h5ad(file_name, values)
    obs_dtype = np.dtype(
        [
            ("_index", "S8"),
            ("state", np.int8),
            ("batch", "S1"),
            ("vector", np.float32, (2,)),
        ]
    )
    obs_values = np.zeros(values.shape[0], dtype=obs_dtype)
    obs_values["_index"] = [b"cell_0", b"cell_1", b"cell_2", b"cell_3"]
    obs_values["state"] = [0, -1, 1, 9]
    obs_values["batch"] = [b"A", b"B", b"A", b"B"]
    obs_values["vector"] = np.arange(8, dtype=np.float32).reshape(4, 2)
    with h5py.File(file_name, mode="r+") as h5:
        del h5["obs"]
        h5.create_dataset("obs", data=obs_values)
        uns = h5.create_group("uns")
        uns.create_dataset(
            "state_categories",
            data=np.array([b"cycling", b"resting"]),
        )

    reader = H5adReader(
        str(file_name),
        feature_name_key="feature_name",
        cluster_keys=("state",),
    )
    try:
        assert reader.clusterKeys == ("state",)
        assert reader._cell_column_value_dtype("state") == np.dtype("S7")
        decoded, missing = reader._cell_column_block("state", 1, 4)
        np.testing.assert_array_equal(
            decoded,
            np.array([None, b"resting", None], dtype=object),
        )
        np.testing.assert_array_equal(missing, [True, False, True])
        # Blocks of cell IDs are the text that the import stores.
        np.testing.assert_array_equal(
            reader._cell_ids_block(1, 3), ["cell_1", "cell_2"]
        )
        # A vector field does not fit one metadata value per cell.
        assert set(dict(reader.get_cell_columns())) == {"batch"}
    finally:
        reader.h5.close()

    with pytest.raises(
        TypeError,
        match="Cluster key 'vector' must contain one scalar value per cell",
    ):
        H5adReader(
            str(file_name),
            feature_name_key="feature_name",
            cluster_keys=("vector",),
        )


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
        # A mask that does not line up with its values makes the column unusable.
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
        assert "weight" not in dict(reader.get_feat_columns())
    finally:
        reader.h5.close()


def test_h5ad_reader_falls_back_without_metadata_groups(tmp_path):
    import h5py

    from scarf.readers import H5adReader

    file_name = tmp_path / "sparse_without_metadata.h5ad"
    with h5py.File(file_name, mode="w") as h5:
        matrix = h5.create_group("X")
        matrix.attrs["encoding-type"] = "csr_matrix"
        matrix.attrs["shape"] = (2, 2)
        matrix.create_dataset("data", data=np.array([5, 8], dtype=np.int16))
        matrix.create_dataset("indices", data=np.array([0, 1]))
        matrix.create_dataset("indptr", data=np.array([0, 1, 2]))

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


@pytest.mark.parametrize("include_metadata", [True, False])
def test_csv_reader_preserves_batches_skipped_columns_and_cell_metadata(
    tmp_path, include_metadata
):
    from scarf.readers import CSVReader

    path = tmp_path / "counts.csv"
    path.write_text("g1,g2,batch,drop\n1,2,a,10\n3,4,b,20\n5,6,c,30\n")
    reader = CSVReader(
        str(path),
        skip_cols=["drop"] if include_metadata else ["batch", "drop"],
        cell_data_cols=["batch"] if include_metadata else [],
        batch_size=2,
    )

    assert reader.nCells == 3
    assert reader.nFeatures == 2
    np.testing.assert_array_equal(reader.cell_ids(), ["cell_0", "cell_1", "cell_2"])
    np.testing.assert_array_equal(reader.feature_ids(), ["g1", "g2"])
    batches = list(reader.consume())
    assert all(counts.dtype.kind in "iu" for counts, _ in batches)
    np.testing.assert_array_equal(batches[0][0], [[1, 2], [3, 4]])
    np.testing.assert_array_equal(batches[1][0], [[5, 6]])
    if include_metadata:
        np.testing.assert_array_equal(batches[0][1], [["a"], ["b"]])
        np.testing.assert_array_equal(batches[1][1], [["c"]])
    else:
        assert all(metadata is None for _, metadata in batches)


def test_h5ad_csc_conversion_rejects_insufficient_workspace(tmp_path):
    from scarf.readers import H5adReader
    from tests.test_writers import _write_h5ad

    path = _write_h5ad(
        tmp_path / "counts.h5ad", np.ones((3, 4), dtype=np.uint16), encoding="csc"
    )
    reader = H5adReader(str(path))
    try:
        with pytest.raises(MemoryError, match="CSC row conversion exceeds"):
            reader.materialize_csc(maxBytes=1)
        assert reader._convertedCsr is None
        assert reader.materialized_csr_bytes() == 0
    finally:
        reader.close()


def test_csv_reader_rejects_features_along_rows(tmp_path):
    from scarf.readers import CSVReader

    path = tmp_path / "counts.csv"
    path.write_text("g1,g2\n1,2\n")
    with pytest.raises(NotImplementedError, match="cells are along the rows"):
        CSVReader(str(path), rows_are_cells=False)


def test_csv_reader_preserves_cell_ids_across_batches(tmp_path):
    from scarf.readers import CSVReader

    path = tmp_path / "counts.csv"
    path.write_text("cell,g1,g2\ncell_A,1,2\ncell_B,3,4\n")
    reader = CSVReader(str(path), id_column=0, batch_size=1)

    assert reader.nCells == 2
    assert reader.nFeatures == 2
    np.testing.assert_array_equal(reader.cell_ids(), ["cell_A", "cell_B"])
    np.testing.assert_array_equal(reader.feature_ids(), ["g1", "g2"])
    np.testing.assert_array_equal(
        np.concatenate([counts for counts, _metadata in reader.consume()]),
        [[1, 2], [3, 4]],
    )


def test_csv_reader_rejects_non_mapping_pandas_kwargs(tmp_path):
    from scarf.readers import CSVReader

    path = tmp_path / "counts.csv"
    path.write_text("g1,g2\n1,2\n", encoding="utf-8")
    with pytest.raises(TypeError, match="pandas_kwargs must be a dictionary"):
        CSVReader(str(path), pandas_kwargs=["header"])


def test_h5ad_reader_range_guards(tmp_path) -> None:
    from scarf.readers import H5adReader
    from tests.test_writers import _write_h5ad

    values = np.arange(6 * 3, dtype=np.uint32).reshape(6, 3)
    path = _write_h5ad(tmp_path / "dense.h5ad", values, encoding="dense")
    reader = H5adReader(str(path))
    try:
        with pytest.raises(ValueError, match="outside the matrix"):
            list(reader.consume_dataset(batch_size=2, row_start=2, row_end=1))
    finally:
        reader.close()

    sparse = _write_h5ad(tmp_path / "sparse.h5ad", values)
    sparse_reader = H5adReader(str(sparse))
    try:
        with pytest.raises(ValueError, match="outside the matrix"):
            list(sparse_reader.consume_group(2, row_start=4, row_end=9))
    finally:
        sparse_reader.h5.close()


def _texts(values) -> list[str]:
    from scarf.readers._text import as_text

    return [as_text(value) for value in values]


def test_inspect_h5ad_reads_indexes_that_anndata_writes(tmp_path):
    anndata = pytest.importorskip("anndata")
    import pandas as pd
    from scipy.sparse import csr_matrix

    from scarf.readers import H5adReader, inspect_h5ad

    n_cells, n_genes = 8, 4
    obs = pd.DataFrame(
        {
            "sample": pd.Categorical(["s1"] * 4 + ["s2"] * 4),
            # A unique numeric column must not be mistaken for the cell IDs.
            "n_genes": np.arange(n_cells, dtype=np.int64) * 10,
        },
        index=[f"AAACCTG-{index}" for index in range(n_cells)],
    )
    var = pd.DataFrame(
        {
            "gene_symbols": [f"SYM{index}" for index in range(n_genes)],
            "feature_types": pd.Categorical(["Gene Expression"] * n_genes),
        },
        index=[f"ENSG{index:05d}" for index in range(n_genes)],
    )
    path = tmp_path / "anndata.h5ad"
    anndata.AnnData(
        X=csr_matrix(np.ones((n_cells, n_genes), dtype=np.float32)),
        obs=obs,
        var=var,
    ).write_h5ad(path)

    inspection = inspect_h5ad(str(path))
    assert inspection.cellIdsKey == "_index"
    assert inspection.featureIdsKey == "_index"
    assert inspection.featureNameKey == "gene_symbols"

    reader = H5adReader.from_inspect(inspection)
    try:
        assert _texts(reader.cell_ids()) == list(obs.index)
        assert _texts(reader.feat_ids()) == list(var.index)
        assert _texts(reader.feat_names()) == list(var["gene_symbols"])
        assert set(dict(reader.get_cell_columns())) == {"sample", "n_genes"}
    finally:
        reader.close()


def test_inspect_h5ad_names_features_from_the_index_beside_gene_ids(tmp_path):
    import h5py

    from scarf.readers import inspect_h5ad

    path = tmp_path / "symbols_in_index.h5ad"
    text = h5py.string_dtype()
    with h5py.File(path, "w") as h5:
        _write_sparse_group(h5, "X", np.array([[1, 0, 2], [0, 3, 1]], dtype=np.uint16))
        obs = h5.create_group("obs")
        obs.attrs["_index"] = "_index"
        obs.create_dataset(
            "_index", data=np.array(["c0", "c1"], dtype=object), dtype=text
        )
        var = h5.create_group("var")
        var.attrs["_index"] = "_index"
        # scanpy.read_10x_h5 keeps symbols in the index and IDs in gene_ids.
        var.create_dataset(
            "_index",
            data=np.array(["MIR1302-2HG", "FAM138A", "OR4F5"], dtype=object),
            dtype=text,
        )
        var.create_dataset(
            "gene_ids",
            data=np.array(
                ["ENSG00000243485", "ENSG00000237613", "ENSG00000186092"], dtype=object
            ),
            dtype=text,
        )
        for name, value in (("feature_types", "Gene Expression"), ("genome", "GRCh38")):
            column = var.create_group(name)
            column.attrs["encoding-type"] = "categorical"
            column.create_dataset("codes", data=np.zeros(3, dtype=np.int8))
            column.create_dataset(
                "categories", data=np.array([value], dtype=object), dtype=text
            )

    inspection = inspect_h5ad(str(path))

    assert inspection.featureIdsKey == "gene_ids"
    assert inspection.featureNameKey == "_index"


def test_h5ad_reader_uses_a_named_dataframe_index_by_default(tmp_path):
    import h5py

    from scarf.readers import H5adReader

    path = tmp_path / "named_index.h5ad"
    with h5py.File(path, "w") as h5:
        _write_sparse_group(h5, "X", np.array([[1, 0], [0, 2]], dtype=np.uint16))
        obs = h5.create_group("obs")
        obs.attrs["_index"] = "barcode"
        obs.create_dataset("barcode", data=np.array([b"AAAC-1", b"AAAG-1"]))
        obs.create_dataset("batch", data=np.array([b"a", b"b"]))
        var = h5.create_group("var")
        var.attrs["_index"] = "gene_ids"
        var.create_dataset("gene_ids", data=np.array([b"g0", b"g1"]))

    reader = H5adReader(str(path))
    try:
        assert reader.cellIdsKey == "barcode"
        assert reader.featIdsKey == "gene_ids"
        assert _texts(reader.cell_ids()) == ["AAAC-1", "AAAG-1"]
        assert set(dict(reader.get_cell_columns())) == {"batch"}
    finally:
        reader.close()


def test_h5ad_reader_rejects_duplicate_or_missing_identifiers(tmp_path):
    import h5py

    from scarf.readers import H5adReader

    path = tmp_path / "duplicate_ids.h5ad"
    with h5py.File(path, "w") as h5:
        _write_sparse_group(h5, "X", np.array([[1, 0], [0, 2]], dtype=np.uint16))
        h5.create_group("obs").create_dataset("_index", data=np.array([b"c0", b"c0"]))
        h5.create_group("var").create_dataset("_index", data=np.array([b"g0", b""]))

    reader = H5adReader(str(path))
    try:
        with pytest.raises(
            ValueError, match="H5AD cell IDs must contain unique values"
        ):
            reader.cell_ids()
        with pytest.raises(
            ValueError, match="H5AD feature IDs must contain non-empty values"
        ):
            reader.feat_ids()
    finally:
        reader.close()


def test_crh5_reader_prefers_the_matrix_group_over_other_root_groups(tmp_path):
    import h5py

    from scarf.readers import CrH5Reader

    path = tmp_path / "extra_groups.h5"
    values = np.array([[1, 0, 2], [0, 3, 0]], dtype=np.int32)
    _write_cr_h5(path, values)
    with h5py.File(path, "r+") as h5:
        # CellBender writes latent groups beside the Cell Ranger matrix.
        h5.create_group("droplet_latents").create_dataset("x", data=np.zeros(2))
        h5.create_group("global_latents")

    reader = CrH5Reader(str(path))
    try:
        assert (reader.nCells, reader.nFeatures) == values.shape
        assert np.vstack([chunk.toarray() for chunk in reader.consume(2)]).tolist() == (
            values.tolist()
        )
    finally:
        reader.close()


def test_crh5_reader_rejects_several_genome_groups(tmp_path):
    import h5py

    from scarf.readers import CrH5Reader

    path = tmp_path / "barnyard_v2.h5"
    _write_cr_h5(path, np.eye(2, dtype=np.int32), legacy=True)
    with h5py.File(path, "r+") as h5:
        h5.copy("genome", "mm10")

    with pytest.raises(
        ValueError, match="exactly one genome group; found: genome, mm10"
    ):
        CrH5Reader(str(path))


def test_crh5_reader_filters_a_matrix_without_barcodes(tmp_path):
    from scarf.readers import CrH5Reader

    path = tmp_path / "empty.h5"
    _write_cr_h5(path, np.zeros((0, 2), dtype=np.int32))

    reader = CrH5Reader(str(path), is_filtered=False)
    try:
        assert reader.nCells == 0
        assert reader.validBarcodeIdx.tolist() == []
    finally:
        reader.close()


def test_csv_reader_resolves_dtypes_from_every_chunk(tmp_path):
    from scarf.readers import CSVReader

    path = tmp_path / "counts.csv"
    path.write_text("cell,g1,g2,score,label\nc1,1,0,5,a\nc2,0,2,6,b\nc3,2.5,7,7.9,\n")

    reader = CSVReader(
        str(path),
        id_column=0,
        batch_size=2,
        cell_data_cols=["score", "label"],
    )

    assert reader.countDtype == np.dtype(np.float64)
    assert reader.cellDataDtypes == [np.dtype(np.float64), np.dtype(object)]


@pytest.mark.parametrize(
    ("row", "message"),
    [("c3,,7", "missing or non-finite"), ("c3,-1,7", "must not be negative")],
)
def test_csv_reader_rejects_missing_and_negative_counts(tmp_path, row, message):
    from scarf.readers import CSVReader

    path = tmp_path / "counts.csv"
    path.write_text(f"cell,g1,g2\nc1,1,0\nc2,0,2\n{row}\n")

    with pytest.raises(ValueError, match=message):
        CSVReader(str(path), id_column=0, batch_size=2)


@pytest.mark.parametrize(
    ("text", "kwargs", "error", "message"),
    [
        ("g1,g2\n", {}, ValueError, "contains no data rows"),
        ("g1,g2\n1,x\n", {}, ValueError, "must contain numbers"),
        ("g1,g2\n1,True\n2,False\n", {}, ValueError, "must contain numbers"),
        (
            "g1,g2\n1,2\n",
            {"cell_data_cols": ["batch"]},
            KeyError,
            "cell_data_cols are not CSV columns",
        ),
        # Unknown skip_cols names used to be ignored.
        ("g1,g2\n1,2\n", {"skip_cols": ["drop"]}, KeyError, "skip_cols are not CSV"),
    ],
)
def test_csv_reader_rejects_unusable_files(tmp_path, text, kwargs, error, message):
    from scarf.readers import CSVReader

    path = tmp_path / "counts.csv"
    path.write_text(text)
    with pytest.raises(error, match=message):
        CSVReader(str(path), **kwargs)


def test_h5ad_column_helpers_decode_legacy_layouts(tmp_path) -> None:
    import h5py

    from scarf.readers._h5ad_columns import (
        read_column,
        read_table_column,
        sparse_encoding,
        sparse_shape,
        table_column_dtype,
    )

    with h5py.File(
        tmp_path / "columns.h5", "w", driver="core", backing_store=False
    ) as h5:
        # AnnData 0.6 stored obs as a compound dataset with codes whose
        # categories live in uns.
        obs = h5.create_dataset(
            "obs",
            data=np.array(
                [(b"c1", 1), (b"c2", 0)], dtype=[("index", "S2"), ("batch", "<i4")]
            ),
        )
        h5.create_group("uns")["batch_categories"] = np.array([b"a", b"b"])
        values, missing = read_table_column(h5, obs, "batch", ())
        assert values.tolist() == [b"b", b"a"]
        assert not missing.any()
        assert table_column_dtype(h5, obs, "batch", ()) == np.dtype("S1")
        for read in (read_table_column, table_column_dtype):
            with pytest.raises(KeyError, match="Column 'absent' was not found"):
                read(h5, obs, "absent", ())

        unsupported = h5.create_group("awkward")
        unsupported.attrs["encoding-type"] = "awkward-array"
        with pytest.raises(TypeError, match="unsupported H5AD encoding"):
            read_column(unsupported)

        matrix = h5.create_group("matrix")
        assert (sparse_encoding(matrix), sparse_shape(matrix)) == (None, None)
        matrix.attrs["h5sparse_format"] = "csc"
        matrix.attrs["h5sparse_shape"] = [2, 3]
        assert (sparse_encoding(matrix), sparse_shape(matrix)) == ("csc", (2, 3))
        matrix.attrs["encoding-type"] = "array"
        matrix.attrs["shape"] = [1, 2, 3]
        assert (sparse_encoding(matrix), sparse_shape(matrix)) == (None, None)


def _add_weird_cluster(obs) -> None:
    obs.create_group("labels").attrs["encoding-type"] = "awkward-array"


def _add_grid_cluster(obs) -> None:
    obs.create_dataset("labels", data=np.zeros((3, 2), dtype=np.int32))


def _add_misaligned_nullable_cluster(obs) -> None:
    group = obs.create_group("labels")
    group.create_dataset("values", data=np.arange(3, dtype=np.int32))
    group.create_dataset("mask", data=np.zeros(2, dtype=bool))


@pytest.mark.parametrize(
    ("add_labels", "error", "message"),
    [
        (_add_weird_cluster, TypeError, "unsupported H5AD encoding"),
        (_add_grid_cluster, TypeError, "one scalar value per cell"),
        (_add_misaligned_nullable_cluster, ValueError, "misaligned missingness"),
    ],
)
def test_h5ad_reader_validates_cluster_key_encodings(
    tmp_path, add_labels, error, message
) -> None:
    import h5py

    from scarf.readers import H5adReader
    from tests.test_writers import _write_h5ad

    path = _write_h5ad(tmp_path / "clusters.h5ad", np.ones((3, 4), dtype=np.uint16))
    with h5py.File(path, "a") as h5:
        add_labels(h5["obs"])
    with pytest.raises(error, match=message):
        H5adReader(str(path), cluster_keys=["labels"])


def test_h5ad_reader_requires_a_sparse_matrix_shape(tmp_path) -> None:
    import h5py

    from scarf.readers import H5adReader
    from tests.test_writers import _write_h5ad

    path = _write_h5ad(tmp_path / "noshape.h5ad", np.ones((3, 4), dtype=np.uint16))
    with h5py.File(path, "a") as h5:
        # Without obs, the cell count comes from the matrix shape.
        del h5["obs"]
        del h5["X"].attrs["shape"]
    with pytest.raises(ValueError, match="has no shape attribute"):
        H5adReader(str(path))


def _inspect_file(tmp_path, build):
    """Write an H5AD file through ``build`` and inspect it."""
    import h5py

    from scarf.readers import inspect_h5ad

    path = tmp_path / "inspected.h5ad"
    with h5py.File(path, mode="w") as h5:
        build(h5)
    return inspect_h5ad(str(path))


def test_inspect_h5ad_without_obs_columns_generates_cell_ids(tmp_path):
    def build(h5):
        _write_sparse_group(h5, "X", np.array([[1, 0], [0, 2], [3, 0]], np.uint16))
        # An obs group without columns holds neither IDs nor a length.
        h5.create_group("obs").create_group("__categories")
        h5.create_group("var").create_dataset("gene_ids", data=np.array([b"a", b"b"]))

    inspection = _inspect_file(tmp_path, build)
    assert inspection.cellIdsKey == "_index"
    assert (inspection.nCells, inspection.nFeatures) == (3, 2)
    assert inspection.featureIdsKey == "gene_ids"

    def build_without_obs(h5):
        _write_sparse_group(h5, "X", np.array([[1, 0], [0, 2]], np.uint16))
        h5.create_group("var").create_dataset("gene_ids", data=np.array([b"a", b"b"]))

    assert _inspect_file(tmp_path, build_without_obs).cellIdsKey == "_index"


@pytest.mark.parametrize(
    ("values", "integer_like"),
    [
        # A sparse matrix without entries holds no evidence of counts.
        (np.zeros((2, 2), dtype=np.float32), False),
        # Boolean values are not numbers.
        (np.array([[True, False], [False, True]]), False),
    ],
    ids=["empty-float", "boolean"],
)
def test_inspect_h5ad_sparse_values_without_numeric_counts_are_not_integer_like(
    tmp_path, values, integer_like
):
    def build(h5):
        _write_sparse_group(h5, "X", values)
        h5.create_group("obs").create_dataset("_index", data=np.array([b"c0", b"c1"]))

    inspection = _inspect_file(tmp_path, build)
    assert inspection.matrixEncoding == "csr"
    assert inspection.integerLike is integer_like


def test_inspect_h5ad_dense_candidates_by_dtype_and_size(tmp_path):
    def build(h5):
        h5.create_dataset("X", data=np.array([[1, 0], [0, 2]], dtype=np.int32))
        layers = h5.create_group("layers")
        # A text grid is not a count matrix candidate.
        layers.create_dataset("labels", data=np.array([[b"a", b"b"], [b"c", b"d"]]))
        h5.create_group("obs").create_dataset("_index", data=np.array([b"c0", b"c1"]))

    inspection = _inspect_file(tmp_path, build)
    assert inspection.matrixCandidates == ("X",)
    assert inspection.matrixEncoding == "dense"
    # Integer storage is integer-like without sampling its values.
    assert inspection.integerLike is True

    def build_empty(h5):
        h5.create_dataset("X", data=np.zeros((0, 3), dtype=np.float32))

    empty = _inspect_file(tmp_path, build_empty)
    assert (empty.nCells, empty.nFeatures) == (0, 3)
    # An empty matrix offers no sample to call integer-like.
    assert empty.integerLike is False


def test_inspect_h5ad_skips_cell_id_columns_that_do_not_fit_the_cells(tmp_path):
    def build(h5):
        _write_sparse_group(h5, "X", np.array([[1, 0], [0, 2]], np.uint16))
        obs = h5.create_group("obs")
        obs.attrs["_index"] = "names"
        # The named index repeats a value, and the preferred cell_id column
        # holds one value more than there are cells.
        obs.create_dataset("names", data=np.array([b"x", b"x"]))
        obs.create_dataset("cell_id", data=np.array([b"a", b"b", b"c"]))
        obs.create_dataset("sequence", data=np.array([b"AC", b"GT"]))

    assert _inspect_file(tmp_path, build).cellIdsKey == "sequence"


def test_inspect_h5ad_skips_columns_it_cannot_decode(tmp_path):
    def build(h5):
        _write_sparse_group(h5, "X", np.array([[1, 0], [0, 2]], np.uint16))
        obs = h5.create_group("obs")
        broken = obs.create_group("barcode")
        broken.attrs["encoding-type"] = "nullable-string-array"
        broken.create_dataset("values", data=np.array([b"a", b"b"]))
        broken.create_dataset("mask", data=np.array([False]))
        obs.create_dataset("well", data=np.array([b"A1", b"A2"]))

    # The barcode mask does not align with its values, so the next unique
    # text column holds the IDs.
    assert _inspect_file(tmp_path, build).cellIdsKey == "well"


def test_inspect_h5ad_uses_feature_names_as_ids_without_unique_ids(tmp_path):
    def build(h5):
        _write_sparse_group(h5, "X", np.array([[1, 0], [0, 2]], np.uint16))
        h5.create_group("obs").create_dataset("_index", data=np.array([b"c0", b"c1"]))
        h5.create_group("var").create_dataset(
            "gene_symbols", data=np.array([b"GENE", b"GENE"])
        )

    inspection = _inspect_file(tmp_path, build)
    assert inspection.featureIdsKey == "gene_symbols"
    assert inspection.featureNameKey == "gene_symbols"


def test_inspect_h5ad_reads_text_lengths_of_columns_without_values(tmp_path):
    def build(h5):
        _write_sparse_group(h5, "X", np.array([[1], [2]], np.uint16))
        h5.create_group("obs").create_dataset("_index", data=np.array([b"c0", b"c1"]))
        var = h5.create_group("var")
        # One feature, whose only text column is missing for it.
        missing = var.create_group("alias")
        missing.attrs["encoding-type"] = "nullable-string-array"
        missing.create_dataset("values", data=np.array([b""]))
        missing.create_dataset("mask", data=np.array([True]))

    inspection = _inspect_file(tmp_path, build)
    assert inspection.featureIdsKey == "alias"
    assert inspection.featureNameKey == "alias"


def test_inspect_h5ad_matches_matrices_to_feature_groups(tmp_path):
    from loguru import logger

    def build(h5):
        # raw/X is integer and ranks first, but raw/var holds two features
        # for its three columns.
        _write_sparse_group(h5, "raw/X", np.array([[1, 0, 3], [0, 2, 0]], np.uint16))
        h5.create_group("raw/var").create_dataset(
            "gene_ids", data=np.array([b"r1", b"r2"])
        )
        _write_sparse_group(h5, "X", np.array([[1, 0], [0, 2]], np.uint16))
        h5.create_group("var").create_dataset("gene_ids", data=np.array([b"a", b"b"]))
        h5.create_group("obs").create_dataset("_index", data=np.array([b"c0", b"c1"]))

    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]), level="WARNING"
    )
    try:
        inspection = _inspect_file(tmp_path, build)
    finally:
        logger.remove(sink)
    assert (inspection.matrixKey, inspection.featureAttrsKey) == ("X", "var")
    assert any(
        "Ignoring matrix candidate raw/X: feature metadata group `raw/var` "
        "length 2 does not match feature count 3" in message
        for message in messages
    )


def test_inspect_h5ad_borrows_the_dimension_matched_feature_group(tmp_path):
    def build(h5):
        _write_sparse_group(h5, "raw/X", np.array([[1, 0, 3], [0, 2, 0]], np.uint16))
        h5.create_group("var").create_dataset(
            "gene_ids", data=np.array([b"a", b"b", b"c"])
        )
        h5.create_group("obs").create_dataset("_index", data=np.array([b"c0", b"c1"]))

    inspection = _inspect_file(tmp_path, build)
    # raw/X has no raw/var, so it takes the var group of its three features.
    assert (inspection.matrixKey, inspection.featureAttrsKey) == ("raw/X", "var")
    assert inspection.featureIdsKey == "gene_ids"


def test_inspect_h5ad_rejects_matrices_without_matching_feature_groups(tmp_path):
    from loguru import logger

    from scarf.readers import inspect_h5ad

    def build(h5):
        _write_sparse_group(h5, "raw/X", np.array([[1, 0, 3], [0, 2, 0]], np.uint16))
        _write_sparse_group(h5, "X", np.array([[1, 0, 3], [0, 2, 0]], np.uint16))
        h5.create_group("var").create_dataset("gene_ids", data=np.array([b"a", b"b"]))
        h5.create_group("obs").create_dataset("_index", data=np.array([b"c0", b"c1"]))

    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]), level="WARNING"
    )
    try:
        with pytest.raises(
            ValueError, match="No matrix candidate matches the obs and var dimensions"
        ):
            _inspect_file(tmp_path, build)
    finally:
        logger.remove(sink)
    # raw/X has no raw/var and var does not match it; X does not match var.
    assert any(
        "Ignoring matrix candidate raw/X: no feature metadata group has 3 rows"
        in message
        for message in messages
    )
    assert any(
        "Ignoring matrix candidate X: feature metadata group `var` length 2" in message
        for message in messages
    )
    with pytest.raises(
        ValueError, match=r"matrix_key 'layers/counts' not found. Available: raw/X, X"
    ):
        inspect_h5ad(str(tmp_path / "inspected.h5ad"), matrix_key="layers/counts")


def test_inspect_h5ad_drops_a_split_key_whose_length_differs(tmp_path):
    def build(h5):
        _write_sparse_group(h5, "X", np.array([[1, 0], [0, 2]], np.uint16))
        h5.create_group("obs").create_dataset("_index", data=np.array([b"c0", b"c1"]))
        var = h5.create_group("var")
        var.attrs["_index"] = "gene_ids"
        var.create_dataset("gene_ids", data=np.array([b"a", b"b"]))
        var.create_dataset(
            "feature_types",
            data=np.array([b"Gene Expression", b"Antibody Capture", b"Peaks"]),
        )

    inspection = _inspect_file(tmp_path, build)
    assert inspection.assaySplitKey is None
    assert inspection.suggestedAssays == {}


def test_h5ad_reader_rejects_identifiers_that_are_not_one_dimensional(tmp_path):
    import h5py

    from scarf.readers import H5adReader

    path = tmp_path / "grid_ids.h5ad"
    with h5py.File(path, mode="w") as h5:
        _write_sparse_group(h5, "X", np.array([[1, 0], [0, 2]], np.uint16))
        obs = h5.create_group("obs")
        obs.create_dataset("_index", data=np.array([b"c0", b"c1"]))
        obs.create_dataset("grid", data=np.array([[b"a", b"b"], [b"c", b"d"]]))
    reader = H5adReader(str(path), cell_ids_key="grid")
    try:
        with pytest.raises(ValueError, match="H5AD cell IDs must be one-dimensional"):
            reader.cell_ids()
    finally:
        reader.close()


def test_crh5_reader_rejects_a_matrix_without_features(tmp_path):
    from scarf.readers import CrH5Reader

    path = tmp_path / "no_features.h5"
    _write_cr_h5(path, np.zeros((2, 0), dtype=np.int32))

    with pytest.raises(
        ValueError, match="Cannot build an assay table without features"
    ):
        CrH5Reader(str(path))


def test_h5ad_table_column_names_of_a_node_that_is_not_a_table(tmp_path):
    import h5py

    from scarf.readers._h5ad_columns import table_column_names

    with h5py.File(tmp_path / "types.h5", mode="w") as h5:
        # A committed datatype is neither a dataframe group nor a dataset.
        h5["var"] = np.dtype("f8")
        assert isinstance(h5["var"], h5py.Datatype)
        assert table_column_names(h5["var"]) == []
