"""Validation and edge paths of the H5AD reader and the Seurat matrix sources."""

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from scipy.sparse import csc_matrix, csr_matrix

from scarf.readers import H5adReader, inspect_h5ad
from scarf.readers._seurat import (
    BaseMatrixSource,
    CscMatrixSource,
    DelayedSubassignmentMatrixSource,
    DenseMatrixSource,
    LayerStitchMatrixSource,
    MatrixSourceError,
    RenamedMatrixSource,
    ResourceLimitError,
    SourceLimits,
    Subassignment,
    TransposeMatrixSource,
)
from scarf.readers._seurat.sources import (
    prepare_matrix_sources,
    release_temporary_storage,
)

_COUNTS = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5], [0, 6, 0]], dtype=np.uint16)


def _write_h5ad(
    path: Path, *, encoding: str = "csr", counts: np.ndarray = _COUNTS
) -> Path:
    """Write ``counts`` with obs and var tables of matching length.

    The file holds the values in the dtype of ``counts``, byte order included.
    """
    with h5py.File(path, "w") as h5:
        if encoding == "dense":
            h5.create_dataset("X", data=counts)
        else:
            # SciPy sparse matrices hold values in native byte order only.
            native = counts.astype(counts.dtype.newbyteorder("="))
            matrix = (csr_matrix if encoding == "csr" else csc_matrix)(native)
            group = h5.create_group("X")
            group.attrs["encoding-type"] = f"{encoding}_matrix"
            group.attrs["shape"] = counts.shape
            group.create_dataset("data", data=matrix.data.astype(counts.dtype))
            group.create_dataset("indices", data=matrix.indices)
            group.create_dataset("indptr", data=matrix.indptr)
        h5.create_group("obs").create_dataset(
            "_index", data=np.array([b"c0", b"c1", b"c2", b"c3"])
        )
        h5.create_group("var").create_dataset(
            "_index", data=np.array([b"f0", b"f1", b"f2"])
        )
    return path


def _replace(group: h5py.Group, name: str, data: Any) -> None:
    del group[name]
    group.create_dataset(name, data=data)


def _shorten_obs(h5: h5py.File) -> None:
    _replace(h5["obs"], "_index", np.array([b"c0", b"c1", b"c2"]))


def _lengthen_var(h5: h5py.File) -> None:
    _replace(h5["var"], "_index", np.array([b"f0", b"f1", b"f2", b"f3"]))


def _add_row_pointer(h5: h5py.File) -> None:
    _replace(h5["X"], "indptr", np.array([0, 2, 3, 5, 6, 6]))


def _drop_shape(h5: h5py.File) -> None:
    del h5["X"].attrs["shape"]


def _flatten_matrix(h5: h5py.File) -> None:
    del h5["X"]
    h5.create_dataset("X", data=np.arange(4, dtype=np.float32))


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        # Each mismatch used to import: rows past obs were dropped, an extra
        # var row became an empty feature, and surplus row pointers were
        # ignored. A flat matrix failed only while its counts were written.
        (_shorten_obs, r"has shape \(4, 3\), but `obs` has 3 rows"),
        (_lengthen_var, r"`var` has 4 rows"),
        (_add_row_pointer, r"needs an indptr of shape \(5,\); found \(6,\)"),
        (_drop_shape, "has no shape attribute"),
        (_flatten_matrix, "must be two-dimensional"),
    ],
)
def test_h5ad_reader_rejects_a_matrix_that_disagrees_with_its_tables(
    tmp_path, edit, message
) -> None:
    path = _write_h5ad(tmp_path / "counts.h5ad")
    with h5py.File(path, "a") as h5:
        edit(h5)
    with pytest.raises(ValueError, match=message):
        H5adReader(str(path))


def test_h5ad_import_matches_numeric_cell_ids_to_embedding_rows(tmp_path) -> None:
    from scarf import DataStore
    from scarf.writers import H5adToZarr

    path = _write_h5ad(tmp_path / "numeric_ids.h5ad")
    umap = np.arange(8, dtype=np.float32).reshape(4, 2)
    with h5py.File(path, "a") as h5:
        h5["obs"].create_dataset("cell_number", data=np.arange(100, 104))
        h5.create_group("obsm").create_dataset("X_umap", data=umap)
    reader = H5adReader(
        str(path), cell_ids_key="cell_number", embedding_roles={"X_umap": "umap"}
    )
    try:
        # The embedding's cell IDs used to be the source integers, which the
        # import could not match to the text IDs that cellData stores.
        result = H5adToZarr(reader, str(tmp_path / "ids.zarr"), nthreads=1).dump()
    finally:
        reader.close()
    store = DataStore(str(tmp_path / "ids.zarr"))
    np.testing.assert_array_equal(
        store.cells.fetch_all("ids"), ["100", "101", "102", "103"]
    )
    embedding = store.load_artifact(result.embeddingArtifacts["X_umap"])
    np.testing.assert_allclose(embedding["values"][:], umap)


def test_h5ad_import_without_tables_generates_ids_for_embeddings(tmp_path) -> None:
    from scarf import DataStore
    from scarf.writers import H5adToZarr

    path = _write_h5ad(tmp_path / "bare.h5ad")
    umap = np.arange(8, dtype=np.float32).reshape(4, 2)
    with h5py.File(path, "a") as h5:
        del h5["obs"]
        del h5["var"]
        h5.create_group("obsm").create_dataset("X_umap", data=umap)
    reader = H5adReader(str(path), embedding_roles={"X_umap": "umap"})
    try:
        result = H5adToZarr(reader, str(tmp_path / "bare.zarr"), nthreads=1).dump()
    finally:
        reader.close()
    store = DataStore(str(tmp_path / "bare.zarr"))
    np.testing.assert_array_equal(
        store.cells.fetch_all("ids"), ["cell_0", "cell_1", "cell_2", "cell_3"]
    )
    np.testing.assert_array_equal(
        store.RNA.feats.fetch_all("ids"), ["feature_0", "feature_1", "feature_2"]
    )
    embedding = store.load_artifact(result.embeddingArtifacts["X_umap"])
    np.testing.assert_allclose(embedding["values"][:], umap)


def test_h5ad_reader_reads_a_table_stored_as_a_plain_dataset(tmp_path) -> None:
    path = _write_h5ad(tmp_path / "plain_obs.h5ad")
    with h5py.File(path, "a") as h5:
        del h5["obs"]
        h5.create_dataset("obs", data=np.array([b"a", b"b", b"c", b"d"]))
    # Inspection accepts a table without fields; the reader used to fail
    # looking up the ID column in it.
    reader = H5adReader.from_inspect(inspect_h5ad(str(path)))
    try:
        assert reader.nCells == 4
        np.testing.assert_array_equal(
            reader.cell_ids(), ["cell_0", "cell_1", "cell_2", "cell_3"]
        )
        assert list(reader.get_cell_columns()) == []
    finally:
        reader.close()


def test_h5ad_reader_rejects_unusable_matrix_and_table_slots(tmp_path) -> None:
    missing_indices = _write_h5ad(tmp_path / "missing_indices.h5ad")
    with h5py.File(missing_indices, "a") as h5:
        del h5["X/indices"]
    with pytest.raises(ValueError, match="`X` is missing: indices"):
        H5adReader(str(missing_indices))

    without_matrix = _write_h5ad(tmp_path / "without_matrix.h5ad")
    with h5py.File(without_matrix, "a") as h5:
        del h5["X"]
    with pytest.raises(ValueError, match="X is neither Dataset or Group type"):
        H5adReader(str(without_matrix))

    unsized = _write_h5ad(tmp_path / "unsized.h5ad")
    with h5py.File(unsized, "a") as h5:
        del h5["obs/_index"]
        h5["obs"].create_group("nested").attrs["encoding-type"] = "dict"
    with pytest.raises(KeyError, match="`obs` key doesn't contain any child node"):
        H5adReader(str(unsized))


def test_h5ad_reader_treats_named_datatypes_as_absent(tmp_path) -> None:
    path = _write_h5ad(tmp_path / "datatypes.h5ad")
    with h5py.File(path, "a") as h5:
        # Assigning a dtype commits a named datatype, neither group nor dataset.
        h5["obs"]["value_type"] = np.dtype("f4")
        del h5["var"]
        h5["var"] = np.dtype("i8")
    reader = H5adReader(str(path))
    try:
        assert reader.groupCodes["var"] == 0
        assert reader.nFeatures == 3
        np.testing.assert_array_equal(
            reader.feat_ids(), ["feature_0", "feature_1", "feature_2"]
        )
        np.testing.assert_array_equal(reader.cell_ids(), [b"c0", b"c1", b"c2", b"c3"])
        assert list(reader.get_cell_columns()) == []
    finally:
        reader.close()


def test_h5ad_reader_generates_ids_for_an_absent_id_key(tmp_path) -> None:
    reader = H5adReader(
        str(_write_h5ad(tmp_path / "counts.h5ad")), cell_ids_key="barcode"
    )
    try:
        assert reader.cellIdsKey == "barcode"
        np.testing.assert_array_equal(
            reader.cell_ids(), ["cell_0", "cell_1", "cell_2", "cell_3"]
        )
    finally:
        reader.close()


def _write_analysis_h5ad(path: Path) -> Path:
    _write_h5ad(path)
    with h5py.File(path, "a") as h5:
        obs = h5["obs"]
        state = obs.create_group("state")
        state.create_dataset("codes", data=np.array([0, 1, 0, 1], dtype=np.int8))
        state.create_dataset("categories", data=np.array([b"a", b"b"]))
        obs.create_dataset("short", data=np.arange(3))
        obs.create_dataset("phase", data=np.ones(4, dtype=np.complex64))
        obsm = h5.create_group("obsm")
        obsm.create_dataset("X_umap", data=np.zeros((4, 2), dtype=np.float32))
        obsm.create_dataset("X_short", data=np.zeros((3, 2), dtype=np.float32))
        obsm.create_dataset("X_labels", data=np.full((4, 2), b"a"))
        sparse = obsm.create_group("X_sparse")
        for name in ("data", "indices", "indptr"):
            sparse.create_dataset(name, data=np.zeros(1))
    return path


@pytest.mark.parametrize(
    ("roles", "error", "message"),
    [
        (["X_umap"], TypeError, "embedding_roles must be a mapping"),
        ({"": "umap"}, ValueError, "keys must be non-empty strings"),
        ({"X_umap": "pca"}, ValueError, "values must be 'umap' or 'tsne'"),
        ({"X_missing": "umap"}, KeyError, "'X_missing' was not found in obsm"),
        ({"X_sparse": "umap"}, TypeError, "must be a dense H5AD array"),
        ({"X_short": "umap"}, ValueError, r"incompatible shape \(3, 2\)"),
        ({"X_labels": "tsne"}, TypeError, "must contain numeric values"),
    ],
)
def test_h5ad_reader_validates_embedding_roles(tmp_path, roles, error, message) -> None:
    path = _write_analysis_h5ad(tmp_path / "analysis.h5ad")
    with pytest.raises(error, match=message):
        H5adReader(str(path), embedding_roles=roles)


@pytest.mark.parametrize(
    ("keys", "error", "message"),
    [
        ("state", TypeError, "must be a sequence of column names"),
        (("state", "state"), ValueError, "cluster_keys must be unique"),
        (("",), ValueError, "must contain non-empty strings"),
        (("_index",), ValueError, "'_index' is reserved H5AD metadata"),
        (("missing",), KeyError, "'missing' was not found in obs"),
        (("short",), ValueError, "'short' has 3 rows; expected 4"),
        (("phase",), TypeError, "'phase' uses unsupported dtype complex64"),
    ],
)
def test_h5ad_reader_validates_cluster_keys(tmp_path, keys, error, message) -> None:
    path = _write_analysis_h5ad(tmp_path / "analysis.h5ad")
    with pytest.raises(error, match=message):
        H5adReader(str(path), cluster_keys=keys)


def test_h5ad_reader_rejects_cluster_keys_of_a_two_dimensional_table(
    tmp_path,
) -> None:
    path = _write_h5ad(tmp_path / "grid_obs.h5ad")
    table = np.zeros((4, 1), dtype=[("_index", "S2"), ("state", "i1")])
    with h5py.File(path, "a") as h5:
        del h5["obs"]
        h5.create_dataset("obs", data=table)
    with pytest.raises(TypeError, match="'state' must contain one scalar value"):
        H5adReader(str(path), cluster_keys=("state",))


def test_h5ad_reader_feature_types_need_one_value_per_feature(tmp_path) -> None:
    path = _write_h5ad(tmp_path / "counts.h5ad")
    with h5py.File(path, "a") as h5:
        h5["var"].create_dataset("feature_types", data=np.array([b"Peaks"] * 2))
    reader = H5adReader(str(path))
    try:
        with pytest.raises(KeyError, match="Feature type key `assay` was not found"):
            reader.feature_types("assay")
        with pytest.raises(ValueError, match="has 2 values; expected 3"):
            reader.feature_types("feature_types")
    finally:
        reader.close()


@pytest.mark.parametrize("batch_size", [0, -1])
def test_dense_h5ad_consume_rejects_nonpositive_batch_sizes(
    tmp_path, batch_size
) -> None:
    reader = H5adReader(str(_write_h5ad(tmp_path / "dense.h5ad", encoding="dense")))
    try:
        # A negative size used to yield no batches at all, and zero failed
        # inside range().
        with pytest.raises(ValueError, match="batch_size must be positive"):
            list(reader.consume(batch_size))
    finally:
        reader.close()


def test_h5ad_reader_batch_bounds(tmp_path) -> None:
    reader = H5adReader(str(_write_h5ad(tmp_path / "csr.h5ad")))
    try:
        with pytest.raises(ValueError, match="batch_size must be positive"):
            reader.max_batch_nnz(0)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            list(reader.consume(0))
        assert reader.max_batch_nnz(3) == 5
    finally:
        reader.close()

    csc = H5adReader(str(_write_h5ad(tmp_path / "csc.h5ad", encoding="csc")))
    try:
        # CSC columns hold no row pointers, so a window is bounded by its
        # dense size until the rows are converted.
        assert csc.max_batch_nnz(3) == 3 * 3
        csc.materialize_csc()
        assert csc.max_batch_nnz(3) == 5
    finally:
        csc.close()


@pytest.mark.parametrize("encoding", ["csr", "csc", "dense"])
@pytest.mark.parametrize(
    ("data_dtype", "values", "read_dtype", "stored_dtype"),
    [
        (">i4", [[1, 0, 300], [0, 3, 0], [4, 0, 5], [0, 6, 0]], np.int32, np.uint16),
        (">i4", [[-1, 0, 2], [0, 3, 0], [4, 0, 5], [0, 6, 0]], np.int32, np.int32),
        (">f4", [[1.5, 0, 2], [0, 3, 0], [4, 0, 5], [0, 6, 0]], np.float32, np.float32),
        (">f8", [[1, 0, 2], [0, 3, 0], [4, 0, 5], [0, 6, 0]], np.float64, np.uint8),
    ],
    ids=["integral-int", "negative-int", "fractional-float", "integral-float"],
)
def test_h5ad_import_reads_big_endian_counts_natively(
    tmp_path,
    encoding: str,
    data_dtype: str,
    values: list[list[float]],
    read_dtype: type,
    stored_dtype: type,
) -> None:
    import zarr

    from scarf.writers import H5adToZarr

    counts = np.array(values, dtype=data_dtype)
    path = _write_h5ad(tmp_path / "counts.h5ad", encoding=encoding, counts=counts)
    reader = H5adReader(str(path))
    try:
        # SciPy sparse matrices hold no other byte order, so these imports
        # used to fail while writing, after the destination was created.
        assert reader.sourceMatrixDtype == read_dtype
        H5adToZarr(reader, str(tmp_path / "counts.zarr"), nthreads=1).dump()
    finally:
        reader.close()

    stored = zarr.open_group(str(tmp_path / "counts.zarr"), mode="r")["RNA/counts"]
    assert stored.dtype == stored_dtype
    np.testing.assert_array_equal(stored[:], values)


def test_matrix_source_bases_are_abstract() -> None:
    with pytest.raises(TypeError, match="abstract"):
        BaseMatrixSource((1, 1), np.float32, is_sparse=False)


def test_source_preparation_reaches_sources_inside_subassignments(tmp_path) -> None:
    values = np.arange(12, dtype=np.float64).reshape(3, 4)
    replacement = csc_matrix(np.array([[7.0, 0.0], [9.0, 8.0]]))
    transposed = TransposeMatrixSource(
        CscMatrixSource(
            replacement.data, replacement.indices, replacement.indptr, (2, 2)
        )
    )
    transposed._tempDir = tmp_path
    source = DelayedSubassignmentMatrixSource(
        DenseMatrixSource(values),
        [Subassignment([0], [0], -1.0), Subassignment([1, 2], [2, 3], transposed)],
    )
    expected = values.T.copy()
    expected[0, 0] = -1.0
    expected[np.ix_([2, 3], [1, 2])] = replacement.toarray()
    try:
        # A scalar assignment holds no source; the sparse transpose inside the
        # other assignment is converted once into temporary row storage.
        prepare_matrix_sources(source)
        assert len(list(tmp_path.iterdir())) == 1
        np.testing.assert_array_equal(source.read_cells(0, 4), expected)
    finally:
        release_temporary_storage(source)
    assert list(tmp_path.iterdir()) == []


def test_transpose_of_an_in_memory_dense_matrix_needs_no_storage(tmp_path) -> None:
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    source = TransposeMatrixSource(DenseMatrixSource(values))
    source._tempDir = tmp_path
    prepare_matrix_sources(source)
    assert list(tmp_path.iterdir()) == []
    np.testing.assert_array_equal(source.read_cells(0, 2), values)


def test_dense_source_requires_an_array_with_a_shape_or_length() -> None:
    with pytest.raises(TypeError, match="must expose shape or len"):
        DenseMatrixSource(object())


class _MatrixLike:
    """A two-dimensional array-like whose column slices can misbehave."""

    dtype = np.dtype(np.float64)
    shape = (2, 3)

    def __init__(self, block: np.ndarray | None) -> None:
        self._block = block

    def __getitem__(self, key: Any) -> np.ndarray:
        if self._block is None:
            raise TypeError("no two-dimensional indexing")
        return self._block


def test_dense_source_requires_bounded_column_slices() -> None:
    with pytest.raises(MatrixSourceError, match="does not support bounded column"):
        DenseMatrixSource(_MatrixLike(None)).read_cells(0, 2)
    with pytest.raises(
        MatrixSourceError, match=r"dense block has shape \(2, 1\); expected \(2, 2\)"
    ):
        DenseMatrixSource(_MatrixLike(np.ones((2, 1)))).read_cells(0, 2)


def test_csc_pointers_are_checked_across_chunk_boundaries() -> None:
    limits = SourceLimits(compressedChunkNnz=2)
    source = CscMatrixSource([1, 2, 3], [0, 0, 0], [0, 1, 2, 3], (1, 3), limits=limits)
    np.testing.assert_array_equal(source.read_cells(0, 3).toarray(), [[1], [2], [3]])
    with pytest.raises(MatrixSourceError, match="p slot must be nondecreasing"):
        CscMatrixSource([1, 1, 1], [0, 0, 0], [0, 2, 1, 3], (1, 3), limits=limits)


def test_row_and_column_names_share_the_metadata_budget() -> None:
    # Each axis needs 18 bytes, one UTF-8 byte and an 8-byte reference per name.
    limits = SourceLimits(maxMetadataBytes=20)
    values = np.ones((2, 2))
    named = DenseMatrixSource(values, row_names=["a", "b"], limits=limits)
    assert named.row_names == ("a", "b")
    assert named.resident_bytes == 2 * 2 * 8 + 18
    with pytest.raises(ResourceLimitError, match="names exceed maxMetadataBytes=20"):
        DenseMatrixSource(
            values, row_names=["a", "b"], column_names=["c", "d"], limits=limits
        )


def test_layer_stitching_reads_unsorted_and_empty_layers() -> None:
    backwards = DenseMatrixSource(
        np.array([[1, 2, 3]]), row_names=["f1"], column_names=["c3", "c2", "c1"]
    )
    zeros = DenseMatrixSource(
        np.zeros((1, 3)), row_names=["f2"], column_names=["c1", "c2", "c3"]
    )
    stitched = LayerStitchMatrixSource(
        [backwards, zeros], row_names=["f1", "f2"], column_names=["c1", "c2", "c3"]
    )
    np.testing.assert_array_equal(
        stitched.read_cells(0, 3).toarray(), [[3, 0], [2, 0], [1, 0]]
    )
    np.testing.assert_array_equal(stitched.read_cells(1, 3).toarray(), [[2, 0], [1, 0]])


def test_renamed_source_counts_its_names_and_its_child() -> None:
    child = DenseMatrixSource(np.ones((2, 3)))
    renamed = RenamedMatrixSource(child, row_names=["f1", "f2"], column_names=None)
    assert renamed.row_names == ("f1", "f2")
    assert renamed.resident_bytes == child.resident_bytes + 2 * (2 + 8)
    np.testing.assert_array_equal(renamed.read_cells(0, 3), np.ones((3, 2)))
