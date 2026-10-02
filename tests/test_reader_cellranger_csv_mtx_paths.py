"""Behavior of the 10x HDF5, Matrix Market, and CSV readers at their edges."""

import csv
import gzip
import shutil
import stat
import zipfile
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import zarr
from scipy.sparse import coo_matrix, csr_matrix
from zarr.storage import MemoryStore

from scarf.readers import (
    CSVReader,
    CrDirReader,
    CrH5Reader,
    CrReader,
    MtxReader,
    inspect_mtx,
)
from scarf.storage.count_matrix import CountMatrixPolicy
from scarf.utils.count_values import CountValueRange, compressed_count_ranges
from scarf.writers import CrToZarr, CSVtoZarr, MtxToZarr

_POLICY = CountMatrixPolicy(unitBytes=16, chunkBytes=8)
_BANNER = "%%MatrixMarket matrix coordinate integer general\n"


def _write_h5(
    path: Path,
    values: np.ndarray,
    *,
    legacy: bool = False,
    data_dtype: str | None = None,
) -> None:
    matrix = csr_matrix(values)
    n_cells, n_features = values.shape
    data = matrix.data if data_dtype is None else matrix.data.astype(data_dtype)
    with h5py.File(path, mode="w") as handle:
        group = handle.create_group("genome" if legacy else "matrix")
        group.create_dataset("data", data=data)
        group.create_dataset("indices", data=matrix.indices.astype(np.int64))
        group.create_dataset("indptr", data=matrix.indptr.astype(np.int64))
        group.create_dataset(
            "barcodes", data=np.array([f"cell-{i}".encode() for i in range(n_cells)])
        )
        ids = np.array([f"feature-{i}".encode() for i in range(n_features)])
        names = np.array([f"gene-{i}".encode() for i in range(n_features)])
        if legacy:
            group.create_dataset("genes", data=ids)
            group.create_dataset("gene_names", data=names)
            return
        features = group.create_group("features")
        features.create_dataset("id", data=ids)
        features.create_dataset("name", data=names)
        features.create_dataset(
            "feature_type", data=np.array([b"Gene Expression"] * n_features)
        )


def _write_mex(
    directory: Path,
    entries: str,
    *,
    n_features: int,
    n_cells: int,
    n_entries: int | None = None,
    matrix_name: str = "matrix.mtx",
) -> None:
    """Write a 10x MEX triplet whose matrix holds ``entries``, one per line."""
    lines = [line for line in entries.splitlines() if line]
    declared = len(lines) if n_entries is None else n_entries
    (directory / matrix_name).write_text(
        f"{_BANNER}{n_features} {n_cells} {declared}\n{entries}"
    )
    (directory / "features.tsv").write_text(
        "".join(f"feature-{i}\tgene-{i}\tGene Expression\n" for i in range(n_features))
    )
    (directory / "barcodes.tsv").write_text(
        "".join(f"cell-{i}\n" for i in range(n_cells))
    )


def _write_parse(directory: Path, matrix: str) -> None:
    (directory / "DGE.mtx").write_text(matrix)
    (directory / "genes.csv").write_text(
        "gene_id,gene_name\nfeature-0,Gene 0\nfeature-1,Gene 1\nfeature-2,Gene 2\n"
    )
    (directory / "cell_metadata.csv").write_text(
        "bc_wells,sample\ncell-0,A\ncell-1,B\n"
    )


def _counts(store: MemoryStore, assay: str = "RNA") -> np.ndarray:
    return np.asarray(zarr.open_group(store=store, mode="r")[f"{assay}/counts"][:])


class _MemoryReader(CrReader):
    """A reader that implements only the abstract members of ``CrReader``."""

    def __init__(self, values: np.ndarray) -> None:
        self._matrix = csr_matrix(values)
        super().__init__(self._handle_version())

    def _handle_version(self) -> dict[str, str | None]:
        return {
            "feature_ids": "ids",
            "feature_names": "names",
            "feature_types": None,
            "cell_names": "cells",
        }

    def _read_dataset(self, key: str) -> list[str]:
        n_cells, n_features = self._matrix.shape
        return {
            "feature_ids": [f"feature-{i}" for i in range(n_features)],
            "feature_names": [f"gene-{i}" for i in range(n_features)],
            "cell_names": [f"cell-{i}" for i in range(n_cells)],
        }[key]

    @property
    def matrix_dtype(self) -> np.dtype:
        return self._matrix.dtype

    def count_value_ranges(
        self, maxBytes: int, featureGroups: np.ndarray | None = None
    ) -> list[CountValueRange]:
        return compressed_count_ranges(
            self._matrix.indptr,
            self._matrix.indices,
            self._matrix.data,
            minorSize=self._matrix.shape[1],
            maxBytes=maxBytes,
            groups=featureGroups,
        )

    def consume(self, batch_size: int, lines_in_mem: int):
        for start in range(0, self._matrix.shape[0], batch_size):
            yield self._matrix[start : start + batch_size].tocoo()


def test_crreader_subclass_imports_with_the_base_planning_defaults() -> None:
    values = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=np.int64)
    reader = _MemoryReader(values)

    # The base bounds a window by its dense size and reserves no staging.
    assert reader.max_window_nnz(2) == 2 * 3
    assert reader.max_window_nnz(10) == 3 * 3
    with pytest.raises(ValueError, match="window_rows must be positive"):
        reader.max_window_nnz(0)
    assert reader.producer_staging_bytes(2, 1) == 0
    assert list(reader.assayFeats.columns) == ["RNA"]

    store = MemoryStore()
    CrToZarr(reader, store, mem_budget="64M", nthreads=1, policy=_POLICY).dump()

    counts = zarr.open_group(store=store, mode="r")["RNA/counts"]
    assert counts.dtype == np.uint8
    np.testing.assert_array_equal(counts[:], values)


def _write_mixed_mex(directory: Path) -> None:
    (directory / "features.tsv").write_text(
        "f1\tg1\tGene Expression\nf2\th1\tAntibody Capture\nf3\tg2\tGene Expression\n"
    )
    (directory / "barcodes.tsv").write_text("c1\n")
    (directory / "matrix.mtx").write_text(f"{_BANNER}3 1 2\n1 1 2\n2 1 4\n")


@pytest.mark.parametrize(
    ("indexes", "feature_type", "error", "message"),
    [
        ([1], "", ValueError, "non-empty string"),
        ([1], 7, ValueError, "non-empty string"),
        ("1", "HTO", TypeError, "sequence of integer"),
        ([[1]], "HTO", ValueError, "one-dimensional"),
        ([], "HTO", ValueError, "at least one"),
        ([1.0], "HTO", TypeError, "only integers"),
    ],
)
def test_reclassify_features_rejects_malformed_requests(
    tmp_path: Path,
    indexes: object,
    feature_type: object,
    error: type[Exception],
    message: str,
) -> None:
    _write_mixed_mex(tmp_path)
    reader = CrDirReader(str(tmp_path))
    try:
        before = reader.assayFeats.copy()
        with pytest.raises(error, match=message):
            reader.reclassify_features(indexes, feature_type)  # type: ignore[arg-type]
        assert reader.assayFeats.equals(before)
    finally:
        reader.close()


def test_reclassify_features_without_a_required_previous_type(tmp_path: Path) -> None:
    _write_mixed_mex(tmp_path)
    reader = CrDirReader(str(tmp_path))
    try:
        reader.reclassify_features([0], "Multiplexing Capture", require_previous=None)

        assert reader.feature_types() == [
            "Multiplexing Capture",
            "Antibody Capture",
            "Gene Expression",
        ]
        assert list(reader.assayFeats.columns) == ["HTO", "ADT", "RNA"]
    finally:
        reader.close()


def test_unfiltered_cellranger_h5_rejects_pointers_past_its_data(
    tmp_path: Path,
) -> None:
    path = tmp_path / "counts.h5"
    _write_h5(path, np.array([[1, 2], [3, 0]], dtype=np.uint16))
    with h5py.File(path, mode="a") as handle:
        indptr = handle["matrix/indptr"]
        indptr[-1] = int(indptr[-1]) + 1

    with pytest.raises(ValueError, match="pointers do not match its data"):
        CrH5Reader(str(path), is_filtered=False, filtering_cutoff=0)
    # The failed reader closed the file, so it opens for writing.
    h5py.File(path, mode="r+").close()


def test_cellranger_h5_rejects_a_negative_filtering_cutoff(tmp_path: Path) -> None:
    path = tmp_path / "counts.h5"
    _write_h5(path, np.array([[1, 2], [0, 0]], dtype=np.uint16))

    # MtxReader rejected it, while this reader kept every barcode.
    with pytest.raises(ValueError, match="filtering_cutoff cannot be negative"):
        CrH5Reader(str(path), is_filtered=False, filtering_cutoff=-1)


def test_cellranger_h5_feature_tags_skip_reserved_absent_and_group_keys(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tagged.h5"
    _write_h5(path, np.array([[1, 0], [0, 2]], dtype=np.uint8))
    with h5py.File(path, mode="a") as handle:
        features = handle["matrix/features"]
        features.create_dataset(
            "_all_tag_keys", data=np.array([b"id", b"absent", b"nested", b"genome"])
        )
        features.create_group("nested")
        features.create_dataset("genome", data=np.array([b"GRCh38", b"GRCh38"]))
    reader = CrH5Reader(str(path))
    try:
        columns = dict(reader.get_feature_columns())
    finally:
        reader.close()

    assert list(columns) == ["feature_type", "genome"]
    np.testing.assert_array_equal(columns["genome"].astype(str), ["GRCh38"] * 2)


def test_cellranger_h5_rejects_a_feature_tag_of_another_length(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tagged.h5"
    _write_h5(path, np.array([[1, 0], [0, 2]], dtype=np.uint8))
    with h5py.File(path, mode="a") as handle:
        features = handle["matrix/features"]
        features.create_dataset("_all_tag_keys", data=np.array([b"genome"]))
        features.create_dataset("genome", data=np.array([b"GRCh38"] * 3))
    reader = CrH5Reader(str(path))
    try:
        with pytest.raises(ValueError, match="'genome' has shape \\(3,\\)"):
            list(reader.get_feature_columns())
    finally:
        reader.close()


def test_cellranger2_h5_reports_only_feature_types(tmp_path: Path) -> None:
    path = tmp_path / "legacy.h5"
    _write_h5(path, np.array([[1, 0], [0, 2]], dtype=np.uint8), legacy=True)
    reader = CrH5Reader(str(path))
    try:
        columns = list(reader.get_feature_columns())
        assert reader.feature_names() == ["gene-0", "gene-1"]
    finally:
        reader.close()

    assert [name for name, _values in columns] == ["feature_type"]
    np.testing.assert_array_equal(columns[0][1], ["Gene Expression"] * 2)


def test_cellranger_h5_yields_empty_batches_for_barcodes_without_entries(
    tmp_path: Path,
) -> None:
    values = np.array([[1, 0], [0, 0], [0, 0], [0, 3]], dtype=np.uint16)
    path = tmp_path / "counts.h5"
    _write_h5(path, values)
    reader = CrH5Reader(str(path))
    try:
        single = list(reader.consume(1))
        paired = list(reader.consume(2))
    finally:
        reader.close()

    assert [batch.nnz for batch in single] == [1, 0, 0, 1]
    assert all(isinstance(batch, coo_matrix) for batch in single)
    assert all(batch.dtype == np.uint16 for batch in single + paired)
    np.testing.assert_array_equal(
        np.vstack([batch.toarray() for batch in single]), values
    )
    np.testing.assert_array_equal(
        np.vstack([batch.toarray() for batch in paired]), values
    )


@pytest.mark.parametrize(
    ("data_dtype", "values", "read_dtype", "stored_dtype"),
    [
        ("<f2", [[1, 0], [0, 2]], np.float32, np.uint8),
        ("<f2", [[1.5, 0], [0, 2]], np.float32, np.float32),
        (">i4", [[1, 0], [0, 300]], np.int32, np.uint16),
        (">f8", [[1.5, 0], [0, 2]], np.float64, np.float64),
    ],
    ids=["float16-integral", "float16-fraction", "big-endian-int", "big-endian-float"],
)
def test_cellranger_h5_reads_float16_and_big_endian_counts_natively(
    tmp_path: Path,
    data_dtype: str,
    values: list[list[float]],
    read_dtype: type,
    stored_dtype: type,
) -> None:
    path = tmp_path / "counts.h5"
    _write_h5(path, np.array(values, dtype=np.float64), data_dtype=data_dtype)
    reader = CrH5Reader(str(path))
    store = MemoryStore()
    try:
        # SciPy sparse matrices hold neither dtype, so these imports used to
        # fail while writing, after the destination was created.
        assert reader.matrix_dtype == read_dtype
        CrToZarr(reader, store, mem_budget="64M", nthreads=1, policy=_POLICY).dump()
    finally:
        reader.close()

    counts = zarr.open_group(store=store, mode="r")["RNA/counts"]
    assert counts.dtype == stored_dtype
    np.testing.assert_array_equal(counts[:], values)


@pytest.mark.parametrize(
    ("text", "kwargs"),
    [
        ("id\nc1\nc2\n", {"id_column": 0}),
        ("id,batch\nc1,x\nc2,y\n", {"id_column": 0, "cell_data_cols": ["batch"]}),
        ("a,b\n1,2\n3,4\n", {"skip_cols": ["a", "b"]}),
    ],
    ids=["id-only", "metadata-only", "skipped-only"],
)
def test_csv_without_count_columns_is_rejected(
    tmp_path: Path, text: str, kwargs: dict[str, object]
) -> None:
    path = tmp_path / "counts.csv"
    path.write_text(text)

    # Such a file used to import an assay without features, which no
    # DataStore could open.
    with pytest.raises(ValueError, match="no count columns"):
        CSVReader(str(path), **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [{"skip_cols": ["a"]}, {"cell_data_cols": ["a"]}],
    ids=["skip_cols", "cell_data_cols"],
)
def test_csv_named_columns_require_a_header(
    tmp_path: Path, kwargs: dict[str, list[str]]
) -> None:
    path = tmp_path / "counts.csv"
    path.write_text("1,2\n3,4\n")

    with pytest.raises(ValueError, match="Named columns require a CSV header"):
        CSVReader(str(path), has_header=False, **kwargs)  # type: ignore[arg-type]


def test_csv_without_a_header_names_cells_and_features_by_position(
    tmp_path: Path,
) -> None:
    path = tmp_path / "counts.csv"
    path.write_text("1,0,2\n0,3,0\n")
    reader = CSVReader(str(path), has_header=False)

    assert reader.feature_ids().tolist() == ["feature_0", "feature_1", "feature_2"]
    assert reader.cell_ids().tolist() == ["cell_0", "cell_1"]
    store = MemoryStore()
    CSVtoZarr(reader, store, assay_name="RNA", nthreads=1, policy=_POLICY).dump()

    root = zarr.open_group(store=store, mode="r")
    np.testing.assert_array_equal(root["RNA/counts"][:], [[1, 0, 2], [0, 3, 0]])
    np.testing.assert_array_equal(
        root["RNA/featureData/ids"][:], ["feature_0", "feature_1", "feature_2"]
    )


def test_csv_fraction_in_an_early_chunk_keeps_float_storage(tmp_path: Path) -> None:
    path = tmp_path / "counts.csv"
    path.write_text("a,b\n0.5,1\n2,3\n4,5\n")
    reader = CSVReader(str(path), batch_size=1)

    assert reader.countRange.integral is False
    assert reader.countDtype == np.float64
    store = MemoryStore()
    CSVtoZarr(reader, store, assay_name="RNA", nthreads=1, policy=_POLICY).dump()

    counts = zarr.open_group(store=store, mode="r")["RNA/counts"]
    assert counts.dtype == np.float64
    np.testing.assert_array_equal(counts[:], [[0.5, 1], [2, 3], [4, 5]])


def test_csv_reader_leaves_the_callers_pandas_kwargs_unchanged(
    tmp_path: Path,
) -> None:
    path = tmp_path / "counts.csv"
    path.write_text("a, b\n1, 2\n")
    pandas_kwargs = {"skipinitialspace": True}

    reader = CSVReader(str(path), batch_size=5, pandas_kwargs=pandas_kwargs)

    # The reader used to add its read_csv settings, such as chunksize, here.
    assert pandas_kwargs == {"skipinitialspace": True}
    assert reader.pandas_kwargs["chunksize"] == 5
    assert reader.feature_ids().tolist() == ["a", "b"]


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ("1,2,x\n4,5,6,7\n", "line 3 has 4 fields, but line 1 has 3"),
        ("1,2,x\n4,5,6,\n", "line 3 has 4 fields, but line 1 has 3"),
        ("1,2,x\n4,5\n", "line 3 has 2 fields, but line 1 has 3"),
        ("c1,1,2,x\nc2,3,4,y\n", "line 2 has 4 fields, but line 1 has 3"),
    ],
    ids=["extra-field", "extra-empty-field", "missing-field", "unnamed-row-names"],
)
def test_csv_rows_without_the_header_field_count_are_rejected(
    tmp_path: Path, rows: str, message: str
) -> None:
    path = tmp_path / "counts.csv"
    path.write_text(f"a,b,note\n{rows}")

    # pandas dropped the extra fields of a row that starts a chunk, padded a
    # row without its skipped field, and took row names that the header does
    # not name as its index, so each of these files imported without an error.
    with pytest.raises(ValueError, match=message):
        CSVReader(str(path), skip_cols=["note"], batch_size=1)


@pytest.mark.parametrize(
    ("name", "text", "options"),
    [
        ("counts.csv", 'a,note,b\n1,"x,y",2\n3,"two\nlines",4\n', {}),
        (
            "counts.csv",
            "a,note,b\n1,'x,y',2\n3,z,4\n",
            {"pandas_kwargs": {"quotechar": "'"}},
        ),
        (
            "counts.csv",
            'a,note,b\n1,"x,2\n3,y,4\n',
            {"pandas_kwargs": {"quoting": csv.QUOTE_NONE}},
        ),
        (
            "counts.csv",
            "a,note,b\n1,x\\,y,2\n3,z,4\n",
            {"pandas_kwargs": {"escapechar": "\\"}},
        ),
        (
            "counts.csv",
            'a,note,b\n1, "x,y",2\n3,z,4\n',
            {"pandas_kwargs": {"skipinitialspace": True}},
        ),
        ("counts.csv", '\ufeff"a,1",note,b\n1,x,2\n3,y,4\n', {}),
        ("counts.csv", "a,note,b\n1,x,2\n\n \t \n3,y,4\n", {}),
        ("counts.csv", '"skipped\nrow"\na,note,b\n1,x,2\n3,y,4\n', {"skip_rows": 1}),
        ("counts.csv.gz", "a,note,b\n1,x,2\n3,y,4\n", {}),
    ],
    ids=[
        "quoted",
        "quotechar",
        "quote-none",
        "escapechar",
        "skipinitialspace",
        "byte-order-mark",
        "blank-lines",
        "skip-rows",
        "gzip",
    ],
)
def test_csv_field_check_splits_rows_as_pandas_does(
    tmp_path: Path, name: str, text: str, options: dict[str, object]
) -> None:
    path = tmp_path / name
    data = text.encode()
    path.write_bytes(gzip.compress(data) if name.endswith(".gz") else data)

    # A field check that split rows otherwise than pandas, or that read other
    # text than pandas, would reject each of these files.
    reader = CSVReader(str(path), skip_cols=["note"], batch_size=1, **options)  # type: ignore[arg-type]

    counts = np.vstack([batch for batch, _metadata in reader.consume()])
    np.testing.assert_array_equal(counts, [[1, 2], [3, 4]])


@pytest.mark.parametrize(
    ("text", "options", "message"),
    [
        ("a\tb\n1\t2\n", {"sep": r"\s+"}, "sep must be one character"),
        (
            "a\tb\n1\t2\n",
            {"pandas_kwargs": {"dialect": "excel-tab"}},
            "pandas_kwargs cannot set dialect:",
        ),
        (
            "a,b\n1,2\n",
            {"pandas_kwargs": {"lineterminator": "\n", "comment": "#"}},
            "pandas_kwargs cannot set comment, lineterminator:",
        ),
    ],
    ids=["regex-separator", "dialect", "comment-and-lineterminator"],
)
def test_csv_reader_rejects_settings_that_its_field_check_cannot_follow(
    tmp_path: Path, text: str, options: dict[str, object], message: str
) -> None:
    path = tmp_path / "counts.csv"
    path.write_text(text)

    # pandas reads each file under its setting, but the field check cannot
    # split rows as the setting does.
    with pytest.raises(ValueError, match=message):
        CSVReader(str(path), **options)  # type: ignore[arg-type]


def test_matrix_names_without_the_matrix_suffix_use_plain_sidecars(
    tmp_path: Path,
) -> None:
    _write_mex(
        tmp_path, "1 1 3\n2 2 4\n", n_features=2, n_cells=2, matrix_name="counts.mtx"
    )

    candidates = inspect_mtx(tmp_path)

    assert [Path(candidate.matrixPath).name for candidate in candidates] == [
        "counts.mtx"
    ]
    assert Path(candidates[0].featurePath).name == "features.tsv"
    assert Path(candidates[0].cellPath).name == "barcodes.tsv"


def test_parse_matrix_without_its_sidecars_is_not_a_candidate(tmp_path: Path) -> None:
    _write_mex(tmp_path, "1 1 3\n", n_features=1, n_cells=1)
    # A Parse matrix needs Parse gene and cell tables, not 10x sidecars.
    (tmp_path / "count_matrix.mtx").write_text(f"{_BANNER}1 1 1\n1 1 3\n")

    candidates = inspect_mtx(tmp_path)

    assert [Path(candidate.matrixPath).name for candidate in candidates] == [
        "matrix.mtx"
    ]


def test_parse_matrix_stored_features_by_cells_is_read_transposed(
    tmp_path: Path,
) -> None:
    _write_parse(tmp_path, f"{_BANNER}3 2 3\n1 1 4\n3 1 1\n2 2 7\n")
    candidate = inspect_mtx(tmp_path)[0]
    assert candidate.matrixOrientation == "featuresByCells"
    assert (candidate.nCells, candidate.nFeatures) == (2, 3)

    with pytest.raises(KeyError, match="'barcode' was not found"):
        MtxReader(candidate, cell_id_key="barcode")
    reader = MtxReader(candidate)
    store = MemoryStore()
    try:
        MtxToZarr(reader, store, mem_budget="64M", nthreads=1, policy=_POLICY).dump()
        assert reader.cell_names() == ["cell-0", "cell-1"]
    finally:
        reader.close()

    np.testing.assert_array_equal(_counts(store), [[4, 0, 1], [0, 7, 0]])


def test_missing_matrix_market_sources_are_rejected(tmp_path: Path) -> None:
    _write_mex(tmp_path, "1 1 3\n", n_features=1, n_cells=1)

    with pytest.raises(FileNotFoundError):
        inspect_mtx(tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="missing.tsv"):
        MtxReader(
            str(tmp_path / "matrix.mtx"),
            str(tmp_path / "features.tsv"),
            str(tmp_path / "missing.tsv"),
        )


def test_mtx_reader_rejects_invalid_construction_arguments(tmp_path: Path) -> None:
    _write_mex(tmp_path, "1 1 3\n", n_features=1, n_cells=1)
    candidate = inspect_mtx(tmp_path)[0]

    with pytest.raises(ValueError, match="Do not pass explicit sidecars"):
        MtxReader(candidate, feature_path=str(tmp_path / "features.tsv"))
    # The reader extracted this file from a ZIP archive but never read it.
    with pytest.raises(TypeError, match="cell_metadata_path"):
        MtxReader(candidate, cell_metadata_path=str(tmp_path / "barcodes.tsv"))  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="filtering_cutoff cannot be negative"):
        MtxReader(candidate, is_filtered=False, filtering_cutoff=-1)
    with pytest.raises(FileNotFoundError, match="missing"):
        MtxReader(candidate, temp_dir=str(tmp_path / "missing"))


def test_mtx_reader_planning_and_streaming_arguments_must_be_positive(
    tmp_path: Path,
) -> None:
    _write_mex(tmp_path, "1 1 3\n2 2 4\n", n_features=2, n_cells=2)
    reader = MtxReader(inspect_mtx(tmp_path)[0])
    try:
        with pytest.raises(ValueError, match="batch_size must be positive"):
            list(reader.consume(0))
        with pytest.raises(ValueError, match="lines_in_mem must be positive"):
            list(reader.consume(1, lines_in_mem=0))
        with pytest.raises(ValueError, match="batch_size must be positive"):
            reader.producer_staging_bytes(0, 1)
        with pytest.raises(ValueError, match="lines_in_mem must be positive"):
            reader.producer_staging_bytes(1, 0)
        with pytest.raises(ValueError, match="window_rows must be positive"):
            reader.max_window_nnz(0)
        assert reader.max_window_nnz(2) == 2
    finally:
        reader.close()


def _flag_first_member_encrypted(path: Path) -> None:
    content = bytearray(path.read_bytes())
    # The general purpose flags follow the central directory signature,
    # the creator version, and the extractor version.
    position = content.index(b"PK\x01\x02") + 8
    content[position] |= 0x1
    path.write_bytes(bytes(content))


@pytest.mark.parametrize("member", ["encrypted", "fifo", "nested"])
def test_zip_members_that_cannot_be_read_safely_are_rejected(
    tmp_path: Path, member: str
) -> None:
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, mode="w") as archive:
        if member == "fifo":
            info = zipfile.ZipInfo("matrix.mtx")
            info.create_system = 3
            info.external_attr = (stat.S_IFIFO | 0o644) << 16
            archive.writestr(info, b"")
        elif member == "nested":
            archive.writestr("inner.zip", b"")
        else:
            archive.writestr("matrix.mtx", f"{_BANNER}1 1 0\n")
    if member == "encrypted":
        _flag_first_member_encrypted(path)
    message = {
        "encrypted": "Encrypted ZIP archive member: matrix.mtx",
        "fifo": "Unsupported ZIP archive member: matrix.mtx",
        "nested": "cannot contain another ZIP",
    }[member]

    with pytest.raises(ValueError, match=message):
        inspect_mtx(path)


@pytest.mark.parametrize(
    ("sidecar", "content", "message"),
    [
        (
            "features.tsv",
            "feature-0\tgene-0\n",
            "Feature sidecar has 1 rows, expected 2",
        ),
        ("barcodes.tsv", "", "Cell sidecar has 0 rows, expected 2"),
        ("barcodes.tsv", "cell-0\n", "Cell sidecar has 1 rows, expected 2"),
        ("peaks.bed", "chr1\t10\t20\n", "Feature sidecar has 1 rows, expected 2"),
    ],
)
def test_sidecars_that_changed_after_inspection_are_rejected(
    tmp_path: Path, sidecar: str, content: str, message: str
) -> None:
    _write_mex(tmp_path, "1 1 3\n2 2 4\n", n_features=2, n_cells=2)
    if sidecar == "peaks.bed":
        (tmp_path / "features.tsv").unlink()
        (tmp_path / "peaks.bed").write_text("chr1\t10\t20\nchr1\t30\t40\n")
    candidate = inspect_mtx(tmp_path)[0]
    feature_sidecar = "peaks.bed" if sidecar == "peaks.bed" else "features.tsv"
    assert Path(candidate.featurePath).name == feature_sidecar
    (tmp_path / sidecar).write_text(content)

    with pytest.raises(ValueError, match=message):
        MtxReader(candidate)


def test_matrix_declaring_entries_that_it_lacks_is_rejected(tmp_path: Path) -> None:
    _write_mex(tmp_path, "", n_features=2, n_cells=2, n_entries=2)

    with pytest.raises(ValueError, match="declares 2 entries, but 0 were read"):
        MtxReader(inspect_mtx(tmp_path)[0])


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ("", "declares 2 entries, but 0 were read"),
        ("2 2 4\n1 1 3\n", "cellMajor order is invalid at entry 2"),
    ],
    ids=["entries-removed", "entries-reordered"],
)
def test_matrix_changed_after_the_scan_is_rejected_while_streaming(
    tmp_path: Path, entries: str, message: str
) -> None:
    _write_mex(tmp_path, "1 1 3\n2 2 4\n", n_features=2, n_cells=2)
    reader = MtxReader(inspect_mtx(tmp_path)[0])
    try:
        assert reader.coordinateOrder == "cellMajor"
        (tmp_path / "matrix.mtx").write_text(f"{_BANNER}2 2 2\n{entries}")
        with pytest.raises(ValueError, match=message):
            list(reader.consume(1))
    finally:
        reader.close()


def test_feature_major_matrix_whose_cell_entries_changed_is_rejected(
    tmp_path: Path,
) -> None:
    # Feature-major entries: cell 0 holds one entry and cell 1 holds two.
    _write_mex(tmp_path, "1 2 1\n2 1 1\n2 2 1\n", n_features=2, n_cells=2)
    reader = MtxReader(inspect_mtx(tmp_path)[0], temp_dir=str(tmp_path))
    try:
        assert reader.coordinateOrder == "featureMajor"
        # Still feature-major with as many entries, but cell 0 now holds two.
        (tmp_path / "matrix.mtx").write_text(f"{_BANNER}2 2 3\n1 1 1\n2 1 1\n2 2 1\n")
        with pytest.raises(RuntimeError, match="inconsistent row pointers"):
            list(reader.consume(1))
    finally:
        reader.close()
    assert not list(tmp_path.glob("scarf-mtx-csr-*"))


def test_feature_major_matrix_without_kept_cells_streams_nothing(
    tmp_path: Path,
) -> None:
    _write_mex(tmp_path, "1 2 1\n2 1 1\n", n_features=2, n_cells=2)
    reader = MtxReader(
        inspect_mtx(tmp_path)[0],
        is_filtered=False,
        filtering_cutoff=5,
        temp_dir=str(tmp_path),
    )
    try:
        assert reader.coordinateOrder == "featureMajor"
        assert reader.nCells == 0
        assert list(reader.consume(1)) == []
    finally:
        reader.close()
    assert not list(tmp_path.glob("scarf-mtx-csr-*"))


def test_feature_major_import_without_free_temporary_disk_fails_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_mex(tmp_path, "1 2 1\n2 1 1\n", n_features=2, n_cells=2)
    reader = MtxReader(inspect_mtx(tmp_path)[0], temp_dir=str(tmp_path))
    store = MemoryStore()
    try:
        writer = MtxToZarr(reader, store, mem_budget="64M", nthreads=1, policy=_POLICY)
        monkeypatch.setattr(shutil, "disk_usage", lambda _path: SimpleNamespace(free=0))
        with pytest.raises(OSError, match="temporary bytes, but 0 bytes are free"):
            writer.dump()
        monkeypatch.undo()
        # The failed preparation left nothing behind, so the import can retry.
        writer.dump()
    finally:
        reader.close()
    assert not list(tmp_path.glob("scarf-mtx-csr-*"))
    np.testing.assert_array_equal(_counts(store), [[0, 1], [1, 0]])


def test_zip_matrix_imports_repeatedly_and_removes_each_extraction(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sample.zip"
    with zipfile.ZipFile(path, mode="w") as archive:
        archive.writestr("matrix.mtx", f"{_BANNER}2 2 2\n1 1 3\n2 2 4\n")
        archive.writestr("features.tsv", "feature-0\tgene-0\nfeature-1\tgene-1\n")
        archive.writestr("barcodes.tsv", "cell-0\ncell-1\n")
    reader = MtxReader(inspect_mtx(path)[0], temp_dir=str(tmp_path))
    stores = [MemoryStore(), MemoryStore()]
    try:
        for store in stores:
            MtxToZarr(
                reader, store, mem_budget="64M", nthreads=1, policy=_POLICY
            ).dump()
            assert not list(tmp_path.glob("scarf-mtx-archive-*"))
    finally:
        reader.close()

    for store in stores:
        np.testing.assert_array_equal(_counts(store), [[3, 0], [0, 4]])
