"""The count storage dtype and the count layout of every import writer.

Each writer stores counts in the dtype that ``count_storage_dtype`` resolves
from their canonical (duplicate-summed) values, and admits its count layout
against its budget before it creates the destination: an import names the
layout that fits when the default does not, and subset and repack fit it
themselves. Subset and repack keep the source dtype.
"""

import inspect
import re
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
import zarr
from scipy.sparse import csr_matrix
from zarr.storage import MemoryStore

from scarf import DataStore
from scarf.merge import DataStoreMerge
from scarf.readers import CrH5Reader, CSVReader, H5adReader, MtxReader, inspect_mtx
from scarf.readers.seurat import SeuratReader
from scarf.storage.count_matrix import (
    DEFAULT_COUNT_MATRIX_POLICY,
    CountMatrixPolicy,
    load_count_matrix_plan,
    policy_from_payload,
)
from scarf.tools.repack_zarr import repack_store
from scarf.utils.count_values import CountValueRange
from scarf.writers import (
    CrToZarr,
    CSVtoZarr,
    H5adToZarr,
    MtxToZarr,
    SeuratToZarr,
    SparseToZarr,
    SubsetZarr,
    create_zarr_count_assay,
    subset_assay_zarr,
)
from tests.storage_helpers import finalize_test_counts, write_count_store
from tests.test_seurat_reader import _Wire

# An empty cell, an empty feature, and counts up to 2500.
_VALUES = np.array(
    [
        [0, 3, 0, 0, 2500],
        [7, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [12, 1, 0, 0, 0],
        [0, 0, 0, 300, 5],
        [1, 1, 0, 1, 1],
    ],
    dtype=np.int64,
)


def _cells(values: np.ndarray) -> list[str]:
    return [f"c{index}" for index in range(values.shape[0])]


def _features(values: np.ndarray) -> list[str]:
    return [f"g{index}" for index in range(values.shape[1])]


def _split_duplicates(row: np.ndarray) -> list[tuple[int, Any]]:
    """Return a row's entries with its first value split into two entries."""
    entries = [(int(column), row[column]) for column in np.flatnonzero(row)]
    if not entries:
        return entries
    column, value = entries[0]
    half = value // 2
    return [(column, value - half), *entries[1:], (column, half)]


def _write_h5ad(
    path: Path,
    values: np.ndarray,
    dtype: str,
    feature_types: list[str] | None = None,
) -> Path:
    matrix = csr_matrix(values.astype(np.float64))
    with h5py.File(path, "w") as h5:
        group = h5.create_group("X")
        group.attrs["encoding-type"] = "csr_matrix"
        group.attrs["shape"] = values.shape
        group.create_dataset("data", data=matrix.data.astype(dtype))
        group.create_dataset("indices", data=matrix.indices)
        group.create_dataset("indptr", data=matrix.indptr)
        h5.create_group("obs").create_dataset(
            "_index", data=np.array([name.encode() for name in _cells(values)])
        )
        var = h5.create_group("var")
        names = np.array([name.encode() for name in _features(values)])
        var.create_dataset("_index", data=names)
        var.create_dataset("feature_name", data=names)
        if feature_types is not None:
            var.create_dataset(
                "feature_types",
                data=np.array([name.encode() for name in feature_types]),
            )
        h5.create_group("obsm")
    return path


def _write_cellranger(
    path: Path,
    values: np.ndarray,
    encoding: str,
    feature_types: list[str] | None = None,
) -> Path:
    """Write a 10x HDF5 matrix; ``encoding`` is a data dtype, optionally with
    ``-duplicates`` for split, unsorted coordinates."""
    dtype, _, variant = encoding.partition("-")
    pointers, indices, data = [0], [], []
    for row in values:
        entries = (
            _split_duplicates(row)
            if variant == "duplicates"
            else [(int(column), row[column]) for column in np.flatnonzero(row)]
        )
        indices.extend(column for column, _value in entries)
        data.extend(value for _column, value in entries)
        pointers.append(len(indices))
    with h5py.File(path, "w") as h5:
        group = h5.create_group("matrix")
        group.create_dataset(
            "data", data=np.asarray(data, dtype=np.float64).astype(dtype)
        )
        group.create_dataset("indices", data=np.asarray(indices, dtype=np.int64))
        group.create_dataset("indptr", data=np.asarray(pointers, dtype=np.int64))
        group.create_dataset(
            "barcodes", data=np.array([name.encode() for name in _cells(values)])
        )
        features = group.create_group("features")
        names = np.array([name.encode() for name in _features(values)])
        features.create_dataset("id", data=names)
        features.create_dataset("name", data=names)
        features.create_dataset(
            "feature_type",
            data=np.array(
                [
                    name.encode()
                    for name in feature_types or ["Gene Expression"] * len(names)
                ]
            ),
        )
    return path


def _write_mtx(
    directory: Path,
    values: np.ndarray,
    encoding: str,
    feature_types: list[str] | None = None,
) -> Path:
    """Write a Matrix Market triplet; ``encoding`` is ``integer``, ``real``,
    ``featureMajor``, or ``duplicates``."""
    directory.mkdir()
    entries = [
        (int(cell), column, value)
        for cell, row in enumerate(values)
        for column, value in (
            _split_duplicates(row)
            if encoding == "duplicates"
            else [(int(column), row[column]) for column in np.flatnonzero(row)]
        )
    ]
    if encoding == "duplicates":
        # A cell-major file keeps duplicate coordinates next to each other.
        entries.sort(key=lambda entry: (entry[0], entry[1]))
    if encoding == "featureMajor":
        entries.sort(key=lambda entry: (entry[1], entry[0]))
    field = "real" if encoding == "real" else "integer"
    lines = [
        f"%%MatrixMarket matrix coordinate {field} general",
        f"{values.shape[1]} {values.shape[0]} {len(entries)}",
        *(
            f"{column + 1} {cell + 1} "
            + (f"{float(value):.1f}" if field == "real" else str(int(value)))
            for cell, column, value in entries
        ),
    ]
    (directory / "matrix.mtx").write_text("\n".join(lines) + "\n")
    types = feature_types or ["Gene Expression"] * values.shape[1]
    (directory / "features.tsv").write_text(
        "".join(
            f"{name}\t{name}\t{kind}\n"
            for name, kind in zip(_features(values), types, strict=True)
        )
    )
    (directory / "barcodes.tsv").write_text(
        "".join(f"{name}\n" for name in _cells(values))
    )
    return directory


def _write_csv(path: Path, values: np.ndarray, encoding: str) -> Path:
    """Write counts as CSV text; ``encoding`` is ``integer`` or ``real``."""
    rows = (
        ",".join(f"{value:.1f}" if encoding == "real" else str(value) for value in row)
        for row in values.tolist()
    )
    path.write_text(",".join(_features(values)) + "\n" + "\n".join(rows) + "\n")
    return path


def _sparse(values: np.ndarray, encoding: str) -> csr_matrix:
    """Return a CSR matrix; ``encoding`` is a dtype, optionally with
    ``-duplicates`` for split, unsorted coordinates."""
    dtype, _, variant = encoding.partition("-")
    if variant != "duplicates":
        return csr_matrix(values.astype(dtype))
    pointers, indices, data = [0], [], []
    for row in values:
        entries = _split_duplicates(row)
        indices.extend(column for column, _value in entries)
        data.extend(value for _column, value in entries)
        pointers.append(len(indices))
    return csr_matrix(
        (np.asarray(data, dtype=dtype), indices, pointers), shape=values.shape
    )


def _write_seurat(path: Path, values: np.ndarray, encoding: str) -> Path:
    """Write a Seurat object with dense counts; ``encoding`` is ``real`` (R
    double) or ``integer`` (R integer)."""
    wire = _Wire()
    cells, features = _cells(values), _features(values)
    # R matrices are column-major, so the features-by-cells counts are the
    # row-major cells-by-features values.
    counts = wire.matrix(
        values.reshape(-1).tolist(),
        (len(features), len(cells)),
        rows=features,
        columns=cells,
        real=encoding == "real",
    )
    assay = wire.s4(
        [
            ("counts", counts),
            (
                "meta.features",
                wire.data_frame([("symbol", wire.string_vector(features))], features),
            ),
            ("class", wire.string_vector(["Assay"])),
        ]
    )
    root = wire.s4(
        [
            ("assays", wire.vector([assay], names=["RNA"])),
            (
                "meta.data",
                wire.data_frame(
                    [("group", wire.string_vector(["a"] * len(cells)))], cells
                ),
            ),
            ("active.assay", wire.string_vector(["RNA"])),
            ("active.ident", wire.factor([1] * len(cells), ["cells"], names=cells)),
            ("reductions", wire.vector([], names=[])),
            ("class", wire.string_vector(["Seurat"])),
        ]
    )
    path.write_bytes(wire.document(root))
    return path


def _source(writer: str, directory: Path, values: np.ndarray, encoding: str) -> Any:
    """Write ``values`` in the source format of ``writer`` and open its reader."""
    if writer == "h5ad":
        return H5adReader(
            str(_write_h5ad(directory / "counts.h5ad", values, encoding)),
            feature_name_key="feature_name",
        )
    if writer == "cellranger":
        return CrH5Reader(
            str(_write_cellranger(directory / "counts.h5", values, encoding))
        )
    if writer == "mtx":
        return MtxReader(
            inspect_mtx(_write_mtx(directory / "mtx", values, encoding))[0]
        )
    if writer == "csv":
        return CSVReader(
            str(_write_csv(directory / "counts.csv", values, encoding)), batch_size=250
        )
    if writer == "sparse":
        return _sparse(values, encoding)
    return SeuratReader(
        _write_seurat(directory / "counts.rds", values, encoding), reductions=[]
    )


def _writer(writer: str, source: Any, destination: str, **options: Any) -> Any:
    if writer == "h5ad":
        return H5adToZarr(source, destination, **options)
    if writer in {"cellranger", "mtx"}:
        return CrToZarr(source, destination, **options)
    if writer == "csv":
        return CSVtoZarr(source, destination, assay_name="RNA", **options)
    if writer == "sparse":
        return SparseToZarr(
            source,
            destination,
            cell_ids=[f"c{index}" for index in range(source.shape[0])],
            feature_ids=[f"g{index}" for index in range(source.shape[1])],
            **options,
        )
    return SeuratToZarr(source, destination, **options)


def _close(source: Any) -> None:
    close = getattr(source, "close", None)
    if callable(close):
        close()


def _construct(
    writer: str, directory: Path, values: np.ndarray, encoding: str, **options: Any
) -> Any:
    """Construct ``writer`` for ``values`` with a destination in ``directory``.

    Returns a callable that dumps the import and closes its reader.
    """
    source = _source(writer, directory, values, encoding)
    try:
        instance = _writer(
            writer,
            source,
            str(directory / "counts.zarr"),
            **{"nthreads": 1, **options},
        )
    except BaseException:
        _close(source)
        raise

    def dump() -> None:
        try:
            instance.dump()
        finally:
            _close(source)

    return dump


def _import(
    writer: str, directory: Path, values: np.ndarray, encoding: str, **options: Any
) -> zarr.Group:
    directory.mkdir(parents=True, exist_ok=True)
    _construct(writer, directory, values, encoding, **options)()
    return zarr.open_group(str(directory / "counts.zarr"), mode="r")


def _planned_policy(
    writer: str, directory: Path, values: np.ndarray, encoding: str, **options: Any
) -> CountMatrixPolicy:
    """Return the layout an import writer records when it creates its counts."""
    directory.mkdir(parents=True, exist_ok=True)
    source = _source(writer, directory, values, encoding)
    try:
        _writer(writer, source, str(directory / "counts.zarr"), nthreads=1, **options)
    finally:
        _close(source)
    return _policy(zarr.open_group(str(directory / "counts.zarr"), mode="r"))


def _fingerprint(values: np.ndarray, dtype: Any) -> str:
    """Return the content fingerprint of ``values`` stored in ``dtype``."""
    root = zarr.open_group(store=MemoryStore(), mode="w")
    return finalize_test_counts(root.create_array("counts", data=values.astype(dtype)))


def _policy(root: zarr.Group) -> CountMatrixPolicy:
    return policy_from_payload(load_count_matrix_plan(root["RNA/counts"]))


def _named_policy(error: BaseException) -> CountMatrixPolicy:
    """Return the policy that a refused import names in its error."""
    ((unit, chunk),) = re.findall(
        r"policy=CountMatrixPolicy\(unitBytes=(\d+), chunkBytes=(\d+)\)", str(error)
    )
    return CountMatrixPolicy(unitBytes=int(unit), chunkBytes=int(chunk))


_ENCODINGS = [
    ("h5ad", "float32"),
    ("cellranger", "int32"),
    ("cellranger", "float32"),
    ("cellranger", "float64"),
    ("cellranger", "int64"),
    ("cellranger", "uint32"),
    ("cellranger", "int32-duplicates"),
    ("mtx", "integer"),
    ("mtx", "real"),
    ("mtx", "featureMajor"),
    ("mtx", "duplicates"),
    ("csv", "integer"),
    ("csv", "real"),
    ("sparse", "float32"),
    ("sparse", "float64"),
    ("sparse", "int32"),
    ("sparse", "int64"),
    ("sparse", "uint32"),
    ("sparse", "float32-duplicates"),
    ("seurat", "real"),
    ("seurat", "integer"),
]


@pytest.mark.parametrize(("writer", "encoding"), _ENCODINGS)
@pytest.mark.parametrize(
    ("maximum", "expected"), [(2500, np.uint16), (70_000, np.uint32)]
)
def test_every_writer_stores_the_same_counts_identically(
    tmp_path, writer, encoding, maximum, expected
):
    values = _VALUES.copy()
    values[0, 4] = maximum
    root = _import(writer, tmp_path, values, encoding)
    counts = root["RNA/counts"]
    assert counts.dtype == np.dtype(expected)
    np.testing.assert_array_equal(counts[:], values)
    np.testing.assert_array_equal(root["RNA/countsT"][:], values.T)
    # The counts fingerprint depends on the canonical values alone.
    assert counts.attrs["content_fingerprint"] == _fingerprint(values, expected)


def test_cellranger_h5_counts_range_over_the_selected_barcodes(tmp_path, monkeypatch):
    from scarf.readers import cellranger as cellranger_module

    # Barcodes 1 and 3 hold at most the cutoff of 300 and are not kept; the
    # 300 of barcode 1 would need uint16.
    values = np.array(
        [[200, 101, 0], [0, 300, 0], [250, 60, 0], [0, 0, 0], [150, 0, 160]],
        dtype=np.int64,
    )
    path = _write_cellranger(tmp_path / "counts.h5", values, "int32")
    scans: list[np.ndarray] = []
    scan = cellranger_module.compressed_count_ranges

    def recorded(indptr, *args, **kwargs):
        scans.append(np.asarray(indptr))
        return scan(indptr, *args, **kwargs)

    monkeypatch.setattr(cellranger_module, "compressed_count_ranges", recorded)
    reader = CrH5Reader(str(path), is_filtered=False, filtering_cutoff=300)
    try:
        np.testing.assert_array_equal(reader.validBarcodeIdx, [0, 2, 4])
        assert reader.count_value_ranges(1 << 20) == [CountValueRange(maximum=250)]
        # Barcode 1 holds values between kept barcodes, so the scan skips it;
        # the empty barcode 3 lets barcodes 2 and 4 be scanned together.
        assert [list(pointers) for pointers in scans] == [[0, 2], [3, 5, 5, 7]]
        CrToZarr(reader, str(tmp_path / "counts.zarr"), nthreads=1).dump()
    finally:
        reader.close()
    counts = zarr.open_group(str(tmp_path / "counts.zarr"), mode="r")["RNA/counts"]
    assert counts.dtype == np.uint8
    np.testing.assert_array_equal(counts[:], values[[0, 2, 4]])


@pytest.mark.parametrize(
    ("writer", "encoding", "value", "expected"),
    [
        ("cellranger", "float32", 1.5, np.float32),
        ("cellranger", "int32", -3, np.int32),
        ("csv", "real", 1.5, np.float64),
        ("sparse", "float32", 1.5, np.float32),
        ("sparse", "float64", -0.5, np.float64),
        ("sparse", "int32", -3, np.int32),
        ("seurat", "real", 1.5, np.float64),
        ("seurat", "integer", -3, np.int32),
    ],
)
def test_counts_that_are_not_non_negative_integers_keep_their_dtype(
    tmp_path, writer, encoding, value, expected
):
    values = _VALUES.astype(np.float64)
    values[3, 2] = value
    counts = _import(writer, tmp_path, values, encoding)["RNA/counts"]
    assert counts.dtype == np.dtype(expected)
    np.testing.assert_array_equal(counts[:], values.astype(expected))


@pytest.mark.parametrize(
    ("writer", "encoding"),
    [
        ("cellranger", "float32"),
        ("mtx", "real"),
        ("csv", "real"),
        ("sparse", "float32"),
        ("seurat", "real"),
    ],
)
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_non_finite_counts_fail_before_the_destination_exists(
    tmp_path, writer, encoding, value
):
    values = _VALUES.astype(np.float64)
    values[1, 3] = value
    # Readers that scan their counts at construction reject them there.
    with pytest.raises(ValueError, match="finite|missing"):
        _construct(writer, tmp_path, values, encoding)
    assert not (tmp_path / "counts.zarr").exists()


@pytest.mark.parametrize(
    ("writer", "encoding"),
    [("h5ad", "float32"), ("cellranger", "float32"), ("sparse", "float32")],
)
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_non_finite_counts_after_a_fraction_fail_before_the_destination_exists(
    tmp_path, monkeypatch, writer, encoding, value
):
    from scarf.utils import count_values

    # Windows of four values put the fraction and the NaN or infinity in
    # different windows, as a matrix of more than 2**20 values does.
    monkeypatch.setattr(count_values, "_SCAN_WINDOW_VALUES", 4)
    values = np.tile(_VALUES.astype(np.float64), (8, 1))
    values[0, 1] = 0.5
    values[-1, -1] = value
    with pytest.raises(ValueError, match="finite values"):
        _construct(writer, tmp_path, values, encoding)
    assert not (tmp_path / "counts.zarr").exists()


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_seurat_non_finite_counts_after_a_fraction_fail_before_the_destination_exists(
    tmp_path, value
):
    values = _large_values(600, 200).astype(np.float64)
    values[0, 0] = 0.5
    values[-1, 7] = value
    # This budget reads the counts in several blocks.
    with pytest.raises(ValueError, match="finite|missing"):
        _construct("seurat", tmp_path, values, "real", mem_budget=2 * 1024**2)
    assert not (tmp_path / "counts.zarr").exists()


def test_cellranger_h5_scans_every_kept_barcode_group_for_non_finite_counts(
    tmp_path,
):
    # Odd barcodes hold no more than the cutoff and are dropped, so the kept
    # barcodes are scanned in separate groups.
    values = np.tile(np.array([[400.0, 1.0], [1.0, 0.0]]), (20, 1))
    values[0, 1] = 0.5
    values[38, 0] = np.inf
    path = _write_cellranger(tmp_path / "counts.h5", values, "float32")
    reader = CrH5Reader(str(path), is_filtered=False, filtering_cutoff=300)
    try:
        assert reader.nCells == 20
        with pytest.raises(ValueError, match="finite values"):
            CrToZarr(reader, str(tmp_path / "counts.zarr"), nthreads=1)
    finally:
        reader.close()
    assert not (tmp_path / "counts.zarr").exists()


def test_cellranger_h5_import_that_keeps_no_barcode_stores_uint8(tmp_path):
    values = np.array([[1, 2, 0], [0, 3, 1]])
    path = _write_cellranger(tmp_path / "counts.h5", values, "int32")
    reader = CrH5Reader(str(path), is_filtered=False, filtering_cutoff=100)
    try:
        assert reader.count_value_ranges(1 << 20) == [CountValueRange()]
        CrToZarr(reader, str(tmp_path / "counts.zarr"), nthreads=1).dump()
    finally:
        reader.close()
    counts = zarr.open_group(str(tmp_path / "counts.zarr"), mode="r")["RNA/counts"]
    assert (counts.shape, counts.dtype) == ((0, 3), np.uint8)


# Six cells of four RNA features with counts up to 300 and three ADT features
# with counts up to 5.
_CITE_TYPES = ["Gene Expression"] * 4 + ["Antibody Capture"] * 3
_CITE_VALUES = np.array(
    [
        [300, 1, 0, 2, 5, 0, 1],
        [0, 4, 9, 0, 0, 2, 0],
        [7, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 4],
        [0, 0, 6, 0, 3, 0, 0],
        [2, 0, 0, 3, 0, 5, 2],
    ],
    dtype=np.int64,
)


def _import_split(
    writer: str, directory: Path, values: np.ndarray, encoding: str, **options: Any
) -> zarr.Group:
    """Import RNA and ADT features of one source matrix as two assays."""
    destination = str(directory / "counts.zarr")
    if writer == "h5ad":
        source: Any = H5adReader(
            str(_write_h5ad(directory / "counts.h5ad", values, encoding, _CITE_TYPES)),
            feature_name_key="feature_name",
        )
        create = lambda: H5adToZarr(  # noqa: E731
            source, destination, assay_split_key="feature_types", nthreads=1
        )
    else:
        source = (
            CrH5Reader(
                str(
                    _write_cellranger(
                        directory / "counts.h5", values, encoding, _CITE_TYPES
                    )
                )
            )
            if writer == "cellranger"
            else MtxReader(
                inspect_mtx(
                    _write_mtx(directory / "mtx", values, encoding, _CITE_TYPES)
                )[0],
                **options,
            )
        )
        create = lambda: CrToZarr(source, destination, nthreads=1)  # noqa: E731
    try:
        create().dump()
    finally:
        source.close()
    return zarr.open_group(destination, mode="r")


@pytest.mark.parametrize(
    ("writer", "encoding"),
    [("cellranger", "int32"), ("mtx", "integer"), ("h5ad", "float32")],
)
def test_split_assays_store_the_dtype_of_their_own_counts(tmp_path, writer, encoding):
    root = _import_split(writer, tmp_path, _CITE_VALUES, encoding)
    for name, values, dtype in (
        ("RNA", _CITE_VALUES[:, :4], np.uint16),
        ("ADT", _CITE_VALUES[:, 4:], np.uint8),
    ):
        counts = root[f"{name}/counts"]
        assert counts.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(counts[:], values)
        # Each assay has the identity of the same counts imported on their own.
        assert counts.attrs["content_fingerprint"] == _fingerprint(values, dtype)


@pytest.mark.parametrize("writer", ["cellranger", "h5ad"])
def test_a_fractional_assay_keeps_its_dtype_without_widening_the_others(
    tmp_path, writer
):
    values = _CITE_VALUES.astype(np.float64)
    values[1, 5] = 0.5
    root = _import_split(writer, tmp_path, values, "float32")
    assert root["RNA/counts"].dtype == np.uint16
    assert root["RNA/countsT"].dtype == np.uint16
    assert root["ADT/counts"].dtype == np.float32
    np.testing.assert_array_equal(root["ADT/counts"][:], values[:, 4:])


def test_matrix_market_split_assays_range_over_the_kept_cells(tmp_path):
    values = _CITE_VALUES.copy()
    values[:, 0] += 400
    # The last cell holds the largest ADT count, but its total of 300 does not
    # exceed the cutoff, so it is dropped and ADT stores as uint8.
    dropped = np.zeros((1, values.shape[1]), dtype=np.int64)
    dropped[0, 5] = 300
    source = np.vstack([values, dropped])
    root = _import_split(
        "mtx", tmp_path, source, "integer", is_filtered=False, filtering_cutoff=300
    )
    assert root["RNA/counts"].dtype == np.uint16
    assert root["ADT/counts"].dtype == np.uint8
    np.testing.assert_array_equal(root["ADT/counts"][:], values[:, 4:])
    np.testing.assert_array_equal(root["RNA/counts"][:], values[:, :4])


def _duplicate_sum_csr(duplicates: list[float]) -> csr_matrix:
    """Return float64 counts whose cell 0 holds three duplicates at gene 2."""
    return csr_matrix(
        (
            np.array([duplicates[0], 3.0, duplicates[1], 2.0, duplicates[2], 4.0]),
            np.array([2, 0, 2, 5, 2, 1]),
            np.array([0, 5, 6]),
        ),
        shape=(2, 6),
    )


@pytest.mark.parametrize("writer", ["sparse", "h5ad", "cellranger"])
@pytest.mark.parametrize(
    ("duplicates", "expected"),
    [([0.1, 0.2, 0.7], np.float64), ([0.7, 0.2, 0.1], np.uint8)],
)
def test_float_duplicates_resolve_the_dtype_of_the_sums_the_import_stores(
    tmp_path, writer, duplicates, expected
):
    from scarf.utils.arrays import canonicalize_sparse

    matrix = _duplicate_sum_csr(duplicates)
    # The sums of the three terms depend on their order: one is 1.0, the
    # other 0.9999999999999999.
    stored = canonicalize_sparse(matrix.tocoo()).toarray()
    destination = str(tmp_path / "counts.zarr")
    if writer == "sparse":
        SparseToZarr(
            matrix,
            destination,
            cell_ids=["c0", "c1"],
            feature_ids=_features(stored),
            nthreads=1,
        ).dump()
    else:
        path = (
            _write_h5ad(tmp_path / "counts.h5ad", stored, "float64")
            if writer == "h5ad"
            else _write_cellranger(tmp_path / "counts.h5", stored, "float64")
        )
        # Store the duplicate coordinates themselves, in source order.
        with h5py.File(path, "r+") as h5:
            group = h5["X" if writer == "h5ad" else "matrix"]
            for name in ("data", "indices", "indptr"):
                del group[name]
                group.create_dataset(name, data=getattr(matrix, name))
        if writer == "h5ad":
            reader: Any = H5adReader(str(path), feature_name_key="feature_name")
            create = lambda: H5adToZarr(reader, destination, nthreads=1)  # noqa: E731
        else:
            reader = CrH5Reader(str(path))
            create = lambda: CrToZarr(reader, destination, nthreads=1)  # noqa: E731
        try:
            create().dump()
        finally:
            reader.close()
    counts = zarr.open_group(destination, mode="r")["RNA/counts"]
    assert counts.dtype == np.dtype(expected)
    np.testing.assert_array_equal(counts[:], stored.astype(expected))


def test_bands_of_canonical_and_unsorted_rows_keep_exact_large_counts(tmp_path):
    # Row 0 is canonical int64 and row 1 has unsorted indices, so only row 1
    # is canonicalized; both rows land in one uint64 band.
    big = 2**53 + 1
    matrix = csr_matrix(
        (np.array([big, 5, 3], dtype=np.int64), np.array([0, 2, 0]), [0, 1, 3]),
        shape=(2, 3),
    )
    destination = str(tmp_path / "counts.zarr")
    SparseToZarr(
        matrix,
        destination,
        cell_ids=["c0", "c1"],
        feature_ids=["g0", "g1", "g2"],
        nthreads=1,
    ).dump(batch_size=1)
    counts = zarr.open_group(destination, mode="r")["RNA/counts"]
    assert counts.dtype == np.uint64
    np.testing.assert_array_equal(counts[:], matrix.toarray().astype(np.uint64))


def test_sparse_writer_refuses_a_matrix_larger_than_mem_budget(tmp_path):
    from scarf.utils.arrays import sparse_matrix_bytes

    matrix = _sparse(_large_values(200, 50), "float32")
    held = sparse_matrix_bytes(matrix)
    destination = tmp_path / "counts.zarr"
    with pytest.raises(
        MemoryError,
        match=f"holds {held} bytes, but mem_budget is {held // 2} bytes",
    ):
        _writer("sparse", matrix, str(destination), mem_budget=held // 2, nthreads=1)
    assert not destination.exists()


def test_seurat_import_of_an_assay_without_cells_stores_uint8(tmp_path):
    destination = MemoryStore()
    with SeuratReader(
        _write_seurat(tmp_path / "empty.rds", np.zeros((0, 3)), "real"), reductions=[]
    ) as reader:
        SeuratToZarr(reader, destination, nthreads=1).dump()
    counts = zarr.open_group(store=destination, mode="r")["RNA/counts"]
    assert (counts.shape, counts.dtype) == ((0, 3), np.uint8)


def test_csv_counts_the_stored_dtype_cannot_hold_raise_instead_of_wrapping(
    tmp_path,
):
    values = np.minimum(_VALUES, 200)
    path = _write_csv(tmp_path / "counts.csv", values, "integer")
    reader = CSVReader(str(path), batch_size=2)
    writer = CSVtoZarr(reader, str(tmp_path / "counts.zarr"), assay_name="RNA")
    # The file changes after the reader scanned it, so the counts no longer
    # fit the uint8 dtype resolved from the scan; 300 would wrap to 44.
    values[0, 4] = 300
    _write_csv(path, values, "integer")
    with pytest.raises(OverflowError, match="exceed the destination dtype"):
        writer.dump()
    counts = zarr.open_group(str(tmp_path / "counts.zarr"), mode="r")["RNA/counts"]
    assert counts.dtype == np.uint8
    assert counts.attrs["complete"] is False


def test_seurat_dense_counts_the_stored_dtype_cannot_hold_raise_instead_of_wrapping(
    tmp_path, monkeypatch
):
    values = np.minimum(_VALUES, 200)
    with SeuratReader(
        _write_seurat(tmp_path / "counts.rds", values, "real"), reductions=[]
    ) as reader:
        writer = SeuratToZarr(reader, str(tmp_path / "counts.zarr"), nthreads=1)
        assert writer.counts["RNA"].dtype == np.uint8
        counts = reader.get_assay("RNA").counts
        original = counts.read_cells

        def changed(start, stop):
            block = np.array(original(start, stop))
            block[block == 200] = 300
            return block

        monkeypatch.setattr(counts, "read_cells", changed)
        with pytest.raises(OverflowError, match="exceed the destination dtype"):
            writer.dump()


@pytest.mark.parametrize(
    ("target", "keyword"),
    [
        (CrToZarr, "dtype"),
        (MtxToZarr, "dtype"),
        (MtxReader, "dtype"),
        (CSVtoZarr, "dtype"),
        (SparseToZarr, "matrix_dtype"),
        (DataStoreMerge, "dtype"),
    ],
)
def test_count_dtype_overrides_are_removed(target, keyword):
    assert keyword not in inspect.signature(target).parameters


def test_count_assays_are_created_with_an_explicit_dtype():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    with pytest.raises(TypeError, match="dtype"):
        create_zarr_count_assay(root, "RNA", None, 2, ["a"], ["a"])  # type: ignore[call-arg]
    counts = create_zarr_count_assay(root, "RNA", None, 2, ["a"], ["a"], np.uint8)
    assert counts.dtype == np.uint8


@pytest.mark.parametrize("dtype", [np.int32, np.float32])
def test_subset_and_repack_keep_the_source_dtype(tmp_path, dtype):
    source = tmp_path / "source.zarr"
    write_count_store(str(source), {"RNA": _VALUES}, dtype)
    store = DataStore(str(source), min_features_per_cell=0, nthreads=1)
    source_fingerprint = store.RNA.matrixGroup["counts"].attrs["content_fingerprint"]

    SubsetZarr(
        str(tmp_path / "subset.zarr"),
        assays=[store.RNA],
        cell_idx=np.arange(_VALUES.shape[0]),
        nthreads=1,
    ).dump()
    repack_store(str(source), str(tmp_path / "repacked.zarr"), data_only=True)

    for name in ("subset.zarr", "repacked.zarr"):
        counts = zarr.open_group(str(tmp_path / name), mode="r")["RNA/counts"]
        # The rebuilt dataset keeps its identity, so integral counts of an
        # earlier store stay in their source dtype.
        assert counts.dtype == np.dtype(dtype)
        assert counts.attrs["content_fingerprint"] == source_fingerprint
        np.testing.assert_array_equal(counts[:], _VALUES)


def _large_values(n_cells: int, n_features: int) -> np.ndarray:
    values = np.random.default_rng(0).poisson(0.3, size=(n_cells, n_features))
    # One count past uint16 keeps uint32 storage and wide count rows.
    values[0, 0] = 70_000
    return values


@pytest.fixture(scope="module")
def large_store(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("layout") / "source.zarr"
    values = _large_values(2_000, 400)
    SparseToZarr(
        csr_matrix(values),
        str(path),
        cell_ids=_cells(values),
        feature_ids=_features(values),
        nthreads=1,
    ).dump()
    return path


def _rebuild(writer: str, directory: Path, source: Path, **options: Any) -> zarr.Group:
    """Rebuild the counts of ``source`` with the subset or repack writer."""
    destination = str(directory / "counts.zarr")
    store = DataStore(str(source), min_features_per_cell=0, nthreads=1)
    if writer == "subset":
        SubsetZarr(
            destination,
            assays=[store.RNA],
            cell_idx=np.arange(store.cells.N),
            nthreads=1,
            **options,
        ).dump()
    else:
        repack_store(str(source), destination, data_only=True, nthreads=1, **options)
    return zarr.open_group(destination, mode="r")


# Budgets between the need of one-row count shards and the need of the default
# layout, and budgets below the one-row need but above what each writer needs
# before its layout fit: the sparse writer holds its 1.7 MB matrix, and the
# Seurat writer prepares its source. The Seurat fixture is smaller, because its
# dense RDS counts are slow to build.
_LAYOUT_CASES = {
    "cellranger": ("int32", (2_000, 400), 16 * 1024**2, 128 * 1024),
    "mtx": ("integer", (2_000, 400), 16 * 1024**2, 128 * 1024),
    "csv": ("integer", (2_000, 400), 16 * 1024**2, 128 * 1024),
    "sparse": ("float32", (2_000, 400), 16 * 1024**2, 1_740_000),
    "seurat": ("real", (600, 200), 2 * 1024**2, 40 * 1024),
    "subset": (None, (2_000, 400), 16 * 1024**2, 128 * 1024),
    "repack": (None, (2_000, 400), 16 * 1024**2, 128 * 1024),
}


def _layout_import(
    writer: str, directory: Path, large_store: Path, **options: Any
) -> zarr.Group:
    encoding, shape, _budget, _tiny = _LAYOUT_CASES[writer]
    if encoding is None:
        directory.mkdir(parents=True, exist_ok=True)
        return _rebuild(writer, directory, large_store, **options)
    return _import(writer, directory, _large_values(*shape), encoding, **options)


@pytest.mark.parametrize("writer", list(_LAYOUT_CASES))
def test_every_writer_finds_the_count_layout_that_fits_its_budget(
    tmp_path, large_store, writer
):
    _encoding, shape, budget, _tiny = _LAYOUT_CASES[writer]
    values = _large_values(*shape)
    if writer in ("subset", "repack"):
        if writer == "subset":
            # The default layout does not fit, and it is refused before the
            # destination exists.
            with pytest.raises(MemoryError, match="default count-matrix policy"):
                _layout_import(
                    writer,
                    tmp_path / "default",
                    large_store,
                    mem_budget=budget,
                    policy=DEFAULT_COUNT_MATRIX_POLICY,
                )
            assert not (tmp_path / "default" / "counts.zarr").exists()
        fitted = _layout_import(
            writer, tmp_path / "fitted", large_store, mem_budget=budget
        )
    else:
        # An import keeps the default layout, which does not fit. It is
        # refused before the destination exists, naming the layout that fits.
        with pytest.raises(MemoryError, match="default count-matrix policy") as refused:
            _layout_import(writer, tmp_path / "default", large_store, mem_budget=budget)
        assert not (tmp_path / "default" / "counts.zarr").exists()
        fitted = _layout_import(
            writer,
            tmp_path / "fitted",
            large_store,
            mem_budget=budget,
            policy=_named_policy(refused.value),
        )

    policy = _policy(fitted)
    halvings = DEFAULT_COUNT_MATRIX_POLICY.unitBytes // policy.unitBytes
    assert halvings > 1 and halvings & (halvings - 1) == 0
    assert policy.chunksPerShard == DEFAULT_COUNT_MATRIX_POLICY.chunksPerShard
    assert fitted["RNA/counts"].dtype == np.uint32
    assert fitted["RNA/countsT"].attrs["complete"] is True
    np.testing.assert_array_equal(fitted["RNA/counts"][:], values)
    np.testing.assert_array_equal(fitted["RNA/countsT"][:], values.T)
    # Identity does not depend on the layout: the fingerprint is that of the
    # same counts in a plain array.
    assert fitted["RNA/counts"].attrs["content_fingerprint"] == _fingerprint(
        values, np.uint32
    )

    # An ample budget keeps the default layout. An import records its layout
    # when it creates the counts; subset and repack fit it as they write.
    if writer in ("subset", "repack"):
        roomy = _layout_import(writer, tmp_path / "roomy", large_store, mem_budget="1G")
        assert _policy(roomy) == DEFAULT_COUNT_MATRIX_POLICY
        assert (
            roomy["RNA/counts"].attrs["content_fingerprint"]
            == fitted["RNA/counts"].attrs["content_fingerprint"]
        )
    else:
        assert (
            _planned_policy(
                writer, tmp_path / "roomy", values, _encoding, mem_budget="1G"
            )
            == DEFAULT_COUNT_MATRIX_POLICY
        )


@pytest.mark.parametrize("writer", list(_LAYOUT_CASES))
def test_every_writer_below_one_row_shards_fails_before_the_destination_exists(
    tmp_path, large_store, writer
):
    _encoding, _shape, _budget, tiny = _LAYOUT_CASES[writer]
    with pytest.raises(MemoryError, match="count shards of one row"):
        _layout_import(writer, tmp_path, large_store, mem_budget=tiny)
    assert not (tmp_path / "counts.zarr").exists()


@pytest.mark.parametrize("budget", [3_000_000, 6_000_000, 12_000_000])
def test_the_named_layout_writes_one_band_per_source_batch(budget):
    from scarf.storage.layout import array_shard_rows

    values = _large_values(2_000, 400)
    with pytest.raises(MemoryError, match="default count-matrix policy") as refused:
        _writer(
            "sparse",
            _sparse(values, "float32"),
            MemoryStore(),
            mem_budget=budget,
            nthreads=1,
        )
    store = MemoryStore()
    writer = _writer(
        "sparse",
        _sparse(values, "float32"),
        store,
        mem_budget=budget,
        nthreads=1,
        policy=_named_policy(refused.value),
    )
    writer.dump()
    counts = zarr.open_group(store=store, mode="r")["RNA/counts"]
    # A layout whose one-row batches fit, but not its band-high batches, would
    # starve the write to batches narrower than its bands.
    assert writer._lastImportPlan.batchRows == array_shard_rows(counts) < 2_000
    np.testing.assert_array_equal(counts[:], values)


@pytest.mark.parametrize("budget", [900_000, 2_100_000])
def test_the_named_layout_reads_one_band_of_dense_seurat_cells_per_batch(
    tmp_path, budget
):
    from scarf.storage.layout import array_shard_rows

    values = _large_values(600, 200).astype(np.float64)
    with SeuratReader(
        _write_seurat(tmp_path / "counts.rds", values, "real"), reductions=[]
    ) as reader:
        with pytest.raises(MemoryError, match="default count-matrix policy") as refused:
            SeuratToZarr(reader, MemoryStore(), mem_budget=budget, nthreads=1)
        writer = SeuratToZarr(
            reader,
            MemoryStore(),
            mem_budget=budget,
            nthreads=1,
            policy=_named_policy(refused.value),
        )
        writer.dump()
    shard_rows = array_shard_rows(writer.counts["RNA"])
    assert writer._lastDenseBatchRows["RNA"] == shard_rows < 600
    np.testing.assert_array_equal(writer.counts["RNA"][:], values)


def test_an_explicit_batch_holds_at_most_one_destination_band(tmp_path):
    values = _large_values(200, 50)
    store = MemoryStore()
    # uint32 rows of 200 bytes give 16-row shards.
    writer = _writer(
        "sparse",
        _sparse(values, "float32"),
        store,
        policy=CountMatrixPolicy(unitBytes=3_200, chunkBytes=800),
        nthreads=1,
    )
    writer.dump(batch_size=10_000)
    assert writer._lastImportPlan.batchRows == 16
    np.testing.assert_array_equal(
        zarr.open_group(store=store, mode="r")["RNA/counts"][:], values
    )


def test_subset_assay_zarr_fits_the_count_layout_to_its_budget(tmp_path, large_store):
    values = _large_values(2_000, 400)

    def subset(name: str, **options: Any) -> zarr.Group:
        store = tmp_path / name
        shutil.copytree(large_store, store)
        subset_assay_zarr(
            str(store),
            "RNA/counts",
            "selected",
            np.arange(values.shape[0]),
            np.arange(values.shape[1]),
            nthreads=1,
            **options,
        )
        return zarr.open_group(str(store), mode="r")

    # The subset writes no countsT, so the default layout needs less than an
    # import does, but more than this budget.
    budget = 8 * 1024**2
    with pytest.raises(MemoryError, match="default count-matrix policy"):
        subset("default", mem_budget=budget, policy=DEFAULT_COUNT_MATRIX_POLICY)
    # The source chunks that the subset reads do not fit at all.
    with pytest.raises(MemoryError, match="mem_budget"):
        subset("tiny", mem_budget=128 * 1024)
    for name in ("default", "tiny"):
        assert "selected" not in zarr.open_group(str(tmp_path / name), mode="r")

    fitted = subset("fitted", mem_budget=budget)["selected"]
    policy = policy_from_payload(load_count_matrix_plan(fitted))
    halvings = DEFAULT_COUNT_MATRIX_POLICY.unitBytes // policy.unitBytes
    assert halvings > 1 and halvings & (halvings - 1) == 0
    assert policy.chunksPerShard == DEFAULT_COUNT_MATRIX_POLICY.chunksPerShard
    assert fitted.dtype == np.uint32
    np.testing.assert_array_equal(fitted[:], values)


def test_the_named_layout_does_not_depend_on_the_worker_count(tmp_path):
    values = _large_values(2_000, 400)
    layouts = set()
    for workers in (1, 4):
        with pytest.raises(MemoryError, match="default count-matrix policy") as refused:
            _import(
                "cellranger",
                tmp_path / str(workers),
                values,
                "int32",
                mem_budget=16 * 1024**2,
                nthreads=workers,
            )
        layouts.add(_named_policy(refused.value))
    assert len(layouts) == 1
    assert next(iter(layouts)) != DEFAULT_COUNT_MATRIX_POLICY


@pytest.mark.parametrize(
    ("dtype", "converted"), [("int16", "int64"), ("uint8", "uint64")]
)
def test_csc_imports_plan_batches_in_the_dtype_consume_yields(
    tmp_path, monkeypatch, dtype, converted
):
    from scipy.sparse import csc_matrix

    values = np.array([[1, 0, 3], [0, 2, 0], [4, 0, 5]])
    path = _write_h5ad(tmp_path / "csc.h5ad", values, dtype)
    matrix = csc_matrix(values)
    with h5py.File(path, "r+") as h5:
        group = h5["X"]
        group.attrs["encoding-type"] = "csc_matrix"
        for name, array in (
            ("data", matrix.data.astype(dtype)),
            ("indices", matrix.indices),
            ("indptr", matrix.indptr),
        ):
            del group[name]
            group.create_dataset(name, data=array)
    reader = H5adReader(str(path))
    try:
        assert reader.matrixOrientation == "csc"
        assert reader.consumeDtype == np.dtype(dtype)
        reader.materialize_csc()
        assert reader.consumeDtype == np.dtype(converted)
        batch = next(reader.consume(batch_size=3))
        assert batch.dtype == np.dtype(converted)
        np.testing.assert_array_equal(batch.toarray(), values)
    finally:
        reader.close()

    # The layout fit and the batch plan size values in that dtype.
    from scarf.storage import sharding

    planned: list[tuple[str, np.dtype]] = []
    for name in ("resolve_sparse_import_spec", "resolve_sparse_import_batch"):
        original = getattr(sharding, name)

        def record(*args, _original=original, _name=name, **kwargs):
            planned.append((_name, np.dtype(kwargs["sourceDtype"])))
            return _original(*args, **kwargs)

        monkeypatch.setattr(sharding, name, record)
    reader = H5adReader(str(path))
    try:
        H5adToZarr(reader, str(tmp_path / "counts.zarr"), nthreads=1).dump()
    finally:
        reader.close()
    assert {name for name, _dtype in planned} == {
        "resolve_sparse_import_spec",
        "resolve_sparse_import_batch",
    }
    assert {dtype for _name, dtype in planned} == {np.dtype(converted)}
    counts = zarr.open_group(str(tmp_path / "counts.zarr"), mode="r")["RNA/counts"]
    np.testing.assert_array_equal(counts[:], values)
