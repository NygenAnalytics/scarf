import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
import zarr
from scipy.sparse import csr_matrix
from zarr.storage import MemoryStore

from scarf import DataStore
from scarf.readers import CSVReader, H5adReader, MtxReader, inspect_mtx
from scarf.utils.logging import logger
from scarf.writers import CSVtoZarr, H5adToZarr, MtxToZarr
from scarf.writers._store import keyed_metadata_columns

EGFR = "Baseline eGFR (ml/min/1.73m2) (Binned)"
EGFR_KEY = "Baseline eGFR (ml_min_1.73m2) (Binned)"


@dataclass(frozen=True)
class _Imported:
    store: MemoryStore
    cells: dict[str, list[Any]]
    features: dict[str, list[Any]] = field(default_factory=dict)
    # Axis, source name, and stored key of each renamed column.
    renamed: tuple[tuple[str, str, str], ...] = ()


def _capture_warnings(action: Callable[[], Any]) -> tuple[Any, list[str]]:
    messages: list[str] = []
    sink = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        result = action()
    finally:
        logger.remove(sink)
    return result, messages


def _write_separator_h5ad(
    path: Path,
    *,
    column_order: bool = True,
    extra_order: tuple[str, ...] = (),
) -> Path:
    """Write the layout old AnnData versions produced for '/' in a column name."""
    counts = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=np.uint16)
    matrix = csr_matrix(counts)
    with h5py.File(path, mode="w") as h5:
        group = h5.create_group("X")
        group.attrs["encoding-type"] = "csr_matrix"
        group.attrs["shape"] = counts.shape
        group.create_dataset("data", data=matrix.data)
        group.create_dataset("indices", data=matrix.indices)
        group.create_dataset("indptr", data=matrix.indptr)

        obs = h5.create_group("obs")
        obs.attrs["encoding-type"] = "dataframe"
        obs.attrs["_index"] = "_index"
        obs.create_dataset("_index", data=np.array([b"c0", b"c1", b"c2"]))
        # HDF5 nests a name with '/' into groups; column-order keeps the name.
        binned = obs.create_group(EGFR)
        binned.attrs["encoding-type"] = "categorical"
        binned.create_dataset("codes", data=np.array([0, 1, 0], dtype=np.int8))
        binned.create_dataset("categories", data=np.array([b"<60", b">=60"]))
        obs.create_dataset("c\\d", data=np.array([1.5, 2.5, 3.5]))
        obs.create_dataset("a/b", data=np.array([b"x", b"y", b"z"]))
        obs.create_dataset("a_b", data=np.array([10, 20, 30]))

        var = h5.create_group("var")
        var.attrs["encoding-type"] = "dataframe"
        var.attrs["_index"] = "_index"
        var.create_dataset("_index", data=np.array([b"f0", b"f1", b"f2"]))
        var.create_dataset("feature_name", data=np.array([b"g0", b"g1", b"g2"]))
        var.create_dataset("g/s", data=np.array([b"A", b"B", b"C"]))
        if column_order:
            obs.attrs["column-order"] = [EGFR, "c\\d", "a/b", "a_b", *extra_order]
            var.attrs["column-order"] = ["feature_name", "g/s"]
    return path


def _import_h5ad(tmp_path: Path) -> _Imported:
    source = _write_separator_h5ad(tmp_path / "separators.h5ad")
    store = MemoryStore()
    reader = H5adReader(str(source), feature_name_key="feature_name")
    try:
        H5adToZarr(reader, zarr_loc=store, mem_budget="64M", nthreads=1).dump()
    finally:
        reader.close()
    return _Imported(
        store=store,
        cells={
            "a_b": [10, 20, 30],
            EGFR_KEY: ["<60", ">=60", "<60"],
            "c_d": [1.5, 2.5, 3.5],
            "a_b_2": ["x", "y", "z"],
        },
        features={"g_s": ["A", "B", "C"]},
        renamed=(
            ("cell", EGFR, EGFR_KEY),
            ("cell", "c\\d", "c_d"),
            ("cell", "a/b", "a_b_2"),
            ("feature", "g/s", "g_s"),
        ),
    )


def _import_csv(tmp_path: Path) -> _Imported:
    path = tmp_path / "counts.csv"
    path.write_text(
        "a/b,a_b,c\\d,geneA,geneB\nx,10,1.5,1,0\ny,20,2.5,0,3\nz,30,3.5,4,5\n",
        encoding="utf-8",
    )
    reader = CSVReader(str(path), cell_data_cols=["a/b", "a_b", "c\\d"], batch_size=2)
    store = MemoryStore()
    CSVtoZarr(reader, zarr_loc=store, assay_name="RNA", nthreads=1).dump()
    return _Imported(
        store=store,
        cells={"a_b": [10, 20, 30], "a_b_2": ["x", "y", "z"], "c_d": [1.5, 2.5, 3.5]},
        renamed=(("cell", "a/b", "a_b_2"), ("cell", "c\\d", "c_d")),
    )


def _import_mtx(tmp_path: Path) -> _Imported:
    # A Parse Biosciences output carries its cell metadata in a CSV sidecar.
    (tmp_path / "DGE.mtx").write_text(
        "%%MatrixMarket matrix coordinate integer general\n"
        "3 2 4\n"
        "1 1 1\n"
        "1 2 2\n"
        "2 1 3\n"
        "3 2 4\n"
    )
    (tmp_path / "genes.csv").write_text(
        "gene_id,gene_name\nfeature-0,Gene 0\nfeature-1,Gene 1\n"
    )
    (tmp_path / "cell_metadata.csv").write_text(
        "bc_wells,a/b,a_b,c\\d\ncell-0,x,10,1.5\ncell-1,y,20,2.5\ncell-2,z,30,3.5\n"
    )
    reader = MtxReader(inspect_mtx(tmp_path)[0])
    store = MemoryStore()
    try:
        MtxToZarr(reader, store, mem_budget="64M", lines_in_mem=2).dump()
    finally:
        reader.close()
    return _Imported(
        store=store,
        cells={"a_b": [10, 20, 30], "a_b_2": ["x", "y", "z"], "c_d": [1.5, 2.5, 3.5]},
        renamed=(("cell", "a/b", "a_b_2"), ("cell", "c\\d", "c_d")),
    )


def _assert_imported(imported: _Imported, messages: list[str]) -> None:
    root = zarr.open_group(store=imported.store, mode="r")
    for path, expected in (
        ("cellData", imported.cells),
        ("RNA/featureData", imported.features),
    ):
        group = root[path]
        assert list(group.group_keys()) == []
        for key, values in expected.items():
            np.testing.assert_array_equal(group[key][:], values)
    for axis, source, key in imported.renamed:
        assert any(
            f"Stored source {axis} metadata column {source!r} as {key!r}" in message
            for message in messages
        ), (source, messages)

    datastore = DataStore(
        imported.store,
        default_assay="RNA",
        min_features_per_cell=0,
        mem_budget="64M",
        nthreads=1,
    )
    for table, expected in (
        (datastore.cells, imported.cells),
        (datastore.RNA.feats, imported.features),
    ):
        assert set(expected) <= set(table.columns)
        assert set(expected) <= set(table.head().columns)
        frame = table.to_pandas_dataframe(table.columns)
        for key, values in expected.items():
            assert frame[key].tolist() == values


@pytest.mark.parametrize(
    "build",
    [_import_h5ad, _import_csv, _import_mtx],
    ids=["h5ad", "csv", "mtx"],
)
def test_imports_store_separator_named_columns_under_underscore_keys(
    tmp_path: Path,
    build: Callable[[Path], _Imported],
) -> None:
    imported, messages = _capture_warnings(lambda: build(tmp_path))

    _assert_imported(imported, messages)
    # An exact source name keeps its key; a renamed one takes the next suffix.
    assert any(
        "'a/b' as 'a_b_2'" in message and "'a_b' is already used" in message
        for message in messages
    )


@pytest.mark.parametrize("source", ["c\\d", "a/b"], ids=["backslash", "slash"])
def test_anndata_written_separator_columns_round_trip(
    tmp_path: Path,
    source: str,
) -> None:
    anndata = pytest.importorskip("anndata")
    import pandas as pd

    obs = pd.DataFrame(
        {source: pd.Categorical(["x", "y", "x"]), "plain": [1.0, 2.0, 3.0]},
        index=["c0", "c1", "c2"],
    )
    var = pd.DataFrame({source: ["A", "B"]}, index=["f0", "f1"])
    path = tmp_path / "anndata.h5ad"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        adata = anndata.AnnData(
            X=csr_matrix(np.array([[1, 0], [0, 2], [3, 4]], dtype=np.float32)),
            obs=obs,
            var=var,
        )
        try:
            adata.write_h5ad(path)
        except ValueError as error:
            if "not allowed" not in str(error):
                raise
            pytest.skip(f"This AnnData version refuses the column name {source!r}")

    def build() -> _Imported:
        store = MemoryStore()
        reader = H5adReader(str(path))
        try:
            H5adToZarr(reader, zarr_loc=store, mem_budget="64M", nthreads=1).dump()
        finally:
            reader.close()
        return _Imported(
            store=store,
            cells={"plain": [1.0, 2.0, 3.0], key: ["x", "y", "x"]},
            features={key: ["A", "B"]},
            renamed=(("cell", source, key), ("feature", source, key)),
        )

    key = source.replace("/", "_").replace("\\", "_")
    imported, messages = _capture_warnings(build)

    _assert_imported(imported, messages)


def test_h5ad_without_column_order_skips_nested_columns(tmp_path: Path) -> None:
    source = _write_separator_h5ad(tmp_path / "unordered.h5ad", column_order=False)
    store = MemoryStore()

    def dump() -> None:
        reader = H5adReader(str(source), feature_name_key="feature_name")
        try:
            H5adToZarr(reader, zarr_loc=store, mem_budget="64M", nthreads=1).dump()
        finally:
            reader.close()

    _result, messages = _capture_warnings(dump)

    cells = zarr.open_group(store=store, mode="r")["cellData"]
    assert set(cells.array_keys()) == {"I", "ids", "names", "a_b", "c_d"}
    assert list(cells.group_keys()) == []
    for group in ("a", "Baseline eGFR (ml"):
        assert any(
            f"Skipping obs column {group!r} because its H5AD encoding" in message
            for message in messages
        ), group


def test_h5ad_reader_reports_column_order_names_the_file_lacks(
    tmp_path: Path,
) -> None:
    source = _write_separator_h5ad(tmp_path / "ghost.h5ad", extra_order=("ghost",))
    reader = H5adReader(str(source), feature_name_key="feature_name")
    try:
        assert reader._source_column_names("obs") == [
            EGFR,
            "c\\d",
            "a/b",
            "a_b",
            "_index",
        ]
        columns, messages = _capture_warnings(lambda: dict(reader.get_cell_columns()))
    finally:
        reader.close()

    assert list(columns) == [EGFR, "c\\d", "a/b", "a_b"]
    assert any(
        "Skipping obs column 'ghost' because column-order lists it but the file "
        "does not contain it" in message
        for message in messages
    )


def test_h5ad_reader_rejects_malformed_column_order_before_any_write(
    tmp_path: Path,
) -> None:
    source = _write_separator_h5ad(tmp_path / "malformed.h5ad")
    with h5py.File(source, "r+") as h5:
        h5["obs"].attrs["column-order"] = np.array([1.0, 2.0])
    store = tmp_path / "existing.zarr"
    zarr.open_group(str(store), mode="w").attrs["kept"] = True

    with pytest.raises(ValueError, match="does not list column names"):
        H5adReader(str(source), feature_name_key="feature_name")

    # The reader fails before a writer could open and replace the destination.
    assert zarr.open_group(str(store), mode="r").attrs["kept"] is True


def test_keyed_metadata_columns_explain_each_skipped_or_renamed_column() -> None:
    keys = {"a_b": "a_b", "a/b": "a_b_2", "c\\d": "c_d"}
    columns = [
        ("a_b", 1),
        ("a/b", 2),
        ("c\\d", 3),
        ("a/b", 4),
        ("ids", 5),
        ("__scarf/missing__x", 6),
        (".", 7),
        ("taken", 8),
    ]

    written, messages = _capture_warnings(
        lambda: list(keyed_metadata_columns(columns, keys, "cell"))
    )

    assert written == [("a_b", 1), ("a_b_2", 2), ("c_d", 3)]
    expected = [
        "Stored source cell metadata column 'a/b' as 'a_b_2' because Zarr reads "
        "'/' and '\\' as path separators, and 'a_b' is already used",
        "Stored source cell metadata column 'c\\\\d' as 'c_d' because Zarr reads "
        "'/' and '\\' as path separators",
        "Skipped source cell metadata column 'a/b' because the source repeats "
        "that column name",
        "Skipped source cell metadata column 'ids' because Scarf reserves the "
        "column names 'I', 'ids', and 'names' and the prefix '__scarf_missing__'",
        "Skipped source cell metadata column '__scarf/missing__x' because Scarf "
        "reserves the column",
        "Skipped source cell metadata column '.' because that name cannot name a "
        "Zarr array",
        "Skipped source cell metadata column 'taken' because the destination "
        "already has a column named 'taken'",
    ]
    assert len(messages) == len(expected)
    for message, text in zip(messages, expected, strict=True):
        assert message.startswith(text), (message, text)
