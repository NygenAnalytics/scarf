"""H5AD inspection of dataframe columns that old AnnData nested into groups."""

from pathlib import Path

import h5py
import numpy as np
from scipy.sparse import csr_matrix

from scarf.readers.h5ad import H5adReader, inspect_h5ad
from scarf.readers._h5ad_inspect import _read_text_scalar

CELL_IDS = ["cell-a", "cell-b", "cell-c"]


def test_text_scalars_accept_only_scalar_or_one_element_datasets(
    tmp_path: Path,
) -> None:
    path = tmp_path / "text.h5"
    with h5py.File(path, "w") as h5:
        uns = h5.create_group("uns")
        uns.create_dataset("title", data=np.asarray([b"one"]))
        uns.create_dataset("description", data=np.asarray([b"a", b"b"]))
        uns.create_dataset("citation", data=np.bytes_("cited"))
    with h5py.File(path, "r") as h5:
        assert _read_text_scalar(h5, "uns/title", 500) == "one"
        assert _read_text_scalar(h5, "uns/description", 500) is None
        assert _read_text_scalar(h5, "uns/citation", 4) == "cite"
        assert _read_text_scalar(h5, "uns", 500) is None
        assert _read_text_scalar(h5, "uns/missing", 500) is None


def _write_table(
    h5: h5py.File,
    key: str,
    columns: dict[str, np.ndarray],
    *,
    column_order: bool,
) -> None:
    table = h5.create_group(key)
    table.attrs["encoding-type"] = "dataframe"
    table.attrs["_index"] = "_index"
    for name, values in columns.items():
        table.create_dataset(name, data=values)
    if column_order:
        # AnnData lists every column except the index by its full name.
        names = [name for name in columns if name != "_index"]
        table.attrs.create("column-order", names, dtype=h5py.string_dtype())


def _write_nested_cell_ids(path: Path, *, column_order: bool) -> None:
    """Write an H5AD whose only unique cell column is nested at its ``/``."""
    counts = csr_matrix(np.asarray([[1, 0], [2, 3], [0, 4]], dtype=np.int32))
    with h5py.File(path, "w") as h5:
        matrix = h5.create_group("X")
        matrix.attrs["encoding-type"] = "csr_matrix"
        matrix.attrs["shape"] = counts.shape
        matrix.create_dataset("data", data=counts.data)
        matrix.create_dataset("indices", data=counts.indices)
        matrix.create_dataset("indptr", data=counts.indptr)
        _write_table(
            h5,
            "obs",
            {
                "_index": np.asarray([b"dup", b"dup", b"other"]),
                "cell/id": np.asarray(CELL_IDS, dtype="S"),
                "sample": np.asarray([b"s1", b"s1", b"s2"]),
            },
            column_order=column_order,
        )
        _write_table(
            h5,
            "var",
            {"_index": np.asarray([b"ENSG01", b"ENSG02"])},
            column_order=column_order,
        )


def test_inspect_h5ad_sees_id_columns_that_hdf5_nests(tmp_path):
    path = tmp_path / "nested.h5ad"
    _write_nested_cell_ids(path, column_order=True)

    inspection = inspect_h5ad(str(path))

    # The source name is reported because the reader resolves it as a path.
    assert inspection.cellIdsKey == "cell/id"
    reader = H5adReader.from_inspect(inspection)
    try:
        assert reader.cell_ids().tolist() == [name.encode() for name in CELL_IDS]
    finally:
        reader.close()


def test_inspect_h5ad_without_column_order_sees_only_direct_children(tmp_path):
    path = tmp_path / "nested.h5ad"
    _write_nested_cell_ids(path, column_order=False)

    inspection = inspect_h5ad(str(path))

    assert inspection.cellIdsKey == "_index"
