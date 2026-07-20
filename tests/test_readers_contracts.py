import gzip
import io
import inspect
import subprocess
import sys

import h5py
import numpy as np

import scarf.readers as readers_module
from scarf.readers import (
    CSVReader,
    CrDirReader,
    CrH5Reader,
    CrReader,
    H5adReader,
    LoomReader,
)
from tests.signature_contracts import signature_digest


_PUBLIC_CLASS_METHODS = {
    CrReader: (
        "__init__",
        "consume",
        "rename_assays",
        "feature_ids",
        "feature_names",
        "feature_types",
        "cell_names",
    ),
    CrH5Reader: (
        "__init__",
        "cell_names",
        "consume",
        "close",
    ),
    CrDirReader: (
        "__init__",
        "read_header",
        "process_batch",
        "to_sparse",
        "cell_names",
        "rename_batches",
        "consume",
    ),
    H5adReader: (
        "__init__",
        "cell_ids",
        "feat_ids",
        "feat_names",
        "get_cell_columns",
        "get_feat_columns",
        "consume_dataset",
        "consume_group",
        "consume",
    ),
    LoomReader: (
        "__init__",
        "cell_names",
        "cell_ids",
        "get_cell_attrs",
        "feature_names",
        "feature_ids",
        "get_feature_attrs",
        "consume",
    ),
    CSVReader: (
        "__init__",
        "cell_ids",
        "feature_ids",
        "consume",
    ),
}
_PUBLIC_CLASS_SIGNATURE_DIGESTS = {
    CrReader: "43d9d19bf016cf8c1d4720cc78442d0ae469620c50f32fef06539f2a05e213a1",
    CrH5Reader: "59f74373515157899063f0b9588402aa206a08215519e6032e9187eb7b4e6cc8",
    CrDirReader: "ad6ba1260fc28761847440e4da24a89d7c325a076462ef7bde4e12d902bbae78",
    H5adReader: "f4bea40d0fde5404a4276de316e83dfc4dc31ee68f105592b011cdb327410417",
    LoomReader: "c886ad21a9681366c5434d4f71c280fae22f01d17615daf00dfe78e2b3a3d569",
    CSVReader: "75e056fab441bf1c12f593a19f053c80134c44606562286c6b08dd30486e73a0",
}
_MODULE_SIGNATURE_DIGEST = (
    "c0f145a4bb685fb1d07050860d2cd8aded4f78bc315d01576685470da8adf8da"
)


def _write_minimal_cellranger_directory(path) -> None:
    (path / "features.tsv").write_text(
        "f1\tg1\tGene Expression\nf2\tg2\tGene Expression\n"
    )
    (path / "barcodes.tsv").write_text("c1\n")
    (path / "matrix.mtx").write_text(
        "%%MatrixMarket matrix coordinate integer general\n2 1 1\n1 1 1\n"
    )


def test_readers_facade_surface_is_stable():
    assert readers_module.__all__ == [
        "CrH5Reader",
        "CrDirReader",
        "CrReader",
        "H5adReader",
        "LoomReader",
        "CSVReader",
    ]
    expected = {
        "CSVReader",
        "CrDirReader",
        "CrH5Reader",
        "CrReader",
        "H5adReader",
        "LoomReader",
        "get_file_handle",
        "read_file",
    }
    assert expected.issubset(vars(readers_module))


def test_readers_facade_loads_format_modules_lazily():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import scarf.readers as readers; "
                "assert 'scarf.readers.cellranger' not in sys.modules; "
                "assert 'scarf.readers.csv' not in sys.modules; "
                "assert 'scarf.readers.h5ad' not in sys.modules; "
                "assert 'scarf.readers.loom' not in sys.modules; "
                "assert 'h5py' not in sys.modules; "
                "assert 'pandas' not in sys.modules; "
                "assert 'scipy' not in sys.modules; "
                "assert 'CSVReader' in dir(readers); "
                "reader = readers.CSVReader; "
                "assert reader.__module__ == 'scarf.readers'; "
                "assert 'scarf.readers.csv' in sys.modules; "
                "assert 'scarf.readers.cellranger' not in sys.modules; "
                "assert 'scarf.readers.h5ad' not in sys.modules; "
                "assert 'scarf.readers.loom' not in sys.modules; "
                "assert 'pandas' in sys.modules; "
                "assert 'h5py' not in sys.modules; "
                "assert 'scipy' not in sys.modules"
            ),
        ],
        check=True,
    )


def test_reader_class_and_method_signatures_are_stable():
    for cls, names in _PUBLIC_CLASS_METHODS.items():
        methods = {name: getattr(cls, name) for name in names}
        assert signature_digest(methods) == _PUBLIC_CLASS_SIGNATURE_DIGESTS[cls]


def test_reader_module_function_signatures_are_stable():
    methods = {
        name: getattr(readers_module, name) for name in ("get_file_handle", "read_file")
    }
    assert signature_digest(methods) == _MODULE_SIGNATURE_DIGEST


def test_reader_public_metadata_remains_on_facade():
    for cls, names in _PUBLIC_CLASS_METHODS.items():
        assert cls.__module__ == "scarf.readers"
        for name in names:
            descriptor = inspect.getattr_static(cls, name)
            assert descriptor.__module__ == "scarf.readers"
            assert descriptor.__qualname__.startswith(f"{cls.__name__}.")

    for name in ("get_file_handle", "read_file"):
        assert getattr(readers_module, name).__module__ == "scarf.readers"


def test_cellranger_reader_hierarchy_and_abstract_contracts_are_stable():
    assert inspect.isabstract(CrReader)
    assert issubclass(CrH5Reader, CrReader)
    assert issubclass(CrDirReader, CrReader)
    for name in ("_handle_version", "_read_dataset", "consume"):
        assert getattr(CrReader, name).__isabstractmethod__


def test_reader_text_helpers_support_plain_gzip_and_missing_files(tmp_path):
    plain = tmp_path / "plain.txt"
    compressed = tmp_path / "compressed.txt.gz"
    plain.write_text("one \n two\n")
    with gzip.open(compressed, mode="wt") as handle:
        handle.write("three\nfour \n")

    assert list(readers_module.read_file(str(plain))) == ["one", " two"]
    assert list(readers_module.read_file(str(compressed))) == ["three", "four"]

    missing = tmp_path / "missing.txt"
    try:
        readers_module.get_file_handle(str(missing))
    except FileNotFoundError as error:
        assert str(error) == f"ERROR: FILE NOT FOUND: {missing}"
    else:
        raise AssertionError("Missing reader input did not raise FileNotFoundError")


def test_get_file_handle_facade_remains_patchable_by_read_file(monkeypatch):
    handle = io.StringIO("one\ntwo\n")
    monkeypatch.setattr(
        readers_module,
        "get_file_handle",
        lambda filename: handle,
    )

    assert list(readers_module.read_file("virtual.txt")) == ["one", "two"]
    assert handle.closed


def test_read_file_facade_remains_patchable_by_cellranger_reader(
    tmp_path,
    monkeypatch,
):
    _write_minimal_cellranger_directory(tmp_path)
    original = readers_module.read_file
    calls = []

    def tracked_read_file(filename):
        calls.append(filename)
        yield from original(filename)

    monkeypatch.setattr(readers_module, "read_file", tracked_read_file)
    reader = CrDirReader(str(tmp_path))

    assert reader.feature_names() == ["g1", "g2"]
    assert any(filename.endswith("features.tsv") for filename in calls)
    assert any(filename.endswith("barcodes.tsv") for filename in calls)


def test_loom_reader_preserves_cell_feature_orientation(tmp_path):
    values = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
    path = tmp_path / "orientation.loom"
    with h5py.File(path, mode="w") as handle:
        handle.create_dataset("matrix", data=values)
        handle.create_group("col_attrs")
        handle.create_group("row_attrs")

    reader = LoomReader(str(path))
    try:
        chunks = [chunk.toarray() for chunk in reader.consume(batch_size=2)]
    finally:
        reader.h5.close()

    np.testing.assert_array_equal(np.concatenate(chunks), values.T)
