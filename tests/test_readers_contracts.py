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
    CrReader: "016c1b61d01f8809606e469fadd4614c8dd026ddc098a18ca8cec54653da5254",
    CrH5Reader: "053373f2af2f2fc74a3e00cde9b067c5818aba92c09da3a9ac2e129566ca87b9",
    CrDirReader: "51884c2390ad90fba1cdbd71808fc4dd98de548bd901e810c9caba6dbb9cf49a",
    H5adReader: "952244c6ef7048f599958bcf3b0e853f2934c7ed141736484b5b4bb776966967",
    LoomReader: "85c3ff965cb94a4fa201915b9d43890081e1f26e931327fcda6531bee4c3782a",
    CSVReader: "8aa6c17c876afb62765584fc7ff64d2838c66ef53095da10d7198ca60ab83851",
}
_MODULE_SIGNATURE_DIGEST = (
    "f0889415a5820081a464e6dd0c65580fb42e47c40b44d1a85cb507ca10388186"
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
