import gzip
import io
import inspect
import pickle
import subprocess
import sys
from typing import get_type_hints

import h5py
import numpy as np
import pytest
import zarr

import scarf.readers as readers_module
from scarf.readers import (
    CSVReader,
    CrDirReader,
    CrH5Reader,
    CrReader,
    H5adInspectResult,
    H5adReader,
    LoomReader,
    MtxCandidate,
    MtxReader,
    SeuratInspectResult,
    SeuratReader,
)
from tests.signature_contracts import signature_digest


_PUBLIC_CLASS_METHODS = {
    CrReader: (
        "__init__",
        "consume",
        "rename_assays",
        "reclassify_features",
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
        "from_inspect",
        "cell_ids",
        "feat_ids",
        "feat_names",
        "get_cell_columns",
        "get_feat_columns",
        "feature_types",
        "assay_feature_slices",
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
    MtxReader: (
        "__init__",
        "consume",
        "close",
    ),
    SeuratReader: (
        "__init__",
        "close",
        "get_assay",
        "get_reduction",
        "inspect",
    ),
}
_PUBLIC_CLASS_SIGNATURE_DIGESTS = {
    CrReader: "cfeac7ccf7bc316f1db1d9e177d6556b37a0169b3cbb2800a92e561b75f4fc4a",
    CrH5Reader: "053373f2af2f2fc74a3e00cde9b067c5818aba92c09da3a9ac2e129566ca87b9",
    CrDirReader: "51884c2390ad90fba1cdbd71808fc4dd98de548bd901e810c9caba6dbb9cf49a",
    H5adReader: "474da31ca7268b340606557f8119f19003123e846699f201879091d43f159aff",
    LoomReader: "85c3ff965cb94a4fa201915b9d43890081e1f26e931327fcda6531bee4c3782a",
    CSVReader: "8aa6c17c876afb62765584fc7ff64d2838c66ef53095da10d7198ca60ab83851",
    MtxReader: "06376a32ff98ff0153ae1cc35f327509c88784ce027cb045bf9833b45dbccf2a",
    SeuratReader: "f90ad28ab745692321245b3fd48e66273cbbe52795a9711f0cf17fb171820d3f",
}
_MODULE_SIGNATURE_DIGEST = (
    "06ff8febbf86e3fef017f7126c9f810ed2a0fa5101e9988945bc2417f5e8eae3"
)


def _write_minimal_cellranger_directory(path) -> None:
    (path / "features.tsv").write_text(
        "f1\tg1\tGene Expression\nf2\tg2\tGene Expression\n"
    )
    (path / "barcodes.tsv").write_text("c1\n")
    (path / "matrix.mtx").write_text(
        "%%MatrixMarket matrix coordinate integer general\n2 1 1\n1 1 1\n"
    )


def _write_mixed_cellranger_directory(path) -> None:
    (path / "features.tsv").write_text(
        "f1\tg1\tGene Expression\n"
        "f2\th1\tAntibody Capture\n"
        "f3\tg2\tGene Expression\n"
        "f4\th2\tAntibody Capture\n"
        "f5\ta1\tAntibody Capture\n"
    )
    (path / "barcodes.tsv").write_text("c1\n")
    (path / "matrix.mtx").write_text(
        "%%MatrixMarket matrix coordinate integer general\n5 1 3\n2 1 2\n4 1 4\n5 1 5\n"
    )


def test_readers_facade_surface_is_stable():
    assert readers_module.__all__ == [
        "CrH5Reader",
        "CrDirReader",
        "CrReader",
        "H5adInspectResult",
        "H5adReader",
        "inspect_h5ad",
        "MtxCandidate",
        "MtxReader",
        "inspect_mtx",
        "SeuratInspectResult",
        "SeuratReader",
        "inspect_seurat",
        "LoomReader",
        "CSVReader",
    ]
    expected = {
        "CSVReader",
        "CrDirReader",
        "CrH5Reader",
        "CrReader",
        "H5adInspectResult",
        "H5adReader",
        "LoomReader",
        "MtxCandidate",
        "MtxReader",
        "SeuratInspectResult",
        "SeuratReader",
        "get_file_handle",
        "inspect_h5ad",
        "inspect_mtx",
        "inspect_seurat",
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
                "assert 'scarf.readers.mtx' not in sys.modules; "
                "assert 'scarf.readers.seurat' not in sys.modules; "
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
                "assert 'scarf.readers.mtx' not in sys.modules; "
                "assert 'scarf.readers.seurat' not in sys.modules; "
                "assert 'pandas' in sys.modules; "
                "assert 'h5py' not in sys.modules; "
                "assert 'scipy' not in sys.modules"
            ),
        ],
        check=True,
    )


def test_matrix_market_exports_load_together_lazily():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import scarf.readers as readers; "
                "assert 'scarf.readers.mtx' not in sys.modules; "
                "reader = readers.MtxReader; "
                "assert reader.__module__ == 'scarf.readers'; "
                "assert readers.MtxCandidate.__module__ == 'scarf.readers'; "
                "assert readers.inspect_mtx.__module__ == 'scarf.readers'; "
                "assert 'scarf.readers.mtx' in sys.modules; "
                "assert 'scarf.readers.h5ad' not in sys.modules; "
                "assert 'scarf.readers.loom' not in sys.modules"
            ),
        ],
        check=True,
    )


def test_seurat_exports_load_together_without_loading_the_writer():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import scarf.readers as readers; "
                "assert 'scarf.readers.seurat' not in sys.modules; "
                "reader = readers.SeuratReader; "
                "assert reader.__module__ == 'scarf.readers'; "
                "assert readers.SeuratInspectResult.__module__ == 'scarf.readers'; "
                "assert readers.inspect_seurat.__module__ == 'scarf.readers'; "
                "assert 'scarf.readers.seurat' in sys.modules; "
                "assert 'scarf.writers.seurat' not in sys.modules"
            ),
        ],
        check=True,
    )


def test_seurat_reader_facade_objects_resolve_annotations_and_pickle():
    for name in ("__init__", "get_assay", "get_reduction", "inspect"):
        assert get_type_hints(getattr(SeuratReader, name))
    for value in (
        SeuratReader,
        SeuratInspectResult,
        readers_module.inspect_seurat,
    ):
        assert pickle.loads(pickle.dumps(value)) is value


def test_reader_class_and_method_signatures_are_stable():
    for cls, names in _PUBLIC_CLASS_METHODS.items():
        methods = {name: getattr(cls, name) for name in names}
        assert signature_digest(methods) == _PUBLIC_CLASS_SIGNATURE_DIGESTS[cls]


def test_reader_module_function_signatures_are_stable():
    methods = {
        name: getattr(readers_module, name)
        for name in (
            "get_file_handle",
            "inspect_h5ad",
            "inspect_mtx",
            "inspect_seurat",
            "read_file",
        )
    }
    assert signature_digest(methods) == _MODULE_SIGNATURE_DIGEST


def test_reader_public_metadata_remains_on_facade():
    for cls, names in _PUBLIC_CLASS_METHODS.items():
        assert cls.__module__ == "scarf.readers"
        for name in names:
            descriptor = inspect.getattr_static(cls, name)
            method = (
                descriptor.__func__
                if isinstance(descriptor, classmethod | staticmethod)
                else descriptor
            )
            assert method.__module__ == "scarf.readers"
            assert method.__qualname__.startswith(f"{cls.__name__}.")

    assert H5adInspectResult.__module__ == "scarf.readers"
    assert MtxCandidate.__module__ == "scarf.readers"
    assert SeuratInspectResult.__module__ == "scarf.readers"
    for name in (
        "get_file_handle",
        "inspect_h5ad",
        "inspect_mtx",
        "inspect_seurat",
        "read_file",
    ):
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


def test_crreader_reclassifies_noncontiguous_features_atomically(tmp_path):
    _write_mixed_cellranger_directory(tmp_path)
    reader = CrDirReader(str(tmp_path))

    reader.reclassify_features([1, 3], "HTO")

    assert reader.feature_types() == [
        "Gene Expression",
        "HTO",
        "Gene Expression",
        "HTO",
        "Antibody Capture",
    ]
    assert list(reader.assayFeats.columns) == ["RNA", "HTO", "RNA", "HTO", "ADT"]
    assert reader.feature_names("HTO") == ["h1", "h2"]
    assert reader.feature_names("ADT") == ["a1"]

    before = reader.assayFeats.copy()
    reader.reclassify_features([1, 3], "HTO")
    assert reader.assayFeats.equals(before)

    with pytest.raises(ValueError, match="conflicting"):
        reader.reclassify_features([1], "RNA", require_previous=None)
    assert reader.feature_types()[1] == "HTO"


def test_crreader_reclassification_validates_before_mutation(tmp_path):
    _write_mixed_cellranger_directory(tmp_path)
    reader = CrDirReader(str(tmp_path))
    original_types = reader.feature_types()
    original_table = reader.assayFeats.copy()

    with pytest.raises(ValueError, match="currently have type"):
        reader.reclassify_features([1, 2], "HTO")
    with pytest.raises(ValueError, match="unique"):
        reader.reclassify_features([1, 1], "HTO")
    with pytest.raises(IndexError, match="out-of-range"):
        reader.reclassify_features([5], "HTO")

    assert reader.feature_types() == original_types
    assert reader.assayFeats.equals(original_table)


def test_crreader_reclassification_locks_when_writer_captures_schema(tmp_path):
    from scarf.writers import CrToZarr

    source = tmp_path / "source"
    source.mkdir()
    _write_mixed_cellranger_directory(source)
    reader = CrDirReader(str(source))
    reader.reclassify_features([1, 3], "HTO")

    destination = tmp_path / "out.zarr"
    writer = CrToZarr(reader, str(destination))
    writer.dump()
    root = zarr.open_group(str(destination), mode="r")

    assert set(root.group_keys()) >= {"RNA", "HTO", "ADT"}
    np.testing.assert_array_equal(
        np.asarray(root["HTO/featureData/ids"][:]).astype(str),
        ["f2", "f4"],
    )
    np.testing.assert_array_equal(
        np.asarray(root["ADT/featureData/ids"][:]).astype(str),
        ["f5"],
    )
    np.testing.assert_array_equal(root["HTO/counts"][:], [[2, 4]])
    np.testing.assert_array_equal(root["ADT/counts"][:], [[5]])
    with pytest.raises(RuntimeError, match="captures the schema"):
        reader.reclassify_features([4], "HTO")


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
