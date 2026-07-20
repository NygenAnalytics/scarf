import importlib
import inspect
import pickle
import subprocess
import sys
from typing import get_type_hints

import scarf.writers as writers_module
from scarf.storage import arrays as storage_arrays
from scarf.storage import materialize as storage_materialize
from scarf.storage import schema as storage_schema
from scarf.writers import (
    CSVtoZarr,
    CrToZarr,
    H5adToZarr,
    LoomToZarr,
    SparseToZarr,
    SubsetZarr,
)
from tests.signature_contracts import signature_digest


_PUBLIC_CLASS_METHODS = {
    CrToZarr: (
        "__init__",
        "dump",
    ),
    H5adToZarr: (
        "__init__",
        "dump",
    ),
    LoomToZarr: (
        "__init__",
        "dump",
    ),
    SparseToZarr: (
        "__init__",
        "dump",
    ),
    CSVtoZarr: (
        "__init__",
        "dump",
    ),
    SubsetZarr: (
        "__init__",
        "dump",
    ),
}
_PUBLIC_CLASS_SIGNATURE_DIGESTS = {
    CrToZarr: "344d41debb0f9586364ce6295cdb3813fd103d9d9957521eb16b9c5cb17b88e4",
    H5adToZarr: "0433cbbfc23f494de3643a12ee9a62fe5877575f6c80395f1c6214ab52da22eb",
    LoomToZarr: "dd522fe4083c3f6b1012f6ded940b529b1f7c541083ac312375710c96fcf011f",
    SparseToZarr: "c71d015c4d67465d82d6edeba4a732de686b231a6c05f9de34c279fe7fc3fbb9",
    CSVtoZarr: "f1ddc14e7474f0be1a5ee37d97125d959599e9dfa66804c8300f5272d2708bc9",
    SubsetZarr: "206f316920e194ae26dc85be63abfc6058cbfdaec5b89b779e193da48cfc786c",
}
_MODULE_FUNCTIONS = (
    "create_zarr_count_assay",
    "create_zarr_dataset",
    "create_zarr_obj_array",
    "dask_to_zarr",
    "subset_assay_zarr",
    "to_h5ad",
    "to_mtx",
    "write_renorm_subset_to_zarr",
)
_MODULE_SIGNATURE_DIGEST = (
    "5bd8ea971cdefc36dfa06bd1059a7193b000e296eaf53bcc392971e2f8626131"
)


def test_writers_facade_surface_is_stable():
    assert writers_module.__all__ == [
        "create_zarr_dataset",
        "create_zarr_obj_array",
        "create_zarr_count_assay",
        "subset_assay_zarr",
        "dask_to_zarr",
        "write_renorm_subset_to_zarr",
        "SubsetZarr",
        "CrToZarr",
        "H5adToZarr",
        "LoomToZarr",
        "SparseToZarr",
        "to_h5ad",
        "to_mtx",
        "CSVtoZarr",
    ]
    expected = set(writers_module.__all__) | {
        "_apply_budget_override",
        "bed_to_sparse_array",
        "create_cell_data",
        "finalize_writer_counts",
        "load_count_store",
        "load_zarr",
        "sparse_writer",
    }
    assert expected.issubset(dir(writers_module))
    assert all(getattr(writers_module, name) is not None for name in expected)


def test_writer_class_and_method_signatures_are_stable():
    for cls, names in _PUBLIC_CLASS_METHODS.items():
        methods = {name: getattr(cls, name) for name in names}
        assert signature_digest(methods) == _PUBLIC_CLASS_SIGNATURE_DIGESTS[cls]


def test_writer_module_function_signatures_are_stable():
    methods = {name: getattr(writers_module, name) for name in _MODULE_FUNCTIONS}
    assert signature_digest(methods) == _MODULE_SIGNATURE_DIGEST


def test_writer_public_metadata_remains_on_facade():
    for cls, names in _PUBLIC_CLASS_METHODS.items():
        assert cls.__module__ == "scarf.writers"
        for name in names:
            descriptor = inspect.getattr_static(cls, name)
            if isinstance(descriptor, staticmethod):
                method = descriptor.__func__
            else:
                method = descriptor
            assert method.__module__ == "scarf.writers"
            assert method.__qualname__.startswith(f"{cls.__name__}.")

    for name in _MODULE_FUNCTIONS:
        assert getattr(writers_module, name).__module__ == "scarf.writers"


def test_writer_static_method_contracts_are_stable():
    for cls, names in {
        CrToZarr: ("_prep_assay_input_ranges", "_prep_feat_index_offset"),
        SubsetZarr: ("_check_assays",),
    }.items():
        for name in names:
            assert isinstance(inspect.getattr_static(cls, name), staticmethod)


def test_writer_storage_wrappers_remain_distinct_objects():
    assert writers_module.create_zarr_dataset is not storage_arrays.create_zarr_dataset
    assert (
        writers_module.create_zarr_obj_array is not storage_arrays.create_zarr_obj_array
    )
    assert (
        writers_module.create_zarr_count_assay
        is not storage_schema.create_zarr_count_assay
    )
    assert writers_module.create_cell_data is not storage_schema.create_cell_data
    assert writers_module.finalize_writer_counts is not storage_schema.finalize_counts
    assert writers_module.dask_to_zarr is not storage_materialize.dask_to_zarr
    assert (
        writers_module.write_renorm_subset_to_zarr
        is not storage_materialize.write_renorm_subset_to_zarr
    )


def test_writer_type_hints_resolve_from_facade_objects():
    for cls, names in _PUBLIC_CLASS_METHODS.items():
        for name in names:
            assert get_type_hints(getattr(cls, name))
    for name in _MODULE_FUNCTIONS:
        assert get_type_hints(getattr(writers_module, name))


def test_writer_facade_objects_remain_pickle_resolvable():
    for cls in _PUBLIC_CLASS_METHODS:
        assert pickle.loads(pickle.dumps(cls)) is cls
    for name in _MODULE_FUNCTIONS:
        function = getattr(writers_module, name)
        assert pickle.loads(pickle.dumps(function)) is function


def test_writers_facade_loads_format_implementations_lazily():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import inspect
import sys

import scarf.writers as writers

writer_formats = {
    "scarf.writers.cellranger",
    "scarf.writers.csv",
    "scarf.writers.h5ad",
    "scarf.writers.loom",
    "scarf.writers.sparse",
    "scarf.writers.subset",
}
reader_formats = {
    "scarf.readers.cellranger",
    "scarf.readers.csv",
    "scarf.readers.h5ad",
    "scarf.readers.loom",
}
assert not {
    name for name in sys.modules if name.startswith("scarf.writers.")
}
assert reader_formats.isdisjoint(sys.modules)

writer_class = writers.CSVtoZarr
assert writer_formats.intersection(sys.modules) == {"scarf.writers.csv"}
assert reader_formats.intersection(sys.modules) == {"scarf.readers.csv"}
assert "scarf.writers._store" not in sys.modules
assert "h5py" not in sys.modules
assert "scipy" not in sys.modules
assert writer_class.__module__ == "scarf.writers"
method = inspect.getattr_static(writer_class, "__init__")
assert method.__module__ == "scarf.writers"
assert method.__qualname__ == "CSVtoZarr.__init__"
assert writers.CSVtoZarr is writer_class
""",
        ],
        check=True,
    )


def test_writers_facade_reload_discards_cached_exports():
    expected = writers_module.CSVtoZarr
    writers_module.CSVtoZarr = object()

    reloaded = importlib.reload(writers_module)

    assert reloaded.CSVtoZarr is expected
