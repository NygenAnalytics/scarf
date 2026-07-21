import importlib
import inspect
import pickle
import subprocess
import sys
from typing import get_type_hints

import pytest
import zarr
from zarr.storage import MemoryStore

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
    CrToZarr: "dd6edcf6a9fe30470e72a3e7444c6dbd657d6d72301656e668b70abd9b4b0e23",
    H5adToZarr: "ad87c8500db8379f61e70bd1e1e5cc9c4f9282d879cc522b4b099c7ad3df617a",
    LoomToZarr: "fbf990e0ca352387e78ad58689a322de2408e434255f594c7ce30e8c413a5d01",
    SparseToZarr: "72f1da0ae143a964d9cc361bcd526cf92d66baffe16a37a495274d88eed19e43",
    CSVtoZarr: "29792109de0712bb812788fc7d2d5d3f35f8fc6f826931d27a1c714237708548",
    SubsetZarr: "5f190befb39892bf22e836fe8d92137bb349c3669a3319652aea523f5cce3622",
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
    "0694afa2ccf5190ec57dca8b8e362d2092c05be4589e0dcb910d01262951271e"
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


def test_writer_rejects_reserved_assay_name_before_mutation():
    root = zarr.open_group(store=MemoryStore(), mode="w")

    with pytest.raises(ValueError, match="reserved for DataStore.plots"):
        writers_module.create_zarr_count_assay(
            root,
            "plots",
            None,
            (2, 2),
            2,
            ["g1", "g2"],
            ["Gene 1", "Gene 2"],
        )

    assert list(root.group_keys()) == []


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
