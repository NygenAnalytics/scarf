import importlib
import inspect
import pickle
import subprocess
import sys
from typing import get_type_hints

import numpy as np
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
    H5adImportResult,
    H5adToZarr,
    MtxToZarr,
    SeuratImportResult,
    SeuratToZarr,
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
    SeuratToZarr: (
        "__init__",
        "dump",
    ),
}
# Every importer takes the keyword ``overwrite``.
_PUBLIC_CLASS_SIGNATURE_DIGESTS = {
    CrToZarr: "52d442411d74637bc13ef200681f3a73ba5e32b085c1313332327eb9106823d9",
    H5adToZarr: "af557d28c7a99d860acfeed7a95158a7c3ca07a24db85702acf087cbcbeaf8a3",
    SparseToZarr: "93e2b19b931a3fb26315a5bcc32ed45da2633cfa38ae177c832e37236e09b1e2",
    CSVtoZarr: "b005b7bd9fccc8d0c1ad987954f7f7d6de909987b42d577f4f416ccd9b5835de",
    SubsetZarr: "76fd6326d68a545065ce07fed9411ec7f02d3f080041db627c3623bda5fe4ed8",
    SeuratToZarr: "ae7e55e7c2e34d71b6fc1755babb987f2957e05cfd255d026327b989e1ec0c6d",
}
_MODULE_FUNCTIONS = (
    "create_zarr_count_assay",
    "create_zarr_dataset",
    "create_zarr_obj_array",
    "chunked_to_zarr",
    "subset_assay_zarr",
    "to_h5ad",
    "to_mtx",
    "write_renorm_subset_to_zarr",
)
# to_h5ad takes the keyword-only matrix="raw" or "normed" beside run.
_MODULE_SIGNATURE_DIGEST = (
    "d2b52149156853c361e7a78b8bbeaee30788df456db0b32da6acae49677a1076"
)


def test_writers_facade_surface_is_stable():
    assert writers_module.__all__ == [
        "create_zarr_dataset",
        "create_zarr_obj_array",
        "create_zarr_count_assay",
        "subset_assay_zarr",
        "chunked_to_zarr",
        "write_renorm_subset_to_zarr",
        "SubsetZarr",
        "CrToZarr",
        "MtxToZarr",
        "H5adImportResult",
        "H5adToZarr",
        "SeuratImportResult",
        "SeuratToZarr",
        "SparseToZarr",
        "to_h5ad",
        "to_mtx",
        "CSVtoZarr",
    ]
    expected = set(writers_module.__all__)
    assert expected.issubset(dir(writers_module))
    assert all(getattr(writers_module, name) is not None for name in expected)
    for removed in (
        "sparse_writer",
        "bed_to_sparse_array",
        "create_cell_data",
        "load_count_store",
        "load_zarr",
        "LoomToZarr",
    ):
        assert not hasattr(writers_module, removed)
    assert MtxToZarr is CrToZarr


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
    assert H5adImportResult.__module__ == "scarf.writers"
    assert SeuratImportResult.__module__ == "scarf.writers"


def test_writer_static_method_contracts_are_stable():
    for cls, names in {
        CrToZarr: ("_prep_feat_index_offset",),
        SubsetZarr: ("_check_assays",),
    }.items():
        for name in names:
            assert isinstance(inspect.getattr_static(cls, name), staticmethod)
    # Called on the class, the offsets move each source feature range of an
    # assay to the next free columns of that assay.
    assert CrToZarr._prep_feat_index_offset(
        {"RNA": ((0, 2), (5, 8)), "ADT": ((2, 5),)}
    ) == {"RNA": [0, -3], "ADT": [-2]}


def test_writer_storage_wrappers_remain_distinct_objects(
    monkeypatch: pytest.MonkeyPatch,
):
    from scarf.assay import normalization

    assert writers_module.create_zarr_dataset is not storage_arrays.create_zarr_dataset
    assert (
        writers_module.create_zarr_obj_array is not storage_arrays.create_zarr_obj_array
    )
    assert (
        writers_module.create_zarr_count_assay
        is not storage_schema.create_zarr_count_assay
    )
    assert writers_module.chunked_to_zarr is not storage_materialize.chunked_to_zarr
    assert (
        writers_module.write_renorm_subset_to_zarr
        is not normalization.write_renorm_subset_to_zarr
    )
    assert (
        "stats_group"
        in inspect.signature(storage_materialize.chunked_to_zarr).parameters
    )
    assert (
        "stats_group"
        not in inspect.signature(writers_module.chunked_to_zarr).parameters
    )

    forwarded: dict[str, object] = {}

    def capture(*args, **kwargs):
        forwarded["args"] = args
        forwarded["kwargs"] = kwargs

    monkeypatch.setattr("scarf.writers._materialize._chunked_to_zarr", capture)
    data = object()
    root = object()
    mirror = object()
    resources = object()
    writers_module.chunked_to_zarr(
        data,
        root,
        "normalized/data",
        3,
        msg="Writing",
        mirror=mirror,
        resources=resources,
    )

    assert forwarded == {
        "args": (data, root, "normalized/data", 3),
        "kwargs": {
            "msg": "Writing",
            "mirror": mirror,
            "resources": resources,
        },
    }


def test_writer_rejects_reserved_assay_name_before_mutation():
    root = zarr.open_group(store=MemoryStore(), mode="w")

    with pytest.raises(ValueError, match="reserved for DataStore.plots"):
        writers_module.create_zarr_count_assay(
            root,
            "plots",
            None,
            2,
            ["g1", "g2"],
            ["Gene 1", "Gene 2"],
            np.uint8,
        )

    assert list(root.group_keys()) == []

    with pytest.raises(ValueError, match=r"reserved for DataStore\.summary"):
        writers_module.create_zarr_count_assay(
            root,
            "summary",
            None,
            2,
            ["g1", "g2"],
            ["Gene 1", "Gene 2"],
            np.uint8,
        )

    assert list(root.group_keys()) == []

    with pytest.raises(ValueError, match="artifact storage"):
        writers_module.create_zarr_count_assay(
            root,
            "artifacts",
            None,
            2,
            ["g1", "g2"],
            ["Gene 1", "Gene 2"],
            np.uint8,
        )

    assert list(root.group_keys()) == []


def test_conversion_writers_reject_summary_before_truncating_destination():
    import pandas as pd
    from scipy.sparse import csr_matrix

    class SummaryCellRangerReader:
        assayFeats = pd.DataFrame({"summary": [0, 1]})

    constructors = {
        "cellranger": lambda store: CrToZarr(
            SummaryCellRangerReader(),
            zarr_loc=store,
        ),
        "csv": lambda store: CSVtoZarr(
            object(),
            zarr_loc=store,
            assay_name="summary",
        ),
        "h5ad": lambda store: H5adToZarr(
            object(),
            zarr_loc=store,
            assay_name="summary",
        ),
        "sparse": lambda store: SparseToZarr(
            csr_matrix((1, 1), dtype=np.uint32),
            zarr_loc=store,
            cell_ids=["c1"],
            feature_ids=["g1"],
            assay_name="summary",
        ),
    }

    for writer_name, construct in constructors.items():
        store = MemoryStore()
        root = zarr.open_group(store=store, mode="w")
        root.create_group("sentinel")

        with pytest.raises(
            ValueError,
            match=r"reserved for DataStore\.summary",
        ):
            construct(store)

        preserved = zarr.open_group(store=store, mode="r")
        assert set(preserved.group_keys()) == {"sentinel"}, writer_name


def test_writer_type_hints_resolve_from_facade_objects():
    for cls, names in _PUBLIC_CLASS_METHODS.items():
        for name in names:
            assert get_type_hints(getattr(cls, name))
    for name in _MODULE_FUNCTIONS:
        assert get_type_hints(getattr(writers_module, name))
    assert get_type_hints(H5adImportResult)
    assert get_type_hints(SeuratImportResult)


def test_writer_facade_objects_remain_pickle_resolvable():
    for cls in _PUBLIC_CLASS_METHODS:
        assert pickle.loads(pickle.dumps(cls)) is cls
    assert pickle.loads(pickle.dumps(H5adImportResult)) is H5adImportResult
    assert pickle.loads(pickle.dumps(SeuratImportResult)) is SeuratImportResult
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
    "scarf.writers.sparse",
    "scarf.writers.subset",
    "scarf.writers.seurat",
}
reader_formats = {
    "scarf.readers.cellranger",
    "scarf.readers.csv",
    "scarf.readers.h5ad",
    "scarf.readers.seurat",
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


def test_seurat_writer_exports_load_together_lazily():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys

import scarf.writers as writers

assert "scarf.writers.seurat" not in sys.modules
writer = writers.SeuratToZarr
assert writer.__module__ == "scarf.writers"
assert writers.SeuratImportResult.__module__ == "scarf.writers"
assert "scarf.writers.seurat" in sys.modules
assert "scarf.readers.seurat" in sys.modules
assert "scarf.datastore.datastore" not in sys.modules
assert "scarf.writers.h5ad" not in sys.modules
""",
        ],
        check=True,
    )


def test_writers_facade_reload_discards_cached_exports():
    expected = writers_module.CSVtoZarr
    writers_module.CSVtoZarr = object()

    reloaded = importlib.reload(writers_module)

    assert reloaded.CSVtoZarr is expected
