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
    H5adToZarr,
    LoomToZarr,
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
    SeuratToZarr: (
        "__init__",
        "dump",
    ),
}
_PUBLIC_CLASS_SIGNATURE_DIGESTS = {
    CrToZarr: "6ddc400f3edcce221a338b60bc528a83f086409ee59b7d3e6baebfa0980f8f75",
    H5adToZarr: "c1993f1caa8cdb973c1bee9969cf107a3f39dbf2a2269d3e0a86f6c5f0b1d94b",
    LoomToZarr: "a840271f303b73a2187148efb1e4651a6671b4283af83c0f28251d449f5754dd",
    SparseToZarr: "78dcbc0d0d7d48a3d971ba0654260be39fd69157a5d3cb87f4d7191f0dee450d",
    CSVtoZarr: "30b241f964ead70d6a44e0f09be571ea88d227cd78335651b3f1a38cadf3268e",
    SubsetZarr: "ae6eb9a73c50d32b3716ea33002993bcd005461c68e0e784aec87a66ac3061e5",
    SeuratToZarr: "824c0169d8b921397643c54fd57b8d238720669da376e02c21d226be87adb86f",
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
    "dbe97d2ed890118780c3618b3c3433ea56954dc3ed23b67c93b665cbae46debf"
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
        "MtxToZarr",
        "H5adToZarr",
        "LoomToZarr",
        "SeuratImportResult",
        "SeuratToZarr",
        "SparseToZarr",
        "to_h5ad",
        "to_mtx",
        "CSVtoZarr",
    ]
    expected = set(writers_module.__all__) | {
        "bed_to_sparse_array",
        "create_cell_data",
        "load_count_store",
        "load_zarr",
    }
    assert expected.issubset(dir(writers_module))
    assert all(getattr(writers_module, name) is not None for name in expected)
    assert not hasattr(writers_module, "sparse_writer")
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
    assert SeuratImportResult.__module__ == "scarf.writers"


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
            2,
            ["g1", "g2"],
            ["Gene 1", "Gene 2"],
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
            dtype=np.dtype("uint32"),
        ),
        "h5ad": lambda store: H5adToZarr(
            object(),
            zarr_loc=store,
            assay_name="summary",
        ),
        "loom": lambda store: LoomToZarr(
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
    assert get_type_hints(SeuratImportResult)


def test_writer_facade_objects_remain_pickle_resolvable():
    for cls in _PUBLIC_CLASS_METHODS:
        assert pickle.loads(pickle.dumps(cls)) is cls
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
    "scarf.writers.loom",
    "scarf.writers.sparse",
    "scarf.writers.subset",
    "scarf.writers.seurat",
}
reader_formats = {
    "scarf.readers.cellranger",
    "scarf.readers.csv",
    "scarf.readers.h5ad",
    "scarf.readers.loom",
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
