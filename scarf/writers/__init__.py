"""Methods and classes for writing data to disk."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._materialize import chunked_to_zarr, write_renorm_subset_to_zarr
    from ._store import (
        create_cell_data as create_cell_data,
        create_zarr_count_assay,
        create_zarr_dataset,
        create_zarr_obj_array,
        load_count_store as load_count_store,
        load_zarr as load_zarr,
    )
    from .cellranger import CrToZarr, MtxToZarr
    from .csv import CSVtoZarr
    from .export import to_h5ad, to_mtx
    from .h5ad import H5adImportResult, H5adToZarr
    from .loom import LoomToZarr
    from .seurat import SeuratImportResult, SeuratToZarr
    from .sparse import SparseToZarr, bed_to_sparse_array as bed_to_sparse_array
    from .subset import SubsetZarr, subset_assay_zarr

__all__ = [
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
    "LoomToZarr",
    "SeuratImportResult",
    "SeuratToZarr",
    "SparseToZarr",
    "to_h5ad",
    "to_mtx",
    "CSVtoZarr",
]

_LAZY_EXPORTS = {
    "create_cell_data": "_store",
    "create_zarr_count_assay": "_store",
    "create_zarr_dataset": "_store",
    "create_zarr_obj_array": "_store",
    "load_count_store": "_store",
    "load_zarr": "_store",
    "chunked_to_zarr": "_materialize",
    "write_renorm_subset_to_zarr": "_materialize",
    "CrToZarr": "cellranger",
    "MtxToZarr": "cellranger",
    "CSVtoZarr": "csv",
    "to_h5ad": "export",
    "to_mtx": "export",
    "H5adImportResult": "h5ad",
    "H5adToZarr": "h5ad",
    "LoomToZarr": "loom",
    "SeuratImportResult": "seurat",
    "SeuratToZarr": "seurat",
    "SparseToZarr": "sparse",
    "bed_to_sparse_array": "sparse",
    "SubsetZarr": "subset",
    "subset_assay_zarr": "subset",
}

for _export_name in _LAZY_EXPORTS:
    globals().pop(_export_name, None)
del _export_name

_METADATA_EXPORTS = frozenset(_LAZY_EXPORTS) - {"load_zarr"}


def _normalize_export_metadata(value: Any) -> None:
    value.__module__ = __name__
    if not isinstance(value, type):
        return
    for descriptor in value.__dict__.values():
        if isinstance(descriptor, (classmethod, staticmethod)):
            method = descriptor.__func__
        else:
            method = descriptor
        if callable(method) and hasattr(method, "__module__"):
            method.__module__ = __name__


def _cache_module_exports(module_name: str, module: Any) -> None:
    for export_name, export_module in _LAZY_EXPORTS.items():
        if export_module != module_name:
            continue
        value = getattr(module, export_name)
        if export_name in _METADATA_EXPORTS:
            _normalize_export_metadata(value)
        globals().setdefault(export_name, value)


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(f".{module_name}", __name__)
    _cache_module_exports(module_name, module)
    return globals()[name]


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
