"""Methods and classes for writing data to disk."""

from typing import TYPE_CHECKING

from .._facade import lazy_facade as _lazy_facade

if TYPE_CHECKING:
    from ._materialize import chunked_to_zarr, write_renorm_subset_to_zarr
    from ._store import (
        create_zarr_count_assay,
        create_zarr_dataset,
        create_zarr_obj_array,
    )
    from .cellranger import CrToZarr, MtxToZarr
    from .csv import CSVtoZarr
    from .export import to_h5ad, to_mtx
    from .h5ad import H5adImportResult, H5adToZarr
    from .seurat import SeuratImportResult, SeuratToZarr
    from .sparse import SparseToZarr
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
    "SeuratImportResult",
    "SeuratToZarr",
    "SparseToZarr",
    "to_h5ad",
    "to_mtx",
    "CSVtoZarr",
]

__getattr__, __dir__ = _lazy_facade(
    __name__,
    {
        "create_zarr_count_assay": "._store",
        "create_zarr_dataset": "._store",
        "create_zarr_obj_array": "._store",
        "chunked_to_zarr": "._materialize",
        "write_renorm_subset_to_zarr": "._materialize",
        "CrToZarr": ".cellranger",
        "MtxToZarr": ".cellranger",
        "CSVtoZarr": ".csv",
        "to_h5ad": ".export",
        "to_mtx": ".export",
        "H5adImportResult": ".h5ad",
        "H5adToZarr": ".h5ad",
        "SeuratImportResult": ".seurat",
        "SeuratToZarr": ".seurat",
        "SparseToZarr": ".sparse",
        "SubsetZarr": ".subset",
        "subset_assay_zarr": ".subset",
    },
    set_module=True,
)
