"""A collection of classes for reading in different data formats.

- Classes:
    - CrH5Reader: A class to read in CellRanger (Cr) data, in the form of an H5 file.
    - CrDirReader: A class to read in CellRanger (Cr) data, in the form of a directory.
    - CrReader: A class to read in CellRanger (Cr) data.
    - H5adReader: A class to read in data in the form of a H5ad file (h5 file with AnnData information).
    - LoomReader: A class to read in data in the form of a Loom file.
"""

from typing import TYPE_CHECKING

from .._facade import lazy_facade as _lazy_facade

if TYPE_CHECKING:
    from .cellranger import CrH5Reader, CrReader
    from .csv import CSVReader
    from .h5ad import H5adInspectResult, H5adReader, inspect_h5ad
    from .loom import LoomReader
    from .mtx import CrDirReader, MtxCandidate, MtxReader, inspect_mtx
    from .seurat import SeuratInspectResult, SeuratReader, inspect_seurat

__all__ = [
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

__getattr__, __dir__ = _lazy_facade(
    __name__,
    {
        "CrDirReader": ".mtx",
        "CrH5Reader": ".cellranger",
        "CrReader": ".cellranger",
        "CSVReader": ".csv",
        "H5adInspectResult": ".h5ad",
        "H5adReader": ".h5ad",
        "inspect_h5ad": ".h5ad",
        "MtxCandidate": ".mtx",
        "MtxReader": ".mtx",
        "inspect_mtx": ".mtx",
        "SeuratInspectResult": ".seurat",
        "SeuratReader": ".seurat",
        "inspect_seurat": ".seurat",
        "LoomReader": ".loom",
    },
    set_module=True,
)
