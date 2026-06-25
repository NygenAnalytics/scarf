"""
====================================
Scarf - Single-cell atlases reformed
====================================

Scarf is a Python package that performs memory-efficient analysis of single-cell genomics data.

- Analyze atlas scale datasets on your laptop (tested with up to 4 million cells)
- Perform analysis of scATAC-Seq data (datasets with up to 700K cells with 1 million peaks tested) under 10 GB RAM
- Make parallel implementations of UMAP and tSNE (SG-tSNE) for quick cell embedding
- Perform hierarchical clustering that gives interpretable cluster relationships
- Sub-sample highly representative cells using state-of-the-art method TopACeDo
- Perform quick and accurate projections of cells from one dataset onto another or integrate multiple datasets.

Exports:
--------

- Modules
    - datastore: Contains the primary interface to interact with data (i.e. DataStore) and its superclasses.
    - readers: A collection of classes for reading in different data formats.
    - writers: Methods and classes for writing data to disk.
    - meld_assay:
    - utils: Utility methods.
    - downloader: Used to download datasets included in Scarf.

GitHub: https://github.com/parashardhapola/scarf

Documentation: https://scarf.readthedocs.io/en/latest/index.html

Pre-print: https://www.biorxiv.org/content/10.1101/2021.05.02.441899v1

PyPI: https://pypi.org/project/scarf/
"""

import warnings

from importlib.metadata import version

from .datastore.datastore import DataStore
from .downloader import fetch_dataset, show_available_datasets
from .meld_assay import GffReader, coordinate_melding
from .merge import AssayMerge, DatasetMerge, ZarrMerge
from .readers import (
    CSVReader,
    CrDirReader,
    CrH5Reader,
    CrReader,
    H5adReader,
    LoomReader,
    NaboH5Reader,
)
from .utils import (
    clean_array,
    controlled_compute,
    get_log_level,
    load_zarr,
    logger,
    permute_into_chunks,
    prefetch_blocks,
    rescale_array,
    rolling_window,
    set_verbosity,
    show_dask_progress,
    system_call,
    tqdmbar,
    tqdm_params,
)
from .writers import (
    CSVtoZarr,
    CrToZarr,
    H5adToZarr,
    LoomToZarr,
    NaboH5ToZarr,
    SparseToZarr,
    SubsetZarr,
    create_zarr_count_assay,
    create_zarr_dataset,
    create_zarr_obj_array,
    dask_to_zarr,
    subset_assay_zarr,
    to_h5ad,
    to_mtx,
    write_renorm_subset_to_zarr,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    __version__ = version("scarf")
except ImportError:
    print("Scarf is not installed", flush=True)

__all__ = [
    "AssayMerge",
    "CSVReader",
    "CSVtoZarr",
    "CrDirReader",
    "CrH5Reader",
    "CrReader",
    "CrToZarr",
    "DataStore",
    "DatasetMerge",
    "GffReader",
    "H5adReader",
    "H5adToZarr",
    "LoomReader",
    "LoomToZarr",
    "NaboH5Reader",
    "NaboH5ToZarr",
    "SparseToZarr",
    "SubsetZarr",
    "ZarrMerge",
    "clean_array",
    "controlled_compute",
    "coordinate_melding",
    "create_zarr_count_assay",
    "create_zarr_dataset",
    "create_zarr_obj_array",
    "dask_to_zarr",
    "fetch_dataset",
    "get_log_level",
    "load_zarr",
    "logger",
    "permute_into_chunks",
    "prefetch_blocks",
    "rescale_array",
    "rolling_window",
    "set_verbosity",
    "show_available_datasets",
    "show_dask_progress",
    "subset_assay_zarr",
    "system_call",
    "to_h5ad",
    "to_mtx",
    "tqdmbar",
    "tqdm_params",
    "write_renorm_subset_to_zarr",
]
