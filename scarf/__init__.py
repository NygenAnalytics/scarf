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
    - cytebase: Browse and download public datasets from Cytebase.
    - datastore: Contains the primary interface to interact with data (i.e. DataStore) and its superclasses.
    - embeddings: Dimension reduction, Harmony correction, and cell embeddings.
    - matrix: Lazy blockwise matrix operations over NumPy and Zarr arrays.
    - readers: Classes for reading supported data formats.
    - writers: Methods and classes for writing data to disk.
    - features: Variability, marker, scoring, and genomic feature algorithms.
    - mapping: Reference mapping, alignment, and confidence algorithms.
    - metrics: Integration and neighborhood quality metrics.
    - quality_control: Quality-control algorithms and cell-cycle references.
    - utils: Utility methods.

GitHub: https://github.com/parashardhapola/scarf

Documentation: https://scarf.readthedocs.io/en/latest/index.html

Pre-print: https://www.biorxiv.org/content/10.1101/2021.05.02.441899v1

PyPI: https://pypi.org/project/scarf/
"""

import warnings as _warnings
from importlib import import_module as _import_module
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _distribution_version
from pathlib import Path as _Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from . import cytebase as cytebase
    from .datastore.datastore import DataStore as DataStore
    from .graph.state import AssayState as AssayState
    from .features.enrichment import (
        EnrichmentResult as EnrichmentResult,
        read_gmt as read_gmt,
    )
    from .mapping.models import MappingResult as MappingResult
    from .mapping.reference import MappingReference as MappingReference
    from .storage.artifacts import (
        ArtifactRef as ArtifactRef,
        ArtifactStatus as ArtifactStatus,
    )
    from .features.genomic.gff import GffReader as GffReader
    from .features.genomic.melding import coordinate_melding as coordinate_melding
    from .merge import (
        AssayMerge as AssayMerge,
        DatasetMerge as DatasetMerge,
        ZarrMerge as ZarrMerge,
    )
    from .readers import (
        CSVReader as CSVReader,
        CrDirReader as CrDirReader,
        CrH5Reader as CrH5Reader,
        CrReader as CrReader,
        H5adInspectResult as H5adInspectResult,
        H5adReader as H5adReader,
        LoomReader as LoomReader,
        inspect_h5ad as inspect_h5ad,
    )
    from .trajectory.results import (
        FateMappingResult as FateMappingResult,
        PseudotimeAggregationResult as PseudotimeAggregationResult,
        PseudotimeMarkerResult as PseudotimeMarkerResult,
        PseudotimeScoreResult as PseudotimeScoreResult,
    )
    from .utils import (
        clean_array as clean_array,
        controlled_compute as controlled_compute,
        get_log_level as get_log_level,
        load_zarr as load_zarr,
        logger as logger,
        permute_into_chunks as permute_into_chunks,
        prefetch_blocks as prefetch_blocks,
        rescale_array as rescale_array,
        rolling_window as rolling_window,
        set_verbosity as set_verbosity,
        show_dask_progress as show_dask_progress,
        system_call as system_call,
        tqdmbar as tqdmbar,
        tqdm_params as tqdm_params,
    )
    from .writers import (
        CSVtoZarr as CSVtoZarr,
        CrToZarr as CrToZarr,
        H5adToZarr as H5adToZarr,
        LoomToZarr as LoomToZarr,
        SparseToZarr as SparseToZarr,
        SubsetZarr as SubsetZarr,
        create_zarr_count_assay as create_zarr_count_assay,
        create_zarr_dataset as create_zarr_dataset,
        create_zarr_obj_array as create_zarr_obj_array,
        dask_to_zarr as dask_to_zarr,
        subset_assay_zarr as subset_assay_zarr,
        to_h5ad as to_h5ad,
        to_mtx as to_mtx,
        write_renorm_subset_to_zarr as write_renorm_subset_to_zarr,
    )

_warnings.filterwarnings(
    "ignore",
    message=r"The data type .* does not have a Zarr V3 specification\.",
    module=r"zarr\.core\.dtype\..*",
)

try:
    __version__ = _distribution_version("scarf")
except _PackageNotFoundError:
    _version_path = _Path(__file__).resolve().parents[1] / "VERSION"
    __version__ = (
        _version_path.read_text().strip() if _version_path.is_file() else "unavailable"
    )

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ArtifactRef": (".storage.artifacts", "ArtifactRef"),
    "ArtifactStatus": (".storage.artifacts", "ArtifactStatus"),
    "AssayMerge": (".merge", "AssayMerge"),
    "AssayState": (".graph.state", "AssayState"),
    "CSVReader": (".readers", "CSVReader"),
    "CSVtoZarr": (".writers", "CSVtoZarr"),
    "CrDirReader": (".readers", "CrDirReader"),
    "CrH5Reader": (".readers", "CrH5Reader"),
    "CrReader": (".readers", "CrReader"),
    "CrToZarr": (".writers", "CrToZarr"),
    "DataStore": (".datastore.datastore", "DataStore"),
    "DatasetMerge": (".merge", "DatasetMerge"),
    "EnrichmentResult": (".features.enrichment", "EnrichmentResult"),
    "FateMappingResult": (".trajectory.results", "FateMappingResult"),
    "GffReader": (".features.genomic.gff", "GffReader"),
    "H5adInspectResult": (".readers", "H5adInspectResult"),
    "H5adReader": (".readers", "H5adReader"),
    "H5adToZarr": (".writers", "H5adToZarr"),
    "LoomReader": (".readers", "LoomReader"),
    "LoomToZarr": (".writers", "LoomToZarr"),
    "MappingReference": (".mapping.reference", "MappingReference"),
    "MappingResult": (".mapping.models", "MappingResult"),
    "PseudotimeAggregationResult": (
        ".trajectory.results",
        "PseudotimeAggregationResult",
    ),
    "PseudotimeMarkerResult": (
        ".trajectory.results",
        "PseudotimeMarkerResult",
    ),
    "PseudotimeScoreResult": (
        ".trajectory.results",
        "PseudotimeScoreResult",
    ),
    "SparseToZarr": (".writers", "SparseToZarr"),
    "SubsetZarr": (".writers", "SubsetZarr"),
    "ZarrMerge": (".merge", "ZarrMerge"),
    "clean_array": (".utils", "clean_array"),
    "controlled_compute": (".utils", "controlled_compute"),
    "coordinate_melding": (".features.genomic.melding", "coordinate_melding"),
    "create_zarr_count_assay": (".writers", "create_zarr_count_assay"),
    "create_zarr_dataset": (".writers", "create_zarr_dataset"),
    "create_zarr_obj_array": (".writers", "create_zarr_obj_array"),
    "dask_to_zarr": (".writers", "dask_to_zarr"),
    "get_log_level": (".utils", "get_log_level"),
    "inspect_h5ad": (".readers", "inspect_h5ad"),
    "load_zarr": (".utils", "load_zarr"),
    "logger": (".utils", "logger"),
    "permute_into_chunks": (".utils", "permute_into_chunks"),
    "prefetch_blocks": (".utils", "prefetch_blocks"),
    "read_gmt": (".features.enrichment", "read_gmt"),
    "rescale_array": (".utils", "rescale_array"),
    "rolling_window": (".utils", "rolling_window"),
    "set_verbosity": (".utils", "set_verbosity"),
    "show_dask_progress": (".utils", "show_dask_progress"),
    "subset_assay_zarr": (".writers", "subset_assay_zarr"),
    "system_call": (".utils", "system_call"),
    "to_h5ad": (".writers", "to_h5ad"),
    "to_mtx": (".writers", "to_mtx"),
    "tqdmbar": (".utils", "tqdmbar"),
    "tqdm_params": (".utils", "tqdm_params"),
    "write_renorm_subset_to_zarr": (
        ".writers",
        "write_renorm_subset_to_zarr",
    ),
}

_LAZY_MODULES = frozenset(
    {
        "assay",
        "cytebase",
        "datastore",
        "embeddings",
        "features",
        "mapping",
        "matrix",
        "merge",
        "metadata",
        "metrics",
        "quality_control",
        "readers",
        "storage",
        "utils",
        "writers",
    }
)

for _lazy_name in (*_LAZY_EXPORTS, *_LAZY_MODULES):
    globals().pop(_lazy_name, None)
del _lazy_name

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _LAZY_MODULES:
        value = _import_module(f".{name}", __name__)
    elif export := _LAZY_EXPORTS.get(name):
        module_name, attribute_name = export
        value = getattr(_import_module(module_name, __name__), attribute_name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(_LAZY_EXPORTS, _LAZY_MODULES))
