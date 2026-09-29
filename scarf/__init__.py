import warnings as _warnings
from collections.abc import Callable as _Callable
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _distribution_version
from pathlib import Path as _Path
from re import search as _re_search
from typing import TYPE_CHECKING

from ._facade import lazy_facade as _lazy_facade

if TYPE_CHECKING:
    from . import cytebase as cytebase
    from .datastore.datastore import DataStore as DataStore
    from .datastore.pipeline_run import (
        PipelineExecutionError as PipelineExecutionError,
        PipelineRun as PipelineRun,
    )
    from .datastore.summary import DataStoreSummary as DataStoreSummary
    from .storage.lineage import ArtifactLineage as ArtifactLineage
    from .features.enrichment import (
        EnrichmentResult as EnrichmentResult,
        read_gmt as read_gmt,
    )
    from .mapping.models import MappingResult as MappingResult
    from .mapping.reference import MappingReference as MappingReference
    from .storage.artifacts import ArtifactStatus as ArtifactStatus
    from .storage.errors import ArtifactResolutionError as ArtifactResolutionError
    from .storage.refs import ArtifactRef as ArtifactRef
    from .storage.stores import load_zarr as load_zarr
    from .merge import (
        DataStoreMerge as DataStoreMerge,
    )
    from .readers import (
        CSVReader as CSVReader,
        CrDirReader as CrDirReader,
        CrH5Reader as CrH5Reader,
        CrReader as CrReader,
        H5adInspectResult as H5adInspectResult,
        H5adReader as H5adReader,
        MtxReader as MtxReader,
        SeuratInspectResult as SeuratInspectResult,
        SeuratReader as SeuratReader,
        inspect_h5ad as inspect_h5ad,
        inspect_mtx as inspect_mtx,
        inspect_seurat as inspect_seurat,
    )
    from .trajectory.results import (
        FateMappingResult as FateMappingResult,
        PseudotimeAggregationResult as PseudotimeAggregationResult,
        PseudotimeMarkerResult as PseudotimeMarkerResult,
        PseudotimeScoreResult as PseudotimeScoreResult,
    )
    from .utils import (
        clean_array as clean_array,
        configure_output as configure_output,
        controlled_compute as controlled_compute,
        logger as logger,
        permute_into_chunks as permute_into_chunks,
        rescale_array as rescale_array,
        rolling_window as rolling_window,
        set_verbosity as set_verbosity,
        compute_with_progress as compute_with_progress,
        tqdmbar as tqdmbar,
        tqdm_params as tqdm_params,
    )
    from .writers import (
        CSVtoZarr as CSVtoZarr,
        CrToZarr as CrToZarr,
        H5adImportResult as H5adImportResult,
        H5adToZarr as H5adToZarr,
        MtxToZarr as MtxToZarr,
        SeuratImportResult as SeuratImportResult,
        SeuratToZarr as SeuratToZarr,
        SparseToZarr as SparseToZarr,
        SubsetZarr as SubsetZarr,
        create_zarr_count_assay as create_zarr_count_assay,
        create_zarr_dataset as create_zarr_dataset,
        create_zarr_obj_array as create_zarr_obj_array,
        chunked_to_zarr as chunked_to_zarr,
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


def _resolve_version(
    distribution_version: _Callable[[str], str] = _distribution_version,
    version_path: _Path | None = None,
) -> str:
    try:
        return distribution_version("scarf")
    except _PackageNotFoundError:
        path = version_path or _Path(__file__).with_name("_version.py")
        if not path.is_file():
            return "unavailable"
        match = _re_search(
            r"\bversion\s*=\s*['\"]([^'\"]+)['\"]",
            path.read_text(encoding="utf-8"),
        )
        return match.group(1) if match else "unavailable"


__version__ = _resolve_version()

_LAZY_EXPORTS: dict[str, str] = {
    "ArtifactLineage": ".storage.lineage",
    "ArtifactRef": ".storage.refs",
    "ArtifactResolutionError": ".storage.errors",
    "ArtifactStatus": ".storage.artifacts",
    "CSVReader": ".readers",
    "CSVtoZarr": ".writers",
    "CrDirReader": ".readers",
    "CrH5Reader": ".readers",
    "CrReader": ".readers",
    "CrToZarr": ".writers",
    "DataStore": ".datastore.datastore",
    "DataStoreSummary": ".datastore.summary",
    "DataStoreMerge": ".merge",
    "EnrichmentResult": ".features.enrichment",
    "FateMappingResult": ".trajectory.results",
    "H5adInspectResult": ".readers",
    "H5adImportResult": ".writers",
    "H5adReader": ".readers",
    "H5adToZarr": ".writers",
    "MtxReader": ".readers",
    "MtxToZarr": ".writers",
    "SeuratImportResult": ".writers",
    "SeuratInspectResult": ".readers",
    "SeuratReader": ".readers",
    "SeuratToZarr": ".writers",
    "MappingReference": ".mapping.reference",
    "MappingResult": ".mapping.models",
    "mount_datastore": ".datastore.datastore",
    "PseudotimeAggregationResult": ".trajectory.results",
    "PseudotimeMarkerResult": ".trajectory.results",
    "PseudotimeScoreResult": ".trajectory.results",
    "PipelineExecutionError": ".datastore.pipeline_run",
    "PipelineRun": ".datastore.pipeline_run",
    "SparseToZarr": ".writers",
    "SubsetZarr": ".writers",
    "clean_array": ".utils",
    "configure_output": ".utils",
    "controlled_compute": ".utils",
    "create_zarr_count_assay": ".writers",
    "create_zarr_dataset": ".writers",
    "create_zarr_obj_array": ".writers",
    "chunked_to_zarr": ".writers",
    "inspect_h5ad": ".readers",
    "inspect_mtx": ".readers",
    "inspect_seurat": ".readers",
    "load_zarr": ".storage.stores",
    "logger": ".utils",
    "permute_into_chunks": ".utils",
    "read_gmt": ".features.enrichment",
    "rescale_array": ".utils",
    "rolling_window": ".utils",
    "set_verbosity": ".utils",
    "compute_with_progress": ".utils",
    "subset_assay_zarr": ".writers",
    "to_h5ad": ".writers",
    "to_mtx": ".writers",
    "tqdmbar": ".utils",
    "tqdm_params": ".utils",
    "write_renorm_subset_to_zarr": ".writers",
}

__all__ = list(_LAZY_EXPORTS)

__getattr__, __dir__ = _lazy_facade(
    __name__,
    _LAZY_EXPORTS,
    modules=(
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
    ),
)
