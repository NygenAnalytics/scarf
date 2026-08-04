import warnings as _warnings
from collections.abc import Callable as _Callable
from importlib import import_module as _import_module
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _distribution_version
from pathlib import Path as _Path
from re import search as _re_search
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from . import cytebase as cytebase
    from .datastore.datastore import DataStore as DataStore
    from .datastore.summary import DataStoreSummary as DataStoreSummary
    from .graph.state import (
        ArtifactSelectionError as ArtifactSelectionError,
        AssayState as AssayState,
    )
    from .lineage import ArtifactLineage as ArtifactLineage
    from .features.enrichment import (
        EnrichmentResult as EnrichmentResult,
        read_gmt as read_gmt,
    )
    from .mapping.models import MappingResult as MappingResult
    from .mapping.reference import MappingReference as MappingReference
    from .storage.artifacts import ArtifactStatus as ArtifactStatus
    from .storage.refs import ArtifactRef as ArtifactRef
    from .features.genomic.gff import GffReader as GffReader
    from .features.genomic.melding import coordinate_melding as coordinate_melding
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
        LoomReader as LoomReader,
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
        get_log_level as get_log_level,
        load_zarr as load_zarr,
        logger as logger,
        permute_into_chunks as permute_into_chunks,
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
        MtxToZarr as MtxToZarr,
        SeuratImportResult as SeuratImportResult,
        SeuratToZarr as SeuratToZarr,
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

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ArtifactLineage": (".lineage", "ArtifactLineage"),
    "ArtifactRef": (".storage.refs", "ArtifactRef"),
    "ArtifactSelectionError": (".graph.state", "ArtifactSelectionError"),
    "ArtifactStatus": (".storage.artifacts", "ArtifactStatus"),
    "AssayState": (".graph.state", "AssayState"),
    "CSVReader": (".readers", "CSVReader"),
    "CSVtoZarr": (".writers", "CSVtoZarr"),
    "CrDirReader": (".readers", "CrDirReader"),
    "CrH5Reader": (".readers", "CrH5Reader"),
    "CrReader": (".readers", "CrReader"),
    "CrToZarr": (".writers", "CrToZarr"),
    "DataStore": (".datastore.datastore", "DataStore"),
    "DataStoreSummary": (".datastore.summary", "DataStoreSummary"),
    "DataStoreMerge": (".merge", "DataStoreMerge"),
    "EnrichmentResult": (".features.enrichment", "EnrichmentResult"),
    "FateMappingResult": (".trajectory.results", "FateMappingResult"),
    "GffReader": (".features.genomic.gff", "GffReader"),
    "H5adInspectResult": (".readers", "H5adInspectResult"),
    "H5adReader": (".readers", "H5adReader"),
    "H5adToZarr": (".writers", "H5adToZarr"),
    "LoomReader": (".readers", "LoomReader"),
    "LoomToZarr": (".writers", "LoomToZarr"),
    "MtxReader": (".readers", "MtxReader"),
    "MtxToZarr": (".writers", "MtxToZarr"),
    "SeuratImportResult": (".writers", "SeuratImportResult"),
    "SeuratInspectResult": (".readers", "SeuratInspectResult"),
    "SeuratReader": (".readers", "SeuratReader"),
    "SeuratToZarr": (".writers", "SeuratToZarr"),
    "MappingReference": (".mapping.reference", "MappingReference"),
    "MappingResult": (".mapping.models", "MappingResult"),
    "mount_datastore": (".datastore.datastore", "mount_datastore"),
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
    "clean_array": (".utils", "clean_array"),
    "configure_output": (".utils", "configure_output"),
    "controlled_compute": (".utils", "controlled_compute"),
    "coordinate_melding": (".features.genomic.melding", "coordinate_melding"),
    "create_zarr_count_assay": (".writers", "create_zarr_count_assay"),
    "create_zarr_dataset": (".writers", "create_zarr_dataset"),
    "create_zarr_obj_array": (".writers", "create_zarr_obj_array"),
    "dask_to_zarr": (".writers", "dask_to_zarr"),
    "get_log_level": (".utils", "get_log_level"),
    "inspect_h5ad": (".readers", "inspect_h5ad"),
    "inspect_mtx": (".readers", "inspect_mtx"),
    "inspect_seurat": (".readers", "inspect_seurat"),
    "load_zarr": (".utils", "load_zarr"),
    "logger": (".utils", "logger"),
    "permute_into_chunks": (".utils", "permute_into_chunks"),
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
