from typing import TYPE_CHECKING, Any, cast

from ..storage.types import ZarrMode
from ..assay import Assay
from ..storage.profiles import StorageProfile, ZarrLocation
from ..storage.stores import create_matrix_source
from ._operations.features import _FeatureOperationsMixin
from ._operations.integration_metrics import _IntegrationMetricsOperationsMixin
from ._operations.presentation import _PresentationOperationsMixin
from ._operations.quality_control import _QualityControlOperationsMixin
from ._operations.trajectory import _TrajectoryFeatureOperationsMixin
from .mapping_datastore import MappingDatastore

if TYPE_CHECKING:
    from .pipeline_accessor import PipelineAccessor
    from .plot_accessor import DataStorePlotAccessor

__all__ = ["DataStore", "mount_datastore"]


def mount_datastore(
    source: str,
    at: ZarrLocation,
    *,
    workspace: str | None = None,
    storage_options: dict[str, Any] | None = None,
    **datastore_options: Any,
) -> "DataStore":
    """Create a writable DataStore whose count matrices come from ``source``.

    The target store receives copied cell and feature metadata plus all new
    analysis artifacts. Count matrices remain in the read-only source and are
    remounted automatically when the target is reopened with ``DataStore``.
    """
    if "zarr_loc" in datastore_options:
        raise TypeError("mount_datastore takes the target location through 'at'")
    zarr_mode = datastore_options.pop("zarr_mode", "r+")
    if zarr_mode != "r+":
        raise ValueError(
            f"A mounted datastore is writable and needs zarr_mode 'r+', got "
            f"{zarr_mode!r}. Reopen the target with DataStore to read it."
        )

    create_matrix_source(
        source,
        at,
        workspace=workspace,
        storage_options=storage_options,
    )
    return DataStore(
        at,
        zarr_mode=zarr_mode,
        workspace=workspace,
        storage_options=storage_options,
        **datastore_options,
    )


class DataStore(
    _QualityControlOperationsMixin,
    _FeatureOperationsMixin,
    _TrajectoryFeatureOperationsMixin,
    _IntegrationMetricsOperationsMixin,
    _PresentationOperationsMixin,
    MappingDatastore,
):
    """This class extends MappingDatastore and consequently inherits methods of
    all the other DataStore classes.

    DataStore is the primary interface for filtering cells, selecting features,
    building graphs, mapping datasets, finding markers, aggregating cells, and
    exporting data. Store-backed plots are available through `DataStore.plots`;
    the same functions remain available through `scarf.plotting`.

    Args:
        zarr_loc: Path to Zarr file created using one of writer functions of Scarf.
        assay_types: A dictionary with keys as assay names present in the Zarr file and values as either one of:
                     'RNA', 'ADT', 'ATAC' or 'GeneActivity'.
        default_assay: Name of assay that should be considered as default. It is mandatory to provide this value
                       when DataStore loads a Zarr file for the first time.
        min_features_per_cell: Minimum number of non-zero features in a cell. If lower than this then the cell
                               will be filtered out.
        min_cells_per_feature: Minimum number of cells where a feature has a non-zero value. Genes with values
                               less than this will be filtered out.
        mito_pattern: Regex pattern to capture mitochondrial genes. When None, uses ``MT-|mt``.
        ribo_pattern: Regex pattern to capture ribosomal genes. When None, uses
                      ``RPS|RPL|MRPS|MRPL``.
        nthreads: Number of maximum threads to use in all multi-threaded functions
        zarr_mode: For read-write mode use r+' or for read-only use 'r'. (Default value: 'r+')
        workspace: Workspace for the data
        mem_budget: Memory budget bounding streaming and concurrency. Accepts bytes, a suffixed size
                    (e.g. '8G'), or a fraction of total system memory (e.g. '0.6'). When None, it is
                    auto-detected (SCARF_MEM_BUDGET env var, else total system memory). Override it to
                    simulate reading on a machine with a different memory size than the writer.
    """

    def __init__(
        self,
        zarr_loc: ZarrLocation,
        assay_types: dict[str, str] | None = None,
        default_assay: str | None = None,
        min_features_per_cell: int = 10,
        min_cells_per_feature: int = 20,
        mito_pattern: str | None = None,
        ribo_pattern: str | None = None,
        nthreads: int = 2,
        zarr_mode: ZarrMode = "r+",
        workspace: str | None = None,
        zarrProfile: StorageProfile | None = None,
        storage_options: dict[str, Any] | None = None,
        mem_budget: int | str | None = None,
    ) -> None:
        from ..storage.budget import resolve_budget
        from ..storage.profiles import resolve_storage_profile

        resources = resolve_budget(memory=mem_budget, workers=nthreads)
        profile = resolve_storage_profile(zarr_loc, zarrProfile)
        if zarr_mode not in ["r", "r+"]:
            raise ValueError(
                "ERROR: Zarr file can only be accessed using either 'r' or 'r+' mode"
            )
        super().__init__(
            zarr_loc=zarr_loc,
            assay_types=assay_types,
            default_assay=default_assay,
            min_features_per_cell=min_features_per_cell,
            min_cells_per_feature=min_cells_per_feature,
            mito_pattern=mito_pattern,
            ribo_pattern=ribo_pattern,
            zarr_mode=zarr_mode,
            workspace=workspace,
            resources=resources,
            storage_profile=profile,
            storage_options=storage_options,
        )

    @property
    def plots(self) -> "DataStorePlotAccessor":
        """Return store-bound equivalents of store-first plotting functions."""
        from .plot_accessor import DataStorePlotAccessor

        return DataStorePlotAccessor(self)

    @property
    def pipeline(self) -> "PipelineAccessor":
        """Return the store-bound analysis recipe runner."""
        from .pipeline_accessor import PipelineAccessor

        return PipelineAccessor(self)

    def get_assay(self, assay_name: str) -> Assay:
        """Returns the assay object for the given assay name.

        Args:
            assay_name: Name of the assay to be returned.

        Returns:
            Assay object
        """
        if assay_name not in self.assay_names:
            raise ValueError(f"ERROR: Assay {assay_name} not found in the Zarr file")
        else:
            return cast(Assay, getattr(self, assay_name))

    def _create_temporary_datastore(
        self,
        zarr_loc: ZarrLocation,
        *,
        default_assay: str,
        assay_types: dict[str, str],
        nthreads: int,
    ) -> "DataStore":
        return DataStore(
            zarr_loc,
            default_assay=default_assay,
            assay_types=assay_types,
            nthreads=nthreads,
            mem_budget=self.memoryBytes,
        )
