from typing import Any

from ..storage.types import as_zarr_group
from ..readers import H5adReader
from ..storage.stores import ZARRLOC
from ..utils.logging import logger


class H5adToZarr:
    """A class for converting data in anndata's H5ad format to Zarr hierarchy.

    Args:
        h5ad: A H5adReader object, containing the Cellranger data.
        zarr_loc: The file name for the Zarr hierarchy or a store
        assay_name: the name of the assay (e. g. 'RNA')
        workspace: An optional workspace id.
        chunk_size: The requested size of chunks to load into memory and process.
        mem_budget: Memory budget driving write-time chunk and shard geometry. Accepts bytes, a
                    suffixed size (e.g. '8G'), or a fraction of total system memory (e.g. '0.6').
                    Set it to simulate writing on a machine with a different memory size.
        nthreads: Worker count for write-time concurrency. When None, auto-detected.
        working_copies: Number of concurrent in-memory working copies the memory budget is divided
                        across. When None, uses SCARF_WORKING_COPIES env var or the default.

    Attributes:
        h5ad: A h5ad object (h5 file with added AnnData structure).
        chunkSizes: The requested size of chunks to load into memory and process.
        assayName: The Zarr hierarchy (array or group).
        z: The Zarr hierarchy (array or group).
    """

    def __init__(
        self,
        h5ad: H5adReader,
        zarr_loc: ZARRLOC,
        assay_name: str | None = None,
        workspace: str | None = None,
        chunk_size: tuple[int, int] = (1000, 1000),
        storage_options: dict[str, Any] | None = None,
        mem_budget: int | str | None = None,
        nthreads: int | None = None,
        working_copies: int | None = None,
        targetChunkBytes: int | None = None,
        minFeatureChunk: int | None = None,
        maxFeatureChunk: int | None = None,
    ) -> None:
        from . import (
            _apply_budget_override,
            create_zarr_count_assay,
            load_zarr,
        )

        # TODO: support for multiple assay. One of the `var` datasets can be used to group features in separate assays
        _apply_budget_override(mem_budget, nthreads, working_copies)
        self.h5ad = h5ad
        self.chunkSizes = chunk_size
        self.workspace = workspace
        self.storage_options = storage_options
        if assay_name is None:
            logger.info(
                "No value provided for assay names. Will use default value: 'RNA'"
            )
            self.assayName = "RNA"
        else:
            self.assayName = assay_name
        self.z = load_zarr(zarr_loc=zarr_loc, mode="w", storage_options=storage_options)
        self._ini_cell_data()
        create_zarr_count_assay(
            z=self.z,
            assay_name=self.assayName,
            workspace=workspace,
            chunk_size=chunk_size,
            n_cells=self.h5ad.nCells,
            feat_ids=self.h5ad.feat_ids(),
            feat_names=self.h5ad.feat_names(),
            dtype=self.h5ad.matrixDtype,
            targetChunkBytes=targetChunkBytes,
            minFeatureChunk=minFeatureChunk,
            maxFeatureChunk=maxFeatureChunk,
        )
        self._ini_feature_data()

    def _ini_cell_data(self) -> None:
        from . import create_cell_data, create_zarr_obj_array

        ids = self.h5ad.cell_ids()
        g = create_cell_data(
            z=self.z,
            workspace=self.workspace,
            ids=ids,
            names=ids,
        )
        for i, j in self.h5ad.get_cell_columns():
            create_zarr_obj_array(g, i, j, j.dtype)

    def _ini_feature_data(self) -> None:
        from . import create_zarr_obj_array

        if self.workspace is None:
            feat_group = as_zarr_group(
                self.z[f"{self.assayName}/featureData"],
                name=f"{self.assayName}/featureData",
            )
        else:
            feat_group = as_zarr_group(
                self.z[f"{self.workspace}/{self.assayName}/featureData"],
                name=f"{self.workspace}/{self.assayName}/featureData",
            )
        for i, j in self.h5ad.get_feat_columns():
            if i not in feat_group:
                create_zarr_obj_array(feat_group, i, j, j.dtype)

    def dump(self, batch_size: int = 1000) -> None:
        """Write h5ad matrix data into the Zarr counts array.

        Args:
            batch_size: Number of cells written per sparse_writer batch.

        Raises:
            AssertionError: If written cell count does not match expected nCells.

        Returns:
            None
        """
        from . import finalize_writer_counts, load_count_store, sparse_writer

        store = load_count_store(self.z, self.assayName, self.workspace)
        total_cells_written = sparse_writer(
            store=store,
            data_stream=self.h5ad.consume(batch_size),
            n_cells=self.h5ad.nCells,
            batch_size=batch_size,
        )
        if total_cells_written != self.h5ad.nCells:
            raise AssertionError(
                "ERROR: This is a bug in H5adToZarr. All cells might not have been successfully "
                "written into the zarr file. Please report this issue"
            )
        finalize_writer_counts(self.z, self.assayName, self.workspace)
