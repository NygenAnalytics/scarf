from typing import Any

import numpy as np

from ..storage.types import as_zarr_group
from ..readers import LoomReader
from ..storage.profiles import (
    StorageProfile,
    ZarrLocation,
    resolve_storage_profile,
)
from ..utils.logging import logger


class LoomToZarr:
    """A class for converting data in a Loom file to a Zarr hierarchy. Converts
    a Loom file read using scarf.LoomReader into Scarf's Zarr format.

    Args:
        loom: LoomReader object used to open Loom format file
        zarr_loc: Output Zarr filename with path
        assay_name: Name for the output assay. If not provided then automatically set to RNA

    Attributes:
        loom: A scarf.LoomReader object used to open Loom format file.
        fn: The file name for the Zarr hierarchy.
        assayName: The Zarr hierarchy (array or group).
        z: The Zarr hierarchy (array or group).
    """

    def __init__(
        self,
        loom: LoomReader,
        zarr_loc: ZarrLocation,
        assay_name: str | None = None,
        workspace: str | None = None,
        storage_options: dict[str, Any] | None = None,
        mem_budget: int | str | None = None,
        nthreads: int | None = None,
        profile: StorageProfile | None = None,
        targetChunkBytes: int | None = None,
        targetShardBytes: int | None = None,
    ) -> None:
        from ..storage.budget import resolve_budget
        from ..storage.schema import create_zarr_count_assay, validate_assay_name
        from ..storage.stores import load_zarr

        # TODO: support for multiple assay. Data from within individual layers can be treated as separate assays
        self.loom = loom
        self.resources = resolve_budget(mem_budget, nthreads)
        self.profile = resolve_storage_profile(zarr_loc, profile)
        self.workspace = workspace
        self.storage_options = storage_options
        if assay_name is None:
            logger.debug("Using RNA as the default assay name")
            self.assayName = "RNA"
        else:
            self.assayName = assay_name
        validate_assay_name(self.assayName)
        self.z = load_zarr(zarr_loc, mode="w", storage_options=storage_options)
        self._ini_cell_data()
        create_zarr_count_assay(
            z=self.z,
            assay_name=self.assayName,
            workspace=workspace,
            n_cells=self.loom.nCells,
            feat_ids=self.loom.feature_ids(),
            feat_names=self.loom.feature_names(),
            dtype=self.loom.matrixDtype,
            profile=self.profile,
            targetChunkBytes=targetChunkBytes,
            targetShardBytes=targetShardBytes,
        )
        self._ini_feature_data()

    def _ini_cell_data(self) -> None:
        from ..storage.arrays import create_zarr_obj_array
        from ..storage.schema import create_cell_data

        ids = np.array(self.loom.cell_ids())
        cell_group = create_cell_data(
            root=self.z,
            workspace=self.workspace,
            ids=ids,
            names=ids,
            profile=self.profile,
        )
        for i, j in self.loom.get_cell_attrs():
            try:
                create_zarr_obj_array(
                    cell_group,
                    i,
                    j,
                    j.dtype,
                    profile=self.profile,
                )
            except UnicodeDecodeError:
                logger.warning(f"Could not import {i} cell(column) attribute")

    def _ini_feature_data(self) -> None:
        from ..storage.arrays import create_zarr_obj_array

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
        for i, j in self.loom.get_feature_attrs():
            create_zarr_obj_array(
                feat_group,
                i,
                j,
                j.dtype,
                profile=self.profile,
            )

    def dump(self, batch_size: int = 1000) -> None:
        """Write Loom matrix data into the Zarr counts array.

        Args:
            batch_size: Number of cells read from the source per batch.

        Raises:
            AssertionError: If written cell count does not match expected nCells.

        Returns:
            None
        """
        from ..storage.layout import array_shard_rows
        from ..storage.sharding import (
            accumulate_sparse_to_shards,
            sparse_producer_peak_bytes,
        )
        from ..storage.schema import load_count_array

        store = load_count_array(self.z, self.assayName, self.workspace)
        source_rows = min(batch_size, self.loom.nCells)
        buffered_rows = min(
            self.loom.nCells,
            batch_size + array_shard_rows(store),
        )
        source_nnz = source_rows * self.loom.nFeatures
        buffered_nnz = buffered_rows * self.loom.nFeatures
        value_bytes = max(
            np.dtype(self.loom.sourceMatrixDtype).itemsize,
            np.dtype(self.loom.matrixDtype).itemsize,
            np.dtype(store.dtype).itemsize,
        )
        dense_source_bytes = source_nnz * np.dtype(self.loom.sourceMatrixDtype).itemsize
        total_cells_written = accumulate_sparse_to_shards(
            store,
            self.loom.consume(batch_size),
            resources=self.resources,
            producerReserveBytes=sparse_producer_peak_bytes(
                buffered_nnz,
                source_nnz,
                value_bytes,
            )
            + dense_source_bytes,
            msg="Writing Loom counts",
        )
        if total_cells_written != self.loom.nCells:
            raise AssertionError(
                "ERROR: This is a bug in LoomToZarr. All cells might not have been successfully "
                "written into the zarr file. Please report this issue"
            )
        logger.info(
            f"Wrote {self.loom.nCells} cells and {self.loom.nFeatures} features "
            f"from Loom to assay {self.assayName}"
        )
