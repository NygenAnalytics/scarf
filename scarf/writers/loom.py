from typing import Any

import numpy as np

from ..storage.types import as_zarr_group
from ..readers import LoomReader
from ..storage.count_matrix import CountMatrixPolicy
from ..storage.io_policy import StorageIoPolicy
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
        workspace: Workspace name in the destination store. None uses the
                   legacy layout without a workspace group.
        storage_options: Backend options passed when opening the Zarr store.
        mem_budget: Memory available to the conversion. Accepts bytes, a
                    suffixed size (e.g. '8G'), or a fraction of total system memory (e.g. '0.6').
        nthreads: Worker count for write-time concurrency. When None, auto-detected.
        profile: Zarr encoding profile (``fast_local`` or ``cloud``). When
                 None, chosen from the destination location.
        policy: Count-matrix geometry policy. When None, the default
                unitBytes and chunkBytes plan is used.
        io: Optional explicit read, compute, and write widths. Unset values
            stay under automatic planning.

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
        policy: CountMatrixPolicy | None = None,
        io: StorageIoPolicy | None = None,
    ) -> None:
        from ..storage.budget import resolve_budget
        from ..storage.schema import create_zarr_count_assay, validate_assay_name
        from ..storage.stores import load_zarr

        # TODO: support for multiple assay. Data from within individual layers can be treated as separate assays
        self.loom = loom
        self.resources = resolve_budget(mem_budget, nthreads)
        self.profile = resolve_storage_profile(zarr_loc, profile)
        self.policy = policy
        self.io = io
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
            policy=policy,
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
            batch_size: Maximum number of cells read from the source per batch.

        Raises:
            AssertionError: If written cell count does not match expected nCells.

        Returns:
            None
        """
        from ..storage.budget import ResourceBudget
        from ..storage.partition import affordable_width
        from ..storage.sharding import _writer_count, write_dense_from_row_batches
        from ..storage.schema import load_count_array

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        store = load_count_array(self.z, self.assayName, self.workspace)
        matrix = self.loom.h5[self.loom.matrixKey]
        source_bytes = np.dtype(self.loom.sourceMatrixDtype).itemsize
        target_bytes = np.dtype(self.loom.matrixDtype).itemsize
        chunk_bytes = (
            0 if matrix.chunks is None else int(np.prod(matrix.chunks)) * source_bytes
        )
        cache_bytes = int(matrix.id.get_access_plist().get_chunk_cache()[1])
        if matrix.chunks is None:
            cache_bytes = 0
        else:
            n_chunks = int(
                np.prod(
                    [
                        (size + chunk - 1) // chunk
                        for size, chunk in zip(matrix.shape, matrix.chunks, strict=True)
                    ]
                )
            )
            cache_bytes = min(cache_bytes, n_chunks * chunk_bytes)

        def producer_bytes(rows: int) -> int:
            # A yielded batch may remain live while the next HDF5 slice is read.
            return int(
                2 * rows * self.loom.nFeatures * (source_bytes + target_bytes)
                + chunk_bytes
                + cache_bytes
            )

        def fits(rows: int) -> bool:
            remaining = self.resources.memoryBytes - producer_bytes(rows)
            if remaining < 1:
                return False
            try:
                _writer_count(
                    store,
                    ResourceBudget(remaining, self.resources.workers),
                    1,
                    io=self.io,
                )
            except MemoryError:
                return False
            return True

        rows = affordable_width(fits, min(batch_size, self.loom.nCells))
        if self.loom.nCells and rows == 0:
            raise MemoryError(
                "Loom import cannot fit one source row and one destination row band "
                "within mem_budget"
            )
        writer_resources = (
            ResourceBudget(
                self.resources.memoryBytes - producer_bytes(rows),
                self.resources.workers,
            )
            if rows
            else self.resources
        )
        total_cells_written = write_dense_from_row_batches(
            store,
            self.loom.consume_dense(max(1, rows)),
            resources=writer_resources,
            msg="Writing Loom counts",
            io=self.io,
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
        from .counts_t import finalize_writer_counts_t

        finalize_writer_counts_t(
            self.z,
            self.assayName,
            self.workspace,
            resources=self.resources,
            profile=self.profile,
            policy=self.policy,
            io=self.io,
        )
