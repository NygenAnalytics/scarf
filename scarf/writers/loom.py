from typing import Any

import numpy as np

from ..storage.types import as_zarr_group
from ..readers import LoomReader
from ..storage.stores import ZARRLOC
from ..utils.logging import logger


class LoomToZarr:
    """A class for converting data in a Loom file to a Zarr hierarchy. Converts
    a Loom file read using scarf.LoomReader into Scarf's Zarr format.

    Args:
        loom: LoomReader object used to open Loom format file
        zarr_loc: Output Zarr filename with path
        assay_name: Name for the output assay. If not provided then automatically set to RNA
        chunk_size: Chunk size for the count matrix saved in Zarr file.

    Attributes:
        loom: A scarf.LoomReader object used to open Loom format file.
        fn: The file name for the Zarr hierarchy.
        chunkSizes: The requested size of chunks to load into memory and process.
        assayName: The Zarr hierarchy (array or group).
        z: The Zarr hierarchy (array or group).
    """

    def __init__(
        self,
        loom: LoomReader,
        zarr_loc: ZARRLOC,
        assay_name: str | None = None,
        workspace: str | None = None,
        chunk_size: tuple[int, int] = (1000, 1000),
        storage_options: dict[str, Any] | None = None,
    ) -> None:
        from . import create_zarr_count_assay, load_zarr

        # TODO: support for multiple assay. Data from within individual layers can be treated as separate assays
        self.loom = loom
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
        self.z = load_zarr(zarr_loc, mode="w", storage_options=storage_options)
        self._ini_cell_data()
        create_zarr_count_assay(
            z=self.z,
            assay_name=self.assayName,
            workspace=workspace,
            chunk_size=chunk_size,
            n_cells=self.loom.nCells,
            feat_ids=self.loom.feature_ids(),
            feat_names=self.loom.feature_names(),
            dtype=self.loom.matrixDtype,
        )
        self._ini_feature_data()

    def _ini_cell_data(self) -> None:
        from . import create_cell_data, create_zarr_obj_array

        ids = np.array(self.loom.cell_ids())
        cell_group = create_cell_data(
            z=self.z,
            workspace=self.workspace,
            ids=ids,
            names=ids,
        )
        for i, j in self.loom.get_cell_attrs():
            try:
                create_zarr_obj_array(cell_group, i, j, j.dtype)
            except UnicodeDecodeError:
                logger.warning(f"Could not import {i} cell(column) attribute")

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
        for i, j in self.loom.get_feature_attrs():
            create_zarr_obj_array(feat_group, i, j, j.dtype)

    def dump(self, batch_size: int = 1000) -> None:
        """Write Loom matrix data into the Zarr counts array.

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
            data_stream=self.loom.consume(batch_size),
            n_cells=self.loom.nCells,
            batch_size=batch_size,
        )
        if total_cells_written != self.loom.nCells:
            raise AssertionError(
                "ERROR: This is a bug in LoomToZarr. All cells might not have been successfully "
                "written into the zarr file. Please report this issue"
            )
        finalize_writer_counts(self.z, self.assayName, self.workspace)
