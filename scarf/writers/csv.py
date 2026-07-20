from collections.abc import Iterator
from typing import Any

import numpy as np

from ..storage.types import as_zarr_group
from ..readers import CSVReader
from ..storage.sharding import write_dense_from_row_batches
from ..storage.stores import ZARRLOC


class CSVtoZarr:
    """A class for converting data from CSV format to a Zarr hierarchy.

    Args:
        cr: A CSVReader object
        zarr_loc: The file name for the Zarr hierarchy.
        assay_name: A label for the assay. Ex. "RNA" or "ATAC"
        chunk_size: The requested size of chunks to load into memory and process.
        dtype: the dtype of the data.

    Attributes:
        csvr: A CSVReader object
        fn: The file name for the Zarr hierarchy.
        chunkSizes: The requested size of chunks to store in Zarr file
        z: The Zarr hierarchy (array or group).
    """

    def __init__(
        self,
        cr: CSVReader,
        zarr_loc: ZARRLOC,
        assay_name: str,
        chunk_size: tuple[int, int] = (1000, 1000),
        workspace: str | None = None,
        dtype: np.dtype | None = None,
        storage_options: dict[str, Any] | None = None,
    ) -> None:
        from . import (
            create_cell_data,
            create_zarr_count_assay,
            load_zarr,
        )

        self.csvr = cr
        self.assayName = assay_name
        self.chunkSizes = chunk_size
        self.workspace = workspace
        self.storage_options = storage_options
        self.z = load_zarr(zarr_loc, mode="w", storage_options=storage_options)
        if dtype is not None:
            self.dtype = dtype
        else:
            self.dtype = next(self.csvr.consume())[0].dtype
        cell_ids = self.csvr.cell_ids()
        _ = create_cell_data(
            z=self.z,
            workspace=workspace,
            ids=cell_ids,
            names=cell_ids,
        )
        create_zarr_count_assay(
            z=self.z,
            assay_name=self.assayName,
            workspace=workspace,
            chunk_size=chunk_size,
            n_cells=self.csvr.nCells,
            feat_ids=self.csvr.feature_ids(),
            feat_names=self.csvr.feature_ids(),
            dtype=str(self.dtype),
        )

    def dump(self) -> None:
        """Writes the count values into the Zarr matrix.

        Args:

        Raises:
            AssertionError: Catches eventual bugs in the class, if number of cells does not match after transformation.

        Returns:
            None
        """
        from . import (
            create_zarr_obj_array,
            finalize_writer_counts,
            load_count_store,
        )

        store = load_count_store(self.z, self.assayName, self.workspace)
        cell_data_grp = as_zarr_group(self.z["cellData"], name="cellData")
        cell_data = [
            create_zarr_obj_array(
                cell_data_grp, name=x, data=None, dtype=y, shape=self.csvr.nCells
            )
            for x, y in zip(self.csvr.cellDataCols, self.csvr.cellDataDtypes or [])
        ]

        def count_batches() -> Iterator[np.ndarray]:
            s = 0
            for a, c in self.csvr.consume():
                e = s + a.shape[0]
                if c is not None:
                    for n, i in enumerate(c.T):
                        cell_data[n][s:e] = i
                s = e
                if self.dtype is not None:
                    yield a.astype(self.dtype)
                else:
                    yield a

        e = write_dense_from_row_batches(
            store,
            count_batches(),
            msg="Writing CSV counts",
        )
        if e != self.csvr.nCells:
            raise AssertionError(
                "ERROR: This is a bug in CSVtoZarr. All cells might not have been successfully "
                "written into the zarr file. Please report this issue"
            )
        finalize_writer_counts(self.z, self.assayName, self.workspace)
