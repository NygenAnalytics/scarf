from collections.abc import Iterator
from typing import Any

import numpy as np

from ..storage.types import as_zarr_group
from ..readers import CSVReader
from ..storage.profiles import (
    StorageProfile,
    ZarrLocation,
    resolve_storage_profile,
)
from ..storage.sharding import write_dense_from_row_batches
from ..utils.logging import logger


class CSVtoZarr:
    """A class for converting data from CSV format to a Zarr hierarchy.

    Args:
        cr: A CSVReader object
        zarr_loc: The file name for the Zarr hierarchy.
        assay_name: A label for the assay. Ex. "RNA" or "ATAC"
        dtype: the dtype of the data.

    Attributes:
        csvr: A CSVReader object
        fn: The file name for the Zarr hierarchy.
        z: The Zarr hierarchy (array or group).
    """

    def __init__(
        self,
        cr: CSVReader,
        zarr_loc: ZarrLocation,
        assay_name: str,
        workspace: str | None = None,
        dtype: np.dtype | None = None,
        storage_options: dict[str, Any] | None = None,
        mem_budget: int | str | None = None,
        nthreads: int | None = None,
        profile: StorageProfile | None = None,
        targetChunkBytes: int | None = None,
        targetShardBytes: int | None = None,
    ) -> None:
        from ..storage.budget import resolve_budget
        from ..storage.schema import (
            create_cell_data,
            create_zarr_count_assay,
            validate_assay_name,
        )
        from ..storage.stores import load_zarr

        self.csvr = cr
        self.assayName = assay_name
        validate_assay_name(self.assayName)
        self.resources = resolve_budget(mem_budget, nthreads)
        self.profile = resolve_storage_profile(zarr_loc, profile)
        self.workspace = workspace
        self.storage_options = storage_options
        self.z = load_zarr(zarr_loc, mode="w", storage_options=storage_options)
        if dtype is not None:
            self.dtype = dtype
        else:
            self.dtype = next(self.csvr.consume())[0].dtype
        cell_ids = self.csvr.cell_ids()
        _ = create_cell_data(
            root=self.z,
            workspace=workspace,
            ids=cell_ids,
            names=cell_ids,
            profile=self.profile,
        )
        create_zarr_count_assay(
            z=self.z,
            assay_name=self.assayName,
            workspace=workspace,
            n_cells=self.csvr.nCells,
            feat_ids=self.csvr.feature_ids(),
            feat_names=self.csvr.feature_ids(),
            dtype=str(self.dtype),
            profile=self.profile,
            targetChunkBytes=targetChunkBytes,
            targetShardBytes=targetShardBytes,
        )

    def dump(self) -> None:
        """Writes the count values into the Zarr matrix.

        Args:

        Raises:
            AssertionError: Catches eventual bugs in the class, if number of cells does not match after transformation.

        Returns:
            None
        """
        from ..storage.arrays import create_zarr_obj_array
        from ..storage.schema import load_count_array

        store = load_count_array(self.z, self.assayName, self.workspace)
        cell_data_path = (
            "cellData" if self.workspace is None else f"{self.workspace}/cellData"
        )
        cell_data_grp = as_zarr_group(
            self.z[cell_data_path],
            name=cell_data_path,
        )
        cell_data = [
            create_zarr_obj_array(
                cell_data_grp,
                name=x,
                data=None,
                dtype=y,
                shape=self.csvr.nCells,
                profile=self.profile,
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
            resources=self.resources,
        )
        if e != self.csvr.nCells:
            raise AssertionError(
                "ERROR: This is a bug in CSVtoZarr. All cells might not have been successfully "
                "written into the zarr file. Please report this issue"
            )
        logger.info(
            f"Wrote {self.csvr.nCells} cells and {self.csvr.nFeatures} features "
            f"from CSV to assay {self.assayName}"
        )
