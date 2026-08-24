from collections.abc import Iterator
from typing import Any

import numpy as np

from ..storage.types import as_zarr_group
from ..readers import CSVReader
from ..storage.count_matrix import CountMatrixPolicy
from ..storage.io_policy import StorageIoPolicy
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
        workspace: Workspace name in the destination store. None uses the
                   legacy layout without a workspace group.
        dtype: the dtype of the data.
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
        policy: CountMatrixPolicy | None = None,
        io: StorageIoPolicy | None = None,
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
        self.policy = policy
        self.io = io
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
            policy=policy,
        )

    def dump(self) -> None:
        """Writes the count values into the Zarr matrix.

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
            io=self.io,
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
