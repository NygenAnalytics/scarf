from collections.abc import Iterator
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from ..storage.count_matrix import CountMatrixPolicy
from ..storage.identity import CountSummary, finalize_counts
from ..storage.io_policy import StorageIoPolicy
from ..storage.profiles import (
    StorageProfile,
    ZarrLocation,
    resolve_storage_profile,
)
from ..storage.sharding import accumulate_sparse_to_shards
from ..utils.logging import logger


class SparseToZarr:
    """A class for converting data in a sparse matrix to a Zarr hierarchy.

    Args:
        csr_mat: A CSR format sparse matrix
        zarr_loc: Output Zarr filename with path
        cell_ids: Cell IDs for the cells in the dataset.
        feature_ids: Feature IDs for the features in the dataset.
        assay_name: Name for the output assay. If not provided then automatically set to RNA.
        workspace: Workspace name in the destination store. None uses the
                   legacy layout without a workspace group.
        feature_names: Optional display names aligned with ``feature_ids``.
        matrix_dtype: Storage dtype for counts. When None, the sparse matrix
                      dtype is used.
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
        assay_type: Preset assay type, such as ``RNA``, for an assay whose name
                    is not a preset. When None, the assay name decides the type.

    Raises:
        ValueError: Raised if number of input cell or feature IDs does not match the matrix.
        AssertionError: Catches eventual bugs in the class, if number of cells does not match after transformation.

    Attributes:
        mat: Input CSR matrix
        fn: The file name for the Zarr hierarchy.
        assayName: The Zarr hierarchy (array or group).
        z: The Zarr hierarchy (array or group).
    """

    def __init__(
        self,
        csr_mat: csr_matrix,
        zarr_loc: ZarrLocation,
        cell_ids: np.ndarray | list[str],
        feature_ids: np.ndarray | list[str],
        assay_name: str | None = None,
        workspace: str | None = None,
        feature_names: np.ndarray | list[str] | None = None,
        matrix_dtype: np.dtype | None = None,
        storage_options: dict[str, Any] | None = None,
        mem_budget: int | str | None = None,
        nthreads: int | None = None,
        profile: StorageProfile | None = None,
        policy: CountMatrixPolicy | None = None,
        io: StorageIoPolicy | None = None,
        assay_type: str | None = None,
    ) -> None:
        from ..storage.budget import resolve_budget
        from ..storage.schema import (
            create_cell_data,
            create_zarr_count_assay,
            validate_assay_name,
        )
        from ..storage.stores import load_zarr
        from .counts_t import validate_assay_type

        validate_assay_type(assay_type)
        self.mat = csr_mat
        self.assayType = assay_type
        self.resources = resolve_budget(mem_budget, nthreads)
        self.profile = resolve_storage_profile(zarr_loc, profile)
        self.policy = policy
        self.io = io
        self.workspace = workspace
        self.storage_options = storage_options
        cell_ids = np.array(cell_ids)
        if matrix_dtype is None:
            self.matrixDtype = self.mat.dtype
        else:
            self.matrixDtype = matrix_dtype
        if assay_name is None:
            logger.debug("Using RNA as the default assay name")
            self.assayName = "RNA"
        else:
            self.assayName = assay_name
        validate_assay_name(self.assayName)
        self.nCells, self.nFeatures = self.mat.shape
        if len(cell_ids) != self.nCells:
            raise ValueError(
                "ERROR: Number of cell ids are not same as number of cells in the matrix"
            )
        if len(feature_ids) != self.nFeatures:
            raise ValueError(
                "ERROR: Number of feature ids are not same as number of features in the matrix"
            )

        self.z = load_zarr(zarr_loc, mode="w", storage_options=storage_options)
        _ = create_cell_data(
            root=self.z,
            workspace=self.workspace,
            ids=cell_ids,
            names=cell_ids,
            profile=self.profile,
        )
        if feature_names is None:
            feature_names = feature_ids
        create_zarr_count_assay(
            z=self.z,
            assay_name=self.assayName,
            workspace=workspace,
            n_cells=self.nCells,
            feat_ids=feature_ids,
            feat_names=feature_names,
            dtype=str(self.matrixDtype),
            profile=self.profile,
            policy=policy,
        )

    def dump(self, batch_size: int | None = None) -> None:
        """Write out the data matrix into the Zarr hierarchy.

        Args:
            batch_size: Number of source cells per batch. By default, a
                        destination-aligned value is selected within the memory budget.

         Raises:
            ValueError: Raised if there is any unexpected errors when writing to the Zarr hierarchy.
            AssertionError: Catches eventual bugs in the class, if number of cells does not match after transformation.

        Returns:
            None
        """
        from ..storage.schema import load_count_array
        from ..storage.sharding import resolve_sparse_import_batch
        from ..utils.arrays import max_window_nnz, sparse_matrix_bytes

        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive")
        store = load_count_array(self.z, self.assayName, self.workspace)
        summary = CountSummary(store)
        resident_bytes = sparse_matrix_bytes(self.mat) + summary.nbytes
        indptr = np.asarray(self.mat.indptr)
        plan = resolve_sparse_import_batch(
            (store,),
            nRows=self.nCells,
            resources=self.resources,
            maxWindowNnz=lambda rows: max_window_nnz(indptr, rows),
            sourceDtype=self.mat.dtype,
            batchRows=batch_size,
            residentBytes=resident_bytes,
        )
        self._lastImportPlan = plan
        resolved_batch_rows = plan.batchRows
        logger.debug(
            f"Resolved sparse source batch rows={resolved_batch_rows} "
            f"write_tasks={plan.writeTasks}"
        )

        def row_batches() -> Iterator[coo_matrix]:
            s = 0
            for end in range(
                resolved_batch_rows,
                self.nCells + resolved_batch_rows,
                resolved_batch_rows,
            ):
                if s == self.nCells:
                    break
                if end > self.nCells:
                    end = self.nCells
                yield self.mat[s:end].tocoo()
                s = end

        e = accumulate_sparse_to_shards(
            store,
            row_batches(),
            resources=self.resources,
            residentBytes=resident_bytes,
            producerReserveBytes=plan.producerReserveBytes,
            msg="Writing sparse counts",
            io=self.io,
            countSummary=summary,
        )
        if e != self.nCells:
            raise AssertionError(
                "ERROR: This is a bug in SparseToZarr. All cells might not have been successfully "
                "written into the zarr file. Please report this issue"
            )
        logger.info(
            f"Wrote {self.nCells} cells and {self.nFeatures} features "
            f"to assay {self.assayName}"
        )
        finalize_counts(store, summary=summary)
        from .counts_t import finalize_writer_counts_t

        finalize_writer_counts_t(
            self.z,
            self.assayName,
            self.workspace,
            assay_type=self.assayType,
            resources=self.resources,
            profile=self.profile,
            policy=self.policy,
            io=self.io,
        )
