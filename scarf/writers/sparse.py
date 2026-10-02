from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from ..storage.count_dtype import count_storage_dtype
from ..storage.count_matrix import CountMatrixPolicy
from ..storage.identity import CountSummary, finalize_counts
from ..storage.io_policy import StorageIoPolicy
from ..storage.profiles import (
    StorageProfile,
    ZarrLocation,
    resolve_storage_profile,
)
from ..storage.sharding import accumulate_sparse_to_shards
from ..utils.arrays import max_window_nnz, sparse_matrix_bytes
from ..utils.count_values import compressed_count_ranges
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
        storage_options: Backend options passed when opening the Zarr store.
        mem_budget: Memory available to the conversion. Accepts bytes, a
                    suffixed size (e.g. '8G'), or a fraction of total system memory (e.g. '0.6').
        nthreads: Worker count for write-time concurrency. When None, auto-detected.
        profile: Zarr encoding profile (``fast_local`` or ``cloud``). When
                 None, chosen from the destination location.
        policy: Count-matrix geometry policy, used exactly. When None, the
                default policy is used with unitBytes and chunkBytes halved
                together until the counts write and the countsT transpose
                fit ``mem_budget``. Either way, an import that does not fit
                raises MemoryError before the destination is created.
        io: Optional explicit read, compute, and write widths. Unset values
            stay under automatic planning.
        assay_type: Preset assay type, such as ``RNA``, for an assay whose name
                    is not a preset. When None, the assay name decides the type.

    The counts are stored in the dtype that
    :func:`~scarf.storage.count_dtype.count_storage_dtype` resolves from the
    canonical (duplicate-summed) values of the matrix.

    Raises:
        ValueError: Raised if number of input cell or feature IDs does not match
            the matrix, or if a count is NaN or infinite.

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
        self.io = io
        self.workspace = workspace
        self.storage_options = storage_options
        cell_ids = np.array(cell_ids)
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
        # The scan reads windows beside the matrix it holds.
        held = sparse_matrix_bytes(self.mat)
        if held >= self.resources.memoryBytes:
            raise MemoryError(
                f"The sparse matrix holds {held} bytes, but mem_budget is "
                f"{self.resources.memoryBytes} bytes. Increase mem_budget."
            )
        (value_range,) = compressed_count_ranges(
            self.mat.indptr,
            self.mat.indices,
            self.mat.data,
            minorSize=self.nFeatures,
            maxBytes=self.resources.memoryBytes - held,
        )
        storage_dtype = count_storage_dtype(self.mat.dtype, value_range)
        # A layout that does not fit fails here, before the destination exists.
        layout = self._fit_count_layout(storage_dtype, policy)

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
            dtype=storage_dtype,
            profile=self.profile,
            policy=layout,
        )

    def _count_import_requirements(self) -> tuple[int, Callable[[int], int]]:
        """Return the resident bytes and the window entries of the counts write.

        The layout fit and the write plan the same import from these.
        """
        indptr = np.asarray(self.mat.indptr)
        resident = sparse_matrix_bytes(self.mat) + CountSummary.nbytes_for(
            self.nCells, self.nFeatures
        )
        return resident, lambda rows: max_window_nnz(indptr, rows)

    def _fit_count_layout(
        self, storage_dtype: Any, requested: CountMatrixPolicy | None
    ) -> CountMatrixPolicy:
        """Return the count layout whose import and ``countsT`` fit the budget."""
        from ..storage.sharding import fit_count_layout, sparse_counts_admission
        from .counts_t import counts_t_assays

        resident, window_nnz = self._count_import_requirements()
        assay_types = {} if self.assayType is None else {self.assayName: self.assayType}
        return fit_count_layout(
            {self.assayName: (self.nFeatures, storage_dtype)},
            nCells=self.nCells,
            profile=self.profile,
            memoryBytes=self.resources.memoryBytes,
            transposed=counts_t_assays((self.assayName,), assay_types),
            admitCounts=sparse_counts_admission(
                nRows=self.nCells,
                maxWindowNnz=window_nnz,
                sourceDtype=self.mat.dtype,
                residentBytes=resident,
            ),
            requested=requested,
        )

    def dump(self, batch_size: int | None = None) -> None:
        """Write out the data matrix into the Zarr hierarchy.

        Args:
            batch_size: Number of source cells per batch, at most one
                        destination row band, which is the default.

        Raises:
            ValueError: If ``batch_size`` is not positive.

        Returns:
            None
        """
        from ..storage.schema import load_count_array
        from ..storage.sharding import resolve_sparse_import_batch

        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive")
        store = load_count_array(self.z, self.assayName, self.workspace)
        summary = CountSummary(store)
        resident_bytes, window_nnz = self._count_import_requirements()
        plan = resolve_sparse_import_batch(
            (store,),
            nRows=self.nCells,
            resources=self.resources,
            maxWindowNnz=window_nnz,
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
            for start in range(0, self.nCells, resolved_batch_rows):
                yield self.mat[start : start + resolved_batch_rows].tocoo()

        accumulate_sparse_to_shards(
            store,
            row_batches(),
            resources=self.resources,
            residentBytes=resident_bytes,
            producerReserveBytes=plan.producerReserveBytes,
            msg="Writing sparse counts",
            io=self.io,
            countSummary=summary,
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
            io=self.io,
        )
