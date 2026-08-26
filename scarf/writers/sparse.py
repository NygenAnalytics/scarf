from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix

from ..storage.count_matrix import CountMatrixPolicy
from ..storage.io_policy import StorageIoPolicy
from ..storage.profiles import (
    StorageProfile,
    ZarrLocation,
    resolve_storage_profile,
)
from ..storage.sharding import accumulate_sparse_to_shards
from ..utils.logging import logger
from ..utils.progress import iter_progress


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
    ) -> None:
        from ..storage.budget import resolve_budget
        from ..storage.schema import (
            create_cell_data,
            create_zarr_count_assay,
            validate_assay_name,
        )
        from ..storage.stores import load_zarr

        self.mat = csr_mat
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
        from ..storage.sharding import (
            resolve_sparse_import_batch,
            sparse_matrix_bytes,
        )

        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive")
        store = load_count_array(self.z, self.assayName, self.workspace)
        resident_bytes = sparse_matrix_bytes(self.mat)
        indptr = np.asarray(self.mat.indptr)

        def max_window_nnz(window_rows: int) -> int:
            width = min(window_rows, self.nCells)
            if width == 0:
                return 0
            return int(np.max(indptr[width:] - indptr[:-width]))

        plan = resolve_sparse_import_batch(
            (store,),
            nRows=self.nCells,
            resources=self.resources,
            maxWindowNnz=max_window_nnz,
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


def bed_to_sparse_array(
    bed_fn: str,
    bin_size: int,
    chrom_sizes: dict[str, int],
    min_counts_per_cell: int = 500,
    read_chunk_size: float = 1e6,
    sep: str = "\t",
    chrom_col: int = 0,
    start_col: int = 1,
    end_col: int = 2,
    barcode_col: int = 3,
    count_col: int = 4,
    comments_startswith: str = "#",
    disable_tqdm: bool = False,
    chrom_modifier: Any = None,
) -> tuple[csr_matrix, pd.Series, pd.Series]:
    """

    Args:
        bed_fn:
        bin_size:
        chrom_sizes:
        min_counts_per_cell:
        read_chunk_size:
        sep:
        chrom_col:
        start_col:
        end_col:
        barcode_col:
        count_col:
        comments_startswith:
        disable_tqdm:
        chrom_modifier:

    Returns:

    """
    import gc

    def feat_mapper(x: str) -> int:
        return feat_idx.get(x, n_feats)

    def default_chrom_modifier(x: str) -> str:
        return x + "_"

    feat_idx: dict[str, int] = {}
    for i in iter_progress(
        chrom_sizes,
        disable=disable_tqdm,
        desc="Calculating bin indices",
        total=len(chrom_sizes),
    ):
        for j in range((chrom_sizes[i] // bin_size) + 1):
            feat_idx[f"{i}_{j}"] = len(feat_idx)
    cell_idx: dict[Any, int] = {}
    mat_chunks: list[np.ndarray] = []
    n_feats = len(feat_idx)
    if chrom_modifier is None:
        chrom_modifier = default_chrom_modifier

    stream = pd.read_csv(
        bed_fn,
        sep=sep,
        header=None,
        comment=comments_startswith,
        usecols=[chrom_col, start_col, end_col, barcode_col, count_col],
        chunksize=int(read_chunk_size),
    )
    for df in iter_progress(
        stream, disable=disable_tqdm, desc="Building in memory sparse matrix"
    ):
        df[chrom_col] = df[chrom_col].map(chrom_modifier) + (
            (df[start_col] + (df[end_col] - df[start_col]) // 2).values // bin_size
        ).astype(str)
        for i in df[barcode_col].unique():
            if i not in cell_idx:
                cell_idx[i] = len(cell_idx)
        mat_chunks.append(
            np.vstack(
                [
                    np.fromiter(map(cell_idx.get, df[barcode_col].values), dtype=int),
                    np.fromiter(map(feat_mapper, df[chrom_col].values), dtype=int),
                    df[count_col].values,
                ]
            ).T
        )
    mat_arr = np.vstack(mat_chunks)
    gc.collect()
    mat = csr_matrix(
        (mat_arr[:, 2], (mat_arr[:, 0], mat_arr[:, 1])),
        shape=(len(cell_idx), n_feats + 1),
    )
    gc.collect()
    idx = np.array(mat.sum(axis=1))[:, 0] > min_counts_per_cell
    return mat[idx, :-1], pd.Series(cell_idx.keys())[idx], pd.Series(feat_idx.keys())
