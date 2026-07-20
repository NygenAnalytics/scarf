from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix

from ..storage.layout import array_shard_rows
from ..storage.sharding import accumulate_sparse_to_shards
from ..storage.stores import ZARRLOC
from ..utils.logging import logger
from ..utils.progress import tqdmbar


class SparseToZarr:
    """A class for converting data in a sparse matrix to a Zarr hierarchy.

    Args:
        csr_mat: A CSR format sparse matrix
        zarr_loc: Output Zarr filename with path
        cell_ids: Cell IDs for the cells in the dataset.
        feature_ids: Feature IDs for the features in the dataset.
        assay_name: Name for the output assay. If not provided then automatically set to RNA.
        chunk_size: The requested size of chunks to load into memory and process.

    Raises:
        ValueError: Raised if number of input cell or feature IDs does not match the matrix.
        AssertionError: Catches eventual bugs in the class, if number of cells does not match after transformation.

    Attributes:
        mat: Input CSR matrix
        fn: The file name for the Zarr hierarchy.
        chunkSizes: The requested size of chunks to load into memory and process.
        assayName: The Zarr hierarchy (array or group).
        z: The Zarr hierarchy (array or group).
    """

    def __init__(
        self,
        csr_mat: csr_matrix,
        zarr_loc: ZARRLOC,
        cell_ids: np.ndarray | list[str],
        feature_ids: np.ndarray | list[str],
        assay_name: str | None = None,
        workspace: str | None = None,
        feature_names: np.ndarray | list[str] | None = None,
        chunk_size: tuple[int, int] = (1000, 1000),
        matrix_dtype: np.dtype | None = None,
        storage_options: dict[str, Any] | None = None,
    ) -> None:
        from . import (
            create_cell_data,
            create_zarr_count_assay,
            load_zarr,
        )

        self.mat = csr_mat
        self.chunkSizes = chunk_size
        self.workspace = workspace
        self.storage_options = storage_options
        cell_ids = np.array(cell_ids)
        if matrix_dtype is None:
            self.matrixDtype = self.mat.dtype
        else:
            self.matrixDtype = matrix_dtype
        if assay_name is None:
            logger.info(
                "No value provided for assay names. Will use default value: 'RNA'"
            )
            self.assayName = "RNA"
        else:
            self.assayName = assay_name
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
            z=self.z,
            workspace=self.workspace,
            ids=cell_ids,
            names=cell_ids,
        )
        if feature_names is None:
            feature_names = feature_ids
        create_zarr_count_assay(
            z=self.z,
            assay_name=self.assayName,
            workspace=workspace,
            chunk_size=chunk_size,
            n_cells=self.nCells,
            feat_ids=feature_ids,
            feat_names=feature_names,
            dtype=str(self.matrixDtype),
        )

    def dump(self, batch_size: int | None = None) -> None:
        """Write out the data matrix into the Zarr hierarchy.

        Args:
            batch_size: Number of cells to be written in one go. By default, this value will automatically be chosen
                        based on the chunk size in the cell dimension.

         Raises:
            ValueError: Raised if there is any unexpected errors when writing to the Zarr hierarchy.
            AssertionError: Catches eventual bugs in the class, if number of cells does not match after transformation.

        Returns:
            None
        """
        from . import finalize_writer_counts, load_count_store

        store = load_count_store(self.z, self.assayName, self.workspace)
        if batch_size is None:
            batch_size = array_shard_rows(store)

        def row_batches() -> Iterator[coo_matrix]:
            s = 0
            for end in range(batch_size, self.nCells + batch_size, batch_size):
                if s == self.nCells:
                    break
                if end > self.nCells:
                    end = self.nCells
                yield self.mat[s:end].tocoo()
                s = end

        e = accumulate_sparse_to_shards(
            store,
            row_batches(),
            dtype=self.matrixDtype,
        )
        if e != self.nCells:
            raise AssertionError(
                "ERROR: This is a bug in SparseToZarr. All cells might not have been successfully "
                "written into the zarr file. Please report this issue"
            )
        finalize_writer_counts(self.z, self.assayName, self.workspace)


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
    for i in tqdmbar(chrom_sizes, disable=disable_tqdm, desc="Calculating bin indices"):
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
    for df in tqdmbar(
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
