import math
import os
from abc import ABC, abstractmethod
from collections.abc import Generator, Sequence
from typing import Any

import h5py
import numpy as np
import pandas as pd
from numpy.typing import DTypeLike
from scipy.sparse import coo_matrix

from ..utils.logging import logger
from ..utils.progress import iter_progress
from ._assay_names import (
    AUTO_ASSAY_NAMES,
    auto_name_feat_table,
    make_feat_table_from_types,
)


class CrReader(ABC):
    """A class to read in CellRanger (Cr) data.

    Args:
        grp_names (Dict): A dictionary that specifies where to find the matrix, features and barcodes.

    Attributes:
        autoNames: Specifies if the data is from RNA or ATAC sequencing.
        grpNames: A dictionary that specifies where to find the matrix, features and barcodes.
        nFeatures: Number of features in dataset.
        nCells: Number of cells in dataset.
        assayFeats: A DataFrame with information about the features in the assay.
    """

    def __init__(self, grp_names: dict[str, Any]) -> None:
        self.autoNames = dict(AUTO_ASSAY_NAMES)
        self._featureTypeOverrides: dict[int, str] = {}
        self._schemaCaptured = False
        self.grpNames: dict[str, Any] = grp_names
        self.nFeatures: int = len(self.feature_names())
        self.nCells: int = len(self.cell_names())
        self.assayFeats = self._make_feat_table()
        self._auto_rename_assay_names()

    @abstractmethod
    def _handle_version(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def _read_dataset(self, key: str | None = None) -> list[Any] | None:
        pass

    @abstractmethod
    def consume(
        self, batch_size: int, lines_in_mem: int
    ) -> Generator[coo_matrix, None, None]:
        """Yield CSR matrix chunks of cell rows.

        Args:
            batch_size: Number of cells per yielded chunk.
            lines_in_mem: MTX lines buffered in memory (CrDirReader only).

        Yields:
            scipy.sparse.coo_matrix chunks.
        """
        pass

    def max_window_nnz(self, window_rows: int) -> int:
        """Bound nnz in any source row window."""
        if window_rows <= 0:
            raise ValueError("window_rows must be positive")
        return min(window_rows, self.nCells) * self.nFeatures

    @property
    def matrix_dtype(self) -> np.dtype[Any]:
        """Return the count dtype yielded by the default consume call."""
        return np.dtype(np.uint32)

    def producer_staging_bytes(
        self,
        batch_size: int,
        lines_in_mem: int,
    ) -> int:
        """Return source-reader bytes retained while a matrix batch is yielded."""
        valid_idx = getattr(self, "validBarcodeIdx", None)
        return int(valid_idx.nbytes) if isinstance(valid_idx, np.ndarray) else 0

    def _subset_by_assay(self, v: list[Any], assay: str | None) -> list[Any]:
        if assay is None:
            return v
        elif assay not in self.assayFeats:
            raise ValueError(f"ERROR: Assay ID {assay} is not valid")
        if len(self.assayFeats[assay].shape) == 2:
            ret_val: list[Any] = []
            for i in self.assayFeats[assay].values[1:3].T:
                ret_val.extend(list(v[i[0] : i[1]]))
            return ret_val
        elif len(self.assayFeats[assay].shape) == 1:
            idx = self.assayFeats[assay]
            return v[idx.start : idx.end]
        else:
            raise ValueError(
                "ERROR: assay feats is 3D. Something went really wrong. Create a github issue"
            )

    @staticmethod
    def _make_feat_table_from_types(feature_types: Sequence[str]) -> pd.DataFrame:
        return make_feat_table_from_types(feature_types)

    def _make_feat_table(self) -> pd.DataFrame:
        return self._make_feat_table_from_types(self.feature_types())

    def _auto_named_feat_table(self, assay_feats: pd.DataFrame) -> pd.DataFrame:
        return auto_name_feat_table(assay_feats, self.autoNames)

    def _auto_rename_assay_names(self) -> None:
        self.assayFeats = self._auto_named_feat_table(self.assayFeats)

    def _mark_schema_captured(self) -> None:
        self._schemaCaptured = True

    def reclassify_features(
        self,
        indexes: Sequence[int],
        feature_type: str,
        *,
        require_previous: str | None = "Antibody Capture",
    ) -> None:
        """Reclassify global feature rows before a writer captures the schema."""
        if self._schemaCaptured:
            raise RuntimeError(
                "Features cannot be reclassified after a writer captures the schema"
            )
        if not isinstance(feature_type, str) or feature_type == "":
            raise ValueError("feature_type must be a non-empty string")
        if isinstance(indexes, str):
            raise TypeError("indexes must be a sequence of integer feature indexes")
        index_array = np.asarray(indexes)
        if index_array.ndim != 1:
            raise ValueError("indexes must be one-dimensional")
        if index_array.size == 0:
            raise ValueError("indexes must contain at least one feature index")
        if not np.issubdtype(index_array.dtype, np.integer):
            raise TypeError("indexes must contain only integers")
        index_array = index_array.astype(np.int64, copy=False)
        if np.unique(index_array).size != index_array.size:
            raise ValueError("indexes must contain unique feature indexes")
        if np.any(index_array < 0) or np.any(index_array >= self.nFeatures):
            raise IndexError("indexes contains an out-of-range feature index")

        conflicting = [
            int(index)
            for index in index_array
            if index in self._featureTypeOverrides
            and self._featureTypeOverrides[int(index)] != feature_type
        ]
        if conflicting:
            raise ValueError(
                "Features already have a conflicting reclassification: "
                + ", ".join(map(str, conflicting))
            )

        current_types = self.feature_types()
        pending = np.asarray(
            [
                index
                for index in index_array
                if current_types[int(index)] != feature_type
            ],
            dtype=np.int64,
        )
        if pending.size == 0:
            return None
        if require_previous is not None:
            invalid = [
                int(index)
                for index in pending
                if current_types[int(index)] != require_previous
            ]
            if invalid:
                raise ValueError(
                    f"Features must currently have type {require_previous!r}: "
                    + ", ".join(map(str, invalid))
                )

        updated_types = list(current_types)
        updated_overrides = dict(self._featureTypeOverrides)
        for index in pending:
            updated_types[int(index)] = feature_type
            updated_overrides[int(index)] = feature_type
        updated_table = self._auto_named_feat_table(
            self._make_feat_table_from_types(updated_types)
        )

        self._featureTypeOverrides = updated_overrides
        self.assayFeats = updated_table
        return None

    def rename_assays(self, name_map: dict[str, str]) -> None:
        """Renames specified assays in the Reader.

        Args:
            name_map: A Dictionary containing current name as key and new name as value.
        """
        self.assayFeats.rename(columns=name_map, inplace=True)

    def feature_ids(self, assay: str | None = None) -> list[str]:
        """Returns a list of feature IDs in a specified assay.

        Args:
            assay: Select which assay to retrieve feature IDs from.
        """
        vals = self._read_dataset("feature_ids")
        if vals is None:
            return []
        return self._subset_by_assay(vals, assay)

    def feature_names(self, assay: str | None = None) -> list[str]:
        """Returns a list of features in the dataset.

        Args:
            assay: Select which assay to retrieve features from.
        """
        vals = self._read_dataset("feature_names")
        if vals is None:
            logger.warning("Feature names extraction failed using feature IDs")
            vals = self._read_dataset("feature_ids")
        if vals is None:
            return []
        return self._subset_by_assay(vals, assay)

    def feature_types(self) -> list[str]:
        """Returns a list of feature types in the dataset."""
        if self.grpNames["feature_types"] is not None:
            ret_val = self._read_dataset("feature_types")
            if ret_val is not None:
                feature_types = list(ret_val)
            else:
                feature_types = []
        else:
            feature_types = []
        if not feature_types:
            default_name = list(self.autoNames.keys())[0]
            feature_types = [default_name for _ in range(self.nFeatures)]
        for index, feature_type in self._featureTypeOverrides.items():
            feature_types[index] = feature_type
        return feature_types

    def cell_names(self) -> list[str]:
        """Returns a list of names of the cells in the dataset."""
        vals = self._read_dataset("cell_names")
        if vals is None:
            return []
        return vals


class CrH5Reader(CrReader):
    # noinspection PyUnresolvedReferences
    """A class to read in CellRanger (Cr) data, in the form of an H5 file.

    Subclass of CrReader.

    Args:
        h5_fn: File name for the h5 file.

    Attributes:
        autoNames: Specifies if the data is from RNA or ATAC sequencing.
        grpNames: A dictionary that specifies where to find the matrix, features and barcodes.
        nFeatures: Number of features in dataset.
        nCells: Number of cells in dataset.
        assayFeats: A DataFrame with information about the features in the assay.
        h5obj: A File object from the h5py package.
        grp: Current active group in the hierarchy.
    """

    def __init__(
        self,
        h5_fn: str,
        is_filtered: bool = True,
        filtering_cutoff: int = 500,
    ) -> None:
        self.h5obj: h5py.File = h5py.File(h5_fn, mode="r")
        self.grp: h5py.Group
        self.validBarcodeIdx: np.ndarray | None = None
        super().__init__(self._handle_version())
        if is_filtered:
            self.validBarcodeIdx = np.array(range(self.nCells))
        else:
            self.validBarcodeIdx = self._get_valid_barcodes(filtering_cutoff)
        self.nCells = len(self.validBarcodeIdx)

    def _handle_version(self) -> dict[str, str | None]:
        root_key = list(self.h5obj.keys())[0]
        self.grp = self.h5obj[root_key]
        if root_key == "matrix":
            grps: dict[str, str | None] = {
                "feature_ids": "features/id",
                "feature_names": "features/name",
                "feature_types": "features/feature_type",
                "cell_names": "barcodes",
            }
        else:
            grps = {
                "feature_ids": "genes",
                "feature_names": "gene_names",
                "feature_types": None,
                "cell_names": "barcodes",
            }
        return grps

    def _get_valid_barcodes(
        self, filtering_cutoff: int, batch_size: int = 1000
    ) -> np.ndarray:
        valid_idx = []
        test_counter = 0
        indptr = self.grp["indptr"][:]
        for s in iter_progress(
            range(0, len(indptr) - 1, batch_size),
            desc="Filtering out background barcodes",
        ):
            idx = indptr[s : s + batch_size + 1]
            data = self.grp["data"][idx[0] : idx[-1]]
            indices = self.grp["indices"][idx[0] : idx[-1]]
            cell_idx = np.repeat(range(len(idx) - 1), np.diff(idx))
            mat = coo_matrix(
                (data, (cell_idx, indices)), shape=(len(idx) - 1, self.nFeatures)
            )
            valid_idx.append(np.array(mat.sum(axis=1)).T[0] > filtering_cutoff)
            test_counter += data.shape[0]
        assert test_counter == self.grp["data"].shape[0]
        assert len(indptr) == (s + len(idx))
        return np.where(np.hstack(valid_idx))[0]

    @property
    def matrix_dtype(self) -> np.dtype[Any]:
        dtype: np.dtype[Any] = np.dtype(self.grp["data"].dtype)
        return dtype

    def _read_dataset(self, key: str | None = None) -> list[str]:
        if key is None:
            raise ValueError("Dataset key must be provided")
        grp_key = self.grpNames[key]
        return [x.decode("UTF-8") for x in self.grp[grp_key][:]]

    def cell_names(self) -> list[str]:
        """Returns a list of names of the cells in the dataset."""
        vals = np.array(self._read_dataset("cell_names"))
        if self.validBarcodeIdx is not None:
            vals = vals[self.validBarcodeIdx]
        return list(vals)

    # noinspection DuplicatedCode
    def consume(
        self, batch_size: int, lines_in_mem: int | None = None
    ) -> Generator[coo_matrix, None, None]:
        """Yield CSR chunks from the Cell Ranger H5 matrix.

        Args:
            batch_size: Number of cells per chunk.
            lines_in_mem: Unused; kept for CrReader API compatibility.
        """
        valid_idx = self.validBarcodeIdx
        assert valid_idx is not None
        indptr = self.grp["indptr"][:]
        for s in range(0, len(valid_idx), batch_size):
            v_pos = valid_idx[s : s + batch_size]
            starts = indptr[v_pos]
            ends = indptr[v_pos + 1]
            counts = ends - starts
            cell_idx = np.repeat(
                np.arange(len(v_pos)),
                counts,
            )
            nonempty = np.flatnonzero(counts)
            idx = (
                np.concatenate([np.arange(starts[i], ends[i]) for i in nonempty])
                if nonempty.size
                else np.array([], dtype=np.int64)
            )
            if idx.size == 0:
                yield coo_matrix(
                    ([], ([], [])),
                    shape=(len(v_pos), self.nFeatures),
                    dtype=self.matrix_dtype,
                )
                continue
            data = np.asarray(self.grp["data"][idx])
            indices = np.asarray(self.grp["indices"][idx])
            yield coo_matrix(
                (data, (cell_idx, indices)), shape=(len(v_pos), self.nFeatures)
            )

    def max_window_nnz(self, window_rows: int) -> int:
        """Return the largest selected-cell row-window nnz."""
        if window_rows <= 0:
            raise ValueError("window_rows must be positive")
        valid_idx = self.validBarcodeIdx
        assert valid_idx is not None
        width = min(window_rows, len(valid_idx))
        if width == 0:
            return 0
        indptr = np.asarray(self.grp["indptr"])
        row_nnz = np.diff(indptr)[valid_idx]
        cumulative = np.empty(row_nnz.size + 1, dtype=np.int64)
        cumulative[0] = 0
        np.cumsum(row_nnz, dtype=np.int64, out=cumulative[1:])
        return int(np.max(cumulative[width:] - cumulative[:-width]))

    def producer_staging_bytes(
        self,
        batch_size: int,
        lines_in_mem: int,
    ) -> int:
        """Count selected-cell indexes and the CSR row pointer loaded by consume."""
        indptr = self.grp["indptr"]
        indptr_bytes = int(indptr.size * indptr.dtype.itemsize)
        planning_arrays = 3 * (self.nCells + 1) * np.dtype(np.int64).itemsize
        return super().producer_staging_bytes(batch_size, lines_in_mem) + int(
            2 * indptr_bytes + planning_arrays
        )

    def close(self) -> None:
        """Closes file connection."""
        self.h5obj.close()


class CrDirReader(CrReader):
    """A class to read in CellRanger (Cr) data, in the form of a directory.

    Subclass of CrReader.

    Args:
        loc (str): Path for the directory containing the cellranger output.
        mtx_separator (str): Column delimiter in the MTX file (Default value: ' ')
        index_offset (int): This value is added to each feature index (Default value: -1)

    Attributes:
        loc: Path for the directory containing the cellranger output.
        matFn: The file name for the matrix file.
        sep (str): Column delimiter in the MTX file (Default value: ' ')
        indexOffset (int): This value is added to each feature index (Default value: -1)
    """

    def __init__(
        self,
        loc: str,
        mtx_separator: str = " ",
        index_offset: int = -1,
        is_filtered: bool = True,
        filtering_cutoff: int = 500,
    ) -> None:
        self.loc: str = loc.rstrip("/") + "/"
        self.matFn: str
        self.sep = mtx_separator
        self.indexOffset = index_offset
        self.validBarcodeIdx: np.ndarray | None = None
        super().__init__(self._handle_version())
        self.matrixEntryCount = (
            0 if self.nCells == 0 else int(self.read_header().iloc[0]["nCounts"])
        )
        if is_filtered:
            self.validBarcodeIdx = np.array(range(self.nCells))
            self.validBarcodeIdx -= self.indexOffset
        else:
            self.validBarcodeIdx = self._get_valid_barcodes(filtering_cutoff)
        self.nCells = len(self.validBarcodeIdx)

    def _handle_version(self) -> dict[str, tuple[str, int] | None]:
        show_error = False
        mat_fn: str | None = None
        feat_fn: str | None = None
        cell_fn: str | None = None
        if os.path.isfile(self.loc + "matrix.mtx.gz"):
            mat_fn = self.loc + "matrix.mtx.gz"
        elif os.path.isfile(self.loc + "matrix.mtx"):
            mat_fn = self.loc + "matrix.mtx"
        else:
            show_error = True
        if os.path.isfile(self.loc + "features.tsv.gz"):
            feat_fn = "features.tsv.gz"
        elif os.path.isfile(self.loc + "features.tsv"):
            feat_fn = "features.tsv"
        elif os.path.isfile(self.loc + "genes.tsv.gz"):
            feat_fn = "genes.tsv.gz"
        elif os.path.isfile(self.loc + "genes.tsv"):
            feat_fn = "genes.tsv"
        elif os.path.isfile(self.loc + "peaks.bed"):
            feat_fn = "peaks.bed"
        elif os.path.isfile(self.loc + "peaks.bed.gz"):
            feat_fn = "peaks.bed.gz"
        else:
            feat_fn = None
            show_error = True
        if os.path.isfile(self.loc + "barcodes.tsv.gz"):
            cell_fn = "barcodes.tsv.gz"
        elif os.path.isfile(self.loc + "barcodes.tsv"):
            cell_fn = "barcodes.tsv"
        else:
            cell_fn = None
            show_error = True
        if show_error or mat_fn is None or feat_fn is None or cell_fn is None:
            raise OSError(
                "ERROR: Couldn't find either of these expected combinations of files:\n"
                "\t- matrix.mtx, barcodes.tsv and genes.tsv\n"
                "\t- matrix.mtx.gz, barcodes.tsv.gz and features.tsv.gz\n"
                "Please make sure that you have not compressed or uncompressed the Cellranger output files "
                "manually"
            )
        self.matFn = mat_fn
        return {
            "feature_ids": (feat_fn, 0),
            "feature_names": (feat_fn, 1),
            "feature_types": (feat_fn, 2),
            "cell_names": (cell_fn, 0),
        }

    def _read_dataset(self, key: str | None = None) -> list[str] | None:
        if key is None:
            raise ValueError("Dataset key must be provided")
        from . import read_file

        grp_entry = self.grpNames[key]
        try:
            vals = [
                x.split("\t")[grp_entry[1]] for x in read_file(self.loc + grp_entry[0])
            ]
        except IndexError:
            logger.warning(
                f"Could not extract {key} from {grp_entry[0]} column {grp_entry[1]}"
            )
            vals = None
        return vals

    def read_header(self) -> pd.DataFrame:
        header = pd.read_csv(
            self.matFn,
            comment="%",
            sep=self.sep,
            header=None,
            nrows=1,
            names=["nFeatures", "nCells", "nCounts"],
        )
        if header["nCells"][0] == 0 and self.nCells > 0:
            raise ValueError(
                "ERROR: Barcode count in MTX header is 0 but barcodes are present in the barcodes file"
            )
        if header["nCells"][0] > 0 and self.nCells == 0:
            raise ValueError(
                "ERROR: Barcode count in MTX header is greater than 0 but no barcodes are present in the barcodes file"
            )
        if header["nCells"][0] == 0 and self.nCells == 0:
            raise ValueError(
                "ERROR: Barcode count in MTX header and barcodes file is 0. No data to read"
            )
        return header

    def process_batch(
        self, dfs: list[pd.DataFrame], filtering_cutoff: int
    ) -> np.ndarray:
        """Returns a list of valid barcodes after filtering out background barcodes for a given batch.

        Args:
            dfs: A Polar DataFrame containing a chunk of data from the MTX file.
            filtering_cutoff: The cutoff value for filtering out background barcodes
        """
        merged = pd.concat(dfs, ignore_index=True)
        summed = merged.groupby("barcode")["count"].sum()
        valid = summed[summed > filtering_cutoff].index.to_numpy()
        return np.sort(valid)

    def _get_valid_barcodes(
        self,
        filtering_cutoff: int,
        batch_size: int = int(10e3),
        lines_in_mem: int = int(10e5),
    ) -> np.ndarray:
        """Returns a list of valid barcodes after filtering out background barcodes.

        Args:
            filtering_cutoff: The cutoff value for filtering out background barcodes.
            batch_size: The number of barcodes to process in each batch.
            lines_in_mem: The number of lines to read into memory
        """
        test_counter = 0
        matrixIO = pd.read_csv(
            self.matFn,
            comment="%",
            sep=self.sep,
            header=0,
            chunksize=lines_in_mem,
            names=["gene", "barcode", "count"],
        )

        header = self.read_header()
        nChunks = math.ceil(header["nCounts"][0] / lines_in_mem)
        test_counter = 0
        valid_idx = []
        start = 1

        dfs = []
        for chunk in iter_progress(
            # range(nChunks),
            matrixIO,
            total=nChunks,
            desc="Filtering out background barcodes",
        ):
            if (
                (chunk.iloc[-1]["barcode"] - start) >= batch_size
            ):  # If the last "cell id" is greater than the start + batch size
                # Filter rows in the current chunk that belong to the current batch
                idx = np.array(
                    chunk["barcode"].values < (batch_size + start)
                )  # This is the crucial line. This makes sure that if any cell ID is spread over multiple chunks, it is not missed, as any cell ID that is less than the batch size + start is included.
                # If no rows belong to the current batch, move to the next batch.
                if idx.sum() == 0:
                    dfs.append(chunk)
                    start += batch_size
                    test_counter += len(chunk)
                    continue
                # Process the rows belonging to the current batch
                mask_pos = np.where(idx)[0]
                mask_neg = np.where(~idx)[0]
                dfs.append(chunk.iloc[mask_pos])
                valid_idx.append(self.process_batch(dfs, filtering_cutoff))
                # Prepare for the next batch
                del dfs
                dfs = [chunk.iloc[mask_neg]]
                start += batch_size
            else:
                # If we haven't reached the batch boundary, accumulate the chunk
                dfs.append(chunk)
            test_counter += len(chunk)
        # Process any remaining data after the main loop
        if len(dfs) > 0:
            valid_idx.append(self.process_batch(dfs, filtering_cutoff))
        # Verify that all rows were processed
        assert test_counter == header["nCounts"][0]
        return np.sort(np.unique(np.hstack(valid_idx)))

    def to_sparse(self, a: np.ndarray, dtype: DTypeLike) -> coo_matrix:
        """Returns the input data as a sparse (COO) matrix.

        Args:
            a: Sparse matrix, contains a chunk of data from the MTX file.
            dtype:
        """
        c = (a[:, 1] - a[0, 1]).astype(int)
        return coo_matrix(
            (
                a[:, 2],
                (
                    c,
                    (a[:, 0] + self.indexOffset).astype(int),
                ),
            ),
            shape=(c[-1] + 1, self.nFeatures),
            dtype=dtype,
        )

    def cell_names(self) -> list[str]:
        """Returns a list of names of the cells in the dataset."""
        vals = np.array(self._read_dataset("cell_names"))
        if self.validBarcodeIdx is not None:
            vals = vals[(self.validBarcodeIdx + self.indexOffset)]
        return list(vals)

    def rename_batches(self, collect: list[pd.DataFrame]) -> np.ndarray:
        df = pd.concat(collect, ignore_index=True)
        barcodes = df["barcode"].to_numpy()
        count_hash = {}
        for i, x in enumerate(np.unique(barcodes)):
            count_hash[x] = i
        cell_idx = np.array([count_hash[x] for x in barcodes])
        df = df.copy()
        df["barcode"] = cell_idx
        return np.array(df)

    def producer_staging_bytes(
        self,
        batch_size: int,
        lines_in_mem: int,
    ) -> int:
        """Bound pandas input and split state retained across a yielded batch."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if lines_in_mem <= 0:
            raise ValueError("lines_in_mem must be positive")
        integer_bytes = np.dtype(np.int64).itemsize
        mask_bytes = np.dtype(np.bool_).itemsize
        frame_and_index = 3 * integer_bytes + integer_bytes
        grouping_scratch = 4 * integer_bytes
        pending_entries = min(
            min(batch_size, self.nCells) * self.nFeatures,
            self.matrixEntryCount,
        )
        pending_bytes = pending_entries * (
            3 * frame_and_index + grouping_scratch + mask_bytes
        )
        return super().producer_staging_bytes(batch_size, lines_in_mem) + int(
            lines_in_mem * (4 * frame_and_index + grouping_scratch + mask_bytes)
            + pending_bytes
        )

    def max_window_nnz(self, window_rows: int) -> int:
        return min(
            super().max_window_nnz(window_rows),
            self.matrixEntryCount,
        )

    # noinspection DuplicatedCode
    def consume(
        self,
        batch_size: int,
        lines_in_mem: int = int(1e5),
        dtype: DTypeLike = np.uint32,
    ) -> Generator[coo_matrix, None, None]:
        """Yields chunks of data from the MTX file.

        Args:
            batch_size: The number of barcodes to process in each batch.
            lines_in_mem: The number of lines to read into memory.
            dtype: The data type of the matrix.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        matrixIO = pd.read_csv(
            self.matFn,
            comment="%",
            sep=self.sep,
            header=0,
            chunksize=lines_in_mem,
            names=["gene", "barcode", "count"],
        )
        pending: pd.DataFrame | None = None
        for chunk in matrixIO:
            chunk = chunk[chunk["barcode"].isin(self.validBarcodeIdx)]
            if chunk.empty:
                continue
            pending = (
                chunk.reset_index(drop=True)
                if pending is None
                else pd.concat((pending, chunk), ignore_index=True)
            )
            pending = (
                pending.groupby(
                    ["gene", "barcode"],
                    sort=False,
                    as_index=False,
                )["count"]
                .sum()
                .reset_index(drop=True)
            )
            barcodes = pending["barcode"].to_numpy()
            if np.any(barcodes[1:] < barcodes[:-1]):
                raise ValueError("Cell Ranger MTX entries must be sorted by barcode")
            unique_barcodes = np.unique(barcodes)
            complete = (len(unique_barcodes) - 1) // batch_size
            consumed = 0
            for boundary in range(batch_size, complete * batch_size + 1, batch_size):
                end = int(np.searchsorted(barcodes, unique_barcodes[boundary]))
                batch_arr = self.rename_batches([pending.iloc[consumed:end]])
                yield self.to_sparse(batch_arr, dtype=dtype)
                consumed = end
            if consumed:
                pending = pending.iloc[consumed:].reset_index(drop=True)
        if pending is not None and not pending.empty:
            batch_arr = self.rename_batches([pending])
            yield self.to_sparse(batch_arr, dtype=dtype)
