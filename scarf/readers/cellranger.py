import math
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterator, Sequence
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

    def _prepare_sparse_import(self) -> None:
        """Prepare optional reader-owned state used by import planning."""

    def _release_sparse_import(self) -> None:
        """Release optional reader-owned state created for one import."""

    def _sparse_import_resident_bytes(self) -> int:
        """Return reader arrays retained for the duration of an import."""
        return 0

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

    def get_cell_columns(self) -> Iterator[tuple[str, np.ndarray]]:
        """Yield optional cell metadata columns supplied by the reader."""
        yield from ()

    def get_feature_columns(self) -> Iterator[tuple[str, np.ndarray]]:
        """Yield source feature types and optional feature metadata."""
        yield "feature_type", np.asarray(self.feature_types(), dtype=object)


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
        self._indptrCache: np.ndarray | None = None
        self._cumulativeRowNnz: np.ndarray | None = None
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
        indptr = self._source_indptr()
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

    def _source_indptr(self) -> np.ndarray:
        if self._indptrCache is None:
            self._indptrCache = np.asarray(self.grp["indptr"][:])
        return self._indptrCache

    def _selected_cumulative_nnz(self) -> np.ndarray:
        if self._cumulativeRowNnz is None:
            valid_idx = self.validBarcodeIdx
            assert valid_idx is not None
            row_nnz = np.diff(self._source_indptr())[valid_idx]
            cumulative = np.empty(row_nnz.size + 1, dtype=np.int64)
            cumulative[0] = 0
            np.cumsum(row_nnz, dtype=np.int64, out=cumulative[1:])
            self._cumulativeRowNnz = cumulative
        return self._cumulativeRowNnz

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

    def get_feature_columns(self) -> Iterator[tuple[str, np.ndarray]]:
        yield from super().get_feature_columns()
        if "features" not in self.grp:
            return
        features = self.grp["features"]
        if not isinstance(features, h5py.Group) or "_all_tag_keys" not in features:
            return
        raw_keys = np.asarray(features["_all_tag_keys"][:]).reshape(-1)
        for raw_key in raw_keys:
            key = (
                raw_key.decode("utf-8")
                if isinstance(raw_key, bytes | np.bytes_)
                else str(raw_key)
            )
            if key in {"id", "name", "feature_type"} or key not in features:
                continue
            values = features[key]
            if not isinstance(values, h5py.Dataset):
                continue
            if values.ndim != 1 or int(values.shape[0]) != self.nFeatures:
                raise ValueError(
                    f"10x feature tag {key!r} has shape {values.shape}; "
                    f"expected ({self.nFeatures},)"
                )
            yield key, np.asarray(values[:])

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
        indptr = self._source_indptr()
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
        cumulative = self._selected_cumulative_nnz()
        return int(np.max(cumulative[width:] - cumulative[:-width]))

    def producer_staging_bytes(
        self,
        batch_size: int,
        lines_in_mem: int,
    ) -> int:
        """Count row-pointer arrays created while one H5 batch is produced."""
        rows = min(max(1, int(batch_size)), self.nCells)
        if rows == 0:
            return 0
        itemsize = self._source_indptr().dtype.itemsize
        return int(rows * (3 * itemsize + np.dtype(np.int64).itemsize))

    def _prepare_sparse_import(self) -> None:
        self._source_indptr()
        self._selected_cumulative_nnz()

    def _sparse_import_resident_bytes(self) -> int:
        arrays = (
            self.validBarcodeIdx,
            self._indptrCache,
            self._cumulativeRowNnz,
        )
        return int(
            sum(array.nbytes for array in arrays if isinstance(array, np.ndarray))
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
        from .mtx import _MtxEngine, inspect_mtx

        self.loc: str = loc.rstrip("/") + "/"
        self.sep = mtx_separator
        self.indexOffset = index_offset
        candidates = inspect_mtx(loc)
        if len(candidates) != 1:
            raise ValueError(
                "CrDirReader requires one complete Matrix Market triplet; "
                "use inspect_mtx and MtxReader to select among candidates"
            )
        self._engine = _MtxEngine(
            candidates[0],
            cell_id_key=None,
            separator=mtx_separator,
            index_offset=index_offset,
            is_filtered=is_filtered,
            filtering_cutoff=filtering_cutoff,
            temp_dir=None,
            dtype=np.uint32,
        )
        self.matFn = self._engine.matrixPath
        self.validBarcodeIdx = self._engine.validCellIndexes - self.indexOffset
        self.matrixEntryCount = self._engine.matrixEntryCount
        self.coordinateOrder = self._engine.coordinateOrder
        CrReader.__init__(
            self,
            {
                "feature_ids": "feature_ids",
                "feature_names": "feature_names",
                "feature_types": "feature_types",
                "cell_names": "cell_names",
            },
        )
        self.nCells = self._engine.nCells

    def _handle_version(self) -> dict[str, str]:
        return {
            "feature_ids": "feature_ids",
            "feature_names": "feature_names",
            "feature_types": "feature_types",
            "cell_names": "cell_names",
        }

    def _read_dataset(self, key: str | None = None) -> list[str]:
        if key is None:
            raise ValueError("Dataset key must be provided")
        values = {
            "feature_ids": self._engine.feature_ids,
            "feature_names": self._engine.feature_names,
            "feature_types": self._engine.feature_types,
            "cell_names": self._engine.cell_names,
        }
        if key not in values:
            raise KeyError(key)
        return values[key]()

    def read_header(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "nFeatures": self._engine.nFeatures,
                    "nCells": self._engine.rawCellCount,
                    "nCounts": self._engine.matrixEntryCount,
                }
            ]
        )

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
        return self._engine.cell_names()

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
        return self._engine.producer_staging_bytes(
            batch_size,
            lines_in_mem,
        )

    def max_window_nnz(self, window_rows: int) -> int:
        return self._engine.max_window_nnz(window_rows)

    @property
    def matrix_dtype(self) -> np.dtype[Any]:
        return self._engine.matrixDtype

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
        yield from self._engine.consume(batch_size, lines_in_mem, dtype)

    def get_cell_columns(self) -> Iterator[tuple[str, np.ndarray]]:
        yield from self._engine.cell_columns()

    def get_feature_columns(self) -> Iterator[tuple[str, np.ndarray]]:
        yield from self._engine.feature_columns()

    def _set_sparse_import_lines_in_mem(self, lines_in_mem: int) -> None:
        self._engine.configure_import_lines(lines_in_mem)

    def _prepare_sparse_import(self) -> None:
        self._engine.prepare()

    def _release_sparse_import(self) -> None:
        self._engine.release()

    def _sparse_import_resident_bytes(self) -> int:
        return self._engine.resident_bytes()

    def close(self) -> None:
        self._engine.release()
