from abc import ABC, abstractmethod
from collections.abc import Generator, Iterator, Sequence
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from ..utils.logging import logger
from ..utils.progress import iter_progress
from ._assay_names import (
    AUTO_ASSAY_NAMES,
    auto_name_feat_table,
    make_feat_table_from_types,
)
from ._text import as_text, require_unique_identifiers
from ..utils.arrays import (
    assay_feature_ranges,
    cumulative_nnz,
    has_duplicates,
    max_window_nnz,
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
            lines_in_mem: MTX lines buffered in memory (Matrix Market readers
                only).

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
        ranges = assay_feature_ranges(self.assayFeats)
        if assay not in ranges:
            raise ValueError(f"ERROR: Assay ID {assay} is not valid")
        return [value for start, end in ranges[assay] for value in v[start:end]]

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
        if has_duplicates(index_array):
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
        try:
            super().__init__(self._handle_version())
            require_unique_identifiers(self.cell_names(), "Cell Ranger barcodes")
            require_unique_identifiers(self.feature_ids(), "Cell Ranger feature IDs")
            if is_filtered:
                self.validBarcodeIdx = np.arange(self.nCells)
            else:
                self.validBarcodeIdx = self._get_valid_barcodes(filtering_cutoff)
            self.nCells = len(self.validBarcodeIdx)
        except BaseException:
            self.h5obj.close()
            raise

    def _handle_version(self) -> dict[str, str | None]:
        root_keys = list(self.h5obj.keys())
        if "matrix" in root_keys:
            root_key = "matrix"
        else:
            # Cell Ranger 2 keeps one group per genome at the root.
            genomes = [
                key for key in root_keys if isinstance(self.h5obj[key], h5py.Group)
            ]
            if len(genomes) != 1:
                raise ValueError(
                    "Cell Ranger HDF5 input needs a `matrix` group or exactly one "
                    f"genome group; found: {', '.join(genomes) or 'none'}"
                )
            root_key = genomes[0]
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
        indptr = self._source_indptr()
        n_barcodes = len(indptr) - 1
        if int(indptr[-1]) != int(self.grp["data"].shape[0]):
            raise ValueError("Cell Ranger matrix pointers do not match its data")
        valid = np.zeros(n_barcodes, dtype=bool)
        for start in iter_progress(
            range(0, n_barcodes, batch_size),
            desc="Filtering out background barcodes",
        ):
            stop = min(start + batch_size, n_barcodes)
            data = np.asarray(self.grp["data"][indptr[start] : indptr[stop]])
            rows = np.repeat(
                np.arange(stop - start),
                np.diff(indptr[start : stop + 1]),
            )
            totals = np.bincount(rows, weights=data, minlength=stop - start)
            valid[start:stop] = totals > filtering_cutoff
        return np.flatnonzero(valid)

    def _source_indptr(self) -> np.ndarray:
        if self._indptrCache is None:
            self._indptrCache = np.asarray(self.grp["indptr"][:])
        return self._indptrCache

    def _selected_cumulative_nnz(self) -> np.ndarray:
        if self._cumulativeRowNnz is None:
            valid_idx = self.validBarcodeIdx
            assert valid_idx is not None
            self._cumulativeRowNnz = cumulative_nnz(
                np.diff(self._source_indptr())[valid_idx]
            )
        return self._cumulativeRowNnz

    @property
    def matrix_dtype(self) -> np.dtype[Any]:
        dtype: np.dtype[Any] = np.dtype(self.grp["data"].dtype)
        return dtype

    def _read_dataset(self, key: str | None = None) -> list[str]:
        if key is None:
            raise ValueError("Dataset key must be provided")
        grp_key = self.grpNames[key]
        return [as_text(x) for x in self.grp[grp_key][:]]

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
            key = as_text(raw_key)
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
            nnz = int(counts.sum())
            if nnz == 0:
                yield coo_matrix(
                    ([], ([], [])),
                    shape=(len(v_pos), self.nFeatures),
                    dtype=self.matrix_dtype,
                )
                continue
            data = np.empty(nnz, dtype=self.matrix_dtype)
            indices = np.empty(nnz, dtype=self.grp["indices"].dtype)
            boundaries = np.r_[
                0, np.flatnonzero(starts[1:] != ends[:-1]) + 1, len(v_pos)
            ]
            offset = 0
            for first, stop in zip(boundaries[:-1], boundaries[1:]):
                start, end = int(starts[first]), int(ends[stop - 1])
                size = end - start
                if size:
                    source = np.s_[start:end]
                    destination = np.s_[offset : offset + size]
                    self.grp["data"].read_direct(data, source, destination)
                    self.grp["indices"].read_direct(indices, source, destination)
                    offset += size
            yield coo_matrix(
                (data, (cell_idx, indices)), shape=(len(v_pos), self.nFeatures)
            )

    def max_window_nnz(self, window_rows: int) -> int:
        """Return the largest selected-cell row-window nnz."""
        return max_window_nnz(self._selected_cumulative_nnz(), window_rows)

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
