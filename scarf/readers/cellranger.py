import math
import os
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any

import h5py
import numpy as np
import pandas as pd
from numpy.typing import DTypeLike
from scipy.sparse import coo_matrix

from ..utils.logging import logger
from ..utils.progress import tqdmbar


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
        self.autoNames: dict[str, str] = {
            "Gene Expression": "RNA",
            "Peaks": "ATAC",
            "Antibody Capture": "ADT",
            "RNA": "RNA",
            "ADT": "ADT",
            "HTO": "HTO",
        }
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

    def _make_feat_table(self) -> pd.DataFrame:
        s = self.feature_types()
        span: list[tuple] = []
        last = s[0]
        last_n: int = 0
        for n, i in enumerate(s[1:], 1):
            if i != last:
                span.append((last, last_n, n))
                last_n = n
            elif n == len(s) - 1:
                span.append((last, last_n, n + 1))
            last = i
        df = pd.DataFrame(span, columns=["type", "start", "end"])
        df.index = ["ASSAY%s" % str(x + 1) for x in df.index]
        df["nFeatures"] = df.end - df.start
        return df.T

    def _auto_rename_assay_names(self) -> None:
        new_names = []
        for k, v in self.assayFeats.T["type"].to_dict().items():
            if v in self.autoNames:
                new_names.append(self.autoNames[v])
            else:
                new_names.append(k)
        self.assayFeats.columns = new_names

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
                return ret_val
        default_name = list(self.autoNames.keys())[0]
        return [default_name for _ in range(self.nFeatures)]

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
        for s in tqdmbar(
            range(0, len(indptr) - 1, batch_size),
            desc=f"Filtering out background barcodes",  # noqa: F541
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
            row_ranges = [
                np.arange(x, y) for x, y in zip(indptr[v_pos], indptr[v_pos + 1])
            ]
            cell_idx = np.repeat(
                np.arange(len(row_ranges)), [len(x) for x in row_ranges]
            )
            idx = np.hstack(row_ranges) if row_ranges else np.array([], dtype=np.int64)
            if idx.size == 0:
                yield coo_matrix(
                    ([], ([], [])),
                    shape=(len(v_pos), self.nFeatures),
                )
                continue
            data = np.asarray(self.grp["data"][idx])
            indices = np.asarray(self.grp["indices"][idx])
            yield coo_matrix(
                (data, (cell_idx, indices)), shape=(len(v_pos), self.nFeatures)
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
                f"{key} extraction failed from {grp_entry[0]} in column {grp_entry[1]}",
                flush=True,
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
        for chunk in tqdmbar(
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
        matrixIO = pd.read_csv(
            self.matFn,
            comment="%",
            sep=self.sep,
            header=0,
            chunksize=lines_in_mem,
            names=["gene", "barcode", "count"],
        )
        unique_list: list[Any] = []
        collect: list[pd.DataFrame] = []
        for chunk in matrixIO:
            chunk = chunk[chunk["barcode"].isin(self.validBarcodeIdx)]
            in_uniques = np.unique(chunk["barcode"].values)
            unique_list.extend(in_uniques.tolist())
            unique_list = list(set(unique_list))
            if len(unique_list) > batch_size:
                diff = batch_size - (len(unique_list) - len(in_uniques))
                mask_pos = in_uniques[:diff]
                mask_neg = in_uniques[diff:]
                extra = chunk[chunk["barcode"].isin(mask_pos)]
                collect.append(extra)
                batch_arr = self.rename_batches(collect)
                mtx = self.to_sparse(batch_arr, dtype=dtype)
                yield mtx
                left_out = chunk[chunk["barcode"].isin(mask_neg)]
                collect = [left_out]
                unique_list = mask_neg.tolist()
            else:
                collect.append(chunk)
        if len(collect) > 0:
            batch_arr = self.rename_batches(collect)
            mtx = self.to_sparse(batch_arr, dtype=dtype)
            yield mtx
