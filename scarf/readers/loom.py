from collections.abc import Generator

import h5py
import numpy as np
from scipy.sparse import coo_matrix

from ..utils.logging import logger
from ..utils.progress import iter_progress


class LoomReader:
    """A class to read in data in the form of a Loom file.

    Args:
        loom_fn: Path to loom format file.
        matrix_key: Child node under HDF5 file root wherein the chunked matrix is stored. (Default value: matrix).
                    This matrix is expected to be of form (nFeatures x nCells)
        cell_attrs_key: Child node under the HDF5 file wherein the cell attributes are stored.
                        (Default value: col_attrs)
        cell_names_key: Child node under the `cell_attrs_key` wherein the cell names are stored.
                        (Default value: obs_names)
        feature_attrs_key: Child node under the HDF5 file wherein the feature/gene attributes are stored.
                           (Default value: row_attrs)
        feature_names_key: Child node under the `feature_attrs_key` wherein the feature/gene names are stored.
                           (Default value: var_names)
        feature_ids_key: Child node under the `feature_attrs_key` wherein the feature/gene ids are stored.
                         (Default value: None)
        dtype: Numpy dtype of the matrix data. This dtype is enforced when streaming the data through `consume`
               method. (Default value: Automatically determined)

    Attributes:
        h5: A File object from the h5py package.
        matrixKey: Child node under HDF5 file root wherein the chunked matrix is stored.
        cellAttrsKey: Child node under the HDF5 file wherein the cell attributes are stored.
        featureAttrsKey: Child node under the HDF5 file wherein the feature/gene attributes are stored.
        cellNamesKey: Child node under the `cell_attrs_key` wherein the cell names are stored.
        featureNamesKey: Child node under the `feature_attrs_key` wherein the feature/gene names are stored.
        featureIdsKey: Child node under the `feature_attrs_key` wherein the feature/gene ids are stored.
        matrixDtype: Numpy dtype of the matrix data.
        nFeatures: Number of features in dataset.
        nCells: Number of cells in dataset.
    """

    def __init__(
        self,
        loom_fn: str,
        matrix_key: str = "matrix",
        cell_attrs_key: str = "col_attrs",
        cell_names_key: str = "obs_names",
        feature_attrs_key: str = "row_attrs",
        feature_names_key: str = "var_names",
        feature_ids_key: str | None = None,
        dtype: str | None = None,
    ) -> None:
        self.h5: h5py.File = h5py.File(loom_fn, mode="r")
        self.matrixKey = matrix_key
        self.cellAttrsKey, self.featureAttrsKey = cell_attrs_key, feature_attrs_key
        self.cellNamesKey, self.featureNamesKey = cell_names_key, feature_names_key
        self.featureIdsKey = feature_ids_key
        self.sourceMatrixDtype = self.h5[self.matrixKey].dtype
        self.matrixDtype = self.sourceMatrixDtype if dtype is None else dtype
        self._check_integrity()
        self.nFeatures, self.nCells = self.h5[self.matrixKey].shape

    def _check_integrity(self) -> bool:
        if self.matrixKey not in self.h5:
            raise KeyError(
                f"ERROR: Matrix key (location): {self.matrixKey} is missing in the H5 file"
            )
        if self.cellAttrsKey not in self.h5:
            logger.warning(
                f"Cell attributes are missing. Key {self.cellAttrsKey} was not found"
            )
        if self.featureAttrsKey not in self.h5:
            logger.warning(
                f"Feature attributes are missing. Key {self.featureAttrsKey} was not found"
            )
        return True

    def cell_names(self) -> list[str]:
        """Returns a list of names of the cells in the dataset."""
        if self.cellAttrsKey not in self.h5:
            pass
        elif self.cellNamesKey not in self.h5[self.cellAttrsKey]:
            logger.warning(
                f"Cell names/ids key ({self.cellNamesKey}) is missing in attributes"
            )
        else:
            return list(self.h5[self.cellAttrsKey][self.cellNamesKey][:])
        return [f"cell_{x}" for x in range(self.nCells)]

    def cell_ids(self) -> list[str]:
        """Returns a list of cell IDs."""
        return self.cell_names()

    def _stream_attrs(
        self, key: str, ignore: str | list[str] | None
    ) -> Generator[tuple[str, np.ndarray], None, None]:
        ignored: set[str]
        if isinstance(ignore, str):
            ignored = {ignore}
        elif ignore is None:
            ignored = set()
        else:
            ignored = {name for name in ignore if name is not None}
        if key in self.h5:
            for i in iter_progress(
                self.h5[key].keys(),
                desc=f"Reading {key} attributes",
            ):
                if i in ignored:
                    continue
                vals = self.h5[key][i][:]
                if vals.dtype.names is None:
                    yield i, vals
                else:
                    # Attribute is a structured array
                    for j in vals.dtype.names:
                        yield i + "_" + str(j), vals[j]

    def get_cell_attrs(self) -> Generator[tuple[str, np.ndarray], None, None]:
        """Returns a Generator that yields the cells' attributes."""
        return self._stream_attrs(self.cellAttrsKey, [self.cellNamesKey])

    def feature_names(self) -> list[str]:
        """Returns a list of feature names."""
        if self.featureAttrsKey not in self.h5:
            pass
        elif self.featureNamesKey not in self.h5[self.featureAttrsKey]:
            logger.warning(
                f"Feature names key ({self.featureNamesKey}) is missing in attributes"
            )
        else:
            return list(self.h5[self.featureAttrsKey][self.featureNamesKey][:])
        return [f"feature_{x}" for x in range(self.nFeatures)]

    def feature_ids(self) -> list[str]:
        """Returns a list of feature IDs."""
        if self.featureAttrsKey not in self.h5:
            pass
        elif self.featureIdsKey is None:
            pass
        elif self.featureIdsKey not in self.h5[self.featureAttrsKey]:
            logger.warning(
                f"Feature names key ({self.featureIdsKey}) is missing in attributes"
            )
        else:
            return list(self.h5[self.featureAttrsKey][self.featureIdsKey][:])
        return [f"feature_{x}" for x in range(self.nFeatures)]

    def get_feature_attrs(self) -> Generator[tuple[str, np.ndarray], None, None]:
        """Returns a Generator that yields the features' attributes."""
        ignore_keys = [
            key for key in (self.featureIdsKey, self.featureNamesKey) if key is not None
        ]
        return self._stream_attrs(self.featureAttrsKey, ignore_keys)

    def consume(self, batch_size: int = 1000) -> Generator[np.ndarray, None, None]:
        """Returns a generator that yield chunks of data."""
        dset = self.h5[self.matrixKey]
        s = 0
        for e in range(batch_size, dset.shape[1] + batch_size, batch_size):
            if e > dset.shape[1]:
                e = dset.shape[1]
            yield coo_matrix(dset[:, s:e]).T.astype(self.matrixDtype)
            s = e
