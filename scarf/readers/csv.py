from collections.abc import Generator
from typing import Any

import numpy as np
import pandas as pd

from ..utils.progress import iter_progress


class CSVReader:
    """A class to read in data from a CSV file.

    Args:
        csv_fn: Path to the CSV file
        has_header: Does the CSV file has a header. (Default value: True)
        id_column: The column number which contains row name. (Default value: None)
        rows_are_cells: If True then each row represents a cell and hence each column is a feature. If False then each
                        row is feature and each column in a cell
        sep: The column separator in the CSV file (Default value: ',')
        skip_rows: Number of rows to skip from the top of the file. (Default value: 0)
        skip_cols: Names of columns to skip. Must be provided as a list even if just one column.
                        (Default value: None)
        cell_data_cols: Names of columns to include in cell metadata rather than count matrix. Must be provided as a
                       list even if just one column. (Default value: None)
        batch_size: Number of lines to read at a time. Decrease this value if you have too many columns.
                    (Default value: 50,000)
        pandas_kwargs: A dictionary of keyword arguments to be passed to Pandas read_csv function.

    Attributes:
        nFeatures: Number of features in dataset.
        nCells: Number of cells in dataset.
    """

    def __init__(
        self,
        csv_fn: str,
        has_header: bool = True,
        id_column: int | None = None,
        rows_are_cells: bool = True,
        sep: str = ",",
        skip_rows: int = 0,
        skip_cols: list[str] | None = None,
        cell_data_cols: list[str] | None = None,
        batch_size: int = 10000,
        pandas_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._fn = csv_fn
        if rows_are_cells is False:
            raise NotImplementedError(
                "Currently Scarf supports only those CSV files where cells are along the rows"
            )
        if pandas_kwargs is None:
            pandas_kwargs = {}
        elif not isinstance(pandas_kwargs, dict):
            raise TypeError("pandas_kwargs must be a dictionary")
        header_row: int | None
        if has_header is False:
            header_row = None
        else:
            header_row = 0
        self.pandas_kwargs: dict[str, Any] = pandas_kwargs
        self.pandas_kwargs["sep"] = sep
        self.pandas_kwargs["header"] = header_row
        self.pandas_kwargs["skiprows"] = skip_rows
        self.pandas_kwargs["chunksize"] = batch_size
        self.pandas_kwargs["index_col"] = id_column

        if skip_cols is None:
            self.skipCols = []
        else:
            self.skipCols = skip_cols
        if cell_data_cols is None:
            self.cellDataCols = []
        else:
            self.cellDataCols = cell_data_cols
        (
            self.nCells,
            self.nFeatures,
            self.cellIds,
            self.featureIds,
            self.keepCols,
            self.cellDataDtypes,
            self.cellDataIdx,
        ) = self._consistency_check()

    def _get_streamer(self) -> Generator[pd.DataFrame, None, None]:
        reader = pd.read_csv(self._fn, **self.pandas_kwargs)
        yield from reader

    def _consistency_check(
        self,
    ) -> tuple[
        int,
        int,
        np.ndarray | None,
        np.ndarray | None,
        list[int] | None,
        list[np.dtype] | None,
        list[int] | None,
    ]:
        stream = self._get_streamer()
        n_cells = 0
        n_features = 0
        feature_ids: np.ndarray | None = None
        cell_data_dtypes: list[np.dtype] | None = None
        cell_data_idx: list[int] | None = None
        cell_ids: np.ndarray | None = None
        collected_cell_ids: list[Any] | None = None
        if self.pandas_kwargs["index_col"] is not None:
            collected_cell_ids = []
        for df in iter_progress(
            stream,
            desc="Checking CSV consistency",
        ):
            n_cells += df.shape[0]
            if n_features == 0:
                n_features = df.shape[1]
                if self.pandas_kwargs["header"] is not None:
                    feature_ids = np.asarray(df.columns.values)
                    if len(feature_ids) != n_features:
                        raise ValueError(
                            "Header length not same as number of features. This can happen if you did not"
                            " skip the right number of rows."
                        )
                    if len(self.cellDataCols) > 0:
                        cell_data_dtypes = list(df[self.cellDataCols].dtypes.values)
                        cell_data_idx = [
                            n
                            for n, x in enumerate(feature_ids)
                            if x in self.cellDataCols
                        ]
            else:
                if n_features != df.shape[1]:
                    raise ValueError(
                        "Number of columns changed in the CSV during consistency check."
                        " Maybe a problem with the delimiter."
                    )
        if collected_cell_ids is not None:
            cell_ids = np.asarray(collected_cell_ids)
        keep_cols: list[int] | None = None
        if feature_ids is not None:
            skip_names = list(set(self.skipCols).union(self.cellDataCols))
            if len(skip_names) > 0:
                keep_cols = [
                    n for n, x in enumerate(feature_ids) if x not in skip_names
                ]
                feature_ids = feature_ids[keep_cols]
                n_features = len(keep_cols)
        return (
            n_cells,
            n_features,
            cell_ids,
            feature_ids,
            keep_cols,
            cell_data_dtypes,
            cell_data_idx,
        )

    def cell_ids(self) -> np.ndarray:
        """Returns a list of cell IDs."""
        if self.cellIds is None:
            return np.array([f"cell_{x}" for x in range(self.nCells)])
        else:
            return self.cellIds

    def feature_ids(self) -> np.ndarray:
        """Returns a list of feature IDs."""
        if self.featureIds is None:
            return np.array([f"feature_{x}" for x in range(self.nFeatures)])
        else:
            return self.featureIds

    def consume(self) -> Generator[tuple[np.ndarray, np.ndarray | None], None, None]:
        """Returns a generator that yield chunks of data."""
        stream = self._get_streamer()
        if self.keepCols is None:
            for df in stream:
                yield df.values, None
        else:
            if self.cellDataIdx is not None:
                for df in stream:
                    yield df.values[:, self.keepCols], df.values[:, self.cellDataIdx]
            else:
                for df in stream:
                    yield df.values[:, self.keepCols], None
