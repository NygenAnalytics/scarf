import csv
import itertools
from collections.abc import Generator
from typing import Any

import numpy as np
import pandas as pd
from pandas.io.common import get_handle

from ..utils.count_values import CountValueRange
from ..utils.progress import iter_progress
from ._text import require_unique_identifiers

# read_csv settings that, besides the separator and ``quoting``, decide how
# pandas splits a line into fields. The field check splits with the same ones.
_QUOTING_SETTINGS = ("quotechar", "doublequote", "escapechar", "skipinitialspace")
# read_csv settings that split lines in ways Python's csv module cannot follow.
_UNSUPPORTED_SETTINGS = frozenset({"comment", "dialect", "lineterminator"})


def _count_dtype(frame: pd.DataFrame, value_range: CountValueRange) -> np.dtype[Any]:
    """Return the dtype of one chunk of count columns after validating it.

    The chunk's values are added to ``value_range``.
    """
    for dtype in frame.dtypes:
        if pd.api.types.is_bool_dtype(dtype) or not pd.api.types.is_numeric_dtype(
            dtype
        ):
            raise ValueError(
                "CSV count columns must contain numbers; move text columns to "
                "cell_data_cols or skip_cols"
            )
    if frame.shape[1] == 0:
        # An assay without features cannot be opened, so the import stops here.
        raise ValueError(
            "CSV file contains no count columns; every column is the ID column "
            "or is listed in cell_data_cols or skip_cols"
        )
    # Column reductions avoid copying the chunk into one array.
    minimum = frame.min(axis=0, skipna=False).to_numpy(dtype=np.float64)
    maximum = frame.max(axis=0, skipna=False).to_numpy(dtype=np.float64)
    if not (np.isfinite(minimum).all() and np.isfinite(maximum).all()):
        raise ValueError("CSV counts contain missing or non-finite values")
    if bool((minimum < 0).any()):
        raise ValueError("CSV counts must not be negative")
    if value_range.integral:
        value_range.update(frame.to_numpy())
    return np.dtype(np.result_type(*frame.dtypes))


def _metadata_dtype(series: pd.Series) -> np.dtype[Any]:
    """Return a numeric dtype for numeric columns and ``object`` for text."""
    dtype = series.dtype
    resolved: np.dtype[Any] = np.dtype(object)
    if pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_numeric_dtype(dtype):
        resolved = np.dtype(dtype)
    return resolved


def _merged_dtype(current: np.dtype[Any] | None, dtype: np.dtype[Any]) -> np.dtype[Any]:
    if current is None:
        return dtype
    if current.kind == "O" or dtype.kind == "O":
        return np.dtype(object)
    return np.result_type(current, dtype)


class CSVReader:
    """A class to read in data from a CSV file.

    Construction checks that every row has as many fields as the header, or
    as the first row of a file without one, and raises ValueError otherwise:
    pandas reads in chunks, pads a short row with missing values, and drops
    the extra fields of a row that starts a chunk. Python's csv module counts
    the fields with ``sep`` and the ``quotechar``, ``quoting``,
    ``doublequote``, ``escapechar``, and ``skipinitialspace`` settings that
    pandas splits the rows with.

    Args:
        csv_fn: Path to the CSV file
        has_header: Does the CSV file has a header. (Default value: True)
        id_column: The column number which contains row name. (Default value: None)
        rows_are_cells: If True then each row represents a cell and hence each column is a feature. If False then each
                        row is feature and each column in a cell
        sep: The column separator in the CSV file, one character. (Default value: ',')
        skip_rows: Number of rows to skip from the top of the file. (Default value: 0)
        skip_cols: Names of columns to skip. Must be provided as a list even if just one column.
                        (Default value: None)
        cell_data_cols: Names of columns to include in cell metadata rather than count matrix. Must be provided as a
                       list even if just one column. (Default value: None)
        batch_size: Number of lines to read at a time. Decrease this value if you have too many columns.
                    (Default value: 50,000)
        pandas_kwargs: A dictionary of keyword arguments to be passed to Pandas read_csv function.
                       It cannot set ``comment``, ``dialect``, or ``lineterminator``.

    Attributes:
        nFeatures: Number of features in dataset.
        nCells: Number of cells in dataset.
        countDtype: Dtype that holds the count values of every row.
        countRange: Range of the count values of every row, which decides
                    their storage dtype.
        cellDataDtypes: Dtype of each ``cell_data_cols`` column across every
                        row; ``object`` marks a text column.
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
        if len(sep) != 1:
            raise ValueError("sep must be one character")
        unsupported = sorted(_UNSUPPORTED_SETTINGS.intersection(pandas_kwargs))
        if unsupported:
            raise ValueError(
                f"pandas_kwargs cannot set {', '.join(unsupported)}: the reader "
                "checks the fields of every row with sep and the quoting "
                "settings alone"
            )
        header_row: int | None
        if has_header is False:
            if skip_cols or cell_data_cols:
                raise ValueError("Named columns require a CSV header")
            header_row = None
        else:
            header_row = 0
        # A copy, so that the reader's settings stay out of the caller's dict.
        self.pandas_kwargs: dict[str, Any] = dict(pandas_kwargs)
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
            self.countDtype,
            self.countRange,
        ) = self._consistency_check()

    def _get_streamer(self) -> Generator[pd.DataFrame, None, None]:
        reader = pd.read_csv(self._fn, **self.pandas_kwargs)
        yield from reader

    def _check_field_counts(self) -> None:
        """Raise ValueError unless every row has as many fields as the first.

        The rows are the ones pandas reads: the text that ``read_csv`` opens,
        after ``skip_rows`` rows, without blank lines.
        """
        kwargs = self.pandas_kwargs
        settings = {key: kwargs[key] for key in _QUOTING_SETTINGS if key in kwargs}
        # read_csv opens its source with this function, so the check reads the
        # text that pandas parses.
        with get_handle(
            self._fn,
            "r",
            encoding=kwargs.get("encoding"),
            compression=kwargs.get("compression", "infer"),
            errors=kwargs.get("encoding_errors", "strict"),
            storage_options=kwargs.get("storage_options"),
        ) as handles:
            lines = iter(handles.handle)
            # pandas drops a byte order mark before it splits the first line.
            first = next(lines, "").removeprefix("\ufeff")
            rows = csv.reader(
                itertools.chain((first,), lines),
                delimiter=kwargs["sep"],
                # pandas reads quotes under every quoting setting but
                # QUOTE_NONE, and the csv module would convert unquoted fields
                # under QUOTE_NONNUMERIC.
                quoting=(
                    csv.QUOTE_NONE
                    if kwargs.get("quoting") == csv.QUOTE_NONE
                    else csv.QUOTE_MINIMAL
                ),
                **settings,
            )
            expected: int | None = None
            expected_line = 0
            for row in itertools.islice(rows, kwargs["skiprows"], None):
                count = len(row)
                # pandas skips lines that hold nothing but spaces and tabs.
                if count == expected or (count <= 1 and not "".join(row).strip(" \t")):
                    continue
                if expected is not None:
                    raise ValueError(
                        f"CSV line {rows.line_num} has {count} fields, but line "
                        f"{expected_line} has {expected}"
                    )
                expected = count
                expected_line = rows.line_num

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
        np.dtype,
        CountValueRange,
    ]:
        """Stream every row once to fix the shape, dtypes, and count range.

        The field count of every row is checked first.
        """
        self._check_field_counts()
        stream = self._get_streamer()
        n_cells = 0
        n_features = 0
        feature_ids: np.ndarray | None = None
        keep_cols: list[int] | None = None
        cell_data_dtypes: list[np.dtype] | None = None
        cell_data_idx: list[int] | None = None
        count_dtype: np.dtype | None = None
        count_range = CountValueRange()
        collected_cell_ids: list[Any] | None = None
        if self.pandas_kwargs["index_col"] is not None:
            collected_cell_ids = []
        for df in iter_progress(
            stream,
            desc="Checking CSV consistency",
        ):
            if df.shape[0] == 0:
                # A header-only file yields one empty chunk of untyped columns.
                continue
            n_cells += df.shape[0]
            if collected_cell_ids is not None:
                collected_cell_ids.extend(df.index.to_numpy())
            if n_features == 0:
                n_features = df.shape[1]
                if self.pandas_kwargs["header"] is not None:
                    feature_ids = np.asarray(df.columns.values)
                    for option, names in (
                        ("cell_data_cols", self.cellDataCols),
                        ("skip_cols", self.skipCols),
                    ):
                        absent = [name for name in names if name not in df.columns]
                        if absent:
                            raise KeyError(f"{option} are not CSV columns: {absent}")
                    if len(self.cellDataCols) > 0:
                        cell_data_idx = df.columns.get_indexer(
                            self.cellDataCols
                        ).tolist()
                    skip_names = set(self.skipCols).union(self.cellDataCols)
                    if skip_names:
                        keep_cols = [
                            n for n, x in enumerate(feature_ids) if x not in skip_names
                        ]
            counts = df if keep_cols is None else df.iloc[:, keep_cols]
            count_dtype = _merged_dtype(count_dtype, _count_dtype(counts, count_range))
            if cell_data_idx is not None:
                chunk_dtypes = [
                    _metadata_dtype(df.iloc[:, index]) for index in cell_data_idx
                ]
                cell_data_dtypes = (
                    chunk_dtypes
                    if cell_data_dtypes is None
                    else [
                        _merged_dtype(current, dtype)
                        for current, dtype in zip(
                            cell_data_dtypes, chunk_dtypes, strict=True
                        )
                    ]
                )
        if count_dtype is None:
            raise ValueError("CSV file contains no data rows")
        cell_ids: np.ndarray | None = None
        if collected_cell_ids is not None:
            cell_ids = np.asarray(collected_cell_ids)
            require_unique_identifiers(cell_ids, "CSV cell IDs")
        if feature_ids is not None and keep_cols is not None:
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
            count_dtype,
            count_range,
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
                    yield (
                        df.iloc[:, self.keepCols].to_numpy(),
                        df.iloc[:, self.cellDataIdx].to_numpy(dtype=object),
                    )
            else:
                for df in stream:
                    yield df.iloc[:, self.keepCols].to_numpy(), None
