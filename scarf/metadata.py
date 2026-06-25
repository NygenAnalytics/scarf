"""Contains the MetaData class, which is used for storing metadata about cells
and features."""

import re
from typing import Any
from collections.abc import Iterable

import numpy as np
import pandas as pd
import zarr

from ._types import as_zarr_array
from .feat_utils import fit_lowess
from .utils import logger
from .writers import create_zarr_obj_array

zarrGroup = zarr.Group

__all__ = ["MetaData"]


def _all_true(bools: np.ndarray) -> np.ndarray:
    """

    Args:
        bools:

    Returns:

    """
    a = bools.sum(axis=0)
    a[a < bools.shape[0]] = 0
    return np.asarray(a, dtype=bool)


class MetaData:
    """MetaData class for cells and features.

    This class provides an interface to perform CRUD operations on metadata, saved in the Zarr hierarchy.
    All the changes at the metadata are synchronized on disk.

    Attributes:
        locations: The locations for where the metadata is stored.
        N: The size of the primary data.
        index: A numpy array with the indices of the cells/features.
    """

    def __init__(self, zgrp: zarrGroup):
        """
        Args:
            zgrp: Zarr hierarchy object wherein metadata arrays are saved.
        """
        self.locations: dict[str, zarrGroup] = {"primary": zgrp}
        self.N = self._get_size(self.locations["primary"], strict_mode=True)
        self.index = np.array(range(self.N))

    def _get_size(self, zgrp: zarrGroup, strict_mode: bool = False) -> int:
        """

        Args:
            zgrp:
            strict_mode:

        Returns:

        """
        sizes = []
        for i in zgrp.keys():
            try:
                child = zgrp[i]
                if isinstance(child, zarr.Array):
                    sizes.append(child.shape[0])
            except Exception:
                pass
        if len(sizes) > 0:
            if len(set(sizes)) != 1:
                raise ValueError(
                    "ERROR: Metadata table is corrupted. Not all columns are of same length"
                )
            return sizes[0]
        else:
            if strict_mode:
                raise ValueError("Attempted to get size of empty zarr group")
            else:
                return self.N

    @staticmethod
    def _col_renamer(loc: str, col: str) -> str:
        """

        Args:
            loc:
            col:

        Returns:

        """
        if loc != "primary":
            return f"{loc}_{col}"
        return col

    def _column_map(self) -> dict[str, str | tuple[str, str]]:
        """

        Returns:

        """
        reserved_cols = ["I", "ids", "names"]
        col_map: dict[str, str | tuple[str, str]] = {
            x: "primary" for x in reserved_cols
        }
        for loc, zgrp in self.locations.items():
            for i in zgrp.keys():
                j = self._col_renamer(loc, i)
                if j in col_map and j not in reserved_cols:
                    logger.warning(
                        f" {i} is duplicate in metadata loc {loc}. This means something has failed "
                        f"upstream. This is quite unexpected. Please report this issue."
                    )
                col_map[j] = (loc, i)
        return col_map

    def _get_loc(self, column: str) -> tuple[str, str]:
        """

        Args:
            column:

        Returns:

        """
        col_map = self._column_map()
        if column not in col_map:
            raise KeyError(f"{column} does not exist in the metadata columns.")
        entry = col_map[column]
        if isinstance(entry, str):
            return "primary", column
        loc, col = entry
        return loc, col

    def _get_array(self, column: str) -> zarr.Array:
        """

        Args:
            column:

        Returns:

        """
        loc, col = self._get_loc(column)
        return as_zarr_array(self.locations[loc][col], name=col)

    def get_dtype(self, column: str) -> np.dtype[Any]:
        """Returns the dtype for the given column.

        Args:
            column: Column name of the table.
        """
        return self._get_array(column).dtype

    def _verify_bool(self, key: str) -> bool:
        """Validates if a give table column (parameter 'key') is bool type.

        Args:
            key: Name of the column to query

        Returns: None
        """

        if self.get_dtype(key) is not bool:
            raise TypeError(
                "ERROR: `key` should be name of a boolean type column in Metadata table"
            )
        return True

    def mount_location(self, zgrp: zarrGroup, identifier: str) -> None:
        """

        Args:
            zgrp:
            identifier:

        Returns:

        """
        if identifier in self.locations:
            raise ValueError(
                f"ERROR: a location with identifier '{identifier}' already mounted"
            )
        size = self._get_size(zgrp)
        if size != self.N:
            raise ValueError(
                f"ERROR: The index size of the mount location ({size}) is not same as primary ({self.N})"
            )
        new_cols = [self._col_renamer(identifier, x) for x in zgrp.keys()]
        cols = self.columns
        conflict_names = [x for x in new_cols if x in cols]
        if len(conflict_names) > 0:
            conflict_str = " ".join(conflict_names)
            raise ValueError(
                f"ERROR: These names in location conflict with existing names: {conflict_str}\n. "
                f"Please try with a different identifier value."
            )
        self.locations[identifier] = zgrp

    def unmount_location(self, identifier: str) -> None:
        """

        Args:
            identifier:

        Returns:

        """
        if identifier == "primary":
            raise ValueError("Cannot unmount the primary location")
        if identifier not in self.locations:
            logger.warning(f"{identifier} is not mounted. Nothing to unmount")
            return None
        self.locations.pop(identifier)

    @property
    def columns(self) -> list[str]:
        """

        Returns:

        """
        return list(self._column_map().keys())

    def fetch_all(self, column: str) -> np.ndarray:
        """

        Args:
            column:

        Returns:

        """
        return np.asarray(self._get_array(column)[:])

    def active_index(self, key: str) -> np.ndarray:
        """

        Args:
            key:

        Returns:

        """
        if self._verify_bool(key):
            return np.asarray(self.index[self.fetch_all(key)])
        else:
            raise ValueError(
                "ERROR: Unexpected error when verifying boolean key. Please report this issue"
            )

    def fetch(self, column: str, key: str = "I") -> np.ndarray:
        """Get column values for only valid rows.

        Args:
            column:
            key:

        Returns:
        """

        return np.asarray(self.fetch_all(column)[self.active_index(key)])

    def _save(
        self, column_name: str, values: np.ndarray, location: str = "primary"
    ) -> None:
        """

        Args:
            column_name:
            values:

        Returns:

        """
        if location not in self.locations:
            raise KeyError(
                f"ERROR: '{location}' has not been mounted. Save data request failed!"
            )
        if values.shape != (self.N,):
            raise ValueError(
                f"ERROR: Values are of shape: {values.shape}. Expected shape is: ({self.N},)"
            )
        create_zarr_obj_array(
            self.locations[location], column_name, values, values.dtype
        )
        return None

    def _fill_to_index(
        self, values: np.ndarray, fill_value: Any, key: str, auto_fill_disable: bool = False
    ) -> np.ndarray:
        """Makes sure that the array being added to the table is of the same
        shape. If the array has same shape as the table then the input array is
        added straightaway. It is assumed that the input array is in same order
        as table rows.

        Args:
            values:
            fill_value:
            key:
            auto_fill_disable (bool = False): `fill_value` autocorrection based on dytype is disabled.

        Returns:
        """

        if isinstance(values, np.ndarray) is False:
            values = np.array(values)
        if auto_fill_disable is False:
            if values.dtype == bool:
                fill_value = False
            elif np.issubdtype(values.dtype, np.integer):
                try:
                    if np.isnan(fill_value):
                        if min(values) > -1:
                            fill_value = 0
                        else:
                            raise ValueError("`fill_value should be an integer value. ")
                except TypeError:
                    raise ValueError("`fill_value should be an interger value. ")

        n = values.shape[0]
        if n == self.N:
            return values
        else:
            self._verify_bool(key)
            k = self.fetch_all(key)
            v_n = k.sum()
            if len(values) != v_n:
                raise ValueError(
                    f"ERROR: `values`  are of incorrect length ({n}). "
                    f" Chosen key ({key}) has {v_n} active rows"
                )
            else:
                a = np.empty(self.N).astype(values.dtype)
                a[k] = values
                a[~k] = fill_value
                return a

    def get_index_by(
        self, value_targets: list[Any], column: str, key: str | None = None
    ) -> np.ndarray:
        """

        Args:
            value_targets:
            column:
            key:
        Returns:

        """
        if isinstance(value_targets, Iterable) and not isinstance(value_targets, str):
            if key is None:
                values = self.fetch_all(column)
            else:
                values = self.fetch(column, key)
            value_map: dict[str, list[int]] = {}
            for n, x in enumerate(values):
                x = x.upper()
                if x not in value_map:
                    value_map[x] = []
                value_map[x].append(n)
            ret_val = []
            missing_count = 0
            for i in value_targets:
                i = i.upper()
                if i in value_map:
                    ret_val.extend(value_map[i])
                else:
                    missing_count += 1
            if missing_count > 0:
                logger.warning(
                    f"{missing_count} values were not found in the table column {column}"
                )
            return np.array(ret_val)
        else:
            raise TypeError("ERROR: Please provide the `value_targets` as list")

    def index_to_bool(self, idx: np.ndarray, invert: bool = False) -> np.ndarray:
        """

        Args:
            idx:
            invert:

        Returns:

        """
        a = np.zeros(self.N, dtype=bool)
        if len(idx) > 0:
            a[idx] = True
        if invert:
            a = ~a
        return a

    def insert(
        self,
        column_name: str,
        values: np.ndarray | list,
        fill_value: Any = np.nan,
        key: str = "I",
        overwrite: bool = False,
        location: str = "primary",
        force: bool = False,
    ) -> None:
        """Insert a column into the table.

        Args:
            column_name (str): Name of column to modify.
            values (np.array): Values the column should contain.
            fill_value (Any = np.nan): Value to fill unassigned slots with.
            key (str = 'I'):
            overwrite (bool = False): Should function overwrite column if it already exists?
            location (str = 'primary'):
            force (bool = False): Enforce change to column, even if column is a protected column name ('I' or 'ids').

        Returns:
            None
        """
        col = self._col_renamer(location, column_name)
        if col in ["I", "ids"] and force is False:
            raise ValueError(
                f"ERROR: {col} is a protected column name in MetaData class."
            )
        if col in self.columns and overwrite is False:
            raise ValueError(
                f"ERROR: {col} already exists. Please set `overwrite` to True to overwrite."
            )
        if isinstance(values, list):
            logger.debug(
                "'values' parameter is of `list` type and not `np.ndarray` as expected. The correct dtype "
                "may not be assigned to the column"
            )
        v = self._fill_to_index(np.array(values), fill_value, key)
        self._save(column_name, v, location=location)
        return None

    def update_key(self, values: np.ndarray, key: str) -> None:
        """Modify a column in the metadata table, specified with `key`.

        Args:
            values: The values to update the column with.
            key: Which column in the metadata table to update.

        Returns:
            None
        """
        v = self._fill_to_index(values, False, key)
        v = _all_true(np.array([v, self.fetch_all(key)]))
        self._save(key, v)
        return None

    def reset_key(self, key: str) -> None:
        """

        Args:
            key:

        Returns:

        """
        values = np.array([True for _ in range(self.N)]).astype(bool)
        self._save(key, values)
        return None

    def drop(self, column: str) -> None:
        """

        Args:
            column:

        Returns:

        """
        if column in ["I", "ids", "names"]:
            raise ValueError(
                f"ERROR: {column} is a protected name in MetaData class. Cannot be deleted"
            )
        # noinspection PyUnusedLocal
        loc, col = self._get_loc(column)
        del self.locations[loc][col]
        return None

    def sift(
        self,
        column: str,
        min_v: float = -np.inf,
        max_v: float = np.inf,
        keep_bounds: bool = False,
    ) -> np.ndarray:
        """

        Args:
            column:
            min_v:
            max_v:
            keep_bounds:

        Returns:

        """
        values = self.fetch_all(column)
        if keep_bounds:
            return (values >= min_v) & (values <= max_v)
        else:
            return (values > min_v) & (values < max_v)

    def multi_sift(
        self,
        columns: list[str],
        lows: Iterable,
        highs: Iterable,
        keep_bounds: bool = False,
    ) -> np.ndarray:
        """Return boolean mask where all column filters are satisfied."""
        ret_val = _all_true(
            np.array(
                [
                    self.sift(i, j, k, keep_bounds=keep_bounds)
                    for i, j, k in zip(columns, lows, highs)
                ]
            )
        )
        return ret_val

    def head(self, n: int = 5) -> pd.DataFrame:
        """Return first ``n`` rows of all metadata columns as a DataFrame."""
        df = pd.DataFrame({x: self.fetch_all(x)[:n] for x in self.columns})
        return df

    def to_pandas_dataframe(
        self, columns: list[str], key: str | None = None
    ) -> pd.DataFrame:
        """Return requested columns as a DataFrame, optionally filtered by ``key``.

        Args:
            columns: Metadata column names to include.
            key: If set, restrict rows to those active for this boolean key.

        Returns:
            Pandas DataFrame of the selected columns.
        """
        valid_cols = self.columns
        df = pd.DataFrame({x: self.fetch_all(x) for x in columns if x in valid_cols})
        if key is not None:
            df = df.reindex(self.active_index(key))
        return df

    def grep(self, pattern: str, only_valid: bool = False) -> list[str]:
        """Return feature names matching a regex pattern (case insensitive).

        Args:
            pattern: Regular expression matched against ``names`` column.
            only_valid: Restrict to features active in ``I``.

        Returns:
            Sorted list of matching names.
        """
        names = np.array(list(map(str.upper, self.fetch_all("names"))))
        if only_valid:
            names = names[self.active_index("I")]
        return sorted(
            set([x for x in names if re.match(pattern.upper(), x) is not None])
        )

    def remove_trend(
        self,
        x: str,
        y: str,
        n_bins: int = 200,
        lowess_frac: float = 0.1,
        fill_value: float = 0,
    ) -> np.ndarray:
        """Remove a LOWESS trend of column ``y`` with respect to column ``x``.

        Args:
            x: Column name for the independent variable.
            y: Column name for the dependent variable.
            n_bins: Bins for LOWESS fitting.
            lowess_frac: LOWESS smoothing fraction.
            fill_value: Value returned where trend cannot be estimated.

        Returns:
            Residual values of ``y`` after trend removal.
        """
        a = self.fetch(x).astype(float)
        b = self.fetch(y).astype(float)
        idx = a > 0
        c = fit_lowess(
            a[idx],
            b[idx],
            n_bins,
            lowess_frac,
        )
        ret_val = np.repeat(fill_value, len(a)).astype(float)
        ret_val[idx] = c
        return ret_val

    def __repr__(self) -> str:
        return f"MetaData of {self.fetch_all('I').sum()}({self.N}) elements"
