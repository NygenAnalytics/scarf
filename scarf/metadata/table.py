from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
import pandas as pd
import zarr

from ..storage.types import as_zarr_array
from ..storage.arrays import create_zarr_obj_array
from ..utils.logging import logger
from .queries import (
    _all_true,
    grep as _grep,
    head as _head,
    multi_sift as _multi_sift,
    remove_trend as _remove_trend,
    sift as _sift,
    to_pandas_dataframe as _to_pandas_dataframe,
)
from .rows import (
    MetaDataRowBlock,
    default_block_rows as _default_block_rows,
    iter_row_blocks as _iter_row_blocks,
)

zarrGroup = zarr.Group
_INTERNAL_METADATA_PREFIX = "__scarf_missing__"


class MetaData:
    """Metadata table for cells and features backed by Zarr arrays.

    Changes made through this class are synchronized with the backing store.
    """

    def __init__(self, zgrp: zarrGroup):
        self.locations: dict[str, zarrGroup] = {"primary": zgrp}
        self.N = self._get_size(self.locations["primary"], strict_mode=True)
        self.index = np.array(range(self.N))

    def _get_size(self, zgrp: zarrGroup, strict_mode: bool = False) -> int:
        sizes = []
        for key in zgrp.keys():
            try:
                child = zgrp[key]
                if isinstance(child, zarr.Array):
                    sizes.append(child.shape[0])
            except Exception:
                pass
        if sizes:
            if len(set(sizes)) != 1:
                raise ValueError(
                    "ERROR: Metadata table is corrupted. Not all columns are "
                    "of same length"
                )
            return sizes[0]
        if strict_mode:
            raise ValueError("Attempted to get size of empty zarr group")
        return self.N

    @staticmethod
    def _col_renamer(loc: str, col: str) -> str:
        if loc != "primary":
            return f"{loc}_{col}"
        return col

    def _column_map(self) -> dict[str, str | tuple[str, str]]:
        reserved_cols = ["I", "ids", "names"]
        col_map: dict[str, str | tuple[str, str]] = {
            column: "primary" for column in reserved_cols
        }
        for location, group in self.locations.items():
            for column in group.keys():
                if column.startswith(_INTERNAL_METADATA_PREFIX):
                    continue
                public_name = self._col_renamer(location, column)
                if public_name in col_map and public_name not in reserved_cols:
                    logger.warning(
                        f" {column} is duplicate in metadata loc {location}. "
                        "This means something has failed upstream. This is "
                        "quite unexpected. Please report this issue."
                    )
                col_map[public_name] = (location, column)
        return col_map

    def _get_loc(self, column: str) -> tuple[str, str]:
        col_map = self._column_map()
        if column not in col_map:
            raise KeyError(f"{column} does not exist in the metadata columns.")
        entry = col_map[column]
        if isinstance(entry, str):
            return "primary", column
        location, stored_column = entry
        return location, stored_column

    def _get_array(self, column: str) -> zarr.Array:
        location, stored_column = self._get_loc(column)
        return as_zarr_array(
            self.locations[location][stored_column],
            name=stored_column,
        )

    def _get_missing_mask_array(self, column: str) -> zarr.Array | None:
        location, stored_column = self._get_loc(column)
        group = self.locations[location]
        output = as_zarr_array(group[stored_column], name=stored_column)
        missing_name = output.attrs.get("missing_mask")
        if not isinstance(missing_name, str) or missing_name not in group:
            return None
        return as_zarr_array(group[missing_name], name=missing_name)

    def get_dtype(self, column: str) -> np.dtype[Any]:
        """Return the dtype of a metadata column."""
        return self._get_array(column).dtype

    def _verify_bool(self, key: str) -> bool:
        if self.get_dtype(key) != bool:  # noqa: E721
            raise TypeError(
                "ERROR: `key` should be name of a boolean type column in Metadata table"
            )
        return True

    def mount_location(self, zgrp: zarrGroup, identifier: str) -> None:
        if identifier in self.locations:
            raise ValueError(
                f"ERROR: a location with identifier '{identifier}' already mounted"
            )
        size = self._get_size(zgrp)
        if size != self.N:
            raise ValueError(
                f"ERROR: The index size of the mount location ({size}) is not "
                f"same as primary ({self.N})"
            )
        new_cols = [self._col_renamer(identifier, column) for column in zgrp.keys()]
        conflict_names = [column for column in new_cols if column in self.columns]
        if conflict_names:
            conflict_str = " ".join(conflict_names)
            raise ValueError(
                "ERROR: These names in location conflict with existing names: "
                f"{conflict_str}\n. Please try with a different identifier value."
            )
        self.locations[identifier] = zgrp

    def unmount_location(self, identifier: str) -> None:
        if identifier == "primary":
            raise ValueError("Cannot unmount the primary location")
        if identifier not in self.locations:
            logger.warning(f"{identifier} is not mounted. Nothing to unmount")
            return None
        self.locations.pop(identifier)

    @property
    def columns(self) -> list[str]:
        """Return all mounted metadata column names."""
        return list(self._column_map().keys())

    def fetch_all(self, column: str) -> np.ndarray:
        """Return all values from a metadata column."""
        return np.asarray(self._get_array(column)[:])

    def active_index(self, key: str) -> np.ndarray:
        """Return global row indices selected by a boolean column."""
        if self._verify_bool(key):
            return np.asarray(self.index[self.fetch_all(key)])
        raise ValueError(
            "ERROR: Unexpected error when verifying boolean key. "
            "Please report this issue"
        )

    def fetch(self, column: str, key: str = "I") -> np.ndarray:
        """Return column values for rows selected by ``key``."""
        return np.asarray(self.fetch_all(column)[self.active_index(key)])

    def default_block_rows(self, column: str = "I") -> int:
        """Prefer the Zarr chunk length of ``column`` for row iteration."""
        return _default_block_rows(self, column)

    def iter_row_blocks(
        self,
        *,
        cell_key: str = "I",
        columns: Iterable[str] | None = None,
        block_rows: int | None = None,
    ) -> Iterator[MetaDataRowBlock]:
        """Yield contiguous row blocks over this table.

        Each block covers a half-open global index range ``[start, stop)``.
        ``active_global_indices`` lists rows in that range that pass
        ``cell_key``. Column arrays are aligned to those active indices only.
        """
        return _iter_row_blocks(
            self,
            cell_key=cell_key,
            columns=columns,
            block_rows=block_rows,
        )

    def _save(
        self,
        column_name: str,
        values: np.ndarray,
        location: str = "primary",
    ) -> None:
        if location not in self.locations:
            raise KeyError(
                f"ERROR: '{location}' has not been mounted. Save data request failed!"
            )
        if values.shape != (self.N,):
            raise ValueError(
                f"ERROR: Values are of shape: {values.shape}. "
                f"Expected shape is: ({self.N},)"
            )
        create_zarr_obj_array(
            self.locations[location],
            column_name,
            values,
            values.dtype,
        )

    def _fill_to_index(
        self,
        values: np.ndarray,
        fill_value: Any,
        key: str,
        auto_fill_disable: bool = False,
    ) -> np.ndarray:
        """Fill values that do not cover every metadata row."""
        if not isinstance(values, np.ndarray):
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
                            raise ValueError("`fill_value` should be an integer value.")
                except TypeError:
                    raise ValueError("`fill_value` should be an integer value.")

        n_values = values.shape[0]
        if n_values == self.N:
            return values

        self._verify_bool(key)
        selected = self.fetch_all(key)
        selected_count = selected.sum()
        if len(values) != selected_count:
            raise ValueError(
                f"ERROR: `values`  are of incorrect length ({n_values}). "
                f" Chosen key ({key}) has {selected_count} active rows"
            )
        filled = np.empty(self.N, dtype=values.dtype)
        filled[selected] = values
        filled[~selected] = fill_value
        return filled

    def get_index_by(
        self,
        value_targets: list[Any],
        column: str,
        key: str | None = None,
    ) -> np.ndarray:
        """Return row indices for requested values in a metadata column."""
        if not isinstance(value_targets, Iterable) or isinstance(value_targets, str):
            raise TypeError("ERROR: Please provide the `value_targets` as list")
        if key is None:
            values = self.fetch_all(column)
        else:
            values = self.fetch(column, key)
        value_map: dict[str, list[int]] = {}
        for index, value in enumerate(values):
            normalized = value.upper()
            value_map.setdefault(normalized, []).append(index)
        result = []
        missing_count = 0
        for target in value_targets:
            normalized = target.upper()
            if normalized in value_map:
                result.extend(value_map[normalized])
            else:
                missing_count += 1
        if missing_count > 0:
            logger.warning(
                f"{missing_count} values were not found in the table column {column}"
            )
        return np.array(result)

    def index_to_bool(self, idx: np.ndarray, invert: bool = False) -> np.ndarray:
        """Convert row indices into a table-sized boolean array."""
        values = np.zeros(self.N, dtype=bool)
        if len(idx) > 0:
            values[idx] = True
        if invert:
            values = ~values
        return values

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
        """Insert a column into the table."""
        column = self._col_renamer(location, column_name)
        if column in ["I", "ids"] and force is False:
            raise ValueError(
                f"ERROR: {column} is a protected column name in MetaData class."
            )
        if column in self.columns and overwrite is False:
            raise ValueError(
                f"ERROR: {column} already exists. Please set `overwrite` to "
                "True to overwrite."
            )
        if isinstance(values, list):
            logger.debug(
                "'values' parameter is of `list` type and not `np.ndarray` as "
                "expected. The correct dtype may not be assigned to the column"
            )
        filled = self._fill_to_index(np.array(values), fill_value, key)
        self._save(column_name, filled, location=location)

    def update_key(self, values: np.ndarray, key: str) -> None:
        """Restrict a boolean metadata key using the supplied values."""
        filled = self._fill_to_index(values, False, key)
        filled = _all_true(np.array([filled, self.fetch_all(key)]))
        self._save(key, filled)

    def reset_key(self, key: str) -> None:
        """Set every value in a boolean metadata key to true."""
        values = np.array([True for _ in range(self.N)]).astype(bool)
        self._save(key, values)

    def drop(self, column: str) -> None:
        """Delete an unprotected metadata column."""
        if column in ["I", "ids", "names"]:
            raise ValueError(
                f"ERROR: {column} is a protected name in MetaData class. "
                "Cannot be deleted"
            )
        location, stored_column = self._get_loc(column)
        del self.locations[location][stored_column]

    def sift(
        self,
        column: str,
        min_v: float = -np.inf,
        max_v: float = np.inf,
        keep_bounds: bool = False,
    ) -> np.ndarray:
        """Return rows whose values fall within the requested bounds."""
        return _sift(self, column, min_v, max_v, keep_bounds)

    def multi_sift(
        self,
        columns: list[str],
        lows: Iterable,
        highs: Iterable,
        keep_bounds: bool = False,
    ) -> np.ndarray:
        """Return a boolean mask where all column filters are satisfied."""
        return _multi_sift(self, columns, lows, highs, keep_bounds)

    def head(self, n: int = 5) -> pd.DataFrame:
        """Return the first ``n`` rows of all metadata columns."""
        return _head(self, n)

    def to_pandas_dataframe(
        self,
        columns: list[str],
        key: str | None = None,
    ) -> pd.DataFrame:
        """Return requested columns as a DataFrame, optionally filtered by key."""
        return _to_pandas_dataframe(self, columns, key)

    def grep(self, pattern: str, only_valid: bool = False) -> list[str]:
        """Return feature names matching a case-insensitive regex."""
        return _grep(self, pattern, only_valid)

    def remove_trend(
        self,
        x: str,
        y: str,
        n_bins: int = 200,
        lowess_frac: float = 0.1,
        fill_value: float = 0,
    ) -> np.ndarray:
        """Remove a LOWESS trend of column ``y`` with respect to column ``x``."""
        return _remove_trend(
            self,
            x,
            y,
            n_bins,
            lowess_frac,
            fill_value,
        )

    def __repr__(self) -> str:
        return f"MetaData of {self.fetch_all('I').sum()}({self.N}) elements"
