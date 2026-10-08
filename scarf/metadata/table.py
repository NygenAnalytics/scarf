from collections.abc import Iterable, Iterator
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import zarr

from ..storage.stores import metadata_workers, run_concurrently
from ..storage.types import as_zarr_array
from ..storage.arrays import (
    MISSING_MASK_PREFIX,
    linked_missing_mask,
)
from ..storage.metadata_keys import (
    is_metadata_column_key,
    metadata_column_key,
    nested_group_error,
    validate_metadata_column_name,
)
from ..utils.logging import logger
from .encoding import checked_fill_value, missing_placeholder, nullable_values
from .queries import (
    grep as _grep,
    head as _head,
    multi_sift as _multi_sift,
    sift as _sift,
    to_pandas_dataframe as _to_pandas_dataframe,
)
from .rows import (
    MetaDataRowBlock,
    default_block_rows as _default_block_rows,
    iter_row_blocks as _iter_row_blocks,
)

_RESERVED_COLUMNS = ("I", "ids", "names")


class CaseInsensitiveIndex:
    """Positions of values, matched as case-insensitive text.

    Values and looked-up names both compare as ``str(value).upper()``, so a
    value of any type can be indexed and looked up. A name that matches no
    value has no positions; callers decide whether that is an error.
    """

    __slots__ = ("_positions",)

    def __init__(self, values: Iterable[Any]) -> None:
        positions: dict[str, list[int]] = {}
        for position, value in enumerate(values):
            positions.setdefault(str(value).upper(), []).append(position)
        self._positions = positions

    def positions(self, name: Any) -> list[int]:
        """Return the positions of the values that match ``name``, in order."""
        return list(self._positions.get(str(name).upper(), ()))


class MetaData:
    """Metadata table for cells and features backed by one Zarr group.

    Changes made through this class are synchronized with the backing store.
    """

    def __init__(self, zgrp: zarr.Group):
        self.locations: dict[str, zarr.Group] = {"primary": zgrp}
        self.N = self._get_size(zgrp)
        self.index = np.array(range(self.N))

    @property
    def _group(self) -> zarr.Group:
        return self.locations["primary"]

    @staticmethod
    def _get_size(zgrp: zarr.Group) -> int:
        # members() reads child metadata concurrently; opening each key in turn
        # costs one sequential round trip per column on object stores.
        sizes = {
            child.shape[0]
            for _key, child in zgrp.members()
            if isinstance(child, zarr.Array)
        }
        if not sizes:
            raise ValueError("Attempted to get size of empty zarr group")
        if len(sizes) != 1:
            raise ValueError(
                "ERROR: Metadata table is corrupted. Not all columns are of same length"
            )
        return sizes.pop()

    @staticmethod
    def _is_public(column: str) -> bool:
        return is_metadata_column_key(column) and not column.startswith(
            MISSING_MASK_PREFIX
        )

    @staticmethod
    def _missing_column(column: str) -> KeyError:
        message = f"{column} does not exist in the metadata columns."
        key = metadata_column_key(column) if isinstance(column, str) else column
        if key != column:
            message += (
                " Scarf stores imported columns with '_' in place of '/' and "
                f"'\\'; look for {key!r} or a numbered variant such as "
                f"'{key}_2'."
            )
        return KeyError(message)

    def _column_names(self) -> list[str]:
        # Zarr lists members in no particular order, so sort them.
        names = sorted(
            column
            for column in self._group.keys()
            if self._is_public(column) and column not in _RESERVED_COLUMNS
        )
        return [*_RESERVED_COLUMNS, *names]

    def _has_column(self, column: str) -> bool:
        # Looking up one column avoids listing and opening every column.
        return column in _RESERVED_COLUMNS or (
            self._is_public(column) and column in self._group
        )

    def _get_array(self, column: str) -> zarr.Array:
        """Return the array of a public column.

        Raises:
            KeyError: If the table has no public column of that name.
            TypeError: If the name holds a group. Stores written before Scarf
                renamed imported columns nested a source column whose name
                contains ``/`` or ``\\`` into groups; such a store must be
                imported again.
        """
        # One metadata read per column: a membership test before indexing
        # doubles the round trips on object stores.
        if self._is_public(column):
            try:
                node = self._group[column]
            except KeyError:
                pass
            else:
                if isinstance(node, zarr.Group):
                    raise nested_group_error(column)
                return as_zarr_array(node, name=column)
        raise self._missing_column(column)

    def _get_missing_mask_array(self, column: str) -> zarr.Array | None:
        if not self._has_column(column):
            raise self._missing_column(column)
        return linked_missing_mask(self._group, column, label=f"Column {column!r}")

    def get_dtype(self, column: str) -> np.dtype[Any]:
        """Return the dtype of a metadata column."""
        return self._get_array(column).dtype

    def _bool_array(self, key: str) -> zarr.Array:
        array = self._get_array(key)
        if array.dtype != bool:  # noqa: E721
            raise TypeError(
                "ERROR: `key` should be name of a boolean type column in Metadata table"
            )
        return array

    @property
    def columns(self) -> list[str]:
        """Return all metadata column names."""
        return self._column_names()

    def fetch_all(self, column: str) -> np.ndarray:
        """Return all stored values from a metadata column.

        Rows flagged by the column's linked missing mask hold stored
        placeholders; ``to_pandas_dataframe`` shows them as missing.
        """
        return np.asarray(self._get_array(column)[:])

    def fetch_all_columns(self, columns: Iterable[str]) -> list[np.ndarray]:
        """Return several whole columns; object stores read them concurrently."""
        return run_concurrently(
            [partial(self.fetch_all, column) for column in columns],
            workers=metadata_workers(self._group),
        )

    def active_index(self, key: str) -> np.ndarray:
        """Return global row indices selected by a boolean column."""
        return np.asarray(self.index[np.asarray(self._bool_array(key)[:])])

    def fetch(self, column: str, key: str = "I") -> np.ndarray:
        """Return stored column values for rows selected by ``key``.

        Rows flagged by the column's linked missing mask hold stored
        placeholders; ``to_pandas_dataframe`` shows them as missing.
        """
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

    def _protect(self, column: str) -> None:
        """Refuse a write that the store protects, before any value is read.

        Whether or not ``column`` exists, a cell table refuses the columns of
        prepared data and the membership column ``<assay>_I`` of each of its
        assays, and a feature table the columns of prepared data (see
        :func:`~scarf.storage.identity.protect_metadata_column`).
        """
        from ..storage.identity import protect_metadata_column

        protect_metadata_column(self._group, column)

    def _save(
        self,
        column_name: str,
        values: np.ndarray,
        missing: np.ndarray | None = None,
    ) -> None:
        """Write one value per row, flagging the ``missing`` rows in a mask.

        The column is replaced through
        :func:`~scarf.storage.identity.replace_metadata_column`, so a failed
        write leaves the previous column, its attributes, and its mask.
        Callers run :meth:`_protect` for the column before they read any
        value, so the replacement does not check protection again.
        """
        validate_metadata_column_name(column_name)
        if isinstance(self._group.get(column_name), zarr.Group):
            raise nested_group_error(column_name)
        if values.shape != (self.N,):
            raise ValueError(
                f"ERROR: Values are of shape: {values.shape}. "
                f"Expected shape is: ({self.N},)"
            )
        if missing is not None and np.shape(missing) != (self.N,):
            raise ValueError(
                f"ERROR: The missing mask is of shape: {np.shape(missing)}. "
                f"Expected shape is: ({self.N},)"
            )
        from ..storage.identity import replace_metadata_column

        replace_metadata_column(self._group, column_name, values, missing)

    def _expand_to_rows(
        self,
        values: Any,
        key: str,
        fill_value: Any = None,
        *,
        name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return one value per row and the rows that are missing.

        Values are typed by
        :func:`~scarf.metadata.encoding.nullable_values`, which also finds the
        missing values among them. Values for every row are returned as they
        are. Values for the rows that the boolean column ``key`` selects are
        spread to those rows, and each other row holds ``fill_value``: with
        None, the placeholder of the dtype, flagged as missing; otherwise the
        fill as a real value, which
        :func:`~scarf.metadata.encoding.checked_fill_value` checks against the
        dtype.

        Raises:
            ValueError: If the values cover neither every row nor the rows
                that ``key`` selects, or the fill does not fit their dtype.
        """
        supplied, supplied_missing = nullable_values(values, name=name)
        n_values = supplied.shape[0]
        if n_values == self.N:
            return supplied, supplied_missing
        selected = np.asarray(self._bool_array(key)[:], dtype=bool)
        selected_count = int(np.count_nonzero(selected))
        if n_values != selected_count:
            raise ValueError(
                f"ERROR: `values`  are of incorrect length ({n_values}). "
                f" Chosen key ({key}) has {selected_count} active rows"
            )
        if fill_value is None:
            dtype = supplied.dtype
            fill = missing_placeholder(dtype)
            missing = ~selected
        else:
            fill, dtype = checked_fill_value(fill_value, supplied.dtype)
            missing = np.zeros(self.N, dtype=bool)
        expanded = np.full(self.N, fill, dtype=dtype)
        expanded[selected] = supplied
        missing[selected] = supplied_missing
        return expanded, missing

    def get_index_by(
        self,
        value_targets: list[Any],
        column: str,
        key: str | None = None,
    ) -> np.ndarray:
        """Return row indices for requested values in a metadata column.

        Values match as case-insensitive text, as in ``CaseInsensitiveIndex``,
        and every row that matches a target is returned, in target order. With
        ``key``, indices are positions among the rows that ``key`` selects. A
        target that matches no row adds no index and is counted in a warning.
        Rows flagged by the column's missing mask never match.
        """
        if not isinstance(value_targets, Iterable) or isinstance(value_targets, str):
            raise TypeError("ERROR: Please provide the `value_targets` as list")
        if key is None:
            values = self.fetch_all(column)
        else:
            values = self.fetch(column, key)
        index = CaseInsensitiveIndex(values)
        mask = self._get_missing_mask_array(column)
        missing = None if mask is None else np.asarray(mask[:], dtype=bool)
        if missing is not None and key is not None:
            missing = missing[self.active_index(key)]
        result: list[int] = []
        missing_count = 0
        for target in value_targets:
            positions = index.positions(target)
            if missing is not None:
                # A missing row holds a placeholder, not a value.
                positions = [
                    position for position in positions if not missing[position]
                ]
            if not positions:
                missing_count += 1
            result.extend(positions)
        if missing_count > 0:
            logger.warning(
                f"{missing_count} values were not found in the table column {column}"
            )
        return np.asarray(result, dtype=np.int64)

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
        values: np.ndarray | list | pd.Series | pd.api.extensions.ExtensionArray,
        fill_value: Any = None,
        key: str = "I",
        overwrite: bool = False,
        force: bool = False,
    ) -> None:
        """Insert a column into the table.

        Raises:
            ValueError: If ``column_name`` is protected, already exists
                without ``overwrite``, or contains ``/`` or ``\\``, which
                Zarr reads as path separators.
        """
        validate_metadata_column_name(column_name)
        if column_name in ["I", "ids"] and force is False:
            raise ValueError(
                f"ERROR: {column_name} is a protected column name in MetaData class."
            )
        self._protect(column_name)
        if overwrite is False and self._has_column(column_name):
            raise ValueError(
                f"ERROR: {column_name} already exists. Please set `overwrite` to "
                "True to overwrite."
            )
        if isinstance(values, list):
            logger.debug(
                "'values' parameter is of `list` type and not `np.ndarray` as "
                "expected. The correct dtype may not be assigned to the column"
            )
        expanded, missing = self._expand_to_rows(
            values, key, fill_value, name=column_name
        )
        if column_name in _RESERVED_COLUMNS and missing.any():
            raise ValueError(
                f"ERROR: {column_name} cannot hold missing values. Pass a value "
                "for every row, or a `fill_value` for the rows that `key` does "
                "not select."
            )
        self._save(column_name, expanded, missing)

    def update_key(self, values: np.ndarray, key: str) -> None:
        """Restrict a boolean metadata key using the supplied values."""
        validate_metadata_column_name(key)
        self._protect(key)
        supplied, missing = nullable_values(values, name=key)
        if supplied.dtype.kind != "b":
            raise TypeError(
                f"ERROR: `values` must be booleans to restrict key {key!r}; "
                f"found dtype {supplied.dtype}"
            )
        restriction, _ = self._expand_to_rows(supplied & ~missing, key, False, name=key)
        current = np.asarray(self._bool_array(key)[:], dtype=bool)
        self._save(key, restriction & current)

    def reset_key(self, key: str) -> None:
        """Set every value in a boolean metadata key to true."""
        validate_metadata_column_name(key)
        self._protect(key)
        values = np.array([True for _ in range(self.N)]).astype(bool)
        self._save(key, values)

    def drop(self, column: str) -> None:
        """Delete an unprotected metadata column."""
        if column in ["I", "ids", "names"]:
            raise ValueError(
                f"ERROR: {column} is a protected name in MetaData class. "
                "Cannot be deleted"
            )
        # Raises for a missing column and for a nested group left by an
        # earlier import.
        self._get_array(column)
        from ..storage.identity import clear_column

        clear_column(self._group, column)

    def sift(
        self,
        column: str,
        min_v: float = -np.inf,
        max_v: float = np.inf,
        keep_bounds: bool = False,
    ) -> np.ndarray:
        """Return rows whose values fall within the requested bounds.

        Rows flagged by the column's linked missing mask never pass.
        """
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
        """Return the first ``n`` rows of all columns, with masked rows missing."""
        return _head(self, n)

    def to_pandas_dataframe(
        self,
        columns: list[str],
        key: str | None = None,
    ) -> pd.DataFrame:
        """Return requested columns as a DataFrame, optionally filtered by key.

        Rows flagged by a column's linked missing mask are shown as missing:
        NaN for numeric columns, which become float64, a nullable boolean for
        boolean columns, and a missing value for other columns.
        """
        return _to_pandas_dataframe(self, columns, key)

    def grep(self, pattern: str, only_valid: bool = False) -> list[str]:
        """Return feature names matching a case-insensitive regex."""
        return _grep(self, pattern, only_valid)

    def __repr__(self) -> str:
        return f"MetaData of {self.fetch_all('I').sum()}({self.N}) elements"
