from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Hashable, Protocol, cast

import numpy as np
import pandas as pd

from ..storage.arrays import MISSING_MASK_PREFIX
from ..utils.arrays import regex_match_mask, within_bounds
from .rows import MetaDataRowBlock, apply_missing_mask, metadata_missing_mask


class _QueryableMetaData(Protocol):
    @property
    def columns(self) -> list[str]: ...

    @property
    def N(self) -> int: ...

    def active_index(self, key: str) -> np.ndarray: ...

    def fetch(self, column: str, key: str = "I") -> np.ndarray: ...

    def fetch_all(self, column: str) -> np.ndarray: ...

    def _get_array(self, column: str) -> Any: ...

    def sift(
        self,
        column: str,
        min_v: float = -np.inf,
        max_v: float = np.inf,
        keep_bounds: bool = False,
    ) -> np.ndarray: ...

    def iter_row_blocks(
        self,
        *,
        cell_key: str = "I",
        columns: Iterable[str] | None = None,
        block_rows: int | None = None,
    ) -> Iterator[Any]: ...


def _all_true(bools: np.ndarray) -> np.ndarray:
    combined = bools.sum(axis=0)
    combined[combined < bools.shape[0]] = 0
    return np.asarray(combined, dtype=bool)


def sift(
    metadata: _QueryableMetaData,
    column: str,
    min_v: float = -np.inf,
    max_v: float = np.inf,
    keep_bounds: bool = False,
) -> np.ndarray:
    """Return rows whose values fall within the requested bounds.

    Rows flagged by the column's linked missing mask never pass.
    """
    selected = within_bounds(
        metadata.fetch_all(column), min_v, max_v, keep_bounds=keep_bounds
    )
    mask = metadata_missing_mask(metadata, column)
    if mask is not None:
        selected &= ~np.asarray(mask[:], dtype=bool)
    return np.asarray(selected)


def _filter_values(values: Iterable, name: str, kind: str) -> list[Any]:
    """Return one ``multi_sift`` argument as a list, refusing a bare string."""
    if isinstance(values, str | bytes):
        raise TypeError(f"{name} must be a sequence of {kind}, not a string")
    try:
        return list(values)
    except TypeError:
        raise TypeError(f"{name} must be a sequence of {kind}") from None


def multi_sift(
    metadata: _QueryableMetaData,
    columns: list[str],
    lows: Iterable,
    highs: Iterable,
    keep_bounds: bool = False,
) -> np.ndarray:
    """Return rows that satisfy every requested column filter."""
    column_names = _filter_values(columns, "columns", "column names")
    low_bounds = _filter_values(lows, "lows", "lower bounds")
    high_bounds = _filter_values(highs, "highs", "upper bounds")
    if not column_names:
        raise ValueError("multi_sift requires at least one column")
    if not len(column_names) == len(low_bounds) == len(high_bounds):
        raise ValueError(
            "multi_sift needs one lower and one upper bound for each column: "
            f"{len(column_names)} columns, {len(low_bounds)} lows, and "
            f"{len(high_bounds)} highs"
        )
    return _all_true(
        np.array(
            [
                metadata.sift(column, low, high, keep_bounds=keep_bounds)
                for column, low, high in zip(
                    column_names, low_bounds, high_bounds, strict=True
                )
            ]
        )
    )


def missing_frame_values(values: np.ndarray, missing: np.ndarray | None) -> Any:
    """Return one table column whose masked rows are pandas missing values.

    Masked numeric and text rows follow :func:`apply_missing_mask`. Boolean
    columns with masked rows become a nullable boolean array.
    """
    array = np.asarray(values)
    if missing is None or not np.any(missing) or array.dtype.kind != "b":
        return apply_missing_mask(array, missing)
    return pd.arrays.BooleanArray(array, np.asarray(missing, dtype=bool))


def _frame_column(
    metadata: _QueryableMetaData,
    column: str,
    stop: int | None = None,
) -> Any:
    values = np.asarray(metadata._get_array(column)[:stop])
    mask = metadata_missing_mask(metadata, column)
    return missing_frame_values(values, None if mask is None else mask[:stop])


def head(metadata: _QueryableMetaData, n: int = 5) -> pd.DataFrame:
    """Return the first rows of every metadata column.

    Rows that a column's linked missing mask flags are shown as missing.
    """
    return pd.DataFrame(
        {column: _frame_column(metadata, column, n) for column in metadata.columns}
    )


def to_pandas_dataframe(
    metadata: _QueryableMetaData,
    columns: list[str],
    key: str | None = None,
) -> pd.DataFrame:
    """Return requested metadata columns as a pandas DataFrame.

    Rows that a column's linked missing mask flags are shown as missing.
    """
    valid_columns = metadata.columns
    frame = pd.DataFrame(
        {
            column: _frame_column(metadata, column)
            for column in columns
            if column in valid_columns
        }
    )
    if key is not None:
        frame = frame.reindex(metadata.active_index(key))
    return frame


def grep(
    metadata: _QueryableMetaData,
    pattern: str,
    only_valid: bool = False,
) -> list[str]:
    """Return feature names that match a case-insensitive regex."""
    names = metadata.fetch_all("names")
    if only_valid:
        names = names[metadata.active_index("I")]
    return sorted(
        {str(name).upper() for name in names[regex_match_mask(names, pattern)]}
    )


_MISSING_LEVEL: Hashable = (MISSING_MASK_PREFIX,)


def level_key(value: Any) -> Hashable:
    """Return a hashable equality key that treats missing as one level."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return _MISSING_LEVEL
    try:
        if pd.isna(value):
            return _MISSING_LEVEL
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, list | tuple | set | dict | np.ndarray):
        return ("__scarf_repr__", repr(value))
    return cast(Hashable, value)


def _missing_masks(
    metadata: _QueryableMetaData,
    columns: Sequence[str],
) -> dict[str, Any]:
    return {column: metadata_missing_mask(metadata, column) for column in columns}


def _block_values(
    block: MetaDataRowBlock,
    masks: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Return block columns whose masked rows read as missing (None)."""
    rows = np.asarray(block.active_global_indices, dtype=np.int64) - block.start
    values: dict[str, np.ndarray] = {}
    for column, mask in masks.items():
        raw = np.asarray(block.values[column])
        if mask is None:
            values[column] = raw
            continue
        missing = np.asarray(mask[block.start : block.stop], dtype=bool)[rows]
        values[column] = apply_missing_mask(raw, missing, labels=True)
    return values


@dataclass(frozen=True, slots=True)
class PartitionDigest:
    digest: bytes
    nLevels: int
    nMissing: int
    nRows: int


def column_partition_digest(
    metadata: _QueryableMetaData,
    column: str,
    *,
    cell_key: str = "I",
) -> PartitionDigest:
    """Hash active-row partition codes with one global first-seen codebook."""
    import hashlib

    hasher = hashlib.sha256()
    codes: dict[Hashable, int] = {}
    n_missing = 0
    n_rows = 0
    next_code = 0
    masks = _missing_masks(metadata, [column])
    for block in metadata.iter_row_blocks(cell_key=cell_key, columns=[column]):
        for value in _block_values(block, masks)[column]:
            n_rows += 1
            key = level_key(value)
            if key is _MISSING_LEVEL:
                code = -1
                n_missing += 1
            else:
                existing = codes.get(key)
                if existing is None:
                    code = next_code
                    codes[key] = code
                    next_code += 1
                else:
                    code = existing
            hasher.update(int(code).to_bytes(4, byteorder="little", signed=True))
    return PartitionDigest(
        digest=hasher.digest(),
        nLevels=len(codes) + (1 if n_missing else 0),
        nMissing=n_missing,
        nRows=n_rows,
    )


def columns_same_partition(
    metadata: _QueryableMetaData,
    left: str,
    right: str,
    *,
    cell_key: str = "I",
    sample_limit: int = 8,
) -> tuple[bool, str]:
    """Exact partition equality with a bounded label correspondence string."""
    forward: dict[Hashable, Hashable] = {}
    backward: dict[Hashable, Hashable] = {}
    samples: list[tuple[Any, Any]] = []
    masks = _missing_masks(metadata, list(dict.fromkeys([left, right])))
    for block in metadata.iter_row_blocks(cell_key=cell_key, columns=[left, right]):
        values = _block_values(block, masks)
        left_values = values[left]
        right_values = values[right]
        for left_value, right_value in zip(left_values, right_values, strict=True):
            left_key = level_key(left_value)
            right_key = level_key(right_value)
            mapped = forward.get(left_key)
            if mapped is not None and mapped != right_key:
                return False, ""
            reverse = backward.get(right_key)
            if reverse is not None and reverse != left_key:
                return False, ""
            if mapped is None:
                forward[left_key] = right_key
                backward[right_key] = left_key
                if len(samples) < sample_limit:
                    samples.append((left_value, right_value))
    shown = "; ".join(
        " = ".join(
            "missing" if level_key(value) is _MISSING_LEVEL else str(value)
            for value in pair
        )
        for pair in samples
    )
    if len(forward) > sample_limit:
        shown = f"{shown}; ..."
    return True, shown


def column_constant_within(
    metadata: _QueryableMetaData,
    inner: str,
    outer: str,
    *,
    cell_key: str = "I",
) -> bool:
    """True when ``inner`` does not vary inside each ``outer`` level."""
    seen: dict[Hashable, Hashable] = {}
    masks = _missing_masks(metadata, list(dict.fromkeys([inner, outer])))
    for block in metadata.iter_row_blocks(cell_key=cell_key, columns=[inner, outer]):
        values = _block_values(block, masks)
        for outer_value, inner_value in zip(
            values[outer],
            values[inner],
            strict=True,
        ):
            outer_key = level_key(outer_value)
            inner_key = level_key(inner_value)
            previous = seen.get(outer_key)
            if previous is None:
                seen[outer_key] = inner_key
            elif previous != inner_key:
                return False
    return True


def reduce_observation_units(
    metadata: _QueryableMetaData,
    observation_unit: str,
    columns: list[str],
    *,
    cell_key: str = "I",
) -> pd.DataFrame:
    """Collapse active rows to one record per observation-unit level."""
    ordered = list(dict.fromkeys([observation_unit, *columns]))
    records: dict[Hashable, dict[str, Any]] = {}
    masks = _missing_masks(metadata, ordered)
    for block in metadata.iter_row_blocks(cell_key=cell_key, columns=ordered):
        values = _block_values(block, masks)
        for index, unit_value in enumerate(values[observation_unit]):
            unit_key = level_key(unit_value)
            if unit_key in records:
                continue
            records[unit_key] = {name: values[name][index] for name in ordered}
    if not records:
        return pd.DataFrame(columns=ordered)
    return pd.DataFrame(list(records.values()), columns=ordered)
