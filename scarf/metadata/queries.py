import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Hashable, Protocol, cast

import numpy as np
import pandas as pd


class _QueryableMetaData(Protocol):
    @property
    def columns(self) -> list[str]: ...

    @property
    def N(self) -> int: ...

    def active_index(self, key: str) -> np.ndarray: ...

    def fetch(self, column: str, key: str = "I") -> np.ndarray: ...

    def fetch_all(self, column: str) -> np.ndarray: ...

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
    """Return rows whose values fall within the requested bounds."""
    values = metadata.fetch_all(column)
    if keep_bounds:
        return (values >= min_v) & (values <= max_v)
    return (values > min_v) & (values < max_v)


def multi_sift(
    metadata: _QueryableMetaData,
    columns: list[str],
    lows: Iterable,
    highs: Iterable,
    keep_bounds: bool = False,
) -> np.ndarray:
    """Return rows that satisfy every requested column filter."""
    return _all_true(
        np.array(
            [
                metadata.sift(low_column, low, high, keep_bounds=keep_bounds)
                for low_column, low, high in zip(columns, lows, highs)
            ]
        )
    )


def head(metadata: _QueryableMetaData, n: int = 5) -> pd.DataFrame:
    """Return the first rows of every metadata column."""
    return pd.DataFrame(
        {column: metadata.fetch_all(column)[:n] for column in metadata.columns}
    )


def to_pandas_dataframe(
    metadata: _QueryableMetaData,
    columns: list[str],
    key: str | None = None,
) -> pd.DataFrame:
    """Return requested metadata columns as a pandas DataFrame."""
    valid_columns = metadata.columns
    frame = pd.DataFrame(
        {
            column: metadata.fetch_all(column)
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
    names = np.array(list(map(str.upper, metadata.fetch_all("names"))))
    if only_valid:
        names = names[metadata.active_index("I")]
    return sorted(
        {name for name in names if re.match(pattern.upper(), name) is not None}
    )


def remove_trend(
    metadata: _QueryableMetaData,
    x: str,
    y: str,
    n_bins: int = 200,
    lowess_frac: float = 0.1,
    fill_value: float = 0,
) -> np.ndarray:
    """Remove a LOWESS trend from one metadata column."""
    from ..features.variability import fit_lowess

    x_values = metadata.fetch(x).astype(float)
    y_values = metadata.fetch(y).astype(float)
    positive = x_values > 0
    trend = fit_lowess(
        x_values[positive],
        y_values[positive],
        n_bins,
        lowess_frac,
        bin_strategy="fixed",
    )
    residuals = np.repeat(fill_value, len(x_values)).astype(float)
    residuals[positive] = trend
    return residuals


_MISSING_LEVEL: Hashable = ("__scarf_missing__",)


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
    for block in metadata.iter_row_blocks(cell_key=cell_key, columns=[column]):
        for value in block.values[column]:
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
    for block in metadata.iter_row_blocks(cell_key=cell_key, columns=[left, right]):
        left_values = block.values[left]
        right_values = block.values[right]
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
    for block in metadata.iter_row_blocks(cell_key=cell_key, columns=[inner, outer]):
        for outer_value, inner_value in zip(
            block.values[outer],
            block.values[inner],
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
    for block in metadata.iter_row_blocks(cell_key=cell_key, columns=ordered):
        unit_values = block.values[observation_unit]
        for index, unit_value in enumerate(unit_values):
            unit_key = level_key(unit_value)
            if unit_key in records:
                continue
            records[unit_key] = {name: block.values[name][index] for name in ordered}
    if not records:
        return pd.DataFrame(columns=ordered)
    return pd.DataFrame(list(records.values()), columns=ordered)
