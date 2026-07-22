import re
from collections.abc import Iterable
from typing import Protocol

import numpy as np
import pandas as pd


class _QueryableMetaData(Protocol):
    @property
    def columns(self) -> list[str]: ...

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
