from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ..storage.geometry import array_geometry
from ..storage.partition import row_band


class _RowReadableMetaData(Protocol):
    N: int

    @property
    def columns(self) -> list[str]: ...

    def _get_array(self, column: str) -> Any: ...

    def _verify_bool(self, key: str) -> bool: ...

    def default_block_rows(self, column: str = "I") -> int: ...


@dataclass(frozen=True, slots=True)
class MetaDataRowBlock:
    """One contiguous slice of a metadata table for blockwise scans."""

    start: int
    stop: int
    active_global_indices: np.ndarray
    values: dict[str, np.ndarray]


def default_block_rows(metadata: _RowReadableMetaData, column: str = "I") -> int:
    """Return a row block size aligned with the backing Zarr chunks."""
    return row_band(
        array_geometry(metadata._get_array(column)),
        unit="chunk",
        fallback=min(metadata.N, 100_000),
    )


def iter_row_blocks(
    metadata: _RowReadableMetaData,
    *,
    cell_key: str = "I",
    columns: Iterable[str] | None = None,
    block_rows: int | None = None,
) -> Iterator[MetaDataRowBlock]:
    """Yield contiguous active row blocks from a metadata table."""
    metadata._verify_bool(cell_key)
    if block_rows is None:
        block_rows = metadata.default_block_rows(cell_key)
    if block_rows < 1:
        raise ValueError("block_rows must be >= 1")

    if columns is None:
        column_names: list[str] = []
    else:
        column_names = list(columns)
        for column in column_names:
            if column not in metadata.columns:
                raise KeyError(f"{column} does not exist in the metadata columns.")

    key_array = metadata._get_array(cell_key)
    column_arrays = {column: metadata._get_array(column) for column in column_names}

    for start in range(0, metadata.N, block_rows):
        stop = min(start + block_rows, metadata.N)
        key_slice = np.asarray(key_array[start:stop], dtype=bool)
        local_indices = np.flatnonzero(key_slice)
        active_global_indices = (local_indices + start).astype(
            np.int64,
            copy=False,
        )
        values: dict[str, np.ndarray] = {}
        for column, array in column_arrays.items():
            block = np.asarray(array[start:stop])
            values[column] = block[local_indices]
        yield MetaDataRowBlock(
            start=start,
            stop=stop,
            active_global_indices=active_global_indices,
            values=values,
        )
