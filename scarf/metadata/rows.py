from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ..storage.geometry import array_geometry
from ..storage.layout import _encoded_chunk_bound
from ..storage.partition import (
    checked_indices,
    is_contiguous,
    partition_indices,
    row_band,
)


_SELECTION_INDEX_ARRAYS = 16


class _RowReadableMetaData(Protocol):
    N: int

    @property
    def columns(self) -> list[str]: ...

    def _get_array(self, column: str) -> Any: ...

    def _verify_bool(self, key: str) -> bool: ...

    def default_block_rows(self, column: str = "I") -> int: ...


def _read_array_rows(array: Any, rows: np.ndarray) -> np.ndarray:
    indices = np.asarray(rows, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("Metadata row indices must be one-dimensional")
    if indices.size == 0:
        return np.asarray(array[0:0])
    start = int(indices.min())
    stop = int(indices.max()) + 1
    if start < 0 or stop > int(array.shape[0]):
        raise IndexError("Metadata row indices are out of bounds")
    if is_contiguous(indices):
        return np.asarray(array[int(indices[0]) : int(indices[-1]) + 1])

    orthogonal_selection = getattr(array, "get_orthogonal_selection", None)
    if callable(orthogonal_selection):
        try:
            return np.asarray(orthogonal_selection((indices,)))
        except (AttributeError, NotImplementedError, TypeError):
            pass
    coordinate_selection = getattr(array, "get_coordinate_selection", None)
    if callable(coordinate_selection):
        try:
            return np.asarray(coordinate_selection((indices,)))
        except (AttributeError, NotImplementedError, TypeError):
            pass
    return np.asarray(array[indices])


def array_row_selection_parts(array: Any) -> tuple[int, int]:
    """Return fixed and per-row bytes for one chunk-serial selection."""
    itemsize = max(1, int(np.dtype(array.dtype).itemsize))
    index_bytes = np.dtype(np.int64).itemsize
    per_row = 3 * itemsize + _SELECTION_INDEX_ARRAYS * index_bytes
    geometry = array_geometry(array)
    if geometry is None:
        return 0, int(per_row)
    decoded = geometry.nominalChunkBytes()
    return int(decoded + _encoded_chunk_bound(decoded)), int(per_row)


def array_row_selection_peak_bytes(array: Any, rows: int) -> int:
    """Bound one chunk-serial row selection."""
    width = max(0, int(rows))
    if width == 0:
        return 0
    fixed, per_row = array_row_selection_parts(array)
    return int(fixed + width * per_row)


def read_array_rows_chunkwise(array: Any, rows: np.ndarray) -> np.ndarray:
    """Read distinct selected rows one physical chunk at a time."""
    indices = np.asarray(rows, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("Metadata row indices must be one-dimensional")
    if indices.size == 0:
        return np.asarray(array[0:0])

    geometry = array_geometry(array)
    if geometry is None:
        checked = checked_indices(indices, limit=int(array.shape[0]), name="rows")
        return _read_array_rows(array, checked)

    blocks = partition_indices(geometry, 0, indices)
    output = np.empty(indices.size, dtype=np.dtype(array.dtype))
    for block in blocks:
        values = _read_array_rows(array, block.indices)
        if values.shape != block.indices.shape:
            raise ValueError("Metadata row selection returned an invalid shape")
        output[block.destinations] = values
    return output


def read_metadata_rows(
    metadata: _RowReadableMetaData,
    column: str,
    rows: np.ndarray,
) -> np.ndarray:
    """Read selected metadata rows without expanding scattered ranges."""
    return _read_array_rows(metadata._get_array(column), rows)


def read_metadata_rows_chunkwise(
    metadata: _RowReadableMetaData,
    column: str,
    rows: np.ndarray,
) -> np.ndarray:
    """Read distinct metadata rows one physical chunk at a time."""
    return read_array_rows_chunkwise(metadata._get_array(column), rows)


def metadata_row_selection_peak_bytes(
    metadata: _RowReadableMetaData,
    column: str,
    rows: int,
) -> int:
    """Bound one chunk-serial metadata row selection."""
    return array_row_selection_peak_bytes(metadata._get_array(column), rows)


def iter_metadata_column_blocks(
    metadata: _RowReadableMetaData,
    column: str,
    *,
    block_rows: int | None = None,
) -> Iterator[np.ndarray]:
    """Yield every value in a metadata column through bounded slices."""
    array = metadata._get_array(column)
    requested_rows = (
        metadata.default_block_rows(column) if block_rows is None else int(block_rows)
    )
    if requested_rows < 1:
        raise ValueError("block_rows must be >= 1")
    chunk_rows = row_band(
        array_geometry(array),
        unit="chunk",
        fallback=metadata.default_block_rows(column),
    )
    resolved_rows = min(requested_rows, chunk_rows)
    for chunk_start in range(0, metadata.N, chunk_rows):
        chunk_stop = min(chunk_start + chunk_rows, metadata.N)
        for start in range(chunk_start, chunk_stop, resolved_rows):
            stop = min(start + resolved_rows, chunk_stop)
            yield np.asarray(array[start:stop])


def metadata_missing_mask(metadata: Any, column: str) -> Any | None:
    """Return a column's internal missing-mask array when one exists."""
    get_mask = getattr(metadata, "_get_missing_mask_array", None)
    if not callable(get_mask):
        return None
    return get_mask(column)


def read_metadata_missing_rows(
    metadata: Any,
    column: str,
    rows: np.ndarray,
) -> np.ndarray | None:
    """Read a column's internal missing mask for selected rows."""
    mask = metadata_missing_mask(metadata, column)
    if mask is None:
        return None
    return np.asarray(_read_array_rows(mask, rows), dtype=bool)


def read_metadata_missing_rows_chunkwise(
    metadata: Any,
    column: str,
    rows: np.ndarray,
) -> np.ndarray | None:
    """Read a missing mask one physical chunk at a time."""
    mask = metadata_missing_mask(metadata, column)
    if mask is None:
        return None
    return np.asarray(read_array_rows_chunkwise(mask, rows), dtype=bool)


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
