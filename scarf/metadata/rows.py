import hashlib
import json
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

    def _get_array(self, column: str) -> Any: ...

    def _bool_array(self, key: str) -> Any: ...

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


def metadata_column_fingerprint(metadata: _RowReadableMetaData, column: str) -> str:
    """Hash one metadata column in bounded blocks.

    The digest covers the stored dtype, the type and representation of each
    object value, and the linked missing mask, so an edit to any of them
    changes it.
    """
    digest = hashlib.sha256()
    for block in iter_metadata_column_blocks(metadata, column):
        digest.update(str(block.dtype).encode())
        digest.update(str(block.shape).encode())
        digest.update(
            json.dumps(
                [(type(value).__name__, repr(value)) for value in block.tolist()]
            ).encode()
            if block.dtype.hasobject
            else block.tobytes()
        )
    missing = metadata_missing_mask(metadata, column)
    digest.update(b"missing:none" if missing is None else b"missing:present")
    if missing is not None:
        for start in range(0, len(missing), 65_536):
            digest.update(
                np.asarray(missing[start : start + 65_536], dtype=bool).tobytes()
            )
    return digest.hexdigest()


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


def apply_missing_mask(
    values: np.ndarray,
    missing: np.ndarray | None,
    *,
    labels: bool = False,
) -> np.ndarray:
    """Show rows flagged by a linked missing mask as missing values.

    Nullable columns and artifacts store a placeholder in each masked row. By
    default, masked numeric rows become NaN in a float64 copy, masked boolean
    rows become False so that boolean filters exclude them, and other masked
    rows become None in an object copy. With ``labels=True``, every masked row
    becomes None in an object copy, so categorical labels keep their values.
    ``values`` is returned unchanged when no row is masked.
    """
    array = np.asarray(values)
    if missing is None:
        return array
    mask = np.asarray(missing, dtype=bool)
    if mask.shape != array.shape:
        raise ValueError("Missing mask does not align with its values")
    if not mask.any():
        return array
    kind = "O" if labels else array.dtype.kind
    if kind == "b":
        output = array.copy()
        output[mask] = False
    elif kind in {"f", "i", "u"}:
        output = array.astype(np.float64, copy=True)
        output[mask] = np.nan
    else:
        output = array.astype(object, copy=True)
        output[mask] = None
    return output


@dataclass(frozen=True, slots=True)
class MetaDataRowBlock:
    """One contiguous slice of a metadata table for blockwise scans."""

    start: int
    stop: int
    active_global_indices: np.ndarray
    values: dict[str, np.ndarray]


def array_block_rows(array: Any, n_rows: int) -> int:
    """Return a row block size aligned with ``array``'s chunks.

    Arrays without chunk geometry use at most 100,000 of their ``n_rows`` rows.
    """
    return row_band(array_geometry(array), unit="chunk", fallback=min(n_rows, 100_000))


def default_block_rows(metadata: _RowReadableMetaData, column: str = "I") -> int:
    """Return a row block size aligned with the backing Zarr chunks."""
    return array_block_rows(metadata._get_array(column), metadata.N)


def iter_row_blocks(
    metadata: _RowReadableMetaData,
    *,
    cell_key: str = "I",
    columns: Iterable[str] | None = None,
    block_rows: int | None = None,
) -> Iterator[MetaDataRowBlock]:
    """Yield contiguous active row blocks from a metadata table."""
    key_array = metadata._bool_array(cell_key)
    if block_rows is None:
        block_rows = array_block_rows(key_array, metadata.N)
    if block_rows < 1:
        raise ValueError("block_rows must be >= 1")
    column_arrays = {
        column: metadata._get_array(column)
        for column in ([] if columns is None else list(columns))
    }

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
