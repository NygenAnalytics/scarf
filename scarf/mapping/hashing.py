import hashlib
from typing import Any

import numpy as np

from ..storage.geometry import array_geometry
from ..storage.partition import row_band

# Fixed-width, object, byte, and variable-width strings hash as the same text.
_TEXT_KINDS = frozenset({"O", "S", "T", "U"})


def _update_field(digest: Any, payload: bytes) -> None:
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _update_header(digest: Any, shape: tuple[int, ...], dtype: np.dtype[Any]) -> bool:
    text = dtype.kind in _TEXT_KINDS
    _update_field(digest, repr(shape).encode("ascii"))
    _update_field(digest, b"text" if text else dtype.str.encode("ascii"))
    return text


def _update_values(digest: Any, values: np.ndarray, *, text: bool) -> None:
    if not text:
        digest.update(np.ascontiguousarray(values).tobytes())
        return
    for value in values.reshape(-1):
        if isinstance(value, bytes | bytearray | np.bytes_):
            encoded = bytes(value)
        else:
            encoded = str(value).encode("utf-8")
        _update_field(digest, encoded)


def array_hash(values: np.ndarray | list[Any]) -> str:
    """Return a stable content hash for numeric or identifier arrays.

    Each string is length-prefixed, so different identifier lists never share
    an encoding. Strings hash as text whatever their NumPy dtype, and the
    result equals ``array_store_hash`` for the same values.
    """
    arr = np.asarray(values)
    digest = hashlib.sha256()
    text = _update_header(digest, tuple(int(size) for size in arr.shape), arr.dtype)
    _update_values(digest, arr, text=text)
    return digest.hexdigest()


def array_store_hash(values: Any) -> str:
    """Hash a row-addressable array without materializing it in memory.

    The result equals ``array_hash`` of the same values.
    """
    shape = tuple(int(value) for value in values.shape)
    digest = hashlib.sha256()
    text = _update_header(digest, shape, np.dtype(values.dtype))
    if not shape:
        _update_values(digest, np.asarray(values[...]), text=text)
        return digest.hexdigest()
    row_chunk = row_band(
        array_geometry(values),
        unit="chunk",
        fallback=min(max(shape[0], 1), 10_000),
    )
    for start in range(0, shape[0], row_chunk):
        stop = min(start + row_chunk, shape[0])
        _update_values(digest, np.asarray(values[start:stop]), text=text)
    return digest.hexdigest()
