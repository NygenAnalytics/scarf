import hashlib
from typing import Any

import numpy as np

from ..storage.geometry import array_geometry
from ..storage.partition import row_band


def array_hash(values: np.ndarray | list[Any]) -> str:
    """Return a stable content hash for numeric or identifier arrays."""
    arr = np.asarray(values)
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode())
    digest.update(arr.dtype.str.encode())
    if arr.dtype.kind in {"O", "S", "U"}:
        digest.update(
            "\x1f".join(str(value) for value in arr.reshape(-1)).encode("utf-8")
        )
    else:
        digest.update(np.ascontiguousarray(arr).tobytes())
    return digest.hexdigest()


def array_store_hash(values: Any) -> str:
    """Hash a row-addressable array without materializing it in memory."""
    shape = tuple(int(value) for value in values.shape)
    dtype = np.dtype(values.dtype)
    digest = hashlib.sha256()
    digest.update(str(shape).encode())
    digest.update(dtype.str.encode())
    if not shape:
        digest.update(np.asarray(values[...]).tobytes())
        return digest.hexdigest()
    row_chunk = row_band(
        array_geometry(values),
        unit="chunk",
        fallback=min(max(shape[0], 1), 10_000),
    )
    for start in range(0, shape[0], row_chunk):
        stop = min(start + row_chunk, shape[0])
        block = np.asarray(values[start:stop])
        if dtype.kind in {"O", "S", "U"}:
            digest.update(
                "\x1f".join(str(value) for value in block.reshape(-1)).encode("utf-8")
            )
        else:
            digest.update(np.ascontiguousarray(block).tobytes())
    return digest.hexdigest()
