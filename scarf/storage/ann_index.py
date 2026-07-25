import os
import tempfile
from typing import Any

import numpy as np
import zarr

from .types import as_zarr_array
from .layout import _group_zarr_format, get_compressors
from .profiles import StorageProfile, get_storage_profile

ANN_INDEX_ARRAY = "ann_idx_bytes"
ANN_INDEX_CHUNK_BYTES = 8 * 1024 * 1024


def has_ann_index(group: zarr.Group, name: str = ANN_INDEX_ARRAY) -> bool:
    """Return whether a group contains an ANN index byte array."""
    return name in group


def legacy_ann_index_path(zw_root: str | None, ann_loc: str) -> str | None:
    """Return the legacy filesystem path for an ANN index."""
    if zw_root is None:
        return None
    return os.path.join(zw_root, ann_loc, "ann_idx")


def save_ann_index(
    group: zarr.Group,
    ann_idx: Any,
    name: str = ANN_INDEX_ARRAY,
    profile: StorageProfile | None = None,
) -> None:
    """Persist an hnswlib index as a chunked byte array."""
    data = serialize_ann_index(ann_idx)

    if name in group:
        del group[name]
    chunk_size = min(ANN_INDEX_CHUNK_BYTES, max(int(data.shape[0]), 1))
    zarr_format = _group_zarr_format(group)
    array = group.create_array(
        name,
        shape=data.shape,
        chunks=(chunk_size,),
        dtype="uint8",
        overwrite=True,
        compressors=get_compressors(
            profile or get_storage_profile(),
            zarrFormat=zarr_format,
        ),
    )
    array[:] = data
    array.attrs["byte_length"] = int(data.shape[0])


def serialize_ann_index(ann_idx: Any) -> np.ndarray:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        path = tmp.name
    try:
        ann_idx.save_index(path)
        data = np.fromfile(path, dtype=np.uint8)
    finally:
        os.unlink(path)
    return data


def load_ann_index(
    group: zarr.Group,
    space: str,
    dim: int,
    name: str = ANN_INDEX_ARRAY,
) -> Any:
    """Load an hnswlib index from a Zarr byte array."""
    import hnswlib

    if name not in group:
        raise FileNotFoundError(f"ANN index array {name!r} not found in group")
    data = np.asarray(as_zarr_array(group[name], name=name)[:], dtype=np.uint8)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        path = tmp.name
    try:
        data.tofile(path)
        index = hnswlib.Index(space=space, dim=dim)
        index.load_index(path)
        return index
    finally:
        os.unlink(path)


def load_ann_index_from_path(path: str, space: str, dim: int) -> Any:
    """Load an hnswlib index from a legacy filesystem path."""
    import hnswlib

    index = hnswlib.Index(space=space, dim=dim)
    index.load_index(path)
    return index
